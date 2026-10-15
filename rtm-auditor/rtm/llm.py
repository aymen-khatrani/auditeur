"""Client Ollama local (embeddings + juge) avec cache disque.

Contrainte PoC : aucune donnee ne sort du poste. Tout passe par l'API HTTP
d'Ollama sur 127.0.0.1 et les resultats sont mis en cache SQLite pour que
deux executions successives soient quasi instantanees et reproductibles.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import requests


class Cache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS kv "
                         "(k TEXT PRIMARY KEY, v BLOB, ts REAL)")
        self.con.commit()

    def get(self, key: str) -> Optional[bytes]:
        row = self.con.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        self.con.execute("INSERT OR REPLACE INTO kv VALUES (?,?,?)",
                         (key, value, time.time()))
        self.con.commit()


def _key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()


class OllamaUnavailable(RuntimeError):
    pass


class Ollama:
    def __init__(self, cfg: dict, cache_path: str = ".cache/llm.sqlite"):
        self.host = cfg["host"].rstrip("/")
        self.embed_model = cfg["embed_model"]
        self.judge_model = cfg["judge_model"]
        self.timeout = cfg.get("timeout_s", 180)
        self.num_ctx = cfg.get("num_ctx", 8192)
        self.num_predict = cfg.get("num_predict", 800)
        self.temperature = cfg.get("temperature", 0.0)
        self.cache = Cache(cache_path)
        self.calls = {"embed": 0, "judge": 0, "cached": 0}

    # ------------------------------------------------------------------ health
    def available(self) -> bool:
        try:
            requests.get(f"{self.host}/api/tags", timeout=5).raise_for_status()
            return True
        except Exception:
            return False

    def models(self) -> List[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    # -------------------------------------------------------------- embeddings
    def embed(self, texts: Sequence[str], batch: int = 16,
              progress=None) -> np.ndarray:
        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        todo: List[int] = []
        for i, t in enumerate(texts):
            hit = self.cache.get(_key("embed", self.embed_model, t))
            if hit is not None:
                vectors[i] = np.frombuffer(hit, dtype=np.float32)
                self.calls["cached"] += 1
            else:
                todo.append(i)

        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            payload = {"model": self.embed_model,
                       "input": [texts[i] or " " for i in chunk]}
            data = self._post("/api/embed", payload)
            embs = data.get("embeddings") or []
            if len(embs) != len(chunk):
                raise OllamaUnavailable(
                    f"reponse d'embedding inattendue ({len(embs)}/{len(chunk)})")
            for i, vec in zip(chunk, embs):
                arr = np.asarray(vec, dtype=np.float32)
                vectors[i] = arr
                self.cache.put(_key("embed", self.embed_model, texts[i]), arr.tobytes())
            self.calls["embed"] += len(chunk)
            if progress:
                progress(min(start + batch, len(todo)), len(todo))

        dim = max((v.shape[0] for v in vectors if v is not None), default=1)
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, v in enumerate(vectors):
            if v is not None and v.shape[0] == dim:
                out[i] = v
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    # -------------------------------------------------------------------- chat
    def judge(self, system: str, user: str, schema_hint: str = "") -> Dict:
        ck = _key("judge", self.judge_model, system, user)
        hit = self.cache.get(ck)
        if hit is not None:
            self.calls["cached"] += 1
            return json.loads(hit)
        payload = {
            "model": self.judge_model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": self.num_predict},
        }
        data = self._post("/api/chat", payload)
        content = (data.get("message") or {}).get("content", "{}")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"_raw": content, "_error": "json invalide"}
        self.calls["judge"] += 1
        self.cache.put(ck, json.dumps(parsed).encode())
        return parsed

    # ------------------------------------------------------------------ interne
    def _post(self, route: str, payload: dict) -> dict:
        try:
            r = requests.post(f"{self.host}{route}", json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            raise OllamaUnavailable(f"Ollama injoignable sur {self.host} : {exc}") from exc
