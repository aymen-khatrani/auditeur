"""Moteur de mise en correspondance entre deux niveaux de referentiel."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from . import features as ft
from . import judge as jd
from . import normalize as nz
from .llm import Ollama
from .model import Corpus, Requirement


@dataclass
class Candidate:
    tgt: Requirement
    feats: Dict[str, float]
    score: float                 # score deterministe fusionne, brut (0..1)
    confidence: float            # taux de confiance final (apres calibration + LLM)
    ratio: float = 1.0           # score / meilleur score du vivier (sans echelle)
    existing: bool = False       # lien deja present dans DOORS
    in_pool: bool = True         # issu de la recherche (False = lien existant rajoute)
    synthetic: bool = False      # negatif aleatoire, uniquement pour la calibration
    link_types: List[str] = field(default_factory=list)
    llm_verdict: str = ""
    llm_conf: float = 0.0
    llm_aspects: List[int] = field(default_factory=list)
    llm_reason: str = ""


@dataclass
class SourceResult:
    src: Requirement
    aspects: List[str]
    candidates: List[Candidate]
    existing_ids: Set[str]
    status: str = ""
    coverage: str = ""            # complete / partielle / absente / indetermine
    missing_aspects: List[str] = field(default_factory=list)
    gap_note: str = ""
    judged: bool = False


# --------------------------------------------------------------------- selection
def eligible_sources(corpus: Corpus, level: str, cfg: dict,
                     include_info: bool = False) -> List[Requirement]:
    types = set(cfg["requirement_types"])
    if include_info:
        types |= {"Information"}
    out = []
    for r in corpus.requirements(level):
        if r.is_heading or not nz.clean(r.text):
            continue
        if r.obj_type in types:
            out.append(r)
    return out


def eligible_targets(corpus: Corpus, level: str, cfg: dict) -> List[Requirement]:
    """Cote aval on garde tout objet porteur de contenu (les EL n'ont pas d'ObjectType)."""
    mini = cfg["matching"]["min_text_chars"]
    return [r for r in corpus.requirements(level)
            if not r.is_heading and len(nz.clean(r.text)) >= mini]


def existing_targets(src: Requirement, target_modules: Set[str]) -> Dict[str, List[str]]:
    """Liens DOORS deja presents vers le niveau aval : {ID cible: [types de lien]}."""
    out: Dict[str, List[str]] = {}
    for link in src.links:
        if link.target_module not in target_modules:
            continue
        rid = link.target_id
        if not rid:
            continue
        out.setdefault(rid, []).append(f"{link.direction}/{link.link_type}")
    return out


# ------------------------------------------------------------------------ moteur
class Matcher:
    def __init__(self, corpus: Corpus, cfg: dict, llm: Optional[Ollama] = None,
                 calibration: Optional[dict] = None, verbose: bool = True):
        self.corpus = corpus
        self.cfg = cfg
        self.llm = llm
        self.calibration = calibration
        self.verbose = verbose
        self.stopwords = cfg["matching"].get("domain_stopwords", [])
        # sans LLM, la similarite d'embeddings est absente : on l'exclut du score
        self.active = [f for f in ft.FEATURE_NAMES
                       if not f.startswith("emb") or llm is not None]
        self._emb_cache: Dict[str, np.ndarray] = {}
        # budget d'appels au juge partage par l'ensemble de l'audit (toutes
        # chaines confondues), pour que le temps total reste maitrise
        self.judge_budget = cfg["ollama"].get("judge_budget", 0)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _embed(self, reqs: Sequence[Requirement], role: str = "document") -> np.ndarray:
        if self.llm is None:
            return np.zeros((len(reqs), 1), dtype=np.float32)
        texts = [self._embed_text(r, role) for r in reqs]

        def progress(done_n, total):
            self._log(f"    embeddings {done_n}/{total}")
        return self.llm.embed(texts, progress=progress if self.verbose else None)

    @staticmethod
    def _embed_text(r: Requirement, role: str = "document") -> str:
        # nomic-embed-text attend un prefixe de tache : amont = requete,
        # aval = document indexe.
        prefix = "search_query: " if role == "query" else "search_document: "
        section = r.section.split(" > ")[-1] if r.section else ""
        return f"{prefix}[{r.level}] {section}\n{nz.clean(r.text)[:2000]}"

    # ------------------------------------------------------------------- run
    def run_chain(self, src_level: str, tgt_level: str, *,
                  use_llm: bool = True, topk: Optional[int] = None,
                  include_info: bool = False,
                  limit: Optional[int] = None,
                  random_negatives: int = 0,
                  seed: int = 12345) -> List[SourceResult]:
        mcfg = self.cfg["matching"]
        topk = topk or mcfg["topk"]
        pool = mcfg["candidate_pool"]
        weights = mcfg["weights"]
        thresholds = dict(mcfg["thresholds"])
        thresholds.update(mcfg.get("thresholds_per_chain", {}).get(
            f"{src_level}->{tgt_level}", {}))

        sources = eligible_sources(self.corpus, src_level, self.cfg, include_info)
        targets = eligible_targets(self.corpus, tgt_level, self.cfg)
        if limit:
            sources = sources[:limit]
        self._log(f"[{src_level} -> {tgt_level}] {len(sources)} exigences amont, "
                  f"{len(targets)} objets aval")
        if not sources or not targets:
            return []

        tgt_modules = {m.name for m in self.corpus.by_level(tgt_level)}
        t_index = {t.uid: j for j, t in enumerate(targets)}
        t_by_rid = {t.rid: t for t in targets}
        s_space = ft.FeatureSpace(sources, self.stopwords)
        t_space = ft.FeatureSpace(targets, self.stopwords)
        vec = ft.build_tfidf([s_space, t_space])
        S = ft.tfidf_matrix(vec, s_space)
        T = ft.tfidf_matrix(vec, t_space)

        self._log("  calcul des embeddings locaux (Ollama)...")
        E_s = self._embed(sources, role="query")
        E_t = self._embed(targets, role="document")

        tfidf_sims = (S @ T.T).toarray() if S.shape[0] else np.zeros((0, 0))
        emb_sims = (E_s @ E_t.T) if self.llm is not None else np.zeros_like(tfidf_sims)
        rcfg = self.cfg["matching"].get("retrieval", {"emb": 0.5, "tfidf": 0.5})
        blend = (rcfg["emb"] * emb_sims + rcfg["tfidf"] * tfidf_sims
                 if self.llm is not None else tfidf_sims)
        rng = np.random.default_rng(seed)
        chain_key = f"{src_level}->{tgt_level}"

        results: List[SourceResult] = []
        judging = bool(use_llm and self.llm)
        policy = self.cfg["ollama"].get("judge_policy", "targeted")

        for i, src in enumerate(sources):
            row = blend[i]
            k = min(pool, row.shape[0])
            pool_idx = np.argpartition(-row, k - 1)[:k] if k > 0 else np.array([], int)
            pool_idx = pool_idx[np.argsort(-row[pool_idx])]

            existing = existing_targets(src, tgt_modules)
            stats = {
                "emb_mean": float(np.mean(emb_sims[i, pool_idx])) if len(pool_idx) else 0.0,
                "emb_std": float(np.std(emb_sims[i, pool_idx])) if len(pool_idx) else 0.0,
                "tfidf_mean": float(np.mean(tfidf_sims[i, pool_idx])) if len(pool_idx) else 0.0,
                "tfidf_std": float(np.std(tfidf_sims[i, pool_idx])) if len(pool_idx) else 0.0,
            }
            scored: List[Candidate] = []
            for j in pool_idx:
                tgt = targets[j]
                feats = ft.pair_features(src, tgt, s_space, t_space,
                                         float(tfidf_sims[i, j]), float(emb_sims[i, j]),
                                         stats)
                score = self._score(feats, weights)
                scored.append(Candidate(
                    tgt=tgt, feats=feats, score=score, confidence=score,
                    existing=tgt.rid in existing,
                    link_types=existing.get(tgt.rid, []),
                    in_pool=True,
                ))
            # on garde aussi les liens existants meme s'ils sortent du pool
            present = {c.tgt.rid for c in scored}
            for rid, types in existing.items():
                if rid in present:
                    continue
                tgt = t_by_rid.get(rid)
                if tgt is None:
                    continue
                j = t_index[tgt.uid]
                feats = ft.pair_features(src, tgt, s_space, t_space,
                                         float(tfidf_sims[i, j]), float(emb_sims[i, j]),
                                         stats)
                score = self._score(feats, weights)
                scored.append(Candidate(tgt=tgt, feats=feats, score=score,
                                        confidence=score, existing=True,
                                        link_types=types, in_pool=False))

            # negatifs aleatoires : uniquement pour entrainer la calibration
            if random_negatives:
                taken = {c.tgt.uid for c in scored}
                for j in rng.choice(len(targets), size=min(random_negatives * 3,
                                                           len(targets)), replace=False):
                    tgt = targets[int(j)]
                    if tgt.uid in taken or tgt.rid in existing:
                        continue
                    feats = ft.pair_features(src, tgt, s_space, t_space,
                                             float(tfidf_sims[i, j]),
                                             float(emb_sims[i, j]), stats)
                    scored.append(Candidate(tgt=tgt, feats=feats,
                                            score=self._score(feats, weights),
                                            confidence=0.0, in_pool=False,
                                            synthetic=True))
                    taken.add(tgt.uid)
                    if sum(c.synthetic for c in scored) >= random_negatives:
                        break

            pool_max = max((c.score for c in scored if c.in_pool), default=0.0) or 1e-9
            for c in scored:
                c.ratio = round(min(1.0, c.score / pool_max), 4)
                c.confidence = self._confidence(c.score, c.ratio, chain_key)

            scored.sort(key=lambda c: (-int(c.existing), -c.confidence))
            if random_negatives:
                keep = scored            # on conserve tout pour la calibration
            else:
                keep = scored[: max(topk, len(existing))]
            keep.sort(key=lambda c: -c.confidence)
            res = SourceResult(src=src, aspects=nz.atomize(src.text)[:10],
                               candidates=keep, existing_ids=set(existing))

            self._finalize(res, thresholds)      # pre-statut deterministe
            if (judging and self.judge_budget > 0 and keep
                    and self._needs_judge(res, policy, thresholds)):
                jtopk = self.cfg["ollama"].get("judge_topk", 3)
                before = self.llm.calls["judge"]
                verdict = jd.judge_requirement(
                    self.llm, src, [c.tgt for c in keep[:jtopk]],
                    self.cfg["report"]["max_text_chars"])
                # une reponse servie par le cache ne coute rien : elle ne doit
                # pas consommer le budget d'appels
                if self.llm.calls["judge"] > before:
                    self.judge_budget -= 1
                res.judged = True
                self._apply_llm(res, verdict)
                self._finalize(res, thresholds)   # statut revu apres arbitrage
            results.append(res)
            if self.verbose and (i + 1) % 25 == 0:
                self._log(f"    {i+1}/{len(sources)} exigences analysees")
        return results

    @staticmethod
    def _needs_judge(res: SourceResult, policy: str, th: Dict[str, float]) -> bool:
        """Le juge LLM est appele la ou une decision humaine est attendue."""
        if policy == "all":
            return True
        best = res.candidates[0].confidence if res.candidates else 0.0
        if policy == "gaps":
            return not res.existing_ids and best >= th["weak"]
        # targeted : propositions, liens existants douteux, zones grises
        if res.status == "LIEN_EXISTANT_CONFIRME":
            return False
        return best >= th["weak"] or res.status == "LIEN_EXISTANT_A_REVOIR"

    # -------------------------------------------------------------- scoring
    def _score(self, feats: Dict[str, float], weights: Dict[str, float]) -> float:
        """Score de fusion brut (0..1) : moyenne ponderee des signaux actifs."""
        usable = {k: v for k, v in feats.items() if k in self.active}
        return round(ft.linear_score(usable, weights), 4)

    def _confidence(self, raw: float, ratio: float, chain: str) -> float:
        """Taux de confiance calibre.

        Deux informations complementaires :
          - `raw`   : similarite absolue -> permet de declarer un ecart quand
                      *aucun* candidat ne tient la route ;
          - `ratio` : position relative dans le vivier de l'exigence -> corrige
                      le fait que le niveau de similarite absolu depend
                      enormement du style de redaction du referentiel.
        Sans fichier de calibration, on retombe sur le score brut.
        """
        cal = self._chain_calibration(chain)
        if not cal or cal.get("mode") != "platt2":
            return round(max(0.0, min(1.0, raw)), 4)
        z = cal["b"] + cal["a_score"] * raw + cal["a_ratio"] * ratio
        return round(float(1.0 / (1.0 + np.exp(-z))), 4)

    def _chain_calibration(self, chain: str) -> Optional[dict]:
        if not self.calibration:
            return None
        chains = self.calibration.get("chaines", {})
        cal = chains.get(chain)
        if cal and cal.get("mode") == "platt2":
            return cal
        if cal and cal.get("mode") == "rejetee":
            return None          # chaine explicitement non calibrable
        return chains.get("_global")

    def _apply_llm(self, res: SourceResult, verdict: Dict) -> None:
        by_rank = {c["rank"]: c for c in verdict.get("chaine", [])}
        for rank, cand in enumerate(res.candidates, 1):
            item = by_rank.get(rank)
            if not item:
                continue
            cand.llm_verdict = item["verdict"]
            cand.llm_conf = item["confiance"]
            cand.llm_aspects = item["aspects_couverts"]
            cand.llm_reason = item["justification"]
            w = jd.VERDICT_WEIGHT.get(item["verdict"], 0.0)
            # fusion : le deterministe cadre, le LLM arbitre
            cand.confidence = round(0.55 * cand.score + 0.45 * (w * item["confiance"]), 4)
        res.coverage = verdict.get("couverture", "indetermine")
        res.gap_note = verdict.get("ecart", "")
        # Les aspects non couverts sont *recalcules* : complementaire de l'union
        # des aspects couverts par la chaine proposee. Le champ declare par le
        # modele n'est utilise qu'en complement (les LLM l'oublient souvent).
        covered = {a for item in verdict.get("chaine", [])
                   if item["verdict"] != "hors_sujet"
                   for a in item["aspects_couverts"]}
        declared = {a for a in verdict.get("aspects_non_couverts", [])}
        missing = sorted((set(range(1, len(res.aspects) + 1)) - covered) | declared)
        res.missing_aspects = [res.aspects[a - 1] for a in missing
                               if 1 <= a <= len(res.aspects)]
        if res.missing_aspects and res.coverage == "complete":
            res.coverage = "partielle"
        if verdict.get("_error"):
            res.gap_note = f"[LLM indisponible] {verdict['_error']}"
        res.candidates.sort(key=lambda c: -c.confidence)

    def _finalize(self, res: SourceResult, th: Dict[str, float]) -> None:
        best = res.candidates[0] if res.candidates else None
        best_conf = best.confidence if best else 0.0
        if res.existing_ids:
            conf_existing = max((c.confidence for c in res.candidates if c.existing),
                                default=0.0)
            res.status = ("LIEN_EXISTANT_CONFIRME" if conf_existing >= th["weak"]
                          else "LIEN_EXISTANT_A_REVOIR")
        elif best_conf >= th["propose"]:
            res.status = "PROPOSITION"
        elif best_conf >= th["weak"]:
            res.status = "A_VERIFIER"
        else:
            res.status = "ECART_SANS_CANDIDAT"
        if not res.coverage:
            res.coverage = ("indetermine" if res.status.startswith("LIEN")
                            else ("partielle" if res.status == "A_VERIFIER" else
                                  "absente" if res.status == "ECART_SANS_CANDIDAT"
                                  else "indetermine"))
        if res.status in {"ECART_SANS_CANDIDAT"} and not res.gap_note:
            res.gap_note = "aucun requis aval semantiquement proche"


# ------------------------------------------------------------------ calibration
def load_calibration(path: str) -> Optional[dict]:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None
