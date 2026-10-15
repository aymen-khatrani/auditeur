"""Juge LLM local : valide les candidats et liste les aspects non couverts.

Un seul appel par exigence amont (avec ses k meilleurs candidats) :
 - limite le temps de calcul sur poste de travail,
 - permet au modele de proposer une *chaine* de plusieurs requis aval,
 - fait ressortir explicitement les aspects orphelins (les ecarts).
Le LLM ne cree jamais de lien seul : il pondere un score deterministe.
"""
from __future__ import annotations

from typing import Dict, List

from . import normalize as nz
from .llm import Ollama, OllamaUnavailable
from .model import Requirement

SYSTEM = (
    "Tu es ingenieur systeme senior en ingenierie des exigences ferroviaire. "
    "Tu evalues si des exigences de niveau inferieur couvrent une exigence de "
    "niveau superieur. Tu es strict : une reformulation vague ne couvre pas une "
    "exigence chiffree. Tu reponds UNIQUEMENT en JSON valide, sans commentaire."
)

TEMPLATE = """Exigence amont ({src_level} - {src_id}) :
Section : {src_section}
Texte : {src_text}

Aspects verifiables identifies dans l'exigence amont :
{aspects}

Candidats de niveau {tgt_level} :
{candidates}

Rends ce JSON exactement :
{{
  "chaine": [
    {{"candidat": "C1",
      "verdict": "couvre|couvre_partiellement|hors_sujet",
      "aspects_couverts": [1, 2],
      "confiance": 0.0,
      "justification": "une phrase, en francais"}}
  ],
  "aspects_non_couverts": [3],
  "couverture": "complete|partielle|absente",
  "confiance_globale": 0.0,
  "ecart": "description courte de ce qui manque, ou vide"
}}

Regles :
- n'invente aucun candidat : utilise les etiquettes C1..C{n} fournies ;
- un candidat hors sujet doit avoir une confiance < 0.2 ;
- "aspects_couverts" renvoie les numeros d'aspects reellement traites ;
- si une valeur chiffree, une unite ou une norme de l'amont n'apparait pas
  dans les candidats, l'aspect correspondant est non couvert ;
- confiance_globale = ta certitude que la chaine proposee couvre l'amont.
"""

VERDICT_WEIGHT = {"couvre": 1.0, "couvre_partiellement": 0.6, "hors_sujet": 0.0}


def build_prompt(src: Requirement, candidates: List[Requirement],
                 max_chars: int = 1200) -> tuple[str, List[str]]:
    aspects = nz.atomize(src.text)[:10]
    aspects_txt = "\n".join(f"{i+1}. {a[:300]}" for i, a in enumerate(aspects)) or "1. (texte court)"
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"C{i} [{c.rid} | {c.module} | {c.obj_type}]\n"
            f"   Section : {c.section[:160]}\n"
            f"   Texte : {nz.clean(c.text)[:max_chars]}"
        )
    prompt = TEMPLATE.format(
        src_level=src.level, src_id=src.rid, src_section=src.section[:200],
        src_text=nz.clean(src.text)[:max_chars], aspects=aspects_txt,
        tgt_level=candidates[0].level if candidates else "?",
        candidates="\n".join(lines), n=len(candidates))
    return prompt, aspects


def judge_requirement(llm: Ollama, src: Requirement,
                      candidates: List[Requirement],
                      max_chars: int = 1200) -> Dict:
    """Retourne le verdict LLM normalise (ou une structure d'erreur exploitable)."""
    if not candidates:
        return {"couverture": "absente", "confiance_globale": 0.0,
                "chaine": [], "aspects_non_couverts": [], "ecart": "aucun candidat"}
    prompt, aspects = build_prompt(src, candidates, max_chars)
    try:
        raw = llm.judge(SYSTEM, prompt)
    except OllamaUnavailable as exc:
        return {"_error": str(exc), "chaine": [], "couverture": "indetermine",
                "confiance_globale": 0.0, "aspects_non_couverts": [], "ecart": ""}
    return _normalize(raw, len(candidates), len(aspects))


def _normalize(raw: Dict, n_cand: int, n_aspects: int) -> Dict:
    out: Dict = {"chaine": [], "aspects_non_couverts": [], "ecart": "",
                 "couverture": "indetermine", "confiance_globale": 0.0}
    if not isinstance(raw, dict):
        return out
    for item in raw.get("chaine") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("candidat", "")).strip().upper()
        if not label.startswith("C") or not label[1:].isdigit():
            continue
        rank = int(label[1:])
        if not 1 <= rank <= n_cand:
            continue
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in VERDICT_WEIGHT:
            verdict = "hors_sujet"
        try:
            conf = float(item.get("confiance", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        covered = [int(a) for a in (item.get("aspects_couverts") or [])
                   if str(a).isdigit() and 1 <= int(a) <= n_aspects]
        out["chaine"].append({
            "rank": rank, "verdict": verdict,
            "confiance": max(0.0, min(1.0, conf)),
            "aspects_couverts": covered,
            "justification": str(item.get("justification", ""))[:400],
        })
    out["aspects_non_couverts"] = [int(a) for a in (raw.get("aspects_non_couverts") or [])
                                   if str(a).isdigit() and 1 <= int(a) <= n_aspects]
    cov = str(raw.get("couverture", "")).strip().lower()
    out["couverture"] = cov if cov in {"complete", "partielle", "absente"} else "indetermine"
    try:
        out["confiance_globale"] = max(0.0, min(1.0, float(raw.get("confiance_globale", 0.0))))
    except (TypeError, ValueError):
        out["confiance_globale"] = 0.0
    out["ecart"] = str(raw.get("ecart", ""))[:500]
    if "_error" in raw:
        out["_error"] = raw["_error"]
    return out
