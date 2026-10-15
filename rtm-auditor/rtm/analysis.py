"""Synthese de couverture, controle d'allocation, metriques et calibration."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .match import SourceResult



# ------------------------------------------------------------------- synthese
def summarize(results: Sequence[SourceResult], src_level: str, tgt_level: str) -> Dict:
    status = Counter(r.status for r in results)
    coverage = Counter(r.coverage for r in results)
    n = len(results) or 1
    linked = sum(1 for r in results if r.existing_ids)
    proposals = sum(1 for r in results if not r.existing_ids
                    and r.status in {"PROPOSITION", "A_VERIFIER"})
    gaps = sum(1 for r in results if r.status == "ECART_SANS_CANDIDAT")
    partial = sum(1 for r in results if r.coverage == "partielle")
    conf = [r.candidates[0].confidence for r in results if r.candidates]
    return {
        "chaine": f"{src_level} -> {tgt_level}",
        "exigences_amont": len(results),
        "deja_tracees": linked,
        "taux_tracabilite_existant": round(linked / n, 4),
        "propositions_de_chainage": proposals,
        "taux_couverture_apres_proposition": round((linked + proposals) / n, 4),
        "ecarts_sans_candidat": gaps,
        "couverture_partielle": partial,
        "confiance_mediane_meilleur_candidat": round(float(np.median(conf)) if conf else 0.0, 4),
        "statuts": dict(status),
        "couvertures": dict(coverage),
    }


def reuse_rate(results: Sequence[SourceResult], high: float = 0.80) -> Dict:
    """Taux de reutilisation : part d'exigences amont dont un requis aval
    existe deja presque a l'identique (candidat a tres forte confiance)."""
    n = len(results) or 1
    strong = sum(1 for r in results if r.candidates and r.candidates[0].confidence >= high)
    identical = sum(1 for r in results
                    if r.candidates and r.candidates[0].feats.get("fuzzy", 0) >= 0.95)
    return {
        "exigences_amont": len(results),
        "reutilisation_forte": strong,
        "taux_reuse_estime": round(strong / n, 4),
        "quasi_identiques": identical,
        "taux_quasi_identique": round(identical / n, 4),
    }


def allocation_check(results: Sequence[SourceResult]) -> List[Dict]:
    """Controle des allocations SE -> sous-systemes.

    Une exigence allouee a une equipe/sous-systeme doit avoir une descendance
    au niveau inferieur : sinon l'allocation n'est pas tracee.
    """
    rows: List[Dict] = []
    for r in results:
        alloc = r.src.allocated_to.strip()
        if not alloc:
            continue
        if r.existing_ids:
            continue
        rows.append({
            "id_amont": r.src.rid,
            "module_amont": r.src.module,
            "allocation": alloc,
            "statut_allocation": r.src.alloc_status,
            "statut_tracabilite": r.status,
            "meilleur_candidat": r.candidates[0].tgt.rid if r.candidates else "",
            "confiance": r.candidates[0].confidence if r.candidates else 0.0,
            "commentaire": "allocation declaree sans lien vers le niveau inferieur",
        })
    return rows


# ------------------------------------------------------------------ metriques
def evaluate(results: Sequence[SourceResult], ks: Sequence[int] = (1, 3, 5)) -> Dict:
    """Mesure le moteur contre les liens DOORS existants (verite terrain).

    Seules les exigences deja tracees sont evaluees : on verifie que le moteur
    retrouve le(s) lien(s) valides par les ingenieurs.
    """
    evaluated = [r for r in results if r.existing_ids]
    if not evaluated:
        return {"exigences_evaluables": 0}
    out: Dict[str, float] = {"exigences_evaluables": len(evaluated)}
    # IMPORTANT : on n'evalue que les candidats *trouves* par le moteur.
    # Les liens existants rajoutes d'office (in_pool=False) sont exclus du
    # classement, sinon le rappel serait artificiellement de 100 %.
    def ranked(r, k):
        found = sorted((c for c in r.candidates if c.in_pool),
                       key=lambda c: -c.confidence)
        return [c.tgt.rid for c in found[:k]]

    for k in ks:
        hits = sum(1 for r in evaluated if r.existing_ids & set(ranked(r, k)))
        out[f"rappel@{k}"] = round(hits / len(evaluated), 4)
    top1 = sum(1 for r in evaluated
               if set(ranked(r, 1)) & r.existing_ids)
    out["precision@1"] = round(top1 / len(evaluated), 4)
    pos = [c.confidence for r in evaluated for c in r.candidates if c.existing]
    neg = [c.confidence for r in evaluated for c in r.candidates
           if not c.existing and c.in_pool]
    out["confiance_moyenne_liens_valides"] = round(float(np.mean(pos)) if pos else 0.0, 4)
    out["confiance_moyenne_non_liens"] = round(float(np.mean(neg)) if neg else 0.0, 4)
    out["separation"] = round(out["confiance_moyenne_liens_valides"]
                              - out["confiance_moyenne_non_liens"], 4)
    return out


def suspicious_links(results: Sequence[SourceResult], weak: float) -> List[Dict]:
    """Liens DOORS existants que le moteur ne confirme pas (a auditer)."""
    rows = []
    for r in results:
        for c in r.candidates:
            if c.existing and c.confidence < weak:
                rows.append({
                    "id_amont": r.src.rid, "module_amont": r.src.module,
                    "id_aval": c.tgt.rid, "module_aval": c.tgt.module,
                    "types_de_lien": ", ".join(c.link_types),
                    "confiance": c.confidence,
                    "verdict_llm": c.llm_verdict,
                    "commentaire": c.llm_reason or "faible similarite semantique",
                })
    return rows


# ----------------------------------------------------------------- calibration
def _samples(results: Sequence[SourceResult], max_hard_neg: int, max_easy_neg: int):
    """Couples etiquetes a partir des liens DOORS existants, par chaine."""
    data: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    for r in results:
        if not r.existing_ids:
            continue
        chain = f"{r.src.level}->{r.candidates[0].tgt.level}" if r.candidates else ""
        if not chain:
            continue
        hard = easy = 0
        for c in r.candidates:
            if c.existing:
                data[chain].append((c.score, c.ratio, 1))
            elif c.synthetic:
                if easy < max_easy_neg:
                    data[chain].append((c.score, c.ratio, 0)); easy += 1
            elif hard < max_hard_neg:
                data[chain].append((c.score, c.ratio, 0)); hard += 1
    return data


def _fit_one(samples: List[Tuple[float, float, int]]) -> Optional[Dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    X = np.asarray([[s[0], s[1]] for s in samples])
    y = np.asarray([s[2] for s in samples])
    if len(set(y.tolist())) < 2 or len(y) < 40:
        return None
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    auc_cv = float(np.mean(cross_val_score(model, X, y, cv=5, scoring="roc_auc")))
    model.fit(X, y)
    a_score, a_ratio = float(model.coef_[0][0]), float(model.coef_[0][1])
    # Garde-fou indispensable : si le modele apprend qu'une *faible* similarite
    # annonce un vrai lien, c'est que la chaine n'est pas de nature semantique
    # (cas TPS -> EL : un requis pointe vers une fiche de preuve, pas vers un
    # texte qui lui ressemble). On refuse alors de calibrer plutot que de
    # produire des taux de confiance inverses.
    if a_score <= 0 or a_ratio <= 0 or auc_cv < 0.55:
        motif = ("le score de similarite n'est pas predictif des liens existants "
                 "sur cette chaine (coefficients negatifs) : chaine non semantique "
                 "ou liens de nature differente" if (a_score <= 0 or a_ratio <= 0)
                 else f"pouvoir predictif insuffisant en validation croisee "
                      f"(AUC={auc_cv:.2f} < 0.55)")
        return {"mode": "rejetee",
                "motif": motif,
                "a_score": a_score, "a_ratio": a_ratio,
                "n_exemples": int(len(y)), "n_positifs": int(y.sum()),
                "auc_validation_croisee": round(auc_cv, 4), "fiabilite": []}
    proba = model.predict_proba(X)[:, 1]
    edges = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    reliability = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi)
        if mask.sum() == 0:
            continue
        reliability.append({"tranche": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                            "couples": int(mask.sum()),
                            "vrais_liens": int(y[mask].sum()),
                            "taux_observe": round(float(y[mask].mean()), 4)})
    return {
        "mode": "platt2",
        "a_score": a_score,
        "a_ratio": a_ratio,
        "b": float(model.intercept_[0]),
        "n_exemples": int(len(y)),
        "n_positifs": int(y.sum()),
        "auc_validation_croisee": round(auc_cv, 4),
        "auc_apprentissage": round(float(roc_auc_score(y, proba)), 4),
        "fiabilite": reliability,
    }


def fit_calibration(results: Sequence[SourceResult],
                    max_hard_neg: int = 6, max_easy_neg: int = 4) -> Optional[Dict]:
    """Calibre score -> taux de confiance, par chaine de tracabilite.

    On ne reapprend pas les poids des signaux : la verite terrain disponible
    (liens deja saisis) ne fournit que des negatifs "durs", ce qui biaise un
    apprentissage multivarie et degrade le classement (mesure sur PKP :
    precision@1 0.81 -> 0.36). On apprend seulement la transformation monotone
    du score en probabilite, a partir du score absolu et de la position
    relative dans le vivier. Le classement reste donc inchange et le taux de
    confiance devient lisible : "0.8" = 8 couples sur 10 a ce niveau etaient de
    vrais liens DOORS.
    """
    data = _samples(results, max_hard_neg, max_easy_neg)
    if not data:
        return None
    out: Dict = {"chaines": {}}
    allsamples: List[Tuple[float, float, int]] = []
    for chain, samples in data.items():
        allsamples += samples
        fit = _fit_one(samples)
        if fit:
            out["chaines"][chain] = fit
    glob = _fit_one(allsamples)
    if glob and glob.get("mode") == "platt2":
        out["chaines"]["_global"] = glob
    return out if out["chaines"] else None


def save_calibration(calib: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2, ensure_ascii=False)
