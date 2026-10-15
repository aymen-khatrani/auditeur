"""Signaux de similarite deterministes (hors LLM) entre exigences."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

from . import normalize as nz
from .model import Requirement

FEATURE_NAMES = ["emb", "emb_rel", "tfidf", "tfidf_rel", "fuzzy", "numeric",
                 "section", "standard"]


class FeatureSpace:
    """Prepare les representations lexicales d'un ensemble d'exigences."""

    def __init__(self, reqs: Sequence[Requirement], stopwords: Sequence[str] = ()):
        self.reqs = list(reqs)
        self.stopwords = set(stopwords)
        self.sig: Dict[str, dict] = {
            r.uid: nz.signals(r.text, self.stopwords) for r in self.reqs
        }
        self.section_sig: Dict[str, set] = {
            r.uid: set(nz.normalize(r.section, self.stopwords).split())
            for r in self.reqs
        }

    def norm(self, r: Requirement) -> str:
        return self.sig[r.uid]["norm"]


def build_tfidf(spaces: List[FeatureSpace]):
    """TF-IDF appris sur l'union des niveaux compares (vocabulaire commun)."""
    corpus = [sp.norm(r) for sp in spaces for r in sp.reqs]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                          max_features=200_000)
    vec.fit(corpus or ["placeholder"])
    return vec


def tfidf_matrix(vec, space: FeatureSpace):
    mat = vec.transform([space.norm(r) for r in space.reqs])
    return mat


def zscore(value: float, mean: float, std: float) -> float:
    """Position relative d'un candidat dans son propre vivier, ramenee en [0,1].

    Les similarites absolues d'embeddings sont peu discriminantes (deux
    exigences ferroviaires quelconques se ressemblent). La position relative au
    sein des candidats d'une meme exigence amont l'est davantage. 0.5 = moyenne
    du vivier, 1 = nettement au-dessus, 0 = nettement en-dessous.
    """
    if std <= 1e-9:
        return 0.5
    z = (value - mean) / (3.0 * std)
    return float(max(0.0, min(1.0, 0.5 + 0.5 * z)))


def pair_features(src: Requirement, tgt: Requirement,
                  s_space: FeatureSpace, t_space: FeatureSpace,
                  tfidf: float, emb: float,
                  stats: Dict[str, float] | None = None) -> Dict[str, float]:
    """Vecteur de caracteristiques explicable pour un couple (amont, aval)."""
    ss, ts = s_space.sig[src.uid], t_space.sig[tgt.uid]
    fuzzy = fuzz.token_set_ratio(ss["norm"], ts["norm"]) / 100.0
    numeric = nz.jaccard(ss["quantities"], ts["quantities"])
    standard = nz.jaccard(ss["standards"], ts["standards"])
    sec_a, sec_b = s_space.section_sig[src.uid], t_space.section_sig[tgt.uid]
    section = (len(sec_a & sec_b) / max(1, min(len(sec_a), len(sec_b)))) if sec_a and sec_b else 0.0
    stats = stats or {}
    return {
        "emb": float(emb),
        "emb_rel": zscore(emb, stats.get("emb_mean", 0.0), stats.get("emb_std", 0.0)),
        "tfidf": float(tfidf),
        "tfidf_rel": zscore(tfidf, stats.get("tfidf_mean", 0.0), stats.get("tfidf_std", 0.0)),
        "fuzzy": float(fuzzy),
        "numeric": float(numeric),
        "section": float(min(1.0, section)),
        "standard": float(standard),
    }


def linear_score(feats: Dict[str, float], weights: Dict[str, float]) -> float:
    total = sum(weights.get(k, 0.0) for k in feats)
    if total <= 0:
        return 0.0
    return sum(feats[k] * weights.get(k, 0.0) for k in feats) / total


def cosine_topk(a: np.ndarray, b: np.ndarray, k: int):
    """Top-k par ligne de a contre b (matrices de vecteurs L2-normalises)."""
    sims = a @ b.T
    k = min(k, sims.shape[1]) if sims.shape[1] else 0
    if k == 0:
        return np.zeros((sims.shape[0], 0), dtype=int), np.zeros((sims.shape[0], 0))
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    vals = sims[rows, idx]
    order = np.argsort(-vals, axis=1)
    return idx[rows, order], vals[rows, order]
