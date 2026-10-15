"""Normalisation de texte d'exigence et extraction de signaux metier."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set

# Unites rencontrees dans les specifications ferroviaires.
# ATTENTION : alternatives triees du plus long au plus court, sinon "200 km/h"
# serait lu comme "200 km" (la regex s'arrete a la premiere alternative qui
# correspond).
UNITS = "|".join(sorted(
    ["mm", "cm", "m", "km", "km/h", "m/s2", "m/s\u00b2", "m/s", "s", "ms", "min", "h",
     "kg", "t", "kn", "n", "nm", "bar", "pa", "kpa", "mpa", "\u00b0c", "\u00b0", "%",
     "v", "kv", "a", "ma", "w", "kw", "kva", "hz", "db\\(a\\)", "db", "lux", "lx",
     "l", "dm3", "m3", "mm2", "m2"],
    key=len, reverse=True))
RE_QUANTITY = re.compile(rf"(?P<val>[+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>{UNITS})\b", re.I)
RE_STANDARD = re.compile(
    r"\b(?:EN|ISO|IEC|UIC|NF|DIN|TSI|PN|GOST|ASTM)\s?[-–]?\s?\d{2,5}(?:[-–]\d+)*\b", re.I)
RE_CROSSREF = re.compile(r"§\s?\d+(?:\.\d+)*")
RE_MODAL = re.compile(r"\b(shall|must|should|may|will)\b", re.I)
RE_BULLET = re.compile(r"^[\s•\-\*•─-╿]+", re.M)
RE_WS = re.compile(r"\s+")
RE_WORD = re.compile(r"[a-z0-9][a-z0-9\-/]*")

BASE_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "be", "is", "are", "as",
    "by", "with", "on", "at", "it", "its", "this", "that", "these", "those", "from",
    "all", "any", "each", "if", "not", "no", "which", "when", "where", "shall", "will",
    "can", "may", "must", "should", "been", "has", "have", "such", "other", "than",
    "per", "via", "into", "between", "according", "relevant", "case", "used", "use",
}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def clean(text: str) -> str:
    """Texte lisible : retire puces, artefacts d'export et espaces multiples."""
    t = text.replace(" ", " ")
    t = RE_BULLET.sub(" ", t)
    t = re.sub(r"<no [^>]+>", " ", t)
    return RE_WS.sub(" ", t).strip()


def normalize(text: str, stopwords: Set[str] | None = None) -> str:
    """Forme canonique pour la comparaison lexicale."""
    t = strip_accents(clean(text)).lower()
    t = re.sub(r"[^a-z0-9%°./\-\s]", " ", t)
    words = RE_WORD.findall(t)
    sw = BASE_STOPWORDS | (stopwords or set())
    return " ".join(w for w in words if w not in sw and len(w) > 1)


def quantities(text: str) -> Set[str]:
    """Couples valeur+unite normalises ({'200km/h', '1.5m'})."""
    out = set()
    for m in RE_QUANTITY.finditer(text):
        val = m.group("val").replace(",", ".")
        if "." in val:            # ne tronquer que les decimales : 200 reste 200
            val = val.rstrip("0").rstrip(".")
        out.add(f"{val or '0'}{m.group('unit').lower().replace('²', '2')}")
    return out


def standards(text: str) -> Set[str]:
    return {RE_WS.sub("", s.upper().replace("–", "-")) for s in RE_STANDARD.findall(text)}


def crossrefs(text: str) -> Set[str]:
    return {RE_WS.sub("", c) for c in RE_CROSSREF.findall(text)}


def modals(text: str) -> List[str]:
    return [m.lower() for m in RE_MODAL.findall(text)]


def signals(text: str, stopwords: Set[str] | None = None) -> Dict[str, object]:
    """Tous les signaux deterministes d'une exigence, calcules une seule fois."""
    return {
        "norm": normalize(text, stopwords),
        "quantities": quantities(text),
        "standards": standards(text),
        "crossrefs": crossrefs(text),
        "modals": modals(text),
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def atomize(text: str) -> List[str]:
    """Decoupe une exigence en 'aspects' verifiables (puces et phrases modales).

    Sert a mesurer la couverture partielle : un requis aval peut ne couvrir
    qu'une partie des aspects du requis amont.
    """
    t = clean(text)
    parts: List[str] = []
    for chunk in re.split(r"(?<=[.;:])\s+(?=[A-Z0-9])|\n+", t):
        chunk = chunk.strip(" -•\t")
        if len(chunk) >= 15:
            parts.append(chunk)
    if not parts and t:
        parts = [t]
    return parts
