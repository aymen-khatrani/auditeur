"""Tests du PoC : parsing DOORS, signaux, calibration, bout en bout sans LLM."""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtm import analysis, doors, judge  # noqa: E402
from rtm import normalize as nz  # noqa: E402
from rtm.match import Matcher, existing_targets  # noqa: E402
from rtm.model import set_requirement_types  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("RTM_DATA", os.path.dirname(ROOT))


@pytest.fixture(scope="session")
def cfg():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as fh:
        c = yaml.safe_load(fh)
    set_requirement_types(c["requirement_types"])
    return c


@pytest.fixture(scope="session")
def corpus(cfg):
    files = doors.discover(DATA)
    if not files:
        pytest.skip(f"aucune extraction DOORS dans {DATA}")
    return doors.load_corpus(files, cfg)


# ---------------------------------------------------------------- parsing pur
def test_split_id():
    assert doors.split_id("SA.10.RS 1014") == ("SA.10.RS", 1014)
    assert doors.split_id("PKP-ANNEX1.CUS-3339") == ("PKP-ANNEX1.CUS", 3339)
    assert doors.split_id("sans numero") == ("sans numero", None)


def test_parse_object_links():
    cell = ("In [Specifies]\nRolling Stock System.RS#303\nRolling Stock System.RS#365\n"
            "Out [Evidence]\nEvidence.EL#462")
    links = doors.parse_object_links(cell)
    assert [(l.direction, l.link_type, l.target_abs) for l in links] == [
        ("In", "Specifies", 303), ("In", "Specifies", 365), ("Out", "Evidence", 462)]


def test_parse_requirement_refs():
    cell = ("RSSys.RS 273, Rolling Stock System.RS\n§2.1 Purpose\nLe RSS...\n"
            + "┈" * 20 + "\nRSSys.RS 365, Rolling Stock System.RS\n§2.2\n")
    refs = doors.parse_requirement_refs(cell, "ParentRequirements", "In")
    assert [r.target_id for r in refs] == ["RSSys.RS 273", "RSSys.RS 365"]
    assert refs[0].target_abs == 273


def test_detect_level(cfg):
    assert doors.detect_level(cfg["levels"], "Doors and entrances.RS") == "RS"
    assert doors.detect_level(cfg["levels"], "S14-01 Passenger access doors.RSTPS") == "TPS"
    assert doors.detect_level(cfg["levels"], "quelconque.docx") == "UNKNOWN"


# ------------------------------------------------------------------- signaux
def test_quantities_and_standards():
    t = "The door shall close in 3,5 s at 200 km/h according to EN 14752 and TSI 1302."
    assert "3.5s" in nz.quantities(t)
    assert "200km/h" in nz.quantities(t)
    assert nz.standards(t) >= {"EN14752", "TSI1302"}


def test_normalize_drops_stopwords():
    out = nz.normalize("The system shall provide the door", {"system"})
    assert "shall" not in out and "system" not in out and "door" in out


def test_atomize_splits_aspects():
    text = "The door shall open in 3 s. The door shall close in 4 s; and lock."
    assert len(nz.atomize(text)) >= 2


# ----------------------------------------------------------------- juge (hors ligne)
def test_judge_normalisation_filtre_les_hallucinations():
    raw = {"chaine": [{"candidat": "C9", "verdict": "couvre", "confiance": 1.0},
                      {"candidat": "C1", "verdict": "inconnu", "confiance": 5,
                       "aspects_couverts": [1, 99]}],
           "aspects_non_couverts": [2], "couverture": "partielle",
           "confiance_globale": 0.9}
    out = judge._normalize(raw, n_cand=3, n_aspects=2)
    assert [c["rank"] for c in out["chaine"]] == [1]          # C9 inexistant -> rejete
    assert out["chaine"][0]["verdict"] == "hors_sujet"        # verdict inconnu -> prudent
    assert out["chaine"][0]["confiance"] == 1.0               # borne a [0,1]
    assert out["chaine"][0]["aspects_couverts"] == [1]        # aspect 99 rejete
    assert out["aspects_non_couverts"] == [2]


# ----------------------------------------------------------- bout en bout (sans LLM)
def test_corpus_charge_les_niveaux(corpus):
    levels = {m.level for m in corpus.modules}
    assert {"CUS", "RS", "AD", "TPS"} <= levels
    assert all(m.prefix for m in corpus.modules)


def test_liens_resolus_en_identifiants(corpus):
    reqs = [r for r in corpus.requirements("CUS") if r.links]
    assert reqs, "la specification client doit porter des liens"
    resolved = [l for r in reqs for l in r.links if l.target_id]
    assert resolved, "au moins un lien doit etre resolu vers un ID DOORS"


def test_audit_deterministe_retrouve_les_liens_existants(corpus, cfg):
    matcher = Matcher(corpus, cfg, llm=None, verbose=False)
    results = matcher.run_chain("AD", "TPS", use_llm=False)
    assert results
    ev = analysis.evaluate(results)
    assert ev["exigences_evaluables"] > 100, "verite terrain insuffisante"
    # sans embeddings, le TF-IDF seul doit deja retrouver la majorite des liens
    assert ev["rappel@5"] >= 0.7, ev
    summary = analysis.summarize(results, "AD", "TPS")
    assert summary["exigences_amont"] == len(results)
    assert set(summary["statuts"]) <= {
        "LIEN_EXISTANT_CONFIRME", "LIEN_EXISTANT_A_REVOIR", "PROPOSITION",
        "A_VERIFIER", "ECART_SANS_CANDIDAT"}


def test_existing_targets_filtre_par_module(corpus):
    src = next(r for r in corpus.requirements("CUS")
               if any(l.target_id for l in r.links))
    assert existing_targets(src, set()) == {}


def test_calibration_refuse_les_chaines_non_semantiques(corpus, cfg):
    matcher = Matcher(corpus, cfg, llm=None, verbose=False)
    results = matcher.run_chain("TPS", "EL", use_llm=False, limit=200)
    calib = analysis.fit_calibration(results)
    if calib:   # la chaine preuve n'est pas de nature semantique
        fit = calib["chaines"].get("TPS->EL")
        assert fit is None or fit["mode"] == "rejetee"
