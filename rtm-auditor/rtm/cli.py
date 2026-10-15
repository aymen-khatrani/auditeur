"""Interface en ligne de commande de l'auditeur de tracabilite.

Exemples :
  python -m rtm.cli inspect --data-dir ..
  python -m rtm.cli calibrate --data-dir .. --chain CUS:RS
  python -m rtm.cli audit --data-dir .. --chain CUS:RS --chain RS:AD --llm on
  python -m rtm.cli audit --data-dir .. --all-chains --no-llm
"""
from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
from typing import Dict, List, Optional, Tuple

import yaml

from . import analysis, doors, report
from .llm import Ollama
from .match import Matcher, load_calibration
from .model import set_requirement_types

DEFAULT_CFG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


# ------------------------------------------------------------------ chargement
def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    set_requirement_types(cfg["requirement_types"])
    return cfg


def load(args) -> Tuple[dict, "doors.Corpus"]:
    cfg = load_cfg(args.config)
    files = doors.discover(args.data_dir)
    if not files:
        sys.exit(f"Aucune extraction Excel trouvee dans {args.data_dir}")
    print(f"Lecture de {len(files)} extraction(s) DOORS...")
    corpus = doors.load_corpus(files, cfg)
    return cfg, corpus


def make_llm(cfg: dict, enabled: bool) -> Optional[Ollama]:
    if not enabled:
        print("Mode deterministe : LLM desactive.")
        return None
    llm = Ollama(cfg["ollama"], cache_path=os.path.join(".cache", "llm.sqlite"))
    if not llm.available():
        print(f"ATTENTION : Ollama injoignable sur {cfg['ollama']['host']} -> "
              f"bascule en mode deterministe.")
        return None
    available = llm.models()
    for key, name in (("embed_model", llm.embed_model), ("judge_model", llm.judge_model)):
        if name not in available:
            print(f"ATTENTION : modele '{name}' absent. Modeles presents : "
                  f"{', '.join(available) or 'aucun'}")
            print(f"            -> `ollama pull {name}`")
            return None
    print(f"Ollama OK ({llm.embed_model} + {llm.judge_model})")
    return llm


def parse_chains(cfg: dict, args) -> List[Tuple[str, str]]:
    if args.all_chains or not args.chain:
        return [(c["from"], c["to"]) for c in cfg["chains"]]
    out = []
    for spec in args.chain:
        if ":" not in spec:
            sys.exit(f"Chaine invalide : {spec} (format attendu AMONT:AVAL)")
        a, b = spec.split(":", 1)
        out.append((a.strip().upper(), b.strip().upper()))
    return out


# ------------------------------------------------------------------- commandes
def cmd_inspect(args) -> None:
    cfg, corpus = load(args)
    print(f"\n{'Niveau':8} {'Prefixe':16} {'Module':45} {'Objets':>7} {'Exigences':>10} {'Liens':>7}")
    print("-" * 100)
    for m in corpus.modules:
        reqs = [r for r in m.requirements if r.is_requirement]
        links = sum(len(r.links) for r in m.requirements)
        print(f"{m.level:8} {m.prefix:16} {m.name[:45]:45} {len(m.requirements):7} "
              f"{len(reqs):10} {links:7}")
    print("\nChaines configurees :")
    for a, b in parse_chains(cfg, args):
        na = len(corpus.by_level(a))
        nb = len(corpus.by_level(b))
        flag = "OK" if na and nb else "modules manquants"
        print(f"  {a:5} -> {b:5}  ({na} module(s) amont / {nb} aval)  [{flag}]")


def cmd_calibrate(args) -> None:
    cfg, corpus = load(args)
    llm = make_llm(cfg, args.llm == "on")
    matcher = Matcher(corpus, cfg, llm=llm, verbose=not args.quiet)
    results = []
    for a, b in parse_chains(cfg, args):
        results += matcher.run_chain(a, b, use_llm=False, limit=args.limit,
                                     random_negatives=args.easy_negatives)
    calib = analysis.fit_calibration(results)
    if not calib:
        sys.exit("Pas assez de liens existants pour calibrer (il en faut ~40).")
    analysis.save_calibration(calib, args.calibration)
    print(f"\nCalibration ecrite dans {args.calibration}")
    for chain, fit in calib["chaines"].items():
        print(f"\n  [{chain}] {fit['n_exemples']} couples dont {fit['n_positifs']} "
              f"liens reels - AUC(cv)={fit['auc_validation_croisee']}")
        if fit.get("mode") == "rejetee":
            print(f"    CALIBRATION REFUSEE : {fit['motif']}")
            continue
        for row in fit["fiabilite"]:
            print(f"    confiance {row['tranche']:8} : {row['taux_observe']:.0%} de "
                  f"vrais liens sur {row['couples']:5} couples")


def cmd_audit(args) -> None:
    cfg, corpus = load(args)
    llm = make_llm(cfg, args.llm == "on")
    calib = load_calibration(args.calibration)
    if calib:
        chains_cal = ", ".join(k for k in calib.get("chaines", {}) if k != "_global")
        print(f"Calibration chargee ({args.calibration}) : {chains_cal or 'globale'}")
    if args.topk:
        cfg["matching"]["topk"] = args.topk
    if args.judge_budget is not None:
        cfg["ollama"]["judge_budget"] = args.judge_budget
    if args.judge_policy:
        cfg["ollama"]["judge_policy"] = args.judge_policy

    matcher = Matcher(corpus, cfg, llm=llm, calibration=calib, verbose=not args.quiet)
    chains: List[Dict] = []
    for a, b in parse_chains(cfg, args):
        if not corpus.by_level(a) or not corpus.by_level(b):
            print(f"[{a} -> {b}] ignoree : module amont ou aval absent du perimetre")
            continue
        results = matcher.run_chain(a, b, use_llm=(llm is not None and args.llm == "on"),
                                    include_info=args.include_info, limit=args.limit)
        if not results:
            continue
        chains.append({
            "label": f"{a}->{b}",
            "results": results,
            "summary": analysis.summarize(results, a, b),
            "reuse": analysis.reuse_rate(results),
            "eval": analysis.evaluate(results),
            "allocation": analysis.allocation_check(results),
            "suspicious": analysis.suspicious_links(
                results, cfg["matching"]["thresholds"]["weak"]),
        })
    if not chains:
        sys.exit("Aucune chaine exploitable.")

    context = {
        "timestamp": report.timestamp(),
        "host": f"{platform.system()} {platform.release()} / {socket.gethostname()}",
        "embed_model": cfg["ollama"]["embed_model"],
        "judge_model": cfg["ollama"]["judge_model"],
        "llm": llm is not None and args.llm == "on",
        "llm_calls": llm.calls if llm else {},
        "modules": corpus.modules,
        "calibration": calib,
        "calibration_note": (f"calibration Platt par chaine ({args.calibration})" if calib
                             else "poids par defaut (config.yaml)"),
    }
    out_dir = args.out or cfg["report"]["out_dir"]
    xlsx = report.write_excel(os.path.join(out_dir, "matrice_tracabilite.xlsx"),
                              chains, cfg, context)
    md = report.write_markdown(os.path.join(out_dir, "synthese.md"), chains, context)

    print("\n=== Synthese ===")
    for ch in chains:
        s, ev = ch["summary"], ch["eval"]
        print(f"{s['chaine']:12} amont={s['exigences_amont']:5} "
              f"tracees={s['deja_tracees']:5} ({s['taux_tracabilite_existant']:.0%})  "
              f"propositions={s['propositions_de_chainage']:4}  "
              f"ecarts={s['ecarts_sans_candidat']:4}  "
              f"partielles={s['couverture_partielle']:4}  "
              f"rappel@5={ev.get('rappel@5', 'n/a')}")
    print(f"\nLivrables :\n  {xlsx}\n  {md}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="rtm", description="Auditeur de tracabilite et de couverture des exigences "
                                "(extractions DOORS, 100% local)")
    p.add_argument("--config", default=DEFAULT_CFG, help="fichier de configuration")
    p.add_argument("--data-dir", default=".", help="repertoire des extractions Excel")
    p.add_argument("--chain", action="append", default=[],
                   help="chaine a auditer, ex. CUS:RS (repetable)")
    p.add_argument("--all-chains", action="store_true", help="auditer toutes les chaines")
    p.add_argument("--llm", choices=["on", "off"], default="on", help="activer Ollama")
    p.add_argument("--no-llm", dest="llm", action="store_const", const="off",
                   help="alias de --llm off")
    p.add_argument("--calibration", default="calibration.json")
    p.add_argument("--topk", type=int, default=None)
    p.add_argument("--easy-negatives", type=int, default=0,
                   help="calibrate : nb de negatifs aleatoires ajoutes par exigence "
                        "(0 = calibration conditionnelle au vivier, recommande)")
    p.add_argument("--judge-policy", choices=["all", "targeted", "gaps"], default=None,
                   help="ou appeler le juge LLM (defaut : config.yaml)")
    p.add_argument("--judge-budget", type=int, default=None,
                   help="nombre max d'appels au juge LLM")
    p.add_argument("--include-info", action="store_true",
                   help="inclure les objets 'Information' comme exigences amont")
    p.add_argument("--limit", type=int, default=None, help="limiter les exigences amont")
    p.add_argument("--out", default=None, help="repertoire de sortie")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("command", choices=["inspect", "calibrate", "audit"])
    args = p.parse_args(argv)

    {"inspect": cmd_inspect, "calibrate": cmd_calibrate, "audit": cmd_audit}[args.command](args)


if __name__ == "__main__":
    main()
