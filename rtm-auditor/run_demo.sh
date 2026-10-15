#!/usr/bin/env bash
# Demonstration de bout en bout sur un repertoire d'extractions DOORS.
#   ./run_demo.sh /chemin/vers/extractions [budget_llm]
set -euo pipefail
DATA="${1:-..}"
BUDGET="${2:-40}"
PY="${PY:-python3}"

echo "== 1/4 perimetre =="
"$PY" -m rtm.cli inspect --data-dir "$DATA"

echo "== 2/4 tests =="
"$PY" -m pytest tests -q

echo "== 3/4 calibration du taux de confiance =="
"$PY" -m rtm.cli calibrate --data-dir "$DATA" --all-chains --quiet

echo "== 4/4 audit + livrables =="
"$PY" -m rtm.cli audit --data-dir "$DATA" --all-chains \
      --judge-policy targeted --judge-budget "$BUDGET" --out out
