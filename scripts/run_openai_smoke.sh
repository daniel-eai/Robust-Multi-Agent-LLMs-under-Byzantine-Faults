#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-gpt-4o-mini}
DATA=${DATA:-data/byzantine/math500/math500_smoke.json}
OUT=${OUT:-results/smoke}

if [ ! -f "$DATA" ]; then
  python tools/make_math500_full.py --out "$DATA" --limit 5
fi

for method in sac cp_wbft; do
  extra=()
  [ "$method" = "sac" ] && extra=(--adversary-bound 3)
  python methods/unified_entry.py "$method" \
    --dataset-type math500 --data-path "$DATA" \
    --agents 7 --malicious 1 --weak-honest 2 --byzantine-mode fixed \
    --mode all --rounds 2 \
    --seed 1234 --position-seed 1234 \
    --strong-model "$MODEL" --weak-model "$MODEL" \
    --topology merg --output-dir "$OUT" --log-level INFO \
    "${extra[@]}"
done

python tools/paper_metrics.py --dir "$OUT"
