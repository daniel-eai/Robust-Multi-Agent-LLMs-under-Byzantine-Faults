#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:-data/byzantine/commonsense}
DATASETS=${DATASETS:-"arc_c hellaswag boolq obqa rte"}
OUT=${OUT:-results/commonsense}
STRONG_MODELS=${STRONG_MODELS:-qwen3-4b}
WEAK_MODELS=${WEAK_MODELS:-qwen2.5-1.5b-instruct}
export STRONG_MODELS WEAK_MODELS
export STRONG_API_BASE_URL=${STRONG_API_BASE_URL:-http://127.0.0.1:8002/v1}
export WEAK_API_BASE_URL=${WEAK_API_BASE_URL:-http://127.0.0.1:8001/v1}
export API_BASE_URL=$STRONG_API_BASE_URL
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}
export API_KEY=$OPENAI_API_KEY

for dataset in $DATASETS; do
  for method in sac cp_wbft; do
    for topology in merg k_circulant robust_random; do
      echo "=== $dataset / $method / $topology ==="
      extra=()
      [ "$method" = "sac" ] && extra=(--adversary-bound 3)
      python methods/unified_entry.py "$method" \
        --dataset-type commonsense170k_mix --data-path "$DATA_DIR/$dataset.json" \
        --agents 7 --malicious 1 --weak-honest 2 --byzantine-mode fixed \
        --mode all --rounds 6 \
        --seed 1234 --position-seed 1234 \
        --strong-model "$STRONG_MODELS" --weak-model "$WEAK_MODELS" \
        --topology "$topology" --output-dir "$OUT/$dataset" --log-level INFO \
        "${extra[@]}"
    done
  done
  python tools/paper_metrics.py --dir "$OUT/$dataset" --out "$OUT/$dataset/tables.md"
done
