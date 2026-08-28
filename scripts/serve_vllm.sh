#!/usr/bin/env bash
set -euo pipefail

STRONG_HF_MODEL=${STRONG_HF_MODEL:-Qwen/Qwen3-4B}
WEAK_HF_MODEL=${WEAK_HF_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
STRONG_PORT=${STRONG_PORT:-8002}
WEAK_PORT=${WEAK_PORT:-8001}
GPU_FRACTION=${GPU_FRACTION:-0.45}

python -m vllm.entrypoints.openai.api_server \
    --model "$STRONG_HF_MODEL" --served-model-name qwen3-4b \
    --port "$STRONG_PORT" --gpu-memory-utilization "$GPU_FRACTION" \
    --max-model-len 8192 &

python -m vllm.entrypoints.openai.api_server \
    --model "$WEAK_HF_MODEL" --served-model-name qwen2.5-1.5b-instruct \
    --port "$WEAK_PORT" --gpu-memory-utilization "$GPU_FRACTION" \
    --max-model-len 8192 &

wait
