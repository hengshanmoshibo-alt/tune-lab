#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv-wsl/bin/activate
prepare-chnsenticorp
train-sentiment-lora \
  --output-dir checkpoints/qwen2.5-0.5b-chnsenticorp-lora \
  --max-train-samples 2000 \
  --max-valid-samples 400 \
  --epochs 2 \
  --grad-accum 8 \
  --save-steps 100 \
  --eval-steps 100 \
  --logging-steps 20
