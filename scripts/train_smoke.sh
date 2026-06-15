#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab
source .venv-wsl/bin/activate
prepare-chnsenticorp
train-sentiment-lora \
  --output-dir checkpoints/qwen2.5-0.5b-chnsenticorp-lora-smoke \
  --max-train-samples 64 \
  --max-valid-samples 64 \
  --epochs 1 \
  --grad-accum 4 \
  --save-steps 20 \
  --eval-steps 20 \
  --logging-steps 5
