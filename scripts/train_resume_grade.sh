#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab
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
