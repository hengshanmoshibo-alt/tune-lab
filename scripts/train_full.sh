#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab
source .venv-wsl/bin/activate
prepare-chnsenticorp
train-sentiment-lora \
  --output-dir checkpoints/qwen2.5-0.5b-chnsenticorp-lora \
  --epochs 3 \
  --grad-accum 8 \
  --save-steps 200 \
  --eval-steps 200 \
  --logging-steps 20
