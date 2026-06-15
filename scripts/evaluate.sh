#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab
source .venv-wsl/bin/activate

evaluate-sentiment-models \
  --test-file data/processed/chnsenticorp_sentiment/test.jsonl \
  --output-dir outputs/chnsenticorp_eval \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter-path checkpoints/qwen2.5-0.5b-chnsenticorp-lora \
  --modes base,lora \
  --load-in-4bit
