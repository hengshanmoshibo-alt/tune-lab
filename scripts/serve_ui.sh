#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab
source .venv-wsl/bin/activate
serve-sentiment-ui --host 0.0.0.0 --port 7861
