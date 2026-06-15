#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv-wsl/bin/activate
serve-sentiment-ui --host 0.0.0.0 --port 7861
