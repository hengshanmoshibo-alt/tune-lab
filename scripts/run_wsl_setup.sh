#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/home/work/projects/PycharmProjects/domain-tune-lab

rm -rf .venv-wsl

if python3 -m venv .venv-wsl >/dev/null 2>&1; then
  true
else
  rm -rf .venv-wsl
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv .venv-wsl
  elif [ -x "$HOME/.local/bin/virtualenv" ]; then
    "$HOME/.local/bin/virtualenv" .venv-wsl
  else
    echo "Missing python3-venv or virtualenv. Install one of them first:"
    echo "  sudo apt update && sudo apt install -y python3.12-venv python3-pip"
    echo "or:"
    echo "  python3 /tmp/get-pip.py --user --break-system-packages && ~/.local/bin/pip3 install --user --break-system-packages virtualenv"
    exit 1
  fi
fi

source .venv-wsl/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install --retries 10 --timeout 120 torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-wsl.txt
pip install -e .

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
