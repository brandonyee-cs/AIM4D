#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip

if [[ "${FULL:-0}" == "1" ]]; then
  pip install -r requirements.txt
else
  pip install requests pandas pyarrow
fi

echo
echo "Setup complete. Activate with: source .venv/bin/activate"
