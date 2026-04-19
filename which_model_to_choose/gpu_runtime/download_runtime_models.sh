#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

python which_model_to_choose/gpu_runtime/download_runtime_models.py "$@"
