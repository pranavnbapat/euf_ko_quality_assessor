#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo " which_model_to_choose SGLang Setup"
echo "========================================"

if [[ ! -d "/workspace" ]]; then
  echo "Error: /workspace not found."
  exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
  echo "Error: nvidia-smi not found. GPU not available?"
  exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

cd "$REPO_DIR/.."

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
if [[ -f requirements.txt ]]; then
  pip install -r requirements.txt
fi
pip install sglang openai requests python-dotenv pyyaml tqdm huggingface-hub hf_transfer

mkdir -p /workspace/models
mkdir -p /workspace/.cache/huggingface
mkdir -p "$REPO_DIR/artifacts/candidate_runs"
mkdir -p "$REPO_DIR/artifacts/evaluation_results"

echo ""
echo "Setup complete."
echo ""
echo "Next steps:"
echo "1. Put HF_TOKEN and SGLANG_BASE_URL in which_model_to_choose/gpu_runtime/.env"
echo "2. Generate a GPU-fit config:"
echo "   python which_model_to_choose/gpu_runtime/build_sglang_model_config.py a40"
echo "3. Optionally predownload the configured models:"
echo "   bash which_model_to_choose/gpu_runtime/download_runtime_models.sh"
echo "4. Start SGLang for a chosen model key, or run the full model cycle:"
echo "   python which_model_to_choose/gpu_runtime/start_sglang_server.py --model-key <model_key>"
echo "   bash which_model_to_choose/gpu_runtime/run_model_cycle.sh --task summary --sample-size 50"
