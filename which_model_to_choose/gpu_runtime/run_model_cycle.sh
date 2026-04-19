#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

if [[ -f "which_model_to_choose/gpu_runtime/.env" ]]; then
  set -a
  source "which_model_to_choose/gpu_runtime/.env"
  set +a
fi

TASK=""
RUN_ID="$(date +%Y%m%d_%H%M%S)"
SUMMARY_RUN_ID=""
SAMPLE_SIZE=""
SAMPLE_SEED="42"
EXPORT_FORMAT="both"
INPUT_PATH=""
PORT="${SGLANG_PORT:-8000}"
HOST="${SGLANG_HOST:-127.0.0.1}"
DOWNLOAD_FIRST="0"

usage() {
  cat <<EOF
Usage:
  bash which_model_to_choose/gpu_runtime/run_model_cycle.sh --task summary [options]
  bash which_model_to_choose/gpu_runtime/run_model_cycle.sh --task metadata --summary-run-id <run_id> [options]

Options:
  --task <summary|metadata>
  --run-id <run_id>
  --summary-run-id <run_id>
  --sample-size <n>
  --sample-seed <seed>
  --export-format <json|csv|both>
  --input <path>
  --port <port>
  --host <host>
  --download-first
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --summary-run-id) SUMMARY_RUN_ID="$2"; shift 2 ;;
    --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
    --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
    --export-format) EXPORT_FORMAT="$2"; shift 2 ;;
    --input) INPUT_PATH="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --download-first) DOWNLOAD_FIRST="1"; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$TASK" != "summary" && "$TASK" != "metadata" ]]; then
  echo "Error: --task must be summary or metadata" >&2
  exit 2
fi

if [[ "$TASK" == "metadata" && -z "$SUMMARY_RUN_ID" ]]; then
  echo "Error: --summary-run-id is required for metadata" >&2
  exit 2
fi

if [[ "$DOWNLOAD_FIRST" == "1" ]]; then
  bash which_model_to_choose/gpu_runtime/download_runtime_models.sh
fi

MODEL_KEYS="$(python - <<'PY'
from which_model_to_choose.generation.config import load_runtime_models
print("\n".join(load_runtime_models().keys()))
PY
)"

if [[ -z "$MODEL_KEYS" ]]; then
  echo "Error: no model keys found in runtime_config.yaml" >&2
  exit 1
fi

wait_for_server() {
  local base_url="$1"
  local attempts=0
  local auth_args=()
  if [[ -n "${SGLANG_API_KEY:-}" ]]; then
    auth_args=(-H "Authorization: Bearer ${SGLANG_API_KEY}")
  fi
  until curl -fsS "${auth_args[@]}" "${base_url}/models" >/dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [[ $attempts -ge 120 ]]; then
      echo "Error: timed out waiting for SGLang at ${base_url}" >&2
      return 1
    fi
    sleep 2
  done
}

for MODEL_KEY in $MODEL_KEYS; do
  echo ""
  echo "=== MODEL: ${MODEL_KEY} ==="

  LOG_DIR="which_model_to_choose/output/${RUN_ID}/server_logs"
  mkdir -p "$LOG_DIR"
  LOG_FILE="${LOG_DIR}/${MODEL_KEY}.log"

  python which_model_to_choose/gpu_runtime/start_sglang_server.py \
    --model-key "$MODEL_KEY" \
    --host "$HOST" \
    --port "$PORT" \
    >"$LOG_FILE" 2>&1 &
  SERVER_PID=$!

  cleanup() {
    if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      kill "$SERVER_PID" >/dev/null 2>&1 || true
      wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup EXIT

  wait_for_server "http://${HOST}:${PORT}/v1"

  CMD=(python -m which_model_to_choose.generation.run_candidates
    --task "$TASK"
    --run-id "$RUN_ID"
    --model "$MODEL_KEY"
    --sample-seed "$SAMPLE_SEED"
    --export-format "$EXPORT_FORMAT"
  )
  if [[ -n "$INPUT_PATH" ]]; then
    CMD+=(--input "$INPUT_PATH")
  fi
  if [[ -n "$SAMPLE_SIZE" ]]; then
    CMD+=(--sample-size "$SAMPLE_SIZE")
  fi
  if [[ "$TASK" == "metadata" ]]; then
    CMD+=(--summary-run-id "$SUMMARY_RUN_ID")
  fi

  "${CMD[@]}"

  cleanup
  trap - EXIT
done
