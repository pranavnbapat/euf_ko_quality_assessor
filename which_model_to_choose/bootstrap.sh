#!/usr/bin/env bash
set -euxo pipefail

# Clean up any previous state
pkill -f 'ollama serve' || true
rm -f /workspace/ollama-gpu0.pid /workspace/ollama-gpu1.pid

# Install / update Ollama (idempotent)
REQUIRED_OLLAMA_VER="0.12.6"
if command -v ollama >/dev/null 2>&1; then
  if ! ollama --version | grep -q " $REQUIRED_OLLAMA_VER$"; then
    echo "Upgrading Ollama to $REQUIRED_OLLAMA_VER…"
    curl -fsSL https://ollama.com/install.sh | sh
  fi
else
  echo "Installing Ollama $REQUIRED_OLLAMA_VER…"
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Common environment
export OLLAMA_LOG_LEVEL=debug
export OLLAMA_KEEP_ALIVE=12h
export OLLAMA_MODELS=/workspace/models
mkdir -p /workspace/models /workspace/models/blobs /workspace/models/manifests
CPU_TOTAL=$(nproc); export OLLAMA_NUM_THREADS=$(( CPU_TOTAL / 2 ))

# Start servers (per-instance settings; sequential to avoid races)
# GPU0 -> 11434 (allow 2 models / 2 parallel reqs)
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST="0.0.0.0:11434" \
OLLAMA_MAX_LOADED_MODELS=2 \
OLLAMA_NUM_PARALLEL=2 \
nohup ollama serve > /workspace/ollama-gpu0.log 2>&1 &
echo $! > /workspace/ollama-gpu0.pid

# Wait until GPU0 is ready (max 60s)
deadline=$((SECONDS+60))
until curl -fsS "http://127.0.0.1:11434/api/version" >/dev/null; do
  [ $SECONDS -ge $deadline ] && { echo "GPU0 failed to start"; tail -n 200 /workspace/ollama-gpu0.log >&2; exit 1; }
  sleep 1
done
echo "GPU0 up on 11434"

## GPU1 -> 11435 (conservative for 30B)
#CUDA_VISIBLE_DEVICES=1 \
#OLLAMA_HOST="0.0.0.0:11435" \
#OLLAMA_MAX_LOADED_MODELS=1 \
#OLLAMA_NUM_PARALLEL=1 \
#nohup ollama serve > /workspace/ollama-gpu1.log 2>&1 &
#echo $! > /workspace/ollama-gpu1.pid

# Wait until GPU1 is ready (max 60s)
#deadline=$((SECONDS+60))
#until curl -fsS "http://127.0.0.1:11435/api/version" >/dev/null; do
#  [ $SECONDS -ge $deadline ] && { echo "GPU1 failed to start"; tail -n 200 /workspace/ollama-gpu1.log >&2; exit 1; }
#  sleep 1
#done
#echo "GPU1 up on 11435"

# Pull & warm models on BOTH instances
#for H in 127.0.0.1:11434 127.0.0.1:11435; do
hosts=(127.0.0.1:11434)
models=(
  "gpt-oss:20b"
  "nomic-embed-text"
  # "qwen3:4b-instruct-2507-q4_K_M"
  # "qwen3:30b-a3b-instruct-2507-q8_0"
)

for H in "${hosts[@]}"; do
  for m in "${models[@]}"; do
    if OLLAMA_HOST="$H" ollama pull "$m"; then
      printf "Hello" | OLLAMA_HOST="$H" ollama run "$m" >/dev/null || true
    else
      echo "Skip: $m not found on $H" >&2
    fi
  done
done

echo "Ready: GPU0 on 11434 (PID $(cat /workspace/ollama-gpu0.pid))"
#echo "Ready: GPU0 on 11434 (PID $(cat /workspace/ollama-gpu0.pid)), GPU1 on 11435 (PID $(cat /workspace/ollama-gpu1.pid))"
