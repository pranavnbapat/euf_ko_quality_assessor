#!/usr/bin/env bash
set -euxo pipefail

# Clean up any previous state
pkill -f 'ollama serve' || true
rm -f /workspace/ollama-gpu0.pid

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

CPU_TOTAL=$(nproc)
export OLLAMA_NUM_THREADS=$(( CPU_TOTAL / 2 ))

# --- Start Ollama on GPU0 → port 11434 ---
CUDA_VISIBLE_DEVICES=0 \
OLLAMA_HOST="0.0.0.0:11434" \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_PARALLEL=1 \
nohup ollama serve > /workspace/ollama-gpu0.log 2>&1 &

echo $! > /workspace/ollama-gpu0.pid

# Wait until GPU0 is ready (max 60s)
deadline=$((SECONDS+60))
until curl -fsS "http://127.0.0.1:11434/api/version" >/dev/null; do
  if [ $SECONDS -ge $deadline ]; then
    echo "GPU0 Ollama server failed to start"
    tail -n 200 /workspace/ollama-gpu0.log >&2 || true
    exit 1
  fi
  sleep 1
done
echo "GPU0 Ollama up on 11434"

# --- Start Ollama on GPU1 → port 11435 ---
CUDA_VISIBLE_DEVICES=1 \
OLLAMA_HOST="0.0.0.0:11435" \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NUM_PARALLEL=1 \
nohup ollama serve > /workspace/ollama-gpu1.log 2>&1 &

echo $! > /workspace/ollama-gpu1.pid

# Wait until GPU1 is ready (max 60s)
deadline=$((SECONDS+60))
until curl -fsS "http://127.0.0.1:11435/api/version" >/dev/null; do
  if [ $SECONDS -ge $deadline ]; then
    echo "GPU1 Ollama server failed to start"
    tail -n 200 /workspace/ollama-gpu1.log >&2 || true
    exit 1
  fi
  sleep 1
done
echo "GPU1 Ollama up on 11435"

# Pull & warm models on GPU0 (11434)
for m in \
  "qwen3:30b-a3b-instruct-2507-q8_0" \
  "nomic-embed-text"
do
  if OLLAMA_HOST="127.0.0.1:11434" ollama pull "$m"; then
    printf "Hello" | OLLAMA_HOST="127.0.0.1:11434" ollama run "$m" >/dev/null || true
  else
    echo "Skip: $m not found on 127.0.0.1:11434" >&2
  fi
done

# Warm up Qwen on GPU1 as well (reuse already-downloaded weights)
printf "Hello" | OLLAMA_HOST="127.0.0.1:11435" ollama run "qwen3:30b-a3b-instruct-2507-q8_0" >/dev/null || true

echo "Ready: dual Ollama (GPU0:11434, GPU1:11435)"

