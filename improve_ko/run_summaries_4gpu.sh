#!/usr/bin/env bash
set -e

# Shard 0
RUNPOD_OLLAMA_HOST="https://<id>-8000.proxy.runpod.net" \
LLM_BACKEND="vllm" \
python main.py summary --shard-index 0 --num-shards 4 &

# Shard 1
RUNPOD_OLLAMA_HOST="https://<id>-9000.proxy.runpod.net" \
LLM_BACKEND="vllm" \
python main.py summary --shard-index 1 --num-shards 4 &

# Shard 2
RUNPOD_OLLAMA_HOST="https://<id>-10000.proxy.runpod.net" \
LLM_BACKEND="vllm" \
python main.py summary --shard-index 2 --num-shards 4 &

# Shard 3
RUNPOD_OLLAMA_HOST="https://<id>-11000.proxy.runpod.net" \
LLM_BACKEND="vllm" \
python main.py summary --shard-index 3 --num-shards 4 &
