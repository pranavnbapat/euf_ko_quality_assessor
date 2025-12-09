# Getting started

## Using vLLM

### 1. Inside runpod

```
cd /workspace
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate

pip install --upgrade pip
pip install vllm transformers accelerate hf_transfer huggingface_hub
```

This ensures:
- vLLM runtime is installed 
- HuggingFace download code works 
- The model loads on GPU without errors


### 2. Start vLLM on GPU 0 (port 8000)
```
CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --served-model-name qwen3:30b-a3b-instruct-2507-q8_0 \
    --dtype auto \
    --max-model-len 32768 \
    --port 8000
```

### 3. Start vLLM on GPU 1 (port 9000)
```
CUDA_VISIBLE_DEVICES=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct \
    --served-model-name qwen3:30b-a3b-instruct-2507-q8_0 \
    --dtype auto \
    --max-model-len 32768 \
    --port 9000
```

vLLM loads the real 14B Qwen2.5 model internally,
but exposes it under custom alias:
```
qwen3:30b-a3b-instruct-2507-q8_0
```

#### Now you have:
- GPU 0 → 8000 → https://<your_id>-8000.proxy.runpod.net 
- GPU 1 → 9000 → https://<your_id>-9000.proxy.runpod.net

### 4. Verify the vLLM servers
In browser:
```
https://<your_id>-8000.proxy.runpod.net/docs
https://<your_id>-9000.proxy.runpod.net/docs
```
- / → gives 404 (normal)
- /docs → Swagger UI (correct)

In terminal:
```
curl https://<your_id>-8000.proxy.runpod.net/v1/models
curl https://<your_id>-9000.proxy.runpod.net/v1/models
```

### 5. Run Both Shards in Parallel
- From your laptop, in one terminal of your code:
```
RUNPOD_OLLAMA_HOST="https://<your_id>-8000.proxy.runpod.net" \
LLM_BACKEND="vllm" \
python main.py summary \
  --shard-index 0 --num-shards 2 &
```

- In another terminal:
```
RUNPOD_OLLAMA_HOST="https://<your_id>-9000.proxy.runpod.net" \
LLM_BACKEND="vllm" \
python main.py summary \
  --shard-index 1 --num-shards 2 &
```

