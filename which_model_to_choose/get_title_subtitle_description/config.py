# which_model_to_choose/get_title_subtitle_description/config.py

from __future__ import annotations

import os

from pathlib import Path

# ---------- MODELS ----------
models_available = [
    "gpt-oss:20b",
    # "qwen3:30b-a3b-instruct-2507-q8_0",
]

MODEL_OVERRIDES = {
    "gpt-oss:latest": {
        "no_schema": True,
        "use_chat": True,
        "num_predict": 2048,
    },
}

# ---------- SCALING / CHUNKING ----------
EXTREME_CTX_THRESHOLD_TOK = 128_000
NEAR_LIMIT_CTX_THRESHOLD_TOK = 110_000
CHUNK_TARGET_TOK = 16_000
CHUNK_OVERLAP_TOK = 400

DEFAULT_NUM_PREDICT = 2048
LONG_NUM_PREDICT = 4096
COMBINE_NUM_PREDICT = 8192

# ---------- NETWORK / TIMEOUTS ----------
MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 5
PER_REQUEST_TIMEOUT = 600
REQUEST_KEEP_ALIVE = "24h"

# ---------- PATHS ----------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
INPUT_DIR = PROJECT_ROOT / "input"

# ---------- OLLAMA HOSTS (2× GPU) ----------
# Primary host (kept for compatibility)
OLLAMA_HOST = os.environ.get("RUNPOD_OLLAMA_HOST", "https://sezky39f3fx90v-11434.proxy.runpod.net").rstrip("/")

OLLAMA_HOSTS = {
    "gpu0": "https://sezky39f3fx90v-11434.proxy.runpod.net",
    # "gpu1": "https://2xmqzl3bfw42y7-11435.proxy.runpod.net",
}

MODEL_TO_HOST = {
    "gpt-oss:20b": OLLAMA_HOSTS["gpu0"],
    # "qwen3:30b-a3b-instruct-2507-q8_0": OLLAMA_HOSTS["gpu1"],
}
