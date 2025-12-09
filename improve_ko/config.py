# which_model_to_choose/get_summaries/config.py

from __future__ import annotations

import os

from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_OLLAMA_HOST = os.environ.get("RUNPOD_OLLAMA_HOST",).rstrip("/")
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").strip().lower()

# ---------- MODELS ----------
models_available = [
    "qwen3:30b-a3b-instruct-2507-q8_0",
    # "gpt-oss:20b"
]

PRIMARY_MODEL = models_available[0]

MODEL_OVERRIDES: dict[str, dict] = {}

# ---------- SCALING / CHUNKING ----------
EXTREME_CTX_THRESHOLD_TOK = 128_000
NEAR_LIMIT_CTX_THRESHOLD_TOK = 110_000
CHUNK_TARGET_TOK = 16_000
CHUNK_OVERLAP_TOK = 400

DEFAULT_NUM_PREDICT = 2048
LONG_NUM_PREDICT = 4096
COMBINE_NUM_PREDICT = 8192

# How many times to re-call the model if parsing/summary fails or is empty
SUMMARY_MAX_ATTEMPTS = 3

# ---------- NETWORK / TIMEOUTS ----------
MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 5
PER_REQUEST_TIMEOUT = 600
REQUEST_KEEP_ALIVE = "24h"

# ---------- PATHS ----------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
INPUT_DIR = PROJECT_ROOT / "input"

OLLAMA_HOSTS = {
    "gpu0": BASE_OLLAMA_HOST,
}

MODEL_TO_HOST = {
    "qwen3:30b-a3b-instruct-2507-q8_0": OLLAMA_HOSTS["gpu0"],
}
