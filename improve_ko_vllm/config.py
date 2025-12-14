# improve_ko_vllm/config.py

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# vLLM / OpenAI-compatible endpoint
# ------------------------------------------------------------------
BASE_VLLM_HOST = os.environ["RUNPOD_VLLM_HOST"].rstrip("/")


# ------------------------------------------------------------------
# Models
# ------------------------------------------------------------------
models_available = [
    "qwen3:30b-a3b-instruct-2507-q8_0",
]

PRIMARY_MODEL = models_available[0]

MODEL_TO_HOST = {
    PRIMARY_MODEL: BASE_VLLM_HOST,
}


# ------------------------------------------------------------------
# Chunking / token logic
# ------------------------------------------------------------------
EXTREME_CTX_THRESHOLD_TOK = 128_000
NEAR_LIMIT_CTX_THRESHOLD_TOK = 110_000

CHUNK_TARGET_TOK = 16_000
CHUNK_OVERLAP_TOK = 400

DEFAULT_NUM_PREDICT = 2048
LONG_NUM_PREDICT = 4096
COMBINE_NUM_PREDICT = 8192

SUMMARY_MAX_ATTEMPTS = 3


# ------------------------------------------------------------------
# Networking
# ------------------------------------------------------------------
MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 5
PER_REQUEST_TIMEOUT = 600


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent
INPUT_DIR = PROJECT_ROOT / "input"
