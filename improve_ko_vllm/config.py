# improve_ko_vllm/config.py

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------
# vLLM / OpenAI-compatible endpoint
# ------------------------------------------------------------------
RUNPOD_VLLM_HOST = os.environ["RUNPOD_VLLM_HOST"].rstrip("/")

# Canonical base URL for vLLM (OpenAI-compatible)
BASE_VLLM_HOST = RUNPOD_VLLM_HOST

PRIMARY_MODEL = os.environ["VLLM_MODEL"].strip()

MODEL_TO_HOST = {PRIMARY_MODEL: RUNPOD_VLLM_HOST}

VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "131072"))

# ------------------------------------------------------------------
# Chunking / token logic
# ------------------------------------------------------------------
EXTREME_CTX_THRESHOLD_TOK = VLLM_MAX_MODEL_LEN - 3_000  # safety margin
NEAR_LIMIT_CTX_THRESHOLD_TOK = int(VLLM_MAX_MODEL_LEN * 0.85)

CHUNK_TARGET_TOK = 16_000
CHUNK_OVERLAP_TOK = 400

DEFAULT_NUM_PREDICT = 2048
LONG_NUM_PREDICT = 4096
COMBINE_NUM_PREDICT = 24576

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
