from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
RUNTIME_DIR = PROJECT_ROOT / "which_model_to_choose" / "gpu_runtime"
ARTIFACTS_DIR = PROJECT_ROOT / "which_model_to_choose" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "which_model_to_choose" / "output"
CANDIDATE_RUNS_DIR = ARTIFACTS_DIR / "candidate_runs"
EVALUATION_RESULTS_DIR = ARTIFACTS_DIR / "evaluation_results"
INPUT_DIR = PROJECT_ROOT / "input"
RUNTIME_CONFIG_PATH = RUNTIME_DIR / "runtime_config.yaml"

if load_dotenv is not None:
    load_dotenv(RUNTIME_DIR / ".env")

SGLANG_BASE_URL = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
SGLANG_API_KEY = os.environ.get("SGLANG_API_KEY", "sk-sglang-local")

DEFAULT_TEMPERATURE = float(os.environ.get("WMC_TEMPERATURE", "0.2"))
DEFAULT_MAX_TOKENS = int(os.environ.get("WMC_MAX_TOKENS", "2048"))


def load_runtime_config() -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load runtime_config.yaml")
    if not RUNTIME_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Runtime config not found: {RUNTIME_CONFIG_PATH}")
    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("runtime_config.yaml must load as a dictionary")
    return data


def load_runtime_models() -> Dict[str, Dict[str, Any]]:
    cfg = load_runtime_config()
    models = cfg.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("runtime_config.yaml models block must be a dictionary")
    return models
