#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

import requests

try:
    from dotenv import dotenv_values
except Exception:  # pragma: no cover - optional dependency
    dotenv_values = None


TARGET_GPU_VRAM_GB: Dict[str, int] = {
    "3090": 24,
    "a40": 48,
    "l40": 48,
    "l40s": 48,
    "a100": 80,
    "a100sxm": 80,
    "h100": 80,
    "h200": 141,
    "h200_sxm": 141,
    "b200": 180,
}


def normalize_target_gpu(name: str) -> Optional[str]:
    val = name.strip().lower().replace("-", "").replace(" ", "").replace("_", "")
    aliases = {
        "a40": "a40",
        "3090": "3090",
        "l40": "l40",
        "l40s": "l40s",
        "a100": "a100",
        "a100sxm": "a100sxm",
        "h100": "h100",
        "h200": "h200",
        "h200sxm": "h200_sxm",
        "b200": "b200",
    }
    return aliases.get(val)


def load_env(env_path: Optional[Path] = None) -> Dict[str, str]:
    if env_path is None:
        env_path = Path(__file__).with_name(".env")
    merged: Dict[str, str] = dict(os.environ)
    if dotenv_values is not None and env_path.exists():
        for key, value in dotenv_values(env_path).items():
            if value is not None:
                merged.setdefault(key, value)
    return merged


def fetch_config(repo_id: str, hf_token: Optional[str] = None) -> Dict:
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    resp = requests.get(url, timeout=20, headers=headers)
    resp.raise_for_status()
    return resp.json()


def _infer_params_from_config(cfg: Dict) -> Optional[float]:
    text_cfg = cfg.get("text_config") or {}
    hidden = text_cfg.get("hidden_size", cfg.get("hidden_size"))
    layers = text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers"))
    vocab = text_cfg.get("vocab_size", cfg.get("vocab_size"))
    intermediate = text_cfg.get("intermediate_size", cfg.get("intermediate_size"))
    if not all(isinstance(x, int) and x > 0 for x in (hidden, layers)):
        return None

    vocab = vocab if isinstance(vocab, int) and vocab > 0 else 32000
    intermediate = intermediate if isinstance(intermediate, int) and intermediate > 0 else (hidden * 4)

    attn_proj = 4 * hidden * hidden
    mlp_proj = 3 * hidden * intermediate
    layer_norm = 8 * hidden
    per_layer = attn_proj + mlp_proj + layer_norm
    embedding = vocab * hidden
    final_norm = hidden
    return float((per_layer * layers) + embedding + final_norm)


def estimate_weights_mb(cfg: Dict, dtype_bytes: int = 2) -> Optional[float]:
    for key in ("num_parameters", "n_parameters", "parameter_count"):
        val = cfg.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return float(val) * dtype_bytes / (1024 ** 2)

    inferred = _infer_params_from_config(cfg)
    if inferred is None:
        return None
    # Add conservative 8% overhead for metadata / underestimated architectures.
    return (inferred * dtype_bytes / (1024 ** 2)) * 1.08


def estimate_kv_cache_mb(
    *,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int,
) -> float:
    per_token_per_layer = 2 * num_kv_heads * head_dim * dtype_bytes
    total_bytes = per_token_per_layer * seq_len * num_layers
    return total_bytes / (1024 ** 2)


def classify_fit(total_mb: float, target_vram_gb: int) -> str:
    ratio = total_mb / (target_vram_gb * 1024)
    if ratio <= 0.75:
        return "comfortable"
    if ratio <= 0.90:
        return "tight"
    if ratio <= 1.0:
        return "very tight"
    return "unlikely"


def choose_gpu_mem_util(ratio: float) -> float:
    if ratio <= 0.70:
        return 0.82
    if ratio <= 0.80:
        return 0.86
    if ratio <= 0.90:
        return 0.90
    return 0.92
