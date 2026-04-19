#!/usr/bin/env python3
"""
Batch-generate `models:` entries in gpu_runtime/runtime_config.yaml for one target GPU.

Rules:
- Read HF repos from a plain text file (one repo per line).
- Estimate fit heuristically using model_static_check helpers.
- Include only models whose fit label is in --allow-fits (default: comfortable).
- Optionally cap max_model_len for target concurrent users (--concurrent-users).
- Rewrite only the `models:` block in the config file.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_static_check import (  # noqa: E402
    TARGET_GPU_VRAM_GB,
    choose_gpu_mem_util,
    classify_fit,
    estimate_kv_cache_mb,
    estimate_weights_mb,
    fetch_config,
    load_env,
    normalize_target_gpu,
)

TRUST_REMOTE_CODE_REPOS = {
    "openGPT-X/Teuken-7B-instruct-commercial-v0.4",
}


def parse_seq_lens(raw: str) -> List[int]:
    seq = sorted({int(x.strip()) for x in raw.split(",") if x.strip().isdigit()})
    if not seq:
        raise ValueError("No valid --seq-lens values provided.")
    return seq


def load_repo_list(path: Path) -> List[str]:
    repos: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if not val or val.startswith("#"):
            continue
        repos.append(val)
    return repos


def sanitize_repo_key(repo_id: str) -> str:
    return repo_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def build_name(repo_id: str, gpu: str) -> str:
    base = repo_id.split("/")[-1].lower().replace("_", "-").replace(".", "-")
    return f"{base}-{gpu.replace('_', '-')}"


def extract_model_dims(cfg: Dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    text_cfg = cfg.get("text_config") or {}
    hidden_size = text_cfg.get("hidden_size", cfg.get("hidden_size"))
    num_layers = text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers"))
    num_heads = text_cfg.get("num_attention_heads", cfg.get("num_attention_heads"))
    num_kv_heads = text_cfg.get("num_key_value_heads", cfg.get("num_key_value_heads", num_heads))
    head_dim = (hidden_size // num_heads) if (hidden_size and num_heads) else None
    return hidden_size, num_layers, num_heads, num_kv_heads, head_dim


def extract_max_position_embeddings(cfg: Dict) -> Optional[int]:
    text_cfg = cfg.get("text_config") or {}
    max_pos = text_cfg.get("max_position_embeddings", cfg.get("max_position_embeddings"))
    if isinstance(max_pos, int) and max_pos > 0:
        return max_pos
    return None


def detect_required_dtype(cfg: Dict) -> str:
    quant_cfg = cfg.get("quantization_config") or {}
    quant_method = str(quant_cfg.get("quant_method", "")).lower()
    if quant_method == "mxfp4":
        return "bfloat16"
    return "float16"


def fetch_checkpoint_total_size_mb(repo_id: str, hf_token: Optional[str] = None) -> Optional[float]:
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    candidates = [
        f"https://huggingface.co/{repo_id}/resolve/main/model.safetensors.index.json",
        f"https://huggingface.co/{repo_id}/resolve/main/pytorch_model.bin.index.json",
    ]

    for url in candidates:
        try:
            resp = requests.get(url, timeout=20, headers=headers)
            if resp.status_code != 200:
                continue
            data = resp.json()
            meta = data.get("metadata") or {}
            total_size_bytes = meta.get("total_size")
            if isinstance(total_size_bytes, int) and total_size_bytes > 0:
                return total_size_bytes / (1024 ** 2)
        except Exception:
            continue
    return None


FIT_ORDER = {
    "comfortable": 0,
    "tight": 1,
    "very tight": 2,
    "unlikely": 3,
}


def best_candidate_summary(
    candidates: List[Tuple[int, int, float, float, str]],
) -> Optional[Tuple[int, int, str]]:
    if not candidates:
        return None
    best = min(
        candidates,
        key=lambda x: (
            FIT_ORDER.get(x[4], 99),
            -x[0],
        ),
    )
    return best[0], best[1], best[4]


def choose_len_by_fit(
    weights_mb: float,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    seq_lens: List[int],
    max_supported_len: Optional[int],
    concurrency_seq_cap: Optional[int],
    target_max_output_tokens: int,
    target_vram_gb: int,
    allow_fits: List[str],
) -> Tuple[Optional[Tuple[int, int, float, float, str]], List[Tuple[int, int, float, float, str]]]:
    candidates: List[Tuple[int, int, float, float, str]] = []
    vram_mb = target_vram_gb * 1024
    candidate_lens = sorted(set(seq_lens))
    if max_supported_len:
        candidate_lens = [s for s in candidate_lens if 0 < s <= max_supported_len]
        if not candidate_lens:
            candidate_lens = [max_supported_len]
    if concurrency_seq_cap:
        candidate_lens = [s for s in candidate_lens if s <= concurrency_seq_cap]
        if not candidate_lens and concurrency_seq_cap > 0:
            fallback = max(256, (concurrency_seq_cap // 256) * 256)
            if max_supported_len:
                fallback = min(fallback, max_supported_len)
            if fallback > 0:
                candidate_lens = [fallback]

    for seq_len in candidate_lens:
        usable_input_tokens = seq_len - target_max_output_tokens
        if usable_input_tokens <= 0:
            continue
        kv_mb = estimate_kv_cache_mb(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            dtype_bytes=dtype_bytes,
        )
        total_mb = weights_mb + kv_mb
        fit = classify_fit(total_mb, target_vram_gb)
        ratio = total_mb / vram_mb
        candidates.append((seq_len, usable_input_tokens, total_mb, ratio, fit))

    allowed = [c for c in candidates if c[4] in allow_fits]
    if not allowed:
        return None, candidates
    return max(allowed, key=lambda x: x[0]), candidates


def estimate_seq_cap_for_concurrency(
    *,
    weights_mb: float,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    target_vram_gb: int,
    concurrent_users: int,
    target_max_output_tokens: int,
) -> Optional[int]:
    if concurrent_users <= 1:
        return None

    vram_mb = target_vram_gb * 1024
    reserve_mb = max(vram_mb * 0.20, 4096.0)
    kv_budget_total_mb = vram_mb - weights_mb - reserve_mb
    if kv_budget_total_mb <= 0:
        return 0

    kv_budget_per_user_mb = kv_budget_total_mb / concurrent_users
    kv_per_token_mb = estimate_kv_cache_mb(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=1,
        dtype_bytes=dtype_bytes,
    )
    if kv_per_token_mb <= 0:
        return None

    raw_cap = int((kv_budget_per_user_mb / kv_per_token_mb) * 0.75)
    if raw_cap <= target_max_output_tokens:
        return 0
    return max(raw_cap, 0)


def render_models_yaml(models: List[Dict[str, str]]) -> str:
    lines = ["models:"]
    for m in models:
        lines.append(f"  {m['key']}:")
        lines.append(f"    name: \"{m['name']}\"")
        lines.append(f"    served_model_name: \"{m['served_model_name']}\"")
        lines.append(f"    repo: \"{m['repo']}\"")
        lines.append(f"    local_path: \"{m['local_path']}\"")
        lines.append("    quant: null")
        lines.append(f"    dtype: \"{m['dtype']}\"")
        lines.append(f"    max_model_len: {m['max_model_len']}")
        lines.append(f"    usable_input_tokens: {m['usable_input_tokens']}")
        lines.append(f"    gpu_memory_util: {m['gpu_memory_util']}")
        if m.get("trust_remote_code") == "true":
            lines.append("    trust_remote_code: true")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def replace_models_block(config_text: str, new_models_block: str) -> str:
    lines = config_text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^models:\s*$", line.strip()):
            start = i
            break
    if start is None:
        raise ValueError("Could not find `models:` block in config file.")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$", lines[j]) and not lines[j].startswith("  "):
            end = j
            break

    before = "".join(lines[:start])
    after = "".join(lines[end:])
    return f"{before}{new_models_block}{after}"


def replace_evaluation_max_tokens(config_text: str, max_tokens: int) -> str:
    lines = config_text.splitlines(keepends=True)
    eval_start = None
    for i, line in enumerate(lines):
        if re.match(r"^evaluation:\s*$", line.strip()):
            eval_start = i
            break
    if eval_start is None:
        raise ValueError("Could not find `evaluation:` block in config file.")

    eval_end = len(lines)
    for j in range(eval_start + 1, len(lines)):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$", lines[j]) and not lines[j].startswith("  "):
            eval_end = j
            break

    replaced = False
    for k in range(eval_start + 1, eval_end):
        if re.match(r"^\s{2}max_tokens:\s*", lines[k]):
            lines[k] = re.sub(r"(:\s*).*$", rf"\g<1>{max_tokens}\n", lines[k], count=1)
            replaced = True
            break

    if not replaced:
        insert_at = eval_start + 1
        lines.insert(insert_at, f"  max_tokens: {max_tokens}\n")

    return "".join(lines)


def render_generation_profile_yaml(
    *,
    gpu: str,
    concurrent_users: int,
    target_max_output_tokens: int,
    seq_lens_raw: str,
    allow_fits_raw: str,
) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "generation_profile:",
        f"  gpu: \"{gpu}\"",
        f"  concurrent_users: {concurrent_users}",
        f"  target_max_output_tokens: {target_max_output_tokens}",
        f"  seq_lens: \"{seq_lens_raw}\"",
        f"  allow_fits: \"{allow_fits_raw}\"",
        f"  generated_at: \"{ts}\"",
        "",
    ]
    return "\n".join(lines)


def replace_generation_profile_block(config_text: str, profile_block: str) -> str:
    lines = config_text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^generation_profile:\s*$", line.strip()):
            start = i
            break

    if start is not None:
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$", lines[j]) and not lines[j].startswith("  "):
                end = j
                break
        before = "".join(lines[:start])
        after = "".join(lines[end:])
        return f"{before}{profile_block}{after}"

    models_start = None
    for i, line in enumerate(lines):
        if re.match(r"^models:\s*$", line.strip()):
            models_start = i
            break
    if models_start is None:
        return config_text.rstrip() + "\n\n" + profile_block

    before = "".join(lines[:models_start])
    after = "".join(lines[models_start:])
    return f"{before}{profile_block}{after}"


def extract_evaluation_max_tokens(config_text: str) -> Optional[int]:
    lines = config_text.splitlines()
    eval_start = None
    for i, line in enumerate(lines):
        if re.match(r"^evaluation:\s*$", line.strip()):
            eval_start = i
            break
    if eval_start is None:
        return None

    for j in range(eval_start + 1, len(lines)):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$", lines[j]) and not lines[j].startswith("  "):
            break
        m = re.match(r"^\s{2}max_tokens:\s*(\d+)\s*$", lines[j])
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GPU-specific SGLang models config block.")
    parser.add_argument("gpu", help="Target GPU: a40|l40|l40s|3090|a100|a100sxm|h100|h200|h200_sxm|b200")
    parser.add_argument("--repos-file", default=str(SCRIPT_DIR / "model_repos.txt"))
    parser.add_argument("--config-file", default=str(SCRIPT_DIR / "runtime_config.yaml"))
    parser.add_argument("--seq-lens", default="4096,8192,16384")
    parser.add_argument("--dtype-bytes", type=int, default=2, help="fp16/bf16=2")
    parser.add_argument("--concurrent-users", type=int, default=1)
    parser.add_argument("--target-max-output-tokens", type=int, default=None)
    parser.add_argument("--allow-fits", default="comfortable")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_gpu = normalize_target_gpu(args.gpu)
    if not target_gpu or target_gpu not in TARGET_GPU_VRAM_GB:
        print("Error: invalid GPU. Use one of: 3090, a40, l40, l40s, a100, a100sxm, h100, h200, h200_sxm, b200")
        return 2

    seq_lens = parse_seq_lens(args.seq_lens)
    if args.concurrent_users < 1:
        print("Error: --concurrent-users must be >= 1")
        return 2
    if args.target_max_output_tokens is not None and args.target_max_output_tokens < 1:
        print("Error: --target-max-output-tokens must be >= 1")
        return 2
    allow_fits = [x.strip().lower() for x in args.allow_fits.split(",") if x.strip()]
    valid_fits = {"comfortable", "tight", "very tight", "unlikely"}
    if any(f not in valid_fits for f in allow_fits):
        print("Error: --allow-fits accepts: comfortable,tight,very tight,unlikely")
        return 2

    repos_file = Path(args.repos_file)
    config_file = Path(args.config_file)
    if not repos_file.exists():
        print(f"Error: repos file not found: {repos_file}")
        return 2
    if not config_file.exists():
        print(f"Error: config file not found: {config_file}")
        return 2

    original = config_file.read_text(encoding="utf-8")
    effective_max_output_tokens = (
        args.target_max_output_tokens
        if args.target_max_output_tokens is not None
        else extract_evaluation_max_tokens(original) or 512
    )

    env = load_env(SCRIPT_DIR / ".env")
    hf_token = env.get("HF_TOKEN")
    target_vram_gb = TARGET_GPU_VRAM_GB[target_gpu]

    repos = load_repo_list(repos_file)
    if not repos:
        print(f"No repos found in {repos_file}")
        return 1

    generated: List[Dict[str, str]] = []
    skipped: List[Tuple[str, str]] = []

    for repo in repos:
        try:
            cfg = fetch_config(repo, hf_token=hf_token)
        except Exception as exc:
            skipped.append((repo, f"fetch failed: {type(exc).__name__}"))
            continue

        hidden_size, num_layers, _, num_kv_heads, head_dim = extract_model_dims(cfg)
        max_supported_len = extract_max_position_embeddings(cfg)
        dtype = detect_required_dtype(cfg)
        text_cfg = cfg.get("text_config") or {}
        weights_mb = fetch_checkpoint_total_size_mb(repo, hf_token=hf_token)
        if weights_mb is None:
            weights_mb = estimate_weights_mb(text_cfg if text_cfg else cfg)

        if not all([hidden_size, num_layers, num_kv_heads, head_dim]) or weights_mb is None:
            skipped.append((repo, "missing dimensions or weight estimate"))
            continue

        concurrency_seq_cap = estimate_seq_cap_for_concurrency(
            weights_mb=weights_mb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=args.dtype_bytes,
            target_vram_gb=target_vram_gb,
            concurrent_users=args.concurrent_users,
            target_max_output_tokens=effective_max_output_tokens,
        )
        if args.concurrent_users > 1 and concurrency_seq_cap == 0:
            skipped.append(
                (
                    repo,
                    (
                        "insufficient KV budget for "
                        f"{args.concurrent_users} concurrent users with "
                        f"target_max_output_tokens={effective_max_output_tokens}"
                    ),
                )
            )
            continue

        chosen, candidates = choose_len_by_fit(
            weights_mb=weights_mb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=args.dtype_bytes,
            seq_lens=seq_lens,
            max_supported_len=max_supported_len,
            concurrency_seq_cap=concurrency_seq_cap,
            target_max_output_tokens=effective_max_output_tokens,
            target_vram_gb=target_vram_gb,
            allow_fits=allow_fits,
        )
        if chosen is None:
            best_candidate = best_candidate_summary(candidates)
            if best_candidate is None:
                reason = (
                    "no valid candidate seq_len remained after applying model/context/concurrency "
                    f"constraints (target_max_output_tokens={effective_max_output_tokens})"
                )
            else:
                best_seq_len, best_usable_input_tokens, best_fit = best_candidate
                reason = (
                    "no candidate seq_len matched requested fit labels "
                    f"[{','.join(allow_fits)}]; "
                    f"best observed fit={best_fit} at seq_len={best_seq_len} "
                    f"(usable_input_tokens={best_usable_input_tokens}, "
                    f"target_max_output_tokens={effective_max_output_tokens})"
                )
            skipped.append(
                (
                    repo,
                    reason,
                )
            )
            continue

        max_model_len, usable_input_tokens, _, ratio, fit = chosen
        model_key_base = sanitize_repo_key(repo)
        top_key = f"{model_key_base}_fp16_{target_gpu}"
        generated.append(
            {
                "key": top_key,
                "name": build_name(repo, target_gpu),
                "served_model_name": build_name(repo, target_gpu),
                "repo": repo,
                "local_path": f"/workspace/models/{top_key}",
                "dtype": dtype,
                "max_model_len": str(max_model_len),
                "usable_input_tokens": str(usable_input_tokens),
                "gpu_memory_util": f"{choose_gpu_mem_util(ratio):.2f}",
                "fit": fit,
                "trust_remote_code": "true" if repo in TRUST_REMOTE_CODE_REPOS else "false",
            }
        )

    generated.sort(key=lambda x: x["key"])
    new_models_block = render_models_yaml(generated)
    generation_profile_block = render_generation_profile_yaml(
        gpu=target_gpu,
        concurrent_users=args.concurrent_users,
        target_max_output_tokens=effective_max_output_tokens,
        seq_lens_raw=args.seq_lens,
        allow_fits_raw=args.allow_fits,
    )

    if args.dry_run:
        print(generation_profile_block)
        print(new_models_block)
        if args.target_max_output_tokens is not None:
            print(f"\nWould set evaluation.max_tokens: {args.target_max_output_tokens}")
    else:
        if not generated and not args.allow_empty:
            print("Refusing to overwrite config with empty models block. Use --allow-empty to force.")
            return 1
        updated = replace_models_block(original, new_models_block)
        if args.target_max_output_tokens is not None:
            updated = replace_evaluation_max_tokens(updated, args.target_max_output_tokens)
        updated = replace_generation_profile_block(updated, generation_profile_block)
        config_file.write_text(updated, encoding="utf-8")
        print(f"Updated models and generation_profile blocks in: {config_file}")
        if args.target_max_output_tokens is not None:
            print(f"Updated evaluation.max_tokens in: {config_file}")

    print(
        f"\nTarget GPU: {target_gpu} ({target_vram_gb}GB) | "
        f"Included: {len(generated)} | Skipped: {len(skipped)}"
    )
    print(f"Target max output tokens for sizing: {effective_max_output_tokens}")
    if generated:
        print("Included models:")
        for m in generated:
            print(
                "  - "
                f"{m['repo']} | max_model_len={m['max_model_len']} | "
                f"usable_input_tokens={m['usable_input_tokens']} | fit={m['fit']}"
            )
    if skipped:
        print("Skipped models:")
        for repo, reason in skipped:
            print(f"  - {repo} | {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
