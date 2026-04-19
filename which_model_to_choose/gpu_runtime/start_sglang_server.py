#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyyaml is required to read runtime_config.yaml") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = SCRIPT_DIR / "runtime_config.yaml"
ENV_PATH = SCRIPT_DIR / ".env"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_runtime_config() -> Dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Runtime config not found: {RUNTIME_CONFIG_PATH}")
    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("runtime_config.yaml must load as a dictionary")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start an SGLang server for a configured model key.")
    p.add_argument("--model-key", required=True, help="Model key from runtime_config.yaml")
    p.add_argument("--host", default=os.environ.get("SGLANG_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("SGLANG_PORT", "8000")))
    p.add_argument("--api-key", default=os.environ.get("SGLANG_API_KEY"))
    p.add_argument("--python", default=sys.executable, help="Python executable to use")
    p.add_argument("--dry-run", action="store_true", help="Print the launch command without executing it")
    return p.parse_args()


def build_command(model_cfg: Dict[str, Any], host: str, port: int, api_key: str | None, python_bin: str) -> list[str]:
    local_path = str(model_cfg.get("local_path") or "").strip()
    repo = str(model_cfg.get("repo") or "").strip()
    model_path = local_path if local_path and Path(local_path).exists() else repo
    if not model_path:
        raise ValueError("Model config is missing both local_path and repo")

    served_model_name = (
        str(model_cfg.get("served_model_name") or "").strip()
        or str(model_cfg.get("name") or "").strip()
        or repo
    )
    if not served_model_name:
        raise ValueError("Model config is missing served_model_name/name/repo")

    cmd = [
        python_bin,
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--served-model-name",
        served_model_name,
        "--host",
        host,
        "--port",
        str(port),
    ]

    context_length = model_cfg.get("max_model_len")
    if context_length:
        cmd.extend(["--context-length", str(context_length)])

    mem_fraction_static = model_cfg.get("gpu_memory_util")
    if mem_fraction_static:
        cmd.extend(["--mem-fraction-static", str(mem_fraction_static)])

    if model_cfg.get("trust_remote_code"):
        cmd.append("--trust-remote-code")

    if api_key:
        cmd.extend(["--api-key", api_key])

    return cmd


def main() -> int:
    load_dotenv(ENV_PATH)
    args = parse_args()
    cfg = load_runtime_config()
    models = cfg.get("models") or {}
    if args.model_key not in models:
        available = ", ".join(sorted(models.keys()))
        raise KeyError(f"Unknown model key '{args.model_key}'. Available: {available}")

    cmd = build_command(models[args.model_key], args.host, args.port, args.api_key, args.python)
    print("Launch command:")
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
