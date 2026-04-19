#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

try:
    from huggingface_hub import snapshot_download
except Exception as exc:  # pragma: no cover
    raise RuntimeError("huggingface_hub is required to download runtime models") from exc

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pyyaml is required to read runtime_config.yaml") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
RUNTIME_CONFIG_PATH = SCRIPT_DIR / "runtime_config.yaml"
ENV_PATH = SCRIPT_DIR / ".env"


def load_dotenv(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def load_runtime_config() -> Dict[str, Any]:
    if not RUNTIME_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Runtime config not found: {RUNTIME_CONFIG_PATH}")
    with RUNTIME_CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("runtime_config.yaml must load as a dictionary")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download model repos from runtime_config.yaml into their local_path.")
    p.add_argument("--model-key", action="append", default=None, help="Restrict download to one or more model keys.")
    p.add_argument("--force", action="store_true", help="Download even if the local_path already exists and is non-empty.")
    return p.parse_args()


def select_models(cfg: Dict[str, Any], wanted: list[str] | None) -> Dict[str, Dict[str, Any]]:
    models = cfg.get("models") or {}
    if not isinstance(models, dict):
        raise ValueError("runtime_config.yaml models block must be a dictionary")
    if not wanted:
        return models
    missing = [key for key in wanted if key not in models]
    if missing:
        raise KeyError(f"Unknown model keys: {', '.join(missing)}")
    return {key: models[key] for key in wanted}


def should_skip(path: Path, force: bool) -> bool:
    if force:
        return False
    return path.exists() and any(path.iterdir())


def main() -> int:
    args = parse_args()
    cfg = load_runtime_config()
    env = load_dotenv(ENV_PATH)
    hf_token = env.get("HF_TOKEN") or None
    models = select_models(cfg, args.model_key)

    for model_key, model_cfg in models.items():
        repo = str(model_cfg.get("repo") or "").strip()
        local_path = Path(str(model_cfg.get("local_path") or "")).expanduser()
        if not repo:
            raise ValueError(f"Model {model_key} is missing repo")
        if not local_path:
            raise ValueError(f"Model {model_key} is missing local_path")

        local_path.mkdir(parents=True, exist_ok=True)
        if should_skip(local_path, args.force):
            print(f"[SKIP] {model_key} already present at {local_path}")
            continue

        print(f"[DOWNLOAD] {model_key} <- {repo}")
        snapshot_download(
            repo_id=repo,
            local_dir=str(local_path),
            token=hf_token,
            resume_download=True,
        )
        print(f"[DONE] {model_key} -> {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
