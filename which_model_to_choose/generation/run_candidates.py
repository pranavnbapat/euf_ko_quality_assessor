from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from .config import load_runtime_models
from .io_helpers import candidate_run_dir, find_latest_json, load_json, write_json
from .metadata_pipeline import generate_metadata_dataset
from .summary_pipeline import summarize_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run candidate generation for model selection.")
    p.add_argument("--task", choices=["summary", "metadata"], required=True, help="Generation task to run.")
    p.add_argument("--input", type=str, default=None, help="Explicit input JSON path. Defaults to latest root input JSON.")
    p.add_argument("--run-id", type=str, default=None, help="Optional run id. Defaults to timestamp.")
    p.add_argument("--model", action="append", dest="models", default=None, help="Runtime config model key to run. Can be repeated.")
    p.add_argument("--all-models", action="store_true", help="Run all models from runtime config.")
    p.add_argument("--summary-run-id", type=str, default=None, help="Required for metadata task: run id containing summary artifacts.")
    return p.parse_args()


def select_models(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    runtime_models = load_runtime_models()
    if args.all_models:
        return runtime_models
    if args.models:
        missing = [m for m in args.models if m not in runtime_models]
        if missing:
            raise KeyError(f"Unknown runtime model keys: {', '.join(missing)}")
        return {k: runtime_models[k] for k in args.models}
    # default: first runtime model only
    first_key = next(iter(runtime_models))
    return {first_key: runtime_models[first_key]}


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    input_path = Path(args.input) if args.input else find_latest_json()
    data = load_json(input_path)
    models = select_models(args)

    run_dir = candidate_run_dir(run_id)
    manifest = {
        "run_id": run_id,
        "task": args.task,
        "input_file": str(input_path),
        "models": list(models.keys()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "manifest.json", manifest)

    for model_key, model_cfg in models.items():
        model_dir = run_dir / model_key
        if args.task == "summary":
            rows = summarize_dataset(data, model_key)
            payload = {
                "task": "summary",
                "model_key": model_key,
                "model_name": model_cfg.get("name"),
                "model_repo": model_cfg.get("repo"),
                "input_file": str(input_path),
                "rows": rows,
            }
            write_json(model_dir / "summaries.json", payload)
            print(f"[DONE] wrote {model_dir / 'summaries.json'}")
        elif args.task == "metadata":
            if not args.summary_run_id:
                raise ValueError("--summary-run-id is required for metadata generation")
            summary_path = candidate_run_dir(args.summary_run_id) / model_key / "summaries.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Summary artifact not found for model {model_key}: {summary_path}")
            summary_payload = load_json(summary_path)
            summary_rows = summary_payload.get("rows") or []
            rows = generate_metadata_dataset(data, model_key, summary_rows)
            payload = {
                "task": "metadata",
                "model_key": model_key,
                "model_name": model_cfg.get("name"),
                "model_repo": model_cfg.get("repo"),
                "input_file": str(input_path),
                "summary_run_id": args.summary_run_id,
                "rows": rows,
            }
            write_json(model_dir / "metadata.json", payload)
            print(f"[DONE] wrote {model_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
