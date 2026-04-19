from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List

from .config import load_runtime_models
from .io_helpers import (
    candidate_run_dir,
    ensure_dir,
    find_latest_json,
    load_json,
    normalize_records,
    output_run_dir,
    record_id,
    sample_records,
    write_json,
)
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
    p.add_argument("--sample-size", type=int, default=None, help="Randomly sample this many records from the latest/input JSON.")
    p.add_argument("--sample-seed", type=int, default=42, help="Random seed used when --sample-size is set.")
    p.add_argument("--export-format", choices=["json", "csv", "both"], default="json", help="Write consolidated outputs to which_model_to_choose/output.")
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


def _suffix(model_key: str) -> str:
    return model_key.replace("-", "_").replace(".", "_")


def _source_row(record: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "record_index": idx,
        "id": record_id(record, idx),
        "title": record.get("title"),
        "subtitle": record.get("subtitle"),
        "description": record.get("description"),
        "keywords": record.get("keywords"),
        "title_llm": record.get("title_llm"),
        "subtitle_llm": record.get("subtitle_llm"),
        "description_llm": record.get("description_llm"),
        "keywords_llm": record.get("keywords_llm"),
        "ko_content_flat": record.get("ko_content_flat"),
    }


def _rows_to_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def build_summary_export(records: List[Dict[str, Any]], model_outputs: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    per_model = {model_key: _rows_to_map(rows) for model_key, rows in model_outputs.items()}
    payload: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        row = _source_row(record, idx)
        rec_id = row["id"]
        for model_key, model_rows in per_model.items():
            suffix = _suffix(model_key)
            result = model_rows.get(rec_id, {})
            row[f"summary_{suffix}"] = result.get("summary")
            row[f"status_{suffix}"] = result.get("status")
            row[f"error_{suffix}"] = result.get("error")
        payload.append(row)
    return payload


def build_metadata_export(records: List[Dict[str, Any]], model_outputs: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    per_model = {model_key: _rows_to_map(rows) for model_key, rows in model_outputs.items()}
    payload: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        row = _source_row(record, idx)
        rec_id = row["id"]
        for model_key, model_rows in per_model.items():
            suffix = _suffix(model_key)
            result = model_rows.get(rec_id, {})
            row[f"title_{suffix}"] = result.get("title")
            row[f"subtitle_{suffix}"] = result.get("subtitle")
            row[f"description_{suffix}"] = result.get("description")
            row[f"keywords_{suffix}"] = result.get("keywords")
            row[f"status_{suffix}"] = result.get("status")
            row[f"error_{suffix}"] = result.get("error")
        payload.append(row)
    return payload


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat_row: Dict[str, Any] = {}
            for key, value in row.items():
                if isinstance(value, list):
                    flat_row[key] = " | ".join(str(x) for x in value)
                else:
                    flat_row[key] = value
            writer.writerow(flat_row)


def merge_export_rows(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in existing + new_rows:
        rec_id = str(row.get("id"))
        if rec_id not in merged:
            merged[rec_id] = {}
            order.append(rec_id)
        merged[rec_id].update(row)
    return [merged[rec_id] for rec_id in order]


def write_consolidated_outputs(run_id: str, task: str, rows: List[Dict[str, Any]], export_format: str) -> None:
    out_dir = output_run_dir(run_id)
    json_path = out_dir / f"{task}_candidates.json"
    csv_path = out_dir / f"{task}_candidates.csv"
    existing_rows: List[Dict[str, Any]] = []
    if json_path.exists():
        data = load_json(json_path)
        if isinstance(data, list):
            existing_rows = [row for row in data if isinstance(row, dict)]
    merged_rows = merge_export_rows(existing_rows, rows)
    if export_format in {"json", "both"}:
        write_json(json_path, merged_rows)
        print(f"[DONE] wrote {json_path}")
    if export_format in {"csv", "both"}:
        write_csv(csv_path, merged_rows)
        print(f"[DONE] wrote {csv_path}")


def filter_records_by_ids(records: List[Dict[str, Any]], wanted_ids: List[str]) -> List[Dict[str, Any]]:
    wanted = set(wanted_ids)
    filtered: List[Dict[str, Any]] = []
    for idx, record in enumerate(records, 1):
        if record_id(record, idx) in wanted:
            filtered.append(record)
    return filtered


def main() -> None:
    args = parse_args()
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    input_path = Path(args.input) if args.input else find_latest_json()
    data = load_json(input_path)
    records = normalize_records(data)
    models = select_models(args)

    if args.task == "metadata" and args.summary_run_id and args.sample_size is None:
        first_model_key = next(iter(models))
        summary_path = candidate_run_dir(args.summary_run_id) / first_model_key / "summaries.json"
        if summary_path.exists():
            summary_payload = load_json(summary_path)
            summary_rows = summary_payload.get("rows") or []
            sampled_records = filter_records_by_ids(
                records,
                [str(row.get("id")) for row in summary_rows if row.get("id") is not None],
            )
        else:
            sampled_records = records
    else:
        sampled_records = sample_records(records, args.sample_size, args.sample_seed)

    run_dir = candidate_run_dir(run_id)
    manifest = {
        "run_id": run_id,
        "task": args.task,
        "input_file": str(input_path),
        "models": list(models.keys()),
        "sample_size_requested": args.sample_size,
        "sample_size_actual": len(sampled_records),
        "sample_seed": args.sample_seed,
        "sample_record_ids": [record_id(record, idx) for idx, record in enumerate(sampled_records, 1)],
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_dir / "manifest.json", manifest)

    model_outputs: Dict[str, List[Dict[str, Any]]] = {}
    for model_key, model_cfg in models.items():
        model_dir = run_dir / model_key
        served_model_name = (
            model_cfg.get("served_model_name")
            or model_cfg.get("name")
            or model_cfg.get("repo")
            or model_key
        )
        if args.task == "summary":
            rows = summarize_dataset(sampled_records, model_key, served_model_name)
            payload = {
                "task": "summary",
                "model_key": model_key,
                "model_name": model_cfg.get("name"),
                "served_model_name": served_model_name,
                "model_repo": model_cfg.get("repo"),
                "input_file": str(input_path),
                "sample_size": len(sampled_records),
                "rows": rows,
            }
            write_json(model_dir / "summaries.json", payload)
            model_outputs[model_key] = rows
            print(f"[DONE] wrote {model_dir / 'summaries.json'}")
        elif args.task == "metadata":
            if not args.summary_run_id:
                raise ValueError("--summary-run-id is required for metadata generation")
            summary_path = candidate_run_dir(args.summary_run_id) / model_key / "summaries.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Summary artifact not found for model {model_key}: {summary_path}")
            summary_payload = load_json(summary_path)
            summary_rows = summary_payload.get("rows") or []
            rows = generate_metadata_dataset(sampled_records, model_key, served_model_name, summary_rows)
            payload = {
                "task": "metadata",
                "model_key": model_key,
                "model_name": model_cfg.get("name"),
                "served_model_name": served_model_name,
                "model_repo": model_cfg.get("repo"),
                "input_file": str(input_path),
                "summary_run_id": args.summary_run_id,
                "sample_size": len(sampled_records),
                "rows": rows,
            }
            write_json(model_dir / "metadata.json", payload)
            model_outputs[model_key] = rows
            print(f"[DONE] wrote {model_dir / 'metadata.json'}")

    if args.task == "summary":
        export_rows = build_summary_export(sampled_records, model_outputs)
    else:
        export_rows = build_metadata_export(sampled_records, model_outputs)
    write_consolidated_outputs(run_id, args.task, export_rows, args.export_format)


if __name__ == "__main__":
    main()
