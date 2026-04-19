from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from ..generation.io_helpers import candidate_run_dir, find_latest_json, load_json, write_json
from ..generation.summary_pipeline import iter_records
from .metrics import score_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate metadata candidate artifacts.")
    p.add_argument("--run-id", required=True, help="Candidate run id containing metadata artifacts.")
    p.add_argument("--input", type=str, default=None, help="Explicit source input JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else find_latest_json()
    data = load_json(input_path)
    records = list(iter_records(data))
    rec_by_id = {
        (r.get("_orig_id") or r.get("@id") or r.get("id") or f"row_{i+1}"): r
        for i, r in enumerate(records)
    }

    run_dir = candidate_run_dir(args.run_id)
    model_dirs = [p for p in run_dir.iterdir() if p.is_dir()]
    output_root = run_dir / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: List[Dict[str, Any]] = []

    for model_dir in model_dirs:
        metadata_file = model_dir / "metadata.json"
        summary_file = model_dir / "summaries.json"
        if not metadata_file.exists() or not summary_file.exists():
            continue
        metadata_payload = load_json(metadata_file)
        summary_payload = load_json(summary_file)
        summary_by_id = {
            row.get("id"): row for row in (summary_payload.get("rows") or [])
            if row.get("status") == "ok"
        }

        rows = metadata_payload.get("rows") or []
        scored_rows: List[Dict[str, Any]] = []
        scores: List[float] = []
        ok_count = 0
        for row in rows:
            rec_id = row.get("id")
            record = rec_by_id.get(rec_id)
            summary_row = summary_by_id.get(rec_id)
            if not record or not summary_row:
                continue
            if row.get("status") == "ok":
                md = {
                    "title": row.get("title", ""),
                    "subtitle": row.get("subtitle", ""),
                    "description": row.get("description", ""),
                    "keywords": row.get("keywords", []),
                }
                metrics = score_metadata(record, str(summary_row.get("summary") or ""), md)
                ok_count += 1
                scores.append(float(metrics["final_score"]))
            else:
                metrics = {
                    "final_score": 0.0,
                    "title_score": 0.0,
                    "subtitle_score": 0.0,
                    "description_score": 0.0,
                    "keywords_score": 0.0,
                    "summary_overlap": 0.0,
                    "title_consistency": 0.0,
                    "keywords_consistency": 0.0,
                    "reason": "generation_error",
                }
            scored_rows.append({**row, **metrics})

        model_summary = {
            "run_id": args.run_id,
            "model_key": metadata_payload.get("model_key"),
            "model_name": metadata_payload.get("model_name"),
            "model_repo": metadata_payload.get("model_repo"),
            "input_file": str(input_path),
            "rows": scored_rows,
            "aggregate": {
                "records_total": len(rows),
                "records_ok": ok_count,
                "success_rate": round(ok_count / len(rows), 4) if rows else 0.0,
                "mean_final_score": round(statistics.mean(scores), 4) if scores else 0.0,
                "median_final_score": round(statistics.median(scores), 4) if scores else 0.0,
            },
        }
        write_json(output_root / f"{model_dir.name}_metadata_eval.json", model_summary)
        aggregate_rows.append({
            "model_key": metadata_payload.get("model_key"),
            "model_name": metadata_payload.get("model_name"),
            "model_repo": metadata_payload.get("model_repo"),
            **model_summary["aggregate"],
        })

    aggregate_rows.sort(key=lambda x: (x["mean_final_score"], x["success_rate"]), reverse=True)
    write_json(output_root / "metadata_model_ranking.json", {
        "run_id": args.run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ranking": aggregate_rows,
    })


if __name__ == "__main__":
    main()
