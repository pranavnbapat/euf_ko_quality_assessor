from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from ..generation.io_helpers import candidate_run_dir, find_latest_json, load_json, write_json
from ..generation.summary_pipeline import iter_records
from .metrics import score_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate summary candidate artifacts.")
    p.add_argument("--run-id", required=True, help="Candidate run id containing summary artifacts.")
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
        summary_file = model_dir / "summaries.json"
        if not summary_file.exists():
            continue
        payload = load_json(summary_file)
        rows = payload.get("rows") or []
        scored_rows: List[Dict[str, Any]] = []
        scores: List[float] = []
        ok_count = 0
        for row in rows:
            rec_id = row.get("id")
            record = rec_by_id.get(rec_id)
            if not record:
                continue
            source = str(record.get("ko_content_flat") or "").strip()
            summary = str(row.get("summary") or "").strip()
            metrics = score_summary(source, summary) if row.get("status") == "ok" else {
                "final_score": 0.0,
                "coverage": 0.0,
                "length_score": 0.0,
                "repetition": 0.0,
                "num_penalty": 0.0,
                "reason": "generation_error",
            }
            if row.get("status") == "ok":
                ok_count += 1
                scores.append(float(metrics["final_score"]))
            scored_rows.append({**row, **metrics})

        model_summary = {
            "run_id": args.run_id,
            "model_key": payload.get("model_key"),
            "model_name": payload.get("model_name"),
            "model_repo": payload.get("model_repo"),
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
        write_json(output_root / f"{model_dir.name}_summary_eval.json", model_summary)
        aggregate_rows.append({
            "model_key": payload.get("model_key"),
            "model_name": payload.get("model_name"),
            "model_repo": payload.get("model_repo"),
            **model_summary["aggregate"],
        })

    aggregate_rows.sort(key=lambda x: (x["mean_final_score"], x["success_rate"]), reverse=True)
    write_json(output_root / "summary_model_ranking.json", {
        "run_id": args.run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ranking": aggregate_rows,
    })


if __name__ == "__main__":
    main()
