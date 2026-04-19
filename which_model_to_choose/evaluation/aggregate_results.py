from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

from ..generation.io_helpers import candidate_run_dir, load_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate summary and metadata evaluation rankings.")
    p.add_argument("--run-id", required=True, help="Candidate run id.")
    p.add_argument("--summary-weight", type=float, default=0.6)
    p.add_argument("--metadata-weight", type=float, default=0.4)
    return p.parse_args()


def _ranking_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {item["model_key"]: item for item in items if item.get("model_key")}


def main() -> None:
    args = parse_args()
    run_dir = candidate_run_dir(args.run_id)
    eval_dir = run_dir / "evaluation"
    summary = load_json(eval_dir / "summary_model_ranking.json") if (eval_dir / "summary_model_ranking.json").exists() else {"ranking": []}
    metadata = load_json(eval_dir / "metadata_model_ranking.json") if (eval_dir / "metadata_model_ranking.json").exists() else {"ranking": []}

    summary_map = _ranking_map(summary.get("ranking") or [])
    metadata_map = _ranking_map(metadata.get("ranking") or [])
    keys = sorted(set(summary_map) | set(metadata_map))

    rows: List[Dict[str, Any]] = []
    for key in keys:
        s = summary_map.get(key, {})
        m = metadata_map.get(key, {})
        s_score = float(s.get("mean_final_score", 0.0) or 0.0)
        m_score = float(m.get("mean_final_score", 0.0) or 0.0)
        final = args.summary_weight * s_score + args.metadata_weight * m_score
        rows.append({
            "model_key": key,
            "model_name": s.get("model_name") or m.get("model_name"),
            "model_repo": s.get("model_repo") or m.get("model_repo"),
            "summary_mean_score": round(s_score, 4),
            "metadata_mean_score": round(m_score, 4),
            "combined_score": round(final, 4),
            "summary_success_rate": s.get("success_rate", 0.0),
            "metadata_success_rate": m.get("success_rate", 0.0),
        })

    rows.sort(key=lambda x: (x["combined_score"], x["summary_mean_score"], x["metadata_mean_score"]), reverse=True)
    payload = {
        "run_id": args.run_id,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": {
            "summary": args.summary_weight,
            "metadata": args.metadata_weight,
        },
        "ranking": rows,
        "best_summary_model": (summary.get("ranking") or [{}])[0] if (summary.get("ranking") or []) else None,
        "best_metadata_model": (metadata.get("ranking") or [{}])[0] if (metadata.get("ranking") or []) else None,
        "best_overall_model": rows[0] if rows else None,
    }
    write_json(eval_dir / "combined_model_ranking.json", payload)


if __name__ == "__main__":
    main()
