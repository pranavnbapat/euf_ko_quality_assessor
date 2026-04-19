from __future__ import annotations

from ..generation.io_helpers import candidate_run_dir, load_json


def build_markdown_report(run_id: str) -> str:
    eval_dir = candidate_run_dir(run_id) / "evaluation"
    combined = load_json(eval_dir / "combined_model_ranking.json")
    lines = [
        f"# Model Selection Report: {run_id}",
        "",
        "## Best Overall Model",
        "",
        f"`{(combined.get('best_overall_model') or {}).get('model_key', 'n/a')}`",
        "",
        "## Best Summary Model",
        "",
        f"`{(combined.get('best_summary_model') or {}).get('model_key', 'n/a')}`",
        "",
        "## Best Metadata Model",
        "",
        f"`{(combined.get('best_metadata_model') or {}).get('model_key', 'n/a')}`",
        "",
    ]
    return "\n".join(lines)
