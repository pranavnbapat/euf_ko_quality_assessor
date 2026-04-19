from __future__ import annotations

import argparse

from ..generation.io_helpers import candidate_run_dir, load_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Print best models from aggregate evaluation outputs.")
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = candidate_run_dir(args.run_id) / "evaluation"
    combined = load_json(eval_dir / "combined_model_ranking.json")
    print("Best summary model:")
    print(combined.get("best_summary_model"))
    print("\nBest metadata model:")
    print(combined.get("best_metadata_model"))
    print("\nBest overall model:")
    print(combined.get("best_overall_model"))


if __name__ == "__main__":
    main()
