# Validation run artifacts

Evidence behind the choices in `stages/scoring.py` and `stages/gate.py`. These are
inputs to decisions, not pipeline output — pipeline output goes to `output/`.

| File | What it is |
|---|---|
| `human_reviewed_92_kos.json` | The 92 KOs covered by both the 2024 review and the June-2026 export. The sample every correlation in the README was computed on. |
| `panel_<model>.json` | The same 92 KOs judged by four model families against the review rubric. The evidence for choosing glm-5.2 and for using one judge rather than a panel. |
| `judge_validation.json` | Per-question correlations of judge answers against human answers. |
| `human_vs_auto_3rater.xlsx` | The human-vs-automatic comparison including `collating` as a third rater, as a robustness check. |

Regenerate the correlations with `python -m validation.validate_metrics`.
