"""
Superseded: the reviewer-fitted KO quality score (September 2026).

This generation fitted the quality score to the 2024 human review (ridge regression on
content length, MTLD and an LLM judge's verdict) and kept the structural / semantic /
functional pillar modules only as diagnostics. It was replaced because the project
decided not to use reviewer ratings as the yardstick - they are subjective and
error-prone - and to score METADATA against the file on literature-grounded,
reproducible criteria instead (see ../../pipeline.py, METHODS.md, LITERATURE_REVIEW.md).

Kept for reference and reproducibility of the earlier results; nothing in the active
pipeline imports it. Its fitted artifacts (score_model.json, diagnostic_thresholds.json)
are here too but are ignored by git, as all *.json outside the allowlist are; the
committed versions remain in the repository history.
"""
