# 03 Plot NLI Results

## Purpose

`03_plot_nli_results.py` visualises the output of `03_evaluate_chunks.py`.

## Input

Expected input:

- `summary_evaluation/output/03_evaluate_chunks.json`

## Plots

### Mean entailment gap by candidate

Useful for:

- ranking candidates at a high level

### Coverage vs reverse-direction entailment scatter

Useful for:

- spotting candidates that are well-supported by the source
- comparing models visually

### Mean contradiction by candidate

Useful for:

- identifying candidates that look less safe

## Interpretation

This plotter helps compare candidates, but it inherits all the limitations of
`03`:

- chunk-level approximation
- not human ground truth

Treat it as a candidate-comparison tool, not a final truth source.
