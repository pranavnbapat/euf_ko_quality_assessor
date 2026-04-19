# 01 Plot Evaluation

## Purpose

`01_plot_evaluation.py` visualises the output of `01_evaluate_chunks.py`.

It turns row-level intrinsic metrics into:

- manager-friendly summary plots
- engineer-oriented failure-mode plots
- ops-style outlier plots

## Input

Expected input:

- `summary_evaluation/output/01_evaluate_selected.json`

Optional second input:

- another `01` JSON file for before/after dumbbell comparisons

## What It Produces

PNG chart packs under the output directory you provide, split into:

- `managers/`
- `engineers/`
- `ops/`

## Main Plot Types

### Manager boxplots

Shows metric distributions by field/variant.

Useful for:

- comparing original vs LLM fields at a high level
- seeing whether LLM variants are shorter or less repetitive

### Completeness chart

Shows empty-rate by field variant.

Useful for:

- checking whether generated fields are missing too often

### Engineer scatter plots

Shows relationships like:

- length vs TTR
- length vs repetition

Useful for:

- spotting clusters of bad records
- identifying field-specific failure modes

### Density plots

Shows whether generated content normalises the distribution of text lengths.

Useful for:

- checking whether summaries are consistently smaller and more stable

### Correlation heatmaps

Shows which metrics are redundant.

Useful for:

- simplifying future metric sets

### Ops outlier plots

Shows longest and most repetitive content fields.

Useful for:

- triaging problematic records manually

## Interpretation

This script does not create new evaluation logic.

It is only a visual layer on top of `01_evaluate_chunks.py`.

So its usefulness depends entirely on whether `01` is the right method for your
question.

## Limits

- plots can look authoritative even when the underlying metric is heuristic
- readability and stopword-based plots remain English-biased
- outlier plots can be dominated by a few pathological records
