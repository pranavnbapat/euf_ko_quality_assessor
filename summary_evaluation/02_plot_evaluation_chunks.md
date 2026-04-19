# 02 Plot Evaluation Chunks

## Purpose

`02_plot_evaluation_chunks.py` visualises the semantic-similarity output from
`02_evaluate_chunks.py`.

## Input

Expected input:

- `summary_evaluation/output/02_evaluate_chunks.json`

## What It Produces

Interactive matplotlib plots showing:

- mean semantic scores by candidate/model
- summary length vs similarity
- compression ratio vs semantic score

## Why It Is Useful

This makes it easier to answer:

- which candidate is strongest on average?
- does stronger compression hurt semantic quality?
- are some models producing short but still semantically strong summaries?

## Interpretation

Look mainly at:

- `bertscore_f1`
- cosine similarity
- length/compression together

High similarity with extreme length reduction is generally desirable.

## Limits

- it does not add new evidence beyond `02`
- averages can hide bad outliers
