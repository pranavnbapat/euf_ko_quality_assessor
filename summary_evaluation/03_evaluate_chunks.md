# 03 Evaluate Chunks

## Purpose

`03_evaluate_chunks.py` performs a chunk-aware bidirectional NLI evaluation.

It is a stronger method than `02` when the question becomes:

- Is the source text supporting the candidate summary?
- Does the candidate roughly cover the source content?

## Core Idea

The script uses an MNLI model:

- `roberta-large-mnli`

It computes two directions:

- `L -> S`
  Source supports summary chunk-by-chunk
- `S -> L`
  Reverse-direction overlap/compression proxy

## Why Chunking Was Added

Long KO content does not fit in a single 512-token transformer window.

So this script now:

- splits source and candidate into token chunks
- scores chunk pairs
- selects the best-matching source chunk for each hypothesis chunk
- averages the selected scores

This is much better than a single first-window pass.

## Metrics

### `L_to_S_entail`

How strongly the source supports the candidate.

Interpretation:

- higher is better
- this is the most useful score in the file

### `L_to_S_contradiction`

How strongly the source appears to contradict the candidate.

Interpretation:

- lower is better

### `S_to_L_entail`

Reverse-direction support.

Interpretation:

- not literal “faithfulness”
- better thought of as a rough coverage/compression overlap signal

### `S_to_L_contradiction`

Reverse-direction contradiction.

Interpretation:

- lower is generally better

### `entail_gap_LS_minus_SL`

`L_to_S_entail - S_to_L_entail`

Interpretation:

- often used as a ranking convenience
- higher can indicate the summary is supported by the source while staying compressed

### `large_chunk_count`, `small_chunk_count`

Number of chunks used for source and candidate.

Interpretation:

- useful for understanding how approximate the evaluation was
- larger chunk counts imply more chunk aggregation and more approximation

## Output

Default output:

- `summary_evaluation/output/03_evaluate_chunks.json`

Each row also includes:

- `evaluation_method`
- `score_scope`
- `limitation_note`

## What This Method Is Good For

- chunk-aware support checking
- comparing summary candidates on long documents
- stronger proxy than plain similarity metrics

## What It Is Not

- full-document entailment
- human-judged factuality
- ground truth

This is an aggregated chunk-level approximation.
