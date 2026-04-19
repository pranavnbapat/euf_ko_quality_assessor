# 04 Evaluate Chunks

## Purpose

`04_evaluate_chunks.py` is the most ambitious evaluator in this folder.

It combines:

- chunk-aware NLI
- question generation
- question answering

This is meant to be a stronger factuality-style check than `03`.

## Models Used

- NLI: `microsoft/deberta-large-mnli`
- question generation: `iarfmoose/t5-base-question-generator`
- QA: `deepset/deberta-v3-base-squad2`

## Core Workflow

For each source/candidate pair:

1. Run chunk-aware bidirectional NLI.
2. Generate questions from chunks of the source text.
3. Answer those questions on:
   - the source
   - the candidate
4. Compare answers using token-level F1.

## Metrics

### NLI metrics

Same role as in `03`:

- `nli_L_to_S_entail`
- `nli_L_to_S_contra`
- `nli_S_to_L_entail`
- `nli_S_to_L_contra`

Interpretation:

- `L_to_S` is the stronger support-style signal
- reverse direction is weaker and more proxy-like

### `qa_mean_f1`

Average token-level F1 between answers from source and candidate.

Interpretation:

- higher means the candidate can answer source-derived questions more similarly
- lower means important source information may be missing or distorted

Impact:

- this is a useful factual-consistency-style signal

### `qa_num_questions`

How many usable questions were generated and evaluated.

Interpretation:

- low values make the QA score less stable

### `nli_L_chunk_count`, `nli_S_chunk_count`

Chunk counts used during NLI aggregation.

Interpretation:

- higher chunk counts imply more approximation and more aggregation

## Why This Method Is Stronger

Unlike plain similarity, it asks:

- can the candidate answer source-derived questions?

That makes it more informative for factual consistency.

## Why It Is Still Limited

- question generation quality can vary
- QA extraction is still approximate
- chunk aggregation is still a heuristic
- runtime cost is high

So this is still not human ground truth.

## Output

Default outputs:

- `summary_evaluation/output/04_evaluate_chunks.jsonl`
- `summary_evaluation/output/04_evaluate_chunks.csv`

Rows also include:

- `evaluation_method`
- `score_scope`
- `limitation_note`

## Recommended Use

Use `04` last, after `01`, `02`, and `03`.

It is best treated as an expensive secondary check on top candidates, not the
first thing to run across a huge dataset.
