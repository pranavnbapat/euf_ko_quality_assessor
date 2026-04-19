# 02 Evaluate Chunks

## Purpose

`02_evaluate_chunks.py` measures semantic similarity between raw content and its
candidate summary variants.

Unlike `01`, this script uses neural similarity metrics. It is trying to answer:

- Does the summary stay semantically close to the source?
- Which `ko_content_flat*` candidate preserves source meaning best?

## What It Compares

For each record:

- source: `ko_content_flat`
- candidates: every other non-empty field starting with `ko_content_flat`

Typical candidates include:

- `ko_content_flat_summarised`
- model-specific summary variants if present

## Metrics Used

### Cosine similarity

Computed from sentence-transformer embeddings using:

- `sentence-transformers/all-MiniLM-L6-v2`

Interpretation:

- higher means the summary embedding is closer to the source embedding

Impact:

- cheap semantic similarity baseline
- useful for broad preservation checks

Limitation:

- one embedding can hide local omissions or hallucinations

### BERTScore

Computed with `roberta-large`.

Outputs:

- `bertscore_p`
- `bertscore_r`
- `bertscore_f1`

Interpretation:

- precision: how much of the candidate text is supported by the reference
- recall: how much of the reference is reflected in the candidate
- F1: overall balance

Impact:

- stronger than lexical overlap
- often the most useful metric in this script

Limitation:

- still similarity, not factuality
- expensive on large datasets

### MoverScore

Optional.

Interpretation:

- higher means stronger semantic alignment based on token movement in embedding space

Impact:

- extra semantic signal

Limitation:

- slower
- not essential if runtime is a concern

## Output

Default output:

- `summary_evaluation/output/02_evaluate_chunks.json`

During long runs it also writes:

- `summary_evaluation/output/02_evaluate_chunks.json.partial`

The partial file is a checkpoint for monitoring progress.

## Runtime Characteristics

This script is much heavier than `01`.

Main reasons:

- embedding inference
- BERTScore with `roberta-large`
- optional MoverScore

## How To Interpret Results

Good candidates usually show:

- high `bertscore_f1`
- high cosine similarity
- reasonable compression relative to source length

If one candidate is much shorter but keeps strong semantic scores, it is usually
better for indexing.

## What This Method Is Good For

- semantic preservation
- comparing multiple summary candidates
- checking whether compression destroyed meaning

## What It Is Not Good For

- hallucination detection
- full factuality
- document-level reasoning

For stronger support-style checks, use `03` or `04`.
