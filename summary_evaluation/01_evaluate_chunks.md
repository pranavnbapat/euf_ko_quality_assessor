# 01 Evaluate Chunks

## Purpose

`01_evaluate_chunks.py` is the cheapest evaluator in this folder.

It does not try to decide whether a summary is factually correct. Instead, it
computes intrinsic text-quality diagnostics so you can compare:

- `title` vs `title_llm`
- `subtitle` vs `subtitle_llm`
- `description` vs `description_llm`
- `keywords` vs `keywords_llm`
- `ko_content_flat` vs `ko_content_flat_summarised`

The goal is to answer questions like:

- Is the generated field much shorter or much longer?
- Is it more repetitive?
- Does it look like normal English?
- Does it look noisy, OCR-heavy, or list-heavy?
- Is the summarised content field actually more indexable than the raw one?

## What It Uses

This script uses only local text processing:

- regex tokenisation
- stopword statistics from `nltk`
- character and token counts
- simple repetition counts
- a rough Flesch-Kincaid-like readability proxy

It does not use embeddings, NLI, QA, or external APIs.

## What It Outputs

For each input record, it writes one row containing metrics for every original
and generated variant.

Default outputs:

- `summary_evaluation/output/01_evaluate_selected.json`
- `summary_evaluation/output/01_evaluate_selected.xlsx`

The Excel file is mainly for manual inspection and sorting/filtering.

## Metrics

### `*_len_chars`

Raw character length.

Interpretation:

- high value means the field is large
- useful for spotting huge raw content or unexpectedly long summaries

Impact:

- large values increase storage and indexing cost
- extremely large content is a warning for chunking/summarisation need

### `*_len_tokens`

Token count after normalised tokenisation.

Interpretation:

- this is the main length metric for comparing original vs generated text

Impact:

- lower is often better for search/indexing, unless the model has over-compressed

### `*_ttr`

Type-token ratio = unique tokens / total tokens.

Interpretation:

- higher means more lexical variety
- very high on short text is not meaningful
- very low on long text often means repetition or boilerplate

Impact:

- useful for spotting repetitive outputs or degraded text

### `*_stopword_ratio`

Fraction of tokens that are English stopwords.

Interpretation:

- moderate values often look like normal English prose
- very low values may indicate keyword soup, tables, or non-English text

Impact:

- helps distinguish readable text from fragments or extracted debris

Limitation:

- English-only heuristic

### `*_punct_ratio`

Punctuation characters / total characters.

Interpretation:

- high values can indicate bullets, tables, markup fragments, or noisy extraction

Impact:

- useful for spotting raw PDF-like or structured junk

### `*_readability_fk_like`

Rough Flesch-Kincaid-like score using vowel-count approximation instead of real
syllables.

Interpretation:

- higher generally means easier reading
- lower means denser or harder-to-read text
- best used comparatively, not absolutely

Impact:

- helps see whether generated text became easier to consume

Limitation:

- not a publishable readability metric
- weak for short strings like titles and keyword lists

### `*_top5_repetition_ratio`

Share of tokens explained by the top 5 non-stopword tokens.

Interpretation:

- high values mean repetition dominates the text
- more useful on medium/long text than on very short fields

Impact:

- good for catching looping, boilerplate, and repeated structure

### `*_english_suspect`

Boolean flag set when:

- token count is reasonably large
- stopword ratio is unusually low

Interpretation:

- `true` means the field may not look like normal English prose

Impact:

- quick QA flag, not a language-identification system

## How To Read The Output

Typical reading pattern:

1. Compare presence:
   - is the generated field missing less often or more often?
2. Compare lengths:
   - is `description_llm` shorter but still substantial?
   - is `ko_content_flat_summarised` dramatically smaller than `ko_content_flat`?
3. Compare repetition and punctuation:
   - did the generated version remove noise?
4. Compare readability:
   - did the generated version become easier to read?

## What This Method Is Good For

- QA of generated fields
- identifying records that still have giant raw content
- checking whether generated fields are cleaner for indexing
- spotting missing or empty generated values

## What It Is Not Good For

- semantic faithfulness
- hallucination detection
- factual correctness
- retrieval quality

For those, use `02`, `03`, and `04`.
