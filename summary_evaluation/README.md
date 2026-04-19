# Summary Evaluation

This folder is a curated copy of the useful evaluation code from
[`which_model_to_choose/methods`](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/which_model_to_choose/methods).

The goal is to keep the important evaluation logic in one clean place without
changing the historical `methods` folder.

## Why This Folder Exists

The original `methods` folder mixes:

- active evaluator scripts
- plotting scripts
- historical JSON outputs
- methodology notes
- one stale plot script
- sample input files

That is workable as a scratch area, but not a clean long-term layout.

This folder keeps only the important evaluation code and documentation:

- `01_evaluate_chunks.py`
  Intrinsic text-quality checks for selected fields.
- `01_plot_evaluation.py`
  Plotting for the `01` evaluator output.
- `01_evaluate_chunks_methodology.txt`
  Plain-English metric explanation.
- `02_evaluate_chunks.py`
  Semantic similarity evaluation of `ko_content_flat*` candidates.
- `02_plot_evaluation_chunks.py`
  Plotting for `02`.
- `03_evaluate_chunks.py`
  Bidirectional NLI evaluation of `ko_content_flat*` candidates.
- `03_plot_nli_results.py`
  Plotting for `03`.
- `04_evaluate_chunks.py`
  Experimental NLI + QG/QA evaluation.

Detailed per-script documentation:

- [01_evaluate_chunks.md](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/01_evaluate_chunks.md)
- [01_plot_evaluation.md](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/01_plot_evaluation.md)
- [02_evaluate_chunks.md](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/02_evaluate_chunks.md)
- [02_plot_evaluation_chunks.md](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/02_plot_evaluation_chunks.md)
- [03_evaluate_chunks.md](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/03_evaluate_chunks.md)
- [03_plot_nli_results.md](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/03_plot_nli_results.md)
- [04_evaluate_chunks.md](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/04_evaluate_chunks.md)

## Intentionally Not Copied

- `01_plot_evaluation_chunks.py`
  This looks stale and does not match the current `01_evaluate_chunks.py` output schema.
- historical output JSON files
- historical text summaries
- sample input files

Those remain in the original `methods` folder as run history or legacy material.

## What Was Fixed Here

The curated copies were adjusted so they are less brittle:

- they accept CLI input/output paths instead of relying only on hard-coded paths
- they default to the newest file under the repo root `input/` folder when `--input` is not passed
- `02_evaluate_chunks.py` now evaluates all `ko_content_flat*` candidates per record,
  not just one hard-coded summary field
- `01` and `03` can read list JSON, single-object JSON, and wrapped `{ "docs": [...] }`
- `01_evaluate_chunks.py` evaluates preferred final fields only:
  `title_llm`, `subtitle_llm`, `description_llm`, `keywords_llm`, and
  `ko_content_flat_summarised`; it does not fall back to the older/original fields

## Recommended Use

Run the curated copies from the repo root.

If `--input` is omitted, the evaluators use the newest file under the repo root
[`input/`](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/input) folder.

If `--output` is omitted, results are written under
[`summary_evaluation/output/`](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/summary_evaluation/output).

The scripts resolve these paths relative to the repo and script location, so
they work whether you run them from the repo root or from inside
`summary_evaluation/`.

### 01. Intrinsic quality

```bash
python3 summary_evaluation/01_evaluate_chunks.py \
  --output summary_evaluation/output/01_evaluate_selected.json
```

This writes both:

- `summary_evaluation/output/01_evaluate_selected.json`
- `summary_evaluation/output/01_evaluate_selected.xlsx`

Then:

```bash
python3 summary_evaluation/01_plot_evaluation.py \
  --input summary_evaluation/output/01_evaluate_selected.json \
  --output summary_evaluation/output/plots_01
```

### 02. Semantic similarity

```bash
python3 summary_evaluation/02_evaluate_chunks.py \
  --output summary_evaluation/output/02_evaluate_chunks.json
```

Then:

```bash
python3 summary_evaluation/02_plot_evaluation_chunks.py \
  --input summary_evaluation/output/02_evaluate_chunks.json
```

### 03. NLI

```bash
python3 summary_evaluation/03_evaluate_chunks.py \
  --output summary_evaluation/output/03_evaluate_chunks.json
```

Then:

```bash
python3 summary_evaluation/03_plot_nli_results.py \
  --input summary_evaluation/output/03_evaluate_chunks.json
```

### 04. Experimental factuality

```bash
python3 summary_evaluation/04_evaluate_chunks.py \
  --output-jsonl summary_evaluation/output/04_evaluate_chunks.jsonl \
  --output-csv summary_evaluation/output/04_evaluate_chunks.csv
```

## Suggested Order

Run them in this order:

1. `01_evaluate_chunks.py`
   Cheap intrinsic hygiene check. Run this first.
2. `02_evaluate_chunks.py`
   Semantic similarity / preservation.
3. `03_evaluate_chunks.py`
   Chunked NLI support proxy.
4. `04_evaluate_chunks.py`
   Most expensive and most experimental. Run last, only if needed.

Recommended plotting order:

1. `01_plot_evaluation.py`
2. `02_plot_evaluation_chunks.py`
3. `03_plot_nli_results.py`

Example from inside `summary_evaluation/`:

```bash
python 01_evaluate_chunks.py
python 02_evaluate_chunks.py
python 03_evaluate_chunks.py
python 04_evaluate_chunks.py
```

This will:

- read the newest file from the repo root `input/` folder
- write outputs to `summary_evaluation/output/`

## Trust Level

- `01` is useful for hygiene and intrinsic diagnostics.
- `02` is useful for semantic preservation.
- `03` is useful as a rough factual-support proxy.
- `04` is experimental and expensive.

`03` and `04` now use chunk-aware scoring instead of a single first-window pass,
which makes them more defensible on long KOs.

Even so:

- `03` is still aggregated chunk-level NLI, not full-document entailment
- `04` is still aggregated chunk-level NLI + QG/QA, not ground-truth factuality
- both should be used comparatively across candidates, not as literal truth labels
