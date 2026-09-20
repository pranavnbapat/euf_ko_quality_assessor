### Goal

Decide, from measurement rather than argument, whether EU-FarmBook search should
be keyword-based (BM25), semantic (dense embeddings), or both — and if both, how
to weight them.

This is the companion to [`which_fields_to_choose`](../which_fields_to_choose),
which decides *which fields* to index. This folder decides *how to search them*.

### What you need

Python, numpy, and `sentence-transformers` (with torch). **No LLM API, no network
access at run time, and no OpenSearch instance** — the embedding models run
locally on CPU.

```bash
cd ko_quality_assessor
source .venv/bin/activate
pip install numpy sentence-transformers langdetect
```

Both embedding models are the ones the platform actually indexes with, so results
transfer to production:

| Mode | Model | Production index |
|---|---|---|
| `dense_en` | `sentence-transformers/msmarco-distilbert-base-tas-b` (768-dim) | `neural_search_index_msmarco_distilbert` |
| `dense_multi` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim) | `neural_search_index_mlang_minilm_v1` |

They are downloaded once from Hugging Face and cached in `~/.cache/huggingface`.
Embeddings are cached in `semantic_vs_search/.cache/`, keyed by model and exact
text, so re-runs with the same corpus are close to instant.

**Input.** The same KO export the field audit uses: JSON or JSONL via `--input`,
or the newest file in `../input/` by default.

### How to run

Run from the repository root (`ko_quality_assessor/`):

```bash
# default: 3000 documents, 400 query-documents, both models, per-field breakdown
python semantic_vs_search/compare_search_modes.py

# quick pass while iterating
python semantic_vs_search/compare_search_modes.py --limit 800 --queries 150 --skip-per-field

# the whole corpus (slow on CPU - roughly 25 minutes per model)
python semantic_vs_search/compare_search_modes.py --limit 0
```

The report is printed to stdout and written to
`../reports/search_mode_comparison_<timestamp>.txt`.

| Flag | Default | Effect |
|---|---|---|
| `--limit` | `3000` | Documents in the corpus (`0` = all). Larger is harder and more discriminating. |
| `--queries` | `400` | Documents to draw queries from; each yields up to four queries. |
| `--seed` | `13` | Sampling seed, so runs are reproducible. |
| `--skip-per-field` | off | Skip the per-field comparison, which is the slow part. |

### How the comparison is made fair

The hard part of this question is not running two retrievers. It is making sure
neither one is handed the answer.

**Indexed text and query text are kept apart.** Retrievers only ever see the
fields the field audit recommends indexing (`title_llm`, `subtitle_llm`,
`description_llm`, `keywords_llm`, `ko_content_flat_summarised`, `project_name`,
`project_acronym`). Queries are built only from text that is *not* indexed
(`ko_content_flat` and the original, pre-LLM metadata). No mode can win by
matching a string it was also given to index.

**Four query types, chosen because they separate the modes:**

| Type | How it is built | What it tests |
|---|---|---|
| `topical` | The document's highest tf-idf terms from held-out text | The ordinary case |
| `unseen-vocab` | The same, but restricted to words that appear **nowhere** in the indexed text | Vocabulary mismatch. BM25 cannot match these by construction, so this isolates what only meaning-based retrieval can do |
| `entity` | The project acronym plus the rarest capitalised token | Exact match on codes and proper nouns, where dense retrieval is known to be weak |
| `natural` | A whole sentence from held-out text | A real, verbose, natural-language query |

**Language relationship is measured, not assumed.** Each query and each document
has its language detected, and results are broken down into same-language
(English), same-language (non-English) and cross-language. That last bucket is
the one keyword search structurally cannot serve.

**Complementarity is counted.** For every query the report records whether the
document was found by both modes, by one alone, or by neither. That is the
evidence for or against a hybrid: if the modes retrieve the same documents,
fusing them buys nothing.

**Fusion uses RRF**, not tuned linear weights, because there is nothing to
calibrate weights against yet. RRF combines two rankings by reciprocal rank and
needs no score normalisation between modes.

### Metrics

Each query has exactly one correct document — the one it was derived from — so
the metrics are deliberately simple:

- **MRR@10** — how near the top the correct document lands, averaged. The
  headline number.
- **Hit@1** — the right answer came first.
- **Hit@10** — the right answer was somewhere on page one.

nDCG is not reported: with a single relevant document it is a monotone function
of rank and would add nothing over MRR.

### What the report contains

| Section | What it gives you |
|---|---|
| `SETUP` | Corpus size, query count, which fields were indexed and which held out |
| `HEADLINE: ALL QUERIES` | The four modes side by side |
| `BY QUERY TYPE` | Where each mode wins — the most informative section |
| `BY LANGUAGE RELATIONSHIP` | Same-language vs cross-language performance |
| `DO THE MODES FIND THE SAME DOCUMENTS?` | Both / keyword-only / semantic-only / neither |
| `SINGLE-FIELD RETRIEVAL` | Which mode suits which field, for routing a hybrid |
| `READING OF THE RESULTS` | A rule-based interpretation of the numbers above |
| `VERDICT` | The recommendation, with its reasons and its limits |

### Replacing the synthetic evidence: `build_judging_set.py`

Everything above rests on queries an LLM invented and relevance an LLM graded.
Neither has to stay that way. scout logs every search to ClickHouse, so the
queries users actually typed already exist. This script turns them into a
spreadsheet a domain expert can grade, in three stages:

```bash
# 1. on the server, where ClickHouse is reachable
python build_judging_set.py export --out real_queries.json --days 90 --size 50

# 2. anywhere the search API is reachable
python build_judging_set.py sheet --queries real_queries.json --out judging.xlsx

# 3. once the `grade` column is filled in
python build_judging_set.py convert --sheet judging.xlsx --out judgments.json
python ../which_fields_to_choose/evaluate_field_bundles.py \
    --input ../input/<export>.json --judgments judgments.json --id-field _id
```

The sample is stratified by (language, outcome) using proportions computed from
the logs themselves, so it reflects real usage rather than an assumption about
it - and it deliberately keeps zero-result searches, which are the only way to
evaluate the no-match behaviour at all.

The spreadsheet carries a dropdown for the 0-3 grade, a second sheet explaining
the scale, and one row per retrieved document. Grading the same sheet twice,
independently, gives an agreement rate - which is the number that says how far
any of this can be trusted, including the LLM judge used elsewhere in this
folder.

Use `--id-field _id`, not `@id`: the sheet records the parent document id the
search API returns, which joins to the export's `_id`.

### What this test cannot tell you

Stated in the report itself, and worth repeating:

- **The queries are generated, not observed.** They are realistic in shape but
  nobody actually typed them. Real users phrase things differently and want
  different things.
- **"Correct" means one specific document.** A real search usually has many
  acceptable answers. Single-target scoring rewards precision in a way that does
  not map exactly onto user satisfaction.
- **Documents are truncated, not chunked.** Production splits long documents into
  chunks and indexes each; here each document is truncated to the model's context
  window. Long documents therefore do worse in this test than in production, and
  the multilingual model — with a 128-token limit — is affected most.
- **Judged queries remain the only decisive evidence.** Thirty to fifty real
  searches with someone marking which results were correct would settle both the
  keyword/semantic weighting and the field ablation in
  `which_fields_to_choose`. That is the single highest-value missing asset.

### Relationship to the rest of the repo

- `../which_fields_to_choose` — which fields to index. Its recommended bundle is
  what this folder searches over.
- `../reports/` — both tools write timestamped reports here. The folder is
  gitignored, so reports are local.
- The BM25 implementation is a deliberate self-contained copy of the one in
  `analyse_fields.py`, following this workspace's convention of copying rather
  than importing across service folders.
