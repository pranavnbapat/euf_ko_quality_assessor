### Goal

Analyse a KO JSON file and decide which fields should be indexed in OpenSearch.

### What you need

Nothing beyond Python and numpy. **No LLM, no model download, no network
access, no OpenSearch instance** — this folder is pure classical IR (BM25,
IDF, entropy) over a local JSON file. It analyses fields that other services
produced with an LLM (`*_llm`, `ko_content_flat_summarised`), but it never
calls one itself.

```bash
cd ko_quality_assessor
python3 -m venv .venv && source .venv/bin/activate
pip install numpy          # the only non-stdlib import
```

**Input.** A KO export as JSON or JSONL, either passed with `--input` or left
to default to the newest file in `../input/`. Note the default picks the newest
file of *any* extension in that folder, so pass `--input` explicitly if
anything else lives there.

The export is loaded into memory whole. The current one
(`input/final_improved_mysql_export_18_06-2026_08-00-14.json`) is ~441 MB, so a
full run needs several GB of RAM and a few minutes. While iterating, use
`AF_MAX_DOCS` to work on a seeded random sample:

```bash
AF_MAX_DOCS=2000 python3 which_fields_to_choose/analyse_fields.py
```

**Expected fields.** Field *discovery* is automatic and works on any JSON, but
the recommendation layer is not: the policy lists in `analyse_fields.py`
(`POLICY_FULL_TEXT_PRIORITY`, `POLICY_FACETS`) name EU-FarmBook fields, and the
relevance labels read `project_id` / `project_acronym` / `themes` / `topics` /
`subcategories`. Run it on an unrelated schema and the diagnostics still work
while the recommendation comes back empty and the benchmark is skipped with a
reason. Edit those lists to retarget it.

### Files in this folder

| File | What it is |
|---|---|
| `analyse_fields.py` | The main field audit. Writes `../reports/field_audit_<timestamp>.txt`. |
| `evaluate_field_bundles.py` | Optional second workflow: the same bundles scored against *judged* queries. |
| `query_judgments.sample.json` | Format template only — its ids are `PUT_DOC_ID_HERE` placeholders. Replace them with real `@id` values before use. |
| `opensearch_mapping_final_improved_13_02_2026.json` | The mapping that was actually written from an earlier run. See below. |

### What `analyse_fields.py` does now

- Accepts a JSON or JSONL file via `--input`, or falls back to the newest file under `../input`.
- Supports common JSON shapes:
  - top-level list: `[ {...}, {...} ]`
  - wrapped records: `{ "docs": [ {...}, ... ] }`
  - JSONL: one object per line
- Discovers fields automatically from the records instead of relying on a hard-coded field list.
- Normalises values to text for diagnostics and classifies fields by likely role:
  - full-text search fields
  - facet / filter fields
  - sort fields
  - skip / internal / display-only fields
- Computes diagnostics such as:
  - coverage
  - average token length
  - average IDF (normalised by corpus size)
  - distinct value count and value entropy
  - boilerplate / repetition
  - per-field MRR@10 against held-out pseudo-queries
- Uses heuristics to score fields for:
  - full-text suitability
  - facet / filter suitability
  - sort suitability
- Reports every case where the policy field lists and the measurements disagree.
- Runs a pseudo-query ablation benchmark over the final recommended full-text
  bundle, with queries drawn from text held out of that bundle, using:
  - `MRR@10`
  - `nDCG@10`
  - `Recall@10`, shown against the ceiling it could reach
- Keeps internal hashes / fingerprints in `skip_internal` and separates true identifiers into `identifier_only`.
- Writes a report to `../reports/field_audit_YYYY-MM-DD_hh-mm-ss.txt`.

### Final field choice

Derived from `final_improved_13_02-2026_10-54-48.json`, which is no longer
in `input/` — treat the lists below as the standing decision, and re-run the
audit against the current export before relying on them.

This is the **policy decision**, not the script's output. The script gates it
and flags disagreements; see "The recommendation is policy, gated by evidence"
below.

The intended policy is:

- prefer `_llm` fields over the original plain fields
- exclude fingerprints / hashes / internal helper fields
- exclude URL / DOI style fields from indexing decisions

#### Full-text fields

- `title_llm`
- `subtitle_llm`
- `description_llm`
- `keywords_llm`
- `ko_content_flat_summarised`
- `project_name`
- `project_acronym`

#### Facet / filter / sort fields

- `themes`
- `subcategories`
- `locations_flat`
- `languages`
- `project_acronym`
- `date_of_completion`
- `ko_created_at`
- `ko_updated_at`
- `creators`
- `category`
- `project_type`
- `license`

#### Fields to exclude

- `ko_content_flat`
- `title`
- `subtitle`
- `description`
- `keywords`
- `_field_hashes.*`
- `_source_fp`
- `_content_fp`
- `_enrich_inputs_fp`
- `project_url`
- `project_doi`
- `ko_content_url`
- `resolved_url`

#### Identifier-only fields

- `project_id`
- `ko_file_id`

These may be useful for exact lookup or pipeline joins, but they are not part of the recommended semantic search fields.

Note:

- `project_name` is included as readable semantic text
- `project_acronym` is included as a searchable project label, but it is still best mapped as `keyword` or exact-match oriented metadata

### How the scores are calculated

The script uses hand-tuned heuristic scores, not learned model scores. Every
input is an **absolute** quantity in `[0, 1]`, so a field's score does not move
because some other field in the same file happened to be the longest or the
rarest, and the role thresholds mean the same thing on a 300-record sample as
on the full export.

- `full_text_score` — is this field worth ranking on?
  - `0.30 * coverage`
  - `+ 0.20 * avg_idf_norm` — average IDF divided by `log(n populated)`, so it
    is comparable across input files of different sizes
  - `+ 0.20 * length_fitness` — `log1p(avg_tokens) / log1p(200)`, saturating
  - `+ 0.10 * (1 - boilerplate)`
  - `+ 0.20 * retrieval` — MRR@10 on the held-out pseudo-queries below

- `facet_score` — is this field worth offering as a filter?
  - `0.25 * coverage`
  - `+ 0.25 * cardinality_fitness` — on the **absolute** number of distinct
    values (2–40 ideal, tapering to 0 at 200), because that is what a facet UI
    has to render
  - `+ 0.25 * facet_reuse` — `log10` of the mean number of documents each value
    selects; a field whose values never repeat is a label, not a filter
  - `+ 0.15 * value_entropy` — normalised Shannon entropy of the value
    distribution; 1.0 when documents spread evenly over the values
  - `+ 0.10 * type_bonus`

  Nothing here penalises a field for repeating its values. Repetition is what
  makes a facet work.

- `sort_score` — can documents be meaningfully ordered by this field?
  - `sort_bias * (0.60 + 0.40 * coverage)`, where `sort_bias` is `1.0` for
    dates, `0.9` for numerics, `0.4` for a single-valued near-unique keyword,
    and **`0.0` otherwise**

  Being well populated is never on its own enough to make a field sortable.

After the additive part, `full_text_score` and `facet_score` are multiplied
down by penalties for field shape and naming — keyword-typed non-enums
(`× 0.45`), keyword fields that are not name-like (`× 0.35`), non-string kinds
(`× 0.2`), hashes / ids / URLs (`× 0.05` full-text, `× 0.1` facet) and internal
fields (`× 0.15` / `× 0.5`). These compound, and they dominate: an id-like
keyword field ends up below 1% of its additive score. The multipliers are in
`analyse_fields.py`, which remains the source of truth.

Roles are then assigned by comparing the three scores against each other and
against `AF_FULL_TEXT_THRESHOLD` (0.40), `AF_FACET_THRESHOLD` (0.35) and
`AF_SORT_THRESHOLD` (0.35).

### The recommendation is policy, gated by evidence

The final field lists come from policy lists in `analyse_fields.py`
(`POLICY_FULL_TEXT_PRIORITY`, `POLICY_FACETS`): prefer `_llm` fields over the
originals, keep hashes and URLs out. The scores gate those lists but do not
generate them.

The report has a `WHERE POLICY AND EVIDENCE DISAGREE` section that names every
case where the two differ — a policy pick whose measured role is something
else, a policy pick the measurements reject outright, and fields that score
well but are not on the list. Read that section before treating the field
lists as a finding rather than a decision.

### Pseudo-query benchmark

The recommended bundle is benchmarked with `MRR@10`, `nDCG@10` and `Recall@10`,
plus a drop-one-field ablation.

**Queries are held out from the bundle.** For each sampled document, its
highest tf-idf terms from `ko_content_flat, title, keywords, description,
subtitle` (configurable with `AF_QUERY_SOURCE`) become a short keyword query,
and that document is the target. Any field that would also appear in a bundle
is removed from the query source and named in the report.

This matters: building the query out of the same fields being ranked makes the
ablation answer itself. Dropping a field that is in the query removes the exact
string being matched, so it must hurt; a field that is not in the query only
lengthens documents, so BM25 length normalisation must penalise it. The result
is decided before any data is read.

Residual leakage is still measured and printed — the share of each bundle
field's query terms that it already contains — because a summary of the query
source still overlaps it. Fields above 25% get a caution line.

Relevance labels: target document = 3, same `project_id`/`project_acronym` = 2,
**two or more** shared `themes`/`topics`/`subcategories` values = 1. Taxonomy
values covering more than `AF_QREL_MAX_BUCKET_RATIO` (25%) of the corpus create
no judgements at all, and `category` is not used — a field with five values
otherwise marks a fifth of the corpus relevant to every query and pins
`Recall@k` at its arithmetic ceiling.

`Recall@k` is reported as `achieved/ceiling`, where the ceiling is
`k / |relevant|`. No ranking can beat the ceiling, so the raw number alone is
not interpretable.

How to read the ablation:

- `all_recommended` — baseline using the whole recommended full-text bundle
- `drop_<field>` — the same benchmark after removing one field
- dropping a field lowers `nDCG@10` → that field was helping
- barely changes it → likely optional
- raises it → the field may be noise in this weak benchmark

If the whole ablation spans less than 0.01 nDCG, the report says so and tells
you to treat every field as untested rather than confirmed.

This is still weak supervision. Judged queries with real relevance labels
remain the only decisive evidence.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `AF_MAX_DOCS` | `0` (all) | Analyse a random sample of N records (seeded, not the first N) |
| `AF_MIN_COVERAGE` | `0.05` | Coverage below which a field is skipped |
| `AF_FACET_IDEAL_MAX` | `40` | Distinct values still considered an ideal facet |
| `AF_FACET_USABLE_MAX` | `200` | Distinct values beyond which a facet scores 0 |
| `AF_BENCH_QUERIES` | `1500` | Pseudo-queries sampled for the benchmark |
| `AF_QUERY_TERMS` | `8` | Terms per pseudo-query |
| `AF_QUERY_SOURCE` | `ko_content_flat,title,keywords,description,subtitle` | Fields queries are drawn from |
| `AF_QREL_MIN_OVERLAP` | `2` | Shared taxonomy values needed for a grade-1 judgement |
| `AF_QREL_MAX_BUCKET_RATIO` | `0.25` | Taxonomy values above this corpus share create no judgements |
| `AF_SEED` | `13` | Seed for query and document sampling |
| `AF_FULL_TEXT_THRESHOLD` / `AF_FACET_THRESHOLD` / `AF_SORT_THRESHOLD` | `0.40` / `0.35` / `0.35` | Role assignment cutoffs |

### Example

Run from the repository root (`ko_quality_assessor/`):

```bash
# newest file in input/, full corpus
python3 which_fields_to_choose/analyse_fields.py

# a specific export
python3 which_fields_to_choose/analyse_fields.py \
  --input input/final_improved_mysql_export_18_06-2026_08-00-14.json

# fast iteration on a seeded 2000-record sample
AF_MAX_DOCS=2000 python3 which_fields_to_choose/analyse_fields.py
```

The report is printed to stdout and written to
`../reports/field_audit_<timestamp>.txt`. Read the `WHERE POLICY AND EVIDENCE
DISAGREE` section first: it is the part that tells you whether the field lists
still match the data.

### Judged retrieval evaluation

For a stronger evaluation than heuristics or pseudo-qrels, use judged queries.

Important:

- this is a separate workflow
- it does **not** change the intended report/output of `analyse_fields.py`
- `analyse_fields.py` remains the main field-audit script for the current indexing decision
- `evaluate_field_bundles.py` is only an additional evaluation tool for later use if you create judged queries

Script:

```bash
python3 which_fields_to_choose/evaluate_field_bundles.py \
  --input input/final_improved_13_02-2026_10-54-48.json \
  --judgments which_fields_to_choose/query_judgments.sample.json \
  --id-field @id
```

Judgments file format:

```json
[
  {
    "query": "anaerobic digestion nutrient management",
    "relevant": {
      "DOC_ID_1": 3,
      "DOC_ID_2": 2,
      "DOC_ID_3": 1
    }
  }
]
```

Meaning of relevance grades:

- `0`: not relevant
- `1`: somewhat relevant
- `2`: relevant
- `3`: highly relevant

The evaluator:

- loads the judged queries
- ranks documents for the recommended full-text bundle
- runs drop-one-field ablations automatically
- reports:
  - `nDCG@10`
  - `MRR@10`
  - `Recall@10` as `achieved/ceiling`
  - `MAP` over the same top-k window

Interpretation:

- if dropping a field lowers `nDCG@10`, that field is helping retrieval
- if dropping a field barely changes the metrics, that field is likely optional
- if dropping a field improves the metrics, that field may be noisy

### The committed mapping

`opensearch_mapping_final_improved_13_02_2026.json` is the mapping written from
the February 2026 audit: `dynamic: false`, a `lowercase` + `asciifolding`
normalizer, and 22 properties. Its design choice is a `search_text` sink — the
eight text fields all `copy_to` it, so queries can hit one analysed field
instead of fanning out across eight.

It has since drifted from the field lists above, in three places:

| Field | This README | The mapping |
|---|---|---|
| `ko_content_flat` | excluded | indexed as `text`, copied to `search_text` |
| `ko_object_name` | not mentioned | indexed as `text`, copied to `search_text` |
| `project_id` | identifier-only | indexed as a `keyword` facet |

The mapping is a hand-maintained artifact — nothing in this folder generates or
validates it. Reconcile it against a fresh audit before treating either as
authoritative.
