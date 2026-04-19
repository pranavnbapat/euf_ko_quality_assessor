### Goal

Analyse a KO JSON file and decide which fields should be indexed in OpenSearch.

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
  - average IDF
  - cardinality ratio
  - boilerplate / repetition
- Uses heuristics to score fields for:
  - full-text suitability
  - facet / filter suitability
  - sort suitability
- Runs a pseudo-query ablation benchmark over the final recommended full-text bundle using:
  - `MRR@10`
  - `Recall@10`
  - `nDCG@10`
- Keeps internal hashes / fingerprints in `skip_internal` and separates true identifiers into `identifier_only`.
- Writes a report to `../reports/field_audit_YYYY-MM-DD_hh-mm-ss.txt`.

### Final field choice for `final_improved_13_02-2026_10-54-48.json`

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

The script uses hand-tuned heuristic scores, not learned model scores.

- `full_text_score`
  - rewards coverage, term distinctiveness, length, low boilerplate, and self-retrieval performance
  - formula:
    - `0.35 * coverage_norm`
    - `+ 0.20 * avg_idf_norm`
    - `+ 0.15 * avg_tokens_norm`
    - `+ 0.10 * low_boilerplate_norm`
    - `+ 0.20 * retrieval_norm`

- `facet_score`
  - rewards coverage, stable categorical/date/numeric behavior, and balanced facet cardinality
  - formula:
    - `0.40 * coverage_norm`
    - `+ 0.35 * facet_balance`
    - `+ 0.15 * low_boilerplate_norm`
    - `+ 0.10 * type_bonus`

- `sort_score`
  - rewards populated scalar fields, especially dates and numbers
  - formula:
    - `0.50 * coverage_norm`
    - `+ 0.35 * sort_bias`
    - `+ 0.15 * low_boilerplate_norm`

These scores are only used to rank candidates. Final recommendations also apply policy rules such as preferring `_llm` fields and excluding internal/hash fields.

### Better evaluation than heuristics

The script now also includes a pseudo-query benchmark with standard IR metrics:

- `MRR@10`
- `Recall@10`
- `nDCG@10`

This benchmark is still weakly supervised because it builds pseudo relevance from:

- the source record itself
- shared `project_id` / `project_acronym`
- shared `themes` / `topics` / `subcategories`
- shared `category`

This is more defensible than relying only on raw heuristic scores, but it is still not a substitute for real judged query relevance.

How to read the ablation output:

- `all_recommended`
  - baseline using the whole recommended full-text bundle
- `drop_<field>`
  - the same benchmark after removing one field from that bundle
- interpretation:
  - if dropping a field lowers `nDCG@10`, that field was helping
  - if dropping a field barely changes `nDCG@10`, that field is likely optional
  - if dropping a field improves `nDCG@10`, that field may be noisy in this weak benchmark

The report also prints a `PLAIN-ENGLISH ABLATION SUMMARY` section that translates those metric changes into a short recommendation.

### About the self-retrieval section

The report may also print a `SELF-RETRIEVAL PROXY (MRR@10)` section.

This section is only a lightweight legacy proxy:

- it does **not** test all possible field combinations
- it evaluates a small candidate set of single fields and a few hand-picked combinations
- historically those combinations were centered around fields like `title`, `description`, `keywords`, and `ko_content_flat`

So if you see combinations like:

- `title`
- `description + title`
- `keywords + title`
- `ko_content_flat + title`

that is because the code intentionally uses a small bounded set for speed, not because those are the only valid combinations.

For actual decision-making, the pseudo-query ablation benchmark is more important than the self-retrieval proxy.

The most scientifically sound next step would be:

- a real query set
- labeled relevance judgments
- field ablation experiments evaluated with `nDCG@k`, `MRR`, `Recall@k`, and optionally `MAP`

### Example

```bash
python3 which_fields_to_choose/analyse_fields.py --input input/final_improved_13_02-2026_10-54-48.json
```

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
  - `Recall@10`
  - `MAP`

Interpretation:

- if dropping a field lowers `nDCG@10`, that field is helping retrieval
- if dropping a field barely changes the metrics, that field is likely optional
- if dropping a field improves the metrics, that field may be noisy
