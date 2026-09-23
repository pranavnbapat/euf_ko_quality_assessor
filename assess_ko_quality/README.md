# assess_ko_quality

Assesses the quality of EU-FarmBook Knowledge Objects (KOs). A KO is a file plus the
metadata that describes it. **The score measures the metadata** - title, subtitle,
description and keywords - against the file: is it correct, informative, well-formed and
findable? The file itself is the original author's work, so its quality is measured and
reported, and never changes the score.

Every criterion is measurable and reproducible, taken from published research and
standards; nothing is fitted to human ratings. Each part is checked technically: the
pipeline damages metadata on purpose and confirms that the matching measurement notices.

| document | what it covers |
|---|---|
| this README | running the pipeline and reading its outputs |
| [`METHODS.md`](METHODS.md) | every method, threshold and weight, precisely enough to reimplement |
| [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) | where each method comes from, why it was chosen, and the evidence behind every number |
| `How_the_KO_quality_pipeline_works.docx` | the two documents above as a Word guide, generated from them |
| `validation/metadata_spec.json` | every configurable setting, with its provenance |

---

## Quick start

```bash
cd assess_ko_quality
python pipeline.py --limit 20                  # 20 KOs, one per language
python pipeline.py --limit 20 --selftest       # plus the perturbation self-test
python pipeline.py --limit 10 --llm-review     # plus LLM review: 10 calls
python pipeline.py --no-gate-llm --limit 200   # no LLM calls at all
```

The first run on an export builds a corpus index: about 20 minutes the very first time
(it embeds every KO once), about 5 minutes after a settings change, seconds otherwise. It
is cached in `output/_corpus_cache/`. Assessment then takes about 25-30 seconds per KO on
CPU, mostly the entailment model.

**LLM calls.** Only two stages can call an LLM: the domain gate, for the few KOs its
classifier is unsure about (typically 0-10%; `--no-gate-llm` turns it off), and the
optional LLM review (`--llm-review`, one call per KO, capped by `--llm-limit`, default
10). Responses are cached, so rerunning costs nothing.

---

## Choosing what to assess

You rarely want all 10,468 at once. Every filter reports what it removed rather than
dropping it quietly, and the selection is written to the run's `run.json` so a run
describes itself.

| Flag | Default | What it does |
|---|---|---|
| `--limit N` | 0 (all) | Assess at most N records |
| `--min-chars N` | 500 | Skip KOs with less extracted text than this |
| `--file-type` | all | `pdf`, `mp4`, `Video`, `Factsheet`, … (comma list) |
| `--languages` | all | `English,French` (comma list) |
| `--lang-balance` | equal | How `--limit` is spread across languages |
| `--seed` | 42 | Sampling seed, so a run repeats |
| `--no-gate-llm` | off | Classifier-only gate; with no `--llm-review`, no LLM calls at all |

```bash
# 100 records, spread evenly across languages, real text only
python pipeline.py --limit 100

# 50 English PDFs
python pipeline.py --limit 50 --file-type pdf --languages English

# every video, including ones whose text never extracted
python pipeline.py --file-type mp4 --min-chars 0
```

### File type

`--file-type` is matched against **category, extension, mimetype and subcategory** at
once, so `pdf`, `Video` and `Factsheet` all work without you having to know which field
holds the answer. That matters: the extension is missing on 16.7% of records and the
mimetype on 10.6%, so matching a single field would silently drop rows.

The corpus is 90.8% `Document` (81.2% of everything is `.pdf`), 4.7% `Video`, 2.5%
`Slideshow/Presentation`, and small numbers of `Image`, `Dataset`, `Audio` and
`Software Application`.

### Language

A KO's language is the **first one listed**. About 12% carry more than one; taking the
first keeps filtering and counting consistent, so a request never returns anything
outside it. A KO listed as English then Spanish is treated as English and will not come
back from a Spanish filter.

### `--lang-balance`: equal or proportional

Only relevant when `--limit` is set **and** more than one language is in play. With
`--languages English` there is nothing to balance between, so omit it.

**proportional** mirrors the corpus. **equal** gives each language roughly the same
number of records. With a 61%-English corpus and `--limit 40`:

| language | corpus | equal | proportional |
|---|---|---|---|
| English | 61.4% | 2 | 23 |
| Spanish | 11.5% | 2 | 5 |
| German | 4.3% | 2 | **0** |
| Greek | 1.8% | 2 | 2 |
| **languages covered** | 21 | **21** | **10** |

Use **equal** (the default) to check the pipeline or compare across languages. Use
**proportional** when the aggregate is the point, such as "what is the average quality
of the platform" - an equal sample over-weights small languages and describes no real
population.

### The 500-character floor

`--min-chars 500` is the default and it excludes **35.9% of the corpus** - the KOs whose
text never extracted. There is nothing to check the metadata against, but those records
are a real finding about ingestion. The count is always printed. Pass `--min-chars 0` to
include them; their semantic and functional sub-pillars are then "not evaluable".

---

## The pipeline

```
select ─► 1 GATE ─► 2 COMPLIANCE ─► 3 CONTENT ─► 4 METADATA SCORE ─► 5 LLM REVIEW ─► 6 ANALYSIS
```

| stage | question | affects the score |
|---|---|---|
| 1 gate | Is the KO about agriculture in the broad sense (farming, forestry, livestock, food, rural life, and their economics, environment and society)? | off-domain KOs are not scored |
| 2 compliance | Are the fields every record carries filled in, with values from the current vocabularies? Optional fields are never a failure; old vocabulary values are reported as drift; age is reported, not failed. | no |
| 3 content | What is the file itself like? The published web-text quality filters (Gopher, C4, FineWeb, FineWeb2), declared vs detected language, lexical diversity, readability, DOIs and links. | no |
| 4 metadata score | How well do title, subtitle, description and keywords describe the file? | **the score, 0-100** |
| 5 LLM review | Is each field correct? If not, what should it say? A proposal is kept only if it scores higher without becoming less faithful to the file. | proposals are scored by stage 4 |
| 6 analysis | Why is a score low, and would the ranking change under other weights? | no |

### The score

Each field is scored on three pillars and ten sub-pillars, then averaged with equal weights:

| pillar | sub-pillars |
|---|---|
| **Structural** - is the field well-formed? | length (published limits) · completeness (present, not a placeholder or copy) · noise (markup, URLs, encoding debris) · formatting (casing, line breaks, keyword list form) |
| **Semantic** - is it correct and informative? | consistency (supported by the file, coherent with the other fields) · density (specific rather than generic words) · clarity (not cut off, acronyms explained, readable) · usefulness (covers the file's key terms, adds to the title, keywords not repetitive) |
| **Functional** - can search find the KO by it? | BM25 findability · semantic findability (search every KO with the field; where does this KO rank?) |

The same scorer scores three sets of metadata, so they can be compared directly: the
contributor's, the LLM-improved `*_llm` fields already in the export, and stage 5's
proposals.

---

## Outputs

Each run writes a folder, `output/run_<timestamp>/` or `--output-dir`:

| file | one row per | what to look at |
|---|---|---|
| `kos.tsv` | KO | `metadata_score` (contributor) and `metadata_score_llm_fields` (existing `*_llm`), `delta`, pillar and field scores, `weakest` (the lowest sub-pillars with the reason), gate, compliance and content flags |
| `fields.tsv` | KO × metadata set × field | every sub-pillar value |
| `llm_review.tsv` | KO × field (with `--llm-review`) | verdict, failing criteria, proposal, both scores, decision, and `disagreement` where the model says "correct" but the measurement says weak |
| `selftest.tsv` | KO × deliberate defect (with `--selftest`) | whether each sub-pillar noticed |
| `analysis.json` | run | rank stability under alternative weights, redundant sub-pillars, length bias, self-test totals |
| `details.jsonl` | KO | everything, including the evidence behind each sub-pillar |
| `run.json` | run | arguments, selection, settings version, corpus fingerprint, timing |

**Reading a low score.** Start from `weakest`: it names the field, the sub-pillar and the
reason, for example `title.bm25_findability=0.08 ranks 8628 of 10468 when used as a
search query`, or `description.clarity=0.33 unexplained acronyms: GESVAC`.

---

## Layout

| path | what |
|---|---|
| `pipeline.py` | the entry point; selection and orchestration |
| `stages/` | `gate`, `compliance`, `metadata_score`, `metadata_llm`, `analysis` |
| `core/` | `corpus_index` (search index, frequencies, baselines), `metadata_metrics` (the ten sub-pillars, NLI), `content_report` (web-text filters), `llm`, `text_utils`, `io_utils` |
| `validation/` | `metadata_spec.json` (settings), `compliance_profile.json`, `gate_validation/`, `runs/` (evidence such as the NLI comparison) |
| `tools/` | `score_metadata` (scoring and self-test on their own), `compare_nli` (NLI model choice), `md_to_docx` (the Word guide), older standalone analyses |
| `legacy/` | superseded generations, including `reviewer_fitted/` (the score fitted to the 2024 human review); nothing active imports them |
| `docs/` | longer write-ups and the presentation |

---

## Nothing is a hand-written list

| thing | where it comes from |
|---|---|
| length rules | published external sources only: the EIP-AGRI Common Format (title) and Google's Dataset guidelines (description), quoted in `LITERATURE_REVIEW.md` §3.1; none where no source gives a number |
| controlled vocabularies | globbed from `data_model_v2/` |
| required fields | filled on every record (the form enforces them), and contributor-supplied |
| generic vs specific words | platform document frequencies and general-language word frequencies (wordfreq) for each language found |
| stop words | each language's commonest words (wordfreq), as FineWeb2 does |
| "similar enough", "coherent enough" | chance baselines learned from random corpus keywords and KOs |
| scale anchors | the corpus's own 5th-95th percentiles |
| the NLI model | chosen by measurement (`tools/compare_nli.py`) |

---

## Recalibrating

```bash
python pipeline.py --learn --limit 1          # compliance profile (1 LLM call)
python -m stages.gate --calibrate             # domain gate
python -m tools.compare_nli --limit 40        # NLI model
python -m tools.md_to_docx METHODS.md LITERATURE_REVIEW.md \
    --out How_the_KO_quality_pipeline_works.docx --title "How the KO quality pipeline works"
```

---

## LLM access

Configuration and quirks: [`../USING_LLMS.md`](../USING_LLMS.md). `core/llm.py` adapts
`semantic_vs_search/llm_client.py` (disk cache, retries, `reasoning_effort` handling) and
resolves a model preference against the provider's model list. Everything except the
gate's uncertain band and the optional review works offline.

New dependency: `wordfreq` (Apache-2.0 code, CC BY-SA 4.0 data), for general-language
word frequencies. Install with `pip install wordfreq`.

---

## Latest run

Run of 20 KOs, one per language (Bulgarian, Croatian, Czech, Danish, Dutch, Finnish, French, German, Greek, Hungarian, Irish, Italian, Latvian, Lithuanian, Polish, Portuguese, Romanian, Slovene, Spanish, Swedish); seed 42;
`python pipeline.py --limit 20 --selftest`. Output folder: `output/run_final_20ko_v4/`.
No LLM calls beyond the gate's uncertain band (3 KOs, answered from cache).

**Gate and checks.** 0 of 20 rejected by the gate. Compliance: every KO complete. Content
report: 4 files flagged by at least one web-text filter; declared language
differs from the detected one on 5.

**Metadata score** (mean, 0-100): the contributor's metadata against the LLM-improved
`*_llm` fields already in the export, on identical criteria.

| | contributor | existing `*_llm` |
|---|---|---|
| overall | 80.2 | 79.2 |
| structural | 98.8 | 99.6 |
| semantic | 71.5 | 68.5 |
| functional | 70.4 | 69.4 |
| title | 82.5 | 77.1 |
| subtitle (n=3 / 18) | 82.3 | 79.8 |
| description | 79.4 | 80.2 |
| keywords | 79.0 | 80.9 |

The `*_llm` version scored more than a point higher on 8 KOs and lower on 8. It
is stronger structurally and on descriptions and keywords, and weaker on titles and on
the semantic pillar - partly because it rewrote several non-English titles in English.

**Self-test:** 232 of 240 deliberate defects detected
(97%).

| sub-pillar | detected |
|---|---|
| bm25_findability | 19 / 20 |
| clarity | 20 / 20 |
| completeness | 20 / 20 |
| consistency | 40 / 40 |
| density | 18 / 20 |
| formatting | 20 / 20 |
| length | 20 / 20 |
| noise | 20 / 20 |
| semantic_findability | 19 / 20 |
| usefulness | 36 / 40 |

The misses are cases where the original was already at the floor (a title that ranks
8,628th of 10,468 cannot be made less findable) and the cross-language coverage limit
described in METHODS.md §12.

**Robustness** (OECD/JRC step): Spearman correlation of the KO ranking with the
equal-weights ranking under data-driven weights - pillars: entropy 0.953,
CRITIC 0.964, PCA 0.983; sub-pillars: entropy
0.967, CRITIC 0.992, PCA 0.992. Rankings
barely change under other defensible weights. No sub-pillar pair is redundant
(|rho| >= 0.9); the strongest length association is clarity (rho -0.451).

**LLM review** (tested earlier on 10 KOs, under the previous length rules; 9 calls):
14 of 40 fields replaced; 5 proposals rejected as
less faithful to the file and 2 as not measurably better; 11
fields judged correct and kept. Rerun with `--llm-review --llm-limit 10` (10 calls) to
repeat it under the current rules.
