# assess_ko_quality

Assesses the quality of EU-FarmBook Knowledge Objects (KOs). A KO is a file plus the
metadata that describes it. **The score measures the metadata** - title, subtitle,
description and keywords - against the file's summary (`ko_content_flat_summarised`):
is it correct, informative, well-formed and findable? Every language is assessed by
default. Each field is scored twice, on identical criteria:
- the contributor's version;
- an LLM version written fresh from the summary by the current model (one call per KO;
  `--use-existing-llm` scores the export's existing `*_llm` fields instead, as every
  run did before version 12).

Each field score averages four pillars: three measured ones (structural, semantic,
functional; no LLM) and an LLM's 0-100 score of the field against the summary. The file
itself is the original author's work, so its quality is measured and reported, and never
changes the score.

Every criterion is measurable and reproducible, taken from published research and
standards; nothing is fitted to human ratings. Each part is checked technically: the
pipeline damages metadata on purpose and confirms that the matching measurement notices.

| document | what it covers |
|---|---|
| this README | running the pipeline and reading its outputs |
| [`METHODS.md`](METHODS.md) | every method, threshold and weight, precisely enough to reimplement |
| [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) | where each method comes from, why it was chosen, and the evidence behind every number |
| `How_the_KO_quality_pipeline_works.docx` | the two documents above as a Word guide, generated from them |
| [`METHOD_NO_SOURCES.md`](METHOD_NO_SOURCES.md), [`SOURCES.md`](SOURCES.md) | the method without citations, and the sources on their own (each also as a Word file) |
| `validation/metadata_spec.json` | every configurable setting, with its provenance |

---

## Quick start

```bash
cd assess_ko_quality
python pipeline.py --limit 10                  # 10 KOs: 10 generation calls + up to 80 for the LLM pillar
python pipeline.py --limit 10 --selftest       # plus the perturbation self-test
python pipeline.py --limit 200 --no-llm-score --no-gate-llm --use-existing-llm
                                               # measured pillars only, no LLM calls
```

The first run on an export builds a corpus index: about 20 minutes the very first time
(it embeds every KO once), about 5 minutes after a settings change, seconds otherwise. It
is cached in `output/_corpus_cache/`, and switching the reference text between the
summary and the file rebuilds it. Assessment then takes about 100 seconds per KO on CPU:
98-102 s in the October 2026 runs with the summary as the reference and the LLM answers
mostly cached, and 134 s for two long PDFs whose summaries were rewritten, on an Intel
Core Ultra 9 185H laptop. Nearly all of it is the two transformer models; in a profile of
one KO the entailment model took about a third and the embedding model about half. At
this rate the full export of 11,627 KOs needs about 13 days of CPU time, plus the wait for
any LLM calls not already cached.

**LLM calls.** Only five things can call an LLM:
- **The `*_llm` fields**: one call per KO with a summary, writing all four fields fresh
  from it (the current model, in the document's own language). `--use-existing-llm`
  turns this off and scores the export's `*_llm` fields instead.
- **The LLM pillar**: one call per field that has text, per version, so up to 8 per KO
  (about 7.4 on English KOs, as most have no contributor subtitle). The number is
  printed before any call is made. `--no-llm-score` turns it off, and the field scores
  then average the three measured pillars only.
- **The domain gate**, for the few KOs its classifier is unsure about (typically 0-10%).
  `--no-gate-llm` turns this off.
- **`--resummarise`**: 2-4 calls per KO for a document of up to 12 pages (draft,
  verification, a revision when needed, a translation to English when the document is
  not English), about two more per further 12 pages, plus one transcription call per
  10 minutes of a video's or audio file's sound. The file itself, downloaded from its
  `@id` URL and read as it is; export-text fallback when it cannot be. The estimate is
  printed before any call.
- **`--learn`**: 1 call.

Responses are cached, so rerunning costs nothing.

---

## Choosing what to assess

You rarely want all 10,468 at once. Every filter reports what it removed rather than
dropping it quietly, and the selection is written to the Run sheet of the run's `report.xlsx` so a run
describes itself.

| Flag | Default | What it does |
|---|---|---|
| `--limit N` | 0 (all) | Assess at most N records |
| `--min-summary-words N` | 50 | Skip KOs whose summary is shorter than this (0 = no filter) |
| `--file-type` | all | `pdf`, `mp4`, `Video`, `Factsheet`, … (comma list) |
| `--languages` | all | `English,French` (comma list); `all` (the default) for every language |
| `--lang-balance` | equal | How `--limit` is spread across languages |
| `--licenses` / `--exclude-licenses` | all | Keep only / drop these licences, e.g. `CC BY-NC-ND` |
| `--upload-sources` | all | Keep only these `ko_upload_source` values, e.g. `upload_form` (others: `public_api`, `ai_uploader`) |
| `--seed` | 42 | Sampling seed, so a run repeats |
| `--no-gate-llm` | off | Classifier-only gate; with `--no-llm-score` and `--use-existing-llm` too, no LLM calls at all |
| `--use-existing-llm` | off | Score the export's `*_llm` fields as the LLM variant instead of writing them fresh from the summary (the pre-version-12 behaviour) |
| `--no-llm-score` | off | Leave the LLM pillar out: the field scores average the three measured pillars |
| `--resummarise` | off | Regenerate each assessed KO's reference summary with the current vision model from the file itself, **in the document's own language** (downloaded from `@id` and read as it is: pages as images, office files converted, a video's sound transcribed and its frames shown; long documents summarised in parts and merged; every draft verified against its pages and revised; a non-English summary also translated to English for the search index; export-text fallback, basis recorded per KO). That summary becomes the reference the fields are measured against; the summaries are stored in the outputs. Not comparable with runs on the export's summaries |
| `--ids` | all | Comma list of `_orig_id` values to assess (debugging; the other filters still apply) |
| `--no-summary-check` | off | Skip stage 3b, the compression/coverage/NLI reading on the reference summary |
| `--generate-model` | spec (`''` = the client's default) | Model that writes the `*_llm` fields; name it in full as the provider lists it |
| `--judge-model` | spec (`glm-5.2`) | Model for the LLM pillar; never the generator's family (self-preference) — the pipeline warns |
| `--resummarise-model` | spec (`''` = the client's default) | Model for `--resummarise` |

```bash
# 100 records, spread evenly across every language (the default scope)
python pipeline.py --limit 100

# 50 English PDFs
python pipeline.py --limit 50 --file-type pdf --languages English

# every video, including ones without a usable summary
python pipeline.py --file-type mp4 --min-summary-words 0

# score against fresh summaries of the files themselves instead of the export's (1 LLM call per KO)
python pipeline.py --limit 50 --resummarise
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

**Every language is in scope (the default is `all`).** The platform is multilingual and
so is the measurement stack (multilingual E5, XLM-RoBERTa XNLI, wordfreq in each corpus
language). One structural caveat remains: the fields are measured against the summary,
and for 82% of 2,308 non-English files checked the summary is in English, so a
non-English contributor's fields are measured cross-lingually. Report non-English
results per language rather than pooled with English; `--languages English` restores
the old single-language scope. (English-only was the default until October 2026.)

### `--lang-balance`: equal or proportional

Only relevant when `--limit` is set **and** more than one language is in play
(the default `all`, or a list). With a single-language filter there is nothing to
balance between.

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

### The summary floor

`--min-summary-words 50` is the default. It keeps the KOs whose summary
(`ko_content_flat_summarised`) has at least 50 words and excludes **655 of 10,468 (6.3%)**:
300 with a shorter summary, 284 whose summary is just "No content present", and 71 with
none. For every one of those with a summary (584 of 584), all four `*_llm` fields are
copies of the contributor's. The count is always printed; `--min-summary-words 0`
includes them.

The summary is also what the fields are measured against, and what the LLM reads. A KO
without one cannot be measured on the semantic and functional sub-pillars or given an
LLM score, whether or not its text extracted.

---

## The pipeline

```
select ─► 1 GATE ─► 2 COMPLIANCE ─► 3 CONTENT ─► 4 METADATA SCORE (3 measured pillars + LLM score) ─► 5 ANALYSIS
```

| stage | question | affects the score |
|---|---|---|
| 1 gate | Is the KO about agriculture in the broad sense (farming, forestry, livestock, food, rural life, and their economics, environment and society)? | off-domain KOs are not scored |
| 2 compliance | Are the fields every record carries filled in, with values from the current vocabularies? Optional fields are never a failure; old vocabulary values are reported as drift; age is reported, not failed. | no |
| 3 content | What is the file itself like? The published web-text quality filters (Gopher, C4, FineWeb, FineWeb2), declared vs detected language, lexical diversity, readability, DOIs and links. | no |
| 4 metadata score | How well do title, subtitle, description and keywords describe the file, measured against its summary, for the contributor's version and an LLM version written fresh from that summary (`--use-existing-llm`: the export's `*_llm` fields)? Three measured pillars plus an LLM's 0-100 score per field; an empty field scores 0. The better version of each field is recommended. | **the score, 0-100** |
| 5 analysis | Why is a score low, and would the ranking change under other weights? | no |

### The score

Each field is scored on three measured pillars (ten sub-pillars) and an LLM's score, and
the four are averaged with equal weights:

| pillar | sub-pillars |
|---|---|
| **Structural** - is the field well-formed? | length (published limits) · completeness (present, not a placeholder or copy) · noise (markup, URLs, encoding debris) · formatting (casing, line breaks, keyword list form) |
| **Semantic** - is it correct and informative? | consistency (supported by the summary, coherent with the other fields) · density (specific rather than generic words) · clarity (not cut off, acronyms explained, readable) · usefulness (covers the summary's key terms, adds to the title, keywords not repetitive) |
| **Functional** - can search find the KO by it? | BM25 findability · semantic findability (search every KO's summary with the field; where does this KO rank?). A field not in the reference's language skips BM25 findability (not evaluable - lexical search cannot cross languages); semantic findability is multilingual |
| **LLM score** - what does an LLM make of it? | one call per field per version: the summary, that one field and the criteria above go in; a 0-100 score and a one-sentence reason come out (glm-5.2, which wrote neither version) |

The same scorer scores the contributor's metadata and the LLM version, so they can be
compared directly. The LLM version is written blind by the current model - it sees the
summary and the published field rules only, never the contributor's fields - in the
document's own language (`file_language`, read from the pages during `--resummarise`;
the judge is told that language and asked not to penalise it). With `--use-existing-llm` it is instead the `*_llm`
fields already in the export, which the platform's improver wrote from the summary *and*
the contributor's fields. For each field the better of the two is recommended: the
higher field score, unless it is less faithful to the summary than the contributor's,
by the measured accuracy (METHODS.md §5.5).

**The LLM score is not validated.** LLM judges may favour LLM-written text, which here is
the `*_llm` version; in this project's earlier tests, judges never once preferred the
contributor's version (METHODS.md §6). It is one pillar of four for that reason.

**Measured against the summary.** The LLM version is written from the summary (and the
export's `*_llm` fields were too), so measuring against the summary favours it. On
English KOs, word overlap with the summary favoured the existing `*_llm` titles by +0.11
over the contributor's. Against the file itself it was −0.02. Report results as measured
against the summary (METHODS.md §12).

---

## Outputs

Each run writes a folder, `output/run_<timestamp>/` or `--output-dir`:

Everything you need to read is in **one file, `report.xlsx`**:

| sheet | one row per | what to look at |
|---|---|---|
| Run | run | arguments, selection counts, what the fields were measured against, models, the LLM pillar's model and calls, self-test and rank-stability results |
| KOs | KO | the score of each version (contributor, the LLM one), the contributor's four pillars and field scores, the recommended version per field, gate / compliance / content flags, and the weakest parts with the reason for each |
| Fields | KO × field × version | the text, the field score, the four pillars (the LLM score next to the three measured ones), all 10 sub-pillars, faithfulness to the summary, whether it is the recommended version, and the LLM's reason |
| Summaries | KO | the reference summary the fields were measured against, its source and language; with `--resummarise` also its basis (file kind, pages, chunks, calls, what the verification found and, for a document longer than one chunk, how many sentences got no verdict), its English version and the platform's summary it replaced |
| Self-test | deliberate defect (with `--selftest`) | whether each sub-pillar noticed |

`details.jsonl`, written next to it, is the raw archive with every piece of evidence,
for code rather than for reading.

**Reading a low score.** Start from the `weakest` column of the KOs sheet: it names the field, the sub-pillar and the
reason, for example `title.bm25_findability=0.08 ranks 8628 of 10468 when used as a
search query`, or `description.clarity=0.33 unexplained acronyms: GESVAC`. A low LLM
score appears the same way, as `title.llm=...` with the LLM's reason.

---

## Layout

| path | what |
|---|---|
| `pipeline.py` | the entry point; selection and orchestration |
| `stages/` | `gate`, `compliance`, `metadata_score`, `llm_score` (the LLM pillar), `analysis`, `report` (`report.xlsx`) |
| `core/` | `corpus_index` (the reference text, search index, frequencies, baselines), `metadata_metrics` (the ten sub-pillars, NLI), `content_report` (web-text filters), `llm`, `text_utils`, `io_utils` |
| `validation/` | `metadata_spec.json` (settings), `compliance_profile.json`, `gate_validation/`, `runs/` (evidence such as the NLI comparison) |
| `tools/` | `score_metadata` (scoring and self-test on their own), `compare_nli` (NLI model choice), `md_to_docx` (the Word guides), `build_flow_docx` (the flow diagram), older standalone analyses |
| `legacy/` | superseded generations, including `reviewer_fitted/` (the score fitted to the 2024 human review) and `metadata_llm.py` (the removed stage 5); nothing active imports them |
| `docs/` | longer write-ups and the presentation |

---

## Nothing is a hand-written list

| thing | where it comes from |
|---|---|
| length rules | published external sources only: the EIP-AGRI Common Format (title) and Google's Dataset guidelines (description), quoted in `LITERATURE_REVIEW.md` §3.1; none where no source gives a number |
| controlled vocabularies | globbed from `data_model_v2/` |
| required fields | filled on every record (the form enforces them), and contributor-supplied |
| generic vs specific words | platform document frequencies and general-language word frequencies (wordfreq) for each language found |
| stop words | each language's 8 commonest words (wordfreq), adapted from FineWeb2 |
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
resolves a model preference against the provider's model list; the shortest matching id
wins, which is why the LLM pillar's model is named in full (`glm-5.2`,
`llm_score.model` in `validation/metadata_spec.json`); choose a model that wrote neither
version. Everything except the LLM pillar and the gate's uncertain band works offline.

New dependency: `wordfreq` (Apache-2.0 code, CC BY-SA 4.0 data), for general-language
word frequencies. Listed in the repo's `requirements.in` / `requirements.txt`.

---

## Latest run

**This run predates the current rules.** It was made under settings version 7:
- measured against the file, not the summary;
- every language;
- the existing `*_llm` fields compared;
- an empty subtitle left out rather than scored 0.

Under version 8's empty-field rule its overall means become about 63.3 (contributor) and
77.5 (existing `*_llm`); 17 of the 20 contributors left the subtitle empty. No full run
has been made yet under version 12 (the summary as the reference, every language, the
LLM pillar as a quarter of every field score, the LLM version written fresh from the
summary).

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

**Recommended versions** (10 English KOs, `output/llm_blind_en10/`, 40 fields): choosing
between the contributor's and the existing `*_llm` version gives recommended fields a mean
score of 81.8 and faithfulness 0.883, with no LLM call. Adding a new glm-5.2 version gives
84.4 and 0.892 at one call per KO. Details in METHODS.md §5.5 and §6.
