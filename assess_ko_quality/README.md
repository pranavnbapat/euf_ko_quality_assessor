# assess_ko_quality

Assesses the quality of EU-FarmBook Knowledge Objects.

One entry point, `pipeline.py`, running four stages. Every number it emits states
whether it has been validated against human judgement, because most of what the
previous generation measured had not been, and most of that turned out not to work.

```bash
cd assess_ko_quality

python pipeline.py --input ../input/kos.json              # assess
python pipeline.py --input ../input/kos.json --judge      # add the LLM judge
python pipeline.py --input ../input/kos.json --learn      # refit corpus profiles first
```

Run everything from inside this folder. Modules import as `core.*` / `stages.*`, so
scripts under a subfolder run as `python -m validation.validate_metrics`.

---

## Choosing what to assess

You rarely want all 10,468 at once. Every filter reports what it removed rather than
dropping it quietly, and the selection is written to `output/selection.json` so a run
describes itself.

| Flag | Default | What it does |
|---|---|---|
| `--limit N` | 0 (all) | Assess at most N records |
| `--min-chars N` | 500 | Skip KOs with less extracted text than this |
| `--file-type` | all | `pdf`, `mp4`, `Video`, `Factsheet`, … (comma list) |
| `--languages` | all | `English,French` (comma list) |
| `--lang-balance` | equal | How `--limit` is spread across languages |
| `--seed` | 42 | Sampling seed, so a run repeats |
| `--no-gate-llm` | off | Classifier-only gate; no LLM calls at stage 0 |

```bash
# 100 records, spread evenly across languages, real text only
python pipeline.py --input ../input/kos.json --limit 100

# 50 English PDFs
python pipeline.py --input ../input/kos.json --limit 50 --file-type pdf --languages English

# every video, including ones whose text never extracted
python pipeline.py --input ../input/kos.json --file-type mp4 --min-chars 0
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

A proportional run of 40 touches 10 of your 21 languages and gives German nothing — you
would finish it knowing nothing about how the pipeline handles German. The equal run
covers all 21.

Use **equal** (the default) to check the pipeline or compare across languages. Use
**proportional** when the aggregate is the point, such as "what is the average quality
of the platform" — an equal sample over-weights small languages and describes no real
population. Do not read corpus statistics off an equal sample.

### The 500-character floor

`--min-chars 500` is the default and it excludes **35.9% of the corpus** — the KOs whose
text never extracted. That is right for assessing quality, since there is nothing to
assess, but those records are a real finding about ingestion rather than records that do
not matter. The count is always printed. Pass `--min-chars 0` to include them.

---

## The four stages

```
  KO
   │
   ├── STAGE 0  GATE ........... is this in scope at all?
   │                             accept / review / reject · off-domain KOs are not scored
   │
   ├── STAGE 1  COMPLIANCE ..... obligations, as a pass/fail checklist
   │                             deliberately NOT part of the score
   │
   ├── STAGE 2  SCORE .......... the one validated number, 0-100
   │                             content depth + lexical variety
   │
   └── STAGE 3  DIAGNOSTICS .... everything else, as "unusual for this corpus"
                                 plus LLM judge wording, never as a score term
```

### Why relevance is a gate, not a pillar

Domain relevance used to be 10% of a weighted total. Three off-domain documents written
to be structurally strong — Bach's Brandenburg Concertos, asynchronous server handlers,
atrial fibrillation — scored **75.2 to 75.8**, above *every* real agricultural KO in the
same run, with the domain pillar switched on. At 10% weight the most an irrelevant
upload can lose is 10 points, so a polished off-topic document always passes.

Relevance is a precondition, not a quality you can trade against good formatting.

The old measure could not have caught them anyway: cosine to the AGROVOC/NALT centroid
gave a crop-rotation guide 0.071 and literal gibberish 0.069. Averaging 96,691 thesaurus
concepts produces a vector pointing nowhere in particular.

| gate method | AUC |
|---|---|
| cosine to anchor centroid (old) | 0.647 |
| mean of top-10 anchor similarities | 0.768 |
| **logistic regression on embeddings** | **1.000** |

The gate is a **hybrid**: that classifier decides every confident case, and an LLM
decides only the uncertain band, `[0.20, 0.70)`. Real KOs almost never land there — 98.7%
of even the Economics/Society/Environment-only ones are accepted outright — so on a
real corpus the LLM sees a small minority of records. Its scope comes from
`data_model.topics.json`. Use `--no-gate-llm` for a classifier-only gate with no LLM
calls. Details and the prompt's failure history are in `METHODS.md` section 1.

The 10,468 real KOs define the domain far better than a thesaurus does. At the
calibrated threshold: **100% of real KOs kept in all 14 languages**, 94.6% of
off-domain caught. It is not set higher because a language gap opens above 0.60 —
at 0.70, German retention falls to 83% and Greek to 33%, and roughly a third of the
corpus is not in English.

### Why compliance is not in the score

As a scored component, metadata completeness correlated with reviewer judgement at
**-0.217** (p=0.039) — the wrong direction. A thorough record is not the same thing as
good content. Completeness still matters, as an obligation, so it is a checklist.

### What the score is made of

Of sixteen sub-scores in the previous design, eleven had no relationship with reviewer
judgement, one ran backwards, and the rest were dominated by two measurements.

| scorer | ρ vs human | % of ceiling |
|---|---|---|
| old four-pillar total | 0.174 | 21% |
| four pillars, *optimally* reweighted | -0.170 | below chance |
| **content depth + lexical variety** | **0.526** | **62%** |

Ridge regression given free choice of weights over the four pillars scored below
chance, so the failure was never the 30/35/25/10 split — it was the contents.

Two specifics worth keeping in mind when changing anything here:

- **Type-token ratio was inverted.** It falls mechanically with length (ρ = -0.93), so
  93.7% of the corpus scored 5/5 and failed extractions scored best. Replaced with
  MTLD, which is length-robust: -0.29 → **+0.47**.
- **Banding destroys signal.** Discretising content length into 0-5 bands lost 53% of
  it, because the band table penalised documents over 6,000 tokens while reviewers
  consistently preferred longer ones. The score uses continuous inputs.

The ceiling is **0.847**: two reviewers agree with each other at ρ 0.558, and no scorer
can exceed the square root of the criterion's reliability.

---

## Layout

| Path | What |
|---|---|
| `pipeline.py` | The entry point. Everything else is a component of it. |
| `core/` | Measurement. `text_utils`, `io_utils`, `llm`, and the four metric modules. |
| `stages/` | `gate`, `compliance`, `scoring`, `diagnostics`, `llm_judge`. |
| `validation/` | `validate_metrics`, `human_review`, learned profiles, gate model. |
| `tools/` | Standalone analyses: content and compression diagnostics, anchor builders. |
| `legacy/` | The superseded generation, frozen and self-contained. Do not extend. |
| `docs/` | Longer write-ups. |
| `data/` | Human review workbook and other local inputs. |
| `anchors/` | AGROVOC / NALT anchor texts and the old centroid. |

---

## Nothing here is a hand-written list

| Thing | Where it comes from |
|---|---|
| Controlled vocabularies | Globbed from `data_model_v2/`; field name from the filename |
| Which fields are obligations | Corpus coverage, then an LLM classifies contributor / system / derived |
| Staleness cut-off | A percentile of the corpus's completion years |
| Diagnostic thresholds | Per-metric tails learned from the corpus |
| The judge's rubric | The question texts in the human review workbook itself |
| Which LLM is used | `GET /v1/models`, matched against a preference expression |
| Score weights | Fitted against human review; refit with `validation/validate_metrics.py` |

Refit the corpus-derived profiles with `--learn` whenever the export changes.

---

## Validating and recalibrating

```bash
python -m validation.validate_metrics                       # metrics vs human review
python -m validation.validate_metrics --calibrate --kos ../input/export.json
python -m stages.gate --calibrate                           # refit the domain gate
```

`validate_metrics` reports the inter-rater ceiling, per-metric correlations with
partial-on-length, a cross-validated feature-set comparison against a permutation
baseline, and robustness across content-length floors.

**The rule this folder is built on:** nothing enters the score until it has been
measured against human judgement. The LLM judge is a good instrument — gpt-oss-120b
reaches ρ 0.491 with no fitting at all — but it correlates 0.65 with the existing
signal and adds 0.009 when combined, so it is reported as diagnostics, not scored.
A bigger model is not an exemption from validation.

---

## LLM access

Configuration and quirks are documented in [`../USING_LLMS.md`](../USING_LLMS.md).
`core/llm.py` adapts `semantic_vs_search/llm_client.py` (disk caching, retries,
`reasoning_effort` handling) and adds runtime model discovery. Set `KOQ_LLM_MODEL` to a
preference expression such as `"gpt-oss"` to pin a family without pinning a checkpoint.

Everything degrades without an LLM: the gate, the score and the corpus-derived
thresholds all work offline. Only field-role classification and the judge need one.
