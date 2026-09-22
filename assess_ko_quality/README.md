# assess_ko_quality

Assesses the quality of EU-FarmBook Knowledge Objects.

One entry point, `pipeline.py`, running four stages. Every number it emits states
whether it has been validated against human judgement, because most of what the
previous generation measured had not been, and most of that turned out not to work.

```bash
cd assess_ko_quality

python pipeline.py --input input/kos.json              # assess
python pipeline.py --input input/kos.json --learn      # refit corpus profiles first
python pipeline.py --input input/kos.json --judge      # add the LLM judge (diagnostics)
```

Run everything from inside this folder. Modules import as `core.*` / `stages.*`, so
scripts under a subfolder run as `python -m validation.validate_metrics`.

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
