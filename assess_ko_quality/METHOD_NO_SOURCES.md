# The KO quality pipeline: method

Every step, rule, threshold and weight the KO quality pipeline uses, precisely enough to
reimplement. The sources behind them are kept in a separate document, `SOURCES.md`
(Word: `KO_quality_pipeline_sources.docx`); `README.md` says how to run the pipeline.

**What is scored.** A Knowledge Object (KO) is a file plus the metadata that describes
it. The score measures the **metadata** - title, subtitle, description, keywords -
against the file's **summary** (`ko_content_flat_summarised`, Section 5.1): is it
correct, informative, well-formed and findable? Every language is assessed by default,
and results are reported per file language, never pooled across languages.
Each field is scored twice, on identical criteria: the contributor's version and an
LLM version written fresh from the summary each run (Section 4d; `--use-existing-llm`
scores the export's `*_llm` fields instead). A field score averages four pillars: three
measured ones (ten sub-pillars, no LLM) and an LLM's 0-100 score of the field against
the summary (Section 6). The file's own quality is the original author's work; it is
measured and reported, and never changes the score.

**How it is checked.** Only measurable, reproducible criteria are used. Human reviewer
ratings are not used to fit or validate anything. Validation is technical: every
sub-pillar must detect a deliberate defect in its own property (Section 9).

---

## 0. The pipeline at a glance

```
select ─► 1 GATE ─► 2 COMPLIANCE ─► 3 CONTENT (+ 3b SUMMARY CHECK) ─► 4 METADATA SCORE (write the LLM version, then 3 measured pillars + LLM score) ─► 5 ANALYSIS
```

| stage | question | output | affects the score | LLM calls |
|---|---|---|---|---|
| select | which KOs to assess | Run sheet | - | 0 |
| 1 gate | is the KO about agriculture in the broad sense? | `gate_*` | rejected KOs are not scored | only KOs in the uncertain band (typically 0-10%) |
| 2 compliance | are the platform's required fields filled, with current vocabulary values? | `compliance_*` | no | 0 |
| 3 content | what is the file itself like? | `content_*` | no | 0 |
| 3b summary check | is the reference summary itself a reliable instrument? | `summary_*`, a `reference_reliability` verdict | no | 0 (one vision call per KO with `--resummarise`, Section 4c) |
| 4 metadata | how well does each version of the metadata describe the file, measured against its summary, and which version of each field is better? | `metadata_score`, the four pillars, fields, sub-pillars, the LLM's score and reason, the recommended version per field | **this is the score** | one generation call per KO (the LLM version, Section 4d), then one judge call per field with text, per version: up to 9 per KO |
| 5 analysis | why is a score low, and how much do the choices matter? | KOs and Run sheets | no | 0 |

---

## 1. Selection

| flag | effect |
|---|---|
| `--limit N` | at most N KOs |
| `--min-summary-words` (50) | skip KOs whose summary is shorter; the summary is what the fields are measured against, and below 50 words all four `*_llm` fields in the export are copies of the contributor's (584 of 584); 6.3% of the export |
| `--file-type` | matched against category, extension, mimetype and subcategory ("pdf", "Video", "Factsheet") |
| `--languages` (all) | a KO's language is the **first** one listed. Every language is assessed by default; because the reference summary is English, non-English results are reported per file language and never pooled or ranked across languages. A filter (e.g. `en`) restricts the run |
| `--licenses`, `--exclude-licenses` | keep only, or drop, these licence values (e.g. `CC BY-NC-ND`) |
| `--upload-sources` | keep only KOs with these `ko_upload_source` values, e.g. `upload_form` (24 September 2026 export: upload_form 7,155, public_api 2,477, missing 1,963, ai_uploader 28); the search corpus stays the whole export |
| `--lang-balance equal` (default) | round-robin across languages, so a 20-KO run is not 60% English |
| `--seed` (42) | reproducible sampling |

---

## 2. Stage 1 - Domain gate

**A classifier for every KO, an LLM only where the classifier is unsure.** The gate is
a precondition, not a quality dimension: a polished off-topic upload must not outrank
real content.

| property | value |
|---|---|
| Classifier | logistic regression on `all-mpnet-base-v2` sentence embeddings of the first 2,000 characters |
| Training / validation | `validation/gate_validation/` - 678 items: real KOs vs off-domain documents (Wikipedia, five languages) |
| Performance | AUC 1.000 (5-fold CV); at the accept threshold 100% of real KOs kept in all 14 languages tested, 94.6% of off-domain caught |
| Accept / reject | P ≥ 0.40 accept; P < 0.20 reject |
| LLM band | P in [0.20, 0.70) is decided by an LLM (`KOQ_GATE_MODEL`, default glm): only the cases the classifier is unsure of are deferred |
| LLM scope | read from `data_model.topics.json` (Forestry, Livestock, Crop farming, Economics, Environment, Society), with Economics/Environment/Society in scope when they are *of* agriculture, food, forestry or rural life |
| Fallback | if the LLM fails, the classifier's thresholds decide; the gate never drops a KO silently |

Measured on 140 labelled items: the classifier alone accepted 62% of near-miss documents
(packaging, logistics, urban planning); the LLM rejected them; on the uncertain band the
LLM was right 12 of 12 times. 98.7% of Economics/Society/Environment-only KOs are
accepted by the classifier outright.

The LLM prompt failed twice before it worked, and both clauses exist because of it:
"judge the SUBJECT only" (otherwise it judged document type and rejected Wikipedia's
*Dairy farming*), and "Economics, Environment and Society OF agriculture" (otherwise it
accepted VAT law and cement). Re-run the comparison after changing the prompt: a failing
LLM gate fails silently.

Rejected method: cosine similarity to an agricultural-thesaurus centroid (AUC 0.647) -
it scored gibberish (0.069) like a crop-rotation guide (0.071).

---

## 3. Stage 2 - Compliance

Obligations, as a pass/fail checklist next to the score and never inside it.

| property | value |
|---|---|
| Profile | `validation/compliance_profile.json`, learned from the export (`--learn`) |
| Required fields | filled on **100%** of records and classified as contributor-supplied: category, date_of_completion, date_of_completion_source, description, keywords, ko_resource_language, languages, license, project_acronym, project_name, project_type, title |
| Optional fields | creators, intended_purposes, locations, subcategories, themes, topics - filled on 97.9-99.9% of records, i.e. the form does not enforce them. **Never a failure**; if present, fine |
| Field roles | an LLM classified each well-covered field as contributor / system / derived (stored in the profile); `title_llm` or `_id` can never fail a KO |
| Vocabularies | every `data_model.*.json` in `data_model_v2/` |
| Vocabulary drift | values from a superseded vocabulary (58% of records use old `intended_purposes` values) are reported in `compliance_vocab_drift`, not failed: the contributor chose from the list they were shown |
| Age | `compliance_year`, and `compliance_older_than_corpus_p5` (before 2016). Reported only: an old document is not a missed obligation, and age alone does not say content is outdated |

The data model does not declare which fields are mandatory; "filled on every record" is
the observable trace of a field being enforced at upload.

---

## 4. Stage 3 - Content report (reported, never scored)

Text-quality filters of the kind used to clean web text, with fixed thresholds. A flag
means "this text would be removed by this filter".

| filter | rule (flag when) |
|---|---|
| Document quality | < 50 or > 100,000 words; mean word length outside 3-10; symbol/word ratio > 0.1; > 90% bullet lines; > 30% lines ending in an ellipsis; < 80% words with a letter; < 2 of the language's 8 commonest words |
| Repetition | duplicate lines > 30% or their characters > 20%; same for paragraphs; top 2/3/4-gram character share > 0.20/0.18/0.16; duplicated 5-10-gram character share > 0.15 ... 0.10 |
| Line structure | lines ending in punctuation ≤ 12%; characters in duplicated lines ≥ 10%; lines under 30 characters ≥ 67% |
| Boilerplate | "lorem ipsum"; a curly bracket; lines mentioning javascript |
| Stop words per language | the language's 8 commonest words (wordfreq) replace a fixed English list |
| Declared vs detected language | first declared language differs from the detected one |

Also reported: word count, MTLD lexical diversity (50+ words), LIX readability (100+
words), counts of DOIs, URLs and distinct years mentioned (provenance cues),
`content_extraction_failed` (< 50 words).

### 4b. Stage 3b - Summary reference check (reported, never scored)

`stages/summary_check.py` (`--no-summary-check` skips it). Every field score is
measured against the KO's summary, so a bad summary silently degrades every reading
for that KO. This stage measures the summary itself against the file text and reports
a reliability reading on the reference; it never enters the score. Per KO: the
compression ratio (summary words / file words), key-term coverage (share of the
file's TF-IDF key terms found literally in the summary), NLI support (up to 6 summary
sentences, each read against its 2 best-matching file chunks; mean P(entailment)),
and the summary's language (the platform writes summaries in English by design;
anything else is flagged). There are no absolute thresholds: the run's own 5th-95th
percentiles flag outliers, anchored **within each file-language group of at least 30
checked KOs** (run-wide fallback, the anchor recorded per KO) - cross-lingual NLI and
coverage readings run systematically lower, so a run-wide percentile would flag
non-English KOs for being non-English. Flags and a `reference_reliability` verdict
(`ok`/`check`) appear on the KOs sheet; a flagged KO's scores read against a
questionable reference.

### 4c. Regenerating the reference summary (`--resummarise`, opt-in)

`stages/resummarise.py`. The export's summaries were written by the platform's
summariser of their day from the platform's own extraction, and that stored
extraction can be wrong (one Lithuanian factsheet carries an English *translation*
as its "extracted text"). With `--resummarise`, each assessed KO gets a fresh summary
**from its own file, in the document's own language**, and that summary becomes the
reference every field of the KO is measured against: a French KO's fields are judged
in French end to end. An **English version** of it (one translation call for a
non-English document, none for an English one) is the KO's row in the corpus index,
which stays English like every other row, and the secondary reference for
English-language fields on a non-English document (5.2). The summary, its English
version and the platform's summary it replaced are stored with the run (`summary` in
`details.jsonl`; the Summaries sheet of `report.xlsx`). The flag is never the
default: without it the export's summary is used as it is.

**The file as it is.** The file is downloaded from the `@id` URL (cached locally; up
to 600 MB) and read as what it is, with no page cap: a PDF or an image as its pages
rendered to images, one page per image (no text extraction anywhere; the endpoint
rejects PDFs sent natively); an office document (doc/docx/ppt/pptx/xls/xlsx/odt/odp/
ods/rtf) converted to PDF by LibreOffice first; a video with its sound transcribed
(in pieces of 10 minutes, one transcription call each) and 12 frames sampled evenly
across the recording shown next to the transcript; an audio file as its transcript.
A KO whose file cannot be fetched or read falls back to the export's text, and a file
flow that fails retries once from the export's text; the basis of every summary is
recorded per KO (`resummarised.from`, with the kind, pages, chunks, calls and
findings) and counted on the Run sheet.

**How the summary is written.** A document of up to 12 pages (the provider's
images-per-prompt limit), a transcript with frames, or a text is drafted in one call.
A longer document is summarised in parts: detailed notes per chunk of 12 pages,
merged eight at a time until one summary remains (hierarchical merging; coherence
over the extra detail an incrementally updated summary gives). Every draft is then
**verified against its source**. A source of one chunk is checked in one call, which
lists the statements the source contradicts or does not support and the major content
the draft lacks. A longer document is checked one chunk per call, sentence by
sentence, because one chunk cannot tell a wrong statement from one about other pages:
each chunk gives every numbered sentence of the draft a verdict - supported,
contradicted, or absent from these pages - and the decision is taken across chunks. A
sentence is kept when any chunk supports it, is an error when a chunk contradicts it
and none supports it, and is unsupported when every chunk reports it absent. One
revision call then applies every finding; with no findings there is no revision call.
Last, a non-English summary is translated to English for the corpus row. The prompts
keep the platform summariser's brief - every major topic, entities and numbers
preserved, proportional length - and ask for the document's language, which the
model names from the pages (`file_language`, on the KOs sheet next to the declared
languages; reported, never scored). The declared languages are contributor input and
the stored text can be a translation, so the file itself is the only trusted source
for what the document is written in.

Calls per KO: 2-4 for a document of up to 12 pages, about two more per further chunk
plus the merges; one transcription call per 10 minutes of audio. The number is
estimated and printed before any call; every answer is cached. The KO's own
findability row is recomputed from the English version while corpus statistics
remain the export's (fine for samples; a whole-corpus rerun should rebuild the
index). Runs with and without the flag are not comparable, nor are they comparable
with earlier versions' (English reference, one call, 50-page cap). The model
(`--resummarise-model`) must be vision-capable.

### 4d. Writing the LLM version of the fields (on by default)

`stages/generate_metadata.py`. The pipeline writes the LLM variant itself: one call
per KO with a summary - counted before any call, cached - after `--resummarise` (so
the fields are written from the very summary they are measured against) and before
the LLM pillar. Generation is **blind**: the model sees the reference summary and the
published field rules only, never the contributor's fields, so the LLM version is an
independent rewrite rather than an edit anchored to the contributor's wording; the
rules in the prompt are the same published limits the structural pillar measures. The
brief caps keywords at 10 (a brief choice; the count is never scored). Each field is
written in **the document's own language**: `file_language` read from the pages
during `--resummarise`, else the detected language of the stored text - never the
contributor's declared languages, which are input under test, not an instrument. The
summary the fields are written from is the platform's English one or - with
`--resummarise` - the fresh one in the document's own language (4c), so the fields
and the summary then share a language; the prompt names the summary's language, and
the judge is told the field's language and not to penalise it (Section 6).

The generator (empty model = the default client's, qwen3.5 at the time of writing)
must be a different model from the LLM pillar's judge, because LLM judges favour
their own model's text; the pipeline warns when they match. A KO whose summary is
empty, or whose call fails, keeps the export's `*_llm` fields and is marked per KO.
`--use-existing-llm` restores the pre-version-12 comparison against the export's
`*_llm` fields (written by the platform's improver from the summary **and** the
contributor's value) - and, with `--no-gate-llm` and `--no-llm-score`, the
no-LLM-calls path. LLM-version columns are not comparable across that switch: a
different author wrote the fields.

---

## 5. Stage 4 - Metadata score

### 5.1 What the fields are checked against

**Length rules.** Title: at most 150 characters, spaces not counted. Description: 50 to
5,000 characters. Subtitle and keywords: no length or count is imposed. The first
version's ranges (title 5-12 words, description 40-80 words, keywords 4-10) were removed.

**The reference text** (`reference.source`, a choice): the KO's summary,
`ko_content_flat_summarised`, and nothing else - or the fresh file-based summary with
`--resummarise` (4c). The LLM version of the fields is written from the same summary
(4d), so both versions are measured against the text it was written from. The summary
is not the file. For 86% of KOs it is
a light polish of the vision model's own summary of the file (`ko_content_flat_vision`),
and it stays near 200 words whatever the file's length. Fields are scored only when the
summary has at least 200 characters; otherwise the semantic and functional sub-pillars
are "not evaluable" and left out. With `--resummarise`, the reference is the fresh
summary in the document's own language, and its English version is the KO's row in
the corpus index and the secondary reference for English-language fields on a
non-English document (5.2); the summary used is stored with every run.
`reference.source: "file"` restores the reference used
up to version 8: the vision reading followed by the extracted text (`ko_content_flat`),
cut at 100,000 characters. Section 11 gives the effect of the choice.

**The corpus index** (`core/corpus_index.py`, cached in `output/_corpus_cache/`): a BM25
index and E5 passage embeddings of every KO's reference text (never its metadata, so a
title cannot find itself), corpus document frequencies, general-language word
frequencies (wordfreq) for every detected language, per-field value-sharing counts,
null distributions for the chance baselines, and anchors for the scales below.

**Models**: `intfloat/multilingual-e5-small` for embeddings (a larger model of the same
family is a configuration change), and `joeddav/xlm-roberta-large-xnli` for natural
language inference (NLI), **chosen by measurement** (Section 9.2).

### 5.2 The ten sub-pillars

Every value is in [0, 1]. "Chance" baselines and scale anchors are learned from the
corpus, never typed in.

| pillar | sub-pillar | measures | how |
|---|---|---|---|
| **Structural** | length | within the length rules (5.1) | each limit 1 when met, linear to 0 at 2× a maximum or at 0 below a minimum; the lowest limit counts; not evaluated where there is no limit |
| | completeness | present and not a placeholder | empty → 0; otherwise the share of: not a filename, not a copy of another field, not shared verbatim by KOs of other projects; for titles also not an internal project code (D3.1, T2.3, WP2, "Task 4") and not the file name |
| | noise | debris in the field | 1 − share of tokens that are markup, URLs, encoding debris, symbol runs or immediate repeats; rich-text HTML is converted to text first |
| | formatting | field-specific form | share of checks passed (casing, line breaks, no dangling comma/colon at the end; not all capitals for two words or more; description starts capitalised and ends a sentence; keywords a list, one term per item, no duplicates, none longer than the corpus 99th percentile) |
| **Semantic** | consistency | correct about the file (as its summary states it), and coherent with the other fields | mean of **accuracy** - each claim vs its 2 best-matching passages of the summary, max(P(entailment) − P(contradiction)), mean over claims, mapped to [0,1]; keywords: share present in the summary or closer to it than 95% of random keywords - and **coherence** - share of 300 random KOs whose same field is less similar to this field than this KO's own |
| | density | specific rather than generic words | mean over words of min(platform IDF, 1 − Zipf/8), placed between the corpus 5th and 95th percentiles for the field |
| | clarity | understandable | mean of: not visibly cut off; 1 − share of tokens that are unexplained acronyms; for descriptions of 40+ words, LIX (1 at ≤ 40, 0 at ≥ 60, linear between) |
| | usefulness | carries what the file is about, without repetition | coverage of the summary's 10 key terms (function words with Zipf ≥ 6 excluded; ranked by TF-IDF with sublinear tf, (1 + ln tf) × ln(N/df)), literal or beating chance by meaning; subtitle/description: 1 − Jaccard overlap with the title; keywords: diversity, mean of (1 − duplicate-token ratio after stemming) and pairwise embedding similarity placed between corpus p5/p95 |
| **Functional** | bm25_findability | can a keyword search find the file by this field? | search every KO's summary with the field (BM25, k1 1.2, b 0.75); gain 1/log2(1 + rank of this KO); ties share the mid rank |
| | semantic_findability | can a meaning-based search find it, in any language? | same, with E5 dense retrieval |

**Cross-lingual fields.** A field that does not read as the reference's language -
the share of its alphabetic tokens that are known words of that language (word
frequency Zipf >= 2.0) falls under 0.5, with at least 2 tokens; open-set language
detection misreads short fields and is not used - has its purely lexical parts marked
**not evaluable** instead of scored against text they cannot match:
`bm25_findability` (the functional pillar is then semantic findability alone; in a
subset where only one KO carries a language, any query in that language would rank it
first without competition), `usefulness`'s coverage, and keywords' accuracy (its
chance threshold is calibrated on same-language pairs). NLI accuracy for title,
subtitle and description and semantic findability are multilingual and still apply.
With `--resummarise` the reference itself is in the document's language, so
same-language keywords are measured directly; keywords written in English on a
non-English document are measured against the English version of that summary
instead of abstaining (literal match or beating its own chance threshold; the
evidence names the basis). BM25 findability then gates on the corpus index's
language (English) rather than the reference's, for the reason above. A "not
evaluable" part is excluded from its pillar's mean - the weight redistributes, it is
never zeroed.

### 5.3 Which parts apply to which field

| part | title | subtitle | description | keywords |
|---|---|---|---|---|
| length | ≤ 150 characters, spaces not counted | not evaluated | 50-5,000 characters | not evaluated |
| completeness checks | filename, copy, shared, internal project reference, file name | filename, copy, shared | filename, copy, shared | has terms |
| rich text | - | - | editor HTML converted to text first; bullets kept | - |
| accuracy claim | "This text is about {title}." | "This text is about {subtitle}." | each sentence (up to 8) | each term: present or supported |
| clarity: readability | - | - | if 40+ words | - |
| clarity: not cut off | yes | yes | yes | - |
| usefulness: adds to title | - | yes | yes | - |
| usefulness: diversity | - | - | - | yes |

### 5.4 Aggregation

```
pillar       = mean of its evaluable sub-pillars          (structural, semantic, functional)
llm          = the LLM's 0-100 score of the field / 100    (Section 6)
field score  = 100 × weighted mean of the four pillars     (weights: 1 each)
KO score     = weighted mean over the four fields          (weights: 1 each; an empty field, subtitle included, scores 0)
```

Equal weights at every level, the LLM pillar included: a quarter of every field score.
Change them in `validation/metadata_spec.json`; stage 5 reports how much rankings would
move under data-driven alternatives. Every field is expected, subtitle included: an
empty field scores 0 on its field score and on each of the four pillars, with no LLM
call, and counts in the KO score. Its completeness sub-pillar is 0, so the
weakest-sub-pillar list names it.

A field whose LLM call returned no usable score is scored on the three measured pillars
and lists `llm` under "not evaluable". So is every field of a run with
`--no-llm-score`, which is then not comparable with a full run.

The same code scores the contributor's metadata (`original`) and the LLM version
(`llm`: written fresh per run by 4d, or the export's `*_llm` fields with
`--use-existing-llm`), so the two are comparable number for number.

### 5.5 Recommended version per field

Every scored KO gets, for each field, a recommendation between the contributor's version
and the LLM one (named `generated_llm` or `existing_llm` per KO). No further call is
needed: both versions have been scored by 5.1-5.4, the LLM pillar included.

```
candidates   = the contributor's field and the LLM version's, where not empty
eligible     = candidates whose accuracy against the summary (5.2 consistency) is at least
               the contributor's accuracy - 0.02
recommended  = the eligible candidate with the higher field score; a tie goes to the contributor
```

The faithfulness condition stops a version that scores higher by saying more than the
summary supports from being recommended. It uses the measured accuracy, not the LLM's
score. The 0.02 is `accuracy_tolerance` in `validation/metadata_spec.json`. A version
whose faithfulness is not evaluable (cross-lingual keywords, 5.2) passes the guard:
a measurement gap, not a verdict.

When the contributor left a field empty (most often the subtitle: 60.5% of English KOs
with a usable summary in the 24 September 2026 export), the `*_llm` version is
recommended without the faithfulness condition, because there is nothing to compare it
with; no absolute faithfulness floor is imposed. The Fields sheet shows each version's
faithfulness.

With `--use-existing-llm`: the export's `*_llm` fields were generated field by field
from the summary **and** the
contributor's current value, and came back unchanged for 15.5% of English titles, 22.3%
of subtitles (where both exist), 4.8% of descriptions and 6.6% of keyword sets. Where
they are unchanged, both versions get the same score. The fresh generation (4d) is
blind, so this convergence does not apply to it.

---

## 6. The LLM pillar

`stages/llm_score.py`. For every KO the gate lets through, for each version and each
field that has text, one call:

- **in:** the KO's summary (up to 6,000 characters), that one field, and the criteria of
  the three measured pillars in words, with the length rules for title and description.
  The prompt also names the KO's language, says the summary is always English, and asks
  the judge to rate meaning across languages without penalising the field's own
  language - LLM evaluators are less reliable on non-English text, and the platform is
  multilingual;
- **out:** `{"structural": 0-100, "semantic": 0-100, "functional": 0-100,
  "score": 0-100, "reason": "<one sentence>"}`. Only `score` enters the field score;
  the three per-criterion numbers are the LLM's own reading of the measured pillars'
  criteria, reported next to the measured values (two sources per pillar; their
  per-run agreement is on the Run sheet).

Each field is a separate call, because multi-field answers were where the output broke.
An empty field is not sent; it scores 0. The number of calls is printed before any is
made: up to 8 per KO, about 7.4 on English KOs, about 44,000 for the 6,001 English KOs
with a usable summary. Answers are cached, so a rerun costs nothing. Each prompt change
resets comparability with earlier runs' LLM-pillar values.

The model is glm-5.2, which wrote neither version: the contributor wrote one, and a
Qwen model wrote the LLM version (the generator is qwen3.5 by default; the export's
`*_llm` fields came from another Qwen model), which is why no Qwen model scores them -
the judge stays out of the writer's model family, and the pipeline warns on a match.
Each version is scored on its
own, not side by side, so the order in which versions are shown cannot bias it.

**Known and accepted for now.** LLM judges tend to favour LLM-written text, which here is
the `*_llm` version. Earlier side-by-side comparisons in this project (version 7, scored
against the file, 10 English KOs) found the same: the judges repeated their verdict under
a reversed order on only 45-63% of fields, picked the first-shown version 50-56% of the
time (chance 33%), gave 71-77% of their wins to their own model's text, and never picked
the contributor's version. That is why the LLM is one pillar of four, and why the
recommendation's faithfulness condition uses the measured accuracy. The LLM also cannot
see where a field ranks when every KO is searched, or how common a word is on the
platform, and does not count characters reliably.

The Fields sheet of `report.xlsx` shows, per KO × field × version, the four pillars side
by side with the LLM's reason.

---

## 7. Stage 5 - Analysis

| output | what |
|---|---|
| `weakest` (per KO) | the weakest sub-pillars (< 0.5) across fields, each with its evidence in plain words: "ranks 8,628 of 10,468 when used as a search query", "unexplained acronyms: GESVAC" |
| `robustness_pillars`, `robustness_subpillars` | the composite recomputed with entropy, CRITIC and PCA weights; Spearman with the nominal ranking, mean and maximum rank shift, share of KOs moving over 10% of the ranking |
| `redundant_subpillars` | pairs correlated at \|ρ\| ≥ 0.9 across KOs |
| `length_bias` | Spearman correlation of each sub-pillar with field length |
| `llm_agreement` | per-run Spearman agreement between each measured pillar and the LLM's own per-criterion reading of it |
| `reference_flags` | the stage-3b percentile thresholds, anchored per file-language group of at least 30 checked KOs (run-wide fallback), and which KOs they flag |
| per-language tables | the console summary and the Run sheet break scores down by file language; scores are never pooled or ranked across languages, because instrument coverage differs |

Pillar and field scores are always reported next to the composite, because one number
hides the trade-offs between them.

---

## 8. Settings

`validation/metadata_spec.json` holds every configurable number (field limits and
weights, pillar weights, readability cut-offs, models, the reference text's source and
limits, coverage and coherence baselines, diversity anchor sample, BM25, specificity,
the LLM pillar's model, the generation settings, the resummarise settings - the
document's language, 12 pages per chunk, merges of 8, verification on, translation
to English on, no page cap, 120 dpi, JPEG quality 80, 12 video frames, 10-minute
audio segments, the transcription model, 600 MB and 300 s download limits - and the
cross-lingual rule: at least 2 tokens, known-word Zipf 2.0, known share 0.5). Values learned from the corpus are computed from the export
and cached with the index, keyed on the export file and the settings; switching
`reference.source` rebuilds the index.

Per-run model overrides: `--generate-model`, `--judge-model`, `--resummarise-model`
replace the spec's generation, LLM-pillar and resummarise models for one run,
recorded in the run's arguments; the pipeline warns when generator and judge share a
model family.

Environment: `KOQ_GATE_MODEL`, `KOQ_GATE_LLM_LOW`, `KOQ_GATE_LLM_HIGH` (gate);
`KOQ_LLM_MODEL` (default LLM preference); `KO_DM_DIR` (controlled vocabularies).

---

## 9. Validation without human labels

### 9.1 Perturbation self-test (`--selftest`)

For every assessed KO, each sub-pillar receives one deliberate defect aimed at the
property it claims to measure; it passes if its value drops.

| sub-pillar | defect |
|---|---|
| length | title padded past its 150-character limit |
| completeness | title replaced by a filename |
| noise | markup, a URL and encoding debris appended to the description |
| formatting | title set in capitals |
| consistency | description, and separately title, taken from another project's KO |
| density | title rewritten in the language's commonest words |
| clarity | description cut mid-sentence with an ellipsis |
| usefulness | keywords taken from another project's KO; keywords padded with case variants of themselves |
| bm25 / semantic findability | title taken from another project's KO |

The LLM pillar is not in the self-test yet: damaging a field and checking that the LLM's
score drops is the planned check. A value that does not drop because the original was already at the floor is reported as
`unchanged_at_floor`, not as a failure. Latest result: 232 of 240 defects detected (97%).

### 9.2 Choosing the NLI model

Each candidate scores 40 KOs' own title and description, and ones swapped in from other
projects; the share of pairs where the true field scores higher. Measured against the
file, on the June 2026 export; `python -m tools.compare_nli --reference summary` repeats
the check against the summary, which has not been run yet:

| model | overall | its training languages | other EU languages | speed (pairs/s, CPU) |
|---|---|---|---|---|
| `joeddav/xlm-roberta-large-xnli` | **0.938** | **1.000** | **0.875** | 0.97 |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 0.775 | 0.750 | 0.800 | 1.5 |

XLM-R large is used: better in both language groups.

---

## 10. Regenerating everything

```bash
cd assess_ko_quality
python pipeline.py --learn --limit 1              # relearn the compliance profile (1 LLM call)
python -m stages.gate --calibrate                 # gate classifier and thresholds
python -m tools.compare_nli --limit 40            # NLI model choice
python pipeline.py --limit 20 --selftest          # scores + perturbation tests (English KOs; up to 160 LLM calls)
python pipeline.py --limit 20 --no-llm-score --no-gate-llm --use-existing-llm   # the measured pillars only, no LLM calls
```

The corpus index rebuilds automatically when the export or a setting that affects it
changes.

---

## 11. Limitations

- **Weights are a convention**, not a finding; stage 5 shows how much they matter.
- **Some settings are choices**: the length fall-off, chunk sizes, chance-baseline
  percentiles, the key-term count.
- **Coverage is not measured across languages.** Matching a file's key terms to a
  field in another language is weak (correct English keywords for a Latvian file reached
  0.1), so coverage is marked not evaluable for a field that does not read as the
  reference's language (5.2), rather than understating it. Consistency still penalises
  wrong keywords.
- **Accuracy is weaker outside the NLI model's training languages** (0.875 vs 1.000 in
  9.2), and descriptions phrased as statements about the document ("This report
  explores ...") come out neutral rather than supported.
- **The findability index is built from the export**, not the live search cluster, and
  represents each KO by the first 2,000 characters of its summary for dense retrieval.
- **Readability is coarse** across languages; LIX rises with long compound words
  (German, Dutch, Finnish, Greek).
- **A field the contributor left empty** is recommended from the LLM version
  without a faithfulness condition (5.5).
- **The reference text is the summary the LLM version is written from**, so measuring
  against it favours text written from it. On the English KOs of the June 2026 export,
  the share of a field's words found in the summary was higher for the existing `*_llm`
  fields than for the contributor's (titles +0.11, descriptions +0.19), while against
  the file itself the contributor's were ahead (-0.02, -0.05). A detail the contributor
  took from the file but the summary left out counts against the contributor. Results
  are measured against the summary and should be reported that way.
- **The summary is short and does not grow with the file** (about 200 words at any
  length), so for a long file most of its content is not in the reference.
- **Scores are not comparable across languages.** The reference summary is English, so
  the lexical sub-pillars abstain on cross-lingual fields (5.2) and instrument coverage
  differs by language; results are reported per file language and never pooled or
  ranked across languages.
- **The LLM pillar is not validated,** and it is a quarter of every field score. LLM
  judges may favour LLM-written text, which here is the LLM version; it cannot see
  corpus-level findability or specificity, and does not count characters reliably
  (Section 6). Accepted for now; the planned check is to damage a field on purpose and
  confirm that the LLM's score drops.
- **Factual correctness of the file itself is not assessed.** The score checks that the
  metadata agrees with the summary, not that the file's advice is right.
- **Cost.** About 100 s per KO on CPU: 98-102 s in the October 2026 runs with the summary
  as the reference and the LLM answers mostly cached, 134 s for two long PDFs that were
  resummarised (Intel Core Ultra 9 185H laptop). Nearly all of it is the embedding and
  entailment models. The full export of 11,627 KOs needs about 13 days of CPU time at this
  rate, plus the wait for any LLM calls not already cached. The LLM pillar adds up to 8
  calls per KO.
