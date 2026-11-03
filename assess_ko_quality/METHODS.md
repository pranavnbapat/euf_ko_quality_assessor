# Methods reference

Every method, threshold and weight the KO quality pipeline uses, where each comes from,
and how it is checked. `README.md` says how to run it; `LITERATURE_REVIEW.md` gives the
full sources and a per-number evidence table; `validation/metadata_spec.json` holds the
configurable settings with their provenance. This file is precise enough to reimplement.

**What is scored.** A Knowledge Object (KO) is a file plus the metadata that describes
it. The score measures the **metadata** - title, subtitle, description, keywords -
against the file's **summary** (`ko_content_flat_summarised`, Section 5.1): is it
correct, informative, well-formed and findable? Every language is assessed by default.
Each field is scored twice, on identical criteria: the contributor's version and an
LLM version written fresh from the summary by the current model (Section 4d;
`--use-existing-llm` scores the export's `*_llm` fields instead, as every run did
before version 12). A field score averages four pillars: three
measured ones (ten sub-pillars, no LLM) and an LLM's 0-100 score of the field against
the summary (Section 6). The file's own quality is the original author's work; it is
measured and reported, and never changes the score.

**How it is justified.** Only measurable, reproducible criteria from the literature and
standards. Human reviewer ratings are not used to fit or validate anything - they are
subjective and error-prone. Validation is technical: every sub-pillar must detect a
deliberate defect in its own property (Section 9).

---

## 0. The pipeline at a glance

```
select ─► 1 GATE ─► 2 COMPLIANCE ─► 3 CONTENT ─► 4 METADATA SCORE (3 measured pillars + LLM score) ─► 5 ANALYSIS
```

| stage | question | output | affects the score | LLM calls |
|---|---|---|---|---|
| select | which KOs to assess | Run sheet | - | 0 |
| 1 gate | is the KO about agriculture in the broad sense? | `gate_*` | rejected KOs are not scored | only KOs in the uncertain band (typically 0-10%) |
| 2 compliance | are the platform's required fields filled, with current vocabulary values? | `compliance_*` | no | 0 |
| 3 content | what is the file itself like? | `content_*` | no | 0 |
| 4 metadata | how well does each version of the metadata describe the file, measured against its summary, and which version of each field is better? | `metadata_score`, the four pillars, fields, sub-pillars, the LLM's score and reason, the recommended version per field | **this is the score** | one per KO to write the LLM version (§4d; 0 with `--use-existing-llm`), plus one per field with text, per version: up to 9 per KO |
| 5 analysis | why is a score low, and how much do our choices matter? | KOs and Run sheets | no | 0 |

---

## 1. Selection

| flag | effect |
|---|---|
| `--limit N` | at most N KOs |
| `--min-summary-words` (50) | skip KOs whose summary is shorter; the summary is what the fields are measured against, and below 50 words all four `*_llm` fields in the export are copies of the contributor's (584 of 584); 6.3% of the export |
| `--file-type` | matched against category, extension, mimetype and subcategory ("pdf", "Video", "Factsheet") |
| `--languages` (all) | a KO's language is the **first** one listed. Every language by default (version 11; English-only until then). Caveat kept from version 9: for 82% of 2,308 non-English files checked the summary is in English, so non-English contributor fields are measured cross-lingually - report them per language, not pooled with English |
| `--licenses`, `--exclude-licenses` | keep only, or drop, these licence values (e.g. `CC BY-NC-ND`) |
| `--upload-sources` | keep only KOs with these `ko_upload_source` values, e.g. `upload_form` (24 September 2026 export: upload_form 7,155, public_api 2,477, missing 1,963, ai_uploader 28); the search corpus stays the whole export |
| `--lang-balance equal` (default) | round-robin across languages, so a 20-KO run is not 60% English |
| `--seed` (42) | reproducible sampling |

---

## 2. Stage 1 - Domain gate

**Hybrid: a classifier for every KO, an LLM only where the classifier is unsure.** The
gate is a precondition, not a quality dimension: a polished off-topic upload must not
outrank real content.

| Property | Value |
|---|---|
| Classifier | Logistic regression on `all-mpnet-base-v2` sentence embeddings (Reimers & Gurevych, EMNLP 2019) of the first 2,000 characters |
| Training / validation | `validation/gate_validation/` - 678 items: real KOs vs off-domain documents (Wikipedia, five languages) |
| Performance | AUC 1.000 (5-fold CV); at the accept threshold 100% of real KOs kept in all 14 languages tested, 94.6% of off-domain caught |
| Accept / reject | P ≥ 0.40 accept; P < 0.20 reject |
| LLM band | P in [0.20, 0.70) is decided by an LLM (`KOQ_GATE_MODEL`, default glm) - the reject option of Chow (1970) and selective classification (Geifman & El-Yaniv, NeurIPS 2017): defer only the cases the cheap model is unsure of |
| LLM scope | read from `data_model.topics.json` (Forestry, Livestock, Crop farming, Economics, Environment, Society), with Economics/Environment/Society in scope when they are *of* agriculture, food, forestry or rural life |
| Fallback | if the LLM fails, the classifier's thresholds decide; the gate never drops a KO silently |

Measured on 140 labelled items: classifier alone accepted 62% of near-miss documents
(packaging, logistics, urban planning); the LLM rejected them; on the uncertain band
the LLM was right 12 of 12 times. 98.7% of Economics/Society/Environment-only KOs are
accepted by the classifier outright.

The LLM prompt failed twice before it worked, and both clauses exist because of it:
"judge the SUBJECT only" (otherwise it judged document type and rejected Wikipedia's
*Dairy farming*), and "Economics, Environment and Society OF agriculture" (otherwise it
accepted VAT law and cement). Re-run the comparison after changing the prompt: a failing
LLM gate fails silently.

Rejected method: cosine similarity to an AGROVOC/NALT centroid (AUC 0.647) - it scored
gibberish (0.069) like a crop-rotation guide (0.071).

---

## 3. Stage 2 - Compliance

Obligations, as a pass/fail checklist next to the score and never inside it (ISO/IEC
25012 treats compliance as its own characteristic).

| Property | Value |
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

The published text-quality filters used to clean web text for language models, with
their published thresholds. A flag means "this text would be removed by filter X".

| filter | rule (flag when) | source |
|---|---|---|
| Gopher quality | < 50 or > 100,000 words; mean word length outside 3-10; symbol/word ratio > 0.1; > 90% bullet lines; > 30% lines ending in an ellipsis; < 80% words with a letter; < 2 of the language's 8 commonest words | Rae et al. 2021 (MassiveText); datatrove defaults |
| Gopher repetition | duplicate lines > 30% or their characters > 20%; same for paragraphs; top 2/3/4-gram character share > 0.20/0.18/0.16; duplicated 5-10-gram character share > 0.15 ... 0.10 | same |
| FineWeb | lines ending in punctuation ≤ 12%; characters in duplicated lines ≥ 10%; lines under 30 characters ≥ 67% | Penedo et al., NeurIPS 2024 - kept from over 50 candidates by ablation |
| C4 (as kept by FineWeb) | "lorem ipsum"; a curly bracket; lines mentioning javascript | Raffel et al., JMLR 2020; Dodge et al., EMNLP 2021 |
| Multilingual stop words | Gopher's English stop words replaced by the language's 8 commonest words (wordfreq) | adapted from FineWeb2, Penedo et al., COLM 2025 (a frequency threshold, at least 8 words) |
| Declared vs detected language | first declared language differs from the detected one | Kreutzer et al., TACL 2022 (mislabelled language is a common, easily detected fault) |

Also reported: word count, MTLD lexical diversity (McCarthy & Jarvis 2010, 50+ words),
LIX readability (100+ words), counts of DOIs, URLs and distinct years mentioned
(provenance cues), `content_extraction_failed` (< 50 words, Gopher's minimum).

### 4b. Stage 3b - Summary reference check (reported, never scored; version 11)

`stages/summary_check.py` (`--no-summary-check` skips it). Every field score is
measured against the KO's summary, so a bad summary silently degrades every reading
for that KO. This stage measures the summary itself against the file text and reports
the result as a reliability reading on the reference - like the content report, it
never enters the score, because the reference is the pipeline's instrument, not its
subject. Per KO:

| reading | how | basis |
|---|---|---|
| compression ratio | summary words / file words | no published universal ratio exists, so no absolute threshold; the run's own p5-p95 flags outliers (`analysis.reference_flags`, the learned-from-the-corpus principle the density anchors use; at least 30 summaries needed) |
| key-term coverage | share of the file's TF-IDF key terms (the coverage machinery of §5) found literally in the summary | Salton & Buckley 1988; Manning, Raghavan & Schütze 2008 |
| NLI support | up to 6 summary sentences, spread evenly, each read against its 2 best-matching file chunks; mean P(entailment) | the bidirectional-NLI design of `summary_evaluation/03`; chunking and premise counts reuse the `reference` settings |
| language | langdetect on the summary | the platform writes summaries in English by design; anything else is flagged |

Flags (`no_summary`, `summary_language_mismatch` - the detected language against the one the summary is meant to be in: the document's own with `--resummarise`, else English - `compression_outlier`, `low_nli_support`,
`low_key_term_coverage`) and a `reference_reliability` verdict (`ok`/`check`) appear on
the KOs sheet of `report.xlsx`; a flagged KO's scores read against a questionable
reference.

### 4c. Regenerating the reference summary (`--resummarise`, opt-in; reworked in version 15)

`stages/resummarise.py`. The export's summaries were written by the platform's
summariser of their day, from the platform's own extraction of the file; newer models
may summarise better, and the stored extraction can be wrong (one Lithuanian factsheet
carries an English *translation* as its "extracted text"). With `--resummarise`, each
assessed KO gets a fresh summary **from its own file, in the document's own language**
(version 15, decided 30 October 2026), and that summary becomes the
reference every field of the KO is measured against (§5.1): a French KO's fields are
judged in French end to end. An **English version** of it (one translation call for a
non-English document, none for an English one) is the KO's row in the corpus index,
which stays English like every other row (`CorpusIndex.override_reference`), and the
secondary reference for English-language fields on a non-English document (§5.3). The
summary, its English version and the platform's summary it replaced are stored with
the run (`details.jsonl` -> `summary`; the Summaries sheet of `report.xlsx`). The flag
is never the default: without it the export's summary is used as it is.

**The file as it is.** The file is downloaded from the `@id` URL (cached locally; up
to 600 MB) and read as what it is, with no page cap:
- a PDF or an image: the pages rendered to images (PyMuPDF, a visual rendering; the
  text layer is never read), one page per image at 120 dpi. The provider's endpoint
  rejects PDFs sent natively, so pages-as-images is the delivery (verified 8 October
  2026);
- an office document (doc/docx/ppt/pptx/xls/xlsx/odt/odp/ods/rtf): converted to PDF
  by LibreOffice headless first, then as above;
- a video: its sound transcribed by `whisper-large-v3` through the provider's
  `/v1/audio/transcriptions` (Radford et al., ICML 2023; in pieces of 10 minutes, one
  transcription call each) and 12 frames sampled evenly across the recording, shown
  next to the transcript;
- an audio file: the transcript alone.

A KO whose file cannot be fetched or read falls back to the export's text, and a file
flow that fails retries once from the export's text; the basis of every summary is
recorded per KO (`resummarised.from`: "file" or "export_text", with the kind, pages,
chunks, calls and findings) and counted on the Run sheet.

**How the summary is written.** A document of up to 12 pages (`chunk_pages`, the
provider's images-per-prompt limit), a transcript with frames, or a text is drafted in
one call. A longer document is summarised by **hierarchical merging** (Chang, Lo,
Goyal & Iyyer, ICLR 2024, "BooookScore"): detailed notes per chunk of 12 pages, merged
eight at a time (`merge_fanin`) until one summary remains. That study compared the two
workflows - "(1) hierarchically merging chunk-level summaries, and (2) incrementally
updating a running summary" - and found that "Incremental updating yields lower
BooookScore but higher level of detail than hierarchical merging"; coherence is chosen
here. Every draft is then **verified against its source**, following
Chain-of-Verification (Dhuliawala et al., Findings of ACL 2024), adapted to a
document. A source of one chunk is checked in one call, which lists the statements the
source contradicts or does not support and the major content the draft lacks. A
longer document is checked one chunk per call, **sentence by sentence**, because one
chunk cannot tell a wrong statement from one about other pages: the draft is split
into numbered sentences, each chunk gives every sentence a verdict - supported,
contradicted, or absent from these pages - and lists the content the draft lacks, and
the decision is taken across chunks. A sentence is kept when any chunk supports it.
This is how chunked factual-consistency checkers combine evidence: AlignScore (Zha et
al., ACL 2023, §3.2) splits the context into chunks, and "The highest alignment score
of each claim sentence is selected and then averaged to derive the factual
consistency score"; SummaC-ZS (Laban et al., TACL 2022, §3.2), the rule the accuracy
sub-pillar already uses (§5.2), keeps for each summary sentence "the score for the
document sentence that provides the strongest support for each summary sentence". A
sentence is an error when a chunk contradicts it and none supports it, and
unsupported when every chunk reports it absent. A sentence that a chunk gave no
verdict on is never called unsupported, so a malformed answer cannot delete a true
statement; sentences with no verdict at all are counted (`unverified`). One revision
call then applies every finding; with no findings there is no revision call. Last, a
non-English summary is
**translated to English** for the corpus row. The prompts keep the platform
summariser's brief - every major topic, entities and numbers preserved, proportional
length - and ask for the document's language, which the model names from the pages
(`file_language`, on the KOs sheet next to the declared languages; reported, never
scored). The declared languages are contributor input and the stored text can be a
translation, so the file itself is the only trusted source for what the document is
written in; on the export-text fallback the reading describes that stored text
instead.

Calls per KO: 2-4 for a document of up to 12 pages (draft, verification, a revision
when needed, a translation when not English), about two more per further chunk plus
the merges; one transcription call per 10 minutes of audio. The number is estimated
and printed before any call is made; every answer is cached (the page images are part
of the cache key). Scoring, the LLM pillar, metadata generation (§4d) and the summary
check then read the fresh summary. The KO's own row in the findability rankings is
recomputed from the English version; corpus statistics (document frequencies,
anchors, the other KOs' rows) remain the export's - a documented approximation, fine
for samples, wrong for whole-corpus reruns, which should rebuild the index from a
re-summarised export instead. Runs with and without the flag are not comparable, nor
are version-15 runs with version 14's (English reference, one call per KO, 50-page
cap, stacked tiles). The model (`resummarise.model`, `--resummarise-model`) must be
vision-capable.

### 4d. Writing the LLM version of the fields (version 12, on by default)

`stages/generate_metadata.py`. Until version 11 the LLM variant was the `*_llm` fields
already in the export, written by the platform's improver (`data-prep-opensearch`,
configured for qwen3-30b-a3b-awq) from the summary **and** the contributor's fields.
Since version 12 the pipeline writes the variant itself: one call per KO with a
summary - counted before any call is made, cached - after `--resummarise` (so the
fields are written from the very summary they will be measured against) and before
the LLM pillar. Generation is **blind**: the model sees the reference summary and the
published field rules only, never the contributor's fields, so the LLM version is an
independent rewrite rather than an edit anchored to the contributor's wording; the
rules in the prompt are the same published limits the structural pillar measures, so
writer and scorer share one yardstick. Each field is written in **the document's own language** (rule of 8
October 2026): `file_language` read from the pages during `--resummarise`, else the
detected language of the stored text - never the contributor's declared languages,
which are input under test, not an instrument. The summary the fields are written
from is the platform's English one or - with `--resummarise` - the fresh one in the
document's own language (§4c), so the fields and the summary then share a language;
the prompt names the summary's language, and the LLM pillar's judge is told the
field's language and not to penalise it (§6).

The generator (`generate_metadata.model`; empty = the default client's model, qwen3.5
at the time of writing) must be a different model from the LLM pillar's judge, because
LLM judges favour their own model's text (Panickssery et al., NeurIPS 2024); the
pipeline warns when they match. A KO whose summary is empty, or whose call fails,
keeps the export's `*_llm` fields and is marked per KO (`generated_llm` in
`details.jsonl`, the version label in the Fields sheet). `--use-existing-llm` restores
the pre-version-12 comparison - and, with `--no-gate-llm` and `--no-llm-score`, the
no-LLM-calls path. Version-12 `llm` columns are not comparable with earlier runs: a
different author wrote the fields.

---

## 5. Stage 4 - Metadata score

### 5.1 What the fields are checked against

**Length rules** come only from published external sources - the EIP-AGRI Common Format
(title) and Google's Dataset guidelines (description); `LITERATURE_REVIEW.md` §3.1 quotes
each. Subtitle and keywords have no published length, so none is imposed. The first
version's ranges (title 5-12 words, description 40-80 words, keywords 4-10) came from an
LLM prompt and were removed.

**The reference text** (`reference.source`, a choice): the KO's summary,
`ko_content_flat_summarised`, and nothing else - or, with `--resummarise`, the fresh
summary of the file in the document's own language (§4c), which replaces it on the
record; the summary used is stored with every run. The LLM version of the fields is
written from the same summary (§4d) - as were the export's `*_llm` fields (5.5) - so
both versions are measured against the text the LLM version was written from. The summary is not the file. For 86% of KOs it is
a light polish of the vision model's own summary of the file (`ko_content_flat_vision`),
and it stays near 200 words whatever the file's length: median 186-235 words for files
of 200 to over 20,000 extracted words. Fields are scored only when the summary has at
least 200 characters; otherwise the semantic and functional sub-pillars are "not
evaluable" and left out. `reference.source: "file"` restores the reference used up to
version 8: the vision reading followed by the extracted text (`ko_content_flat`), cut
at 100,000 characters. Section 12 gives the effect of the choice.

**The corpus index** (`core/corpus_index.py`, cached in `output/_corpus_cache/`): a BM25
index and E5 passage embeddings of every KO's reference text (never its metadata, so a
title cannot find itself), corpus document frequencies, general-language word
frequencies (wordfreq) for every detected language, per-field value-sharing counts,
null distributions for the chance baselines, and anchors for the scales below.

**Models**: `intfloat/multilingual-e5-small` for embeddings (Wang et al. 2024; the best
public model on MMTEB, Enevoldsen et al., ICLR 2025, is `multilingual-e5-large-instruct`
from the same family - the small variant is used for CPU cost), and `joeddav/xlm-roberta-large-xnli` for NLI (XLM-R,
Conneau et al., ACL 2020, fine-tuned on MNLI + XNLI, Conneau et al., EMNLP 2018),
**chosen by measurement** (Section 9.2).

### 5.2 The ten sub-pillars

Every value is in [0, 1]. "Chance" baselines and scale anchors are learned from the
corpus, never typed in.

| pillar | sub-pillar | measures | how | source |
|---|---|---|---|---|
| **Structural** | length | within the published limits (5.3) | each limit 1 when met, linear to 0 at 2× a maximum or at 0 below a minimum; the lowest limit counts; not evaluated where no source gives a limit | EIP-AGRI Common Format; Google Dataset guidelines; desirability function (Derringer & Suich 1980) |
| | completeness | present and not a placeholder | required and empty → 0; otherwise share of: not a filename, not a copy of another field, not shared verbatim by KOs of other projects; for titles also not an internal project code (D3.1, T2.3, WP2, "Task 4") and not the file name | Bruce & Hillmann 2004; Ochoa & Duval 2006; EIP-AGRI; Google title-link guidance |
| | noise | debris in the field | 1 − share of tokens that are markup, URLs, encoding debris, symbol runs or immediate repeats | C4 / Gopher rules; ISO 25012 syntactic accuracy |
| | formatting | field-specific form | share of checks passed (casing, line breaks, no dangling comma/colon at the end; description starts capitalised and ends a sentence; keywords a list, one term per item, no duplicates, none longer than the corpus 99th percentile) | ISO 25012 consistency; ANSI/NISO Z39.19; Tinker 1963 |
| **Semantic** | consistency | correct about the file (as its summary states it), and coherent with the other fields | mean of **accuracy** - NLI adapted from SummaC's zero-shot rule: each claim vs its 2 best-matching passages of the summary, max(P(entail) − P(contradict)), mean over claims, mapped to [0,1]; keywords: share present in the summary or closer to it than 95% of random keywords - and **coherence** - share of 300 random KOs whose same field is less similar to this field than this KO's own | Ochoa & Duval 2006 (accuracy, coherence); Laban et al., TACL 2022; Yin et al., EMNLP 2019; Meng et al., ACL 2017 |
| | density | specific rather than generic words | mean over words of min(platform IDF, 1 − Zipf/8), placed between the corpus 5th and 95th percentiles for the field | Ochoa & Duval text information content; Spärck Jones 1972; wordfreq (Speer 2022); Zipf scale (van Heuven et al. 2014) |
| | clarity | understandable | mean of: not visibly cut off; 1 − share of tokens that are unexplained acronyms; for descriptions of 40+ words, LIX (1 at ≤ 40, 0 at ≥ 60) | Barnett & Doubleday, eLife 2020; Schwartz & Hearst 2003; Veyseh et al., COLING 2020; Björnsson 1968/1983; Ochoa & Duval 2006 |
| | usefulness | carries what the file is about, without repetition | coverage of the summary's 10 key terms (function words excluded, TF-IDF with sublinear tf), literal or beating chance by meaning; subtitle/description: 1 − Jaccard overlap with the title; keywords: KPEval diversity, mean of (1 − duplicate-token ratio after stemming) and emb_sim placed between corpus p5/p95 | Lin 2004 (ROUGE recall); Salton & Buckley 1988 (TF-IDF); Manning et al. 2008 (sublinear tf); Stvilia et al. 2007; KPEval (Wu, Yin & Chang, Findings of ACL 2024) |
| **Functional** | bm25_findability | can a keyword search find the file by this field? | search every KO's summary with the field (BM25, k1 1.2, b 0.75); gain 1/log2(1 + rank of this KO); ties share the mid rank | Azzopardi & Vinay 2008; Robertson & Zaragoza 2009; OpenSearch defaults (k1, b); Burges et al., ICML 2005 (nDCG discount); KPEval utility; BEIR (Thakur et al., NeurIPS 2021) |
| | semantic_findability | can a meaning-based search find it, in any language? | same, with E5 dense retrieval | as above; Karpukhin et al., EMNLP 2020 |

### 5.3 Which parts apply to which field

Since version 14, **keywords** that match the document-language summary's
language are measured against *that* summary (same literal-or-chance test, chunks
and chance threshold from the document-language text; the evidence names the
basis), so keyword faithfulness abstains only when the keywords match neither
summary's language. BM25 findability stays not evaluable cross-lingually even
then - findability is a corpus property, the corpus is indexed on English
summaries, and a query in a language only one KO carries would rank first
without competition. Coverage also stays not evaluable: key-term extraction
needs document-frequency and word-frequency statistics in the field's language.

Since version 13, a field that does not read as the reference's language (the
share of its alphabetic tokens that are known words of that language, wordfreq
`zipf_frequency` >= 2.0, falls under 0.5 with at least 2 tokens) additionally
marks `bm25_findability`, `usefulness.coverage` and - for keywords -
`consistency.accuracy` as not evaluable: a lexical instrument cannot read
across languages, and the keyword chance threshold is calibrated on
same-language pairs. The NLI accuracy of the text fields and semantic
findability are multilingual and still apply.

| part | title | subtitle | description | keywords |
|---|---|---|---|---|
| length | ≤ 150 characters, spaces not counted (EIP-AGRI) | not evaluated (no published limit) | 50-5,000 characters (Google Dataset) | not evaluated (no published count) |
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
field score  = 100 × weighted mean of the four pillars     (weights: structural 1, semantic 1, functional 1, llm 1)
KO score     = weighted mean over the four fields          (weights: 1 each; an empty field, subtitle included, scores 0)
```

Equal weights at every level: what most composite indicators use (OECD/JRC *Handbook
on Constructing Composite Indicators*, 2008), and hard to beat without an external
criterion (Dawes 1979). Giving the LLM pillar the weight of each measured pillar - a
quarter of every field score - is a choice (version 10). Change weights in
`validation/metadata_spec.json`; stage 5 reports how much rankings would move under
data-driven alternatives. Every field is expected, subtitle included: an empty field
scores 0 on its field score and on each of the four pillars, with no LLM call, and
counts in the KO score, as in Ochoa & Duval's (2006) completeness metric, which counts
the fields that hold a value, so a missing field adds nothing to the score. Its
completeness sub-pillar is 0, so the weakest-sub-pillar list names it.

A field whose LLM call returned no usable score is scored on the three measured pillars
and lists `llm` under "not evaluable". So is every field of a run with
`--no-llm-score`, which is then not comparable with a full run.

The same code scores the contributor's metadata (`original`) and the LLM version
(`llm`: written fresh from the summary, §4d, or the export's `*_llm` fields with
`--use-existing-llm`), so the two are comparable number for number.

### 5.5 Recommended version per field

Every scored KO gets, for each field, a recommendation between the contributor's version
and the LLM one (`stages/metadata_score.recommend`; its output names the LLM candidate
`generated_llm`, or `existing_llm` where this KO's fields are the export's). No
further call is needed: both versions have been scored by 5.1-5.4, the LLM pillar
included.

```
candidates   = the contributor's field and the LLM field, where not empty
eligible     = candidates whose accuracy against the summary (5.2 consistency) is at least
               the contributor's accuracy - 0.02
recommended  = the eligible candidate with the higher field score; a tie goes to the contributor
```

The faithfulness condition is there because optimising a proxy score degrades the true
objective (Gao, Schulman & Hilton, ICML 2023): a version that scores higher by saying more
than the summary supports is not recommended. The condition uses the measured accuracy,
not the LLM's score. The 0.02 is `accuracy_tolerance` in
`validation/metadata_spec.json`.

When the contributor left a field empty (most often the subtitle: 60.5% of English KOs
with a usable summary in the 24 September 2026 export), the `*_llm` version is
recommended without the faithfulness condition, because there is nothing to compare it
with. No published source gives an absolute faithfulness floor, so none is imposed; the
Fields sheet shows each version's faithfulness.

With `--use-existing-llm`, the export's `*_llm` fields apply: the platform's improver
generated them field by field from the summary **and** the contributor's current value,
and they came back unchanged for 15.5% of English titles, 22.3% of subtitles (where
both exist), 4.8% of descriptions and 6.6% of keyword sets (24 September 2026 export).
Where they are unchanged, both versions get the same score. The version-12 generated
fields are written blind (§4d), so they cannot copy the contributor's.

---

## 6. The LLM pillar

`stages/llm_score.py`. For every KO the gate lets through, for each version and each
field that has text, one call:

- **in:** the KO's summary (up to 6,000 characters), that one field, and the criteria of
  the three measured pillars in words, with the published length rules for title and
  description. Since version 12 the prompt also names the KO's declared language, says
  the summary is always English, and asks the judge to rate meaning across languages
  without penalising the field's own language - LLM evaluators are less reliable on
  non-English text (Hada et al., Findings of EACL 2024), and the platform is
  multilingual;
- **out:** `{"structural": 0-100, "semantic": 0-100, "functional": 0-100,
  "score": 0-100, "reason": "<one sentence>"}` (version 11; G-Eval's form-filling
  design, Liu et al., EMNLP 2023). Only `score` enters the field score. The three
  per-criterion numbers are the LLM's own reading of the measured pillars' criteria,
  shown next to the measured values in the Fields sheet, so every pillar has two
  sources - code and LLM; their per-run Spearman agreement is on the Run sheet
  (`analysis.llm_agreement`). Each prompt change resets comparability: version-11
  LLM-pillar values are not comparable with version 10's, nor version 12's (the
  language line) with version 11's.

Each field is a separate call, because multi-field answers were where the output broke.
An empty field is not sent; it scores 0. The calls are made concurrently after the gate
and before the measured scoring, and their number is printed first: up to 8 per KO,
about 7.4 on English KOs because 60% have no contributor subtitle, about 44,000 for the
6,001 English KOs with a usable summary in the 24 September 2026 export. Answers are
cached, so a rerun costs nothing, and a `*_llm` field copied from the contributor's gets
the same score.

The model is glm-5.2 (`llm_score.model`). It wrote neither version: the contributor
wrote one, and the LLM version is written by the generator of §4d (qwen3.5 by
default; with `--use-existing-llm`, by the platform's qwen3-30b-a3b-awq).
Self-preference (Panickssery et al., NeurIPS 2024) therefore cannot favour one; no
Qwen model is used as the judge because it would be of the writer's family, and the
pipeline warns when generator and judge match. Of the two other models measured as judges (below), glm-5.2 repeated
its verdict more often with the order reversed (60-63% against 55%), and gpt-oss-120b
always reasons and rejects `reasoning_effort`, so tight answers can come back truncated
(`../USING_LLMS.md`). Scoring each
version on its own, rather than side by side, removes the position bias Wang et al.
(ACL 2024) found in pairwise comparison. Rating named criteria with an LLM is the
approach of G-Eval (Liu et al., EMNLP 2023).

**Known and accepted for now.** The G-Eval authors report that LLM evaluators may favour
LLM-generated text, which here is the `*_llm` version. The side-by-side comparisons
made earlier in this project (version 7: scored against the file, 10 English KOs, 40
fields, `output/llm_compare_en10/`) found the same:

| | glm-5.2 as judge | gpt-oss-120b as judge |
|---|---|---|
| same verdict when the order is reversed | 60% | 55% |
| picked whichever version was shown first (chance 33%) | 50% | 56% |
| verdict agrees with the measured best | 54% | 46% |
| share of wins given to its own model's text | **71%** | **77%** |
| times it picked the contributor's original | 0 | 0 |

Both judges showed the position bias and the self-preference reported in the literature,
never preferred the contributor's version even where it measured best, and their 1-5
ratings barely followed the measured properties (Spearman 0.06-0.34). A second
comparison with qwen3.5-397b as judge (`output/llm_compare_glm_qwen_en10/`) repeated its
verdict with the order reversed on only 45% of fields. That is why the LLM is one pillar
of four, and why the recommendation's faithfulness condition uses the measured accuracy.

The LLM also cannot see what the corpus-based parts measure - where a field ranks when
every KO is searched, or how common a word is on the platform - and does not count
characters reliably. Its structural and functional judgement is a reading of the text,
next to the measured pillars rather than instead of them.

The Fields sheet of `report.xlsx` shows, per KO × field × version, the four pillars side
by side with the LLM's reason; `details.jsonl` keeps every call's score and reason.

---

## 7. Stage 5 - Analysis

| output | what | source |
|---|---|---|
| `weakest` (per KO) | the weakest sub-pillars (< 0.5) across fields, and an LLM score under 50 with the LLM's reason, each with its evidence in plain words: "ranks 8,628 of 10,468 when used as a search query", "unexplained acronyms: GESVAC" | - |
| `robustness_pillars`, `robustness_subpillars` | the composite recomputed with entropy, CRITIC and PCA weights; Spearman with the nominal ranking, mean and maximum rank shift, share of KOs moving over 10% of the ranking | OECD/JRC 2008; Saisana, Saltelli & Tarantola 2005; Diakoulaki et al. 1995 |
| `redundant_subpillars` | pairs correlated at \|ρ\| ≥ 0.9 across KOs | OECD/JRC 2008 multivariate step |
| `length_bias` | Spearman correlation of each sub-pillar with field length | the reason type-token ratio was dropped |

Per-KO multi-dimensional reporting - pillar and field scores next to the composite, not
only a single number - follows HELM's argument that one aggregate hides trade-offs
(Liang et al., TMLR 2023).

---

## 8. Settings

`validation/metadata_spec.json` holds every configurable number with its provenance
(field ranges and weights, pillar weights, readability cut-offs, models, the reference
text's source and limits, coverage and coherence baselines, diversity anchor sample,
BM25, specificity, the LLM pillar's model). Values learned from the corpus are not
stored there: they are computed from the export and cached with the index, keyed on the
export file and the settings. `reference.source` is part of both cache keys, so
switching between the summary and the file rebuilds the index and re-embeds the
passages.

Environment: `KOQ_GATE_MODEL`, `KOQ_GATE_LLM_LOW`, `KOQ_GATE_LLM_HIGH` (gate);
`KOQ_LLM_MODEL` (default LLM preference); `KO_DM_DIR` (controlled vocabularies).

Per run, `--generate-model`, `--judge-model` and `--resummarise-model` override the
spec's `generate_metadata.model`, `llm_score.model` and `resummarise.model`; the
override is recorded in the run's arguments on the Run sheet. Names are resolved
against the provider's live model list, shortest match first, so name a model in full.
The pipeline warns when the generator and the judge share a model family (§4d).

---

## 9. Validation without human labels

### 9.1 Perturbation self-test (`--selftest`)

For every assessed KO, each sub-pillar receives one deliberate defect aimed at the
property it claims to measure; it passes if its value drops. This is behavioural testing
of an evaluation metric (CheckList, Ribeiro et al., ACL 2020; Sai et al., EMNLP 2021;
metamorphic testing, Chen et al. 1998).

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
score drops is the planned check. A value that does not drop because the original was already at the floor (a two-word
title cannot get shorter) is reported as `unchanged_at_floor`, not as a failure.

Results: see `README.md` "Latest run" and the Self-test sheet of the run's `report.xlsx`.

### 9.2 Choosing the NLI model (`tools/compare_nli.py`)

Each candidate scores 40 KOs' own title and description, and ones swapped in from other
projects; the share of pairs where the true field scores higher. Measured against the
file, on the June 2026 export (`--reference file`, the tool's default, with `--input` set
to that export); `--reference summary` repeats the check against the summary, the
reference the score now uses, which has not been run yet:

| model | overall | XNLI languages | other EU languages | speed (pairs/s, CPU) |
|---|---|---|---|---|
| `joeddav/xlm-roberta-large-xnli` | **0.938** | **1.000** | **0.875** | 0.97 |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | 0.775 | 0.750 | 0.800 | 1.5 |

XLM-R large is used. It was better in both language groups despite mDeBERTa's broader
multilingual training data. Evidence: `validation/runs/nli_model_comparison.json`.

---

## 10. LLM configuration

Documented in [`../USING_LLMS.md`](../USING_LLMS.md). `core/llm.py` adapts
`semantic_vs_search/llm_client.py` (disk cache keyed on the whole request, retries,
`reasoning_effort` handling, token accounting) and resolves a model preference against
the provider's `/v1/models` (the shortest matching id wins, so the LLM pillar's model is
named in full: `glm-5.2`). Provider: Scaleway, OpenAI-compatible; credentials from
`farm_assistant_um/.env`, then `scout/.env`, then this repo's `.env`. `reasoning_effort=none` matters:
without it qwen3.5 spent 1,640 s on work that takes 113 s.

---

## 11. Regenerating everything

```bash
cd assess_ko_quality
python pipeline.py --learn --limit 1              # relearn the compliance profile (1 LLM call)
python -m stages.gate --calibrate                 # gate classifier and thresholds
python -m tools.compare_nli --limit 40            # NLI model choice
python pipeline.py --limit 20 --selftest          # scores + perturbation tests (English KOs; up to 160 LLM calls)
python pipeline.py --limit 20 --no-llm-score --no-gate-llm   # the measured pillars only, no LLM calls
python -m tools.md_to_docx METHODS.md LITERATURE_REVIEW.md \
    --out How_the_KO_quality_pipeline_works.docx --title "How the KO quality pipeline works"
```

The corpus index rebuilds automatically when the export or a setting that affects it
changes.

---

## 12. Limitations

- **Weights are a convention.** Equal weights are the common choice, not a finding;
  no method can show the "right" weights without an external criterion. Stage 5 shows
  how much they matter.
- **Some settings are choices**, marked as such in `LITERATURE_REVIEW.md` §7.4: the length
  fall-off, chunk sizes, chance-baseline percentiles, the key-term count.
- **Coverage is under-measured across languages.** Matching a file's key terms to a
  field in another language is weak (correct English keywords for a Latvian file reached
  0.1), so usefulness understates cross-language metadata, and diversity can then
  outweigh coverage. Consistency still penalises wrong keywords.
- **Accuracy is weaker outside the XNLI languages** (0.875 vs 1.000 in 9.2), and NLI
  reads descriptions phrased as statements about the document ("This report explores
  ...") as neutral.
- **The findability index is built from the export**, not the live OpenSearch cluster,
  and represents each KO by the first 2,000 characters of its summary for dense
  retrieval.
- **Readability formulas are coarse** outside the languages they were built on; LIX
  rises with long compound words (German, Dutch, Finnish, Greek).
- **A field the contributor left empty** is recommended from the `*_llm` version
  without a faithfulness condition (5.5); no published floor exists to impose one.
- **The reference text is the summary the `*_llm` fields were written from.** Measuring
  against it favours text written from it. On the English KOs of the June 2026 export,
  the share of a field's words found in the summary was higher for the existing `*_llm`
  fields than for the contributor's (titles +0.11, descriptions +0.19), while against
  the file itself the contributor's were ahead (-0.02, -0.05). The contributor wrote
  from the file and the `*_llm` from the summary, so a detail taken from the file that
  the summary left out counts against the contributor. Results are measured against the
  summary and should be reported that way.
- **The summary is short and does not grow with the file** (about 200 words at any
  length; Spearman 0.38 with the extracted text's length), so for a long file most of
  its content is not in the reference, and a field is not credited for describing what
  the summary omits.
- **Cross-lingual measurement without `--resummarise`.** For 82% of 2,308 non-English
  files checked, the export's summary is in English, so a non-English contributor's
  fields are measured against another language: the lexical parts abstain (§5.3) and
  NLI reads cross-lingually. `--resummarise` (version 15) removes this for the KOs it
  covers by writing the reference in the document's own language; those results are
  not comparable with runs on the export's summaries, and scores are never pooled
  across languages either way.
- **The LLM pillar is not validated,** and it is a quarter of every field score. LLM
  judges may favour LLM-written text, which here is the `*_llm` version; it cannot see
  corpus-level findability or specificity, and does not count characters reliably
  (Section 6). Accepted for now; the planned check is the self-test's (9.1).
- **Factual correctness of the file itself is not assessed.** The score checks that the
  metadata agrees with the summary, not that the file's advice is right.
- **Cost.** About 100 s per KO on CPU: 98-102 s in the October 2026 runs with the summary
  as the reference and the LLM answers mostly cached, 134 s for two long PDFs that were
  resummarised (Intel Core Ultra 9 185H laptop). Nearly all of it is the embedding and
  entailment models. The full export of 11,627 KOs needs about 13 days of CPU time at this
  rate, plus the wait for any LLM calls not already cached. The LLM pillar adds up to 8
  calls per KO.

---

## 13. What changed from the previous version

**Version 15 (30 October 2026).**
- `--resummarise` (still opt-in) writes the reference summary **in the document's
  own language**, and that summary becomes the reference for every per-KO measure,
  the LLM pillar and the generated fields; its English version is the KO's row in
  the corpus index and the secondary reference for English-language fields on a
  non-English document (replacing version 14's document-language secondary). The
  summary, its English version and the platform's are stored with every run
  (`summary` in `details.jsonl`; the Summaries sheet).
- The file is read as it is, whatever it is: no page cap, one page per image (no
  stacking); office files converted by LibreOffice; videos and audio downloaded, the
  sound transcribed by Whisper, frames shown with the transcript. Long documents are
  summarised by hierarchical merging of per-chunk notes; every draft is verified
  against its pages and revised when the check finds errors or gaps (§4c).
- Amended 31 October 2026: a document longer than one chunk is verified sentence by
  sentence, with the verdicts combined across chunks (§4c). Before, each chunk listed
  errors itself and flagged statements that were simply on other pages: on a 33-page
  document 12 of 13 listed errors were of that kind, and the revision removed two
  true figures. Documents of one chunk are verified as before, so their summaries and
  cached calls are unchanged.
- BM25 findability gates on the corpus index's language (English) rather than the
  reference's; key terms for coverage are filtered with the reference's own language;
  the keyword diversity stemmer follows the reference's language.
- The summary check's language flag is `summary_language_mismatch`, against the
  language the summary is meant to be in. `--ids` selects named KOs.
- Runs with `--resummarise` are not comparable with version 14's.

**Version 14 (9 October 2026).**
- `--resummarise` returns two summaries from the one call: English (still *the*
  reference) and, for non-English documents, the same summary in the document's
  language. Keywords matching that language get their faithfulness measured
  against it instead of abstaining (§4c, §5.3). BM25 findability and coverage
  stay not evaluable cross-lingually, with the reasons documented.
- PDFs over the provider's 12-images-per-prompt limit are sent as vertically
  stacked JPEG tiles - every page, one call; video/audio/office files are
  skipped before download.

**Version 13 (8 October 2026).**
- Cross-lingual fields: a field that does not read as the reference's language
  (wordfreq known-word share; open-set langdetect misreads short fields) has its
  purely lexical parts marked not evaluable instead of scored against text they
  cannot match: `bm25_findability` (functional is then semantic findability
  alone), `usefulness.coverage`, and `consistency.accuracy` for keywords (its
  chance threshold is same-language-calibrated). NLI accuracy for
  title/subtitle/description stays - XNLI is cross-lingual by construction. A
  version whose faithfulness is not evaluable passes the recommendation guard: a
  measurement gap, not a verdict. Non-English field scores are not comparable
  with version 12.
- The generation brief caps keywords at 10 (a brief choice; count is never
  scored).
- Per-language reporting and calibration: the run summary and the Run sheet
  break scores down by file language (scores are not comparable across
  languages: instrument coverage differs), and the stage-3b percentile flags are
  anchored within each file-language group of at least 30 checked KOs (run-wide
  fallback, anchor recorded per KO) - cross-lingual NLI/coverage readings run
  systematically lower, so a run-wide percentile would flag non-English KOs for
  being non-English.

**Version 12 (8 October 2026).**
- The LLM variant of the fields is written fresh from the reference summary by the
  current model, one call per KO, blind - the summary and the published rules only,
  never the contributor's fields (§4d). The export's `*_llm` fields are scored only
  with `--use-existing-llm`. This returns to version 9's blind generation, now with
  the writer and the judge in different model families and the per-field scoring of
  versions 10-11. LLM-variant columns are not comparable with earlier versions: a
  different author wrote the fields.
- The LLM pillar's prompt names the KO's declared language and asks the judge to rate
  meaning across languages without penalising the language itself (Hada et al.,
  Findings of EACL 2024) (§6). LLM-pillar values are not comparable with version 11's.
- `--resummarise` now summarises the KO's own file - downloaded from `@id` and shown
  to the vision model as images - instead of the export's extracted text, which can be
  wrong (an English translation was found stored as a Lithuanian factsheet's text).
  Export text remains the per-KO fallback, and the basis is recorded (§4c). The same
  call names the document's language (`file_language`); the generated fields are
  written in that language, and the KOs sheet drops the stored-text "detected
  language" column in its favour.
- JSON answers broken by unescaped quotes inside strings (non-English typography:
  `„Horizontas 2020"`) are repaired instead of discarded (core/llm.py
  `complete_json_robust`, used by every LLM stage).
- Per-run model selection: `--generate-model`, `--judge-model`, `--resummarise-model`
  (§8).

**Version 11 (8 October 2026).**
- Every language is assessed by default (`--languages all`); English-only remains
  available as a filter. Non-English results are reported per language (§1, §12).
- Stage 3b, the summary reference check (§4b): compression, key-term coverage, NLI
  support and language of the reference summary, reported next to the scores with a
  `reference_reliability` verdict; corpus-relative flags are learned from the run.
- `--resummarise` (§4c): the reference summary can be regenerated from the file text
  with the currently served model, one counted call per KO; the platform's summary is
  kept on the record. Such runs are not comparable with runs on the export's summaries.
- The LLM pillar's call also returns the LLM's own 0-100 reading of each measured
  pillar's criteria, reported next to the measured values (two sources per pillar);
  only the overall score enters the field score (§6). LLM-pillar values are therefore
  not comparable with version 10's.

**Version 10 (24 September 2026).**
- The comparison is the contributor's version against the `*_llm` fields already in the
  export; no new LLM version is written. Stage 5 (generation plus a side-by-side judge)
  was removed; its code is kept in `legacy/metadata_llm.py`.
- An LLM's 0-100 score of each field, one call per field per version, is the fourth
  pillar of the field score, with the weight of each measured pillar.

**Version 9 (24 September 2026).**
- The fields are measured against the KO's summary instead of the file (5.1, 12).
- English KOs are assessed by default.
- The LLM version compared with the contributor's was written from the summary alone
  (`glm@blind`), and an LLM judge (gpt-oss-120b) scored both versions next to the
  measured score. Both were replaced in version 10.

Version 8 made an empty field score 0 and count in the KO score. Numbers quoted in
Sections 5.5 and 6, and in `README.md` "Latest run", come from versions 7 and 8,
measured against the file.

The previous version (September 2026, now in `legacy/reviewer_fitted/`) fitted a quality
score to the 2024 human review with ridge regression on content length, MTLD and an LLM
judge's verdict, and scored the **content**. It was replaced because reviewer ratings are
not used as the yardstick, and because the contributor controls the metadata, not the
file. Its content measures live on, reported only, in stage 3; the domain gate and
compliance stages carried over with the changes in Sections 2-3.
