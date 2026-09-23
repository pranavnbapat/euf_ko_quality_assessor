# Methods reference

Every method, threshold and weight the KO quality pipeline uses, where each comes from,
and how it is checked. `README.md` says how to run it; `LITERATURE_REVIEW.md` gives the
full sources and a per-number evidence table; `validation/metadata_spec.json` holds the
configurable settings with their provenance. This file is precise enough to reimplement.

**What is scored.** A Knowledge Object (KO) is a file plus the metadata that describes
it. The score measures the **metadata** - title, subtitle, description, keywords -
against the **file**: is it correct, informative, well-formed and findable? The file's
own quality is the original author's work; it is measured and reported, and never
changes the score.

**How it is justified.** Only measurable, reproducible criteria from the literature and
standards. Human reviewer ratings are not used to fit or validate anything - they are
subjective and error-prone. Validation is technical: every sub-pillar must detect a
deliberate defect in its own property (Section 9).

---

## 0. The pipeline at a glance

```
select ─► 1 GATE ─► 2 COMPLIANCE ─► 3 CONTENT ─► 4 METADATA SCORE ─► 5 LLM REVIEW ─► 6 ANALYSIS
```

| stage | question | output | affects the score | LLM calls |
|---|---|---|---|---|
| select | which KOs to assess | `selection` in `run.json` | - | 0 |
| 1 gate | is the KO about agriculture in the broad sense? | `gate_*` | rejected KOs are not scored | only KOs in the uncertain band (typically 0-10%) |
| 2 compliance | are the platform's required fields filled, with current vocabulary values? | `compliance_*` | no | 0 |
| 3 content | what is the file itself like? | `content_*` | no | 0 |
| 4 metadata | how well does the metadata describe the file? | `metadata_score`, pillars, fields, sub-pillars | **this is the score** | 0 |
| 5 LLM review | is each field correct, and if not, what should it say? | `llm_review.tsv` | proposals are scored by stage 4 | 1 per KO, only with `--llm-review` |
| 6 analysis | why is a score low, and how much do our choices matter? | `weakest`, `analysis.json` | no | 0 |

---

## 1. Selection

| flag | effect |
|---|---|
| `--limit N` | at most N KOs |
| `--min-chars` (500) | skip KOs whose extracted text is shorter; about 36% of the corpus is failed extractions |
| `--file-type` | matched against category, extension, mimetype and subcategory ("pdf", "Video", "Factsheet") |
| `--languages` | a KO's language is the **first** one listed |
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
| Multilingual stop words | Gopher's English stop words replaced by the language's commonest words (wordfreq) | FineWeb2, Penedo et al., COLM 2025 |
| Declared vs detected language | first declared language differs from the detected one | Kreutzer et al., TACL 2022 (mislabelled language is a common, easily detected fault) |

Also reported: word count, MTLD lexical diversity (McCarthy & Jarvis 2010, 50+ words),
LIX readability (100+ words), counts of DOIs, URLs and distinct years mentioned
(provenance cues), `content_extraction_failed` (< 50 words, Gopher's minimum).

---

## 5. Stage 4 - Metadata score

### 5.1 What the fields are checked against

**Length rules** come only from published external sources - the EIP-AGRI Common Format
(title) and Google's Dataset guidelines (description); `LITERATURE_REVIEW.md` §3.1 quotes
each. Subtitle and keywords have no published length, so none is imposed. The first
version's ranges (title 5-12 words, description 40-80 words, keywords 4-10) came from an
LLM prompt and were removed.

**The reference text**: the vision model's reading of the file (`ko_content_flat_vision`,
86% of KOs) followed by the extracted text (`ko_content_flat`), cut at 100,000
characters. Fields are scored only when the reference has at least 200 characters;
otherwise the semantic and functional sub-pillars are "not evaluable" and left out.

**The corpus index** (`core/corpus_index.py`, cached in `output/_corpus_cache/`): a BM25
index and E5 passage embeddings of every KO's reference text (never its metadata, so a
title cannot find itself), corpus document frequencies, general-language word
frequencies (wordfreq) for every detected language, per-field value-sharing counts,
null distributions for the chance baselines, and anchors for the scales below.

**Models**: `intfloat/multilingual-e5-small` for embeddings (Wang et al. 2024; the E5
family is the best public model on MMTEB, Enevoldsen et al., ICLR 2025 - the small
variant is used for CPU cost), and `joeddav/xlm-roberta-large-xnli` for NLI (XLM-R,
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
| **Semantic** | consistency | correct about the file, and coherent with the other fields | mean of **accuracy** - SummaC zero-shot NLI: each claim vs its 2 best-matching passages, max(P(entail) − P(contradict)), mean over claims, mapped to [0,1]; keywords: share present in the file or closer to it than 95% of random keywords - and **coherence** - share of 300 random KOs whose same field is less similar to this field than this KO's own | Ochoa & Duval 2006/2009 (accuracy, coherence); Laban et al., TACL 2022; Yin et al., EMNLP 2019; Meng et al., ACL 2017 |
| | density | specific rather than generic words | mean over words of min(platform IDF, 1 − Zipf/8), placed between the corpus 5th and 95th percentiles for the field | Ochoa & Duval text information content; Spärck Jones 1972; wordfreq (Speer 2022); Zipf scale (van Heuven et al. 2014) |
| | clarity | understandable | mean of: not visibly cut off; 1 − share of tokens that are unexplained acronyms; for descriptions of 40+ words, LIX (1 at ≤ 40, 0 at ≥ 60) | Barnett & Doubleday, eLife 2020; Schwartz & Hearst 2003; Veyseh et al., COLING 2020; Björnsson 1968/1983; Ochoa & Duval 2006 |
| | usefulness | carries what the file is about, without repetition | coverage of the file's 10 key terms (function words excluded, TF-IDF with sublinear tf), literal or beating chance by meaning; subtitle/description: 1 − Jaccard overlap with the title; keywords: KPEval diversity, mean of (1 − duplicate-token ratio after stemming) and emb_sim placed between corpus p5/p95 | Lin 2004 (ROUGE recall); Salton & Buckley 1988; Stvilia et al. 2007; KPEval (Wu, Yin & Chang, Findings of ACL 2024) |
| **Functional** | bm25_findability | can a keyword search find the file by this field? | search every KO's text with the field (BM25, k1 1.2, b 0.75); gain 1/log2(1 + rank of this KO); ties share the mid rank | Azzopardi & Vinay 2008; Robertson & Zaragoza 2009; Järvelin & Kekäläinen 2002; KPEval utility; BEIR (Thakur et al., NeurIPS 2021) |
| | semantic_findability | can a meaning-based search find it, in any language? | same, with E5 dense retrieval | as above; Karpukhin et al., EMNLP 2020 |

### 5.3 Which parts apply to which field

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
pillar       = mean of its evaluable sub-pillars
field score  = 100 × weighted mean of pillars              (weights: structural 1, semantic 1, functional 1)
KO score     = weighted mean over fields present           (weights: 1 each; an empty optional subtitle is left out)
```

Equal weights at every level: the default of the OECD/JRC *Handbook on Constructing
Composite Indicators* (2008), hard to beat without an external criterion (Dawes 1979).
Change them in `validation/metadata_spec.json`; stage 6 reports how much rankings would
move under data-driven alternatives. A required field that is empty scores 0.

The same code scores the contributor's metadata (`original`), the LLM-improved fields
already in the export (`llm`, from `*_llm`), and stage 5's proposals, so the three are
comparable number for number.

---

## 6. Stage 5 - LLM review (`--llm-review`)

One call per KO (`stages/metadata_llm.py`, model from `llm_review.model`, default glm).

1. The model receives the file's content (first 6,000 characters of the reference
   text), the current metadata, the six criteria, and the platform length rules.
2. For each field it returns a verdict - `correct`, `needs_change`, or `absent` for an
   empty optional subtitle - the criteria that fail, and a proposal **only** when a
   change is needed. It is told not to create a subtitle that does not exist.
3. Every proposal is scored by stage 4 exactly like the original.
4. A proposal replaces the field only if **its field score is at least 1 point higher
   AND its accuracy against the file is not lower by more than 0.02.**

The six criteria given to the model are the scorer's own properties in plain words:
faithful, specific, coherent, clear, findable, well-formed.

Why the accuracy constraint: the scorer rewards things an LLM can produce cheaply
(lengths in range, search words); optimising against a proxy measure degrades the true
objective (Gao, Schulman & Hilton, ICML 2023). Faithfulness is the objective, so it is a
constraint, not a term to trade. Why a verdict first: the existing `*_llm` prompt allowed
returning a field unchanged, yet 76 of 96 titles and 90 of 96 descriptions were
rewritten with no record of whether the original was wrong; LLM judges have documented
biases (Zheng et al., NeurIPS 2023; Wang et al., ACL 2024).

`--llm-language keep|source|english` sets the language of proposals (default: the same
language as the field it replaces). Where the model calls a field correct but the field
scores under 50, the row is flagged as a **disagreement** - those are the cases a person
should look at.

---

## 7. Stage 6 - Analysis

| output | what | source |
|---|---|---|
| `weakest` (per KO) | the weakest sub-pillars (< 0.5) across fields, each with its evidence in plain words: "ranks 8,628 of 10,468 when used as a search query", "unexplained acronyms: GESVAC" | - |
| `robustness_pillars`, `robustness_subpillars` | the composite recomputed with entropy, CRITIC and PCA weights; Spearman with the nominal ranking, mean and maximum rank shift, share of KOs moving over 10% of the ranking | OECD/JRC 2008; Saisana, Saltelli & Tarantola 2005; Diakoulaki et al. 1995 |
| `redundant_subpillars` | pairs correlated at \|ρ\| ≥ 0.9 across KOs | OECD/JRC 2008 multivariate step |
| `length_bias` | Spearman correlation of each sub-pillar with field length | the reason type-token ratio was dropped |

Per-KO multi-dimensional reporting - pillar and field scores next to the composite, not
only a single number - follows HELM's argument that one aggregate hides trade-offs
(Liang et al., TMLR 2023).

---

## 8. Settings

`validation/metadata_spec.json` holds every configurable number with its provenance
(field ranges and weights, pillar weights, readability cut-offs, models, reference
limits, coverage and coherence baselines, diversity anchor sample, BM25, specificity,
LLM review). Values learned from the corpus are not stored there: they are computed from
the export and cached with the index, keyed on the export file and the settings.

Environment: `KOQ_GATE_MODEL`, `KOQ_GATE_LLM_LOW`, `KOQ_GATE_LLM_HIGH` (gate);
`KOQ_LLM_MODEL` (default LLM preference); `KO_DM_DIR` (controlled vocabularies).

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

A value that does not drop because the original was already at the floor (a two-word
title cannot get shorter) is reported as `unchanged_at_floor`, not as a failure.

Results: see `README.md` "Latest run" and the run's `selftest.tsv`.

### 9.2 Choosing the NLI model (`tools/compare_nli.py`)

Each candidate scores 40 KOs' own title and description, and ones swapped in from other
projects; the share of pairs where the true field scores higher:

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
the provider's `/v1/models`. Provider: Scaleway, OpenAI-compatible; credentials from
`farm_assistant_um/.env`, then this repo's `.env`. `reasoning_effort=none` matters:
without it qwen3.5 spent 1,640 s on work that takes 113 s.

---

## 11. Regenerating everything

```bash
cd assess_ko_quality
python pipeline.py --learn --limit 1              # relearn the compliance profile (1 LLM call)
python -m stages.gate --calibrate                 # gate classifier and thresholds
python -m tools.compare_nli --limit 40            # NLI model choice
python pipeline.py --limit 20 --selftest          # scores + perturbation tests
python -m tools.md_to_docx METHODS.md LITERATURE_REVIEW.md \
    --out How_the_KO_quality_pipeline_works.docx --title "How the KO quality pipeline works"
```

The corpus index rebuilds automatically when the export or a setting that affects it
changes.

---

## 12. Limitations

- **Weights are a convention.** Equal weights are the documented default, not a finding;
  no method can show the "right" weights without an external criterion. Stage 6 shows
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
  and represents each KO by its first 2,000 characters for dense retrieval.
- **Readability formulas are coarse** outside the languages they were built on; LIX
  rises with long compound words (German, Dutch, Finnish, Greek).
- **Factual correctness of the file itself is not assessed.** The score checks that the
  metadata agrees with the file, not that the file's advice is right.
- **Cost.** About 25-30 s per KO on CPU, mostly NLI; the full corpus takes about 3 days
  on one machine unless NLI runs on a GPU.

---

## 13. What changed from the previous version

The previous version (September 2026, now in `legacy/reviewer_fitted/`) fitted a quality
score to the 2024 human review with ridge regression on content length, MTLD and an LLM
judge's verdict, and scored the **content**. It was replaced because reviewer ratings are
not used as the yardstick, and because the contributor controls the metadata, not the
file. Its content measures live on, reported only, in stage 3; the domain gate and
compliance stages carried over with the changes in Sections 2-3.
