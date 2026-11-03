# The KO quality pipeline: sources

The sources behind every rule, threshold and weight of the KO quality pipeline. The
method itself, without sources, is in `METHOD_NO_SOURCES.md` (Word:
`How_the_KO_quality_pipeline_works_without_sources.docx`); the section numbers below
refer to it.

**Kinds of backing.** *Published method*: the setting is the one the cited work defines.
*Published method, adapted*: the cited method with a stated change. *Standard*: a
formal standard. *Published guideline, format, guidance or finding*: a rule or result a published guideline, format or study states. *Common
practice / software default*: what the cited work reports as usual, or a tool's default.
*Learned*: computed from the corpus by a stated rule. *Choice*: no published basis; set
for a stated reason and open to change.

---

## 1. Frameworks behind the pillars and sub-pillars

| framework | what the pipeline takes from it |
|---|---|
| ISO/IEC 25012:2008, measures in ISO/IEC 25024:2015 | the data-quality characteristics: accuracy (syntactic → noise; semantic → consistency), completeness, consistency (→ formatting; coherence), precision (→ density), understandability (→ clarity), accessibility (→ findability), compliance (→ the separate compliance stage) |
| Wang & Strong (1996) | quality is more than accuracy: intrinsic, contextual, representational and accessibility dimensions - the argument for a functional pillar |
| Bruce & Hillmann (2004) | seven metadata-specific criteria: completeness, accuracy, provenance, conformance to expectations, logical consistency and coherence, timeliness, accessibility |
| Ochoa & Duval (2006, 2009) | computable versions of those criteria: accuracy (semantic distance between metadata and document), coherence (same topic in title, description and keywords), text information content (TF-IDF), readability of the description, completeness (the fields that hold a value) |
| Stvilia, Gasser, Twidale & Smith (2007) | intrinsic / relational / reputational dimensions; the informativeness/redundancy dimension behind "adds to the title" |
| Park (2009) | accuracy, completeness and consistency are the criteria most used in practice |
| Wilkinson et al. (2016), FAIR; Devaraju & Huber (2021), F-UJI | findability as a quality criterion of the metadata (F2 rich metadata, F4 indexed in a searchable resource), assessable automatically |

---

## 2. Stage by stage

### Stage 1 - Domain gate (Section 2)

| part | source |
|---|---|
| sentence-embedding classifier | Sentence-BERT (Reimers & Gurevych, EMNLP 2019) |
| deferring only the uncertain band to an LLM | the reject option (Chow 1970); selective classification (Geifman & El-Yaniv, NeurIPS 2017) |
| why the LLM decides only the uncertain band | documented LLM-judge biases: position, verbosity, self-preference (Zheng et al., NeurIPS 2023) |

### Stage 2 - Compliance (Section 3)

| part | source |
|---|---|
| compliance as its own pass/fail characteristic, outside the score | ISO/IEC 25012 (compliance characteristic) |
| required fields, vocabulary drift, age | learned from the export; no external source |

### Stage 3 - Content report (Section 4)

| measurement | thresholds | source |
|---|---|---|
| Gopher quality | 50-100,000 words; mean word length 3-10; symbol/word <= 0.1; bullet lines <= 90%; ellipsis lines <= 30%; >= 80% words with a letter; >= 2 stop words | Rae et al. (2021); datatrove defaults |
| Gopher repetition | duplicate lines <= 30% (chars <= 20%), paragraphs likewise; top 2/3/4-gram char share <= 0.20/0.18/0.16; duplicated 5-10-gram share <= 0.15-0.10 | Rae et al. (2021) |
| FineWeb filters | lines ending in punctuation > 12%; chars in duplicated lines < 10%; lines under 30 chars < 67% | Penedo et al., NeurIPS 2024 (selected by ablation from over 50 candidates) |
| C4 rules kept by FineWeb | no "lorem ipsum", curly brackets or javascript lines (FineWeb drops C4's terminal-punctuation rule: it removed 30% of tokens) | Raffel et al. (2020); Dodge et al., EMNLP 2021 |
| Stop words per language | the language's 8 commonest words (wordfreq), not Gopher's English list | adapted from FineWeb2 (Penedo et al., COLM 2025), which takes each language's stop words above a frequency threshold, at least eight |
| Declared vs detected language | first declared language vs detected | Kreutzer et al., TACL 2022: mislabelled language is common and easy to detect |
| Lexical richness | MTLD, 50+ words | McCarthy & Jarvis (2010) |
| Readability | LIX, 100+ words | Björnsson (1968, 1983) |
| Provenance cues | DOIs, URLs, distinct years mentioned | Bruce & Hillmann (2004) provenance |

### Stage 4 - Metadata score: the length rules, quoted (Section 5.1)

| field | rule | source, verbatim |
|---|---|---|
| title | at most 150 characters, spaces not counted | EIP-AGRI Common Format (European Commission, 2015), project title: "short and easily understandable (one key sentence on the project; max 150 characters, word count – no spaces)" |
| description | 50 to 5,000 characters | Google Search Central, Dataset structured data: "The summary must be between 50 and 5000 characters long." |
| subtitle, keywords | none | no published source gives a length or a count |

What the sources do not support: Google sets no length for titles or meta descriptions
("While there's no limit on how long a `<title>` element can be, the title link is
truncated in Google Search results as needed"; "There's no limit on how long a meta
description can be"). Titles should be "descriptive and concise", avoiding "vague
descriptors" (Google Search Central, title links), and datasets should have unique names
(Dataset structured data).

### Stage 4 - Metadata score: every setting (Sections 5.2-5.4)

| sub-pillar | setting | value | kind | basis |
|---|---|---|---|---|
| length | title | at most 150 characters, spaces not counted | published format | EIP-AGRI Common Format (European Commission 2015) |
| length | subtitle | none | - | no published source gives a length |
| length | description | 50-5,000 characters | published guideline | Google Search Central, Dataset structured data |
| length | keywords | none | - | no published source gives a count |
| length | fall-off outside a limit | linear to 0 at 2x a maximum, or at 0 below a minimum | choice | desirability form (Derringer & Suich 1980); the 2x is a choice |
| completeness | title: not an internal project code, not the file name | - | published guidance | EIP-AGRI ("short and easily understandable"); Google ("descriptive", "avoid vague descriptors") |
| completeness | checks | present; not a filename; not a copy of another field; not shared verbatim across projects | published method | Bruce & Hillmann (2004) completeness and conformance to expectations; Ochoa & Duval (2006) |
| noise | token types | markup, URLs, encoding debris, symbol runs, stutters | published method | C4 (Raffel et al. 2020) and Gopher (Rae et al. 2021) text-quality rules; ISO 25012 syntactic accuracy |
| formatting | keyword form | one concept per term, no duplicates, concise | standard | ANSI/NISO Z39.19-2005 |
| formatting | not all capitals | - | published finding | Tinker (1963) |
| formatting | longest keyword | corpus 99th percentile (5 words) | learned | Z39.19 asks for concise terms but sets no limit |
| consistency | accuracy definition | metadata vs document | published method | Ochoa & Duval (2006, 2009) |
| consistency | accuracy scoring | max(P(entail) - P(contradict)) over the best passages, mean over claims | published method, adapted | SummaC zero-shot (Laban et al. 2022): maximum over premises, mean over claims; entailment minus contradiction per passage is the adaptation |
| consistency | title hypothesis | "This text is about {title}." | published method | Yin, Hay & Roth (2019) |
| consistency | passages per claim / description sentences / passages searched | 2 / 8 / 40 x 120 words | choice | compute bounds; SummaC also scores against the best-matching premise |
| consistency | keyword support threshold | 95th percentile of random keywords | choice | the usual one-sided 5% level; learned per file |
| consistency | coherence definition | title, description, keywords about the same topic | published method | Ochoa & Duval (2006) |
| consistency | coherence baseline | 300 random KOs | choice | sample size; ranks against chance instead of a raw cosine |
| density | word specificity | min(platform IDF, 1 - Zipf/8) | published method | Ochoa & Duval text information content (TF-IDF); Spärck Jones (1972); wordfreq (Speer 2022), Zipf scale (van Heuven et al. 2014) |
| density | scale anchors | corpus 5th-95th percentile | choice | min-max normalisation with outlier trimming, one of the options in the OECD/JRC handbook |
| clarity | readability index | LIX | published method | Björnsson (1968), applied in 11 languages (Björnsson 1983); applied to descriptions as in Ochoa & Duval (2006) |
| clarity | readability cut-offs | full at 40 or below, zero at 60 or above | published scale | Björnsson's scale: about 20 very easy to 60 very difficult |
| clarity | minimum words for readability | 40 | choice | readability formulas assume samples of about 100 words; below 40 words one or two sentences dominate the index |
| clarity | acronyms | share of tokens that are unexplained acronyms | published finding + method | Barnett & Doubleday (2020); Schwartz & Hearst (2003) |
| usefulness | coverage | recall of the file's salient words | published method | ROUGE-style recall (Lin 2004); TF-IDF (Salton & Buckley 1988); sublinear tf (Manning, Raghavan & Schütze 2008) |
| usefulness | number of salient words | 10 | convention | the usual keyphrase evaluation cut-offs are 5 and 10 (e.g. Meng et al. 2017) |
| usefulness | function-word exclusion for key terms | Zipf >= 6 in any corpus language | choice | Zipf scale (van Heuven et al. 2014): 6 = once per 1,000 words, the function-word band; stopword exclusion is standard in keyphrase extraction |
| usefulness | keyword diversity | mean(1 - dup_token_ratio, anchored emb_sim) | published method | KPEval (Findings of ACL 2024); Bahuleyan & El Asri (COLING 2020) |
| usefulness | diversity anchors | corpus 5th-95th percentile over 1,000 keyword sets | choice | emb_sim has no natural scale with E5 |
| usefulness | redundancy | 1 - Jaccard with the title | published method | Stvilia et al. (2007) informativeness/redundancy; Jaccard (1912) |
| findability | measure | rank of the KO when searching with the field | published method | retrievability (Azzopardi & Vinay 2008); known-item queries (Azzopardi, de Rijke & Balog 2007) |
| findability | rank to score | 1 / log2(1 + rank) | published method | nDCG discount in its standard form (Burges et al., ICML 2005), after Järvelin & Kekäläinen (2002) |
| findability | BM25 parameters | k1 = 1.2, b = 0.75 | software default | OpenSearch defaults; at the edge of the ranges Robertson & Zaragoza (2009) report as reasonably good |
| findability | passage per KO | first 2,000 characters of the reference text | choice | the embedding model reads at most 512 tokens |
| all semantic and functional parts | reference text | the KO's summary (`ko_content_flat_summarised`); every language since version 11. With `--resummarise` (version 15), a fresh summary of the downloaded file **in the document's own language**, which replaces it as the reference; its English version is the KO's row in the corpus index and the secondary reference for English-language fields on a non-English document | choice | the LLM version is written from the same summary, so both versions are measured against the text it was written from; it favours text written from the summary (Section 11). Up to version 8: the vision reading followed by the extracted text. The sources on accuracy (Ochoa & Duval; SummaC; FactCC) check against the document itself |
| cross-lingual evaluability | known-word share | a field reads as a language when at least half of its (at least 2) alphabetic tokens are known words of it, wordfreq Zipf >= 2.0; below that, the lexical parts - bm25_findability, usefulness.coverage, keywords' accuracy - are not evaluable and their weight redistributes; with `--resummarise` (version 15) the reference is in the document's language, so bm25_findability gates on the corpus index's language (English) instead of the reference's | choice | wordfreq (Speer 2022); open-set langdetect rejected by measurement (it misreads short fields); exclusion rather than a zero keeps an unmeasurable property out of the score |
| generation (version 12) | brief | blind - the summary and the published rules only, never the contributor's fields; written in the document's language; at most 10 keywords | choice + convention | blind generation keeps the text under assessment out of the writer's context; 10 follows the usual keyphrase cut-offs (Meng et al. 2017); the count is never scored |
| resummarise (version 15) | file handling | the file downloaded from its URL and read as it is, never cut short: an image, or a PDF of up to 100 pages, as page images (one per image); a PDF of more than 100 pages as its own text layer, in chunks of at most 100,000 characters (amended 3 November 2026); a PDF over 100 MB not downloaded (the export's text instead); office files converted to PDF first, a video's sound transcribed by Whisper and its frames shown, an audio file as its transcript | choice + published model + published method | the export's stored text can be a translation of the file, so the file itself is read; 12 images per prompt is the provider's measured limit; the 100-page and 100 MB limits are choices that bound a run's downloads and image requests; hierarchical merging over text is the setting BooookScore (Chang et al., ICLR 2024) studies; an unusable text layer is caught by the Gopher quality filter's word rules (Rae et al. 2021: at least 50 words, at least 80% of them containing a letter); Whisper (Radford et al., ICML 2023) transcribes zero-shot across languages |
| resummarise (version 15) | long documents | notes per chunk of 12 pages, merged 8 at a time until one summary remains | published method | hierarchical merging, the more coherent of the two workflows BooookScore compares (Chang et al., ICLR 2024) |
| resummarise (version 15) | verification | every draft read against its source for contradicted or unsupported statements and missing content; a source longer than one chunk checked sentence by sentence, each chunk giving every sentence a verdict (supported, contradicted, absent) and a sentence kept when any chunk supports it; one revision call applies the findings; none when there are none | published methods, adapted | Chain-of-Verification (Dhuliawala et al., Findings of ACL 2024): draft, verify independently, revise; across chunks, each sentence's best support counts, as in AlignScore (Zha et al., ACL 2023) and SummaC-ZS (Laban et al., TACL 2022) |
| resummarise (version 15) | summary language | the document's own, named by the model from the pages; an English translation for the corpus row | design decision (30 October 2026) | a non-English KO's fields are then measured in their own language; the corpus index stays English, so BM25 findability gates on the index's language |
| aggregation | weights | equal at every level | common practice | OECD/JRC (2008): most composite indicators use equal weights; Dawes (1979) |
| aggregation | empty field, subtitle included | scores 0 | published method | Ochoa & Duval (2006) completeness: counts the fields that hold a value |

### Recommendation and the LLM pillar (Sections 5.4-5.5, 6)

| part | source |
|---|---|
| faithfulness as a hard condition on accepting a higher-scoring version | optimising a proxy degrades the true objective (Gao, Schulman & Hilton, ICML 2023) |
| accuracy tolerance 0.02 | choice |
| an LLM rating named criteria of each field | the approach of G-Eval (Liu et al., EMNLP 2023) |
| each version scored on its own, one call per field | scoring alone removes the position bias found in pairwise comparison (Wang et al., ACL 2024); one field per call is a choice (multi-field answers broke) |
| the LLM model wrote neither version (glm-5.2; the generator is qwen3.5 by default and the export's `*_llm` fields came from another Qwen model, so no Qwen model scores them - the judge stays out of the writer's model family) | self-preference (Panickssery, Bowman & Feng, NeurIPS 2024) cannot favour a version it did not write; glm-5.2 over gpt-oss-120b is a choice (more order-consistent in Section 6) |
| the judge's prompt names the KO's language, says the summary is English, and asks for meaning to be rated across languages without penalising the language itself | LLM evaluators are less reliable on non-English text (Hada et al., Findings of EACL 2024) |
| the LLM pillar weighs as much as each measured pillar | choice |
| known bias toward LLM-generated text, accepted for now | G-Eval (Liu et al., EMNLP 2023), whose authors report that LLM evaluators may favour LLM-generated text; the judges measured in Section 6 |

### Stage 5 - Analysis (Section 7)

| part | source |
|---|---|
| robustness under alternative weights | OECD/JRC Handbook on Constructing Composite Indicators (2008); Saisana, Saltelli & Tarantola (2005) |
| CRITIC weights | Diakoulaki, Mavrotas & Papayannakis (1995) |
| redundancy check (\|ρ\| ≥ 0.9) | OECD/JRC (2008), multivariate analysis step |
| reporting pillars and fields next to the composite | HELM (Liang et al., TMLR 2023) |

### Validation (Section 9)

| part | source |
|---|---|
| perturbation self-test | CheckList (Ribeiro et al., ACL 2020); Perturbation CheckLists for NLG metrics (Sai et al., EMNLP 2021); metamorphic testing (Chen, Cheung & Yiu 1998) |
| NLI candidates | XNLI (Conneau et al., EMNLP 2018); XLM-R (Conneau et al., ACL 2020); DeBERTaV3 (He, Gao & Chen, ICLR 2023) |
| embedding model | multilingual E5 (Wang et al. 2024); on MMTEB (Enevoldsen et al., ICLR 2025) the best publicly available model is `multilingual-e5-large-instruct`, from the same family; MTEB (Muennighoff et al., EACL 2023) |

---

## 3. Where the methods sit in the core NLP/ML venues

| part of the pipeline | paper (venue) | what was taken | used in code? |
|---|---|---|---|
| keywords: faithfulness, diversity, utility | KPEval (Wu, Yin & Chang, Findings of ACL 2024) | the three reference-free aspects; the diversity metrics dup_token_ratio and emb_sim; utility as retrieval with BM25 + dense | **yes** - diversity in usefulness; utility as findability |
| keyword diversity | Bahuleyan & El Asri (COLING 2020) | origin of KPEval's diversity metrics | via KPEval |
| keywords present / absent in the text | Meng et al. (ACL 2017) | an absent keyword can still be right, hence the semantic fallback in accuracy | **yes** |
| accuracy as faithfulness | SummaC (Laban et al., TACL 2022); Falke et al. (ACL 2019); FactCC (Kryscinski et al., EMNLP 2020) | NLI against the source, max over best premises, mean over claims | **yes** |
| title as a topic hypothesis | Yin, Hay & Roth (EMNLP 2019) | "This text is about ..." entailment template | **yes** |
| NLI models | XNLI (Conneau et al., EMNLP 2018); XLM-R (Conneau et al., ACL 2020); DeBERTaV3 (He et al., ICLR 2023) | the two candidates compared; XLM-R chosen by measurement | **yes** |
| description quality dimensions | SummEval (Fabbri et al., TACL 2021); Kryscinski et al. (EMNLP 2019) | the standard summary dimensions - consistency, relevance, coherence, fluency - map onto consistency, usefulness (coverage), coherence and noise/formatting | informs design |
| factual consistency, stronger scorers | AlignScore (Zha et al., ACL 2023); UniEval (Zhong et al., EMNLP 2022); FActScore (Min et al., EMNLP 2023) | considered; AlignScore and UniEval are English-only, FActScore needs an LLM call per atomic claim | no - noted as upgrades |
| specificity | Li & Nenkova (AAAI 2015); Ko, Durrett & Li (AAAI 2019) | specificity is an established, measurable property of text | informs density; their models are English-only |
| acronyms | Veyseh et al. (COLING 2020) | acronym identification as a task; supports the acronym part of clarity | informs clarity |
| multilingual readability | ReadMe++ (Naous et al., EMNLP 2024) | benchmark for five languages; supports treating formula readability as coarse | informs limits; not EU coverage |
| dense retrieval | DPR (Karpukhin et al., EMNLP 2020); BEIR (Thakur et al., NeurIPS 2021); Burges et al. (ICML 2005) | dense retrieval for semantic findability; BM25 as a robust baseline; the nDCG discount | **yes** |
| embedding model choice | MTEB (Muennighoff et al., EACL 2023); MMTEB (Enevoldsen et al., ICLR 2025) | the best public model on MMTEB is multilingual-e5-large-instruct; the same family's small variant is used | **yes** |
| web-text quality filters | FineWeb (Penedo et al., NeurIPS 2024); FineWeb2 (Penedo et al., COLM 2025); Dodge et al. (EMNLP 2021) | the filter set and thresholds; per-language stop words | **yes** - content report |
| language mislabelling | Kreutzer et al. (TACL 2022) | declared vs detected language check | **yes** - content report |
| metric validation | CheckList (Ribeiro et al., ACL 2020); Sai et al. (EMNLP 2021) | perturbation self-test | **yes** |
| LLM judges and their biases | MT-Bench (Zheng et al., NeurIPS 2023); Wang et al. (ACL 2024); G-Eval (Liu et al., EMNLP 2023) | verdicts are checked by measurement, never trusted alone | **yes** - review design |
| multilingual LLM evaluators | Hada et al. (Findings of EACL 2024) | LLM evaluators are less reliable on non-English text; the judge's prompt names the languages involved and asks for meaning to be rated across them without penalising the language | **yes** - the LLM pillar's language line |
| over-optimising a proxy | Gao, Schulman & Hilton (ICML 2023) | faithfulness as a hard constraint on accepting LLM proposals | **yes** |
| deferring uncertain cases | Geifman & El-Yaniv (NeurIPS 2017) | the gate's LLM band | **yes** |
| multi-metric reporting | HELM (Liang et al., TMLR 2023) | report pillars and fields next to the composite | **yes** |
| sentence embeddings | Sentence-BERT (Reimers & Gurevych, EMNLP 2019) | the gate classifier's encoder | **yes** |
| long-document summarisation (`--resummarise`) | BooookScore (Chang et al., ICLR 2024) | hierarchical merging of chunk-level notes, the more coherent of the two workflows it compares | **yes** |
| summary verification (`--resummarise`) | Chain-of-Verification (Dhuliawala et al., Findings of ACL 2024) | draft, verify against the source independently, revise | **yes**, adapted to documents |
| verdicts across chunks (`--resummarise`) | AlignScore (Zha et al., ACL 2023); SummaC-ZS (Laban et al., TACL 2022) | a draft sentence counts as supported when any chunk of a long document supports it | **yes** |
| speech transcription (`--resummarise`) | Whisper (Radford et al., ICML 2023) | zero-shot multilingual transcription of video and audio files | **yes** |

**What these venues do not cover.** The metadata-quality criteria themselves - accuracy,
completeness, coherence, conformance, provenance - come from digital-library and
information-science research (Bruce & Hillmann 2004; Ochoa & Duval 2006, 2009; Stvilia
et al., JASIST 2007; Park 2009), and from standards (ISO/IEC 25012, ANSI/NISO Z39.19,
FAIR). Known-item retrievability and BM25 come from information retrieval (CIKM, SIGIR,
Foundations and Trends in IR), and composite-index construction from the OECD/JRC
handbook and statistics journals. No NLP/ML-venue paper defines a metadata-quality
framework, so those foundations are kept; the NLP/ML papers supply the measurement
techniques.

---

## 4. How the references were checked

On 24 September 2026 every reference below was checked for existence and bibliographic
details (authors, title, venue, year, volume, pages) against Crossref, the publisher,
ACL Anthology, arXiv, conference pages, ERIC or Zenodo. The claims attributed
to them were checked against the source text where it was accessible: the Google and
EIP-AGRI quotations, Ochoa & Duval (2006), Salton & Buckley (1988), Järvelin &
Kekäläinen (2002), Burges et al. (2005), Robertson & Zaragoza (2009), the OECD/JRC
handbook, SummaC, KPEval, Yin et al. (2019), FineWeb, FineWeb2, Gopher, MMTEB, Barnett &
Doubleday (2020), Kreutzer et al. (2022) and the OpenSearch documentation.

Not read in full: Ochoa & Duval (2009), Björnsson (1968, 1983), Tinker (1963), and the
texts of ISO/IEC 25012, ISO/IEC 25024 and ANSI/NISO Z39.19. Their bibliographic details
are confirmed; the claims attributed to them rest on their abstracts, on secondary
sources, or on the 2006 paper by the same authors.

Added on 9 October 2026 with pipeline version 12: Hada et al. (2024), checked the same
way against the ACL Anthology (2024.findings-eacl.71) and arXiv; the claim attributed
to it (LLM evaluators are less reliable on non-English text, so they need calibration)
rests on the paper's abstract and findings.

Added on 30 October 2026 with pipeline version 15: Chang et al. (2024), Dhuliawala et
al. (2024) and Radford et al. (2023). Bibliographic details were checked against the
ACL Anthology (2024.findings-acl.212, pages 3563-3578), PMLR (v202/radford23a, pages
28492-28518) and arXiv (2310.00785, whose comment field names the ICLR 2024
camera-ready). The sentences quoted from them - the two prompting workflows and their
coherence/detail trade-off, the four steps of Chain-of-Verification, and the 680,000
hours of supervision with zero-shot transfer - are from the papers' abstracts.

Added on 31 October 2026, for the sentence-level verification of long documents: the
sentences quoted from AlignScore (§3.2 and the Figure 2 caption) and SummaC (§3.2,
SummaC-ZS) were checked word for word against the published PDFs on the ACL Anthology
(2023.acl-long.634, pages 11328-11348; 2022.tacl-1.10, TACL volume 10, pages
163-177). The max over the source's parts is SummaC's zero-shot variant only; its
convolutional variant uses the whole distribution of scores instead.

---

## 5. References

- Anderson, J. (1983). Lix and Rix: variations on a little-known readability index. *Journal of Reading*, 26(6), 490-496.
- Azzopardi, L., & Vinay, V. (2008). Retrievability: an evaluation measure for higher order information access tasks. *CIKM '08*, 561-570.
- Azzopardi, L., de Rijke, M., & Balog, K. (2007). Building simulated queries for known-item topics: an analysis using six European languages. *SIGIR '07*, 455-462.
- Barnett, A., & Doubleday, Z. (2020). Meta-Research: the growth of acronyms in the scientific literature. *eLife*, 9, e60080.
- Bahuleyan, H., & El Asri, L. (2020). Diverse keyphrase generation with neural unlikelihood training. *COLING 2020*.
- Barzilay, R., & Lapata, M. (2008). Modeling local coherence: an entity-based approach. *Computational Linguistics*, 34(1), 1-34.
- Björnsson, C. H. (1968). *Läsbarhet*. Stockholm: Liber.
- Björnsson, C. H. (1983). Readability of newspapers in 11 languages. *Reading Research Quarterly*, 18(4), 480-497.
- Bruce, T. R., & Hillmann, D. I. (2004). The continuum of metadata quality: defining, expressing, exploiting. In D. I. Hillmann & E. L. Westbrooks (Eds.), *Metadata in Practice* (pp. 238-256). ALA Editions.
- Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., & Hullender, G. (2005). Learning to rank using gradient descent. *ICML 2005*, 89-96.
- Chen, T. Y., Cheung, S. C., & Yiu, S. M. (1998). *Metamorphic testing: a new approach for generating next test cases*. Technical Report HKUST-CS98-01, Hong Kong University of Science and Technology.
- Chang, Y., Lo, K., Goyal, T., & Iyyer, M. (2024). BooookScore: a systematic exploration of book-length summarization in the era of LLMs. *ICLR 2024*. arXiv:2310.00785.
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff. *IEEE Transactions on Information Theory*, 16(1), 41-46.
- Conneau, A., Rinott, R., Lample, G., Williams, A., Bowman, S., Schwenk, H., & Stoyanov, V. (2018). XNLI: evaluating cross-lingual sentence representations. *EMNLP 2018*. arXiv:1809.05053.
- Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*. arXiv:1911.02116.
- Dawes, R. M. (1979). The robust beauty of improper linear models in decision making. *American Psychologist*, 34(7), 571-582.
- Derringer, G., & Suich, R. (1980). Simultaneous optimization of several response variables. *Journal of Quality Technology*, 12(4), 214-219.
- Devaraju, A., & Huber, R. (2021). An automated solution for measuring the progress toward FAIR research data. *Patterns*, 2(11), 100370.
- Dhuliawala, S., Komeili, M., Xu, J., Raileanu, R., Li, X., Celikyilmaz, A., & Weston, J. (2024). Chain-of-Verification reduces hallucination in large language models. *Findings of ACL 2024*, 3563-3578. arXiv:2309.11495.
- Diakoulaki, D., Mavrotas, G., & Papayannakis, L. (1995). Determining objective weights in multiple criteria problems: the CRITIC method. *Computers & Operations Research*, 22(7), 763-770.
- Dodge, J., Sap, M., Marasović, A., Agnew, W., Ilharco, G., Groeneveld, D., Mitchell, M., & Gardner, M. (2021). Documenting large webtext corpora: a case study on the Colossal Clean Crawled Corpus. *EMNLP 2021*.
- Enevoldsen, K., et al. (2025). MMTEB: Massive Multilingual Text Embedding Benchmark. *ICLR 2025*. arXiv:2502.13595.
- Fabbri, A. R., Kryściński, W., McCann, B., Xiong, C., Socher, R., & Radev, D. (2021). SummEval: re-evaluating summarization evaluation. *Transactions of the ACL*, 9, 391-409.
- European Commission (2015). *EIP-AGRI common format for interactive innovation projects* (version of 14 October 2015). EIP-AGRI Service Point.
- Falke, T., Ribeiro, L. F. R., Utama, P. A., Dagan, I., & Gurevych, I. (2019). Ranking generated summaries by correctness: an interesting but challenging application for natural language inference. *ACL 2019*.
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML 2023*.
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS 2017*.
- Google Search Central (2026). *Dataset structured data*; *Influencing title links*; *Control your snippets in search results*. developers.google.com/search/docs.
- Hada, R., Gumma, V., de Wynter, A., Diddee, H., Ahmed, M., Choudhury, M., Bali, K., & Sitaram, S. (2024). Are large language model-based evaluators the solution to scaling up multilingual evaluation? *Findings of EACL 2024*, 1051-1070. arXiv:2309.07462.
- He, P., Gao, J., & Chen, W. (2023). DeBERTaV3: improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *ICLR 2023*.
- ISO/IEC 25012:2008. *Software engineering - Software product Quality Requirements and Evaluation (SQuaRE) - Data quality model*.
- ISO/IEC 25024:2015. *Systems and software engineering - SQuaRE - Measurement of data quality*.
- Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*, 11(2), 37-50.
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. *EMNLP 2020*.
- Ko, W.-J., Durrett, G., & Li, J. J. (2019). Domain agnostic real-valued specificity prediction. *AAAI 2019*, 6610-6617.
- Kreutzer, J., Caswell, I., Wang, L., et al. (2022). Quality at a glance: an audit of web-crawled multilingual datasets. *Transactions of the ACL*, 10, 50-72.
- Kryściński, W., Keskar, N. S., McCann, B., Xiong, C., & Socher, R. (2019). Neural text summarization: a critical evaluation. *EMNLP-IJCNLP 2019*.
- Kryscinski, W., McCann, B., Xiong, C., & Socher, R. (2020). Evaluating the factual consistency of abstractive text summarization. *EMNLP 2020*. arXiv:1910.12840.
- Laban, P., Schnabel, T., Bennett, P. N., & Hearst, M. A. (2022). SummaC: re-visiting NLI-based models for inconsistency detection in summarization. *Transactions of the ACL*, 10, 163-177. arXiv:2111.09525.
- Li, J. J., & Nenkova, A. (2015). Fast and accurate prediction of sentence specificity. *AAAI 2015*.
- Liang, P., et al. (2023). Holistic evaluation of language models. *Transactions on Machine Learning Research*.
- Lin, C.-Y. (2004). ROUGE: a package for automatic evaluation of summaries. *Text Summarization Branches Out (ACL Workshop)*, 74-81.
- Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG evaluation using GPT-4 with better human alignment. *EMNLP 2023*.
- Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. Chapter 6, "Sublinear tf scaling".
- McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: a validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381-392.
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P. W., Iyyer, M., Zettlemoyer, L., & Hajishirzi, H. (2023). FActScore: fine-grained atomic evaluation of factual precision in long form text generation. *EMNLP 2023*.
- Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2023). MTEB: Massive Text Embedding Benchmark. *EACL 2023*.
- Naous, T., Ryan, M. J., Lavrouk, A., Chandra, M., & Xu, W. (2024). ReadMe++: benchmarking multilingual language models for multi-domain readability assessment. *EMNLP 2024*.
- Meng, R., Zhao, S., Han, S., He, D., Brusilovsky, P., & Chi, Y. (2017). Deep keyphrase generation. *ACL 2017*, 582-592. arXiv:1704.06879.
- NISO (2005, reaffirmed 2010). *ANSI/NISO Z39.19-2005 (R2010): Guidelines for the Construction, Format, and Management of Monolingual Controlled Vocabularies*. NISO Press.
- OECD & European Commission Joint Research Centre (2008). *Handbook on Constructing Composite Indicators: Methodology and User Guide*. OECD Publishing.
- Ochoa, X., & Duval, E. (2006). Quality metrics for learning object metadata. *Proceedings of ED-MEDIA 2006, World Conference on Educational Multimedia, Hypermedia and Telecommunications*, 1004-1011.
- Ochoa, X., & Duval, E. (2009). Automatic evaluation of metadata quality in digital repositories. *International Journal on Digital Libraries*, 10(2-3), 67-91.
- Park, J.-R. (2009). Metadata quality in digital repositories: a survey of the current state of the art. *Cataloging & Classification Quarterly*, 47(3-4), 213-228.
- OpenSearch (2026). *Keyword search: BM25 similarity parameters* (k1 default 1.2, b default 0.75). docs.opensearch.org.
- Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM evaluators recognize and favor their own generations. *NeurIPS 2024*.
- Penedo, G., Kydlíček, H., Ben Allal, L., Lozhkov, A., Mitchell, M., Raffel, C., Von Werra, L., & Wolf, T. (2024). The FineWeb datasets: decanting the web for the finest text data at scale. *NeurIPS 2024 Datasets and Benchmarks*. arXiv:2406.17557.
- Penedo, G., Kydlíček, H., Sabolčec, V., Messmer, B., Foroutan, N., Kargaran, A. H., Raffel, C., Jaggi, M., Von Werra, L., & Wolf, T. (2025). FineWeb2: one pipeline to scale them all - adapting pre-training data processing to every language. *COLM 2025*. arXiv:2506.20920.
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *ICML 2023*, PMLR 202, 28492-28518. arXiv:2212.04356.
- Rae, J. W., et al. (2021). Scaling language models: methods, analysis & insights from training Gopher. arXiv:2112.11446.
- Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research*, 21(140), 1-67. arXiv:1910.10683.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: sentence embeddings using Siamese BERT-networks. *EMNLP-IJCNLP 2019*. arXiv:1908.10084.
- Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond accuracy: behavioral testing of NLP models with CheckList. *ACL 2020*. arXiv:2005.04118.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). BEIR: a heterogeneous benchmark for zero-shot evaluation of information retrieval models. *NeurIPS 2021 Datasets and Benchmarks*.
- Sai, A. B., Dixit, T., Sheth, D. Y., Mohan, S., & Khapra, M. M. (2021). Perturbation CheckLists for evaluating NLG evaluation metrics. *EMNLP 2021*.
- Saisana, M., Saltelli, A., & Tarantola, S. (2005). Uncertainty and sensitivity analysis techniques as tools for the quality assessment of composite indicators. *Journal of the Royal Statistical Society: Series A*, 168(2), 307-323.
- Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.
- Schwartz, A. S., & Hearst, M. A. (2003). A simple algorithm for identifying abbreviation definitions in biomedical text. *Pacific Symposium on Biocomputing*, 8, 451-462.
- Speer, R. (2022). *rspeer/wordfreq: v3.0* (v3.0.2). Zenodo. doi:10.5281/zenodo.7199437.
- Spärck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. *Journal of Documentation*, 28(1), 11-21.
- Stvilia, B., Gasser, L., Twidale, M. B., & Smith, L. C. (2007). A framework for information quality assessment. *Journal of the American Society for Information Science and Technology*, 58(12), 1720-1733.
- Tinker, M. A. (1963). *Legibility of Print*. Ames: Iowa State University Press.
- Ure, J. (1971). Lexical density and register differentiation. In G. Perren & J. L. M. Trim (Eds.), *Applications of Linguistics* (pp. 443-452). Cambridge University Press.
- van Heuven, W. J. B., Mandera, P., Keuleers, E., & Brysbaert, M. (2014). SUBTLEX-UK: a new and improved word frequency database for British English. *Quarterly Journal of Experimental Psychology*, 67(6), 1176-1190.
- Wang, L., Yang, N., Huang, X., Yang, L., Majumder, R., & Wei, F. (2024). Multilingual E5 text embeddings: a technical report. arXiv:2402.05672.
- Veyseh, A. P. B., Dernoncourt, F., Tran, Q. H., & Nguyen, T. H. (2020). What does this acronym mean? Introducing a new dataset for acronym identification and disambiguation. *COLING 2020*.
- Wang, P., Li, L., Chen, L., Cai, Z., Zhu, D., Lin, B., Cao, Y., Liu, Q., Liu, T., & Sui, Z. (2024). Large language models are not fair evaluators. *ACL 2024*.
- Wang, R. Y., & Strong, D. M. (1996). Beyond accuracy: what data quality means to data consumers. *Journal of Management Information Systems*, 12(4), 5-33.
- Wenzek, G., Lachaux, M.-A., Conneau, A., Chaudhary, V., Guzmán, F., Joulin, A., & Grave, E. (2020). CCNet: extracting high quality monolingual datasets from web crawl data. *LREC 2020*. arXiv:1911.00359.
- Wilkinson, M. D., et al. (2016). The FAIR Guiding Principles for scientific data management and stewardship. *Scientific Data*, 3, 160018.
- Wu, D., Yin, D., & Chang, K.-W. (2024). KPEval: towards fine-grained semantic-based keyphrase evaluation. *Findings of ACL 2024*, 1959-1981.
- Yin, W., Hay, J., & Roth, D. (2019). Benchmarking zero-shot text classification: datasets, evaluation and entailment approach. *EMNLP-IJCNLP 2019*. arXiv:1909.00161.
- Zha, Y., Yang, Y., Li, R., & Hu, Z. (2023). AlignScore: evaluating factual consistency with a unified alignment function. *Proceedings of ACL 2023 (Volume 1: Long Papers)*, 11328-11348. arXiv:2305.16739.
- Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks*. arXiv:2306.05685.
- Zhong, M., Liu, Y., Yin, D., Mao, Y., Jiao, Y., Liu, P., Zhu, C., Ji, H., & Han, J. (2022). Towards a unified multi-dimensional evaluator for text generation. *EMNLP 2022*.
