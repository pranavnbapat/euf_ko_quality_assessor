# Literature review: methods behind the KO quality assessment

This document explains where each method in `assess_ko_quality/` comes from, why it
was chosen over the alternatives, and how it is implemented. Code references point at
the implementation; every number the code uses is either cited here or learned from
the corpus, and `validation/metadata_spec.json` records the provenance of the ones that
are configured.

Status markers used below:

| marker | meaning |
|---|---|
| **built** | implemented and exercised by `tools/score_metadata.py` |
| **planned** | agreed design, not implemented yet |
| **dropped** | tried and removed; kept here so it is not re-introduced by accident |

---

## 1. What is assessed, and on what principle

A Knowledge Object (KO) is a file plus the metadata that describes it. The two are
treated differently:

| | who controls it | role in the assessment |
|---|---|---|
| **The file's content** (text, vision reading) | the original author | measured and reported; **never affects the score** |
| **The metadata** (title, subtitle, description, keywords) | the contributor's version and an LLM version written fresh from the summary each run (the export's `*_llm` with `--use-existing-llm`) | **the score**: three measured pillars and an LLM's 0-100 score (Section 9), a quarter each |
| **Domain relevance** | - | a **gate**: off-domain KOs are not scored at all |
| **Required fields** | the contributor | a **compliance checklist**, reported, not scored |

The metadata score asks one question: *does the metadata correctly tell a reader and a
search engine what this file is?* The answer is built from measurable, reproducible
criteria only. Human reviewer ratings are **not** used to fit or validate it; reviewer
judgement is itself subjective and error-prone, so validation is technical instead
(Section 7).

**What the metadata is checked against.** Since version 9, the metadata is checked
against the KO's summary (`ko_content_flat_summarised`), not the file's full text.
This is a choice, not a published rule. Since version 11 every language is assessed
by default, with results reported per file language; since version 12 the LLM
version is written fresh from that summary each run - blind, and in the document's
own language (Section 9) - so both versions are measured against the text the LLM
version was written from (`--use-existing-llm` restores the export's `*_llm`
fields, which came from the same summary plus the contributor's value). The sources below measure metadata against the
document itself (Ochoa & Duval's accuracy). SummaC and FactCC check a summary against
its source document. Using the file's summary as the reference departs from them, and
it favours text written from the summary (METHODS.md §12).

---

## 2. Quality frameworks: where the pillars come from

### ISO/IEC 25012 - Data quality model
*ISO/IEC 25012:2008, Software engineering - SQuaRE - Data quality model; measures in ISO/IEC 25024:2015.*

The international standard for data quality. It defines 15 characteristics split into
**inherent** (accuracy, completeness, consistency, credibility, currentness),
**inherent and system-dependent** (accessibility, compliance, confidentiality,
efficiency, precision, traceability, understandability) and **system-dependent**
(availability, portability, recoverability).

**Why:** it is the standard, it is domain-neutral, and it separates what belongs to the
data (the metadata) from what belongs to the system around it. That separation is the
same one this assessment draws between metadata and content.

**How it maps onto the sub-pillars:**

| ISO 25012 characteristic | sub-pillar |
|---|---|
| Accuracy (syntactic) | noise |
| Accuracy (semantic) | consistency |
| Completeness | completeness |
| Consistency | formatting; consistency (coherence between fields) |
| Precision | density |
| Understandability | clarity |
| Accessibility / availability | findability (functional pillar) |
| Compliance | compliance checklist (separate stage) |
| Currentness | reported only (document age) |
| Credibility, traceability | planned content measurements (provenance) |

### Wang & Strong (1996)
*Beyond accuracy: What data quality means to data consumers. Journal of Management Information Systems 12(4), 5-33.*

The foundational study that grouped data quality into intrinsic, contextual,
representational and accessibility dimensions, from the consumer's point of view.
**Why:** it establishes that quality is more than accuracy - a correct title that nobody
can find (accessibility) or understand (representational) is still poor metadata. This
is the argument for having a functional pillar at all.

### Bruce & Hillmann (2004)
*The continuum of metadata quality: defining, expressing, exploiting. In Hillmann & Westbrooks (eds.), Metadata in Practice, ALA Editions, 238-256.*

Seven criteria written specifically for metadata: completeness, accuracy, provenance,
conformance to expectations, logical consistency and coherence, timeliness,
accessibility. **Why:** the most-cited metadata-specific framework; ISO 25012 is about
data in general. Its "logical consistency and coherence" is the basis for checking
that title, description and keywords describe the same thing.

### Ochoa & Duval (2006, 2009)
*Quality metrics for learning object metadata. Proceedings of ED-MEDIA 2006, 1004-1011. Automatic evaluation of metadata quality in digital repositories. International Journal on Digital Libraries 10(2-3), 67-91.*

Turned the Bruce & Hillmann criteria into **computable** metrics for learning-object
repositories and showed they can run at repository scale without manual review.
**Why:** it is the direct precedent for what this assessment does - automatic,
per-record metadata quality in a repository of educational objects. Four of its
metrics map one-to-one onto sub-pillars here, which is why this assessment follows
their definitions rather than inventing new ones:

| Ochoa & Duval metric | their definition | sub-pillar here |
|---|---|---|
| Accuracy | "the semantic distance between the information that a user could extract from the metadata record and the information that the same user could obtain from the document itself"; for text documents, the distance between the title or description and the document's text | consistency (accuracy part) |
| Coherence | a coherent record describes the same topic in title, description and keywords; measured as the semantic distance between those free-text fields (they used LSA) | consistency (coherence part) |
| Text information content | TF-IDF over the field's words: term frequency x log(1 / document frequency); in the 2009 evaluation the authors single it out as agreeing especially well with human review | density |
| Readability (cognitive accessibility) | a readability index applied to the description | clarity |

Where this assessment departs from them, it is to replace a 2006-era technique with a
current one measuring the same thing: NLI instead of word-overlap cosine or LSA for
accuracy (NLI also detects contradiction), multilingual embeddings instead of LSA for
coherence, and LIX instead of Flesch because Flesch is calibrated for English only.

### Stvilia, Gasser, Twidale & Smith (2007)
*A framework for information quality assessment. JASIST 58(12), 1720-1733.*

A taxonomy of information-quality dimensions in three classes: intrinsic (measured
against the information itself or reference standards), relational/contextual
(measured against its context of use) and reputational. **Why:** it separates what can
be measured from the object alone (structural pillar) from what needs its context -
the file and the rest of the corpus (semantic and functional pillars). Its
informativeness/redundancy dimension is the basis for penalising a description that
only repeats the title.

### Park (2009)
*Metadata quality in digital repositories: a survey of the current state of the art. Cataloging & Classification Quarterly 47(3-4), 213-228.*

Survey finding that accuracy, completeness and consistency are the criteria most used
in practice. **Why:** confirms the three criteria this assessment weights most heavily
are the field's consensus, not a local choice.

### FAIR principles and F-UJI
*Wilkinson et al. (2016), The FAIR Guiding Principles for scientific data management and stewardship, Scientific Data 3, 160018. Devaraju & Huber (2021), An automated solution for measuring the progress toward FAIR research data, Patterns 2(11), 100370.*

FAIR asks that objects be Findable, Accessible, Interoperable and Reusable; F2 requires
rich metadata and F4 that it be indexed in a searchable resource. F-UJI shows FAIRness
can be assessed automatically. **Why:** findability is the platform's purpose, and
FAIR makes it a quality criterion of the metadata rather than a property of the search
engine. That is the justification for the functional pillar.

### The three pillars
Structural, semantic and functional follow the 2025 EU-FarmBook KO quality diagram.
The fourth pillar of that diagram, domain, became the gate (Section 6).

---

## 3. Structural pillar - is the field well-formed? (**built**)

Implemented in `core/metadata_metrics.py`: `length`, `completeness`, `noise`, `formatting`.

### 3.1 Length
Only limits that a published external source states are applied; where no source gives
a number, no number is imposed and the length sub-pillar is not evaluated.

| field | rule | source |
|---|---|---|
| title | at most 150 characters, not counting spaces | EIP-AGRI Common Format (European Commission, 2015): project title "short and easily understandable (one key sentence on the project; max 150 characters, word count – no spaces)" |
| description | 50 to 5,000 characters | Google Search Central, Dataset structured data: "The summary must be between 50 and 5000 characters long" |
| subtitle | none | no published source gives a length |
| keywords | none | no published source gives a count; their form (one concept per term, no duplicates; ANSI/NISO Z39.19) and diversity (KPEval) are still checked |

- **Scoring:** each limit gives 1 when met and falls off linearly outside it (to 0 at
  twice a maximum, or at zero below a minimum), the desirability form of Derringer &
  Suich (1980); a field scores the lowest of its limits.
- **Why these sources:** the EIP-AGRI Common Format is the European Commission's format
  for practice abstracts and project summaries in agricultural innovation, the closest
  published standard to an EU-FarmBook KO; Google's Dataset guidelines are the only
  published range for a resource description that was found.
- **What the sources do not support:** Google gives no length for titles or meta
  descriptions ("There's no limit on how long a meta description can be"; both are
  truncated to fit the device), and Dublin Core defines its elements without lengths.
  The often-quoted "about 160 characters" for a description is SEO practice, not a
  published rule. No source sets a minimum title length, a subtitle length or a keyword
  count.
- **Replaced:** the first version used title 5-12 words, subtitle 8-20 words,
  description 40-80 words and at most 600 characters, keywords 4-10. Those came from an
  LLM metadata-generation prompt (`which_model_to_choose/generation/prompts.py`) with no
  source, and were removed.
- **How the corpus compares** (10,468 KOs): 1.6% of titles exceed 150 characters (spaces
  not counted); 2.0% of descriptions are under 50 characters and 3 exceed 5,000.
- **Rich text:** the description field is stored as editor HTML (17% of descriptions
  contain markup). It is converted to text, with bullets kept as bullets, before any
  check, rather than counted as noise.

### 3.2 Completeness
- **Source:** ISO 25012 completeness; Bruce & Hillmann; Ochoa & Duval.
- **How:** a field that is required and empty scores 0. A present field scores the
  share of the checks it passes: *not a filename* (no spaces with `_`/`.` separators,
  or a trailing file extension), *not a copy of another field*, and *not shared
  verbatim with KOs from other projects* (Google's Dataset guidelines likewise ask for
  unique names). For the title two more: *not an internal project code* (deliverable,
  task, work-package and milestone numbers such as D3.1, T2.3, WP2, "Task 4") and *not
  the file name* (checked against `ko_object_name`). Neither is "short and easily
  understandable" (EIP-AGRI Common Format) or "descriptive" (Google Search Central, title
  links: "Write descriptive and concise text... Avoid vague descriptors").
- **Why the last check:** a title reused across unrelated projects is boilerplate, not
  a description of this file. Detected from the corpus, so no list of "placeholder
  words" is typed in.
- **Missing fields:** every field is expected, subtitle included. An empty field scores
  0 and counts in the KO score, as in Ochoa & Duval's (2006) completeness metric, which
  counts the fields that hold a value, so a missing field adds nothing to the score.

### 3.3 Noise
- **Source:** the rule-based text-quality filters used to clean web corpora for
  language-model training: C4 (Raffel et al. 2020) and MassiveWeb/Gopher (Rae et al.
  2021). These are the best-documented, reproducible heuristics for "this text contains
  debris". ISO 25012 syntactic accuracy.
- **How:** the share of tokens that are markup (`<br>`, `&amp;`), URLs or e-mails,
  encoding debris (mojibake such as `Ã©`, the replacement character), runs of symbols,
  or an immediate word repetition. Score = 1 - that share.
- **Why a share and not thresholds:** it needs no cut-off and degrades in proportion
  to how much of the field is debris.

### 3.4 Formatting
- **Sources:** ISO 25012 consistency; ANSI/NISO Z39.19-2005 (R2010), the standard for controlled vocabularies
  and index terms, for keyword form (one concept per term, concise noun phrases, no
  duplicates); Tinker (1963) for text set in capitals, which is read measurably more
  slowly than mixed case.
- **How:** field-specific checks, scored as the share passed. Title and subtitle: no
  line breaks, clean whitespace, no dangling comma, semicolon or colon at the end, not
  all capitals, not all
  lower case. Description: no line break inside a sentence (a PDF copy artefact), clean
  whitespace, starts with a capital, not all capitals (two words or more; a single word
  is usually an acronym or a name), ends with sentence punctuation.
  Keywords: stored as a list, one term per item (not `"a;b"`), no duplicates, no
  trivial terms, no term longer than the corpus's 99th-percentile keyword length, no
  trailing punctuation.
- **Why the keyword length cut-off is learned:** Z39.19 asks for concise terms but sets
  no word limit; the corpus's own 99th percentile flags sentences pasted in as keywords
  without typing a number in.
- **Changed after review:** "ends with sentence punctuation" was first counted under
  clarity as a sign of truncation. A missing final full stop is a formatting fault,
  not evidence that text was cut, so it moved here.

---

## 4. Semantic pillar - is the field correct and informative? (**built**)

### 4.1 Consistency (is the field right about the file, and do the fields agree?)
Two parts, averaged, following Ochoa & Duval's split of Bruce & Hillmann's criteria
into **accuracy** and **coherence**.

**Accuracy - does the file support the field?**
- **Sources:**
  - Ochoa & Duval (2006, 2009): accuracy is the semantic distance between what the
    metadata says and what the document says.
  - Natural language inference as a test of factual consistency: Falke et al. (2019)
    and SummaC (Laban et al. 2022); FactCC (Kryscinski et al. 2020), a consistency
    classifier trained on synthetic errors, frames the same problem. A description is,
    in effect, a summary of the file, so this is the summarisation-faithfulness problem.
  - Cross-lingual NLI: XNLI (Conneau et al. 2018) on XLM-RoBERTa (Conneau et al. 2020).
    Which model is used is decided by measurement (Section 7.3).
  - For titles, the entailment framing of zero-shot topic classification, "This text
    is about {title}." (Yin, Hay & Roth 2019).
  - For keywords, the distinction between "present" keyphrases, which occur in the
    text, and "absent" ones, which are correct without occurring (Meng et al. 2017).
- **How (text fields):** each claim - the title, the subtitle, each description
  sentence - is compared with the two passages of the reference text most similar to
  it (found with multilingual E5). Since version 9 the reference text is the KO's
  summary (Section 1); "the file" below means that reference. A claim scores max(P(entailment) - P(contradiction)) over
  those passages and the field scores the mean, mapped from [-1, 1] to [0, 1]. This is
  adapted from SummaC's zero-shot rule, which takes the maximum over premises and the
  mean over claims; its paper scores the entailment probability, and its released code
  subtracts the highest contradiction probability from the highest entailment
  probability. Here the difference is taken per passage, then maximised.
- **How (keywords):** the share of terms that occur in the file ("present"), or whose
  similarity to the file's closest passage beats the 95th percentile of randomly drawn
  corpus keywords ("absent" but supported). The threshold is learned per file from that
  null distribution, not typed in.
- **Cross-lingual keywords (versions 13-14):** that chance threshold is calibrated on
  same-language pairs, so keywords that do not read as the reference's language
  (wordfreq known-word share: fewer than half of at least two alphabetic tokens known
  at Zipf >= 2.0; open-set langdetect misreads short fields and is not used) are marked
  **not evaluable** rather than scored, and their weight redistributes. With
  `--resummarise` (version 15) the reference itself is in the document's language, so
  same-language keywords are measured directly; keywords written in English on a
  non-English document are measured against the English version of that summary
  instead of abstaining, with the evidence naming the basis. The NLI accuracy of
  title, subtitle and description stays in every case - XNLI is cross-lingual by
  construction.
- **Why NLI rather than similarity:** similarity says two texts are about the same
  thing; NLI says one supports the other and can detect contradiction. A description
  that names the wrong country or crop can be very similar to the file and still wrong.
- **Limits:** NLI models are weaker on long premises, hence the retrieval of the two
  best passages first. With the summary as the reference the whole summary is searched
  (about 200 words, rarely over 600); with the file, the first 40 passages (about 4,800
  words) of a long document are searched, not the whole document. XNLI covers 15 languages, of which only English,
  French, Spanish, German, Greek and Bulgarian are EU languages; for the rest the model
  depends on cross-lingual transfer or on machine-translated training data. Descriptions
  written as statements about the document ("This report explores ...") tend to come
  out neutral rather than entailed, because the file never says that about itself.

**Coherence - do title, description and keywords describe the same thing?**
- **Source:** Ochoa & Duval (2006): "a coherent metadata record describes the same
  topic in title, description and keywords", measured as the semantic distance between
  the free-text fields (they used LSA).
- **How:** multilingual E5 similarity between the field and each of the KO's other
  descriptive fields, expressed as the share of 300 randomly drawn KOs whose
  corresponding field is less similar. 1.0 means the fields belong together more
  clearly than any random pairing; 0.5 means no better than chance.
- **Why against chance:** a raw cosine has no natural scale (0.8 can be high for one
  model and ordinary for another); a rank against random pairings reads the same way
  for every field, language and model.
- **Added after review.** The first version checked each field against the file only,
  never the fields against each other, which left out a criterion both Bruce & Hillmann
  and Ochoa & Duval name.

### 4.2 Density (how specific are the words?)
- **Sources:** Ochoa & Duval's *text information content*: TF-IDF over a field's words,
  term frequency x log(1 / document frequency) - in their 2009 evaluation, singled out
  as agreeing especially well with human review of metadata quality. Inverse document frequency
  itself is Spärck Jones (1972). ISO 25012 precision.
- **How:** each word's specificity is the lower of two readings, then averaged over the
  field and mapped to 0-1 between the corpus's 5th and 95th percentiles for that field
  (the same anchors apply to original and LLM metadata):
  - *platform-generic*: normalised IDF across all KOs, so "agriculture" and "project"
    count as generic here even though they are not common in general language;
  - *language-generic*: 1 - the word's Zipf frequency / 8 in general usage of any
    language found in the corpus, from wordfreq (Speer 2022), whose Zipf scale is that
    of van Heuven et al. (2014). "The" is 7.7; a word seen once per million is 3.
- **Why a general-language list:** a per-language IDF would need enough KOs in every
  language to estimate. Only five languages have 200 or more KOs, so in the first
  version every Finnish, Czech, Greek or Slovene word looked rare and those titles
  scored 1.0 on density - an artefact that favoured non-English metadata over English
  by up to 0.9. wordfreq covers 41 languages including all but three EU languages
  (Estonian, Irish, Maltese).
- **Why no stopword list:** both readings give function words near-zero specificity in
  every language, which is how the method avoids a typed-in list.
- **Difference from Ochoa & Duval:** their measure accumulates over the field's words,
  so a longer field carries more information content. Here the mean is taken instead,
  because length is already scored on its own (3.1) and summing would count it twice.
  Term frequency within a title or keyword list is almost always 1, so it is dropped.

### 4.3 Clarity (can a reader understand it?)
- **Sources:**
  - Readability of the description as a metadata-quality metric: Ochoa & Duval (2006),
    who used the Flesch index. LIX (Björnsson 1968) is used here instead because it
    needs no syllable counting and was applied to newspapers in 11 languages (Björnsson 1983);
    Flesch-type formulas are calibrated for English. Björnsson's scale runs from about
    20 (very easy) to 60 (very difficult); Anderson (1983) relates it to reading grades.
  - Acronyms: Barnett & Doubleday (2020) show acronyms in over 24 million titles and 18
    million abstracts have grown steadily while few are ever reused - 79% appear ten
    times or fewer - so an unexplained acronym is a genuine barrier to a reader. The
    matcher is a simplified Schwartz & Hearst (2003), the standard algorithm for finding
    where an abbreviation is spelled out.
  - ISO 25012 understandability.
- **How:** the mean of up to three parts: *not visibly truncated* (no trailing ellipsis
  or dangling punctuation), *acronyms explained* (1 minus the share of tokens that are
  acronyms spelled out nowhere in the file or the other fields), and, for descriptions
  of 40 words or more, *readability* (full score at LIX 40 or below, zero at 60 or
  above - "very difficult" on Björnsson's scale - linear between).
- **Changed after review:**
  - the full-score cut-off moved from LIX 45 to 40, the boundary below which texts
    count as easy or ordinary on Björnsson's scale;
  - readability is no longer computed on descriptions under 40 words. Readability
    formulas are defined on samples of about 100 words and are unreliable on fragments;
    40 words is a choice, below which one or two sentences dominate the index;
  - a missing final full stop no longer counts as truncation (moved to formatting).
- **Limits:** LIX rises with long compound words, so German, Dutch, Finnish and Greek
  descriptions read as harder than equivalent English ones. The acronym matcher accepts
  initials in order and allows short function words to be skipped; it does not handle
  acronyms formed from the middle of words.

### 4.4 Usefulness (does it carry what the file is about?)
- **Sources:** coverage is recall of the document's salient content in its summary,
  the idea behind ROUGE (Lin 2004); the salient words are found by TF-IDF weighting
  (Salton & Buckley 1988) and matched across languages with multilingual E5 (Wang et
  al. 2024). Redundancy is Stvilia et al.'s (2007) informativeness/redundancy dimension,
  measured with Jaccard overlap (Jaccard 1912).
- **How:** *coverage* is the share of the file's ten most characteristic words that
  the field carries, literally or by meaning; "by meaning" must beat the 95th percentile
  similarity of random corpus keywords to the same field. The characteristic words are
  ranked by TF-IDF (Salton & Buckley 1988) with sublinear term frequency, (1 + ln tf) x
  ln(N/df) (Manning, Raghavan & Schütze 2008, "Sublinear tf scaling"), after excluding
  function words: any word with a Zipf frequency of 6 or
  more in one of the corpus languages (more frequent than 1 in 1,000 words), as
  keyphrase extraction always excludes stopwords. For subtitle and description, *adds to
  the title* is 1 minus their word overlap with the title. For keywords, *diversity*
  follows KPEval (Wu, Yin & Chang, Findings of ACL 2024), which takes it from Bahuleyan &
  El Asri (COLING 2020): the mean of 1 - dup_token_ratio (share of tokens repeated after
  Snowball stemming in the file's language) and the semantic emb_sim (mean pairwise
  keyword similarity) placed between the corpus's 5th and 95th percentiles.
- **Changed after review:** the first version let function words through: a Dutch
  file's "key terms" came out as *een, het, kan*, because IDF alone cannot demote the
  function words of a language only a few KOs are written in. Keywords had no diversity
  check; KPEval's was added.
- **Key terms are words:** tokens with digits or underscores (identifiers, codes such as
  `bp_pwgz9g80`) are excluded; they are rare, so TF-IDF would otherwise rank them first.
- **Limit - coverage across languages:** single-word matching across languages is
  weak. English keywords for a Latvian file ("meat production, livestock" for *gaļas,
  liellopu*) reached a coverage of only 0.1 although they are correct translations. When
  coverage is near zero for right and wrong keywords alike, the diversity part decides
  usefulness, so the self-test found cases where another project's keywords scored as
  useful. Wrong keywords are still caught by consistency, which dropped in every
  swapped-field test. Since version 13 coverage is marked **not evaluable** for a
  field that does not read as the reference's language, instead of letting a
  near-zero reading stand: key-term extraction needs document-frequency and
  word-frequency statistics in the field's language.
- **Coherence vs redundancy:** these are not in tension. Coherence (4.1) asks that the
  fields be about the same thing; redundancy asks that they not use the same words to
  say it. A description that restates the title is coherent and redundant.
- **Why TF-IDF rather than an LLM keyword extractor:** it is deterministic,
  language-neutral, costs nothing, and gives the same answer every time.

---

## 5. Functional pillar - can a search find the KO by this field? (**built**)

### 5.1 Known-item retrievability
- **Sources:** retrievability (Azzopardi & Vinay 2008), which measures how easily a
  document can be retrieved by a search system; simulated known-item queries built
  from a document, tested in six European languages (Azzopardi, de Rijke & Balog 2007).
- **How:** the field is used as a query against an index of every KO's file text (not
  their metadata, so a title cannot match itself). The rank of the KO the field belongs
  to is recorded, and converted to a gain of 1 / log2(1 + rank), the discount of
  normalised discounted cumulative gain in its standard form (Burges et al., ICML 2005;
  used by BEIR): rank 1 scores 1.0, rank 3 scores 0.5, rank 10 scores 0.29. Ties share
  the middle of their block. Järvelin & Kekäläinen's (2002) original DCG discount
  differs: no discount before rank 2, then 1 / log2(rank).
- **Why this is a metadata-quality measure:** a title like "Final report" is not wrong,
  but it does not identify its file among 10,000; this measures exactly that. Ochoa &
  Duval (2006) treat findability as the "physical/logical accessibility" of metadata
  and note that retrieval frequency only measures accessibility "with the current
  searching tool"; they ask for a measure of *potential* accessibility, and propose
  linkage (how many other records reference it). Retrievability under a fixed, standard
  retrieval model is another measure of potential accessibility, closer to how a
  search-driven platform is used.

- **Top-venue support:** KPEval's *utility* aspect (Findings of ACL 2024) evaluates
  keyphrases by retrieval effectiveness, with BM25 and a dense retriever averaged -
  the same two retrievers used here; BEIR (Thakur et al., NeurIPS 2021) established
  BM25 as a robust zero-shot baseline across domains and nDCG, whose discount is used
  here, as the standard retrieval measure.

### 5.2 BM25 findability
- **Source:** Okapi BM25 (Robertson & Zaragoza 2009), with k1 = 1.2 and b = 0.75, the
  defaults of OpenSearch, the platform's search engine (OpenSearch documentation). They
  sit at the edge of the ranges Robertson & Zaragoza report as reasonably good in many
  circumstances (1.2 < k1 < 2, 0.5 < b < 0.8).
- **Why:** it measures findability for someone typing the words, in the file's language.
- **Cross-lingual limit (version 13):** BM25 is lexical and the index holds the English
  reference summaries, so a field that does not read as the index's language is marked
  **not evaluable** rather than given a meaningless rank - in a subset where only one
  KO carries a language, any query in that language would rank it first without
  competition. The functional pillar is then carried by semantic findability alone,
  which is multilingual.

### 5.3 Semantic findability
- **Source:** dense retrieval (Karpukhin et al., EMNLP 2020) with multilingual E5 (Wang
  et al. 2024). On the Massive Multilingual Text Embedding Benchmark (MMTEB, Enevoldsen
  et al., ICLR 2025) the best publicly available model is `multilingual-e5-large-instruct`,
  from the same family. The small variant is used here for CPU cost; a larger one is a
  configuration change.
- **Why:** it measures findability by meaning, and across languages: an English title
  for a Spanish file can still find its file. The two together separate "wrong words"
  from "wrong language".
- **Limit:** each KO is represented by the first 2,000 characters of its file for this
  measure, and the index is built from the export rather than the live OpenSearch
  cluster, so it approximates the platform's search rather than reproducing it.

---

## 6. The domain gate (**built**)

- **Classifier:** logistic regression on sentence embeddings (all-mpnet-base-v2,
  Reimers & Gurevych 2019), calibrated on a labelled gate-validation set.
- **Hybrid with a reject option:** KOs the classifier is confident about are decided by
  it; only the uncertain band [0.20, 0.70) is sent to an LLM. This is the classic reject
  option (Chow 1970), studied for neural classifiers as selective classification
  (Geifman & El-Yaniv, NeurIPS 2017): defer only the cases where the cheap decision is
  unreliable. It kept LLM calls to about 10% of KOs.
- **LLM as judge:** used only in the uncertain band, because LLM judges have
  documented biases (position, verbosity, self-preference; Zheng et al. 2023). The
  scope given to the LLM is the platform's own topic vocabulary
  (`data_model.topics.json`), not a list written for the prompt.
- **Dropped:** cosine similarity to an AGROVOC/NALT centroid. It scored gibberish
  (0.069) about the same as a crop-rotation guide (0.071).

---

## 7. Combining sub-pillars, and validating without human labels

### 7.1 Normalisation and weighting
- **Source:** OECD/JRC, *Handbook on Constructing Composite Indicators* (2008), the
  standard reference for building a composite index: normalise, weight, aggregate,
  test robustness.
- **Normalisation:** desirability functions (Derringer & Suich 1980) map each metric to
  0-1 with a declared direction: higher-is-better, lower-is-better, or a target range.
- **Weights:** equal, at every level (sub-pillars within a pillar, pillars within a
  field, fields within a KO). Without an external criterion no weighting can be shown
  to be correct. Equal weights are what most composite indicators use (the handbook:
  "Most composite indicators rely on equal weighting") and are hard to beat in practice
  (Dawes 1979). The handbook also warns that equal weights can disguise the lack of an
  empirical basis, which is why stage 6 tests alternatives. Weights are configurable in
  `validation/metadata_spec.json`.
- **Robustness (built, stage 6):** the handbook's robustness step - recompute with
  data-driven weights (entropy; CRITIC, Diakoulaki, Mavrotas & Papayannakis 1995; first
  principal component) and report how far KO rankings move (Saisana, Saltelli & Tarantola
  2005). Also the handbook's redundancy check (sub-pillars correlated at |rho| >= 0.9)
  and a length-bias check on every sub-pillar.
- **Reporting:** pillar and field scores are always reported next to the composite,
  following HELM's case (Liang et al., TMLR 2023) that one aggregate hides trade-offs
  between dimensions.

### 7.2 Validation by perturbation
- **Sources:** behavioural testing of NLP systems with CheckList (Ribeiro et al. 2020);
  perturbation checklists for evaluation metrics (Sai et al. 2021); metamorphic testing
  (Chen, Cheung & Yiu 1998).
- **How:** for every KO, each sub-pillar is given one deliberate defect aimed at the
  property it claims to measure - title padded past its 150-character limit (length), replaced by a
  filename (completeness), markup appended (noise), set in capitals
  (formatting), another project's description or title swapped in (consistency),
  rewritten in the language's commonest words (density), description cut off
  (clarity), another project's keywords and keywords padded with case variants of
  themselves (usefulness), another project's title (findability). A sub-pillar passes if its value drops.
- **Why:** it is the only way to show, without human labels, that a metric measures
  what its name says. A metric that does not move when its property is broken is not
  measuring that property.
- **Results:** see Section 10.

### 7.3 Choosing the NLI model by measurement
Two cross-lingual NLI models were compared on the
same task the accuracy check performs: for each KO, score its own title and
description against its file, and a title and description swapped in from another
project; the better model ranks the true field above the swapped one more often. The
comparison is split by whether the KO's language is one of the 15 XNLI training
languages, because that is where multilingual NLI models are known to differ
(`tools/compare_nli.py`, results in `validation/runs/nli_model_comparison.json`).

| model | all (n=80) | XNLI languages (n=40) | other EU languages (n=40) | pairs/s (CPU) |
|---|---|---|---|---|
| `joeddav/xlm-roberta-large-xnli` (XLM-R, ACL 2020, on XNLI, EMNLP 2018) | **0.938** | **1.000** | **0.875** | 0.97 |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (DeBERTaV3, ICLR 2023) | 0.775 | 0.750 | 0.800 | 1.5 |

The share of pairs in which the true field scores higher than the swapped one. XLM-R
large is used: better in both language groups, despite mDeBERTa's training on machine-
translated NLI data in more languages.

### 7.4 Evidence behind every number

What backs each setting, and how strongly. "Published method" means the setting is the
one the cited work defines; "standard" a formal standard; "published guideline" a
rule stated by a published guideline; "learned" computed from the corpus by a stated rule;
"choice" a value with no published basis, set for a stated reason and open to change.

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
| all semantic and functional parts | reference text | the KO's summary (`ko_content_flat_summarised`); every language since version 11. With `--resummarise` (version 15), a fresh summary of the downloaded file **in the document's own language**, which replaces it as the reference; its English version is the KO's row in the corpus index and the secondary reference for English-language fields on a non-English document | choice | the LLM version is written from the same summary, so both versions are measured against the text it was written from; it favours text written from the summary (METHODS.md §12). Up to version 8: the vision reading followed by the extracted text |
| cross-lingual evaluability | known-word share | a field reads as a language when at least half of its (at least 2) alphabetic tokens are known words of it, wordfreq Zipf >= 2.0; below that, the lexical parts - bm25_findability, usefulness.coverage, keywords' accuracy - are not evaluable and their weight redistributes; with `--resummarise` (version 15) the reference is in the document's language, so bm25_findability gates on the corpus index's language (English) instead of the reference's | choice | wordfreq (Speer 2022); open-set langdetect rejected by measurement (it misreads short fields); exclusion rather than a zero keeps an unmeasurable property out of the score |
| generation (version 12) | brief | blind - the summary and the published rules only, never the contributor's fields; written in the document's language; at most 10 keywords | choice + convention | blind generation keeps the text under assessment out of the writer's context; 10 follows the usual keyphrase cut-offs (Meng et al. 2017); the count is never scored |
| resummarise (version 15) | file handling | the file downloaded from its URL and read as it is: PDF and image pages as images (one per image, no page cap), office files converted to PDF first, a video's sound transcribed by Whisper and its frames shown, an audio file as its transcript | choice + published model | the export's stored text can be a translation of the file, so the file itself is read; 12 images per prompt is the provider's measured limit; Whisper (Radford et al., ICML 2023) transcribes zero-shot across languages |
| resummarise (version 15) | long documents | notes per chunk of 12 pages, merged 8 at a time until one summary remains | published method | hierarchical merging, the more coherent of the two workflows BooookScore compares (Chang et al., ICLR 2024) |
| resummarise (version 15) | verification | every draft read against its source for contradicted or unsupported statements and missing content; a source longer than one chunk checked sentence by sentence, each chunk giving every sentence a verdict (supported, contradicted, absent) and a sentence kept when any chunk supports it; one revision call applies the findings; none when there are none | published methods, adapted | Chain-of-Verification (Dhuliawala et al., Findings of ACL 2024): draft, verify independently, revise; across chunks, each sentence's best support counts, as in AlignScore (Zha et al., ACL 2023) and SummaC-ZS (Laban et al., TACL 2022) |
| resummarise (version 15) | summary language | the document's own, named by the model from the pages; an English translation for the corpus row | design decision (30 October 2026) | a non-English KO's fields are then measured in their own language; the corpus index stays English, so BM25 findability gates on the index's language |
| LLM pillar | model | glm-5.2, which wrote neither version (the generator is qwen3.5 by default; the export's `*_llm` fields came from another Qwen model) | choice | self-preference (Panickssery et al., NeurIPS 2024) cannot favour a version it did not write, so the judge stays out of the writer's model family (the pipeline warns on a match); of the judges measured, glm-5.2 was the more order-consistent (60-63% vs 55%) |
| LLM pillar | score | one 0-100 score per field per version, one call each, from the summary and that field; a quarter of the field score | published approach + choice | rating named criteria with an LLM is the approach of G-Eval (Liu et al., EMNLP 2023); scoring each version alone removes the position bias of pairwise comparison (Wang et al., ACL 2024); the one-call-per-field form and the equal weight are choices |
| LLM pillar | language line (version 12) | the prompt names the KO's language, says the summary is English, and asks for meaning to be judged across languages without penalising the language itself | published finding | LLM evaluators are less reliable on non-English text (Hada et al., Findings of EACL 2024) |
| aggregation | weights | equal at every level | common practice | OECD/JRC (2008): most composite indicators use equal weights; Dawes (1979) |
| aggregation | empty field, subtitle included | scores 0 | published method | Ochoa & Duval (2006) completeness: counts the fields that hold a value |

**What the review changed** (all reflected in the code and `validation/metadata_spec.json`):

| change | why |
|---|---|
| Consistency now includes **coherence** between title, description and keywords | Named by Bruce & Hillmann and operationalised by Ochoa & Duval; the first version left it out |
| Density uses **wordfreq general-language frequencies** instead of per-language document frequencies | Per-language tables need enough KOs per language; only five languages have them, and the rest scored as maximally specific |
| LIX full-score cut-off **45 → 40** | Björnsson's scale; 40 is the boundary of easy and ordinary prose |
| Readability only on descriptions of **40+ words** | Readability formulas are unreliable on short samples |
| **Length rules re-sourced** from external sources only (title <= 150 characters; description 50-5,000; no subtitle length or keyword count) | The previous ranges came from an LLM prompt; the new ones come from the EIP-AGRI Common Format and Google's Dataset guidelines; where no source gives a number, none is imposed |
| **Title checks** added to completeness: not a project code, not the file name | Neither is "easily understandable" (EIP-AGRI) or "descriptive" (Google) |
| **Rich-text descriptions** converted before checking | 17% of descriptions carry the editor's HTML; it was being counted as noise |
| "Ends with a full stop" moved from **clarity to formatting** | A missing full stop is not evidence of truncation |
| Keyword **diversity** added to usefulness | KPEval's diversity aspect (Findings of ACL 2024) |
| **Function words excluded** from key terms | A Dutch file's key terms were "een, het, kan" |
| **NLI model chosen by measurement** | XLM-R large 0.938 vs mDeBERTa 0.775 on true-vs-swapped fields |

**Not changed, but weaker than the rest:** the few "choice" rows above. They are
engineering bounds, not findings; the self-test shows the sub-pillars respond to
defects with them, not that the values are optimal.

---

## 8. Measurements of the content, reported and not scored (**built**, `core/content_report.py`)

The file's own quality is the original author's, not the contributor's, and it does
not change the metadata score. It is reported with the published filters used to clean
web text for language models, at their published thresholds - so each flag reads "this
text would be removed by filter X":

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

**Considered, not built:** perplexity under per-language language models (CCNet, Wenzek
et al. 2020) needs a KenLM model per language, several GB each; lexical density (Ure
1971) and entity-grid coherence (Barzilay & Lapata 2008) need a tagger or parser per
language.

## 9. The LLM pillar (**built**, `stages/llm_score.py`)

An LLM scores each field of each version, 0-100, from the KO's summary, that one field
and the criteria of the three measured pillars; one call per field, because multi-field
answers were where the output broke. The score is the fourth pillar of the field score,
with the weight of each measured pillar. Rating named criteria with an LLM is the
approach of G-Eval (Liu et al., EMNLP 2023). Each version is scored on its own, not side
by side, which removes the position bias Wang et al. (ACL 2024) found in pairwise
comparison. The model, glm-5.2, wrote neither version - the `*_llm` fields came from a
Qwen model - so self-preference (Panickssery et al., NeurIPS 2024) cannot favour one. The recommendation between the two
versions keeps a faithfulness condition on the measured accuracy, because optimising
against a proxy degrades the true objective (Gao, Schulman & Hilton, ICML 2023).

**Known and accepted for now.** The G-Eval authors report that LLM evaluators may favour
LLM-generated text, which here is the `*_llm` version. Side-by-side comparisons made
earlier in this project (version 7: scored against the file, 10 English KOs, 40 fields,
`output/llm_compare_en10/`) found the same:

| | glm-5.2 as judge | gpt-oss-120b as judge |
|---|---|---|
| same verdict when the order is reversed | 60% | 55% |
| picked whichever version was shown first (chance 33%) | 50% | 56% |
| verdict agrees with the measured best | 54% | 46% |
| share of wins given to its own model's text | **71%** | **77%** |
| times it picked the contributor's original | 0 | 0 |

Both judges showed the position bias and the self-preference reported in the literature,
never preferred the contributor's version even where it measured best, and their 1-5
ratings barely followed the measured properties (Spearman 0.06-0.34); qwen3.5-397b as a
judge repeated its verdict with the order reversed on only 45% of fields
(`output/llm_compare_glm_qwen_en10/`). This is why the LLM is one pillar of four rather
than the whole score.

Versions 7 to 9 also had a stage 5 that wrote new LLM versions (glm-5.2, compared with
gpt-oss-120b and qwen3.5-397b in the runs above) and judged them side by side. Version 10
removed it and compared the contributor's version with the `*_llm` fields already in the
export; its code is kept in `legacy/metadata_llm.py`. Version 12 brought back only the
blind generation: the LLM version is written fresh from the summary each run - one call
per KO, the summary and the published rules only, never the contributor's fields, in the
document's own language - and scored by the ordinary per-field machinery, never judged
side by side. The writer (qwen3.5 by default) is kept out of the judge's model family
for the self-preference reason above; the pipeline warns on a family match. Since
version 12 the judge's prompt also names the KO's language and asks for meaning to be
rated across languages without penalising the language itself, because LLM evaluators
are less reliable on non-English text (Hada et al., Findings of EACL 2024).

## 10. Results so far

**These results predate the current rules.** They come from version 7:
- measured against the file;
- the existing `*_llm` fields compared;
- an empty subtitle left out.

Under version 8's rule that an empty field scores 0, the same run's overall means become
about 63.3 (contributor) and 77.5 (existing `*_llm`). No full run under the current
rules (version 14: the summary as the reference, every language, the LLM version
generated fresh in the document's language, the LLM pillar, cross-lingual gating) has
been written up here yet.

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
score of 81.8 and faithfulness 0.883, with no LLM call; adding a new glm-5.2 version gives
84.4 and 0.892 at one call per KO (Section 9).

---

## 11. Tried and dropped

| what | why it was dropped |
|---|---|
| Score fitted to 2024 reviewer ratings (ridge regression) | Reviewer judgement is itself biased; the assessment must rest on measurable criteria |
| Type-token ratio for lexical richness | Falls mechanically with length (ρ -0.93 with length); replaced by MTLD (McCarthy & Jarvis 2010) |
| Banding continuous measures into 0-5 bins | Lost half the information in content length |
| Domain as a scored pillar (centroid similarity) | Could not separate gibberish from agriculture; replaced by the gate |
| Hand-typed term lists (generic words, agricultural terms) | Not reproducible across languages; replaced by corpus IDF and the platform vocabularies |
| Per-language document frequencies for density | Only five languages have 200+ KOs; the rest scored as maximally specific. Replaced by wordfreq general-language frequencies |
| `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` for NLI | 0.775 vs 0.938 for XLM-R large on the true-vs-swapped test |
| Open-set langdetect for the cross-lingual gate | Reads short fields wrongly ("personal development, dairy farming" came out French); replaced by the wordfreq known-word share |
| `--resummarise` from the export's extracted text | The stored text can be a translation of the file (a Lithuanian factsheet's stored text was English); the file itself is downloaded and read as page images, export text kept only as the per-KO fallback |
| `--resummarise` in one call, 50-page cap, pages stacked into 12 tiles (versions 12-14) | Pages lost resolution in the tiles, longer documents were cut, a 6,700-word PDF was rejected by the endpoint, and videos were skipped; replaced by per-chunk notes merged hierarchically, verified and revised, with videos transcribed (version 15) |
| `--resummarise` verification listing errors per chunk (30 October 2026) | A chunk cannot tell a wrong statement from one about other pages: on a 33-page document 12 of 13 listed errors were statements found elsewhere, and the revision removed two true figures. Replaced by verdicts per sentence combined across chunks |
| English reference with a document-language secondary summary (version 14) | Non-English fields were still measured against English; replaced by the document-language summary as the reference, with an English version for the corpus row (version 15) |

---

## 12. Where the methods sit in the core NLP/ML venues

Papers were sought in EMNLP, ACL (including Findings and TACL, the ACL journal), EACL,
COLM, COLING, TMLR, ICML, ICLR, NeurIPS, AAAI and IJCAI. The table maps each part of the
pipeline to what those venues offer, what was taken, and what was considered but not
used.

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
| LLM judges and their biases | MT-Bench (Zheng et al., NeurIPS 2023); Wang et al. (ACL 2024); G-Eval (Liu et al., EMNLP 2023) | the LLM is one pillar of four, and the recommendation's faithfulness condition uses the measured accuracy | **yes** - the LLM pillar |
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

## References

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
- Chang, Y., Lo, K., Goyal, T., & Iyyer, M. (2024). BooookScore: a systematic exploration of book-length summarization in the era of LLMs. *ICLR 2024*. arXiv:2310.00785.
- Chen, T. Y., Cheung, S. C., & Yiu, S. M. (1998). *Metamorphic testing: a new approach for generating next test cases*. Technical Report HKUST-CS98-01, Hong Kong University of Science and Technology.
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
