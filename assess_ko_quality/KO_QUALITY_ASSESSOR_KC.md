# KO Quality Assessor KC

`ko_quality_assessor_kc.py` performs four-pillar Knowledge Object (KO) quality assessment and writes a TSV report.

## Overview

This script evaluates each KO across four quality pillars:

1. **Structural** (30%): Physical completeness, formatting, noise levels
2. **Semantic** (35%): Clarity, usefulness, information density, consistency
3. **Functional** (25%): Searchability (BM25, embeddings, hybrid search, RAG readiness)
4. **Domain** (10%): Agricultural terminology relevance and context

Each pillar returns both a pillar score and supporting diagnostics. The script then produces:

- per-pillar 0-25 scores
- an unweighted total on 0-100
- a weighted total on 0-100
- a `Notes` field with common problems and runtime warnings

## Features

- **Four-pillar assessment**: Comprehensive quality evaluation across dimensions
- **Weighted scoring**: Configurable pillar weights (must sum to 100)
- **MNLI semantic consistency**: Cross-checks content alignment with metadata
- **Language detection**: Automatic language identification for metadata
- **Domain embedding**: Agriculture-specific domain relevance scoring
- **Error resilience**: Continues processing if individual KOs fail
- **Batch processing**: Reads one input file, scores each KO, and writes one TSV report

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    KO Quality Assessor                      │
├─────────────────────────────────────────────────────────────┤
│  Input: JSON/JSONL/NDJSON(.gz) file with KO objects         │
├─────────────────────────────────────────────────────────────┤
│  Processing Pipeline:                                        │
│  ┌─────────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐  │
│  │ Structural  │ │ Semantic  │ │ Functional│ │  Domain  │  │
│  │   (30%)     │ │   (35%)   │ │   (25%)   │ │  (10%)   │  │
│  └──────┬──────┘ └─────┬─────┘ └─────┬─────┘ └────┬─────┘  │
│         └──────────────┴─────────────┴────────────┘         │
│                           │                                 │
│                    ┌──────┴──────┐                         │
│                    │   MNLI      │ (Semantic consistency)   │
│                    │ Consistency │                         │
│                    └──────┬──────┘                         │
│                           │                                 │
│              ┌────────────┴────────────┐                   │
│              │  Weighted Aggregation   │                   │
│              │    (0-100 scale)        │                   │
│              └────────────┬────────────┘                   │
├───────────────────────────┼─────────────────────────────────┤
│  Output: TSV with detailed scores and diagnostics           │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Core dependencies
pip install pandas numpy

# Domain scoring requires sentence-transformers
pip install sentence-transformers torch

# MNLI consistency requires transformers
pip install transformers

# spaCy for NLP components (via quality modules)
pip install spacy
python -m spacy download en_core_web_lg
```

## Usage

### Basic Usage (Auto-detect latest input)

```bash
python ko_quality_assessor_kc.py
```

Scans `./input` for the latest supported input file and writes a TSV report to `./output`.

### With Explicit Input File

```bash
python ko_quality_assessor_kc.py \
    --input path/to/kos.json \
    --output-dir ./reports
```

### With Custom Directories

```bash
python ko_quality_assessor_kc.py \
    --input-dir ./data/kos \
    --output-dir ./quality_reports
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | ❌ No | Latest from `--input-dir` | Explicit path to JSON/JSONL/NDJSON(.gz) file |
| `--input-dir` | ❌ No | `./input` or `$KO_INPUT_DIR` | Directory to scan for latest input |
| `--output-dir` | ❌ No | `./output` or `$KO_OUTPUT_DIR` | Directory for output TSV |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KO_INPUT_DIR` | `./input` | Input directory (overridden by `--input-dir`) |
| `KO_OUTPUT_DIR` | `./output` | Output directory (overridden by `--output-dir`) |
| `AGRI_DOMAIN_CENTROID` | `./anchors/centroids/agri_anchor_centroid.npy` | Path to domain centroid for agriculture relevance |
| `W_STRUCT` | `30` | Structural pillar weight (0-100) |
| `W_SEM` | `35` | Semantic pillar weight (0-100) |
| `W_FUNC` | `25` | Functional pillar weight (0-100) |
| `W_DOM` | `10` | Domain pillar weight (0-100) |

**Note**: Weights must sum to exactly 100.

### Custom Weight Examples

```bash
# Emphasize domain relevance
export W_STRUCT=25
export W_SEM=30
export W_FUNC=20
export W_DOM=25
python ko_quality_assessor_kc.py

# Focus on search functionality
export W_STRUCT=20
export W_SEM=25
export W_FUNC=45
export W_DOM=10
python ko_quality_assessor_kc.py
```

## Input Format

Accepts JSON array, JSONL (newline-delimited JSON), or NDJSON (optionally gzip-compressed):

```json
[
  {
    "_orig_id": "ko_123",
    "@id": "https://example.com/ko/123",
    "title": "Crop Rotation Best Practices",
    "subtitle": "Sustainable farming techniques",
    "description": "This guide covers...",
    "keywords": ["agriculture", "rotation", "sustainability"],
    "ko_content_flat": "Full article content here..."
  }
]
```

### Required Fields

| Field | Required | Description |
|-------|----------|-------------|
| `_orig_id` or `@id` | ✅ Yes | Unique KO identifier |
| `title` | ⚠️ Recommended | KO title (affects all scores) |
| `description` | ⚠️ Recommended | Short description |
| `ko_content_flat` | ⚠️ Recommended | Main content body |
| `keywords` | ❌ No | List of keywords |
| `subtitle` | ❌ No | Subtitle or alternative title |

## Output Format

Produces a TSV file with quality metrics and diagnostics.

### Key Output Columns

| Column | Description | Range |
|--------|-------------|-------|
| `_orig_id` | KO identifier | string |
| `title` | KO title (truncated to 300 chars) | string |
| `lang_meta_detected` | Detected language (e.g., "en", "nl") | string |
| `Structural_Score_0_25` | Structural quality | 0-25 |
| `Semantic_Score_0_25` | Semantic quality | 0-25 |
| `Functional_Score_0_25` | Functional quality | 0-25 |
| `Domain_Score_0_25` | Domain relevance | 0-25 |
| `Total_Quality_unweighted_0_100` | Sum of 4 pillars | 0-100 |
| `Total_Quality_weighted_0_100` | Weighted aggregate | 0-100 |
| `Weights_used` | Weight configuration | e.g., "S30/Se35/F25/D10" |
| `Notes` | Diagnostic notes and warnings | string |

### Structural Diagnostics

| Column | Description |
|--------|-------------|
| `Structural_title_tokens` | Token count in title |
| `Structural_desc_tokens` | Token count in description |
| `Structural_content_tokens` | Token count in content |
| `Structural_keyword_count` | Number of keywords |
| `Structural_has_subtitle` | Boolean subtitle presence |
| `Structural_title_noise_ratio` | Special character ratio in title |
| `Structural_completeness_check` | Missing fields noted |
| `Structural_flags` | Semicolon-separated structural issues |

### Semantic Diagnostics

| Column | Description |
|--------|-------------|
| `Semantic_avg_sentence_length` | Mean sentence length |
| `Semantic_type_token_ratio` | Lexical diversity |
| `Semantic_stopword_ratio` | Stopword frequency |
| `Semantic_info_density_tokens` | Information-carrying tokens |
| `Semantic_title_desc_similarity` | Similarity proxy between title and description |
| `Semantic_content_coherence` | Internal content consistency |
| `Semantic_mnli_consistency` | MNLI-style metadata/content consistency score |
| `Semantic_mnli_diagnostics` | MNLI consistency details |
| `Semantic_flags` | Semicolon-separated semantic issues |

### Functional Diagnostics

| Column | Description |
|--------|-------------|
| `Functional_bm25_readiness` | BM25 search suitability |
| `Functional_embedding_density` | Vector embedding quality |
| `Functional_unique_term_ratio` | Term diversity for search |
| `Functional_content_searchability` | Overall search score |
| `Functional_flags` | Semicolon-separated functional issues |

### Domain Diagnostics

| Column | Description |
|--------|-------------|
| `Domain_cosine_to_centroid` | Similarity to agri centroid |
| `Domain_content_tokens_used` | Tokens analyzed (truncation) |
| `Domain_content_truncated` | Whether content was cut |
| `Domain_anchor_hits` | Matching domain anchors |
| `Domain_flags` | Semicolon-separated domain issues |

## Quality Pillars Explained

### 1. Structural Quality (0-25)

Evaluates physical completeness and formatting:

| Factor | Measurement |
|--------|-------------|
| **Title presence** | Exists and non-empty |
| **Description presence** | Exists and non-empty |
| **Content length** | Adequate token count |
| **Keywords** | Sufficient keywords present |
| **Noise level** | Low special character ratio |
| **Formatting** | Proper structure |

**Flags**: `missing_title`, `missing_description`, `missing_content`, `very_short_content`, `high_noise`, `few_keywords`

### 2. Semantic Quality (0-25)

Evaluates clarity, usefulness, and information density:

| Factor | Measurement |
|--------|-------------|
| **Sentence length** | Appropriate complexity |
| **Lexical diversity** | Type-Token Ratio (TTR) |
| **Stopword ratio** | Appropriate density |
| **Information density** | Content-to-fluff ratio |
| **Cross-field consistency** | Title/desc/content alignment |
| **MNLI entailment** | Semantic consistency check |

**Flags**: `very_short_sentences`, `low_diversity`, `high_stopword_ratio`, `low_info_density`, `inconsistent_metadata`

### 3. Functional Quality (0-25)

Evaluates search and retrieval suitability:

| Factor | Measurement |
|--------|-------------|
| **BM25 readiness** | Term frequency distribution |
| **Embedding density** | Vectorizable content ratio |
| **Unique terms** | Vocabulary diversity |
| **Search coverage** | Hybrid search compatibility |

**Flags**: `poor_bm25_distribution`, `low_embedding_yield`, `high_term_repetition`

### 4. Domain Quality (0-25)

Evaluates agricultural domain relevance:

| Factor | Measurement |
|--------|-------------|
| **Centroid similarity** | Distance to agriculture embedding centroid |
| **Anchor matches** | Hits on domain-specific terminology |
| **Content relevance** | Agricultural context strength |

**Flags**: `low_domain_relevance`, `truncated_content`, `no_anchor_matches`

## MNLI Semantic Consistency

Uses Natural Language Inference (NLI) as a consistency proxy between content and metadata:

```
Premise: content text
Hypothesis: title / subtitle / description
Check: does the content support the metadata?
```

This is a useful semantic alignment signal, but it is still a model-based proxy rather than human judgment.

## Scoring Examples

### High-Quality KO (Score: 85-100)

```
Structural_Score_0_25: 23 (Good structure, all fields present)
Semantic_Score_0_25: 22 (Clear, diverse, consistent)
Functional_Score_0_25: 21 (Searchable, good term distribution)
Domain_Score_0_25: 19 (Strong agricultural relevance)

Total_Quality_unweighted_0_100: 85
Total_Quality_weighted_0_100: 84.2
Weights_used: S30/Se35/F25/D10
Notes: -
```

### Low-Quality KO (Score: 0-40)

```
Structural_Score_0_25: 8 (Missing description, very short content)
Semantic_Score_0_25: 12 (Low diversity, inconsistent metadata)
Functional_Score_0_25: 10 (Poor searchability)
Domain_Score_0_25: 5 (Low agricultural relevance)

Total_Quality_unweighted_0_100: 35
Total_Quality_weighted_0_100: 32.4
Weights_used: S30/Se35/F25/D10
Notes: Missing description; Few keywords (<2); Detected non-EN metadata language: nl
```

## Typical Workflow

### 1. Prepare Input
```bash
mkdir -p input
cp kos.json input/
```

### 2. Run Assessment
```bash
python ko_quality_assessor_kc.py
```

### 3. Review Output
```bash
# View summary
head -20 output/*.tsv

# Open in spreadsheet
libreoffice output/*.tsv

# Filter poor quality KOs
awk -F'\t' '$NF < 50 {print $1, $2}' output/*.tsv
```

### 4. Analyze by Pillar
```bash
# Find KOs with poor semantic quality
awk -F'\t' '$5 < 10 {print $1, $2}' output/*.tsv

# Find KOs with domain issues
awk -F'\t' '$7 < 5 {print $1, $2}' output/*.tsv
```

## Troubleshooting

### "No valid JSON objects found"
**Cause**: Input file is empty or malformed.

**Solution**: Check JSON validity:
```bash
python -c "import json; json.load(open('input/kos.json'))"
```

### "Weights must sum to 100"
**Cause**: Environment variable weights don't total 100.

**Solution**: Check and adjust weights:
```bash
echo $W_STRUCT $W_SEM $W_FUNC $W_DOM
# Should sum to 100
```

### "Domain centroid not found"
**Cause**: Agriculture centroid file missing.

**Solution**: Set correct path:
```bash
export AGRI_DOMAIN_CENTROID=/path/to/centroid.npy
python ko_quality_assessor_kc.py
```

### Many "ERROR" entries in Notes
**Cause**: Individual KO processing failures.

**Solution**: Check specific errors in Notes column. Common causes:
- Malformed KO JSON
- Missing required fields
- Memory issues with very large content

## Performance

| Corpus Size | Approx. Time | Notes |
|-------------|--------------|-------|
| 100 KOs | 1-2 minutes | With MNLI and embeddings |
| 1,000 KOs | 10-15 minutes | GPU recommended for domain scores |
| 10,000 KOs | 2-3 hours | Batch processing advised |

**Tips for faster processing**:
- Use GPU for domain scoring (`sentence-transformers` auto-detects)
- Disable MNLI if not needed (requires code modification)
- Process in chunks for very large corpora

## Integration with Other Tools

### Chain with Content Diagnostics
```bash
# First: content quality
python ko_content_quality_diagnostics.py --input kos.json --field ko_content_flat

# Then: comprehensive quality assessment
python ko_quality_assessor_kc.py --input output/kos_with_*.json
```

### Compare Quality Before/After
```bash
# Before improvement
python ko_quality_assessor_kc.py --input kos_v1.json --output-dir output/before

# After improvement
python ko_quality_assessor_kc.py --input kos_v2.json --output-dir output/after

# Compare distributions
diff output/before/*.tsv output/after/*.tsv
```

## See Also

- `quality_structural_kc.py` — Structural quality implementation
- `quality_semantic_kc.py` — Semantic quality implementation
- `quality_functional_kc.py` — Functional quality implementation
- `quality_domain_kc.py` — Domain quality implementation
- `KO_CONTENT_QUALITY_DIAGNOSTICS.md` — Content-level quality analysis
- `KO_CONTENT_QUALITY_ANALYSIS.md` — Content quality reporting
