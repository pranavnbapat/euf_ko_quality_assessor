# KO Content Quality Diagnostics

`ko_content_quality_diagnostics.py` - Compute content quality metrics for Knowledge Objects (KOs).

## Overview

This script analyzes text content within KO JSON files and computes comprehensive quality metrics including readability, lexical diversity, entity density, and semantic coherence. It adds quality scores (0-100) with grades (A/B/C/D) and flags potential issues.

## Features

- **Multi-format input support**: Direct JSON arrays or wrapped format (`{meta, stats, docs: [...]}`)
- **NLP-powered analysis**: Uses spaCy for entity extraction, keyword identification, and sentence parsing
- **Semantic coherence**: Optional SentenceTransformer embeddings for topic drift and redundancy detection
- **Weighted scoring**: Combines multiple metrics into a single quality score
- **Comprehensive flags**: Identifies issues like short text, high noise, repetition, low entity density
- **Preserves structure**: Output maintains original JSON format (meta/stats/docs)

## Installation

```bash
pip install spacy numpy pandas
python -m spacy download en_core_web_lg

# Optional: for better token counting
pip install tiktoken

# Optional: for semantic coherence metrics
pip install sentence-transformers torch
```

## Usage

### Basic Usage

```bash
python ko_content_quality_diagnostics.py \
    --input path/to/kos.json \
    --field ko_content_flat
```

### With Options

```bash
python ko_content_quality_diagnostics.py \
    --input path/to/kos.json \
    --field ko_content_flat \
    --tsv output/custom_metrics.tsv \
    --out-json output/custom_output.json \
    --id-field @id \
    --spacy-max-chars 100000 \
    --disable-embeddings \
    --log-every 100
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | ✅ Yes | - | Path to input JSON file (KO data) |
| `--field` | ✅ Yes | - | Field name to analyze (e.g., `ko_content_flat`) |
| `--tsv` | ❌ No | `output/{input}_content_quality_check.tsv` | Output TSV path |
| `--out-json` | ❌ No | `output/{input}_with_{field}_metrics.json` | Output JSON path |
| `--id-field` | ❌ No | `_orig_id` → `@id` → `id` | Preferred ID field for reporting |
| `--spacy-max-chars` | ❌ No | `200000` | Max characters to feed spaCy (truncation limit) |
| `--disable-embeddings` | ❌ No | - | Disable SentenceTransformer semantic coherence |
| `--st-model` | ❌ No | `sentence-transformers/all-mpnet-base-v2` | SentenceTransformer model name |
| `--device` | ❌ No | `auto` (`cuda` if available) | Device for embeddings (`auto`, `cpu`, `cuda`) |
| `--log-every` | ❌ No | `50` | Progress logging frequency (KOs) |

## Input Format

### Format 1: Direct Array (Legacy)
```json
[
  {
    "@id": "https://example.com/ko/1",
    "title": "Example KO",
    "ko_content_flat": "This is the content to analyze..."
  },
  ...
]
```

### Format 2: Wrapped Format (Recommended)
```json
{
  "meta": { "env_mode": "DEV", ... },
  "stats": { "total_docs": 1262, ... },
  "docs": [
    {
      "@id": "https://example.com/ko/1",
      "title": "Example KO",
      "ko_content_flat": "This is the content to analyze..."
    },
    ...
  ]
}
```

## Output Files

### 1. Augmented JSON (`{input}_with_{field}_metrics.json`)

Original JSON with added `{field}_metrics` field on each KO:

```json
{
  "@id": "https://example.com/ko/1",
  "title": "Example KO",
  "ko_content_flat": "Content text...",
  "ko_content_flat_metrics": {
    "tokens": 1250,
    "noise_ratio_non_alnum": 0.05,
    "flesch_reading_ease": 45.2,
    "ttr": 0.42,
    "unigram_entropy": 7.8,
    "bigram_repetition_ratio": 0.12,
    "entity_mentions": 15,
    "entity_unique": 8,
    "entity_density_per_100_tokens": 1.2,
    "keyword_top40_unique": 28,
    "embedding_sentence_count_used": 25,
    "mean_adjacent_cosine_similarity": 0.82,
    "mean_centroid_cosine_similarity": 0.71,
    "first_last_block_cosine_similarity": 0.68,
    "field_quality_score_0_100": 72.5,
    "field_quality_grade": "B",
    "field_quality_flags": [],
    "assessed_at_utc": "2026-02-19T21:31:50+00:00",
    "spacy_max_chars_used": 200000
  }
}
```

### 2. Summary TSV (`{input}_content_quality_check.tsv`)

One row per KO with key metrics for easy filtering in Excel/sheets:

| Column | Description |
|--------|-------------|
| `id` | KO identifier |
| `title` | KO title (truncated) |
| `field` | Analyzed field name |
| `tokens` | Token count |
| `noise_ratio_non_alnum` | Ratio of non-alphanumeric characters |
| `flesch_reading_ease` | Flesch reading ease score |
| `ttr` | Type-Token Ratio |
| `unigram_entropy` | Shannon entropy of unigrams |
| `bigram_repetition_ratio` | Ratio of repeated bigrams |
| `entity_unique` | Count of unique entities |
| `entity_mentions` | Total entity mentions |
| `entity_density_per_100_tokens` | Entities per 100 tokens |
| `keyword_top40_unique` | Unique keywords (top 40) |
| `embedding_sentence_count_used` | Sentences used for embeddings |
| `mean_adjacent_cosine_similarity` | Semantic similarity of adjacent sentences |
| `mean_centroid_cosine_similarity` | Similarity to document centroid |
| `first_last_block_cosine_similarity` | Topic drift (first vs last 20%) |
| `score_0_100` | Overall quality score |
| `grade` | Quality grade (A/B/C/D) |
| `flags` | Semicolon-separated quality flags |

### 3. Flags TSV (`{input}_content_quality_check_flags.tsv`)

One row per flag occurrence for detailed analysis:

| Column | Description |
|--------|-------------|
| `id` | KO identifier |
| `title` | KO title |
| `field` | Analyzed field |
| `flag` | Individual flag name |

## Quality Flags

| Flag | Trigger Condition |
|------|-------------------|
| `empty_text` | Field is empty or missing |
| `very_short_text` | Less than 80 tokens |
| `short_text` | Less than 250 tokens |
| `truncated_for_spacy` | Text exceeded `--spacy-max-chars` |
| `high_noise_ratio` | Non-alphanumeric ratio ≥ 18% |
| `no_entities_found` | No entities detected (tokens > 300) |
| `low_keyword_diversity` | Less than 10 unique keywords (tokens > 300) |
| `very_hard_to_read` | Flesch score < 5 |
| `readability_skipped_non_english` | Stopword ratio too low (non-English) |
| `high_repetition` | Bigram repetition ≥ 25% (tokens > 400) |
| `embedding_coherence_failed` | Embedding computation failed |

## Scoring Formula

Quality score (0-100) is computed from weighted components:

```
score = (
    0.08 × length_score +           # Adequate length (caps at 600 tokens)
    0.18 × noise_score +            # Low special characters
    0.16 × readability_score +      # Flesch reading ease
    0.16 × ttr_score +              # Lexical diversity
    0.12 × entropy_score +          # Information density
    0.16 × entity_density_score +   # Information richness
    0.14 × keyword_score            # Keyword diversity
)
- repetition_penalty
+ embedding_bonus
- embedding_penalty
```

### Grade Thresholds

| Score | Grade |
|-------|-------|
| ≥ 85 | A |
| ≥ 70 | B |
| ≥ 55 | C |
| < 55 | D |

## Examples

### Analyze default field with embeddings
```bash
python ko_content_quality_diagnostics.py \
    --input kos.json \
    --field ko_content_flat
```

### Quick analysis without embeddings (faster)
```bash
python ko_content_quality_diagnostics.py \
    --input kos.json \
    --field description \
    --disable-embeddings \
    --log-every 200
```

### Custom output paths
```bash
python ko_content_quality_diagnostics.py \
    --input ../data/kos.json \
    --field content \
    --tsv ../reports/quality.tsv \
    --out-json ../reports/kos_with_quality.json
```

### GPU acceleration for embeddings
```bash
python ko_content_quality_diagnostics.py \
    --input kos.json \
    --field ko_content_flat \
    --device cuda
```

## Troubleshooting

### "No KOs with valid metrics found"
- Ensure the `--field` exists in your KOs
- Check that the field contains text content (not null/empty)
- Verify input JSON format (direct array or wrapped with `docs`)

### spaCy model not found
```bash
python -m spacy download en_core_web_lg
# or fallback:
python -m spacy download en_core_web_sm
```

### Out of memory with embeddings
```bash
# Use CPU instead of GPU
python ko_content_quality_diagnostics.py ... --device cpu

# Or disable embeddings entirely
python ko_content_quality_diagnostics.py ... --disable-embeddings
```

### Very large files
```bash
# Reduce spaCy processing limit
python ko_content_quality_diagnostics.py ... --spacy-max-chars 100000
```

## Performance

| Configuration | Approx. Speed |
|--------------|---------------|
| With embeddings (CPU) | ~2-5 KOs/second |
| With embeddings (GPU) | ~10-20 KOs/second |
| Without embeddings | ~20-50 KOs/second |

## See Also

- `ko_content_quality_analysis.py` - Generate reports from diagnostics output
- `KO_CONTENT_QUALITY_ANALYSIS.md` - Documentation for analysis script
