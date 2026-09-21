# KO Content Quality Analysis

`ko_content_quality_analysis.py` — Generate human-readable reports from KO quality diagnostics output.

## Overview

This script reads the JSON output from `ko_content_quality_diagnostics.py` and produces a comprehensive console report with grade distributions, score statistics, common flags, and top/bottom quality rankings.

## Features

- **Multi-format input**: Handles both direct arrays and wrapped format (`{meta, stats, docs: [...]}`)
- **Statistical summaries**: Mean, percentiles (p10, p50, p90), grade distributions
- **Flag analysis**: Most common quality issues across the corpus
- **Ranked lists**: Best and worst KOs by quality score
- **Debug output**: Shows available keys to help troubleshoot missing metrics

## Usage

### Basic Usage

```bash
python ko_content_quality_analysis.py \
    --input output/kos_with_ko_content_flat_metrics.json \
    --text-field ko_content_flat \
    --topk 10
```

### With Custom Top-K

```bash
python ko_content_quality_analysis.py \
    --input output/kos_with_metrics.json \
    --text-field ko_content_flat \
    --topk 20
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input` | ✅ Yes | — | Path to JSON file with quality metrics (output from diagnostics) |
| `--text-field` | ❌ No | `ko_content_flat` | Base field name that was analyzed |
| `--topk` | ❌ No | `10` | Number of top/bottom KOs to display |

## Input Format

The script expects JSON with `{field}_metrics` objects added by the diagnostics script.

### Format 1: Direct Array
```json
[
  {
    "@id": "https://example.com/ko/1",
    "title": "Example KO",
    "ko_content_flat": "Content...",
    "ko_content_flat_metrics": {
      "field_quality_score_0_100": 72.5,
      "field_quality_grade": "B",
      "field_quality_flags": [],
      "tokens": 1250,
      ...
    }
  }
]
```

### Format 2: Wrapped Format
```json
{
  "meta": { ... },
  "stats": { ... },
  "docs": [
    {
      "@id": "https://example.com/ko/1",
      "title": "Example KO",
      "ko_content_flat_metrics": { ... }
    }
  ]
}
```

## Output Report

The script prints a formatted report to console:

```
==============================
KO CONTENT QUALITY — SUMMARY
==============================
Input file: output/kos_with_ko_content_flat_metrics.json
Input format: wrapped (meta/stats/docs)
Field analyzed: ko_content_flat
KOs with usable quality metrics: 1262
KOs skipped (missing/invalid): 0

--- Grade counts ---
 A:    245  ( 19.4%)
 B:    512  ( 40.6%)
 C:    380  ( 30.1%)
 D:    125  (  9.9%)
 ?:      0  (  0.0%)

--- Score stats ---
mean=68.3  p10=42.5  p50=70.2  p90=88.1

--- Most common flags ---
                   short_text: 342
                very_short_text: 89
           readability_skipped_non_english: 45
                      no_entities_found: 23

--- Token stats ---
mean=485  min=12  max=5000

--- Bottom 10 (lowest quality score) ---
 1. score= 12.5  tokens=    45  ents=     0  ttr=  0.15  [Short Title]  [id123]  flags=very_short_text,high_noise_ratio
 2. score= 15.2  tokens=    67  ents=     1  ttr=  0.22  [Another KO]  [id124]  flags=short_text,low_keyword_diversity
...

--- Top 10 (highest quality score) ---
 1. score= 94.2  tokens=  1250  ents=    45  ttr=  0.52  [Excellent Content]  [id456]  flags=
 2. score= 92.8  tokens=   980  ents=    38  ttr=  0.48  [Great Article]  [id457]  flags=
...

Done.
```

## Report Sections Explained

### Summary Header
- **Input file**: Path to analyzed JSON
- **Input format**: Whether wrapped (`meta/stats/docs`) or direct array
- **Field analyzed**: Which field was quality-assessed
- **KOs with usable metrics**: Successfully analyzed count
- **KOs skipped**: Missing or invalid quality data

### Grade Counts
Distribution of quality grades (A/B/C/D):
- **A** (≥85): Excellent quality
- **B** (≥70): Good quality
- **C** (≥55): Acceptable quality
- **D** (<55): Poor quality
- **?**: Missing/unknown grade

### Score Stats
- **mean**: Average quality score across all KOs
- **p10**: 10th percentile (bottom 10% threshold)
- **p50**: Median (50th percentile)
- **p90**: 90th percentile (top 10% threshold)

### Most Common Flags
Lists the quality flags that appear most frequently, indicating common issues in the corpus.

### Token Stats
Basic statistics on content length:
- **mean**: Average token count
- **min**: Shortest content
- **max**: Longest content

### Bottom K (Lowest Quality)
Ranked list of poorest quality KOs showing:
- Quality score and grade
- Token count
- Entity mentions count
- Type-Token Ratio (lexical diversity)
- KO title and ID
- Quality flags (issues found)

### Top K (Highest Quality)
Ranked list of best quality KOs with the same metrics as Bottom K.

## Typical Workflow

### Step 1: Run Diagnostics
```bash
python ko_content_quality_diagnostics.py \
    --input kos.json \
    --field ko_content_flat
```
Output: `output/kos_with_ko_content_flat_metrics.json`

### Step 2: Analyze Results
```bash
python ko_content_quality_analysis.py \
    --input output/kos_with_ko_content_flat_metrics.json \
    --text-field ko_content_flat \
    --topk 15
```

### Step 3: Review TSV (Optional)
```bash
# Open in Excel/LibreOffice
libreoffice output/kos_content_quality_check.tsv

# Or filter with command line
awk -F'\t' '$18 == "D" {print $1, $2}' output/kos_content_quality_check.tsv
```

## Troubleshooting

### "No KOs with valid metrics found"
**Cause**: Input file doesn't have computed metrics.

**Solution**: Run diagnostics first:
```bash
python ko_content_quality_diagnostics.py \
    --input original_kos.json \
    --field ko_content_flat
```

### "Looking for metrics key: 'xxx_metrics'"
**Cause**: `--text-field` doesn't match the field used in diagnostics.

**Solution**: Use the same field name:
```bash
# If diagnostics used --field description
python ko_content_quality_analysis.py \
    --input output/kos_with_description_metrics.json \
    --text-field description
```

### "Input format: wrapped" but metrics are missing
**Cause**: The `docs` array doesn't contain KOs with metrics.

**Solution**: Check if diagnostics completed successfully and look at the first KO keys in the debug output.

### Empty grade counts or score stats
**Cause**: All KOs were skipped due to missing/invalid metrics.

**Solution**: Verify the input JSON was generated by the diagnostics script and contains `{field}_metrics` objects.

## Examples

### Quick quality check
```bash
python ko_content_quality_analysis.py \
    --input output/kos_with_metrics.json \
    --text-field ko_content_flat
```

### Deep dive into poorest content
```bash
python ko_content_quality_analysis.py \
    --input output/kos_with_metrics.json \
    --text-field ko_content_flat \
    --topk 50 > poor_quality_kos.txt
```

### Compare different fields
```bash
# Analyze description field
python ko_content_quality_diagnostics.py --input kos.json --field description

python ko_content_quality_analysis.py \
    --input output/kos_with_description_metrics.json \
    --text-field description

# Analyze full content field
python ko_content_quality_diagnostics.py --input kos.json --field ko_content_flat

python ko_content_quality_analysis.py \
    --input output/kos_with_ko_content_flat_metrics.json \
    --text-field ko_content_flat
```

## Interpreting Results

### Healthy Corpus Indicators
- **Grade distribution**: Majority in A/B grades (>60%)
- **Mean score**: Above 70
- **p10 threshold**: Above 50 (bottom 10% still acceptable)
- **Common flags**: Mostly empty or minor issues only

### Concerning Patterns
- **High D-grade percentage** (>20%): Many poor quality KOs
- **Mean score below 60**: Overall quality issues
- **Frequent `very_short_text`**: Many KOs lack substantive content
- **Frequent `no_entities_found`**: Content may be too generic

### Actionable Insights
| Pattern | Possible Action |
|---------|-----------------|
| Many `very_short_text` flags | Review minimum content requirements |
| High `high_noise_ratio` | Check for encoding issues or special characters |
| Frequent `readability_skipped_non_english` | Consider language-specific processing |
| Low TTR scores | Content may be repetitive; check for duplication |

## See Also

- `ko_content_quality_diagnostics.py` — Compute quality metrics
- `KO_CONTENT_QUALITY_DIAGNOSTICS.md` — Documentation for diagnostics script
- `output/*_content_quality_check.tsv` — Machine-readable quality data
