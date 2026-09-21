# Summary Generation Framework for Knowledge Objects

A comprehensive, scientifically-backed guide for generating high-quality summaries from KO content (`ko_content_flat` and original documents).

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Summarization Analysis](#pre-summarization-analysis)
3. [Compression Difficulty Score (CDS)](#compression-difficulty-score-cds)
4. [Content Quality Flags](#content-quality-flags)
5. [Multimodal Content Handling](#multimodal-content-handling)
6. [Summarization Strategy Selection](#summarization-strategy-selection)
7. [Summary Generation Steps](#summary-generation-steps)
8. [Post-Generation Quality Checks](#post-generation-quality-checks)
9. [Decision Matrix](#decision-matrix)
10. [Scientific References](#scientific-references)

---

## Overview

This framework provides a systematic approach to generating summaries from Knowledge Object content. It considers:

- **Statistical properties** of the text (entropy, density, repetition)
- **Quality metrics** from existing diagnostics
- **Multimodal elements** (tables, images, charts)
- **Domain-specific requirements** (agricultural content)
- **Length and complexity** factors

### Core Principle

> *"One size does not fit all. Summary generation should be adaptive based on content characteristics."*

---

## Pre-Summarization Analysis

Before generating a summary, analyze these factors:

### 1. Content Length Metrics

| Metric | Thresholds | Action |
|--------|------------|--------|
| **Token Count** | < 80 | Skip summarization; use full text |
| | 80-250 | Minimal compression (preserve all) |
| | 250-600 | Light summarization (60-80% ratio) |
| | 600-1500 | Standard summarization (20-40% ratio) |
| | 1500-3000 | Moderate compression (15-30% ratio) |
| | 3000-6000 | Aggressive compression (10-25% ratio) |
| | > 6000 | Hierarchical/multi-stage summarization |
| **Sentence Count** | < 5 | Use full text |
| | 5-20 | Extractive methods preferred |
| | > 50 | Consider section-based summarization |
| **Character Count** | < 600 | Use full text |

### 2. Structural Analysis

```python
# Check for document structure
structure_markers = {
    "has_sections": ["Introduction", "Methods", "Results", "Conclusion"],
    "has_steps": ["Step 1", "Step 2", "Step 3", "First", "Second", "Finally"],
    "has_headers": ["# ", "## ", "### "],  # Markdown headers
    "has_lists": ["- ", "* ", "1. ", "2. "],
}
```

**Implication:** Structured documents benefit from section-aware summarization.

### 3. Language Detection

| Language | Strategy |
|----------|----------|
| English | Use standard pipeline |
| Non-English | Use language-specific model or translation |
| Mixed | Segment by language, summarize separately |

**Implementation:** Check `field_quality_flags` for `readability_skipped_non_english`

---

## Compression Difficulty Score (CDS)

CDS is computed by `ko_compression_diagnostics.py` and determines how difficult the content is to compress.

### CDS Components

| Component | Weight | Range | Interpretation |
|-----------|--------|-------|----------------|
| **Lexical Density** | 0.22 | 0.35-0.65 | Higher = more content words = harder to compress |
| **Unigram Entropy** | 0.22 | 5.5-8.5 | Higher = more unpredictable vocab = harder |
| **Entity Density** | 0.22 | 2-10/100tok | Higher = more concepts = harder |
| **Sentence Length** | 0.18 | 12-35 words | Longer = more complex syntax = harder |
| **Bigram Repetition** | -0.12 | 0-0.35 | Higher = more redundancy = easier |
| **Adjacent Similarity** | -0.10 | 0.20-0.85 | Higher = local repetition = easier |
| **Centroid Similarity** | -0.08 | 0.25-0.90 | Higher = topic stability = neutral |

### CDS Formula

```
CDS = (0.22 × norm(lexical_density)) 
    + (0.22 × norm(unigram_entropy))
    + (0.22 × norm(entity_density))
    + (0.18 × norm(sentence_length))
    - (0.12 × norm(bigram_repetition))
    - (0.10 × norm(adjacent_similarity))
    - (0.08 × norm(centroid_similarity))
```

Normalized to [0, 1] range.

### CDS-Based Ratio Bands

| CDS Range | Band | Summary Ratio | Model Recommendation |
|-----------|------|---------------|---------------------|
| < 0.35 | Low | 10-20% | Small model (14B/GPT-OSS) |
| 0.35-0.60 | Medium | 20-30% | Small model |
| > 0.60 | High | 30-45% | Large model (30B+) |

**Special Case:** If token_count >= 900 and CDS != Low → use Large model

---

## Content Quality Flags

Check `ko_content_quality_diagnostics.py` output flags and adjust strategy:

### Critical Flags (May Skip Summarization)

| Flag | Condition | Action |
|------|-----------|--------|
| `empty_text` | No content | Skip; flag for review |
| `very_short_text` | < 80 tokens | Use full text as "summary" |

### Warning Flags (Adjust Strategy)

| Flag | Issue | Summarization Adjustment |
|------|-------|-------------------------|
| `short_text` | < 250 tokens | Minimal compression (preserve 80%+) |
| `high_noise_ratio` | > 18% non-alphanumeric | Pre-clean text before summarization |
| `no_entities_found` | No NER detected | Use extractive methods; abstractive may hallucinate |
| `low_keyword_diversity` | < 10 unique keywords | Be conservative; content may be repetitive |
| `high_repetition` | > 25% bigram repeat | Aggressive deduplication first |
| `very_hard_to_read` | Flesch < 5 | Simplify during summarization |
| `truncated_for_spacy` | Content > 200k chars | Consider hierarchical approach |

### Informational Flags (Context Only)

| Flag | Meaning |
|------|---------|
| `readability_skipped_non_english` | Non-English content; use appropriate model |
| `embedding_coherence_failed` | Semantic analysis failed; rely on lexical methods |

---

## Multimodal Content Handling

This is a **critical gap** in the current pipeline. Original documents may contain tables, images, and charts.

### Detection Pipeline

```
PDF/Document → Parser → Element Detection → Processing → Textual Representation
```

### 1. Table Detection and Processing

#### Detection Methods
- **PDF Parsers:** Camelot, Tabula, pdfplumber
- **Heuristics:** Grid-like structures, repeated patterns
- **Metadata:** PDF tags indicating table regions

#### Processing Steps

```python
table_processing = {
    "extraction": "Extract cells with coordinates",
    "structure_parsing": "Identify headers, rows, columns",
    "content_extraction": "Extract text from each cell",
    "summary_generation": "Generate natural language description",
}
```

#### Table-to-Text Conversion

**Example:**
```markdown
# Original Table (visual)
| Crop    | Yield (t/ha) | Water Use (mm) |
|---------|-------------|----------------|
| Wheat   | 8.5         | 450            |
| Barley  | 7.2         | 380            |

# Textual Representation for Summary Context
"Table showing crop yields and water usage: Wheat produces 8.5 tonnes 
per hectare using 450mm of water, while Barley produces 7.2 tonnes per 
hectare using 380mm of water."
```

#### Tools
- **Camelot:** Best for scanned PDFs
- **Tabula:** Best for digital PDFs
- **Pandas:** Data manipulation after extraction

### 2. Image and Figure Processing

#### Detection Methods
- **Image metadata:** Bounding boxes from PDF parser
- **Caption proximity:** Text containing "Figure N:" or "Fig. N:"
- **Aspect ratio:** Images typically have aspect ratios different from text blocks

#### Processing Strategy

```python
image_processing = {
    "extract_image": "Save image from PDF",
    "extract_caption": "Find nearest caption text",
    "vlm_analysis": "Use Vision-Language Model for description",
    "context_integration": "Insert description into summary context",
}
```

#### Vision-Language Models (VLMs)

| Model | Use Case | Cost | Quality |
|-------|----------|------|---------|
| GPT-4V / GPT-4o | High-stakes agricultural content | High | Excellent |
| Claude 3.5 Sonnet | Balanced quality/cost | Medium | Very Good |
| LLaVA 1.6 | Open-source option | Low | Good |
| Qwen-VL | Multilingual support | Low | Good |

#### VLM Prompt Template

```
You are analyzing an image from an agricultural knowledge object.

Image Context: {extracted_caption}
Document Topic: {ko_title}

Describe what this image shows, focusing on:
1. Main subject and visual elements
2. Any data, measurements, or quantities visible
3. Agricultural relevance and practical implications
4. Relationship to the document topic

Provide a concise 2-3 sentence description suitable for inclusion in a text summary.
```

### 3. Chart and Diagram Processing

#### Types and Strategies

| Chart Type | Extraction Method | Description Focus |
|------------|------------------|-------------------|
| **Bar Chart** | OCR + axis detection | Trends, comparisons, key values |
| **Line Graph** | OCR + line detection | Trends over time, rates of change |
| **Pie Chart** | OCR + segment detection | Proportions, percentages |
| **Scatter Plot** | OCR + point detection | Correlations, distributions |
| **Flowchart** | VLM | Process flow, decision points |
| **Diagram** | VLM | Components, relationships |

#### Data Extraction Approach

```python
chart_processing = {
    "detect_chart_type": "Use heuristics or VLM",
    "extract_data_points": "OCR for text, CV for visual elements",
    "describe_trends": "Generate natural language description",
    "include_key_values": "Preserve specific numbers in description",
}
```

---

## Summarization Strategy Selection

Based on all factors, select the appropriate method:

### Decision Tree

```
START
  │
  ├── Is content very short (< 250 tokens)?
  │   ├── YES → Use full text
  │   └── NO → Continue
  │
  ├── Does content have critical quality flags?
  │   ├── YES (empty/noisy) → Flag for manual review
  │   └── NO → Continue
  │
  ├── Does content have multimodal elements?
  │   ├── YES → Use Multimodal LLM (GPT-4V/Claude 3.5)
  │   └── NO → Continue
  │
  ├── What is CDS?
  │   ├── CDS < 0.35 → Extractive summarization (small model)
  │   ├── 0.35 ≤ CDS < 0.60 → Abstractive (small model)
  │   └── CDS ≥ 0.60 → Abstractive (large model)
  │
  └── Is content very long (> 4000 tokens)?
      ├── YES → Hierarchical summarization
      └── NO → Standard single-pass
```

### Method Definitions

#### 1. Extractive Summarization
- **Algorithm:** TextRank, BM25-based selection, or centroid-based
- **When to use:** Low CDS, high repetition, need for factual precision
- **Pros:** Faithful to source, no hallucination
- **Cons:** Less fluent, limited compression

#### 2. Abstractive Summarization
- **Algorithm:** Seq2Seq LLM (T5, BART, GPT)
- **When to use:** Medium-High CDS, need for fluent summaries
- **Pros:** Fluent, high compression, paraphrasing
- **Cons:** Risk of hallucination, may lose specifics

#### 3. Hierarchical Summarization
- **Algorithm:** Chunk → Summarize → Combine → Final summary
- **When to use:** Very long documents (> 4000 tokens)
- **Process:**
  ```
  Document → [Chunk 1, Chunk 2, ...] → [Summary 1, Summary 2, ...] 
  → Intermediate Summary → Final Summary
  ```

#### 4. Multimodal Summarization
- **Algorithm:** Vision-Language Model
- **When to use:** Documents with tables, figures, images
- **Input:** Text + Image descriptions + Table summaries
- **Output:** Unified text summary

---

## Summary Generation Steps

### Step 1: Preprocessing

```python
def preprocess(content, quality_flags):
    """
    1. Clean noise (if high_noise_ratio flagged)
    2. Deduplicate (if high_repetition flagged)
    3. Detect language
    4. Detect multimodal elements
    5. Compute CDS
    """
```

### Step 2: Multimodal Processing (if applicable)

```python
def process_multimodal(document):
    """
    1. Extract tables → Convert to markdown/text
    2. Extract images → VLM captioning
    3. Extract charts → Data extraction + description
    4. Combine all into context window
    """
```

### Step 3: Strategy Selection

```python
def select_strategy(cds, token_count, has_multimodal, quality_flags):
    """
    Returns: {
        "method": "extractive|abstractive|hierarchical|multimodal",
        "model_size": "small|large",
        "compression_ratio": (min, max),
        "chunking_strategy": "none|section|fixed",
    }
    """
```

### Step 4: Summary Generation

#### Prompt Engineering Guidelines

**Base Prompt Template:**

```markdown
You are generating a summary for an agricultural knowledge object.

# Source Content
{preprocessed_content}

# Context
- Document Title: {title}
- Document Type: {category}
- Target Audience: Farmers, practitioners, researchers
- Content Quality Score: {quality_score}/100

# Instructions
1. Create a summary between {min_tokens} and {max_tokens} tokens
2. Preserve specific numbers, measurements, and dates
3. Maintain agricultural terminology accuracy
4. Include practical recommendations if present
5. Focus on: {key_topics_from_keywords}

# Output Format
{structured_output_format}
```

**Conditional Prompt Additions:**

| Condition | Addition |
|-----------|----------|
| Has tables | "Include key data points from tables" |
| Has figures | "Reference important visual information" |
| Very high CDS | "Prioritize key concepts over details" |
| Hard to read (Flesch < 5) | "Simplify complex language while preserving meaning" |

### Step 5: Post-Processing

```python
def postprocess(summary):
    """
    1. Trim to target length
    2. Fix formatting issues
    3. Ensure complete sentences
    4. Remove hallucinated content (if detectable)
    """
```

---

## Post-Generation Quality Checks

### 1. Intrinsic Metrics (No Reference)

| Metric | Target | Computation |
|--------|--------|-------------|
| **Compression Ratio** | Match CDS band | summary_tokens / source_tokens |
| **Entity Coverage** | > 70% | entities_in_summary / entities_in_source |
| **Readability Preservation** | ±10 Flesch points | Compare Flesch scores |
| **Length Compliance** | Within target range | Token count check |

### 2. Faithfulness Metrics

| Metric | Method | Target |
|--------|--------|--------|
| **Claim Entailment** | NLI model (MNLI) | > 0.80 |
| **Keyword Overlap** | Jaccard similarity | > 0.30 |
| **Semantic Similarity** | BERTScore / Sentence-BERT | > 0.75 |

### 3. Quality Flags for Generated Summaries

| Flag | Condition | Action |
|------|-----------|--------|
| `summary_too_short` | < 50% of target min | Regenerate with relaxed constraints |
| `summary_too_long` | > 150% of target max | Truncate or regenerate |
| `low_entity_coverage` | < 50% | Use more extractive approach |
| `hallucination_detected` | Claims not in source | Flag for review; consider extractive |
| `poor_readability` | Flesch < 0 | Simplify summary |

---

## Decision Matrix

Quick reference for strategy selection:

| Tokens | CDS | Multimodal | Quality Flags | Strategy | Model | Ratio |
|--------|-----|------------|---------------|----------|-------|-------|
| < 250 | Any | Any | Any | **Use Full Text** | - | 100% |
| 250-600 | < 0.35 | No | Clean | Extractive | Small | 60-80% |
| 250-600 | >= 0.35 | No | Clean | Abstractive | Small | 40-60% |
| 600-1500 | < 0.35 | No | Clean | Extractive | Small | 40-60% |
| 600-1500 | 0.35-0.60 | No | Clean | Abstractive | Small | 25-35% |
| 600-1500 | > 0.60 | No | Clean | Abstractive | Large | 30-45% |
| 1500-3000 | < 0.35 | No | Clean | Extractive | Small | 30-40% |
| 1500-3000 | >= 0.35 | No | Clean | Hierarchical | Large | 15-25% |
| 3000-6000 | Any | No | Clean | Hierarchical | Large | 10-25% |
| > 6000 | Any | No | Clean | Multi-stage Hierarchical | Large | 5-15% |
| Any | Any | **Yes** | Clean | **Multimodal LLM** | VLM | Adaptive |
| Any | Any | Any | **High Noise** | Pre-clean + Standard | Adaptive | Adaptive |
| Any | Any | Any | **High Repetition** | Deduplicate + Standard | Adaptive | Adaptive |

---

## Implementation Checklist

### Phase 1: Foundation (Current)
- [x] Compute CDS (`ko_compression_diagnostics.py`)
- [x] Quality flag detection (`ko_content_quality_diagnostics.py`)
- [x] Content length analysis
- [ ] Integrate CDS into summarization decisions

### Phase 2: Multimodal Support
- [ ] Table extraction (Camelot/Tabula)
- [ ] Table-to-text conversion
- [ ] Image detection and extraction
- [ ] VLM integration for image captioning
- [ ] Chart data extraction

### Phase 3: Advanced Summarization
- [ ] Hierarchical summarization pipeline
- [ ] Extractive summarization module
- [ ] Abstractive summarization with model selection
- [ ] Faithfulness checking (NLI)
- [ ] Post-generation quality validation

### Phase 4: Evaluation
- [ ] ROUGE/BERTScore evaluation pipeline
- [ ] Human evaluation framework
- [ ] A/B testing for strategy effectiveness

---

## Scientific References

### Information Theory & Text Metrics
1. **Shannon, C.E. (1948).** A Mathematical Theory of Communication. *Bell System Technical Journal*.
2. **Halliday, M.A.K. (1985).** An Introduction to Functional Grammar. *Edward Arnold*.

### Summarization
3. **Mihalcea, R., & Tarau, P. (2004).** TextRank: Bringing Order into Text. *EMNLP*.
4. **Nallapati, R., et al. (2016).** Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond. *CoNLL*.
5. **Lewis, M., et al. (2020).** BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation. *ACL*.

### Evaluation
6. **Lin, C.Y. (2004).** ROUGE: A Package for Automatic Evaluation of Summaries. *ACL Workshop*.
7. **Zhang, T., et al. (2020).** BERTScore: Evaluating Text Generation with BERT. *ICLR*.
8. **Maynez, J., et al. (2020).** On Faithfulness and Factuality in Abstractive Summarization. *ACL*.

### Long Document & Hierarchical Summarization
9. **Zaheer, M., et al. (2020).** Big Bird: Transformers for Longer Sequences. *NeurIPS*.
10. **Zhang, J., et al. (2022).** SummIt: Iterative Text Summarization via ChatGPT. *arXiv*.

### Multimodal Processing
11. **Li, J., et al. (2020).** Multimodal Summarization with Guiding Markup. *NAACL*.
12. **Liu, H., et al. (2024).** Visual Instruction Tuning. *NeurIPS* (LLaVA).

### Table Extraction
13. **Ammar, W., et al. (2014).** Table Extraction and Understanding in Scientific Documents. *JCDL*.

### Domain-Specific
14. **Durrett, G., et al. (2016).** Learning-based Single-Document Summarization with Compression and Anaphoricity Constraints. *TACL*.

---

## Related Files in This Repository

| File | Purpose |
|------|---------|
| `ko_compression_diagnostics.py` | Computes CDS and compression metrics |
| `ko_content_quality_diagnostics.py` | Content quality assessment and flags |
| `ko_content_quality_analysis.py` | Quality report generation |
| `quality_semantic.py` | Semantic quality scoring |
| `quality_structural.py` | Structural quality scoring |
| `quality_functional.py` | Functional/RAG quality scoring |
| `calibrate_compression.py` | Compression calibration and faithfulness testing |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-02-20 | Initial framework documentation |

---

## Notes

- This framework assumes access to the outputs from `ko_compression_diagnostics.py` and `ko_content_quality_diagnostics.py`
- Multimodal processing requires additional dependencies (Camelot, VLMs)
- Model recommendations are based on current (2024) capabilities; update as models improve
- Always validate summaries for agricultural domain accuracy
