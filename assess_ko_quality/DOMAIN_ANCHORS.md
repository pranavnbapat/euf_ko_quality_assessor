# Domain Anchors System

## Overview

The `anchors/` folder contains **domain anchor texts** derived from authoritative agricultural thesauri and controlled vocabularies. These anchors serve as the foundation for computing a **domain centroid vector** - a mathematical representation of "agricultural-ness" in embedding space.

The centroid is used by `quality_domain.py` to score how agriculturally-relevant a Knowledge Object (KO) is, based on cosine similarity between the KO's content and the pre-computed agricultural domain prototype.

---

## Directory Structure

```
anchors/
├── agrovoc/
│   └── agrovoc_anchor_texts.jsonl      # AGROVOC concepts (FAO) ~20K entries
├── nalt/
│   ├── nalt-full_dwn_20240716.rdf      # Raw USDA NALT thesaurus (source)
│   └── nalt_anchor_texts.jsonl         # Parsed NALT concepts ~100K+ entries
├── centroids/
│   ├── agri_anchor_centroid.npy        # Pre-computed centroid vector (384-dim)
│   └── agri_anchor_centroid.meta.json  # Metadata (model, sources, counts)
└── cabi/                               # (Optional) CABI thesaurus anchors
└── wikibooks/                          # (Optional) Wikibooks agriculture chunks
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DOMAIN ANCHOR GENERATION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   AGROVOC (FAO)     │     │   NALT (USDA)       │     │   CABI (Optional)   │
│   ─────────────     │     │   ───────────       │     │   ─────────────     │
│   SPARQL Endpoint   │     │   Local RDF File    │     │   Local RDF File    │
│   aims.fao.org      │     │   nalt-full_*.rdf   │     │   cabi_thesaurus.rdf│
└──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│build_agrovoc_anchor_│     │build_nalt_anchor_   │     │build_cabi_anchor_   │
│texts.py             │     │texts.py             │     │texts.py             │
│─────────────────────│     │─────────────────────│     │─────────────────────│
│• Fetches via SPARQL │     │• Parses local RDF   │     │• Parses local RDF   │
│• 20K concepts max   │     │• Extracts SKOS      │     │• Extracts SKOS      │
│• Labels + definitions│    │• prefLabel/altLabel │     │• prefLabel/altLabel │
│• Broader terms      │     │• Broader concepts   │     │• Broader concepts   │
└──────────┬──────────┘     └──────────┬──────────┘     └──────────┬──────────┘
           │                           │                           │
           ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ANCHOR TEXTS (JSONL Format)                               │
│  ─────────────────────────────────────────────                               │
│  Each line: {"uri": "...", "anchor_text": "rich descriptive text..."}        │
│                                                                              │
│  Example AGROVOC:                                                            │
│  {"uri": "http://aims.fao.org/aos/agrovoc/c_00008d7b",                       │
│   "prefLabel_en": "air-water exchanges",                                     │
│   "anchor_text": "Concept: air-water exchanges. Alternative labels:         │
│    air-sea exchanges; air-sea transfer; water-air exchanges.                 │
│    Broader terms: natural phenomena."}                                       │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         compute_centroid.py                                  │
│  ─────────────────────────────────────────                                   │
│  • Loads anchor texts from all available sources                             │
│  • Embeds using SentenceTransformer (default: all-mpnet-base-v2)             │
│  • Computes mean vector → L2-normalizes → saves as .npy                      │
│  • Writes metadata JSON (model, counts, sources)                             │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGRICULTURAL DOMAIN CENTROID                              │
│  ─────────────────────────────────────────────                               │
│  File: anchors/centroids/agri_anchor_centroid.npy                            │
│  Shape: (384,)  ← embedding dimension (mpnet-base)                           │
│  Meta:  anchors/centroids/agri_anchor_centroid.meta.json                     │
│                                                                              │
│  Current centroid stats:                                                     │
│  • Model: all-mpnet-base-v2                                                  │
│  • Texts: 96,691 anchors (AGROVOC + NALT combined)                           │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RUNTIME: KO QUALITY ASSESSMENT                            │
│  ─────────────────────────────────────────────                               │
│  File: ko_quality_assessor.py → imports quality_domain.py                    │
│                                                                              │
│  1. Load centroid via load_domain_centroid()                                 │
│  2. Model compatibility check (centroid vs runtime model)                    │
│  3. For each KO, compute cosine similarity to centroid:                      │
│     • title_sim, desc_sim, kw_sim, content_sim                               │
│  4. Map similarity [0,1] → score [0-5]                                       │
│  5. Aggregate: Domain_Total_Raw (0-20) → Domain_Score_0_25 (0-25)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Anchor Text Format

### AGROVOC Format
```json
{
  "uri": "http://aims.fao.org/aos/agrovoc/c_xxxx",
  "prefLabel_en": "preferred term",
  "altLabels_en": ["synonym1", "synonym2"],
  "broader_prefLabels_en": ["parent term"],
  "note_en": "definition or scope note",
  "anchor_text": "Concept: preferred term. Alternative labels: ..."
}
```

### NALT Format
```json
{
  "uri": "https://lod.nal.usda.gov/nalt/xxxx",
  "source": "nalt",
  "anchor_text": "Preferred term: term. Synonyms: ... Broader concepts: ..."
}
```

---

## Build Scripts

### 1. `scripts/build_agrovoc_anchor_texts.py`

**Purpose**: Fetch agricultural concepts from FAO's AGROVOC thesaurus via SPARQL.

**Source**: `https://agrovoc.fao.org/sparql`

**Output**: `anchors/agrovoc/agrovoc_anchor_texts.jsonl`

**Process**:
1. Queries SPARQL endpoint with pagination (2000 concepts/page)
2. Extracts: prefLabel, altLabel, scopeNote/definition, broader terms
3. Constructs rich anchor text combining all fields
4. Writes JSONL with ~20,000 concepts (configurable via `MAX_CONCEPTS`)

**Usage**:
```bash
python scripts/build_agrovoc_anchor_texts.py
```

---

### 2. `scripts/build_nalt_anchor_texts.py`

**Purpose**: Parse USDA's National Agricultural Library Thesaurus from local RDF.

**Source**: `anchors/nalt/nalt-full_dwn_20240716.rdf` (must be downloaded separately)

**Output**: `anchors/nalt/nalt_anchor_texts.jsonl`

**Process**:
1. Parses RDF/XML using rdflib
2. Extracts SKOS concepts: prefLabel, altLabel, definition, broader
3. Filters for English language labels
4. Builds anchor texts with context

**Usage**:
```bash
python scripts/build_nalt_anchor_texts.py \
    --input anchors/nalt/nalt-full_dwn_20240716.rdf \
    --out anchors/nalt/nalt_anchor_texts.jsonl
```

---

### 3. `scripts/build_cabi_anchor_texts.py` (Optional)

**Purpose**: Parse CABI Thesaurus for additional agricultural concepts.

**Source**: Local CABI RDF file (obtain from CABI on request)

**Output**: `anchors/cabi/cabi_anchor_texts.jsonl`

**Usage**:
```bash
python scripts/build_cabi_anchor_texts.py \
    --input /path/to/cabi_thesaurus.rdf \
    --out anchors/cabi/cabi_anchor_texts.jsonl
```

---

### 4. `scripts/compute_centroid.py`

**Purpose**: Compute the domain centroid vector from anchor texts.

**Inputs**: One or more anchor JSONL files

**Outputs**: 
- `anchors/centroids/agri_anchor_centroid.npy` - the centroid vector
- `anchors/centroids/agri_anchor_centroid.meta.json` - metadata

**Process**:
1. Loads anchor texts from available sources (AGROVOC, NALT, CABI, Wikibooks)
2. Embeds all texts using SentenceTransformer
3. Computes mean embedding → L2 normalizes
4. Saves .npy vector and metadata JSON

**Usage**:
```bash
# Default: use all available anchor sources
python scripts/compute_centroid.py

# Explicit sources with custom output
python scripts/compute_centroid.py \
    --inputs anchors/agrovoc/agrovoc_anchor_texts.jsonl \
             anchors/nalt/nalt_anchor_texts.jsonl \
    --out anchors/centroids/agri_agrovoc_nalt.npy \
    --meta anchors/centroids/agri_agrovoc_nalt.meta.json \
    --model all-mpnet-base-v2 \
    --batch-size 64
```

---

## Runtime Usage

### In `ko_quality_assessor.py`

```python
# Load precomputed centroid (default path)
centroid_path = os.environ.get(
    "AGRI_DOMAIN_CENTROID",
    str((Path(__file__).resolve().parent / "anchors" / "centroids" / "agri_anchor_centroid.npy"))
)
load_domain_centroid(centroid_path)
```

### In `quality_domain.py`

```python
def domain_scores(title, desc, content, keywords):
    """
    Returns domain relevance scores:
    - Domain_term_density: content similarity (0-5)
    - Domain_in_title: title similarity (0-5)
    - Domain_in_keywords: keyword similarity (0-5)
    - Domain_consistency: description similarity (0-5)
    - Domain_Total_Raw: sum (0-20)
    - Domain_Score_0_25: scaled (0-25)
    - Domain_similarity_*: raw cosine similarities [0,1]
    """
```

**Scoring Logic**:
1. Embed the text field using SentenceTransformer
2. Compute cosine similarity with domain centroid: `sim = dot(emb, centroid)`
3. Map similarity [0,1] → score [0-5] using thresholds:
   - ≥0.65 → 5 (highly agricultural)
   - ≥0.55 → 4
   - ≥0.45 → 3
   - ≥0.35 → 2
   - >0.00 → 1
   - 0.00 → 0 (off-domain)

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `AGRI_DOMAIN_CENTROID` | Path to centroid .npy file | `anchors/centroids/agri_anchor_centroid.npy` |
| `AGRI_EMB_MODEL_NAME` | Embedding model for runtime | `all-mpnet-base-v2` |

---

## Model Compatibility

**Critical**: The centroid MUST be built with the same embedding model used at runtime.

The system validates this by checking `agri_anchor_centroid.meta.json`:

```python
if centroid_model != runtime_model:
    raise ValueError("Embedding model mismatch!")
```

**Current compatible models**:
- `all-mpnet-base-v2` (default, 384-dim)
- `BAAI/bge-base-en-v1.5` (768-dim) - requires rebuilding centroid

---

## Rebuilding the Centroid

If you update anchor sources or change the embedding model:

```bash
# 1. Rebuild individual anchors (if needed)
python scripts/build_agrovoc_anchor_texts.py
python scripts/build_nalt_anchor_texts.py --input anchors/nalt/nalt-full_dwn_20240716.rdf

# 2. Recompute centroid
python scripts/compute_centroid.py

# 3. Verify
python -c "import numpy as np; print(np.load('anchors/centroids/agri_anchor_centroid.npy').shape)"
# Output: (384,)
```

---

## Summary Table

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| `build_agrovoc_anchor_texts.py` | SPARQL (FAO) | `agrovoc_anchor_texts.jsonl` | AGROVOC concepts |
| `build_nalt_anchor_texts.py` | Local RDF | `nalt_anchor_texts.jsonl` | NALT concepts |
| `build_cabi_anchor_texts.py` | Local RDF | `cabi_anchor_texts.jsonl` | CABI concepts (optional) |
| `compute_centroid.py` | JSONL files | `agri_anchor_centroid.npy` | Domain prototype vector |
| `quality_domain.py` | KO content + centroid | Domain scores | Agricultural relevance scoring |

---

## Key Design Decisions

1. **External Vocabularies**: Uses authoritative agricultural thesauri (AGROVOC, NALT) rather than hardcoded terms
2. **Rich Anchor Texts**: Combines labels, synonyms, definitions, and broader terms for better embeddings
3. **Pre-computed Centroid**: Centroid is computed once offline, not at runtime - faster and stable
4. **Model Validation**: Runtime checks ensure centroid and embedding model compatibility
5. **Extensible**: Easy to add new anchor sources (CABI, Wikibooks, etc.)
