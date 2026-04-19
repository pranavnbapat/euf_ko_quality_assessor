# Domain Anchors System

## Overview

The `anchors/` folder contains domain anchor texts derived from agricultural thesauri and related controlled vocabularies. These anchors are embedded offline and averaged into a domain centroid vector.

At runtime, the centroid is loaded by [quality_domain_kc.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/quality_domain_kc.py) and used by [ko_quality_assessor_kc.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/ko_quality_assessor_kc.py) to score how agriculturally relevant a KO is.

The runtime score is embedding-based. It is a domain-relevance proxy, not a classifier.

## Directory Structure

```text
anchors/
├── agrovoc/
│   └── agrovoc_anchor_texts.jsonl
├── nalt/
│   ├── nalt-full_dwn_20240716.rdf
│   └── nalt_anchor_texts.jsonl
├── centroids/
│   ├── agri_anchor_centroid.npy
│   └── agri_anchor_centroid.meta.json
├── cabi/        # optional additional anchors
└── wikibooks/   # optional additional anchors
```

## Offline Build Flow

1. Build anchor JSONL files from one or more sources such as AGROVOC and NALT.
2. Embed each `anchor_text` using a SentenceTransformer model.
3. Compute the mean embedding and L2-normalize it.
4. Save:
   - `agri_anchor_centroid.npy`
   - `agri_anchor_centroid.meta.json`

The centroid metadata is important because runtime code checks model compatibility before scoring.

## Runtime Flow

Current runtime code path:

- [ko_quality_assessor_kc.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/ko_quality_assessor_kc.py)
- [quality_domain_kc.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/quality_domain_kc.py)

Runtime behavior:

1. Load the centroid from `AGRI_DOMAIN_CENTROID` or the default `anchors/centroids/agri_anchor_centroid.npy`.
2. Optionally check centroid metadata against `AGRI_EMB_MODEL_NAME`.
3. Embed title, description, content, and keywords in a batch.
4. Compute cosine similarity to the centroid for each field.
5. Map similarity bands to 0-5 sub-scores.
6. Aggregate those sub-scores into `Domain_Score_0_25`.

## Important Implementation Notes

### Model compatibility

The centroid must be built with the same embedding model that runtime uses.

Current default runtime model in code:

- `AGRI_EMB_MODEL_NAME=all-mpnet-base-v2`

The metadata check happens in [quality_domain_kc.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/quality_domain_kc.py).

### Do not rely on a fixed embedding dimension in docs

The centroid is a 1-D vector, but its exact length depends on the embedding model used to build it.

That means this documentation intentionally avoids hard-coding a dimension such as `384` or `768` as a universal truth. If you need the actual dimension for a given centroid file, inspect the `.npy` file or its metadata.

### Truncation behavior

Long content is truncated before embedding. The current implementation keeps head and tail portions using token-aware truncation when possible, with a hard character cap before tokenization to avoid pathological slowdowns.

This means domain scoring for very large KOs is approximate and biased toward the beginning and end of the content.

## Scoring Logic

The runtime domain scorer produces field-level similarities and maps them to integer sub-scores.

Current threshold table in code:

- similarity `>= 0.65` -> `5`
- similarity `>= 0.55` -> `4`
- similarity `>= 0.45` -> `3`
- similarity `>= 0.35` -> `2`
- similarity `> 0.00` -> `1`
- similarity `== 0.00` -> `0`

The exact output column names come from [quality_domain_kc.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/quality_domain_kc.py), so that file should be treated as the source of truth.

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `AGRI_DOMAIN_CENTROID` | Path to centroid `.npy` file | `anchors/centroids/agri_anchor_centroid.npy` |
| `AGRI_EMB_MODEL_NAME` | Runtime embedding model name | `all-mpnet-base-v2` |
| `AGRI_VALIDATE_DIM` | Optional extra dimension validation | `0` |
| `AGRI_MAX_SEQ_LEN` | Max embedding tokens used at runtime | `512` |
| `AGRI_TRUNC_KEEP_END_TOKENS` | Tail tokens preserved during truncation | `112` |
| `AGRI_MAX_CHARS_PRETOKENISE` | Hard char cap before tokenization | `200000` |

## Interpretation

What a high score means:

- the KO text is semantically close to the agricultural anchor centroid
- the KO likely uses domain-relevant agricultural vocabulary and context

What it does not mean:

- the KO is factually correct
- the KO is high quality overall
- the KO belongs to a specific agricultural subdomain with certainty

Domain scoring is one pillar inside the broader four-pillar KO quality assessment.

## Summary

This folder supports an embedding-based agricultural relevance signal. The current implementation is centered on `_kc` runtime modules, model-metadata compatibility, and truncated long-text embedding. Treat the centroid as model-dependent and the domain score as a proxy, not ground truth.
