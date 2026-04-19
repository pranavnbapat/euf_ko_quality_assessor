# Scientific KO Quality Assessment Framework

This folder contains an ML-oriented framework for KO quality assessment. It is intended to support a more rigorous workflow than fixed heuristic rules, but it is still a framework rather than a pre-validated production model.

## What This Folder Provides

- feature extraction for KO quality signals
- model training infrastructure
- validation and calibration components
- SHAP-based explainability hooks
- a main API in [assessor_scientific.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/scientific_quality/assessor_scientific.py)

## What It Does Not Guarantee By Itself

This codebase does not, by itself, prove that you already have a scientifically validated quality assessor.

That only becomes true after you:

1. collect reliable human labels
2. train a model on those labels
3. evaluate it on held-out data
4. inspect calibration and error behavior

Without that, this folder should be understood as infrastructure for scientific quality assessment, not as a completed validated result.

## Positioning Against the Heuristic Assessor

| Aspect | Heuristic Assessor | Scientific Framework |
|--------|--------------------|----------------------|
| Scoring | Fixed rules and thresholds | Trainable model |
| Weights | Fixed or manually configured | Can be learned from data |
| Features | Hand-built diagnostics | Broader feature pipeline |
| Validation | Manual interpretation | Validation workflow supported |
| Explainability | Rule tracing | SHAP-style explanations available |
| Uncertainty | Not explicit | Model-side uncertainty support |

## Main Entry Point

The main API is [assessor_scientific.py](/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/assess_ko_quality/scientific_quality/assessor_scientific.py).

It exposes:

- `ScientificQualityAssessor.train(...)`
- `ScientificQualityAssessor.assess_ko(...)`
- `ScientificQualityAssessor.assess_batch(...)`
- `ScientificQualityAssessor.save(...)`
- `ScientificQualityAssessor.load(...)`

## Typical Workflow

### 1. Prepare labeled data

You need KOs plus human judgments for the relevant quality dimensions.

Example:

```python
from scientific_quality.annotation import (
    QualityRubric,
    stratified_sample_for_annotation,
)

QualityRubric.export_markdown("annotation_guidelines.md")
sample = stratified_sample_for_annotation(all_kos, n_samples=300)
```

### 2. Train a model

```python
from scientific_quality import ScientificQualityAssessor

assessor = ScientificQualityAssessor()
results = assessor.train(kos, labels)
assessor.save("models/quality_v1")
```

### 3. Run inference

```python
assessor = ScientificQualityAssessor.load("models/quality_v1")
result = assessor.assess_ko(ko_dict)

print(result["overall_quality"])
print(result["confidence"])
```

## Project Structure

```text
scientific_quality/
├── __init__.py
├── config.py
├── assessor_scientific.py
├── USAGE_EXAMPLE.py
├── annotation/
├── features/
├── models/
├── validation/
├── explainability/
└── monitoring/
```

## Feature Families

### Embedding Features

- title-content similarity
- description-content similarity
- paragraph or segment coherence
- semantic density proxies

### Readability Features

Implemented readability features include standard textstat-style measures such as:

- Flesch Reading Ease
- Flesch-Kincaid Grade
- SMOG
- Coleman-Liau
- Automated Readability Index
- Dale-Chall

### Information-Theoretic Features

- entropy
- type-token ratio
- hapax-style rarity signals
- lexical diversity measures

### Legacy / Basic Signals

- content length
- token counts
- keyword overlap
- simpler similarity signals

## Validation Expectations

This framework supports validation, but does not ship with universally valid benchmark results.

For a real deployment, you would normally want:

- held-out evaluation against human judgments
- calibration checks
- feature importance review
- error analysis
- stability checks across sources or time

The exact acceptance thresholds depend on your annotation scheme and use case.

## Current Limitations

1. It requires labeled data.
2. Many feature choices are English-centric.
3. Training and explanation can be computationally expensive.
4. Even with SHAP, the trained model is still less directly interpretable than fixed rules.

## Recommended Interpretation

Treat this folder as:

- a serious framework for building a scientific assessor
- useful infrastructure for experiments and model development

Do not treat it as:

- proof that a validated model already exists
- proof that current default metrics meet a predefined benchmark

## Summary

The code in this folder is broadly aligned with an ML-based quality assessment workflow. The correct way to describe it is “framework and infrastructure for scientific quality assessment,” not “already scientifically validated replacement.”
