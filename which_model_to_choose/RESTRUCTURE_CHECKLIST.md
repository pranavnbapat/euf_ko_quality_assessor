# which_model_to_choose Restructure Checklist

This file turns the migration plan into an implementation checklist for rebuilding `which_model_to_choose` as a proper model-selection pipeline for KO summary and metadata generation.

Target runtime choice for the new pipeline: `SGLang`.

## Goal

Replace the current mixed script layout with a staged pipeline:

1. hardware-fit filtering
2. candidate generation
3. evaluation
4. aggregation
5. model selection
6. production handoff

---

## Stage 0: Keep / Remove Decision

Status: completed.

Removed legacy components after migration:

- `which_model_to_choose/get_summaries/`
- `which_model_to_choose/get_title_subtitle_description/`
- `which_model_to_choose/summary_analysis.py`
- `which_model_to_choose/bootstrap.sh`
- `which_model_to_choose/get_summaries.py`

These are no longer part of the active pipeline.

---

## Stage 1: Create New Folder Structure

Create these folders:

- `which_model_to_choose/gpu_runtime/`
- `which_model_to_choose/generation/`
- `which_model_to_choose/evaluation/`
- `which_model_to_choose/selection/`
- `which_model_to_choose/artifacts/`
- `which_model_to_choose/artifacts/candidate_runs/`
- `which_model_to_choose/artifacts/evaluation_results/`

Status: core folders created for runtime, generation, evaluation, and selection. Artifact directories are created on demand by the pipeline.

---

## Stage 2: GPU Runtime Layer

### New files to add

- `which_model_to_choose/gpu_runtime/build_sglang_model_config.py`
- `which_model_to_choose/gpu_runtime/model_static_check.py`
- `which_model_to_choose/gpu_runtime/model_repos.txt`
- `which_model_to_choose/gpu_runtime/runtime_config.yaml`
- `which_model_to_choose/gpu_runtime/setup_sglang.sh`

### Source mapping

- Your current `generate_gpu_config.py`
  -> `which_model_to_choose/gpu_runtime/build_sglang_model_config.py`

- Your current `config.yaml`
  -> `which_model_to_choose/gpu_runtime/runtime_config.yaml`

- Your current `model_repos.txt`
  -> `which_model_to_choose/gpu_runtime/model_repos.txt`

- Your current `setup.sh`
  -> `which_model_to_choose/gpu_runtime/setup_sglang.sh`

- Your current `model_static_check.py`
  -> `which_model_to_choose/gpu_runtime/model_static_check.py`

### Required code changes

- rename vLLM-specific config labels to SGLang-specific or runtime-neutral labels
- keep OpenAI-compatible assumptions, but do not hard-code `vllm`
- support target GPUs such as `a40`, `l40s`, `a100`, `h200`, etc.
- keep `concurrent_users` and `target_max_output_tokens`
- make output config suitable for batch generation as well as serving

### Deliverable

A generated runtime config that answers:

- which candidate models fit on the chosen GPU
- what `max_model_len` to use
- what `usable_input_tokens` remain
- what runtime memory target is safe

---

## Stage 3: Generation Layer

### New files to add

- `which_model_to_choose/generation/client.py`
- `which_model_to_choose/generation/config.py`
- `which_model_to_choose/generation/io_helpers.py`
- `which_model_to_choose/generation/prompts.py`
- `which_model_to_choose/generation/run_candidates.py`
- `which_model_to_choose/generation/summary_pipeline.py`
- `which_model_to_choose/generation/metadata_pipeline.py`

### Source mapping

#### Summary generation

- `which_model_to_choose/get_summaries/config.py`
  -> `which_model_to_choose/generation/config.py`

- `which_model_to_choose/get_summaries/io_helpers.py`
  -> `which_model_to_choose/generation/io_helpers.py`

- `which_model_to_choose/get_summaries/prompts.py`
  -> `which_model_to_choose/generation/prompts.py`

- `which_model_to_choose/get_summaries/pipeline.py`
  -> split into:
  - `which_model_to_choose/generation/summary_pipeline.py`
  - `which_model_to_choose/generation/client.py`

- `which_model_to_choose/get_summaries/main.py`
  -> `which_model_to_choose/generation/run_candidates.py`

#### Metadata generation

- `which_model_to_choose/get_title_subtitle_description/pipeline.py`
  -> `which_model_to_choose/generation/metadata_pipeline.py`

- `which_model_to_choose/get_title_subtitle_description/prompts.py`
  -> merge into `which_model_to_choose/generation/prompts.py`

- `which_model_to_choose/get_title_subtitle_description/main.py`
  -> absorbed into `which_model_to_choose/generation/run_candidates.py`

### Required code changes

- remove hard-coded model-specific input field names
  - for example: `ko_content_flat_summarised_qwenb_30b_instruct`

- use a runtime-neutral candidate artifact format

- separate `summary` and `metadata` tasks cleanly

- implement one client abstraction in `generation/client.py`
  - current backend target: `SGLang`
  - interface should remain runtime-neutral if possible

- store outputs per model in artifacts instead of writing model names into field names directly

### Recommended artifact layout

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/<model_key>/summaries.json`
- `which_model_to_choose/artifacts/candidate_runs/<run_id>/<model_key>/metadata.json`
- `which_model_to_choose/artifacts/candidate_runs/<run_id>/manifest.json`

### Manifest should include

- input file
- task type
- model repo
- served model name
- runtime
- prompt version
- decoding params
- timestamp

---

## Stage 4: Evaluation Layer

### New files to add

- `which_model_to_choose/evaluation/metrics.py`
- `which_model_to_choose/evaluation/evaluate_summaries.py`
- `which_model_to_choose/evaluation/evaluate_metadata.py`
- `which_model_to_choose/evaluation/aggregate_results.py`

### Source mapping

- `which_model_to_choose/summary_analysis.py`
  -> split across:
  - `which_model_to_choose/evaluation/metrics.py`
  - `which_model_to_choose/evaluation/evaluate_summaries.py`
  - `which_model_to_choose/evaluation/evaluate_metadata.py`
  - `which_model_to_choose/evaluation/aggregate_results.py`

### Required code changes

- move language checks into reusable metric helpers
- move embedding scoring into reusable metric helpers
- move NLI or consistency scoring into reusable metric helpers
- separate summary evaluation from metadata evaluation
- add operational metrics:
  - failure rate
  - JSON validity rate
  - average latency
  - average tokens generated

### Summary evaluation should answer

- which model produces the best summaries overall
- which model performs best by language / content size / quality band

### Metadata evaluation should answer

- which model produces the best title / subtitle / description / keywords candidates
- which model has the best schema compliance and usefulness

---

## Stage 5: Selection Layer

### New files to add

- `which_model_to_choose/selection/choose_best_models.py`
- `which_model_to_choose/selection/reporting.py`

### Responsibilities

- choose best summary model
- choose best metadata model
- optionally choose different winners by task
- optionally choose fallback models
- generate human-readable selection reports

### Output examples

- `which_model_to_choose/artifacts/evaluation_results/<run_id>/summary_model_ranking.json`
- `which_model_to_choose/artifacts/evaluation_results/<run_id>/metadata_model_ranking.json`
- `which_model_to_choose/artifacts/evaluation_results/<run_id>/selection_report.md`

---

## Stage 6: Root README / Folder README

### Add or update

- `which_model_to_choose/README.md`

### README should explain

1. what this folder is for
2. pipeline stages
3. how to generate runtime configs
4. how to run candidate generation
5. how to run evaluation
6. how to select the final model

---

## Stage 7: Execution Order

Recommended execution flow after restructure:

1. `gpu_runtime/build_sglang_model_config.py`
2. `generation/run_candidates.py --task summary`
3. `generation/run_candidates.py --task metadata`
4. `evaluation/evaluate_summaries.py`
5. `evaluation/evaluate_metadata.py`
6. `evaluation/aggregate_results.py`
7. `selection/choose_best_models.py`

---

## Stage 8: Cleanup Checklist

Remove only after the new pipeline is working:

- `which_model_to_choose/get_summaries.py`

Consider archiving after migration:

- `which_model_to_choose/get_summaries/`
- `which_model_to_choose/get_title_subtitle_description/`
- `which_model_to_choose/summary_analysis.py`

Do not delete them until:

- new generation pipeline works for summary
- new generation pipeline works for metadata
- new evaluation scripts produce equivalent or better outputs

---

## Stage 9: Immediate Implementation Priority

Recommended order of work:

### Priority 1

- add `gpu_runtime/`
- add SGLang-based `generation/client.py`
- add `generation/config.py`

### Priority 2

- migrate summary generation into `generation/summary_pipeline.py`
- create `generation/run_candidates.py`

### Priority 3

- migrate metadata generation into `generation/metadata_pipeline.py`

### Priority 4

- split `summary_analysis.py` into evaluation modules

### Priority 5

- add selection/reporting
- remove legacy entrypoints

---

## Definition of Done

This restructure is done when:

- candidate models are filtered by GPU fit before generation
- summary and metadata generation run from one unified generation layer
- outputs are stored as run artifacts rather than model-specific field names
- evaluation is separated from generation
- final model choice is reproducible and reportable
- legacy scripts are either removed or clearly archived
