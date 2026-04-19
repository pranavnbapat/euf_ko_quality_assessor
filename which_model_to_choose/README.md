# which_model_to_choose

This folder is the model-selection pipeline for KO summary and metadata generation.

It is being migrated from older ad hoc scripts into a staged workflow with:

1. GPU-fit candidate filtering
2. candidate generation
3. evaluation
4. aggregation
5. model selection

## Folder Layout

- `gpu_runtime/`
  Hardware-fit filtering and runtime configuration for SGLang-served models.

- `generation/`
  Candidate generation for:
  - summaries
  - metadata

- `evaluation/`
  Heuristic evaluation of generated summaries and metadata.

- `selection/`
  Final model ranking and selection utilities.

- `artifacts/`
  Generated candidate outputs and evaluation results.

## What To Run First

Run the pipeline in this order.

### 1. Prepare runtime environment

From repo root:

```bash
bash which_model_to_choose/gpu_runtime/setup_sglang.sh
```

Then put your runtime secrets in:

- `which_model_to_choose/gpu_runtime/.env`

At minimum:

```bash
HF_TOKEN=...
SGLANG_BASE_URL=http://127.0.0.1:8000/v1
SGLANG_API_KEY=sk-sglang-local
```

### 2. Generate GPU-fit model config

Example for A40:

```bash
python which_model_to_choose/gpu_runtime/build_sglang_model_config.py a40
```

This updates:

- `which_model_to_choose/gpu_runtime/runtime_config.yaml`

with the models that fit your chosen GPU under the selected constraints.

### 3. Start SGLang

Start SGLang separately for the model you want to test first.

This pipeline assumes an OpenAI-compatible SGLang endpoint is already running and reachable at:

- `SGLANG_BASE_URL`

### 4. Run summary generation

Use the new generation entrypoint as a module:

```bash
python -m which_model_to_choose.generation.run_candidates --task summary --all-models
```

Or for a single runtime-config model key:

```bash
python -m which_model_to_choose.generation.run_candidates \
  --task summary \
  --model qwen3_5_9b_fp16_a40
```

This creates:

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/manifest.json`
- `which_model_to_choose/artifacts/candidate_runs/<run_id>/<model_key>/summaries.json`

### 5. Run metadata generation

Metadata generation depends on the summaries from a previous summary run.

```bash
python -m which_model_to_choose.generation.run_candidates \
  --task metadata \
  --all-models \
  --summary-run-id <summary_run_id>
```

This creates:

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/<model_key>/metadata.json`

### 6. Evaluate summaries

```bash
python -m which_model_to_choose.evaluation.evaluate_summaries --run-id <run_id>
```

This creates:

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/evaluation/summary_model_ranking.json`

### 7. Evaluate metadata

```bash
python -m which_model_to_choose.evaluation.evaluate_metadata --run-id <run_id>
```

This creates:

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/evaluation/metadata_model_ranking.json`

### 8. Aggregate results

```bash
python -m which_model_to_choose.evaluation.aggregate_results --run-id <run_id>
```

This creates:

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/evaluation/combined_model_ranking.json`

### 9. Choose best models

```bash
python -m which_model_to_choose.selection.choose_best_models --run-id <run_id>
```

## Current Output Structure

For a run `<run_id>`:

- `artifacts/candidate_runs/<run_id>/manifest.json`
- `artifacts/candidate_runs/<run_id>/<model_key>/summaries.json`
- `artifacts/candidate_runs/<run_id>/<model_key>/metadata.json`
- `artifacts/candidate_runs/<run_id>/evaluation/summary_model_ranking.json`
- `artifacts/candidate_runs/<run_id>/evaluation/metadata_model_ranking.json`
- `artifacts/candidate_runs/<run_id>/evaluation/combined_model_ranking.json`

## Current Status

Implemented:

- SGLang-oriented GPU runtime config generator
- summary candidate generation
- metadata candidate generation
- first-pass evaluation layer
- aggregation and best-model selection

Still to improve:

- stronger evaluation metrics
- richer metadata scoring
- better reporting
- production hardening and run-time observability

## Important Notes

- Run the new scripts with `python -m ...`, not as bare files, because they use package-relative imports.
- The evaluation layer is currently heuristic. It is useful for ranking candidates, but not yet a fully validated benchmark framework.
- Summary and metadata winners do not have to be the same model.
