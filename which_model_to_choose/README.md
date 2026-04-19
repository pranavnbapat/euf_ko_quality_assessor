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

- `output/`
  Human-inspectable consolidated JSON / CSV exports for sampled comparison runs.

## What To Run First

Run the pipeline in this order.

## Deployment Modes

You can run this pipeline in two ways.

### Option A: Recommended

Run everything on the GPU machine:

- clone this repo on Runpod
- run SGLang on Runpod
- run generation and evaluation on Runpod

This avoids sending large KO payloads over the network and is the preferred setup for full runs.

### Option B: Local control, remote GPU serving

Run the Python pipeline on your local machine, but point it to an SGLang server running on Runpod.

In that setup:

- this repo stays on your local machine
- SGLang runs on Runpod
- set `SGLANG_BASE_URL` to the Runpod endpoint
- set `SGLANG_API_KEY` if your SGLang endpoint requires one

This works, but full runs will be slower and more network-dependent than running everything on Runpod.

### 1. Prepare runtime environment

From repo root:

```bash
bash which_model_to_choose/gpu_runtime/setup_sglang.sh
```

This automates the Runpod environment setup:

- creates `.venv` if needed
- activates it
- installs `uv`
- installs repo requirements with `uv`
- installs SGLang and runtime dependencies with `uv`
- creates the expected model/cache/output directories

You can then predownload the configured models:

```bash
bash which_model_to_choose/gpu_runtime/download_runtime_models.sh
```

Then put your runtime secrets in:

- `which_model_to_choose/gpu_runtime/.env`

At minimum:

```bash
HF_TOKEN=...
SGLANG_BASE_URL=http://127.0.0.1:8000/v1
SGLANG_API_KEY=sk-sglang-local
```

For local-control / Runpod-serving mode, set `SGLANG_BASE_URL` to your Runpod OpenAI-compatible endpoint instead of `http://127.0.0.1:8000/v1`.

### 2. Generate GPU-fit model config

Example for A40:

```bash
python which_model_to_choose/gpu_runtime/build_sglang_model_config.py a40
```

This updates:

- `which_model_to_choose/gpu_runtime/runtime_config.yaml`

with the models that fit your chosen GPU under the selected constraints.

### 3. Start SGLang

Start SGLang for the model you want to test first.

This pipeline assumes an OpenAI-compatible SGLang endpoint is already running and reachable at:

- `SGLANG_BASE_URL`

When starting SGLang, set `--served-model-name` to the `served_model_name` value for the selected model in:

- `which_model_to_choose/gpu_runtime/runtime_config.yaml`

You can now do this directly from the generated config:

```bash
python which_model_to_choose/gpu_runtime/start_sglang_server.py \
  --model-key qwen3_5_9b_fp16_a40
```

This launcher will:

- read `runtime_config.yaml`
- use the configured `repo` / `local_path`
- set `--served-model-name`
- pass `--context-length`
- pass `--mem-fraction-static`
- pass `--trust-remote-code` when required
- pass `--api-key` from `which_model_to_choose/gpu_runtime/.env` if present

If you only want to inspect the command:

```bash
python which_model_to_choose/gpu_runtime/start_sglang_server.py \
  --model-key qwen3_5_9b_fp16_a40 \
  --dry-run
```

If you want the full per-model cycle automated, you do not need to start SGLang manually. Use:

```bash
bash which_model_to_choose/gpu_runtime/run_model_cycle.sh --task summary --sample-size 50
```

That script will:

- read all model keys from `runtime_config.yaml`
- start SGLang for one model
- run generation for the chosen sample
- stop SGLang
- start the next model
- repeat until all configured models are processed

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

For practical model comparison, use a sampled run so the same random records are sent to every model:

```bash
python -m which_model_to_choose.generation.run_candidates \
  --task summary \
  --all-models \
  --sample-size 50 \
  --sample-seed 42 \
  --export-format both
```

This does the following:

- reads the latest JSON from the root `input/` folder unless `--input` is passed
- samples `X` records once using `--sample-size`
- sends that exact same sample to every selected model
- stores per-model artifacts
- writes consolidated comparison outputs to `which_model_to_choose/output/<run_id>/`

For a full Runpod cycle, prefer:

```bash
bash which_model_to_choose/gpu_runtime/run_model_cycle.sh \
  --task summary \
  --sample-size 50 \
  --sample-seed 42 \
  --export-format both
```

This creates:

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/manifest.json`
- `which_model_to_choose/artifacts/candidate_runs/<run_id>/<model_key>/summaries.json`
- `which_model_to_choose/output/<run_id>/summary_candidates.json`
- `which_model_to_choose/output/<run_id>/summary_candidates.csv`

### 5. Run metadata generation

Metadata generation depends on the summaries from a previous summary run.

```bash
python -m which_model_to_choose.generation.run_candidates \
  --task metadata \
  --all-models \
  --summary-run-id <summary_run_id> \
  --export-format both
```

If `--sample-size` is omitted for metadata, the script reuses the sampled record set from the summary run.

This creates:

- `which_model_to_choose/artifacts/candidate_runs/<run_id>/<model_key>/metadata.json`
- `which_model_to_choose/output/<run_id>/metadata_candidates.json`
- `which_model_to_choose/output/<run_id>/metadata_candidates.csv`

For the same per-model cycling behavior:

```bash
bash which_model_to_choose/gpu_runtime/run_model_cycle.sh \
  --task metadata \
  --run-id <metadata_run_id> \
  --summary-run-id <summary_run_id> \
  --export-format both
```

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
- `output/<run_id>/summary_candidates.json`
- `output/<run_id>/summary_candidates.csv`
- `output/<run_id>/metadata_candidates.json`
- `output/<run_id>/metadata_candidates.csv`

## Runpod First-Run Checklist

On a fresh A40 Runpod instance:

1. Clone the repo.
2. Run:

```bash
bash which_model_to_choose/gpu_runtime/setup_sglang.sh
```

3. Fill in:

- `which_model_to_choose/gpu_runtime/.env`

Example:

```bash
HF_TOKEN=...
SGLANG_API_KEY=some-long-random-string
SGLANG_BASE_URL=http://127.0.0.1:8000/v1
```

4. Generate the allowed model set:

```bash
python which_model_to_choose/gpu_runtime/build_sglang_model_config.py a40
```

5. Start one chosen model:

```bash
python which_model_to_choose/gpu_runtime/start_sglang_server.py \
  --model-key qwen3_5_9b_fp16_a40
```

Or predownload and run the automated per-model cycle:

```bash
bash which_model_to_choose/gpu_runtime/download_runtime_models.sh
```

6. Run a sampled summary comparison:

```bash
bash which_model_to_choose/gpu_runtime/run_model_cycle.sh \
  --task summary \
  --sample-size 50 \
  --sample-seed 42 \
  --export-format both
```

7. Then run metadata generation against the summary run:

```bash
bash which_model_to_choose/gpu_runtime/run_model_cycle.sh \
  --task metadata \
  --run-id <metadata_run_id> \
  --summary-run-id <summary_run_id> \
  --export-format both
```

## Local-Control / Remote-Runpod Usage

If SGLang is running on Runpod but you want to drive generation from your local machine:

- keep this repo locally
- set local `which_model_to_choose/gpu_runtime/.env` to the Runpod endpoint
- use the same `SGLANG_API_KEY` value that you used when launching SGLang on Runpod

Example local `.env` values:

```bash
SGLANG_BASE_URL=https://<runpod-endpoint>/v1
SGLANG_API_KEY=some-long-random-string
```

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
