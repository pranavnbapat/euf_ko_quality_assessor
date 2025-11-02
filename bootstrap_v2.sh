#!/usr/bin/env bash
# End-to-end multi-GPU Ollama bootstrapper with model warm-up and stop command.
# Works on Linux; safe to run from PyCharm Run/Debug configs by passing env vars.
set -Eeuo pipefail

#############################################
# 0) CONFIG (override via env)
#############################################
: "${REQUIRED_OLLAMA_VER:=0.12.6}"

# GPUs to use: "0" or "0,1" or "0,1,2"
: "${GPU_LIST:=0}"

# First port; subsequent instances map to PORT_BASE + index
: "${PORT_BASE:=11434}"

# Per-instance limits: single value for all OR CSV aligned with GPU_LIST
#   e.g. OLLAMA_MAX_LOADED_MODELS="2"      or "2,1,1"
#        OLLAMA_NUM_PARALLEL="2"           or "2,1,1"
: "${OLLAMA_MAX_LOADED_MODELS:=2}"
: "${OLLAMA_NUM_PARALLEL:=2}"

# Global Ollama env
: "${OLLAMA_LOG_LEVEL:=debug}"
: "${OLLAMA_KEEP_ALIVE:=12h}"
: "${OLLAMA_MODELS:=/workspace/models}"

# CPU threads: default to half of available cores unless overridden
CPU_TOTAL="$(nproc)"
: "${OLLAMA_NUM_THREADS:=$(( CPU_TOTAL / 2 ))}"

# Models to pull/warm (space separated) OR via MODELS_FILE (one per line; supports # comments)
: "${MODELS:=gpt-oss:20b nomic-embed-text}"
: "${MODELS_FILE:=}"

# Files for runtime state
STATE_DIR="${STATE_DIR:-/workspace}"
PID_PREFIX="${PID_PREFIX:-${STATE_DIR}/ollama-gpu}"
LOG_PREFIX="${LOG_PREFIX:-${STATE_DIR}/ollama-gpu}"
VERSION_URL="https://ollama.com/install.sh"

#############################################
# 1) UTILS
#############################################

# Print nicely to stderr
say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# Split CSV to array
IFS=',' read -r -a GPU_ARR <<< "${GPU_LIST}"

# Resolve per-instance value: one value for all OR CSV aligned with instance index
resolve_per_instance() {
  local csv="$1" idx="$2"
  IFS=',' read -r -a vals <<< "$csv"
  if [[ "${#vals[@]}" -eq 1 ]]; then
    printf '%s' "${vals[0]}"
  else
    printf '%s' "${vals[$idx]}"
  fi
}

# HTTP readiness check
wait_until_up() {
  local host="$1" timeout="${2:-60}" start=$SECONDS
  until curl -fsS "http://${host}/api/version" >/dev/null 2>&1; do
    (( SECONDS - start >= timeout )) && return 1
    sleep 1
  done
}

# Load models from MODELS or MODELS_FILE
load_models_list() {
  if [[ -n "${MODELS_FILE}" && -f "${MODELS_FILE}" ]]; then
    # strip comments/blank lines
    mapfile -t _MODELS < <(grep -vE '^\s*(#|$)' "${MODELS_FILE}")
  else
    # split by spaces
    read -r -a _MODELS <<< "${MODELS}"
  fi
  printf '%s\n' "${_MODELS[@]}"
}

# Try pull a model a few times (network hiccups happen)
pull_with_retry() {
  local host="$1" model="$2" attempts=3
  for i in $(seq 1 "${attempts}"); do
    if OLLAMA_HOST="${host}" ollama pull "${model}"; then
      return 0
    fi
    say "Pull failed for ${model} on ${host} (try ${i}/${attempts}); retrying…"
    sleep 2
  done
  return 1
}

# Derived helpers
instance_port() { echo $(( PORT_BASE + $1 )); }     # idx -> port
instance_host() { echo "0.0.0.0:$(instance_port "$1")"; }
instance_pidfile() { echo "${PID_PREFIX}${1}.pid"; } # we index by *GPU ID* for readability
instance_logfile() { echo "${LOG_PREFIX}${1}.log"; }

#############################################
# 2) INSTALL / UPDATE OLLAMA (idempotent)
#############################################
ensure_ollama() {
  if command -v ollama >/dev/null 2>&1; then
    local curr
    curr="$(ollama --version 2>/dev/null | awk '{print $NF}')"
    if [[ "${curr}" != "${REQUIRED_OLLAMA_VER}" ]]; then
      say "Upgrading Ollama ${curr} → ${REQUIRED_OLLAMA_VER}…"
      curl -fsSL "${VERSION_URL}" | sh
    else
      say "Ollama ${curr} OK"
    fi
  else
    say "Installing Ollama ${REQUIRED_OLLAMA_VER}…"
    curl -fsSL "${VERSION_URL}" | sh
  fi
}

#############################################
# 3) START / STOP INSTANCES
#############################################
start_instance() {
  local idx="$1" gpu_id="$2"

  local host port max_loaded parallel log pid
  port="$(instance_port "${idx}")"
  host="$(instance_host "${idx}")"
  log="$(instance_logfile "${gpu_id}")"
  pid="$(instance_pidfile "${gpu_id}")"

  max_loaded="$(resolve_per_instance "${OLLAMA_MAX_LOADED_MODELS}" "${idx}")"
  parallel="$(resolve_per_instance "${OLLAMA_NUM_PARALLEL}" "${idx}")"

  # One Ollama server per GPU; we pin CUDA_VISIBLE_DEVICES accordingly.
  CUDA_VISIBLE_DEVICES="${gpu_id}" \
  OLLAMA_HOST="${host}" \
  OLLAMA_MAX_LOADED_MODELS="${max_loaded}" \
  OLLAMA_NUM_PARALLEL="${parallel}" \
  OLLAMA_LOG_LEVEL="${OLLAMA_LOG_LEVEL}" \
  OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE}" \
  OLLAMA_MODELS="${OLLAMA_MODELS}" \
  OLLAMA_NUM_THREADS="${OLLAMA_NUM_THREADS}" \
  nohup ollama serve > "${log}" 2>&1 &

  echo $! > "${pid}"

  if ! wait_until_up "127.0.0.1:${port}" 60; then
    say "❌ GPU${gpu_id} on ${host} failed to start"
    tail -n 200 "${log}" >&2 || true
    return 1
  fi
  say "✅ GPU${gpu_id} up on ${host} (max_loaded=${max_loaded}, parallel=${parallel})"
}

start_all() {
  # Prepare model store dirs *before* we start servers
  mkdir -p "${OLLAMA_MODELS}" "${OLLAMA_MODELS}/blobs" "${OLLAMA_MODELS}/manifests"

  ensure_ollama

  # Clean old PID files from previous runs
  for gpu_id in "${GPU_ARR[@]}"; do
    rm -f "$(instance_pidfile "${gpu_id}")"
  done

  # Start each requested instance sequentially (safer; avoids races on startup)
  for i in "${!GPU_ARR[@]}"; do
    start_instance "${i}" "${GPU_ARR[$i]}"
  done

  # Build list of hosts for warm-up
  HOSTS=()
  for i in "${!GPU_ARR[@]}"; do
    HOSTS+=("127.0.0.1:$(instance_port "${i}")")
  done

  # Pull and warm models on each instance
  mapfile -t MODELS_ARR < <(load_models_list)
  for H in "${HOSTS[@]}"; do
    for m in "${MODELS_ARR[@]}"; do
      if pull_with_retry "${H}" "${m}"; then
        # For chatty models, a tiny prompt forces first load; embeddings may no-op—ignore failures.
        printf "Hello\n" | OLLAMA_HOST="${H}" ollama run "${m}" >/dev/null 2>&1 || true
        say "Warmed ${m} on ${H}"
      else
        say "Skip: ${m} not found or failed to pull on ${H}"
      fi
    done
  done

  # Final status line
  READY=()
  for i in "${!GPU_ARR[@]}"; do
    READY+=("GPU${GPU_ARR[$i]}@$(instance_port "${i}") (PID $(cat "$(instance_pidfile "${GPU_ARR[$i]}")"))")
  done
  say "Ready: ${READY[*]}"
}

stop_all() {
  local any=0
  for gpu_id in "${GPU_ARR[@]}"; do
    local pidf pid
    pidf="$(instance_pidfile "${gpu_id}")"
    if [[ -f "${pidf}" ]]; then
      pid="$(cat "${pidf}")"
      if kill "${pid}" 2>/dev/null; then
        say "Stopped GPU${gpu_id} (pid ${pid})"
      fi
      rm -f "${pidf}"
      any=1
    fi
  done

  # Fallback: kill stray ollama serve (only if we started nothing via PID files)
  if [[ "${any}" -eq 0 ]]; then
    pkill -f 'ollama serve' 2>/dev/null || true
    say "No PID files, sent best-effort stop to any running 'ollama serve'."
  fi
}

#############################################
# 4) ENTRYPOINT
#############################################
main() {
  # Subcommands: start (default) | stop | status
  local cmd="${1:-start}"

  case "${cmd}" in
    start)
      start_all
      ;;
    stop)
      stop_all
      ;;
    status)
      for i in "${!GPU_ARR[@]}"; do
        local gpu_id="${GPU_ARR[$i]}"
        local port="$(instance_port "${i}")"
        local host="127.0.0.1:${port}"
        local pidf="$(instance_pidfile "${gpu_id}")"
        if curl -fsS "http://${host}/api/version" >/dev/null 2>&1; then
          if [[ -f "${pidf}" ]]; then
            say "GPU${gpu_id}@${port} UP (PID $(cat "${pidf}"))"
          else
            say "GPU${gpu_id}@${port} UP (PID unknown)"
          fi
        else
          say "GPU${gpu_id}@${port} DOWN"
        fi
      done
      ;;
    *)
      echo "Usage: $0 [start|stop|status]" >&2
      exit 2
      ;;
  esac
}

main "$@"
