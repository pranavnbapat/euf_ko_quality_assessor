#!/usr/bin/env bash
#
# deploy_to_runpod.sh
#
# Upload a local JSON file and ko_quality_assesor.py to a Runpod instance via scp.

set -euo pipefail

############################
# CONFIG – EDIT THESE
############################

# Remote SSH details
REMOTE_USER="cditf9qfq1di7o-64411be9"
REMOTE_HOST="ssh.runpod.io"

SSH_KEY="${HOME}/.ssh/id_ed25519"

# Remote base directory (project root on Runpod)
REMOTE_BASE_DIR="/workspace/ko_quality_assessor"

# Where the assessor code will live on Runpod
REMOTE_CODE_DIR="${REMOTE_BASE_DIR}/assess_ko_quality"
REMOTE_INPUT_DIR="${REMOTE_CODE_DIR}/input"

# Local project root (parent of assess_ko_quality and which_model_to_choose)
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Local JSON file to upload
LOCAL_JSON_FILE="${LOCAL_PROJECT_DIR}/which_model_to_choose/improve_ko/input/final_output_24_11-2025_03-50-22_for_qa.json"

# Python script to upload
LOCAL_PY_FILE="${LOCAL_PROJECT_DIR}/assess_ko_quality/ko_quality_assessor.py"

############################
# END CONFIG
############################

# Helper: print a nice message
log() {
  echo "[deploy] $*"
}

# Check local files
if [[ ! -f "${LOCAL_JSON_FILE}" ]]; then
  echo "[ERROR] JSON file not found: ${LOCAL_JSON_FILE}" >&2
  exit 1
fi

if [[ ! -f "${LOCAL_PY_FILE}" ]]; then
  echo "[ERROR] Python file not found: ${LOCAL_PY_FILE}" >&2
  exit 1
fi

if [[ ! -f "${SSH_KEY}" ]]; then
  echo "[ERROR] SSH key not found: ${SSH_KEY}" >&2
  exit 1
fi

# Ensure remote directories exist
log "Ensuring remote directories exist: ${REMOTE_CODE_DIR} and ${REMOTE_INPUT_DIR}"
ssh -T -o RequestTTY=no -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new \
  "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_CODE_DIR}' '${REMOTE_INPUT_DIR}'"


# Upload JSON file
log "Uploading JSON file: ${LOCAL_JSON_FILE} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_INPUT_DIR}/"
scp -O -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new \
  "${LOCAL_JSON_FILE}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_INPUT_DIR}/"


# Upload Python script
log "Uploading Python script: ${LOCAL_PY_FILE} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CODE_DIR}/"
scp -O -i "${SSH_KEY}" -o StrictHostKeyChecking=accept-new \
  "${LOCAL_PY_FILE}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_CODE_DIR}/"

log "Done."
