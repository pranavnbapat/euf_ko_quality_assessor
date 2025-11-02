# config.py

"""
Configuration module: defines folder paths and ensures required data exists.
Used by the main runner and assessor scripts.
"""

import os

from pathlib import Path

from langdetect import DetectorFactory

# Input/output and controlled vocabulary directories.
# These can be overridden with environment variables.
INPUT_FOLDER = Path(os.environ.get("KO_INPUT_DIR", "./input")).resolve()
OUTPUT_FOLDER = Path(os.environ.get("KO_OUTPUT_DIR", "./output")).resolve()
DATA_MODEL_DIR = Path(os.environ.get("KO_DM_DIR", "./data_model_v2")).resolve()

# Fail early if controlled vocabulary folder is missing.
# This prevents half-runs or cryptic import errors later.
if not DATA_MODEL_DIR.exists():
    raise FileNotFoundError(f"Controlled vocab folder not found: {DATA_MODEL_DIR}")

# Fix the random seed for 'langdetect' so language detection results are reproducible.
DetectorFactory.seed = 42
