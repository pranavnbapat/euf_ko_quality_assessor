# io_utils.py

"""
Small, dependency-light I/O utilities used by the runner pipeline.

Design goals:
- Be strict about paths (fail fast with clear exceptions).
- Support common JSON payload styles:
  * A single JSON object
  * A JSON array of objects
  * JSON Lines / NDJSON (optionally gzipped)
- Produce collision-free output filenames with timestamped stems.

Notes:
- We intentionally avoid importing heavy JSON libs beyond 'orjson' for speed.
- All functions raise explicit exceptions instead of returning sentinel values.
"""

import csv
import gzip
import orjson
import os
import pandas as pd

from datetime import datetime
from pathlib import Path


def ensure_directory(p: Path) -> None:
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"Expected a directory: {p}")
    p.mkdir(parents=True, exist_ok=True)

def assert_readable_dir(p: Path) -> None:
    if not p.exists():
        raise FileNotFoundError(f"Input folder not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Expected a directory: {p}")
    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for: {p}")

def latest_json_file(folder: Path) -> Path:
    candidates = []
    for pat in ("*.json", "*.jsonl", "*.ndjson", "*.json.gz", "*.jsonl.gz", "*.ndjson.gz"):
        candidates.extend(folder.glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def read_json_any(path: Path):
    name = path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    try:
        arr = orjson.loads(data)
        if isinstance(arr, dict):
            yield arr
        else:
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
        return
    except orjson.JSONDecodeError:
        pass
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = orjson.loads(line)
            if isinstance(obj, dict):
                yield obj
        except Exception:
            continue

def unique_outfile(base_dir: Path, stem: str = "assessments", ext: str = ".tsv") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{stem}_{ts}{ext}"
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        alt = base_dir / f"{stem}_{ts}_{i}{ext}"
        if not alt.exists():
            return alt
        i += 1

def write_tsv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        lineterminator="\n",
        encoding="utf-8",
    )
