# assess_ko_quality/utils.py

import gzip
import orjson
import os

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _latest_json_file(folder: str) -> Path:
    candidates: List[Path] = []
    for pat in ("*.json", "*.jsonl", "*.ndjson", "*.json.gz", "*.jsonl.gz", "*.ndjson.gz"):
        candidates.extend(Path(folder).glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


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

def _read_json_any(path: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield dicts from JSON array or (gz)NDJSON.
    Entire file is read into memory for simplicity.
    """
    name = path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()

    # Try JSON array / object first
    try:
        obj = orjson.loads(data)
        if isinstance(obj, dict):
            yield obj
            return
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
            return
    except orjson.JSONDecodeError:
        pass

    # Fallback: NDJSON
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = orjson.loads(line)
            if isinstance(item, dict):
                yield item
        except Exception:
            continue


def _unique_outfile(base_dir: Path, stem: str = "quality", ext: str = ".tsv") -> Path:
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

