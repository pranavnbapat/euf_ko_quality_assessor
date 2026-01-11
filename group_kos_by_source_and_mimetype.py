# group_kos_by_source_and_mimetype.py

"""
Group Knowledge Objects (KOs) by:
- ko_object_mimetype
- ko_upload_source
and combinations thereof.

Supports:
- JSON array file: [ {...}, {...}, ... ]
- NDJSON file: one JSON object per line

Usage:
  python group_kos_by_source_and_mimetype.py /path/to/kos.json
"""

from __future__ import annotations

import argparse
import json

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse


def classify_ko(ko: dict) -> str:
    """
    Classify a KO as 'file_only' or 'url_only'.
    Assumes mutual exclusivity by data model.
    """
    if ko.get("ko_is_hosted") is True:
        return "file_only"

    if ko.get("is_url_only") is True:
        return "url_only"

    if ko.get("ko_content_url"):
        return "url_only"

    # Defensive fallback – should not happen
    return "unknown"


def is_video_ko(ko: dict) -> bool:
    """
    Robust video detection:
    - Prefer mimetype video/*
    - Fallback to extension for cases where mimetype is missing/incorrect
    """
    mt = (ko.get("ko_object_mimetype") or "").lower().strip()
    if mt.startswith("video/"):
        return True

    ext = (ko.get("ko_object_extension") or "").lower().strip()
    if ext in {".mp4", ".mov", ".m4v", ".webm", ".avi"}:
        return True

    name = (ko.get("ko_object_name") or "").lower()
    return any(name.endswith(e) for e in [".mp4", ".mov", ".m4v", ".webm", ".avi"])


def is_known_video_host(url: str) -> bool:
    """
    Detect known third-party video hosting platforms.
    """
    u = (url or "").lower().strip()
    if not u:
        return False

    # Use netloc when possible (cleaner than substring matching)
    try:
        netloc = urlparse(u).netloc.lower()
    except Exception:
        netloc = u

    HOSTS = (
        "youtube.com", "youtu.be",
        "vimeo.com",
        "dailymotion.com",
        "wistia.com",
        "twitch.tv",
        "facebook.com", "fb.watch",
        "instagram.com",
        "tiktok.com",
        # add others you care about
    )
    return any(h in netloc for h in HOSTS)

def pick_best_video_identifier(ko: dict) -> str:
    """
    Decide which URL/identifier to test.
    Priority:
    1) ko_file_id (usually the actionable file location)
    2) @id
    """
    return (ko.get("ko_file_id") or ko.get("@id") or "").strip()

def is_video_not_on_video_host(ko: dict) -> bool:
    """
    Your target predicate:
    - video KO
    - hosted file KO (ko_is_hosted == True)
    - identifier is NOT a known video-hosting platform
    """
    if not is_video_ko(ko):
        return False

    if ko.get("ko_is_hosted") is not True:
        return False

    ident = pick_best_video_identifier(ko)
    if not ident:
        return False

    return not is_known_video_host(ident)

def normalise(value: Any, missing: str = "__MISSING__") -> str:
    """Normalise grouping keys to avoid None / blanks / weird types."""
    if value is None:
        return missing
    if isinstance(value, str):
        v = value.strip()
        return v if v else missing
    # If it's not a string, keep it deterministic
    return str(value)

def load_json_array(path: Path) -> List[Dict[str, Any]]:
    """Load a standard JSON array file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # If the file is a single object, treat as one KO
        return [data]
    raise ValueError("Input JSON must be an array of objects, or a single object.")


def load_ndjson(path: Path) -> Iterable[Dict[str, Any]]:
    """Load newline-delimited JSON (NDJSON): one object per line."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            if isinstance(obj, dict):
                yield obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to KO JSON file (array JSON or NDJSON).")
    parser.add_argument(
        "--ndjson",
        action="store_true",
        help="Treat input as NDJSON (one JSON object per line).",
    )
    parser.add_argument(
        "--missing-label",
        default="__MISSING__",
        help="Label to use when a field is missing/blank.",
    )
    args = parser.parse_args()

    path = Path(args.input_path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    # Load KOs
    if args.ndjson:
        kos_iter = load_ndjson(path)
    else:
        kos_iter = load_json_array(path)

    # 1) (upload_source, mimetype) combo counts
    combo_counts: Counter[tuple[str, str]] = Counter()

    # 2) nested groupings
    # mimetype -> upload_source -> count
    by_mimetype_then_source: dict[str, Counter[str]] = defaultdict(Counter)
    # upload_source -> mimetype -> count
    by_source_then_mimetype: dict[str, Counter[str]] = defaultdict(Counter)

    # --- hosted vs url-only counts ---
    ko_type_counts: Counter[str] = Counter()

    # cross-tabs (nice for sanity checks / dashboards)
    type_by_source: dict[str, Counter[str]] = defaultdict(Counter)  # source -> type -> count
    type_by_mimetype: dict[str, Counter[str]] = defaultdict(Counter)  # mimetype -> type -> count
    type_by_source_and_mime: Counter[tuple[str, str, str]] = Counter()  # (type, source, mimetype) -> count

    total = 0
    for ko in kos_iter:
        total += 1
        mimetype = normalise(ko.get("ko_object_mimetype"), missing=args.missing_label)
        source = normalise(ko.get("ko_upload_source"), missing=args.missing_label)

        combo_counts[(source, mimetype)] += 1
        by_mimetype_then_source[mimetype][source] += 1
        by_source_then_mimetype[source][mimetype] += 1

        ko_type = classify_ko(ko)
        ko_type_counts[ko_type] += 1

        type_by_source[source][ko_type] += 1
        type_by_mimetype[mimetype][ko_type] += 1
        type_by_source_and_mime[(ko_type, source, mimetype)] += 1

        if ko_type == "unknown":
            print(f"WARNING: unknown KO type for @id={ko.get('@id')}")

        if is_video_not_on_video_host(ko):
            ident = pick_best_video_identifier(ko)
            print(
                "VIDEO_NOT_ON_VIDEO_HOST",
                ko.get("@id"),
                "mimetype=", ko.get("ko_object_mimetype"),
                "ext=", ko.get("ko_object_extension"),
                "ident=", ident,
            )

    # Pretty printing
    print(f"\nTotal KOs processed: {total}\n")

    print("==== Grouped by ko_object_mimetype → ko_upload_source ====")
    for mimetype in sorted(by_mimetype_then_source.keys()):
        total_for_mime = sum(by_mimetype_then_source[mimetype].values())
        print(f"\nMIME type: {mimetype}  (total: {total_for_mime})")
        for source, count in by_mimetype_then_source[mimetype].most_common():
            print(f"  - {source}: {count}")

    print("\n\n==== Grouped by ko_upload_source → ko_object_mimetype ====")
    for source in sorted(by_source_then_mimetype.keys()):
        total_for_source = sum(by_source_then_mimetype[source].values())
        print(f"\nUpload source: {source}  (total: {total_for_source})")
        for mimetype, count in by_source_then_mimetype[source].most_common():
            print(f"  - {mimetype}: {count}")

    print("\n\n==== Grouped by (ko_upload_source, ko_object_mimetype) ====")
    for (source, mimetype), count in combo_counts.most_common():
        print(f"{source}\t{mimetype}\t{count}")

    print("==== KO Type (file_only vs url_only) ====")
    for ko_type, count in ko_type_counts.most_common():
        print(f"  - {ko_type}: {count}")

    print("\n==== KO Type by ko_upload_source ====")
    for source in sorted(type_by_source.keys()):
        total_for_source = sum(type_by_source[source].values())
        print(f"\nUpload source: {source} (total: {total_for_source})")
        for ko_type, count in type_by_source[source].most_common():
            print(f"  - {ko_type}: {count}")

    print("\n==== Grouped by (ko_type, ko_upload_source, ko_object_mimetype) ====")
    for (ko_type, source, mimetype), count in type_by_source_and_mime.most_common():
        print(f"{ko_type}\t{source}\t{mimetype}\t{count}")

if __name__ == "__main__":
    main()
