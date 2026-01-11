"""
This is a flawed method. Not used anymore.
"""


# sample_kos_for_quality_assessment.py

import csv
import glob
import json
import logging
import os
import random
import sys

from collections import Counter, defaultdict
from datetime import datetime, UTC
from hashlib import md5

from urllib.parse import urlparse


# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]  # Print logs to console
)

ALLOWED_CC = {
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA"
}

JSON_FILE = os.path.join("input/final_output_10_01-2026_10-16-54.json")

def _is_file_backed(doc: dict) -> bool:
    """
    True only for KOs that have an actual file object in our storage / with real file metadata.
    """
    if doc.get("is_url_only") is True:
        return False
    if doc.get("ko_is_hosted") is False:
        return False

    mt = str(doc.get("ko_object_mimetype", "")).strip().lower()
    ext = str(doc.get("ko_object_extension", "")).strip().lower()

    if not mt and not ext:
        return False

    # 'application/octet-stream' is usually "unknown binary" and often useless here
    if mt == "application/octet-stream":
        # allow if we at least have a meaningful extension
        return bool(ext and ext != "<missing>")

    return True


def _extract_file_meta(doc: dict) -> tuple[str, str]:
    """
    Return (mimetype, extension). For URL-only KOs, extension is often missing,
    so infer from name/url when possible.
    """
    mt = str(doc.get("ko_object_mimetype", "")).strip() or "<missing>"
    ext = str(doc.get("ko_object_extension", "")).strip() or ""

    # Infer extension when missing/blank
    if not ext:
        # Try object name first (often has .pdf/.mp4)
        name = str(doc.get("ko_object_name", "")).strip()
        if name:
            ext = os.path.splitext(name)[1]

    if not ext:
        # Try ko_content_url, ko_file_id, @id (in that order)
        for k in ("ko_content_url", "ko_file_id", "@id"):
            u = str(doc.get(k, "")).strip()
            if u:
                ext = _derive_extension_from_url(u)
                if ext != "<missing>":
                    break

    ext = ext or "<missing>"
    return mt, ext


def filetype_bucket_from_mime_ext(mimetype: str, ext: str) -> str | None:
    """
    Decide sampling bucket from mimetype/extension, not JSON 'category'.
    Buckets: PDFs, Video, Audio, Image, Other
    """
    mt = (mimetype or "").strip().lower()
    ex = (ext or "").strip().lower()

    if mt == "application/pdf" or ex == ".pdf":
        return "PDFs"

    if mt.startswith("video/") or ex in {".mp4", ".mov", ".m4v", ".webm", ".avi"}:
        return "Video"

    if mt.startswith("audio/") or ex in {".mp3", ".wav", ".m4a", ".aac", ".ogg"}:
        return "Audio"

    if mt.startswith("image/") or ex in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".tif", ".tiff"}:
        return "Image"

    return "Other"

# Keep only these Creative Commons variants (case- and hyphen-insensitive)
def is_allowed_license(value: str) -> bool:
    if not value:
        return False
    clean = value.upper().replace("-", " ").strip()
    clean = " ".join(clean.split())
    for allowed in ALLOWED_CC:
        if clean.replace(" ", "") == allowed.replace("-", "").replace(" ", ""):
            return True
    return False

def _derive_extension_from_url(url: str | None) -> str:
    """
    Best-effort: try to infer file extension from @id URL path.
    If none, return "<missing>".
    """
    if not url:
        return "<missing>"
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    return ext if ext else "<missing>"

def _map_json_category_to_sampling_cat(json_category: str | None) -> str | None:
    """
    Map JSON 'category' (e.g. 'Document', 'Audio', 'Video', 'Image') to sampling buckets.
    """
    if not json_category:
        return None
    cat = json_category.strip().lower()
    if cat in {"document", "report", "guideline", "paper"}:
        return "PDFs"
    if cat in {"audio", "podcast"}:
        return "Audio"
    if cat in {"video", "webinar"}:
        return "Video"
    if cat in {"image", "photo", "figure"}:
        return "Image"
    return None

def _safe_ko_id(doc: dict) -> str:
    """
    Produce a stable KO id for the sample rows.
    Prefer JSON's '_orig_id' if present, else hash '@id'.
    """
    if doc.get("_orig_id"):
        return str(doc["_orig_id"])
    if doc.get("@id"):
        return md5(doc["@id"].encode("utf-8")).hexdigest()[:24]  # 24-hex-like for readability
    # last resort: hash the title + created_at for uniqueness
    blob = f"{doc.get('title','')}-{doc.get('ko_created_at','')}"
    return md5(blob.encode("utf-8")).hexdigest()[:24]

def _extract_langs_from_json(doc: dict) -> list[str]:
    """
    Normalise 'languages' to list[str].
    """
    raw = doc.get("languages")
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []

def _first_lang_key_label_from_json(doc: dict) -> tuple[str, str]:
    langs = _extract_langs_from_json(doc)
    first_raw = langs[0] if langs else "<missing>"
    return (first_raw.lower(), first_raw)

def _extract_license_from_json(doc: dict) -> str:
    return str(doc.get("license", "")).strip()

TARGETS = {
    "English":   {"PDFs": 67, "Audio": 2,  "Video": 38, "Image": 9},
    "Spanish":   {"PDFs": 4,  "Audio": 5,  "Video": 8,  "Image": 20},
    "Dutch":     {"PDFs": 23, "Audio": 0,  "Video": 3,  "Image": 1},
    "French":    {"PDFs": 20, "Audio": 0,  "Video": 2,  "Image": 0},
    "Italian":   {"PDFs": 15, "Audio": 0,  "Video": 1,  "Image": 0},
    "German":    {"PDFs": 8,  "Audio": 0,  "Video": 4,  "Image": 0},
    "Polish":    {"PDFs": 11, "Audio": 0,  "Video": 1,  "Image": 0},
    "Portuguese":{"PDFs": 9,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Greek":     {"PDFs": 9,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Slovene":   {"PDFs": 5,  "Audio": 0,  "Video": 1,  "Image": 0},
    "Hungarian": {"PDFs": 5,  "Audio": 0,  "Video": 1,  "Image": 0},
    "Romanian":  {"PDFs": 6,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Latvian":   {"PDFs": 6,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Finnish":   {"PDFs": 4,  "Audio": 0,  "Video": 1,  "Image": 0},
    "Lithuanian":{"PDFs": 5,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Czech":     {"PDFs": 1,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Danish":    {"PDFs": 1,  "Audio": 0,  "Video": 1,  "Image": 0},
    "Bulgarian": {"PDFs": 1,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Croatian":  {"PDFs": 1,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Swedish":   {"PDFs": 1,  "Audio": 0,  "Video": 0,  "Image": 0},
    "Irish":     {"PDFs": 1,  "Audio": 0,  "Video": 0,  "Image": 0},
}

if not os.path.isfile(JSON_FILE):
    # Optional convenience: if the exact file isn't present, fallback to latest matching in the same folder
    matches = sorted(glob.glob(os.path.join("final_output_*.json")), key=os.path.getmtime)
    if matches:
        JSON_FILE = matches[-1]
    else:
        raise FileNotFoundError(f"JSON file not found: {JSON_FILE}")

with open(JSON_FILE, "r", encoding="utf-8") as jf:
    try:
        json_data = json.load(jf)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {JSON_FILE}") from e

# Many exports are a list; handle wrapper objects too.
if isinstance(json_data, dict):
    docs_iter = json_data.get("items") or json_data.get("data") or []
elif isinstance(json_data, list):
    docs_iter = json_data
else:
    raise TypeError("Unexpected JSON structure; expected list or {items: [...]}/{data: [...]}")

# EXCLUDE: project_acronym == "Groen Kennisnet"
docs_iter = [d for d in docs_iter if d.get("project_acronym") != "Groen Kennisnet"]

print(f"Total KOs (JSON): {len(docs_iter)}")

lang_label_by_key = {}
lang_totals = Counter()
unlinked_per_lang = Counter()             # not used in JSON mode; stays zero
counts_by_lang_mime = defaultdict(Counter)
counts_by_lang_ext  = defaultdict(Counter)

# Candidate pools: Language label × Category
candidates_by_lang_cat: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

# Map sampled ko_id -> original JSON object (for QA JSON + CSV)
koid_to_doc: dict[str, dict] = {}

# Optional deterministic sampling
seed_val = os.environ.get("ASSESSMENT_RANDOM_SEED")
if seed_val:
    random.seed(seed_val)

# Build candidate pools from JSON
for doc in docs_iter:
    lic_value = _extract_license_from_json(doc)
    if not is_allowed_license(lic_value):
        continue

    # Only sample file-backed KOs (exclude URL-only / no real file metadata)
    if not _is_file_backed(doc):
        continue

    # Exclude very short / empty KOs (quality sampling requirement)
    ko_text = doc.get("ko_content_flat")
    if not isinstance(ko_text, str):
        ko_text = ""  # treat missing/non-string as empty
    if len(ko_text.strip()) < 100:
        continue

    lang_key, lang_label = _first_lang_key_label_from_json(doc)
    lang_label_by_key.setdefault(lang_key, lang_label)
    lang_totals[lang_key] += 1

    mimetype, ext = _extract_file_meta(doc)
    sampling_cat = filetype_bucket_from_mime_ext(mimetype, ext)

    if sampling_cat not in {"PDFs", "Video", "Audio", "Image"}:
        continue

    ko_id = _safe_ko_id(doc)

    # Keep the original object for later export
    koid_to_doc.setdefault(ko_id, doc)

    pid = "<json>"
    mimetype, ext = _extract_file_meta(doc)

    counts_by_lang_mime[lang_key][mimetype] += 1
    counts_by_lang_ext[lang_key][ext] += 1

    candidates_by_lang_cat[lang_label][sampling_cat].append({
        "ko_id": ko_id,
        "pid": pid,
        "language": lang_label,
        "category": sampling_cat,
        "mimetype": mimetype,
        "extension": ext,
        "license": lic_value,
    })


def lang_display(key: str) -> str:
    return lang_label_by_key.get(key, key)

print("\n==== KOs by Language × MIME type (from JSON) ====")
for lang_key, total in sorted(lang_totals.items(), key=lambda kv: (-kv[1], lang_display(kv[0]))):
    label = lang_display(lang_key)
    print(f"\nLanguage: {label}  (total KOs: {total})")
    mime_counter = counts_by_lang_mime.get(lang_key, Counter())
    if mime_counter:
        for mt, cnt in sorted(mime_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {mt}: {cnt}")
    else:
        print("  - <no resolved file MIME types>")

print("\n==== KOs by Language × File Extension (from JSON) ====")
for lang_key, total in sorted(lang_totals.items(), key=lambda kv: (-kv[1], lang_display(kv[0]))):
    label = lang_display(lang_key)
    print(f"\nLanguage: {label}  (total KOs: {total})")
    ext_counter = counts_by_lang_ext.get(lang_key, Counter())
    if ext_counter:
        for ext, cnt in sorted(ext_counter.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {ext}: {cnt}")
    else:
        print("  - <no resolved file extensions>")

# ===================== SAMPLING & CSV WRITE =====================

picked_ko_ids = set()
picked_rows = []

def _pick_some(pool: list[dict], k: int) -> list[dict]:
    """Pick up to k distinct KOs (dedup by ko_id) from pool at random."""
    fresh = [x for x in pool if x["ko_id"] not in picked_ko_ids]
    if not fresh:
        return []
    if len(fresh) < k:
        logging.warning(f"Requested {k} but only {len(fresh)} available in pool.")
        k = len(fresh)
    return random.sample(fresh, k)

# Iterate TARGETS in declared order for legible CSV
for lang_label, by_cat in TARGETS.items():
    for category, need in by_cat.items():
        if need <= 0:
            continue
        pool = candidates_by_lang_cat.get(lang_label, {}).get(category, [])
        chosen = _pick_some(pool, need)
        for row in chosen:
            picked_ko_ids.add(row["ko_id"])
            picked_rows.append(row)

per_cat = Counter(r["category"] for r in picked_rows)
per_lang = Counter(r["language"] for r in picked_rows)

print("\n==== Assessment sample audit ====")
print(f"Total picked: {len(picked_rows)} (expect 300)")
print("By category:", dict(per_cat))
print("By language (top 5):", per_lang.most_common(5))


# -------- Write QA JSON (full selected objects) --------
base_name = os.path.splitext(os.path.basename(JSON_FILE))[0]
qa_json_path = os.path.join(os.path.dirname(JSON_FILE), f"{base_name}_for_qa.json")

# preserve selection order and uniqueness
seen = set()
picked_docs_in_order = []
for r in picked_rows:
    kid = r["ko_id"]
    if kid in seen:
        continue
    seen.add(kid)
    doc = koid_to_doc.get(kid)
    if doc is not None:
        picked_docs_in_order.append(doc)

with open(qa_json_path, "w", encoding="utf-8") as jf:
    json.dump(picked_docs_in_order, jf, ensure_ascii=False, indent=2)

print(f"Wrote QA JSON: {qa_json_path}")



out_path = os.path.join(f"assessment_sample_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["ko_id", "language", "category", "mimetype", "extension", "ko_file_pid", "license", "object_json"])
    for r in picked_rows:
        obj = koid_to_doc.get(r["ko_id"], {})
        obj_min = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))  # compact
        w.writerow([
            str(r["ko_id"]),
            r["language"],
            r["category"],
            r["mimetype"],
            r["extension"],
            str(r.get("pid") or "<json>"),
            r.get("license", ""),
            obj_min,
        ])

print(f"\nWrote selection to: {out_path}")
print("Tip: export ASSESSMENT_RANDOM_SEED=42 for reproducible sampling.")

