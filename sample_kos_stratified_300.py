# sample_kos_stratified_300.py

import csv
import glob
import json
import logging
import math
import os
import random
import sys

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, UTC
from hashlib import md5
from urllib.parse import urlparse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

ALLOWED_CC = {
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA",
    "Other"
}

DISALLOWED_VIDEO_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "vimeo.com",
    "www.vimeo.com",
    "player.vimeo.com",
    "dailymotion.com",
    "www.dailymotion.com",
    "tiktok.com",
    "www.tiktok.com",
    "facebook.com",
    "www.facebook.com",
    "fb.watch",
    "instagram.com",
    "www.instagram.com",
}

DEFAULT_JSON_FILE = os.path.join("input", "final_output_10_01-2026_10-16-54.json")
MIN_TEXT_LEN = 50


# Target sample size (dynamic)
CONFIDENCE = 0.95          # 95% chance to catch at least one defect
MIN_DEFECT_RATE = 0.01     # assume at least 1% are problematic
MIN_PER_LANGUAGE = 1

def sample_size_for_detection(confidence: float, defect_rate: float) -> int:
    # n >= ln(1-c) / ln(1-d)
    return math.ceil(math.log(1 - confidence) / math.log(1 - defect_rate))

def _ko_text_len(doc: dict) -> int:
    """
    Returns length of ko_content_flat (after strip). Missing/non-string => 0.
    """
    txt = doc.get("ko_content_flat")
    if not isinstance(txt, str):
        return 0
    return len(txt.strip())

def _host_no_port(url: str) -> str:
    """
    Return hostname without port, lowercased.
    """
    try:
        netloc = (urlparse(url).netloc or "").strip().lower()
        # Strip credentials if any (rare)
        if "@" in netloc:
            netloc = netloc.split("@", 1)[1]
        # Strip port if any
        if ":" in netloc:
            netloc = netloc.split(":", 1)[0]
        return netloc
    except Exception:
        return ""

def is_disallowed_video_host_url(url: str | None) -> bool:
    """
    True if URL hostname is in, or is a subdomain of, known video hosting platforms.
    """
    if not url:
        return False
    host = _host_no_port(url)
    if not host:
        return False

    # Exact match or subdomain match
    for bad in DISALLOWED_VIDEO_HOSTS:
        bad = bad.lower()
        if host == bad or host.endswith("." + bad):
            return True
    return False

def normalise_license(value: str) -> str:
    """Canonicalise common CC licence spellings."""
    if not value:
        return ""
    v = value.upper()
    v = v.replace("CREATIVE COMMONS", "CC")
    v = v.replace("_", " ").replace("-", " ")
    v = " ".join(v.split())
    return v


def is_allowed_license(value: str) -> bool:
    """
    Accept common CC BY variants including versions like '4.0' and punctuation differences.
    Restrict to ALLOWED_CC families.
    """
    v_raw = (value or "").strip()
    if not v_raw:
        return False

    if v_raw.strip().lower() == "other":
        return True

    v = normalise_license(value)

    # Reject empty / non-CC quickly
    if not v or "CC" not in v:
        return False

    # Explicitly reject ND if present
    if " ND" in v or "NO DERIV" in v or "NODERIV" in v:
        return False

    # Check allowed families in descending specificity
    if "CC BY NC SA" in v:
        return True
    if "CC BY NC" in v:
        return True
    if "CC BY SA" in v:
        return True
    if "CC BY" in v:
        return True

    return False


def _safe_ko_id(doc: dict) -> str:
    """Stable id for sampling/dedup."""
    if doc.get("_orig_id"):
        return str(doc["_orig_id"])
    if doc.get("@id"):
        return md5(doc["@id"].encode("utf-8")).hexdigest()[:24]
    blob = f"{doc.get('title','')}-{doc.get('ko_created_at','')}"
    return md5(blob.encode("utf-8")).hexdigest()[:24]


def _extract_langs(doc: dict) -> list[str]:
    raw = doc.get("languages")
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def _first_language(doc: dict) -> str:
    langs = _extract_langs(doc)
    return langs[0] if langs else "<missing>"


def _derive_extension_from_url(url: str | None) -> str:
    if not url:
        return "<missing>"
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    return ext if ext else "<missing>"


def _extract_file_meta(doc: dict) -> tuple[str, str]:
    """
    Return (mimetype, extension) with best-effort extension inference for URL-only KOs.
    """
    mt = str(doc.get("ko_object_mimetype", "")).strip() or "<missing>"
    ext = str(doc.get("ko_object_extension", "")).strip() or ""

    if not ext:
        name = str(doc.get("ko_object_name", "")).strip()
        if name:
            ext = os.path.splitext(name)[1]

    if not ext:
        for k in ("ko_content_url", "ko_file_id", "@id"):
            u = str(doc.get(k, "")).strip()
            if u:
                ext = _derive_extension_from_url(u)
                if ext != "<missing>":
                    break

    ext = ext or "<missing>"
    return mt, ext


def filetype_bucket_from_mime_ext(mimetype: str, ext: str) -> str:
    """
    Coarse buckets for stratification.
    """
    mt = (mimetype or "").strip().lower()
    ex = (ext or "").strip().lower()

    if mt == "application/pdf" or ex == ".pdf":
        return "PDF"
    if mt.startswith("video/") or ex in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
        return "Video"
    if mt.startswith("audio/") or ex in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}:
        return "Audio"
    if mt.startswith("image/") or ex in {".png", ".jpg", ".jpeg", ".svg", ".webp", ".tif", ".tiff"}:
        return "Image"

    # If we have literally nothing to go on, label as Unknown (still included)
    if mt in {"", "<missing>"} and ex in {"", "<missing>"}:
        return "Unknown"

    # Everything else stays included, just labelled
    return "Other"


@dataclass(frozen=True)
class Stratum:
    language: str
    ftype: str


def latest_json_fallback() -> str:
    if os.path.isfile(DEFAULT_JSON_FILE):
        return DEFAULT_JSON_FILE
    matches = sorted(glob.glob(os.path.join("input", "final_output_*.json")), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(f"No JSON found. Expected {DEFAULT_JSON_FILE} or input/final_output_*.json")
    return matches[-1]


def allocate_quotas(counts: dict[Stratum, int], target_n: int) -> dict[Stratum, int]:
    """
    Proportional allocation using Largest Remainder (Hamilton method).
    Ensures sum(quotas) == target_n, unless there are fewer total items than target_n.
    """
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}

    if total <= target_n:
        # Not enough items; take everything
        return {k: v for k, v in counts.items()}

    # Ideal fractional quotas
    ideal = {k: (v / total) * target_n for k, v in counts.items()}
    base = {k: int(ideal[k]) for k in counts}
    used = sum(base.values())

    # Distribute remaining by largest fractional remainder
    remaining = target_n - used
    remainders = sorted(
        ((ideal[k] - base[k], k) for k in counts),
        key=lambda t: (t[0], t[1].language.lower(), t[1].ftype.lower()),
        reverse=True
    )

    quotas = dict(base)
    for i in range(remaining):
        _, k = remainders[i]
        quotas[k] += 1

    # Safety: cap by availability (should hold already, but keep robust)
    for k in list(quotas.keys()):
        quotas[k] = min(quotas[k], counts[k])

    # If capping caused shortfall (rare), top up from strata with spare capacity
    short = target_n - sum(quotas.values())
    if short > 0:
        spare = sorted(
            ((counts[k] - quotas[k], k) for k in counts if counts[k] > quotas[k]),
            reverse=True
        )
        idx = 0
        while short > 0 and idx < len(spare):
            cap, k = spare[idx]
            if cap <= 0:
                idx += 1
                continue
            take = min(cap, short)
            quotas[k] += take
            short -= take
            idx += 1

    return quotas

def pick_one_per_language(
    pools: dict[Stratum, list[dict]],
    picked_ids: set[str],
) -> list[dict]:
    """
    Guarantee at least one KO per language (if eligible).
    We pick from that language’s available pools, weighted by pool size,
    so we don’t accidentally always pick the rarest/oddest filetype.
    """
    # language -> list of (stratum, pool_size)
    by_lang: dict[str, list[tuple[Stratum, int]]] = defaultdict(list)
    for stratum, pool in pools.items():
        # Exclude already-picked
        fresh_count = sum(1 for x in pool if x["ko_id"] not in picked_ids)
        if fresh_count > 0:
            by_lang[stratum.language].append((stratum, fresh_count))

    chosen_rows: list[dict] = []

    for lang in sorted(by_lang.keys(), key=lambda s: s.lower()):
        options = by_lang[lang]
        if not options:
            continue

        # Weighted choice of stratum within the language by available count
        total = sum(c for _, c in options)
        r = random.randint(1, total)
        picked_stratum = None
        acc = 0
        for s, c in options:
            acc += c
            if r <= acc:
                picked_stratum = s
                break

        if picked_stratum is None:
            continue

        # Pick one KO from that stratum
        fresh = [x for x in pools[picked_stratum] if x["ko_id"] not in picked_ids]
        if not fresh:
            continue

        row = random.choice(fresh)
        picked_ids.add(row["ko_id"])
        chosen_rows.append(row)

    return chosen_rows


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else latest_json_fallback()

    seed_val = os.environ.get("ASSESSMENT_RANDOM_SEED")
    if seed_val:
        seed = int(seed_val)
    else:
        seed = int(datetime.now(UTC).timestamp())  # or secrets.randbits(32)
    random.seed(seed)
    logging.info("Sampling seed: %d", seed)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        docs = data.get("items") or data.get("data") or []
    elif isinstance(data, list):
        docs = data
    else:
        raise TypeError("Unexpected JSON structure; expected list or {items:[...]}/{data:[...]}")

    logging.info("Total KOs in JSON: %d", len(docs))

    # Build eligible pools stratified by (language, filetype)
    pools: dict[Stratum, list[dict]] = defaultdict(list)
    koid_to_doc: dict[str, dict] = {}

    docs = [d for d in docs if d.get("project_acronym") != "Groen Kennisnet"]

    eligible_total = 0
    dropped_by_license = 0
    dropped_by_video_host = 0

    for doc in docs:
        lic = str(doc.get("license", "")).strip()
        if not is_allowed_license(lic):
            dropped_by_license += 1
            continue

        lang = _first_language(doc)
        mt, ext = _extract_file_meta(doc)
        ftype = filetype_bucket_from_mime_ext(mt, ext)

        # Drop KOs whose URL points to known video hosting platforms (YouTube/Vimeo/etc.).
        # We check multiple fields because different sources populate different keys.
        ko_urls = [
            str(doc.get("ko_file_id") or "").strip(),
            str(doc.get("ko_content_url") or "").strip(),
            str(doc.get("@id") or "").strip(),
        ]

        if any(is_disallowed_video_host_url(u) for u in ko_urls if u):
            dropped_by_video_host += 1
            continue

        ko_id = _safe_ko_id(doc)
        koid_to_doc.setdefault(ko_id, doc)

        pools[Stratum(lang, ftype)].append({
            "ko_id": ko_id,
            "language": lang,
            "filetype": ftype,
            "mimetype": mt,
            "extension": ext,
            "license": lic,
        })
        eligible_total += 1

    if eligible_total == 0:
        logging.warning("No eligible KOs after licence filtering.")
        return

    target_n = min(
        eligible_total,
        sample_size_for_detection(CONFIDENCE, MIN_DEFECT_RATE)
    )

    # Ensure we can still include at least one per eligible language
    eligible_langs = sorted({s.language for s in pools.keys()}, key=lambda s: s.lower())
    target_n = max(target_n, min(len(eligible_langs) * MIN_PER_LANGUAGE, eligible_total))

    logging.info("Dropped by licence: %d", dropped_by_license)
    logging.info("Dropped by video-host URL (YouTube/Vimeo/etc.): %d", dropped_by_video_host)

    # Counts per stratum
    counts = {k: len(v) for k, v in pools.items()}

    # Languages seen in the entire JSON (before licence filter) vs eligible (after filter)
    all_langs_in_json = Counter(_first_language(d) for d in docs)
    eligible_langs = sorted({s.language for s in pools.keys()}, key=lambda s: s.lower())

    logging.info("Languages in JSON (raw): %d", len(all_langs_in_json))
    logging.info("Languages eligible (licence-filtered): %d", len(eligible_langs))

    # Warn about languages that exist in the dataset but have zero eligible items
    missing_eligible = [l for l in all_langs_in_json.keys() if l not in set(eligible_langs)]
    if missing_eligible:
        logging.warning(
            "Languages present in JSON but 0 eligible under licence filter (cannot include): %s",
            sorted(missing_eligible, key=lambda s: s.lower())[:50],
        )

    logging.info("Eligible KOs after licence filter: %d", eligible_total)
    logging.info("Distinct strata (language×filetype): %d", len(counts))
    logging.info("Target sample size: %d", min(target_n, eligible_total))

    # Sample per stratum
    picked: list[dict] = []
    picked_ids: set[str] = set()
    target_final = min(target_n, eligible_total)

    # ---- Phase A: guarantee language coverage ----
    must_cover = sorted({s.language for s in pools.keys()}, key=lambda s: s.lower())
    if len(must_cover) > target_final:
        logging.warning(
            "Eligible languages (%d) exceed target sample size (%d). "
            "Cannot include all languages.",
            len(must_cover), target_final
        )

    # Pick one per eligible language (bounded by target_final)
    one_each = pick_one_per_language(pools, picked_ids)
    if len(one_each) > target_final:
        one_each = one_each[:target_final]

    picked.extend(one_each)

    # ---- Phase B: allocate remaining proportionally by (language × filetype) ----
    remaining_n = target_final - len(picked)
    if remaining_n > 0:
        # Recompute availability after the guaranteed picks
        remaining_counts: dict[Stratum, int] = {}
        for stratum, pool in pools.items():
            remaining_counts[stratum] = sum(1 for x in pool if x["ko_id"] not in picked_ids)

        # Remove empty strata
        remaining_counts = {k: v for k, v in remaining_counts.items() if v > 0}

        remaining_quotas = allocate_quotas(remaining_counts, remaining_n)

        for stratum in sorted(remaining_quotas.keys(), key=lambda s: (s.language.lower(), s.ftype.lower())):
            need = remaining_quotas[stratum]
            if need <= 0:
                continue

            fresh = [x for x in pools[stratum] if x["ko_id"] not in picked_ids]
            if not fresh:
                continue

            if len(fresh) < need:
                need = len(fresh)

            chosen = random.sample(fresh, need)
            for row in chosen:
                picked_ids.add(row["ko_id"])
                picked.append(row)

    # ---- Phase C: final top-up if still short (due to dedupe edge cases) ----
    if len(picked) < target_final:
        remaining = []
        for pool in pools.values():
            remaining.extend([x for x in pool if x["ko_id"] not in picked_ids])

        if remaining:
            extra = random.sample(remaining, min(target_final - len(picked), len(remaining)))
            for row in extra:
                picked_ids.add(row["ko_id"])
                picked.append(row)

    # -----------------------
    # Allocation / picked summary (print ALL, not top-10)
    # -----------------------
    per_lang = Counter(r["language"] for r in picked)
    per_ftype = Counter(r["filetype"] for r in picked)

    # Pretty print: languages sorted A→Z (case-insensitive)
    logging.info("Picked by language (ALL):")
    for lang in sorted(per_lang.keys(), key=lambda s: s.lower()):
        logging.info("  %-20s %4d", lang, per_lang[lang])

    # Filetypes sorted by count desc
    logging.info("Picked by filetype (ALL):")
    for ft, cnt in sorted(per_ftype.items(), key=lambda kv: (-kv[1], kv[0].lower())):
        logging.info("  %-10s %4d", ft, cnt)

    # Optional: also show the (language × filetype) matrix for deeper audit
    per_lang_ftype = Counter((r["language"], r["filetype"]) for r in picked)
    logging.info("Picked by (language × filetype):")
    for (lang, ft), cnt in sorted(
        per_lang_ftype.items(),
        key=lambda kv: (kv[0][0].lower(), kv[0][1].lower())
    ):
        logging.info("  %-20s %-10s %4d", lang, ft, cnt)


    # -----------------------
    # TARGETS-style summary (actual picked)
    # -----------------------
    FILETYPE_ORDER = ["PDF", "Audio", "Video", "Image"]

    # Build language → filetype → count
    actual_targets: dict[str, dict[str, int]] = defaultdict(
        lambda: {ft: 0 for ft in FILETYPE_ORDER}
    )

    for r in picked:
        lang = r["language"]
        ft = r["filetype"]
        if ft in actual_targets[lang]:
            actual_targets[lang][ft] += 1

    logging.info("Actual sample distribution (TARGETS-style):")
    logging.info("ACTUAL_TARGETS = {")

    for lang in sorted(actual_targets.keys(), key=lambda s: s.lower()):
        parts = []
        for ft in FILETYPE_ORDER:
            parts.append(f'"{ft}s": {actual_targets[lang][ft]:>3}')
        logging.info(f'    "{lang}": {{{", ".join(parts)}}},')

    logging.info("}")



    picked_url_only = 0
    picked_hosted = 0

    for r in picked:
        doc = koid_to_doc.get(r["ko_id"], {})
        if doc.get("is_url_only") is True or doc.get("ko_is_hosted") is False:
            picked_url_only += 1
        if doc.get("ko_is_hosted") is True:
            picked_hosted += 1

    logging.info("Picked hosted: %d", picked_hosted)
    logging.info("Picked URL-only / non-hosted: %d", picked_url_only)



    picked_video_hosts = Counter()

    for r in picked:
        doc = koid_to_doc.get(r["ko_id"], {})
        for k in ("ko_file_id", "ko_content_url", "@id"):
            u = str(doc.get(k) or "").strip()
            if is_disallowed_video_host_url(u):
                picked_video_hosts[_host_no_port(u)] += 1

    if picked_video_hosts:
        logging.warning("Picked still contains disallowed video hosts: %s", dict(picked_video_hosts))
    else:
        logging.info("Picked contains no disallowed video-host URLs (YouTube/Vimeo/etc.).")

    # Audit
    per_lang = Counter(r["language"] for r in picked)
    per_ftype = Counter(r["filetype"] for r in picked)
    per_mime = Counter(r["mimetype"] for r in picked)

    logging.info("Picked: %d", len(picked))
    logging.info("Picked by filetype: %s", dict(per_ftype))
    logging.info("Picked by language (top 10): %s", per_lang.most_common(10))
    logging.info("Picked by mimetype (top 10): %s", per_mime.most_common(10))

    # Output files
    base = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = os.path.dirname(json_path) or "."

    qa_json_path = os.path.join(out_dir, f"{base}_sample_{len(picked)}.json")
    csv_path = os.path.join(out_dir, f"{base}_sample_{len(picked)}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.tsv")

    # Preserve selection order
    picked_docs = []
    seen = set()
    for r in picked:
        kid = r["ko_id"]
        if kid in seen:
            continue
        seen.add(kid)
        doc = koid_to_doc.get(kid)
        if doc is not None:
            picked_docs.append(doc)

    with open(qa_json_path, "w", encoding="utf-8") as f:
        json.dump(picked_docs, f, ensure_ascii=False, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ko_id", "language", "filetype", "mimetype", "extension", "license", "ko_file_id"])
        for r in picked:
            doc = koid_to_doc.get(r["ko_id"], {})
            w.writerow([
                r["ko_id"],
                r["language"],
                r["filetype"],
                r["mimetype"],
                r["extension"],
                r["license"],
                str(doc.get("ko_file_id") or doc.get("@id") or ""),
            ])

    logging.info("Wrote QA JSON: %s", qa_json_path)
    logging.info("Wrote TSV: %s", csv_path)


if __name__ == "__main__":
    main()
