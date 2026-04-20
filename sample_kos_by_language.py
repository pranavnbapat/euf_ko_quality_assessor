#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"

EU24_LANGUAGES = [
    "Bulgarian",
    "Croatian",
    "Czech",
    "Danish",
    "Dutch",
    "English",
    "Estonian",
    "Finnish",
    "French",
    "German",
    "Greek",
    "Hungarian",
    "Irish",
    "Italian",
    "Latvian",
    "Lithuanian",
    "Maltese",
    "Polish",
    "Portuguese",
    "Romanian",
    "Slovak",
    "Slovenian",
    "Spanish",
    "Swedish",
]

ALLOWED_CC = {
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA",
    "Other",
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

LANG_ALIASES = {
    "bg": "Bulgarian",
    "bulgarian": "Bulgarian",
    "hr": "Croatian",
    "croatian": "Croatian",
    "cs": "Czech",
    "czech": "Czech",
    "da": "Danish",
    "danish": "Danish",
    "nl": "Dutch",
    "dutch": "Dutch",
    "en": "English",
    "english": "English",
    "et": "Estonian",
    "estonian": "Estonian",
    "fi": "Finnish",
    "finnish": "Finnish",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "el": "Greek",
    "greek": "Greek",
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "ga": "Irish",
    "irish": "Irish",
    "it": "Italian",
    "italian": "Italian",
    "lv": "Latvian",
    "latvian": "Latvian",
    "lt": "Lithuanian",
    "lithuanian": "Lithuanian",
    "mt": "Maltese",
    "maltese": "Maltese",
    "pl": "Polish",
    "polish": "Polish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ro": "Romanian",
    "romanian": "Romanian",
    "sk": "Slovak",
    "slovak": "Slovak",
    "sl": "Slovenian",
    "slovenian": "Slovenian",
    "es": "Spanish",
    "spanish": "Spanish",
    "sv": "Swedish",
    "swedish": "Swedish",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample up to N KOs per EU language from the latest input JSON."
    )
    p.add_argument(
        "per_language",
        type=int,
        help="Maximum number of records to take per EU language.",
    )
    p.add_argument(
        "--input",
        type=str,
        default=None,
        help="Explicit input JSON path. Defaults to latest JSON under input/.",
    )
    p.add_argument(
        "--min-content-chars",
        type=int,
        default=500,
        help="Minimum ko_content_flat length required.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    p.add_argument(
        "--allowed-license-only",
        action="store_true",
        help="Keep only KOs with allowed CC-style licenses (same family as sample_kos_stratified_300.py).",
    )
    p.add_argument(
        "--exclude-video-hosts",
        action="store_true",
        help="Exclude KOs whose URLs point to disallowed video-hosting platforms.",
    )
    return p.parse_args()


def latest_input_file() -> Path:
    files = list(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files found in {INPUT_DIR}")
    return max(files, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_docs(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        docs = payload.get("docs")
        if isinstance(docs, list):
            return [row for row in docs if isinstance(row, dict)]
        return [payload]
    raise TypeError(f"Unsupported JSON top-level type: {type(payload).__name__}")


def normalize_language(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return LANG_ALIASES.get(text.lower())


def primary_language(doc: Dict[str, Any]) -> str | None:
    langs = doc.get("languages")
    if isinstance(langs, list) and langs:
        lang = normalize_language(langs[0])
        if lang:
            return lang
    return normalize_language(doc.get("ko_resource_language"))


def _host_no_port(url: str) -> str:
    try:
        netloc = (urlparse(url).netloc or "").strip().lower()
        if "@" in netloc:
            netloc = netloc.split("@", 1)[1]
        if ":" in netloc:
            netloc = netloc.split(":", 1)[0]
        return netloc
    except Exception:
        return ""


def is_disallowed_video_host_url(url: str | None) -> bool:
    if not url:
        return False
    host = _host_no_port(url)
    if not host:
        return False
    for bad in DISALLOWED_VIDEO_HOSTS:
        if host == bad or host.endswith("." + bad):
            return True
    return False


def normalise_license(value: str) -> str:
    if not value:
        return ""
    v = value.upper()
    v = v.replace("CREATIVE COMMONS", "CC")
    v = v.replace("_", " ").replace("-", " ")
    return " ".join(v.split())


def is_allowed_license(value: str) -> bool:
    v_raw = (value or "").strip()
    if not v_raw:
        return False
    if v_raw.lower() == "other":
        return True
    v = normalise_license(value)
    if not v or "CC" not in v:
        return False
    if " ND" in v or "NO DERIV" in v or "NODERIV" in v:
        return False
    return any(
        token in v
        for token in ("CC BY NC SA", "CC BY NC", "CC BY SA", "CC BY")
    )


def eligible_docs(
    docs: Iterable[Dict[str, Any]],
    min_content_chars: int,
    allowed_license_only: bool = False,
    exclude_video_hosts: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {lang: [] for lang in EU24_LANGUAGES}
    for doc in docs:
        lang = primary_language(doc)
        if lang not in grouped:
            continue
        if allowed_license_only and not is_allowed_license(str(doc.get("license") or "").strip()):
            continue
        if exclude_video_hosts:
            urls = [
                str(doc.get("ko_file_id") or "").strip(),
                str(doc.get("ko_content_url") or "").strip(),
                str(doc.get("@id") or "").strip(),
            ]
            if any(is_disallowed_video_host_url(url) for url in urls if url):
                continue
        content = doc.get("ko_content_flat")
        if not isinstance(content, str) or len(content.strip()) < min_content_chars:
            continue
        grouped[lang].append(doc)
    return grouped


def build_output_path(input_path: Path, per_language: int) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_{per_language}_24.json")


def rebuild_payload(original: Any, sampled_docs: List[Dict[str, Any]]) -> Any:
    if isinstance(original, list):
        return sampled_docs
    if isinstance(original, dict) and isinstance(original.get("docs"), list):
        out = dict(original)
        out["docs"] = sampled_docs
        return out
    if len(sampled_docs) == 1:
        return sampled_docs[0]
    return sampled_docs


def main() -> int:
    args = parse_args()
    input_path = Path(args.input) if args.input else latest_input_file()
    payload = load_json(input_path)
    docs = get_docs(payload)
    grouped = eligible_docs(
        docs,
        args.min_content_chars,
        allowed_license_only=args.allowed_license_only,
        exclude_video_hosts=args.exclude_video_hosts,
    )

    rng = random.Random(args.seed)
    sampled_docs: List[Dict[str, Any]] = []

    print(f"Input file: {input_path}")
    print(f"Minimum ko_content_flat length: {args.min_content_chars}")
    print(f"Per-language target: {args.per_language}")
    print(f"Allowed-license-only: {args.allowed_license_only}")
    print(f"Exclude video hosts: {args.exclude_video_hosts}")
    print("")

    for lang in EU24_LANGUAGES:
        pool = grouped[lang]
        take = min(args.per_language, len(pool))
        if take > 0:
            picked = rng.sample(pool, take)
            sampled_docs.extend(picked)
        print(f"{lang:12} available={len(pool):4d} selected={take:4d}")

    output_path = build_output_path(input_path, args.per_language)
    output_payload = rebuild_payload(payload, sampled_docs)
    dump_json(output_path, output_payload)

    print("")
    print(f"Total selected: {len(sampled_docs)}")
    print(f"Wrote sample to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
