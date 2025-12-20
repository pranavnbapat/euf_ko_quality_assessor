# main.py

import csv
import re
import os
import gzip
import orjson
import string
import urllib.parse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple, Optional

import pandas as pd
from langdetect import detect, DetectorFactory
from rapidfuzz import fuzz, process
from nltk.corpus import stopwords

# Optional EN-only spell checks (skip for other langs)
try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker(language="en")
except Exception:
    _SPELL = None

DetectorFactory.seed = 42

INPUT_FOLDER = Path(os.environ.get("KO_INPUT_DIR", "./input")).resolve()
OUTPUT_FOLDER = Path(os.environ.get("KO_OUTPUT_DIR", "./output")).resolve()
DATA_MODEL_DIR = Path(os.environ.get("KO_DM_DIR", "./data_model_v2")).resolve()

if not DATA_MODEL_DIR.exists():
    raise FileNotFoundError(f"Controlled vocab folder not found: {DATA_MODEL_DIR}")

LT = None

# ---------- Utility: file loading ----------

def ensure_directory(p: Path) -> None:
    """
    Ensure directory exists; create it (including parents) if missing.
    Raise if the path exists but is not a directory.
    """
    if p.exists() and not p.is_dir():
        raise NotADirectoryError(f"Expected a directory: {p}")
    p.mkdir(parents=True, exist_ok=True)

def assert_readable_dir(p: Path) -> None:
    """
    Validate that 'p' exists, is a directory, and is readable.
    """
    if not p.exists():
        raise FileNotFoundError(f"Input folder not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Expected a directory: {p}")
    # Optional: check access
    if not os.access(p, os.R_OK):
        raise PermissionError(f"No read permission for: {p}")

def _latest_json_file(folder: str) -> Path:
    """Pick the most recently modified *.json or *.jsonl(.gz) file in folder."""
    candidates = []
    for pat in ("*.json", "*.jsonl", "*.ndjson", "*.json.gz", "*.jsonl.gz", "*.ndjson.gz"):
        candidates.extend(Path(folder).glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No JSON files found in {folder}")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _read_json_any(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield dicts from JSON array or (gz)NDJSON."""
    name = path.name.lower()
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    # Try JSON array first
    try:
        arr = orjson.loads(data)
        if isinstance(arr, dict):
            # Single object file
            yield arr
        else:
            for obj in arr:
                if isinstance(obj, dict):
                    yield obj
        return
    except orjson.JSONDecodeError:
        pass
    # Fallback: NDJSON
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

def _unique_outfile(base_dir: Path, stem: str = "assessments", ext: str = ".csv") -> Path:
    """
    Build a timestamped output path; if it (unexpectedly) exists, add a counter suffix.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir / f"{stem}_{ts}{ext}"
    if not candidate.exists():
        return candidate
    # extremely unlikely, but handle collisions
    i = 1
    while True:
        alt = base_dir / f"{stem}_{ts}_{i}{ext}"
        if not alt.exists():
            return alt
        i += 1

# ---------- Controlled vocabulary loader ----------

def _load_list_of_names(path: Path, name_keys=("name",)) -> set:
    """
    Load a JSON array of objects and return a set of allowed lowercased strings
    for the given name_keys (e.g., ('name', 'english_name')).
    """
    if not path.exists():
        return set()
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    allowed = set()
    for obj in data:
        for k in name_keys:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                allowed.add(v.strip().lower())
    return allowed

def _load_subcategories(path: Path) -> tuple[set, dict]:
    """
    Return (all_subcat_names_lower, subcat_name_lower -> parent_category_names_lower_set)
    """
    if not path.exists():
        return set(), {}
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    all_names = set()
    parent_map = {}
    for obj in data:
        name = (obj.get("name") or "").strip()
        if not name:
            continue
        name_l = name.lower()
        all_names.add(name_l)
        parents = set()
        for p in (obj.get("parent_category") or []):
            if isinstance(p, str) and p.strip():
                parents.add(p.strip().lower())
        parent_map[name_l] = parents
    return all_names, parent_map

def load_controlled_vocabs(base_dir: Path) -> dict:
    """
    Load all CV files we care about into sets (lowercased).
    Expected filenames:
      - data_model.category.json
      - data_model.themes.json
      - data_model.topics.json
      - data_model.subcategories.json
      - data_model.languages.json
      - data_model.locations.json
      - data_model.license.json
      - data_model.intended_purposes.json
      - (optional) data_model.project_type.json
    """
    files = {
        "category": base_dir / "data_model.category.json",
        "themes": base_dir / "data_model.themes.json",
        "topics": base_dir / "data_model.topics.json",
        "subcategories": base_dir / "data_model.subcategories.json",
        "languages": base_dir / "data_model.languages.json",
        "locations": base_dir / "data_model.locations.json",
        "license": base_dir / "data_model.license.json",
        "intended_purposes": base_dir / "data_model.intended_purposes.json",
        "project_type": base_dir / "data_model.project_type.json",
    }

    cv = {}
    cv["category"] = _load_list_of_names(files["category"], ("name",))
    cv["themes"] = _load_list_of_names(files["themes"], ("name",))
    cv["topics"] = _load_list_of_names(files["topics"], ("name",))
    cv["subcategories_all"], cv["subcat_parents"] = _load_subcategories(files["subcategories"])
    # languages: accept both 'name' and 'english_name'
    cv["languages"] = _load_list_of_names(files["languages"], ("name", "english_name"))
    cv["locations"] = _load_list_of_names(files["locations"], ("name",))
    cv["license"] = _load_list_of_names(files["license"], ("name",))
    cv["intended_purposes"] = _load_list_of_names(files["intended_purposes"], ("name",))
    cv["project_type"] = _load_list_of_names(files["project_type"], ("name",))  # optional
    return cv

# Load once at import time (fail gracefully if folder missing)
try:
    CV = load_controlled_vocabs(DATA_MODEL_DIR)
except Exception as _e:
    CV = {}


# ---------- Text normalisation & tokenisation ----------

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_OCR_SPLIT = re.compile(r"([A-Za-z])\s*-\s*\n?\s*([A-Za-z])")  # hyphen line-breaks
_NON_ASCII = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")

def norm_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.replace("\u00AD", "")  # soft hyphen
    s = _OCR_SPLIT.sub(r"\1\2", s)  # fix hyphenated breaks
    s = s.replace(" \n", " ").replace("\n", " ")
    s = _WS.sub(" ", s).strip()
    return s

def tokens(s: str, lang: str = "en") -> List[str]:
    """Simple tokeniser: lowercase, strip punctuation, keep words ≥2 chars."""
    s = s.lower()
    # remove URLs before tokenising
    s = _URL.sub(" ", s)
    tb = s.translate(str.maketrans("", "", string.punctuation))
    toks = [t for t in tb.split() if len(t) >= 2]
    return toks

_STOP_EN = set(stopwords.words("english"))

def strip_stops(toks: List[str], lang: str) -> List[str]:
    if lang.startswith("en"):
        return [t for t in toks if t not in _STOP_EN]
    return toks

def unique_ratio(toks: List[str]) -> float:
    return 0.0 if not toks else len(set(toks))/len(toks)

def compression_proxy(s: str) -> float:
    """Very cheap duplication proxy: compressibility using ORJSON + length ratio."""
    if not s:
        return 1.0
    raw = s.encode("utf-8", errors="ignore")
    # simulate 'compression' by counting repeats of 3-grams
    counts = {}
    for i in range(0, len(raw)-3, 3):
        tri = raw[i:i+3]
        counts[tri] = counts.get(tri, 0) + 1
    if not counts:
        return 1.0
    # higher average count => more duplicative; scale to 0..1
    avg = sum(counts.values())/len(counts)
    return min(1.0, 0.5 + 0.5*(1.0/avg))

# ---------- Field helpers ----------

_DOI_PAT = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

def validate_doi(s: Optional[str]) -> bool:
    s = (s or "").strip()
    return bool(_DOI_PAT.match(s))

def normalise_url(s: Optional[str]) -> Tuple[str, bool]:
    s = (s or "").strip()
    if not s:
        return "", False
    if not s.lower().startswith(("http://", "https://")):
        s = "https://" + s
    parsed = urllib.parse.urlparse(s)
    ok = bool(parsed.scheme and parsed.netloc)
    return s, ok

# ---------- Light validators for metadata ----------

_LANG_NAME_TO_CODE = {
    "english": "en", "en": "en", "en-gb": "en", "en-us": "en",
    "dutch": "nl", "nederlands": "nl", "nl": "nl",
    "french": "fr", "français": "fr", "fr": "fr",
    "german": "de", "deutsch": "de", "de": "de",
    "spanish": "es", "español": "es", "es": "es",
    "italian": "it", "italiano": "it", "it": "it",
    "portuguese": "pt", "português": "pt", "pt": "pt",
}

def normalise_lang_label(s: Optional[str]) -> str:
    if not s:
        return ""
    key = s.strip().lower().replace("_", "-")
    return _LANG_NAME_TO_CODE.get(key, key)

def token_overlap_ratio(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

_CC_LIC_PAT = re.compile(r"^(cc0|cc\s*by|cc\s*by-?sa|cc\s*by-?nc|cc\s*by-?nd|cc\s*by-?nc-?sa|cc\s*by-?nc-?nd)$", re.IGNORECASE)
def license_status(s: Optional[str]) -> str:
    """
    Return 'ok' for common Creative Commons or 'All rights reserved',
    'unknown' for anything else, 'missing' for empty.
    """
    s = (s or "").strip()
    if not s:
        return "missing"
    if s.lower().strip() in {"all rights reserved", "public domain"}:
        return "ok"
    if _CC_LIC_PAT.match(s.replace("–", "-").replace("—", "-").replace(" ", "")):
        return "ok"
    if s.lower().startswith("cc "):  # e.g. 'CC BY-NC-ND'
        return "ok"
    return "unknown"


def detect_lang_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

# ---------- Simple length & coherence helpers ----------

def length_stats(text: str) -> Dict[str, int]:
    """
    Compute basic length stats for a given text.
    Returns: {"chars": int, "tokens": int}
    """
    t = norm_text(text)
    toks = tokens(t)
    return {
        "chars": len(t),
        "tokens": len(toks),
    }


def length_flag(stats: Dict[str, int], min_tokens: int, max_tokens: int) -> bool:
    """
    True if token length is within [min_tokens, max_tokens], else False.
    """
    n = stats.get("tokens", 0)
    return min_tokens <= n <= max_tokens


def translate_to_en(text: str, src_lang: str) -> str:
    """
    Stub for future translation from src_lang to English.
    """
    if not text or not src_lang or src_lang.startswith("en"):
        return text
    # TODO: integrate with translation service
    return text


def coherence_score(
    title: str,
    subtitle: str,
    desc: str,
    content_text: str,
    keywords: List[str],
) -> int:
    """
    Crude 0–10 coherence score based on lexical overlap between key fields.
    Higher = fields talk about roughly the same thing.
    """
    kw_text = " ".join(keywords or [])
    pairs = [
        (title, desc),
        (title, content_text[:2000]),
        (desc, content_text[:2000]),
        (kw_text, title + " " + desc + " " + content_text[:2000]),
    ]

    overlaps: List[float] = []
    for a, b in pairs:
        ov = token_overlap_ratio(a, b)
        if ov > 0:
            overlaps.append(ov)

    if not overlaps:
        return 0

    avg_ov = sum(overlaps) / len(overlaps)

    # Map average overlap (0..1) → score 0..10
    if avg_ov >= 0.45:
        return 10
    if avg_ov >= 0.35:
        return 8
    if avg_ov >= 0.25:
        return 6
    if avg_ov >= 0.15:
        return 4
    if avg_ov >= 0.08:
        return 2
    return 0

# ---------- Scoring components ----------

def score_title(title: str) -> int:
    """0–10 based on token length and stopword share."""
    if not title:
        return 0
    toks = tokens(title)
    n = len(toks)
    # Domain-ish range: 4–18 tokens is ideal, otherwise penalise a bit
    if 4 <= n <= 18:
        base = 8
    else:
        base = 5
    # stopword penalty (EN only heuristic)
    sw = sum(1 for t in toks if t in _STOP_EN)
    ratio = 0 if not toks else sw / len(toks)
    if ratio > 0.5:
        base -= 3
    return max(0, min(10, base))


def score_description(desc: str) -> int:
    """0–15: length and presence of aim/method/result cues."""
    if not desc:
        return 0
    toks = tokens(desc)
    score = 0
    if len(toks) >= 40:
        score += 7
    if any(k in desc.lower() for k in ("objective", "aim", "goal", "purpose")):
        score += 3
    if any(k in desc.lower() for k in ("method", "approach", "methodology")):
        score += 3
    if any(k in desc.lower() for k in ("result", "output", "findings", "outcome")):
        score += 2
    return min(15, score)

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    return 0.0 if not A or not B else len(A & B)/len(A | B)

def score_keyword_alignment(keywords: List[str], text_blobs: List[str]) -> int:
    """0–10 using overlap across title/desc/content."""
    if not keywords:
        return 0
    kws = tokens(" ".join(keywords))
    text = tokens(" ".join(text_blobs))
    sim = jaccard(set(kws), set(text))
    if sim >= 0.5: return 10
    if sim >= 0.35: return 8
    if sim >= 0.2: return 6
    if sim >= 0.1: return 3
    return 1

def score_content_depth(content: str) -> int:
    """0–10 by token count."""
    n = len(tokens(content))
    if n >= 1500: return 10
    if n >= 800: return 8
    if n >= 500: return 6
    if n >= 250: return 4
    if n >= 100: return 2
    return 0

def score_diversity(content: str, lang: str) -> int:
    toks = strip_stops(tokens(content), lang)
    r = unique_ratio(toks)
    if r >= 0.35: return 5
    if r >= 0.28: return 4
    if r >= 0.22: return 3
    if r >= 0.18: return 2
    if r >= 0.14: return 1
    return 0

def score_duplication(content: str) -> int:
    cp = compression_proxy(content)  # lower => more duplicated; here proxy ~ (0.5..1]
    # We mapped so that 1.0 is "least duplicative"
    if cp >= 0.95: return 5
    if cp >= 0.9: return 4
    if cp >= 0.8: return 3
    if cp >= 0.7: return 2
    if cp >= 0.6: return 1
    return 0

def score_topics_themes(topics: List[str], themes: List[str], text_blobs: List[str]) -> int:
    text = tokens(" ".join(text_blobs))
    tt = tokens(" ".join(topics + themes))
    sim = jaccard(set(tt), set(text))
    if sim >= 0.35: return 6
    if sim >= 0.2: return 4
    if sim >= 0.1: return 2
    return 0

def score_project_echo(acronym: str, name: str, text_blobs: List[str]) -> int:
    text = " ".join(text_blobs).lower()
    s = 0
    if acronym and acronym.lower() in text:
        s += 3
    # conservative fuzzy match for name
    if name:
        if name.lower() in text:
            s += 3
        else:
            if fuzz.partial_ratio(name.lower(), text) >= 85:
                s += 2
    return min(6, s)

def en_spell_score(title: str, desc: str, content: str) -> Tuple[int, float]:
    """Returns (score 0–8, misspell_ratio)."""
    if _SPELL is None:
        return 8, 0.0  # can't check => don't punish
    sample = " ".join([title, desc, content[:4000]])  # limit for speed
    toks = [t for t in tokens(sample) if t.isalpha()]
    if not toks:
        return 8, 0.0
    miss = _SPELL.unknown(toks)
    ratio = len(miss) / len(toks)
    # map ratio to score
    if ratio <= 0.01: return 8, ratio
    if ratio <= 0.02: return 7, ratio
    if ratio <= 0.03: return 6, ratio
    if ratio <= 0.05: return 4, ratio
    if ratio <= 0.08: return 2, ratio
    return 0, ratio

def cleanliness_score(texts: List[str]) -> Tuple[int, Dict[str, int]]:
    s = " ".join(texts)
    issues = {
        "non_ascii": len(_NON_ASCII.findall(s)),
        "ocr_hyphen": len(_OCR_SPLIT.findall(" ".join(texts))),
        "url_count": len(_URL.findall(s)),
        "all_caps_runs": len(re.findall(r"\b[A-Z]{6,}\b", s)),
        "weird_spaces": len(re.findall(r"\S\s{3,}\S", s)),
    }
    score = 6
    # penalise progressively
    if issues["non_ascii"] > 20: score -= 2
    if issues["ocr_hyphen"] > 10: score -= 2
    if issues["url_count"] > 25: score -= 1
    if issues["all_caps_runs"] > 10: score -= 1
    if issues["weird_spaces"] > 5: score -= 1
    return max(0, score), issues

def structure_punct_score(texts: List[str]) -> int:
    s = " ".join(texts)
    sents = re.split(r"[.!?]\s+", s)
    lens = [len(tokens(x)) for x in sents if x.strip()]
    if not lens:
        return 4
    avg = sum(lens)/len(lens)
    if 8 <= avg <= 35: return 6
    if 6 <= avg <= 45: return 4
    if 4 <= avg <= 55: return 2
    return 1

# ---------- Per-KO assessor ----------

def assess_ko(ko: Dict[str, Any]) -> Dict[str, Any]:
    title = norm_text(ko.get("title"))
    subtitle = norm_text(ko.get("subtitle"))
    desc = norm_text(ko.get("description"))
    content = norm_text(ko.get("ko_content_flat_qwen3_30b_a3b_thinking_2507_q8_0"))
    keywords = [norm_text(x) for x in (ko.get("keywords") or []) if isinstance(x, str)]
    topics = [norm_text(x) for x in (ko.get("topics") or []) if isinstance(x, str)]
    themes = [norm_text(x) for x in (ko.get("themes") or []) if isinstance(x, str)]
    category = norm_text(ko.get("category"))
    subcategories = [norm_text(x) for x in (ko.get("subcategories") or []) if isinstance(x, str)]
    languages = ko.get("languages") or []
    creators = [norm_text(x) for x in (ko.get("creators") or []) if isinstance(x, str)]
    intended_purposes = [norm_text(x) for x in (ko.get("intended_purposes") or []) if isinstance(x, str)]
    locations_flat = [norm_text(x) for x in (ko.get("locations_flat") or []) if isinstance(x, str)]
    license_raw = norm_text(ko.get("license"))
    category_raw = category
    project_name = norm_text(ko.get("project_name"))
    project_acronym = norm_text(ko.get("project_acronym"))
    project_url_raw = ko.get("project_url")
    doi_raw = ko.get("project_doi")
    _id = ko.get("_orig_id") or ko.get("@id") or ""

    # --- Basic length stats for key fields ---
    title_len = length_stats(title)
    subtitle_len = length_stats(subtitle)
    desc_len = length_stats(desc)
    content_len = length_stats(content)

    # Domain-tuned "OK" ranges
    title_length_ok = length_flag(title_len, min_tokens=4, max_tokens=18)
    subtitle_length_ok = (not subtitle) or length_flag(subtitle_len, 4, 25)
    description_length_ok = length_flag(desc_len, 40, 150)      # ~40–150 tokens
    content_length_ok = length_flag(content_len, 100, 4000)     # 100–4000 tokens

    keyword_count = len(keywords)
    keyword_count_ok = 4 <= keyword_count <= 12

    # Detect languages separately for metadata vs content
    meta_text_basis = " ".join([title, desc])
    content_basis = content[:2000]

    meta_lang = detect_lang_safe(meta_text_basis) if meta_text_basis else "unknown"
    content_lang = detect_lang_safe(content_basis) if content_basis else "unknown"

    # Normalise first declared language (if any) and compare with detected metadata language
    declared_lang = normalise_lang_label(languages[0]) if languages else ""
    detected_base = normalise_lang_label(meta_lang.split("-", 1)[0])
    lang_match = (declared_lang == "" or declared_lang == detected_base)

    # For alignment with EN metadata, use an EN version of content (stub for now)
    content_for_alignment = content
    if content_lang and not content_lang.startswith("en"):
        content_for_alignment = translate_to_en(content, content_lang)

    # Fix and validate URL/DOI
    fixed_url, url_ok = normalise_url(project_url_raw)
    doi_ok = validate_doi(doi_raw)

    # --- Semantic Precision & Clarity (0–40)
    sp_title = score_title(title)
    sp_desc = score_description(desc)
    sp_kw = score_keyword_alignment(
        keywords,
        [title, desc, content_for_alignment[:2000]],
    )
    sp_url_doi = (3 if url_ok else 0) + (2 if doi_ok else 0)
    sem_total = sp_title + sp_desc + sp_kw + sp_url_doi
    sem_total = min(40, sem_total)  # clamp

    # --- Content Richness & Relevance (0–20)
    cr_depth = score_content_depth(content)
    cr_div = score_diversity(content, content_lang)
    cr_dup = score_duplication(content)
    cr_total = min(20, cr_depth + cr_div + cr_dup)

    # --- Cross-field Consistency (0–20)
    cf_kw = min(8, int(round(sp_kw * 0.8)))  # reuse alignment strength
    cf_tt = score_topics_themes(topics, themes, [title, desc, content_for_alignment[:2000]])

    # Project echo: high echo is bad, so invert it into a "good" contribution
    proj_echo_raw = score_project_echo(project_acronym, project_name, [desc, content[:4000]])
    cf_proj = max(0, 6 - proj_echo_raw)  # 0..6 where low echo → higher score

    cf_total = min(20, cf_kw + cf_tt + cf_proj)

    # --- Linguistic Integrity (0–20)
    li_spell, miss_ratio = (8, 0.0)
    if meta_lang.startswith("en"):
        li_spell, miss_ratio = en_spell_score(title, desc, content)
    li_clean, clean_issues = cleanliness_score([title, desc, content[:6000]])
    li_struct = structure_punct_score([desc, content[:4000]])
    li_total = min(20, li_spell + li_clean + li_struct)

    # --- Coherence across title/description/content/keywords (0–10) ---
    coh_score = coherence_score(
        title,
        subtitle,
        desc,
        content_for_alignment,
        keywords,
    )

    # Subtitle quality + duplicates
    subtitle_ok = False
    subtitle_duplicate_title = False
    subtitle_duplicate_description = False
    description_duplicate_title = False

    if subtitle:
        # exact-duplicate checks (after normalisation & lowercase)
        st = subtitle.strip().lower()
        tt = title.strip().lower()
        dt = desc.strip().lower()
        subtitle_duplicate_title = (st == tt) and bool(st)
        subtitle_duplicate_description = (st == dt) and bool(st)
        description_duplicate_title = (dt == tt) and bool(dt) and bool(tt)

        # If not exact dupes, check it adds some information (low overlap, reasonable length)
        if not subtitle_duplicate_title and not subtitle_duplicate_description:
            ov = token_overlap_ratio(title, subtitle)
            subtitle_ok = (ov <= 0.8) and (5 <= len(tokens(subtitle)) <= 40)
        else:
            ov = 1.0  # treat as high overlap for diagnostics
    else:
        ov = 0.0  # no subtitle

    # Creators must be at least 1
    creators_count = len([c for c in creators if c])
    creators_ok = creators_count >= 1

    # Intended purposes: can be one or many, but must not be blank
    purposes_count = len([p for p in intended_purposes if p])
    purposes_ok = purposes_count >= 1

    # Locations_flat: presence and bounded size (keep <=5)
    locations_count = len([l for l in locations_flat if l])
    locations_ok = 1 <= locations_count <= 5

    # License must not be blank; accept CC/ARR/Public Domain as 'ok', else 'unknown'
    license_eval = license_status(license_raw)
    license_ok = (license_eval == "ok") or (license_eval == "unknown")  # not blank is the hard rule
    license_missing = (license_eval == "missing")

    # Category must be present
    category_ok = bool(category_raw)

    # Subcategories: at least 1 (and keep the upper guard <=5)
    subcats = subcategories
    subcats_count = len([s for s in subcats if s])
    subcats_ok = 1 <= subcats_count <= 5

    # ---- Controlled vocabulary membership checks ----
    cv_issues = []

    # Normalise helpers
    def _lower_list(lst):
        return [x.lower() for x in lst if isinstance(x, str) and x.strip()]

    topics_l = _lower_list(topics)
    themes_l = _lower_list(themes)
    subcats_l = _lower_list(subcategories)
    languages_l = _lower_list(languages)
    locations_l = _lower_list(locations_flat)
    purposes_l = _lower_list(intended_purposes)
    category_l = category.strip().lower() if category else ""
    license_l = license_raw.strip().lower() if license_raw else ""

    # 0) Presence/cardinality hard rules
    #    - intended_purposes: >=1
    #    - license: exactly 1 (non-blank)
    #    - category: exactly 1 (non-blank)
    #    - subcategories: >=1 (upper bound already handled)
    if purposes_count < 1:
        cv_issues.append("intended_purposes missing (cardinality)")
    if not license_l:
        cv_issues.append("License missing (cardinality)")
    if not category_l:
        cv_issues.append("Category missing (cardinality)")
    if subcats_count < 1:
        cv_issues.append("At least one subcategory required (cardinality)")

    # 1) Membership: topics/themes can be length 1..N; each must be in list
    if CV.get("topics"):
        bad_topics = [t for t in topics_l if t not in CV["topics"]]
        if bad_topics:
            cv_issues.append("Unknown topics: " + ", ".join(sorted(set(bad_topics))))
    if CV.get("themes"):
        bad_themes = [t for t in themes_l if t not in CV["themes"]]
        if bad_themes:
            cv_issues.append("Unknown themes: " + ", ".join(sorted(set(bad_themes))))

    # 2) Category must be exactly one and in list
    cv_category_ok = True
    if category_l:
        if CV.get("category") and category_l not in CV["category"]:
            cv_category_ok = False
            cv_issues.append(f"Unknown category: {category}")
    else:
        cv_category_ok = False

    # 3) Subcategories: each must be in list; also parent must include category (if both present)
    bad_subcats = []
    parent_mismatch = []
    if CV.get("subcategories_all"):
        for s in subcats_l:
            if s not in CV["subcategories_all"]:
                bad_subcats.append(s)
            else:
                # parent check only if category is present and known
                if category_l and CV.get("subcat_parents"):
                    parents = CV["subcat_parents"].get(s, set())
                    if parents and (category_l not in parents):
                        parent_mismatch.append((s, list(parents)[:3]))
    if bad_subcats:
        cv_issues.append("Unknown subcategories: " + ", ".join(sorted(set(bad_subcats))))
    if parent_mismatch:
        msgs = [f"{s}→allowed:{'|'.join(p)}" for s,p in parent_mismatch]
        cv_issues.append("Subcategory parent/category mismatch: " + "; ".join(msgs))

    # 4) Languages
    #    Validate membership against CV['languages'] if available
    bad_langs = []
    if CV.get("languages") and languages_l:
        for lg in languages_l:
            if lg not in CV["languages"]:
                bad_langs.append(lg)
        if bad_langs:
            cv_issues.append("Unknown languages: " + ", ".join(sorted(set(bad_langs))))

    # 5) Locations_flat: membership against CV['locations'] if available
    bad_locs = []
    if CV.get("locations") and locations_l:
        for loc in locations_l:
            if loc not in CV["locations"]:
                bad_locs.append(loc)
        if bad_locs:
            cv_issues.append("Unknown locations: " + ", ".join(sorted(set(bad_locs))))

    # 6) License: must be exactly one and in list (hard rule)
    cv_license_ok = True
    if license_l:
        if CV.get("license") and license_l not in CV["license"]:
            cv_license_ok = False
            cv_issues.append(f"Unknown license: {license_raw}")
    else:
        cv_license_ok = False

    # 7) Intended purposes: each must be in list (>=1 already checked)
    bad_purposes = []
    if CV.get("intended_purposes") and purposes_l:
        for p in purposes_l:
            if p not in CV["intended_purposes"]:
                bad_purposes.append(p)
        if bad_purposes:
            cv_issues.append("Unknown intended_purposes: " + ", ".join(sorted(set(bad_purposes))))


    total = sem_total + cr_total + cf_total + li_total

    # Flags and notes
    empty_fields = [k for k in ["title", "description", "ko_content_flat_qwen3_30b_a3b_thinking_2507_q8_0"] if not norm_text(ko.get(k))]
    notes = []
    if not url_ok and project_url_raw:
        notes.append("Invalid project_url")
    if not doi_ok and doi_raw:
        notes.append("Invalid DOI format")
    if meta_lang.startswith("en") and miss_ratio > 0.05:
        notes.append(f"High misspelling ratio: {miss_ratio:.2%}")
    if len(tokens(content)) < 250:
        notes.append("Low content depth (<250 tokens)")
    if clean_issues["ocr_hyphen"] > 10:
        notes.append("OCR hyphenation artefacts detected")

    if not subtitle:
        notes.append("Subtitle missing")
    if subtitle_duplicate_title:
        notes.append("Subtitle duplicates title")
    if subtitle_duplicate_description:
        notes.append("Subtitle duplicates description")
    if description_duplicate_title:
        notes.append("Description duplicates title")

    if not creators_ok:
        notes.append("Creators missing")
    if not lang_match and declared_lang:
        notes.append(f"Language mismatch (declared={declared_lang}, detected={detected_base})")
    if not purposes_ok:
        notes.append("intended_purposes missing")
    if not locations_ok:
        notes.append("locations_flat missing or >5 items")

    if license_missing:
        notes.append("License missing")

    if (license_eval == "unknown") and not license_missing and license_raw:
        notes.append(f"Unrecognised license: {license_raw}")

    if not category_ok:
        notes.append("Category missing")
    if not subcats_ok:
        if subcats_count == 0:
            notes.append("At least one subcategory required")
        else:
            notes.append("Too many subcategories (>5)")

    # Controlled vocabulary inconsistencies
    if cv_issues:
        notes.append("CV inconsistencies: " + " | ".join(cv_issues))

    return {
        "_orig_id": _id,
        "title": title[:300],
        "lang_detected": meta_lang,
        "lang_meta_detected": meta_lang,
        "lang_content_detected": content_lang,

        # Subscores for transparency
        "Semantic_Precision": sem_total,
        "Content_Richness": cr_total,
        "Cross_Field_Consistency": cf_total,
        "Linguistic_Integrity": li_total,
        "Coherence_Score": coh_score,
        "Total_Score": total,

        # Diagnostics
        "sp_title": sp_title,
        "sp_desc": sp_desc,
        "sp_keyword_anchoring": sp_kw,
        "cr_depth": cr_depth,
        "cr_diversity": cr_div,
        "cr_duplication": cr_dup,
        "cf_topics_themes": cf_tt,
        "cf_project_echo": cf_proj,                 # inverted (0..6 good)
        "cf_project_echo_raw": proj_echo_raw,       # original 0..6 leakage
        "li_spell_score": li_spell,
        "li_misspell_ratio": round(miss_ratio, 4),
        "url_valid": url_ok,
        "project_url_fixed": fixed_url,
        "doi_valid": doi_ok,
        "empty_fields": ";".join(empty_fields),
        "notes": "; ".join(notes),
        "category": category,
        "subcategories": orjson.dumps(subcategories).decode("utf-8"),
        "keywords": orjson.dumps(keywords).decode("utf-8"),
        "topics": orjson.dumps(topics).decode("utf-8"),
        "themes": orjson.dumps(themes).decode("utf-8"),
        "subtitle_ok": subtitle_ok,
        "subtitle_overlap_with_title": round(ov, 3),
        "subtitle_duplicate_title": subtitle_duplicate_title,
        "subtitle_duplicate_description": subtitle_duplicate_description,
        "description_duplicate_title": description_duplicate_title,
        "creators_count": creators_count,
        "purposes_count": purposes_count,
        "locations_count": locations_count,
        "license_status": license_eval,
        "category_ok": category_ok,
        "subcategories_count": subcats_count,
        "lang_declared": declared_lang,
        "lang_match": lang_match,

        # Length + keyword diagnostics
        "title_len_chars": title_len["chars"],
        "title_len_tokens": title_len["tokens"],
        "title_length_ok": title_length_ok,
        "subtitle_len_chars": subtitle_len["chars"],
        "subtitle_len_tokens": subtitle_len["tokens"],
        "subtitle_length_ok": subtitle_length_ok,
        "description_len_chars": desc_len["chars"],
        "description_len_tokens": desc_len["tokens"],
        "description_length_ok": description_length_ok,
        "content_len_chars": content_len["chars"],
        "content_len_tokens": content_len["tokens"],
        "content_length_ok": content_length_ok,
        "keyword_count": keyword_count,
        "keyword_count_ok": keyword_count_ok,

        # CV diagnostics
        "cv_category_ok": cv_category_ok,
        "cv_license_ok": cv_license_ok,
        "cv_topics_unknown_count": len([t for t in topics_l if CV.get('topics') and t not in CV['topics']]),
        "cv_themes_unknown_count": len([t for t in themes_l if CV.get('themes') and t not in CV['themes']]),
        "cv_subcategories_unknown_count": len(
            [s for s in subcats_l if CV.get('subcategories_all') and s not in CV['subcategories_all']]),
        "cv_subcat_parent_mismatch_count": len(parent_mismatch),
        "cv_languages_unknown_count": len(bad_langs),
        "cv_locations_unknown_count": len(bad_locs),
        "cv_intended_purposes_unknown_count": len(bad_purposes),
    }

# ---------- Driver ----------

def main() -> None:
    """
    Read the latest JSON/NDJSON from INPUT_FOLDER and write assessments CSV to OUTPUT_FOLDER.
    Creates OUTPUT_FOLDER if it doesn't exist. Performs defensive checks on both folders.
    """
    # 1) Validate I/O folders
    assert_readable_dir(INPUT_FOLDER)
    ensure_directory(OUTPUT_FOLDER)

    # 2) Pick latest input file
    latest = _latest_json_file(str(INPUT_FOLDER))
    print(f"[INFO] Using latest file: {latest}")

    # 3) Assess
    rows = []
    count = 0
    errors = 0
    for ko in _read_json_any(latest):
        try:
            rows.append(assess_ko(ko))
        except Exception as e:
            errors += 1
            rid = ko.get("_orig_id") or ko.get("@id") or f"row_{count}"
            rows.append({
                "_orig_id": rid,
                "title": ko.get("title", ""),
                "lang_detected": "unknown",
                "Semantic_Precision": 0,
                "Content_Richness": 0,
                "Cross_Field_Consistency": 0,
                "Linguistic_Integrity": 0,
                "Total_Score": 0,
                "notes": f"ERROR: {type(e).__name__}: {e}"
            })
        count += 1
        if count % 1000 == 0:
            print(f"[INFO] Processed {count} KOs...")

    if count == 0:
        raise RuntimeError(f"No valid JSON objects found in input file: {latest}")

    # 4) Write output (always to OUTPUT_FOLDER)
    df = pd.DataFrame(rows)
    out_path = _unique_outfile(OUTPUT_FOLDER, stem="assessments", ext=".tsv")
    df.to_csv(
        out_path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        lineterminator="\n",
        encoding="utf-8",
    )
    print(f"[OK] Wrote {len(df)} rows -> {out_path}")
    if errors:
        print(f"[WARN] {errors} item(s) had exceptions; details recorded in 'notes' column.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Non-zero exit to help CI or shell scripts detect failure
        print(f"[FATAL] {type(exc).__name__}: {exc}")
        raise


