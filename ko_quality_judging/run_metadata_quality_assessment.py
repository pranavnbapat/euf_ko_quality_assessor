#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "input" / "final_improved_mysql_export_18_06-2026_08-00-14.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ko_quality_judging" / "output"
DEFAULT_DOTENV = PROJECT_ROOT / ".env"

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']{1,}")
SENT_RE = re.compile(r"[.!?]+")

FIELD_WEIGHTS = {
    "title": 0.30,
    "subtitle": 0.15,
    "description": 0.35,
    "keywords": 0.20,
}

GENERIC_TERMS = {
    "overview", "introduction", "summary", "report", "document", "information",
    "guide", "project", "results", "data", "study", "research", "article",
}

AGRI_TERMS = {
    "agriculture", "agricultural", "agroecology", "agronomy", "arable", "barn",
    "biodiversity", "biosecurity", "cattle", "climate", "compost", "crop",
    "crops", "cultivation", "dairy", "disease", "farm", "farmer", "farmers",
    "farming", "fertiliser", "fertilizer", "field", "forestry", "grassland",
    "harvest", "horticulture", "irrigation", "livestock", "manure", "pasture",
    "pest", "pesticide", "plant", "plants", "poultry", "rural", "seed",
    "seeds", "soil", "soils", "sustainability", "sustainable", "tillage",
    "vineyard", "water", "weed", "welfare", "yield",
}


LLM_SYSTEM_PROMPT = """You are a strict quality evaluator for EU-FarmBook Knowledge Object metadata.

Judge all eight metadata fields, grouped as two competing metadata sets:

Original metadata:
- title
- subtitle
- description
- keywords

LLM-generated metadata:
- title_llm
- subtitle_llm
- description_llm
- keywords_llm

Use ko_content_flat as the evidence source. You may judge whether metadata is supported by this content, but do not claim real-world truth beyond the provided content.

Compare each field pair directly:
- title vs title_llm
- subtitle vs subtitle_llm
- description vs description_llm
- keywords vs keywords_llm

For each field pair, assign original_score_0_25 and llm_score_0_25 based on:
- clarity and readability
- SEO/search usefulness
- semantic completeness
- support from ko_content_flat
- domain relevance

Also assign a combined score for each metadata set on 0-100.

Return only valid JSON with this exact shape:
{
  "field_comparison": {
    "title": {"original_score_0_25": 0, "llm_score_0_25": 0, "winner": "original|llm|tie|neither", "reason": "short evidence-based reason"},
    "subtitle": {"original_score_0_25": 0, "llm_score_0_25": 0, "winner": "original|llm|tie|neither", "reason": "short evidence-based reason"},
    "description": {"original_score_0_25": 0, "llm_score_0_25": 0, "winner": "original|llm|tie|neither", "reason": "short evidence-based reason"},
    "keywords": {"original_score_0_25": 0, "llm_score_0_25": 0, "winner": "original|llm|tie|neither", "reason": "short evidence-based reason"}
  },
  "original_combined_score_0_100": 0,
  "llm_combined_score_0_100": 0,
  "overall_winner": "original|llm|tie|neither",
  "flags": ["short snake_case issue labels"],
  "judge_notes": "2-4 concise sentences explaining the comparison using evidence from the content."
}

Be conservative. Polished metadata should lose points if it is not supported by ko_content_flat. Original and LLM fields must be judged by the same standard."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample English KOs and score original vs LLM metadata fields."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-content-chars", type=int, default=200)
    parser.add_argument("--max-per-project", type=int, default=20)
    parser.add_argument("--object-extension", default=".pdf")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--model", default=None, help="OpenAI judge model. Defaults to OPENAI_MODEL from .env, then gpt-5.5.")
    parser.add_argument("--workers", type=int, default=1, help="Deprecated; LLM judging is sequential to avoid overloading model endpoints.")
    parser.add_argument("--max-content-chars", type=int, default=4000)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--judge-jsonl", type=Path, default=None)
    parser.add_argument("--no-private-fallback", action="store_true", help="Disable fallback to LLM_URL/LLM_MODEL/LLM_API_KEY when OpenAI fails.")
    return parser.parse_args()


def load_dotenv(path: Path = DEFAULT_DOTENV) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, list):
        value = "; ".join(str(item).strip() for item in value if str(item).strip())
    text = TAG_RE.sub(" ", str(value))
    text = text.replace("\u00a0", " ")
    return WS_RE.sub(" ", text).strip()


def tokens(value: Any) -> List[str]:
    return [m.group(0).lower().strip("-'") for m in TOKEN_RE.finditer(clean_text(value))]


def content_tokens(value: Any) -> List[str]:
    return [tok for tok in tokens(value) if tok not in ENGLISH_STOP_WORDS and len(tok) > 2]


def keyword_items(value: Any) -> List[str]:
    if isinstance(value, list):
        return [clean_text(item).lower() for item in value if clean_text(item)]
    text = clean_text(value)
    if not text:
        return []
    return [clean_text(part).lower() for part in re.split(r";|,|\n", text) if clean_text(part)]


def load_records(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("docs"), list):
        return [row for row in payload["docs"] if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    raise TypeError(f"Unsupported JSON root type: {type(payload).__name__}")


def record_id(record: Dict[str, Any]) -> str:
    for key in ("_orig_id", "_id", "@id", "ko_content_url"):
        value = clean_text(record.get(key))
        if value:
            return value
    return ""


def primary_language(record: Dict[str, Any]) -> str:
    languages = record.get("languages")
    if isinstance(languages, list) and languages:
        return clean_text(languages[0])
    return clean_text(languages)


def project_key(record: Dict[str, Any]) -> str:
    for key in ("project_id", "project_acronym", "project_name"):
        value = clean_text(record.get(key))
        if value:
            return value
    return "__missing_project__"


def normalise_extension(value: Any) -> str:
    ext = clean_text(value).lower()
    if ext and not ext.startswith("."):
        ext = "." + ext
    return ext


def selected_content(record: Dict[str, Any]) -> Tuple[str, str]:
    summarised = clean_text(record.get("ko_content_flat_summarised"))
    if summarised:
        return summarised, "ko_content_flat_summarised"
    return clean_text(record.get("ko_content_flat")), "ko_content_flat"


def eligible(record: Dict[str, Any], min_content_chars: int, object_extension: str) -> bool:
    if primary_language(record).lower() != "english":
        return False
    if normalise_extension(record.get("ko_object_extension")) != normalise_extension(object_extension):
        return False
    content, _ = selected_content(record)
    if content.lower() == "no content present" or len(content) <= min_content_chars:
        return False
    required = (
        "title", "title_llm", "subtitle", "subtitle_llm",
        "description", "description_llm", "keywords", "keywords_llm",
    )
    return all(clean_text(record.get(key)) for key in required)


def sample_records(
    records: List[Dict[str, Any]],
    sample_size: int,
    seed: int,
    min_content_chars: int,
    max_per_project: int,
    object_extension: str,
) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    candidates: List[Dict[str, Any]] = []
    for record in records:
        rid = record_id(record)
        if not rid or rid in seen:
            continue
        if not eligible(record, min_content_chars, object_extension):
            continue
        seen.add(rid)
        candidates.append(record)

    rng = random.Random(seed)
    rng.shuffle(candidates)

    sampled: List[Dict[str, Any]] = []
    project_counts: Dict[str, int] = {}
    for record in candidates:
        key = project_key(record)
        if project_counts.get(key, 0) >= max_per_project:
            continue
        sampled.append(record)
        project_counts[key] = project_counts.get(key, 0) + 1
        if len(sampled) == sample_size:
            return sampled

    if sampled:
        print(
            f"[WARN] Requested {sample_size} records, but only {len(sampled)} fit the "
            f"eligibility filters and max_per_project={max_per_project}. "
            f"Eligible records before project cap: {len(candidates)}."
        )
        return sampled

    raise RuntimeError(
        "No records can be sampled under the eligibility filters "
        f"and max_per_project={max_per_project}. "
        f"Eligible records before project cap: {len(candidates)}."
    )


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    a_set, b_set = set(a), set(b)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / len(a_set | b_set)


def overlap_ratio(query_terms: Iterable[str], target_terms: Iterable[str]) -> float:
    query_set, target_set = set(query_terms), set(target_terms)
    if not query_set or not target_set:
        return 0.0
    return len(query_set & target_set) / len(query_set)


def clamp(value: float, lo: int = 0, hi: int = 25) -> int:
    return max(lo, min(hi, int(round(value))))


def sentence_count(text: str) -> int:
    return len([part for part in SENT_RE.split(clean_text(text)) if part.strip()])


def score_title(text: str, content: str, sim: float) -> Tuple[int, str]:
    terms = content_tokens(text)
    content_terms = content_tokens(content)
    words = tokens(text)
    chars = len(clean_text(text))
    score = 0.0
    notes = []

    score += 5 if 6 <= len(words) <= 14 else (3 if 4 <= len(words) <= 18 else 1)
    score += 3 if 45 <= chars <= 90 else (2 if 25 <= chars <= 110 else 0)
    score += min(7, sim * 10)
    score += min(5, overlap_ratio(terms, content_terms) * 5)
    score += 3 if len(set(terms) - GENERIC_TERMS) >= 3 else 1
    score += 2 if set(terms) & AGRI_TERMS else 0

    if overlap_ratio(terms, content_terms) < 0.25:
        notes.append("weak_content_support")
    if set(terms) <= GENERIC_TERMS:
        notes.append("generic_title")
    return clamp(score), ";".join(notes)


def score_subtitle(text: str, title_text: str, content: str, sim: float) -> Tuple[int, str]:
    terms = content_tokens(text)
    title_terms = content_tokens(title_text)
    content_terms = content_tokens(content)
    words = tokens(text)
    chars = len(clean_text(text))
    score = 0.0
    notes = []

    score += 5 if 8 <= len(words) <= 20 else (3 if 5 <= len(words) <= 26 else 1)
    score += 3 if chars <= 140 else 1
    score += min(7, sim * 10)
    score += min(5, overlap_ratio(terms, content_terms) * 5)
    title_overlap = jaccard(terms, title_terms)
    score += 3 if 0.05 <= title_overlap <= 0.65 else 1
    score += 2 if set(terms) & AGRI_TERMS else 0

    if title_overlap > 0.75:
        notes.append("duplicates_title")
    if overlap_ratio(terms, content_terms) < 0.25:
        notes.append("weak_content_support")
    return clamp(score), ";".join(notes)


def score_description(text: str, content: str, sim: float) -> Tuple[int, str]:
    terms = content_tokens(text)
    content_terms = content_tokens(content)
    words = tokens(text)
    chars = len(clean_text(text))
    sents = sentence_count(text)
    score = 0.0
    notes = []

    score += 4 if 40 <= len(words) <= 80 else (3 if 25 <= len(words) <= 120 else 1)
    score += 3 if 300 <= chars <= 600 else (2 if 150 <= chars <= 900 else 0)
    score += 2 if 2 <= sents <= 4 else 1
    score += min(7, sim * 10)
    score += min(5, overlap_ratio(terms, content_terms) * 5)
    score += 2 if len(set(terms)) >= 25 else 1
    score += 2 if set(terms) & AGRI_TERMS else 0

    if overlap_ratio(terms, content_terms) < 0.20:
        notes.append("weak_content_support")
    if len(set(terms)) < 18:
        notes.append("low_information_detail")
    return clamp(score), ";".join(notes)


def score_keywords(value: Any, content: str, sim: float) -> Tuple[int, str]:
    items = keyword_items(value)
    kw_terms = content_tokens(" ".join(items))
    content_terms = content_tokens(content)
    unique_items = set(items)
    score = 0.0
    notes = []

    score += 6 if 4 <= len(items) <= 10 else (3 if 2 <= len(items) <= 15 else 0)
    score += 3 if len(unique_items) == len(items) else 1
    score += min(6, overlap_ratio(kw_terms, content_terms) * 6)
    score += min(4, sim * 7)
    score += 3 if len(set(kw_terms) - GENERIC_TERMS) >= 4 else 1
    score += min(3, len(set(kw_terms) & AGRI_TERMS))

    if len(items) != len(unique_items):
        notes.append("duplicate_keywords")
    if overlap_ratio(kw_terms, content_terms) < 0.25:
        notes.append("weak_content_support")
    return clamp(score), ";".join(notes)


def build_similarity_pairs(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Tuple[int, str, str]]]:
    meta_texts: List[str] = []
    content_texts: List[str] = []
    slots: List[Tuple[int, str, str]] = []
    for idx, record in enumerate(records):
        content, _ = selected_content(record)
        content = content_for_similarity(content)
        for variant, suffix in (("orig", ""), ("llm", "_llm")):
            for field in ("title", "subtitle", "description", "keywords"):
                key = field + suffix
                meta_texts.append(clean_text(record.get(key)))
                content_texts.append(content)
                slots.append((idx, variant, field))
    return meta_texts, content_texts, slots


def content_for_similarity(content: str) -> str:
    if len(content) <= 6000:
        return content
    return content[:4500] + " " + content[-1500:]


def tfidf_similarities(meta_texts: List[str], content_texts: List[str]) -> List[float]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(meta_texts + content_texts)
    meta = matrix[: len(meta_texts)]
    content = matrix[len(meta_texts):]
    return [float(cosine_similarity(meta[i], content[i])[0][0]) for i in range(len(meta_texts))]


def embedding_or_tfidf_similarities(meta_texts: List[str], content_texts: List[str], model_name: str, no_embeddings: bool) -> Tuple[List[float], str]:
    if no_embeddings:
        return tfidf_similarities(meta_texts, content_texts), "tfidf_cosine"
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, local_files_only=True)
        embeddings = model.encode(meta_texts + content_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        meta_emb = embeddings[: len(meta_texts)]
        content_emb = embeddings[len(meta_texts):]
        sims = np.sum(meta_emb * content_emb, axis=1)
        return [float(v) for v in sims], f"sentence_transformers:{model_name}"
    except Exception as exc:
        print(f"[WARN] Embedding model unavailable locally ({exc}); falling back to TF-IDF cosine.")
        return tfidf_similarities(meta_texts, content_texts), "tfidf_cosine"


def quality_band(score: int) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Needs improvement"
    return "Poor"


def combined_score(scores: Dict[str, int]) -> int:
    # Field scores are 0-25; scale weighted average to 0-100.
    weighted_0_25 = sum(scores[field] * FIELD_WEIGHTS[field] for field in FIELD_WEIGHTS)
    return clamp(weighted_0_25 * 4, 0, 100)


def base_row(record: Dict[str, Any]) -> Dict[str, Any]:
    content, content_source = selected_content(record)
    return {
        "_orig_id": clean_text(record.get("_orig_id")),
        "ko_content_url": clean_text(record.get("ko_content_url")),
        "language": primary_language(record),
        "project_id": clean_text(record.get("project_id")),
        "project_acronym": clean_text(record.get("project_acronym")),
        "ko_object_extension": normalise_extension(record.get("ko_object_extension")),
        "content_source": content_source,
        "title": clean_text(record.get("title")),
        "title_llm": clean_text(record.get("title_llm")),
        "subtitle": clean_text(record.get("subtitle")),
        "subtitle_llm": clean_text(record.get("subtitle_llm")),
        "description": clean_text(record.get("description")),
        "description_llm": clean_text(record.get("description_llm")),
        "keywords": clean_text(record.get("keywords")),
        "keywords_llm": clean_text(record.get("keywords_llm")),
        "selected_content_chars": len(content),
        "selected_content_excerpt": content[:1200],
        "ko_content_flat_chars": len(content),
        "ko_content_flat_excerpt": content[:1200],
    }


def automated_score_records(records: List[Dict[str, Any]], model_name: str, no_embeddings: bool) -> Tuple[pd.DataFrame, str]:
    meta_texts, content_texts, slots = build_similarity_pairs(records)
    sims, method = embedding_or_tfidf_similarities(meta_texts, content_texts, model_name, no_embeddings)
    sim_map: Dict[Tuple[int, str, str], float] = {slot: sims[i] for i, slot in enumerate(slots)}

    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        content, _ = selected_content(record)
        row = base_row(record)

        for variant, suffix, prefix in (("orig", "", "orig"), ("llm", "_llm", "llm")):
            title_text = clean_text(record.get("title" + suffix))
            subtitle_text = clean_text(record.get("subtitle" + suffix))
            desc_text = clean_text(record.get("description" + suffix))
            kw_value = record.get("keywords" + suffix)

            title_score, title_notes = score_title(title_text, content, sim_map[(idx, variant, "title")])
            subtitle_score, subtitle_notes = score_subtitle(subtitle_text, title_text, content, sim_map[(idx, variant, "subtitle")])
            desc_score, desc_notes = score_description(desc_text, content, sim_map[(idx, variant, "description")])
            kw_score, kw_notes = score_keywords(kw_value, content, sim_map[(idx, variant, "keywords")])

            field_scores = {
                "title": title_score,
                "subtitle": subtitle_score,
                "description": desc_score,
                "keywords": kw_score,
            }
            combined = combined_score(field_scores)

            row.update({
                f"{prefix}_title_score_0_25": title_score,
                f"{prefix}_subtitle_score_0_25": subtitle_score,
                f"{prefix}_description_score_0_25": desc_score,
                f"{prefix}_keywords_score_0_25": kw_score,
                f"{prefix}_combined_score_0_100": combined,
                f"{prefix}_quality_band": quality_band(combined),
                f"{prefix}_title_notes": title_notes,
                f"{prefix}_subtitle_notes": subtitle_notes,
                f"{prefix}_description_notes": desc_notes,
                f"{prefix}_keywords_notes": kw_notes,
            })

        row["automated_winner"] = (
            "tie" if abs(row["orig_combined_score_0_100"] - row["llm_combined_score_0_100"]) <= 2
            else "llm" if row["llm_combined_score_0_100"] > row["orig_combined_score_0_100"]
            else "original"
        )
        row["similarity_method"] = method
        rows.append(row)

    return pd.DataFrame(rows), method


def row_id_from_series(row: pd.Series) -> str:
    for key in ("_orig_id", "_id", "@id", "ko_content_url"):
        value = clean_text(row.get(key))
        if value:
            return value
    return str(row.name)


def truncate_content(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + "\n\n[...middle omitted for length...]\n\n" + text[-tail:]


def make_llm_prompt(row: pd.Series, content: str, max_content_chars: int) -> str:
    return json.dumps(
        {
            "ko_id": row_id_from_series(row),
            "link": clean_text(row.get("ko_content_url")),
            "title": clean_text(row.get("title")),
            "title_llm": clean_text(row.get("title_llm")),
            "subtitle": clean_text(row.get("subtitle")),
            "subtitle_llm": clean_text(row.get("subtitle_llm")),
            "description": clean_text(row.get("description")),
            "description_llm": clean_text(row.get("description_llm")),
            "keywords": clean_text(row.get("keywords")),
            "keywords_llm": clean_text(row.get("keywords_llm")),
            "content_source": clean_text(row.get("content_source")),
            "ko_content_flat": truncate_content(content, max_content_chars),
            "automated_scores": {
                "orig_title_score_0_25": row.get("orig_title_score_0_25"),
                "orig_subtitle_score_0_25": row.get("orig_subtitle_score_0_25"),
                "orig_description_score_0_25": row.get("orig_description_score_0_25"),
                "orig_keywords_score_0_25": row.get("orig_keywords_score_0_25"),
                "orig_combined_score_0_100": row.get("orig_combined_score_0_100"),
                "llm_title_score_0_25": row.get("llm_title_score_0_25"),
                "llm_subtitle_score_0_25": row.get("llm_subtitle_score_0_25"),
                "llm_description_score_0_25": row.get("llm_description_score_0_25"),
                "llm_keywords_score_0_25": row.get("llm_keywords_score_0_25"),
                "llm_combined_score_0_100": row.get("llm_combined_score_0_100"),
            },
            "instruction": "Evaluate independently. Use automated_scores only as weak context; do not copy them if your judgment differs.",
        },
        ensure_ascii=False,
    )


def parse_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_llm_judgment(judgment: Dict[str, Any]) -> Dict[str, Any]:
    def score(value: Any, hi: int) -> int:
        try:
            return max(0, min(hi, int(round(float(value)))))
        except Exception:
            return 0

    out: Dict[str, Any] = {}
    comparisons = judgment.get("field_comparison", {})
    if not isinstance(comparisons, dict):
        comparisons = {}
    allowed = {"original", "llm", "tie", "neither"}
    for field in ("title", "subtitle", "description", "keywords"):
        item = comparisons.get(field, {})
        if not isinstance(item, dict):
            item = {}
        out[f"judge_orig_{field}_score_0_25"] = score(item.get("original_score_0_25"), 25)
        out[f"judge_llm_{field}_score_0_25"] = score(item.get("llm_score_0_25"), 25)
        winner = clean_text(item.get("winner")).lower()
        out[f"judge_{field}_winner"] = winner if winner in allowed else "tie"
        out[f"judge_{field}_reason"] = clean_text(item.get("reason"))

    out["judge_orig_combined_score_0_100"] = score(judgment.get("original_combined_score_0_100"), 100)
    out["judge_llm_combined_score_0_100"] = score(judgment.get("llm_combined_score_0_100"), 100)
    winner = clean_text(judgment.get("overall_winner")).lower()
    if winner not in allowed:
        winner = (
            "tie" if abs(out["judge_orig_combined_score_0_100"] - out["judge_llm_combined_score_0_100"]) <= 2
            else "llm" if out["judge_llm_combined_score_0_100"] > out["judge_orig_combined_score_0_100"]
            else "original"
        )
    out["judge_overall_winner"] = winner
    flags = judgment.get("flags", [])
    if isinstance(flags, str):
        flags = [flags]
    out["judge_flags"] = "; ".join(clean_text(flag) for flag in flags if clean_text(flag))
    out["judge_notes"] = clean_text(judgment.get("judge_notes"))
    return out


def load_completed_judgments(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    completed: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            completed[obj["row_id"]] = obj["judgment"]
    return completed


def insufficient_quota(message: str) -> bool:
    lower = message.lower()
    return "insufficient_quota" in lower or "exceeded your current quota" in lower


def private_llm_configured() -> bool:
    return bool(os.environ.get("LLM_URL") and os.environ.get("LLM_MODEL"))


def normalise_openai_base_url(url: str | None) -> str | None:
    if not url:
        return url
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return cleaned + "/v1"


def create_judge_client(provider: str, api_key: str, request_timeout: float) -> OpenAI:
    if provider == "openai":
        return OpenAI(api_key=api_key, timeout=request_timeout, max_retries=4)
    base_url = normalise_openai_base_url(os.environ.get("LLM_URL"))
    private_key = os.environ.get("LLM_API_KEY") or "not-needed"
    return OpenAI(api_key=private_key, base_url=base_url, timeout=request_timeout, max_retries=4)


def call_judge_model(client: OpenAI, provider: str, model: str, row: pd.Series, content: str, chars: int) -> str:
    user_prompt = make_llm_prompt(row, content, chars)
    if provider == "openai":
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
        )
        return response.output_text

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
    return response.choices[0].message.content or ""


def judge_one(
    row_dict: Dict[str, Any],
    provider: str,
    api_key: str,
    model: str,
    max_content_chars: int,
    request_timeout: float,
) -> Dict[str, Any]:
    row = pd.Series(row_dict)
    client = create_judge_client(provider, api_key, request_timeout)
    content = clean_text(row.get("ko_content_flat_full")) or clean_text(row.get("selected_content_excerpt")) or clean_text(row.get("ko_content_flat_excerpt"))
    last_error: Exception | None = None
    for attempt, chars in enumerate((max_content_chars, 2500, 1500), start=1):
        chars = min(chars, max_content_chars)
        try:
            raw_text = call_judge_model(client, provider, model, row, content, chars)
            judgment = validate_llm_judgment(parse_json_response(raw_text))
            judgment["judge_provider"] = provider
            judgment["judge_model"] = model
            return judgment
        except Exception as exc:
            last_error = exc
            time.sleep(min(2 * attempt, 8))
    detail = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"
    raise RuntimeError(detail) from last_error


def run_llm_judge(df: pd.DataFrame, args: argparse.Namespace, output: Path) -> pd.DataFrame:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    private_available = private_llm_configured() and not args.no_private_fallback
    if not api_key and not private_available:
        raise RuntimeError("Set OPENAI_API_KEY or configure LLM_URL and LLM_MODEL in .env before using --llm-judge.")

    model = args.model or os.environ.get("OPENAI_MODEL") or os.environ.get("OPENAI_JUDGE_MODEL") or "gpt-5.5"
    private_model = os.environ.get("LLM_MODEL") or ""
    judge_jsonl = args.judge_jsonl or output.with_suffix(".judgments.jsonl")
    completed = load_completed_judgments(judge_jsonl)
    pending = []
    for _, row in df.iterrows():
        rid = row_id_from_series(row)
        if rid not in completed:
            pending.append((rid, row.to_dict()))

    print(f"OpenAI judge model: {model if api_key else 'not configured'}")
    print(f"Private fallback model: {private_model if private_available else 'not configured/disabled'}")
    print("LLM judge mode: sequential")
    print(f"LLM judge completed: {len(completed)}")
    print(f"LLM judge pending: {len(pending)}")

    failures: List[Tuple[str, str]] = []
    judge_jsonl.parent.mkdir(parents=True, exist_ok=True)
    openai_disabled = not bool(api_key)
    with judge_jsonl.open("a", encoding="utf-8") as handle:
        for rid, row_dict in pending:
            judgment: Dict[str, Any] | None = None
            errors: List[str] = []

            if not openai_disabled:
                try:
                    judgment = judge_one(
                        row_dict,
                        "openai",
                        api_key,
                        model,
                        args.max_content_chars,
                        args.request_timeout,
                    )
                except Exception as exc:
                    msg = str(exc)
                    errors.append(f"openai: {msg}")
                    print(f"[WARN] {rid}: OpenAI judge failed: {msg}", flush=True)
                    if insufficient_quota(msg):
                        openai_disabled = True
                        print("[INFO] OpenAI quota exhausted; switching remaining rows to private LLM fallback.", flush=True)

            if judgment is None and private_available:
                try:
                    judgment = judge_one(
                        row_dict,
                        "private",
                        os.environ.get("LLM_API_KEY") or "not-needed",
                        private_model,
                        args.max_content_chars,
                        args.request_timeout,
                    )
                except Exception as exc:
                    msg = str(exc)
                    errors.append(f"private: {msg}")
                    print(f"[ERROR] {rid}: private LLM judge failed: {msg}", flush=True)

            if judgment is None:
                failures.append((rid, " | ".join(errors)))
                continue

            completed[rid] = judgment
            handle.write(json.dumps({"row_id": rid, "judgment": judgment}, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[JUDGE {len(completed)}/{len(df)}] {rid}: "
                f"{judgment.get('judge_llm_combined_score_0_100')} "
                f"via {judgment.get('judge_provider')}",
                flush=True,
            )

    out_rows = []
    for _, row in df.iterrows():
        row_out = row.to_dict()
        rid = row_id_from_series(row)
        if rid in completed:
            row_out.update(completed[rid])
        out_rows.append(row_out)

    if failures:
        failure_path = judge_jsonl.with_suffix(".failures.jsonl")
        with failure_path.open("w", encoding="utf-8") as handle:
            for rid, msg in failures:
                handle.write(json.dumps({"row_id": rid, "error": msg}, ensure_ascii=False) + "\n")
        print(f"LLM judge failures: {len(failures)}")
        print(f"Failure diagnostics: {failure_path}")

    return pd.DataFrame(out_rows)


def write_excel(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    export_df = df.drop(columns=["ko_content_flat_full"], errors="ignore")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        export_df.to_excel(writer, index=False, sheet_name="metadata_quality")
        worksheet = writer.sheets["metadata_quality"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions


def default_output_path(sample_size: int) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return DEFAULT_OUTPUT_DIR / f"ko_metadata_quality_sample_{sample_size}_{stamp}.xlsx"


def main() -> None:
    args = parse_args()
    if args.max_per_project < 1:
        raise ValueError("--max-per-project must be at least 1")

    records = load_records(args.input)
    sampled = sample_records(
        records,
        args.sample_size,
        args.seed,
        args.min_content_chars,
        args.max_per_project,
        args.object_extension,
    )
    output = args.output or default_output_path(len(sampled))

    df, method = automated_score_records(sampled, args.embedding_model, args.no_embeddings)
    df["ko_content_flat_full"] = [selected_content(record)[0] for record in sampled]
    print(f"Input records: {len(records)}")
    print(f"Sampled records: {len(df)}")
    print(f"Primary language filter: English")
    print(f"Minimum selected content chars: > {args.min_content_chars}")
    print("Content preference: ko_content_flat_summarised, then ko_content_flat")
    print(f"Object extension filter: {normalise_extension(args.object_extension)}")
    print(f"Max KOs per project: {args.max_per_project}")
    print(f"Similarity method: {method}")

    if args.llm_judge:
        df = run_llm_judge(df, args, output)

    write_excel(df, output)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
