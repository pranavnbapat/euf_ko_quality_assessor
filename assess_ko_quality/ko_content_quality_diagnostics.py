# assess_ko_quality/ko_content_quality_diagnostics.py

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Time
# ----------------------------

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------
# JSON shape
# ----------------------------

def ensure_list(data: Any) -> List[Dict[str, Any]]:
    """
    Extract list of KO objects from JSON data.
    Handles:
    - Old format: direct list of KOs
    - New format: {meta, stats, docs: [...]} where docs contains KOs
    - Single object: wrap in list
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        # New format: check for 'docs' field containing the KOs
        if "docs" in data and isinstance(data["docs"], list):
            return [x for x in data["docs"] if isinstance(x, dict)]
        # Single object (legacy single KO format)
        return [data]
    raise ValueError("Input JSON must be a list, an object, or an object with 'docs' field")


# ----------------------------
# Token counting (best-effort)
# ----------------------------

_WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def approx_token_count(text: str) -> int:
    return len(_WORD_RE.findall(text)) if text else 0

def try_tiktoken_count(text: str) -> Optional[int]:
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return None

def count_tokens(text: str) -> int:
    tt = try_tiktoken_count(text)
    return tt if tt is not None else approx_token_count(text)


# ----------------------------
# spaCy setup (auto-install + auto-download model)
# ----------------------------

def _run_cmd(cmd: List[str]) -> None:
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed ({e.returncode}): {' '.join(cmd)}") from e

def load_spacy_model(model_name: str = "en_core_web_lg"):
    try:
        import spacy  # type: ignore
    except ModuleNotFoundError:
        print("[INFO] spaCy not found; installing via pip…")
        _run_cmd([sys.executable, "-m", "pip", "install", "spacy"])
        import spacy  # type: ignore

    try:
        nlp = spacy.load(model_name, disable=["lemmatizer"])
        return nlp
    except Exception:
        print(f"[INFO] spaCy model '{model_name}' not available; downloading…")
        _run_cmd([sys.executable, "-m", "spacy", "download", model_name])
        try:
            return spacy.load(model_name, disable=["lemmatizer"])
        except Exception:
            fallback = "en_core_web_sm"
            print(f"[WARN] Failed to load '{model_name}'. Falling back to '{fallback}'…")
            try:
                return spacy.load(fallback, disable=["lemmatizer"])
            except Exception:
                print(f"[INFO] spaCy model '{fallback}' not available; downloading…")
                _run_cmd([sys.executable, "-m", "spacy", "download", fallback])
                return spacy.load(fallback, disable=["lemmatizer"])


# ----------------------------
# SentenceTransformer (optional)
# ----------------------------

def load_sentence_transformer(model_name: str, device: str = "auto"):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers not installed. Install with:\n"
            "  pip install sentence-transformers torch\n"
        ) from e

    import torch  # type: ignore
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


# ----------------------------
# Readability (simple Flesch)
# ----------------------------

VOWELS_RE = re.compile(r"[aeiouy]+", re.I)

def count_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    sylls = len(VOWELS_RE.findall(w))
    if w.endswith("e") and sylls > 1:
        sylls -= 1
    return max(1, sylls)

def flesch_reading_ease(text: str) -> Optional[float]:
    sentences = re.split(r"[.!?]+\s*", text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return None
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)
    if not words:
        return None
    syllables = sum(count_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syllables / len(words)
    return float(206.835 - 1.015 * wps - 84.6 * spw)

def spacy_stopword_ratio(doc, max_tokens: int = 500) -> Optional[float]:
    # Uses spaCy's own stopword flags; language-sensitive to the loaded model.
    # With en_core_web_*, a non-English doc will usually have a low stopword hit-rate.
    toks = [t for t in doc if not (t.is_space or t.is_punct)]
    if not toks:
        return None
    toks = toks[:max_tokens]
    return sum(1 for t in toks if t.is_stop) / len(toks)


# ----------------------------
# Entities + keywords (no hardcoded stopwords)
# ----------------------------

def normalise_entity(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"'`]", "", s)
    return s

def extract_entities(doc) -> Counter:
    # Counts normalised mentions (so repeated mentions increase the count)
    ents = [normalise_entity(e.text) for e in getattr(doc, "ents", []) if e.text.strip()]
    ents = [e for e in ents if len(e) >= 3]
    return Counter(ents)


def keyword_set_spacy(doc, topk: int = 40) -> List[str]:
    """
    Data-driven keyword proxy:
    - spaCy tokenisation
    - token.is_stop (language model stopwords)
    - no manual stopword list
    """
    words: List[str] = []
    for t in doc:
        if t.is_space or t.is_punct:
            continue
        if t.like_num:
            continue
        if t.is_stop:
            continue
        txt = (t.lemma_ or t.text).strip().lower()
        if len(txt) < 3:
            continue
        words.append(txt)

    counts = Counter(words)
    return [w for w, _ in counts.most_common(topk)]


# ----------------------------
# Scoring helpers
# ----------------------------

@dataclass
class QualityResult:
    metrics: Dict[str, Any]
    score_0_100: float
    grade: str
    flags: List[str]

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def grade_from_score(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    return "D"


# ----------------------------
# Embedding coherence metrics (optional)
# ----------------------------

def sentence_semantic_coherence(
    doc,
    st_model,
    max_sentences: int = 40,
    min_sentence_chars: int = 20,
) -> Dict[str, Any]:
    sent_texts = [s.text.strip() for s in doc.sents]
    sent_texts = [s for s in sent_texts if len(s) >= min_sentence_chars]

    if len(sent_texts) < 2:
        return {
            "embedding_sentence_count_used": int(len(sent_texts)),
            "mean_adjacent_cosine_similarity": None,
            "mean_centroid_cosine_similarity": None,
            "first_last_block_cosine_similarity": None,
        }

    if len(sent_texts) > max_sentences:
        idxs = np.linspace(0, len(sent_texts) - 1, num=max_sentences, dtype=int)
        sent_texts = [sent_texts[i] for i in idxs]

    emb = st_model.encode(
        sent_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # dot product == cosine
    )

    # Adjacent similarity (local repetition)
    adj = np.sum(emb[:-1] * emb[1:], axis=1)
    mean_adj = float(np.mean(adj)) if adj.size else None

    # Centroid similarity (global redundancy / cohesion)
    centroid = np.mean(emb, axis=0, keepdims=True)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    cent_sims = np.sum(emb * centroid, axis=1)
    mean_cent = float(np.mean(cent_sims)) if cent_sims.size else None

    # Drift: first 20% vs last 20%
    n = emb.shape[0]
    k = max(1, int(round(0.2 * n)))
    first = np.mean(emb[:k], axis=0)
    last = np.mean(emb[-k:], axis=0)
    drift = float(np.dot(first, last) / ((np.linalg.norm(first) * np.linalg.norm(last)) + 1e-12))

    return {
        "embedding_sentence_count_used": int(len(sent_texts)),
        "mean_adjacent_cosine_similarity": mean_adj,
        "mean_centroid_cosine_similarity": mean_cent,
        "first_last_block_cosine_similarity": drift,
    }


# ----------------------------
# Single-field quality scoring (no reference)
# ----------------------------

def compute_field_quality(text: str, nlp, st_model=None, spacy_max_chars: int = 200_000) -> QualityResult:
    flags: List[str] = []
    text = text or ""

    # tok = count_tokens(text)

    # Truncate once so ALL metrics refer to the same text slice
    truncated = text[:spacy_max_chars]
    was_truncated = len(text) > len(truncated)

    tok = count_tokens(truncated)

    if tok == 0:
        return QualityResult(
            metrics={
                "tokens": 0,
                "noise_ratio_non_alnum": None,
                "flesch_reading_ease": None,
                "ttr": None,
                "unigram_entropy": None,
                "bigram_repetition_ratio": None,
                "entity_unique": 0,
                "entity_mentions": 0,
                "entity_density_per_100_tokens": None,
                "keyword_top40_unique": 0,
                "field_quality_score_0_100": 0.0,
                "field_quality_grade": "D",
                "field_quality_flags": ["empty_text"],
                "embedding_sentence_count_used": None,
                "mean_adjacent_cosine_similarity": None,
                "mean_centroid_cosine_similarity": None,
                "first_last_block_cosine_similarity": None,
            },
            score_0_100=0.0,
            grade="D",
            flags=["empty_text"],
        )

    if tok < 80:
        flags.append("very_short_text")
    elif tok < 250:
        flags.append("short_text")

    # Noise ratio
    non_ws = [ch for ch in truncated if not ch.isspace()]
    noise_ratio = (sum(1 for ch in non_ws if not ch.isalnum()) / len(non_ws)) if non_ws else None
    if isinstance(noise_ratio, float) and noise_ratio >= 0.18:
        flags.append("high_noise_ratio")

    # spaCy doc (truncate)
    # doc = nlp(text[:spacy_max_chars])
    if was_truncated:
        flags.append("truncated_for_spacy")
    doc = nlp(truncated)

    # Optional embedding coherence
    coh: Dict[str, Any] = {
        "embedding_sentence_count_used": None,
        "mean_adjacent_cosine_similarity": None,
        "mean_centroid_cosine_similarity": None,
        "first_last_block_cosine_similarity": None,
    }
    if st_model is not None:
        try:
            coh = sentence_semantic_coherence(doc, st_model)
        except Exception:
            flags.append("embedding_coherence_failed")

    # Entities
    ents = extract_entities(doc)
    entity_unique = int(len(ents))
    entity_mentions = int(sum(ents.values()))
    # Use mentions for density; keep unique as a separate metric
    ent_density = float((entity_mentions / tok) * 100.0) if tok else None

    if entity_mentions == 0 and tok > 300:
        flags.append("no_entities_found")

    # Keywords
    kws = keyword_set_spacy(doc, topk=40)
    kw_unique = int(len(set(kws)))
    if kw_unique < 10 and tok > 300:
        flags.append("low_keyword_diversity")

    # Readability
    # flesch = flesch_reading_ease(text)
    stop_ratio = spacy_stopword_ratio(doc)
    if stop_ratio is not None and stop_ratio >= 0.12:
        flesch = flesch_reading_ease(truncated)
        if isinstance(flesch, float) and flesch < 5:
            flags.append("very_hard_to_read")
    else:
        flesch = None
        flags.append("readability_skipped_non_english")

    # Lexical richness + repetition
    # words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", truncated.lower())
    n = len(words)
    if n > 0:
        counts = Counter(words)
        ttr = float(len(counts) / n)

        # Shannon entropy
        total = sum(counts.values())
        h = 0.0
        for c in counts.values():
            p = c / total
            h -= p * np.log2(p)
        unigram_entropy = float(h)

        # Bigram repetition ratio
        if n >= 2:
            bigrams = list(zip(words[:-1], words[1:]))
            bc = Counter(bigrams)
            repeated = sum(1 for v in bc.values() if v >= 2)
            bigram_rep = float(repeated / len(bc)) if bc else 0.0
        else:
            bigram_rep = 0.0
    else:
        ttr = None
        unigram_entropy = None
        bigram_rep = None

    if isinstance(bigram_rep, float) and bigram_rep >= 0.25 and tok > 400:
        flags.append("high_repetition")

    # ---- scoring (0..100)
    # Length is a sufficiency check (after ~400 tokens, stop rewarding more length)
    len_score = clamp01(min(tok, 600) / 600)

    noise_score = 0.5 if noise_ratio is None else clamp01(1.0 - (noise_ratio / 0.25))
    read_score = clamp01((flesch + 20.0) / 100.0) if isinstance(flesch, float) else 0.5
    ttr_score = 0.5 if ttr is None else clamp01((ttr - 0.15) / (0.55 - 0.15))
    entropy_score = 0.5 if unigram_entropy is None else clamp01((unigram_entropy - 5.0) / (9.0 - 5.0))
    entdens_score = 0.5 if ent_density is None else clamp01(ent_density / 8.0)
    kw_score = clamp01((kw_unique - 5) / (30 - 5))

    rep_penalty = 0.0
    if isinstance(bigram_rep, float):
        rep_penalty = 0.15 * clamp01((bigram_rep - 0.15) / (0.35 - 0.15))

    # Optional: don’t reward boring redundancy if embeddings enabled
    embed_bonus = 0.0
    embed_penalty = 0.0
    if st_model is not None:
        drift = coh.get("first_last_block_cosine_similarity")
        cent = coh.get("mean_centroid_cosine_similarity")
        adj = coh.get("mean_adjacent_cosine_similarity")

        # Drift reward: stable topic is good
        if isinstance(drift, float):
            embed_bonus += 0.06 * clamp01((drift - 0.40) / (0.85 - 0.40))

        # Centroid: penalise extreme high (too repetitive) and extreme low (incoherent)
        if isinstance(cent, float):
            if cent > 0.90:
                embed_penalty += 0.06 * clamp01((cent - 0.90) / (0.97 - 0.90))
            if cent < 0.45:
                embed_penalty += 0.04 * clamp01((0.45 - cent) / (0.45 - 0.30))

        # Adjacent: high adjacent similarity suggests local repetition
        if isinstance(adj, float) and adj > 0.90:
            embed_penalty += 0.05 * clamp01((adj - 0.90) / (0.97 - 0.90))

    score01 = (
        0.08 * len_score +
        0.18 * noise_score +
        0.16 * read_score +
        0.16 * ttr_score +
        0.12 * entropy_score +
        0.16 * entdens_score +
        0.14 * kw_score
    )
    score01 = clamp01(score01 - rep_penalty + embed_bonus - embed_penalty)

    score_0_100 = float(score01 * 100.0)
    grade = grade_from_score(score_0_100)

    metrics_out: Dict[str, Any] = {
        "tokens": int(tok),
        "noise_ratio_non_alnum": float(noise_ratio) if isinstance(noise_ratio, float) else None,
        "flesch_reading_ease": float(flesch) if isinstance(flesch, float) else None,
        "ttr": float(ttr) if isinstance(ttr, float) else None,
        "unigram_entropy": float(unigram_entropy) if isinstance(unigram_entropy, float) else None,
        "bigram_repetition_ratio": float(bigram_rep) if isinstance(bigram_rep, float) else None,
        "entity_mentions": int(entity_mentions),
        "entity_unique": int(entity_unique),
        "entity_density_per_100_tokens": float(ent_density) if isinstance(ent_density, float) else None,
        "keyword_top40_unique": int(kw_unique),
        "field_quality_score_0_100": float(score_0_100),
        "field_quality_grade": grade,
        "field_quality_flags": flags,
    }
    metrics_out.update(coh)

    return QualityResult(metrics=metrics_out, score_0_100=score_0_100, grade=grade, flags=flags)


# ----------------------------
# Main CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Assess KO field quality (single-field, no reference text)")
    parser.add_argument("--input", required=True, help="Input JSON path (list or object)")
    parser.add_argument("--field", required=True, help="Which field to assess (e.g. ko_content_flat)")
    parser.add_argument("--tsv", default=None, help="Output .tsv path (optional). Also writes a _flags.tsv summary.")
    parser.add_argument("--id-field", default=None, help="Preferred ID field; else tries _orig_id, @id, id")
    parser.add_argument("--spacy-max-chars", type=int, default=200_000, help="Max chars to feed spaCy")
    parser.add_argument("--disable-embeddings", action="store_true", help="Disable embedding coherence metrics")
    parser.add_argument("--st-model", default="sentence-transformers/all-mpnet-base-v2", help="SentenceTransformer model")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="ST device")
    parser.add_argument("--log-every", type=int, default=50, help="Progress logging frequency")
    parser.add_argument("--out-json", default=None, help="Output JSON path (optional). If set, writes input JSON augmented with '<field>_metrics'.",)
    args = parser.parse_args()

    inp = Path(args.input)

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tsv:
        tsv_path = Path(args.tsv)
    else:
        tsv_path = out_dir / f"{inp.stem}_content_quality_check.tsv"

    t0 = time.perf_counter()
    print(f"[{now_utc_iso()}] START ko_content_quality_diagnostics field='{args.field}' input='{inp}'")

    # Load spaCy
    nlp = load_spacy_model()
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    # Optional ST model
    st_model = None
    if not args.disable_embeddings:
        st_model = load_sentence_transformer(args.st_model, device=args.device)
        print(f"[{now_utc_iso()}] SentenceTransformer device: {st_model.device}")

    # Load JSON
    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if input is wrapped format (has docs field)
    is_wrapped_format = isinstance(data, dict) and "docs" in data and isinstance(data["docs"], list)
    is_input_list = isinstance(data, list)

    items = ensure_list(data)

    total = len(items)

    assessed = 0
    missing = 0
    rows: List[Dict[str, Any]] = []

    for i, obj in enumerate(items, start=1):
        val = obj.get(args.field)

        if not isinstance(val, str) or not val.strip():
            missing += 1
            continue

        qr = compute_field_quality(
            text=val,
            nlp=nlp,
            st_model=st_model,
            spacy_max_chars=args.spacy_max_chars,
        )

        metrics_key = f"{args.field}_metrics"
        obj[metrics_key] = qr.metrics
        obj[metrics_key]["assessed_at_utc"] = now_utc_iso()
        obj[metrics_key]["spacy_max_chars_used"] = int(min(len(val), args.spacy_max_chars))

        # Pick ID
        if args.id_field and obj.get(args.id_field):
            oid = obj.get(args.id_field)
        else:
            oid = obj.get("_orig_id") or obj.get("@id") or obj.get("id")

        rows.append({
            "id": oid,
            "title": (obj.get("title") or "")[:200],
            "field": args.field,
            "tokens": qr.metrics.get("tokens"),
            "noise_ratio_non_alnum": qr.metrics.get("noise_ratio_non_alnum"),
            "flesch_reading_ease": qr.metrics.get("flesch_reading_ease"),
            "ttr": qr.metrics.get("ttr"),
            "unigram_entropy": qr.metrics.get("unigram_entropy"),
            "bigram_repetition_ratio": qr.metrics.get("bigram_repetition_ratio"),
            "entity_unique": qr.metrics.get("entity_unique"),
            "entity_mentions": qr.metrics.get("entity_mentions"),
            "entity_density_per_100_tokens": qr.metrics.get("entity_density_per_100_tokens"),
            "keyword_top40_unique": qr.metrics.get("keyword_top40_unique"),
            "embedding_sentence_count_used": qr.metrics.get("embedding_sentence_count_used"),
            "mean_adjacent_cosine_similarity": qr.metrics.get("mean_adjacent_cosine_similarity"),
            "mean_centroid_cosine_similarity": qr.metrics.get("mean_centroid_cosine_similarity"),
            "first_last_block_cosine_similarity": qr.metrics.get("first_last_block_cosine_similarity"),
            "score_0_100": qr.metrics.get("field_quality_score_0_100"),
            "grade": qr.metrics.get("field_quality_grade"),
            "flags": ";".join(qr.metrics.get("field_quality_flags") or []),
        })

        assessed += 1
        if assessed == 1 or assessed == total or (assessed % args.log_every == 0):
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(1, assessed)
            eta_s = (total - i) * avg
            print(f"[{now_utc_iso()}] assessed={assessed} missing={missing} | avg={avg:.2f}s ETA~{eta_s/60:.1f}m")

    total_s = time.perf_counter() - t0
    print(f"[{now_utc_iso()}] DONE scoring in {total_s/60:.2f} minutes | assessed={assessed} missing={missing}")

    # TSV outputs
    df = pd.DataFrame(rows)
    df.to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")

    print(f"[{now_utc_iso()}] Wrote TSV: {tsv_path}")

    # Write flags summary TSV
    flags_path = tsv_path.with_name(tsv_path.stem + "_flags.tsv")
    flags_rows = []
    for r in rows:
        if r["flags"]:
            for fl in r["flags"].split(";"):
                if fl:
                    flags_rows.append({"id": r["id"], "title": r["title"], "field": r["field"], "flag": fl})
    if flags_rows:
        df_flags = pd.DataFrame(flags_rows)
        df_flags.to_csv(flags_path, sep="\t", index=False, encoding="utf-8")
        print(f"[{now_utc_iso()}] Wrote flags TSV: {flags_path} ({len(flags_rows)} flag entries)")

    if args.out_json:
        out_json_path = Path(args.out_json)
    else:
        out_json_path = out_dir / f"{inp.stem}_with_{args.field}_metrics.json"

    # Write augmented JSON - preserve original structure
    if is_wrapped_format:
        # For wrapped format, update the docs in place and write the whole wrapper
        out_payload = data
    elif is_input_list:
        out_payload = items
    else:
        out_payload = items[0] if items else {}

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)

    print(f"[{now_utc_iso()}] Wrote JSON: {out_json_path}")

if __name__ == "__main__":
    main()
