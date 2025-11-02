# which_model_to_choose/summary_analysis.py

from __future__ import annotations

import json
import os
import math
import re
import sys
import unicodedata

import requests

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Any, Dict, Tuple

try:
    from nltk.corpus import stopwords as nltk_stopwords
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False

try:
    from langdetect import detect as _ld_detect
    _LD_OK = True
except Exception:
    _LD_OK = False

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

try:
    from sentence_transformers import CrossEncoder
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

LANG_PIVOT = os.getenv("LANG_PIVOT", "en")   # pivot language for coverage/stopwords
LANG_MISMATCH_WEIGHT_SHIFT = float(os.getenv("LANG_MISMATCH_WEIGHT_SHIFT", "0.15"))
NON_LATIN_FRACTION_THRESH = float(os.getenv("NON_LATIN_FRACTION_THRESH", "0.20"))

RUNPOD_OLLAMA_HOST = os.getenv("RUNPOD_OLLAMA_HOST", "https://qaigjfchbeuczr-11435.proxy.runpod.net")
RUNPOD_OLLAMA_EMBED_MODEL = os.getenv("RUNPOD_OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "30"))

USE_LOCAL_NLI = os.getenv("USE_LOCAL_NLI", "1") not in {"0","false","False"}
LOCAL_NLI_MODEL = os.getenv("LOCAL_NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
LOCAL_NLI_DEVICE = os.getenv("LOCAL_NLI_DEVICE", "cpu")

HF_NLI_MODEL = os.getenv("HF_NLI_MODEL", "roberta-large-mnli")
HF_NLI_FALLBACKS = [
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
]
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")
HF_NLI_TIMEOUT = float(os.getenv("HF_NLI_TIMEOUT", "30"))
NLI_MAX_SENTS = int(os.getenv("NLI_MAX_SENTS", "15"))
NLI_CHUNK_CHARS = int(os.getenv("NLI_CHUNK_CHARS", "1500"))
HF_API_BASE = os.getenv("HF_API_BASE", "https://api-inference.huggingface.co/models")

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]          # <repo_root>/.../this_script.py  -> go up 2 levels
INPUT_DIR = PROJECT_ROOT / "input"             # e.g., /path/to/project/input
OUTPUT_DIR = PROJECT_ROOT / "output"           # e.g., /path/to/project/output
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # ensure output dir exists

# Optional: toggle advanced scorers centrally
USE_NLI = True
USE_EMBEDDINGS = True
FINAL_THRESHOLD = 0.55     # accept winner only if final_score >= this
MIN_ENTAILMENT = 0.30      # accept winner only if entailment >= this

LOG_CSV = OUTPUT_DIR / "benchmark_log.csv"


def guess_lang(s: str) -> str:
    """Cheap detector with fallback; returns ISO-639-1 like 'en', 'el', 'fr'."""
    if _LD_OK:
        try:
            return _ld_detect(s) or "en"
        except Exception:
            pass
    return "en"

def frac_non_latin(s: str) -> float:
    n = sum(1 for ch in s if ch.strip())
    k = sum(1 for ch in s if ch.strip() and "LATIN" not in unicodedata.name(ch, ""))
    return (k / n) if n else 0.0


# ---------------------------------------------
# 1) Files: find latest, load JSON
# ---------------------------------------------
def find_latest_json(input_dir: Path) -> Path:
    """Return the path to the most recently modified *.json file in input_dir."""
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")
    return max(json_files, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> Dict[str, Any] | list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

@lru_cache(maxsize=1)
def _get_local_nli_model():
    if not USE_LOCAL_NLI or not _ST_AVAILABLE:
        return None
    try:
        return CrossEncoder(LOCAL_NLI_MODEL, device=LOCAL_NLI_DEVICE)
    except Exception as e:
        print(f"[warn] Local NLI load failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None

@lru_cache(maxsize=4096)
def _local_nli_call(premise: str, hypothesis: str) -> tuple[float, float, float] | float:
    """
    Returns (p_contra, p_neutral, p_entail) or -1.0 sentinel if local unavailable.
    """
    model = _get_local_nli_model()
    if model is None:
        return -1.0
    try:
        import numpy as np
        logits = model.predict([(premise, hypothesis)], convert_to_tensor=False)
        vec = logits[0] if isinstance(logits, list) and hasattr(logits[0], "__len__") else logits
        vec = np.array(vec, dtype="float64").reshape(-1)
        e = np.exp(vec - np.max(vec))
        probs = (e / e.sum()).tolist()
        # Most CE NLI heads: [contradiction, neutral, entailment]
        if len(probs) == 3:
            return probs[0], probs[1], probs[2]
        # Fallback: treat max as entailment
        m = max(probs); idx = probs.index(m)
        if idx == 2 and len(probs) == 3:
            return probs[0], probs[1], probs[2]
        return 0.0, 1.0-m, m
    except Exception as e:
        print(f"[warn] Local NLI inference failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 0.0, 1.0, 0.33

@lru_cache(maxsize=1)
def _ollama_preflight() -> None:
    """
    Verify the host is Ollama and the embed model exists (pulled).
    Raises RuntimeError with a helpful message if something is wrong.
    """
    base = RUNPOD_OLLAMA_HOST.rstrip("/")
    # 1) Check Ollama is there
    try:
        r = requests.get(f"{base}/api/version", timeout=OLLAMA_TIMEOUT)
        if r.status_code == 404:
            raise RuntimeError(
                f"Ollama not found at {base} (404 on /api/version). "
                f"Double-check RUNPOD_OLLAMA_HOST and any reverse proxy path."
            )
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Failed to reach Ollama at {base}: {e}")

    # 2) Check model is present
    try:
        r = requests.get(f"{base}/api/tags", timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        names = {m.get("name", "") for m in data.get("models", [])} if isinstance(data, dict) else set()
        # Allow either exact name or prefix match (e.g., "nomic-embed-text:latest")
        if not any(n.startswith(RUNPOD_OLLAMA_EMBED_MODEL) for n in names):
            raise RuntimeError(
                f"Embedding model '{RUNPOD_OLLAMA_EMBED_MODEL}' is not pulled on the Ollama host.\n"
                f"On the server, run:  ollama pull {RUNPOD_OLLAMA_EMBED_MODEL}"
            )
    except Exception as e:
        raise RuntimeError(f"Failed to inspect models at {base}/api/tags: {e}")

@lru_cache(maxsize=4096)
def _hf_nli_call(premise: str, hypothesis: str) -> float:
    """
    Query HF Inference API for NLI with (premise, hypothesis).
    Tries the primary model (HF_NLI_MODEL) and a fallback list.
    For each model, tries several common payload schemas.
    Returns ENTAILMENT probability in [0,1]; 0.33 on neutral/unknown.
    """
    if not HF_TOKEN:
        return 0.0

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    # Common payload variants seen across hosted NLI handlers
    def payloads(p: str, h: str):
        base_opts = {"wait_for_model": True}
        # 1) text/text_pair (works for some generic sequence classifiers)
        yield {"inputs": {"text": p, "text_pair": h}, "options": base_opts}
        # 2) premise/hypothesis (common for MoritzLaurer models)
        yield {"inputs": {"premise": p, "hypothesis": h}, "options": base_opts}
        # 3) sentence1/sentence2 (some handlers)
        yield {"inputs": {"sentence1": p, "sentence2": h}, "options": base_opts}
        # 4) list of two strings (rare but harmless to try)
        yield {"inputs": [p, h], "options": base_opts}

    # Normalise different output shapes to an entailment probability
    def parse_output(repo_id: str, out) -> Optional[float]:
        # A) [{label, score}, ...] or [[{..}, {..}, {..}]]
        if isinstance(out, list):
            if out and isinstance(out[0], list):
                out = out[0]
            if out and isinstance(out[0], dict):
                for item in out:
                    lab = str(item.get("label", "")).upper()
                    if lab.startswith("ENTAIL"):
                        try:
                            return float(item.get("score", 0.0))
                        except Exception:
                            return None
        # B) {"labels":[...], "scores":[...]}
        if isinstance(out, dict) and "labels" in out and "scores" in out:
            labels = [str(x).upper() for x in out["labels"]]
            for lab, sc in zip(labels, out["scores"]):
                if lab.startswith("ENTAIL"):
                    try:
                        return float(sc)
                    except Exception:
                        return None
        # C) zero-shot style: {"sequence":..., "labels":[...], "scores":[...]}
        if isinstance(out, dict) and "labels" in out and "scores" in out and "sequence" in out:
            labels = [str(x).upper() for x in out["labels"]]
            for lab, sc in zip(labels, out["scores"]):
                if lab.startswith("ENTAIL"):
                    try:
                        return float(sc)
                    except Exception:
                        return None
        # Unknown shape
        prev = str(out)
        if len(prev) > 220: prev = prev[:220] + "…"
        print(f"[warn] NLI unexpected output from '{repo_id}': {prev}", file=sys.stderr)
        return None

    model_ids = [HF_NLI_MODEL] + [m for m in HF_NLI_FALLBACKS if m != HF_NLI_MODEL]

    for repo_id in model_ids:
        url = f"{HF_API_BASE.rstrip('/')}/{repo_id}"
        for pl in payloads(premise, hypothesis):
            try:
                r = requests.post(url, headers=headers, json=pl, timeout=HF_NLI_TIMEOUT)
                # Some handlers reply 400 for wrong schema; 404 for unserved model; 503 for cold start.
                if r.status_code == 404:
                    print(f"[warn] NLI 404 for '{repo_id}', trying next model…", file=sys.stderr)
                    break  # try next model id
                if r.status_code == 503:
                    # cold start; neutral fallback
                    return 0.33
                if r.status_code == 400:
                    # wrong payload shape; try next schema for the same model
                    continue
                r.raise_for_status()
                prob = parse_output(repo_id, r.json())
                if prob is not None:
                    return prob if prob > 0 else 0.33
                # parsed but no entailment label found → neutral
                return 0.33
            except Exception as e:
                # Transport error; try next schema/model
                print(f"[warn] NLI call failed on '{repo_id}': {type(e).__name__}: {e}", file=sys.stderr)
                continue

    # All attempts failed
    return 0.33


def _embed_text_chunked(text: str, *, max_chars: int = 1500, embed_fn: Callable[[str], Optional[list[float]]]) -> Optional[list[float]]:
    if not text:
        return None
    chunks = _chunk_text(text, max_chars=max_chars)
    vecs = []
    for ch in chunks:
        v = embed_fn(ch)   # <-- always use the supplied function
        if not v:
            continue
        n = math.sqrt(sum(x*x for x in v)) or 1.0
        vecs.append([x / n for x in v])
    if not vecs:
        return None
    pooled = [sum(col)/len(vecs) for col in zip(*vecs)]
    n = math.sqrt(sum(x*x for x in pooled)) or 1.0
    return [x / n for x in pooled]


# ---------------------------------------------
# 2) Candidate extraction
# ---------------------------------------------
CANDIDATE_PREFIX = "ko_content_flat"


def extract_ko_object(data: Dict[str, Any] | list[dict]) -> Dict[str, Any]:
    """
    Accepts either a KO object or a list containing at least one KO object.
    Returns a single dict (first element if list).
    """
    if isinstance(data, list):
        if not data:
            raise ValueError("JSON is an empty list; expected at least one KO object.")
        ko = data[0]
        if not isinstance(ko, dict):
            raise TypeError("First list element is not a dict.")
        return ko
    elif isinstance(data, dict):
        return data
    else:
        raise TypeError("JSON must be a dict or a list of dicts.")


def extract_candidates(ko: dict) -> Tuple[str, Dict[str, str]]:
    """
    Returns (source_text, candidates_map).
    - source_text = ko['ko_content_flat']
    - candidates_map = { field_name: text } for keys starting with 'ko_content_flat_'
    """
    source = (ko.get(CANDIDATE_PREFIX) or "").strip()
    if not source:
        raise ValueError(f"Missing '{CANDIDATE_PREFIX}' as source text.")

    candidates: Dict[str, str] = {}
    for k, v in ko.items():
        if k.startswith(CANDIDATE_PREFIX + "_") and isinstance(v, str):
            vv = v.strip()
            if vv:
                candidates[k] = vv

    if not candidates:
        raise ValueError("No candidate fields found (expected keys like 'ko_content_flat_*').")

    return source, candidates


# ---------------------------------------------
# 3) Text utilities (tokenisation, normalisation)
# ---------------------------------------------
_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9%]+(?:['-][A-Za-z0-9]+)?")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def words(s: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(s)]


def sentences(s: str) -> list[str]:
    return [t.strip() for t in _SENT_SPLIT_RE.split(s) if t.strip()]


def get_stopwords() -> set[str]:
    """
    Use NLTK English stopwords only. If unavailable, return an empty set.
    """
    if not NLTK_AVAILABLE:
        print("[warn] NLTK not available; using empty stopword set.", file=sys.stderr)
        return set()
    try:
        return set(nltk_stopwords.words("english"))
    except LookupError:
        print("[warn] NLTK 'stopwords' corpus not found. Run:\n"
              "       python -c \"import nltk; nltk.download('stopwords')\"",
              file=sys.stderr)
        return set()
    except Exception as e:
        print(f"[warn] Failed to load NLTK stopwords: {type(e).__name__}: {e}", file=sys.stderr)
        return set()


@lru_cache(maxsize=2048)
def get_embed_ollama(text: str) -> Optional[list[float]]:
    """
    Pure Ollama client. Uses /api/embeddings with 'prompt'.
    Returns a single vector or None on error.
    """
    if not text:
        return None
    try:
        _ollama_preflight()  # will raise once with a clear message if misconfigured
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        return None

    base = RUNPOD_OLLAMA_HOST.rstrip("/")
    url = f"{base}/api/embeddings"
    try:
        payload = {"model": RUNPOD_OLLAMA_EMBED_MODEL, "prompt": text}
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Accept both shapes seen in the wild
        if isinstance(data, dict):
            if "embedding" in data and isinstance(data["embedding"], list):
                return data["embedding"]
            if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                return data["embeddings"][0]
        print(f"[warn] Unexpected Ollama embedding payload: {str(data)[:200]}...", file=sys.stderr)
        return None
    except requests.HTTPError as e:
        # Show the server message if any — very helpful for 404/400
        msg = e.response.text[:300] if getattr(e, "response", None) else str(e)
        print(f"[warn] Ollama embed failed ({e}): {msg}", file=sys.stderr)
        return None
    except requests.RequestException as e:
        print(f"[warn] Ollama embed failed: {type(e).__name__}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------
# 4) Metrics (coverage, length, repetition, numbers/dates)
# ---------------------------------------------
def top_keywords(text: str, top_n: int = 50, stopset: Optional[set[str]] = None) -> set[str]:
    stopset = stopset or get_stopwords()
    toks = [t for t in words(text) if t not in stopset and len(t) > 2]
    most = [w for w, _ in Counter(toks).most_common(top_n)]
    return set(most)


def coverage_recall(source: str, summary: str, top_n: int = 50, stopset: Optional[set[str]] = None) -> float:
    kw = top_keywords(source, top_n=top_n, stopset=stopset)
    if not kw:
        return 0.0
    sum_toks = set(words(summary))
    found = sum(1 for k in kw if k in sum_toks)
    return found / len(kw)


def repetition_ratio(summary: str, n: int = 3) -> float:
    toks = words(summary)
    if len(toks) < n:
        return 0.0
    grams = [" ".join(toks[i:i+n]) for i in range(len(toks)-n+1)]
    counts = Counter(grams)
    reps = sum(c-1 for c in counts.values() if c > 1)
    return reps / max(1, len(grams))


def length_score(source: str, summary: str) -> float:
    """Band-based length fitness relative to source length."""
    s_len = len(source)
    m_len = len(summary)

    if s_len < 2000:             # short source
        tgt_min, tgt_max = 400, 1600
    elif s_len < 10000:          # medium source
        tgt_min, tgt_max = 800, 3000
    else:                        # long source
        tgt_min, tgt_max = 1500, 6000

    if m_len <= 0:
        return 0.0
    if m_len < tgt_min:
        return max(0.0, m_len / tgt_min * 0.7)   # partial credit if under
    if m_len > tgt_max:
        return max(0.0, tgt_max / m_len * 0.7)   # partial if over
    return 1.0


_NUM_RE = re.compile(r"\b\d{1,4}(?:[.,]\d{1,3})*(?:%|[A-Za-z]*)?\b")

def salient_mismatch_penalty(source: str, summary: str) -> float:
    """
    Penalty in [0,1]. 0 = no penalty. 1 = heavy mismatch.
    Checks numbers/years appearing in summary but not in source.
    """
    src = norm_space(source).lower()
    summ = norm_space(summary).lower()

    nums = set(_NUM_RE.findall(summ))
    years = set(re.findall(r"\b(19|20)\d{2}\b", summ))

    bad = 0
    total = 0

    # numeric strings
    for m in nums:
        total += 1
        if m not in src:
            bad += 1

    # years
    for y in years:
        total += 1
        if y not in src:
            bad += 1

    if total == 0:
        return 0.0
    ratio = bad / total
    # allow a couple slips; scale a bit
    return min(1.0, ratio)


# ---------------------------------------------
# 5) (Optional) Embeddings & NLI stubs
# ---------------------------------------------
def cosine(a, b) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def embedding_cosine_score(src_vec: Optional[list[float]], sum_vec: Optional[list[float]]) -> float:
    if not src_vec or not sum_vec:
        return 0.0
    c = cosine(src_vec, sum_vec)
    return (c + 1) / 2  # map [-1,1] -> [0,1]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    text = norm_space(text)
    if len(text) <= max_chars:
        return [text]
    chunks = []
    i = 0
    # 20% overlap to keep sentence continuity
    step = int(max_chars * 0.8)
    while i < len(text):
        chunks.append(text[i:i+max_chars])
        i += step
    return chunks

def _word_set(s: str) -> set[str]:
    return set(words(s))

def _best_premise_chunk(sentence: str, chunks: list[str]) -> str:
    """Pick the chunk with highest token overlap (cheap & decent)."""
    sset = _word_set(sentence)
    best = chunks[0]
    best_score = -1.0
    for c in chunks:
        cset = _word_set(c)
        inter = len(sset & cset)
        denom = (len(sset) + len(cset)) or 1
        score = inter / denom
        if score > best_score:
            best_score = score
            best = c
    return best


def nli_entailment_score(source: str, summary: str, max_sents: int = NLI_MAX_SENTS) -> float:
    sents = sentences(summary)
    if not sents:
        return 0.0
    sents = sents[:max_sents]
    chunks = _chunk_text(source, max_chars=NLI_CHUNK_CHARS)

    ents = []
    for s in sents:
        premise = _best_premise_chunk(s, chunks)
        out = _local_nli_call(premise, s)
        if out == -1.0:  # local not available → HF fallback (kept as-is)
            p_ent = _hf_nli_call(premise, s)
        else:
            # tuple from local path
            p_contra, p_neu, p_ent = out
        ents.append(p_ent if out != -1.0 else (p_ent if p_ent is not None else 0.33))

    ents.sort()
    n = len(ents)
    lo = int(0.1 * n); hi = n - lo if (n - lo) > lo else n
    kept = ents[lo:hi] if hi > lo else ents
    return sum(kept) / len(kept) if kept else 0.0



# ---------------------------------------------
# 6) Scoring & selection
# ---------------------------------------------
@dataclass
class CandidateScore:
    final_score: float
    entailment: float
    coverage: float
    embed_cosine: float
    length_score: float
    repetition: float
    num_penalty: float
    reason: str = "ok"


def score_candidate(
    source: str,
    summary: str,
    *,
    entailment: float = 0.0,
    embed_cosine: float = 0.0,
    stopset: Optional[set[str]] = None,
    weights: Optional[dict] = None,   # <- new
) -> CandidateScore:
    """
    weights keys: ent, cov, emb, len, rep, instr
    They should already be normalised to sum to 1.0 by the caller.
    """
    if not summary or not summary.strip():
        return CandidateScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, reason="empty")

    # Defaults (used if caller didn't pass weights)
    w = weights or {"ent": 0.45, "cov": 0.20, "emb": 0.15, "len": 0.10, "rep": 0.05, "instr": 0.05}

    ls  = length_score(source, summary)
    cov = coverage_recall(source, summary, top_n=75, stopset=stopset)
    rep = repetition_ratio(summary, n=3)
    pen = salient_mismatch_penalty(source, summary)

    bullet_heavy = summary.lstrip().startswith(("-", "•"))
    instr_bonus = 0.0 if bullet_heavy else 0.1

    # Hard fail only if numeric mismatches are egregious AND entailment is really low (when used)
    if pen > 0.5 and entailment < 0.3:
        return CandidateScore(0.0, entailment, cov, embed_cosine, ls, rep, pen, reason="hallucination_numeric")

    ent_gated = entailment * (0.5 + 0.5 * cov)  # if cov=0 → halves NLI weight; if cov=1 → full

    final = (
            w["ent"] * ent_gated +
            w["cov"] * cov +
            w["emb"] * embed_cosine +
            w["len"] * ls +
            w["rep"] * max(0.0, 1.0 - rep) +
            w["instr"] * instr_bonus
    )

    return CandidateScore(
        final_score=round(final, 4),
        entailment=round(entailment, 4),
        coverage=round(cov, 4),
        embed_cosine=round(embed_cosine, 4),
        length_score=round(ls, 4),
        repetition=round(rep, 4),
        num_penalty=round(pen, 4),
        reason="ok",
    )


def pick_best(
    source: str,
    candidate_map: Dict[str, str],
    *,
    use_nli: bool = False,
    use_embeddings: bool = False,
    get_embed=None,
    stopset: Optional[set[str]] = None,
    final_threshold: float = 0.45,
    min_entailment: float = 0.25,
    lang_hint: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "winner": field_name | None,
        "scores": { field_name: { ...metrics... } },
        "reason": "ok" | "low_confidence"
      }
    """
    # 1) Build dynamic weights based on enabled signals
    w_ent = 0.45 if use_nli else 0.0
    w_emb = 0.15 if use_embeddings else 0.0
    w_cov = 0.20
    w_len = 0.10
    w_rep = 0.05
    w_ins = 0.05

    # If source text is mostly non-Latin (e.g., Greek) but we only have English stopwords,
    # shift some mass from coverage to NLI/embeddings (cross-lingual-friendly signals).
    if lang_hint and lang_hint.get("source_non_latin"):
        shift = min(LANG_MISMATCH_WEIGHT_SHIFT, w_cov / 2)
        w_cov -= shift
        if use_nli:        w_ent += shift * 0.7
        if use_embeddings: w_emb += shift * 0.3

    total = w_ent + w_emb + w_cov + w_len + w_rep + w_ins
    # normalise so weights always sum to 1.0
    weights = {k: v / total for k, v in {
        "ent": w_ent, "cov": w_cov, "emb": w_emb, "len": w_len, "rep": w_rep, "instr": w_ins
    }.items()}

    # 2) Source embedding (optional)
    src_vec = _embed_text_chunked(source, embed_fn=get_embed) if (use_embeddings and get_embed) else None

    scores: Dict[str, Dict[str, Any]] = {}
    for field, summ in candidate_map.items():
        ent = nli_entailment_score(source, summ) if use_nli else 0.0
        emb = 0.0
        if use_embeddings and get_embed:
            try:
                emb = embedding_cosine_score(src_vec, _embed_text_chunked(summ, embed_fn=get_embed))
            except Exception:
                emb = 0.0

        sc = score_candidate(source, summ, entailment=ent, embed_cosine=emb, stopset=stopset, weights=weights)
        scores[field] = sc.__dict__

    ranked = sorted(scores.items(), key=lambda kv: kv[1]["final_score"], reverse=True)
    winner, meta = ranked[0]

    # 3) Acceptance rule:
    #    - If NLI is disabled, ignore min_entailment.
    #    - If nothing passes final_threshold, still pick top but mark low_confidence.
    passes = (meta["final_score"] >= final_threshold) and (meta["entailment"] >= min_entailment if use_nli else True)
    if passes:
        return {"winner": winner, "scores": scores, "reason": "ok"}

    # Graceful fallback
    return {"winner": winner, "scores": scores, "reason": "low_confidence"}


# ---------------------------------------------
# 7) Main / CLI
# ---------------------------------------------
def main() -> None:
    """
    Reads the latest KO JSON from INPUT_DIR, evaluates all 'ko_content_flat_*' candidates
    against 'ko_content_flat' as source, selects a winner, and writes artefacts to OUTPUT_DIR.
    """
    # 1) locate & load the most recent KO
    latest_path = find_latest_json(INPUT_DIR)
    data = load_json(latest_path)
    ko = extract_ko_object(data)

    meta_bits = []
    title = ko.get("title") or ""
    desc = ko.get("description") or ""
    kw = ", ".join(ko.get("keywords") or [])
    if title: meta_bits.append(f"Title: {title}")
    if desc:  meta_bits.append(f"Description: {desc}")
    if kw:    meta_bits.append(f"Keywords: {kw}")
    metadata_context = "\n".join(meta_bits)

    source, candidates = extract_candidates(ko)

    # 2) resources (stopwords + optional embedding hook)
    stopset = get_stopwords()

    get_embed_fn = (get_embed_ollama if USE_EMBEDDINGS else None)
    print(f"[info] Embeddings via {RUNPOD_OLLAMA_HOST} model={RUNPOD_OLLAMA_EMBED_MODEL}", file=sys.stderr)
    try:
        _ollama_preflight()
        print("[info] Ollama embeddings ready", file=sys.stderr)
    except RuntimeError as e:
        print(f"[warn] {e} — embeddings will be 0.0", file=sys.stderr)

    # 3) run selection
    # Auto-disable NLI if token is missing to avoid 0.0 entailment pulling scores down
    use_nli_effective = USE_NLI and (_get_local_nli_model() is not None or bool(HF_TOKEN))

    # language analysis for source and candidates
    src_lang = guess_lang(source)
    has_non_latin = frac_non_latin(source) >= NON_LATIN_FRACTION_THRESH

    # Augment source with metadata to reinforce salient facts (names, entities, year)
    scoring_source = source
    if metadata_context:
        scoring_source = f"{source}\n\n[METADATA]\n{metadata_context}"

    # If coverage/stopwords are English-only and source is mostly non-Latin,
    # rely more on multilingual NLI/embeddings (weights shift happens inside pick_best)
    lang_hint = {
        "source_lang": src_lang,
        "source_non_latin": has_non_latin
    }

    result = pick_best(
        scoring_source,
        candidates,
        use_nli=use_nli_effective,
        use_embeddings=USE_EMBEDDINGS,
        get_embed=get_embed_fn,
        stopset=stopset,
        final_threshold=FINAL_THRESHOLD,
        min_entailment=MIN_ENTAILMENT,
        lang_hint=lang_hint,
    )

    # 4) prepare output objects/paths (mirror input name in OUTPUT_DIR)
    base = latest_path.stem                               # e.g., "myfile"
    selected_json_path = OUTPUT_DIR / f"{base}.selected.json"

    out_obj = {
        "input_file": str(latest_path),
        "winner": result["winner"],
        "reason": result["reason"],
        "scores": result["scores"],                       # per-candidate metrics
    }

    out_obj["nli_diagnostics"] = {
        "use_nli_requested": USE_NLI,
        "use_nli_effective": use_nli_effective,
        "local_nli": bool(_get_local_nli_model()),
        "hf_token_present": bool(HF_TOKEN),
        "hf_model": HF_NLI_MODEL,
        "local_model": LOCAL_NLI_MODEL,
    }

    # 5) write selected result JSON
    selected_json_path.write_text(
        json.dumps(out_obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 6) append/update CSV log (one line per run)
    try:
        top_field = result["winner"]
        top_score = result["scores"].get(top_field, {}) if top_field else {}
        new_file = not LOG_CSV.exists()
        with LOG_CSV.open("a", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.writer(f)
            if new_file:
                writer.writerow([
                    "input_file", "winner", "reason",
                    "final_score", "entailment", "coverage", "embed_cosine",
                    "length_score", "repetition", "num_penalty"
                ])
            writer.writerow([
                str(latest_path),
                top_field or "",
                result["reason"],
                top_score.get("final_score", ""),
                top_score.get("entailment", ""),
                top_score.get("coverage", ""),
                top_score.get("embed_cosine", ""),
                top_score.get("length_score", ""),
                top_score.get("repetition", ""),
                top_score.get("num_penalty", ""),
            ])
    except Exception:
        # CSV is non-critical; continue silently on logging issues
        pass

    # 7) also print a compact JSON to stdout (handy in PyCharm/run configs)
    print(json.dumps(out_obj, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)