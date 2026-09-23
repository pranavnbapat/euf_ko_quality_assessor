# assess_ko_quality/core/corpus_index.py
"""
The corpus the metadata score measures against.

Several metadata checks only mean something relative to the rest of the platform:

  - findability: if someone searches with this title, does THIS KO come back, or one
    of the 10,000 others? That needs a search index over every KO.
  - information density: is a word specific or generic? That needs document
    frequencies across the platform, plus general-language word frequencies so a
    word is judged in its own language, not only in English.
  - boilerplate: is this exact title shared by KOs from unrelated projects?

All of it is computed from the export itself - nothing typed in - and cached, keyed on
the export file and the settings that affect it, so a run is reproducible and the
second run on the same export costs seconds.

The reference text of a KO is what the file says: the vision model's reading of the
file followed by the extracted text. Metadata fields are deliberately NOT indexed, so a
title cannot find its own KO by matching itself.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import pickle
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from core.text_utils import norm_text

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SPEC_PATH = BASE / "validation" / "metadata_spec.json"
CACHE_DIR = BASE / "output" / "_corpus_cache"

TOKEN_PATTERN = r"(?u)\b\w\w+\b"
_TOKEN_RE = re.compile(TOKEN_PATTERN)
_HAS_LETTER = re.compile(r"[^\W\d_]")
_KW_SPLIT = re.compile(r"\s*[;,|]\s*")

TEXT_FIELDS = ("title", "subtitle", "description")
FIELDS = ("title", "subtitle", "description", "keywords")


# --------------------------------------------------------------------- record access
def load_spec(path: Path = SPEC_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ko_id(rec: Dict[str, Any]) -> str:
    return str(rec.get("_orig_id") or rec.get("_id") or rec.get("@id") or "")


def project_key(rec: Dict[str, Any]) -> str:
    return str(rec.get("project_id") or rec.get("project_acronym") or rec.get("project_name") or "")


def field_value(rec: Dict[str, Any], field: str, variant: str) -> Any:
    """`variant` is "original" or "llm"; the LLM proposal lives in `<field>_llm`."""
    return rec.get(field if variant == "original" else f"{field}_llm")


def split_keywords(value: Any) -> List[str]:
    """
    The keyword terms a field actually carries.

    Some records store several keywords in one item ("Adaptation to climate
    change;Decision Support System"). That is a formatting fault and is scored as one;
    for everything else the terms are split out, so the same fault is not also charged
    to length, consistency and coverage.
    """
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else [value]
    out: List[str] = []
    for item in items:
        for part in _KW_SPLIT.split(str(item)):
            part = " ".join(part.split())
            if part:
                out.append(part)
    return out


_TAG = re.compile(r"<[^>]+>")
_BLOCK = re.compile(r"</?(?:p|div|li|ul|ol|br|h[1-6]|tr)\b[^>]*>", re.I)


def rich_text_to_plain(value: str) -> str:
    """
    The text of a rich-text field. The platform's description editor stores HTML and
    asks contributors to use bullet points, so its markup is structure, not noise:
    block tags become line breaks, the rest is dropped, entities are decoded.
    """
    import html

    if "<" not in value and "&" not in value:
        return value
    text = re.sub(r"<li\b[^>]*>", "\n• ", value, flags=re.I)   # keep bullets visible as bullets
    text = _BLOCK.sub("\n", text)
    text = html.unescape(_TAG.sub("", text))
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def field_text(field: str, value: Any) -> str:
    """One string per field, for the checks that read text."""
    if field == "keywords":
        return ", ".join(split_keywords(value))
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    if not isinstance(value, str):
        return ""
    if field == "description":
        value = rich_text_to_plain(value)
    return norm_text(value)


def reference_text(rec: Dict[str, Any], max_chars: int) -> str:
    """What the KO says: the vision reading of the file, then the extracted text."""
    vision = norm_text(rec.get("ko_content_flat_vision"))
    content = norm_text(rec.get("ko_content_flat"))
    return f"{vision} {content}".strip()[:max_chars]


def word_tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def has_letter(token: str) -> bool:
    return bool(_HAS_LETTER.search(token))


def _norm_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return " ".join(str(value or "").split()).lower()


def _value_key(value: Any) -> str:
    return hashlib.sha1(_norm_value(value).encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------- language
def _detect(text: str) -> str:
    from langdetect import DetectorFactory, detect

    DetectorFactory.seed = 42
    try:
        return detect(text) if len(text) >= 50 else "unknown"
    except Exception:
        return "unknown"


def detect_languages(texts: Sequence[str], workers: int = 0) -> List[str]:
    """Detected language of each text. langdetect is seeded, so this is deterministic."""
    probes = [t[:1000] for t in texts]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if len(probes) < 200 or workers == 1:
        return [_detect(p) for p in probes]
    from multiprocessing import Pool

    with Pool(workers) as pool:
        return pool.map(_detect, probes, chunksize=64)


# --------------------------------------------------------------------- models
class Embedder:
    """Multilingual E5. Queries and passages are prefixed as the model was trained."""

    def __init__(self, name: str):
        self.name = name
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(os.cpu_count() or 4)
            self._model = SentenceTransformer(self.name, device="cpu")
            self._model.max_seq_length = 512
        return self._model

    def _encode(self, texts: Sequence[str], prefix: str, batch_size: int = 32,
                progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self._load().encode(
            [prefix + t for t in texts], batch_size=batch_size,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=progress).astype(np.float32)

    def queries(self, texts: Sequence[str], **kw) -> np.ndarray:
        return self._encode(texts, "query: ", **kw)

    def passages(self, texts: Sequence[str], **kw) -> np.ndarray:
        return self._encode(texts, "passage: ", **kw)


# --------------------------------------------------------------------- specificity
# wordfreq's language codes; langdetect reports Croatian, Serbian and Bosnian separately
# where wordfreq has one Serbo-Croatian list.
_WORDFREQ_ALIAS = {"hr": "sh", "sr": "sh", "bs": "sh", "no": "nb"}


@functools.lru_cache(maxsize=None)
def _general_specificity(token: str, langs: Tuple[str, ...], zipf_max: float) -> float:
    """1 - (highest Zipf frequency of the word in any of `langs`) / zipf_max."""
    from wordfreq import zipf_frequency

    top = max((zipf_frequency(token, l) for l in langs), default=0.0)
    return min(1.0, max(0.0, 1.0 - top / zipf_max))


def _wordfreq_languages(detected: Iterable[str]) -> Tuple[str, ...]:
    """
    The detected languages wordfreq can serve. Some (Japanese, Korean, Chinese) need
    extra tokenisers that are not installed; those are dropped rather than failing.
    """
    from wordfreq import available_languages, zipf_frequency

    usable = []
    for l in sorted({_WORDFREQ_ALIAS.get(l, l) for l in detected} & set(available_languages())):
        try:
            zipf_frequency("test", l)
            usable.append(l)
        except Exception:
            continue
    return tuple(usable)


def ko_language_text(rec: Dict[str, Any]) -> str:
    """
    The text to detect a KO's language from: the extracted text when there is enough
    of it, otherwise the vision reading. Not the vision reading first - in about one
    non-English KO in five the vision model answered in English.
    """
    content = norm_text(rec.get("ko_content_flat"))
    return content if len(content) >= 50 else norm_text(rec.get("ko_content_flat_vision"))


def keyword_set_similarity(vecs: np.ndarray) -> float:
    """KPEval's emb_sim: mean cosine over all ordered pairs of distinct keywords."""
    m = len(vecs)
    if m < 2:
        return 0.0
    sim = vecs @ vecs.T
    return float((sim.sum() - np.trace(sim)) / (m * (m - 1)))


# --------------------------------------------------------------------- index
class CorpusIndex:
    """Search index and corpus statistics over one export."""

    def __init__(self) -> None:
        self.ids: List[str] = []
        self.pos: Dict[str, int] = {}
        self.langs: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.bm25 = None                                  # scipy CSC, docs x terms
        self.df_all: Optional[np.ndarray] = None
        self.passages: Optional[np.ndarray] = None       # docs x dim, normalised
        self.shared: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.density_anchors: Dict[str, Tuple[float, float]] = {}
        self.keyword_max_words: int = 0
        self.null_terms: List[str] = []
        self.null_term_vecs: Optional[np.ndarray] = None
        self.coherence_pool: Dict[str, np.ndarray] = {}     # field -> other KOs' embeddings
        self.keyword_sim_anchors: Tuple[float, float] = (0.0, 1.0)  # p5, p95 of emb_sim
        self.wordfreq_langs: Tuple[str, ...] = ()
        self.zipf_max: float = 8.0
        self.key_term_max_zipf: float = 6.0
        self.meta: Dict[str, Any] = {}

    # ----------------------------------------------------------------- lookups
    @property
    def n_docs(self) -> int:
        return len(self.ids)

    def corpus_specificity(self, token: str) -> float:
        """Normalised IDF over every KO on the platform: 0 in every KO, 1 in none."""
        j = self.vocab.get(token)
        d = int(self.df_all[j]) if j is not None else 0
        return math.log((self.n_docs + 1) / (d + 1)) / math.log(self.n_docs + 1)

    def specificity(self, tokens: Sequence[str]) -> List[float]:
        """
        How specific each word is, 0 (generic) to 1 (rare).

        The lower of two readings. Platform-generic: normalised IDF across all KOs, so
        "agriculture" and "project" count as generic here even though they are not
        common in general language. Language-generic: the word's frequency in general
        usage of any language the corpus contains (wordfreq, Speer 2022), so function
        words and everyday vocabulary count as generic in Finnish or Slovene as much as
        in English - which per-language document frequencies could not do, because most
        languages have too few KOs to estimate them.
        """
        return [min(self.corpus_specificity(t),
                    _general_specificity(t, self.wordfreq_langs, self.zipf_max))
                for t in tokens]

    def key_terms(self, ref_tokens: Sequence[str], k: int) -> List[str]:
        """
        The k most characteristic words of a reference text.

        Candidates exclude function words - any word with a Zipf frequency at or above
        `key_term_max_zipf` in one of the corpus languages (more frequent than one in a
        thousand words: "the", "een", "und") - as keyphrase extraction always excludes
        stopwords. The rest are ranked by TF-IDF with sublinear term frequency,
        (1 + ln tf) x ln(N / df) (Salton & Buckley 1988). Without the exclusion a Dutch
        file's "key terms" came out as "een, het, kan": IDF alone cannot demote function
        words of a language that only a few KOs are written in.
        """
        from wordfreq import zipf_frequency

        # Words only: identifiers and codes ("bp_pwgz9g80", "2019") are rare, so TF-IDF
        # would otherwise rank them as the file's most characteristic terms.
        counts = Counter(t for t in ref_tokens if len(t) >= 3 and t.replace("-", "").isalpha())
        scored = []
        for t, tf in counts.items():
            if max((zipf_frequency(t, l) for l in self.wordfreq_langs), default=0.0) >= self.key_term_max_zipf:
                continue
            j = self.vocab.get(t)
            d = int(self.df_all[j]) if j is not None else 0
            scored.append(((1.0 + math.log(tf)) * math.log((self.n_docs + 1) / (d + 1)), t))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [t for _, t in scored[:k]]

    def generic_words(self, lang: str, n: int = 100) -> List[str]:
        """The language's commonest words, from wordfreq; English when unsupported."""
        from wordfreq import available_languages, top_n_list

        code = _WORDFREQ_ALIAS.get(lang, lang)
        code = code if code in available_languages() else "en"
        return [w for w in top_n_list(code, n * 2) if has_letter(w)][:n]

    def shared_projects(self, variant: str, field: str, value: Any) -> int:
        """How many distinct projects carry exactly this value in this field."""
        return self.shared.get(variant, {}).get(field, {}).get(_value_key(value), 0)

    @staticmethod
    def _rank(scores: np.ndarray, own: int) -> float:
        """1-based rank of `own`; ties share the middle of their block."""
        s = scores[own]
        greater = int((scores > s).sum())
        ties = int((scores == s).sum()) - 1
        return 1.0 + greater + ties / 2.0

    def rank_bm25(self, query_tokens: Sequence[str], own: int) -> Optional[float]:
        cols = sorted({self.vocab[t] for t in query_tokens if t in self.vocab})
        if not cols:
            return float(self.n_docs)            # nothing in the query exists anywhere
        scores = np.asarray(self.bm25[:, cols].sum(axis=1)).ravel()
        return self._rank(scores, own)

    def rank_semantic(self, query_vec: np.ndarray, own: int) -> float:
        scores = self.passages @ query_vec.ravel()
        return self._rank(scores, own)

    # ----------------------------------------------------------------- build
    @classmethod
    def build(cls, records: Sequence[Dict[str, Any]], spec: Dict[str, Any],
              embedder: Embedder, export: Path, log=print) -> "CorpusIndex":
        from scipy import sparse
        from sklearn.feature_extraction.text import CountVectorizer

        ref_cfg = spec["reference"]
        idx = cls()
        idx.ids = [ko_id(r) for r in records]
        idx.pos = {k: i for i, k in enumerate(idx.ids) if k}
        refs = [reference_text(r, ref_cfg["max_chars"]) for r in records]
        n = len(refs)

        log(f"[corpus] detecting the language of {n} KOs ...")
        idx.langs = detect_languages([ko_language_text(r) for r in records])
        lang_counts = Counter(idx.langs)
        idx.wordfreq_langs = _wordfreq_languages(lang_counts)
        idx.zipf_max = float(spec["specificity"]["zipf_max"])
        idx.key_term_max_zipf = float(spec["coverage"]["key_term_max_zipf"])
        log(f"[corpus] general-language frequencies for: {', '.join(idx.wordfreq_langs)}")

        log("[corpus] counting terms and building the BM25 index ...")
        cv = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, dtype=np.float32)
        X = cv.fit_transform(refs).tocsr()
        idx.vocab = {t: int(j) for t, j in cv.vocabulary_.items()}
        present = X.copy()
        present.data[:] = 1.0
        idx.df_all = np.asarray(present.sum(axis=0)).ravel().astype(np.int32)
        del present

        k1, b = float(spec["bm25"]["k1"]), float(spec["bm25"]["b"])
        dl = np.asarray(X.sum(axis=1)).ravel()
        avgdl = float(dl.mean()) or 1.0
        idf = np.log((n - idx.df_all + 0.5) / (idx.df_all + 0.5) + 1.0).astype(np.float32)
        rows = np.repeat(np.arange(n), np.diff(X.indptr))
        tf = X.data
        W = X.copy()
        W.data = (idf[X.indices] * tf * (k1 + 1.0) /
                  (tf + k1 * (1.0 - b + b * dl[rows] / avgdl))).astype(np.float32)
        idx.bm25 = sparse.csc_matrix(W)
        del X, W

        idx.passages = cls._passages(refs, spec, embedder, export, log)

        log("[corpus] field statistics ...")
        projects: Dict[str, Dict[str, Dict[str, set]]] = {}
        for rec in records:
            proj = project_key(rec)
            for variant in ("original", "llm"):
                for field in TEXT_FIELDS:
                    v = field_value(rec, field, variant)
                    if _norm_value(v):
                        projects.setdefault(variant, {}).setdefault(field, {}) \
                                .setdefault(_value_key(v), set()).add(proj)
        idx.shared = {var: {f: {k: len(p) for k, p in d.items()} for f, d in fields.items()}
                      for var, fields in projects.items()}

        # Density anchors: the corpus's own 5th and 95th percentiles of ORIGINAL fields,
        # applied unchanged to anything scored later, including LLM proposals.
        for field in FIELDS:
            vals = []
            for rec in records:
                toks = word_tokens(field_text(field, field_value(rec, field, "original")))
                if toks:
                    vals.append(float(np.mean(idx.specificity(toks))))
            if vals:
                idx.density_anchors[field] = (float(np.percentile(vals, 5)),
                                              float(np.percentile(vals, 95)))

        terms = [t for rec in records for t in split_keywords(field_value(rec, "keywords", "original"))]
        words = [len(t.split()) for t in terms]
        idx.keyword_max_words = int(np.percentile(words, 99)) if words else 0

        rng = random.Random(int(spec["seed"]))
        pool = sorted({t.lower() for t in terms})
        idx.null_terms = rng.sample(pool, min(int(spec["coverage"]["null_terms"]), len(pool)))
        idx.null_term_vecs = embedder.queries(idx.null_terms)

        # Coherence baseline: the same fields from randomly drawn KOs, so "how similar
        # are this KO's title and description" can be compared with chance.
        n_pool = int(spec["coherence"]["null_pool"])
        donors = [r for r in records if all(field_text(f, field_value(r, f, "original"))
                                            for f in ("title", "description", "keywords"))]
        pool = rng.sample(donors, min(n_pool, len(donors)))
        for f in FIELDS:
            texts = [field_text(f, field_value(r, f, "original")) for r in pool]
            texts = [t for t in texts if t]
            if texts:
                idx.coherence_pool[f] = embedder.queries(texts)

        # KPEval's semantic diversity: mean pairwise similarity within a keyword set.
        # It has no natural scale with E5 (cosines sit high), so it is anchored to the
        # corpus's own 5th-95th percentiles over a sample of original keyword sets.
        kw_sets = [split_keywords(field_value(r, "keywords", "original")) for r in records]
        kw_sets = [k for k in kw_sets if len(k) >= 2]
        sample = rng.sample(kw_sets, min(int(spec["diversity"]["anchor_sample"]), len(kw_sets)))
        sims = [keyword_set_similarity(embedder.queries(k)) for k in sample]
        if sims:
            idx.keyword_sim_anchors = (float(np.percentile(sims, 5)), float(np.percentile(sims, 95)))

        idx.meta = {
            "records": n,
            "languages": dict(lang_counts.most_common()),
            "wordfreq_languages": list(idx.wordfreq_langs),
            "vocabulary": len(idx.vocab),
            "avg_doc_tokens": round(avgdl, 1),
            "density_anchors": idx.density_anchors,
            "keyword_max_words": idx.keyword_max_words,
            "keyword_sim_anchors": idx.keyword_sim_anchors,
        }
        return idx

    # ----------------------------------------------------------------- cache
    @staticmethod
    def _export_stamp(export: Path) -> List[str]:
        st = export.stat()
        return [str(export.resolve()), str(st.st_size), str(int(st.st_mtime))]

    @classmethod
    def passages_key(cls, export: Path, spec: Dict[str, Any]) -> str:
        """Embeddings depend only on the export, the model and the passage length."""
        parts = cls._export_stamp(export) + [spec["models"]["embedding"],
                                             str(spec["reference"]["max_chars"]),
                                             str(spec["reference"]["passage_chars"])]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]

    @classmethod
    def _passages(cls, refs: Sequence[str], spec: Dict[str, Any], embedder: Embedder,
                  export: Path, log=print) -> np.ndarray:
        """
        Passage embeddings, cached on their own. They are the expensive part of the
        index (about 15 minutes on CPU) and change far less often than the statistics,
        so a change to any other setting does not re-embed the corpus.
        """
        path = CACHE_DIR / "passages" / f"{cls.passages_key(export, spec)}.npy"
        if path.exists():
            log(f"[corpus] cached passage embeddings {path.stem}")
            return np.load(path)
        log(f"[corpus] embedding {len(refs)} passages with {embedder.name} (CPU; cached after the first run) ...")
        vecs = embedder.passages([r[:spec["reference"]["passage_chars"]] for r in refs], progress=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    @classmethod
    def fingerprint(cls, export: Path, spec: Dict[str, Any]) -> str:
        # Only the settings the index is built from; a change to scoring rules (lengths,
        # weights, platform rules) must not force a rebuild.
        keys = ("reference", "coverage", "coherence", "diversity", "bm25", "specificity", "seed")
        parts = cls._export_stamp(export) + [
            spec["models"]["embedding"],
            json.dumps({k: spec[k] for k in keys}, sort_keys=True)]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]

    @classmethod
    def load_or_build(cls, records: Sequence[Dict[str, Any]], export: Path,
                      spec: Dict[str, Any], embedder: Embedder, log=print) -> "CorpusIndex":
        key = cls.fingerprint(export, spec)
        path = CACHE_DIR / key / "index.pkl"
        if path.exists():
            log(f"[corpus] cached index {key}")
            with path.open("rb") as fh:
                return pickle.load(fh)
        idx = cls.build(records, spec, embedder, export, log=log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(idx, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (path.parent / "meta.json").write_text(json.dumps(
            {**idx.meta, "export": str(export), "fingerprint": key}, indent=1, default=str))
        log(f"[corpus] cached to {path.parent}")
        return idx


__all__ = [
    "CorpusIndex", "Embedder", "load_spec", "ko_id", "project_key", "field_value",
    "split_keywords", "field_text", "reference_text", "word_tokens", "has_letter",
    "detect_languages", "ko_language_text", "keyword_set_similarity", "FIELDS", "TEXT_FIELDS",
]
