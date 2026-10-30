# assess_ko_quality/core/metadata_metrics.py
"""
The ten metadata sub-pillars.

Each one takes a single metadata field (title, subtitle, description or keywords) and
returns a value in [0, 1] plus the evidence behind it, or None when it cannot be
evaluated for that field. Nothing is scored against an opinion: every sub-pillar is a
count, a rank, a model probability, or a published cut-off, and the same inputs always
give the same output.

"The file" below means the KO's reference text, set by `reference.source` in the spec:
its summary (ko_content_flat_summarised, the default) or the file's own text (see
corpus_index.reference_text).

    STRUCTURAL   length ........ within the published limits for the field
                 completeness .. present, not a filename, not a copy of another field,
                                 not shared verbatim with unrelated projects
                 noise ......... share of tokens that are markup, URLs, encoding debris,
                                 symbol runs or stutters
                 formatting .... field-specific form: casing, stray line breaks, keyword
                                 lists split and de-duplicated

    SEMANTIC     consistency ... accuracy: does the file support the field? (NLI, adapted
                                 from SummaC's zero-shot rule, Laban et al. 2022) and
                                 coherence: do the fields describe the same thing? (vs
                                 chance) - both as defined by Ochoa & Duval (2006)
                 density ....... how specific the words are: platform IDF and general
                                 word frequency in the field's language (wordfreq)
                 clarity ....... not visibly cut off, acronyms explained somewhere, readable
                                 (LIX, Bjornsson 1968)
                 usefulness .... covers the file's key terms, adds to the title rather
                                 than repeating it

    FUNCTIONAL   bm25 .......... search the whole corpus with the field: where does this
                 semantic ...... KO rank? (known-item retrievability, Azzopardi & Vinay
                                 2008; gain 1/log2(1+rank), the nDCG discount of
                                 Burges et al. 2005)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from core.corpus_index import (
    CorpusIndex, Embedder, field_text, has_letter, split_keywords, word_tokens,
)
from core.text_utils import norm_text

Result = Tuple[Optional[float], Dict[str, Any]]

SUBPILLARS = {
    "structural": ("length", "completeness", "noise", "formatting"),
    "semantic": ("consistency", "density", "clarity", "usefulness"),
    "functional": ("bm25_findability", "semantic_findability"),
}


# --------------------------------------------------------------------- NLI
class NLI:
    """Cross-lingual NLI. Returns P(contradiction), P(neutral), P(entailment) per pair."""

    def __init__(self, name: str):
        self.name = name
        self._tok = None
        self._model = None
        self._idx: Dict[str, int] = {}

    def _load(self):
        if self._model is None:
            import os
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(os.cpu_count() or 4)
            self._tok = AutoTokenizer.from_pretrained(self.name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.name).eval()
            labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
            self._idx = {"contradiction": labels["contradiction"],
                         "neutral": labels["neutral"], "entailment": labels["entailment"]}

    def probs(self, pairs: Sequence[Tuple[str, str]], batch: int = 16) -> np.ndarray:
        """Rows of (contradiction, neutral, entailment)."""
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        import torch

        self._load()
        out = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            enc = self._tok([p for p, _ in chunk], [h for _, h in chunk],
                            truncation="only_first", max_length=512,
                            padding=True, return_tensors="pt")
            with torch.no_grad():
                p = self._model(**enc).logits.softmax(-1).numpy()
            out.append(p[:, [self._idx["contradiction"], self._idx["neutral"],
                             self._idx["entailment"]]])
        return np.vstack(out)


# --------------------------------------------------------------------- per-KO context
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_words(text: str, size: int, limit: int) -> List[str]:
    """Sentence-packed chunks of about `size` words, at most `limit` of them."""
    chunks: List[str] = []
    cur: List[str] = []
    n = 0
    for s in sentences(text):
        w = s.split()
        if n and n + len(w) > size:
            chunks.append(" ".join(cur))
            cur, n = [], 0
            if len(chunks) >= limit:
                return chunks
        # A single sentence longer than a chunk (tables, run-on extraction) is cut.
        while len(w) > size:
            chunks.append(" ".join(w[:size]))
            w = w[size:]
            if len(chunks) >= limit:
                return chunks
        cur.extend(w)
        n += len(w)
    if cur and len(chunks) < limit:
        chunks.append(" ".join(cur))
    return chunks


@dataclass
class KOContext:
    """Everything about the file itself, computed once and shared by both variants."""
    pos: Optional[int]
    lang: str
    project: str
    ref_text: str
    ref_tokens: List[str]
    ref_token_set: set
    ref_lower: str
    chunks: List[str]
    chunk_vecs: np.ndarray
    key_terms: List[str]
    keyword_null_threshold: float
    evaluable: bool
    notes: List[str] = dc_field(default_factory=list)
    file_name: Any = None
    ref_lang: str = "unknown"          # detected from the reference text itself
    xling: Dict[str, Any] = dc_field(default_factory=dict)   # spec cross_lingual
    # The English version of a non-English reference (--resummarise on a non-English
    # document): the secondary reference for lexical checks on fields written in English.
    alt_lang: str = "unknown"
    alt_token_set: set = dc_field(default_factory=set)
    alt_chunk_vecs: Optional[np.ndarray] = None
    alt_null_threshold: float = 1.0


def build_context(rec: Dict[str, Any], idx: CorpusIndex, emb: Embedder,
                  spec: Dict[str, Any]) -> KOContext:
    from core.corpus_index import ko_id, project_key, reference_text

    cfg = spec["reference"]
    ref = reference_text(rec, cfg["max_chars"], cfg.get("source", "file"))
    toks = word_tokens(ref)
    pos = idx.pos.get(ko_id(rec))
    lang = idx.langs[pos] if pos is not None else "unknown"
    evaluable = len(ref) >= int(cfg["min_chars"])
    chunks = chunk_words(ref, int(cfg["chunk_words"]), int(cfg["max_chunks"])) if evaluable else []
    vecs = emb.passages(chunks) if chunks else np.zeros((0, 384), dtype=np.float32)

    # What "similar enough" means for a keyword and this file, learned from chance:
    # how close do randomly drawn corpus keywords get to this file's best passage?
    thr = 1.0
    if len(vecs) and idx.null_term_vecs is not None and len(idx.null_term_vecs):
        best = (idx.null_term_vecs @ vecs.T).max(axis=1)
        thr = float(np.percentile(best, spec["coverage"]["null_percentile"]))

    notes = [] if evaluable else [f"reference text under {cfg['min_chars']} characters"]
    if pos is None:
        notes.append("KO not in the corpus index; findability not evaluable")
    from core.corpus_index import _detect

    # Detected fresh from the reference text itself, not from the corpus build, so a
    # summary swapped in by --resummarise carries its own language here.
    ref_lang = _detect(ref[:1000])

    # The English version of a non-English reference, for fields written in English.
    alt_lang, alt_tokens, alt_vecs, alt_thr = "unknown", set(), None, 1.0
    alt = str(rec.get("_summary_en") or "").strip()
    if alt and alt != ref:
        alt_lang = _detect(alt[:1000])
        if alt_lang != ref_lang:
            alt_tokens = set(word_tokens(alt))
            alt_chunks = chunk_words(alt, int(cfg["chunk_words"]), int(cfg["max_chunks"]))
            if alt_chunks:
                alt_vecs = emb.passages(alt_chunks)
                if len(alt_vecs) and idx.null_term_vecs is not None and len(idx.null_term_vecs):
                    best = (idx.null_term_vecs @ alt_vecs.T).max(axis=1)
                    alt_thr = float(np.percentile(best, spec["coverage"]["null_percentile"]))
        else:
            alt_lang = "unknown"

    return KOContext(
        pos=pos, lang=lang, project=project_key(rec), ref_text=ref, ref_tokens=toks,
        ref_token_set=set(toks), ref_lower=ref.lower(), chunks=chunks, chunk_vecs=vecs,
        # The reference's own language joins the stopword filter, so a reference in a
        # language few corpus KOs share does not yield its function words as key terms.
        key_terms=idx.key_terms(toks, int(spec["coverage"]["key_terms"]),
                                extra_langs=(ref_lang,)) if evaluable else [],
        keyword_null_threshold=thr, evaluable=evaluable, notes=notes,
        file_name=rec.get((spec.get("title_rules") or {}).get("file_name_field", "ko_object_name")),
        ref_lang=ref_lang,
        xling=dict(spec.get("cross_lingual") or {}),
        alt_lang=alt_lang, alt_token_set=alt_tokens, alt_chunk_vecs=alt_vecs,
        alt_null_threshold=alt_thr,
    )


# --------------------------------------------------------------------- helpers
def _band(x: float, lo: float, hi: float) -> float:
    """1 inside [lo, hi]; falls linearly to 0 at zero below, and at 2*hi above."""
    if lo <= x <= hi:
        return 1.0
    if x < lo:
        return max(0.0, x / lo) if lo else 0.0
    return max(0.0, 1.0 - (x - hi) / hi) if hi else 0.0


def _fraction(checks: Dict[str, bool]) -> Optional[float]:
    return (sum(checks.values()) / len(checks)) if checks else None


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _gain(rank: Optional[float]) -> Optional[float]:
    return None if rank is None else 1.0 / math.log2(1.0 + rank)


def _raw_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return value if isinstance(value, str) else ""


# --------------------------------------------------------------------- structural
def _at_most(x: float, cap: float) -> float:
    """1 up to the cap, then linearly to 0 at twice the cap."""
    return 1.0 if x <= cap else max(0.0, 1.0 - (x - cap) / cap)


def _at_least(x: float, floor: float) -> float:
    return 1.0 if x >= floor else max(0.0, x / floor)


def length(field: str, value: Any, spec: Dict[str, Any]) -> Result:
    """
    Within the published limits for the field (see validation/metadata_spec.json):
    each limit gives 1 when met and falls off linearly outside it; the field scores
    the lowest of its limits. A field with no published limit is not evaluated here.
    """
    rule = spec["fields"][field]["length"]
    if rule.get("unit") == "items":
        n = len(split_keywords(value))
        return _band(n, rule["min"], rule["max"]), {"items": n, "range": [rule["min"], rule["max"]]}
    text = field_text(field, value)
    parts: Dict[str, float] = {}
    ev: Dict[str, Any] = {"chars": len(text), "words": len(text.split())}
    if "max_chars" in rule:
        parts["max_chars"] = _at_most(len(text), rule["max_chars"])
    if "min_chars" in rule:
        parts["min_chars"] = _at_least(len(text), rule["min_chars"])
    if "max_chars_no_spaces" in rule:
        ns = len(text.replace(" ", ""))
        ev["chars_no_spaces"] = ns
        parts["max_chars_no_spaces"] = _at_most(ns, rule["max_chars_no_spaces"])
    if "max_sentences" in rule:
        n_sent = len([s for s in sentences(text) if s.split()])
        ev["sentences"] = n_sent
        parts["max_sentences"] = _at_most(n_sent, rule["max_sentences"])
    ev["limits"] = {k: rule[k] for k in rule}
    ev["parts"] = {k: round(v, 3) for k, v in parts.items()}
    return (min(parts.values()) if parts else None), ev


_TERMINAL = tuple(".!?)]\"'»”’")
_FILE_EXT = re.compile(r"\.(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{3,4}$")


def _stem_of(filename: Any) -> str:
    name = str(filename or "").rsplit("/", 1)[-1]
    name = re.sub(r"\.[A-Za-z0-9]{2,5}$", "", name)
    return " ".join(re.sub(r"[^\w]+", " ", name.lower()).split())


def completeness(field: str, value: Any, others: Dict[str, Any], variant: str,
                 idx: CorpusIndex, spec: Optional[Dict[str, Any]] = None,
                 file_name: Any = None) -> Result:
    text = field_text(field, value)
    if not text:
        return 0.0, {"present": False}
    checks: Dict[str, bool] = {}
    rules = (spec or {}).get("title_rules") or {}
    if field == "title" and rules:
        # A project code or a file name is not "short and easily understandable"
        # (EIP-AGRI Common Format) nor "descriptive" (Google Search Central).
        checks["no_internal_project_reference"] = not re.search(
            rules["internal_reference_pattern"], text, re.I)
        stem = _stem_of(file_name)
        if stem:
            checks["not_the_file_name"] = " ".join(re.sub(r"[^\w]+", " ", text.lower()).split()) != stem
    if field != "keywords":
        no_space_id = " " not in text and bool(re.search(r"[_.]", text))
        checks["not_filename"] = not (no_space_id or _FILE_EXT.search(text))
        mine = " ".join(text.split()).lower()
        copies = [f for f, v in others.items()
                  if f != field and f != "keywords" and " ".join(field_text(f, v).split()).lower() == mine]
        checks["not_copy_of_other_field"] = not copies
        shared = idx.shared_projects(variant, field, value)
        checks["not_shared_across_projects"] = shared <= 1
    else:
        checks["has_terms"] = bool(split_keywords(value))
    return _fraction(checks), {"present": True, "checks": checks}


_HTML = re.compile(r"</?[A-Za-z][^>]*>|&[A-Za-z]+;|&#\d+;")
_URL = re.compile(r"https?://|www\.|\S+@\S+\.\w+", re.I)
_MOJIBAKE = re.compile(r"Ã[\u0080-¿]|â€|�|Â[ -¿]")
_CONTROL = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
# List-marker glyphs (•, ▪, ►) are not here: the platform asks for bullet points in
# descriptions, so a bullet is structure, not debris.
_JUNK_SINGLE = set("|■□★#*_=~^<>{}[]")


def noise(field: str, value: Any) -> Result:
    raw = ", ".join(split_keywords(value)) if field == "keywords" else _raw_text(value)
    if field == "description":
        from core.corpus_index import rich_text_to_plain
        raw = rich_text_to_plain(raw)     # editor HTML is structure, not debris
    toks = raw.split()
    if not toks:
        return None, {}
    kinds: Dict[str, int] = {}
    prev = ""
    for t in toks:
        kind = None
        if _HTML.search(t):
            kind = "markup"
        elif _URL.search(t):
            kind = "url"
        elif _MOJIBAKE.search(t) or _CONTROL.search(t):
            kind = "encoding"
        elif not any(c.isalnum() for c in t) and (len(t) >= 2 or t in _JUNK_SINGLE):
            kind = "symbols"
        elif has_letter(t) and t.lower().strip(".,;:") == prev:
            kind = "repeat"
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
        prev = t.lower().strip(".,;:")
    bad = sum(kinds.values())
    return 1.0 - bad / len(toks), {"tokens": len(toks), "noise_tokens": kinds}


def _cased(text: str) -> Tuple[int, int]:
    up = sum(1 for c in text if c.isupper())
    low = sum(1 for c in text if c.islower())
    return up, low


def formatting(field: str, value: Any, idx: CorpusIndex) -> Result:
    checks: Dict[str, bool] = {}
    if field == "keywords":
        items = value if isinstance(value, (list, tuple)) else ([value] if value else [])
        terms = split_keywords(value)
        if not terms:
            return None, {}
        low = [t.lower() for t in terms]
        checks["stored_as_list"] = isinstance(value, (list, tuple))
        checks["one_term_per_item"] = not any(re.search(r"[;,|]", str(i)) for i in items)
        checks["no_duplicates"] = len(set(low)) == len(low)
        checks["no_trivial_terms"] = all(len(t) >= 2 and has_letter(t) for t in terms)
        if idx.keyword_max_words:
            checks["no_sentence_terms"] = all(len(t.split()) <= idx.keyword_max_words for t in terms)
        checks["no_trailing_punctuation"] = not any(t[-1] in ".;:,!?" for t in terms)
        return _fraction(checks), {"checks": checks}

    raw = _raw_text(value)
    if field == "description":
        from core.corpus_index import rich_text_to_plain
        raw = rich_text_to_plain(raw)
    text = norm_text(raw)
    if not text:
        return None, {}
    up, lo = _cased(text)
    words = text.split()
    checks["clean_whitespace"] = raw == raw.strip() and "  " not in raw and "\t" not in raw
    # Text set in capitals is read more slowly (Tinker 1963), however short; a single
    # word is exempt because it is usually an acronym or a name.
    checks["not_all_caps"] = not (up + lo >= 4 and len(words) >= 2 and up / (up + lo) > 0.8)
    if field in ("title", "subtitle"):
        checks["no_line_breaks"] = "\n" not in raw
        # A title or subtitle ending in a comma, semicolon or colon is dangling - cut off
        # or unfinished. A final full stop is a matter of house style, not a fault.
        checks["no_dangling_punctuation"] = not text.endswith((",", ";", ":"))
        checks["not_all_lowercase"] = not (lo and not up)
    else:
        # A line break between two words of one sentence is a PDF copy artefact;
        # breaks between paragraphs are fine.
        mid = [m for m in re.finditer(r"[^\s.!?:;…]\n[ \t]*([^\W\d_])", raw) if m.group(1).islower()]
        checks["no_mid_sentence_breaks"] = not mid
        first = next((c for c in text if c.isalpha()), "")
        checks["starts_capitalised"] = (not first) or first.isupper() or not first.islower()
        checks["ends_with_sentence_punctuation"] = text.endswith(_TERMINAL)
    return _fraction(checks), {"checks": checks}


# --------------------------------------------------------------------- semantic
def _matches_language(text: str, lang: str, cfg: Dict[str, Any]) -> Optional[bool]:
    """
    Does the text read as words of `lang`? The one question lexical matching needs,
    answered via wordfreq: the share of alphabetic tokens that are known words of
    that language. Open-set detection (langdetect) misfires on short fields
    ("personal development, dairy farming" reads as French); the share test is
    robust and one-sided. None when it cannot be decided: too few tokens, an unknown
    language, or one wordfreq does not cover.
    """
    toks = [tok for tok in word_tokens(text) if tok.isalpha()]
    if len(toks) < int(cfg.get("min_tokens", 2)) or lang in ("", "unknown"):
        return None
    try:
        from wordfreq import available_languages, zipf_frequency

        if lang not in available_languages():
            return None
        floor = float(cfg.get("known_word_zipf", 2.0))
        known = sum(zipf_frequency(tok, lang) >= floor for tok in toks)
    except Exception:
        return None
    return known / len(toks) >= float(cfg.get("known_share", 0.5))


def cross_lingual(text: str, ctx: KOContext) -> bool:
    """
    True when the field clearly does not read as the reference text's language, so the
    purely lexical parts are not evaluable: left out of their means, never zeroed, like
    every other part that cannot be measured. An undecidable case never gates.
    """
    return _matches_language(text, ctx.ref_lang, ctx.xling) is False


def _accuracy(field: str, value: Any, text: str, ctx: KOContext, emb: Embedder, nli: NLI,
              spec: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Accuracy in Ochoa & Duval's sense: how close the field is to what the file says.

    Text fields: each claim (the title, the subtitle, each description sentence) is
    checked against the two passages of the file most similar to it. The claim scores
    max(P(entailment) - P(contradiction)) over those passages, adapted from SummaC's
    zero-shot rule (max over premises, mean over claims); the field scores the mean,
    mapped from [-1, 1] to [0, 1].
    Keywords: the share of terms that occur in the file or sit closer to it than
    chance, where chance is learned from random corpus keywords. A term absent from
    the text can still be right (Meng et al. 2017's "absent keyphrases").
    """
    if field == "keywords":
        # Measured against whichever summary the keywords share a language with: the
        # reference, or its English version (--resummarise on a non-English document).
        against = None
        if not cross_lingual(text, ctx):
            against = (ctx.ref_token_set, ctx.chunk_vecs, ctx.keyword_null_threshold, None)
        elif ctx.alt_chunk_vecs is not None and len(ctx.alt_chunk_vecs) and \
                _matches_language(text, ctx.alt_lang, ctx.xling):
            against = (ctx.alt_token_set, ctx.alt_chunk_vecs, ctx.alt_null_threshold,
                       f"measured against the English version of the summary ({ctx.alt_lang})")
        if against is None:
            return None, {"reason": f"keywords are in neither the reference's language "
                                    f"({ctx.ref_lang}) nor its English version's; the "
                                    "literal match and the same-language chance "
                                    "threshold are not evaluable"}
        token_set, chunk_vecs, thr, note = against
        terms = split_keywords(value)
        vecs = emb.queries(terms)
        sims = (vecs @ chunk_vecs.T).max(axis=1)
        supported = []
        for t, s in zip(terms, sims):
            toks = word_tokens(t)
            lexical = bool(toks) and all(x in token_set for x in toks)
            supported.append(lexical or float(s) > thr)
        ev = {"unsupported": [t for t, ok in zip(terms, supported) if not ok],
              "threshold": round(thr, 4)}
        if note:
            ev["against"] = note
        return sum(supported) / len(terms), ev

    if field in ("title", "subtitle"):
        claims = [f"This text is about {text.rstrip('.')}."]
    else:
        claims = [s for s in sentences(text) if len(s.split()) >= 3]
        claims = claims[: int(spec["reference"]["max_description_sentences"])] or [text]
    k = int(spec["reference"]["premises_per_hypothesis"])
    qv = emb.queries(claims)
    order = np.argsort(-(qv @ ctx.chunk_vecs.T), axis=1)[:, :k]
    pairs = [(ctx.chunks[j], c) for c, row in zip(claims, order) for j in row]
    p = nli.probs(pairs)
    width = order.shape[1]
    per_claim = [float((p[i * width:(i + 1) * width, 2] - p[i * width:(i + 1) * width, 0]).max())
                 for i in range(len(claims))]
    return (float(np.mean(per_claim)) + 1.0) / 2.0, {
        "claims": len(claims),
        "weakest": min(zip(per_claim, claims))[1][:160],
        "per_claim": [round(x, 3) for x in per_claim]}


def _coherence(field: str, text: str, others: Dict[str, Any], emb: Embedder,
               idx: CorpusIndex) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Coherence in Ochoa & Duval's sense: do title, description and keywords describe the
    same thing? Raw similarity has no natural scale, so it is compared with chance: the
    share of randomly drawn KOs whose corresponding field is LESS similar to this field
    than this KO's own. 1.0 means the fields belong together more clearly than any
    random pairing; 0.5 means no better than chance.
    """
    mine = emb.queries([text])[0]
    parts: Dict[str, float] = {}
    for g, v in others.items():
        if g == field:
            continue
        other = field_text(g, v)
        pool = idx.coherence_pool.get(g)
        if not other or pool is None or not len(pool):
            continue
        own = float(mine @ emb.queries([other])[0])
        null = pool @ mine
        parts[g] = float(((null < own).sum() + 0.5 * (null == own).sum()) / len(null))
    if not parts:
        return None, {}
    return float(np.mean(list(parts.values()))), {k: round(v, 3) for k, v in parts.items()}


def consistency(field: str, value: Any, ctx: KOContext, others: Dict[str, Any],
                emb: Embedder, nli: NLI, idx: CorpusIndex, spec: Dict[str, Any]) -> Result:
    """
    Bruce & Hillmann's "accuracy" and "logical consistency and coherence", as
    defined by Ochoa & Duval (2006): the mean of agreement with the file and
    agreement with the KO's other descriptive fields.
    """
    text = field_text(field, value)
    if not text:
        return None, {}
    parts: Dict[str, float] = {}
    detail: Dict[str, Any] = {}
    if ctx.evaluable and len(ctx.chunk_vecs):
        acc, detail["accuracy"] = _accuracy(field, value, text, ctx, emb, nli, spec)
        if acc is not None:
            parts["accuracy"] = acc
    coh, detail["coherence"] = _coherence(field, text, others, emb, idx)
    if coh is not None:
        parts["coherence"] = coh
    if not parts:
        return None, detail
    detail["parts"] = {k: round(v, 3) for k, v in parts.items()}
    return float(np.mean(list(parts.values()))), detail


def density(field: str, value: Any, idx: CorpusIndex) -> Result:
    toks = word_tokens(field_text(field, value))
    if not toks:
        return None, {}
    raw = float(np.mean(idx.specificity(toks)))
    lo, hi = idx.density_anchors.get(field, (0.0, 1.0))
    score = min(1.0, max(0.0, (raw - lo) / (hi - lo))) if hi > lo else 0.5
    return score, {"mean_specificity": round(raw, 4), "corpus_p5_p95": [round(lo, 4), round(hi, 4)]}


_ACRONYM = re.compile(r"^[A-Z0-9][A-Z0-9&\-]*[A-Z0-9]$")


def _is_acronym(tok: str) -> bool:
    return bool(_ACRONYM.match(tok)) and sum(c.isupper() for c in tok) >= 2


def _explained(acronym: str, words: Sequence[str]) -> bool:
    """
    Is there a run of words whose initials spell the acronym?

    A simplified Schwartz & Hearst (2003): letters are matched in order to word
    initials, short words that do not match ("of", "and", "de") may be skipped, and the
    run may not be longer than twice the acronym plus two.
    """
    letters = [c.lower() for c in acronym if c.isalpha()]
    if len(letters) < 2:
        return True
    limit = 2 * len(letters) + 2
    for i, w in enumerate(words):
        if not w or w[0].lower() != letters[0]:
            continue
        k, j = 1, i + 1
        while k < len(letters) and j < len(words) and j - i < limit:
            wj = words[j]
            if wj and wj[0].lower() == letters[k]:
                k += 1
            elif len(wj) > 3:
                break
            j += 1
        if k == len(letters):
            return True
    return False


def _lix(text: str) -> Optional[float]:
    words = [w for w in re.findall(r"[^\W\d_]+", text)]
    if len(words) < 10:
        return None
    sents = max(1, len(re.findall(r"[.!?:;]+(?:\s|$)", text)))
    long_ = sum(1 for w in words if len(w) > 6)
    return len(words) / sents + 100.0 * long_ / len(words)


def clarity(field: str, value: Any, ctx: KOContext, others: Dict[str, Any],
            spec: Dict[str, Any]) -> Result:
    text = field_text(field, value)
    if not text:
        return None, {}
    parts: Dict[str, float] = {}
    detail: Dict[str, Any] = {}

    if field != "keywords":
        # A visible cut: trailing ellipsis or dangling punctuation. A description that
        # merely lacks its final full stop is a formatting fault, scored there.
        cut = text.endswith(("...", "…")) or text[-1] in ",;:-–—("
        parts["not_truncated"] = 0.0 if cut else 1.0

    toks = [t.strip(".,;:()[]\"'") for t in text.split()]
    toks = [t for t in toks if t]
    acronyms = [t for t in toks if _is_acronym(t)]
    if acronyms:
        context = " ".join([ctx.ref_text] + [field_text(f, v) for f, v in others.items() if f != field])
        words = re.findall(r"[^\W\d_][\w'-]*", context)
        unexplained = sorted({a for a in acronyms if not _explained(a, words)})
        n_unexp = sum(1 for a in acronyms if a in unexplained)
        parts["acronyms_explained"] = 1.0 - n_unexp / len(toks)
        detail["unexplained_acronyms"] = unexplained
    else:
        parts["acronyms_explained"] = 1.0

    if field == "description" and len(text.split()) >= int(spec["readability"]["min_words"]):
        lix = _lix(text)
        if lix is not None:
            cfg = spec["readability"]
            full, zero = cfg["lix_full_score_at_or_below"], cfg["lix_zero_score_at_or_above"]
            parts["readability"] = 1.0 if lix <= full else 0.0 if lix >= zero else (zero - lix) / (zero - full)
            detail["lix"] = round(lix, 1)

    detail["parts"] = {k: round(v, 3) for k, v in parts.items()}
    return float(np.mean(list(parts.values()))), detail


def usefulness(field: str, value: Any, ctx: KOContext, others: Dict[str, Any],
               emb: Embedder, idx: CorpusIndex, spec: Dict[str, Any]) -> Result:
    """
    Does the field carry what the file is about, and add to the title?

    Coverage: of the file's ten most characteristic words (TF-IDF in its own language),
    how many the field carries - literally, or by meaning when the field is in another
    language. "By meaning" has to beat chance: the 95th percentile similarity of
    random corpus keywords to this same field.
    """
    text = field_text(field, value)
    if not text:
        return None, {}
    parts: Dict[str, float] = {}
    detail: Dict[str, Any] = {}

    if ctx.evaluable and ctx.key_terms and cross_lingual(text, ctx):
        detail["coverage"] = (f"not evaluable: field not in the reference's language "
                              f"({ctx.ref_lang})")
    elif ctx.evaluable and ctx.key_terms:
        ftoks = set(word_tokens(text))
        fvec = emb.passages([text])[0]
        null = idx.null_term_vecs @ fvec if idx.null_term_vecs is not None else np.array([1.0])
        thr = float(np.percentile(null, spec["coverage"]["null_percentile"]))
        tvecs = emb.queries(ctx.key_terms)
        sims = tvecs @ fvec
        covered = [t for t, s in zip(ctx.key_terms, sims) if t in ftoks or float(s) > thr]
        parts["coverage"] = len(covered) / len(ctx.key_terms)
        detail["key_terms"] = ctx.key_terms
        detail["missed"] = [t for t in ctx.key_terms if t not in covered]

    if field == "keywords":
        # Stemmer of the reference's language (the document's own with --resummarise).
        div, detail["diversity"] = keyword_diversity(
            value, ctx.ref_lang if ctx.ref_lang != "unknown" else ctx.lang, emb, idx)
        if div is not None:
            parts["diversity"] = div

    if field in ("subtitle", "description") and others.get("title"):
        overlap = _jaccard(set(word_tokens(text)), set(word_tokens(field_text("title", others["title"]))))
        parts["adds_to_title"] = 1.0 - overlap
        detail["overlap_with_title"] = round(overlap, 3)

    if not parts:
        return None, detail
    detail["parts"] = {k: round(v, 3) for k, v in parts.items()}
    return float(np.mean(list(parts.values()))), detail


_SNOWBALL = {"ar": "arabic", "da": "danish", "nl": "dutch", "en": "english", "fi": "finnish",
             "fr": "french", "de": "german", "hu": "hungarian", "it": "italian",
             "no": "norwegian", "nb": "norwegian", "pt": "portuguese", "ro": "romanian",
             "ru": "russian", "es": "spanish", "sv": "swedish"}


def keyword_diversity(value: Any, lang: str, emb: Embedder,
                      idx: CorpusIndex) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    KPEval's diversity aspect (Wu, Yin & Chang, Findings of ACL 2024), which it takes
    from Bahuleyan & El Asri (2020): a keyword set should name distinct concepts.

      lexical:  1 - dup_token_ratio, the share of tokens that repeat after stemming
                (Snowball stemmer for the KO's language where one exists)
      semantic: KPEval's emb_sim - mean pairwise similarity of the keywords - placed
                between the corpus's 5th and 95th percentiles; lower similarity is
                more diverse
    """
    from core.corpus_index import keyword_set_similarity

    terms = split_keywords(value)
    if len(terms) < 2:
        return None, {}
    stem = None
    if lang in _SNOWBALL:
        from nltk.stem.snowball import SnowballStemmer
        stem = SnowballStemmer(_SNOWBALL[lang]).stem
    toks = [stem(t) if stem else t for term in terms for t in word_tokens(term)]
    dup_ratio = 1.0 - len(set(toks)) / len(toks) if toks else 0.0
    sim = keyword_set_similarity(emb.queries(terms))
    lo, hi = idx.keyword_sim_anchors
    semantic = min(1.0, max(0.0, (hi - sim) / (hi - lo))) if hi > lo else 0.5
    return (float(np.mean([1.0 - dup_ratio, semantic])),
            {"dup_token_ratio": round(dup_ratio, 3), "emb_sim": round(sim, 4),
             "stemmer": _SNOWBALL.get(lang)})


# --------------------------------------------------------------------- functional
def bm25_findability(field: str, value: Any, ctx: KOContext, idx: CorpusIndex) -> Result:
    if ctx.pos is None or not ctx.evaluable:
        return None, {"reason": "not evaluable"}
    text = field_text(field, value)
    toks = word_tokens(text)
    if not toks:
        return None, {}
    # Findability is a corpus property, and the corpus is indexed on the export's
    # (English) summaries: a query in a language only this KO's row carries would rank
    # first without competition. So the gate is the corpus's language, not the
    # reference's; the same thing while the reference is English.
    corpus_lang = idx.corpus_language
    if _matches_language(text, corpus_lang, ctx.xling) is False:
        return None, {"reason": f"field not in the corpus index's language ({corpus_lang}); "
                                "lexical search not evaluable"}
    rank = idx.rank_bm25(toks, ctx.pos)
    return _gain(rank), {"rank": rank, "of": idx.n_docs}


def semantic_findability(field: str, value: Any, ctx: KOContext, idx: CorpusIndex,
                         emb: Embedder) -> Result:
    if ctx.pos is None or not ctx.evaluable:
        return None, {"reason": "not evaluable"}
    text = field_text(field, value)
    if not text:
        return None, {}
    rank = idx.rank_semantic(emb.queries([text])[0], ctx.pos)
    return _gain(rank), {"rank": rank, "of": idx.n_docs}


__all__ = [
    "SUBPILLARS", "NLI", "KOContext", "build_context",
    "length", "completeness", "noise", "formatting",
    "consistency", "density", "clarity", "usefulness",
    "bm25_findability", "semantic_findability", "cross_lingual", "_matches_language",
]
