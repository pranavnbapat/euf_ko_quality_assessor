# analyse_fields.py

"""
Field selection audit for KO search (JSON-only).

What this script does, end-to-end:
1) Loads the newest file under ../input (JSON array or JSONL of Knowledge Objects).
2) Normalises candidate fields to plain text per KO (lists/dicts -> space-joined strings).
3) Computes per-field corpus diagnostics: coverage, avg length, boilerplate proxy, average IDF.
4) Runs a self-retrieval test (MRR@TOPK): for each KO, query with its own title (and title+keywords)
   against candidate field sets using a minimal in-memory BM25.
5) Ranks single fields by a Field Usefulness Score (FUS) combining MRR, IDF, coverage, boilerplate.
6) Prints: diagnostics, SR-MRR per candidate set, FUS per single field, and a suggested schema.
"""

import json
import logging
import math
import numpy as np
import os
import re
import time

from collections import Counter
from datetime import datetime
from glob import glob
from itertools import combinations
from typing import List, Dict


# ---------- Config ----------
TOPK = 10  # MRR cutoff; approximates "first page" and keeps runtime sane
# This approximates "first page" relevance and keeps runtime sane for ~7k docs.

# ---------- Logging ----------
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("analyse_fields")

# ---------- Tokeniser ----------
_word = re.compile(r"\w+", re.UNICODE)

# Why: documents how and why tokenisation is deliberately simple.
# -----------------------------------------------------------------
# Tokeniser:
# - Lowercases and extracts \w+ "word-like" spans using a Unicode-aware regex.
# - Fast, language-agnostic, and consistent across all fields.
# - Keeps BM25 comparability fair by avoiding per-field NLP tricks.
# - Caveat: no stemming/lemmatisation and punctuation is dropped.
def tok(text: str) -> List[str]:
    """Minimal, language-agnostic tokeniser: lowercase + \\w+ regex."""
    return _word.findall(text.lower()) if text else []


# Select the newest input file under `../input`:
# - Works with either JSON array ([...]) or JSON Lines (one JSON per line).
# - Sorting by (mtime, filename) gives deterministic tie-breaking.
def latest_input_file(folder: str = "input") -> str:
    """
    Return the newest regular file in `folder`.
    We sort by modification time (mtime); ties break by filename.
    """
    candidates = [p for p in glob(os.path.join(folder, "*")) if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError(f"No files found in {folder}/")
    candidates.sort(key=lambda p: (os.path.getmtime(p), p))
    return candidates[-1]


# ---------- BM25 ----------
def bm25_prepare(docs_tokens: List[List[str]], k1=1.2, b=0.75):
    """
    Pre-compute BM25 stats for a "collection" defined by a specific field set:
    - df: document frequency per unique term (set per doc to avoid double-counting),
    - idf: Robertson/Sparck Jones IDF  (log((N - df + 0.5)/(df + 0.5) + 1)),
    - dl / avgdl: document length and average length (for length normalisation).
    """
    N = len(docs_tokens)
    df = Counter()
    dl = np.array([len(toks) for toks in docs_tokens], dtype=float)
    for toks in docs_tokens:
        if toks:
            df.update(set(toks))

    # Robertson/Sparck Jones IDF:
    # idf(t) = log( (N - df[t] + 0.5) / (df[t] + 0.5) + 1.0 )
    # "+1.0" inside log prevents negative/NaN if the ratio is tiny.
    idf = {t: math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0) for t in df}
    avgdl = float(dl.mean()) if N else 0.0
    return idf, dl, avgdl, k1, b


# ------------------------
# Analysis pipeline
# ------------------------
# ---------- Main pipeline ----------
def main():
    t_start = time.time()

    # Resolve ../input relative to this file so working dir doesn't matter.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "..", "input")
    log.info("Scanning for newest input under: %s", input_dir)

    INPUT_JSON = latest_input_file(input_dir)
    log.info("Using newest input file: %s", INPUT_JSON)

    # Load JSON (array or JSONL)
    t0 = time.time()
    with open(INPUT_JSON, "r", encoding="utf-8") as fh:
        first = fh.read(1); fh.seek(0)
        if first == "[":
            data = json.load(fh)
        else:
            data = [json.loads(line) for line in fh]
    log.info("Loaded %d KOs in %.1fs", len(data), time.time() - t0)

    # Optional dev throttle
    MAX_DOCS = int(os.getenv("AF_MAX_DOCS", "0"))
    if MAX_DOCS and len(data) > MAX_DOCS:
        data = data[:MAX_DOCS]
        log.warning("AF_MAX_DOCS=%d active: using first %d KOs", MAX_DOCS, len(data))

    # Candidate fields to inspect. Textual fields may participate in ranking.
    # Structured fields are for diagnostics (and later, facets/filters).
    FIELDS = [
        "title",
        "subtitle",
        "description",
        "ko_content_flat",
        "keywords",
        "topics",
        "themes",
        "languages",
        "locations_flat",
        "category",
        "subcategories",
        "license",
        "project_type",
        "date_of_completion",
        "creators",
        "project_acronym",
        "project_display_name",
        "project_name",
    ]

    # Normalise field values to strings
    def norm(v):
        if v is None:
            return ""
        if isinstance(v, list):
            # If list of dicts (e.g., locations), prefer 'name'
            if v and isinstance(v[0], dict):
                return " ".join([str(d.get("name", "")) for d in v])
            return " ".join(map(str, v))
        if isinstance(v, dict):
            return " ".join([str(x) for x in v.values()])
        return str(v)

    # Build per-field text arrays
    t0 = time.time()
    per_field_text: Dict[str, List[str]] = {f: [] for f in FIELDS}
    titles, keywords = [], []
    for ko in data:
        for f in FIELDS:
            per_field_text[f].append(norm(ko.get(f, "")))

        titles.append(norm(ko.get("title", "")))
        keywords.append(norm(ko.get("keywords", "")))

    log.info("Normalised fields for %d KOs in %.1fs", len(data), time.time() - t0)

    # Cache per-field tokens once; reuse in diagnostics and SR-MRR
    t0 = time.time()
    field_tokens: Dict[str, List[List[str]]] = {
        f: [tok(v) for v in per_field_text[f]] for f in FIELDS
    }
    log.info("Tokenised all fields once (cache built) in %.1fs", time.time() - t0)

    N = len(data)

    # ---- A) Per-field corpus diagnostics ----
    t0 = time.time()
    report = {}
    for f, vals in per_field_text.items():
        toks = field_tokens[f]
        coverage = sum(1 for v in vals if v.strip()) / (N or 1)
        avg_len = float(np.mean([len(t) for t in toks])) if N else 0.0
        # Boilerplate proxy
        uniq_ratio = len(set(vals)) / (N or 1)
        boilerplate = 1 - uniq_ratio

        # Average IDF over the field vocab
        idf, dl, avgdl, *_ = bm25_prepare(toks)
        avg_idf = float(np.mean(list(idf.values()))) if idf else 0.0

        report[f] = dict(
            coverage=coverage, avg_tokens=avg_len,
            boilerplate=boilerplate, avg_idf=avg_idf
        )
    log.info("Computed diagnostics for %d fields in %.1fs", len(report), time.time() - t0)

    # Self-retrieval @TOPK:
    # For the given field_list, we build a mini collection (docs_tokens),
    # precompute BM25 statistics, and build a tiny inverted index (term -> doc_ids, tfs).
    # Then for each KO (query = title or title+keywords), we score sparsely over postings,
    # compute the KO's rank among all docs, and accumulate MRR if rank ≤ TOPK.
    def sr_mrr(field_list: List[str], query_variant="Q1"):
        """
        Lossless but faster scoring: build an inverted index once for the chosen fields,
        then for each query only score docs that contain the query terms.
        """
        # 1) Concatenate tokens per KO for the chosen fields (no re-tokenising)
        docs_tokens = []
        for i in range(N):
            toks = []
            for f in field_list:
                toks.extend(field_tokens[f][i])
            docs_tokens.append(toks)

        # 2) BM25 collection stats
        idf, dl, avgdl, k1, b = bm25_prepare(docs_tokens)
        # Precompute document-wise denominator term for BM25
        den_vec = k1 * (1 - b + b * (dl / (avgdl or 1.0))) + 1e-12  # shape: (N,)

        # 3) Build a tiny inverted index: term -> (np.array(doc_ids), np.array(tfs))
        #    Linear in total tokens; built once per candidate.
        postings_docs: Dict[str, List[int]] = {}
        postings_tfs: Dict[str, List[int]] = {}
        for doc_id, toks in enumerate(docs_tokens):
            if not toks:
                continue
            tf = Counter(toks)
            for term, f in tf.items():
                postings_docs.setdefault(term, []).append(doc_id)
                postings_tfs.setdefault(term, []).append(f)
        # Convert lists to numpy arrays for vectorised scoring
        postings = {}
        for term in postings_docs.keys():
            postings[term] = (
                np.fromiter(postings_docs[term], dtype=np.int32),
                np.fromiter(postings_tfs[term], dtype=np.float64),
            )

        # 4) Score all queries
        mrr = 0.0
        for i in range(N):
            qtext = titles[i] if query_variant == "Q1" else (titles[i] + " " + keywords[i])
            q = tok(qtext)
            if not q:
                continue

            # Sparse scoring: only docs in the union of postings for query terms
            scores = np.zeros(N, dtype=np.float64)  # zeros for docs with no matching terms
            for term in set(q):  # set() keeps unique terms; BM25 uses term freq in doc, not q-freq
                w_idf = idf.get(term)
                if w_idf is None:
                    continue
                if term not in postings:
                    continue
                doc_ids, tfs = postings[term]  # arrays
                # BM25 contribution for this term over affected docs
                # s += idf * ((tf * (k1+1)) / (tf + denom))
                denom = den_vec[doc_ids]
                scores[doc_ids] += w_idf * ((tfs * (k1 + 1.0)) / (tfs + denom))

            # Early rejection: not in top-K (strictly below Kth largest score)
            if len(scores) > TOPK:
                kth = np.partition(scores, -TOPK)[-TOPK]
                if scores[i] < kth:
                    continue

            # Exact rank without sort: 1 + number of docs with strictly higher score
            s_i = scores[i]
            idxs = np.arange(N)
            rank = int(((scores > s_i).sum()) + ((scores == s_i) & (idxs < i)).sum() + 1)

            if rank <= TOPK:
                mrr += 1.0 / rank

        denom = max(1, N)
        return mrr / denom

    # Candidate generation:
    # - BASE_FIELDS are the superset of fields we'll consider at all.
    # - We first filter by COVERAGE_MIN (e.g. 0.10) to avoid sparse fields skewing results.
    BASE_FIELDS = [
        "title",
        "subtitle",
        "description",
        "ko_content_flat",
        "keywords",
        "topics",
        "themes",
        "languages",
        "locations_flat",
        "category",
        "subcategories",
        "license",
        "project_type",
        "date_of_completion",
        "creators",
        "project_acronym",
        "project_display_name",
        "project_name",
    ]

    COVERAGE_MIN = 0.10
    eligible = [f for f in BASE_FIELDS if report.get(f, {}).get("coverage", 0.0) >= COVERAGE_MIN]

    candidates: List[List[str]] = [[f] for f in eligible]  # singles

    # CORE: fields used to form most combinations.
    CORE = [f for f in ["title", "description", "ko_content_flat", "keywords"] if f in eligible]

    FAST = os.getenv("FAST_CANDIDATES", "0") == "1"
    # FAST mode:
    # Add only a few high-value combos (keeps runtime low for large corpora).
    # Full mode (else) explores 2-, 3-, and 4-field combinations from CORE.

    if FAST:
        if "title" in CORE:
            candidates.append(["title"])
        if all(f in CORE for f in ["title", "description"]):
            candidates.append(["title", "description"])
        if all(f in CORE for f in ["title", "description", "ko_content_flat"]):
            candidates.append(["title", "description", "ko_content_flat"])
        if all(f in CORE for f in ["title", "description", "ko_content_flat", "keywords"]):
            candidates.append(["title", "description", "ko_content_flat", "keywords"])
    else:
        # Full mode:
        # Explore 2-, 3-, and 4-field combinations from CORE
        # and optionally augment them with topics/themes.
        for r in (2, 3, 4):
            for combo in combinations(CORE, r):
                candidates.append(list(combo))

        LIGHTS = [f for f in ["topics", "themes"] if f in eligible]
        core_sets = [set(x) for x in candidates if set(x) <= set(CORE)]
        for core_set in core_sets:
            for g in LIGHTS:
                combo = sorted(core_set | {g})
                if combo not in candidates:
                    candidates.append(combo)

    # Deduplicate (order-insensitive)
    candidates = [list(t) for t in {tuple(sorted(c)) for c in candidates}]
    log.info(
        "Prepared %d candidates (FAST_CANDIDATES=%s, eligible=%s)",
        len(candidates), "1" if FAST else "0", ",".join(eligible) or "-"
    )

    # Evaluate candidates
    t0 = time.time()
    results = []
    for idx, combo in enumerate(candidates, start=1):
        mrr_q1 = sr_mrr(combo, "Q1")
        mrr_q2 = sr_mrr(combo, "Q2")
        results.append((tuple(combo), mrr_q1, mrr_q2))
        if idx % 10 == 0 or idx == len(candidates):
            elapsed = time.time() - t0
            log.info("Evaluated %d/%d candidates in %.1fs", idx, len(candidates), elapsed)

    # Min-max normaliser:
    # Scales a list to [0,1] while guarding division by zero with a small epsilon.
    def mm(vals):
        """Min-max normaliser."""
        lo, hi = min(vals), max(vals)
        return [(v - lo) / (hi - lo + 1e-12) for v in vals]

    singles = [f for f in ["title", "description", "ko_content_flat", "keywords", "topics", "themes"] if f in report]
    mrr1_map = {tuple([f]): next((r[1] for r in results if r[0] == (f,)), 0.0) for f in singles}
    # mrr2_map = {tuple([f]): next((r[2] for r in results if r[0] == (f,)), 0.0) for f in singles}

    mrr1N = dict(zip(singles, mm([mrr1_map[(f,)] for f in singles])))
    # mrr2N = dict(zip(singles, mm([mrr2_map[(f,)] for f in singles])))
    idfN  = dict(zip(singles, mm([report[f]["avg_idf"]     for f in singles])))
    covN  = dict(zip(singles, mm([report[f]["coverage"]    for f in singles])))
    boilN = dict(zip(singles, mm([report[f]["boilerplate"] for f in singles])))

    fus = {}
    for f in singles:
        # Field Usefulness Score (FUS):
        # Bias towards self-retrieval with title (0.70), with supporting signals:
        # - avg IDF (0.15): rarer terms help discrimination,
        # - coverage (0.10): more docs populated is better,
        # - boilerplate (-0.05): penalise repetitive field strings across docs.
        fus[f] = (0.70 * mrr1N[f] + 0.15 * idfN[f] + 0.10 * covN[f] - 0.05 * boilN[f])

    # ---------- Output ----------
    # ---------- Output ----------
    lines = []

    # 1) Diagnostics
    lines.append("\n=== FIELD DIAGNOSTICS ===")
    for f, m in sorted(report.items(), key=lambda x: (-x[1]["avg_idf"], x[0])):
        lines.append(f"{f:22} coverage={m['coverage']:.2f}  avg_tokens={m['avg_tokens']:.1f}  "
                     f"avg_idf={m['avg_idf']:.3f}  boilerplate={m['boilerplate']:.2f}")

    # 2) Self-retrieval scores
    lines.append(f"\n=== SELF-RETRIEVAL (MRR@{TOPK}) ===")
    for combo, m1, m2 in results:
        lines.append(f"{' + '.join(combo):40}  Q1(title)={m1:.4f}  Q2(title+kw)={m2:.4f}")

    # 3) FUS (singles only)
    lines.append("\n=== FIELD USEFULNESS SCORE (FUS) ===")
    for f in sorted(fus, key=fus.get, reverse=True):
        lines.append(f"{f:22}  FUS={fus[f]:.4f}")

    # 4) Suggested schema
    suggest_rank = ["title", "description", "ko_content_flat", "keywords",]
    suggest_facets = ["topics", "themes", "languages", "locations_flat", "category", "subcategories", "license",
                      "project_type", "date_of_completion", "creators", "project_acronym", "project_display_name",
                      "project_name",]
    lines.append("\n=== SUGGESTED USAGE ===")
    lines.append("Ranking fields (BM25F): " + ", ".join(suggest_rank))
    lines.append("Facet/filter fields    : " + ", ".join([f for f in suggest_facets if f in per_field_text]))
    lines.append("Display-only/skip      : project_url, project_doi, IDs, duplicates")

    # Join once
    report_text = "\n".join(lines)

    # Print to console (unchanged behaviour)
    print(report_text)

    # Also write to a timestamped file under ../reports
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir = os.path.join(base_dir, "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"field_audit_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    log.info("Wrote report to %s", out_path)
    log.info("Total elapsed: %.1fs", time.time() - t_start)

if __name__ == "__main__":
    main()

