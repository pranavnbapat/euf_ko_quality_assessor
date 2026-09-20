"""
Generic JSON field analyzer for choosing OpenSearch index fields.

What it does:
1) Loads either a specific JSON/JSONL file or the newest file under ../input.
2) Extracts a record list from common JSON shapes such as:
   - [ {...}, {...} ]
   - { "docs": [ {...}, ... ] }
   - JSONL with one object per line
3) Discovers candidate fields from the records automatically.
4) Computes per-field diagnostics: coverage, cardinality, token stats, distinctiveness.
5) Infers likely OpenSearch-oriented field roles:
   - full-text ranking fields
   - facet/filter fields
   - sort fields
   - skip/store-only fields
6) Optionally runs a self-retrieval proxy when a likely query field exists.
7) Prints and writes a report with mapping recommendations.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import time

from collections import Counter, defaultdict
from datetime import datetime
from glob import glob
from typing import Any

import numpy as np


TOPK = 10
MAX_DISCOVERED_FIELDS = int(os.getenv("AF_MAX_FIELDS", "200"))
MAX_NESTED_DEPTH = int(os.getenv("AF_MAX_DEPTH", "1"))
MAX_ENUM_CARDINALITY = int(os.getenv("AF_MAX_ENUM_CARDINALITY", "200"))
MIN_COVERAGE = float(os.getenv("AF_MIN_COVERAGE", "0.05"))

# --- Facet shape (absolute counts, deliberately NOT ratios) -----------------
# A facet is judged by how many distinct values it has, because that is what a
# facet UI has to render. Ratios like distinct/n_docs are corpus-size dependent:
# the same vocabulary scores differently on a 300-doc sample and a 13k export.
FACET_MIN_VALUES = int(os.getenv("AF_FACET_MIN_VALUES", "2"))
FACET_IDEAL_MAX = int(os.getenv("AF_FACET_IDEAL_MAX", "40"))
FACET_USABLE_MAX = int(os.getenv("AF_FACET_USABLE_MAX", "200"))

# --- Pseudo-query benchmark -------------------------------------------------
BENCH_QUERIES = int(os.getenv("AF_BENCH_QUERIES", "1500"))
BENCH_QUERY_TERMS = int(os.getenv("AF_QUERY_TERMS", "8"))
BENCH_SEED = int(os.getenv("AF_SEED", "13"))
# Fields the pseudo-queries are drawn from. They are removed from the pool if
# they also appear in a bundle being evaluated (see select_query_source_fields).
QUERY_SOURCE_PREFERENCE = [
    field.strip()
    for field in os.getenv(
        "AF_QUERY_SOURCE",
        "ko_content_flat,title,keywords,description,subtitle",
    ).split(",")
    if field.strip()
]

# --- Role assignment cutoffs ------------------------------------------------
FULL_TEXT_THRESHOLD = float(os.getenv("AF_FULL_TEXT_THRESHOLD", "0.40"))
FACET_THRESHOLD = float(os.getenv("AF_FACET_THRESHOLD", "0.35"))
SORT_THRESHOLD = float(os.getenv("AF_SORT_THRESHOLD", "0.35"))

# --- Policy layer -----------------------------------------------------------
# These lists are a decision, not a measurement: prefer LLM-improved metadata
# over the originals, keep fingerprints and URLs out of the index. The scores
# above gate them, and every case where policy and evidence disagree is
# reported rather than silently resolved.
POLICY_FULL_TEXT_PRIORITY = [
    "title_llm",
    "subtitle_llm",
    "description_llm",
    "keywords_llm",
    "ko_content_flat_summarised",
]
POLICY_FULL_TEXT_FALLBACKS = ["project_name", "project_acronym"]
POLICY_FACETS = [
    "themes",
    "subcategories",
    "locations_flat",
    "languages",
    "project_id",
    "project_acronym",
    "date_of_completion",
    "ko_created_at",
    "ko_updated_at",
    "creators",
    "category",
    "project_type",
    "license",
]
POLICY_EXCLUDED = [
    "ko_content_flat",
    "title",
    "subtitle",
    "description",
    "keywords",
]

FULL_TEXT_ROLES = {"full_text", "index_optional"}
FACET_ROLES = {"facet_filter", "sort_filter"}

# Tokens too generic to make a useful pseudo-query term.
_QUERY_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "have", "has", "had", "not", "but", "its", "it", "of", "in", "on", "to",
    "as", "by", "at", "an", "a", "is", "be", "or", "can", "will", "which",
    "their", "there", "these", "those", "than", "then", "also", "such", "more",
}

logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("analyse_fields")

_WORD = re.compile(r"\w+", re.UNICODE)
_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"([tT ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?([zZ]|[+-]\d{2}:\d{2})?)?$"
)
_LABEL_KEYS = ("name", "title", "label", "value", "display_name", "id")
_COMMON_RECORD_KEYS = ("docs", "items", "records", "data", "results", "hits")


def tok(text: str) -> list[str]:
    return _WORD.findall(text.lower()) if text else []


def latest_input_file(folder: str = "input") -> str:
    candidates = [p for p in glob(os.path.join(folder, "*")) if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError(f"No files found in {folder}/")
    candidates.sort(key=lambda p: (os.path.getmtime(p), p))
    return candidates[-1]


def first_non_whitespace_char(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        while True:
            ch = fh.read(1)
            if not ch:
                return ""
            if not ch.isspace():
                return ch


def looks_like_record(obj: Any) -> bool:
    return isinstance(obj, dict) and any(isinstance(k, str) for k in obj.keys())


def looks_like_record_list(obj: Any) -> bool:
    # Checked over the whole list, not a 20-element prefix: a stray non-dict
    # further in would otherwise surface much later as an AttributeError
    # inside flatten_record.
    return isinstance(obj, list) and bool(obj) and all(isinstance(x, dict) for x in obj)


def extract_records(root: Any) -> list[dict[str, Any]]:
    if looks_like_record_list(root):
        return root

    if isinstance(root, dict):
        for key in _COMMON_RECORD_KEYS:
            value = root.get(key)
            if looks_like_record_list(value):
                return value

        list_values = [value for value in root.values() if looks_like_record_list(value)]
        if len(list_values) == 1:
            return list_values[0]

        if looks_like_record(root):
            return [root]

    raise ValueError("Could not extract a list of record objects from the input JSON")


def load_records(path: str) -> list[dict[str, Any]]:
    first = first_non_whitespace_char(path)
    t0 = time.time()

    if first in ("[", "{"):
        # A JSONL file also starts with "{", so a failed whole-file parse is
        # the signal to retry line by line rather than an error to report.
        try:
            with open(path, "r", encoding="utf-8") as fh:
                root = json.load(fh)
        except json.JSONDecodeError:
            root = None
        if root is not None:
            records = extract_records(root)
            log.info("Loaded %d records from JSON in %.1fs", len(records), time.time() - t0)
            return records

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} is neither valid JSON nor JSONL (line {line_no}: {exc})") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            records.append(obj)
    log.info("Loaded %d records from JSONL in %.1fs", len(records), time.time() - t0)
    return records


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def classify_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return "empty"
        if _DATE_RE.match(s):
            return "date"
        return "string"
    return "other"


def summarize_list_kind(values: list[Any]) -> str:
    if not values:
        return "empty_list"
    scalar_kinds = [classify_scalar(v) for v in values if not isinstance(v, (list, tuple, set, dict))]
    if len(scalar_kinds) == len(values):
        kinds = set(k for k in scalar_kinds if k != "empty")
        if not kinds:
            return "empty_list"
        if kinds <= {"string"}:
            return "string_list"
        if kinds <= {"date"}:
            return "date_list"
        if kinds <= {"integer"}:
            return "integer_list"
        if kinds <= {"integer", "float"}:
            return "float_list"
        if kinds <= {"boolean"}:
            return "boolean_list"
        return "mixed_scalar_list"
    if all(isinstance(v, dict) for v in values):
        return "object_list"
    return "mixed_list"


def describe_value(value: Any) -> str:
    if is_empty(value):
        return "empty"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, (list, tuple, set)):
        return summarize_list_kind(list(value))
    return classify_scalar(value)


def flatten_dict(value: dict[str, Any], prefix: str, depth: int, out: dict[str, Any]) -> None:
    if depth >= MAX_NESTED_DEPTH:
        out[prefix] = value
        return

    emitted_children = 0
    for key, child in value.items():
        child_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            before = len(out)
            flatten_dict(child, child_key, depth + 1, out)
            emitted_children += len(out) - before
        else:
            out[child_key] = child
            emitted_children += 1

    # Only fall back to storing the container itself when nothing was emitted
    # for it. Counting dict children too avoids reporting the parent object
    # *and* its flattened children as two separate fields.
    if emitted_children == 0 and prefix:
        out[prefix] = value


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, dict):
            flatten_dict(value, str(key), 0, out)
        else:
            out[str(key)] = value
    return out


def list_item_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in _LABEL_KEYS:
            if key in value and value[key] not in (None, ""):
                return str(value[key])
        scalar_bits = [str(v) for v in value.values() if not isinstance(v, (dict, list, tuple, set)) and v not in (None, "")]
        return " ".join(scalar_bits)
    return str(value)


def normalise_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        bits = [list_item_to_text(v) for v in value.values()]
        return " ".join(bit for bit in bits if bit).strip()
    if isinstance(value, (list, tuple, set)):
        bits = [list_item_to_text(v) for v in value]
        return " ".join(bit for bit in bits if bit).strip()
    return str(value).strip()


def atomic_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        text = normalise_text(value)
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        atoms: list[str] = []
        for item in value:
            atoms.extend(atomic_values(item))
        return [x for x in atoms if x]
    return [str(value)]


def bm25_prepare(docs_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75) -> tuple[dict[str, float], np.ndarray, float, float, float]:
    n_docs = len(docs_tokens)
    df = Counter()
    dl = np.array([len(tokens) for tokens in docs_tokens], dtype=float)
    for tokens in docs_tokens:
        if tokens:
            df.update(set(tokens))
    idf = {term: math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1.0) for term in df}
    avgdl = float(dl.mean()) if n_docs else 0.0
    return idf, dl, avgdl, k1, b


class Bm25Index:
    """Postings-list BM25 over a fixed set of documents.

    Extracted so the field audit, the pseudo-query benchmark and the judged
    evaluator all score with exactly the same implementation.
    """

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.n_docs = len(docs_tokens)
        self.idf, dl, avgdl, self.k1, self.b = bm25_prepare(docs_tokens, k1=k1, b=b)
        self.den_vec = self.k1 * (1 - self.b + self.b * (dl / (avgdl or 1.0))) + 1e-12

        postings_docs: dict[str, list[int]] = defaultdict(list)
        postings_tfs: dict[str, list[int]] = defaultdict(list)
        for doc_id, tokens in enumerate(docs_tokens):
            if not tokens:
                continue
            for term, freq in Counter(tokens).items():
                postings_docs[term].append(doc_id)
                postings_tfs[term].append(freq)

        self.postings = {
            term: (
                np.fromiter(doc_ids, dtype=np.int32, count=len(doc_ids)),
                np.fromiter(postings_tfs[term], dtype=np.float64, count=len(doc_ids)),
            )
            for term, doc_ids in postings_docs.items()
        }
        self.empty = not self.postings

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term in set(query_tokens):
            weight = self.idf.get(term)
            if weight is None:
                continue
            hit = self.postings.get(term)
            if hit is None:
                continue
            hit_doc_ids, tfs = hit
            denom = self.den_vec[hit_doc_ids]
            scores[hit_doc_ids] += weight * ((tfs * (self.k1 + 1.0)) / (tfs + denom))
        return scores

    def top_k(self, query_tokens: list[str], k: int) -> list[int]:
        if not query_tokens or self.empty:
            return []
        return top_k_positive(self.score(query_tokens), k)


def top_k_positive(scores: np.ndarray, k: int) -> list[int]:
    """Top-k document ids, considering only documents that actually matched.

    Plain ``argsort(-scores)[:k]`` pads the result with whatever documents sit
    at the front of the corpus whenever fewer than k documents match, because
    every non-matching document ties at 0.0. Those padded ids are then scored
    as if they had been retrieved.
    """
    hit_ids = np.flatnonzero(scores > 0.0)
    if hit_ids.size == 0:
        return []
    order = np.argsort(-scores[hit_ids], kind="stable")[:k]
    return hit_ids[order].tolist()


def saturating(value: float, midpoint: float) -> float:
    """Map a non-negative quantity into [0, 1) with no dependence on the corpus.

    Used instead of min-max scaling for absolute quantities (token counts),
    so a field's score does not change just because some *other* field in the
    same file happened to be the longest one.
    """
    if value <= 0.0 or midpoint <= 0.0:
        return 0.0
    return math.log1p(value) / math.log1p(midpoint) if value < midpoint else 1.0


def normalised_idf(avg_idf: float, n_docs: int) -> float:
    """IDF rescaled by corpus size so it is comparable across input files."""
    ceiling = math.log(n_docs + 1.0)
    if ceiling <= 0.0:
        return 0.0
    return max(0.0, min(1.0, avg_idf / ceiling))


def facet_cardinality_fitness(distinct: int) -> float:
    """How well a distinct-value count suits a facet control.

    Scale-invariant by construction: it reads the absolute number of values a
    user would have to choose between, not distinct/n_docs.
    """
    if distinct < FACET_MIN_VALUES:
        return 0.0  # one value filters nothing
    if distinct <= FACET_IDEAL_MAX:
        return 1.0
    if distinct >= FACET_USABLE_MAX:
        return 0.0
    span = FACET_USABLE_MAX - FACET_IDEAL_MAX
    return max(0.0, 1.0 - (distinct - FACET_IDEAL_MAX) / span)


def facet_reuse(mean_value_frequency: float) -> float:
    """How much work one facet value does, i.e. how many documents it selects.

    A field whose values never repeat is a label, not a filter: picking a value
    would return a single document. Absolute cardinality alone cannot see this
    on a small corpus, where "87 distinct values" looks like a usable facet even
    though there are only 87 records.
    """
    if mean_value_frequency <= 1.0:
        return 0.0
    return min(1.0, math.log(mean_value_frequency) / math.log(10.0))


def normalised_entropy(counts: list[int]) -> float:
    """Shannon entropy of a value distribution, normalised to [0, 1].

    This is what "balanced facet" should mean: 1.0 when documents spread evenly
    over the values, near 0.0 when one value swallows the corpus.
    """
    total = sum(counts)
    if total <= 0 or len(counts) < 2:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return max(0.0, min(1.0, entropy / math.log(len(counts))))


def field_mapping_hint(kind: str, avg_tokens: float, cardinality_ratio: float, array_ratio: float) -> str:
    if kind in {"string", "string_list"}:
        if avg_tokens >= 6:
            return "text"
        if array_ratio > 0.5 or cardinality_ratio < 0.95:
            return "keyword"
        return "text+keyword"
    if kind in {"date", "date_list"}:
        return "date"
    if kind in {"integer", "integer_list"}:
        return "long"
    if kind in {"float", "float_list"}:
        return "float"
    if kind in {"boolean", "boolean_list"}:
        return "boolean"
    if kind == "object":
        return "object"
    if kind == "object_list":
        return "nested?"
    return "review"


def mapping_snippet(field: str, mapping_hint: str) -> str:
    if mapping_hint == "text":
        return f'"{field}": {{"type": "text"}}'
    if mapping_hint == "text+keyword":
        return f'"{field}": {{"type": "text", "fields": {{"keyword": {{"type": "keyword", "ignore_above": 256}}}}}}'
    if mapping_hint == "keyword":
        return f'"{field}": {{"type": "keyword"}}'
    if mapping_hint == "date":
        return f'"{field}": {{"type": "date"}}'
    if mapping_hint == "long":
        return f'"{field}": {{"type": "long"}}'
    if mapping_hint == "float":
        return f'"{field}": {{"type": "float"}}'
    if mapping_hint == "boolean":
        return f'"{field}": {{"type": "boolean"}}'
    if mapping_hint == "object":
        return f'"{field}": {{"type": "object"}}'
    if mapping_hint == "nested?":
        return f'"{field}": {{"type": "nested"}}  # only if you query individual child objects'
    return f'"{field}": {{...}}  # review manually'


def field_flags(field: str) -> dict[str, bool]:
    lower = field.lower()
    return {
        "is_hashy": any(token in lower for token in ("hash", "_fp", "fingerprint")),
        "is_id_like": lower.endswith("id") or lower.endswith("_id") or lower == "@id" or ".id" in lower,
        "is_url_like": "url" in lower or "doi" in lower,
        "is_internal": lower.startswith("_") or lower.startswith("@") or lower.startswith("_field_hashes."),
        "is_name_like": any(token in lower for token in ("title", "name", "subtitle", "description", "content", "keyword")),
    }


def select_query_source_fields(
    available_fields: set[str],
    bundle_fields: set[str],
) -> tuple[list[str], list[str]]:
    """Pick fields to draw pseudo-queries from, held out from every bundle.

    A pseudo-query built from the same field that is being ranked guarantees a
    match, so any field used as a query source is dropped from the pool. What
    is dropped is returned too, so the report can say why.
    """
    chosen: list[str] = []
    dropped: list[str] = []
    for field in QUERY_SOURCE_PREFERENCE:
        if field not in available_fields:
            continue
        if field in bundle_fields:
            dropped.append(field)
            continue
        chosen.append(field)
    return chosen, dropped


def build_pseudo_queries(
    per_field_text: dict[str, list[str]],
    query_source_fields: list[str],
    n_docs: int,
    sample_size: int = BENCH_QUERIES,
    terms_per_query: int = BENCH_QUERY_TERMS,
    seed: int = BENCH_SEED,
) -> tuple[list[int], list[list[str]]]:
    """Short keyword-style queries drawn from held-out text.

    For each sampled document, the highest tf-idf terms of its *query source*
    text become the query, and that document is the target. This imitates a
    user searching for a document by its subject matter, instead of pasting
    the indexed field back in as the query.

    Returns (target_doc_ids, query_token_lists), aligned.
    """
    if not query_source_fields or n_docs == 0:
        return [], []

    token_lists: list[list[str]] = []
    for idx in range(n_docs):
        parts = [per_field_text[field][idx] for field in query_source_fields if field in per_field_text]
        token_lists.append(tok(" ".join(parts)))

    idf, _, _, _, _ = bm25_prepare(token_lists)

    eligible = [idx for idx, tokens in enumerate(token_lists) if tokens]
    rng = random.Random(seed)
    if sample_size and len(eligible) > sample_size:
        target_ids = sorted(rng.sample(eligible, sample_size))
    else:
        target_ids = eligible

    doc_ids: list[int] = []
    queries: list[list[str]] = []
    for idx in target_ids:
        counts = Counter(
            term
            for term in token_lists[idx]
            if len(term) > 2 and term not in _QUERY_STOPWORDS and not term.isdigit()
        )
        if not counts:
            continue
        ranked = sorted(
            counts.items(),
            key=lambda item: (-(item[1] * idf.get(item[0], 0.0)), item[0]),
        )
        terms = [term for term, _ in ranked[:terms_per_query]]
        if not terms:
            continue
        doc_ids.append(idx)
        queries.append(terms)

    return doc_ids, queries


def query_term_leakage(
    doc_ids: list[int],
    queries: list[list[str]],
    per_field_text: dict[str, list[str]],
    field: str,
) -> float:
    """Mean share of a query's terms that occur verbatim in `field` of its target.

    Residual circularity, measured rather than assumed. 1.0 means the field
    literally contains the query; near 0.0 means the benchmark is asking the
    field to match vocabulary it does not already hold.
    """
    if field not in per_field_text or not queries:
        return 0.0
    total = 0.0
    for doc_id, terms in zip(doc_ids, queries):
        if not terms:
            continue
        field_tokens = set(tok(per_field_text[field][doc_id]))
        total += sum(1 for term in terms if term in field_tokens) / len(terms)
    return total / len(queries)


def field_retrieval_scores(
    per_field_text: dict[str, list[str]],
    fields: list[str],
    doc_ids: list[int],
    queries: list[list[str]],
    n_docs: int,
    exclude: set[str] | None = None,
) -> dict[str, float]:
    """MRR@TOPK of each field on its own, against the held-out pseudo-queries.

    Computed for every eligible field rather than a hand-picked handful, so the
    retrieval term of full_text_score means the same thing for all of them.
    """
    results: dict[str, float] = {}
    exclude = exclude or set()
    if not queries:
        return {field: 0.0 for field in fields}

    for field in fields:
        if field in exclude:
            # The queries were drawn from this field, so it would score ~1.0 by
            # construction. Reported as not applicable instead of as evidence.
            continue
        docs_tokens = [tok(per_field_text[field][idx]) for idx in range(n_docs)]
        index = Bm25Index(docs_tokens)
        if index.empty:
            results[field] = 0.0
            continue
        reciprocal_rank = 0.0
        for target_id, terms in zip(doc_ids, queries):
            for rank, hit in enumerate(index.top_k(terms, TOPK), start=1):
                if hit == target_id:
                    reciprocal_rank += 1.0 / rank
                    break
        results[field] = reciprocal_rank / len(queries)
    return results


def list_value_set(values: dict[str, list[Any]], field: str, idx: int) -> set[str]:
    if field not in values:
        return set()
    return {x.strip().lower() for x in atomic_values(values[field][idx]) if x.strip()}


def scalar_value(values: dict[str, list[Any]], field: str, idx: int) -> str:
    if field not in values:
        return ""
    atoms = atomic_values(values[field][idx])
    return atoms[0].strip().lower() if atoms else ""


# A taxonomy value shared by more than this share of the corpus says nothing
# about relevance, so it is not allowed to create judgements. Without it, a
# coarse field such as `category` marks a fifth of the corpus relevant to every
# query and pins Recall@k at its arithmetic ceiling.
QREL_MAX_BUCKET_RATIO = float(os.getenv("AF_QREL_MAX_BUCKET_RATIO", "0.25"))

# One shared taxonomy value is weak evidence of relevance when the vocabulary
# is small. Requiring agreement on several keeps the judgements meaningful.
QREL_MIN_TAXONOMY_OVERLAP = int(os.getenv("AF_QREL_MIN_OVERLAP", "2"))

QREL_PROJECT_FIELDS = ("project_id", "project_acronym")
QREL_TAXONOMY_FIELDS = ("themes", "topics", "subcategories")


def graded_qrels(
    per_field_values: dict[str, list[Any]],
    n_docs: int,
    query_doc_ids: list[int] | None = None,
) -> tuple[list[dict[int, int]], dict[str, Any]]:
    """Weak relevance labels for the pseudo-queries.

    Grades: the target document itself = 3, same project = 2, shared taxonomy
    value = 1. `category` is deliberately not used — see QREL_MAX_BUCKET_RATIO.

    Built through inverted indexes over precomputed per-document keys, so the
    cost is proportional to the postings actually touched rather than to
    n_docs^2. The previous pairwise version re-derived every other document's
    keys inside the inner loop.
    """
    targets = list(range(n_docs)) if query_doc_ids is None else list(query_doc_ids)

    project_keys: list[set[str]] = []
    taxonomy_keys: list[set[str]] = []
    for idx in range(n_docs):
        proj = {
            f"{field}={scalar_value(per_field_values, field, idx)}"
            for field in QREL_PROJECT_FIELDS
            if scalar_value(per_field_values, field, idx)
        }
        taxo: set[str] = set()
        for field in QREL_TAXONOMY_FIELDS:
            taxo |= {f"{field}={value}" for value in list_value_set(per_field_values, field, idx)}
        project_keys.append(proj)
        taxonomy_keys.append(taxo)

    project_index: dict[str, list[int]] = defaultdict(list)
    taxonomy_index: dict[str, list[int]] = defaultdict(list)
    for idx in range(n_docs):
        for key in project_keys[idx]:
            project_index[key].append(idx)
        for key in taxonomy_keys[idx]:
            taxonomy_index[key].append(idx)

    max_bucket = max(1, int(QREL_MAX_BUCKET_RATIO * n_docs))
    skipped_buckets = sorted(
        (key for key, docs in taxonomy_index.items() if len(docs) > max_bucket),
        key=lambda key: -len(taxonomy_index[key]),
    )
    skipped = set(skipped_buckets)

    qrels: list[dict[int, int]] = []
    for idx in targets:
        overlap: Counter[int] = Counter()
        for key in taxonomy_keys[idx]:
            if key in skipped:
                continue
            for other in taxonomy_index[key]:
                if other != idx:
                    overlap[other] += 1

        rels: dict[int, int] = {
            other: 1
            for other, shared in overlap.items()
            if shared >= QREL_MIN_TAXONOMY_OVERLAP
        }
        for key in project_keys[idx]:
            for other in project_index[key]:
                if other != idx:
                    rels[other] = max(rels.get(other, 0), 2)
        rels[idx] = 3
        qrels.append(rels)

    sizes = [len(rels) for rels in qrels] or [0]
    meta = {
        "mean_relevant": sum(sizes) / len(sizes),
        "max_relevant": max(sizes),
        "min_overlap": QREL_MIN_TAXONOMY_OVERLAP,
        "skipped_buckets": skipped_buckets[:5],
        "skipped_bucket_count": len(skipped_buckets),
        "max_bucket": max_bucket,
    }
    return qrels, meta


def rank_docs_for_fields(
    per_field_text: dict[str, list[str]],
    candidate_fields: list[str],
    queries: list[list[str]],
    n_docs: int,
) -> list[list[int]]:
    docs_tokens: list[list[str]] = []
    for idx in range(n_docs):
        tokens: list[str] = []
        for field in candidate_fields:
            if field in per_field_text:
                tokens.extend(tok(per_field_text[field][idx]))
        docs_tokens.append(tokens)

    index = Bm25Index(docs_tokens)
    if index.empty:
        return [[] for _ in queries]
    return [index.top_k(query_tokens, TOPK) for query_tokens in queries]


def dcg_at_k(ranked_ids: list[int], rels: dict[int, int], k: int) -> float:
    total = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        gain = rels.get(doc_id, 0)
        if gain > 0:
            total += (2 ** gain - 1) / math.log2(rank + 1)
    return total


def evaluate_bundle(rankings: list[list[int]], qrels: list[dict[int, int]], k: int) -> dict[str, float]:
    mrr = 0.0
    recall = 0.0
    recall_ceiling = 0.0
    ndcg = 0.0
    n_queries = 0

    for ranked_ids, rels in zip(rankings, qrels):
        if not rels:
            continue
        n_queries += 1
        relevant_ids = {doc_id for doc_id, grade in rels.items() if grade > 0}
        topk = ranked_ids[:k]

        rr = 0.0
        for rank, doc_id in enumerate(topk, start=1):
            if rels.get(doc_id, 0) > 0:
                rr = 1.0 / rank
                break
        mrr += rr

        hits = sum(1 for doc_id in topk if doc_id in relevant_ids)
        recall += hits / max(1, len(relevant_ids))

        ideal = sorted(rels.values(), reverse=True)
        ideal_dcg = sum((2 ** gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal[:k], start=1))
        ndcg += dcg_at_k(topk, rels, k) / ideal_dcg if ideal_dcg > 0 else 0.0

        # Recall@k cannot exceed k/|relevant|. Tracking the ceiling keeps the
        # reported figure interpretable when the qrels are broad.
        recall_ceiling += min(1.0, k / max(1, len(relevant_ids)))

    denom = max(1, n_queries)
    return {
        "mrr_at_k": mrr / denom,
        "recall_at_k": recall / denom,
        "recall_ceiling_at_k": recall_ceiling / denom,
        "ndcg_at_k": ndcg / denom,
        "queries": float(n_queries),
    }


def build_ablation_candidates(preferred_full_text: list[str]) -> list[tuple[str, list[str]]]:
    candidates: list[tuple[str, list[str]]] = []
    base = [field for field in preferred_full_text if field]
    if not base:
        return candidates
    candidates.append(("all_recommended", base))
    for field in base:
        ablated = [other for other in base if other != field]
        if ablated:
            candidates.append((f"drop_{field}", ablated))
    return candidates


def summarise_ablation_results(
    benchmark_results: list[tuple[str, dict[str, float], list[str]]],
    benchmark: dict[str, Any] | None = None,
) -> list[str]:
    if not benchmark_results:
        return ["No ablation benchmark summary available."]

    baseline_label, baseline_metrics, baseline_fields = benchmark_results[0]
    baseline_ndcg = baseline_metrics["ndcg_at_k"]
    lines = [
        f"Baseline bundle `{baseline_label}` uses: {', '.join(baseline_fields)}.",
        f"Primary comparison metric here is nDCG@{TOPK}; lower after dropping a field means that field was helping.",
    ]

    effects: list[tuple[float, str, str]] = []
    for label, metrics, fields in benchmark_results[1:]:
        dropped = label.replace("drop_", "", 1)
        delta = metrics["ndcg_at_k"] - baseline_ndcg
        effects.append((delta, dropped, label))

    helpful = sorted([item for item in effects if item[0] < -0.002], key=lambda item: item[0])
    neutral = sorted([item for item in effects if abs(item[0]) <= 0.002], key=lambda item: item[1])
    harmful = sorted([item for item in effects if item[0] > 0.002], key=lambda item: -item[0])

    if helpful:
        lines.append(
            "Most helpful fields by ablation: "
            + ", ".join(f"{field} (nDCG delta {delta:.4f})" for delta, field, _ in helpful[:3])
            + "."
        )
    if neutral:
        lines.append(
            "Likely optional / low-impact fields: "
            + ", ".join(f"{field} (delta {delta:+.4f})" for delta, field, _ in neutral[:3])
            + "."
        )
    if harmful:
        lines.append(
            "Fields whose removal improved the pseudo benchmark: "
            + ", ".join(f"{field} (delta {delta:+.4f})" for delta, field, _ in harmful[:3])
            + "."
        )

    spread = max(metrics["ndcg_at_k"] for _, metrics, _ in benchmark_results) - min(
        metrics["ndcg_at_k"] for _, metrics, _ in benchmark_results
    )
    if spread < 0.01:
        lines.append(
            f"Caution: the whole ablation spans only {spread:.4f} nDCG. That is too narrow to rank "
            "fields against each other; treat every field above as untested rather than confirmed."
        )

    leakage = (benchmark or {}).get("leakage") or {}
    leaky = sorted((field for field, value in leakage.items() if value >= 0.25), key=lambda f: -leakage[f])
    if leaky:
        lines.append(
            "Caution: "
            + ", ".join(f"{field} ({leakage[field]:.0%} of query terms)" for field in leaky[:3])
            + " already contain much of the query vocabulary, so their ablation deltas are partly circular."
        )
    return lines


def analyse_fields(
    records: list[dict[str, Any]],
    run_benchmark: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[tuple[str, dict[str, float], list[str]]]]:
    """Diagnose every field, score it for each role, and optionally benchmark.

    Set run_benchmark=False when only the field report and the recommended
    bundle are needed; that skips the qrels build and the ablation passes.
    """
    flattened = [flatten_record(record) for record in records]
    all_fields = Counter()
    for record in flattened:
        all_fields.update(record.keys())

    field_names = [name for name, _ in all_fields.most_common(MAX_DISCOVERED_FIELDS)]
    per_field_values: dict[str, list[Any]] = {field: [] for field in field_names}
    per_field_text: dict[str, list[str]] = {field: [] for field in field_names}

    for record in flattened:
        for field in field_names:
            value = record.get(field)
            per_field_values[field].append(value)
            per_field_text[field].append(normalise_text(value))

    report: dict[str, dict[str, Any]] = {}
    n_docs = len(records)

    for field in field_names:
        values = per_field_values[field]
        texts = per_field_text[field]
        flags = field_flags(field)
        non_empty_pairs = [(value, text) for value, text in zip(values, texts) if not is_empty(value)]
        non_empty_values = [value for value, _ in non_empty_pairs]
        non_empty_texts = [text for _, text in non_empty_pairs]
        non_empty_count = len(non_empty_pairs)
        coverage = non_empty_count / max(1, n_docs)

        observed_kinds = Counter(describe_value(value) for value in non_empty_values)
        dominant_kind = observed_kinds.most_common(1)[0][0] if observed_kinds else "empty"
        array_ratio = sum(isinstance(value, (list, tuple, set)) for value in non_empty_values) / max(1, non_empty_count)
        object_ratio = sum(isinstance(value, dict) for value in non_empty_values) / max(1, non_empty_count)

        token_lists = [tok(text) for text in texts]
        non_empty_token_lists = [tokens for tokens, text in zip(token_lists, texts) if text]
        avg_tokens = float(np.mean([len(tokens) for tokens in non_empty_token_lists])) if non_empty_token_lists else 0.0
        avg_chars = float(np.mean([len(text) for text in non_empty_texts])) if non_empty_texts else 0.0

        # IDF over the documents that actually have this field. Including the
        # empty ones inflates IDF for sparse fields: a term present in 1 of 10
        # populated records looks maximally rare against a 13k-document corpus.
        idf, _, _, _, _ = bm25_prepare(non_empty_token_lists)
        avg_idf = float(np.mean(list(idf.values()))) if idf else 0.0
        avg_idf_norm = normalised_idf(avg_idf, len(non_empty_token_lists))

        exact_unique = len(set(non_empty_texts)) / max(1, non_empty_count)
        boilerplate = 1.0 - exact_unique

        atoms: list[str] = []
        for value in non_empty_values:
            atoms.extend(atomic_values(value))
        atom_counts = Counter(atoms)
        distinct_atoms = len(atom_counts)
        cardinality_ratio = distinct_atoms / max(1, non_empty_count)
        value_entropy = normalised_entropy(list(atom_counts.values()))
        mean_value_frequency = (sum(atom_counts.values()) / distinct_atoms) if distinct_atoms else 0.0

        likely_enum = (
            dominant_kind in {"string", "string_list"}
            and avg_tokens <= 4.0
            and distinct_atoms <= MAX_ENUM_CARDINALITY
            and cardinality_ratio < 0.8
        )

        mapping_hint = field_mapping_hint(dominant_kind, avg_tokens, cardinality_ratio, array_ratio)
        if flags["is_hashy"] or flags["is_id_like"]:
            mapping_hint = "keyword"
        elif flags["is_url_like"]:
            mapping_hint = "keyword" if avg_tokens <= 8 else "text+keyword"

        report[field] = {
            "coverage": coverage,
            "non_empty_count": non_empty_count,
            "dominant_kind": dominant_kind,
            "observed_kinds": dict(observed_kinds),
            "avg_tokens": avg_tokens,
            "avg_chars": avg_chars,
            "avg_idf": avg_idf,
            "avg_idf_norm": avg_idf_norm,
            "boilerplate": boilerplate,
            "cardinality_ratio": cardinality_ratio,
            "distinct_atoms": distinct_atoms,
            "value_entropy": value_entropy,
            "mean_value_frequency": mean_value_frequency,
            "array_ratio": array_ratio,
            "object_ratio": object_ratio,
            "likely_enum": likely_enum,
            "mapping_hint": mapping_hint,
            "flags": flags,
        }

    scored_fields = [field for field, stats in report.items() if stats["coverage"] >= MIN_COVERAGE]
    benchmark: dict[str, Any] = {
        "ran": False,
        "reason": "no fields cleared the coverage threshold",
        "query_source": [],
        "query_source_dropped": [],
        "queries": 0,
        "leakage": {},
        "qrels": {},
    }
    if not scored_fields:
        return report, benchmark, []

    eligible_text = [
        field for field in scored_fields
        if report[field]["dominant_kind"] in {"string", "string_list"}
        and report[field]["mapping_hint"] in {"text", "text+keyword", "keyword"}
        and report[field]["avg_tokens"] >= 2.0
    ]

    # Every field the policy layer could put in a bundle is held out of the
    # pseudo-queries, so no bundle is ever asked to retrieve its own text.
    bundle_union = {
        field
        for field in POLICY_FULL_TEXT_PRIORITY + POLICY_FULL_TEXT_FALLBACKS
        if field in report
    }
    query_source, query_source_dropped = select_query_source_fields(set(per_field_text), bundle_union)
    query_doc_ids, queries = build_pseudo_queries(per_field_text, query_source, n_docs)

    if queries:
        log.info(
            "Pseudo-queries: %d queries of <=%d terms from %s",
            len(queries), BENCH_QUERY_TERMS, ", ".join(query_source),
        )
        retrieval_scores = field_retrieval_scores(
            per_field_text, eligible_text, query_doc_ids, queries, n_docs, exclude=set(query_source)
        )
    else:
        retrieval_scores = {}
        benchmark["reason"] = (
            "no held-out query source available: "
            + (", ".join(query_source_dropped) + " are all inside the candidate bundle"
               if query_source_dropped else "none of " + ", ".join(QUERY_SOURCE_PREFERENCE) + " is present")
        )

    for field in scored_fields:
        stats = report[field]
        coverage = stats["coverage"]
        low_boilerplate = 1.0 - stats["boilerplate"]
        length_fitness = saturating(stats["avg_tokens"], 200.0)
        retrieval = retrieval_scores.get(field, 0.0)
        is_query_source = field in query_source
        flags = stats["flags"]

        # ------------------------------------------------------------------
        # full_text_score - is this field worth ranking on?
        #     0.30 * coverage            (raw, already 0-1)
        #   + 0.20 * avg_idf_norm        (IDF / log(n_populated); corpus-size free)
        #   + 0.20 * length_fitness      (log-saturating on avg tokens, 200 = full)
        #   + 0.10 * (1 - boilerplate)
        #   + 0.20 * retrieval           (MRR@10 on held-out pseudo-queries)
        #
        # then multiplied down by the penalties below. All inputs are absolute
        # quantities, so a field's score does not move because some other field
        # in the same file happened to be the longest or the rarest.
        # ------------------------------------------------------------------
        full_text_score = (
            0.30 * coverage
            + 0.20 * stats["avg_idf_norm"]
            + 0.20 * length_fitness
            + 0.10 * low_boilerplate
            + 0.20 * retrieval
        )
        if stats["mapping_hint"] == "keyword" and not stats["likely_enum"]:
            full_text_score *= 0.45
        if stats["mapping_hint"] == "keyword" and not flags["is_name_like"]:
            full_text_score *= 0.35
        if stats["dominant_kind"] not in {"string", "string_list"}:
            full_text_score *= 0.2
        if flags["is_hashy"] or flags["is_id_like"] or flags["is_url_like"]:
            full_text_score *= 0.05
        elif flags["is_internal"] and not flags["is_name_like"]:
            full_text_score *= 0.15

        # ------------------------------------------------------------------
        # facet_score - is this field worth offering as a filter?
        #     0.25 * coverage
        #   + 0.25 * cardinality_fitness (absolute distinct count: 2..40 ideal)
        #   + 0.25 * facet_reuse         (documents selected per value; 1 value
        #                                 per document is a label, not a filter)
        #   + 0.15 * value_entropy       (how evenly documents spread over values)
        #   + 0.10 * type_bonus
        #
        # Repetition is what makes a facet work, so - unlike full_text_score -
        # nothing here penalises a field for reusing its values, and the shape
        # term reads the absolute number of choices a user would be shown
        # rather than distinct/n_docs.
        # ------------------------------------------------------------------
        facet_score = (
            0.25 * coverage
            + 0.25 * facet_cardinality_fitness(stats["distinct_atoms"])
            + 0.25 * facet_reuse(stats["mean_value_frequency"])
            + 0.15 * stats["value_entropy"]
            + 0.10 * (1.0 if stats["mapping_hint"] in {"keyword", "date", "boolean", "long", "float"} else 0.0)
        )
        if stats["dominant_kind"] in {"object", "object_list", "mixed_list"}:
            facet_score *= 0.2
        if flags["is_hashy"] or flags["is_id_like"]:
            facet_score *= 0.1
        elif flags["is_internal"]:
            facet_score *= 0.5

        # ------------------------------------------------------------------
        # sort_score - can documents be meaningfully ordered by this field?
        # Zero unless the field is actually sortable. Coverage alone must never
        # be enough, or every fully-populated text field outranks its own
        # full_text_score and gets filed as a sort field.
        # ------------------------------------------------------------------
        if stats["mapping_hint"] == "date":
            sort_bias = 1.0
        elif stats["mapping_hint"] in {"long", "float"}:
            sort_bias = 0.9
        elif (
            stats["mapping_hint"] in {"keyword", "text+keyword"}
            and stats["array_ratio"] < 0.3
            and stats["cardinality_ratio"] >= 0.5
        ):
            sort_bias = 0.4  # alphabetical ordering of a near-unique label
        else:
            sort_bias = 0.0

        sort_score = 0.0 if sort_bias == 0.0 else sort_bias * (0.60 + 0.40 * coverage)
        if flags["is_hashy"] or flags["is_url_like"]:
            sort_score *= 0.3

        stats["full_text_score"] = full_text_score
        stats["facet_score"] = facet_score
        stats["sort_score"] = sort_score
        stats["retrieval_mrr"] = retrieval
        stats["retrieval_is_query_source"] = is_query_source

    preferred = resolve_preferred_fields(report)
    benchmark.update(
        {
            "query_source": query_source,
            "query_source_dropped": query_source_dropped,
            "queries": len(queries),
        }
    )

    if not run_benchmark:
        benchmark["reason"] = "benchmark skipped by caller"
        return report, benchmark, []
    if not queries:
        return report, benchmark, []  # reason set when the query source was chosen
    if not preferred["full_text"]:
        benchmark["reason"] = "no full-text fields were recommended"
        return report, benchmark, []

    qrels, qrels_meta = graded_qrels(per_field_values, n_docs, query_doc_ids)
    benchmark["qrels"] = qrels_meta
    benchmark["leakage"] = {
        field: query_term_leakage(query_doc_ids, queries, per_field_text, field)
        for field in preferred["full_text"]
    }
    benchmark["ran"] = True
    benchmark["reason"] = ""

    benchmark_results: list[tuple[str, dict[str, float], list[str]]] = []
    for label, fields in build_ablation_candidates(preferred["full_text"]):
        rankings = rank_docs_for_fields(per_field_text, fields, queries, n_docs)
        metrics = evaluate_bundle(rankings, qrels, TOPK)
        benchmark_results.append((label, metrics, fields))

    return report, benchmark, benchmark_results



def choose_role(stats: dict[str, Any]) -> str:
    if stats["coverage"] < MIN_COVERAGE:
        return "skip_low_coverage"
    if stats["dominant_kind"] in {"object", "object_list", "mixed_list"}:
        return "review_structure"
    if stats["flags"]["is_internal"] or stats["flags"]["is_hashy"]:
        return "skip_internal"
    if stats["flags"]["is_id_like"]:
        return "identifier_only"
    full_text = stats.get("full_text_score", 0.0)
    facet = stats.get("facet_score", 0.0)
    sort_score = stats.get("sort_score", 0.0)

    if stats["flags"]["is_url_like"] and full_text < FULL_TEXT_THRESHOLD:
        return "skip_display_only"
    if stats["mapping_hint"] == "keyword" and not stats["flags"]["is_name_like"] and facet >= FACET_THRESHOLD:
        return "facet_filter" if facet >= sort_score else "sort_filter"

    if full_text >= max(facet, sort_score, FULL_TEXT_THRESHOLD):
        return "full_text"
    if facet >= max(full_text, sort_score, FACET_THRESHOLD):
        return "facet_filter"
    if sort_score >= max(SORT_THRESHOLD, full_text):
        return "sort_filter"
    if stats["mapping_hint"] in {"text", "text+keyword", "keyword", "date", "long", "float", "boolean"}:
        return "index_optional"
    return "skip"


def resolve_preferred_fields(report: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the policy lists, gated by the measured roles.

    Returns the chosen fields plus `overrides`: policy picks that the evidence
    does not support and policy drops that it does, so the report can show
    where the two disagree instead of presenting policy as a finding.
    """
    roles = {field: choose_role(stats) for field, stats in report.items()}
    overrides: list[str] = []

    chosen_full_text: list[str] = []
    for field in POLICY_FULL_TEXT_PRIORITY + POLICY_FULL_TEXT_FALLBACKS:
        if field not in report or field in chosen_full_text:
            continue
        role = roles[field]
        score = report[field].get("full_text_score", 0.0)
        if role in FULL_TEXT_ROLES:
            chosen_full_text.append(field)
            continue
        if role in FACET_ROLES:
            # Policy wants it searchable even though it scores better as a
            # filter - keep it, but say so.
            chosen_full_text.append(field)
            overrides.append(
                f"{field}: kept as full-text by policy; measured role is {role} "
                f"(full_text={score:.3f}, facet={report[field].get('facet_score', 0.0):.3f})"
            )
            continue
        overrides.append(f"{field}: dropped from full-text; measured role is {role} (full_text={score:.3f})")

    chosen_facets: list[str] = []
    for field in POLICY_FACETS:
        if field not in report or field in chosen_facets:
            continue
        if roles[field] in FACET_ROLES:
            chosen_facets.append(field)
        else:
            overrides.append(
                f"{field}: dropped from facets; measured role is {roles[field]} "
                f"(facet={report[field].get('facet_score', 0.0):.3f})"
            )

    strong_unlisted = sorted(
        field
        for field, role in roles.items()
        if role == "facet_filter"
        and field not in chosen_facets
        and report[field].get("facet_score", 0.0) >= FACET_THRESHOLD + 0.15
    )
    for field in strong_unlisted[:5]:
        overrides.append(
            f"{field}: scores well as a facet ({report[field]['facet_score']:.3f}) but is not on the policy list"
        )

    excluded = [field for field in POLICY_EXCLUDED if field in report]
    excluded += sorted(
        field for field, role in roles.items()
        if role in {"skip_internal", "skip_display_only"} and field not in excluded
    )

    return {
        "full_text": chosen_full_text,
        "facets": chosen_facets,
        "excluded": excluded,
        "overrides": overrides,
        "roles": roles,
    }


def build_report(
    input_path: str,
    records: list[dict[str, Any]],
    report: dict[str, dict[str, Any]],
    benchmark: dict[str, Any],
    benchmark_results: list[tuple[str, dict[str, float], list[str]]],
) -> str:
    lines: list[str] = []
    preferred = resolve_preferred_fields(report)
    roles = preferred["roles"]

    lines.append("=== DATASET SUMMARY ===")
    lines.append(f"Input file            : {input_path}")
    lines.append(f"Records analysed      : {len(records)}")
    lines.append(f"Fields analysed       : {len(report)}")
    lines.append(f"Coverage threshold    : {MIN_COVERAGE:.2f}")

    lines.append("\n=== FINAL PICK FOR THIS FILE ===")
    lines.append("Full-text fields      : " + (", ".join(preferred["full_text"]) or "-"))
    lines.append("Facet/sort fields     : " + (", ".join(preferred["facets"]) or "-"))
    lines.append("Excluded              : " + (", ".join(preferred["excluded"][:14]) or "-"))

    lines.append("\n=== WHERE POLICY AND EVIDENCE DISAGREE ===")
    if preferred["overrides"]:
        lines.append("The field lists above are a policy decision. These are the cases the")
        lines.append("measurements do not support, or support but policy does not list:")
        for note in preferred["overrides"]:
            lines.append(f"  - {note}")
    else:
        lines.append("None: every policy pick matches its measured role.")

    lines.append("\n=== FIELD DIAGNOSTICS ===")
    lines.append("cov=coverage  tok=avg tokens  idfn=IDF/log(n populated)  mrr=MRR@10 on held-out")
    lines.append("pseudo-queries ('src' = queries came from this field)  dist=distinct values\nent=value entropy  boil=exact-duplicate share")
    header = (
        f"{'field':34} {'role':16} {'map':11} {'cov':>5} {'tok':>7} "
        f"{'idfn':>5} {'mrr':>5} {'dist':>6} {'ent':>5} {'boil':>5} {'kind':18}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    ranked_fields = sorted(
        report.items(),
        key=lambda item: (
            roles[item[0]] not in {"full_text", "facet_filter", "sort_filter", "index_optional"},
            -(item[1].get("full_text_score", 0.0) + item[1].get("facet_score", 0.0)),
            -item[1]["coverage"],
            item[0],
        ),
    )
    for field, stats in ranked_fields:
        if stats.get("retrieval_is_query_source"):
            mrr_cell = f"{'src':>5}"  # queries came from this field; not evidence
        elif "retrieval_mrr" in stats:
            mrr_cell = f"{stats['retrieval_mrr']:5.2f}"
        else:
            mrr_cell = f"{'-':>5}"
        lines.append(
            f"{field[:34]:34} {roles[field]:16} {stats['mapping_hint']:11} "
            f"{stats['coverage']:5.2f} {stats['avg_tokens']:7.1f} {stats['avg_idf_norm']:5.2f} "
            f"{mrr_cell} {stats['distinct_atoms']:6d} "
            f"{stats['value_entropy']:5.2f} {stats['boilerplate']:5.2f} {stats['dominant_kind'][:18]:18}"
        )

    lines.append("\n=== RECOMMENDED FULL-TEXT FIELDS ===")
    if preferred["full_text"]:
        for field in preferred["full_text"]:
            stats = report[field]
            lines.append(
                f"{field:34} score={stats.get('full_text_score', 0.0):.3f}  "
                f"role={roles[field]:14} mapping={stats['mapping_hint']}  "
                f"coverage={stats['coverage']:.2f}  avg_tokens={stats['avg_tokens']:.1f}"
            )
    else:
        lines.append("No strong full-text fields found.")

    lines.append("\n=== RECOMMENDED FILTER / SORT FIELDS ===")
    if preferred["facets"]:
        by_evidence = sorted(
            preferred["facets"],
            key=lambda field: -max(
                report[field].get("facet_score", 0.0), report[field].get("sort_score", 0.0)
            ),
        )
        for field in by_evidence:
            stats = report[field]
            lines.append(
                f"{field:34} facet={stats.get('facet_score', 0.0):.3f}  "
                f"sort={stats.get('sort_score', 0.0):.3f}  role={roles[field]:14} "
                f"mapping={stats['mapping_hint']}  values={stats['distinct_atoms']}"
            )
    else:
        lines.append("No strong facet/sort fields found.")

    other_facets = [
        (field, stats) for field, stats in report.items()
        if roles[field] in FACET_ROLES and field not in preferred["facets"]
    ]
    other_facets.sort(key=lambda item: item[1].get("facet_score", 0.0), reverse=True)
    if other_facets:
        lines.append("\nOther fields that measure well as filters but are not on the policy list:")
        for field, stats in other_facets[:10]:
            lines.append(
                f"{field:34} facet={stats.get('facet_score', 0.0):.3f}  "
                f"sort={stats.get('sort_score', 0.0):.3f}  values={stats['distinct_atoms']}"
            )

    identifier_fields = sorted(field for field, role in roles.items() if role == "identifier_only")
    lines.append("\n=== IDENTIFIER-ONLY FIELDS ===")
    if identifier_fields:
        for field in identifier_fields:
            lines.append(
                f"{field:34} mapping={report[field]['mapping_hint']}  coverage={report[field]['coverage']:.2f}"
            )
    else:
        lines.append("No identifier-only fields found.")

    lines.append("\n=== SUGGESTED MAPPINGS ===")
    for field in preferred["full_text"] + preferred["facets"]:
        if field in report:
            lines.append(mapping_snippet(field, report[field]["mapping_hint"]))

    lines.append(f"\n=== PSEUDO-QUERY ABLATION BENCHMARK (TOP {TOPK}) ===")
    if benchmark.get("ran") and benchmark_results:
        qrels_meta = benchmark.get("qrels", {})
        lines.append(
            "Queries       : "
            f"{benchmark['queries']} keyword queries of <={BENCH_QUERY_TERMS} terms, "
            f"drawn from {', '.join(benchmark['query_source'])}"
        )
        if benchmark["query_source_dropped"]:
            lines.append(
                "Held out      : "
                + ", ".join(benchmark["query_source_dropped"])
                + " (inside the candidate bundle, so unusable as a query source)"
            )
        lines.append(
            "Judgements    : self=3, same project=2, "
            f"{qrels_meta.get('min_overlap', QREL_MIN_TAXONOMY_OVERLAP)}+ shared taxonomy values=1; "
            f"mean {qrels_meta.get('mean_relevant', 0.0):.1f} relevant per query "
            f"(max {qrels_meta.get('max_relevant', 0)})"
        )
        if qrels_meta.get("skipped_bucket_count"):
            lines.append(
                f"              {qrels_meta['skipped_bucket_count']} taxonomy value(s) ignored for covering "
                f">{QREL_MAX_BUCKET_RATIO:.0%} of the corpus, e.g. "
                + ", ".join(qrels_meta.get("skipped_buckets", [])[:3])
            )
        if benchmark.get("leakage"):
            leak = sorted(benchmark["leakage"].items(), key=lambda item: -item[1])
            lines.append(
                "Term overlap  : share of query terms already present in each bundle field "
                "(lower = less circular)"
            )
            lines.append("              " + ", ".join(f"{field}={value:.2f}" for field, value in leak))
        lines.append("")
        for label, metrics, fields in benchmark_results:
            lines.append(
                f"{label:32} MRR@{TOPK}={metrics['mrr_at_k']:.4f}  "
                f"nDCG@{TOPK}={metrics['ndcg_at_k']:.4f}  "
                f"Recall@{TOPK}={metrics['recall_at_k']:.4f}/{metrics['recall_ceiling_at_k']:.4f}  "
                f"fields={', '.join(fields)}"
            )
        lines.append("Recall is shown as achieved/ceiling; the ceiling is k divided by the number")
        lines.append("of relevant documents, which no ranking can beat.")
    else:
        lines.append(f"Skipped: {benchmark.get('reason') or 'no benchmark candidates were generated.'}")

    lines.append("\n=== PLAIN-ENGLISH ABLATION SUMMARY ===")
    for line in summarise_ablation_results(benchmark_results, benchmark):
        lines.append(line)

    lines.append("\n=== NOTES ===")
    lines.append("Use `text` fields for ranking and `keyword`/`date`/numeric fields for facets, filters, and sorting.")
    lines.append("Fields marked `review_structure` are too nested or mixed to map safely without manual design.")
    lines.append("Short repeated label fields often belong in `keyword`; long narrative fields belong in `text`.")
    lines.append("Queries here are generated from held-out text, not from the fields being ranked, but they are")
    lines.append("still synthetic. Judged queries with real relevance labels remain the only decisive evidence;")
    lines.append("run evaluate_field_bundles.py once you have them.")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse JSON fields for OpenSearch indexing decisions.")
    parser.add_argument("--input", help="Path to JSON or JSONL file. Defaults to newest file under ../input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "..", "input")
    input_path = args.input or latest_input_file(input_dir)

    log.info("Using input file: %s", input_path)
    records = load_records(input_path)
    if not records:
        raise ValueError("No records found in input file")

    max_docs = int(os.getenv("AF_MAX_DOCS", "0"))
    if max_docs and len(records) > max_docs:
        # Sampled, not truncated: exports are ordered by project, so the first
        # N records are a biased slice rather than a smaller version of the set.
        keep = sorted(random.Random(BENCH_SEED).sample(range(len(records)), max_docs))
        records = [records[idx] for idx in keep]
        log.warning("AF_MAX_DOCS=%d active: random sample of %d records (seed %d)", max_docs, len(records), BENCH_SEED)

    report, benchmark, benchmark_results = analyse_fields(records)
    report_text = build_report(input_path, records, report, benchmark, benchmark_results)
    print(report_text)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    reports_dir = os.path.join(base_dir, "..", "reports")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"field_audit_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report_text + "\n")

    log.info("Wrote report to %s", out_path)
    log.info("Total elapsed: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
