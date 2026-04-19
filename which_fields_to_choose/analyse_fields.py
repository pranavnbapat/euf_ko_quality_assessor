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
    return isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj[: min(20, len(obj))])


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
        with open(path, "r", encoding="utf-8") as fh:
            root = json.load(fh)
        records = extract_records(root)
        log.info("Loaded %d records from JSON in %.1fs", len(records), time.time() - t0)
        return records

    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
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

    scalar_children = 0
    for key, child in value.items():
        child_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            flatten_dict(child, child_key, depth + 1, out)
        else:
            out[child_key] = child
            scalar_children += 1

    if scalar_children == 0 and prefix:
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


def mm(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def find_query_fields(field_names: list[str]) -> list[str]:
    lower_map = {name.lower(): name for name in field_names}
    preferred = []
    for needle in ("title", "name", "headline", "keyword", "description", "subtitle"):
        for lname, original in lower_map.items():
            if needle in lname and original not in preferred:
                preferred.append(original)
    return preferred[:3]


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


def self_retrieval_score(
    records_by_field: dict[str, list[str]],
    candidate_fields: list[str],
    query_fields: list[str],
) -> float:
    n_docs = len(next(iter(records_by_field.values()))) if records_by_field else 0
    if n_docs == 0:
        return 0.0

    docs_tokens: list[list[str]] = []
    queries: list[list[str]] = []
    for idx in range(n_docs):
        doc_tokens: list[str] = []
        query_text: list[str] = []
        for field in candidate_fields:
            doc_tokens.extend(tok(records_by_field[field][idx]))
        for field in query_fields:
            query_text.append(records_by_field[field][idx])
        docs_tokens.append(doc_tokens)
        queries.append(tok(" ".join(query_text)))

    if not any(doc_tokens for doc_tokens in docs_tokens):
        return 0.0

    idf, dl, avgdl, k1, b = bm25_prepare(docs_tokens)
    den_vec = k1 * (1 - b + b * (dl / (avgdl or 1.0))) + 1e-12

    postings_docs: dict[str, list[int]] = defaultdict(list)
    postings_tfs: dict[str, list[int]] = defaultdict(list)
    for doc_id, tokens in enumerate(docs_tokens):
        if not tokens:
            continue
        tf = Counter(tokens)
        for term, freq in tf.items():
            postings_docs[term].append(doc_id)
            postings_tfs[term].append(freq)

    postings = {
        term: (
            np.fromiter(doc_ids, dtype=np.int32),
            np.fromiter(postings_tfs[term], dtype=np.float64),
        )
        for term, doc_ids in postings_docs.items()
    }

    reciprocal_rank = 0.0
    scored_queries = 0
    idxs = np.arange(n_docs)

    for doc_id, query_tokens in enumerate(queries):
        if not query_tokens:
            continue
        scored_queries += 1
        scores = np.zeros(n_docs, dtype=np.float64)
        for term in set(query_tokens):
            weight = idf.get(term)
            if weight is None or term not in postings:
                continue
            hit_doc_ids, tfs = postings[term]
            denom = den_vec[hit_doc_ids]
            scores[hit_doc_ids] += weight * ((tfs * (k1 + 1.0)) / (tfs + denom))

        if n_docs > TOPK:
            kth = np.partition(scores, -TOPK)[-TOPK]
            if scores[doc_id] < kth:
                continue

        score_i = scores[doc_id]
        rank = int(((scores > score_i).sum()) + ((scores == score_i) & (idxs < doc_id)).sum() + 1)
        if rank <= TOPK:
            reciprocal_rank += 1.0 / rank

    return reciprocal_rank / max(1, scored_queries)


def build_query_texts(per_field_text: dict[str, list[str]], n_docs: int) -> list[list[str]]:
    preferred_query_fields = [
        "title_llm",
        "keywords_llm",
        "description_llm",
        "title",
        "keywords",
        "description",
    ]
    active_fields = [field for field in preferred_query_fields if field in per_field_text]
    queries: list[list[str]] = []
    for idx in range(n_docs):
        parts = [per_field_text[field][idx] for field in active_fields]
        queries.append(tok(" ".join(parts)))
    return queries


def list_value_set(values: dict[str, list[Any]], field: str, idx: int) -> set[str]:
    if field not in values:
        return set()
    return {x.strip().lower() for x in atomic_values(values[field][idx]) if x.strip()}


def scalar_value(values: dict[str, list[Any]], field: str, idx: int) -> str:
    if field not in values:
        return ""
    atoms = atomic_values(values[field][idx])
    return atoms[0].strip().lower() if atoms else ""


def graded_qrels(per_field_values: dict[str, list[Any]], n_docs: int) -> list[dict[int, int]]:
    qrels: list[dict[int, int]] = []
    for idx in range(n_docs):
        rels: dict[int, int] = {idx: 3}
        source_project = scalar_value(per_field_values, "project_id", idx)
        source_acronym = scalar_value(per_field_values, "project_acronym", idx)
        source_category = scalar_value(per_field_values, "category", idx)
        source_themes = list_value_set(per_field_values, "themes", idx)
        source_topics = list_value_set(per_field_values, "topics", idx)
        source_subcats = list_value_set(per_field_values, "subcategories", idx)

        for jdx in range(n_docs):
            if jdx == idx:
                continue
            grade = 0
            if source_project and source_project == scalar_value(per_field_values, "project_id", jdx):
                grade = max(grade, 2)
            if source_acronym and source_acronym == scalar_value(per_field_values, "project_acronym", jdx):
                grade = max(grade, 2)

            overlap = 0
            overlap += len(source_themes & list_value_set(per_field_values, "themes", jdx))
            overlap += len(source_topics & list_value_set(per_field_values, "topics", jdx))
            overlap += len(source_subcats & list_value_set(per_field_values, "subcategories", jdx))
            if overlap > 0:
                grade = max(grade, 1)
            if source_category and source_category == scalar_value(per_field_values, "category", jdx):
                grade = max(grade, 1)

            if grade > 0:
                rels[jdx] = grade
        qrels.append(rels)
    return qrels


def rank_docs_for_fields(
    per_field_text: dict[str, list[str]],
    candidate_fields: list[str],
    queries: list[list[str]],
) -> list[list[int]]:
    n_docs = len(queries)
    docs_tokens: list[list[str]] = []
    for idx in range(n_docs):
        tokens: list[str] = []
        for field in candidate_fields:
            if field in per_field_text:
                tokens.extend(tok(per_field_text[field][idx]))
        docs_tokens.append(tokens)

    if not any(tokens for tokens in docs_tokens):
        return [[] for _ in range(n_docs)]

    idf, dl, avgdl, k1, b = bm25_prepare(docs_tokens)
    den_vec = k1 * (1 - b + b * (dl / (avgdl or 1.0))) + 1e-12

    postings_docs: dict[str, list[int]] = defaultdict(list)
    postings_tfs: dict[str, list[int]] = defaultdict(list)
    for doc_id, tokens in enumerate(docs_tokens):
        if not tokens:
            continue
        tf = Counter(tokens)
        for term, freq in tf.items():
            postings_docs[term].append(doc_id)
            postings_tfs[term].append(freq)

    postings = {
        term: (
            np.fromiter(doc_ids, dtype=np.int32),
            np.fromiter(postings_tfs[term], dtype=np.float64),
        )
        for term, doc_ids in postings_docs.items()
    }

    rankings: list[list[int]] = []
    for query_tokens in queries:
        if not query_tokens:
            rankings.append([])
            continue
        scores = np.zeros(n_docs, dtype=np.float64)
        for term in set(query_tokens):
            weight = idf.get(term)
            if weight is None or term not in postings:
                continue
            hit_doc_ids, tfs = postings[term]
            denom = den_vec[hit_doc_ids]
            scores[hit_doc_ids] += weight * ((tfs * (k1 + 1.0)) / (tfs + denom))
        ranked = np.argsort(-scores, kind="stable")[:TOPK]
        rankings.append(ranked.tolist())
    return rankings


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

    denom = max(1, n_queries)
    return {
        "mrr_at_k": mrr / denom,
        "recall_at_k": recall / denom,
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
    return lines


def analyse_fields(
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[tuple[tuple[str, ...], float]], list[tuple[str, dict[str, float], list[str]]]]:
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

        idf, _, _, _, _ = bm25_prepare(token_lists)
        avg_idf = float(np.mean(list(idf.values()))) if idf else 0.0

        exact_unique = len(set(non_empty_texts)) / max(1, non_empty_count)
        boilerplate = 1.0 - exact_unique

        atoms: list[str] = []
        for value in non_empty_values:
            atoms.extend(atomic_values(value))
        distinct_atoms = len(set(atoms))
        cardinality_ratio = distinct_atoms / max(1, non_empty_count)

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
            "boilerplate": boilerplate,
            "cardinality_ratio": cardinality_ratio,
            "distinct_atoms": distinct_atoms,
            "array_ratio": array_ratio,
            "object_ratio": object_ratio,
            "likely_enum": likely_enum,
            "mapping_hint": mapping_hint,
            "flags": flags,
        }

    eligible_text = [
        field for field, stats in report.items()
        if stats["coverage"] >= MIN_COVERAGE
        and stats["dominant_kind"] in {"string", "string_list"}
        and stats["mapping_hint"] in {"text", "text+keyword", "keyword"}
        and stats["avg_tokens"] >= 2.0
    ]

    query_fields = [field for field in find_query_fields(field_names) if report.get(field, {}).get("coverage", 0.0) >= MIN_COVERAGE]
    retrieval_results: list[tuple[tuple[str, ...], float]] = []
    if eligible_text and query_fields:
        top_fields = sorted(
            eligible_text,
            key=lambda field: (
                report[field]["coverage"],
                report[field]["avg_idf"],
                report[field]["avg_tokens"],
            ),
            reverse=True,
        )[:6]
        candidates = {(field,) for field in top_fields}
        if "title" in top_fields and any(field in top_fields for field in ("description", "ko_content_flat", "keywords")):
            for extra in ("description", "ko_content_flat", "keywords"):
                if extra in top_fields and extra != "title":
                    candidates.add(tuple(sorted(("title", extra))))
            combo = tuple(sorted([field for field in ("title", "description", "ko_content_flat", "keywords") if field in top_fields]))
            if len(combo) >= 2:
                candidates.add(combo)

        for candidate in sorted(candidates):
            score = self_retrieval_score(per_field_text, list(candidate), query_fields)
            retrieval_results.append((candidate, score))

    text_fields = [field for field, stats in report.items() if stats["coverage"] >= MIN_COVERAGE]
    if text_fields:
        cov_norm = dict(zip(text_fields, mm([report[field]["coverage"] for field in text_fields])))
        idf_norm = dict(zip(text_fields, mm([report[field]["avg_idf"] for field in text_fields])))
        tokens_norm = dict(zip(text_fields, mm([min(report[field]["avg_tokens"], 200.0) for field in text_fields])))
        low_boiler_norm = dict(zip(text_fields, mm([1.0 - report[field]["boilerplate"] for field in text_fields])))
        sort_bias = dict(
            zip(
                text_fields,
                [
                    1.0 if report[field]["mapping_hint"] == "date"
                    else 0.9 if report[field]["mapping_hint"] in {"long", "float"}
                    else 0.7 if report[field]["mapping_hint"] == "keyword" and report[field]["array_ratio"] < 0.3
                    else 0.0
                    for field in text_fields
                ],
            )
        )
        facet_balance = dict(
            zip(
                text_fields,
                [
                    1.0 - min(abs(report[field]["cardinality_ratio"] - 0.25) / 0.25, 1.0)
                    if report[field]["mapping_hint"] == "keyword" or report[field]["likely_enum"]
                    else 0.0
                    for field in text_fields
                ],
            )
        )

        retrieval_map = {candidate[0][0]: candidate[1] for candidate in retrieval_results if len(candidate[0]) == 1}
        retr_norm = dict(zip(text_fields, mm([retrieval_map.get(field, 0.0) for field in text_fields])))

        for field in text_fields:
            stats = report[field]
            # Heuristic scoring layer:
            # - full_text_score rewards populated, discriminative, information-rich fields
            #   with some support from the self-retrieval proxy
            #     0.35 * coverage_norm
            #   + 0.20 * avg_idf_norm
            #   + 0.15 * avg_tokens_norm
            #   + 0.10 * low_boilerplate_norm
            #   + 0.20 * retrieval_norm
            # - facet_score rewards populated fields with stable keyword/date/numeric behaviour
            #   and a "reasonable" cardinality for filters/facets
            #     0.40 * coverage_norm
            #   + 0.35 * facet_balance
            #   + 0.15 * low_boilerplate_norm
            #   + 0.10 * type_bonus
            # - sort_score rewards populated scalar fields, especially dates and numerics
            #     0.50 * coverage_norm
            #   + 0.35 * sort_bias
            #   + 0.15 * low_boilerplate_norm
            #
            # These are hand-tuned heuristics for ranking candidates, not calibrated or learned scores.
            full_text_score = (
                0.35 * cov_norm[field]
                + 0.20 * idf_norm[field]
                + 0.15 * tokens_norm[field]
                + 0.10 * low_boiler_norm[field]
                + 0.20 * retr_norm[field]
            )
            if stats["mapping_hint"] == "keyword" and not stats["likely_enum"]:
                full_text_score *= 0.45
            if stats["mapping_hint"] == "keyword" and not stats["flags"]["is_name_like"]:
                full_text_score *= 0.35
            if stats["dominant_kind"] not in {"string", "string_list"}:
                full_text_score *= 0.2
            if stats["flags"]["is_hashy"] or stats["flags"]["is_id_like"] or stats["flags"]["is_url_like"]:
                full_text_score *= 0.05
            elif stats["flags"]["is_internal"] and not stats["flags"]["is_name_like"]:
                full_text_score *= 0.15

            facet_score = (
                0.40 * cov_norm[field]
                + 0.35 * facet_balance[field]
                + 0.15 * low_boiler_norm[field]
                + 0.10 * (1.0 if stats["mapping_hint"] in {"keyword", "date", "boolean", "long", "float"} else 0.0)
            )
            if stats["dominant_kind"] in {"object", "object_list", "mixed_list"}:
                facet_score *= 0.2
            if stats["flags"]["is_internal"] and stats["flags"]["is_hashy"]:
                facet_score *= 0.65

            sort_score = (
                0.50 * cov_norm[field]
                + 0.35 * sort_bias[field]
                + 0.15 * low_boiler_norm[field]
            )
            if stats["array_ratio"] > 0.5:
                sort_score *= 0.2
            if stats["flags"]["is_hashy"] or stats["flags"]["is_url_like"]:
                sort_score *= 0.3

            stats["full_text_score"] = full_text_score
            stats["facet_score"] = facet_score
            stats["sort_score"] = sort_score
        preferred = resolve_preferred_fields(report)
        query_tokens = build_query_texts(per_field_text, n_docs)
        qrels = graded_qrels(per_field_values, n_docs)
        benchmark_results: list[tuple[str, dict[str, float], list[str]]] = []
        for label, fields in build_ablation_candidates(preferred["full_text"]):
            rankings = rank_docs_for_fields(per_field_text, fields, query_tokens)
            metrics = evaluate_bundle(rankings, qrels, TOPK)
            benchmark_results.append((label, metrics, fields))
        return report, retrieval_results, benchmark_results

    return report, retrieval_results, []


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

    if stats["flags"]["is_url_like"] and full_text < 0.35:
        return "skip_display_only"
    if stats["mapping_hint"] == "keyword" and not stats["flags"]["is_name_like"] and facet >= 0.30:
        return "facet_filter" if facet >= sort_score else "sort_filter"

    if full_text >= max(facet, sort_score, 0.45):
        return "full_text"
    if facet >= max(full_text, sort_score, 0.35):
        return "facet_filter"
    if sort_score >= 0.35:
        return "sort_filter"
    if stats["mapping_hint"] in {"text", "text+keyword", "keyword", "date", "long", "float", "boolean"}:
        return "index_optional"
    return "skip"


def resolve_preferred_fields(report: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    full_text_candidates = [
        field for field, stats in report.items()
        if choose_role(stats) in {"full_text", "index_optional", "sort_filter"}
    ]
    facet_candidates = [
        field for field, stats in report.items()
        if choose_role(stats) in {"facet_filter", "sort_filter"}
    ]

    llm_priority = [
        "title_llm",
        "subtitle_llm",
        "description_llm",
        "keywords_llm",
        "ko_content_flat_summarised",
    ]
    chosen_full_text: list[str] = []
    seen = set()
    for field in llm_priority:
        if field in full_text_candidates and field not in seen:
            chosen_full_text.append(field)
            seen.add(field)

    for fallback in ("project_name",):
        if fallback in full_text_candidates and fallback not in seen:
            chosen_full_text.append(fallback)
            seen.add(fallback)

    # Acronyms are operationally keyword-like, but we still want them searchable
    # in the final recommendation because users may search directly by acronym.
    if "project_acronym" in report and "project_acronym" not in seen:
        chosen_full_text.append("project_acronym")
        seen.add("project_acronym")

    preferred_facets = [
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
    chosen_facets = [field for field in preferred_facets if field in facet_candidates]

    return {
        "full_text": chosen_full_text,
        "facets": chosen_facets,
    }


def build_report(
    input_path: str,
    records: list[dict[str, Any]],
    report: dict[str, dict[str, Any]],
    retrieval_results: list[tuple[tuple[str, ...], float]],
    benchmark_results: list[tuple[str, dict[str, float], list[str]]],
) -> str:
    lines: list[str] = []
    lines.append("=== DATASET SUMMARY ===")
    lines.append(f"Input file            : {input_path}")
    lines.append(f"Records analysed      : {len(records)}")
    lines.append(f"Fields analysed       : {len(report)}")
    lines.append(f"Coverage threshold    : {MIN_COVERAGE:.2f}")

    preferred = resolve_preferred_fields(report)
    lines.append("\n=== FINAL PICK FOR THIS FILE ===")
    lines.append("Full-text fields      : " + (", ".join(preferred["full_text"]) or "-"))
    lines.append("Facet/sort fields     : " + (", ".join(preferred["facets"]) or "-"))
    lines.append("Excluded by policy    : ko_content_flat, title, subtitle, description, keywords, fingerprints, hashes, URLs, DOIs")

    lines.append("\n=== FIELD DIAGNOSTICS ===")
    header = (
        f"{'field':34} {'role':16} {'map':11} {'cov':>5} {'tok':>6} "
        f"{'idf':>6} {'card':>6} {'boil':>6} {'kind':18}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    ranked_fields = sorted(
        report.items(),
        key=lambda item: (
            choose_role(item[1]) not in {"full_text", "facet_filter", "sort_filter", "index_optional"},
            -(item[1].get("full_text_score", 0.0) + item[1].get("facet_score", 0.0)),
            -item[1]["coverage"],
            item[0],
        ),
    )
    for field, stats in ranked_fields:
        lines.append(
            f"{field[:34]:34} {choose_role(stats):16} {stats['mapping_hint']:11} "
            f"{stats['coverage']:5.2f} {stats['avg_tokens']:6.1f} {stats['avg_idf']:6.3f} "
            f"{stats['cardinality_ratio']:6.2f} {stats['boilerplate']:6.2f} {stats['dominant_kind'][:18]:18}"
        )

    preferred_full_text_fields = [
        (field, report[field]) for field in preferred["full_text"] if field in report
    ]

    facet_fields = [
        (field, stats) for field, stats in report.items()
        if choose_role(stats) in {"facet_filter", "sort_filter"}
    ]
    facet_fields.sort(key=lambda item: (item[1].get("facet_score", 0.0), item[1].get("sort_score", 0.0)), reverse=True)
    identifier_fields = [
        (field, stats) for field, stats in report.items()
        if choose_role(stats) == "identifier_only"
    ]
    identifier_fields.sort(key=lambda item: item[0])

    lines.append("\n=== RECOMMENDED FULL-TEXT FIELDS ===")
    if preferred_full_text_fields:
        for field, stats in preferred_full_text_fields:
            lines.append(
                f"{field:34} score={stats.get('full_text_score', 0.0):.3f}  "
                f"mapping={stats['mapping_hint']}  coverage={stats['coverage']:.2f}  avg_tokens={stats['avg_tokens']:.1f}"
            )
    else:
        lines.append("No strong full-text fields found.")

    lines.append("\n=== RECOMMENDED FILTER / SORT FIELDS ===")
    if facet_fields:
        for field, stats in facet_fields[:15]:
            lines.append(
                f"{field:34} facet={stats.get('facet_score', 0.0):.3f}  sort={stats.get('sort_score', 0.0):.3f}  "
                f"mapping={stats['mapping_hint']}  card={stats['cardinality_ratio']:.2f}"
            )
    else:
        lines.append("No strong facet/sort fields found.")

    lines.append("\n=== IDENTIFIER-ONLY FIELDS ===")
    if identifier_fields:
        for field, stats in identifier_fields:
            lines.append(
                f"{field:34} mapping={stats['mapping_hint']}  coverage={stats['coverage']:.2f}"
            )
    else:
        lines.append("No identifier-only fields found.")

    lines.append("\n=== SUGGESTED MAPPINGS ===")
    suggested_names = preferred["full_text"] + preferred["facets"]
    suggested = [(field, report[field]) for field in suggested_names if field in report]
    for field, stats in suggested:
        lines.append(mapping_snippet(field, stats["mapping_hint"]))

    lines.append(f"\n=== PSEUDO-QUERY ABLATION BENCHMARK (TOP {TOPK}) ===")
    if benchmark_results:
        lines.append("Uses standard IR metrics on weak pseudo-labels built from self, project, and shared taxonomy overlap.")
        for label, metrics, fields in benchmark_results:
            lines.append(
                f"{label:28} MRR@{TOPK}={metrics['mrr_at_k']:.4f}  "
                f"Recall@{TOPK}={metrics['recall_at_k']:.4f}  "
                f"nDCG@{TOPK}={metrics['ndcg_at_k']:.4f}  "
                f"fields={', '.join(fields)}"
            )
    else:
        lines.append("Skipped: no benchmark candidates were generated.")

    lines.append("\n=== PLAIN-ENGLISH ABLATION SUMMARY ===")
    for line in summarise_ablation_results(benchmark_results):
        lines.append(line)

    if retrieval_results:
        lines.append(f"\n=== SELF-RETRIEVAL PROXY (MRR@{TOPK}) ===")
        for fields, score in sorted(retrieval_results, key=lambda item: item[1], reverse=True):
            lines.append(f"{' + '.join(fields):34} score={score:.4f}")
    else:
        lines.append(f"\n=== SELF-RETRIEVAL PROXY (MRR@{TOPK}) ===")
        lines.append("Skipped: no reliable query-like fields were detected.")

    lines.append("\n=== NOTES ===")
    lines.append("Use `text` fields for ranking and `keyword`/`date`/numeric fields for facets, filters, and sorting.")
    lines.append("Fields marked `review_structure` are too nested or mixed to map safely without manual design.")
    lines.append("Short repeated label fields often belong in `keyword`; long narrative fields belong in `text`.")
    lines.append("The pseudo-query benchmark is stronger than raw heuristics, but still weaker than judged query relevance data.")
    lines.append("The most defensible next step is real query evaluation with labeled relevance and field ablations.")

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
        records = records[:max_docs]
        log.warning("AF_MAX_DOCS=%d active: using first %d records", max_docs, len(records))

    report, retrieval_results, benchmark_results = analyse_fields(records)
    report_text = build_report(input_path, records, report, retrieval_results, benchmark_results)
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
