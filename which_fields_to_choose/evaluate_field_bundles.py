"""
Evaluate OpenSearch field bundles against judged queries using standard IR metrics.

Judgments format (JSON):
[
  {
    "query": "anaerobic digestion nutrient management",
    "relevant": {
      "doc_id_1": 3,
      "doc_id_2": 2,
      "doc_id_3": 1
    }
  }
]
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from analyse_fields import (
    TOPK,
    analyse_fields,
    bm25_prepare,
    load_records,
    normalise_text,
    resolve_preferred_fields,
    tok,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate field bundles with judged queries.")
    parser.add_argument("--input", required=True, help="Path to JSON or JSONL file")
    parser.add_argument("--judgments", required=True, help="Path to query judgments JSON")
    parser.add_argument("--id-field", default="@id", help="Document field used as the qrels identifier")
    parser.add_argument("--topk", type=int, default=TOPK, help="Cutoff for ranking metrics")
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help="Explicit bundle in the form name=field1,field2,field3. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-ablation",
        action="store_true",
        help="Do not auto-evaluate the recommended bundle and its drop-one-field ablations.",
    )
    return parser.parse_args()


def load_judgments(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Judgments file must contain a JSON list")
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict) or "query" not in item or "relevant" not in item:
            raise ValueError(f"Judgment item {idx} must contain 'query' and 'relevant'")
        if not isinstance(item["relevant"], dict):
            raise ValueError(f"Judgment item {idx} field 'relevant' must be an object")
    return data


def build_doc_index(records: list[dict[str, Any]], id_field: str) -> tuple[list[str], dict[str, int]]:
    doc_ids: list[str] = []
    doc_index: dict[str, int] = {}
    for idx, record in enumerate(records):
        doc_id = normalise_text(record.get(id_field))
        if not doc_id:
            raise ValueError(f"Record {idx} missing id field '{id_field}'")
        if doc_id in doc_index:
            raise ValueError(f"Duplicate document id for '{id_field}': {doc_id}")
        doc_ids.append(doc_id)
        doc_index[doc_id] = idx
    return doc_ids, doc_index


def build_per_field_text(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    field_names = sorted({key for record in records for key in record.keys()})
    per_field_text = {field: [] for field in field_names}
    for record in records:
        for field in field_names:
            per_field_text[field].append(normalise_text(record.get(field)))
    return per_field_text


def rank_queries(
    per_field_text: dict[str, list[str]],
    bundle_fields: list[str],
    queries: list[str],
    topk: int,
) -> list[list[int]]:
    n_docs = len(next(iter(per_field_text.values()))) if per_field_text else 0
    docs_tokens: list[list[str]] = []
    for idx in range(n_docs):
        tokens: list[str] = []
        for field in bundle_fields:
            if field in per_field_text:
                tokens.extend(tok(per_field_text[field][idx]))
        docs_tokens.append(tokens)

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
    for query in queries:
        q_tokens = tok(query)
        scores = np.zeros(n_docs, dtype=np.float64)
        for term in set(q_tokens):
            weight = idf.get(term)
            if weight is None or term not in postings:
                continue
            hit_doc_ids, tfs = postings[term]
            denom = den_vec[hit_doc_ids]
            scores[hit_doc_ids] += weight * ((tfs * (k1 + 1.0)) / (tfs + denom))
        rankings.append(np.argsort(-scores, kind="stable")[:topk].tolist())
    return rankings


def dcg_at_k(ranked_doc_ids: list[str], rels: dict[str, int], k: int) -> float:
    total = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        gain = rels.get(doc_id, 0)
        if gain > 0:
            total += (2 ** gain - 1) / math.log2(rank + 1)
    return total


def evaluate_metrics(
    rankings: list[list[int]],
    doc_ids: list[str],
    judgments: list[dict[str, Any]],
    topk: int,
) -> dict[str, float]:
    mrr = 0.0
    recall = 0.0
    ndcg = 0.0
    ap_total = 0.0

    for ranked, item in zip(rankings, judgments):
        rels = {doc_id: int(score) for doc_id, score in item["relevant"].items()}
        relevant_binary = {doc_id for doc_id, score in rels.items() if score > 0}
        ranked_doc_ids = [doc_ids[idx] for idx in ranked]

        rr = 0.0
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids[:topk], start=1):
            if rels.get(doc_id, 0) > 0 and rr == 0.0:
                rr = 1.0 / rank
            if doc_id in relevant_binary:
                hits += 1
                precision_sum += hits / rank
        mrr += rr
        recall += hits / max(1, len(relevant_binary))
        ap_total += precision_sum / max(1, len(relevant_binary))

        ideal = sorted(rels.values(), reverse=True)
        ideal_dcg = sum((2 ** gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal[:topk], start=1))
        ndcg += dcg_at_k(ranked_doc_ids, rels, topk) / ideal_dcg if ideal_dcg > 0 else 0.0

    denom = max(1, len(judgments))
    return {
        "mrr_at_k": mrr / denom,
        "recall_at_k": recall / denom,
        "ndcg_at_k": ndcg / denom,
        "map": ap_total / denom,
    }


def parse_bundle_arg(bundle_arg: str) -> tuple[str, list[str]]:
    if "=" not in bundle_arg:
        raise ValueError(f"Bundle must be in name=field1,field2 form: {bundle_arg}")
    name, raw_fields = bundle_arg.split("=", 1)
    fields = [field.strip() for field in raw_fields.split(",") if field.strip()]
    if not name or not fields:
        raise ValueError(f"Bundle must contain a name and at least one field: {bundle_arg}")
    return name, fields


def default_bundles(records: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    report, _, _ = analyse_fields(records)
    recommended = resolve_preferred_fields(report)["full_text"]
    bundles: list[tuple[str, list[str]]] = []
    if recommended:
        bundles.append(("all_recommended", recommended))
        for field in recommended:
            ablated = [other for other in recommended if other != field]
            if ablated:
                bundles.append((f"drop_{field}", ablated))
    return bundles


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    judgments = load_judgments(args.judgments)
    doc_ids, _ = build_doc_index(records, args.id_field)
    per_field_text = build_per_field_text(records)

    bundles: list[tuple[str, list[str]]] = []
    if not args.no_default_ablation:
        bundles.extend(default_bundles(records))
    for bundle_arg in args.bundle:
        bundles.append(parse_bundle_arg(bundle_arg))
    if not bundles:
        raise ValueError("No field bundles to evaluate. Provide --bundle or allow the default ablation bundles.")

    queries = [item["query"] for item in judgments]
    print(f"Input file       : {args.input}")
    print(f"Judgments file   : {args.judgments}")
    print(f"ID field         : {args.id_field}")
    print(f"Queries          : {len(judgments)}")
    print(f"TopK             : {args.topk}")
    print("\n=== JUDGED FIELD-BUNDLE EVALUATION ===")

    baseline_ndcg = None
    for label, fields in bundles:
        rankings = rank_queries(per_field_text, fields, queries, args.topk)
        metrics = evaluate_metrics(rankings, doc_ids, judgments, args.topk)
        if baseline_ndcg is None:
            baseline_ndcg = metrics["ndcg_at_k"]
        delta = metrics["ndcg_at_k"] - baseline_ndcg
        print(
            f"{label:28} MRR@{args.topk}={metrics['mrr_at_k']:.4f}  "
            f"Recall@{args.topk}={metrics['recall_at_k']:.4f}  "
            f"nDCG@{args.topk}={metrics['ndcg_at_k']:.4f}  "
            f"MAP={metrics['map']:.4f}  "
            f"delta_nDCG={delta:+.4f}  "
            f"fields={', '.join(fields)}"
        )


if __name__ == "__main__":
    main()
