"""
When should search say "nothing matched, but here is related material"?

A cosine similarity of 0.62 means nothing on its own. What matters is the
observed probability that a result at that score is actually relevant. This
script measures that probability directly, by joining retrieval scores to the
LLM relevance grades already collected by compare_search_modes.py, and then
reports the score bands where a result is worth calling a match, worth showing
as related, or worth suppressing.

Everything it needs is cached by the main comparison run, so it is cheap to
re-run whenever the corpus or the model changes - which it must be, because a
threshold calibrated on one embedding model or one corpus does not transfer.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from compare_search_modes import (
    Bm25Index, build_corpus, derive_held_out_fields, derive_stopwords,
    detect_language, embed, judge_pool, latest_file, llm_queries, load_records,
    read_field_audit, statistical_queries, tok, DEFAULT_MODELS, FALLBACK_INDEX_FIELDS,
)
from llm_client import build_client_from_env


def bm25_scores(index: Bm25Index, query_tokens: list[str]) -> np.ndarray:
    scores = np.zeros(index.n_docs, dtype=np.float64)
    for term in set(query_tokens):
        weight = index.idf.get(term)
        hit = index.postings.get(term)
        if weight is None or hit is None:
            continue
        ids, tfs = hit
        scores[ids] += weight * ((tfs * (index.k1 + 1.0)) / (tfs + index.den_vec[ids]))
    return scores


def calibration_table(pairs: list[tuple[float, int]], n_bins: int, min_per_bin: int) -> list[dict[str, Any]]:
    """pairs = [(score, grade)]. Equal-count bins, so every row carries weight."""
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda p: p[0])
    per_bin = max(min_per_bin, len(pairs) // n_bins)
    rows = []
    for start in range(0, len(pairs), per_bin):
        chunk = pairs[start:start + per_bin]
        if len(chunk) < min_per_bin:
            if rows:  # fold a short tail into the previous bin
                prev = rows[-1]
                prev["n"] += len(chunk)
                prev["hi"] = chunk[-1][0]
                prev["p_rel"] = (prev["p_rel"] * (prev["n"] - len(chunk))
                                 + sum(1 for _, g in chunk if g > 0)) / prev["n"]
                prev["p_strong"] = (prev["p_strong"] * (prev["n"] - len(chunk))
                                    + sum(1 for _, g in chunk if g >= 2)) / prev["n"]
            continue
        rows.append({
            "lo": chunk[0][0], "hi": chunk[-1][0], "n": len(chunk),
            "p_rel": sum(1 for _, g in chunk if g > 0) / len(chunk),
            "p_strong": sum(1 for _, g in chunk if g >= 2) / len(chunk),
            "mean_grade": sum(g for _, g in chunk) / len(chunk),
        })
    return rows


def pick_threshold(rows: list[dict[str, Any]], target: float, key: str) -> float | None:
    """Lowest score band whose observed probability reaches `target`, and stays there."""
    for i, row in enumerate(rows):
        if row[key] >= target and all(r[key] >= target * 0.9 for r in rows[i:]):
            return row["lo"]
    return None


def main() -> None:
    t0 = time.time()
    base = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(base, ".."))
    workspace = os.path.abspath(os.path.join(repo, ".."))

    p = argparse.ArgumentParser(description="Calibrate the no-match / related-results threshold.")
    p.add_argument("--input")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--queries", type=int, default=260)
    p.add_argument("--judge-queries", type=int, default=260)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--min-per-bin", type=int, default=60)
    p.add_argument("--terms-per-query", type=int, default=8)
    p.add_argument("--stopword-df", type=float, default=0.25)
    p.add_argument("--stopword-min-docs", type=int, default=40)
    p.add_argument("--excerpt-chars", type=int, default=1800)
    p.add_argument("--judge-batch", type=int, default=10)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cache-dir", default=os.path.join(base, ".cache"))
    p.add_argument("--output-dir", default=os.path.join(base, "..", "reports"))
    args = p.parse_args()

    args.input = args.input or latest_file(os.path.join(repo, "input"))
    records = load_records(args.input)
    rng = random.Random(args.seed)
    if args.limit and len(records) > args.limit:
        keep = sorted(rng.sample(range(len(records)), args.limit))
        records = [records[i] for i in keep]

    reports_dir = os.path.abspath(args.output_dir)
    index_fields, _ = read_field_audit(reports_dir)
    index_fields = [f for f in (index_fields or FALLBACK_INDEX_FIELDS)
                    if any(f in r for r in records[:200])]
    held_out_fields = derive_held_out_fields(records, index_fields, 5.0, 6)
    indexed, held_out = build_corpus(records, index_fields, held_out_fields)
    n_docs = len(indexed)

    doc_langs = [detect_language(t) for t in indexed]
    held_tokens = [tok(t) for t in held_out]
    stop_per_lang, global_stop = derive_stopwords(
        held_tokens, doc_langs, args.stopword_df, args.stopword_min_docs)

    df = Counter()
    for t in held_tokens:
        if t:
            df.update(set(t))
    idf = {t: math.log((n_docs - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}

    eligible = [i for i, t in enumerate(held_tokens) if len(t) >= 40]
    targets = sorted(rng.sample(eligible, min(args.queries, len(eligible))))

    queries = statistical_queries(targets, indexed, held_tokens, idf,
                                  stop_per_lang, global_stop, doc_langs, args.terms_per_query)
    client = build_client_from_env(
        [os.path.join(workspace, "farm_assistant_um", ".env"),
         os.path.join(workspace, "scout", ".env"), os.path.join(repo, ".env")],
        os.path.join(args.cache_dir, "llm"))
    queries += llm_queries(client, targets, held_out, doc_langs,
                           args.excerpt_chars, args.workers, args.seed)
    query_texts = [q["text"] for q in queries]
    q_targets = [q["target"] for q in queries]
    print(f"[info] {n_docs} documents, {len(queries)} queries")

    # Scores from both retrievers
    cfg = DEFAULT_MODELS["dense_multi"]
    doc_vecs = embed(indexed, cfg["name"], args.cache_dir, "docs_dense_multi")
    q_vecs = embed(query_texts, cfg["name"], args.cache_dir, "q_dense_multi")
    bm25 = Bm25Index([tok(t) for t in indexed])

    judge_idx = sorted(rng.sample(range(len(queries)), min(args.judge_queries, len(queries))))
    snippets = [(indexed[i][:600] if indexed[i] else held_out[i][:600]) for i in range(n_docs)]

    dense_top: dict[int, list[tuple[int, float]]] = {}
    lex_top: dict[int, list[tuple[int, float]]] = {}
    pools: list[list[int]] = []
    for qi in judge_idx:
        sims = q_vecs[qi] @ doc_vecs.T
        d_ids = np.argpartition(-sims, args.topk)[: args.topk]
        d_ids = d_ids[np.argsort(-sims[d_ids])]
        dense_top[qi] = [(int(d), float(sims[d])) for d in d_ids]

        bs = bm25_scores(bm25, tok(query_texts[qi]))
        nz = np.flatnonzero(bs > 0)
        l_ids = nz[np.argsort(-bs[nz])][: args.topk] if nz.size else np.array([], dtype=int)
        lex_top[qi] = [(int(d), float(bs[d])) for d in l_ids]

        pool = list(dict.fromkeys([d for d, _ in dense_top[qi]] + [d for d, _ in lex_top[qi]]))
        if q_targets[qi] not in pool:
            pool.append(q_targets[qi])
        pools.append(pool)

    print(f"[info] judging {len(judge_idx)} queries ({sum(len(p) for p in pools)} pairs; cached where seen)")
    qrels = judge_pool(client, [queries[i] for i in judge_idx], pools, snippets,
                       args.workers, args.judge_batch)
    grades = {qi: g for qi, g in zip(judge_idx, qrels)}

    dense_pairs = [(s, grades[qi].get(d, 0)) for qi in judge_idx for d, s in dense_top[qi]]
    lex_pairs = [(s, grades[qi].get(d, 0)) for qi in judge_idx for d, s in lex_top[qi]]

    dense_rows = calibration_table(dense_pairs, args.bins, args.min_per_bin)
    lex_rows = calibration_table(lex_pairs, args.bins, args.min_per_bin)

    # Per-query top score, to decide whether a whole result SET is a match
    set_pairs = []
    for qi in judge_idx:
        if not dense_top[qi]:
            continue
        best = dense_top[qi][0]
        any_rel = 1 if any(grades[qi].get(d, 0) >= 2 for d, _ in dense_top[qi][:3]) else 0
        set_pairs.append((best[1], any_rel))
    set_rows = calibration_table(set_pairs, 8, max(15, len(set_pairs) // 10))

    L: list[str] = []
    L.append("=== REJECTION THRESHOLD CALIBRATION ===")
    L.append(f"Input               : {args.input}")
    L.append(f"Documents           : {n_docs}")
    L.append(f"Judged queries      : {len(judge_idx)}")
    L.append(f"Graded pairs        : dense {len(dense_pairs)}, lexical {len(lex_pairs)}")
    L.append(f"Embedding model     : {cfg['name']}")
    L.append(f"LLM judge           : {client.model}")
    L.append(f"  {client.usage_summary()}")
    L.append(f"Run time            : {time.time() - t0:.0f}s")
    L.append("")
    L.append("A similarity score is not meaningful on its own. These tables give the")
    L.append("observed probability that a result at a given score is relevant, measured")
    L.append("against LLM grades. Bands hold roughly equal numbers of results.")

    for title, rows, unit in (
        ("SEMANTIC (cosine similarity)", dense_rows, "cosine"),
        ("LEXICAL (BM25 score)", lex_rows, "bm25"),
    ):
        L.append("")
        L.append(f"--- {title} ---")
        L.append(f"  {'score band':>22} {'results':>8} {'P(relevant)':>12} {'P(strongly rel)':>16} {'mean grade':>11}")
        for r in rows:
            L.append(f"  {r['lo']:9.4f} .. {r['hi']:8.4f} {r['n']:8} {r['p_rel']:12.2f} "
                     f"{r['p_strong']:16.2f} {r['mean_grade']:11.2f}")

    L.append("")
    L.append("--- WHOLE RESULT SET: is there anything worth showing? ---")
    L.append("Top semantic score for the query, against whether any of the top 3 was")
    L.append("strongly relevant. This is the number a no-match decision should use.")
    L.append(f"  {'top score band':>22} {'queries':>8} {'P(usable answer)':>18}")
    for r in set_rows:
        L.append(f"  {r['lo']:9.4f} .. {r['hi']:8.4f} {r['n']:8} {r['p_rel']:18.2f}")

    match_at = pick_threshold(dense_rows, 0.60, "p_strong")
    show_at = pick_threshold(dense_rows, 0.40, "p_rel")
    set_at = pick_threshold(set_rows, 0.50, "p_rel")

    L.append("")
    L.append("=== RECOMMENDED BANDS ===")
    base_rate = sum(1 for _, g in dense_pairs if g > 0) / max(1, len(dense_pairs))
    strong_rate = sum(1 for _, g in dense_pairs if g >= 2) / max(1, len(dense_pairs))
    L.append(f"Base rate across all retrieved results: P(relevant)={base_rate:.2f}, "
             f"P(strongly relevant)={strong_rate:.2f}.")
    L.append("A threshold is only worth having if it beats that base rate.")
    L.append("")
    if match_at is not None:
        L.append(f"  MATCHES        cosine >= {match_at:.4f}   (strongly relevant more often than not)")
    else:
        L.append("  MATCHES        no band reached P(strongly relevant) >= 0.60 - do not claim matches on score alone")
    if show_at is not None:
        L.append(f"  RELATED        cosine >= {show_at:.4f}   (relevant often enough to be worth offering)")
    else:
        L.append("  RELATED        no band reached P(relevant) >= 0.40")
    L.append("  SUPPRESS       below the RELATED band - showing these costs more trust than it earns")
    L.append("")
    if set_at is not None:
        L.append(f"  Say 'no matching knowledge objects' when the top semantic score is below")
        L.append(f"  {set_at:.4f}; at that point fewer than half of result sets contain a usable answer.")
    else:
        L.append("  No top-score band reached a 50% chance of a usable answer: on this evidence the")
        L.append("  system should be cautious about claiming a match for any query.")

    L.append("")
    L.append("=== HOW TO USE THIS ===")
    L.append("  - Recalibrate whenever the embedding model, the index or the corpus changes.")
    L.append("    These numbers are properties of this model on this corpus, not constants.")
    L.append("  - Thresholds on raw cosine are only valid for the model measured here")
    L.append("    (paraphrase-multilingual-MiniLM-L12-v2). Another model has another scale.")
    L.append("  - Grades come from an LLM, not domain experts. Treat the bands as a starting")
    L.append("    point to validate against real judgements, not as a final answer.")
    L.append("  - A score gate is weaker than a reranker. If a cross-encoder is ever added,")
    L.append("    calibrate the gate on its score instead: it separates relevant from")
    L.append("    related far better than raw similarity does.")

    report = "\n".join(L)
    print(report)
    os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = os.path.join(reports_dir, f"rejection_calibration_{stamp}.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n[info] wrote {out}")


if __name__ == "__main__":
    main()
