# assess_ko_quality/domain_gate.py
"""
Domain relevance as a GATE, not as a scored pillar.

Why a gate. Relevance is a precondition, not a quality you can have more or less of.
Scored as 10% of a weighted total, a well-written off-topic upload still passes: three
off-domain documents (Bach, async servers, atrial fibrillation) written to be
structurally strong scored 75.2-75.8 on the four-pillar total, above every real
agricultural KO in the same run. Relevance has to decide whether the KO is scored at
all, rather than trading against good formatting.

Why a classifier rather than the AGROVOC/NALT centroid. Measured on the validation set
in gate_validation/ (anchor rows on the 280-item v1 set, classifier on the 678-item v2):

    mean cosine to the anchor centroid   AUC 0.647   (what quality_domain_kc.py does)
    mean of top-10 anchor similarities   AUC 0.768
    logistic regression on embeddings    AUC 1.000   (5-fold CV, 400 in / 130 out)

Thesaurus anchors match any text containing biological or land-use vocabulary, so
"Landscape painting" scored above the in-domain mean. The 10,468 real KOs define the
domain far better than a thesaurus does: they ARE the domain.

The classifier is trained on real KOs as positives and off-domain Wikipedia as
negatives. In-domain Wikipedia articles are held out and still classified in-domain,
so the model learned the domain rather than "Wikipedia means reject".

Usage:
    python domain_gate.py --calibrate          # train, report operating points, save model
    python domain_gate.py --score kos.json     # apply the gate to a KO export
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATION_SET = HERE.parent / "validation" / "gate_validation" / "gate_validation_set.json"
MODEL_PATH = HERE.parent / "validation" / "gate_validation" / "domain_gate_model.pkl"

EMB_MODEL_NAME = "all-mpnet-base-v2"
MAX_CHARS = 2000

# Calibrated on the v2 validation set with out-of-fold probabilities. Higher P = more
# clearly in scope. Between the two bounds the KO goes to a human rather than being
# auto-rejected, because the near-miss boundary is not yet adjudicated.
#
# ACCEPT_AT is deliberately not set higher. Retention is 100% for every language at
# 0.45 and below, but a language gap opens above it: at 0.70, German retention falls to
# 83% and Greek to 33%. Rejecting a legitimate KO because of the language it is written
# in is a worse failure than letting a few off-domain items through to review, and
# roughly a third of the corpus is not in English.
ACCEPT_AT = 0.40   # keeps 100% of real KOs in all 14 languages, catches 94.6% of off-domain
REVIEW_AT = 0.20   # below this, auto-reject: 56.2% of off-domain, 0% of real KOs


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMB_MODEL_NAME)


def embed(texts: List[str], model=None) -> np.ndarray:
    model = model or _load_embedder()
    return model.encode([t[:MAX_CHARS] for t in texts], normalize_embeddings=True,
                        batch_size=16, show_progress_bar=False).astype(np.float32)


def load_validation_set(path: Path = VALIDATION_SET) -> List[Dict[str, Any]]:
    if not path.exists():
        sys.exit(f"Validation set not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def _embed_cached(items: List[Dict[str, Any]], path: Path) -> np.ndarray:
    """Embed the validation set, caching by item count and id digest."""
    import hashlib
    digest = hashlib.sha256("|".join(i["id"] for i in items).encode()).hexdigest()[:16]
    cache = path.parent / f".emb_cache_{len(items)}_{digest}.npy"
    if cache.exists():
        print(f"using cached embeddings: {cache.name}")
        return np.load(cache)
    E = embed([i["text"] for i in items])
    np.save(cache, E)
    return E


def calibrate(path: Path = VALIDATION_SET, save: bool = True,
              train_on_near_miss: bool = False) -> None:
    """
    Train on KOs vs off-domain Wikipedia, report honest out-of-fold operating points.

    `near_miss` items are excluded from training by default: their labels are a first
    pass and some are plausibly in scope, so training on them would bake in guesses.
    They are reported instead, so a domain owner can adjudicate and then opt in with
    train_on_near_miss=True.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    items = load_validation_set(path)
    E = _embed_cached(items, path)

    neg_labels = {"off", "near_miss"} if train_on_near_miss else {"off"}
    train = [n for n, i in enumerate(items)
             if (i["source"] == "eu-farmbook" and i["label"] == "in")
             or (i["source"] == "wikipedia" and i["label"] in neg_labels)]
    X = E[train]
    y = np.array([items[n]["label"] == "in" for n in train], dtype=int)

    oof = np.zeros(len(y))
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))

    print(f"5-fold CV AUC: {np.mean(aucs):.3f} (sd {np.std(aucs):.3f})   "
          f"n={len(y)} ({int(y.sum())} in, {int((1 - y).sum())} out; "
          f"near_miss {'in' if train_on_near_miss else 'NOT in'} training)")
    pos, neg = oof[y == 1], oof[y == 0]
    print(f"\n{'P(in-domain) >=':>16} {'in-domain kept':>15} {'off-domain caught':>18}")
    for t in (0.10, 0.30, 0.50, 0.70, 0.90):
        marker = "  <- ACCEPT_AT" if t == ACCEPT_AT else ("  <- REVIEW_AT" if t == REVIEW_AT else "")
        print(f"{t:16.2f} {100 * (pos >= t).mean():14.1f}% {100 * (neg < t).mean():17.1f}%{marker}")

    # No use shipping a gate that rejects a language.
    # A gate must not double as a quality filter: a badly extracted in-domain KO is
    # still in domain. in_ko_short exists to catch that failure mode.
    print("\nin-domain retention by stratum at ACCEPT_AT:")
    strata: Dict[str, List[float]] = {}
    for n, p_ in zip(train, oof):
        it = items[n]
        if it["source"] == "eu-farmbook":
            strata.setdefault(it.get("stratum", "?"), []).append(p_)
    for st, ps in sorted(strata.items()):
        kept = 100 * np.mean(np.array(ps) >= ACCEPT_AT)
        flag = "   <-- CHECK" if kept < 95 else ""
        print(f"  {st:16s} n={len(ps):3d}  kept {kept:5.1f}%{flag}")

    print("\nin-domain retention by language at ACCEPT_AT:")
    langs: Dict[str, List[float]] = {}
    for n, p in zip(train, oof):
        it = items[n]
        if it["source"] != "eu-farmbook":
            continue
        langs.setdefault((it.get("lang") or "?").split("|")[0], []).append(p)
    for lg, ps in sorted(langs.items()):
        if len(ps) < 2:
            continue
        kept = 100 * np.mean(np.array(ps) >= ACCEPT_AT)
        print(f"  {lg:10s} n={len(ps):3d}  kept {kept:5.1f}%" + ("   <-- CHECK" if kept < 95 else ""))

    clf = LogisticRegression(max_iter=2000).fit(X, y)

    held = [n for n, i in enumerate(items) if i["source"] == "wikipedia" and i["label"] == "in"]
    if held:
        p = clf.predict_proba(E[held])[:, 1]
        print(f"\nsource-confound check - in-domain Wikipedia never trained as positive:")
        print(f"  n={len(held)} mean P={p.mean():.3f}, accepted {100 * (p >= ACCEPT_AT).mean():.0f}%")

    near = [n for n, i in enumerate(items) if i["label"] == "near_miss"]
    if near and not train_on_near_miss:
        p = clf.predict_proba(E[near])[:, 1]
        counts = collections.Counter(decide(pi) for pi in p)
        print(f"\nnear-miss items, held out of training (n={len(near)}): {dict(counts)}")
        print("  lowest-scoring (most clearly out):")
        order = sorted(zip(near, p), key=lambda t: t[1])
        for n, pi in order[:8]:
            print(f"    P={pi:.3f}  {decide(pi):6s}  {items[n]['title']}")
        print("  highest-scoring (model says in scope - adjudicate these):")
        for n, pi in order[-8:]:
            print(f"    P={pi:.3f}  {decide(pi):6s}  {items[n]['title']}")

    if save:
        MODEL_PATH.write_bytes(pickle.dumps({"clf": clf, "emb_model": EMB_MODEL_NAME,
                                             "accept_at": ACCEPT_AT, "review_at": REVIEW_AT}))
        print(f"\nsaved {MODEL_PATH}")


def decide(p: float) -> str:
    if p >= ACCEPT_AT:
        return "accept"
    if p >= REVIEW_AT:
        return "review"
    return "reject"


def score_file(kos_path: Path) -> None:
    """Apply the gate to a KO export and print the decision for each record."""
    if not MODEL_PATH.exists():
        sys.exit(f"No model at {MODEL_PATH}; run --calibrate first.")
    bundle = pickle.loads(MODEL_PATH.read_bytes())
    clf = bundle["clf"]

    raw = json.loads(kos_path.read_text(encoding="utf-8"))
    docs = raw["docs"] if isinstance(raw, dict) and "docs" in raw else raw
    texts, ids = [], []
    for d in docs:
        texts.append((d.get("ko_content_flat") or "") or (d.get("description") or ""))
        ids.append(d.get("_orig_id") or d.get("_id") or d.get("@id") or "?")

    probs = clf.predict_proba(embed(texts))[:, 1]
    counts: Dict[str, int] = {}
    for i, p in zip(ids, probs):
        v = decide(p)
        counts[v] = counts.get(v, 0) + 1
        print(f"{p:.3f}\t{v}\t{i}")
    print(f"\n{counts}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibrate", action="store_true", help="Train and report operating points")
    ap.add_argument("--score", type=Path, help="Apply the gate to a KO export JSON")
    ap.add_argument("--set", type=Path, default=VALIDATION_SET, help="Validation set path")
    ap.add_argument("--train-on-near-miss", action="store_true",
                    help="Also treat near_miss items as negatives (only after adjudicating them)")
    args = ap.parse_args()

    if args.calibrate:
        calibrate(args.set, train_on_near_miss=args.train_on_near_miss)
    elif args.score:
        score_file(args.score)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
