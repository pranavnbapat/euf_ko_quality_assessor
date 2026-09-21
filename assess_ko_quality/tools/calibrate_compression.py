# assess_ko_quality/calibrate_compression.py

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_tsv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """
    Minimal TSV writer. Keeps ordering stable for slide-friendly outputs.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(k) is None else str(r.get(k)) for k in fieldnames) + "\n")

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def token_bucket(tok: int) -> str:
    if tok < 500:
        return "<500"
    if tok < 1500:
        return "500-1499"
    if tok < 4000:
        return "1500-3999"
    return "4000+"

def iter_ko_objects(input_json: Path) -> List[Dict[str, Any]]:
    data = read_json(input_json)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError("Input must be a JSON object or list of objects")

def make_sample(
    items: List[Dict[str, Any]],
    n: int,
    seed: int,
    text_field: str,
    lang_field: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Compact sampling:
    - splits by language (if available) and rough token buckets using a cheap proxy (word-ish count).
    - keeps distribution reasonable without going full statistics lab.
    """
    rnd = random.Random(seed)

    def approx_tok(s: str) -> int:
        # cheap-ish token proxy for sampling only
        return len(re.findall(r"\w+|[^\w\s]", s or "", re.UNICODE))

    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for obj in items:
        txt = obj.get(text_field, "")
        if not isinstance(txt, str):
            txt = str(txt)
        tok = approx_tok(txt)
        lang = "unknown"
        if lang_field:
            lang = obj.get(lang_field) or "unknown"
        key = (str(lang), token_bucket(tok))
        groups.setdefault(key, []).append(obj)

    # allocate roughly proportional, with caps
    all_keys = list(groups.keys())
    rnd.shuffle(all_keys)

    sample: List[Dict[str, Any]] = []
    # round-robin draw to keep mix
    while len(sample) < min(n, len(items)) and all_keys:
        progressed = False
        for k in list(all_keys):
            if len(sample) >= n:
                break
            bucket = groups.get(k, [])
            if bucket:
                sample.append(bucket.pop(rnd.randrange(len(bucket))))
                progressed = True
            else:
                all_keys.remove(k)
        if not progressed:
            break

    return sample

def run_compression_diagnostics(
    script_path: Path,
    input_json: Path,
    output_json: Path,
    text_field: str,
    min_tokens: int,
    disable_embeddings: bool,
    device: str,
) -> None:
    cmd = [
        sys.executable,
        str(script_path),
        "--input", str(input_json),
        "--output", str(output_json),
        "--text-field", text_field,
        "--min-tokens", str(min_tokens),
        "--device", device,
    ]
    if disable_embeddings:
        cmd.append("--disable-embeddings")

    print(f"[{now_utc_iso()}] Running diagnostics: {' '.join(cmd)}")
    subprocess.check_call(cmd)

RATIOS = [0.15, 0.25, 0.35, 0.45]

def render_cmd_template(template: str, values: Dict[str, Any]) -> List[str]:
    """
    Template is a shell-like string; we split on whitespace after substitution.
    Keep it simple and repo-friendly.
    """
    s = template
    for k, v in values.items():
        s = s.replace("{" + k + "}", str(v))
    return s.strip().split()

def generate_summary_via_cmd(
    summarise_cmd_template: str,
    source_text: str,
    out_path: Path,
    model_class: str,
    max_tokens: int,
    orig_id: str,
    tmp_dir: Path,
) -> None:
    ensure_dir(tmp_dir)
    ensure_dir(out_path.parent)

    src_path = tmp_dir / f"{orig_id}.source.txt"
    src_path.write_text(source_text, encoding="utf-8")

    cmd = render_cmd_template(
        summarise_cmd_template,
        {
            "input": str(src_path),
            "output": str(out_path),
            "model": model_class,
            "max_tokens": max_tokens,
            "orig_id": orig_id,
        }
    )
    print(f"[{now_utc_iso()}] Summarise: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def split_sentences(text: str) -> List[str]:
    # simple splitter; good enough for sampling
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 30]

def sample_claims(source: str, n: int = 12) -> List[str]:
    sents = split_sentences(source)
    if not sents:
        return []
    if len(sents) <= n:
        return sents
    idxs = [round(i) for i in list(_linspace(0, len(sents) - 1, n))]
    return [sents[i] for i in idxs]

def _linspace(a: int, b: int, n: int) -> Iterable[float]:
    if n <= 1:
        yield float(a)
        return
    step = (b - a) / (n - 1)
    for i in range(n):
        yield a + i * step

def faithfulness_proxy(source: str, summary: str) -> float:
    """
    Baseline proxy: compare sampled claims to summary via keyword overlap.
    """
    claims = sample_claims(source, n=12)
    if not claims:
        return 0.0

    summ_l = summary.lower()
    passed = 0
    for c in claims:
        # take a few longer words as "anchors"
        anchors = [w.lower() for w in re.findall(r"[A-Za-z]{5,}", c)]
        anchors = anchors[:10]
        if not anchors:
            continue
        hit = sum(1 for w in anchors if w in summ_l)
        # require at least 30% anchor hits
        if hit / max(1, len(anchors)) >= 0.30:
            passed += 1

    return passed / max(1, len(claims))

def extract_key_terms(text: str, n: int = 20) -> List[str]:
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    if not words:
        return []

    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    # prefer medium frequency terms (avoid ultra-common)
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in items[:n]]

def retrieval_proxy(source: str, summary: str) -> float:
    """
    Proxy for retrieval preservation: fraction of key terms retained.
    """
    terms = extract_key_terms(source, n=25)
    if not terms:
        return 0.0
    summ_l = summary.lower()
    kept = sum(1 for t in terms if t in summ_l)
    return kept / len(terms)


@dataclass
class GateThresholds:
    faithfulness_min: float = 0.85
    retrieval_min: float = 0.80

def pick_best_ratio(
    scored: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    scored: list of candidate dicts containing ratio + pass boolean.
    Returns the first passing candidate (smallest ratio).
    """
    for row in sorted(scored, key=lambda r: float(r["ratio"])):
        if row.get("final_pass") is True:
            return row
    return None

def get_orig_id(obj: Dict[str, Any]) -> str:
    for k in ("_orig_id", "orig_id", "id"):
        if obj.get(k):
            return str(obj.get(k))
    # fallback: stable-ish hash
    return str(abs(hash(json.dumps(obj, sort_keys=True))) % 10**10)

def get_lang(obj: Dict[str, Any]) -> str:
    for k in ("lang_meta_detected", "lang", "language"):
        if obj.get(k):
            return str(obj.get(k))
    return "unknown"

def midpoint(a: float, b: float) -> float:
    return (a + b) / 2.0

def run_calibration(
    diagnostics_json: Path,
    out_dir: Path,
    text_field: str,
    summarise_cmd_template: str,
    thresholds: GateThresholds,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
    - candidates_rows: per (KO, ratio)
    - results_rows: per KO
    """
    data = read_json(diagnostics_json)
    items = data if isinstance(data, list) else [data]

    summaries_dir = out_dir / "summaries"
    tmp_dir = out_dir / "tmp"
    ensure_dir(summaries_dir)
    ensure_dir(tmp_dir)

    candidates_rows: List[Dict[str, Any]] = []
    results_rows: List[Dict[str, Any]] = []

    for idx, obj in enumerate(items, start=1):
        orig_id = get_orig_id(obj)
        lang = get_lang(obj)

        metrics = obj.get(f"{text_field}_metrics") or {}
        tok_count = int(metrics.get("token_count") or 0)

        pred_min = metrics.get("suggested_summary_ratio_min")
        pred_max = metrics.get("suggested_summary_ratio_max")
        cds = metrics.get("compression_difficulty_score_0_1")
        model_class = metrics.get("suggested_model_class") or "small"

        clean_text = obj.get(f"{text_field}_clean") or obj.get(text_field) or ""
        if not isinstance(clean_text, str):
            clean_text = str(clean_text)

        # If we have no tokens, skip safely
        if tok_count <= 0 or not clean_text.strip():
            results_rows.append({
                "orig_id": orig_id,
                "lang": lang,
                "token_count": tok_count,
                "cds": cds,
                "predicted_ratio_min": pred_min,
                "predicted_ratio_max": pred_max,
                "predicted_ratio_mid": None,
                "best_ratio": None,
                "best_ratio_source": "no_text",
                "band_match": None,
                "notes": "skipped (empty text)",
            })
            continue

        per_ko_candidates: List[Dict[str, Any]] = []

        for r in RATIOS:
            # Token budget for summary based on original token count
            max_tokens = int(math.ceil(tok_count * r))
            out_path = summaries_dir / orig_id / f"summary_r{r:.2f}.txt"

            # Generate summary if not present (idempotent)
            if not out_path.exists():
                generate_summary_via_cmd(
                    summarise_cmd_template=summarise_cmd_template,
                    source_text=clean_text,
                    out_path=out_path,
                    model_class=model_class,
                    max_tokens=max_tokens,
                    orig_id=orig_id,
                    tmp_dir=tmp_dir,
                )

            summary = out_path.read_text(encoding="utf-8", errors="ignore")

            faith = faithfulness_proxy(clean_text, summary)
            retr = retrieval_proxy(clean_text, summary)

            final_pass = (faith >= thresholds.faithfulness_min) and (retr >= thresholds.retrieval_min)

            row = {
                "orig_id": orig_id,
                "lang": lang,
                "token_count": tok_count,
                "cds": cds,
                "predicted_ratio_min": pred_min,
                "predicted_ratio_max": pred_max,
                "predicted_model_class": model_class,
                "ratio": r,
                "summary_tokens_target": max_tokens,
                "faithfulness_score": round(faith, 4),
                "retrieval_proxy": round(retr, 4),
                "final_pass": final_pass,
                "summary_path": str(out_path.relative_to(out_dir)),
            }
            candidates_rows.append(row)
            per_ko_candidates.append(row)

        best = pick_best_ratio(per_ko_candidates)

        pred_mid = None
        band_match = None
        if isinstance(pred_min, (int, float)) and isinstance(pred_max, (int, float)):
            pred_mid = midpoint(float(pred_min), float(pred_max))
            if best is not None:
                band_match = (float(pred_min) <= float(best["ratio"]) <= float(pred_max))

        if best is None:
            results_rows.append({
                "orig_id": orig_id,
                "lang": lang,
                "token_count": tok_count,
                "cds": cds,
                "predicted_ratio_min": pred_min,
                "predicted_ratio_max": pred_max,
                "predicted_ratio_mid": pred_mid,
                "best_ratio": None,
                "best_ratio_source": "none_passed",
                "band_match": band_match,
                "notes": "no ratio passed gates (consider map-reduce / multi-stage)",
            })
        else:
            results_rows.append({
                "orig_id": orig_id,
                "lang": lang,
                "token_count": tok_count,
                "cds": cds,
                "predicted_ratio_min": pred_min,
                "predicted_ratio_max": pred_max,
                "predicted_ratio_mid": pred_mid,
                "best_ratio": best["ratio"],
                "best_ratio_source": "faithfulness+retrieval",
                "band_match": band_match,
                "notes": "",
            })

        if idx == 1 or idx % 25 == 0 or idx == len(items):
            print(f"[{now_utc_iso()}] Calibration progress: {idx}/{len(items)}")

    return candidates_rows, results_rows


def band_from_ratio(r: Optional[float]) -> Optional[str]:
    if r is None:
        return None
    if r <= 0.20:
        return "low"
    if r <= 0.30:
        return "medium"
    return "high"

def fit_band_model_from_results(
    diagnostics_json: Path,
    results_rows: List[Dict[str, Any]],
    text_field: str,
    out_dir: Path,
) -> Optional[Path]:
    """
    Fits a simple classifier: metrics -> best_band (low/medium/high).
    Saves model pickle + a TSV report of coefficients / simple accuracy.
    Requires scikit-learn. If not installed, we skip gracefully.
    """
    try:
        import numpy as np  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.metrics import classification_report  # type: ignore
        import joblib  # type: ignore
    except Exception:
        print(f"[{now_utc_iso()}] scikit-learn not installed; skipping model fit.")
        return None

    data = read_json(diagnostics_json)
    items = data if isinstance(data, list) else [data]

    best_by_id = {r["orig_id"]: r.get("best_ratio") for r in results_rows}
    X_rows: List[List[float]] = []
    y: List[str] = []
    used_ids: List[str] = []

    FEATURE_KEYS = [
        "token_count",
        "lexical_density",
        "unigram_entropy",
        "entity_density_per_100_tokens",
        "avg_sentence_length_words",
        "bigram_repetition_ratio",
        "mean_adjacent_cosine_similarity",
        "mean_centroid_cosine_similarity",
        "noise_ratio_non_alnum",
    ]

    def safe_float(v: Any) -> float:
        if v is None:
            return float("nan")
        try:
            return float(v)
        except Exception:
            return float("nan")

    for obj in items:
        oid = get_orig_id(obj)
        br = best_by_id.get(oid)
        if br is None:
            continue
        band = band_from_ratio(float(br))
        if band is None:
            continue

        m = obj.get(f"{text_field}_metrics") or {}
        row = [safe_float(m.get(k)) for k in FEATURE_KEYS]

        # skip if too many NaNs
        if sum(1 for v in row if math.isnan(v)) >= 4:
            continue

        # simple NaN impute to feature median later
        X_rows.append(row)
        y.append(band)
        used_ids.append(oid)

    if len(X_rows) < 50:
        print(f"[{now_utc_iso()}] Not enough labelled samples to fit model (need ~50+). Skipping.")
        return None

    X = np.array(X_rows, dtype=float)

    # Median imputation
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=200, multi_class="auto")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)

    model_path = out_dir / "calibration_model.pkl"
    joblib.dump(
        {
            "model": clf,
            "feature_keys": FEATURE_KEYS,
            "col_medians": col_medians.tolist(),
        },
        model_path,
    )

    # Write a slide-friendly report TSV
    report_rows = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            report_rows.append({
                "label": label,
                "precision": round(metrics.get("precision", 0.0), 4),
                "recall": round(metrics.get("recall", 0.0), 4),
                "f1": round(metrics.get("f1-score", 0.0), 4),
                "support": int(metrics.get("support", 0) or 0),
            })

    write_tsv(out_dir / "calibration_model_report.tsv", report_rows,
              ["label", "precision", "recall", "f1", "support"])

    # Coefficients TSV (interpretable)
    coef_rows = []
    for cls_idx, cls_label in enumerate(clf.classes_):
        for j, fk in enumerate(FEATURE_KEYS):
            coef_rows.append({
                "class": cls_label,
                "feature": fk,
                "coef": round(float(clf.coef_[cls_idx][j]), 6),
            })
    write_tsv(out_dir / "calibration_model_coefficients.tsv", coef_rows, ["class", "feature", "coef"])

    print(f"[{now_utc_iso()}] Model saved: {model_path}")
    return model_path


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate KO compression policy (predicted vs empirical best ratio).")
    p.add_argument("--input", required=True, help="Input KO JSON (list or object)")
    p.add_argument("--output", default=None, help=("Output directory root. If omitted, we derive it from --input "
                                                   "by appending '_calibrated' next to the input file."))
    p.add_argument("--text-field", default="ko_content_flat", help="Text field name")
    p.add_argument("--lang-field", default="lang_meta_detected", help="Optional language field for sampling")
    p.add_argument("--sample-size", type=int, default=300, help="How many KOs to sample")
    p.add_argument("--seed", type=int, default=42, help="Sampling seed")
    p.add_argument("--min-tokens", type=int, default=50, help="Pass-through to diagnostics")
    p.add_argument("--diagnostics-script", default="assess_ko_quality/ko_compression_diagnostics.py",
                  help="Path to ko_compression_diagnostics.py")
    p.add_argument("--disable-embeddings", action="store_true", help="Pass-through to diagnostics")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Pass-through to diagnostics")
    p.add_argument("--summarise-cmd", required=True,
                  help=("Command template to generate summary. "
                        "Use placeholders: {input} {output} {model} {max_tokens} {orig_id}. "
                        "Example: \"python main.py summarise --input {input} --output {output} "
                        "--model {model} --max-tokens {max_tokens}\""))

    # Gates
    p.add_argument("--faithfulness-min", type=float, default=0.85, help="Faithfulness gate threshold (0-1)")
    p.add_argument("--retrieval-min", type=float, default=0.80, help="Retrieval proxy threshold (0-1)")

    # Optional model fit
    p.add_argument("--fit-model", action="store_true", help="Fit a band classifier from diagnostics metrics")

    args = p.parse_args()

    input_json = Path(args.input).resolve()

    if args.output:
        out_root = Path(args.output).resolve()

        # If user accidentally passes a file path (e.g., ".../temp_calibrated.json"), treat it as a directory stem
        if out_root.suffix:
            out_root = out_root.with_suffix("")
    else:
        out_root = input_json.with_suffix("")
        out_root = out_root.with_name(out_root.name + "_calibrated")

    ensure_dir(out_root)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / "calibration" / run_id
    ensure_dir(out_dir)

    print(f"[{now_utc_iso()}] START calibration run_id={run_id}")
    print(f"[{now_utc_iso()}] Output: {out_dir}")

    # 1) Load and sample
    items = iter_ko_objects(input_json)
    sample = make_sample(items, n=args.sample_size, seed=args.seed, text_field=args.text_field, lang_field=args.lang_field)

    # Save sample as JSONL (one object per line)
    sample_path = out_dir / "sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as f:
        for obj in sample:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Also save as JSON array for diagnostics input
    sample_json_path = out_dir / "sample.json"
    write_json(sample_json_path, sample)

    # 2) Run diagnostics
    diagnostics_out = out_dir / "diagnostics.json"
    run_compression_diagnostics(
        script_path=Path(args.diagnostics_script),
        input_json=sample_json_path,
        output_json=diagnostics_out,
        text_field=args.text_field,
        min_tokens=args.min_tokens,
        disable_embeddings=args.disable_embeddings,
        device=args.device,
    )

    # 3) Generate candidate summaries + score + pick best
    thresholds = GateThresholds(faithfulness_min=args.faithfulness_min, retrieval_min=args.retrieval_min)
    candidates_rows, results_rows = run_calibration(
        diagnostics_json=diagnostics_out,
        out_dir=out_dir,
        text_field=args.text_field,
        summarise_cmd_template=args.summarise_cmd,
        thresholds=thresholds,
    )

    # 4) Write TSVs
    candidates_tsv = out_dir / "calibration_candidates.tsv"
    results_tsv = out_dir / "calibration_results.tsv"

    write_tsv(
        candidates_tsv,
        candidates_rows,
        [
            "orig_id", "lang", "token_count", "cds",
            "predicted_ratio_min", "predicted_ratio_max", "predicted_model_class",
            "ratio", "summary_tokens_target",
            "faithfulness_score", "retrieval_proxy", "final_pass",
            "summary_path",
        ],
    )

    write_tsv(
        results_tsv,
        results_rows,
        [
            "orig_id", "lang", "token_count", "cds",
            "predicted_ratio_min", "predicted_ratio_max", "predicted_ratio_mid",
            "best_ratio", "best_ratio_source",
            "band_match", "notes",
        ],
    )

    print(f"[{now_utc_iso()}] Wrote: {candidates_tsv}")
    print(f"[{now_utc_iso()}] Wrote: {results_tsv}")

    # 5) Optional: fit a better model (band classifier)
    if args.fit_model:
        fit_band_model_from_results(
            diagnostics_json=diagnostics_out,
            results_rows=results_rows,
            text_field=args.text_field,
            out_dir=out_dir,
        )

    print(f"[{now_utc_iso()}] DONE calibration run_id={run_id}")


if __name__ == "__main__":
    main()
