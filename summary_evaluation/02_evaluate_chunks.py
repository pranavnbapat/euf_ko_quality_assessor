from pathlib import Path
import argparse
import json
from glob import glob
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import evaluate

try:
    from moverscore import get_idf_dict, word_mover_score
    HAS_MOVERSCORE = True
except ImportError:
    HAS_MOVERSCORE = False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "02_evaluate_chunks.json"
DEFAULT_PROGRESS_EVERY = 100
ROOT_INPUT_DIR = REPO_ROOT / "input"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def latest_input_file(folder: Path = ROOT_INPUT_DIR) -> Path:
    candidates = [Path(p) for p in glob(str(folder / "*")) if Path(p).is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files found in {folder}/")
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return candidates[-1]


def parse_args():
    p = argparse.ArgumentParser(description="Semantic similarity evaluation for ko_content_flat candidates.")
    p.add_argument("--input", type=Path, help="Input JSON path. Defaults to newest file under input/")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    p.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY, help="Log and checkpoint every N processed records.")
    return p.parse_args()


def load_records(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            docs = data.get("docs")
            if isinstance(docs, list):
                return docs
            return [data]
    except json.JSONDecodeError:
        pass

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_candidates(record):
    source = record.get("ko_content_flat")
    if not isinstance(source, str) or not source.strip():
        return "", []
    candidates = []
    for key, value in record.items():
        if key == "ko_content_flat":
            continue
        if key.startswith("ko_content_flat") and isinstance(value, str) and value.strip():
            candidates.append((key, value))
    return source, candidates


def build_embedder(model_name: str):
    return SentenceTransformer(model_name)


def cosine_sim_from_texts(model, text_a: str, text_b: str) -> float:
    emb_a = model.encode([text_a], convert_to_numpy=True)
    emb_b = model.encode([text_b], convert_to_numpy=True)
    return float(cosine_similarity(emb_a, emb_b)[0][0])


bertscore_metric = evaluate.load("bertscore")


def moverscore_pair(source: str, target: str) -> float:
    if not HAS_MOVERSCORE:
        return -1.0
    refs = [source]
    hyps = [target]
    idf_dict_hyp = get_idf_dict(hyps)
    idf_dict_ref = get_idf_dict(refs)
    scores = word_mover_score(refs, hyps, idf_dict_ref, idf_dict_hyp, stop_words=[])
    return float(scores[0])


def main():
    args = parse_args()
    input_path = args.input or latest_input_file()
    print(f"Using input file: {input_path}")
    records = load_records(input_path)
    print(f"Loaded {len(records)} records")
    embedder = build_embedder(EMBEDDING_MODEL_NAME)
    print(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(args.output.suffix + ".partial")
    rows = []
    processed = 0
    started = time.time()
    for record_index, record in enumerate(records):
        source, candidates = find_candidates(record)
        if not source or not candidates:
            continue

        candidate_names = [name for name, _ in candidates]
        candidate_texts = [text for _, text in candidates]
        references = [source] * len(candidate_texts)

        bs_result = bertscore_metric.compute(
            predictions=candidate_texts,
            references=references,
            model_type="roberta-large",
        )

        for idx, (name, text) in enumerate(candidates):
            rows.append(
                {
                    "record_index": record_index,
                    "candidate": name,
                    "cosine_sim": cosine_sim_from_texts(embedder, source, text),
                    "bertscore_p": bs_result["precision"][idx],
                    "bertscore_r": bs_result["recall"][idx],
                    "bertscore_f1": bs_result["f1"][idx],
                    "moverscore": moverscore_pair(source, text),
                    "len_L": len(source.split()),
                    "len_S": len(text.split()),
                }
            )

        processed += 1
        if args.progress_every > 0 and processed % args.progress_every == 0:
            elapsed = time.time() - started
            print(
                f"[progress] processed={processed} rows={len(rows)} "
                f"elapsed={elapsed:.1f}s last_record_index={record_index}"
            )
            checkpoint_rows = sorted(rows, key=lambda x: (x["record_index"], -x["bertscore_f1"]))
            with checkpoint_path.open("w", encoding="utf-8") as f:
                json.dump(checkpoint_rows, f, ensure_ascii=False, indent=2)
            print(f"[checkpoint] wrote {checkpoint_path}")

    rows.sort(key=lambda x: (x["record_index"], -x["bertscore_f1"]))
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Done. Wrote {args.output}")


if __name__ == "__main__":
    main()
