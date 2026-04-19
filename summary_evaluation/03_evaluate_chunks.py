# which_model_to_choose/methods/03_evaluate_chunks.py

import argparse
import json
from typing import Dict, Any, List, Tuple
from pathlib import Path
from glob import glob
import math

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ===== CONFIG =====
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "03_evaluate_chunks.json"
ROOT_INPUT_DIR = REPO_ROOT / "input"

MODEL_NAME = "roberta-large-mnli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# max tokens the model can take; MNLI models are ~512
MAX_TOKENS = 512
PAIR_CHUNK_TOKENS = 192
PAIR_CHUNK_OVERLAP = 32
LIMITATION_NOTE = (
    "Chunked NLI is a long-document approximation. Scores reflect aggregated chunk-level support, "
    "not full-document entailment ground truth."
)


def latest_input_file(folder: Path = ROOT_INPUT_DIR) -> Path:
    candidates = [Path(p) for p in glob(str(folder / "*")) if Path(p).is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files found in {folder}/")
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return candidates[-1]


def load_json(path: Path) -> List[Dict[str, Any]]:
    """Load JSON array/object, wrapped {docs:[...]}, or JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("docs"), list):
            return data["docs"]
        if isinstance(data, dict):
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


def truncate_pair(tokenizer, premise: str, hypothesis: str, max_tokens: int):
    """
    MNLI models have a 512 token limit.
    We let the tokenizer do truncation in 'pair' mode.
    """
    return tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )


def token_chunks(tokenizer, text: str, chunk_tokens: int = PAIR_CHUNK_TOKENS, overlap: int = PAIR_CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    step = max(1, chunk_tokens - overlap)
    chunks = []
    for start in range(0, len(token_ids), step):
        piece = token_ids[start : start + chunk_tokens]
        if not piece:
            continue
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True).strip())
        if start + chunk_tokens >= len(token_ids):
            break
    return [c for c in chunks if c]


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def nli_score(
    tokenizer,
    model,
    premise: str,
    hypothesis: str,
) -> Dict[str, float]:
    """
    Run MNLI and return the 3 class probabilities:
    - contradiction
    - neutral
    - entailment

    We’ll return them as a dict.
    """
    inputs = truncate_pair(tokenizer, premise, hypothesis, MAX_TOKENS)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)
    # MNLI head is [contradiction, neutral, entailment]
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return {
        "contradiction": probs[0],
        "neutral": probs[1],
        "entailment": probs[2],
    }


def chunked_nli_score(tokenizer, model, premise: str, hypothesis: str) -> Dict[str, float]:
    premise_chunks = token_chunks(tokenizer, premise)
    hypothesis_chunks = token_chunks(tokenizer, hypothesis)

    if not premise_chunks or not hypothesis_chunks:
        return {"contradiction": 0.0, "neutral": 1.0, "entailment": 0.0}

    selected_scores = []
    for h_chunk in hypothesis_chunks:
        chunk_scores = [nli_score(tokenizer, model, p_chunk, h_chunk) for p_chunk in premise_chunks]
        best = max(
            chunk_scores,
            key=lambda s: (s["entailment"] - s["contradiction"], s["entailment"]),
        )
        selected_scores.append(best)

    denom = float(len(selected_scores))
    return {
        "contradiction": sum(s["contradiction"] for s in selected_scores) / denom,
        "neutral": sum(s["neutral"] for s in selected_scores) / denom,
        "entailment": sum(s["entailment"] for s in selected_scores) / denom,
    }


def find_candidate_chunks(obj: Dict[str, Any]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    From one JSON object:
    - get the main large chunk under key 'ko_content_flat'
    - get all other keys that start with 'ko_content_flat' but are not exactly that
    Return:
        large_chunk_text, list of (key, text) for candidates
    """
    if "ko_content_flat" not in obj:
        return "", []

    large = obj["ko_content_flat"] or ""
    candidates = []
    for k, v in obj.items():
        if k.startswith("ko_content_flat") and k != "ko_content_flat":
            if isinstance(v, str) and v.strip():
                candidates.append((k, v))
    return large, candidates


def parse_args():
    p = argparse.ArgumentParser(description="Bidirectional NLI evaluation of ko_content_flat candidates.")
    p.add_argument("--input", type=Path, help="Input JSON path. Defaults to newest file under input/")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = args.input or latest_input_file()
    data = load_json(input_path)
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

    results = []

    print("Method note:", LIMITATION_NOTE)
    print(f"NLI chunking: chunk_tokens={PAIR_CHUNK_TOKENS}, overlap={PAIR_CHUNK_OVERLAP}")

    # go over each record in JSON
    for idx, record in enumerate(data):
        large, candidates = find_candidate_chunks(record)
        if not large or not candidates:
            continue  # nothing to compare here

        for cand_key, cand_text in candidates:
            # L -> S: whether the source text supports the candidate summary.
            l_to_s = chunked_nli_score(tokenizer, model, premise=large, hypothesis=cand_text)

            # S -> L is not literal "faithfulness"; in practice it behaves more like
            # a rough compression / bidirectional-overlap signal.
            s_to_l = chunked_nli_score(tokenizer, model, premise=cand_text, hypothesis=large)

            results.append(
                {
                    "record_index": idx,
                    "candidate_key": cand_key,
                    "evaluation_method": "chunked_bidirectional_nli",
                    "score_scope": "aggregated_chunk_level",
                    "limitation_note": LIMITATION_NOTE,
                    "len_large": len(large),
                    "len_small": len(cand_text),
                    "large_chunk_count": len(token_chunks(tokenizer, large)),
                    "small_chunk_count": len(token_chunks(tokenizer, cand_text)),
                    # L -> S
                    "L_to_S_entail": l_to_s["entailment"],
                    "L_to_S_neutral": l_to_s["neutral"],
                    "L_to_S_contradiction": l_to_s["contradiction"],
                    # S -> L
                    "S_to_L_entail": s_to_l["entailment"],
                    "S_to_L_neutral": s_to_l["neutral"],
                    "S_to_L_contradiction": s_to_l["contradiction"],
                }
            )

    if not results:
        print("No comparable chunks found.")
        return

    df = pd.DataFrame(results)

    # heuristic columns (optional, helps ranking)
    # high L→S_entail AND low S→L_entail is good for “compressed but faithful”
    df["entail_gap_LS_minus_SL"] = df["L_to_S_entail"] - df["S_to_L_entail"]

    print(df.to_csv(index=False))
    print(
        "Summary: scores above are chunk-aggregated proxies. "
        "Use them for comparative ranking of candidates, not as literal document-level factuality labels."
    )

    # ----- SAVE RESULTS -----
    records = df.to_dict(orient="records")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
