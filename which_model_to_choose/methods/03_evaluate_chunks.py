# which_model_to_choose/methods/03_evaluate_chunks.py

import json
from typing import Dict, Any, List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ===== CONFIG =====
JSON_PATH = "../input/final_output_14_10-2025_17-37-04_for_qa_llmed_runpod.json"
MODEL_NAME = "roberta-large-mnli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# max tokens the model can take; MNLI models are ~512
MAX_TOKENS = 512


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON that is either a list of objects or a single object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalise to list
    if isinstance(data, dict):
        return [data]
    return data


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


def main():
    data = load_json(JSON_PATH)
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)

    results = []

    # go over each record in JSON
    for idx, record in enumerate(data):
        large, candidates = find_candidate_chunks(record)
        if not large or not candidates:
            continue  # nothing to compare here

        for cand_key, cand_text in candidates:
            # L -> S (does L entail S?)  → "coverage"
            # if this is high, it means the small one is supported by large one
            l_to_s = nli_score(tokenizer, model, premise=large, hypothesis=cand_text)

            # S -> L (does S entail L?) → "faithfulness / no hallucination"
            # if this is high, the small one may be overclaiming
            s_to_l = nli_score(tokenizer, model, premise=cand_text, hypothesis=large)

            results.append(
                {
                    "record_index": idx,
                    "candidate_key": cand_key,
                    "len_large": len(large),
                    "len_small": len(cand_text),
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


if __name__ == "__main__":
    main()
