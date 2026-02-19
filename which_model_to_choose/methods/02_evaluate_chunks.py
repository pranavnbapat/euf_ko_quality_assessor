# which_model_to_choose/methods/02_evaluate_chunks.py

from pathlib import Path
import json
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# BERTScore from Hugging Face 'evaluate'
import evaluate

try:
    from moverscore import get_idf_dict, word_mover_score
    HAS_MOVERSCORE = True
except ImportError:
    HAS_MOVERSCORE = False


# === CONFIG ===
JSON_PATH = Path("../input/final_output_14_10-2025_17-37-04_for_qa_llmed_runpod.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_chunks(obj):
    """
    Keep only:
      - ko_content_flat
      - ko_content_flat_summarised
    Works for a dict JSON or a list of dicts.
    Returns a list of (label, text).
    """
    allowed = {"ko_content_flat", "ko_content_flat_summarised"}

    def keep(k, v):
        # Only keep allowed keys and only string values
        return (k in allowed) and isinstance(v, str)

    if isinstance(obj, dict):
        return [(k, v) for k, v in obj.items() if keep(k, v)]

    elif isinstance(obj, list):
        collected = []
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                for k, v in item.items():
                    if keep(k, v):
                        collected.append((f"item{i}.{k}", v))
        return collected

    else:
        raise ValueError("Unsupported JSON structure for chunks")


def build_embedder(model_name: str):
    # loads a sentence-transformers model
    return SentenceTransformer(model_name)


def cosine_sim_from_texts(model, text_a: str, text_b: str) -> float:
    emb_a = model.encode([text_a], convert_to_numpy=True)
    emb_b = model.encode([text_b], convert_to_numpy=True)
    sim = cosine_similarity(emb_a, emb_b)[0][0]
    return float(sim)


# load once
bertscore_metric = evaluate.load("bertscore")

def bertscore_pair(candidate: str, reference: str):
    """
    BERTScore expects: predictions=candidate(s), references=reference(s)
    Here: S is candidate (shorter), L is reference (original).
    """
    result = bertscore_metric.compute(
        predictions=[candidate],
        references=[reference],
        model_type="roberta-large",
        # model_type="microsoft/deberta-xlarge-mnli", # Crashes pycharm
    )
    # returns list of scores; take first
    return {
        "precision": result["precision"][0],
        "recall": result["recall"][0],
        "f1": result["f1"][0],
    }


def moverscore_pair(source: str, target: str) -> float:
    """
    source: original long text (L)
    target: candidate short text (S)
    returns a single score
    """
    if not HAS_MOVERSCORE:
        return -1.0

    refs = [source]
    hyps = [target]

    # IDF dicts help the score; we build trivial ones for both sides
    idf_dict_hyp = get_idf_dict(hyps)
    idf_dict_ref = get_idf_dict(refs)

    scores = word_mover_score(
        refs,
        hyps,
        idf_dict_ref,
        idf_dict_hyp,
        stop_words=[],
    )
    return float(scores[0])


def main():
    data = load_json(JSON_PATH)
    chunks = extract_chunks(data)

    # find the main large chunk
    main_candidates = [t for (k, t) in chunks if k == "ko_content_flat" or k.endswith(".ko_content_flat")]
    if not main_candidates:
        raise ValueError("Could not find main 'ko_content_flat' in JSON.")
    L = main_candidates[0]

    # collect candidates
    candidate_names = []
    candidate_texts = []
    for (k, txt) in chunks:
        if txt == L:
            continue
        candidate_names.append(k)
        candidate_texts.append(txt)

    # make a parallel reference list for BERTScore
    # (same L repeated as many times as we have candidates)
    references = [L] * len(candidate_texts)

    # === 1) run BERTScore ONCE for all candidates ===
    # but first, normalise empties so BERTScore doesn't scream
    safe_candidates = []
    safe_mask = []
    for s in candidate_texts:
        if not s or not s.strip():
            safe_candidates.append(" ")  # dummy
            safe_mask.append(False)
        else:
            safe_candidates.append(s)
            safe_mask.append(True)

    bs_result = bertscore_metric.compute(
        predictions=safe_candidates,
        references=references,
        model_type="roberta-large",
    )

    # === 2) now do cosine (cheap) and optional MoverScore per item ===
    embedder = build_embedder(EMBEDDING_MODEL_NAME)

    results = []
    for idx, (name, S) in enumerate(zip(candidate_names, candidate_texts), start=1):
        print(f"[{idx}/{len(candidate_names)}] scoring {name} ...")

        if not S or not S.strip():
            results.append(
                {
                    "candidate": name,
                    "cosine_sim": 0.0,
                    "bertscore_p": 0.0,
                    "bertscore_r": 0.0,
                    "bertscore_f1": 0.0,
                    "moverscore": -1.0,
                    "len_L": len(L.split()),
                    "len_S": 0,
                }
            )
            continue

        cos_sim = cosine_sim_from_texts(embedder, L, S)

        # pick corresponding BERTScore values
        bs_p = bs_result["precision"][idx - 1]
        bs_r = bs_result["recall"][idx - 1]
        bs_f1 = bs_result["f1"][idx - 1]

        ms = moverscore_pair(L, S)

        results.append(
            {
                "candidate": name,
                "cosine_sim": cos_sim,
                "bertscore_p": bs_p,
                "bertscore_r": bs_r,
                "bertscore_f1": bs_f1,
                "moverscore": ms,
                "len_L": len(L.split()),
                "len_S": len(S.split()),
            }
        )

    # sort & save
    results.sort(key=lambda x: x["cosine_sim"], reverse=True)

    output_path = Path("02_evaluate_chunks.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()

