from __future__ import annotations

import json

from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
                          AutoModelForSeq2SeqLM,)


# ================== CONFIG ==================

INPUT_JSON_PATH = Path("../input/final_output_14_10-2025_17-37-04_for_qa.json")
OUTPUT_JSONL_PATH = Path("04_evaluate_chunks.jsonl")
OUTPUT_CSV_PATH = Path("04_evaluate_chunks.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NLI model for factual alignment (AlignScore-like)
NLI_MODEL_NAME = "microsoft/deberta-large-mnli"

# Question generation model (QGen)
QG_MODEL_NAME = "iarfmoose/t5-base-question-generator"

# QA model (answer extraction)
QA_MODEL_NAME = "deepset/deberta-v3-base-squad2"

# Number of questions to generate from L per KO
NUM_QA_QUESTIONS = 10

# ============================================


def load_json_objects(path: Path) -> List[Dict[str, Any]]:
    """
    Load JSON objects from:
    - a single JSON object
    - a JSON list of objects
    - JSONL (one object per line)
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    objs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            objs.append(json.loads(line))
    return objs


def extract_L_and_candidates(obj: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    """
    Get L (ko_content_flat) and all S candidates (other keys starting with ko_content_flat*).
    """
    if "ko_content_flat" not in obj:
        raise KeyError("Object missing 'ko_content_flat'")

    L = obj["ko_content_flat"]
    candidates: Dict[str, str] = {}
    for key, value in obj.items():
        if not isinstance(value, str):
            continue
        if key == "ko_content_flat":
            continue
        if key.startswith("ko_content_flat"):
            candidates[key] = value
    return L, candidates


# ========== NLI-based factual consistency (AlignScore-like) ==========

class NLIScorer:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

        # For MNLI models: label mapping
        # 0: contradiction, 1: neutral, 2: entailment (most HF MNLI heads follow this)
        self.entail_idx = 2
        self.contra_idx = 0

    @torch.no_grad()
    def score_pair(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Return entailment and contradiction probabilities P(entailment), P(contradiction)
        for (premise -> hypothesis).
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits[0], dim=-1).cpu().tolist()

        return {
            "p_entail": float(probs[self.entail_idx]),
            "p_contra": float(probs[self.contra_idx]),
        }

    def bidirectional_scores(self, L: str, S: str) -> Dict[str, float]:
        """
        Compute NLI in both directions:
        - L -> S : does L support S? (hallucination check)
        - S -> L : does S cover L? (coverage / compression)
        """
        forward = self.score_pair(L, S)
        backward = self.score_pair(S, L)

        return {
            "nli_L_to_S_entail": forward["p_entail"],
            "nli_L_to_S_contra": forward["p_contra"],
            "nli_S_to_L_entail": backward["p_entail"],
            "nli_S_to_L_contra": backward["p_contra"],
        }


# ========== QG + QA-based factuality (QAFactEval-style) ==========

class QAEvaluator:
    def __init__(
        self,
        qg_model_name: str,
        qa_model_name: str,
        device: str = "cpu",
    ) -> None:
        self.device = device

        # Question generation model (T5)
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained(qg_model_name).to(device)
        self.qg_model.eval()

        # QA model
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name).to(device)
        self.qa_model.eval()

    @torch.no_grad()
    def generate_questions(self, context: str, num_questions: int = 10) -> List[str]:
        """
        Generate questions from L, similar to FEQA/QAFactEval's QG step.
        """
        # The T5 QG model expects a prefix like "generate questions:" or similar.
        # This repo uses "generate questions:" as a prompt.
        prompt = f"generate questions: {context}"
        inputs = self.qg_tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generate a bit more than needed, then deduplicate
        num_return = min(num_questions * 2, 20)
        outputs = self.qg_model.generate(
            **inputs,
            max_new_tokens=64,
            num_beams=4,
            num_return_sequences=num_return,
            no_repeat_ngram_size=3,
        )

        questions: List[str] = []
        for out in outputs:
            q = self.qg_tokenizer.decode(out, skip_special_tokens=True).strip()
            if q and q not in questions:
                questions.append(q)
            if len(questions) >= num_questions:
                break
        return questions

    @torch.no_grad()
    def answer_question(self, context: str, question: str) -> str:
        """
        Answer a single question on a given context using the QA model.
        """
        inputs = self.qa_tokenizer(
            question,
            context,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.qa_model(**inputs)
        start_scores = outputs.start_logits[0]
        end_scores = outputs.end_logits[0]

        start_idx = int(torch.argmax(start_scores))
        end_idx = int(torch.argmax(end_scores))

        if end_idx < start_idx:
            return ""

        tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
        answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return answer

    def qa_factuality(
        self,
        L: str,
        S: str,
        num_questions: int = NUM_QA_QUESTIONS,
    ) -> Dict[str, float]:
        """
        QAFactEval-style factual check:

        1. Generate questions from L.
        2. Answer them on L -> reference answers.
        3. Answer them on S -> candidate answers.
        4. Compute simple token-level F1 per question, then average.
        """
        questions = self.generate_questions(L, num_questions=num_questions)
        if not questions:
            return {
                "qa_mean_f1": 0.0,
                "qa_num_questions": 0,
            }

        ref_answers: List[str] = []
        cand_answers: List[str] = []

        for q in questions:
            ref = self.answer_question(L, q)
            cand = self.answer_question(S, q)
            ref_answers.append(ref)
            cand_answers.append(cand)

        f1_scores: List[float] = []
        for ref, cand in zip(ref_answers, cand_answers):
            f1_scores.append(token_f1(ref, cand))

        if not f1_scores:
            return {
                "qa_mean_f1": 0.0,
                "qa_num_questions": len(questions),
            }

        mean_f1 = sum(f1_scores) / len(f1_scores)
        return {
            "qa_mean_f1": float(mean_f1),
            "qa_num_questions": len(questions),
        }


def token_f1(ref: str, cand: str) -> float:
    """
    Simple token-level F1 between two answers (case-insensitive, whitespace tokenisation).
    """
    ref_tokens = [t for t in ref.lower().split() if t]
    cand_tokens = [t for t in cand.lower().split() if t]

    if not ref_tokens and not cand_tokens:
        return 1.0
    if not ref_tokens or not cand_tokens:
        return 0.0

    ref_set = set(ref_tokens)
    cand_set = set(cand_tokens)

    common = ref_set & cand_set
    if not common:
        return 0.0

    precision = len(common) / len(cand_set)
    recall = len(common) / len(ref_set)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ===================== MAIN LOOP =====================

def main() -> None:
    print(f"Loading KOs from {INPUT_JSON_PATH} ...")
    objs = load_json_objects(INPUT_JSON_PATH)
    print(f"Loaded {len(objs)} objects.")

    if not objs:
        print("No objects found. Exiting.")
        return

    print("Initialising NLI scorer...")
    nli = NLIScorer(NLI_MODEL_NAME, DEVICE)

    print("Initialising QA evaluator (QG + QA)...")
    qa_eval = QAEvaluator(QG_MODEL_NAME, QA_MODEL_NAME, DEVICE)

    rows: List[Dict[str, Any]] = []

    for obj in tqdm(objs, desc="Evaluating objects"):
        ko_id = obj.get("@id") or obj.get("_id") or None

        try:
            L, candidates = extract_L_and_candidates(obj)
        except KeyError:
            continue

        if not candidates:
            continue

        len_L = len(L.split())

        for key, S in candidates.items():
            len_S = len(S.split())
            compression = len_S / len_L if len_L > 0 else None

            # NLI-based scores
            nli_scores = nli.bidirectional_scores(L, S)

            # QA-based scores
            qa_scores = qa_eval.qa_factuality(L, S)

            row: Dict[str, Any] = {
                "@id": ko_id,
                "candidate_key": key,
                "len_L": len_L,
                "len_S": len_S,
                "compression_ratio_tokens": compression,
            }
            row.update(nli_scores)
            row.update(qa_scores)

            rows.append(row)

    print(f"Writing JSONL results to {OUTPUT_JSONL_PATH} ...")
    with OUTPUT_JSONL_PATH.open("w", encoding="utf-8") as f_out:
        for r in rows:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Writing CSV results to {OUTPUT_CSV_PATH} ...")
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
