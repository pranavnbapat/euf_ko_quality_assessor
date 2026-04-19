from __future__ import annotations

import json
import argparse

from pathlib import Path
from glob import glob
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from collections import Counter

import pandas as pd
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering,
                          AutoModelForSeq2SeqLM,)


# ================== CONFIG ==================

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "output" / "04_evaluate_chunks.jsonl"
DEFAULT_OUTPUT_CSV = SCRIPT_DIR / "output" / "04_evaluate_chunks.csv"
ROOT_INPUT_DIR = REPO_ROOT / "input"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# NLI model for factual alignment (AlignScore-like)
NLI_MODEL_NAME = "microsoft/deberta-large-mnli"

# Question generation model (QGen)
QG_MODEL_NAME = "iarfmoose/t5-base-question-generator"

# QA model (answer extraction)
QA_MODEL_NAME = "deepset/deberta-v3-base-squad2"

# Number of questions to generate from L per KO
NUM_QA_QUESTIONS = 10
PAIR_CHUNK_TOKENS = 192
PAIR_CHUNK_OVERLAP = 32
QG_CHUNK_TOKENS = 256
QA_CONTEXT_CHUNK_TOKENS = 320
LIMITATION_NOTE = (
    "NLI, question generation, and QA are all chunk-aggregated approximations for long documents. "
    "These scores are stronger than single-truncation baselines but still not document-level ground truth."
)

# ============================================


def latest_input_file(folder: Path = ROOT_INPUT_DIR) -> Path:
    candidates = [Path(p) for p in glob(str(folder / "*")) if Path(p).is_file()]
    if not candidates:
        raise FileNotFoundError(f"No files found in {folder}/")
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return candidates[-1]


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
            docs = data.get("docs")
            if isinstance(docs, list):
                return docs
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


def token_chunks(tokenizer, text: str, chunk_tokens: int, overlap: int = PAIR_CHUNK_OVERLAP) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []
    step = max(1, chunk_tokens - overlap)
    chunks: List[str] = []
    for start in range(0, len(token_ids), step):
        piece = token_ids[start : start + chunk_tokens]
        if not piece:
            continue
        chunks.append(tokenizer.decode(piece, skip_special_tokens=True).strip())
        if start + chunk_tokens >= len(token_ids):
            break
    return [c for c in chunks if c]


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

    def chunked_score_pair(self, premise: str, hypothesis: str) -> Dict[str, float]:
        premise_chunks = token_chunks(self.tokenizer, premise, PAIR_CHUNK_TOKENS)
        hypothesis_chunks = token_chunks(self.tokenizer, hypothesis, PAIR_CHUNK_TOKENS)

        if not premise_chunks or not hypothesis_chunks:
            return {
                "p_entail": 0.0,
                "p_contra": 0.0,
                "premise_chunk_count": len(premise_chunks),
                "hypothesis_chunk_count": len(hypothesis_chunks),
            }

        selected = []
        for h_chunk in hypothesis_chunks:
            scores = [self.score_pair(p_chunk, h_chunk) for p_chunk in premise_chunks]
            best = max(scores, key=lambda s: (s["p_entail"] - s["p_contra"], s["p_entail"]))
            selected.append(best)

        denom = float(len(selected))
        return {
            "p_entail": sum(s["p_entail"] for s in selected) / denom,
            "p_contra": sum(s["p_contra"] for s in selected) / denom,
            "premise_chunk_count": len(premise_chunks),
            "hypothesis_chunk_count": len(hypothesis_chunks),
        }

    def bidirectional_scores(self, L: str, S: str) -> Dict[str, float]:
        """
        Compute NLI in both directions:
        - L -> S : does L support S? (hallucination check)
        - S -> L : does S cover L? (coverage / compression)
        """
        forward = self.chunked_score_pair(L, S)
        backward = self.chunked_score_pair(S, L)

        return {
            "nli_L_to_S_entail": forward["p_entail"],
            "nli_L_to_S_contra": forward["p_contra"],
            "nli_S_to_L_entail": backward["p_entail"],
            "nli_S_to_L_contra": backward["p_contra"],
            "nli_L_chunk_count": forward["premise_chunk_count"],
            "nli_S_chunk_count": backward["premise_chunk_count"],
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
        questions: List[str] = []
        context_chunks = token_chunks(self.qg_tokenizer, context, QG_CHUNK_TOKENS, overlap=48)
        for chunk in context_chunks[: max(1, min(len(context_chunks), 6))]:
            prompt = f"generate questions: {chunk}"
            inputs = self.qg_tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            num_return = min(max(2, num_questions), 8)
            outputs = self.qg_model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                num_return_sequences=num_return,
                no_repeat_ngram_size=3,
            )

            for out in outputs:
                q = self.qg_tokenizer.decode(out, skip_special_tokens=True).strip()
                if q and q not in questions:
                    questions.append(q)
                if len(questions) >= num_questions:
                    return questions
        return questions

    @torch.no_grad()
    def answer_question(self, context: str, question: str) -> str:
        """
        Answer a single question on a given context using the QA model.
        """
        best_answer = ""
        best_score = float("-inf")
        context_chunks = token_chunks(self.qa_tokenizer, context, QA_CONTEXT_CHUNK_TOKENS, overlap=64)
        if not context_chunks:
            return ""

        for chunk in context_chunks:
            inputs = self.qa_tokenizer(
                question,
                chunk,
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
                continue

            score = float(start_scores[start_idx] + end_scores[end_idx])
            tokens = inputs["input_ids"][0][start_idx : end_idx + 1]
            answer = self.qa_tokenizer.decode(tokens, skip_special_tokens=True).strip()
            if answer and score > best_score:
                best_score = score
                best_answer = answer

        return best_answer

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

    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    common = sum(min(ref_counts[t], cand_counts[t]) for t in ref_counts.keys() & cand_counts.keys())
    if common == 0:
        return 0.0

    precision = common / len(cand_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


# ===================== MAIN LOOP =====================

def parse_args():
    p = argparse.ArgumentParser(description="Experimental NLI + QG/QA evaluation of ko_content_flat candidates.")
    p.add_argument("--input", type=Path, help="Input JSON path. Defaults to newest file under input/")
    p.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL, help="Output JSONL path.")
    p.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input or latest_input_file()
    print(f"Loading KOs from {input_path} ...")
    objs = load_json_objects(input_path)
    print(f"Loaded {len(objs)} objects.")

    if not objs:
        print("No objects found. Exiting.")
        return

    print("Initialising NLI scorer...")
    nli = NLIScorer(NLI_MODEL_NAME, DEVICE)

    print("Initialising QA evaluator (QG + QA)...")
    qa_eval = QAEvaluator(QG_MODEL_NAME, QA_MODEL_NAME, DEVICE)
    print("Method note:", LIMITATION_NOTE)
    print(
        "Chunking config:",
        f"nli_chunk_tokens={PAIR_CHUNK_TOKENS},",
        f"qg_chunk_tokens={QG_CHUNK_TOKENS},",
        f"qa_context_chunk_tokens={QA_CONTEXT_CHUNK_TOKENS}",
    )

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
                "evaluation_method": "chunked_nli_plus_chunked_qg_qa",
                "score_scope": "aggregated_chunk_level",
                "limitation_note": LIMITATION_NOTE,
                "len_L": len_L,
                "len_S": len_S,
                "compression_ratio_tokens": compression,
            }
            row.update(nli_scores)
            row.update(qa_scores)

            rows.append(row)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing JSONL results to {args.output_jsonl} ...")
    with args.output_jsonl.open("w", encoding="utf-8") as f_out:
        for r in rows:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Writing CSV results to {args.output_csv} ...")
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print(
        "Summary: output scores are chunk-aggregated proxies for long-document support and answer consistency. "
        "Use them comparatively across candidates, not as literal factuality truth."
    )
    print("Done.")


if __name__ == "__main__":
    main()
