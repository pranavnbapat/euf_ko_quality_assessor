# assess_ko_quality/semantic.py

import os

from typing import Any, Dict, List, Tuple, Optional

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel

from quality_text_utils import tokens, strip_stops, unique_ratio, jaccard, count_non_ascii, sentence_lengths


_NLP = None



def _get_nlp():
    """
    Lazy-load spaCy English model once.
    If 'en_core_web_lg' is not installed, download it on the fly.
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        _NLP = spacy.load("en_core_web_lg")
    except OSError:
        # Model not installed: download then load
        from spacy.cli import download

        print("[SEMANTIC] spaCy model 'en_core_web_lg' not found. Downloading...")
        download("en_core_web_lg")
        _NLP = spacy.load("en_core_web_lg")

    return _NLP

def _usefulness_from_structure(text: str, content_toks: List[str]) -> int:
    """
    Estimate 'usefulness' based on how much the text looks like
    procedural / instructional content, using syntax instead of
    a hard-coded phrase list.

    Signals:
      - Imperative sentences (root verb, no explicit subject).
      - Enumerated sentences / bullet lists (e.g. '1.', '-', '•').
    """
    # If there is no body at all, usefulness is 0.
    if not content_toks:
        return 0
    if not text.strip():
        # Some content, but nothing in meta_text; treat as minimally useful.
        return 1

    nlp = _get_nlp()
    doc = nlp(text)

    sentences = list(doc.sents)
    if not sentences:
        return 1

    num_sents = len(sentences)
    imperative_sents = 0
    enumerated_sents = 0

    for sent in sentences:
        # Imperative heuristic:
        #   - root is a VERB
        #   - no explicit nominal subject in the sentence
        root = sent.root
        has_subject = any(tok.dep_.startswith("nsubj") for tok in sent)
        if root.pos_ == "VERB" and not has_subject:
            imperative_sents += 1

        # Enumerated / bullet-like sentences:
        #   '1.', '2.' etc. or '-', '•', '*'
        first = sent[0]
        if first.like_num and len(sent) > 1 and sent[1].text == ".":
            enumerated_sents += 1
        elif first.text in {"-", "•", "*"}:
            enumerated_sents += 1

    imp_ratio = imperative_sents / num_sents
    enum_ratio = enumerated_sents / num_sents

    # Weighted signal: imperatives are stronger than enumerations.
    usefulness_signal = 2.0 * imp_ratio + 1.5 * enum_ratio

    # Map the continuous signal to 0–5 buckets.
    if usefulness_signal >= 1.2:
        return 5
    elif usefulness_signal >= 0.8:
        return 4
    elif usefulness_signal >= 0.5:
        return 3
    elif usefulness_signal >= 0.2:
        return 2
    else:
        # Some content but essentially descriptive / narrative.
        return 1



# ---------- Lexical semantic scores (clarity, usefulness, etc.) ----------
def semantic_scores(
    title: str,
    subtitle: str,
    desc: str,
    content: str,
    keywords: List[str],
) -> Dict[str, Any]:
    """
    Compute Semantic sub-scores:
      - clarity (0–5)      -> sentence length, few weird chars
      - usefulness (0–5)   -> presence of guidance / recommendation cues
      - info_density (0–5) -> lexical diversity
      - consistency (0–5)  -> overlap between title/desc/keywords and content
    """
    meta_text = " ".join([title, subtitle, desc])
    content_toks = tokens(content)
    content_toks_nostop = strip_stops(content_toks)

    # clarity: sentence length + low non-ascii
    s_lens = sentence_lengths(meta_text + " " + content)
    if s_lens:
        avg_len = sum(s_lens) / len(s_lens)
    else:
        avg_len = 0
    non_ascii = count_non_ascii(meta_text + " " + content)

    if 8 <= avg_len <= 35 and non_ascii <= 10:
        clarity = 5
    elif 6 <= avg_len <= 45 and non_ascii <= 30:
        clarity = 4
    elif 4 <= avg_len <= 55 and non_ascii <= 60:
        clarity = 3
    else:
        clarity = 1 if (avg_len > 0 or non_ascii > 0) else 0

    combined_text = meta_text + " " + content
    usefulness = _usefulness_from_structure(combined_text, content_toks)

    # information density: lexical diversity of content
    if content_toks_nostop:
        ur = unique_ratio(content_toks_nostop)
    else:
        ur = 0.0
    if ur >= 0.38:
        info_density = 5
    elif ur >= 0.30:
        info_density = 4
    elif ur >= 0.24:
        info_density = 3
    elif ur >= 0.18:
        info_density = 2
    elif ur > 0:
        info_density = 1
    else:
        info_density = 0

    # consistency: overlap between (title+desc+keywords) and content
    kw_text = " ".join(keywords)
    meta_toks = tokens(title + " " + desc + " " + kw_text)
    jc = jaccard(meta_toks, content_toks)
    if jc >= 0.5:
        consistency = 5
    elif jc >= 0.35:
        consistency = 4
    elif jc >= 0.2:
        consistency = 3
    elif jc >= 0.1:
        consistency = 2
    elif jc > 0:
        consistency = 1
    else:
        consistency = 0

    sem_raw = clarity + usefulness + info_density + consistency
    sem_scaled = round(sem_raw * 25.0 / 20.0)

    return {
        "Semantic_clarity": clarity,
        "Semantic_usefulness": usefulness,
        "Semantic_information_density": info_density,
        "Semantic_consistency": consistency,
        "Semantic_Total_Raw": sem_raw,
        "Semantic_Score_0_25": sem_scaled,
    }


# ---------- MNLI model (semantic consistency) ----------

MNLI_MODEL_NAME = os.environ.get("MNLI_MODEL_NAME", "roberta-large-mnli")

_MNLI_TOKENIZER = None  # type: Optional[AutoTokenizer]
_MNLI_MODEL: Optional[PreTrainedModel] = None
_MNLI_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_mnli_model() -> None:
    """
    Lazy-load roberta-large-mnli once and cache in globals.
    """
    global _MNLI_MODEL, _MNLI_TOKENIZER

    # If already loaded, do nothing
    if _MNLI_MODEL is not None and _MNLI_TOKENIZER is not None:
        return

    print(f"[MNLI] Loading model: {MNLI_MODEL_NAME} on device={_MNLI_DEVICE} ...")

    # Create local objects first
    tokenizer = AutoTokenizer.from_pretrained(MNLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MNLI_MODEL_NAME)

    model.to(_MNLI_DEVICE)
    model.eval()

    # Assign to globals *after* they are fully initialised
    _MNLI_TOKENIZER = tokenizer
    _MNLI_MODEL = model

    print("[MNLI] Model loaded.")


def run_mnli(premise: str, hypothesis: str) -> Tuple[str, float, float, float, float]:
    """
    Run MNLI on (premise, hypothesis) and return:
      (label, p_entail, p_neutral, p_contra, score_entail)
    """
    global _MNLI_MODEL, _MNLI_TOKENIZER

    _load_mnli_model()  # ensure globals are initialised

    if _MNLI_MODEL is None or _MNLI_TOKENIZER is None:
        raise RuntimeError("MNLI model/tokenizer not loaded correctly")

    # This tells type-checkers that from here on, _MNLI_MODEL is not None
    assert _MNLI_MODEL is not None

    # Shorten premise to avoid overlong sequences
    premise_short = premise[:3000]

    inputs = _MNLI_TOKENIZER(
        premise_short,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    # Move tensors to device (handle both dict and BatchEncoding)
    try:
        inputs = inputs.to(_MNLI_DEVICE)
    except AttributeError:
        inputs = {k: v.to(_MNLI_DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _MNLI_MODEL(**inputs)
        logits = outputs.logits[0]

    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()

    id2label = _MNLI_MODEL.config.id2label
    # Typically: {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
    label_map = {id2label[i].upper(): i for i in id2label}

    idx_ent = label_map.get("ENTAILMENT")
    idx_neu = label_map.get("NEUTRAL")
    idx_con = label_map.get("CONTRADICTION")

    p_ent = float(probs[idx_ent]) if idx_ent is not None else 0.0
    p_neu = float(probs[idx_neu]) if idx_neu is not None else 0.0
    p_con = float(probs[idx_con]) if idx_con is not None else 0.0

    best_idx = int(probs.argmax())
    label = id2label[best_idx]

    # We return p_ent twice: once explicitly, once as "entailment score" shortcut
    return label, p_ent, p_neu, p_con, p_ent


def semantic_mnli_consistency(
    title: str,
    subtitle: str,
    desc: str,
    content: str,
    lang_meta: str,
) -> Dict[str, Any]:
    """
    MNLI-based semantic consistency between metadata (title / subtitle / desc)
    and the KO body (ko_content_flat).

    We treat:
      premise    = content
      hypothesis = title / subtitle / description

    Only run for English metadata to keep it simple.
    Returns 0–5 scores and the underlying labels/probabilities for inspection.
    """
    # If not English metadata (or no content), skip MNLI
    if not content or not lang_meta.startswith("en"):
        return {
            "Semantic_consistency_mnli_title": 0.0,
            "Semantic_consistency_mnli_subtitle": 0.0,
            "Semantic_consistency_mnli_desc": 0.0,
            "Semantic_consistency_mnli": 0.0,
            "Semantic_mnli_title_label": "",
            "Semantic_mnli_desc_label": "",
            "Semantic_mnli_subtitle_label": "",
        }

    # Helper: map (label, p_entail, p_neutral, p_contra) -> score 0–5
    def _mnli_to_score(label: str, p_entail: float, p_neutral: float, p_contra: float) -> float:
        # Strongly punish contradictions
        if p_contra >= 0.5:
            return 0.0
        # Reward strong entailment
        if p_entail >= 0.7:
            return 5.0
        if p_entail >= 0.5:
            return 4.0
        # Otherwise treat as some kind of neutral / weak relation
        if p_neutral >= 0.5:
            return 2.0
        return 1.0

    # Title vs content
    if title:
        label_t, pe_t, pn_t, pc_t, _ = run_mnli(content, title)
        score_t = _mnli_to_score(label_t, pe_t, pn_t, pc_t)
    else:
        label_t, pe_t, pn_t, pc_t, score_t = "", 0.0, 0.0, 0.0, 0.0

    # Description vs content
    if desc:
        label_d, pe_d, pn_d, pc_d, _ = run_mnli(content, desc)
        score_d = _mnli_to_score(label_d, pe_d, pn_d, pc_d)
    else:
        label_d, pe_d, pn_d, pc_d, score_d = "", 0.0, 0.0, 0.0, 0.0

    # Subtitle vs content (optional)
    if subtitle:
        label_s, pe_s, pn_s, pc_s, _ = run_mnli(content, subtitle)
        score_s = _mnli_to_score(label_s, pe_s, pn_s, pc_s)
    else:
        label_s, pe_s, pn_s, pc_s, score_s = "", 0.0, 0.0, 0.0, 0.0

    # Aggregate 0–5 consistency score from the three components
    scores = [score_t, score_d, score_s]
    non_zero = [s for s in scores if s > 0]
    if non_zero:
        combined = sum(non_zero) / len(non_zero)  # average in 0–5
    else:
        combined = 0.0

    return {
        "Semantic_consistency_mnli_title": round(score_t, 2),
        "Semantic_consistency_mnli_subtitle": round(score_s, 2),
        "Semantic_consistency_mnli_desc": round(score_d, 2),
        "Semantic_consistency_mnli": round(combined, 2),
        "Semantic_mnli_title_label": label_t,
        "Semantic_mnli_desc_label": label_d,
        "Semantic_mnli_subtitle_label": label_s,
    }
