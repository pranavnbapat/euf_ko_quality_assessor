# assess_ko_quality/transformer_likelihood/ko_perplexity_4fields.py

"""
Perplexity-based quality scoring for KO metadata (4 fields only):
  - title, subtitle, description, keywords
and their improved variants:
  - title_llm, subtitle_llm, description_llm, keywords_llm

Outputs per KO:
  - ppl_{field}_orig
  - ppl_{field}_llm
  - delta_ppl_{field} = ppl_orig - ppl_llm  (positive = improved fluency)
  - weighted aggregates using FUS-driven weights

Usage:
  python -m assess_ko_quality.transformer_likelihood.ko_perplexity_4fields \
    --input /path/to/kos.json \
    --out_csv /path/to/out/ko_ppl_4fields.csv \
    --model Qwen/Qwen3-14B-Base \
    --batch_size 8 \
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast


# ----------------------------
# Normalisation helpers
# ----------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def strip_html(text: str) -> str:
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", text)).strip()

def normalise_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = strip_html(text)
    text = text.replace("\u00a0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text

def keywords_to_text(keywords: Any) -> str:
    if keywords is None:
        return ""
    if isinstance(keywords, list):
        parts = [str(x).strip() for x in keywords if x is not None and str(x).strip()]
        return "; ".join(parts)
    return str(keywords).strip()

def load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    raise ValueError("Input JSON must be a list[object] or a single object.")


# ----------------------------
# Perplexity computation
# ----------------------------

def compute_batch_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    max_length: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> List[float]:
    """
    Compute perplexity for a batch of texts (one PPL per text).
    Uses token-level negative log-likelihood averaged over non-padding tokens.
    """
    # Handle empties: return NaN so downstream can ignore
    if not texts:
        return []

    # Ensure every text yields at least one real token.
    # We map empty/whitespace-only text to EOS, which is a consistent baseline.
    eos_text = tokenizer.eos_token if tokenizer.eos_token is not None else ""
    safe_texts = [(t if (t and t.strip()) else eos_text) for t in texts]

    # Tokenise with padding + truncation
    enc = tokenizer(
        safe_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Some tokenizers can produce a 0-length sequence in edge cases.
    # Qwen3 cannot handle seq_len==0, so force at least 1 token.
    if input_ids.size(1) == 0:
        # Create a 1-token batch filled with pad/eos tokens
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        input_ids = torch.full((len(texts), 1), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), 1), dtype=torch.long)

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Shift for next-token prediction
    # logits: (B, T, V) ; labels should be next token => shift left
    with torch.no_grad():
        if amp_dtype is not None and device.type == "cuda":
            # Autocast speeds up on GPU; bf16 preferred if supported
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits  # (B, T, V)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Flatten for CE
        vocab_size = shift_logits.size(-1)
        shift_logits_2d = shift_logits.view(-1, vocab_size)
        shift_labels_1d = shift_labels.view(-1)

        # Ignore padding positions
        ignore_index = -100
        shift_labels_1d = shift_labels_1d.clone()
        shift_labels_1d[shift_mask.view(-1) == 0] = ignore_index

        # CrossEntropyLoss is safer in fp32
        shift_logits_2d = shift_logits_2d.float()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        token_losses = loss_fct(shift_logits_2d, shift_labels_1d)  # (B*(T-1),)

        token_losses = token_losses.view(shift_labels.size(0), -1)  # (B, T-1)
        valid_counts = shift_mask.sum(dim=1).clamp(min=1)          # (B,)
        sent_nll = (token_losses.sum(dim=1) / valid_counts)        # (B,)

        # NumPy can't handle bf16 on CPU; cast to fp32 first
        ppl = torch.exp(sent_nll.float()).detach().cpu().numpy().astype(np.float64)

    out: List[float] = []
    for v in ppl.tolist():
        # Always return a numeric PPL, even if the original text was empty.
        out.append(float(v))
    return out


def compute_ppl_for_texts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> List[float]:
    """
    Batched PPL computation for a list of texts.
    """
    results: List[float] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Perplexity", unit="batch"):
        batch = texts[i:i + batch_size]
        results.extend(
            compute_batch_ppl(
                model=model,
                tokenizer=tokenizer,
                texts=batch,
                max_length=max_length,
                device=device,
                amp_dtype=amp_dtype,
            )
        )
    return results


# ----------------------------
# Main pipeline
# ----------------------------

FIELD_SPECS = [
    # (field_name, orig_key, llm_key, max_length)
    ("title", "title", "title_llm", 64),
    ("subtitle", "subtitle", "subtitle_llm", 192),
    ("description", "description", "description_llm", 512),
    ("keywords", "keywords", "keywords_llm", 128),
]

# FUS-derived weights (normalised across the 4 fields)
# title=0.9468, description=0.6310, keywords=0.5447, subtitle=0.1010
WEIGHTS = {
    "title": 0.4258,
    "subtitle": 0.0454,
    "description": 0.2838,
    "keywords": 0.2450,
}


def weighted_mean(values: Dict[str, float], weights: Dict[str, float]) -> float:
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        v = values.get(k, float("nan"))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        num += w * float(v)
        den += w
    return (num / den) if den > 1e-12 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    ap.add_argument("--out_jsonl", default=None, type=Path)
    ap.add_argument("--model", default="Qwen/Qwen3-14B-Base")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu | cuda | auto")
    ap.add_argument("--dtype", default="bf16", help="bf16 | fp16 | fp32")
    args = ap.parse_args()

    # Device selection
    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # AMP dtype selection for CUDA
    amp_dtype: Optional[torch.dtype] = None
    if device.type == "cuda":
        if args.dtype == "bf16":
            amp_dtype = torch.bfloat16
        elif args.dtype == "fp16":
            amp_dtype = torch.float16
        elif args.dtype == "fp32":
            amp_dtype = None
        else:
            raise ValueError("--dtype must be bf16|fp16|fp32")

    records = load_json_records(args.input)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Ensure pad token exists (some causal LMs don't set it)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=(amp_dtype if amp_dtype is not None else None),
        device_map=None,
    ).to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []

    # Pre-build all texts per field to batch efficiently
    for field_name, orig_key, llm_key, max_len in FIELD_SPECS:
        # Collect normalised texts in record order
        orig_texts: List[str] = []
        llm_texts: List[str] = []

        orig_empty: List[bool] = []
        llm_empty: List[bool] = []

        for r in records:
            if field_name == "keywords":
                o = normalise_text(keywords_to_text(r.get(orig_key)))
                l = normalise_text(keywords_to_text(r.get(llm_key)))

                orig_empty.append(not bool(o.strip()))
                llm_empty.append(not bool(l.strip()))

                orig_texts.append(o)
                llm_texts.append(l)
            else:
                o = normalise_text(r.get(orig_key))
                l = normalise_text(r.get(llm_key))

                orig_empty.append(not bool(o.strip()))
                llm_empty.append(not bool(l.strip()))

                orig_texts.append(o)
                llm_texts.append(l)

        # Compute PPLs
        orig_ppl = compute_ppl_for_texts(
            model=model,
            tokenizer=tokenizer,
            texts=orig_texts,
            batch_size=args.batch_size,
            max_length=max_len,
            device=device,
            amp_dtype=amp_dtype,
        )
        llm_ppl = compute_ppl_for_texts(
            model=model,
            tokenizer=tokenizer,
            texts=llm_texts,
            batch_size=args.batch_size,
            max_length=max_len,
            device=device,
            amp_dtype=amp_dtype,
        )

        # Attach to rows
        if not rows:
            # initialise base row info once
            for i, r in enumerate(records):
                ko_id = r.get("@id") or r.get("_orig_id") or r.get("id") or f"idx:{i}"
                rows.append({
                    "ko_id": ko_id,
                    "language": (r.get("languages") or [""])[0] if isinstance(r.get("languages"), list) else (r.get("languages") or ""),
                    "category": r.get("category", ""),
                })

        for i in range(len(records)):
            rows[i][f"ppl_{field_name}_orig"] = float(orig_ppl[i])
            rows[i][f"ppl_{field_name}_llm"] = float(llm_ppl[i])

            rows[i][f"{field_name}_orig_empty"] = bool(orig_empty[i])
            rows[i][f"{field_name}_llm_empty"] = bool(llm_empty[i])

            # Positive delta means the LLM text is less surprising => more fluent/natural
            rows[i][f"delta_ppl_{field_name}"] = float(orig_ppl[i] - llm_ppl[i])

    # Compute weighted aggregates per KO
    for row in rows:
        ppl_orig_map = {f: row.get(f"ppl_{f}_orig", float("nan")) for f, *_ in FIELD_SPECS}
        ppl_llm_map = {f: row.get(f"ppl_{f}_llm", float("nan")) for f, *_ in FIELD_SPECS}
        delta_map = {f: row.get(f"delta_ppl_{f}", float("nan")) for f, *_ in FIELD_SPECS}

        row["ppl_weighted_orig"] = weighted_mean(ppl_orig_map, WEIGHTS)
        row["ppl_weighted_llm"] = weighted_mean(ppl_llm_map, WEIGHTS)
        row["delta_ppl_weighted"] = weighted_mean(delta_map, WEIGHTS)

        # Simple flag: LLM made things *less* fluent overall (negative delta)
        row["flag_fluency_regressed"] = bool(
            (row["delta_ppl_weighted"] == row["delta_ppl_weighted"]) and (row["delta_ppl_weighted"] < 0.0)
        )

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    if args.out_jsonl:
        args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.out_jsonl.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.out_csv}")
    if args.out_jsonl:
        print(f"Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()
