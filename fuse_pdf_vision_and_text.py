# fuse_pdf_vision_and_text.py

"""
Fuse (vision chunk summaries + optional vision reduce) with raw extracted text (ko_content_flat),
and ask a text LLM (Qwen3) to produce a clean final summary.

Inputs:
- --input-json: JSON array of KO objects containing "_orig_id" and "ko_content_flat"
- --vision-out-dir: directory containing per-KO vision output JSON files named "<id>.json"
- --out-dir: output directory for fused JSON files + an aggregate JSONL

Assumptions:
- Vision output JSON has: id, chunk_summaries (list), optional final_summary.summary
- Input JSON has: _orig_id and ko_content_flat

This script does NOT re-download PDFs and does NOT do rendering.
It only performs the fusion reduce using Qwen3.
"""

from __future__ import annotations

import argparse
import json
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


DEFAULT_VLLM_BASE_URL = "https://tld3x82ya8trj4-8000.proxy.runpod.net/v1"
DEFAULT_MODEL = "qwen3-30b-a3b-awq"


# ----------------------------
# Helpers
# ----------------------------

def _truncate_chars(s: str, max_chars: int) -> str:
    """Hard truncate by characters to protect token budget (simple + reliable)."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars].rstrip() + "\n\n[TRUNCATED]"


def _format_chunk_summaries(chunk_summaries: List[Dict[str, Any]], max_total_chars: int) -> str:
    """
    Format chunk summaries into a structured block.
    We also bound the total size by chars (cheap safeguard).
    """
    lines: List[str] = []
    for cs in chunk_summaries or []:
        pr = (cs.get("page_range") or "").strip()
        sm = (cs.get("summary") or "").strip()
        if not sm:
            continue
        header = f"- {pr}:" if pr else "- (chunk):"
        lines.append(f"{header}\n  {sm}")

    joined = "\n".join(lines).strip()
    return _truncate_chars(joined, max_total_chars)


def _build_fusion_prompt(
    title: str,
    raw_text: str,
    chunk_summaries_text: str,
    vision_reduce_hint: str,
) -> Tuple[str, str]:
    """
    Returns (system_prompt, user_prompt).
    The system prompt sets the rules; user prompt contains the evidence payloads.
    """
    system_prompt = (
        "You are an expert summariser for search indexing (OpenSearch), embeddings, and RAG-style chatbots.\n"
        "Your job is to produce a FACT-PRESERVING fused summary of a document.\n\n"
        "You will be given:\n"
        "1) RAW_TEXT (PRIMARY EVIDENCE)\n"
        "2) VISION_CHUNK_SUMMARIES (SECONDARY; may include figure/table info; may be noisy)\n"
        "3) VISION_REDUCE_HINT (TERTIARY; derived; may contain repeats/errors)\n\n"
        "NON-NEGOTIABLE RULES:\n"
        "- Prefer RAW_TEXT whenever the same topic appears in multiple places.\n"
        "- You may use VISION_CHUNK_SUMMARIES ONLY to add details that are clearly present in figures/tables/slides.\n"
        "- Do NOT upgrade visual concepts into factual outcomes. Examples of forbidden upgrades:\n"
        "  * A diagram/mockup ≠ 'implemented'/'deployed'/'tested'\n"
        "  * A roadmap arrow ≠ 'completed'\n"
        "  * An architecture slide ≠ 'system exists'\n"
        "- If a visual element does not explicitly state status, phrase it cautiously:\n"
        "  use 'illustrates', 'describes', 'proposes', 'shows a design for', 'outlines'.\n"
        "- Treat VISION_REDUCE_HINT only as a STRUCTURAL hint. Never introduce claims solely from it.\n"
        "- Remove duplicated/boilerplate/corrupted fragments. Do not speculate about missing parts.\n"
        "- British English only. Neutral tone.\n\n"
        "STRICT OUTPUT (MANDATORY):\n"
        "Return ONLY a single JSON object. No extra text, no markdown.\n"
        "The JSON MUST have exactly these keys and nothing else:\n"
        "{\"summary\": \"<...>\", \"visual_only_additions\": [\"...\"], \"uncertainties\": [\"...\"]}\n\n"
        "FIELD RULES:\n"
        "- summary: 3–8 short paragraphs, information-dense, fact-preserving.\n"
        "- visual_only_additions: 0–6 bullets. ONLY include items that appear in visuals AND are not already in RAW_TEXT.\n"
        "  Each bullet must be phrased with evidence-safe language unless the visual text explicitly states completion.\n"
        "- uncertainties: 0–4 bullets. ONLY include if there is an explicit conflict or ambiguity between sources.\n"
        "  Do not invent general risks (e.g. 'sustainability uncertain') unless the document states them.\n\n"
        "JSON STRING SAFETY:\n"
        "- Ensure valid JSON string escaping for quotes/newlines.\n"
    )

    user_prompt = (
        f"TITLE:\n{title.strip()}\n\n"
        "EVIDENCE BINS:\n"
        "- RAW_TEXT is primary truth.\n"
        "- VISION_CHUNK_SUMMARIES may contain OCR/interpretation noise.\n"
        "- VISION_REDUCE_HINT is ONLY for structure; do not extract facts from it.\n\n"
        f"RAW_TEXT (PRIMARY):\n{raw_text.strip()}\n\n"
        f"VISION_CHUNK_SUMMARIES (SECONDARY):\n{chunk_summaries_text.strip()}\n\n"
        f"VISION_REDUCE_HINT (TERTIARY):\n{vision_reduce_hint.strip()}\n"
    )

    return system_prompt, user_prompt


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
)
def _chat_completion(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float) -> str:
    """
    Calls an OpenAI-compatible Chat Completions endpoint.
    Retries transient failures.
    """
    messages = cast(List[Dict[str, Any]], [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=cast(Any, messages),
    )

    return (resp.choices[0].message.content or "").strip()


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", required=True, help="Path to input KO JSON array")
    ap.add_argument("--out-dir", required=True, help="Directory to write fused outputs")
    ap.add_argument("--api-base", default=DEFAULT_VLLM_BASE_URL, help="OpenAI-compatible base URL (must include /v1)")
    ap.add_argument("--api-key", default="EMPTY", help="API key (vLLM often ignores but OpenAI client requires a value)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model name exposed by vLLM (--served-model-name)")

    # Budget controls (chars ~= crude token control)
    ap.add_argument("--max-raw-text-chars", type=int, default=25000, help="Max chars from ko_content_flat")
    ap.add_argument("--max-chunk-summaries-chars", type=int, default=12000, help="Max chars for formatted chunk summaries")
    ap.add_argument("--max-reduce-hint-chars", type=int, default=4000, help="Max chars for vision reduce hint")

    ap.add_argument("--output-json", default=None, help="Path to write the enriched JSON array. Defaults to <out-dir>/enriched.json")
    ap.add_argument("--no-per-ko-files", action="store_true", help="Do not write <id>.json files; only write the enriched JSON array + JSONL")

    ap.add_argument("--temperature", type=float, default=0.2, help="Lower = more consistent")
    args = ap.parse_args()

    input_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load source KOs
    items = json.loads(input_path.read_text(encoding="utf-8"))
    input_stem = input_path.stem
    default_output_name = f"{input_stem}_extracted.json"
    output_json_path = (out_dir / default_output_name) if not args.output_json else (
                out_dir / Path(args.output_json).name)

    if not isinstance(items, list):
        raise ValueError("input-json must be a JSON array")

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    agg_path = out_dir / "fused_summaries.jsonl"
    agg_f = agg_path.open("w", encoding="utf-8")

    processed = 0
    missing_vision = 0
    failed = 0

    for obj in items:
        ko_id = (obj.get("_orig_id") or "").strip()
        title = (obj.get("title") or obj.get("title_meta") or "").strip()
        raw_text = (obj.get("ko_content_flat") or "").strip()

        if not ko_id:
            continue

        t0 = time.perf_counter()
        print(f"[{processed + failed + 1}/{len(items)}] KO={ko_id} starting…")

        # Vision summary is embedded in the same KO object now
        vision = obj.get("vision_summary")
        if not isinstance(vision, dict):
            vision = None

        has_raw_text = bool(raw_text.strip())
        raw_text_trunc = _truncate_chars(raw_text, args.max_raw_text_chars)

        # Simple heuristic: treat very short text as unusable
        raw_text_usable = has_raw_text and len(raw_text_trunc) >= 400

        if vision and (vision.get("chunk_summaries") or []):
            # Vision exists → #3 or #1
            chunk_summaries = vision.get("chunk_summaries") or []
            chunk_summaries_text = _format_chunk_summaries(chunk_summaries, args.max_chunk_summaries_chars)
            chunk_summaries_count = len(chunk_summaries)

            fs = vision.get("final_summary") or {}
            vision_reduce_hint = ""
            if isinstance(fs, dict):
                vision_reduce_hint = (fs.get("summary") or "").strip()
            vision_reduce_hint = _truncate_chars(vision_reduce_hint, args.max_reduce_hint_chars)
            has_vision_reduce_hint = bool(vision_reduce_hint)

            if raw_text_usable:
                # ✅ #3 — chunks + raw text
                raw_text_for_prompt = raw_text_trunc
                route = "chunks_plus_raw_text"
            else:
                # ✅ #1 — chunks only (fallback when text is poor)
                raw_text_for_prompt = "[RAW TEXT NOT USED – LOW QUALITY OR MISSING]"
                route = "chunks_only"
        else:
            # No usable vision → last resort #4 (raw text only)
            missing_vision += 1
            chunk_summaries_text = "[NO CHUNK SUMMARIES AVAILABLE]"
            chunk_summaries_count = 0
            vision_reduce_hint = "[NO VISION REDUCE SUMMARY AVAILABLE]"
            raw_text_for_prompt = raw_text_trunc if raw_text_usable else "[NO RAW TEXT AVAILABLE]"
            route = "raw_text_only"
            has_vision_reduce_hint = False

        print(f"  route={route} chunks={chunk_summaries_count} raw_chars={len(raw_text_trunc)}")

        system_prompt, user_prompt = _build_fusion_prompt(
            title=title or f"KO {ko_id}",
            raw_text=raw_text_for_prompt,
            chunk_summaries_text=chunk_summaries_text,
            vision_reduce_hint=vision_reduce_hint,
        )

        try:
            fused = _chat_completion(
                client=client,
                model=args.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=args.temperature,
            )

            elapsed_s = time.perf_counter() - t0

            fusion_inputs_obj = {
                "route": route,
                "has_raw_text": bool(raw_text),
                "raw_text_chars_used": len(raw_text_trunc),
                "chunk_count": chunk_summaries_count,
                "has_vision_reduce_hint": has_vision_reduce_hint,
            }

            fusion_timing_obj = {
                "elapsed_seconds": round(elapsed_s, 3),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

            # Enrich the original KO object in-place
            obj["fusion_inputs"] = fusion_inputs_obj
            obj["fused_summary"] = fused
            obj["fusion_timing"] = fusion_timing_obj

            # Optional: write per-KO JSON files
            if not args.no_per_ko_files:
                (out_dir / f"{ko_id}.json").write_text(
                    json.dumps(
                        {
                            "id": ko_id,
                            "title": title,
                            "fusion_inputs": fusion_inputs_obj,
                            "fusion_timing": fusion_timing_obj,
                            "fused_summary": fused,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            # Aggregate JSONL (one line per KO)
            agg_f.write(
                json.dumps(
                    {
                        "id": ko_id,
                        "title": title,
                        "fusion_inputs": fusion_inputs_obj,
                        "fusion_timing": fusion_timing_obj,
                        "fused_summary": fused,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            processed += 1
            print(f"  ✅ done in {elapsed_s:.2f}s")

        except Exception as e:
            failed += 1
            err_obj = {
                "id": ko_id,
                "title": title,
                "status": "error",
                "error": str(e),
            }
            (out_dir / f"{ko_id}.error.json").write_text(
                json.dumps(err_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            elapsed_s = time.perf_counter() - t0
            print(f"  ❌ failed in {elapsed_s:.2f}s: {e}")

    agg_f.close()

    # Write the enriched full JSON array (single file output)
    output_json_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Enriched JSON: {output_json_path}")

    print(f"Done. processed={processed} missing_vision={missing_vision} failed={failed}")
    print(f"Aggregate JSONL: {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
