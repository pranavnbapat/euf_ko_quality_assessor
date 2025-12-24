# summarise_from_pdf.py

"""
End-to-end: JSON (with PDF links) -> download -> render PDF pages to images -> vLLM(VLM) -> summaries.

How it works:
- Downloads PDFs from URLs found in a JSON file.
- Renders each PDF page to PNG images using PyMuPDF (fitz).
- Summarises pages in chunks (map step) via vLLM's OpenAI-compatible Chat Completions endpoint.
- Synthesises a final summary from chunk summaries (reduce step).
- Writes one output JSON per document + an aggregate JSONL file.

Requirements:
- vLLM server running with a vision-language model (e.g. Qwen2.5-VL) on http://localhost:8000/v1
- pip install pymupdf pillow requests openai tenacity
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import fitz  # PyMuPDF
import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_exponential


# -----------------------------
# Configuration defaults
# -----------------------------

DEFAULT_VLLM_BASE_URL = "https://tld3x82ya8trj4-8000.proxy.runpod.net/"
DEFAULT_MODEL = "internvl3_5-14b"
DEFAULT_PDF_URL_FIELD = "@id"

# Rendering: 150-200 dpi is usually a good trade-off
DEFAULT_DPI = 180

DEFAULT_PAGES_PER_CHUNK = 6

# Guardrail: don't render absurdly many pages by default
DEFAULT_MAX_PAGES = 60  # set 0 to disable

VisionSummary = Dict[str, Any]
JsonObject = Dict[str, Any]


# -----------------------------
# Prompts (tune as needed)
# -----------------------------

MAP_PROMPT = """
You are an expert summariser for search indexing (OpenSearch), embeddings, and RAG-style chatbots.

You will be given extracted text per page and sometimes page images. The PDF may be multilingual.

Task (MAP / chunk-level):
Write an information-rich, fact-preserving summary of ONLY these pages (the current chunk).
Write in BRITISH English. If the source is not English, translate into British English while keeping original names of projects, programmes, organisations, datasets, and tools.

STRICT OUTPUT (MANDATORY):
Return ONLY a single JSON object. No extra text, no preamble, no markdown, no comments.
The JSON MUST have exactly these keys and nothing else:
{"page_range":"pages X-Y","summary":"..."}


ROBUSTNESS (MANDATORY):
- Ignore unreadable, duplicated, boilerplate, or corrupted fragments; do not speculate about missing parts.
- Preserve key named entities, locations, dates, numbers, units, and distinctive domain terms.
- Prefer concrete facts and conditions over vague generalities.
- Do not include instructions or explanations—only the final JSON.

COMPLETENESS (MANDATORY):
- Maximise factual coverage of these pages.
- If the pages contain headings/sections/lists, ensure each substantive section is represented in the summary.
- Do not repeat the same point twice; merge duplicates.
- If the pages include explicit contact details (emails, websites, phone numbers), include them.

CITATIONS:
- When you state a specific claim, include an inline page citation like "(p3)" or "(p3–p4)" in the same sentence.

STYLE:
- British English only.
- Neutral, factual tone.
- Use multiple short paragraphs (separated by "\\n\\n") inside the JSON string.
- Avoid bullet points unless the original page is purely a list; if you must use lists, keep them short.
""".strip()


REDUCE_PROMPT = """
You are an expert summariser for search indexing (OpenSearch), embeddings, and RAG-style chatbots.

You will be given chunk summaries (each summarising a page range). Your task is to write one complete document-level summary in BRITISH English.

STRICT OUTPUT (MANDATORY):
Return ONLY a single JSON object. No extra text, no preamble, no markdown, no comments.
The JSON MUST have exactly this key and nothing else:
{"summary":"<summary>"}

ROBUSTNESS (MANDATORY):
- Preserve facts; do not invent.
- Merge duplicates; do not repeat yourself.
- Keep original names of projects/programmes/organisations/tools.
- If there are contradictions or uncertainty in the chunks, mention them explicitly.

CITATIONS:
- Preserve inline page citations from chunks where possible (e.g. "(p12)").

STYLE:
- British English only.
- Neutral, factual tone.
- Use multiple short paragraphs (separated by "\\n\\n") inside the JSON string.
- Do not output bullet points unless unavoidable.
""".strip()


# -----------------------------
# Utilities
# -----------------------------

def _safe_filename(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    return s[:180] if len(s) > 180 else s


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _autosave(out_dir: Path, input_json: Path, data_list: List[Dict[str, Any]]) -> None:
    out_path = out_dir / f"{input_json.stem}_visioned.json"
    _write_json(out_path, data_list)


def _b64_data_url(png_bytes: bytes) -> str:
    # vLLM OpenAI-compatible multimodal uses the same "data:" URL style as many OpenAI-compatible clients.
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=20))
def _download_file(url: str, out_path: Path, timeout_s: int = 60) -> None:
    # Retry-friendly download
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def render_pdf_to_pngs(
    pdf_path: Path,
    render_dir: Path,
    dpi: int = DEFAULT_DPI,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> List[Path]:
    """
    Render each PDF page to a PNG using PyMuPDF page.get_pixmap().
    PyMuPDF recommends using a Matrix / dpi to control resolution. :contentReference[oaicite:7]{index=7}
    """
    render_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    if max_pages and total_pages > max_pages:
        total_pages = max_pages

    out_paths: List[Path] = []
    for i in range(total_pages):
        page = doc.load_page(i)
        # Using dpi gives a predictable resolution; PyMuPDF supports this directly.
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        out_path = render_dir / f"page_{i+1:04d}.png"
        pix.save(out_path.as_posix())
        out_paths.append(out_path)

    doc.close()
    return out_paths


def extract_page_text(doc: fitz.Document, page_index: int) -> str:
    """
    Extract selectable text from a PDF page.
    For proper PDFs this is usually very accurate and cheaper than vision.
    """
    page = doc.load_page(page_index)
    # "text" is a reasonable default;
    text = page.get_text("text") or ""
    # Normalise a bit: collapse excessive whitespace
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def should_render_image(page_text: str, min_chars: int = 250) -> bool:
    """
    Decide whether to include a page image.
    Heuristic:
    - If little/no extractable text, it may be tables/figures or a scanned page -> use image.
    """
    return len(page_text) < min_chars


def chunk_list(items: List[Path], chunk_size: int) -> List[List[Path]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def parse_strict_json(text: str) -> Dict[str, Any]:
    """
    Many models mostly comply, but sometimes add leading/trailing text.
    We salvage the first JSON object block if needed.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Try to find the first {...} block
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def extract_first_json_object(text: str) -> str:
    """
    Extract the first top-level JSON object from a string by brace matching.
    This is more robust than a greedy regex when the text contains extra braces.
    """
    s = text.strip()
    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return s[start:]


def parse_strict_json_robust(text: str) -> Dict[str, Any]:
    """
    First try normal parsing.
    If that fails, try brace-matched extraction of first JSON object and parse again.
    """
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        t2 = extract_first_json_object(t)
        return json.loads(t2)


def _fmt_duration(seconds: float) -> str:
    """
    Human-friendly duration string.
    Examples: "12.3s", "3m 08.1s", "1h 02m 05.0s"
    """
    s = float(seconds)
    if s < 60:
        return f"{s}s"

    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"

    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def repair_json_with_vllm(
    client: OpenAI,
    model: str,
    raw_text: str,
    schema_hint: str,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """
    Ask the model to return VALID JSON only, preserving meaning.
    schema_hint: brief reminder of required keys/types to reduce model creativity.
    """
    repair_prompt = f"""
    You will be given a model output that was intended to be STRICT JSON, but it is invalid.

    Task:
    - Output a corrected version that is valid JSON.
    - Preserve the original meaning and content as much as possible.
    - Do NOT add new keys, do NOT drop keys.
    - Ensure all string values are valid JSON strings (escape internal double-quotes as \\", and newlines as \\n).
    - Output exactly one JSON object.
    - Do NOT wrap in markdown. Output JSON only.

    Schema hint (must match):
    {schema_hint}

    Invalid output:
    {raw_text}
    """.strip()

    messages = cast(
        List[ChatCompletionMessageParam],
        cast(object, [{"role": "user", "content": [{"type": "text", "text": repair_prompt}]}]),
    )

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    fixed = resp.choices[0].message.content or ""
    return parse_strict_json_robust(fixed)


# -----------------------------
# vLLM call (OpenAI-compatible)
# -----------------------------

def build_mm_message(chunk_pages: List[Dict[str, Any]], prompt: str) -> List[ChatCompletionMessageParam]:
    """
    Build a multimodal message:
    - Always include extracted text per page (cheap + accurate for proper PDFs)
    - Include images only for pages that need them (tables/figures/layout-heavy)
    """
    # Add page texts as a structured block (model can reference page numbers easily)
    page_text_blocks: List[str] = []
    for p in chunk_pages:
        t = (p.get("text") or "").strip()
        if t:
            page_text_blocks.append(f"--- PAGE {p['page_num']} TEXT ---\n{t}")
        else:
            page_text_blocks.append(f"--- PAGE {p['page_num']} TEXT ---\n<no extractable text>")

    combined_text = prompt + "\n\n" + "\n\n".join(page_text_blocks)

    content: List[Dict[str, Any]] = [{"type": "text", "text": combined_text}]

    # Append images only where available
    for p in chunk_pages:
        img_path: Optional[Path] = p.get("image_path")
        if img_path and img_path.exists():
            png_bytes = img_path.read_bytes()
            content.append({"type": "image_url", "image_url": {"url": _b64_data_url(png_bytes)}})

    # return [{"role": "user", "content": content}]
    return cast(List[ChatCompletionMessageParam], cast(object, [{"role": "user", "content": content}]))


def summarise_chunk_with_vllm(
    client: OpenAI,
    model: str,
    chunk_pages: List[Dict[str, Any]],
    page_start: int,
    page_end: int,
    temperature: float = 0.2,
    max_tokens: int = 3000,
) -> Dict[str, Any]:
    prompt = MAP_PROMPT + f"\n\nThese are pages {page_start}-{page_end}."
    messages: Any = build_mm_message(chunk_pages, prompt)

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    try:
        obj = parse_strict_json_robust(text)

        summary = obj.get("summary", "")
        has_citation = bool(re.search(r"\(p\d+(\u2013p?\d+)?\)", summary))

        if not has_citation:
            # Retry once with a strict reminder: add citations but do not change meaning.
            fix_msg = (
                "Your JSON is valid but you missed citations. "
                "Rewrite the SAME summary text, adding inline page citations like (p1) or (p1–p2) "
                "to the relevant sentences. Return the same JSON schema only."
            )
            fix_messages = cast(
                List[ChatCompletionMessageParam],
                cast(object, list(messages) + [{
                    "role": "user",
                    "content": [{"type": "text", "text": fix_msg}],
                }]),
            )

            resp2 = client.chat.completions.create(
                model=model,
                messages=fix_messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )

            obj = parse_strict_json_robust(resp2.choices[0].message.content or "")
    except Exception:
        # last resort: ask model to repair its JSON
        obj = repair_json_with_vllm(
            client=client,
            model=model,
            raw_text=text,
            schema_hint='{"page_range":"pages X-Y","summary":"..."}',
            max_tokens=2000,
        )
    obj.setdefault("page_range", f"pages {page_start}-{page_end}")
    return obj


def reduce_summaries_with_vllm(
    client: OpenAI,
    model: str,
    chunk_summaries: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 4200,
) -> Dict[str, Any]:
    """
    Final synthesis: give the model the chunk summaries (text only) and ask for the combined JSON.
    """
    # Keep input simple: the VLM does not need images at this stage.
    chunks_text = json.dumps(chunk_summaries, ensure_ascii=False, indent=2)
    messages = cast(
        List[ChatCompletionMessageParam],
        cast(object, [{
            "role": "user",
            "content": [
                {"type": "text", "text": REDUCE_PROMPT},
                {"type": "text", "text": "Chunk summaries JSON:\n" + chunks_text},
            ],
        }]),
    )

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    try:
        return parse_strict_json_robust(text)
    except Exception:
        return repair_json_with_vllm(
            client=client,
            model=model,
            raw_text=text,
            schema_hint='{"summary":"..."}',
            max_tokens=2400,
        )


# -----------------------------
# Main pipeline
# -----------------------------

@dataclass
class DocJob:
    doc_id: str
    pdf_url: str
    title: Optional[str] = None
    item: JsonObject = field(default_factory=dict)  # always a dict


def load_jobs(json_path: Path, url_field: str) -> Tuple[List[Dict[str, Any]], List[DocJob]]:
    data = _read_json(json_path)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    # Narrow type
    data_list: List[Dict[str, Any]] = [x for x in data if isinstance(x, dict)]

    jobs: List[DocJob] = []
    for idx, item in enumerate(data_list):
        url = item.get(url_field)
        if not url:
            continue

        doc_id = str(item.get("_orig_id") or item.get("id") or f"doc_{idx:05d}")
        title = item.get("title")

        jobs.append(DocJob(
            doc_id=doc_id,
            pdf_url=str(url),
            title=title,
            item=item,
        ))

    return data_list, jobs



def resolve_pdf_url(url: str, timeout_s: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve URL to something that looks like a PDF.
    Accept:
      - Content-Type application/pdf
      - Content-Type application/octet-stream (common for S3) if it smells like a PDF
    """
    headers = {"User-Agent": "pdf-summariser/1.0"}

    def _looks_like_pdf_from_headers(ct: str, cd: str) -> bool:
        ct = (ct or "").lower()
        cd = (cd or "").lower()
        if "application/pdf" in ct:
            return True
        # S3 often uses octet-stream for PDFs
        if "application/octet-stream" in ct and ".pdf" in cd:
            return True
        return False

    # Try GET (streamed) so we can also peek at the first bytes
    try:
        with requests.get(url, allow_redirects=True, stream=True, timeout=timeout_s, headers=headers) as r:
            if not r.ok:
                return None, f"http_status={r.status_code}"

            ct = r.headers.get("Content-Type", "")
            cd = r.headers.get("Content-Disposition", "")

            # If headers already strongly indicate PDF, accept
            if _looks_like_pdf_from_headers(ct, cd):
                return r.url, None

            # Otherwise peek at the first few bytes (PDF signature is %PDF-)
            first = r.raw.read(5, decode_content=True)  # small peek
            if first == b"%PDF-":
                return r.url, None

            return None, f"not_pdf_content_type={ct or 'unknown'}"
    except Exception as e:
        return None, f"resolve_error={e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-json", required=True, type=str, help="Path to JSON list containing PDF URLs.")
    p.add_argument("--url-field", default=DEFAULT_PDF_URL_FIELD, help="JSON field name that contains PDF URL.")
    p.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL, help="vLLM OpenAI-compatible base URL, e.g. http://localhost:8000/v1")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model name as served by vLLM")
    p.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    p.add_argument("--pages-per-chunk", type=int, default=DEFAULT_PAGES_PER_CHUNK)
    p.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="0 = no limit")
    p.add_argument("--out-dir", default="outputs", type=str)
    p.add_argument("--downloads-dir", default="downloads", type=str)
    p.add_argument("--renders-dir", default="renders", type=str)
    p.add_argument("--min-text-chars-for-no-image", type=int, default=250,
                   help="If extracted page text is shorter than this, also send page image.")
    p.add_argument("--force-images", action="store_true", help="Render an image for every page.")

    args = p.parse_args()

    # Ensure OpenAI-compatible base URL includes /v1 (prevents /chat/completions 404)
    args.vllm_base_url = args.vllm_base_url.rstrip("/")
    if not args.vllm_base_url.endswith("/v1"):
        args.vllm_base_url = args.vllm_base_url + "/v1"

    print(f"Using vLLM base_url = {args.vllm_base_url}")
    print(f"Using model = {args.model}")


    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = Path(args.downloads_dir); downloads_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = Path(args.renders_dir); renders_dir.mkdir(parents=True, exist_ok=True)

    data_list, jobs = load_jobs(input_json, url_field=args.url_field)
    if not jobs:
        raise SystemExit(f"No jobs found. Check --url-field '{args.url_field}' in {input_json}")

    # vLLM requires an API key header for OpenAI client, but it can be a dummy value.
    client = OpenAI(
        base_url=args.vllm_base_url,
        api_key=os.environ.get("VLLM_API_KEY", "local-vllm"),
    )

    total_t0 = time.perf_counter()

    for job in jobs:
        job_t0 = time.perf_counter()

        # Skip if already enriched (vision_summary exists and looks sane)
        vs = job.item.get("vision_summary")
        if isinstance(vs, dict) and vs.get("status") in {
            "ok",
            "partial_ok_with_chunk_errors",
            "reduce_failed",
            "map_failed_all_chunks",
            "render_failed",
            "download_failed",
            "download_not_pdf",
            "not_a_pdf",
            "no_pages",
        }:
            # Keep it idempotent but record timing for this run
            vs.setdefault("timing", {})
            vs["timing"]["skipped"] = True
            vs["timing"]["duration_seconds"] = round(time.perf_counter() - job_t0, 3)
            vs["timing"]["duration_human"] = _fmt_duration(time.perf_counter() - job_t0)
            continue

        safe_id = _safe_filename(job.doc_id)

        pdf_path = downloads_dir / f"{safe_id}.pdf"

        resolved_pdf_url, reason = resolve_pdf_url(job.pdf_url)
        if not resolved_pdf_url:
            job.item["vision_summary"] = {
                "id": job.doc_id,
                "title_meta": job.title,
                "pdf_url": job.pdf_url,
                "status": "not_a_pdf",
                "reason": reason,
            }
            _autosave(out_dir, input_json, data_list)
            job.item["vision_summary"]["timing"] = {
                "duration_seconds": round(time.perf_counter() - job_t0, 3),
                "duration_human": _fmt_duration(time.perf_counter() - job_t0),
            }
            continue

        # 1) Download
        try:
            # Sanity check: verify we actually have a PDF (starts with %PDF-)
            with pdf_path.open("rb") as f:
                magic = f.read(5)

            if magic != b"%PDF-":
                # Cached file is not a PDF; remove and try downloading once more
                try:
                    pdf_path.unlink(missing_ok=True)
                except Exception:
                    pass

                _download_file(resolved_pdf_url, pdf_path)

                with pdf_path.open("rb") as f:
                    magic2 = f.read(5)

                if magic2 != b"%PDF-":
                    job.item["vision_summary"] = {
                        "id": job.doc_id,
                        "title_meta": job.title,
                        "pdf_url": job.pdf_url,
                        "status": "download_not_pdf",
                        "magic": repr(magic2),
                    }
                    job.item["vision_summary"]["timing"] = {
                        "duration_seconds": round(time.perf_counter() - job_t0, 3),
                        "duration_human": _fmt_duration(time.perf_counter() - job_t0),
                    }
                    continue



        except Exception as e:
            job.item["vision_summary"] = {
                "id": job.doc_id,
                "title_meta": job.title,
                "pdf_url": job.pdf_url,
                "status": "download_failed",
                "error": str(e),
            }
            job.item["vision_summary"]["timing"] = {
                "duration_seconds": round(time.perf_counter() - job_t0, 3),
                "duration_human": _fmt_duration(time.perf_counter() - job_t0),
            }
            continue

        # 2) Render
        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            page_limit = args.max_pages if args.max_pages else total_pages
            page_limit = min(page_limit, total_pages)

            doc_render_dir = renders_dir / safe_id
            doc_render_dir.mkdir(parents=True, exist_ok=True)

            pages: List[Dict[str, Any]] = []
            for i in range(page_limit):
                page_text = extract_page_text(doc, i)

                img_path: Optional[Path] = None

                pipe_heavy = page_text.count("|") >= 8
                looks_tableish = pipe_heavy or any(
                    m in page_text for m in ["Table ", "TABLE ", "Fig.", "Figure", "Δ", "Σ"])

                if args.force_images or should_render_image(page_text, min_chars=args.min_text_chars_for_no_image) or looks_tableish:
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=args.dpi, alpha=False)
                    img_path = doc_render_dir / f"page_{i + 1:04d}.png"
                    pix.save(img_path.as_posix())

                pages.append({
                    "page_num": i + 1,
                    "text": page_text,
                    "image_path": img_path,
                })

            doc.close()
        except Exception as e:
            job.item["vision_summary"] = {
                "id": job.doc_id,
                "title_meta": job.title,
                "pdf_url": job.pdf_url,
                "status": "render_failed",
                "error": str(e),
            }
            job.item["vision_summary"]["timing"] = {
                "duration_seconds": round(time.perf_counter() - job_t0, 3),
                "duration_human": _fmt_duration(time.perf_counter() - job_t0),
            }
            continue

        if not pages:
            job.item["vision_summary"] = {
                "id": job.doc_id,
                "title_meta": job.title,
                "pdf_url": job.pdf_url,
                "status": "no_pages",
            }
            job.item["vision_summary"]["timing"] = {
                "duration_seconds": round(time.perf_counter() - job_t0, 3),
                "duration_human": _fmt_duration(time.perf_counter() - job_t0),
            }
            continue

        # 3) Map step: chunk page images and summarise each chunk
        chunk_summaries: List[Dict[str, Any]] = []
        page_chunks: List[List[Dict[str, Any]]] = [
            pages[i:i + args.pages_per_chunk] for i in range(0, len(pages), args.pages_per_chunk)
        ]

        for ci, chunk_pages in enumerate(page_chunks):
            page_start = chunk_pages[0]["page_num"]
            page_end = chunk_pages[-1]["page_num"]

            try:
                chunk_summary = summarise_chunk_with_vllm(
                    client=client,
                    model=args.model,
                    chunk_pages=chunk_pages,
                    page_start=page_start,
                    page_end=page_end,
                )
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                chunk_summaries.append({"page_range": f"pages {page_start}-{page_end}", "error": str(e)})

        chunk_error_count = sum(1 for cs in chunk_summaries if "error" in cs)

        # 4) Reduce step: final synthesis
        if chunk_error_count == len(chunk_summaries):
            status = "map_failed_all_chunks"
            final_summary = {"error": "All map chunks failed; skipping reduce."}
        else:
            try:
                final_summary = reduce_summaries_with_vllm(
                    client=client,
                    model=args.model,
                    chunk_summaries=chunk_summaries,
                )
                status = "ok" if chunk_error_count == 0 else "partial_ok_with_chunk_errors"
            except Exception as e:
                final_summary = {"error": str(e)}
                status = "reduce_failed"

        num_images = sum(1 for p in pages if p.get("image_path"))

        output_obj = {
            "id": job.doc_id,
            "title_meta": job.title,
            "pdf_url": job.pdf_url,
            "status": status,
            "rendered_pages": len(pages),
            "pages_per_chunk": args.pages_per_chunk,
            "chunk_count": len(chunk_summaries),
            "chunk_error_count": chunk_error_count,
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary,
            "rendered_images": num_images,
            "duration_seconds": round(time.perf_counter() - job_t0, 3),
            "duration_human": _fmt_duration(time.perf_counter() - job_t0),
        }

        job.item["vision_summary"] = output_obj

        # Autosave progress so results appear during long runs
        out_path = out_dir / f"{input_json.stem}_visioned.json"
        _write_json(out_path, data_list)

        job_dt = time.perf_counter() - job_t0
        print(
            f"{status.upper()}: {job.doc_id} ({len(pages)} pages, chunk_errors={chunk_error_count}/{len(chunk_summaries)}) "
            f"in {_fmt_duration(job_dt)}"
        )

        # Tiny delay to avoid hammering the server if we run many documents
        time.sleep(0.05)

    # Write one enriched JSON file (same structure as input, with per-item vision_summary)
    out_path = out_dir / f"{input_json.stem}_visioned.json"
    _write_json(out_path, data_list)
    print(f"Wrote enriched JSON: {out_path}")

    total_dt = time.perf_counter() - total_t0
    print(f"TOTAL TIME: {_fmt_duration(total_dt)}")

if __name__ == "__main__":
    main()
