# assess_ko_quality/stages/resummarise.py
"""
Regenerate the reference summary from the KO's own file (--resummarise, opt-in).

The summary is written in the document's own language, read from the file itself, and
becomes the reference every field of that KO is measured against (stages/metadata_score,
stages/llm_score, stages/generate_metadata). A non-English summary is also translated
to English: that version keeps the KO's own row in the corpus index English like every
other row (CorpusIndex.override_reference) and serves as the secondary reference for
fields written in English. Both texts, and the platform's own summary, go into the run's
outputs.

The file is read as it is, whatever it is:
  pdf / image      the pages rendered to images (PyMuPDF), one page per image, for a
                   PDF of up to `max_image_pages` pages (100). A longer PDF is read
                   from its own text layer instead, in chunks of pages of at most
                   `max_source_chars` characters, and summarised the same way; a text
                   layer that fails the Gopher quality filter's word rules (a scan,
                   glyph codes) sends the KO to the export's text. The export's stored
                   text is not used while the file is readable: it can be a
                   translation of the file (found on 6749f43db330af0ffc54189f). A PDF
                   over `max_pdf_mb` (100 MB) is not downloaded; other files up to
                   `max_file_mb` (600 MB);
  office           doc/docx/ppt/pptx/xls/xlsx/odt/odp/ods/rtf converted to PDF by
                   LibreOffice headless first, then as above;
  video            the sound transcribed (whisper-large-v3 through the provider's
                   /v1/audio/transcriptions, in segments) and frames sampled evenly
                   across the recording, shown next to the transcript;
  audio            the transcript alone.
A KO whose file cannot be fetched or read falls back to the export's text, and a file
flow that fails retries once from the export's text; the basis is recorded per KO.

How the summary is written:
  one chunk        up to `chunk_pages` pages (12, the provider's images-per-prompt
                   limit), a transcript with frames, or a text: one draft call.
  more pages       hierarchical merging (Chang, Lo, Goyal & Iyyer, ICLR 2024,
                   "BooookScore"): detailed notes per chunk, merged `merge_fanin` at a
                   time until one summary remains - coherence over the extra detail
                   of incremental updating, which the same study measured.
  verification     Chain-of-Verification (Dhuliawala et al., Findings of ACL 2024),
                   adapted: the draft is read against its source for statements the
                   source contradicts or does not support, and for major content it
                   lacks; one revision call applies every finding. No findings, no
                   revision. A source in one chunk is checked in one call. A longer
                   one is checked one chunk per call, sentence by sentence, because a
                   chunk cannot tell a wrong statement from one about other pages:
                   each chunk gives every numbered sentence a verdict (supported,
                   contradicted, absent) and the code decides across chunks. A
                   sentence is supported when any chunk supports it, as AlignScore
                   (Zha et al., ACL 2023) and SummaC-ZS (Laban et al., TACL 2022)
                   keep each sentence's best score over the source's parts; an error
                   when a chunk contradicts it and none supports it; unsupported
                   when every chunk reports it absent.
  translation      a non-English summary is translated to English for the corpus row
                   (the only call whose output is not the reference).

Calls per KO: 2-4 for a document of up to `chunk_pages` pages, about two more per
further chunk plus the merges; one transcription call per audio segment. Every call
is counted before it is made and disk-cached (core/llm.py; page images are part of the
cache key), so a rerun is free. The model (`resummarise.model`, --resummarise-model)
must be vision-capable. Runs made with --resummarise are not comparable with runs on
the export's summaries.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.corpus_index import ko_id, reference_text

SYSTEM = (
    "You are an expert summariser for a European knowledge platform on agriculture, "
    "forestry, livestock, food and rural life. You will be given one document - its "
    "pages shown as images, its text, or the transcript and frames of a recording - in "
    "any language. You read it closely and write in the language the document itself "
    "is written in. You never invent anything."
)

_LANGUAGE_RULE = (
    "Name the main language the document itself is written in (what you see on its "
    "pages or hear in the recording), as an English word, e.g. \"Lithuanian\". Write "
    "everything you produce in THAT language - not in English, unless the document is "
    "in English."
)

_CONTENT_RULES = """- cover ALL major topics, sections and findings, in the document's order;
- preserve named entities (people, organisations, projects, programmes), locations,
  dates, numbers, units, acronyms and distinctive domain terms;
- for technical or research content cover objectives, context, methods, key results,
  conclusions and stated limitations;
- ignore unreadable, duplicated or boilerplate fragments and invent nothing."""

_LENGTH_RULE = """Length: proportional to the document's information density - roughly 150-300 words for
a short document, 400-900 for a medium one, up to 2,000 for a long, dense one. Do not
over-compress rich content."""

# Same goals as the platform's own summariser (every major topic, entities and numbers
# kept, proportional length), plus the language rule.
DRAFT_BRIEF = f"""=== TASK ===
Write a detailed, flowing, fact-preserving summary of this document, for search
indexing and retrieval, in the document's own language. The summary must:
{_CONTENT_RULES}
{_LENGTH_RULE}
{_LANGUAGE_RULE}
Return JSON only: {{"language": "<the document's main language, in English>", \
"summary": "<the summary, in that language>"}}"""


def notes_brief(first: int, last: int, total: int) -> str:
    """Chunk-level notes, the first level of hierarchical merging."""
    return f"""=== TASK ===
These are pages {first}-{last} of {total} of one document. Write detailed notes on THESE
pages only, in the document's own language. The notes must:
{_CONTENT_RULES}
Keep every fact a summary of the whole document could need; do not summarise yet.
{_LANGUAGE_RULE}
Return JSON only: {{"language": "<the document's main language, in English>", \
"notes": "<the notes, in that language>"}}"""


def merge_brief(parts: Sequence[str], language: str, final: bool) -> str:
    """Merge consecutive parts' notes: into notes again, or - at the top - the summary."""
    body = "\n\n".join(f"=== PART {i + 1} of {len(parts)} ===\n{p}" for i, p in enumerate(parts))
    if final:
        task = (f"Merge these notes on consecutive parts of one document into ONE summary of "
                f"the whole document, in {language}, for search indexing and retrieval. The "
                f"summary must:\n{_CONTENT_RULES}\n{_LENGTH_RULE}\n"
                'Return JSON only: {"summary": "<the summary>"}')
    else:
        task = (f"Merge these notes on consecutive parts of one document into ONE set of "
                f"notes, in {language}, keeping every fact, number, name and finding, in the "
                "document's order; remove only repetition.\n"
                'Return JSON only: {"notes": "<the merged notes>"}')
    return body + "\n\n=== TASK ===\n" + task


def verify_brief(draft: str, scope: str) -> str:
    """Chain-of-Verification, adapted: the draft read against the whole source at once."""
    return f"""=== DRAFT SUMMARY ===
{draft}

=== TASK ===
Check the draft summary against {scope}, fact by fact.
- errors: statements in the draft that are wrong about {scope} - a wrong number, name,
  date, finding or claim. List under errors every statement that nothing in the document supports. Quote the statement and say what {scope} actually
  say(s).
- missing: major topics, sections or findings in {scope} that the draft does not cover.
Only what you can see; empty lists when there is nothing. Write in the draft's language.
Return JSON only: {{"errors": ["<statement - correction>", ...], \
"missing": ["<what is missing>", ...]}}"""


def verify_sentences_brief(sentences: Sequence[str], scope: str) -> str:
    """One chunk of a longer source: a verdict per numbered sentence (see combine_verdicts)."""
    numbered = "\n".join(f"[{n}] {s}" for n, s in enumerate(sentences, 1))
    return f"""=== DRAFT SUMMARY, ONE SENTENCE PER NUMBER ===
{numbered}

=== TASK ===
You see only part of the document: {scope}. The rest is on other pages. Check every
numbered sentence against {scope}, fact by fact, and give each one a verdict:
- "supported": {scope} state it, or part of it, and contradict none of it;
- "contradicted": {scope} say something different - a different number, name, date,
  finding or claim. Say what {scope} actually say;
- "absent": {scope} neither state nor contradict it. This is expected for content
  from other pages and is not an error.
Then list under missing the major topics, sections or findings in {scope} that the
draft does not cover. Only what you can see. Write in the draft's language.
Return JSON only: {{"verdicts": [{{"n": 1, "verdict": "supported", "correction": ""}}, ...], \
"missing": ["<what is missing>", ...]}}
One verdict for every number; "correction" only for contradicted sentences."""


# A sentence ends at . ! ? or an ellipsis, then optional closing quotes or brackets, a
# space and a capital letter or an opening quote. Not after a word of one or two letters
# or one with a full stop inside (Dr., e.g., U.S.): a split missed there only gives two
# sentences one verdict.
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’»)\]]*\s+")
_OPENERS = "\"'“‘„«(["


def _sentences(text: str) -> List[str]:
    out: List[str] = []
    start = 0
    for m in _SENTENCE_END.finditer(text):
        after = text[m.end():m.end() + 1]
        if not after or not (after.isupper() or after in _OPENERS):
            continue
        words = text[start:m.start()].split()
        last = words[-1].strip(_OPENERS) if words else ""
        if len(last) <= 2 or "." in last:
            continue
        out.append(text[start:m.end()].strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _verdicts(raw: Dict[str, Any], n_sentences: int) -> Dict[int, Tuple[str, str]]:
    """{sentence number: (verdict, correction)} from one chunk's answer; anything malformed
    is dropped, so that sentence simply has no verdict from this chunk."""
    out: Dict[int, Tuple[str, str]] = {}
    for item in raw.get("verdicts") or []:
        if not isinstance(item, dict):
            continue
        try:
            n = int(item.get("n"))
        except (TypeError, ValueError):
            continue
        word = _clean(item.get("verdict")).lower()
        verdict = ("supported" if word.startswith("support") else
                   "contradicted" if word.startswith("contradict") else
                   "absent" if word.startswith("absent") else "")
        if verdict and 1 <= n <= n_sentences:
            out[n] = (verdict, _clean(item.get("correction")))
    return out


def combine_verdicts(sentences: Sequence[str], per_chunk: Sequence[Dict[int, Tuple[str, str]]]
                     ) -> Tuple[List[str], int]:
    """
    One decision per sentence from every chunk's verdict. Supported by any chunk: kept.
    Contradicted by a chunk and supported by none: an error, with the correction.
    Reported absent by every chunk: unsupported, since nothing in the document states
    it. A sentence that some chunk gave no verdict on is never called unsupported, so a
    malformed answer cannot delete a true statement. Returns (errors, unverified), where
    unverified counts the sentences no chunk gave any verdict on.
    """
    errors: List[str] = []
    unverified = 0
    for n, sentence in enumerate(sentences, 1):
        given = [v[n] for v in per_chunk if n in v]
        if not given:
            unverified += 1
            continue
        if any(verdict == "supported" for verdict, _ in given):
            continue
        corrections = [c for verdict, c in given if verdict == "contradicted"]
        if corrections:
            errors.append(f"{sentence} - " + ("; ".join(c for c in corrections if c)
                                              or "contradicted by the document"))
        elif len(given) == len(per_chunk):
            errors.append(f"{sentence} - nothing in the document states this")
    return errors, unverified


def revise_brief(draft: str, errors: Sequence[str], missing: Sequence[str], language: str) -> str:
    e = "\n".join(f"- {x}" for x in errors) or "- none"
    m = "\n".join(f"- {x}" for x in missing) or "- none"
    return f"""=== DRAFT SUMMARY ===
{draft}

=== REVIEW: ERRORS (statement - correction) ===
{e}

=== REVIEW: MISSING CONTENT ===
{m}

=== TASK ===
Rewrite the summary in {language}: correct or remove every listed error, add the missing
content where it belongs, keep everything else as it is, and keep the summary flowing and
fact-preserving.
{_LENGTH_RULE}
Return JSON only: {{"summary": "<the revised summary>"}}"""


def translate_brief(summary: str, language: str) -> str:
    return f"""=== SUMMARY ({language}) ===
{summary}

=== TASK ===
Translate this summary into British English, complete and faithful: keep every fact,
number, date and name; keep project and programme names as they are.
Return JSON only: {{"summary_en": "<the translation>"}}"""


# ----------------------------------------------------------------- the file itself
_IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "webp": "image/webp", "gif": "image/gif"}
_OFFICE = {"doc", "docx", "ppt", "pptx", "xls", "xlsx", "odt", "odp", "ods", "rtf"}
_VIDEO = {"mp4", "m4v", "mov", "avi", "mkv", "webm", "mpg", "mpeg", "wmv"}
_AUDIO = {"mp3", "wav", "m4a", "ogg", "oga", "flac", "aac", "opus"}
_MIME_EXT = {
    "application/pdf": "pdf",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/rtf": "rtf", "text/rtf": "rtf",
    **{v: k for k, v in _IMAGE_TYPES.items()},
}
_FILE_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "files"


def _declared_ext(rec: Dict[str, Any]) -> str:
    ext = str(rec.get("ko_object_extension") or "").lower().lstrip(".")
    if ext:
        return ext
    mime = str(rec.get("ko_object_mimetype") or "").lower().split(";")[0].strip()
    if mime in _MIME_EXT:
        return _MIME_EXT[mime]
    if mime.startswith("video/"):
        return "mp4"
    if mime.startswith("audio/"):
        return "mp3"
    return ""


def _download(url: str, timeout: int, max_bytes: int) -> Tuple[Optional[bytes], str]:
    """
    The file at `url` and "", or None and why not: "too_large" over the size cap, else
    "download_failed". Cached on disk by URL (.cache/files/, gitignored): a video of
    150 MB is fetched once.
    """
    import requests

    _FILE_CACHE.mkdir(parents=True, exist_ok=True)
    path = _FILE_CACHE / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:24] + ".bin")
    if path.exists():
        if path.stat().st_size <= max_bytes:
            return path.read_bytes(), ""
        return None, "too_large"
    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                return None, "download_failed"
            if int(r.headers.get("Content-Length") or 0) > max_bytes:
                return None, "too_large"
            buf = io.BytesIO()
            for chunk in r.iter_content(chunk_size=1 << 20):
                buf.write(chunk)
                if buf.tell() > max_bytes:
                    return None, "too_large"
            data = buf.getvalue()
    except Exception:
        return None, "download_failed"
    try:
        path.write_bytes(data)
    except OSError:
        pass
    return data, ""


def _file_kind(rec: Dict[str, Any], data: bytes) -> str:
    """"pdf", an image / office / video / audio extension, or "" - magic bytes first."""
    head = data[:16]
    declared = _declared_ext(rec)
    if head[:5] == b"%PDF-":
        return "pdf"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:3] == b"\xff\xd8\xff":
        return "jpg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"PK\x03\x04":                     # OOXML / OpenDocument
        return declared if declared in _OFFICE else "docx"
    if head[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":  # legacy Office (OLE)
        return declared if declared in ("doc", "ppt", "xls") else "doc"
    if head[:5] == b"{\\rtf":
        return "rtf"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[4:8] == b"ftyp":                           # MP4 family: video or m4a
        mime = str(rec.get("ko_object_mimetype") or "").lower()
        return "m4a" if (mime.startswith("audio/") or declared in _AUDIO) else "mp4"
    if head[:4] == b"\x1aE\xdf\xa3":                   # Matroska / WebM
        return "webm" if declared == "webm" else "mkv"
    if declared in _IMAGE_TYPES | _OFFICE | _VIDEO | _AUDIO or declared == "pdf":
        return declared
    return ""


def _run(cmd: List[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)


def _office_to_pdf(data: bytes, ext: str, timeout: int) -> Optional[bytes]:
    """LibreOffice headless, with its own profile so parallel conversions do not collide."""
    if not shutil.which("soffice"):
        return None
    tmp = Path(tempfile.mkdtemp(prefix="koq_office_"))
    try:
        src = tmp / f"in.{ext}"
        src.write_bytes(data)
        _run(["soffice", f"-env:UserInstallation=file://{tmp / 'profile'}", "--headless",
              "--convert-to", "pdf", "--outdir", str(tmp), str(src)], timeout)
        out = tmp / "in.pdf"
        return out.read_bytes() if out.exists() else None
    except Exception:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _render_pdf(doc: Any, dpi: int, quality: int) -> List[Dict[str, Any]]:
    """
    Every page as a JPEG image part, encoded as it is rendered so a long document never
    holds many bitmaps at once.
    """
    from PIL import Image

    parts = []
    for p in doc:
        img = Image.open(io.BytesIO(p.get_pixmap(dpi=dpi).tobytes("png"))).convert("RGB")
        parts.append(_image_part(img, quality))
    return parts


def _pdf_text(doc: Any, max_chars: int) -> Tuple[Optional[List[Tuple[int, int, str]]], int]:
    """
    A PDF read from its own text layer instead of as page images: the pages in order,
    grouped into chunks of at most `max_chars` characters, as (first page, last page,
    text) with a marker before every page; and the number of pages with any text.
    None when the text layer is unusable by the Gopher quality filter's rules (Rae et
    al. 2021, Table A1): fewer than 50 words, or fewer than 80% of words containing a
    letter. That catches a scan without a text layer and a layer of glyph codes.
    """
    from core.content_report import GOPHER_QUALITY

    pages = [re.sub(r"[ \t]+", " ", p.get_text("text") or "").strip() for p in doc]
    words = re.findall(r"\w+", " ".join(pages))
    alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    if len(words) < GOPHER_QUALITY["min_words"] or \
            alpha / len(words) < GOPHER_QUALITY["min_alpha_word_ratio"]:
        return None, sum(1 for t in pages if t)
    groups: List[Tuple[int, int, str]] = []
    size = 0
    for n, text in enumerate(pages, 1):
        if not text:
            continue
        block = f"--- page {n} ---\n{text}"[:max_chars]
        if groups and size + len(block) + 2 <= max_chars:
            first, _, joined = groups[-1]
            groups[-1] = (first, n, f"{joined}\n\n{block}")
            size += len(block) + 2
        else:
            groups.append((n, n, block))
            size = len(block)
    return groups, sum(1 for t in pages if t)


def _media(data: bytes, ext: str, cfg: Dict[str, Any], with_frames: bool
           ) -> Tuple[List[Any], str, Dict[str, Any]]:
    """
    A recording as the model can read it: frames sampled evenly across a video (JPEG,
    PIL) and the transcript of its sound (whisper-large-v3, in segments). Returns
    (frames, transcript, {duration, frames, segments, transcription_calls}).
    """
    from PIL import Image

    from core.llm import transcribe

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("ffmpeg/ffprobe not installed")
    tmp = Path(tempfile.mkdtemp(prefix="koq_media_"))
    meta: Dict[str, Any] = {"transcription_calls": 0}
    try:
        src = tmp / f"in.{ext}"
        src.write_bytes(data)
        probe = _run(["ffprobe", "-v", "error", "-show_entries",
                      "format=duration:stream=codec_type", "-of", "json", str(src)], 120)
        info = json.loads(probe.stdout or b"{}") if probe.returncode == 0 else {}
        duration = float((info.get("format") or {}).get("duration") or 0.0)
        kinds = {s.get("codec_type") for s in info.get("streams") or []}
        meta["duration_s"] = round(duration, 1)

        frames: List[Any] = []
        if with_frames and "video" in kinds and duration > 0:
            n = int(cfg.get("video_frames", 12))
            width = int(cfg.get("frame_width", 1024))
            for i in range(n):
                t = duration * (i + 0.5) / n
                out = tmp / f"frame_{i:03d}.jpg"
                _run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(src),
                      "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(out)], 120)
                if out.exists():
                    frames.append(Image.open(out).convert("RGB"))
        meta["frames"] = len(frames)

        transcript = ""
        if "audio" in kinds:
            seg = int(cfg.get("audio_segment_seconds", 600))
            _run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1",
                  "-ar", "16000", "-b:a", "48k", "-f", "segment", "-segment_time", str(seg),
                  str(tmp / "audio_%03d.mp3")], 1800)
            pieces = sorted(tmp.glob("audio_*.mp3"))
            texts = []
            for p in pieces:
                text, cached = transcribe(p.read_bytes(), str(cfg.get("transcription_model")
                                                              or "whisper-large-v3"), p.name)
                meta["transcription_calls"] += 0 if cached else 1
                if text:
                    texts.append(text)
            transcript = " ".join(texts).strip()
            meta["segments"] = len(pieces)
        meta["transcript_words"] = len(transcript.split())
        return frames, transcript, meta
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _image_part(img: Any, quality: int) -> Dict[str, Any]:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _text_part(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


class Source:
    """One KO's file as chunks the model can read, plus how it was obtained."""

    def __init__(self, chunks: List[List[Dict[str, Any]]], scope: str, part_scope: str,
                 pages: int, basis: Dict[str, Any],
                 ranges: Optional[List[Tuple[int, int]]] = None) -> None:
        self.chunks = chunks          # each: content parts (images and/or text)
        self.scope = scope            # "the document" / "the transcript and frames" / ...
        self.part_scope = part_scope  # "these pages", when there are several chunks
        self.pages = pages
        self.basis = basis            # recorded per KO (_resummarised_from)
        self.ranges = ranges          # (first, last) page of each chunk; None: one per image


def acquire(rec: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Optional[Source], str]:
    """
    Fetch the KO's file from its @id URL and turn it into chunks: page images for a
    PDF / image / office document, frames plus transcript for a video, the transcript
    for an audio file. Returns (source, "") or (None, why-not).
    """
    url = str(rec.get("@id") or "").strip()
    if not url.lower().startswith("http"):
        return None, "no_file_url"
    file_mb = int(cfg.get("max_file_mb", 600))
    pdf_mb = int(cfg.get("max_pdf_mb", 0)) or file_mb
    declared_pdf = _declared_ext(rec) == "pdf"
    data, why = _download(url, int(cfg.get("download_timeout", 300)),
                          (pdf_mb if declared_pdf else file_mb) << 20)
    if data is None:
        if why == "too_large":
            return None, f"pdf_over_{pdf_mb}_mb" if declared_pdf else f"file_over_{file_mb}_mb"
        return None, why
    kind = _file_kind(rec, data)
    if kind == "pdf" and len(data) > pdf_mb << 20:      # a PDF its metadata did not declare
        return None, f"pdf_over_{pdf_mb}_mb"
    quality = int(cfg.get("render_jpeg_quality", 80))
    chunk_pages = int(cfg.get("chunk_pages") or cfg.get("max_images_per_call", 12))
    basis: Dict[str, Any] = {"from": "file", "kind": kind, "bytes": len(data)}

    if kind in _VIDEO or kind in _AUDIO:
        is_video = kind in _VIDEO
        frames, transcript, meta = _media(data, kind, cfg, with_frames=is_video)
        basis.update({"kind": "video" if is_video else "audio", **meta})
        if not frames and not transcript:
            return None, "media_unreadable"
        parts = [_image_part(f, quality) for f in frames]
        label = ("=== TRANSCRIPT OF THE RECORDING ===\n" + transcript) if transcript else \
            "(no speech could be transcribed; the frames are all there is)"
        if frames:
            label = (f"The {len(frames)} images are frames sampled evenly across the video, "
                     "in order.\n") + label
        parts.append(_text_part(label))
        scope = "the transcript and frames" if frames else "the transcript"
        return Source([parts], scope, scope, 0, basis), ""

    if kind in _OFFICE:
        pdf = _office_to_pdf(data, kind, int(cfg.get("office_convert_timeout", 180)))
        if pdf is None:
            return None, f"office_conversion_failed:{kind}"
        basis["kind"] = "office"
        basis["converted_from"] = kind
        data, kind = pdf, "pdf"

    if kind in _IMAGE_TYPES:
        from PIL import Image

        try:
            parts = [_image_part(Image.open(io.BytesIO(data)).convert("RGB"), quality)]
        except Exception:
            return None, "image_unreadable"
        total = 1
        basis["kind"] = "image"
    elif kind == "pdf":
        import pymupdf

        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
            total = doc.page_count
        except Exception:
            return None, "render_failed"
        limit = int(cfg.get("max_image_pages", 0))
        if limit and total > limit:
            # Too long to send as page images: the PDF's own text, in chunks of pages.
            try:
                groups, with_text = _pdf_text(doc, int(cfg.get("max_source_chars", 100000)))
            except Exception:
                return None, "pdf_text_failed"
            if not groups:
                return None, f"pdf_over_{limit}_pages_text_unusable"
            basis.update({"kind": f"{basis.get('kind', 'pdf')}_text", "pages_total": total,
                          "pages_sent": 0, "pages_with_text": with_text,
                          "text_chars": sum(len(t) for _, _, t in groups)})
            if len(groups) == 1:
                chunks = [[_text_part(f"=== THE DOCUMENT'S TEXT, EXTRACTED FROM ITS {total} "
                                      f"PAGES ===\n{groups[0][2]}")]]
            else:
                chunks = [[_text_part(f"=== TEXT OF PAGES {a}-{b} OF {total}, EXTRACTED "
                                      f"FROM THE PDF ===\n{text}")] for a, b, text in groups]
            return Source(chunks, "the document", "these pages", total, basis,
                          ranges=[(a, b) for a, b, _ in groups]), ""
        try:
            parts = _render_pdf(doc, int(cfg.get("render_dpi", 120)), quality)
        except Exception:
            return None, "render_failed"
        if not parts:
            return None, "render_failed"
        basis.setdefault("kind", "pdf")
    else:
        return None, f"unknown_type:{kind or _declared_ext(rec) or 'none'}"

    basis.update({"pages_total": total, "pages_sent": len(parts)})
    chunks = [parts[i:i + chunk_pages] for i in range(0, len(parts), chunk_pages)]
    return Source(chunks, "the document", "these pages", len(parts), basis), ""


def text_source(rec: Dict[str, Any], max_chars: int, after: str = "",
                fallback: str = "") -> Optional[Source]:
    """
    The export's text (the vision reading, then the extracted text) as the fallback.
    `after`: the file flow that failed first; `fallback`: why the file was not used.
    """
    text = reference_text(rec, max_chars, "file")
    if len(text.split()) < 50:
        return None
    basis = {"from": "export_text", "kind": "text", "words": len(text.split())}
    if after:
        basis["after"] = after
    if fallback:
        basis["fallback"] = fallback
    return Source([[_text_part("=== FILE TEXT ===\n" + text)]], "the text", "the text", 0, basis)


# ----------------------------------------------------------------- the summary
def _complete(client, content, max_tokens: int) -> Any:
    from core.llm import complete_json_robust

    return complete_json_robust(client, content, max_tokens, SYSTEM)


def _clean(value: Any) -> str:
    return " ".join(str(value).split()) if isinstance(value, str) and str(value).split() else ""


def _is_english(language: str) -> bool:
    return language.strip().lower().startswith("english")


def _language_name(named: Any, text: str) -> str:
    """The model's own naming of the document's language; detected from the text if absent."""
    name = _clean(named)[:40]
    if name:
        return name
    from core.corpus_index import _detect, language_name

    code = _detect(text[:1000])
    return language_name(code) if code and code != "unknown" else "unknown"


def estimate_calls(src: Source, cfg: Dict[str, Any]) -> int:
    """Upper estimate, for the count printed before any call is made."""
    n = len(src.chunks)
    fanin = max(2, int(cfg.get("merge_fanin", 8)))
    calls = 1 if n == 1 else n + sum(1 for _ in _merge_levels(n, fanin))
    if cfg.get("verify", True):
        calls += n + 1
    if cfg.get("translate_to_english", True):
        calls += 1
    return calls


def _merge_levels(n: int, fanin: int):
    """How many merge calls hierarchical merging needs for n note blocks."""
    while n > 1:
        if n <= fanin:
            yield 1
            return
        n = math.ceil(n / fanin)
        for _ in range(n):
            yield 1


def summarise(client, src: Source, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Draft (one call, or notes per chunk merged hierarchically), verify against every
    chunk and revise when there are findings, translate to English when the document is
    not English. Returns {"summary", "summary_en", "language", ...counts}.
    """
    max_tokens = int(cfg.get("max_tokens", 4000))
    chunk_pages = int(cfg.get("chunk_pages") or cfg.get("max_images_per_call", 12))
    fanin = max(2, int(cfg.get("merge_fanin", 8)))
    calls = 0

    def ask(parts: Sequence[Dict[str, Any]], brief: str) -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        raw = _complete(client, list(parts) + [_text_part(brief)], max_tokens)
        return raw if isinstance(raw, dict) else {}

    if len(src.chunks) == 1:
        raw = ask(src.chunks[0], DRAFT_BRIEF)
        draft = _clean(raw.get("summary"))
        if not draft:
            raise RuntimeError("no draft summary returned")
        language = _language_name(raw.get("language"), draft)
    else:
        notes: List[str] = []
        named: List[str] = []
        first = 1
        for i, chunk in enumerate(src.chunks):
            if src.ranges:
                first, last = src.ranges[i]
            else:
                last = first + len(chunk) - 1
            raw = ask(chunk, notes_brief(first, last, src.pages))
            first = last + 1
            text = _clean(raw.get("notes"))
            if text:
                notes.append(text)
                if _clean(raw.get("language")):
                    named.append(_clean(raw.get("language"))[:40])
        if not notes:
            raise RuntimeError("no notes returned for any chunk")
        language = Counter(named).most_common(1)[0][0] if named else _language_name("", notes[0])
        while True:
            if len(notes) <= fanin:
                raw = ask([], merge_brief(notes, language, final=True))
                draft = _clean(raw.get("summary"))
                break
            merged = []
            for g in range(0, len(notes), fanin):
                group = notes[g:g + fanin]
                raw = ask([], merge_brief(group, language, final=False))
                merged.append(_clean(raw.get("notes")) or "\n\n".join(group))
            notes = merged
        if not draft:
            raise RuntimeError("the merge returned no summary")

    errors: List[str] = []
    missing: List[str] = []
    checked: Dict[str, int] = {}
    revised = False
    if cfg.get("verify", True):
        if len(src.chunks) == 1:
            raw = ask(src.chunks[0], verify_brief(draft, src.scope))
            errors = [_clean(x) for x in (raw.get("errors") or []) if _clean(x)]
            missing = [_clean(x) for x in (raw.get("missing") or []) if _clean(x)]
        else:
            sentences = _sentences(draft)
            per_chunk = []
            for chunk in src.chunks:
                raw = ask(chunk, verify_sentences_brief(sentences, src.part_scope))
                per_chunk.append(_verdicts(raw, len(sentences)))
                missing += [_clean(x) for x in (raw.get("missing") or []) if _clean(x)]
            errors, unverified = combine_verdicts(sentences, per_chunk)
            checked = {"sentences": len(sentences), "unverified": unverified}
        if errors or missing:
            raw = ask([], revise_brief(draft, errors, missing, language))
            text = _clean(raw.get("summary"))
            if text:
                draft, revised = text, True

    summary_en, translated = draft, False
    if cfg.get("translate_to_english", True) and not _is_english(language):
        raw = ask([], translate_brief(draft, language))
        text = _clean(raw.get("summary_en"))
        if text:
            summary_en, translated = text, True

    return {"summary": draft, "summary_en": summary_en, "language": language,
            "calls": calls, "chunks": len(src.chunks), "chunk_pages": chunk_pages,
            "errors": len(errors), "missing": len(missing), "findings": errors + missing,
            "revised": revised, "translated": translated, **checked}


def generate(records: Sequence[Dict[str, Any]], spec: Dict[str, Any],
             workers: int = 8, log=print
             ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    A fresh summary per record in the document's own language, from the file where
    possible, else from the export's text. Returns {ko_id: {"summary", "summary_en",
    "language"}}, the run's counts, and {ko_id: basis}. A KO with no usable source gets
    no call and keeps the export's summary; so does a KO whose calls all fail.
    """
    from core.llm import get_client

    cfg = spec.get("resummarise", {})
    client = get_client(model=cfg.get("model", ""))
    if client is None:
        raise RuntimeError("no LLM available for --resummarise; see ../USING_LLMS.md")
    max_src = int(cfg.get("max_source_chars", 100000))

    # 1. The files: downloads, conversions, renderings and transcriptions, in parallel.
    log(f"[resummarise] fetching {len(records)} files from their @id URLs ...")
    by_id = {ko_id(r): r for r in records}
    fallback_reasons: Dict[str, int] = {}
    sources: Dict[str, Source] = {}
    skipped = 0

    def fetch(rec: Dict[str, Any]) -> Tuple[str, Optional[Source], str]:
        try:
            src, why = acquire(rec, cfg)
        except Exception as exc:        # a conversion or transcription that blew up
            src, why = None, f"{type(exc).__name__}"
        return ko_id(rec), src, why

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for kid, src, why in pool.map(fetch, records):
            rec = by_id[kid]
            if src is None:
                fallback_reasons[why] = fallback_reasons.get(why, 0) + 1
                src = text_source(rec, max_src, fallback=why)
            if src is None:
                skipped += 1
                continue
            sources[kid] = src

    kinds = Counter(s.basis.get("kind") for s in sources.values() if s.basis.get("from") == "file")
    n_file = sum(kinds.values())
    transcription_calls = sum(int(s.basis.get("transcription_calls") or 0) for s in sources.values())
    estimate = sum(estimate_calls(s, cfg) for s in sources.values())
    log(f"[resummarise] {len(sources)} KOs to summarise in the document's own language with "
        f"{client.model}: {n_file} from the file as it is ({dict(kinds)}), "
        f"{len(sources) - n_file} from the export's text"
        + (f" (fallbacks: {fallback_reasons})" if fallback_reasons else "")
        + (f"; {skipped} with neither keep the export's summary" if skipped else "")
        + f"; {transcription_calls} transcription calls made; about {estimate} summary calls "
          "at most (draft/notes, merges, verification, revision, translation); cached calls are free")

    # 2. The summaries, one KO at a time per worker; a failed file flow retries from text.
    def work(kid: str) -> Optional[Dict[str, Any]]:
        src = sources[kid]
        try:
            out = summarise(client, src, cfg)
            out["basis"] = src.basis
            return out
        except Exception as exc:
            if src.basis.get("from") != "file":
                raise
            alt = text_source(by_id[kid], max_src, after=f"file_flow_failed:{type(exc).__name__}")
            if alt is None:
                raise
            out = summarise(client, alt, cfg)
            out["basis"] = alt.basis
            return out

    ids = list(sources)
    results = client.map_concurrent(ids, work, workers=workers, label=f"resummarise:{client.model}")

    out: Dict[str, Dict[str, Any]] = {}
    basis: Dict[str, Dict[str, Any]] = {}
    calls = retried = 0
    for kid, res in zip(ids, results):
        if not res:
            continue
        out[kid] = {"summary": res["summary"], "summary_en": res["summary_en"],
                    "language": res["language"]}
        b = dict(res["basis"])
        b.update({"file_language": res["language"], "chunks": res["chunks"], "calls": res["calls"],
                  "errors": res["errors"], "missing": res["missing"], "revised": res["revised"],
                  "translated": res["translated"], "findings": res["findings"][:20]})
        for key in ("sentences", "unverified"):          # sentence-level check only
            if key in res:
                b[key] = res[key]
        basis[kid] = b
        calls += res["calls"]
        retried += b.get("from") == "export_text" and "after" in b
    failed = len(ids) - len(out)
    if failed:
        log(f"[resummarise] {failed} of {len(ids)} KOs got no usable summary; they keep the "
            "export's summary and are marked")
    from_file = sum(1 for b in basis.values() if b.get("from") == "file")
    info = {"model": client.model, "kos": len(ids), "calls": calls,
            "transcription_calls": transcription_calls,
            "from_file": from_file, "from_export_text": len(out) - from_file,
            "retried_from_text": retried, "kinds": dict(kinds),
            "fallback_reasons": fallback_reasons, "failed": failed, "skipped_no_source": skipped,
            "verified": bool(cfg.get("verify", True)),
            "revised": sum(1 for b in basis.values() if b.get("revised")),
            "translated": sum(1 for b in basis.values() if b.get("translated")),
            "languages": dict(Counter(r["language"] for r in out.values()))}
    return out, info, basis


__all__ = ["generate", "acquire", "text_source", "summarise", "estimate_calls",
           "DRAFT_BRIEF", "SYSTEM"]
