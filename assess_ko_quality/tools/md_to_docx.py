#!/usr/bin/env python3
# assess_ko_quality/tools/md_to_docx.py
"""
Render the Markdown method documents as a Word guide.

The previous guide was written separately from the code's documentation and fell out of
date as soon as the method changed. This builds the .docx from the Markdown files
themselves, so there is one source of truth: edit METHODS.md or LITERATURE_REVIEW.md,
rerun this, and the guide follows.

Handles what those files use: headings, paragraphs, bullet and numbered lists, tables,
fenced code blocks, and **bold**, *italic* and `code` inline.

    cd assess_ko_quality
    python -m tools.md_to_docx METHODS.md LITERATURE_REVIEW.md \
        --out How_the_KO_quality_pipeline_works.docx --title "How the KO quality pipeline works"
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

_INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)")


def add_inline(par, text: str) -> None:
    """Add text to a paragraph, honouring **bold**, *italic* and `code`."""
    for part in _INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            par.add_run(part[2:-2]).bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = par.add_run(part[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            par.add_run(part[1:-1]).italic = True
        else:
            par.add_run(part)


def _shade(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def add_table(doc, rows: List[List[str]]) -> None:
    header, body = rows[0], rows[1:]
    table = doc.add_table(rows=1 + len(body), cols=len(header))
    table.style = doc.styles["Table Grid"]
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate([header] + body):
        for j in range(len(header)):
            cell = table.cell(i, j)
            cell.text = ""
            par = cell.paragraphs[0]
            add_inline(par, row[j] if j < len(row) else "")
            for run in par.runs:
                run.font.size = Pt(8.5)
                if i == 0:
                    run.bold = True
            if i == 0:
                _shade(cell, "E7EEF7")
    doc.add_paragraph()


def _cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(doc, markdown: str) -> None:
    lines = markdown.splitlines()
    i = 0
    para: List[str] = []

    def flush() -> None:
        if para:
            add_inline(doc.add_paragraph(), " ".join(s.strip() for s in para))
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            p = doc.add_paragraph()
            run = p.add_run("\n".join(block))
            run.font.name = "Consolas"
            run.font.size = Pt(8)
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|?\s*:?-{2,}", lines[i + 1].strip()):
            flush()
            rows = [_cells(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            add_table(doc, rows)
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", s)
        if m:
            flush()
            level = len(m.group(1))
            h = doc.add_heading(level=min(level, 3))
            add_inline(h, m.group(2))
            i += 1
            continue
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
        if m:
            flush()
            style = "List Number" if m.group(2)[0].isdigit() else "List Bullet"
            if len(m.group(1)) >= 2:
                style += " 2"
            text = m.group(3)
            # continuation lines of the same item
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and \
                    not re.match(r"^\s*([-*]|\d+\.)\s+", lines[i + 1]) and lines[i + 1].strip():
                i += 1
                text += " " + lines[i].strip()
            try:
                add_inline(doc.add_paragraph(style=style), text)
            except KeyError:
                add_inline(doc.add_paragraph(style="List Bullet"), text)
            i += 1
            continue
        if s in ("---", "***"):
            flush()
            i += 1
            continue
        if not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    doc = Document()
    base = doc.styles["Normal"]
    base.font.name = "Calibri"
    base.font.size = Pt(10.5)
    if args.title:
        t = doc.add_heading(args.title, level=0)
        for run in t.runs:
            run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    for k, src in enumerate(args.sources):
        if k:
            doc.add_page_break()
        render(doc, src.read_text(encoding="utf-8"))
    doc.save(args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
