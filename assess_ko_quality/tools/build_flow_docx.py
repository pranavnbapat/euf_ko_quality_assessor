#!/usr/bin/env python3
# assess_ko_quality/tools/build_flow_docx.py
"""
Draw the pipeline as a diagram and put it, with a step-by-step explanation, in a Word
document.

    cd assess_ko_quality
    python -m tools.build_flow_docx            # -> How_the_KO_quality_pipeline_flows.docx
"""
from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent

LOCAL = "#DCE8F5"      # runs on our machine, no LLM
REPORT = "#ECECEC"     # reported, never scored
SCORE = "#D9EFD9"      # the score
OPTION = "#FFF2CC"     # optional step, LLM calls
EDGE = "#3A4A5C"


def draw(path: Path) -> None:
    """Boxes are sized from their number of lines and stacked top to bottom."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    ink = "#1F2D3D"
    LINE, TOP, PAD, GAP = 3.2, 5.6, 2.2, 5.2

    def height(rows):
        return TOP + len(rows) * LINE + PAD

    # (title, rows, colour, x, w); a row is text or a (left, right) pair
    select = ("SELECT   which KOs to assess",
              [("summary at least 50 words", "- 655"),
               ("file type = pdf", "- 1,608"),
               ("first language = English", "- 3,366"),
               ("licence not CC BY-NC-ND", "- 552"),
               ("eligible", "4,287"),
               "pick 10 at random (seed 42: the same 10 every time)"], LOCAL, 12, 76)
    gate = ("1  DOMAIN GATE   is it about agriculture?",
            ["classifier on every KO (local)",
             "GLM only if the classifier is unsure (P 0.20-0.70)"], LOCAL, 22, 56)
    resum = ("RESUMMARISE   optional (--resummarise)",
             ["the reference summary written fresh from the file itself, in the document's own",
              "language: pages as images, office files converted first, a video's sound",
              "transcribed and its frames shown; long files in parts, merged; every draft checked",
              "against its pages and revised; a non-English summary also translated to English",
              "for the search index. 2-4 calls per short document (qwen3.5 by default). Without",
              "the flag the export's own summary (ko_content_flat_summarised) is the reference"],
             OPTION, 6, 88)
    comp = ("2  COMPLIANCE", ["required fields present,", "current vocabulary values",
                              "-> reported, not scored"], REPORT, 2, 30)
    cont = ("3  CONTENT REPORT", ["published web-text filters", "(Gopher, C4, FineWeb,", "FineWeb2)",
                                  "-> reported, not scored"], REPORT, 35, 30)
    sumc = ("3b  SUMMARY CHECK", ["is the reference summary", "itself reliable? compression,",
                                  "key terms, NLI, language", "-> reported, not scored"], REPORT, 68, 30)
    score = ("4  METADATA SCORE",
             ["first the LLM version is written fresh: one blind call per KO from the reference",
              "summary and the published rules only (never the contributor's fields), in the",
              "document's own language (qwen3.5 by default; --use-existing-llm scores the",
              "export's *_llm fields instead); then, for each field - title . subtitle .",
              "description . keywords - and each version, contributor's and LLM's,",
              "checked against the REFERENCE SUMMARY (the export's, or the one written above):",
              "",
              ("STRUCTURAL", "length (EIP-AGRI, Google) . completeness . noise . formatting"),
              ("SEMANTIC", "consistency (NLI vs the summary + agreement between the fields)"),
              ("", "density (specific words) . clarity (LIX, acronyms) . usefulness (key terms)"),
              ("FUNCTIONAL", "findability: where this KO ranks when all 10,468 summaries"),
              ("", "are searched with the field (BM25 keyword search, E5 meaning search)"),
              ("LLM SCORE", "one call per field: summary + that field + the criteria + the"),
              ("", "KO's language in; one 0-100 score and a reason out (glm-5.2)"),
              "",
              "a lexical part that cannot read the field's language is NOT EVALUABLE,",
              "never zero; field score = mean of the 4 pillars; KO score = mean of the",
              "4 fields; an empty field scores 0 (no LLM call)"],
             SCORE, 6, 88)
    rec = ("RECOMMEND   per field   (no further LLM call)",
           ["the higher-scoring of the contributor's and the LLM version, provided",
            "it is not less faithful to the summary than the contributor's (measured);",
            "an empty contributor field -> the LLM version"], SCORE, 14, 72)
    out = ("5  ANALYSIS + OUTPUT",
           ["report.xlsx (Run . KOs . Fields . Summaries . Self-test)  +  details.jsonl,",
            "both with the summary each KO was measured against"], LOCAL, 14, 72)
    inp = ("KO export (JSON), June 2026", ["10,468 KOs: metadata, summary, file text"], LOCAL, 27, 46)

    stack = [inp, select, gate, resum, (comp, cont, sumc), score, rec, out]
    total = sum(max(height(b[1]) for b in (s if isinstance(s, tuple) and isinstance(s[0], tuple) else (s,)))
                for s in stack) + GAP * (len(stack) - 1) + 2

    fig, ax = plt.subplots(figsize=(8.3, total * 0.078))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, total)
    ax.axis("off")

    def draw_box(b, y_top, h):
        title, rows, color, x, w = b
        ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0.4,rounding_size=1.8",
                                    linewidth=1.1, edgecolor=EDGE, facecolor=color))
        ax.text(x + 1.8, y_top - 1.8, title, fontsize=10.2, fontweight="bold", va="top", color=ink)
        col2 = x + 16 if any(isinstance(r, tuple) and r[0] in ("STRUCTURAL", "SEMANTIC", "FUNCTIONAL") for r in rows) \
            else x + w - 8
        for k, row in enumerate(rows):
            yy = y_top - TOP - k * LINE
            if isinstance(row, tuple):
                ax.text(x + 1.8, yy, row[0], fontsize=8.1, va="top", color=ink)
                right = row[1].startswith("-") or row[1][:1].isdigit()
                ax.text(col2 if not right else x + w - 3, yy, row[1], fontsize=8.1, va="top", color=ink,
                        ha="right" if right else "left")
            else:
                ax.text(x + 1.8, yy, row, fontsize=8.1, va="top", color=ink)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.2, color=EDGE))
        if label:
            ax.text((x1 + x2) / 2 + 1.2, (y1 + y2) / 2, label, fontsize=7.8, va="center",
                    color=ink, style="italic")

    y = total - 1
    geom = {}
    for k, item in enumerate(stack):
        boxes = item if isinstance(item, tuple) and isinstance(item[0], tuple) else (item,)
        h = max(height(b[1]) for b in boxes)
        for b in boxes:
            draw_box(b, y, h)
            geom[b[0]] = (b[3], y - h, b[4], h)
        if k:
            prev = stack[k - 1]
            prev_boxes = prev if isinstance(prev, tuple) and isinstance(prev[0], tuple) else (prev,)
            label = {1: "", 2: "10 KOs", 3: "yes"}.get(k, "")
            for pb in prev_boxes:
                px, py, pw, _ = geom[pb[0]]
                tx = 50 if len(prev_boxes) == 1 else max(40.0, min(60.0, px + pw / 2))
                arrow(px + pw / 2 if len(prev_boxes) > 1 else 50, py, tx, y + 0.4, label if len(prev_boxes) == 1 else "")
        y -= h + GAP

    # gate: the rejected branch
    gx, gy, gw, gh = geom[gate[0]]
    arrow(gx + gw, gy + gh / 2, gx + gw + 9, gy + gh / 2)
    ax.text(gx + gw + 4.5, gy + gh / 2 + 1.8, "no", fontsize=8, ha="center", color=ink)
    ax.text(gx + gw + 10, gy + gh / 2, "stop:\nnot scored", fontsize=7.8, va="center", color=ink)

    # legend
    lx, ly = 2, total - 3
    for k, (c, t) in enumerate([(LOCAL, "local, no LLM"), (SCORE, "the score"),
                                (REPORT, "reported only"), (OPTION, "optional, LLM")]):
        ax.add_patch(FancyBboxPatch((lx, ly - k * 3.4), 2.2, 2.2, boxstyle="round,pad=0.1",
                                    facecolor=c, edgecolor=EDGE, linewidth=0.8))
        ax.text(lx + 3.2, ly - k * 3.4 + 1.1, t, fontsize=7.3, va="center", color=ink)

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build(out: Path, image: Path) -> None:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    t = doc.add_heading("How the KO quality pipeline flows", level=0)
    for r in t.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    doc.add_paragraph(
        "The pipeline scores the metadata of a Knowledge Object - title, subtitle, description "
        "and keywords - against a reference summary: the export's own summary "
        "(ko_content_flat_summarised) or, with --resummarise, one written fresh from the file "
        "itself in the document's own language. Every language is assessed by default, for "
        "both the contributor's version and an LLM version written fresh from the summary each "
        "run (blind, in the document's own language; --use-existing-llm scores the export's "
        "*_llm fields instead). Each field score averages four pillars: three measured on our "
        "machine without an LLM (structural, semantic, functional) and an LLM's 0-100 score of "
        "the field. The better version of each field is recommended, and the summary used is "
        "stored with every run. The worked example below is 10 English PDFs from the June 2026 "
        "export, excluding the CC BY-NC-ND licence:")
    p = doc.add_paragraph()
    r = p.add_run('python pipeline.py --limit 10 --file-type pdf --languages en --exclude-licenses "CC BY-NC-ND"')
    r.font.name = "Consolas"
    r.font.size = Pt(9)
    doc.add_paragraph("The LLM calls per KO: one to write the LLM version, then one judge call per "
                      "field that has text, per version - up to 9 in all, counted and printed before "
                      "any call. --no-llm-score leaves the judge out; --use-existing-llm skips the "
                      "generation. --resummarise adds 2-4 calls per short document, about two more "
                      "per further 12 pages, and one transcription call per 10 minutes of video or "
                      "audio.")
    doc.add_picture(str(image), width=Cm(16))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading("Step by step", level=1)
    steps = [
        ("Select", "Filters the export by summary length, file type, first language and licence, then "
                   "draws the requested number of KOs at random. The seed makes the draw repeatable.", "No"),
        ("1  Domain gate", "A classifier decides whether each KO is about agriculture in the broad "
                           "sense. Off-domain KOs stop here and are not scored.",
         "Only for the few KOs the classifier is unsure about"),
        ("Resummarise (optional)", "With --resummarise, writes each KO's reference summary fresh "
                                   "from the file itself, in the document's own language. The file "
                                   "is fetched from its @id URL and read as it is: pages as images, "
                                   "office files converted first, a video's sound transcribed and "
                                   "its frames shown. Long files are summarised in parts and merged, "
                                   "every draft is checked against its pages and revised, and a "
                                   "non-English summary is also translated to English for the search "
                                   "index. A file that cannot be fetched or read falls back to the "
                                   "export's text. Without the flag the export's own summary is the "
                                   "reference.",
         "Yes: 2-4 calls per short document, about two more per further 12 pages, one "
         "transcription call per 10 minutes of recording"),
        ("2  Compliance", "Checks that the fields every record must carry are present and that "
                          "controlled-vocabulary values are current. Reported, never scored.", "No"),
        ("3  Content report", "Runs the published web-text quality filters over the file's text. "
                              "The file is the author's work, so this is reported, never scored.", "No"),
        ("3b  Summary check", "Measures the reference summary itself against the file text - "
                              "compression, key-term coverage, NLI support, language - and flags "
                              "outliers within each file-language group. Reported, never scored.",
         "No"),
        ("4  Metadata score", "First writes the LLM version: one blind call per KO from the "
                              "reference summary and the published rules, in the document's own "
                              "language. Then scores each field of both versions against that "
                              "summary: 10 "
                              "sub-pillars combined into 3 measured pillars, plus an LLM's 0-100 "
                              "score of the field (one call per field). A lexical part that cannot "
                              "read the field's language is marked not evaluable, never zeroed. The "
                              "field score is the "
                              "mean of the 4 pillars, the KO score the mean of the 4 fields. An "
                              "empty field scores 0, without a call.",
         "Yes: 1 generation call per KO, plus 1 judge call per field with text, per version (up to 9 per KO)"),
        ("Recommend", "Per field, the higher-scoring of the contributor's and the LLM version, "
                      "provided it is not less faithful to the summary than the contributor's "
                      "(measured accuracy). Where the contributor left a field empty, the LLM "
                      "version is taken.", "No"),
        ("5  Analysis + output", "Explains the weakest sub-pillars of each KO, checks how much the "
                                 "ranking depends on the weights, and writes report.xlsx.", "No"),
    ]
    table = doc.add_table(rows=1, cols=3)
    table.style = doc.styles["Table Grid"]
    for i, h in enumerate(["Step", "What happens", "LLM?"]):
        table.rows[0].cells[i].text = h
        table.rows[0].cells[i].paragraphs[0].runs[0].bold = True
    for step, what, llm in steps:
        row = table.add_row().cells
        row[0].text, row[1].text, row[2].text = step, what, llm
    for row in table.rows:
        for cell, w in zip(row.cells, (Cm(3.2), Cm(10.3), Cm(3.0))):
            cell.width = w
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)

    doc.add_heading("What the judge receives", level=1)
    for line in [
        "One field of one version at a time - never several in one prompt, because multi-field "
        "answers were where the output broke.",
        "The reference summary - the export's, or the one --resummarise wrote - the same text "
        "the other pillars measure against. The file itself is not sent.",
        "The criteria of the three measured pillars in words, with the published length rules "
        "for title (EIP-AGRI Common Format) and description (Google Dataset guidelines).",
        "The KO's language, with the instruction to rate meaning across languages and not "
        "penalise the field for the language it is written in.",
        "It returns one 0-100 score, its own reading of each measured pillar's criteria "
        "(reported, never scored) and a one-sentence reason. The model, glm-5.2, wrote "
        "neither version: the generator is qwen3.5 by default, and the judge is kept out of "
        "the writer's model family.",
    ]:
        doc.add_paragraph(line, style="List Bullet")
    doc.add_paragraph(
        "LLM judges may favour LLM-written text, which here is the LLM version; that is why "
        "the LLM score is one pillar of four and the recommendation's faithfulness check uses "
        "the measured accuracy (METHODS.md §6).")

    doc.add_heading("What you get", level=1)
    for line in [
        "report.xlsx - Run: settings, selection, what the fields were measured against and "
        "where that summary came from, models, LLM calls, self-test and robustness; KOs: one "
        "row per KO with each version's score, the contributor's four pillars and the weakest "
        "parts with reasons; Fields: one row per KO, field and version with the 4 pillars, all "
        "10 sub-pillars, the LLM's reason and the recommendation; Summaries: the summary each "
        "KO was measured against, its language and basis, and with --resummarise its English "
        "version and the platform summary it replaced; Self-test when requested.",
        "details.jsonl - the raw archive with every piece of evidence, the summary included, "
        "for code.",
    ]:
        doc.add_paragraph(line, style="List Bullet")
    doc.add_paragraph("Methods and sources: METHODS.md and LITERATURE_REVIEW.md, and the Word guide "
                      "How_the_KO_quality_pipeline_works.docx built from them.")
    doc.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=BASE / "How_the_KO_quality_pipeline_flows.docx")
    args = ap.parse_args()
    image = args.out.with_suffix(".png")
    draw(image)
    build(args.out, image)
    print(f"-> {args.out}\n-> {image}")


if __name__ == "__main__":
    main()
