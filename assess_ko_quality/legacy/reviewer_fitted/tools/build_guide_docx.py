#!/usr/bin/env python3
"""Build How_the_KO_quality_pipeline_works.docx in the EU-FarmBook house style."""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

GREEN = "1F5C4D"
BODY = 10.5
TABLE_W = 9026

# Build on the existing guide so styles, fonts and page setup match the family of
# documents this one joins (How_to_read_the_field_audit.docx and its siblings).
TEMPLATE = ("/home/pranav/PyCharm/EU-FarmBook/ko_quality_assessor/"
            "which_fields_to_choose/How_to_read_the_field_audit.docx")
doc = Document(TEMPLATE)
for _el in list(doc.element.body):
    if _el.tag.split("}")[-1] != "sectPr":
        doc.element.body.remove(_el)
sec = doc.sections[0]

# This template names its built-in styles "Heading 1" where python-docx expects the
# internal "heading 1", so name lookup fails. Resolve to the style objects once.
STYLES = {st.name: st for st in doc.styles if st.name}


def _styled(text, name):
    p = doc.add_paragraph()
    if name in STYLES:
        p.style = STYLES[name]
    if text:
        p.add_run(text)
    return p


def _sz(el, pts):
    for tag in ("w:sz", "w:szCs"):
        e = OxmlElement(tag)
        e.set(qn("w:val"), str(int(pts * 2)))
        el.append(e)


def para(text="", size=BODY, bold=False, italic=False, after=6, before=0, color=None,
         align=None, style=None):
    p = _styled("", style) if style else doc.add_paragraph()
    if align is not None:
        p.alignment = align
    pf = p.paragraph_format
    pf.space_after = Pt(after)
    pf.space_before = Pt(before)
    if text:
        r = p.add_run(text)
        r.bold = bold
        r.italic = italic
        r.font.size = Pt(size)
        if color:
            r.font.color.rgb = RGBColor.from_string(color)
    return p


def h1(text):
    p = _styled(text, "Heading 1")
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    for r in p.runs:
        r.font.size = Pt(15)
        r.bold = True
        r.font.color.rgb = RGBColor.from_string(GREEN)
    return p


def h2(text):
    p = _styled(text, "Heading 2")
    p.paragraph_format.space_before = Pt(11)
    p.paragraph_format.space_after = Pt(4)
    for r in p.runs:
        r.font.size = Pt(12)
        r.bold = True
        r.font.color.rgb = RGBColor.from_string(GREEN)
    return p


def bullets(items, size=BODY):
    for it in items:
        p = _styled("", "List Paragraph")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Pt(18)
        r = p.add_run("•   " + it)
        r.font.size = Pt(size)


# CT_TblPrBase fixes the order of its children. python-docx writes tblLayout and
# tblLook first, so everything added afterwards has to be sorted back into sequence
# or the file fails schema validation (Word still opens it; validators do not).
_TBLPR_ORDER = ("tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize",
                "tblStyleColBandSize", "tblW", "tblJc", "tblCellSpacing", "tblInd",
                "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook",
                "tblCaption", "tblDescription")


def _order_tblPr(tbl):
    tblPr = tbl._tbl.tblPr
    kids = list(tblPr)
    kids.sort(key=lambda el: _TBLPR_ORDER.index(el.tag.split("}")[-1])
              if el.tag.split("}")[-1] in _TBLPR_ORDER else len(_TBLPR_ORDER))
    for el in kids:
        tblPr.append(el)


def _grid(tbl, widths):
    """Set the table grid. Per-cell widths alone are ignored by some renderers."""
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn("w:tblW")):
        tblPr.remove(old)
    w = OxmlElement("w:tblW")
    w.set(qn("w:type"), "dxa")
    w.set(qn("w:w"), str(sum(widths)))
    # The schema fixes child order: tblW must precede tblBorders.
    borders = tblPr.find(qn("w:tblBorders"))
    if borders is not None:
        borders.addprevious(w)
    else:
        tblPr.append(w)
    for old in tbl._tbl.findall(qn("w:tblGrid")):
        tbl._tbl.remove(old)
    grid = OxmlElement("w:tblGrid")
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    tbl._tbl.insert(1, grid)


def _repeat_header(tbl):
    """Repeat the header row when a table breaks across pages."""
    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    h = OxmlElement("w:tblHeader")
    h.set(qn("w:val"), "true")
    trPr.append(h)


def _borders(tbl, spec):
    tblPr = tbl._tbl.tblPr
    for old in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(old)
    b = OxmlElement("w:tblBorders")
    for edge, val, color, sz in spec:
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), val)
        if val != "none":
            e.set(qn("w:color"), color)
            e.set(qn("w:sz"), str(sz))
        b.append(e)
    tblPr.append(b)


def _cell(cell, text, width, bold=False, white=False, size=9.5, shade=None):
    tcPr = cell._tc.get_or_add_tcPr()
    # python-docx has already written a tcW; replace rather than duplicate, and keep
    # the schema's child order (tcW, shd, tcMar).
    for tag in ("w:tcW", "w:shd", "w:tcMar"):
        for old in tcPr.findall(qn(tag)):
            tcPr.remove(old)
    w = OxmlElement("w:tcW")
    w.set(qn("w:type"), "dxa")
    w.set(qn("w:w"), str(width))
    tcPr.append(w)
    if shade:
        sh = OxmlElement("w:shd")
        sh.set(qn("w:fill"), shade)
        sh.set(qn("w:val"), "clear")
        tcPr.append(sh)
    mar = OxmlElement("w:tcMar")
    for side, v in (("top", 90), ("left", 130), ("bottom", 90), ("right", 130)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:type"), "dxa")
        m.set(qn("w:w"), str(v))
        mar.append(m)
    tcPr.append(mar)

    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(size)
    if white:
        r.font.color.rgb = RGBColor.from_string("FFFFFF")


def table(headers, rows, widths):
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.autofit = False
    _borders(t, [("top", "single", "C9D2CF", 2), ("left", "none", "", 0),
                 ("bottom", "single", "C9D2CF", 2), ("right", "none", "", 0),
                 ("insideH", "single", "E3E8E6", 2), ("insideV", "none", "", 0)])
    _grid(t, widths)
    _repeat_header(t)
    for j, htext in enumerate(headers):
        _cell(t.rows[0].cells[j], htext, widths[j], bold=True, white=True, shade=GREEN)
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            _cell(t.rows[i].cells[j], str(val), widths[j])
    _order_tblPr(t)
    para("", size=1, after=6)
    return t


def meta_table(pairs):
    t = doc.add_table(rows=len(pairs) + 1, cols=2)
    t.autofit = False
    _borders(t, [("top", "single", "C9D2CF", 2), ("left", "none", "", 0),
                 ("bottom", "single", "C9D2CF", 2), ("right", "none", "", 0),
                 ("insideH", "single", "E3E8E6", 2), ("insideV", "none", "", 0)])
    _grid(t, [2600, 6426])
    _cell(t.rows[0].cells[0], "About this guide", 2600, bold=True, white=True, shade=GREEN)
    _cell(t.rows[0].cells[1], "", 6426, shade=GREEN)
    for i, (k, v) in enumerate(pairs, start=1):
        _cell(t.rows[i].cells[0], k, 2600, bold=True)
        _cell(t.rows[i].cells[1], v, 6426)
    _order_tblPr(t)
    para("", size=1, after=6)
    return t


def callout(title, body):
    t = doc.add_table(rows=1, cols=1)
    t.autofit = False
    _borders(t, [("top", "single", "D8DEDC", 2), ("left", "single", GREEN, 18),
                 ("bottom", "single", "D8DEDC", 2), ("right", "single", "D8DEDC", 2),
                 ("insideH", "none", "", 0), ("insideV", "none", "", 0)])
    _grid(t, [TABLE_W])
    cell = t.rows[0].cells[0]
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("w:tcW", "w:tcMar"):
        for old in tcPr.findall(qn(tag)):
            tcPr.remove(old)
    w = OxmlElement("w:tcW"); w.set(qn("w:type"), "dxa"); w.set(qn("w:w"), str(TABLE_W))
    tcPr.append(w)
    mar = OxmlElement("w:tcMar")
    for side, v in (("top", 140), ("left", 200), ("bottom", 140), ("right", 200)):
        m = OxmlElement(f"w:{side}"); m.set(qn("w:type"), "dxa"); m.set(qn("w:w"), str(v))
        mar.append(m)
    tcPr.append(mar)
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(title); r.bold = True; r.font.size = Pt(BODY)
    r.font.color.rgb = RGBColor.from_string(GREEN)
    p2 = cell.add_paragraph()
    p2.paragraph_format.space_after = Pt(0)
    r2 = p2.add_run(body); r2.font.size = Pt(BODY)
    _order_tblPr(t)
    para("", size=1, after=6)
    return t


# ============================================================ title block
para("EU-FarmBook", size=10.5, bold=True, after=2, color=GREEN)
para("How the KO Quality Pipeline Works", size=25, bold=True, after=2)
para("A plain-language guide to assess_ko_quality", size=12, after=10)

meta_table([
    ("Explains", "assess_ko_quality/pipeline.py — the four stages that assess a "
                 "knowledge object"),
    ("Produced by", "ko_quality_assessor/assess_ko_quality"),
    ("Data analysed", "10,468 knowledge objects (June 2026 MySQL export)"),
    ("Validated against", "The 2024 KO review: 101 knowledge objects, 199 reviews, "
                          "12 reviewer organisations"),
    ("Audience", "Anyone who needs to act on a KO quality score — no statistics "
                 "background assumed"),
    ("Written", "21 September 2026"),
])

# ============================================================ 1
h1("1. What question does this pipeline answer?")
para("Every knowledge object uploaded to EU-FarmBook is a document, video or dataset with "
     "metadata attached: a title, a description, keywords, a licence, the project it came "
     "from. Some are excellent. Some have a title that does not describe the content, or a "
     "file whose text never extracted, or no licence at all.")
para("This pipeline answers three separate questions about each one, and it keeps them "
     "separate on purpose:")
bullets([
    "Does this belong on the platform at all?",
    "Does the record meet the obligations we place on contributors?",
    "How good is it, compared with what reviewers actually value?",
])
callout("The single most important thing to understand",
        "These three questions have different kinds of answers, so they must not be "
        "combined into one number. Relevance is a yes-or-no precondition. Compliance is a "
        "checklist of obligations. Quality is a matter of degree. The previous version of "
        "this tool averaged all three into a single 0–100 score, and section 2 shows "
        "what that produced.")

# ============================================================ 2
h1("2. Why the previous version had to be rebuilt")
para("The earlier assessor scored four 'pillars' — Structural, Semantic, Functional and "
     "Domain — and combined them with fixed weights of 30, 35, 25 and 10 percent. It was "
     "never checked against human judgement. When it finally was, three things came out.")

h2("It ranked off-topic documents above real farm advice")
para("Three documents with no connection to agriculture were written to be well structured "
     "and well described, then scored by the old assessor alongside genuine knowledge "
     "objects. Every one of them beat every real KO in the run — with the Domain pillar "
     "switched on.")
table(["Document", "Structural", "Semantic", "Functional", "Domain", "Total"],
      [["Bach's Brandenburg Concertos", "24", "15", "20", "15", "75.8"],
       ["Asynchronous server handlers", "24", "16", "18", "15", "75.2"],
       ["Atrial fibrillation treatment", "24", "16", "18", "15", "75.2"],
       ["Real agricultural KO (best of run)", "21", "11", "18", "19", "66.2"],
       ["Real agricultural KO (worst of run)", "20", "8", "8", "19", "50.8"]],
      [3000, 1200, 1200, 1250, 1100, 1276])
para("The Domain pillar could not have stopped this. Asked how agricultural a text is, it "
     "gave a crop-rotation guide 0.071 and literal gibberish 0.069. It compared each "
     "document against the average of 96,691 agricultural thesaurus concepts, and the "
     "average of that many different ideas points nowhere in particular.")
para("Even a perfect relevance detector would not have stopped it either. At 10 percent "
     "weight, the most an irrelevant upload can lose is 10 points out of 100.")

h2("The score barely tracked what reviewers thought")
para("Compared against the 2024 human review, the old total agreed with reviewers at a rank "
     "correlation of 0.174 — not statistically significant. A single measurement, how "
     "much text the document contains, did nearly three times better at 0.484.")

h2("It was not the weights")
para("Given complete freedom to choose its own weights for the four pillars, a statistical "
     "model scored −0.170 out of sample: worse than guessing, and pointing the wrong "
     "way. If the best possible weighting of four numbers cannot beat chance, the problem "
     "is the four numbers, not the percentages applied to them.")
callout("What that means in practice",
        "Individual measurements inside the old score were actively misleading. Metadata "
        "completeness correlated with reviewer judgement at −0.217 — more complete "
        "records were rated slightly worse. Lexical variety, as it was measured, was "
        "inverted: 93.7 percent of the whole corpus scored full marks, and documents whose "
        "text had failed to extract scored best of all.")

# ============================================================ 3
h1("3. The four stages")
table(["Stage", "Question", "Output", "In the score?"],
      [["0. Gate", "Is this in scope for EU-FarmBook?",
        "accept / review / reject", "No — decides whether scoring happens"],
       ["1. Compliance", "Does the record meet its obligations?",
        "Pass or fail per obligation", "No — reported as a checklist"],
       ["2. Score", "How good is it?",
        "0–100", "Yes — this is the score"],
       ["3. Diagnostics", "What is unusual or actionable here?",
        "Flags and written notes", "No — explanatory only"]],
      [1500, 2900, 2200, 2426])
para("A knowledge object the gate rejects is never scored. Scoring an off-topic document "
     "invites exactly the comparison the gate exists to prevent.")

# ============================================================ 4
h1("4. Stage 0 — Is it in scope?")
para("The gate decides whether a knowledge object belongs on the platform before anything "
     "else happens. EU-FarmBook's own topic vocabulary defines the scope, and it is broader "
     "than 'farming': Forestry, Livestock, Crop farming, Economics, Environment and Society. "
     "Thirteen and a half percent of the corpus carries only the last three.")

h2("How it decides")
para("The 10,468 knowledge objects already on the platform define the domain better than any "
     "thesaurus does — they are the domain. The gate compares a new object against them "
     "rather than against a word list.")
table(["Method", "How well it separates in-scope from off-scope"],
      [["Average similarity to thesaurus concepts (the old Domain pillar)", "0.647"],
       ["Similarity to the closest 10 thesaurus concepts", "0.768"],
       ["Learned comparison against real knowledge objects (current)", "1.000"]],
      [5600, 3426])
para("A score of 1.000 means perfect separation on the validation set; 0.500 would mean a "
     "coin flip. The validation set holds 678 items: 400 real knowledge objects across 14 "
     "languages, 130 clearly off-scope documents, 125 borderline cases, and 23 in-scope "
     "articles from an outside source used to check that the gate learned the subject "
     "matter rather than learning to recognise where text came from.")

h2("Three outcomes, not two")
table(["Decision", "Threshold", "What happens"],
      [["Accept", "0.40 and above", "Scored normally"],
       ["Review", "0.20 to 0.40", "Sent to a person"],
       ["Reject", "Below 0.20", "Not scored"]],
      [1800, 2200, 5026])
para("At the accept threshold the gate keeps 100 percent of genuine knowledge objects in "
     "every one of the 14 languages tested, and catches 94.6 percent of clearly off-scope "
     "documents.")
callout("Why the threshold is not set higher",
        "Raising it to 0.70 would catch 100 percent of off-scope documents instead of 94.6. "
        "It would also start rejecting real knowledge objects for the language they are "
        "written in: German retention falls to 83 percent and Greek to 33 percent. Roughly a "
        "third of EU-FarmBook is not in English, so that trade is not worth making. Turning "
        "away a legitimate Greek contribution is a worse failure than letting an off-topic "
        "document through to human review.")
para("Two further checks matter. Knowledge objects whose text failed to extract are still "
     "accepted — a broken PDF is an ingestion problem, not a relevance problem, and the "
     "gate must not quietly double as a quality filter. And objects tagged only Economics, "
     "Society or Environment are accepted at the same rate as the rest.")

# ============================================================ 5
h1("5. Stage 1 — Does the record meet its obligations?")
para("Compliance is a checklist, not a score. Eighteen obligations are checked, covering "
     "core description (title, description, keywords), provenance (creators, project, "
     "identifiers), rights (licence), coverage (languages, locations, topics, themes) and "
     "timeliness.")
para("The list was not written by anyone. It is derived from the corpus: a field that "
     "virtually every record already carries is evidently required in practice. Because "
     "that test alone cannot tell an obligation from something one of our own pipelines "
     "produced, each field is then classified as supplied by the contributor, generated by "
     "the platform, or computed by a downstream process. Only contributor fields become "
     "obligations.")
callout("Why compliance is deliberately kept out of the score",
        "When metadata completeness was part of the old score it correlated with reviewer "
        "judgement at −0.217 — statistically significant, and in the wrong "
        "direction. A thorough record is not the same thing as good content. Completeness "
        "still matters, as an obligation contributors owe, which is why it is reported "
        "separately and in full rather than blended into a number where it did harm.")

# ============================================================ 6
h1("6. Stage 2 — The quality score")
para("The score runs from 0 to 100 and is built from exactly three inputs, each of which had "
     "to prove itself against the 2024 human review before it was allowed in.")
table(["Input", "What it captures", "Why it is there"],
      [["Content depth", "How much substantive text the object contains",
        "Reviewers consistently preferred longer, more developed documents"],
       ["Lexical variety (MTLD)", "How varied the vocabulary is, independent of length",
        "Distinguishes a developed document from a repetitive one"],
       ["Reviewer verdict from an AI model",
        "An overall judgement against the same review form humans used",
        "Agrees with reviewers as closely as reviewers agree with each other"]],
      [2000, 3300, 3726])

h2("How the inputs were chosen")
para("Sixteen measurements from the old design were tested. Eleven had no relationship with "
     "reviewer judgement at all. One ran backwards. The rest were dominated by the two kept "
     "above. Two specific corrections are worth recording, because they are easy to "
     "reintroduce by accident:")
table(["What was wrong", "The effect", "What replaced it"],
      [["Vocabulary variety measured as a type-token ratio",
        "Falls automatically as documents get longer, so 93.7% of the corpus scored full "
        "marks and failed extractions scored best",
        "MTLD, which does not vary with length: agreement with reviewers moved from "
        "−0.29 to +0.47"],
       ["Content length sorted into bands, with a penalty above 6,000 words",
        "Destroyed 53% of the signal in the strongest single measurement available, and "
        "penalised exactly the long documents reviewers preferred",
        "The measurement used directly, with no bands"]],
      [2500, 3400, 3126])

h2("The AI reviewer")
para("An AI model answers the same twelve questions the human reviewers answered, read "
     "directly from the 2024 review form itself. Four different model families were tested. "
     "The best, glm-5.2, agrees with the reviewer consensus at 0.612 — higher than two "
     "human reviewers agree with each other, which is 0.558.")
callout("One AI reviewer, not several",
        "Using more than one model makes the score worse, not better: one gives 0.638, two "
        "0.620, all four 0.600. The families agree with each other at 0.61 to 0.71, so they "
        "largely share their opinions; adding more of them costs complexity without adding "
        "information. The score also works without any AI model at all, using the other two "
        "inputs, at a lower but still useful level of agreement.")

# ============================================================ 7
h1("7. Stage 3 — Diagnostics")
para("Everything measured but not scored is reported here. That is most of what the old "
     "assessor computed, and the wording is deliberately careful about it.")
para("Measured against reviewer judgement, noise scored −0.148, clarity −0.137, "
     "usefulness −0.048 and metadata consistency −0.009. None of them predicts "
     "whether a reviewer rates a knowledge object well. Calling a document 'noisy' on that "
     "evidence would assert something we cannot support — the old noise measure was "
     "largely a length measure in disguise.")
para("So diagnostics report that a value is unusual for this corpus, not that it is wrong, "
     "with the corpus median alongside it. Each report carries the number of flags expected "
     "by chance, so a count can be read properly: with 57 measurements and a 1-in-100 tail "
     "at each end, one or two flags per object are expected and mean nothing.")
para("Two things are reported as genuine problems, because they are not opinions about "
     "quality: text that failed to extract, and failed compliance obligations.")
para("Where an AI reviewer has run, its written justifications supply the plain-language "
     "explanation — for instance that a document is a single unstructured block of "
     "text, or that technical terms are used without explanation. The statistics say what "
     "is unusual; the AI reviewer says what a reader would notice.")

# ============================================================ 8
h1("8. How good is the score, honestly?")
para("Two reviewers looking at the same knowledge object agree with each other at 0.558. "
     "That sets a ceiling: no automatic scorer can agree with 'the reviewers' more closely "
     "than the reviewers agree among themselves. The ceiling works out at 0.847.")
table(["Scorer", "Agreement with reviewers", "Share of what is reachable"],
      [["Old four-pillar total", "0.174", "21%"],
       ["Old four pillars, best possible weighting", "−0.170", "below chance"],
       ["Content depth + lexical variety", "0.535", "63%"],
       ["The current score, with AI reviewer", "0.610", "72%"],
       ["A human reviewer against another human reviewer", "0.558", "—"]],
      [3900, 2600, 2526])
callout("What the score is, and is not",
        "It is a calibrated estimate of how a reviewer would rate this knowledge object, "
        "reaching about 72 percent of what is achievable against a noisy human standard. It "
        "is not a measurement of truth, accuracy or usefulness in the field. A score of 70 "
        "does not mean the advice is correct; it means reviewers would probably rate the "
        "object well.")
para("The fitted model was trained on 91 knowledge objects — the overlap between the "
     "2024 review and the current export. That is a small sample, and the three inputs were "
     "chosen after examining that same sample, so the score should be treated as "
     "provisional until it holds up on a fresh batch of human review. Cross-validation "
     "protects the weights; it does not protect the choice of inputs.")
para("The AI reviewer's verdict carries most of the weight in the fitted model — about "
     "0.59 of the total against 0.18 for content depth and 0.16 for lexical variety. That "
     "is what the evidence supports, but it means the number is substantially an AI "
     "judgement anchored by two text statistics, rather than a statistic in its own right.")

# ============================================================ 9
h1("9. What this pipeline does not measure")
bullets([
    "Factual accuracy. Nothing here checks whether the advice in a knowledge object is "
    "correct. The AI reviewer checks whether the metadata matches the content, which is a "
    "different question.",
    "Retrieval performance. A retrieval-readiness figure is reported but explicitly marked "
    "unvalidated: it has never been checked against whether knowledge objects are actually "
    "found in search. Doing that means joining against the search logs.",
    "The five other review dimensions. Findability, clarity, comprehensibility, usability "
    "and credibility were all collected in 2024, but reviewers agreed with each other on "
    "them at only 0.14 to 0.34. A target that unreliable cannot support a validated score, "
    "so only the overall recommendation is used.",
])

# ============================================================ 10
h1("10. Using it")
para("Run everything from inside the assess_ko_quality folder.", size=BODY)
table(["Command", "What it does"],
      [["python pipeline.py --input input/kos.json", "Assess a batch"],
       ["python pipeline.py --input ... --judge", "Add the AI reviewer"],
       ["python pipeline.py --input ... --learn", "Refit the corpus profiles first"],
       ["python pipeline.py --input ... --no-gate", "Score everything, including off-scope"]],
      [4300, 4726])
para("Output is one row per knowledge object, written as both JSONL and TSV. Every figure "
     "states whether it has been validated: quality_score_is_validated is true, "
     "retrieval_readiness_is_validated is false. The score's contributions are itemised per "
     "input, so any individual score can be explained.")

# ============================================================ 11
h1("11. Keeping it calibrated")
para("Four things are learned from data rather than written down, and all four should be "
     "refreshed when the corpus changes materially.")
table(["What", "Where it comes from", "How to refresh"],
      [["Contributor obligations", "Corpus coverage, plus field classification",
        "pipeline.py --learn"],
       ["Diagnostic thresholds", "Per-measurement tails across the corpus",
        "pipeline.py --learn"],
       ["Score weights", "Fitted against human review",
        "validation/validate_metrics.py"],
       ["Domain gate", "Real knowledge objects versus off-scope documents",
        "stages/gate.py --calibrate"]],
      [2400, 3600, 3026])
para("Nothing in this pipeline contains a hand-written list. The controlled vocabularies are "
     "discovered from the data-model folder; the obligations come from the corpus; the "
     "diagnostic thresholds are corpus percentiles; the AI reviewer's questions are read "
     "from the human review workbook; the AI model is chosen from whatever the provider "
     "currently serves. Add a question to the review form and the AI reviewer asks it.")
callout("The rule the pipeline is built on",
        "Nothing enters the score until it has been measured against human judgement. That "
        "is not caution for its own sake — it is the specific failure that produced a "
        "score rating Bach above farm advice, a completeness measure pointing the wrong way, "
        "and a vocabulary measure that rewarded broken files. A better instrument is not an "
        "exemption: the AI reviewer was measured before it was used, and three of the four "
        "models tested were rejected on the evidence.")

# ============================================================ 12
h1("12. Decisions and the evidence behind them")
table(["Decision", "Evidence"],
      [["Relevance is a gate, not a scored pillar",
        "Three off-topic documents scored 75.2–75.8, above every real KO, with the "
        "Domain pillar active. At 10% weight the most an off-topic upload can lose is 10 "
        "points."],
       ["Compliance is a checklist, not a scored pillar",
        "Completeness correlated with reviewer judgement at −0.217 (significant, wrong "
        "direction)."],
       ["The four pillars were replaced rather than reweighted",
        "Optimal weighting of the four pillars scored −0.170 out of sample, below "
        "chance."],
       ["MTLD replaced the type-token ratio",
        "TTR falls with length (−0.93); 93.7% of the corpus scored full marks. Against "
        "reviewers: −0.29 for TTR, +0.47 for MTLD."],
       ["Content length is used directly, not in bands",
        "Banding lost 53% of the signal and penalised the long documents reviewers "
        "preferred."],
       ["One AI reviewer, not a panel",
        "One 0.638, two 0.620, four 0.600 — the families correlate 0.61–0.71 and "
        "share their opinions."],
       ["Gate threshold held at 0.40",
        "Above 0.60 a language gap opens: German retention 83%, Greek 33% at 0.70."],
       ["Diagnostics describe, they do not accuse",
        "Noise, clarity, usefulness and consistency all scored between −0.15 and "
        "−0.01 against reviewer judgement."]],
      [3200, 5826])

# ============================================================ 13
h1("13. Sources")
h2("Methodological background")
bullets([
    "Wang, R. & Strong, D. (1996). Beyond Accuracy: What Data Quality Means to Data "
    "Consumers. The 'fitness for use' framing, and the intrinsic / contextual / "
    "representational / accessibility dimensions. "
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9912223/",
    "Bruce, T. & Hillmann, D. (2004). The Continuum of Metadata Quality. The seven "
    "characteristics — completeness, accuracy, provenance, conformance to "
    "expectations, logical consistency, timeliness, accessibility — that stages 0 and "
    "1 are organised around. "
    "https://www.researchgate.net/publication/254358241_Quality_Metrics_for_Learning_Object_Metadata",
    "Ochoa, X. & Duval, E. (2009). Automatic evaluation of metadata quality in digital "
    "repositories. International Journal on Digital Libraries. The closest precedent to "
    "this work, and the source of the validation design used here: correlate with manual "
    "review, test discriminatory power, test usefulness as a low-quality filter. "
    "https://link.springer.com/article/10.1007/s00799-009-0054-4",
    "McCarthy, P. & Jarvis, S. (2010). MTLD, vocd-D, and HD-D. Why the type-token ratio "
    "cannot be compared across documents of different lengths, and what to use instead. "
    "https://en.wikipedia.org/wiki/Lexical_diversity",
    "FAIRsFAIR / F-UJI automated FAIR assessment. The model for treating metadata "
    "obligations as an explicit checklist rather than a score. "
    "https://www.fairsfair.eu/f-uji-automated-fair-data-assessment-tool",
])
h2("Internal references")
bullets([
    "ko_quality_assessor/USING_LLMS.md — provider, models, credentials and the three "
    "quirks that otherwise cost an afternoon.",
    "assess_ko_quality/README.md — the technical companion to this guide.",
    "assess_ko_quality/validation/ — the gate validation set, the fitted score model, "
    "and the learned corpus profiles.",
    "which_fields_to_choose/How_to_read_the_field_audit.docx and "
    "semantic_vs_search/How_to_read_the_search_comparison.docx — companion guides for "
    "the neighbouring analyses.",
])
para("")
para("Regenerate the figures in this guide at any time with python -m "
     "validation.validate_metrics from inside assess_ko_quality. Every number here comes "
     "from that report or from stages/gate.py --calibrate.", size=9.5, italic=True)

import sys
out = sys.argv[1] if len(sys.argv) > 1 else "How_the_KO_quality_pipeline_works.docx"
doc.save(out)
print("wrote", out)
