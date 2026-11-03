# assess_ko_quality/stages/report.py
"""
The run's one file to read: report.xlsx.

    Run        settings, selection, reference text, models, LLM calls, self-test and
               robustness results
    KOs        one row per KO: gate, compliance, content flags, the metadata score of each
               version, the contributor's four pillars and field scores, the recommended
               version per field, and the weakest parts with the reason for each
    Fields     one row per KO x field x version (the contributor's and the LLM one -
               written fresh from the summary, or the export's *_llm with
               --use-existing-llm): the text, the field score, the four pillars
               (structural, semantic, functional, LLM), the ten sub-pillars,
               faithfulness to the reference text, whether it is the recommended
               version, and the LLM's reason for its score
    Self-test  one row per deliberate defect (only with --selftest)

details.jsonl is written next to it as the raw archive, with every piece of evidence,
for code rather than for reading.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from core.corpus_index import FIELDS
from stages.analysis import SUBS
from stages.metadata_score import PILLARS

# "llm" is ambiguous on its own - it is the generated fields in a default run and the
# export's with --use-existing-llm - so write() resolves it per run (and per KO, since
# a failed generation falls back to the export's fields for that KO).
VERSION_LABEL = {"original": "contributor", "existing_llm": "existing *_llm",
                 "generated_llm": "generated *_llm"}
PILLAR_LABEL = {"llm": "LLM score"}
REFERENCE_LABEL = {"summary": "summary (ko_content_flat_summarised)",
                   "file": "file (vision reading + extracted text)"}


def _label(v: str) -> str:
    return VERSION_LABEL.get(v, v)


def _pillar_label(p: str) -> str:
    return PILLAR_LABEL.get(p, p)


def _cell(v: Any) -> Any:
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    return json.dumps(v, ensure_ascii=False, default=str)


def _sheet(wb, title: str, header: List[str], rows: List[List[Any]], widths: Dict[int, int] = None):
    from openpyxl.styles import Alignment, Font, PatternFill

    ws = wb.create_sheet(title)
    ws.append(header)
    for r in rows:
        ws.append([_cell(x) for x in r])
    bold = Font(bold=True)
    fill = PatternFill("solid", fgColor="E7EEF7")
    for c in ws[1]:
        c.font = bold
        c.fill = fill
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    for col in range(1, len(header) + 1):
        letter = ws.cell(row=1, column=col).column_letter
        ws.column_dimensions[letter].width = (widths or {}).get(col, 14)
    return ws


def write(out_dir: Path, results: List[Dict[str, Any]], analysis_out: Dict[str, Any],
          selftest: List[Dict[str, Any]], run_info: Dict[str, Any]) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    wb = Workbook()
    wb.remove(wb.active)
    ref = run_info.get("reference_source") or "file"
    llm_info = run_info.get("llm_score") or {}
    gen_info = run_info.get("generate_metadata") or {}
    llm_label = "generated *_llm" if gen_info.get("model") else "existing *_llm"
    has_llm = any((((r.get("metadata") or {}).get("variants")) or {}).get("llm") for r in results)

    # ---------------------------------------------------------------- Run
    from stages.analysis import file_language_of

    scored_rows = [r for r in results if r.get("metadata")]
    by_lang: Dict[str, List[Dict[str, Any]]] = {}
    for r in scored_rows:
        by_lang.setdefault(file_language_of(r), []).append(r)

    def _lang_means(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        def mean_of(variant: str) -> Any:
            xs = [((r["metadata"].get("variants") or {}).get(variant) or {}).get("score") for r in rs]
            xs = [x for x in xs if x is not None]
            return round(sum(xs) / len(xs), 1) if xs else None
        return {"n": len(rs), "contributor": mean_of("original"), "llm": mean_of("llm")}

    lang_line = ({lang: _lang_means(rs)
                  for lang, rs in sorted(by_lang.items(), key=lambda kv: -len(kv[1]))}
                 if len(by_lang) > 1 else "one language in this run")

    ws = wb.create_sheet("Run")
    sel = run_info.get("selection") or {}
    st_total = len(selftest)
    st_ok = sum(1 for s in selftest if s.get("outcome") == "dropped")
    rb = (analysis_out.get("robustness_pillars") or {}).get("rank_change") or {}
    lines = [
        ("KO quality run", ""),
        ("export", run_info.get("export")),
        ("arguments", " ".join(f"--{k}={v}" for k, v in (run_info.get("args") or {}).items()
                               if v not in ("", "None", "False", "0"))),
        ("KOs in export", sel.get("input")),
        *[(f"excluded: {k}", v) for k, v in (sel.get("excluded") or {}).items()],
        ("eligible", sel.get("eligible")),
        ("assessed", sel.get("selected")),
        ("by language", sel.get("by_language")),
        ("settings version", run_info.get("spec_version")),
        ("fields measured against", REFERENCE_LABEL.get(ref, ref)),
        ("models", run_info.get("models")),
        ("LLM score", (f"{llm_info.get('model')}: {llm_info.get('calls')} calls, "
                       f"{llm_info.get('failed', 0)} without a usable score")
         if llm_info.get("model") else "not scored (--no-llm-score): field scores average three pillars"),
        ("*_llm fields", (f"written by {gen_info.get('model')} from the summary: "
                          f"{gen_info.get('calls')} calls, {gen_info.get('failed', 0)} failed "
                          f"(those KOs keep the export's), "
                          f"{gen_info.get('skipped_no_summary', 0)} without a summary")
         if gen_info.get("model") else "the export's (data-prep-opensearch improver)"),
        ("reference summary", (lambda ri: f"regenerated with {ri.get('model')} in the document's own "
                                          f"language: {ri.get('calls')} calls for {ri.get('kos')} KOs "
                                          f"({ri.get('from_file', 0)} from the file as it is "
                                          f"{ri.get('kinds')}, {ri.get('from_export_text', 0)} from "
                                          f"the export's text), {ri.get('transcription_calls', 0)} "
                                          f"transcription calls, {ri.get('revised', 0)} revised after "
                                          f"verification, {ri.get('translated', 0)} translated to "
                                          f"English for the index, {ri.get('failed', 0)} failed, "
                                          f"{ri.get('skipped_no_source', 0)} with no usable source; "
                                          f"languages {ri.get('languages')}; see the Summaries sheet")
         (run_info.get("resummarise")) if run_info.get("resummarise")
         else "the export's (platform summariser); see the Summaries sheet"),
        ("reference check", (lambda rc: f"{rc.get('flagged')} of {rc.get('checked')} summaries flagged "
                                        f"(thresholds learned from this run: {rc.get('thresholds')})"
         if rc and rc.get("checked") else (rc or {}).get("note", "not run"))
         ((analysis_out.get("reference_check") or {}))),
        ("by file language (scores not comparable across languages)", lang_line),
        ("code vs LLM pillar agreement (Spearman)",
         ", ".join(f"{p} {v['spearman']}" for p, v in (analysis_out.get("llm_vs_measured") or {}).items()
                   if isinstance(v, dict) and v.get("spearman") is not None) or "-"),
        ("self-test", f"{st_ok} of {st_total} deliberate defects detected" if st_total else "not run"),
        ("rank stability (Spearman vs equal weights)",
         ", ".join(f"{k} {v['spearman_vs_nominal']}" for k, v in rb.items()) or "-"),
        ("redundant sub-pillars", (analysis_out.get("redundant_subpillars") or {}).get("pairs")),
        ("sub-pillar correlation with field length", analysis_out.get("length_bias")),
        ("seconds", run_info.get("seconds")),
    ]
    for k, v in lines:
        ws.append([k, _cell(v)])
    ws["A1"].font = Font(bold=True, size=13)
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 110
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")

    # ---------------------------------------------------------------- KOs
    # The stored text's detected language and its mismatch flag stay in details.jsonl
    # for code; the sheet shows the file's own language (the trusted one, read from
    # the pages during --resummarise) next to what the contributor declared.
    header = ["_orig_id", "title", "declared language",
              "file language (vision)", "category", "licence",
              "gate decision", "gate probability", "gate decided by",
              "compliance", "missing obligations", "vocabulary drift",
              "content words", "content filters raised",
              "summary words", "compression ratio", "summary language",
              "summary NLI support", "summary key-term coverage",
              "reference flags", "reference reliability",
              "score: contributor"]
    header += [f"score: {llm_label}"] if has_llm else []
    header += [f"{_pillar_label(p)}: contributor" for p in PILLARS] + [f"{f}: contributor" for f in FIELDS]
    header += [f"recommended {f}" for f in FIELDS] + ["weakest parts (contributor)"]
    rows = []
    for r in results:
        g, c, ct = r.get("gate") or {}, r.get("compliance") or {}, r.get("content") or {}
        v = ((r.get("metadata") or {}).get("variants")) or {}
        o, l = v.get("original") or {}, v.get("llm") or {}
        sc = r.get("summary_check") or {}
        row = [r["_orig_id"], r.get("title"), r.get("declared_language"),
               (r.get("resummarised") or {}).get("file_language"),
               r.get("category"), r.get("_licence"), g.get("gate_decision"), g.get("gate_probability"),
               g.get("gate_decided_by"), c.get("compliance_status"), ";".join(c.get("compliance_failed") or []),
               ";".join(sorted((c.get("compliance_vocab_drift") or {}).keys())),
               ct.get("content_words"), ";".join(ct.get("content_filters_raised") or []),
               sc.get("summary_words"), sc.get("compression_ratio"), sc.get("summary_language"),
               sc.get("nli_support"), sc.get("key_term_coverage"),
               ";".join(sc.get("flags") or []) or None, sc.get("reference_reliability"),
               o.get("score")]
        row += [l.get("score")] if has_llm else []
        row += [(o.get("pillars") or {}).get(p) for p in PILLARS]
        row += [((o.get("fields") or {}).get(f) or {}).get("score") for f in FIELDS]
        rec = r.get("recommended") or {}
        row += [_label((rec.get(f) or {}).get("recommended") or "") or None for f in FIELDS]
        row += [" | ".join(f"{x['field']}.{x['subpillar']}={x['value']:.2f} {x['why']}".strip()
                           for x in r.get("weakest") or [])]
        rows.append(row)
    _sheet(wb, "KOs", header, rows, {1: 26, 2: 50, len(header): 90})

    # ---------------------------------------------------------------- Fields
    header = (["_orig_id", "field", "version", "text", "field score"]
              + [_pillar_label(p) for p in PILLARS]
              + ["LLM: structural", "LLM: semantic", "LLM: functional"] + SUBS
              + [f"faithfulness to {ref}", "recommended", "LLM reason", "not evaluable"])
    rows = []
    for r in results:
        v = ((r.get("metadata") or {}).get("variants")) or {}
        rec = r.get("recommended") or {}
        for f in FIELDS:
            chosen = (rec.get(f) or {}).get("recommended")
            for vname, vres in v.items():
                fr = ((vres or {}).get("fields") or {}).get(f)
                if not fr:
                    continue
                # recommend()'s name for this KO's "llm" variant: generated unless this
                # KO fell back to the export's fields (generation off, skipped or failed).
                llm_key = "generated_llm" if r.get("generated_llm") else "existing_llm"
                key = llm_key if vname == "llm" else vname
                acc = ((fr.get("evidence") or {}).get("consistency") or {}).get("parts", {}).get("accuracy")
                crit = ((fr.get("llm") or {}).get("criteria")) or {}
                row = [r["_orig_id"], f, _label(key), fr.get("text"), fr.get("score")]
                row += [(fr.get("pillars") or {}).get(p) for p in PILLARS]
                row += [crit.get("structural"), crit.get("semantic"), crit.get("functional")]
                row += [(fr.get("subpillars") or {}).get(s) for s in SUBS]
                row += [acc, (chosen == key) if chosen else None, (fr.get("llm") or {}).get("reason"),
                        ", ".join(fr.get("not_evaluable") or []) or None]
                rows.append(row)
    _sheet(wb, "Fields", header, rows, {1: 26, 3: 16, 4: 70, len(header) - 1: 60})

    # ---------------------------------------------------------------- Summaries
    # The reference text of every KO, as measured against; with --resummarise also its
    # English version (the corpus row) and the platform's summary it replaced.
    # "sentences without a verdict" applies to documents checked sentence by sentence
    # (longer than one chunk); it is empty for the rest.
    header = ["_orig_id", "source", "language", "basis", "pages", "chunks", "calls",
              "errors found", "missing found", "revised", "sentences without a verdict",
              "words", "summary", "English version", "platform summary"]
    rows = []
    for r in results:
        s = r.get("summary") or {}
        if not s:
            continue
        rs = r.get("resummarised") or {}
        basis = (f"{rs.get('from')}:{rs.get('kind')}" + (f" after {rs['after']}" if rs.get("after") else "")
                 if rs else None)
        unverified = (f"{rs['unverified']} of {rs['sentences']}" if "sentences" in rs else None)
        rows.append([r["_orig_id"], s.get("source"), s.get("language"), basis,
                     rs.get("pages_total"), rs.get("chunks"), rs.get("calls"),
                     rs.get("errors"), rs.get("missing"), rs.get("revised"), unverified,
                     s.get("words"), s.get("text"), s.get("english"), s.get("platform")])
    _sheet(wb, "Summaries", header, rows, {1: 26, 4: 24, 13: 90, 14: 90, 15: 70})

    # ---------------------------------------------------------------- Self-test
    if selftest:
        header = list(selftest[0])
        _sheet(wb, "Self-test", header, [[s.get(k) for k in header] for s in selftest], {1: 26, 4: 45})

    path = out_dir / "report.xlsx"
    wb.save(path)
    return path


__all__ = ["write"]
