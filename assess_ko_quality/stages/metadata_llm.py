# assess_ko_quality/stages/metadata_llm.py
"""
LLM review of a KO's metadata: a verdict per field, a proposal only where needed, and
a measurable decision on whether to take it.

For each KO, ONE call asks the model, field by field, whether the title, subtitle,
description and keywords correctly describe the file, and - only where it says they do
not - to propose a replacement. The model does not get the last word:

  1. every proposal is scored by the same scorer as the original (stages/metadata_score)
  2. a proposal is accepted only if its field score beats the original's by at least
     `min_gain` points AND its accuracy against the file does not drop by more than
     `accuracy_tolerance`

Why the second condition. The scorer rewards things an LLM can produce cheaply -
lengths in range, search-friendly words - so a proposal could score higher while
saying something the file does not support. Optimising hard against a proxy measure
degrades the true objective (Gao, Schulman & Hilton, ICML 2023; Manheim & Garrabrant
2018). Faithfulness to the file is the objective here, so it is held as a constraint,
not traded against the rest.

Why the model is asked for a verdict first. The existing *_llm fields were produced by
a prompt that allowed returning a field unchanged, yet 76 of 96 titles and 90 of 96
descriptions came back rewritten, with no record of whether the original was wrong.
LLM judges also have documented biases - position, verbosity, self-preference (Zheng
et al., NeurIPS 2023; Wang et al., ACL 2024). Asking which criterion fails, and
checking the answer with a measurement, keeps rewrites to the cases that need them.

Costs one LLM call per KO; responses are cached on disk, so a rerun is free.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from core import metadata_metrics as mm
from core.corpus_index import FIELDS, CorpusIndex, Embedder, field_text, reference_text
from stages.metadata_score import score_variant

CRITERIA = {
    "faithful": "everything it says is supported by the file, and nothing contradicts it",
    "specific": "it names what is particular to this file (crop, animal, method, place, project), not generic words",
    "coherent": "title, subtitle, description and keywords describe the same subject",
    "clear": "no unexplained acronyms; the description is complete sentences and is not cut off",
    "findable": "someone searching for this file's topic would use these words",
    "well_formed": "within the length range; no markup, URLs or stray formatting; keywords are one concept each, without duplicates",
}

LANGUAGE_POLICY = {
    "keep": "Write any proposal in the same language as the field it replaces.",
    "source": "Write any proposal in the language of the file.",
    "english": "Write any proposal in English.",
}

SYSTEM = (
    "You review the metadata of entries on EU-FarmBook, a multilingual European platform "
    "of practical knowledge on agriculture, forestry, livestock, food and rural life. You "
    "check whether each metadata field correctly and usefully describes the file it belongs "
    "to. Judge only from the file content you are shown. Never add facts that are not in the "
    "file. Content may be in any European language."
)


def _rules(spec: Dict[str, Any]) -> List[str]:
    """The platform's rules for each field, as the model should apply them."""
    out = []
    for f in FIELDS:
        rule = spec["fields"][f]["length"]
        bits = []
        if rule.get("unit") == "items":
            bits.append(f"{rule['min']}-{rule['max']} terms, each a word or short phrase")
        if "max_chars_no_spaces" in rule:
            bits.append(f"at most {rule['max_chars_no_spaces']} characters not counting spaces")
        if "max_sentences" in rule:
            bits.append("one sentence (two at most)")
        if "min_chars" in rule and "max_chars" in rule:
            bits.append(f"{rule['min_chars']}-{rule['max_chars']} characters")
        elif "max_chars" in rule:
            bits.append(f"at most {rule['max_chars']} characters")
        opt = " (optional)" if not spec["fields"][f]["required"] else ""
        if bits:
            out.append(f"- {f}{opt}: " + "; ".join(bits))
    out += [
        "- title: short and easily understandable, one key sentence; not an internal project "
        "code (like \"D3.1\" or \"Task 4\") and not a file name",
        "- description: an account of the file - its topic, purpose, key results and who it is "
        "useful for (farmers, advisors); explain technical terms, avoid unexplained abbreviations",
        "- keywords: each one concept, a word or short phrase, without duplicates",
    ]
    return out


def build_prompt(rec: Dict[str, Any], spec: Dict[str, Any], language: str,
                 max_file_chars: int) -> str:
    file_text = reference_text(rec, spec["reference"]["max_chars"])[:max_file_chars]
    current = []
    for f in FIELDS:
        v = field_text(f, rec.get(f))
        current.append(f"{f.upper()}: {v if v else '(empty)'}")
    lines = [
        "=== FILE CONTENT (what the file says) ===",
        file_text,
        "",
        "=== CURRENT METADATA ===",
        *current,
        "",
        "=== CRITERIA ===",
        *[f"- {k}: {v}" for k, v in CRITERIA.items()],
        "",
        "=== PLATFORM RULES ===",
        *_rules(spec),
        "",
        "=== TASK ===",
        "For each of title, subtitle, description and keywords decide whether it meets the "
        "criteria. If it does, say so and propose nothing. If it does not, list which criteria "
        "fail and write a replacement that fixes them without adding anything the file does not say.",
        "The subtitle is optional: if it is empty, leave it empty (verdict \"absent\").",
        LANGUAGE_POLICY[language],
        "",
        "Return JSON only, with exactly these keys:",
        '{"title": {"verdict": "correct" | "needs_change", "problems": [criteria names], "proposal": string | null},',
        ' "subtitle": {"verdict": "correct" | "needs_change" | "absent", "problems": [...], "proposal": string | null},',
        ' "description": {"verdict": ..., "problems": [...], "proposal": string | null},',
        ' "keywords": {"verdict": ..., "problems": [...], "proposal": [list of strings] | null}}',
    ]
    return "\n".join(lines)


def _parse(raw: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, dict):
        return out
    for f in FIELDS:
        e = raw.get(f)
        if not isinstance(e, dict):
            continue
        verdict = str(e.get("verdict") or "").strip().lower()
        prop = e.get("proposal")
        if f == "keywords" and isinstance(prop, str):
            prop = [x.strip() for x in prop.split(";") if x.strip()]
        out[f] = {
            "verdict": verdict,
            "problems": [str(p) for p in (e.get("problems") or []) if str(p) in CRITERIA],
            "proposal": prop if prop else None,
        }
    return out


def review(
    records: Sequence[Dict[str, Any]],
    original_scores: Dict[str, Dict[str, Any]],
    idx: CorpusIndex,
    emb: Embedder,
    nli: mm.NLI,
    spec: Dict[str, Any],
    model: str = "",
    language: str = "keep",
    workers: int = 4,
    log=print,
) -> List[Dict[str, Any]]:
    """
    Review and, where justified, replace metadata for each KO.

    `original_scores` maps _orig_id -> the "original" variant result from score_ko, so
    the originals are not scored twice. Returns one result per KO, in order.
    """
    from core.corpus_index import ko_id
    from core.llm import get_client

    cfg = spec["llm_review"]
    client = get_client(model=model or cfg["model"])
    if client is None:
        log("[llm-review] no LLM configured; skipped")
        return [{"_orig_id": ko_id(r), "llm_review": "no LLM configured"} for r in records]
    log(f"[llm-review] {len(records)} KOs, one call each, model {client.model}")

    def ask(rec: Dict[str, Any]) -> Any:
        prompt = build_prompt(rec, spec, language, int(cfg["max_file_chars"]))
        budget = int(cfg["max_tokens"])
        for _ in range(2):
            try:
                raw = client.complete_json(prompt, max_tokens=budget, system=SYSTEM)
                if raw is not None:
                    return raw
            except ValueError:
                pass                     # truncated JSON: retry once with more room
            budget *= 2
        return None

    raws = client.map_concurrent(list(records), ask, workers=workers, label="llm-review")

    results = []
    for rec, raw in zip(records, raws):
        kid = ko_id(rec)
        verdicts = _parse(raw)
        orig = original_scores.get(kid)
        row: Dict[str, Any] = {"_orig_id": kid, "llm_model": client.model,
                               "llm_ok": bool(verdicts), "fields": {}}
        if not verdicts or not orig:
            results.append(row)
            continue

        # Candidate metadata: the proposal where the model asked for a change.
        cand = {f: rec.get(f) for f in FIELDS}
        for f, v in verdicts.items():
            if v["verdict"] == "needs_change" and v["proposal"]:
                cand[f] = v["proposal"]
        changed = [f for f in FIELDS if cand[f] is not rec.get(f)]
        prop_scores = None
        if changed:
            ctx = mm.build_context(rec, idx, emb, spec)
            prop_scores = score_variant(rec, "proposal", ctx, idx, emb, nli, spec, values=cand)

        final: Dict[str, Any] = {}
        for f in FIELDS:
            v = verdicts.get(f, {"verdict": "missing", "problems": [], "proposal": None})
            o = (orig["fields"].get(f) or {})
            entry = {"verdict": v["verdict"], "problems": v["problems"],
                     "original": field_text(f, rec.get(f)),
                     "original_score": o.get("score")}
            if f in changed and prop_scores:
                p = prop_scores["fields"].get(f) or {}
                o_acc = ((o.get("evidence") or {}).get("consistency") or {}).get("parts", {}).get("accuracy")
                p_acc = ((p.get("evidence") or {}).get("consistency") or {}).get("parts", {}).get("accuracy")
                gain = (p.get("score") or 0) - (o.get("score") or 0)
                less_faithful = (o_acc is not None and p_acc is not None
                                 and p_acc < o_acc - float(cfg["accuracy_tolerance"]))
                if less_faithful:
                    decision = "rejected: less faithful to the file"
                elif gain < float(cfg["min_gain"]):
                    decision = "rejected: not better on the measured criteria"
                else:
                    decision = "accepted"
                entry.update({"proposal": field_text(f, cand[f]), "proposal_score": p.get("score"),
                              "gain": round(gain, 1), "accuracy_original": o_acc,
                              "accuracy_proposal": p_acc, "decision": decision})
                final[f] = cand[f] if decision == "accepted" else rec.get(f)
            else:
                entry["decision"] = ("kept: judged correct" if v["verdict"] == "correct"
                                     else "kept: optional and empty" if v["verdict"] == "absent"
                                     else "kept: no proposal")
                final[f] = rec.get(f)
            # Where the model and the measurement disagree, say so: those are the KOs a
            # person should look at.
            weak = (o.get("score") is not None and o["score"] < float(cfg["weak_field_score"]))
            entry["disagreement"] = ("model says correct, measurement says weak"
                                     if v["verdict"] == "correct" and weak else None)
            row["fields"][f] = entry
        row["recommended"] = {f: field_text(f, final[f]) for f in FIELDS}
        row["accepted_fields"] = [f for f, e in row["fields"].items() if e["decision"] == "accepted"]
        results.append(row)

    log(f"[llm-review] API calls {getattr(client, 'calls', '?')}, "
        f"cache hits {getattr(client, 'cache_hits', '?')}")
    return results


__all__ = ["review", "build_prompt", "CRITERIA", "LANGUAGE_POLICY"]
