# assess_ko_quality/scripts/build_agrovoc_anchor_texts.py

"""
Build anchor texts from AGROVOC (controlled vocabulary) via SPARQL.

Output: JSONL file, one record per concept with a constructed anchor_text.
- Designed for centroid building (not for runtime lookup).
- Uses English labels/notes to match your current English-only pipeline.

Source endpoint: https://agrovoc.fao.org/sparql (FAO AGROVOC).  :contentReference[oaicite:4]{index=4}
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

SPARQL_ENDPOINT = "https://agrovoc.fao.org/sparql"

OUT_DIR = Path(__file__).resolve().parents[1] / "anchors" / "agrovoc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "agrovoc_anchor_texts.jsonl"

# Reasonable paging – tune if needed
PAGE_SIZE = 2000
MAX_CONCEPTS: Optional[int] = 20000  # set None for "all", but that can be big


def sparql_select(query: str, timeout_s: int = 60) -> List[Dict[str, str]]:
    """
    Run a SPARQL SELECT query and return bindings as list of dicts.
    """
    headers = {"Accept": "application/sparql-results+json"}
    r = requests.post(
        SPARQL_ENDPOINT,
        data={"query": query},
        headers=headers,
        timeout=timeout_s,
    )

    if r.status_code >= 400:
        print("---- SPARQL query (first 800 chars) ----")
        print(query[:800])
        print("---- Response text (first 800 chars) ----")
        print(r.text[:800])

    r.raise_for_status()
    data = r.json()
    rows = []
    for b in data.get("results", {}).get("bindings", []):
        row = {k: v.get("value", "") for k, v in b.items()}
        rows.append(row)
    return rows


def build_anchor_text(
    pref: str,
    alts: List[str],
    note: str,
    broader: List[str],
) -> str:
    """
    Construct a richer text than just a label, so embeddings have context.
    """
    parts = []
    if pref:
        parts.append(f"Concept: {pref}.")
    if alts:
        parts.append("Alternative labels: " + "; ".join(sorted(set(alts))) + ".")
    if broader:
        parts.append("Broader terms: " + "; ".join(sorted(set(broader))) + ".")
    if note:
        parts.append("Definition: " + note.strip())
    return " ".join(parts).strip()


def main() -> None:
    # Fetch concepts in pages.
    # We fetch: prefLabel (en), optional altLabel (en), optional scopeNote/definition (en),
    # optional broader prefLabels (en).
    #
    # Note: SKOS properties vary a bit; we try scopeNote and definition.
    offset = 0
    written = 0

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        while True:
            query = f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

            SELECT ?c
                   (SAMPLE(STR(?prefLit)) AS ?pref)
                   (GROUP_CONCAT(DISTINCT STR(?altLit); separator="||") AS ?alts)
                   (SAMPLE(STR(?scopeNoteLit)) AS ?scopeNote)
                   (SAMPLE(STR(?defLit)) AS ?definition)
                   (GROUP_CONCAT(DISTINCT STR(?bprefLit); separator="||") AS ?broader)
            WHERE {{
              ?c a skos:Concept ;
                 skos:prefLabel ?prefLit .
              FILTER(lang(?prefLit) = "en")

              OPTIONAL {{
                ?c skos:altLabel ?altLit .
                FILTER(lang(?altLit) = "en")
              }}

              OPTIONAL {{
                ?c skos:scopeNote ?scopeNoteLit .
                FILTER(lang(?scopeNoteLit) = "en")
              }}

              OPTIONAL {{
                ?c skos:definition ?defLit .
                FILTER(lang(?defLit) = "en")
              }}

              OPTIONAL {{
                ?c skos:broader ?b .
                ?b skos:prefLabel ?bprefLit .
                FILTER(lang(?bprefLit) = "en")
              }}
            }}
            GROUP BY ?c
            ORDER BY ?c
            LIMIT {PAGE_SIZE}
            OFFSET {offset}
            """

            rows = sparql_select(query)
            if not rows:
                break

            for r in rows:
                c = r.get("c", "")
                pref = r.get("pref", "").strip()

                alts = [x.strip() for x in r.get("alts", "").split("||") if x.strip()]

                # Prefer scopeNote; fall back to definition
                note = (r.get("scopeNote") or r.get("definition") or "").strip()

                broader = [x.strip() for x in r.get("broader", "").split("||") if x.strip()]

                anchor_text = build_anchor_text(pref=pref, alts=alts, note=note, broader=broader)

                rec = {
                    "uri": c,
                    "prefLabel_en": pref,
                    "altLabels_en": alts,
                    "broader_prefLabels_en": broader,
                    "note_en": note,
                    "anchor_text": anchor_text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

                if MAX_CONCEPTS is not None and written >= MAX_CONCEPTS:
                    break

            print(f"[AGROVOC] Fetched/wrote {written} concepts (offset={offset})")
            if MAX_CONCEPTS is not None and written >= MAX_CONCEPTS:
                break

            offset += PAGE_SIZE
            time.sleep(0.2)  # be polite to the endpoint

    print(f"[OK] Wrote: {OUT_JSONL}")


if __name__ == "__main__":
    main()
