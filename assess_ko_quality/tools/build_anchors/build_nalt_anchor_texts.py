# scripts/build_nalt_anchor_texts.py

"""
Build anchor texts from a local USDA NALT RDF/SKOS file.

Usage:
  python scripts/build_nalt_anchor_texts_local.py \
    --input anchors/nalt/nalt-full_dwn_20240716.rdf \
    --out anchors/nalt/nalt_anchor_texts.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

from rdflib import Graph, RDF
from rdflib.namespace import SKOS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to local NALT RDF file (.rdf/.xml/.ttl/.nt/.jsonld).")
    p.add_argument("--format", default=None, help="rdflib format override: xml/turtle/nt/json-ld. Auto if omitted.")
    p.add_argument("--out", default="anchors/nalt/nalt_anchor_texts.jsonl", help="Output JSONL path.")
    p.add_argument("--limit", type=int, default=0, help="Max concepts to write (0 = no limit).")
    return p.parse_args()


def best_lang(g: Graph, s, p, langs=("en", "en-US", "en-GB")) -> Optional[str]:
    vals = list(g.objects(s, p))
    if not vals:
        return None
    for lang in langs:
        for v in vals:
            if getattr(v, "language", None) == lang and str(v).strip():
                return str(v).strip()
    for v in vals:
        if str(v).strip():
            return str(v).strip()
    return None


def build_anchor(pref: str, alts: List[str], scope: Optional[str], broader: List[str]) -> str:
    bits = [f"Preferred term: {pref}."]
    if alts:
        bits.append("Synonyms: " + "; ".join(alts[:12]) + ".")
    if broader:
        bits.append("Broader concepts: " + "; ".join(broader[:10]) + ".")
    if scope:
        bits.append("Definition: " + scope)
    return " ".join(bits).strip()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = Graph()

    # NALT downloads are commonly RDF/XML, so default to 'xml' if not provided.
    fmt = args.format
    if fmt is None:
        suf = in_path.suffix.lower()
        fmt = {
            ".rdf": "xml",
            ".xml": "xml",
            ".ttl": "turtle",
            ".nt": "nt",
            ".jsonld": "json-ld",
            ".json": "json-ld",
        }.get(suf, "xml")

    print(f"[NALT] Parsing file: {in_path} (format={fmt})")
    g.parse(str(in_path), format=fmt)
    print(f"[NALT] Parsed triples: {len(g)}")

    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for s in g.subjects(RDF.type, SKOS.Concept):
            pref = best_lang(g, s, SKOS.prefLabel)
            if not pref:
                continue

            alts = []
            for v in g.objects(s, SKOS.altLabel):
                if getattr(v, "language", None) in (None, "en", "en-US", "en-GB"):
                    sv = str(v).strip()
                    if sv and sv.lower() != pref.lower():
                        alts.append(sv)

            scope = best_lang(g, s, SKOS.definition) or best_lang(g, s, SKOS.scopeNote)

            broader_labels = []
            for b in g.objects(s, SKOS.broader):
                bl = best_lang(g, b, SKOS.prefLabel)
                if bl:
                    broader_labels.append(bl)

            anchor_text = build_anchor(pref, alts, scope, broader_labels)
            if len(anchor_text) < 30:
                continue

            f.write(json.dumps({"uri": str(s), "source": "nalt", "anchor_text": anchor_text}, ensure_ascii=False) + "\n")
            written += 1

            if args.limit and written >= args.limit:
                break

            if written and written % 5000 == 0:
                print(f"[NALT] Wrote {written} anchors...")

    print(f"[OK] Wrote {written} anchors -> {out_path}")


if __name__ == "__main__":
    main()


