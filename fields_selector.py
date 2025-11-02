# fields_selector

"""
Quick analysis utility for KO JSON files.
Prints coverage, average token length, and a simple entropy (distinctiveness) score
for each selected field — helps decide which fields are most informative.
"""

import json
import math

from collections import Counter

# Path to the input JSON lines file.
PATH = "input/final_output_14_10-2025_17-37-04.json"

# Fields to inspect in each KO (Knowledge Object).
FIELDS = ["title", "subtitle", "intended_purposes", "description", "ko_content_flat", "keywords", "topics", "themes"]

docs = []
# Read all objects (one per line) and flatten list fields into strings.
for line in open(PATH, "r", encoding="utf-8"):
    ko = json.loads(line)
    docs.append({f: (" ".join(ko[f]) if isinstance(ko.get(f), list) else (ko.get(f) or "")) for f in FIELDS})

def entropy(values):
    """Simple Shannon entropy: measures how diverse field values are."""
    c = Counter(values)
    n = sum(c.values())
    return -sum((v/n) * math.log((v/n) + 1e-12) for v in c.values())

N = len(docs)
for f in FIELDS:
    vals = [d[f] for d in docs]
    cov = sum(1 for v in vals if v.strip()) / N
    avg_len = sum(len(v.split()) for v in vals) / max(1, N)
    ent = entropy([v.strip()[:120] or "<EMPTY>" for v in vals])  # coarse distinctiveness
    print(f"{f:18}  coverage={cov:.2%}  avg_tokens={avg_len:.1f}  entropy~={ent:.2f}")