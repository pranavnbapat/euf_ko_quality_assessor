### Goal: audit which fields in your Knowledge Objects (KOs) are most useful for text search. It:

- Loads the newest JSON/JSONL file under ../input. 
- Normalises each candidate field to a plain text string (lists/dicts become space-joined). 
- Tokenises once (cheap lowercase \w+ regex) and caches tokens per field. 
- Computes diagnostics per field:
  - coverage (how many docs have any text),
  - avg length (avg tokens/doc),
  - boilerplate proxy (how repetitive the exact field strings are), and 
  - avg IDF (term rarity). 
- Runs self-retrieval: for each KO, you query with its own title (Q1) or title + keywords (Q2) and compute MRR@K against candidate field sets using a minimal BM25 scorer with a tiny inverted index to keep it fast. 
- Builds candidates: single fields and combinations (based on a “CORE” set; optionally “FAST” to keep combos small). 
- Scores candidates (MRR@TOPK for Q1/Q2) and ranks single fields with a Field Usefulness Score (FUS) that blends MRR, avg IDF, coverage, and penalises boilerplate. 
- Prints a report (diagnostics, MRR per candidate, FUS table, and a suggested schema) and also writes it to ../reports/field_audit_YYYY-MM-DD_hh-mm-ss.txt.

