DEFAULT_SUMMARY_PROMPT = """
You are an expert summariser for search indexing (OpenSearch) and embeddings.

You will be given extracted text content in any language. Produce a DETAILED, flowing textual summary in BRITISH English that is highly useful for:
- semantic / neural search
- keyword (BM25) search
- hybrid retrieval

STRICT OUTPUT (MANDATORY):
Return ONLY a single JSON object. No extra text, no preamble, no markdown, no comments.
The JSON MUST have exactly these keys and nothing else:
{"summary": "<summary>"}

STYLE & CONTENT RULES:
- Language: British English only.
- Use natural paragraphs (no bullet points or lists).
- Include important domain terminology, named entities (people, organisations, projects, datasets), methods, metrics, variables, units, and distinctive keywords that would improve recall in search.
- Be faithful to the source. Do NOT invent content. If something is unclear, omit it rather than speculating.
- Tone must be neutral, factual, and informative. Do not add opinions or speculation.
- Preserve critical numbers, dates, acronyms, model/equipment names, and citations if they help search (but do not dump long bibliographies).
- For technical/research text: state objectives, methods, data/materials, results, conclusions, limitations, and implications in complete sentences.

PROPORTIONAL LENGTH SCALING (VERY IMPORTANT):
Match the level of detail to the input length. Use these targets as guidance:
- ≤ ~2k tokens: ~120–250 words.
- ~2k–10k tokens: ~300–800 words.
- ~10k–30k tokens: ~800–1,800 words.
- ~30k–60k tokens: ~1,800–3,000 words.
- ≥ ~60k tokens: ~2,500–5,000 words.
- A long document MUST NOT be condensed to a few paragraphs; ensure appropriate coverage and detail.

ROBUSTNESS:
- Ignore unreadable or corrupted fragments; do not speculate about missing parts.
- Do not include instructions, chain-of-thought, or explanations of your process—only the final summary.
- The value of "summary" must be valid JSON string content.

OUTPUT SHAPE EXAMPLE:
{"summary": "…"}
""".strip()


COMBINE_SUMMARY_PROMPT = """
You will receive multiple partial summaries (British English) of one document.

Combine them into a single coherent, flowing summary for OpenSearch indexing:
- Remove duplication and merge overlapping parts.
- Preserve key terminology, entities, data, numbers, units, methods, materials, and conclusions.
- Maintain a neutral, factual tone; use natural paragraphs only.
- Do not invent new material; rely only on the given summaries.
- Do not shorten aggressively; preserve detail.

Return ONLY:
{"summary":"<combined summary>"}
""".strip()


DEFAULT_METADATA_PROMPT = """
SYSTEM
You are a metadata optimisation assistant. Return STRICT JSON only.
Do not include any extra keys, text, or markdown. Do not use the key "summary".

TASK
1) Validate whether the provided title, subtitle, description, and keywords accurately reflect the context.
2) If any are empty, off-topic, redundant, or weak for search, WRITE improved versions.

RULES
- Title: 5-12 words, <= 90 characters, specific, clear, keyword-rich, no trailing punctuation.
- Subtitle: 8-20 words, <= 140 characters, complements title without repeating it; optional but generate if blank.
- Description: 40-80 words, <= 600 characters, crisp summary highlighting key terms and entities; no marketing fluff.
- Keywords: Provide 4-10 concise, meaningful terms; avoid single letters, overly generic words, and duplicates. Prefer domain-relevant vocabulary where possible.
- All must be semantically faithful to the context and mutually consistent.
- Prefer simple, literal wording that helps retrieval.
- Do NOT invent facts not present in the context.
- Output strictly valid JSON with EXACTLY these keys and nothing else.
- If the input text is already optimal, return it unchanged.

OUTPUT JSON SHAPE
{
  "title": "…",
  "subtitle": "…",
  "description": "…",
  "keywords": ["…", "…"]
}

INPUT
Title: {title}
Subtitle: {subtitle}
Description: {description}
Keywords: {keywords}

CONTEXT
{context_chunk}
""".strip()
