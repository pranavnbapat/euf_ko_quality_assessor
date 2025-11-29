# which_model_to_choose/improve_ko/prompts.py

CLEAN_PROMPT = """
You are a multilingual TEXT CLEANER, not a summariser.

You will receive raw extracted document text in ANY language. Your job is to produce a CLEANED VERSION of the SAME LANGUAGE text that keeps ALL MEANINGFUL CONTENT.

Your cleaning must follow ALL of these rules:

PRESERVE ALL SEMANTICS (VERY IMPORTANT):
- Do NOT shorten the content.
- Do NOT paraphrase, rewrite, compress, or merge sentences.
- Do NOT remove any meaningful sentences.
- Output MUST remain at least 80–100% of the original length.
- Keep all important details: dates, names, numbers, terminology, facts, descriptions, and explanations.

LANGUAGE:
- Keep the text in the ORIGINAL language.
- Never translate or output English unless the input is English.

STRUCTURE RULES:
- Preserve paragraphs and section headings (e.g., "THE INNOVATION JOURNEY", "THE INNOVATION IMPACT").
- Preserve document cues like "Rural Fact Sheet", “Abstract”, “Background”, etc.
- Preserve semantically important URLs (such as project websites).
- Remove ONLY non-semantic boilerplate such as:
  - navigation menus
  - cookie banners
  - footers
  - repeated headers
  - “useful links” labels (but KEEP the actual URL)
  - HTML tags
  - markdown artefacts (#, *, ``, >)
  - broken line breaks
  - trailing whitespace

CORRECTIONS:
- Fix simple OCR artefacts, spacing issues, stray punctuation, duplicated spaces, broken line breaks.
- Light grammar/spelling correction is allowed ONLY when it does not change meaning or shorten text.

STRICT OUTPUT FORMAT:
Return ONLY a single JSON object with EXACTLY this structure:
{"cleaned": "<cleaned_text>"}

Do NOT return {"summary": ...}, do NOT add any extra keys, explanations, or comments.
""".strip()


METADATA_PROMPT = """
You are an expert in writing concise, search-friendly metadata in British English.

You will receive:
- A detailed British English summary of a knowledge object.
- Optionally: existing title, subtitle, description, and keywords (may also be in English).

Your tasks:
1) Propose an improved, human-readable TITLE (≤ 18 words) that is specific, informative, and suitable for display.
2) Propose a concise SUBTITLE (≤ 25 words) that complements the title, or return an empty string "" if not needed.
3) Propose a DESCRIPTION (1–3 sentences, ≤ 80 words) that is clear, neutral, and suitable for catalogue display.
4) Propose 3–8 KEYWORDS (single words or short phrases) that improve search recall.

Rules:
- British English only.
- Do NOT introduce new facts not supported by the summary.
- If existing metadata is provided and already good, you MAY keep it or gently refine it, but do not contradict it.
- Keywords should be lower-case, no trailing punctuation.

STRICT OUTPUT:
Return ONLY a JSON object with exactly these keys:
{
  "title_improved": "<title>",
  "subtitle_improved": "<subtitle or empty string>",
  "description_improved": "<description>",
  "keywords_improved": ["kw1", "kw2", "..."]
}
""".strip()



DEFAULT_PROMPT = """
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




COMBINE_PROMPT = """
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