# assess_ko_quality/functional.py

from typing import Any, Dict, List

from quality_text_utils import tokens, jaccard, count_urls, count_all_caps_runs, _STOP_EN


def functional_scores(
    title: str,
    desc: str,
    content: str,
    keywords: List[str],
) -> Dict[str, Any]:
    """
    Functional sub-scores (search/RAG friendliness proxies):
      - bm25_readiness (0–5): length + lexical overlap
      - embedding_readiness (0–5): coherence + cleanliness
      - rag_readiness (0–5): segmentation / step cues
      - keyword_indexability (0–5): keywords that actually appear in content
    """
    title_toks = tokens(title)
    desc_toks = tokens(desc)
    content_toks = tokens(content)
    kw_toks = tokens(" ".join(keywords))

    # BM25: overlap and length window
    jc_title = jaccard(title_toks, content_toks)
    jc_desc = jaccard(desc_toks, content_toks)
    len_content = len(content_toks)

    bm25 = 0
    if 150 <= len_content <= 4000:
        bm25 += 2
    elif 80 <= len_content < 150 or 4000 < len_content <= 6000:
        bm25 += 1

    if jc_title >= 0.3:
        bm25 += 2
    elif jc_title >= 0.15:
        bm25 += 1

    if jc_desc >= 0.3:
        bm25 += 1
    elif jc_desc >= 0.15:
        bm25 += 0.5

    bm25 = min(5, bm25)

    # Embeddings: coherence + noise
    meta_toks = tokens(title + " " + desc)
    coherence = jaccard(meta_toks, content_toks)
    noise_score = 5  # start high; reuse structural noise idea
    url_count = count_urls(content)
    all_caps = count_all_caps_runs(content)
    if url_count > 5:
        noise_score -= 1
    if all_caps > 5:
        noise_score -= 1
    noise_score = max(0, noise_score)

    emb = 0
    if coherence >= 0.35:
        emb += 3
    elif coherence >= 0.2:
        emb += 2
    elif coherence > 0:
        emb += 1

    if noise_score >= 4:
        emb += 2
    elif noise_score >= 3:
        emb += 1

    emb = min(5, emb)

    # RAG: segmentation cues (steps/headings) + moderate length
    lower_content = content.lower()
    step_hits = sum(
        1
        for c in [
            "step 1",
            "step 2",
            "step 3",
            "step 4",
            "step 5",
            "section",
            "introduction",
            "conclusion",
            "summary",
        ]
        if c in lower_content
    )
    rag = 0
    if 200 <= len_content <= 3000:
        rag += 2
    if step_hits >= 3:
        rag += 3
    elif step_hits == 2:
        rag += 2
    elif step_hits == 1:
        rag += 1

    rag = min(5, rag)

    # Keyword indexability: non-stopword keywords that appear in content
    kw_clean = [k for k in kw_toks if k not in _STOP_EN]
    kw_in_content = [k for k in kw_clean if k in content_toks]
    if len(kw_in_content) >= 3:
        kw_idx = 5
    elif len(kw_in_content) == 2:
        kw_idx = 4
    elif len(kw_in_content) == 1:
        kw_idx = 3
    elif kw_clean:
        kw_idx = 1
    else:
        kw_idx = 0

    func_raw = bm25 + emb + rag + kw_idx
    func_scaled = round(func_raw * 25.0 / 20.0)

    return {
        "Functional_BM25_readiness": bm25,
        "Functional_embedding_readiness": emb,
        "Functional_RAG_readiness": rag,
        "Functional_keyword_indexability": kw_idx,
        "Functional_Total_Raw": func_raw,
        "Functional_Score_0_25": func_scaled,
        "Functional_keywords_in_content": ";".join(sorted(set(kw_in_content))),
    }
