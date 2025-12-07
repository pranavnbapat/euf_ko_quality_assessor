# assess_ko_quality/structural.py

from typing import Any, Dict, List

from quality_text_utils import tokens, sentence_lengths, count_urls, count_non_ascii, count_all_caps_runs


def structural_scores(
    title: str,
    subtitle: str,
    desc: str,
    content: str,
    keywords: List[str],
) -> Dict[str, Any]:
    """
    Compute Structural sub-scores:
      - length (0–5)
      - completeness (0–5)
      - noise (0–5, higher = cleaner)
      - formatting (0–5)

    And aggregate:
      - Structural_Total_Raw (0–20)
      - Structural_Score_0_25 (0–25)
    """
    text_all = " ".join([title, subtitle, desc, content])
    toks_content = tokens(content)
    n_tokens = len(toks_content)

    # --- length ---
    # Per-field lengths
    title_words = len(tokens(title)) if title else 0
    subtitle_words = len(tokens(subtitle)) if subtitle else 0
    desc_words = len(tokens(desc)) if desc else 0

    title_chars = len(title) if title else 0
    subtitle_chars = len(subtitle) if subtitle else 0
    desc_chars = len(desc) if desc else 0
    content_chars = len(content) if content else 0

    desc_sentences = len(sentence_lengths(desc)) if desc else 0
    kw_count = len(keywords)

    length_components: List[int] = []

    # 1) Content / KO body length (tokens + chars)
    if n_tokens == 0:
        content_len_score = 0
    else:
        if 150 <= n_tokens <= 3000:
            content_len_score = 5
        elif 80 <= n_tokens < 150 or 3000 < n_tokens <= 6000:
            content_len_score = 3
        else:
            content_len_score = 1

        # very short in characters as an extra safeguard
        if content_chars < 600 and content_len_score > 2:
            content_len_score = 2

    length_components.append(content_len_score)

    # 2) Title length: 6–14 words, 45–90 chars
    if title:
        title_score = 5
        if not (6 <= title_words <= 14):
            title_score -= 2
        if not (45 <= title_chars <= 90):
            title_score -= 1
        title_score = max(0, min(5, title_score))
        length_components.append(title_score)

    # 3) Subtitle length: 8–20 words, ≤140 chars
    #    (subtitle is optional; if missing, we just skip it)
    if subtitle:
        subtitle_score = 5
        if not (8 <= subtitle_words <= 20):
            subtitle_score -= 2
        if subtitle_chars > 140:
            subtitle_score -= 1
        subtitle_score = max(0, min(5, subtitle_score))
        length_components.append(subtitle_score)

    # 4) Description: 40–80 words, 300–600 chars, 2–4 sentences
    if desc:
        desc_score = 5
        if not (40 <= desc_words <= 80):
            desc_score -= 2
        if not (300 <= desc_chars <= 600):
            desc_score -= 1
        if not (2 <= desc_sentences <= 4):
            desc_score -= 1
        desc_score = max(0, min(5, desc_score))
        length_components.append(desc_score)

    # 5) Keywords: 4–10 terms
    if kw_count == 0:
        kw_len_score = 0
    elif 4 <= kw_count <= 10:
        kw_len_score = 5
    elif 2 <= kw_count < 4 or 10 < kw_count <= 15:
        kw_len_score = 3
    else:
        kw_len_score = 1
    length_components.append(kw_len_score)

    # Aggregate all available components (0–5) into a single 0–5 length score
    if length_components:
        length_score = round(sum(length_components) / len(length_components))
    else:
        length_score = 0

    # --- completeness ---
    missing_fields = []
    if not title:
        missing_fields.append("title")
    if not desc:
        missing_fields.append("description")
    if not content:
        missing_fields.append("content")
    if len(keywords) < 4:
        missing_fields.append("keywords")

    if not missing_fields:
        completeness_score = 5
    elif len(missing_fields) == 1:
        completeness_score = 3
    elif len(missing_fields) == 2:
        completeness_score = 2
    else:
        completeness_score = 0

    # --- noise ---
    url_count = count_urls(text_all)
    non_ascii = count_non_ascii(text_all)
    all_caps = count_all_caps_runs(text_all)

    noise_score = 5
    if url_count > 5:
        noise_score -= 1
    if url_count > 15:
        noise_score -= 1
    if non_ascii > 20:
        noise_score -= 1
    if all_caps > 5:
        noise_score -= 1
    if all_caps > 20:
        noise_score -= 1
    noise_score = max(0, noise_score)

    # --- formatting ---
    s_lens = sentence_lengths(content)
    if not s_lens:
        formatting_score = 1
    else:
        avg_len = sum(s_lens) / len(s_lens)
        # sweet spot: 8–35 tokens per sentence
        if 8 <= avg_len <= 35:
            formatting_score = 5
        elif 6 <= avg_len <= 45:
            formatting_score = 4
        elif 4 <= avg_len <= 55:
            formatting_score = 3
        else:
            formatting_score = 1

    struct_raw = length_score + completeness_score + noise_score + formatting_score
    struct_scaled = round(struct_raw * 25.0 / 20.0)  # map 0–20 → 0–25

    return {
        "Structural_length": length_score,
        "Structural_completeness": completeness_score,
        "Structural_noise": noise_score,
        "Structural_formatting": formatting_score,
        "Structural_Total_Raw": struct_raw,
        "Structural_Score_0_25": struct_scaled,
    }
