# assessor.py

"""
Main assessment logic for Knowledge Objects (KOs).

This module:
- Normalises and validates each KO field.
- Computes multiple sub-scores:
    1. Semantic Precision & Clarity (0–40)
    2. Content Richness & Relevance (0–20)
    3. Cross-field Consistency (0–20)
    4. Linguistic Integrity (0–20)
- Checks controlled vocabulary compliance (topics, themes, etc.)
- Returns a detailed per-KO result dictionary.

Called by: runner.py
"""

import orjson

from typing import Any, Dict

from config import DATA_MODEL_DIR
from cv_loader import load_controlled_vocabs
from scoring import (score_title, score_description, score_keyword_alignment, score_content_depth,
                      score_diversity, score_duplication, score_topics_themes, score_project_echo, en_spell_score)
from text_utils import norm_text, tokens, detect_lang_safe, token_overlap_ratio
from validators import validate_doi, normalise_url, normalise_lang_label, license_status, cleanliness_score, structure_punct_score


# Load controlled vocabularies once globally for faster scoring.
CV = load_controlled_vocabs(DATA_MODEL_DIR)

def assess_ko(ko: Dict[str, Any]) -> Dict[str, Any]:
    title = norm_text(ko.get("title"))
    subtitle = norm_text(ko.get("subtitle"))
    desc = norm_text(ko.get("description"))
    content = norm_text(ko.get("ko_content_flat"))
    keywords = [norm_text(x) for x in (ko.get("keywords") or []) if isinstance(x, str)]
    topics = [norm_text(x) for x in (ko.get("topics") or []) if isinstance(x, str)]
    themes = [norm_text(x) for x in (ko.get("themes") or []) if isinstance(x, str)]
    category = norm_text(ko.get("category"))
    subcategories = [norm_text(x) for x in (ko.get("subcategories") or []) if isinstance(x, str)]
    languages = ko.get("languages") or []
    creators = [norm_text(x) for x in (ko.get("creators") or []) if isinstance(x, str)]
    intended_purposes = [norm_text(x) for x in (ko.get("intended_purposes") or []) if isinstance(x, str)]
    locations_flat = [norm_text(x) for x in (ko.get("locations_flat") or []) if isinstance(x, str)]
    license_raw = norm_text(ko.get("license"))
    category_raw = category
    project_name = norm_text(ko.get("project_name"))
    project_acronym = norm_text(ko.get("project_acronym"))
    project_url_raw = ko.get("project_url")
    doi_raw = ko.get("project_doi")
    _id = ko.get("_orig_id") or ko.get("@id") or ""

    lang_text_basis = " ".join([title, desc, content[:2000]])
    lang = detect_lang_safe(lang_text_basis)

    # Normalise first declared language (if any) and compare with detected
    declared_lang = normalise_lang_label(languages[0]) if languages else ""
    detected_base = normalise_lang_label(lang.split("-", 1)[0])
    lang_match = (declared_lang == "" or declared_lang == detected_base)

    # Fix and validate URL/DOI
    fixed_url, url_ok = normalise_url(project_url_raw)
    doi_ok = validate_doi(doi_raw)

    # --- Semantic Precision & Clarity (0–40)
    sp_title = score_title(title)
    sp_desc = score_description(desc)
    sp_kw = score_keyword_alignment(keywords, [title, desc, content[:2000]])
    sp_url_doi = (3 if url_ok else 0) + (2 if doi_ok else 0)
    sem_total = sp_title + sp_desc + sp_kw + sp_url_doi
    sem_total = min(40, sem_total)  # clamp

    # --- Content Richness & Relevance (0–20)
    cr_depth = score_content_depth(content)
    cr_div = score_diversity(content, lang)
    cr_dup = score_duplication(content)
    cr_total = min(20, cr_depth + cr_div + cr_dup)

    # --- Cross-field Consistency (0–20)
    cf_kw = min(8, int(round(sp_kw * 0.8)))  # reuse alignment strength
    cf_tt = score_topics_themes(topics, themes, [title, desc, content[:2000]])
    cf_proj = score_project_echo(project_acronym, project_name, [desc, content[:4000]])
    cf_total = min(20, cf_kw + cf_tt + cf_proj)

    # --- Linguistic Integrity (0–20)
    li_spell, miss_ratio = (8, 0.0)
    if lang.startswith("en"):
        li_spell, miss_ratio = en_spell_score(title, desc, content)
    li_clean, clean_issues = cleanliness_score([title, desc, content[:6000]])
    li_struct = structure_punct_score([desc, content[:4000]])
    li_total = min(20, li_spell + li_clean + li_struct)

    # Subtitle quality + duplicates
    subtitle_ok = False
    subtitle_duplicate_title = False
    subtitle_duplicate_description = False
    description_duplicate_title = False

    if subtitle:
        # exact-duplicate checks (after normalisation & lowercase)
        st = subtitle.strip().lower()
        tt = title.strip().lower()
        dt = desc.strip().lower()
        subtitle_duplicate_title = (st == tt) and bool(st)
        subtitle_duplicate_description = (st == dt) and bool(st)
        description_duplicate_title = (dt == tt) and bool(dt) and bool(tt)

        # If not exact dupes, check it adds some information (low overlap, reasonable length)
        if not subtitle_duplicate_title and not subtitle_duplicate_description:
            ov = token_overlap_ratio(title, subtitle)
            subtitle_ok = (ov <= 0.8) and (5 <= len(tokens(subtitle)) <= 40)
        else:
            ov = 1.0  # treat as high overlap for diagnostics
    else:
        ov = 0.0  # no subtitle

    # Creators must be at least 1
    creators_count = len([c for c in creators if c])
    creators_ok = creators_count >= 1

    # Intended purposes: can be one or many, but must not be blank
    purposes_count = len([p for p in intended_purposes if p])
    purposes_ok = purposes_count >= 1

    # Locations_flat: presence and bounded size (keep <=5)
    locations_count = len([l for l in locations_flat if l])
    locations_ok = 1 <= locations_count <= 5

    # License must not be blank; accept CC/ARR/Public Domain as 'ok', else 'unknown'
    license_eval = license_status(license_raw)
    license_ok = (license_eval == "ok") or (license_eval == "unknown")  # not blank is the hard rule
    license_missing = (license_eval == "missing")

    category_ok = bool(category_raw)

    # Subcategories: at least 1 (and keep the upper guard <=5)
    subcats = subcategories
    subcats_count = len([s for s in subcats if s])
    subcats_ok = 1 <= subcats_count <= 5

    # ---- Controlled vocabulary membership checks ----
    cv_issues = []

    # Normalise helpers
    def _lower_list(lst):
        return [x.lower() for x in lst if isinstance(x, str) and x.strip()]

    topics_l = _lower_list(topics)
    themes_l = _lower_list(themes)
    subcats_l = _lower_list(subcategories)
    languages_l = _lower_list(languages)
    locations_l = _lower_list(locations_flat)
    purposes_l = _lower_list(intended_purposes)
    category_l = category.strip().lower() if category else ""
    license_l = license_raw.strip().lower() if license_raw else ""

    # 0) Presence/cardinality hard rules
    #    - intended_purposes: >=1
    #    - license: exactly 1 (non-blank)
    #    - category: exactly 1 (non-blank)
    #    - subcategories: >=1 (upper bound already handled)
    if purposes_count < 1:
        cv_issues.append("intended_purposes missing (cardinality)")
    if not license_l:
        cv_issues.append("License missing (cardinality)")
    if not category_l:
        cv_issues.append("Category missing (cardinality)")
    if subcats_count < 1:
        cv_issues.append("At least one subcategory required (cardinality)")

    # 1) Membership: topics/themes can be length 1..N; each must be in list
    if CV.get("topics"):
        bad_topics = [t for t in topics_l if t not in CV["topics"]]
        if bad_topics:
            cv_issues.append("Unknown topics: " + ", ".join(sorted(set(bad_topics))))
    if CV.get("themes"):
        bad_themes = [t for t in themes_l if t not in CV["themes"]]
        if bad_themes:
            cv_issues.append("Unknown themes: " + ", ".join(sorted(set(bad_themes))))

    # 2) Category must be exactly one and in list
    cv_category_ok = True
    if category_l:
        if CV.get("category") and category_l not in CV["category"]:
            cv_category_ok = False
            cv_issues.append(f"Unknown category: {category}")
    else:
        cv_category_ok = False

    # 3) Subcategories: each must be in list; also parent must include category (if both present)
    bad_subcats = []
    parent_mismatch = []
    if CV.get("subcategories_all"):
        for s in subcats_l:
            if s not in CV["subcategories_all"]:
                bad_subcats.append(s)
            else:
                # parent check only if category is present and known
                if category_l and CV.get("subcat_parents"):
                    parents = CV["subcat_parents"].get(s, set())
                    if parents and (category_l not in parents):
                        parent_mismatch.append((s, list(parents)[:3]))
    if bad_subcats:
        cv_issues.append("Unknown subcategories: " + ", ".join(sorted(set(bad_subcats))))
    if parent_mismatch:
        msgs = [f"{s}→allowed:{'|'.join(p)}" for s, p in parent_mismatch]
        cv_issues.append("Subcategory parent/category mismatch: " + "; ".join(msgs))

    #    Validate membership against CV['languages'] if available
    bad_langs = []
    if CV.get("languages") and languages_l:
        for lg in languages_l:
            if lg not in CV["languages"]:
                bad_langs.append(lg)
        if bad_langs:
            cv_issues.append("Unknown languages: " + ", ".join(sorted(set(bad_langs))))

    # 5) Locations_flat: membership against CV['locations'] if available
    bad_locs = []
    if CV.get("locations") and locations_l:
        for loc in locations_l:
            if loc not in CV["locations"]:
                bad_locs.append(loc)
        if bad_locs:
            cv_issues.append("Unknown locations: " + ", ".join(sorted(set(bad_locs))))

    # 6) License: must be exactly one and in list (hard rule)
    cv_license_ok = True
    if license_l:
        if CV.get("license") and license_l not in CV["license"]:
            cv_license_ok = False
            cv_issues.append(f"Unknown license: {license_raw}")
    else:
        cv_license_ok = False

    # 7) Intended purposes: each must be in list (>=1 already checked)
    bad_purposes = []
    if CV.get("intended_purposes") and purposes_l:
        for p in purposes_l:
            if p not in CV["intended_purposes"]:
                bad_purposes.append(p)
        if bad_purposes:
            cv_issues.append("Unknown intended_purposes: " + ", ".join(sorted(set(bad_purposes))))

    total = sem_total + cr_total + cf_total + li_total

    # Flags and notes
    empty_fields = [k for k in ["title", "description", "ko_content_flat"] if not norm_text(ko.get(k))]
    notes = []
    if not url_ok and project_url_raw:
        notes.append("Invalid project_url")
    if not doi_ok and doi_raw:
        notes.append("Invalid DOI format")
    if lang.startswith("en") and miss_ratio > 0.05:
        notes.append(f"High misspelling ratio: {miss_ratio:.2%}")
    if len(tokens(content)) < 250:
        notes.append("Low content depth (<250 tokens)")
    if clean_issues["ocr_hyphen"] > 10:
        notes.append("OCR hyphenation artefacts detected")

    if not subtitle:
        notes.append("Subtitle missing")
    if subtitle_duplicate_title:
        notes.append("Subtitle duplicates title")
    if subtitle_duplicate_description:
        notes.append("Subtitle duplicates description")
    if description_duplicate_title:
        notes.append("Description duplicates title")

    if not creators_ok:
        notes.append("Creators missing")
    if not lang_match and declared_lang:
        notes.append(f"Language mismatch (declared={declared_lang}, detected={detected_base})")
    if not purposes_ok:
        notes.append("intended_purposes missing")
    if not locations_ok:
        notes.append("locations_flat missing or >5 items")

    if license_missing:
        notes.append("License missing")

    if (license_eval == "unknown") and not license_missing and license_raw:
        notes.append(f"Unrecognised license: {license_raw}")

    if not category_ok:
        notes.append("Category missing")
    if not subcats_ok:
        if subcats_count == 0:
            notes.append("At least one subcategory required")
        else:
            notes.append("Too many subcategories (>5)")

    # Controlled vocabulary inconsistencies
    if cv_issues:
        notes.append("CV inconsistencies: " + " | ".join(cv_issues))

    return {
        "_orig_id": _id,
        "title": title[:300],
        "lang_detected": lang,
        # Subscores for transparency
        "Semantic_Precision": sem_total,
        "Content_Richness": cr_total,
        "Cross_Field_Consistency": cf_total,
        "Linguistic_Integrity": li_total,
        "Total_Score": total,
        # Diagnostics
        "sp_title": sp_title,
        "sp_desc": sp_desc,
        "sp_keyword_anchoring": sp_kw,
        "cr_depth": cr_depth,
        "cr_diversity": cr_div,
        "cr_duplication": cr_dup,
        "cf_topics_themes": cf_tt,
        "cf_project_echo": cf_proj,
        "li_spell_score": li_spell,
        "li_misspell_ratio": round(miss_ratio, 4),
        "url_valid": url_ok,
        "project_url_fixed": fixed_url,
        "doi_valid": doi_ok,
        "empty_fields": ";".join(empty_fields),
        "notes": "; ".join(notes),
        "category": category,
        "subcategories": orjson.dumps(subcategories).decode("utf-8"),
        "keywords": orjson.dumps(keywords).decode("utf-8"),
        "topics": orjson.dumps(topics).decode("utf-8"),
        "themes": orjson.dumps(themes).decode("utf-8"),
        "subtitle_ok": subtitle_ok,
        "subtitle_overlap_with_title": round(ov, 3),
        "subtitle_duplicate_title": subtitle_duplicate_title,
        "subtitle_duplicate_description": subtitle_duplicate_description,
        "description_duplicate_title": description_duplicate_title,
        "creators_count": creators_count,
        "purposes_count": purposes_count,
        "locations_count": locations_count,
        "license_status": license_eval,
        "category_ok": category_ok,
        "subcategories_count": subcats_count,
        "lang_declared": declared_lang,
        "lang_match": lang_match,

        # CV diagnostics
        "cv_category_ok": cv_category_ok,
        "cv_license_ok": cv_license_ok,
        "cv_topics_unknown_count": len([t for t in topics_l if CV.get('topics') and t not in CV['topics']]),
        "cv_themes_unknown_count": len([t for t in themes_l if CV.get('themes') and t not in CV['themes']]),
        "cv_subcategories_unknown_count": len(
            [s for s in subcats_l if CV.get('subcategories_all') and s not in CV['subcategories_all']]),
        "cv_subcat_parent_mismatch_count": len(parent_mismatch),
        "cv_languages_unknown_count": len(bad_langs),
        "cv_locations_unknown_count": len(bad_locs),
        "cv_intended_purposes_unknown_count": len(bad_purposes),
    }
