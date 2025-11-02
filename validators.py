# validators.py

import re

from typing import Optional, Dict, List, Tuple
import urllib.parse

from text_utils import regexes, tokens

_URL, _OCR_SPLIT, _NON_ASCII = regexes()
_DOI_PAT = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

def validate_doi(s: Optional[str]) -> bool:
    s = (s or "").strip()
    return bool(_DOI_PAT.match(s))

def normalise_url(s: Optional[str]) -> tuple[str, bool]:
    s = (s or "").strip()
    if not s: return "", False
    if not s.lower().startswith(("http://", "https://")):
        s = "https://" + s
    parsed = urllib.parse.urlparse(s)
    ok = bool(parsed.scheme and parsed.netloc)
    return s, ok

_LANG_NAME_TO_CODE = {
    "english":"en","en":"en","en-gb":"en","en-us":"en",
    "dutch":"nl","nederlands":"nl","nl":"nl",
    "french":"fr","français":"fr","fr":"fr",
    "german":"de","deutsch":"de","de":"de",
    "spanish":"es","español":"es","es":"es",
    "italian":"it","italiano":"it","it":"it",
    "portuguese":"pt","português":"pt","pt":"pt",
}
def normalise_lang_label(s: Optional[str]) -> str:
    if not s: return ""
    key = s.strip().lower().replace("_","-")
    return _LANG_NAME_TO_CODE.get(key, key)

_CC_LIC_PAT = re.compile(r"^(cc0|cc\s*by|cc\s*by-?sa|cc\s*by-?nc|cc\s*by-?nd|cc\s*by-?nc-?sa|cc\s*by-?nc-?nd)$", re.IGNORECASE)
def license_status(s: Optional[str]) -> str:
    s = (s or "").strip()
    if not s: return "missing"
    if s.lower().strip() in {"all rights reserved", "public domain"}: return "ok"
    if _CC_LIC_PAT.match(s.replace("–","-").replace("—","-").replace(" ","")): return "ok"
    if s.lower().startswith("cc "): return "ok"
    return "unknown"

def cleanliness_score(texts: List[str]) -> Tuple[int, Dict[str,int]]:
    s = " ".join(texts)
    issues = {
        "non_ascii": len(_NON_ASCII.findall(s)),
        "ocr_hyphen": len(_OCR_SPLIT.findall(" ".join(texts))),
        "url_count": len(_URL.findall(s)),
        "all_caps_runs": len(re.findall(r"\b[A-Z]{6,}\b", s)),
        "weird_spaces": len(re.findall(r"\S\s{3,}\S", s)),
    }
    score = 6
    if issues["non_ascii"] > 20: score -= 2
    if issues["ocr_hyphen"] > 10: score -= 2
    if issues["url_count"] > 25: score -= 1
    if issues["all_caps_runs"] > 10: score -= 1
    if issues["weird_spaces"] > 5: score -= 1
    return max(0, score), issues

def structure_punct_score(texts: List[str]) -> int:
    """
    Very light structure/grammar proxy:
    - split on sentence enders . ! ?
    - compute average sentence length (in tokens)
    - map to a 1..6 score (middle ranges rewarded)
    """
    s = " ".join(texts)
    sents = re.split(r"[.!?]\s+", s)
    lens = [len(tokens(x)) for x in sents if x.strip()]
    if not lens:
        return 4  # neutral if we can't segment
    avg = sum(lens) / len(lens)
    if 8 <= avg <= 35: return 6
    if 6 <= avg <= 45: return 4
    if 4 <= avg <= 55: return 2
    return 1