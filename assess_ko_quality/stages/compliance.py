# assess_ko_quality/stages/compliance.py
"""
Stage 1: compliance checks.

These are requirements, not quality signals, so they are reported as a pass/fail
checklist and deliberately kept OUT of the quality score. That separation is not
stylistic: when metadata completeness was a scored component it correlated with
reviewer judgement at -0.217 (p=0.039), the wrong direction, because a thorough record
is not the same thing as good content. Completeness still matters - as an obligation.

Nothing here is a fixed list.

  - The controlled vocabularies are discovered by globbing the data-model folder, so a
    new data_model.*.json is checked automatically and a removed one stops being
    checked. The field name comes from the filename.
  - Which fields count as "expected" is learned from the corpus itself: a field that
    the overwhelming majority of records populate is treated as expected, because that
    is what the platform demonstrably requires in practice. Run `learn_profile` once
    per export and the checklist follows the data rather than an opinion.
  - Coverage alone cannot tell a contributor obligation from something one of our own
    pipelines wrote. `title_llm` and `ko_content_flat_vision` are populated nearly
    everywhere, but failing a KO because OUR enrichment did not run would be nonsense.
    So field roles are classified by an LLM from the field names and sample values, and
    only contributor-supplied fields become obligations. The classification is stored in
    the profile, so it is auditable and costs nothing to re-use.
  - Staleness is a corpus percentile, not a number someone chose.

Without a profile the module still runs, checking only vocabulary conformance and
whatever the caller passes in, rather than inventing requirements.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

HERE = Path(__file__).resolve().parent
DEFAULT_PROFILE = HERE.parent / "validation" / "compliance_profile.json"

# Where the controlled vocabularies live. Gitignored, so every CV check degrades to
# "skipped" rather than "failed" when the folder is absent.
CV_DIR = Path(os.environ.get("KO_DM_DIR", HERE.parents[2] / "data_model_v2"))
CV_GLOB = "data_model.*.json"

_CV_CACHE: Dict[str, Optional[Set[str]]] = {}
_PROFILE_CACHE: Dict[str, Any] = {}


# --------------------------------------------------------------------- vocabularies
def discover_vocabularies(cv_dir: Path = CV_DIR) -> Dict[str, Path]:
    """Map field name -> vocabulary file, from whatever the folder contains."""
    out: Dict[str, Path] = {}
    if not cv_dir.is_dir():
        return out
    for path in sorted(cv_dir.glob(CV_GLOB)):
        field = path.name.split(".", 2)[1] if path.name.count(".") >= 2 else path.stem
        out[field] = path
    return out


def _load_cv(field: str, path: Path) -> Optional[Set[str]]:
    key = str(path)
    if key in _CV_CACHE:
        return _CV_CACHE[key]
    values: Optional[Set[str]] = None
    try:
        acc: List[str] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ("name", "label", "title", "prefLabel", "value") and isinstance(v, str):
                        acc.append(v)
                    else:
                        walk(v)
            elif isinstance(node, list):
                for x in node:
                    walk(x)

        walk(json.loads(path.read_text(encoding="utf-8")))
        values = {v.strip().lower() for v in acc if v.strip()} or None
    except Exception:
        values = None
    _CV_CACHE[key] = values
    return values


# --------------------------------------------------------------------- helpers
def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple, set)):
        return [str(x) for x in v if x is not None and str(x).strip()]
    return [str(v)] if str(v).strip() else []


def _populated(v: Any) -> bool:
    return bool(_as_list(v))


_YEAR = re.compile(r"(19|20)\d{2}")


def _year_of(value: Any) -> Optional[int]:
    for text in _as_list(value):
        m = _YEAR.search(text)
        if m:
            return int(m.group(0))
    return None


# --------------------------------------------------------------------- profile
FIELD_ROLE_SYSTEM = (
    "You are classifying metadata fields of a knowledge-sharing platform for farmers "
    "and foresters. Decide who is responsible for each field's value."
)


def classify_field_roles(
    fields: Sequence[str],
    samples: Dict[str, List[str]],
    client: Any = None,
) -> Dict[str, str]:
    """
    Classify each field as contributor / system / derived, using an LLM.

    contributor - a person uploading the object supplies it (title, description, licence)
    system      - the platform assigns it (ids, timestamps, storage keys)
    derived     - one of our own pipelines computed it (translations, *_llm, extracted text)

    Only 'contributor' fields become compliance obligations: the others are either
    guaranteed by the platform or are our problem, not the contributor's. Returns {} when
    no LLM is configured, in which case the caller keeps every field it learned.
    """
    if client is None:
        from core.llm import get_client
        client = get_client()
    if client is None:
        return {}

    lines = []
    for f in fields:
        ex = "; ".join(str(v)[:60] for v in samples.get(f, [])[:2])
        lines.append(f"- {f}: {ex[:140]}" if ex else f"- {f}:")
    prompt = (
        "Classify each metadata field as exactly one of: contributor, system, derived.\n\n"
        "contributor = a person uploading the knowledge object provides this\n"
        "system      = the platform generates it automatically (identifiers, timestamps, "
        "storage locations, internal bookkeeping)\n"
        "derived     = a downstream pipeline computed it from the object "
        "(machine translation, LLM-rewritten metadata, extracted or summarised text, "
        "vision output, embeddings, computed statistics)\n\n"
        "FIELDS (name: example values)\n" + "\n".join(lines) +
        '\n\nReturn JSON: {"field_name": "contributor|system|derived", ...} for every field.'
    )
    out = client.complete_json(prompt, max_tokens=4000, system=FIELD_ROLE_SYSTEM)
    if not isinstance(out, dict):
        return {}
    valid = {"contributor", "system", "derived"}
    return {k: str(v).strip().lower() for k, v in out.items()
            if k in set(fields) and str(v).strip().lower() in valid}


def learn_profile(
    records: Iterable[Dict[str, Any]],
    expected_at: float = 0.95,
    stale_percentile: float = 0.05,
    cv_dir: Path = CV_DIR,
    classify_roles: bool = True,
    client: Any = None,
) -> Dict[str, Any]:
    """
    Derive the checklist from a corpus instead of asserting it.

    A field populated in at least `expected_at` of records is treated as expected: the
    platform evidently requires it, whatever any spec says. The staleness cut-off is the
    `stale_percentile` quantile of completion years, so "stale" means old relative to
    this corpus rather than older than a number someone picked.
    """
    records = list(records)
    total = len(records)
    if not total:
        raise ValueError("learn_profile needs at least one record")

    counts: Counter = Counter()
    for rec in records:
        for field, value in rec.items():
            if _populated(value):
                counts[field] += 1

    coverage = {f: counts[f] / total for f in counts}
    well_covered = sorted(f for f, c in coverage.items() if c >= expected_at)

    # Sample values so the classifier sees what a field actually holds, not just its name.
    samples: Dict[str, List[str]] = {}
    for rec in records[:40]:
        for f in well_covered:
            if len(samples.setdefault(f, [])) < 2 and _populated(rec.get(f)):
                samples[f].append(", ".join(_as_list(rec.get(f))[:3]))

    roles = classify_field_roles(well_covered, samples, client=client) if classify_roles else {}
    # Without a classification every well-covered field stays in, which is the
    # conservative choice: over-reporting is visible, silently dropping a duty is not.
    expected = sorted(f for f in well_covered if roles.get(f, "contributor") == "contributor")

    years = sorted(y for y in (_year_of(r.get("date_of_completion") or r.get("ko_created_at"))
                               for r in records) if y)
    stale_before = years[int(len(years) * stale_percentile)] if years else None

    return {
        "built": datetime.now(timezone.utc).date().isoformat(),
        "records": total,
        "expected_at": expected_at,
        "expected_fields": expected,
        "field_roles": roles,
        "well_covered_fields": well_covered,
        "coverage": {f: round(c, 4) for f, c in sorted(coverage.items(), key=lambda kv: -kv[1])},
        "stale_before_year": stale_before,
        "stale_percentile": stale_percentile,
        "vocabularies": sorted(discover_vocabularies(cv_dir)),
    }


def save_profile(profile: Dict[str, Any], path: Path = DEFAULT_PROFILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=1), encoding="utf-8")
    return path


def load_profile(path: Path = DEFAULT_PROFILE) -> Optional[Dict[str, Any]]:
    key = str(path)
    if key not in _PROFILE_CACHE:
        try:
            _PROFILE_CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            _PROFILE_CACHE[key] = None
    return _PROFILE_CACHE[key]


# --------------------------------------------------------------------- checks
def check(
    ko: Dict[str, Any],
    profile: Optional[Dict[str, Any]] = None,
    cv_dir: Path = CV_DIR,
) -> Dict[str, Any]:
    """
    Run the checklist over one KO.

    Each check is True (pass), False (fail) or None (not evaluable - no vocabulary
    available, or no profile to say the field is expected). None never counts as a fail.
    """
    profile = profile if profile is not None else load_profile()
    checks: Dict[str, Optional[bool]] = {}

    # Expected fields, learned from the corpus.
    if profile and profile.get("expected_fields"):
        for field in profile["expected_fields"]:
            checks[f"present_{field}"] = _populated(ko.get(field))

    # Vocabulary conformance, for whatever vocabularies exist on disk.
    for field, path in discover_vocabularies(cv_dir).items():
        allowed = _load_cv(field, path)
        values = _as_list(ko.get(field))
        if allowed is None:
            checks[f"cv_{field}"] = None
        elif not values:
            # Only a failure if the corpus says this field is expected.
            expected = bool(profile and field in (profile.get("expected_fields") or []))
            checks[f"cv_{field}"] = False if expected else None
        else:
            checks[f"cv_{field}"] = all(v.strip().lower() in allowed for v in values)

    # Timeliness, against a corpus-derived cut-off.
    stale_before = (profile or {}).get("stale_before_year")
    year = _year_of(ko.get("date_of_completion") or ko.get("ko_created_at"))
    if stale_before is None or year is None:
        checks["not_stale"] = None
    else:
        checks["not_stale"] = year >= stale_before

    evaluated = {k: v for k, v in checks.items() if v is not None}
    failed = sorted(k for k, v in evaluated.items() if not v)

    return {
        "compliance_checks": checks,
        "compliance_passed": sum(1 for v in evaluated.values() if v),
        "compliance_evaluated": len(evaluated),
        "compliance_failed": failed,
        "compliance_skipped": sorted(k for k, v in checks.items() if v is None),
        "compliance_status": "complete" if not failed else "incomplete",
        "compliance_profile_used": bool(profile),
    }


__all__ = [
    "check", "learn_profile", "save_profile", "load_profile",
    "discover_vocabularies", "CV_DIR", "DEFAULT_PROFILE",
]
