"""Client for the deployed OpenSearch retrieval API.

Including the live system as an arm of the experiment is what turns this from
"which paradigm suits this corpus" into "does the deployed system realise it".
The endpoint also reports, per query, the intent router's mode and whether it
enabled the semantic half - so the router itself can be evaluated.

Credentials come from existing service .env files; nothing is hardcoded here.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import random
import threading
import time
import urllib.error
import urllib.request

from typing import Any

log = logging.getLogger("live_search")

DEFAULT_ENDPOINT = "https://api.opensearch.nexavion.com"


class LiveSearchClient:
    def __init__(
        self,
        base_url: str,
        basic_user: str,
        basic_pass: str,
        proxy_token: str,
        cache_dir: str,
        model: str = "mlang_minilm",
        path: str = "neural_search_relevant",
        timeout: float = 60.0,
        max_retries: int = 4,
        session_id: str = "semantic-vs-keyword-eval",
    ) -> None:
        self.url = base_url.rstrip("/") + "/" + path.lstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Basic " + base64.b64encode(
                f"{basic_user}:{basic_pass}".encode()
            ).decode(),
            "x-internal-proxy-token": proxy_token,
            "x-search-session-id": session_id,
        }
        self._lock = threading.Lock()
        self.calls = 0
        self.cache_hits = 0
        self.failures = 0

    # ------------------------------------------------------------------
    def search(self, term: str, size: int = 10, dev: bool = False,
               ui_locale: str | None = None) -> dict[str, Any]:
        """Return {'ids': [...], 'mode': str, 'use_semantic': bool, 'raw_n': int}."""
        body = {
            "search_term": term,
            "model": self.model,
            "page": 1,
            "size": size,
            "dev": dev,
            "include_semantic_score": True,
        }
        # scout only queries the translated *_i18n.<locale> fields when it knows
        # the caller's locale, so this changes the lexical leg substantially.
        if ui_locale:
            body["ui_locale"] = ui_locale
        digest = hashlib.sha256(
            json.dumps({"u": self.url, **body}, sort_keys=True).encode()
        ).hexdigest()[:24]
        path = os.path.join(self.cache_dir, f"{digest}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                with self._lock:
                    self.cache_hits += 1
                return json.load(fh)

        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request = urllib.request.Request(
                    self.url, data=json.dumps(body).encode(), headers=self.headers
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                result = self._parse(payload)
                with self._lock:
                    self.calls += 1
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(result, fh)
                return result
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise
                last = exc
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(min(15.0, (2 ** attempt) * 0.7) * (0.6 + 0.8 * random.random()))

        with self._lock:
            self.failures += 1
        log.warning("live search failed for %r: %s", term[:60], last)
        return {"ids": [], "snippets": {}, "mode": "error", "use_semantic": None,
                "profile_use_semantic": None, "selected_label": None, "attempts": 0, "raw_n": 0}

    # ------------------------------------------------------------------
    @staticmethod
    def _parse(payload: dict[str, Any]) -> dict[str, Any]:
        hits = payload.get("data") or payload.get("results") or payload.get("hits") or []
        ids: list[str] = []
        snippets: dict[str, str] = {}
        for hit in hits:
            # The parent document id. _orig_id carries a "::cN" chunk suffix,
            # so it never joins against an export keyed by document.
            doc_id = hit.get("_id") or hit.get("ko_id") or hit.get("_orig_id") or hit.get("@id")
            if not doc_id:
                continue
            doc_id = str(doc_id).split("::", 1)[0]
            ids.append(doc_id)
            # Keep the text the API itself returned, so a live hit can be judged
            # even when it is outside the locally sampled corpus.
            parts = [
                str(hit.get("projectDisplayName") or hit.get("projectName") or ""),
                str(hit.get("title") or ""),
                str(hit.get("description") or hit.get("description_original") or ""),
                str(hit.get("ko_content_flat_summarised") or ""),
            ]
            snippets[doc_id] = " ".join(p for p in parts if p).strip()
        meta = (payload.get("_meta") or {}).get("default_search") or {}
        selected = meta.get("selected_attempt") or {}
        # `default_search.use_semantic` is the PROFILE's default, not what ran.
        # The cascade may fall through to a semantic attempt after a lexical one
        # returns nothing, so only the selected attempt says what was executed.
        # Reading the profile field instead reports "semantic never ran" even
        # when it did.
        return {
            "ids": ids,
            "snippets": snippets,
            "mode": meta.get("query_mode"),
            "use_semantic": selected.get("use_semantic"),
            "profile_use_semantic": meta.get("use_semantic"),
            "selected_label": selected.get("label"),
            "attempts": len(meta.get("attempts") or []),
            "rewritten": meta.get("rewritten_query"),
            "raw_n": len(hits),
        }

    def usage_summary(self) -> str:
        return (
            f"{self.calls} live searches ({self.cache_hits} from cache, "
            f"{self.failures} failed)"
        )


def build_live_client_from_env(
    env_files: dict[str, str],
    cache_dir: str,
    base_url: str | None = None,
    model: str = "mlang_minilm",
) -> LiveSearchClient | None:
    """Assemble credentials from whichever service .env files carry them.

    env_files maps a label to a path; all are merged, first value wins.
    Returns None when the endpoint is not configured, so the experiment can
    still run purely locally.
    """
    merged: dict[str, str] = {}
    for path in env_files.values():
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    merged.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    url = base_url or merged.get("OPENSEARCH_API_URL") or DEFAULT_ENDPOINT
    user = merged.get("OPENSEARCH_API_USR") or merged.get("BASIC_AUTH_USER") or ""
    password = merged.get("OPENSEARCH_API_PWD") or merged.get("BASIC_AUTH_PASS") or ""
    token = merged.get("EUF_OPENSEARCH_TRUSTED_PROXY_TOKEN") or ""
    if not (url and user and password):
        return None
    if url.startswith("http") is False:
        url = "https://" + url
    return LiveSearchClient(url, user, password, token, cache_dir, model=model)
