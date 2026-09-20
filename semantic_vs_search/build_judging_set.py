"""
Turn real logged searches into a human judging sheet.

Every analysis in this folder has bottomed out at the same gap: the queries are
invented and the relevance grades come from an LLM. Neither needs to stay true.
scout already logs every search to ClickHouse, so the queries users actually
typed exist - this script samples them, runs each one, and writes a spreadsheet
someone can grade.

The output feeds `evaluate_field_bundles.py`, which has been waiting for judged
queries since it was written.

Stage 1 (on the server, where ClickHouse is reachable):
    python build_judging_set.py export --out queries.json

Stage 2 (anywhere, with the search API reachable):
    python build_judging_set.py sheet --queries queries.json --out judging.xlsx

Stage 3 (after a human fills the `grade` column):
    python build_judging_set.py convert --sheet judging.xlsx --out judgments.json
    python ../which_fields_to_choose/evaluate_field_bundles.py \
        --input ../input/<export>.json --judgments judgments.json --id-field _id

Nothing about the sampling is hardcoded to this corpus: the strata are computed
from the logged distribution, so a corpus with different languages or query
shapes yields a proportionate sample without editing this file.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("build_judging_set")

# Requests that are not a person looking for something.
_BOT_HINTS = ("bot", "crawl", "spider", "scrap", "curl", "wget", "python-requests",
              "postman", "monitor", "uptime", "headless")


def load_env(*paths: str) -> dict[str, str]:
    """Merge .env files, then fall back to the real environment.

    Running inside the service container is the only place the ClickHouse
    hostname resolves, and there the credentials arrive as environment
    variables rather than as a file - the image deliberately excludes .env.
    Falling back to os.environ means the container route needs no arguments,
    while an explicitly passed file still wins.
    """
    merged: dict[str, str] = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    merged.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    for key in ("CLICKHOUSE_URL", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD",
                "CLICKHOUSE_DB", "OPENSEARCH_API_USR", "OPENSEARCH_API_PWD",
                "BASIC_AUTH_USER", "BASIC_AUTH_PASS",
                "EUF_OPENSEARCH_TRUSTED_PROXY_TOKEN"):
        if not merged.get(key) and os.getenv(key):
            merged[key] = os.environ[key]
    return merged


# ------------------------------------------------------------------ export ---

def clickhouse_query(env: dict[str, str], sql: str, timeout: float = 60.0) -> str:
    base = (env.get("CLICKHOUSE_URL") or "http://euf_search_clickhouse:8123").rstrip("/")
    db = env.get("CLICKHOUSE_DB") or "euf_search_analytics"
    url = f"{base}/?{urllib.parse.urlencode({'database': db, 'query': sql})}"
    auth = base64.b64encode(
        f"{env.get('CLICKHOUSE_USER','')}:{env.get('CLICKHOUSE_PASSWORD','')}".encode()
    ).decode()
    request = urllib.request.Request(url, headers={"Authorization": "Basic " + auth})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def looks_like_a_person(row: dict[str, Any], min_chars: int, max_chars: int) -> bool:
    query = (row.get("original_query") or "").strip()
    if not (min_chars <= len(query) <= max_chars):
        return False
    if not re.search(r"[^\W\d_]", query, re.UNICODE):   # must contain a letter
        return False
    agent = (row.get("user_agent") or "").lower()
    if any(hint in agent for hint in _BOT_HINTS):
        return False
    return True


def stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Sample proportionally across the strata the data actually exhibits.

    Strata are (language, outcome) where outcome buckets the result count into
    none / few / many. Proportions come from the logs, so a sample reflects real
    usage rather than an assumption about it - including, importantly, the
    zero-result searches that a purely successful-query sample would hide.
    """
    def outcome(row: dict[str, Any]) -> str:
        n = int(row.get("results_count") or 0)
        return "none" if n == 0 else ("few" if n < 5 else "many")

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[((row.get("detected_lang") or "unknown")[:2], outcome(row))].append(row)

    rng = random.Random(seed)
    total = sum(len(v) for v in buckets.values()) or 1
    picked: list[dict[str, Any]] = []
    # Largest strata first, so rounding losses fall on the rarest combinations.
    for key in sorted(buckets, key=lambda k: -len(buckets[k])):
        share = len(buckets[key]) / total
        want = max(1, round(size * share))
        pool = buckets[key]
        rng.shuffle(pool)
        picked.extend(pool[:want])
    rng.shuffle(picked)
    return picked[:size]


def cmd_export(args: argparse.Namespace) -> None:
    env = load_env(*args.env)
    sql = f"""
        SELECT original_query, detected_lang, results_count, user_agent,
               search_source, endpoint, count() AS times
        FROM {args.table}
        WHERE ts >= now() - INTERVAL {int(args.days)} DAY
          AND status_code = 200
          AND original_query != ''
        GROUP BY original_query, detected_lang, results_count, user_agent,
                 search_source, endpoint
        ORDER BY times DESC
        LIMIT {int(args.limit)}
        FORMAT JSONEachRow
    """
    log.info("Querying ClickHouse for the last %d days ...", args.days)
    try:
        raw = clickhouse_query(env, sql)
    except urllib.error.URLError as exc:
        host = (env.get("CLICKHOUSE_URL") or "http://euf_search_clickhouse:8123")
        sys.exit(
            f"Could not reach ClickHouse at {host}: {exc.reason}\n\n"
            "That hostname usually only resolves inside the compose network. Either\n"
            "run this from within the service container, where the credentials are\n"
            "already in the environment and no arguments are needed:\n"
            "    docker compose cp build_judging_set.py scout_search_api:/tmp/bjs.py\n"
            "    docker compose exec scout_search_api python /tmp/bjs.py export \\\n"
            "        --out /tmp/real_queries.json --days 90 --size 50\n\n"
            "or point CLICKHOUSE_URL at a reachable address, e.g. a published port\n"
            "or an SSH tunnel:\n"
            "    CLICKHOUSE_URL=http://127.0.0.1:8123 python3 build_judging_set.py export ..."
        )
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    log.info("  %d distinct logged searches", len(rows))

    people = [r for r in rows if looks_like_a_person(r, args.min_chars, args.max_chars)]
    log.info("  %d after removing bots and non-queries", len(people))

    # One row per distinct query text; keep the most frequent variant.
    best: dict[str, dict[str, Any]] = {}
    for row in sorted(people, key=lambda r: -int(r.get("times") or 0)):
        key = (row.get("original_query") or "").strip().lower()
        best.setdefault(key, row)
    unique = list(best.values())
    log.info("  %d distinct query texts", len(unique))

    sample = stratified_sample(unique, args.size, args.seed)
    langs = Counter((r.get("detected_lang") or "?")[:2] for r in sample)
    zero = sum(1 for r in sample if int(r.get("results_count") or 0) == 0)
    log.info("Sampled %d queries | languages: %s | zero-result: %d",
             len(sample), dict(langs.most_common(8)), zero)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(sample, fh, ensure_ascii=False, indent=2)
    log.info("Wrote %s", args.out)
    print(f"\nNext: copy {args.out} to a machine that can reach the search API, then run")
    print(f"  python build_judging_set.py sheet --queries {args.out} --out judging.xlsx")


# ------------------------------------------------------------------- sheet ---

def search(env: dict[str, str], term: str, size: int, base_url: str, model: str) -> list[dict[str, Any]]:
    body = json.dumps({"search_term": term, "model": model, "page": 1,
                       "size": size, "dev": False}).encode()
    auth = base64.b64encode(
        f"{env.get('OPENSEARCH_API_USR') or env.get('BASIC_AUTH_USER','')}:"
        f"{env.get('OPENSEARCH_API_PWD') or env.get('BASIC_AUTH_PASS','')}".encode()
    ).decode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/neural_search_relevant", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Basic " + auth,
                 "x-internal-proxy-token": env.get("EUF_OPENSEARCH_TRUSTED_PROXY_TOKEN", ""),
                 "x-search-session-id": "judging-set-build"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return (json.loads(response.read().decode()).get("data") or [])


def cmd_sheet(args: argparse.Namespace) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        sys.exit("openpyxl is required for the sheet step: pip install openpyxl")

    env = load_env(*args.env)
    with open(args.queries, "r", encoding="utf-8") as fh:
        queries = json.load(fh)

    wb = Workbook()
    ws = wb.active
    ws.title = "judging"
    headers = ["query", "lang", "rank", "grade", "title", "project", "description", "doc_id"]
    ws.append(headers)
    head_fill = PatternFill("solid", fgColor="1F5C4D")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = head_fill
    ws.freeze_panes = "A2"

    grade_rule = DataValidation(type="list", formula1='"3,2,1,0"', allow_blank=True)
    grade_rule.error = "Grade must be 3, 2, 1 or 0"
    grade_rule.prompt = ("3 = answers the query   2 = clearly useful   "
                         "1 = marginal   0 = not relevant")
    ws.add_data_validation(grade_rule)

    band = False
    row_no = 2
    for item in queries:
        term = (item.get("original_query") or "").strip()
        if not term:
            continue
        try:
            hits = search(env, term, args.topk, args.base_url, args.model)
        except Exception as exc:  # noqa: BLE001 - one bad query must not stop the build
            log.warning("search failed for %r: %s", term[:50], exc)
            hits = []
        band = not band
        shade = PatternFill("solid", fgColor="F2F5F4") if band else None
        if not hits:
            # Zero-result searches are the most valuable rows in the sheet: they
            # are how the no-match behaviour gets evaluated at all.
            ws.append([term, (item.get("detected_lang") or "")[:2], 0, "", "(no results returned)",
                       "", "", ""])
            if shade:
                for cell in ws[row_no]:
                    cell.fill = shade
            row_no += 1
            continue
        for rank, hit in enumerate(hits, start=1):
            ws.append([
                term if rank == 1 else "",
                (item.get("detected_lang") or "")[:2] if rank == 1 else "",
                rank, "",
                str(hit.get("title") or "")[:300],
                str(hit.get("projectDisplayName") or hit.get("projectName") or "")[:120],
                str(hit.get("description") or "")[:600],
                str(hit.get("_id") or ""),
            ])
            grade_rule.add(ws.cell(row=row_no, column=4))
            if shade:
                for cell in ws[row_no]:
                    cell.fill = shade
            row_no += 1

    widths = {"A": 38, "B": 6, "C": 6, "D": 8, "E": 52, "F": 24, "G": 70, "H": 26}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=cell.column_letter in ("E", "G"))

    guide = wb.create_sheet("how to grade")
    for line in [
        ["How to grade"],
        [""],
        ["Fill the `grade` column for every row. Leave nothing blank."],
        [""],
        ["3", "Answers the query. This is what the person was looking for."],
        ["2", "Clearly useful and substantially on the topic, but not the answer."],
        ["1", "Related, only marginally useful."],
        ["0", "Not relevant."],
        [""],
        ["Grade against the QUERY, not against the other results."],
        ["A row reading '(no results returned)' needs no grade - it records that the"],
        ["search found nothing, which is itself a measurement."],
        [""],
        ["Two people grading the same sheet independently is worth the extra effort:"],
        ["their agreement rate tells you how much any of these numbers can be trusted."],
    ]:
        guide.append(line)
    guide.column_dimensions["A"].width = 12
    guide.column_dimensions["B"].width = 90
    guide["A1"].font = Font(bold=True, size=14, color="1F5C4D")

    wb.save(args.out)
    graded_rows = row_no - 2
    log.info("Wrote %s - %d queries, %d rows to grade", args.out, len(queries), graded_rows)
    print(f"\nHand {args.out} to whoever knows the domain. When it comes back:")
    print(f"  python build_judging_set.py convert --sheet {args.out} --out judgments.json")


# ----------------------------------------------------------------- convert ---

def cmd_convert(args: argparse.Namespace) -> None:
    try:
        from openpyxl import load_workbook
    except ImportError:
        sys.exit("openpyxl is required for the convert step: pip install openpyxl")

    ws = load_workbook(args.sheet, data_only=True)["judging"]
    judgments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    graded = blank = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        term, _lang, _rank, grade, _title, _proj, _desc, doc_id = (list(row) + [None] * 8)[:8]
        if term:
            current = {"query": str(term).strip(), "relevant": {}}
            judgments.append(current)
        if current is None or not doc_id:
            continue
        if grade is None or str(grade).strip() == "":
            blank += 1
            continue
        try:
            value = int(str(grade).strip())
        except ValueError:
            blank += 1
            continue
        graded += 1
        if value > 0:
            current["relevant"][str(doc_id)] = value

    judgments = [j for j in judgments if j["relevant"]]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(judgments, fh, ensure_ascii=False, indent=2)

    sizes = [len(j["relevant"]) for j in judgments] or [0]
    log.info("Wrote %s", args.out)
    log.info("  %d queries with at least one relevant document", len(judgments))
    log.info("  %d graded cells, %d left blank", graded, blank)
    log.info("  mean relevant per query: %.1f", sum(sizes) / len(sizes))
    if blank:
        log.warning("  %d blank grades were skipped - they count as 'not judged', not 'not relevant'", blank)
    print(f"\nNow run the judged evaluation:")
    print(f"  python which_fields_to_choose/evaluate_field_bundles.py \\")
    print(f"      --input input/<export>.json --judgments {args.out} --id-field _id")
    print("  (--id-field _id, not @id: the sheet records the parent document id the")
    print("   search API returns, which joins to the export's _id.)")


# -------------------------------------------------------------------- main ---

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.abspath(os.path.join(here, "..", ".."))
    default_env = [os.path.join(workspace, "scout", ".env"),
                   os.path.join(workspace, "farm_assistant_um", ".env")]

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    env_help = "One or more .env files to read credentials from"
    parser.add_argument("--env", action="append", default=None,
                        help=env_help + " (repeat the flag for several)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_env(p):
        """Accept --env on the subcommand too.

        argparse only honours a parent-parser option before the subcommand, and
        the natural spelling puts it after. Declaring it in both places makes
        either order work; `action="append"` keeps it from consuming the
        subcommand name.
        """
        p.add_argument("--env", action="append", default=None, help=env_help)
        return p

    p = sub.add_parser("export", help="Sample real queries from the ClickHouse search log")
    p.add_argument("--out", default="real_queries.json")
    p.add_argument("--table", default="search_events")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--limit", type=int, default=50000, help="Log rows to consider")
    p.add_argument("--size", type=int, default=50, help="Queries to sample")
    p.add_argument("--min-chars", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=120)
    p.add_argument("--seed", type=int, default=13)
    add_env(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("sheet", help="Run the queries and write a judging spreadsheet")
    p.add_argument("--queries", default="real_queries.json")
    p.add_argument("--out", default="judging.xlsx")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--model", default="mlang_minilm")
    p.add_argument("--base-url", default="https://api.opensearch.nexavion.com")
    add_env(p)
    p.set_defaults(func=cmd_sheet)

    p = sub.add_parser("convert", help="Turn a graded spreadsheet into judgments JSON")
    p.add_argument("--sheet", default="judging.xlsx")
    p.add_argument("--out", default="judgments.json")
    add_env(p)
    p.set_defaults(func=cmd_convert)

    args = parser.parse_args()
    # A subcommand --env wins; otherwise fall back to the parent's default.
    if not getattr(args, "env", None):
        args.env = default_env
    args.func(args)


if __name__ == "__main__":
    main()
