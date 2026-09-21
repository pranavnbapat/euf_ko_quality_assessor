"""
What users' behaviour says about search quality, when nobody can grade results.

Relevance judgement is expensive and this platform cannot always buy it. But
every search is logged, and people leave evidence of whether a search worked:
they search again, they go to page two, they repeat themselves, they give up.
None of that is as good as a graded result list. All of it is free, it covers
tens of thousands of searches rather than fifty, and it comes from real intent.

Three things this produces:

1. A failure rate per query, per language and per query shape, derived from
   reformulation, pagination and abandonment.
2. A ranked list of the worst-performing real queries - the ones users fought
   with - which is a work queue for improving retrieval.
3. A correlation between those signals and an LLM panel's grades, which is the
   closest thing to validating the panel that exists without a human.

What it cannot do, stated plainly: reformulation is confounded. Someone may
search again because they found what they wanted and moved on to the next
question. These signals measure aggregate dissatisfaction, not per-document
relevance, and a single query's rate means nothing. Read them in bulk.

Stage 1 (on the server, where ClickHouse is reachable):
    python behavioural_signals.py export --out sessions.json --days 90

Stage 2 (anywhere):
    python behavioural_signals.py analyse --sessions sessions.json
    python behavioural_signals.py validate --sessions sessions.json --judged judged.xlsx
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

logging.basicConfig(level=os.getenv("LOGLEVEL", "INFO"),
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("behavioural_signals")

_BOT_HINTS = ("bot", "crawl", "spider", "scrap", "curl", "wget", "python-requests",
              "postman", "monitor", "uptime", "headless")


def load_env(*paths: str) -> dict[str, str]:
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
    for key in ("CLICKHOUSE_URL", "CLICKHOUSE_USER", "CLICKHOUSE_PASSWORD", "CLICKHOUSE_DB"):
        if not merged.get(key) and os.getenv(key):
            merged[key] = os.environ[key]
    return merged


# ------------------------------------------------------------------ export ---

def cmd_export(args: argparse.Namespace) -> None:
    env = load_env(*args.env)
    base = (env.get("CLICKHOUSE_URL") or "http://euf_search_clickhouse:8123").rstrip("/")
    db = env.get("CLICKHOUSE_DB") or "euf_search_analytics"
    # Ordered by actor then time so sessions can be reconstructed by scanning.
    # user_uuid when present, ip otherwise: there is no session id in the schema,
    # and an ip is a coarse but workable stand-in for one person's sitting.
    sql = f"""
        SELECT ts, original_query, detected_lang, results_count, page,
               user_uuid, ip, user_agent, search_source, endpoint, total_records
        FROM {args.table}
        WHERE ts >= now() - INTERVAL {int(args.days)} DAY
          AND status_code = 200
          AND original_query != ''
        ORDER BY if(user_uuid != '', user_uuid, ip), ts
        LIMIT {int(args.limit)}
        FORMAT JSONEachRow
    """
    url = f"{base}/?{urllib.parse.urlencode({'database': db, 'query': sql})}"
    auth = base64.b64encode(
        f"{env.get('CLICKHOUSE_USER','')}:{env.get('CLICKHOUSE_PASSWORD','')}".encode()).decode()
    log.info("Querying %s days of search events ...", args.days)
    try:
        request = urllib.request.Request(url, headers={"Authorization": "Basic " + auth})
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        sys.exit(
            f"Could not reach ClickHouse at {base}: {exc.reason}\n\n"
            "That hostname usually only resolves inside the compose network. Run this\n"
            "from within the service container:\n"
            "    docker cp behavioural_signals.py scout_search_api:/tmp/bs.py\n"
            "    docker exec scout_search_api python /tmp/bs.py export --out /tmp/sessions.json\n"
        )
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    log.info("  %d events", len(rows))

    kept = [r for r in rows
            if not any(h in (r.get("user_agent") or "").lower() for h in _BOT_HINTS)]
    log.info("  %d after removing declared bots", len(kept))

    for row in kept:
        row.pop("user_agent", None)          # not needed downstream, and identifying
        row["actor"] = _actor_key(row)
        row.pop("user_uuid", None)
        row.pop("ip", None)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(kept, fh, ensure_ascii=False)
    log.info("Wrote %s (%d events)", args.out, len(kept))
    print(f"\nNext: python behavioural_signals.py analyse --sessions {args.out}")


def _actor_key(row: dict[str, Any]) -> str:
    """A stable, non-identifying key for one searcher.

    The raw uuid or ip never leaves the server: only a hash of it is written,
    which is enough to group a sitting together and useless for anything else.
    """
    import hashlib

    raw = (row.get("user_uuid") or "").strip() or (row.get("ip") or "").strip()
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:16] if raw else ""


# ---------------------------------------------------------------- sessions ---

def parse_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("T", " ").rstrip("Z")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def normalise(text: str) -> str:
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def token_overlap(a: str, b: str) -> float:
    ta, tb = set(normalise(a).split()), set(normalise(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def build_sessions(events: list[dict[str, Any]], gap_seconds: int) -> list[list[dict[str, Any]]]:
    """Group consecutive events by actor, splitting on a long pause."""
    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        actor = event.get("actor") or ""
        if actor:
            by_actor[actor].append(event)

    sessions: list[list[dict[str, Any]]] = []
    for actor, rows in by_actor.items():
        rows.sort(key=lambda r: parse_ts(r.get("ts")))
        current: list[dict[str, Any]] = []
        last = None
        for row in rows:
            t = parse_ts(row.get("ts"))
            if last is not None and t - last > gap_seconds:
                if current:
                    sessions.append(current)
                current = []
            current.append(row)
            last = t
        if current:
            sessions.append(current)
    return sessions


def label_searches(sessions: list[list[dict[str, Any]]], follow_seconds: int,
                   overlap_threshold: float) -> list[dict[str, Any]]:
    """Attach an outcome to every search, inferred from what happened next.

    The signals, weakest evidence first:
      page_2        the user paged past the first ten results
      repeat        the same query again - usually a reload or a retry
      reformulated  a different query soon after, sharing some words with it:
                    the user was chasing the same thing and had not found it
      switched      a different query soon after with nothing in common: more
                    likely a new information need than a failure
      zero          the search returned nothing at all
      settled       nothing followed within the window
    """
    labelled: list[dict[str, Any]] = []
    for session in sessions:
        for i, event in enumerate(session):
            nxt = session[i + 1] if i + 1 < len(session) else None
            gap = (parse_ts(nxt["ts"]) - parse_ts(event["ts"])) if nxt else None
            outcome = "settled"
            if int(event.get("results_count") or 0) == 0:
                outcome = "zero"
            elif nxt and gap is not None and gap <= follow_seconds:
                same = normalise(event.get("original_query")) == normalise(nxt.get("original_query"))
                overlap = token_overlap(event.get("original_query"), nxt.get("original_query"))
                if same:
                    outcome = "repeat"
                elif overlap >= overlap_threshold:
                    outcome = "reformulated"
                else:
                    outcome = "switched"
            if int(event.get("page") or 1) > 1:
                outcome = "page_2" if outcome == "settled" else outcome
            labelled.append({**event, "outcome": outcome,
                             "seconds_to_next": round(gap, 1) if gap is not None else None,
                             "session_length": len(session)})
    return labelled


FAILURE_OUTCOMES = {"zero", "reformulated", "repeat", "page_2"}


def query_shape(text: str) -> str:
    words = re.findall(r"\S+", text or "")
    if re.search(r"\?$|^(how|what|why|when|where|which|can|is|are|does|do|comment|wie|c[oó]mo|hoe|wat)\b",
                 text or "", re.I):
        return "question"
    if any(re.fullmatch(r"[A-ZÀ-Þ0-9][A-ZÀ-Þ0-9_\-.]{2,}", w) for w in words):
        return "entity/name"
    return "topical phrase"


def rate_table(rows: list[dict[str, Any]], key, min_n: int) -> list[tuple[str, int, float]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(key(row))].append(row)
    out = []
    for name, group in buckets.items():
        if len(group) < min_n:
            continue
        failed = sum(1 for r in group if r["outcome"] in FAILURE_OUTCOMES)
        out.append((name, len(group), failed / len(group)))
    return sorted(out, key=lambda t: -t[2])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A proportion from few observations needs an interval, not a point."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def cmd_analyse(args: argparse.Namespace) -> None:
    with open(args.sessions, "r", encoding="utf-8") as fh:
        events = json.load(fh)
    sessions = build_sessions(events, args.session_gap)
    rows = label_searches(sessions, args.follow_window, args.overlap)

    lines: list[str] = []
    lines.append("=== WHAT USERS' BEHAVIOUR SAYS ABOUT SEARCH ===")
    lines.append(f"Events analysed     : {len(rows)}")
    lines.append(f"Sessions            : {len(sessions)} "
                 f"(a gap over {args.session_gap}s starts a new one)")
    lines.append(f"Follow-up window    : {args.follow_window}s")
    sizes = [len(s) for s in sessions] or [0]
    lines.append(f"Searches per session: mean {statistics.mean(sizes):.1f}, "
                 f"median {statistics.median(sizes):.0f}, max {max(sizes)}")

    counts = Counter(r["outcome"] for r in rows)
    failed = sum(counts[o] for o in FAILURE_OUTCOMES)
    lo, hi = wilson_interval(failed, len(rows))
    lines.append("")
    lines.append("--- OUTCOME OF EACH SEARCH ---")
    lines.append("  reformulated = a different but overlapping query followed soon after;")
    lines.append("  the strongest sign the user had not found what they wanted.")
    lines.append("")
    for outcome, n in counts.most_common():
        lines.append(f"  {outcome:14} {n:7}  ({100*n/len(rows):5.1f}%)")
    lines.append("")
    lines.append(f"  APPARENT FAILURE RATE: {100*failed/len(rows):.1f}% "
                 f"[{100*lo:.1f}%, {100*hi:.1f}%]")
    lines.append("  (zero + reformulated + repeat + page_2, as a share of all searches)")

    lines.append("")
    lines.append("--- BY LANGUAGE ---")
    lines.append(f"  {'language':10} {'searches':>9} {'failure rate':>13}")
    for name, n, rate in rate_table(rows, lambda r: (r.get("detected_lang") or "?")[:2],
                                    args.min_bucket)[:14]:
        lines.append(f"  {name:10} {n:9} {100*rate:12.1f}%")

    lines.append("")
    lines.append("--- BY QUERY SHAPE ---")
    lines.append(f"  {'shape':16} {'searches':>9} {'failure rate':>13}")
    for name, n, rate in rate_table(rows, lambda r: query_shape(r.get("original_query")),
                                    args.min_bucket):
        lines.append(f"  {name:16} {n:9} {100*rate:12.1f}%")

    lines.append("")
    lines.append("--- BY RESULT COUNT ---")
    lines.append("  Whether getting more results actually helps.")
    lines.append(f"  {'results':16} {'searches':>9} {'failure rate':>13}")

    def bucket(r):
        n = int(r.get("results_count") or 0)
        return "0" if n == 0 else "1-3" if n <= 3 else "4-9" if n <= 9 else "10 (full page)"
    for name, n, rate in sorted(rate_table(rows, bucket, args.min_bucket), key=lambda t: t[0]):
        lines.append(f"  {name:16} {n:9} {100*rate:12.1f}%")

    lines.append("")
    lines.append("--- THE QUERIES USERS FOUGHT WITH ---")
    lines.append("  Real queries, seen often, that most often led to another search.")
    lines.append("  This is a work queue: each one is a retrieval failure users actually hit.")
    lines.append("")
    per_query = rate_table(rows, lambda r: (r.get("original_query") or "").strip(),
                           args.min_query_n)
    lines.append(f"  {'failure':>8} {'n':>5}  query")
    for name, n, rate in per_query[:25]:
        lines.append(f"  {100*rate:7.0f}% {n:5}  {name[:64]}")

    lines.append("")
    lines.append("--- WHAT THIS CANNOT TELL YOU ---")
    lines.append("  Reformulation is confounded: a user may search again because the first")
    lines.append("  search succeeded and moved them on to the next question. These rates")
    lines.append("  measure aggregate dissatisfaction, not per-document relevance, and no")
    lines.append("  single query's rate is meaningful on its own.")
    lines.append("  There is no click logging, so the strongest implicit signal - which")
    lines.append("  result a user actually opened - is unavailable. Adding it would make")
    lines.append("  everything here considerably sharper.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        log.info("Wrote %s", args.out)


# ---------------------------------------------------------------- validate ---

def cmd_validate(args: argparse.Namespace) -> None:
    """Check an LLM panel's grades against what users did next.

    If the panel is tracking reality, searches it rates highly should be
    followed by fewer reformulations than searches it rates poorly. This is the
    only validation available without a human grader, and it is weak - but a
    panel that fails it is certainly wrong.
    """
    from openpyxl import load_workbook

    with open(args.sessions, "r", encoding="utf-8") as fh:
        events = json.load(fh)
    rows = label_searches(build_sessions(events, args.session_gap),
                          args.follow_window, args.overlap)

    behaviour: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        behaviour[normalise(row.get("original_query"))].append(row["outcome"])

    ws = load_workbook(args.judged, data_only=True)["judging"]
    header = [c.value for c in ws[1]]
    try:
        grade_col = header.index(args.column) + 1
    except ValueError:
        sys.exit(f"No column named {args.column!r} in {args.judged}")

    per_query: dict[str, list[int]] = defaultdict(list)
    query = ""
    for row in ws.iter_rows(min_row=2):
        values = [c.value for c in row]
        if values[0]:
            query = str(values[0]).strip()
        grade = ws.cell(row=row[0].row, column=grade_col).value
        if values[7] and grade is not None and str(grade).strip() != "":
            try:
                per_query[normalise(query)].append(int(str(grade).strip()))
            except ValueError:
                continue

    paired = []
    for key, grades in per_query.items():
        outcomes = behaviour.get(key)
        if not outcomes:
            continue
        best = max(grades)
        mean = sum(grades) / len(grades)
        fail = sum(1 for o in outcomes if o in FAILURE_OUTCOMES) / len(outcomes)
        paired.append({"query": key, "best": best, "mean": mean,
                       "fail_rate": fail, "n": len(outcomes)})

    print(f"Queries with both a panel grade and observed behaviour: {len(paired)}")
    if len(paired) < args.min_pairs:
        print(f"\nToo few to correlate (need at least {args.min_pairs}). Export a longer")
        print("window, or sample more queries into the judging sheet.")
        return

    print()
    print("Failure rate by the best grade the panel gave that query:")
    print(f"  {'best grade':12} {'queries':>8} {'mean failure rate':>19}")
    for grade in (3, 2, 1, 0):
        group = [p for p in paired if p["best"] == grade]
        if group:
            rate = statistics.mean(p["fail_rate"] for p in group)
            print(f"  {grade:<12} {len(group):8} {100*rate:18.1f}%")

    xs = [p["mean"] for p in paired]
    ys = [p["fail_rate"] for p in paired]
    r = _pearson(xs, ys)
    print()
    print(f"Correlation (mean grade vs failure rate): r = {r:+.2f}")
    print("A panel that tracks reality gives a NEGATIVE correlation: better-graded")
    print("results should be followed by fewer reformulations.")
    print()
    if r <= -0.3:
        print("VERDICT: the panel's grades move with user behaviour. That is not proof")
        print("         of correctness, but it is evidence the grades are not arbitrary.")
    elif r <= -0.1:
        print("VERDICT: a weak relationship in the expected direction. Suggestive only.")
    elif r < 0.1:
        print("VERDICT: no relationship. The panel's grades and user behaviour are")
        print("         unrelated here - treat the grades as unvalidated.")
    else:
        print("VERDICT: the correlation runs the WRONG WAY. Either the grades are poor,")
        print("         or reformulation is measuring something other than failure in")
        print("         this sample. Investigate before relying on either.")


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return 0.0
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


# -------------------------------------------------------------------- main ---

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    workspace = os.path.abspath(os.path.join(here, "..", ".."))
    default_env = [os.path.join(workspace, "scout", ".env")]

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--session-gap", type=int, default=1800,
                       help="Seconds of inactivity that end a session")
        p.add_argument("--follow-window", type=int, default=120,
                       help="Seconds within which a following search counts as a follow-up")
        p.add_argument("--overlap", type=float, default=0.2,
                       help="Token overlap above which a follow-up counts as a reformulation")
        p.add_argument("--env", action="append", default=None)
        return p

    p = add_common(sub.add_parser("export", help="Pull search events from ClickHouse"))
    p.add_argument("--out", default="sessions.json")
    p.add_argument("--table", default="search_events")
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--limit", type=int, default=500000)
    p.set_defaults(func=cmd_export)

    p = add_common(sub.add_parser("analyse", help="Failure rates by language, shape and query"))
    p.add_argument("--sessions", default="sessions.json")
    p.add_argument("--out", default=None, help="Also write the report to this path")
    p.add_argument("--min-bucket", type=int, default=50)
    p.add_argument("--min-query-n", type=int, default=5)
    p.set_defaults(func=cmd_analyse)

    p = add_common(sub.add_parser("validate", help="Check panel grades against behaviour"))
    p.add_argument("--sessions", default="sessions.json")
    p.add_argument("--judged", default="judged.xlsx")
    p.add_argument("--column", default="llm_grade")
    p.add_argument("--min-pairs", type=int, default=10)
    p.set_defaults(func=cmd_validate)

    args = parser.parse_args()
    if not getattr(args, "env", None):
        args.env = default_env
    args.func(args)


if __name__ == "__main__":
    main()
