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

And once scout is running an experiment (SEARCH_EXPERIMENT), the same signals
become a live A/B - the only measurement here that is not a proxy for anything:

    python behavioural_signals.py experiment --sessions sessions.json --aa   # first
    python behavioural_signals.py experiment --sessions sessions.json
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
    auth = base64.b64encode(
        f"{env.get('CLICKHOUSE_USER','')}:{env.get('CLICKHOUSE_PASSWORD','')}".encode()).decode()

    # experiment_arm and latency_ms arrived later than the rest of the schema, so
    # ask the table what it has rather than assuming. A missing column would
    # otherwise fail the whole export with a syntax error.
    available = _describe(base, db, args.table, auth)
    wanted = ["ts", "original_query", "detected_lang", "results_count", "page",
              "user_uuid", "ip", "user_agent", "search_source", "endpoint",
              "total_records", "experiment_arm", "latency_ms"]
    columns = [c for c in wanted if not available or c in available]
    missing = [c for c in wanted if available and c not in available]
    if missing:
        log.warning("Table %s has no %s; continuing without", args.table, ", ".join(missing))

    sql = f"""
        SELECT {', '.join(columns)}
        FROM {args.table}
        WHERE ts >= now() - INTERVAL {int(args.days)} DAY
          AND status_code = 200
          AND original_query != ''
        ORDER BY if(user_uuid != '', user_uuid, ip), ts
        LIMIT {int(args.limit)}
        FORMAT JSONEachRow
    """
    url = f"{base}/?{urllib.parse.urlencode({'database': db, 'query': sql})}"
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


def _describe(base: str, db: str, table: str, auth: str) -> set[str]:
    """Column names of a table, or an empty set if the question cannot be asked."""
    query = urllib.parse.urlencode({"database": db, "query": f"DESCRIBE TABLE {table} FORMAT TSV"})
    try:
        request = urllib.request.Request(f"{base}/?{query}",
                                         headers={"Authorization": "Basic " + auth})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except Exception:  # noqa: BLE001 - fall back to asking for everything
        return set()
    return {line.split("\t")[0] for line in body.splitlines() if line.strip()}


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

# ---------------------------------------------------------------------------
# Experiment analysis
# ---------------------------------------------------------------------------

def _z(p: float) -> float:
    """Inverse normal CDF, good to ~4.5e-4. Enough for a power calculation."""
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    # Abramowitz & Stegun 26.2.23, mirrored about the median.
    sign, q = (1.0, p) if p > 0.5 else (-1.0, 1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - q))
    num = 2.515517 + 0.802853 * t + 0.010328 * t * t
    den = 1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    return sign * (t - num / den)


def cluster_bootstrap(by_actor: dict[str, list[dict[str, Any]]],
                      arms: dict[str, list[str]],
                      rounds: int,
                      seed: int) -> dict[str, Any]:
    """Failure rates and their difference, resampling PEOPLE rather than searches.

    Searches by one person are not independent: someone who reformulates once
    tends to reformulate again, and a single frustrated user with forty searches
    would otherwise look like forty pieces of evidence. Resampling at the level
    the experiment randomised at - the person - carries that correlation into
    the interval instead of pretending it away. On this data it roughly doubles
    the width of the interval, which is the honest width.
    """
    import random
    rng = random.Random(seed)

    def rate(actor_ids: list[str]) -> tuple[float, int]:
        failed = total = 0
        for actor in actor_ids:
            for row in by_actor[actor]:
                total += 1
                failed += row["outcome"] in FAILURE_OUTCOMES
        return (failed / total if total else 0.0), total

    names = sorted(arms)
    observed = {name: rate(arms[name]) for name in names}
    diffs: list[float] = []
    per_arm: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(rounds):
        sampled = {}
        for name in names:
            pool = arms[name]
            picks = [pool[rng.randrange(len(pool))] for _ in range(len(pool))]
            value, _ = rate(picks)
            sampled[name] = value
            per_arm[name].append(value)
        if len(names) == 2:
            diffs.append(sampled[names[1]] - sampled[names[0]])

    out: dict[str, Any] = {"arms": {}}
    for name in names:
        value, n = observed[name]
        series = sorted(per_arm[name])
        out["arms"][name] = {
            "rate": value,
            "searches": n,
            "actors": len(arms[name]),
            "lo": series[int(0.025 * len(series))],
            "hi": series[int(0.975 * len(series)) - 1],
        }
    if len(names) == 2 and diffs:
        series = sorted(diffs)
        # Two-sided, from the bootstrap distribution itself. The +1s stop a
        # p-value of exactly 0, which no finite resample can justify.
        below = sum(1 for d in diffs if d <= 0)
        above = sum(1 for d in diffs if d >= 0)
        out["diff"] = {
            "value": observed[names[1]][0] - observed[names[0]][0],
            "lo": series[int(0.025 * len(series))],
            "hi": series[int(0.975 * len(series)) - 1],
            "se": statistics.pstdev(diffs),
            "p": min(1.0, 2.0 * (min(below, above) + 1) / (len(diffs) + 1)),
            "baseline": names[0],
            "treated": names[1],
        }
    return out


def cmd_experiment(args: argparse.Namespace) -> None:
    with open(args.sessions, "r", encoding="utf-8") as fh:
        events = json.load(fh)

    if not any("experiment_arm" in e for e in events):
        sys.exit(
            "No experiment_arm in this export, so there is nothing to compare.\n\n"
            "Either the experiment is not running, or ClickHouse has no such column and the\n"
            "arm is being discarded on insert. On the server:\n"
            "  ALTER TABLE euf_search_analytics.search_events\n"
            "    ADD COLUMN IF NOT EXISTS experiment_arm LowCardinality(String) DEFAULT '';\n"
            "then re-export once traffic has accumulated."
        )

    if args.aa:
        # An A/A split: assign every actor by a hash that has nothing to do with
        # the experiment. Any 'significant' difference here is the measurement
        # lying, not the configuration working - run this before believing an
        # A/B result.
        import hashlib
        for row in events:
            digest = hashlib.sha256(f"aa:{row.get('actor','')}".encode()).digest()
            row["experiment_arm"] = "aa_left" if digest[0] < 128 else "aa_right"

    sessions = build_sessions(events, args.session_gap)
    rows = label_searches(sessions, args.follow_window, args.overlap)

    lines: list[str] = []
    lines.append("=== A/B: WHAT THE TWO CONFIGURATIONS DID TO REAL USERS ===")
    if args.aa:
        lines.append("MODE: A/A. Both arms are the same configuration, split by an unrelated")
        lines.append("hash. A significant result here means the pipeline is broken.")
        lines.append("")

    # --- who is in the comparison at all ---
    total = len(rows)
    rows = [r for r in rows if (r.get("experiment_arm") or "").strip()]
    unassigned = total - len(rows)

    if args.source:
        before = len(rows)
        rows = [r for r in rows if (r.get("search_source") or "") == args.source]
        lines.append(f"Restricted to search_source={args.source!r}: "
                     f"{len(rows)} of {before} searches")
    if args.endpoint:
        rows = [r for r in rows if (r.get("endpoint") or "") == args.endpoint]

    if not rows:
        sys.exit("No assigned searches left after filtering.")

    # --- an actor must be in exactly one arm ---
    arms_by_actor: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        arms_by_actor[row.get("actor", "")].add(row["experiment_arm"])
    crossed = {a for a, s in arms_by_actor.items() if len(s) > 1}
    if crossed:
        rows = [r for r in rows if r.get("actor") not in crossed]

    by_actor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_actor[row.get("actor", "")].append(row)
    arms: dict[str, list[str]] = defaultdict(list)
    for actor, group in by_actor.items():
        arms[group[0]["experiment_arm"]].append(actor)

    lines.append(f"Searches analysed   : {len(rows)}")
    lines.append(f"Unassigned, excluded: {unassigned} "
                 f"({100*unassigned/total:.1f}% - anonymous, or before the experiment began)")
    lines.append(f"People              : {len(by_actor)}")
    if crossed:
        lines.append(f"People in BOTH arms : {len(crossed)}, excluded. Assignment should make")
        lines.append("                      this impossible; more than a handful means an actor's")
        lines.append("                      key is not stable and the split cannot be trusted.")
    lines.append("")

    # --- did the split come out as configured ---
    lines.append("--- ASSIGNMENT ---")
    for name in sorted(arms):
        share = len(arms[name]) / len(by_actor)
        lo, hi = wilson_interval(len(arms[name]), len(by_actor))
        lines.append(f"  {name:12} {len(arms[name]):6} people ({100*share:5.1f}%) "
                     f"[{100*lo:.1f}%, {100*hi:.1f}%]  {sum(len(by_actor[a]) for a in arms[name]):6} searches")
    if len(arms) == 2:
        counts = [len(arms[n]) for n in sorted(arms)]
        lo, hi = wilson_interval(counts[0], sum(counts))
        if not (lo <= 0.5 <= hi):
            lines.append("  WARNING: the split is further from even than chance explains. Either the")
            lines.append("  arms are weighted deliberately, or one arm is catching bot traffic.")
    lines.append("")

    if len(arms) < 2:
        sys.exit("\n".join(lines) + "\n\nOnly one arm present; nothing to compare.")

    # --- the outcome ---
    result = cluster_bootstrap(by_actor, dict(arms), args.bootstrap, args.seed)
    lines.append("--- APPARENT FAILURE RATE (zero + reformulated + repeat + page_2) ---")
    lines.append("  Intervals resample people, not searches; see cluster_bootstrap.")
    lines.append("")
    for name in sorted(result["arms"]):
        a = result["arms"][name]
        lines.append(f"  {name:12} {100*a['rate']:5.1f}%  [{100*a['lo']:.1f}%, {100*a['hi']:.1f}%]"
                     f"   n={a['searches']} searches / {a['actors']} people")

    diff = result.get("diff")
    if diff:
        lines.append("")
        direction = "BETTER" if diff["value"] < 0 else "WORSE"
        lines.append(f"  {diff['treated']} vs {diff['baseline']}: "
                     f"{100*diff['value']:+.2f} points "
                     f"[{100*diff['lo']:+.2f}, {100*diff['hi']:+.2f}], p = {diff['p']:.3f}")
        if diff["p"] < 0.05:
            lines.append(f"  -> {diff['treated']} is {direction}. The interval excludes zero.")
        else:
            lines.append(f"  -> No detectable difference. The interval includes zero, so this is")
            lines.append(f"     'not yet measurable', NOT 'the same'.")

    # --- guardrails: things that must not get worse ---
    lines.append("")
    lines.append("--- GUARDRAILS ---")
    for name in sorted(arms):
        group = [r for a in arms[name] for r in by_actor[a]]
        zero = sum(1 for r in group if int(r.get("results_count") or 0) == 0)
        counts = sorted(int(r.get("results_count") or 0) for r in group)
        lat = sorted(int(r["latency_ms"]) for r in group if r.get("latency_ms") is not None)
        median_lat = lat[len(lat) // 2] if lat else None
        p95 = lat[int(0.95 * len(lat))] if lat else None
        lines.append(f"  {name:12} zero-result {100*zero/len(group):5.1f}%   "
                     f"median results {counts[len(counts)//2]:3}   "
                     + (f"latency p50 {median_lat}ms p95 {p95}ms" if lat else "latency not logged"))

    # --- how long must this run ---
    if diff:
        lines.append("")
        lines.append("--- HOW LONG TO RUN ---")
        se = diff["se"]
        span = _span_days(rows)
        per_day = len(rows) / span if span else 0
        z_alpha, z_beta = _z(1 - args.alpha / 2), _z(args.power)
        lines.append(f"  Observed so far: {len(rows)} searches over {span:.1f} days "
                     f"(~{per_day:.0f}/day), standard error {100*se:.2f} points.")
        for mde in args.mde:
            target_se = mde / (z_alpha + z_beta)
            if se <= 0 or target_se <= 0:
                continue
            needed = len(rows) * (se / target_se) ** 2
            more = max(0.0, needed - len(rows))
            when = f"{more/per_day:.0f} more days" if per_day else "unknown - no time span"
            verdict = "already enough" if more <= 0 else when
            lines.append(f"  To detect {100*mde:.1f} points at {100*(1-args.alpha):.0f}% "
                         f"confidence and {100*args.power:.0f}% power: "
                         f"{needed:,.0f} searches -> {verdict}")
        lines.append("  Stopping the moment a result turns significant inflates the false")
        lines.append("  positive rate well past 5%. Pick a target above, then read it once.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
        log.info("Wrote %s", args.out)


def _span_days(rows: list[dict[str, Any]]) -> float:
    stamps = [parse_ts(r["ts"]) for r in rows if r.get("ts")]
    if len(stamps) < 2:
        return 0.0
    return (max(stamps) - min(stamps)) / 86400.0


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

    p = add_common(sub.add_parser("experiment", help="Compare A/B arms on real behaviour"))
    p.add_argument("--sessions", default="sessions.json")
    p.add_argument("--out", default=None, help="Also write the report to this path")
    p.add_argument("--source", default=None,
                   help="Keep only this search_source (e.g. the frontend's tag)")
    p.add_argument("--endpoint", default=None, help="Keep only this endpoint")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--mde", type=float, nargs="+", default=[0.01, 0.02, 0.05],
                   help="Effect sizes to size the run for, as proportions")
    p.add_argument("--aa", action="store_true",
                   help="Ignore the real arms and split by an unrelated hash, to test the pipeline")
    p.set_defaults(func=cmd_experiment)

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
