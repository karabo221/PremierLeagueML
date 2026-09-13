"""
PHASE 6 - COPY THE PREDICTIONS MIRROR INTO SUPABASE

    set SUPABASE_URL=https://<ref>.supabase.co
    set SUPABASE_SERVICE_KEY=<the service role key>
    python scripts/phase6_backfill_supabase.py --check
    python scripts/phase6_backfill_supabase.py --write

WHY THIS EXISTS. The 2026-09-11 round ran WITHOUT credentials in the
environment. It wrote the mirror and recorded, at L4d in
live_log/phase6_live_log_run_audit.csv:

    Supabase insert | NOT WRITTEN - no SUPABASE_URL / SUPABASE_SERVICE_KEY

Mirror-only is a degraded run and not a failed one (L9.4), so nothing needs
repairing in the record. But public.results carries

    match_id text primary key references public.predictions (match_id)

so until the predictions exist in Supabase, phase6_capture_results.py cannot
insert a single scoreline - the foreign key refuses it. This script closes that
gap, once.

WHAT THIS IS NOT, AND THE DISTINCTION MATTERS. It does not generate, recompute
or re-derive anything. It reads rows that were written before their matchweek's
first kickoff, verifies they are the rows that were written, and copies them.
The mirror remains the primary record; this only catches the dashboard's copy
up. A prediction has to be created by phase6_generate_predictions.py --round,
before kickoff, or not at all.

SAFETY. --check is the default and writes nothing. --write refuses any match_id
already present in the database, so a second run cannot overwrite a row: the
refusal is the same immutability guarantee L1.4 puts in the generator. Results
and market are never touched here; those arrive through capture.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = os.path.join(ROOT, "live_log", "phase6_live_predictions.csv")
GENESIS = "0" * 64

# The columns public.predictions declares, in the order the DDL declares them.
# A column in the mirror that is not here is NOT sent, and a column here that is
# missing from the mirror is an error rather than a null.
COLUMNS = [
    "match_id", "season", "round_id", "matchweek_label",
    "scheduled_date", "scheduled_kickoff", "home_team", "away_team",
    "state_cutoff_date", "generated_at_utc",
    "p_home", "p_draw", "p_away", "lambda_home", "lambda_away", "rho",
    "fit_matches", "home_has_history", "away_has_history",
    "freeze_sha", "pin_sha", "protocol_sha", "prev_hash", "row_hash",
]

NUMERIC = {"p_home", "p_draw", "p_away", "lambda_home", "lambda_away", "rho"}
INTEGER = {"round_id", "matchweek_label", "fit_matches"}
BOOLEAN = {"home_has_history", "away_has_history"}


# ============================================================
# 1. THE MIRROR, AND CHECKING IT BEFORE TRUSTING IT
# ============================================================

def read_mirror():
    if not os.path.exists(MIRROR):
        raise SystemExit("no mirror at %s - nothing to copy" % MIRROR)
    with open(MIRROR, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("the mirror is empty - nothing to copy")
    missing = [c for c in COLUMNS if c not in rows[0]]
    if missing:
        raise SystemExit("the mirror is missing columns the schema needs: %s"
                         % ", ".join(missing))
    return rows


def check_chain(rows):
    """
    L4.2. prev_hash of the first row is sixty-four zeroes, and every later row
    carries the previous row's row_hash. A BREAK HERE STOPS THE RUN: copying a
    chain that does not link would put rows into the dashboard that disagree
    with the record they are supposed to mirror.

    This checks the LINKS. The authoritative check, which recomputes each hash
    from the row's own fields, is phase6_generate_predictions.py --verify, and
    --check below tells you to run it.
    """
    problems = []
    for index, row in enumerate(rows):
        expected = GENESIS if index == 0 else rows[index - 1]["row_hash"]
        if row["prev_hash"] != expected:
            problems.append(
                "row %d (%s): prev_hash is %s, expected %s"
                % (index + 1, row["match_id"], row["prev_hash"][:12], expected[:12]))
    return problems


def check_probabilities(rows):
    """The two constraints the schema asserts, checked before the round trip."""
    problems = []
    for row in rows:
        total = float(row["p_home"]) + float(row["p_draw"]) + float(row["p_away"])
        if abs(total - 1.0) >= 1e-9:
            problems.append("%s: probabilities sum to %.15f" % (row["match_id"], total))
        if row["state_cutoff_date"] > row["scheduled_date"]:
            problems.append("%s: cutoff %s is after the fixture %s"
                            % (row["match_id"], row["state_cutoff_date"],
                               row["scheduled_date"]))
    return problems


def check_freeze(rows):
    """
    The mirror records which documents were in force. If the freeze on disk has
    moved since, say so rather than copying rows that name a hash the working
    tree no longer has.
    """
    path = os.path.join(ROOT, "PHASE6_HOLDOUT_FREEZE.txt")
    if not os.path.exists(path):
        return ["PHASE6_HOLDOUT_FREEZE.txt is not on disk"]

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    on_disk = digest.hexdigest()

    declared = {row["freeze_sha"] for row in rows}
    if declared != {on_disk}:
        return ["the mirror names freeze %s; the file on disk hashes to %s"
                % (", ".join(sorted(s[:12] for s in declared)), on_disk[:12])]
    return []


# ============================================================
# 2. TYPING
# ============================================================

def typed(row):
    out = {}
    for column in COLUMNS:
        value = row[column]
        if value == "":
            out[column] = None
        elif column in NUMERIC:
            out[column] = float(value)
        elif column in INTEGER:
            out[column] = int(value)
        elif column in BOOLEAN:
            out[column] = value == "True"
        else:
            out[column] = value
    return out


# ============================================================
# 3. SUPABASE
# ============================================================

def credentials():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise SystemExit(
            "SUPABASE_URL / SUPABASE_SERVICE_KEY are not in the environment.\n"
            "The service key is never read from a file in this repository.")
    return url, key


def request(url, key, method, path, body=None, extra_headers=None):
    headers = {
        "apikey": key,
        "Authorization": "Bearer %s" % key,
        "Content-Type": "application/json",
    }
    headers.update(extra_headers or {})

    payload = json.dumps(body, default=str).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        "%s/rest/v1/%s" % (url, path), data=payload, method=method, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return response.status, raw, dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", "replace"), dict(error.headers)


def existing_ids(url, key):
    status, raw, _ = request(
        url, key, "GET", "predictions?select=match_id&limit=10000")
    if status == 404 or "does not exist" in raw:
        raise SystemExit(
            "public.predictions is not there. Apply the schema first:\n"
            "    psql \"$SUPABASE_DB_URL\" -f ops/supabase_live_log.sql\n"
            "or paste ops/supabase_live_log.sql into the Supabase SQL editor.")
    if not 200 <= status < 300:
        raise SystemExit("reading predictions failed: HTTP %d %s" % (status, raw[:400]))
    return {row["match_id"] for row in json.loads(raw)}


def insert(url, key, rows):
    """INSERT only. No upsert, no PATCH - a duplicate must raise, not overwrite."""
    return request(url, key, "POST", "predictions", rows,
                   {"Prefer": "return=minimal"})


# ============================================================
# 4. MAIN
# ============================================================

def sql_literal(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # round-trip precision. A probability rewritten at 6dp would fail the
        # schema's own sum-to-one check at 1e-9.
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def emit_sql(rows, target):
    """
    Write ONE file that can be pasted into the Supabase SQL editor: the schema,
    then the predictions. No service key, no psql, no Python on the far side.

    Idempotent on both halves. The schema is already `create table if not
    exists`; the inserts carry `on conflict (match_id) do nothing`, so running
    the file twice cannot overwrite a row - which is the same refusal L1.4 puts
    in the generator, expressed in SQL.
    """

    ddl_path = os.path.join(ROOT, "ops", "supabase_live_log.sql")
    with open(ddl_path, encoding="utf-8") as handle:
        ddl = handle.read()

    parts = [
        "-- " + "=" * 74,
        "-- PHASE 6 BOOTSTRAP - SCHEMA PLUS THE PREDICTIONS ALREADY WRITTEN",
        "-- " + "=" * 74,
        "--",
        "-- GENERATED by scripts/phase6_backfill_supabase.py --sql.",
        "-- Do not edit: regenerate it. Safe to run more than once.",
        "--",
        "-- Paste the whole file into the Supabase SQL editor and run it. It does",
        "-- two things, and the second one is why capture is currently blocked:",
        "--",
        "--   1. creates the three tables, their constraints, the RLS policies and",
        "--      the live_log_coverage view;",
        "--",
        "--   2. inserts the %d predictions already in the committed mirror." % len(rows),
        "--      public.results.match_id is a foreign key onto public.predictions,",
        "--      so until these rows exist, phase6_capture_results.py cannot insert",
        "--      a single scoreline - the foreign key refuses it.",
        "--",
        "-- These rows are COPIED, not recomputed. Each was written before its",
        "-- matchweek's first kickoff and is timestamped in git; this file only",
        "-- catches the dashboard's copy up to the record.",
        "--",
        "-- Source   live_log/phase6_live_predictions.csv",
        "-- Freeze   %s" % rows[0]["freeze_sha"],
        "-- Rows     %d, chain verified contiguous from genesis" % len(rows),
        "-- " + "=" * 74,
        "",
        "",
        "-- " + "-" * 74,
        "-- PART 1 OF 2 - THE SCHEMA, verbatim from ops/supabase_live_log.sql",
        "-- " + "-" * 74,
        "",
        ddl.rstrip(),
        "",
        "",
        "-- " + "-" * 74,
        "-- PART 2 OF 2 - THE PREDICTIONS",
        "-- " + "-" * 74,
        "",
        "insert into public.predictions (",
        "    " + ",\n    ".join(COLUMNS),
        ") values",
    ]

    values = []
    for row in rows:
        typed_row = typed(row)
        cells = ", ".join(sql_literal(typed_row[column]) for column in COLUMNS)
        values.append("    -- %s v %s, %s"
                      % (row["home_team"], row["away_team"], row["scheduled_date"]))
        values.append("    (%s)" % cells)

    # join the value tuples with commas, leaving the comment lines alone
    rendered = []
    tuple_lines = [i for i, line in enumerate(values) if line.startswith("    (")]
    for index, line in enumerate(values):
        if index in tuple_lines and index != tuple_lines[-1]:
            rendered.append(line + ",")
        else:
            rendered.append(line)

    parts.extend(rendered)
    parts.append("on conflict (match_id) do nothing;")
    parts.append("")
    parts.append("")
    parts.append("-- " + "-" * 74)
    parts.append("-- CHECK IT LANDED")
    parts.append("-- " + "-" * 74)
    parts.append("")
    parts.append("select * from public.live_log_coverage;")
    parts.append("")
    parts.append("-- predictions_written should read %d. Once it does, capture is"
                 % len(rows))
    parts.append("-- unblocked and results can be inserted against these match_ids.")
    parts.append("")

    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(parts))

    return target


def main():
    parser = argparse.ArgumentParser(
        description="Copy the committed predictions mirror into Supabase.")
    parser.add_argument("--check", action="store_true",
                        help="verify the mirror and report what would be sent (default)")
    parser.add_argument("--write", action="store_true",
                        help="actually insert the rows Supabase does not have")
    parser.add_argument("--sql", action="store_true",
                        help="write ops/supabase_bootstrap.sql instead - schema plus "
                             "the predictions, for pasting into the SQL editor")
    args = parser.parse_args()

    rows = read_mirror()
    print("mirror: %d rows, %s" % (len(rows), os.path.relpath(MIRROR, ROOT)))

    problems = check_chain(rows) + check_probabilities(rows) + check_freeze(rows)
    if problems:
        print("\nTHE MIRROR DID NOT PASS. Nothing was sent.\n")
        for problem in problems:
            print("  %s" % problem)
        return 1

    print("  chain links       contiguous, genesis is sixty-four zeroes")
    print("  probabilities     all sum to 1 within 1e-9")
    print("  cutoffs           all precede their fixture")
    print("  freeze            the mirror agrees with the file on disk")
    print("\n  the authoritative hash check is a separate instrument:")
    print("      python scripts/phase6_generate_predictions.py --verify")

    if args.sql:
        target = emit_sql(rows, os.path.join(ROOT, "ops", "supabase_bootstrap.sql"))
        print("\nwrote %s" % os.path.relpath(target, ROOT))
        print("  schema + %d predictions, idempotent, no credentials needed." % len(rows))
        print("  Paste the whole file into the Supabase SQL editor and run it.")
        return 0

    if not args.write:
        print("\n--check only. Re-run with --write to insert, or --sql to write a "
              "file you can paste into the Supabase SQL editor.")
        url = os.environ.get("SUPABASE_URL", "")
        if not url:
            print("SUPABASE_URL is not set, so the database was not contacted.")
        return 0

    url, key = credentials()
    present = existing_ids(url, key)
    print("\ndatabase: %d predictions already present" % len(present))

    pending = [typed(row) for row in rows if row["match_id"] not in present]
    skipped = len(rows) - len(pending)

    if skipped:
        print("  skipping %d already in the database - a row that exists is "
              "never rewritten" % skipped)

    if not pending:
        print("\nnothing to do. The dashboard's copy already matches the mirror.")
        return 0

    status, raw, _ = insert(url, key, pending)
    if not 200 <= status < 300:
        print("\nINSERT FAILED: HTTP %d\n%s" % (status, raw[:600]))
        return 1

    print("\ninserted %d rows. HTTP %d." % (len(pending), status))
    after = existing_ids(url, key)
    print("database now holds %d of the mirror's %d predictions."
          % (len(after), len(rows)))
    if len(after) != len(rows):
        print("THAT IS NOT A MATCH - investigate before capturing any result.")
        return 1

    print("\nphase6_capture_results.py can now insert scorelines: the foreign "
          "key from public.results has something to point at.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
