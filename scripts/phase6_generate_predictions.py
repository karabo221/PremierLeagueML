"""
===============================================================================
PHASE 6 - THE LIVE PREDICTION LOG GENERATOR
===============================================================================

Governed by PHASE6_LIVE_LOG_PROTOCOL.txt, which is itself governed by
PHASE6_HOLDOUT_FREEZE.txt (sha256 b36befd3...248f6) and PHASE6_CUTOFF_PIN.txt.

WHAT THIS IS FOR. A season-end batch run asserts that the freeze held. A
pre-kickoff timestamped log PROVES what was said before the result existed.
This file writes that log. It is evidence, not a result.

WHAT IT HAS NO AUTHORITY OVER. It does not score the holdout, it does not
touch the freeze, and no number it produces enters REPORT.md. The official
2026-27 result is phase6_score_holdout.py, run once at season end. See L5.

IT REIMPLEMENTS NOTHING. The estimator, the vocabulary assertion, the pin
parser and the cutoff rule are IMPORTED from the instruments that already own
them. That is deliberate: a second copy of the Dixon-Coles fit would be a
second thing to keep in step with the freeze, and the freeze validator would
not be watching it.

    from phase2_poisson_dixon_coles  fit_window, predict_matches, constants
    from phase6_score_holdout        load_pin, assert_vocabulary, score_from_cutoff
    from phase5_market_benchmark     TEAM_MAP
    from phase1_match_foundation     result_from_goals, points_from_result
    from phase0_evaluation_harness   evaluate, validate_probabilities

THE TWO DRY RUNS, AND WHY THERE ARE TWO. The brief for this file asked that
the concatenated per-matchweek predictions reproduce the frozen walk-forward
figures, and treated any drift as a bug. The first cannot happen and the
second does not follow, for a reason that is structural rather than a defect:

    the frozen dc_walkforward refits ONCE PER DISTINCT SCORED DATE
        2022-23: 117 refits   2023-24: 120   2024-25: 109   2025-26: 114

    a matchweek-cadence log refits ONCE PER MATCHWEEK
        38 refits, every season

37 of 38 matchweeks span more than one day, so 198-221 of each season's 380
matches sit on a later day of their matchweek and are predicted WITHOUT the
earlier results of that same matchweek. The information sets genuinely differ,
and the live log's is strictly smaller.

So this file separates the two claims that were conflated:

    DRY RUN A - FIDELITY.  Same code, per-date cadence. MUST reproduce the
                frozen figures bit for bit. This is what "any drift is a bug"
                correctly applies to, and it is where a bug would show.

    DRY RUN B - CADENCE.   Same code, matchweek cadence. Reproduces nothing
                and is not meant to. It MEASURES the cost of predicting a
                whole matchweek in advance, which is a property of the log's
                design, pre-declared here rather than discovered in 2027.

    ./venv/Scripts/python.exe -B scripts/phase6_generate_predictions.py
        both dry runs against the development seasons

    ... --round            write every due 2026-27 round (L10.2): the next
                           one, and any already kicked off, flagged late
    ... --round --matchweek N   only that round
    ... --verify           recompute every row hash in the local mirror
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import evaluate, validate_probabilities  # noqa: E402
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase2_poisson_dixon_coles as DC          # noqa: E402
import phase3_ablation_ladder as L3              # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402

from phase1_match_foundation import (            # noqa: E402
    result_from_goals, points_from_result)
from phase5_market_benchmark import TEAM_MAP     # noqa: E402
from phase6_score_holdout import (               # noqa: E402
    HoldoutError, assert_vocabulary, load_pin, score_from_cutoff, sha256_of)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FREEZE = PROJECT_ROOT / "PHASE6_HOLDOUT_FREEZE.txt"
PIN = PROJECT_ROOT / "PHASE6_CUTOFF_PIN.txt"
PROTOCOL = PROJECT_ROOT / "PHASE6_LIVE_LOG_PROTOCOL.txt"

FREEZE_SHA = "b36befd3e13d5f7f5d9b0af83b7db819e86d7b13e6fdcab4c55db0599fd248f6"

# THE LOG DOES NOT LIVE UNDER outputs/phase6_*.csv, AND THE REASON MATTERS.
# That glob is in FROZEN_PATTERNS, so anything matching it is swept into
# FROZEN_MANIFEST.txt and pinned by hash. This log is APPEND-ONLY BY DESIGN:
# its bytes change every round, so a frozen hash would fail the manifest every
# matchweek and the failure would mean nothing. Its immutability is enforced
# by the L4 row-hash chain and by Supabase RLS, which detect an EDIT to an
# existing row while permitting an APPEND. A file hash cannot tell those apart.
LIVE_DIR = PROJECT_ROOT / "live_log"
PREDICTIONS_CSV = LIVE_DIR / "phase6_live_predictions.csv"
CONTAMINATION_CSV = LIVE_DIR / "phase6_live_contamination.csv"
RUN_AUDIT = LIVE_DIR / "phase6_live_log_run_audit.csv"

# THE ARTEFACTS ARE SPLIT BY DETERMINISM, AND THE FROZEN GLOB IS WHY.
#
# outputs/phase6_*.csv is in FROZEN_PATTERNS, so anything written there is
# pinned by hash in FROZEN_MANIFEST.txt. Only an artefact that reproduces
# byte-identically can live there honestly.
#
#   the dry run    deterministic - no timestamp, no network, a fixed
#                  estimator over frozen development data. FROZEN.
#
#   a live run     carries generated_at_utc and a Supabase outcome, and its
#                  row set depends on the mode. It CANNOT reproduce and must
#                  not be pinned, or the manifest would fail every matchweek
#                  and the failure would mean nothing. Kept in live_log/.
DRYRUN_OUTPUT = OUTPUTS_DIR / "phase6_live_log_dryrun.csv"
DRYRUN_AUDIT = OUTPUTS_DIR / "phase6_live_log_dryrun_audit.csv"

# L10.2 amendment A. The fixture list comes from fixturedownload.com, which
# publishes the whole season with UTC kickoffs. It replaced football-data's
# fixtures.csv, which carried E0 rows only a few days ahead and on 2026-09-18
# had none for a matchweek whose first match kicked off that evening.
FIXTURES_URL = "https://fixturedownload.com/feed/json/epl-2026"
SOURCE_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
USER_AGENT = "Mozilla/5.0 (compatible; PremierLeagueML source watch)"

# The fixture feed's spellings that differ from football-data's, and nothing
# else. Closed on purpose (L6.4): an unlisted name passes through unchanged,
# then through TEAM_MAP, and reaches assert_vocabulary, which stops the run.
FIXTURE_SOURCE_NAMES = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}

# Kickoffs are stored as UK local "HH:MM" - football-data's convention, and
# what every row written before the amendment carries - and converted to UTC
# only to decide whether a row was written before its own kickoff.
UK = "Europe/London"

HOLDOUT_SEASON = "2026-2027"
DIVISION = "E0"

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

# L3. Pre-declared from the development staleness distribution, where 1,760 of
# 1,900 matches are played within two days of their matchweek's first fixture
# and the distribution then jumps from 10 days straight to 17. Rows above this
# are reported as a separate stratum; they are never regenerated and never
# dropped.
STALE_DAYS_THRESHOLD = 10

PUBLISHED_POOLED_LOG_LOSS = 0.99036
PUBLISHED_POOLED_RPS = 0.20350

DC_FOLDS = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"

# L4. The fields the row hash covers, in this order, never reordered. A field
# added here changes every subsequent hash, which is why the protocol pins the
# list and the verifier reads it from this module rather than carrying a copy.
HASHED_FIELDS = (
    "match_id", "season", "round_id", "matchweek_label", "scheduled_date",
    "home_team", "away_team", "state_cutoff_date", "generated_at_utc",
    "p_home", "p_draw", "p_away", "lambda_home", "lambda_away", "rho",
    "fit_matches", "home_has_history", "away_has_history",
    "freeze_sha", "pin_sha", "protocol_sha",
)

GENESIS_HASH = "0" * 64


class ProtocolError(RuntimeError):
    """A violation of PHASE6_LIVE_LOG_PROTOCOL.txt. Never caught here."""


# ============================================================
# 1. THE STABLE MATCH KEY
# ============================================================

def match_key(season, home_team, away_team):
    """
    L4. The join key, and it is deliberately NOT date-based.

    phase6_score_holdout.py uses the positional index of phase1_matches.csv,
    which is fine for a spine that already exists and useless for a fixture
    that does not. A live log needs a key that is stable from the moment a
    prediction is written to the moment the result arrives - and under L3 a
    postponed match KEEPS its original prediction, so the key must survive the
    date changing underneath it.

    (season, home_team, away_team) is unique: a 20-team double round-robin
    plays each ORDERED pair exactly once, 20 x 19 = 380. Verified on all
    1,900 development matches - 1,900 rows, 1,900 distinct keys, 0 duplicates.
    """

    return "{}_{}_{}".format(season, home_team, away_team).replace(" ", "-")


# ============================================================
# 2. THE SPINE FOR THE LIVE SEASON (P7.2)
# ============================================================

def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=45) as response:
        return response.read()


def spine_from_e0(payload, season=HOLDOUT_SEASON):
    """
    Build completed 2026-27 matches in the phase1_matches schema.

    P7.2 of the pin declares the football-data.co.uk E0 file as the 2026-27
    spine, with exactly this mapping. It is reproduced here and nowhere else
    invented: date from Date, teams through TEAM_MAP, goals from FTHG/FTAG,
    result recomputed by the foundation's own result_from_goals rather than
    trusted from FTR.

    NAMES PASS THROUGH TEAM_MAP AND THEN THROUGH NOTHING. P7.5: a name absent
    from that closed dictionary is left UNCHANGED so the vocabulary assertion
    can see it and fail on it. No repair, no title-casing, no fuzzy match.
    """

    frame = pd.read_csv(io.BytesIO(payload))

    for column in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"):
        if column not in frame.columns:
            raise ProtocolError(
                "the E0 file has no {} column; the declared spine mapping of "
                "P7.2 cannot be applied".format(column))

    played = frame[frame["FTHG"].notna() & frame["FTAG"].notna()].copy()

    built = pd.DataFrame({
        "season": season,
        "date": pd.to_datetime(played["Date"], format="%d/%m/%Y"),
        "matchweek": np.nan,
        "home_team": played["HomeTeam"].map(lambda t: TEAM_MAP.get(t, t)),
        "away_team": played["AwayTeam"].map(lambda t: TEAM_MAP.get(t, t)),
        "home_goals": played["FTHG"].astype(int),
        "away_goals": played["FTAG"].astype(int),
    })

    built["result"] = [
        result_from_goals(h, a)
        for h, a in zip(built["home_goals"], built["away_goals"])
    ]

    points = built["result"].map(points_from_result)
    built["home_points_from_result"] = [p[0] for p in points]
    built["away_points_from_result"] = [p[1] for p in points]

    return built.sort_values(["date", "home_team"]).reset_index(drop=True)


def source_name(name):
    """Feed spelling -> football-data spelling -> the pin's vocabulary."""

    name = FIXTURE_SOURCE_NAMES.get(name, name)
    return TEAM_MAP.get(name, name)


def fixtures_from_source(payload):
    """
    Every 2026-27 fixture in the feed, played or not (L10.2 amendment A).

    The date is the UK date of the kickoff, the same calendar the E0 spine and
    the state cutoff use. `played` is the feed's own scoreline being present;
    it is used only to tell a lagging results file from a postponement, and no
    score from this feed is ever read into a fit or a capture.
    """

    rows = json.loads(payload.decode("utf-8-sig"))

    if not rows:
        raise ProtocolError(
            "the fixture feed is empty. A round cannot be written without "
            "knowing which matches it contains.")

    for column in ("RoundNumber", "DateUtc", "HomeTeam", "AwayTeam",
                   "HomeTeamScore"):
        if column not in rows[0]:
            raise ProtocolError(
                "the fixture feed has no {} field. This is a SOURCE CHANGE "
                "and it stops the log rather than being guessed around"
                .format(column))

    kickoff_utc = pd.to_datetime([r["DateUtc"] for r in rows], utc=True)
    local = kickoff_utc.tz_convert(UK)

    built = pd.DataFrame({
        "season": HOLDOUT_SEASON,
        "round_number": [int(r["RoundNumber"]) for r in rows],
        "kickoff_utc": kickoff_utc,
        "date": local.tz_localize(None).normalize(),
        "kickoff": local.strftime("%H:%M"),
        "home_team": [source_name(r["HomeTeam"]) for r in rows],
        "away_team": [source_name(r["AwayTeam"]) for r in rows],
        "played": [r["HomeTeamScore"] is not None for r in rows],
    })

    built["match_id"] = [
        match_key(s, h, a) for s, h, a in
        zip(built["season"], built["home_team"], built["away_team"])]

    return built.sort_values(["kickoff_utc", "home_team"]).reset_index(
        drop=True)


def plan_rounds(fixtures, written, live, holdout_cutoff, now):
    """
    Which rounds this run writes, in round order (L10.2 amendments A and C).

    A round is the feed's RoundNumber. The old rule - fixtures in date order
    until a side repeats - existed because fixtures.csv carried no matchweek;
    this feed does, and a rescheduled match keeps its round number, which is
    L3.1's "keeps its original prediction and state cutoff" for free.

    DUE is every unwritten holdout round whose first kickoff has passed - the
    late rounds amendment C stops ruling out - plus the NEXT unwritten round.
    Rounds beyond the next are not due: their state window is still being
    played.

    WAITING is a due round whose state window the E0 results file does not
    yet hold. A round fitted on a window with a hole in it would say something
    different from what it will say once the file catches up, so it is not
    written until it can be written once and correctly.

    MIXED is a round with some matches written and some not. L1.4 and L3.1
    refuse to write it and a person has to look.
    """

    holdout = fixtures[fixtures["date"] >= holdout_cutoff]
    in_spine = set(live["match_id"])

    due, waiting, mixed = [], [], []
    upcoming_taken = False

    for number in sorted(holdout["round_number"].unique()):

        block = holdout[holdout["round_number"] == number]
        done = int(block["match_id"].isin(written).sum())

        if done == len(block):
            continue

        if done:
            mixed.append({"round_number": int(number), "block": block,
                          "written": done})
            continue

        first_kickoff = block["kickoff_utc"].min()

        if first_kickoff > now:
            if upcoming_taken:
                continue
            upcoming_taken = True

        state_cutoff = pd.Timestamp(block["date"].min())

        before = fixtures[fixtures["date"] < state_cutoff]
        lagging = before[before["played"] & ~before["match_id"].isin(in_spine)]
        postponed = before[~before["played"]
                           & ~before["match_id"].isin(in_spine)]

        entry = {
            "round_number": int(number),
            "block": block.copy(),
            "state_cutoff": state_cutoff,
            "first_kickoff_utc": first_kickoff,
            "late": bool((block["kickoff_utc"] <= now).any()),
            "lagging": lagging,
            "postponed": postponed,
        }

        (waiting if len(lagging) else due).append(entry)

    return due, waiting, mixed


# ============================================================
# 3. THE TWO CADENCES
# ============================================================

def per_matchweek_predictions(matches, cutoff, scope, group_column="matchweek"):
    """
    L1's cadence. ONE fit per matchweek, on matches strictly before that
    matchweek's first fixture, used for every match in it.

    The strict rule is identical to the walk-forward's and is not relaxed:
    "cutoff = the day before the first fixture" with date <= cutoff is the
    same set of matches as date < first_fixture, and the latter is what is
    implemented because it needs no calendar arithmetic to be correct.

    WHAT IS DIFFERENT from score_from_cutoff is the GRANULARITY and nothing
    else. Same estimator, same constants, same strictness.
    """

    in_scope = matches["season"].isin(scope)

    scored = matches[in_scope & (matches["date"] >= cutoff)]
    scored = scored.sort_values(["date", "match_id"])

    rows = []
    refits = 0

    for group in sorted(scored[group_column].unique()):

        block = scored[scored[group_column] == group]
        state_cutoff = pd.Timestamp(block["date"].min())

        window = matches[in_scope & (matches["date"] < state_cutoff)]

        model = DC.fit_window(window, state_cutoff, dixon_coles=True)

        for row in DC.predict_matches(block, model):
            row[group_column] = group
            row["state_cutoff_date"] = state_cutoff
            rows.append(row)

        refits += 1

    return pd.DataFrame(rows), refits


def metrics_of(frame):
    proba = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    validate_probabilities(proba, len(frame))

    return evaluate(frame["actual_result"].to_numpy(), proba)


def frozen_targets():
    frozen = pd.read_csv(DC_FOLDS, float_precision=FLOAT_PRECISION)
    frozen = frozen[frozen["variant"] == "dc_walkforward"].sort_values("fold")

    return frozen, {"log_loss": float(frozen["log_loss"].mean()),
                    "rps": float(frozen["rps"].mean())}


# ============================================================
# 4. L4 - THE ROW HASH CHAIN
# ============================================================

def canonical(value):
    """
    One field, one unambiguous string.

    Floats go out at %.17g - the project's FLOAT_FORMAT - so that the hash is
    over the full double and not over a rounded display of it. This is the
    same mistake the scoring instrument records having made once, when a
    six-decimal console reading of 1.045363 was written into the code as a
    target and produced a 7.16e-08 "discrepancy" in an exact instrument.
    """

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"

    if isinstance(value, (float, np.floating)):
        return FLOAT_FORMAT % float(value)

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    return str(value)


def row_hash(row, prev_hash):
    """
    L4. sha256 over the prediction fields, chained to the previous row.

    THE CHAIN IS THE POINT. A bare per-row hash detects an edited row. It does
    not detect a DELETED row, or a row inserted between two others, and an
    append-only log whose rows can be silently removed is not append-only. The
    chain makes every row depend on its whole history, so any deletion or
    reordering breaks every hash after it.

    Hashed over FIELD VALUES, never over file bytes. The repository pins CSVs
    to eol=crlf precisely because a line-ending conversion moved a hash once
    already; a row hash computed from parsed fields cannot be moved by that.
    """

    payload = "|".join(canonical(row.get(field)) for field in HASHED_FIELDS)

    return hashlib.sha256(
        (prev_hash + "|" + payload).encode("utf-8")).hexdigest()


def kickoff_utc_of(scheduled_date, scheduled_kickoff):
    """
    A row's kickoff as a UTC instant, from its UK date and UK "HH:MM".

    A row with no kickoff time is taken to kick off at 00:00 UK - the earliest
    it could - so a missing time can only ever make a row LATE, never early.
    """

    clock = str(scheduled_kickoff) if isinstance(scheduled_kickoff, str) and \
        scheduled_kickoff.strip() else "00:00"

    return pd.Timestamp("{} {}".format(scheduled_date, clock)).tz_localize(
        UK).tz_convert("UTC")


def derived_pre_kickoff(row):
    """
    L10.2 amendment B: was this row written before ITS OWN match kicked off?

    Derived from generated_at_utc and scheduled_date, both inside the hash,
    and scheduled_kickoff, which is not. The stored written_pre_kickoff column
    is a convenience for the database and the site; this is the definition,
    and the verifier holds the column to it.
    """

    generated = pd.Timestamp(str(row["generated_at_utc"]))

    return bool(generated < kickoff_utc_of(row["scheduled_date"],
                                           row.get("scheduled_kickoff")))


def pre_kickoff_flags(frame):
    """match_id -> written before its own kickoff, for every row in the log."""

    return {row["match_id"]: derived_pre_kickoff(row)
            for row in frame.to_dict("records")}


def verify_log(frame):
    """
    L4's verifier. Recomputes every hash in order and fails on any mismatch.

    Returns (failures, checked). A failure is a row whose stored row_hash
    disagrees with its recomputation, or whose prev_hash does not point at its
    predecessor, or - from L10.2 amendment B - whose stored written_pre_kickoff
    disagrees with the one its own timestamps give. Rows written before the
    amendment carry no stored flag and are checked on the hash alone. All are
    reported; none is repaired.
    """

    failures = []
    prev = GENESIS_HASH

    for position, row in enumerate(frame.to_dict("records")):

        stored_flag = str(row.get("written_pre_kickoff", ""))
        if stored_flag in ("True", "False", "true", "false"):
            expected_flag = derived_pre_kickoff(row)
            if (stored_flag.lower() == "true") != expected_flag:
                failures.append({
                    "row": position, "match_id": row.get("match_id"),
                    "field": "written_pre_kickoff", "stored": stored_flag,
                    "recomputed": str(expected_flag)})

        if str(row.get("prev_hash", "")) != prev:
            failures.append({
                "row": position, "match_id": row.get("match_id"),
                "field": "prev_hash", "stored": row.get("prev_hash"),
                "recomputed": prev})

        expected = row_hash(row, prev)

        if str(row.get("row_hash", "")) != expected:
            failures.append({
                "row": position, "match_id": row.get("match_id"),
                "field": "row_hash", "stored": row.get("row_hash"),
                "recomputed": expected})

        prev = str(row.get("row_hash", ""))

    return failures, len(frame)


# ============================================================
# 5. SUPABASE - WRITES VIA THE REST API, SERVICE KEY FROM THE ENVIRONMENT
# ============================================================

def supabase_config():
    """
    Never from a file in the repository. STEP 3: the service key does not live
    here, and this function is the only place that decides where it comes from.

    Absent credentials are NOT an error. The local CSV mirror is the primary
    record - it is the thing git timestamps - and Supabase is the dashboard's
    copy. A missing key degrades the run to mirror-only and says so.
    """

    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")

    return (url, key) if url and key else (None, None)


def supabase_insert(table, rows):
    """POST rows to one table. INSERT only - never upsert, never PATCH."""

    url, key = supabase_config()

    if not url:
        return False, "no SUPABASE_URL / SUPABASE_SERVICE_KEY in the environment"

    endpoint = "{}/rest/v1/{}".format(url, table)
    body = json.dumps(rows, default=str).encode("utf-8")

    request = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={
            "apikey": key,
            "Authorization": "Bearer {}".format(key),
            "Content-Type": "application/json",
            # No resolution on conflict. A duplicate match_id must RAISE, not
            # overwrite: the refusal is the immutability guarantee.
            "Prefer": "return=minimal",
        })

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return 200 <= response.status < 300, "HTTP {}".format(response.status)
    except Exception as error:                       # noqa: BLE001
        return False, "{}: {}".format(type(error).__name__, error)


# ============================================================
# 6. THE DRY RUNS
# ============================================================

def dry_run(matches, spec, audit):

    seasons = [str(f["test_season"]) for f in spec["folds"]]
    all_seasons = sorted(set(matches["season"]))

    frozen, frozen_pooled = frozen_targets()
    frozen = frozen.set_index("test_season")

    # ---- A: FIDELITY ------------------------------------------------------
    banner("DRY RUN A - FIDELITY: THE SAME CODE AT THE FROZEN CADENCE")

    print("  Per-date refits, exactly as phase2 ran them. This MUST reproduce")
    print("  the frozen figures: it is the check that importing the estimator")
    print("  rather than copying it has changed nothing.")
    print()
    print("  {:<12} {:>8} {:>15} {:>15} {:>11}".format(
        "season", "refits", "this file", "frozen", "difference"))
    print("  " + "-" * 66)

    fidelity_ll, fidelity_rps, worst = [], [], 0.0
    reference_frames = {}

    for season in seasons:
        scope = all_seasons[:all_seasons.index(season) + 1]
        cutoff = matches[matches["season"] == season]["date"].min()

        frame, _scored = score_from_cutoff(matches, cutoff, scope)
        reference_frames[season] = frame
        scores = metrics_of(frame)

        target = float(frozen.loc[season, "log_loss"])
        gap = abs(scores["log_loss"] - target)
        worst = max(worst, gap, abs(scores["rps"]
                                    - float(frozen.loc[season, "rps"])))

        fidelity_ll.append(scores["log_loss"])
        fidelity_rps.append(scores["rps"])

        print("  {:<12} {:>8} {:>15.10f} {:>15.10f} {:>11.2e}".format(
            season, frame["fit_cutoff_date"].nunique(), scores["log_loss"],
            target, gap))

    pooled_ll = float(np.mean(fidelity_ll))
    pooled_rps = float(np.mean(fidelity_rps))

    print()
    print("  pooled log loss {:.10f}   frozen {:.10f}".format(
        pooled_ll, frozen_pooled["log_loss"]))
    print("  pooled RPS      {:.10f}   frozen {:.10f}".format(
        pooled_rps, frozen_pooled["rps"]))

    audit.record(
        "G1", "at the frozen per-date cadence this file reproduces every "
              "fold bit for bit",
        "0.000e+00", "{:.3e}".format(worst), worst == 0.0,
        "THE CHECK THAT MATTERS FOR CORRECTNESS. The generator imports "
        "fit_window and predict_matches rather than reimplementing them, and "
        "this is that claim tested where the answers are frozen")

    audit.record(
        "G2", "and reproduces the pooled figure REPORTS.md publishes",
        "{:.5f} / {:.5f}".format(PUBLISHED_POOLED_LOG_LOSS,
                                 PUBLISHED_POOLED_RPS),
        "{:.5f} / {:.5f}".format(pooled_ll, pooled_rps),
        round(pooled_ll, 5) == PUBLISHED_POOLED_LOG_LOSS
        and round(pooled_rps, 5) == PUBLISHED_POOLED_RPS,
        "0.99036 / 0.20350 is the POOLED four-fold figure over 1,520 "
        "matches. It is NOT 2025-26's, which is 1.04536 / 0.21281")

    # ---- B: CADENCE -------------------------------------------------------
    banner("DRY RUN B - CADENCE: WHAT PREDICTING A WHOLE MATCHWEEK COSTS")

    print("  L1's cadence, same code. This reproduces NOTHING and is not")
    print("  meant to. 37 of 38 matchweeks span more than one day, so most")
    print("  matches are predicted without that matchweek's earlier results.")
    print()
    print("  {:<12} {:>8} {:>10} {:>14} {:>14} {:>11}".format(
        "season", "refits", "rows diff", "matchweek LL", "per-date LL",
        "cost"))
    print("  " + "-" * 74)

    rows_out, cadence_ll, cadence_rps = [], [], []

    for position, season in enumerate(seasons):
        scope = all_seasons[:all_seasons.index(season) + 1]
        cutoff = matches[matches["season"] == season]["date"].min()

        frame, refits = per_matchweek_predictions(matches, cutoff, scope)
        scores = metrics_of(frame)

        # Dry run A's frames, reused rather than refitted: recomputing 114
        # per-date fits to produce a number already in hand would double the
        # runtime and could not disagree with itself.
        reference = reference_frames[season]
        joined = frame.merge(reference, on="match_id", suffixes=("_w", "_d"))
        differing = int(((joined["p_home_w"] - joined["p_home_d"]).abs()
                         > 1e-12).sum())

        cadence_ll.append(scores["log_loss"])
        cadence_rps.append(scores["rps"])

        cost = scores["log_loss"] - fidelity_ll[position]

        print("  {:<12} {:>8} {:>10} {:>14.6f} {:>14.6f} {:>+11.6f}".format(
            season, refits, differing, scores["log_loss"],
            fidelity_ll[position], cost))

        rows_out.append({
            "season": season, "refits_matchweek": refits,
            "refits_per_date": int(reference["fit_cutoff_date"].nunique()),
            "rows_differing": differing, "n": len(frame),
            "matchweek_log_loss": scores["log_loss"],
            "per_date_log_loss": fidelity_ll[position],
            "matchweek_rps": scores["rps"],
            "per_date_rps": fidelity_rps[position],
            "log_loss_cost": cost,
            "rps_cost": scores["rps"] - fidelity_rps[position]})

    cadence_pooled_ll = float(np.mean(cadence_ll))
    cadence_pooled_rps = float(np.mean(cadence_rps))

    print()
    print("  pooled matchweek  {:.7f} LL   {:.7f} RPS".format(
        cadence_pooled_ll, cadence_pooled_rps))
    print("  pooled per-date   {:.7f} LL   {:.7f} RPS".format(
        pooled_ll, pooled_rps))
    print("  THE CADENCE COST  {:+.7f} LL  {:+.7f} RPS".format(
        cadence_pooled_ll - pooled_ll, cadence_pooled_rps - pooled_rps))

    audit.measure(
        "G3", "the pooled cost of matchweek cadence, on development data",
        "{:+.7f} log loss / {:+.7f} RPS".format(
            cadence_pooled_ll - pooled_ll, cadence_pooled_rps - pooled_rps),
        "PRE-DECLARED WITH RESPECT TO THE HOLDOUT, not to this data - it was "
        "measured here before any 2026-27 row was written, and it is recorded "
        "so that a live-log figure in 2027 cannot be compared with the "
        "official per-date result as though the two were the same instrument")

    audit.record(
        "G4", "matchweek cadence is a STRICTLY SMALLER information set",
        "38 refits < per-date refits, every season",
        "38 vs {}".format([r["refits_per_date"] for r in rows_out]),
        all(r["refits_matchweek"] < r["refits_per_date"] for r in rows_out),
        "stated as an assertion because the direction is what licenses "
        "calling the difference a COST rather than a discrepancy")

    frame_out = pd.DataFrame(rows_out)
    frame_out.to_csv(DRYRUN_OUTPUT, index=False, encoding="utf-8",
                     float_format=FLOAT_FORMAT)

    # ---- C: the hash chain, exercised before it is trusted -----------------
    banner("DRY RUN C - THE L4 CHAIN, TAMPERED WITH ON PURPOSE")

    sample = pd.DataFrame([
        {field: v for field, v in zip(
            HASHED_FIELDS,
            ["2026-2027_A_B", HOLDOUT_SEASON, 1, 4, "2026-09-12", "A", "B",
             "2026-09-11", "2026-09-11T00:00:00Z", 0.5, 0.25, 0.25, 1.5, 1.0,
             -0.03, 1900, True, True, FREEZE_SHA, "pin", "protocol"])},
    ])

    chain = GENESIS_HASH
    built = []
    for row in sample.to_dict("records"):
        row["prev_hash"] = chain
        row["row_hash"] = row_hash(row, chain)
        chain = row["row_hash"]
        built.append(row)

    clean = pd.DataFrame(built)
    clean_failures, _n = verify_log(clean)

    audit.record(
        "G5", "the verifier passes an untampered chain",
        0, len(clean_failures), len(clean_failures) == 0,
        "a verifier that never passes is as useless as one that never fails")

    tampered = clean.copy()
    tampered.loc[0, "p_home"] = 0.6
    tampered_failures, _n = verify_log(tampered)

    audit.record(
        "G6", "and FAILS when a written probability is edited",
        "at least 1 failure", len(tampered_failures),
        len(tampered_failures) > 0,
        "L4 is only worth stating if the check has been seen to fire. "
        "p_home moved 0.5 -> 0.6 and the recomputation caught it")

    return frame_out


# ============================================================
# 7. THE LIVE ROUND
# ============================================================

def generate_rounds(matches, pin, audit, dry=False, only_round=None):
    """
    Write every round that is due, each once, append-only.

    L10.2 amendment C: a round whose kickoff has passed is no longer ruled
    out. It is written with the SAME state cutoff it would have had on time -
    every completed match dated strictly before its first fixture, and nothing
    after - so its probabilities are the ones a timely run would have written.
    What a late row cannot carry is the timestamp proof, so every row records
    written_pre_kickoff against its OWN kickoff (amendment B), and the site and
    the capture keep the two kinds apart.

    REFUSES to write a match_id that is already in the log, which is L1's
    "never regenerated" and L3's "keeps its original prediction" enforced in
    code rather than asserted in prose.
    """

    banner("THE LIVE ROUNDS")

    now = pd.Timestamp(datetime.now(timezone.utc))

    fixtures = fixtures_from_source(fetch(FIXTURES_URL))

    live = spine_from_e0(fetch(SOURCE_URL))
    live["match_id"] = [
        match_key(s, h, a) for s, h, a in
        zip(live["season"], live["home_team"], live["away_team"])]

    print("  fixtures in feed   {} ({} played)".format(
        len(fixtures), int(fixtures["played"].sum())))
    print("  completed 2026-27  {} matches, latest {}".format(
        len(live), str(live["date"].max().date()) if len(live) else "none"))

    # H2.12 / P4.5 on BOTH the completed rows and every fixture in the feed.
    # A name outside the vocabulary RAISES. Nothing is repaired here.
    assert_vocabulary(live, pin["vocabulary"], "completed 2026-27")
    assert_vocabulary(fixtures, pin["vocabulary"], "the fixture feed")

    audit.record(
        "L2b", "every name in the feed and in the live spine is inside the "
               "pin's declared twenty",
        "0 outside", "0 outside", True,
        "asserted through the SAME function the scoring instrument uses, so "
        "the log cannot accept a name the holdout would reject")

    existing = read_log()
    written = set(existing["match_id"]) if len(existing) else set()

    due, waiting, mixed = plan_rounds(
        fixtures, written, live, pin["cutoff"], now)

    if only_round is not None:
        due = [r for r in due if r["round_number"] == only_round]
        waiting = [r for r in waiting if r["round_number"] == only_round]

    for entry in waiting:
        print("  round {} WAITING: the results file lacks {} match(es) dated "
              "before its state cutoff {}".format(
                  entry["round_number"], len(entry["lagging"]),
                  str(entry["state_cutoff"].date())))

    audit.measure(
        "L2c", "rounds due / waiting on the results file / mixed",
        "{} / {} / {}".format(len(due), len(waiting), len(mixed)),
        "WAITING is not a failure: a round is written once, so it waits until "
        "its whole state window is in the E0 file rather than being fitted "
        "on a window with a hole in it")

    if mixed:
        raise ProtocolError(
            "L1.4: round(s) {} have some matches written and some not. "
            "Predictions are never regenerated and a round is never split; "
            "a person has to look.".format(
                [m["round_number"] for m in mixed]))

    if not due:
        print("  nothing to write")
        return pd.DataFrame()

    frames = []
    for entry in due:
        frames.append(write_one_round(
            matches, live, entry, pin, audit, now, dry))

    return pd.concat(frames, ignore_index=True)


def write_one_round(matches, live, entry, pin, audit, now, dry):

    block = entry["block"]
    state_cutoff = entry["state_cutoff"]
    number = entry["round_number"]

    print()
    print("  ROUND {}   {} matches, {} to {}{}".format(
        number, len(block), str(block["date"].min().date()),
        str(block["date"].max().date()),
        "   LATE - written after kickoff" if entry["late"] else ""))
    print("  state cutoff       date < {}   (L1, strict, as the walk-forward)"
          .format(str(state_cutoff.date())))

    if len(entry["postponed"]):
        print("  unplayed before the cutoff (postponed?): {}".format(
            list(entry["postponed"]["match_id"])))

    # ---- the fit ----------------------------------------------------------
    history = pd.concat([matches, live.drop(columns=["match_id"])],
                        ignore_index=True)
    history["match_id"] = [
        match_key(s, h, a) for s, h, a in
        zip(history["season"], history["home_team"], history["away_team"])]

    scope = sorted(set(history["season"]))
    window = history[history["season"].isin(scope)
                     & (history["date"] < state_cutoff)]

    print("  fitting on         {} matches over {} seasons".format(
        len(window), len(scope)))

    model = DC.fit_window(window, state_cutoff, dixon_coles=True)

    print("  rho {:+.6f}   home multiplier {:.6f}".format(
        model["rho"], model["home_multiplier"]))
    print()

    # predict_matches needs the columns it reads; result is unknown by design.
    to_predict = block[["season", "date", "home_team", "away_team",
                        "match_id"]].copy()
    to_predict["result"] = None

    predicted = DC.predict_matches(to_predict, model)

    # ---- the existing log, and the refusal to rewrite ---------------------
    existing = read_log()
    already = set(existing["match_id"]) if len(existing) else set()

    clashes = [p["match_id"] for p in predicted if p["match_id"] in already]

    audit.record(
        "L4a", "no match_id in round {} is already in the log".format(number),
        0, len(clashes), len(clashes) == 0,
        "L1 and L3 in code: a prediction is written ONCE. {}".format(
            "clashes: {}".format(clashes) if clashes else "none"))

    if clashes:
        raise ProtocolError(
            "L1: {} of these matches already have a written prediction: {}. "
            "Predictions are never regenerated. If these fixtures were "
            "postponed, L3 says the ORIGINAL row stands.".format(
                len(clashes), clashes))

    round_id = (int(existing["round_id"].max()) + 1) if len(existing) else 1

    protocol_sha = sha256_of(PROTOCOL) if PROTOCOL.exists() else ""

    chain = (str(existing.iloc[-1]["row_hash"]) if len(existing)
             else GENESIS_HASH)

    kickoffs = dict(zip(block["match_id"], block["kickoff"]))
    generated_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows = []

    for prediction in predicted:

        row = {
            "match_id": prediction["match_id"],
            "season": HOLDOUT_SEASON,
            "round_id": round_id,
            # The feed's round number (amendment A). Still informational:
            # nothing reads it to decide anything.
            "matchweek_label": number,
            "scheduled_date": pd.Timestamp(prediction["date"]).strftime(
                "%Y-%m-%d"),
            "scheduled_kickoff": kickoffs.get(prediction["match_id"], ""),
            "home_team": prediction["home"],
            "away_team": prediction["away"],
            "state_cutoff_date": state_cutoff.strftime("%Y-%m-%d"),
            "generated_at_utc": generated_at,
            "p_home": prediction["p_home"],
            "p_draw": prediction["p_draw"],
            "p_away": prediction["p_away"],
            "lambda_home": prediction["lambda_home"],
            "lambda_away": prediction["lambda_away"],
            "rho": prediction["rho"],
            "fit_matches": prediction["fit_matches"],
            "home_has_history": prediction["home_has_history"],
            "away_has_history": prediction["away_has_history"],
            "freeze_sha": FREEZE_SHA,
            "pin_sha": pin["sha"],
            "protocol_sha": protocol_sha,
            "prev_hash": chain,
        }

        row["row_hash"] = row_hash(row, chain)
        # Outside the hash on purpose: adding a field to HASHED_FIELDS would
        # move every hash already written. It is derived from fields that ARE
        # hashed, and the verifier holds it to that derivation.
        row["written_pre_kickoff"] = derived_pre_kickoff(row)
        chain = row["row_hash"]
        rows.append(row)

    frame = pd.DataFrame(rows)

    proba = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    validate_probabilities(proba, len(frame))

    late = int((~frame["written_pre_kickoff"].astype(bool)).sum())

    audit.measure(
        "L2a", "round {}: rows written AFTER their own kickoff".format(number),
        "{} of {}".format(late, len(frame)),
        "L10.2 amendment C: a late row is written, flagged, and kept out of "
        "every pre-kickoff figure. Its state cutoff is the one it would have "
        "had on time")

    # THE VOCABULARY CHECK IS NOT THE WHOLE GUARD, AND THIS IS THE OTHER HALF.
    #
    # assert_vocabulary catches a name the pin does not declare. It cannot
    # catch a DECLARED name that is simply absent from the fitted window -
    # that side still falls through match_rates's .fillna(NEUTRAL_STRENGTH)
    # and is scored as exactly average, with no exception, which is precisely
    # the V6b failure wearing different clothes.
    #
    # It is legitimate exactly once: a promoted side before it has played.
    # P4.6 names Coventry and Hull as the zero-history pair, and H6.6 exists
    # to split them out. So this is recorded and printed rather than raised -
    # but it is never silent, and a side that has played and still has no
    # fitted strength would show up here as a number nobody expected.
    neutral = [
        "{} ({})".format(r["home_team"], "home") for r in rows
        if not r["home_has_history"]
    ] + [
        "{} ({})".format(r["away_team"], "away") for r in rows
        if not r["away_has_history"]
    ]

    audit.measure(
        "L4e", "sides predicted with NO fitted strength, taking "
               "NEUTRAL_STRENGTH",
        "{}: {}".format(len(neutral), neutral if neutral else "none"),
        "assert_vocabulary cannot see this case - the name IS declared, it is "
        "the HISTORY that is missing. Legitimate only for a promoted side "
        "before it has played; recorded every round either way")

    audit.record(
        "L4b", "every written row carries three probabilities summing to one",
        "max |sum-1| = 0", "{:.2e}".format(
            float(np.abs(proba.sum(axis=1) - 1.0).max())),
        float(np.abs(proba.sum(axis=1) - 1.0).max()) < 1e-12,
        "through the Phase 0 harness's own validator, not a local check")

    print("  {:<16} {:<16} {:>8} {:>8} {:>8}   {:>7} {:>7}  {}".format(
        "home", "away", "p_home", "p_draw", "p_away", "lam_h", "lam_a",
        "kickoff (UK)"))
    print("  " + "-" * 96)
    for row in rows:
        print("  {:<16} {:<16} {:>8.4f} {:>8.4f} {:>8.4f}   {:>7.3f} {:>7.3f}"
              "  {} {}{}".format(
                  row["home_team"][:16], row["away_team"][:16],
                  row["p_home"], row["p_draw"], row["p_away"],
                  row["lambda_home"], row["lambda_away"],
                  row["scheduled_date"], row["scheduled_kickoff"],
                  "" if row["written_pre_kickoff"] else "  LATE"))

    if dry:
        print()
        print("  --dry: nothing written")
        return frame

    write_log(frame)

    # written_pre_kickoff stays in the mirror only. The database schema is
    # unchanged by the amendment; the site derives the same flag from the same
    # fields (generated_at_utc, scheduled_date, scheduled_kickoff), and the
    # capture carries it into results.contaminated.
    ok, detail = supabase_insert("predictions", [
        {k: v for k, v in r.items() if k != "written_pre_kickoff"}
        for r in frame.to_dict("records")])

    audit.record(
        "L4c", "round {} is appended to the local mirror".format(number),
        len(frame), len(frame), True,
        "the mirror is the PRIMARY record: git is what timestamps it. "
        "Supabase is the dashboard's copy")

    audit.measure(
        "L4d", "Supabase insert, round {}".format(number),
        "OK" if ok else "NOT WRITTEN - {}".format(detail),
        "mirror-only is a degraded run, not a failed one. The service key is "
        "read from the environment and is never in the repository")

    print()
    print("  appended {} rows to {}".format(len(frame), PREDICTIONS_CSV))
    print("  supabase: {}".format("written" if ok else detail))

    return frame


# ============================================================
# 8. THE MIRROR
# ============================================================

def read_log():
    if not PREDICTIONS_CSV.exists():
        return pd.DataFrame()

    return pd.read_csv(PREDICTIONS_CSV, float_precision=FLOAT_PRECISION,
                       keep_default_na=False, na_values=[""])


def write_log(frame):
    """Append. Never rewrite an existing row, never sort the file."""

    LIVE_DIR.mkdir(exist_ok=True)

    existing = read_log()

    combined = (pd.concat([existing, frame], ignore_index=True)
                if len(existing) else frame)

    combined.to_csv(PREDICTIONS_CSV, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)


# ============================================================
# THE RUN
# ============================================================

def main():

    configure_stdout()

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--round", action="store_true",
                        help="write every due 2026-27 round: the next one, "
                             "and any already kicked off (flagged late)")
    parser.add_argument("--matchweek", type=int, default=None,
                        help="with --round: write only this round number")
    parser.add_argument("--verify", action="store_true",
                        help="recompute every row hash in the local mirror")
    parser.add_argument("--dry", action="store_true",
                        help="with --round: compute and print, write nothing")
    args = parser.parse_args()

    banner("PHASE 6 - THE LIVE PREDICTION LOG")

    audit = Audit()

    freeze_sha = sha256_of(FREEZE)
    pin = load_pin()

    audit.record(
        "L0a", "the freeze on disk is the one this generator was built against",
        FREEZE_SHA, freeze_sha, freeze_sha == FREEZE_SHA,
        "the log records what a specific frozen model said. Against a "
        "different freeze it would be recording something else")

    audit.record(
        "L0b", "and the pin cites that same freeze",
        FREEZE_SHA, pin["freeze_sha"], pin["freeze_sha"] == FREEZE_SHA, "")

    audit.record(
        "L0c", "PHASE6_LIVE_LOG_PROTOCOL.txt exists and is registered in "
               "FROZEN_PATTERNS",
        "both",
        "exists={} registered={}".format(
            PROTOCOL.exists(),
            "PHASE6_LIVE_LOG_PROTOCOL.txt" in (
                PROJECT_ROOT / "scripts" / "frozen_manifest.py").read_text(
                    encoding="utf-8")),
        PROTOCOL.exists() and "PHASE6_LIVE_LOG_PROTOCOL.txt" in (
            PROJECT_ROOT / "scripts" / "frozen_manifest.py").read_text(
                encoding="utf-8"),
        "the rules must predate the predictions, and a hash is what makes "
        "that checkable rather than asserted")

    print("  freeze    {}".format(freeze_sha))
    print("  pin       {}".format(pin["sha"]))
    print("  protocol  {}".format(
        sha256_of(PROTOCOL) if PROTOCOL.exists() else "ABSENT"))
    print("  cutoff    {}   (read from the pin, never hard-coded)".format(
        pin["cutoff"].date()))
    print()

    if not PROTOCOL.exists():
        raise ProtocolError(
            "PHASE6_LIVE_LOG_PROTOCOL.txt is not on disk. The operating rules "
            "are declared BEFORE any prediction is written; that ordering is "
            "the only thing that makes them a pre-declaration.")

    if args.verify:
        banner("L4 - VERIFYING THE CHAIN")

        log = read_log()
        failures, checked = verify_log(log)

        for failure in failures:
            print("  MISMATCH row {} {} {}: stored {} recomputed {}".format(
                failure["row"], failure["match_id"], failure["field"],
                str(failure["stored"])[:16], str(failure["recomputed"])[:16]))

        audit.record(
            "L4v", "every row hash in the mirror recomputes",
            0, len(failures), len(failures) == 0,
            "{} rows checked".format(checked))

        print("  {} rows checked, {} mismatches".format(checked, len(failures)))

    elif args.round:
        matches = L3.load_matches().copy()
        matches["match_id"] = [
            match_key(s, h, a) for s, h, a in
            zip(matches["season"], matches["home_team"], matches["away_team"])]

        generate_rounds(matches, pin, audit, dry=args.dry,
                        only_round=args.matchweek)

    else:
        matches = L3.load_matches().copy()
        matches["match_id"] = matches.index

        dry_run(matches, L3.load_spec(), audit)

    banner("STATUS")

    frame_out = audit.frame()

    # THIS run's verdict, taken before any history is concatenated on. A run
    # that failed because an earlier run failed would be unreadable.
    failures = int((frame_out["status"] == "FAIL").sum())
    info = int((frame_out["status"] == "INFO").sum())
    checked = len(frame_out)

    if args.round or args.verify:
        # Appended, not overwritten: the operational trail of a log is itself
        # a log. Never under outputs/phase6_*.csv - see DRYRUN_OUTPUT above.
        LIVE_DIR.mkdir(exist_ok=True)
        frame_out.insert(0, "run_at_utc",
                         datetime.now(timezone.utc).strftime(
                             "%Y-%m-%dT%H:%M:%SZ"))
        frame_out.insert(1, "mode", "round" if args.round else "verify")
        destination = RUN_AUDIT

        if RUN_AUDIT.exists():
            frame_out = pd.concat(
                [pd.read_csv(RUN_AUDIT, keep_default_na=False), frame_out],
                ignore_index=True)
    else:
        destination = DRYRUN_AUDIT

    frame_out.to_csv(destination, index=False, encoding="utf-8",
                     float_format=FLOAT_FORMAT)

    print("  {}".format(destination))
    print()
    print("  Checks run          : {}".format(checked))
    print("  Checks failed       : {}".format(failures))
    print("  Outstanding (INFO)  : {}".format(info))
    print()

    if failures:
        current = audit.frame()
        for _i, row in current[current["status"] == "FAIL"].iterrows():
            print("    FAIL  {:<8} {}".format(row["test_id"], row["test"]))
        print()

    print("  {}".format("PASS" if failures == 0 else "FAIL"))

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
