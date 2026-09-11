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

    ... --round            write the next 2026-27 round, pre-kickoff
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

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
SOURCE_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
USER_AGENT = "Mozilla/5.0 (compatible; PremierLeagueML source watch)"

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


def fixtures_from_source(payload):
    """The upcoming E0 fixtures, mapped the same way and not otherwise touched."""

    frame = pd.read_csv(io.BytesIO(payload))

    upcoming = frame[frame["Div"] == DIVISION].copy()

    if upcoming.empty:
        raise ProtocolError(
            "the fixture list carries no {} rows. A round cannot be written "
            "without knowing which matches it contains.".format(DIVISION))

    built = pd.DataFrame({
        "season": HOLDOUT_SEASON,
        "date": pd.to_datetime(upcoming["Date"], format="%d/%m/%Y"),
        "kickoff": upcoming["Time"] if "Time" in upcoming.columns else "",
        "home_team": upcoming["HomeTeam"].map(lambda t: TEAM_MAP.get(t, t)),
        "away_team": upcoming["AwayTeam"].map(lambda t: TEAM_MAP.get(t, t)),
    })

    return built.sort_values(["date", "kickoff", "home_team"]).reset_index(
        drop=True)


def next_round(fixtures):
    """
    One prediction round: fixtures in date order until a side would repeat.

    L1 says once per matchweek, and this is what defines the boundary without
    inventing a matchweek LABEL. P7.2 deliberately does not carry matchweek -
    H3.4 makes the cutoff a date rule - so the round is derived from the only
    thing that is actually true of a matchweek: no team plays twice in one.

    That also makes the rule robust to the fixture feed carrying more or fewer
    than ten matches, which it will whenever a midweek card is published
    alongside a weekend one.
    """

    used = set()
    keep = []

    for position, row in enumerate(fixtures.itertuples()):

        if row.home_team in used or row.away_team in used:
            break

        used.add(row.home_team)
        used.add(row.away_team)
        keep.append(position)

    return fixtures.iloc[keep].copy()


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


def verify_log(frame):
    """
    L4's verifier. Recomputes every hash in order and fails on any mismatch.

    Returns (failures, checked). A failure is a row whose stored row_hash
    disagrees with its recomputation, or whose prev_hash does not point at its
    predecessor. Both are reported; neither is repaired.
    """

    failures = []
    prev = GENESIS_HASH

    for position, row in enumerate(frame.to_dict("records")):

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

def generate_round(matches, pin, audit, dry=False):
    """
    Write one round of pre-kickoff predictions. Append-only.

    REFUSES to write a match_id that is already in the log, which is L1's
    "never regenerated" and L3's "keeps its original prediction" enforced in
    code rather than asserted in prose.
    """

    banner("THE LIVE ROUND")

    fixtures = fixtures_from_source(fetch(FIXTURES_URL))
    block = next_round(fixtures)

    if block.empty:
        raise ProtocolError("the fixture feed produced an empty round")

    first_kickoff = pd.Timestamp(block["date"].min())
    state_cutoff = first_kickoff

    print("  fixtures in feed   {}".format(len(fixtures)))
    print("  this round         {} matches, {} to {}".format(
        len(block), str(block["date"].min().date()),
        str(block["date"].max().date())))
    print("  state cutoff       date < {}   (L1, strict, as the walk-forward)"
          .format(str(state_cutoff.date())))

    now = datetime.now(timezone.utc)

    # ---- L1/L2: is this actually pre-kickoff? -----------------------------
    pre_kickoff = now < first_kickoff.tz_localize(timezone.utc)

    audit.record(
        "L2a", "the round is being written BEFORE its first kickoff",
        "generated_at < {}".format(str(first_kickoff.date())),
        now.strftime("%Y-%m-%dT%H:%M:%SZ"), pre_kickoff,
        "L2: a round written after its first kickoff is CONTAMINATED and is "
        "excluded from the live log's own metrics. It is never excluded from "
        "the holdout, which no rule in this protocol may alter")

    if not pre_kickoff:
        raise ProtocolError(
            "L2: the first fixture of this round kicked off at {} and it is "
            "now {}. A prediction written after kickoff is not evidence of "
            "anything. Record the gap in {} instead - do NOT write a late "
            "row.".format(first_kickoff, now, CONTAMINATION_CSV.name))

    # ---- the live spine, and the vocabulary assertion ---------------------
    live = spine_from_e0(fetch(SOURCE_URL))

    print("  completed 2026-27  {} matches, latest {}".format(
        len(live), str(live["date"].max().date()) if len(live) else "none"))

    # H2.12 / P4.5 on BOTH the completed rows and the fixtures being predicted.
    # A name outside the vocabulary RAISES. Nothing is repaired here.
    assert_vocabulary(live, pin["vocabulary"], "completed 2026-27")
    assert_vocabulary(block, pin["vocabulary"], "the round being predicted")

    audit.record(
        "L2b", "every name in the round and in the live spine is inside the "
               "pin's declared twenty",
        "0 outside", "0 outside", True,
        "asserted through the SAME function the scoring instrument uses, so "
        "the log cannot accept a name the holdout would reject")

    # ---- the fit ----------------------------------------------------------
    history = pd.concat([matches, live], ignore_index=True)
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
    to_predict = block.copy()
    to_predict["match_id"] = [
        match_key(s, h, a) for s, h, a in
        zip(to_predict["season"], to_predict["home_team"],
            to_predict["away_team"])]
    to_predict["result"] = None

    predicted = DC.predict_matches(to_predict, model)

    # ---- the existing log, and the refusal to rewrite ---------------------
    existing = read_log()
    already = set(existing["match_id"]) if len(existing) else set()

    clashes = [p["match_id"] for p in predicted if p["match_id"] in already]

    audit.record(
        "L4a", "no match_id in this round is already in the log",
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
    played = int((~live["home_team"].isna()).sum())
    matchweek_label = played // 10 + 1

    protocol_sha = sha256_of(PROTOCOL) if PROTOCOL.exists() else ""

    chain = (str(existing.iloc[-1]["row_hash"]) if len(existing)
             else GENESIS_HASH)

    rows = []
    kickoffs = dict(zip(block["home_team"], block.get("kickoff", "")))

    for prediction in predicted:

        row = {
            "match_id": prediction["match_id"],
            "season": HOLDOUT_SEASON,
            "round_id": round_id,
            "matchweek_label": matchweek_label,
            "scheduled_date": pd.Timestamp(prediction["date"]).strftime(
                "%Y-%m-%d"),
            "scheduled_kickoff": kickoffs.get(prediction["home"], ""),
            "home_team": prediction["home"],
            "away_team": prediction["away"],
            "state_cutoff_date": state_cutoff.strftime("%Y-%m-%d"),
            "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
        chain = row["row_hash"]
        rows.append(row)

    frame = pd.DataFrame(rows)

    proba = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    validate_probabilities(proba, len(frame))

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

    print("  {:<16} {:<16} {:>8} {:>8} {:>8}   {:>7} {:>7}".format(
        "home", "away", "p_home", "p_draw", "p_away", "lam_h", "lam_a"))
    print("  " + "-" * 78)
    for row in rows:
        print("  {:<16} {:<16} {:>8.4f} {:>8.4f} {:>8.4f}   {:>7.3f} {:>7.3f}"
              .format(row["home_team"][:16], row["away_team"][:16],
                      row["p_home"], row["p_draw"], row["p_away"],
                      row["lambda_home"], row["lambda_away"]))

    if dry:
        print()
        print("  --dry: nothing written")
        return frame

    write_log(frame)

    ok, detail = supabase_insert("predictions", frame.to_dict("records"))

    audit.record(
        "L4c", "the round is appended to the local mirror",
        len(frame), len(frame), True,
        "the mirror is the PRIMARY record: git is what timestamps it. "
        "Supabase is the dashboard's copy")

    audit.measure(
        "L4d", "Supabase insert", "OK" if ok else "NOT WRITTEN - {}".format(detail),
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
                        help="write the next 2026-27 round, pre-kickoff")
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

        generate_round(matches, pin, audit, dry=args.dry)

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
