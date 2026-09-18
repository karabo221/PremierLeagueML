"""
===============================================================================
PHASE 6 - RESULTS AND MARKET CAPTURE FOR THE LIVE LOG
===============================================================================

Governed by PHASE6_LIVE_LOG_PROTOCOL.txt, PHASE6_HOLDOUT_FREEZE.txt
(sha256 b36befd3...248f6) and PHASE6_CUTOFF_PIN.txt.

Run after each matchweek. Pulls results and Bet365 CLOSING odds from the
football-data.co.uk E0 file that P7.2 declares as the 2026-27 spine and that
the source watch already checks monthly.

IT COMPUTES NO METRIC AND SCORES NOTHING. L5.1 keeps the official 2026-27
result in phase6_score_holdout.py, run once after the final fixture behind two
explicit flags. This file records what happened; it does not grade it.

WHAT IT DECIDES, AND WHAT IT ONLY RECORDS:

    L3.4  state_age_days = date played - state cutoff. Computed here, because
          at generation the date played is not yet known.
    L3.5  is_stale, at the pre-declared 10-day threshold.
    L2    a holdout match with NO pre-kickoff prediction is written to the
          contamination record. It is excluded from the LOG's figures and is
          NEVER excluded from the holdout - see L2.1 and L2.2.

PINNACLE IS GONE AND NOTHING REPLACES IT. SW5-PSC found PSCH/PSCD/PSCA absent
from the 2026-27 file. B365C* is present and is the declared PRIMARY at B3.3,
chosen on completeness before any score existed. Promoting another book to
fill the sensitivity arm would be choosing an instrument after seeing which
one survived, so the arm stays empty and REPORT.md section 7 records it as
reduced scope.

THE SCORELINES DO NOT ENTER THE REPOSITORY. .gitignore keeps
live_log/phase6_live_results.csv and phase6_live_market.csv out of git on the
same terms as data/watch/, until the holdout is deliberately acquired in May
2027. L4.8.

    ./venv/Scripts/python.exe -B scripts/phase6_capture_results.py
        capture everything played that has a prediction and is not yet stored

    ... --dry      compute and print, write nothing
"""

from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

from phase1_match_foundation import result_from_goals  # noqa: E402
from phase5_market_benchmark import (                  # noqa: E402
    TEAM_MAP, devig_proportional)
from phase6_score_holdout import (                     # noqa: E402
    assert_vocabulary, load_pin, sha256_of)

import phase6_generate_predictions as GEN              # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FREEZE = PROJECT_ROOT / "PHASE6_HOLDOUT_FREEZE.txt"
PROTOCOL = PROJECT_ROOT / "PHASE6_LIVE_LOG_PROTOCOL.txt"

FREEZE_SHA = GEN.FREEZE_SHA

LIVE_DIR = GEN.LIVE_DIR
PREDICTIONS_CSV = GEN.PREDICTIONS_CSV
RESULTS_CSV = LIVE_DIR / "phase6_live_results.csv"
MARKET_CSV = LIVE_DIR / "phase6_live_market.csv"
CONTAMINATION_CSV = GEN.CONTAMINATION_CSV
CAPTURE_AUDIT = LIVE_DIR / "phase6_capture_audit.csv"

# B3.3. The PRIMARY, and the only book this file reads. Closing prices.
PRIMARY_BOOK = "B365C"
PRIMARY_COLUMNS = ("B365CH", "B365CD", "B365CA")

STALE_DAYS_THRESHOLD = GEN.STALE_DAYS_THRESHOLD
FLOAT_FORMAT = GEN.FLOAT_FORMAT
FLOAT_PRECISION = GEN.FLOAT_PRECISION
HOLDOUT_SEASON = GEN.HOLDOUT_SEASON


class CaptureError(RuntimeError):
    """A violation of the protocol or of the declared source shape."""


# ============================================================
# 1. THE SOURCE
# ============================================================

def load_source():
    """
    The E0 file, with its own sha256 recorded on every row it produces.

    The hash is carried into the stored rows deliberately: a capture that
    cannot say WHICH bytes it read is not reproducible, and this source is
    refreshed in place by its publisher rather than versioned.
    """

    payload = GEN.fetch(GEN.SOURCE_URL)
    digest = hashlib.sha256(payload).hexdigest()

    frame = pd.read_csv(io.BytesIO(payload))

    missing = [c for c in ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG")
               if c not in frame.columns]

    if missing:
        raise CaptureError(
            "the E0 file is missing {}, so P7.2's declared spine mapping "
            "cannot be applied. This is a SOURCE CHANGE and is reported, not "
            "worked around.".format(missing))

    return frame, digest


def played_rows(frame):
    """Completed matches only, mapped exactly as P7.2 declares."""

    played = frame[frame["FTHG"].notna() & frame["FTAG"].notna()].copy()

    built = pd.DataFrame({
        "season": HOLDOUT_SEASON,
        "played_date": pd.to_datetime(played["Date"], format="%d/%m/%Y"),
        "home_team": played["HomeTeam"].map(lambda t: TEAM_MAP.get(t, t)),
        "away_team": played["AwayTeam"].map(lambda t: TEAM_MAP.get(t, t)),
        "home_goals": played["FTHG"].astype(int),
        "away_goals": played["FTAG"].astype(int),
    })

    built["result"] = [
        result_from_goals(h, a)
        for h, a in zip(built["home_goals"], built["away_goals"])
    ]

    built["match_id"] = [
        GEN.match_key(s, h, a) for s, h, a in
        zip(built["season"], built["home_team"], built["away_team"])]

    # The odds columns travel with the same rows, so the market table and the
    # results table cannot drift onto different matches.
    for column in PRIMARY_COLUMNS:
        built[column] = (pd.to_numeric(played[column], errors="coerce")
                         if column in played.columns else np.nan)

    return built.sort_values("played_date").reset_index(drop=True)


# ============================================================
# 2. THE CAPTURE
# ============================================================

def capture(audit, dry=False):

    pin = load_pin()

    frame, digest = load_source()
    played = played_rows(frame)

    print("  source sha256      {}".format(digest))
    print("  completed matches  {}".format(len(played)))

    assert_vocabulary(played, pin["vocabulary"], "the captured 2026-27 rows")

    audit.record(
        "C1", "every captured name is inside the pin's declared twenty",
        "0 outside", "0 outside", True,
        "through the scoring instrument's OWN assert_vocabulary. No "
        "normalisation was added to any loader for this capture either")

    audit.record(
        "C2", "Pinnacle closing is absent and NO other book is promoted",
        "book = {} only".format(PRIMARY_BOOK), PRIMARY_BOOK, True,
        "SW5-PSC. B3.3 fixed Bet365 closing as PRIMARY on completeness "
        "before any score existed. Filling the sensitivity arm now would be "
        "choosing an instrument after seeing which one survived")

    # ---- the predictions this capture can be joined to --------------------
    predictions = GEN.read_log()

    if predictions.empty:
        raise CaptureError(
            "there are no written predictions. Nothing can be captured "
            "against an empty log.")

    # L4: never capture against a log that does not verify.
    failures, checked = GEN.verify_log(predictions)

    audit.record(
        "C3", "the prediction log's hash chain verifies BEFORE anything is "
              "joined to it",
        0, len(failures), len(failures) == 0,
        "{} rows checked. Capturing results against a log that does not "
        "verify would attach real outcomes to rows of unknown provenance"
        .format(checked))

    if failures:
        raise CaptureError(
            "L4: the prediction log does not verify ({} mismatches). "
            "Nothing is captured and nothing is repaired. Investigate "
            "before running again.".format(len(failures)))

    cutoffs = dict(zip(predictions["match_id"],
                       pd.to_datetime(predictions["state_cutoff_date"])))

    # L10.2 amendment C. A row written after its own kickoff is in the log now,
    # and its result is captured - but as contaminated, so every pre-kickoff
    # figure keeps excluding it exactly as it excluded a missing row before.
    pre_kickoff = GEN.pre_kickoff_flags(predictions)

    # ---- the holdout rows, and which of them the log covers ---------------
    in_holdout = played[played["played_date"] >= pin["cutoff"]].copy()

    print("  in the holdout     {} (date >= {})".format(
        len(in_holdout), str(pin["cutoff"].date())))

    have_prediction = in_holdout["match_id"].isin(cutoffs)

    # ---- L2: the contamination record -------------------------------------
    contaminated = in_holdout[~have_prediction].copy()

    audit.measure(
        "C4", "holdout matches with NO prediction at all (L2)",
        "{} of {}".format(len(contaminated), len(in_holdout)),
        "EXCLUDED FROM THE LOG's figures, NEVER from the holdout. L2.2: the "
        "holdout's membership is a date rule in a frozen document and no rule "
        "in this protocol may alter it")

    if len(contaminated) and not dry:
        record = contaminated[["match_id", "played_date", "home_team",
                               "away_team"]].copy()
        record["season"] = HOLDOUT_SEASON
        record["reason"] = (
            "no pre-kickoff prediction existed; the live log began after this "
            "match was played (L2.1)")
        record["recorded_at_utc"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        record["excluded_from_log"] = True
        record["excluded_from_holdout"] = False

        LIVE_DIR.mkdir(exist_ok=True)

        existing = (pd.read_csv(CONTAMINATION_CSV)
                    if CONTAMINATION_CSV.exists() else pd.DataFrame())

        keep = record[~record["match_id"].isin(
            set(existing["match_id"]) if len(existing) else set())]

        if len(keep):
            pd.concat([existing, keep], ignore_index=True).to_csv(
                CONTAMINATION_CSV, index=False, encoding="utf-8")
            print("  recorded {} contaminated match(es) in {}".format(
                len(keep), CONTAMINATION_CSV.name))

    # ---- the rows that DO have a prediction -------------------------------
    joinable = in_holdout[have_prediction].copy()

    stored = (pd.read_csv(RESULTS_CSV) if RESULTS_CSV.exists()
              else pd.DataFrame())
    already = set(stored["match_id"]) if len(stored) else set()

    fresh = joinable[~joinable["match_id"].isin(already)].copy()

    print("  already stored     {}".format(len(already)))
    print("  new to capture     {}".format(len(fresh)))

    if fresh.empty:
        audit.measure("C5", "new results captured", 0,
                      "nothing has been played since the last capture that "
                      "carries a prediction")
        return

    # ---- L3.4 / L3.5: staleness ------------------------------------------
    fresh["state_cutoff_date"] = fresh["match_id"].map(cutoffs)
    fresh["state_age_days"] = (
        fresh["played_date"] - fresh["state_cutoff_date"]).dt.days
    fresh["is_stale"] = fresh["state_age_days"] > STALE_DAYS_THRESHOLD

    audit.record(
        "C6", "no captured match precedes the cutoff its prediction was "
              "written against",
        "min state_age_days >= 0", int(fresh["state_age_days"].min()),
        int(fresh["state_age_days"].min()) >= 0,
        "a negative age would mean a result arrived before the state the "
        "prediction was fitted on, which is a leak and not a rounding issue")

    audit.measure(
        "C7", "stale rows, L3.5's pre-declared {}-day stratum".format(
            STALE_DAYS_THRESHOLD),
        "{} of {} (max {} days)".format(
            int(fresh["is_stale"].sum()), len(fresh),
            int(fresh["state_age_days"].max())),
        "L3.1 keeps the ORIGINAL prediction for a rescheduled match, so a "
        "long postponement carries a stale one. Reported as a stratum, never "
        "regenerated and never dropped")

    captured_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results = pd.DataFrame({
        "match_id": fresh["match_id"],
        "season": HOLDOUT_SEASON,
        "played_date": fresh["played_date"].dt.strftime("%Y-%m-%d"),
        "home_goals": fresh["home_goals"],
        "away_goals": fresh["away_goals"],
        "result": fresh["result"],
        "state_age_days": fresh["state_age_days"],
        "is_stale": fresh["is_stale"],
        "contaminated": [not pre_kickoff[m] for m in fresh["match_id"]],
        "source_sha256": digest,
        "captured_at_utc": captured_at,
    })

    # ---- the market, B4.1 proportional ------------------------------------
    prices = fresh[list(PRIMARY_COLUMNS)].to_numpy(dtype=float)
    complete = ~np.isnan(prices).any(axis=1)

    audit.measure(
        "C8", "{} price completeness on the new rows".format(PRIMARY_BOOK),
        "{} of {}".format(int(complete.sum()), len(fresh)),
        "B3.3 chose this book on completeness - 380 of 380 in all five "
        "development seasons. A gap is reported, never filled from another "
        "book")

    market = pd.DataFrame()

    if complete.any():
        # devig_proportional is IMPORTED. B4.1 is defined once, in the
        # instrument that owns it, and is not restated here.
        priced = fresh[complete].copy()
        proba = devig_proportional(prices[complete])

        market = pd.DataFrame({
            "match_id": priced["match_id"].to_numpy(),
            "season": HOLDOUT_SEASON,
            "book": PRIMARY_BOOK,
            "odds_home": prices[complete][:, 0],
            "odds_draw": prices[complete][:, 1],
            "odds_away": prices[complete][:, 2],
            "p_home": proba[:, 0],
            "p_draw": proba[:, 1],
            "p_away": proba[:, 2],
            "overround": (1.0 / prices[complete]).sum(axis=1),
            "source_sha256": digest,
            "captured_at_utc": captured_at,
        })

        worst = float(np.abs(proba.sum(axis=1) - 1.0).max())

        audit.record(
            "C9", "the de-vigged market probabilities sum to one",
            "max |sum-1| < 1e-12", "{:.2e}".format(worst), worst < 1e-12,
            "B4.1 proportional, computed by phase5_market_benchmark's own "
            "devig_proportional rather than a second copy of the formula")

    print()
    print("  {:<34} {:>5} {:>5}  {:>4}  {:>6} {:>6} {:>6}".format(
        "match", "hg", "ag", "age", "mkt H", "mkt D", "mkt A"))
    print("  " + "-" * 76)

    market_lookup = (market.set_index("match_id") if len(market)
                     else pd.DataFrame())

    for _i, row in fresh.iterrows():
        odds = ("{:>6.3f} {:>6.3f} {:>6.3f}".format(
            market_lookup.loc[row["match_id"], "p_home"],
            market_lookup.loc[row["match_id"], "p_draw"],
            market_lookup.loc[row["match_id"], "p_away"])
            if len(market_lookup) and row["match_id"] in market_lookup.index
            else "{:>20}".format("no price"))

        print("  {:<34} {:>5} {:>5}  {:>4}  {}".format(
            "{} v {}".format(row["home_team"][:15], row["away_team"][:15]),
            row["home_goals"], row["away_goals"], row["state_age_days"],
            odds))

    if dry:
        print()
        print("  --dry: nothing written")
        return

    LIVE_DIR.mkdir(exist_ok=True)

    append(RESULTS_CSV, results)

    if len(market):
        append(MARKET_CSV, market)

    ok_r, detail_r = GEN.supabase_insert("results", results.to_dict("records"))
    ok_m, detail_m = ((GEN.supabase_insert("market", market.to_dict("records")))
                      if len(market) else (True, "no rows"))

    audit.record(
        "C10", "results and market appended to the local mirrors",
        "{} results".format(len(results)), "{} results, {} market".format(
            len(results), len(market)), True,
        "these two mirrors are NOT committed: they carry 2026-27 scorelines "
        "and .gitignore keeps those out of the repository until the holdout "
        "is deliberately acquired. L4.8")

    audit.measure(
        "C11", "Supabase insert",
        "results: {} | market: {}".format(
            "OK" if ok_r else detail_r, "OK" if ok_m else detail_m),
        "mirror-only is a degraded run, not a failed one")

    print()
    print("  appended {} results and {} market rows".format(
        len(results), len(market)))
    print("  supabase results: {}".format("written" if ok_r else detail_r))
    print("  supabase market : {}".format("written" if ok_m else detail_m))


def append(path, frame):
    existing = (pd.read_csv(path, float_precision=FLOAT_PRECISION)
                if path.exists() else pd.DataFrame())

    combined = (pd.concat([existing, frame], ignore_index=True)
                if len(existing) else frame)

    combined.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)


# ============================================================
# THE RUN
# ============================================================

def main():

    configure_stdout()

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dry", action="store_true",
                        help="compute and print, write nothing")
    args = parser.parse_args()

    banner("PHASE 6 - RESULTS AND MARKET CAPTURE")

    audit = Audit()

    freeze_sha = sha256_of(FREEZE)

    audit.record(
        "C0a", "the freeze on disk is the one this instrument was built "
               "against",
        FREEZE_SHA, freeze_sha, freeze_sha == FREEZE_SHA, "")

    audit.record(
        "C0b", "the live log protocol is on disk",
        "exists", PROTOCOL.exists(), PROTOCOL.exists(),
        "this file implements section 4 of it and cannot run without it")

    print("  freeze    {}".format(freeze_sha))
    print("  protocol  {}".format(
        sha256_of(PROTOCOL) if PROTOCOL.exists() else "ABSENT"))
    print()

    if not PROTOCOL.exists():
        raise CaptureError("PHASE6_LIVE_LOG_PROTOCOL.txt is not on disk.")

    capture(audit, dry=args.dry)

    banner("STATUS")

    frame_out = audit.frame()

    failures = int((frame_out["status"] == "FAIL").sum())
    info = int((frame_out["status"] == "INFO").sum())
    checked = len(frame_out)

    LIVE_DIR.mkdir(exist_ok=True)
    frame_out.insert(0, "run_at_utc",
                     datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    if CAPTURE_AUDIT.exists():
        frame_out = pd.concat(
            [pd.read_csv(CAPTURE_AUDIT, keep_default_na=False), frame_out],
            ignore_index=True)

    frame_out.to_csv(CAPTURE_AUDIT, index=False, encoding="utf-8",
                     float_format=FLOAT_FORMAT)

    print("  {}".format(CAPTURE_AUDIT))
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
