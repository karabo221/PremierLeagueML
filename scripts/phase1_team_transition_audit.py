"""
===============================================================================
PHASE 1 - INSTRUMENT 4
TEAM TRANSITION AUDIT
===============================================================================

THE QUESTION
    Instruments 1-3 asked whether the matches, the historical state and the
    strength features are trustworthy. This one asks a different question:

        are the team identities, season transitions and historical priors
        COHERENT across the whole five-season period?

    A feature layer can be perfectly leakage-free and still be quietly wrong
    about who a team IS - inheriting a stale prior across a relegation gap,
    or treating a name collision as continuity.

INPUTS - exactly two
    outputs/phase1_matches.csv                    (Instrument 1 - authority
                                                   on team identity)
    outputs/phase1_team_strength_features.csv     (Instrument 3 - the prior
                                                   state being reconciled)

    No FBref aggregate. data/raw/ is never opened, and that is measured by a
    runtime audit hook rather than asserted.

NEW IS NOT PROMOTED - THE CENTRAL DISTINCTION
    A team appearing in season N but not N-1 is, in this data, exactly that:
    new to the current season. Nothing more is knowable from fixtures.

    Three quite different real-world situations produce that same signal:

        continuing                - present in season N-1
        returning_after_absence   - absent in N-1, but present in an EARLIER
                                    season held here (Burnley, Leeds United,
                                    Leicester City, Southampton)
        new_to_dataset            - never seen in any season held here

    Even new_to_dataset does NOT mean "never in the Premier League".
    Sunderland spent years in the Premier League before this dataset begins
    in 2021-22. The dataset boundary is a limit of observation, not a fact
    about the club.

    So this instrument records `new_to_current_season` as the finding, and
    keeps the stronger word "promoted" confined to a clearly-labelled column
    of external interpretation that no test depends on.

RETURNING TEAMS MUST NOT INHERIT A STALE PRIOR
    Burnley plays 2021-22, is absent 2022-23, returns 2023-24. The prior for
    2023-24 must be ABSENT, not 2021-22's. T5 tests this directly, by
    checking that the prior is not merely null but specifically does not
    carry the older season's values.

EXIT CODES
    0  PASS       every test passed
    2  FAIL       an audit test failed - investigate, do not patch
    1  FATAL      the audit could not be run at all (missing or malformed
                  input, unexpected exception)

WHAT IS NOT DONE HERE
    no models, no Elo, no XGBoost, no FBref aggregates, no imputation of
    missing priors, no modelling dataset, no writes to data/raw/, no silent
    renaming, no fuzzy matching, and no reaching back to older seasons to
    fill a gap. An identity mismatch FAILS; it is never repaired.
===============================================================================
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_match_foundation import SANCTION_REGISTRY


# ============================================================
# FILE-ACCESS RECORDER
# ============================================================

_OPENED_PATHS = []


def _record_open(event, args):

    if event != "open":
        return

    target = args[0]

    if isinstance(target, (str, bytes, Path)):
        _OPENED_PATHS.append(str(target))


sys.addaudithook(_record_open)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MATCHES_INPUT = OUTPUTS_DIR / "phase1_matches.csv"
FEATURES_INPUT = OUTPUTS_DIR / "phase1_team_strength_features.csv"

AUDIT_OUTPUT = OUTPUTS_DIR / "phase1_team_transition_audit.csv"
SUMMARY_OUTPUT = OUTPUTS_DIR / "phase1_team_transition_summary.csv"

DECLARED_INPUTS = {MATCHES_INPUT.resolve(), FEATURES_INPUT.resolve()}

RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

EXPECTED_TOTAL_MATCHES = 1900
EXPECTED_TEAMS_PER_SEASON = 20
EXPECTED_MATCHES_PER_TEAM = 38
EXPECTED_TURNOVER = 3

EXIT_PASS = 0
EXIT_FATAL = 1
EXIT_FAIL = 2


# ---- transition vocabulary of this instrument
TRANSITION_BASELINE = "first_season_in_dataset"
TRANSITION_CONTINUING = "continuing"
TRANSITION_RETURNING = "returning_after_absence"
TRANSITION_NEW = "new_to_dataset"

NEW_TO_CURRENT_SEASON = {TRANSITION_RETURNING, TRANSITION_NEW}

# ---- prior-status vocabulary inherited from Instruments 2 and 3
UPSTREAM_STATUS_AVAILABLE = "available"
UPSTREAM_STATUS_NO_PRIOR = "no_prior_season_in_dataset"
UPSTREAM_STATUS_ABSENT = "absent_from_previous_season"

# This instrument's own, more conservative reading of the same three states.
CONSERVATIVE_STATUS = {
    UPSTREAM_STATUS_AVAILABLE: "prior_available",
    UPSTREAM_STATUS_NO_PRIOR: "no_prior_season_in_dataset",
    UPSTREAM_STATUS_ABSENT: "absent_from_previous_season",
}

EXTERNAL_INTERPRETATION_NOTE = (
    "External interpretation, NOT verifiable from this data: a side new to "
    "the current season was most likely promoted from the Championship. "
    "Absence from this dataset is not absence from the Premier League."
)


SUMMARY_COLUMNS = [
    "season",
    "season_index",
    "team",
    "transition_type",
    "new_to_current_season",
    "present_in_previous_season",
    "previous_season",
    "has_previous_season",
    "previous_season_status",
    "previous_season_status_conservative",
    "previous_season_points_available",
    "previous_season_points_used",
    "previous_season_points_raw",
    "previous_season_sanction",
    "previous_season_gf",
    "previous_season_ga",
    "previous_season_gd",
    "previous_season_mp",
    "current_season_pts_from_results",
    "current_season_sanction",
    "current_season_pts_after_sanction",
    "seasons_present_to_date",
    "first_season_seen",
    "last_season_seen",
    "earlier_seasons_seen",
    "seasons_absent_before_return",
    "reconciles_with_instrument3",
    "external_interpretation",
]


# ============================================================
# ERRORS
# ============================================================

class FatalError(Exception):
    """The audit cannot be run at all. Exit 1, not exit 2."""


# ============================================================
# OUTPUT ENCODING
# ============================================================

def configure_stdout():

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ============================================================
# AUDIT LEDGER
# ============================================================

class Audit:
    """
    Three outcome kinds, deliberately kept apart.

        PASS / FAIL   a test with a verdict; FAIL drives exit code 2
        MEASURED      a number reported without a pass/fail claim
        REVIEW        a documented finding for a human, which does NOT fail
                      the run

    REVIEW exists so that a naming or interpretation concern can be raised
    loudly without diluting the FAIL signal - the same reasoning Phase 0
    applied to the referee decision.
    """

    def __init__(self):
        self.rows = []

    def record(self, test_id, test, scope, expected, observed, passed, detail=""):

        # A mis-ordered call would otherwise land a string in `passed` and
        # report a confusing FAIL that looks like a data defect. Refuse it.
        if not isinstance(passed, (bool, np.bool_)):
            raise FatalError(
                f"{test_id}: `passed` must be a bool, got "
                f"{type(passed).__name__} ({passed!r}) - check argument order"
            )

        self.rows.append({
            "test_id": test_id,
            "test": test,
            "scope": scope,
            "expected": expected,
            "observed": observed,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

        return passed

    def measure(self, test_id, test, scope, observed, detail=""):

        self.rows.append({
            "test_id": test_id,
            "test": test,
            "scope": scope,
            "expected": "(measurement)",
            "observed": observed,
            "status": "MEASURED",
            "detail": detail,
        })

    def review(self, test_id, test, scope, observed, detail=""):

        self.rows.append({
            "test_id": test_id,
            "test": test,
            "scope": scope,
            "expected": "(documented finding)",
            "observed": observed,
            "status": "REVIEW",
            "detail": detail,
        })

    def failures(self):
        return [row for row in self.rows if row["status"] == "FAIL"]

    def reviews(self):
        return [row for row in self.rows if row["status"] == "REVIEW"]

    def all_passed(self):
        return not self.failures()

    def frame(self):
        return pd.DataFrame(self.rows, columns=[
            "test_id", "test", "scope",
            "expected", "observed", "status", "detail",
        ])


# ============================================================
# INPUT
# ============================================================

def load_inputs():

    for required in (MATCHES_INPUT, FEATURES_INPUT):
        if not required.exists():
            raise FatalError(f"missing required input: {required}")

    try:
        matches = pd.read_csv(MATCHES_INPUT)
        features = pd.read_csv(FEATURES_INPUT)
    except Exception as error:
        raise FatalError(f"input could not be parsed: {error}") from error

    for frame, name in ((matches, "matches"), (features, "features")):
        if len(frame) != EXPECTED_TOTAL_MATCHES:
            raise FatalError(
                f"{name} has {len(frame)} rows, expected {EXPECTED_TOTAL_MATCHES}"
            )

    required_match_columns = {
        "season", "date", "home_team", "away_team",
        "home_points_from_result", "away_points_from_result",
        "home_goals", "away_goals",
    }

    missing = required_match_columns - set(matches.columns)

    if missing:
        raise FatalError(f"match foundation missing columns: {sorted(missing)}")

    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")

    return matches, features


def team_sides(matches):

    home = matches[[
        "season", "date", "home_team", "home_goals", "away_goals",
        "home_points_from_result",
    ]].rename(columns={
        "home_team": "team", "home_goals": "gf", "away_goals": "ga",
        "home_points_from_result": "pts",
    })

    away = matches[[
        "season", "date", "away_team", "away_goals", "home_goals",
        "away_points_from_result",
    ]].rename(columns={
        "away_team": "team", "away_goals": "gf", "home_goals": "ga",
        "away_points_from_result": "pts",
    })

    return pd.concat([home, away], ignore_index=True)


# ============================================================
# INDEPENDENT RECONSTRUCTION
# ============================================================

def reconstruct_season_totals(matches):
    """Season-end totals from match results, plus the declared sanctions."""

    sides = team_sides(matches)

    totals = sides.groupby(["season", "team"], as_index=False).agg(
        mp=("pts", "size"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
        pts_raw=("pts", "sum"),
    )

    totals["gd"] = totals["gf"] - totals["ga"]

    totals["sanction"] = [
        SANCTION_REGISTRY.get((season, team), 0)
        for season, team in zip(totals["season"], totals["team"])
    ]

    totals["pts_after_sanction"] = totals["pts_raw"] + totals["sanction"]

    return totals


def reconstruct_transitions(matches):
    """
    Rebuild the season-to-season movement of teams, from fixtures alone.

    Identity is the exact fixture string. No normalisation, no aliasing, no
    fuzzy matching - two identities are the same team if and only if the
    strings are equal.
    """

    seasons = sorted(matches["season"].unique())

    teams_by_season = {
        season: set(group["home_team"]) | set(group["away_team"])
        for season, group in matches.groupby("season")
    }

    transitions = []

    for index, season in enumerate(seasons):

        if index == 0:
            continue

        previous_season = seasons[index - 1]

        current_teams = teams_by_season[season]
        previous_teams = teams_by_season[previous_season]

        returning = sorted(current_teams & previous_teams)
        new = sorted(current_teams - previous_teams)
        departing = sorted(previous_teams - current_teams)

        # A new team either played here before (returning after a gap) or has
        # never been seen in this dataset. Those are different claims.
        seen_before = set()

        for earlier in seasons[:index - 1]:
            seen_before |= teams_by_season[earlier]

        returning_after_absence = sorted(set(new) & seen_before)
        new_to_dataset = sorted(set(new) - seen_before)

        transitions.append({
            "previous_season": previous_season,
            "season": season,
            "returning": returning,
            "new_to_current_season": new,
            "departing": departing,
            "returning_after_absence": returning_after_absence,
            "new_to_dataset": new_to_dataset,
        })

    return seasons, teams_by_season, transitions


def build_summary(matches, totals, seasons, teams_by_season):
    """One row per team-season: identity, transition and prior eligibility."""

    totals_by_key = {
        (row.season, row.team): row for row in totals.itertuples()
    }

    rows = []

    for index, season in enumerate(seasons):

        previous_season = seasons[index - 1] if index > 0 else None

        for team in sorted(teams_by_season[season]):

            seen_before = [
                earlier for earlier in seasons[:index]
                if team in teams_by_season[earlier]
            ]

            if index == 0:
                transition = TRANSITION_BASELINE
            elif team in teams_by_season[previous_season]:
                transition = TRANSITION_CONTINUING
            elif seen_before:
                transition = TRANSITION_RETURNING
            else:
                transition = TRANSITION_NEW

            has_prior = transition == TRANSITION_CONTINUING

            if index == 0:
                upstream_status = UPSTREAM_STATUS_NO_PRIOR
            elif has_prior:
                upstream_status = UPSTREAM_STATUS_AVAILABLE
            else:
                upstream_status = UPSTREAM_STATUS_ABSENT

            current = totals_by_key[(season, team)]

            row = {
                "season": season,
                "season_index": index,
                "team": team,
                "transition_type": transition,
                "new_to_current_season": transition in NEW_TO_CURRENT_SEASON,
                "present_in_previous_season": has_prior,
                "previous_season": previous_season if has_prior else "",
                "has_previous_season": has_prior,
                "previous_season_status": upstream_status,
                "previous_season_status_conservative":
                    CONSERVATIVE_STATUS[upstream_status],
                "previous_season_points_available": has_prior,
                "previous_season_points_used": np.nan,
                "previous_season_points_raw": np.nan,
                "previous_season_sanction": np.nan,
                "previous_season_gf": np.nan,
                "previous_season_ga": np.nan,
                "previous_season_gd": np.nan,
                "previous_season_mp": np.nan,
                "current_season_pts_from_results": current.pts_raw,
                "current_season_sanction": current.sanction,
                "current_season_pts_after_sanction": current.pts_after_sanction,
                "seasons_present_to_date": len(seen_before) + 1,
                "first_season_seen": seen_before[0] if seen_before else season,
                "last_season_seen": seen_before[-1] if seen_before else "",
                "earlier_seasons_seen": "|".join(seen_before),
                "seasons_absent_before_return": (
                    index - seasons.index(seen_before[-1]) - 1
                    if transition == TRANSITION_RETURNING else 0
                ),
                "reconciles_with_instrument3": False,
                "external_interpretation": (
                    EXTERNAL_INTERPRETATION_NOTE
                    if transition in NEW_TO_CURRENT_SEASON else ""
                ),
            }

            if has_prior:

                prior = totals_by_key[(previous_season, team)]

                row["previous_season_points_used"] = float(
                    prior.pts_after_sanction)
                row["previous_season_points_raw"] = float(prior.pts_raw)
                row["previous_season_sanction"] = float(prior.sanction)
                row["previous_season_gf"] = float(prior.gf)
                row["previous_season_ga"] = float(prior.ga)
                row["previous_season_gd"] = float(prior.gd)
                row["previous_season_mp"] = float(prior.mp)

            rows.append(row)

    return pd.DataFrame(rows)


def collapse_instrument3_priors(features, audit):
    """
    Collapse Instrument 3's per-match prior columns to one row per team-season.

    A prior that varies within a season would be a serious defect, so the
    collapse verifies constancy rather than assuming it.
    """

    prior_fields = [
        "available", "pts", "pts_raw", "sanction",
        "gf", "ga", "gd", "mp", "source", "status",
    ]

    frames = []

    for side in ("home", "away"):

        block = features[["season", f"{side}_team"]].copy()
        block = block.rename(columns={f"{side}_team": "team"})

        for field in prior_fields:
            block[field] = features[f"{side}_prev_season_{field}"]

        frames.append(block)

    stacked = pd.concat(frames, ignore_index=True)

    inconsistent = []
    collapsed = []

    for (season, team), group in stacked.groupby(["season", "team"]):

        record = {"season": season, "team": team}

        for field in prior_fields:

            values = group[field]

            distinct = set(values.dropna().unique())

            if values.isna().any() and distinct:
                inconsistent.append(
                    f"{season} {team}.{field}: mixed null and {sorted(distinct)}"
                )

            if len(distinct) > 1:
                inconsistent.append(
                    f"{season} {team}.{field}: {sorted(distinct)}"
                )

            record[field] = distinct.pop() if len(distinct) == 1 else np.nan

        record["match_rows"] = len(group)

        collapsed.append(record)

    audit.record(
        "T4a",
        "Instrument 3's prior is constant across all of a team's matches",
        "100 team-seasons",
        "0 inconsistencies",
        f"{len(inconsistent)} inconsistencies",
        not inconsistent,
        "; ".join(inconsistent[:5]),
    )

    frame = pd.DataFrame(collapsed)

    wrong_count = frame[frame["match_rows"] != EXPECTED_MATCHES_PER_TEAM]

    audit.record(
        "T4b",
        "Every team-season is represented by exactly 38 match rows",
        f"{len(frame)} team-seasons",
        f"38 x {len(frame)}",
        f"{len(frame) - len(wrong_count)}/{len(frame)} at 38",
        wrong_count.empty,
        "; ".join(f"{r.season} {r.team}: {r.match_rows}"
                  for r in wrong_count.head(5).itertuples()),
    )

    return frame


# ============================================================
# TESTS
# ============================================================

def test_t1_twenty_teams(teams_by_season, audit):

    wrong = {
        season: len(teams)
        for season, teams in teams_by_season.items()
        if len(teams) != EXPECTED_TEAMS_PER_SEASON
    }

    audit.record(
        "T1",
        "Exactly 20 Premier League teams in every season",
        "all seasons",
        f"20 x {len(teams_by_season)}",
        f"{len(teams_by_season) - len(wrong)}/{len(teams_by_season)} correct",
        not wrong,
        str(wrong) if wrong else "",
    )


def test_t2_continuity(transitions, audit):

    violations = []

    for transition in transitions:

        returning = len(transition["returning"])
        new = len(transition["new_to_current_season"])
        departing = len(transition["departing"])

        label = f"{transition['previous_season']} -> {transition['season']}"

        if returning + new != EXPECTED_TEAMS_PER_SEASON:
            violations.append(
                f"{label}: {returning} returning + {new} new != 20"
            )

        if new != departing:
            violations.append(
                f"{label}: {new} in but {departing} out"
            )

        audit.measure(
            "T2b",
            "Season transition composition",
            label,
            f"{returning} returning, {new} new, {departing} departing",
            f"IN [{', '.join(transition['new_to_current_season'])}] "
            f"OUT [{', '.join(transition['departing'])}]",
        )

    audit.record(
        "T2a",
        "returning + new_to_current_season = 20, and in-count equals out-count",
        f"{len(transitions)} adjacent season pairs",
        "0 violations", f"{len(violations)} violations",
        not violations,
        "; ".join(violations),
    )


def test_t3_turnover(transitions, summary, audit):

    off_pattern = []

    for transition in transitions:

        new = len(transition["new_to_current_season"])

        if new != EXPECTED_TURNOVER:
            off_pattern.append(
                f"{transition['previous_season']} -> {transition['season']}: "
                f"{new} new"
            )

    audit.record(
        "T3a",
        "Three-team turnover holds at every adjacent season pair",
        f"{len(transitions)} pairs",
        f"{EXPECTED_TURNOVER} per pair",
        f"{len(transitions) - len(off_pattern)}/{len(transitions)} at 3",
        not off_pattern,
        "; ".join(off_pattern),
    )

    # The finding is recorded as "new to the current season". Promotion is a
    # strictly stronger claim and is never derived from newness.
    new_rows = summary[summary["new_to_current_season"]]

    breakdown = new_rows["transition_type"].value_counts().to_dict()

    audit.record(
        "T3b",
        "Every new side is typed by evidence, not assumed promoted",
        "all new team-seasons",
        "every new side typed as returning_after_absence or new_to_dataset",
        f"{len(new_rows)} typed as {breakdown}",
        set(breakdown) <= NEW_TO_CURRENT_SEASON,
        "Newness alone never implies promotion",
    )

    unlabelled = new_rows[new_rows["external_interpretation"] == ""]

    audit.record(
        "T3c",
        "The promotion interpretation is documented, not silently assumed",
        "all new team-seasons",
        0, len(unlabelled),
        unlabelled.empty,
        "Stronger 'promoted' reading confined to a labelled column",
    )

    # REGRESSION GUARD.
    #
    # Instruments 2 and 3 once labelled this state `promoted_no_pl_prior`,
    # which asserted more than fixtures can support: 5 of the sides so
    # labelled had returned after an absence rather than being promoted, and
    # the rest were merely unseen in this dataset - not the same as never
    # having been in the Premier League. The label was corrected to
    # `absent_from_previous_season`.
    #
    # This test exists so it cannot come back. The prior VALUES were always
    # right; only the vocabulary overreached.
    absent_labelled = summary[
        summary["previous_season_status"] == UPSTREAM_STATUS_ABSENT
    ]

    observed_statuses = set(summary["previous_season_status"])

    promotion_claims = sorted(
        status for status in observed_statuses
        if "promot" in str(status).casefold()
    )

    audit.record(
        "T3d",
        "No prior-status value claims promotion, which fixtures cannot show",
        "upstream status vocabulary from Instruments 2 and 3",
        "0 statuses asserting promotion",
        f"{len(promotion_claims)} asserting promotion",
        not promotion_claims,
        f"vocabulary in use: {sorted(observed_statuses)}; "
        f"{len(absent_labelled)} team-seasons are absent_from_previous_season",
    )

    # The two are different claims and must stay in different columns:
    # transition_type may say new_to_dataset, prior status may not.
    conflated = summary[
        summary["previous_season_status"] == summary["transition_type"]
    ]

    audit.record(
        "T3e",
        "Transition type and prior status remain distinct vocabularies",
        "100 team-seasons",
        0, len(conflated),
        conflated.empty,
        "new_to_dataset is a transition, not a prior-availability status",
    )


def test_t4_prior_eligibility(summary, instrument3, audit):

    merged = summary.merge(
        instrument3, on=["season", "team"], how="outer", indicator=True
    )

    unmatched = merged[merged["_merge"] != "both"]

    audit.record(
        "T4c",
        "Team-seasons align one-to-one with Instrument 3",
        "100 team-seasons",
        0, len(unmatched),
        unmatched.empty,
        "; ".join(
            f"{r.season} {r.team} ({r._merge})"
            for r in unmatched.head(5).itertuples()
        ),
    )

    if not unmatched.empty:
        return summary

    comparisons = [
        ("has_previous_season", "available", bool),
        ("previous_season_status", "status", str),
        ("previous_season_points_used", "pts", float),
        ("previous_season_points_raw", "pts_raw", float),
        ("previous_season_sanction", "sanction", float),
        ("previous_season_gf", "gf", float),
        ("previous_season_ga", "ga", float),
        ("previous_season_gd", "gd", float),
        ("previous_season_mp", "mp", float),
    ]

    disagreements = []
    per_row_ok = pd.Series(True, index=merged.index)

    for mine, theirs, kind in comparisons:

        left = merged[mine]
        right = merged[theirs]

        if kind is bool:
            differs = left.astype(bool) != right.astype(bool)
        elif kind is str:
            differs = left.astype(str) != right.astype(str)
        else:
            left = pd.to_numeric(left, errors="coerce")
            right = pd.to_numeric(right, errors="coerce")
            differs = ~((left == right) | (left.isna() & right.isna()))

        per_row_ok &= ~differs

        if differs.any():
            disagreements.append(f"{mine} vs {theirs}: {int(differs.sum())} rows")

    audit.record(
        "T4d",
        "Prior eligibility and values reconcile with Instrument 3",
        f"{len(comparisons)} fields x 100 team-seasons",
        "0 disagreements",
        f"{len(disagreements)} disagreeing fields",
        not disagreements,
        "; ".join(disagreements[:5]),
    )

    # Availability must be exactly presence in season N-1 - nothing else.
    wrong_availability = merged[
        merged["previous_season_points_available"].astype(bool)
        != merged["present_in_previous_season"].astype(bool)
    ]

    audit.record(
        "T4e",
        "previous_season_points_available equals presence in season N-1",
        "100 team-seasons",
        0, len(wrong_availability),
        wrong_availability.empty,
    )

    reconciled = dict(zip(
        zip(merged["season"], merged["team"]), per_row_ok
    ))

    summary["reconciles_with_instrument3"] = [
        bool(reconciled.get((season, team), False))
        for season, team in zip(summary["season"], summary["team"])
    ]

    return summary


def test_t5_returning_teams(summary, totals, instrument3, audit):
    """
    T5 - a team returning after an absence must not inherit its older prior.

    Nullness alone is not enough evidence. The test also checks the prior does
    not carry the values of the team's most recent EARLIER season, which is
    the specific thing a naive "last time we saw them" lookup would produce.
    """

    returning = summary[summary["transition_type"] == TRANSITION_RETURNING]

    totals_by_key = {(row.season, row.team): row for row in totals.itertuples()}

    instrument3_by_key = {
        (row.season, row.team): row for row in instrument3.itertuples()
    }

    violations = []
    inherited = []

    for row in returning.itertuples():

        if row.has_previous_season or row.previous_season_points_available:
            violations.append(f"{row.season} {row.team}: marked as having a prior")

        prior = instrument3_by_key.get((row.season, row.team))

        if prior is None:
            violations.append(f"{row.season} {row.team}: absent from Instrument 3")
            continue

        if bool(prior.available):
            violations.append(f"{row.season} {row.team}: Instrument 3 has a prior")

        # The exact failure mode being hunted: an older season's points
        # showing up as the prior. Checked against EVERY earlier season the
        # team appeared in, not just one - the most recent is the value a
        # naive "last time we saw them" lookup would actually return.
        earlier_seasons = [
            s for s in str(row.earlier_seasons_seen).split("|") if s
        ]

        for earlier in earlier_seasons:

            stale = totals_by_key.get((earlier, row.team))

            if stale is None or pd.isna(prior.pts):
                continue

            if float(prior.pts) == float(stale.pts_after_sanction):
                inherited.append(
                    f"{row.season} {row.team}: carries {earlier} "
                    f"points {prior.pts}"
                )

        if pd.notna(prior.pts) or pd.notna(prior.mp) or pd.notna(prior.gf):
            violations.append(f"{row.season} {row.team}: prior values present")

    audit.record(
        "T5a",
        "Returning teams carry no previous-season prior",
        f"{len(returning)} returning team-seasons",
        0, len(violations),
        not violations,
        "; ".join(violations[:5]),
    )

    audit.record(
        "T5b",
        "No returning team inherits an older season's values across the gap",
        f"{len(returning)} returning team-seasons",
        0, len(inherited),
        not inherited,
        "; ".join(inherited[:5]),
    )

    for row in returning.itertuples():
        audit.measure(
            "T5c",
            "Returning team identified",
            f"{row.season} {row.team}",
            f"last seen {row.last_season_seen}, "
            f"absent {row.seasons_absent_before_return} season(s) "
            f"(all earlier: {row.earlier_seasons_seen})",
        )

    return returning


def test_t6_sanctions(summary, totals, audit):

    problems = []
    verified = []

    totals_by_key = {(row.season, row.team): row for row in totals.itertuples()}

    for (sanctioned_season, team), value in sorted(SANCTION_REGISTRY.items()):

        source = totals_by_key.get((sanctioned_season, team))

        if source is None:
            problems.append(f"{sanctioned_season} {team}: not in the foundation")
            continue

        consumers = summary[
            (summary["team"] == team)
            & (summary["previous_season"] == sanctioned_season)
        ]

        if consumers.empty:
            problems.append(
                f"{sanctioned_season} {team}: no season consumes this as a prior"
            )
            continue

        for row in consumers.itertuples():

            expected_used = float(source.pts_raw + value)

            checks = [
                (row.previous_season_points_raw, float(source.pts_raw), "raw"),
                (row.previous_season_sanction, float(value), "sanction"),
                (row.previous_season_points_used, expected_used, "used"),
            ]

            for observed, expected, label in checks:
                if float(observed) != expected:
                    problems.append(
                        f"{row.season} {team} {label}: {observed} != {expected}"
                    )

            # The internal identity that must always hold.
            if (
                float(row.previous_season_points_used)
                != float(row.previous_season_points_raw)
                + float(row.previous_season_sanction)
            ):
                problems.append(
                    f"{row.season} {team}: used != raw + sanction"
                )
            else:
                verified.append(
                    f"{row.season} prior from {sanctioned_season} {team}: "
                    f"{row.previous_season_points_raw:.0f} "
                    f"{row.previous_season_sanction:+.0f} = "
                    f"{row.previous_season_points_used:.0f}"
                )

    audit.record(
        "T6a",
        "Sanctioned points become the correct next-season prior",
        f"{len(SANCTION_REGISTRY)} sanctions",
        "0 problems", f"{len(problems)} problems",
        not problems,
        "; ".join(problems) if problems else "; ".join(verified),
    )

    # Everywhere else: used == raw, and the sanction column is zero.
    with_prior = summary[summary["has_previous_season"]]

    unsanctioned = with_prior[with_prior["previous_season_sanction"] == 0]

    drifted = unsanctioned[
        unsanctioned["previous_season_points_used"]
        != unsanctioned["previous_season_points_raw"]
    ]

    audit.record(
        "T6b",
        "Unsanctioned priors keep used and raw points identical",
        f"{len(unsanctioned)} team-seasons",
        0, len(drifted),
        drifted.empty,
        "; ".join(f"{r.season} {r.team}" for r in drifted.head(5).itertuples()),
    )

    # And the identity holds for every team-season carrying a prior.
    broken = with_prior[
        with_prior["previous_season_points_used"]
        != with_prior["previous_season_points_raw"]
        + with_prior["previous_season_sanction"]
    ]

    audit.record(
        "T6c",
        "used = raw + sanction holds for every prior",
        f"{len(with_prior)} team-seasons",
        0, len(broken),
        broken.empty,
    )


def test_t7_cross_season_contamination(summary, totals, instrument3, audit):
    """
    T7 - the prior must never be the CURRENT season's own totals.

    A numeric coincidence is possible and is not a defect, so the test is on
    provenance: the recorded source season must be N-1, and the prior values
    must equal that season's totals. Coincidences are then listed separately
    so a human can see they were checked rather than waved through.
    """

    seasons = sorted(summary["season"].unique())

    previous_of = {
        current: previous for previous, current in zip(seasons, seasons[1:])
    }

    instrument3_by_key = {
        (row.season, row.team): row for row in instrument3.itertuples()
    }

    totals_by_key = {(row.season, row.team): row for row in totals.itertuples()}

    untracked = []
    wrong_source = []
    wrong_values = []
    coincidences = []

    for row in summary.itertuples():

        prior = instrument3_by_key.get((row.season, row.team))

        if prior is None:
            untracked.append(f"{row.season} {row.team}: missing from Instrument 3")
            continue

        source = "" if pd.isna(prior.source) else str(prior.source)

        if not row.has_previous_season:

            if source:
                wrong_source.append(
                    f"{row.season} {row.team}: source {source!r} without a prior"
                )

            continue

        # The source season must be explicitly recorded ...
        if not source:
            untracked.append(f"{row.season} {row.team}: no source season recorded")
            continue

        # ... and it must be N-1, never the current season.
        if source == row.season:
            wrong_source.append(
                f"{row.season} {row.team}: source is the CURRENT season"
            )
            continue

        if source != previous_of.get(row.season):
            wrong_source.append(
                f"{row.season} {row.team}: source {source!r} is not "
                f"{previous_of.get(row.season)!r}"
            )
            continue

        # And the values must be season N-1's, verified against the foundation.
        expected = totals_by_key[(source, row.team)]

        if (
            float(prior.pts) != float(expected.pts_after_sanction)
            or float(prior.gf) != float(expected.gf)
            or float(prior.ga) != float(expected.ga)
            or float(prior.mp) != float(expected.mp)
        ):
            wrong_values.append(f"{row.season} {row.team}")

        # Numeric coincidence with the current season: allowed, but recorded.
        if float(prior.pts) == float(row.current_season_pts_after_sanction):
            coincidences.append(
                f"{row.season} {row.team}: prior {prior.pts:.0f} equals this "
                f"season's own {row.current_season_pts_after_sanction:.0f} "
                f"(source {source}, verified)"
            )

    audit.record(
        "T7a",
        "Every prior records its source season explicitly",
        "100 team-seasons",
        0, len(untracked),
        not untracked,
        "; ".join(untracked[:5]),
    )

    audit.record(
        "T7b",
        "The source season is N-1 and never the current season",
        "100 team-seasons",
        0, len(wrong_source),
        not wrong_source,
        "; ".join(wrong_source[:5]),
    )

    audit.record(
        "T7c",
        "Prior values equal season N-1's totals in the match foundation",
        "all team-seasons carrying a prior",
        0, len(wrong_values),
        not wrong_values,
        "; ".join(wrong_values[:5]),
    )

    audit.measure(
        "T7d",
        "Priors numerically equal to the team's own current-season points",
        f"{len(coincidences)} coincidences",
        "; ".join(coincidences) if coincidences
        else "none - no prior happens to match its own season",
    )

    return coincidences


def test_t8_identity_integrity(matches, features, teams_by_season, audit):
    """
    T8 - exact fixture identities are authoritative. No fuzzy matching.

    Identity is string equality. This test looks for reasons that assumption
    could be unsafe and FAILS rather than repairing anything.
    """

    foundation_identities = set(matches["home_team"]) | set(matches["away_team"])

    feature_identities = set(features["home_team"]) | set(features["away_team"])

    only_foundation = sorted(foundation_identities - feature_identities)
    only_features = sorted(feature_identities - foundation_identities)

    audit.record(
        "T8a",
        "Identities in Instrument 3 match the foundation exactly, character for character",
        f"{len(foundation_identities)} distinct identities",
        "0 divergences",
        f"{len(only_foundation) + len(only_features)} divergences",
        not only_foundation and not only_features,
        f"only in foundation: {only_foundation}; only in features: {only_features}",
    )

    # Two distinct identities collapsing under normalisation would mean the
    # exact-string assumption is unsafe. That is a FAIL, not a merge.
    normalised = {}

    for identity in sorted(foundation_identities):
        key = " ".join(str(identity).split()).casefold()
        normalised.setdefault(key, []).append(identity)

    collisions = {
        key: names for key, names in normalised.items() if len(names) > 1
    }

    audit.record(
        "T8b",
        "No two distinct identities collide under whitespace/case normalisation",
        f"{len(foundation_identities)} identities",
        0, len(collisions),
        not collisions,
        str(collisions) if collisions else "exact-string identity is safe here",
    )

    # Identities must be clean strings - stray whitespace is how an alias is
    # accidentally born.
    untidy = [
        identity for identity in sorted(foundation_identities)
        if identity != identity.strip() or "  " in identity
    ]

    audit.record(
        "T8c",
        "No identity carries leading, trailing or doubled whitespace",
        f"{len(foundation_identities)} identities",
        0, len(untidy),
        not untidy,
        str(untidy),
    )

    # Every identity must be a real participant in the seasons claimed for it.
    total_team_seasons = sum(len(teams) for teams in teams_by_season.values())

    audit.record(
        "T8d",
        "Team-season count equals seasons x 20",
        f"{len(teams_by_season)} seasons",
        len(teams_by_season) * EXPECTED_TEAMS_PER_SEASON,
        total_team_seasons,
        total_team_seasons == len(teams_by_season) * EXPECTED_TEAMS_PER_SEASON,
    )

    # Informational only: identities where one contains another. Reported so a
    # human can look; NEVER used to merge, alias or rename anything.
    contained = [
        f"{a!r} within {b!r}"
        for a in sorted(foundation_identities)
        for b in sorted(foundation_identities)
        if a != b and a.casefold() in b.casefold()
    ]

    audit.measure(
        "T8e",
        "Identities contained within another identity (informational)",
        len(contained),
        "; ".join(contained) if contained
        else "none - no identity is a substring of another",
    )

    audit.measure(
        "T8f",
        "Distinct team identities across all five seasons",
        len(foundation_identities),
        ", ".join(sorted(foundation_identities)),
    )


def test_t9_independent_reconstruction(matches, summary, instrument3, audit):
    """
    T9 - rebuild transitions a second, independent way and compare.

    The main path derives transitions from per-season team SETS. This path
    walks each team's match dates in order and derives the same facts from
    which seasons it actually appears in - a different route to the same
    claim.
    """

    sides = team_sides(matches)

    seasons = sorted(matches["season"].unique())
    season_index = {season: index for index, season in enumerate(seasons)}

    appearances = {}

    for row in sides.itertuples():
        appearances.setdefault(row.team, set()).add(row.season)

    walked = []

    for team, seen in appearances.items():

        ordered = sorted(seen, key=lambda season: season_index[season])

        for season in ordered:

            index = season_index[season]

            earlier = [s for s in ordered if season_index[s] < index]

            if index == 0:
                transition = TRANSITION_BASELINE
            elif earlier and season_index[earlier[-1]] == index - 1:
                transition = TRANSITION_CONTINUING
            elif earlier:
                transition = TRANSITION_RETURNING
            else:
                transition = TRANSITION_NEW

            walked.append({
                "season": season,
                "team": team,
                "walk_transition": transition,
                "walk_has_prior": transition == TRANSITION_CONTINUING,
                "walk_previous_season": (
                    seasons[index - 1] if transition == TRANSITION_CONTINUING
                    else ""
                ),
            })

    walk_frame = pd.DataFrame(walked)

    merged = summary.merge(walk_frame, on=["season", "team"], how="outer",
                           indicator=True)

    unmatched = merged[merged["_merge"] != "both"]

    audit.record(
        "T9a",
        "Independent walk produces the same team-season universe",
        "100 team-seasons",
        0, len(unmatched),
        unmatched.empty,
        "; ".join(f"{r.season} {r.team}" for r in unmatched.head(5).itertuples()),
    )

    if not unmatched.empty:
        return

    disagreements = []

    for mine, theirs in [
        ("transition_type", "walk_transition"),
        ("has_previous_season", "walk_has_prior"),
        ("previous_season", "walk_previous_season"),
    ]:

        differs = merged[mine].astype(str) != merged[theirs].astype(str)

        if differs.any():
            disagreements.append(f"{mine}: {int(differs.sum())} rows")

    audit.record(
        "T9b",
        "Independent walk agrees on transition type, prior flag and source season",
        "3 fields x 100 team-seasons",
        "0 disagreements",
        f"{len(disagreements)} disagreeing fields",
        not disagreements,
        "; ".join(disagreements),
    )

    # And the reconstructed transitions must agree with Instrument 3's own
    # availability flag, which was built by a different instrument entirely.
    merged3 = summary.merge(instrument3, on=["season", "team"], how="inner")

    differs = merged3["has_previous_season"].astype(bool) != merged3[
        "available"].astype(bool)

    audit.record(
        "T9c",
        "Reconstructed transitions agree with Instrument 3's availability flag",
        f"{len(merged3)} team-seasons",
        0, int(differs.sum()),
        not differs.any(),
        "; ".join(
            f"{r.season} {r.team}"
            for r in merged3[differs].head(5).itertuples()
        ),
    )


def test_provenance(audit):

    opened = []

    for path in _OPENED_PATHS:
        try:
            opened.append(Path(path).resolve())
        except (OSError, ValueError):
            continue

    raw_touches = [
        str(path) for path in opened
        if RAW_DIR == path or RAW_DIR in path.parents
    ]

    audit.record(
        "P1",
        "No file under data/raw/ was opened at any point",
        "runtime file-access record",
        0, len(raw_touches),
        not raw_touches,
        "; ".join(sorted(set(raw_touches))[:5]),
    )

    allowed = DECLARED_INPUTS | {
        AUDIT_OUTPUT.resolve(), SUMMARY_OUTPUT.resolve(),
    }

    data_files = {
        path for path in opened
        if path.suffix.lower() in {".csv", ".xls", ".xlsx", ".json"}
        and PROJECT_ROOT in path.parents
    }

    unexpected = sorted(
        str(path.relative_to(PROJECT_ROOT)) for path in data_files - allowed
    )

    audit.record(
        "P2",
        "Only the two declared inputs were read",
        "runtime file-access record",
        "0 unexpected", f"{len(unexpected)} unexpected",
        not unexpected,
        "; ".join(unexpected[:5]),
    )

    source = Path(__file__).read_text(encoding="utf-8")

    forbidden = ["read_" + "html(", "read_" + "excel("]

    found = [token for token in forbidden if token in source]

    audit.record(
        "P3",
        "No FBref HTML/Excel table reader appears in this source",
        "static source scan",
        "0 occurrences", f"{len(found)} occurrences",
        not found,
        "; ".join(found),
    )


# ============================================================
# REPORT
# ============================================================

def status_text(passed):
    return "PASS" if passed else "FAIL"


def line(label, value, verdict=None):

    if verdict is None:
        print(f"  {label:<34}{value}")
    else:
        print(f"  {label:<34}{value:<30}{verdict}")


def print_test_table(audit):

    print()
    print("=" * 79)
    print("VALIDATION DETAIL")
    print("=" * 79)
    print()

    markers = {
        "PASS": "PASS", "FAIL": "FAIL",
        "MEASURED": "----", "REVIEW": "REV ",
    }

    for row in audit.frame().itertuples():

        print(f"  {markers[row.status]}  {row.test_id:<5} {row.test}")
        print(f"              scope   : {row.scope}")
        print(f"              expected: {row.expected}")
        print(f"              observed: {row.observed}")

        if row.detail:
            print(f"              {row.detail}")


def print_transition_map(transitions, summary):

    print()
    print("=" * 79)
    print("SEASON TRANSITION MAP")
    print("=" * 79)
    print()
    print("  'New to the current season' is the finding. Promotion is a")
    print("  stronger claim this data cannot make.")

    typed = {
        (row.season, row.team): row.transition_type
        for row in summary.itertuples()
    }

    for transition in transitions:

        print()
        print(f"  {transition['previous_season']}  ->  {transition['season']}")
        print(f"    returning  {len(transition['returning']):>2}")

        print(f"    departing  {len(transition['departing']):>2}   "
              f"{', '.join(transition['departing'])}")

        print(f"    new        {len(transition['new_to_current_season']):>2}")

        for team in transition["new_to_current_season"]:

            kind = typed[(transition["season"], team)]

            note = (
                "returned after an absence"
                if kind == TRANSITION_RETURNING
                else "not seen in this dataset before"
            )

            print(f"                    {team:<20}{kind:<26}{note}")


def print_prior_map(summary):

    print()
    print("=" * 79)
    print("PREVIOUS-SEASON PRIOR ELIGIBILITY")
    print("=" * 79)
    print()

    counts = summary["previous_season_status_conservative"].value_counts()

    for status, count in counts.items():
        print(f"    {status:<34}{count:>4} team-seasons")

    print()
    print(f"    {'Season':<12}{'With prior':>11}{'Without':>9}   Without prior")

    for season, group in summary.groupby("season"):

        with_prior = int(group["has_previous_season"].sum())

        without = group[~group["has_previous_season"]]

        names = ", ".join(sorted(without["team"]))

        if len(names) > 46:
            names = names[:43] + "..."

        print(f"    {season:<12}{with_prior:>11}{len(without):>9}   {names}")


def print_returning_report(returning, summary):

    print()
    print("=" * 79)
    print("RETURNING TEAMS - THE STALE-PRIOR TRAP")
    print("=" * 79)
    print()
    print("  Present, then absent, then back. A naive 'last time we saw them'")
    print("  lookup would hand these sides a prior from years earlier.")
    print()

    if returning.empty:
        print("    (none)")
        return

    print(
        f"    {'Season':<12}{'Team':<18}{'Last seen':<12}"
        f"{'Gap':>4}   Prior"
    )

    for row in returning.itertuples():
        print(
            f"    {row.season:<12}{row.team:<18}{row.last_season_seen:<12}"
            f"{row.seasons_absent_before_return:>4}   "
            f"{'ABSENT (correct)' if not row.has_previous_season else 'PRESENT'}"
        )


def print_sanction_report(summary):

    print()
    print("=" * 79)
    print("SANCTION CARRY-THROUGH")
    print("=" * 79)
    print()

    sanctioned = summary[
        summary["has_previous_season"]
        & (summary["previous_season_sanction"] != 0)
    ]

    if sanctioned.empty:
        print("    (no sanctioned prior in use)")
        return

    print(
        f"    {'Consuming':<12}{'Team':<18}{'From':<12}"
        f"{'Raw':>5}{'Sanc':>6}{'Used':>6}   identity"
    )

    for row in sanctioned.itertuples():

        holds = (
            row.previous_season_points_used
            == row.previous_season_points_raw + row.previous_season_sanction
        )

        print(
            f"    {row.season:<12}{row.team:<18}{row.previous_season:<12}"
            f"{row.previous_season_points_raw:>5.0f}"
            f"{row.previous_season_sanction:>6.0f}"
            f"{row.previous_season_points_used:>6.0f}   "
            f"{'holds' if holds else 'BROKEN'}"
        )


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 4: TEAM TRANSITION AUDIT")
    print("=" * 79)
    print()
    print(f"  Inputs     : {MATCHES_INPUT.relative_to(PROJECT_ROOT)}")
    print(f"               {FEATURES_INPUT.relative_to(PROJECT_ROOT)}")
    print("  Question   : are identities, transitions and priors coherent")
    print("               across all five seasons?")
    print("  Identity   : exact fixture string - no fuzzy matching, no alias")
    print("  Scope      : no models, no Elo, no FBref, no imputation, no")
    print("               older-season backfill, no automatic repair")

    matches, features = load_inputs()

    audit = Audit()

    print()
    print("  Reconstructing transitions from the match foundation ...")

    totals = reconstruct_season_totals(matches)

    seasons, teams_by_season, transitions = reconstruct_transitions(matches)

    summary = build_summary(matches, totals, seasons, teams_by_season)

    print(f"  {len(seasons)} seasons, {len(summary)} team-seasons, "
          f"{len(transitions)} adjacent transitions.")

    print("  T8  identity integrity ...")
    test_t8_identity_integrity(matches, features, teams_by_season, audit)

    print("  T1  twenty teams per season ...")
    test_t1_twenty_teams(teams_by_season, audit)

    print("  T2  team continuity ...")
    test_t2_continuity(transitions, audit)

    print("  T3  turnover and the promotion claim ...")
    test_t3_turnover(transitions, summary, audit)

    print("  T4  prior eligibility, reconciled with Instrument 3 ...")
    instrument3 = collapse_instrument3_priors(features, audit)
    summary = test_t4_prior_eligibility(summary, instrument3, audit)

    print("  T5  returning teams ...")
    returning = test_t5_returning_teams(summary, totals, instrument3, audit)

    print("  T6  sanction carry-through ...")
    test_t6_sanctions(summary, totals, audit)

    print("  T7  cross-season contamination ...")
    coincidences = test_t7_cross_season_contamination(
        summary, totals, instrument3, audit
    )

    print("  T9  independent reconstruction ...")
    test_t9_independent_reconstruction(matches, summary, instrument3, audit)

    print("  P   input provenance ...")
    test_provenance(audit)

    # ---- reports
    print_test_table(audit)
    print_transition_map(transitions, summary)
    print_prior_map(summary)
    print_returning_report(returning, summary)
    print_sanction_report(summary)

    # ---- outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    written_summary = summary[SUMMARY_COLUMNS]

    written_summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8")

    audit_frame = audit.frame()
    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(audit_frame)} entries)")
    print(f"  {SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(written_summary)} team-seasons, "
          f"{len(SUMMARY_COLUMNS)} columns)")

    # ---- verdict
    failures = audit.failures()
    reviews = audit.reviews()

    def outcome(prefix):
        rows = [r for r in audit.rows if r["test_id"].startswith(prefix)]
        return status_text(all(r["status"] != "FAIL" for r in rows))

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 4")
    print("=" * 79)
    print()

    line("Seasons:", f"{len(seasons)}")
    line("Team-seasons:", f"{len(summary)}")
    line("Adjacent transitions:", f"{len(transitions)}")
    line("Distinct identities:", f"{summary['team'].nunique()}")
    line("T1 twenty teams per season:", "20 x 5", outcome("T1"))
    line("T2 team continuity:", "returning + new = 20", outcome("T2"))
    line("T3 turnover pattern:", "3 in / 3 out, typed", outcome("T3"))
    line("T4 prior eligibility:", "reconciles with I3", outcome("T4"))
    line(
        "T5 returning teams:",
        f"{len(returning)} found, none inherit",
        outcome("T5"),
    )
    line("T6 sanctions:", "used = raw + sanction", outcome("T6"))
    line(
        "T7 cross-season contamination:",
        f"{len(coincidences)} coincidences checked",
        outcome("T7"),
    )
    line("T8 identity integrity:", "exact strings, no alias", outcome("T8"))
    line("T9 independent reconstruction:", "walk agrees", outcome("T9"))
    line("P  input provenance:", "no FBref, no data/raw", outcome("P"))

    if reviews:
        print()
        print("  DOCUMENTED FINDINGS (do not fail the run):")

        for review in reviews:
            print(f"    {review['test_id']} {review['test']}")
            print(f"        {review['detail']}")

    if failures:
        print()
        print("  FAILURES:")

        for failure in failures:
            print(
                f"    {failure['test_id']} {failure['test']}: "
                f"expected {failure['expected']}, got {failure['observed']} "
                f"{failure['detail']}".rstrip()
            )

    total_tests = len([
        r for r in audit.rows if r["status"] in {"PASS", "FAIL"}
    ])

    print()
    print(f"  Tests run          : {total_tests}")
    print(f"  Tests passed       : {total_tests - len(failures)}")
    print(f"  Tests failed       : {len(failures)}")
    print(f"  Documented findings: {len(reviews)}")
    print()

    if failures:
        print("STATUS: FAIL / INVESTIGATE")
        print()
        return EXIT_FAIL

    print("STATUS: PASS")
    print()

    return EXIT_PASS


def main():

    configure_stdout()

    try:
        return run()

    except FatalError as error:
        print()
        print("=" * 79)
        print("PHASE 1 - INSTRUMENT 4")
        print("=" * 79)
        print()
        print(f"  FATAL: {error}")
        print()
        print("STATUS: FATAL")
        print()
        return EXIT_FATAL

    except Exception:
        print()
        print("=" * 79)
        print("PHASE 1 - INSTRUMENT 4")
        print("=" * 79)
        print()
        print("  FATAL: unexpected exception")
        print()
        traceback.print_exc()
        print()
        print("STATUS: FATAL")
        print()
        return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())
