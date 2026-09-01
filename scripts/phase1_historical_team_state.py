"""
===============================================================================
PHASE 1 - INSTRUMENT 2
HISTORICAL TEAM STATE
===============================================================================

PURPOSE
    Construct, and independently PROVE, the information that would have been
    available about each team immediately before each match.

    This is the information layer every later feature sits on. If it leaks,
    everything above it is worthless, so the proof matters more than the
    features.

CANONICAL SOURCE
    outputs/phase1_matches.csv   (1,900 matches, validated by Instrument 1)

    No FBref aggregate is read. data/raw/ is not touched at all by this
    instrument - it works exclusively off Instrument 1's validated output.

LOCKED TEMPORAL RULE
    For a match at date T, historical information may use only matches where

        historical_date < T

    STRICT. Same-day matches never contribute.

    This is implemented mechanically via np.searchsorted(dates, T, "left"),
    which returns exactly the count of entries strictly less than T. Row
    position is never used as a proxy for date - see section 10 of the spec.
    Phase 0 ruled kickoff Time untrusted for within-day ordering, so date is
    the only eligibility key.

WHAT IS NOT DONE HERE
    no models, no Elo, no XGBoost, no betting predictions, no FBref
    final-season statistics as current predictors, no cold-start imputation,
    no same-day results, no future results, and no silent dropping of
    awkward matches. A failing validation stops the run at FAIL rather than
    being smoothed away.

THE PERTURBATION TEST IS COMPLETE, NOT SAMPLED
    Spec section 9 T2 asks for sampled matches rewritten to 9-0. Sampling is
    not necessary here. Within one matchweek each team plays exactly once, so
    no perturbed match can sit in another perturbed match's history. That
    makes a whole (season, matchweek) group safe to perturb at once, and 190
    such groups cover ALL 1,900 matches.

    Groups are kept season-by-season on purpose. Perturbing season S changes
    season S's totals, which legitimately feed season S+1's previous-season
    prior; perturbing every season at once would flag that as a violation
    when it is in fact correct behaviour.
===============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Single source of truth for the sanctions - re-declaring them here would let
# the two instruments drift apart silently.
from phase1_match_foundation import SANCTION_REGISTRY, SANCTION_NOTES


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MATCHES_INPUT = OUTPUTS_DIR / "phase1_matches.csv"
SUMMARY_INPUT = OUTPUTS_DIR / "phase1_season_team_summary.csv"

STATE_OUTPUT = OUTPUTS_DIR / "phase1_historical_team_state.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase1_historical_state_audit.csv"

EXPECTED_TOTAL_MATCHES = 1900
EXPECTED_TEAM_SIDES = EXPECTED_TOTAL_MATCHES * 2

LAST_N = 5

PERTURBED_HOME_GOALS = 9
PERTURBED_AWAY_GOALS = 0


# Required by spec section 8, in the order given there.
REQUIRED_COLUMNS = [
    "season",
    "date",
    "home_team",
    "away_team",

    "home_matches_before",
    "home_points_before",
    "home_gf_before",
    "home_ga_before",
    "home_last5_points_before",
    "home_previous_match_points_before",

    "away_matches_before",
    "away_points_before",
    "away_gf_before",
    "away_ga_before",
    "away_last5_points_before",
    "away_previous_match_points_before",

    "home_venue_matches_before",
    "home_venue_points_before",
    "home_venue_gf_before",
    "home_venue_ga_before",

    "away_venue_matches_before",
    "away_venue_points_before",
    "away_venue_gf_before",
    "away_venue_ga_before",

    "home_has_previous_season",
    "home_previous_season_points",
    "home_previous_season_gf",
    "home_previous_season_ga",
    "home_previous_season_gd",
    "home_previous_season_matches",

    "away_has_previous_season",
    "away_previous_season_points",
    "away_previous_season_gf",
    "away_previous_season_ga",
    "away_previous_season_gd",
    "away_previous_season_matches",
]

# Carried in addition to the required set.
#
#   *_gd_before                     required by spec section 2
#   *_last5_matches_used            makes "fewer than five if fewer exist"
#                                   auditable instead of asserted
#   *_previous_match_date           the evidence T6 is checked against
#   *_previous_season_points_from_results
#                                   spec section 7 - the sanctioned and
#                                   unsanctioned priors are kept side by side
#   *_previous_season_status        why a prior is absent, which "False"
#                                   alone cannot express
SUPPLEMENTARY_COLUMNS = [
    "matchweek",

    "home_gd_before",
    "away_gd_before",

    "home_last5_matches_used",
    "away_last5_matches_used",

    "home_previous_match_date",
    "away_previous_match_date",

    "home_previous_season_points_from_results",
    "away_previous_season_points_from_results",

    "home_previous_season_status",
    "away_previous_season_status",
]

OUTPUT_COLUMNS = REQUIRED_COLUMNS + SUPPLEMENTARY_COLUMNS


# Feature columns compared by the perturbation test. Identity columns are
# excluded because a perturbation never touches them.
PERTURBATION_COMPARE_COLUMNS = [
    column for column in OUTPUT_COLUMNS
    if column not in {"season", "date", "matchweek", "home_team", "away_team"}
]


PREVIOUS_SEASON_STATUS_AVAILABLE = "available"
PREVIOUS_SEASON_STATUS_NO_PRIOR_SEASON = "no_prior_season_in_dataset"
PREVIOUS_SEASON_STATUS_ABSENT = "absent_from_previous_season"


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

    def __init__(self):
        self.rows = []

    def record(self, test_id, test, expected, observed, passed, detail=""):

        self.rows.append({
            "test_id": test_id,
            "test": test,
            "expected": expected,
            "observed": observed,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

        return passed

    def measure(self, test_id, test, observed, detail=""):
        """A measurement, not a pass/fail gate. Spec section 4: measure first."""

        self.rows.append({
            "test_id": test_id,
            "test": test,
            "expected": "(measurement)",
            "observed": observed,
            "status": "MEASURED",
            "detail": detail,
        })

    def failures(self):
        return [row for row in self.rows if row["status"] == "FAIL"]

    def all_passed(self):
        return not self.failures()

    def frame(self):
        return pd.DataFrame(self.rows, columns=[
            "test_id", "test", "expected", "observed", "status", "detail",
        ])


# ============================================================
# INPUT
# ============================================================

def load_matches():

    matches = pd.read_csv(MATCHES_INPUT)

    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")

    # Deterministic ordering. The sort is by DATE; it never substitutes for
    # the date comparisons that follow.
    matches = matches.sort_values(
        ["season", "date", "home_team", "away_team"]
    ).reset_index(drop=True)

    matches["match_id"] = matches.index

    return matches


def to_team_sides(matches):
    """One row per team per match. 1,900 matches -> 3,800 team-sides."""

    home = pd.DataFrame({
        "match_id": matches["match_id"],
        "season": matches["season"],
        "date": matches["date"],
        "team": matches["home_team"],
        "side": "home",
        "gf": matches["home_goals"],
        "ga": matches["away_goals"],
        "pts": matches["home_points_from_result"],
    })

    away = pd.DataFrame({
        "match_id": matches["match_id"],
        "season": matches["season"],
        "date": matches["date"],
        "team": matches["away_team"],
        "side": "away",
        "gf": matches["away_goals"],
        "ga": matches["home_goals"],
        "pts": matches["away_points_from_result"],
    })

    return pd.concat([home, away], ignore_index=True)


# ============================================================
# PREVIOUS-SEASON PRIOR
# ============================================================

def season_totals(matches):
    """
    Season-end totals rebuilt from match results, plus the sanction registry.

    This is a from-scratch derivation off phase1_matches.csv. It is
    cross-checked against Instrument 1's summary file in T9b, but never
    sourced from it - two independent derivations agreeing is evidence,
    one file read twice is not.
    """

    sides = to_team_sides(matches)

    totals = sides.groupby(["season", "team"], as_index=False).agg(
        matches_played=("match_id", "size"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
        points_from_results=("pts", "sum"),
    )

    totals["gd"] = totals["gf"] - totals["ga"]

    totals["sanction"] = [
        SANCTION_REGISTRY.get((season, team), 0)
        for season, team in zip(totals["season"], totals["team"])
    ]

    totals["points_after_sanction"] = (
        totals["points_from_results"] + totals["sanction"]
    )

    return totals


def build_previous_season_lookup(matches):
    """
    Map (season, team) -> that team's PREVIOUS season final totals.

    A team absent from season N-1 gets no entry. Nothing is invented.

    The prior points use Pts_after_sanction (spec section 7), with
    Pts_from_results carried alongside so the distinction survives.
    """

    totals = season_totals(matches)

    seasons = sorted(matches["season"].unique())

    previous_of = {
        current: previous
        for previous, current in zip(seasons, seasons[1:])
    }

    by_key = {
        (row.season, row.team): row
        for row in totals.itertuples()
    }

    teams_by_season = {
        season: set(group["team"])
        for season, group in totals.groupby("season")
    }

    lookup = {}

    for season in seasons:

        previous_season = previous_of.get(season)

        for team in teams_by_season[season]:

            if previous_season is None:
                # 2021-2022 is the earliest season held. Its teams are not
                # "promoted" - the dataset simply starts here. Saying
                # otherwise would be a claim the data cannot support.
                lookup[(season, team)] = {
                    "has_previous_season": False,
                    "status": PREVIOUS_SEASON_STATUS_NO_PRIOR_SEASON,
                }
                continue

            prior = by_key.get((previous_season, team))

            if prior is None:
                lookup[(season, team)] = {
                    "has_previous_season": False,
                    "status": PREVIOUS_SEASON_STATUS_ABSENT,
                }
                continue

            lookup[(season, team)] = {
                "has_previous_season": True,
                "status": PREVIOUS_SEASON_STATUS_AVAILABLE,
                "source_season": previous_season,
                "points": prior.points_after_sanction,
                "points_from_results": prior.points_from_results,
                "gf": prior.gf,
                "ga": prior.ga,
                "gd": prior.gd,
                "matches": prior.matches_played,
            }

    return lookup


# ============================================================
# CORE - HISTORICAL STATE
# ============================================================

def historical_state_for_group(dates, gf, ga, pts, query_dates):
    """
    The strict boundary, applied mechanically.

    dates/gf/ga/pts describe one team's matches in one season, sorted by
    date ascending. query_dates are the match dates to evaluate.

    np.searchsorted(..., side="left") returns the number of entries STRICTLY
    less than the query. Equal dates land to the left of the cut, so a
    same-day match can never be counted. This is the whole temporal rule, in
    one call, with no room for an off-by-one to hide.
    """

    counts = np.searchsorted(dates, query_dates, side="left")

    # Leading zero so cumulative[k] is the sum over the first k matches.
    cumulative_pts = np.concatenate([[0], np.cumsum(pts)])
    cumulative_gf = np.concatenate([[0], np.cumsum(gf)])
    cumulative_ga = np.concatenate([[0], np.cumsum(ga)])

    window_start = np.maximum(counts - LAST_N, 0)

    last_n_points = cumulative_pts[counts] - cumulative_pts[window_start]
    last_n_used = counts - window_start

    has_history = counts > 0

    previous_points = np.where(has_history, pts[counts - 1], np.nan)

    previous_dates = np.where(
        has_history,
        dates[counts - 1],
        np.datetime64("NaT"),
    )

    return {
        "matches_before": counts,
        "points_before": cumulative_pts[counts],
        "gf_before": cumulative_gf[counts],
        "ga_before": cumulative_ga[counts],
        "last5_points_before": np.where(has_history, last_n_points, np.nan),
        "last5_matches_used": last_n_used,
        "previous_match_points_before": previous_points,
        "previous_match_date": previous_dates,
    }


def build_state(matches, previous_season_lookup):
    """
    Build every team-side's pre-match state.

    Current-season state is scoped to (season, team) so nothing carries over
    a season boundary. Venue state is scoped to (season, team, side).
    """

    sides = to_team_sides(matches)

    sides = sides.sort_values(["season", "team", "date"]).reset_index(drop=True)

    overall_frames = []
    venue_frames = []

    for (season, team), group in sides.groupby(["season", "team"], sort=False):

        dates = group["date"].to_numpy()
        gf = group["gf"].to_numpy()
        ga = group["ga"].to_numpy()
        pts = group["pts"].to_numpy()

        # ---- current-season overall state
        state = historical_state_for_group(dates, gf, ga, pts, dates)

        overall = pd.DataFrame(state)
        overall["match_id"] = group["match_id"].to_numpy()
        overall["side"] = group["side"].to_numpy()

        overall_frames.append(overall)

        # ---- venue state
        #
        # A team's home state is built ONLY from its home matches, and its
        # away state ONLY from its away matches. The two never mix.
        for side in ("home", "away"):

            side_mask = group["side"].to_numpy() == side

            side_dates = dates[side_mask]

            venue_state = historical_state_for_group(
                side_dates,
                gf[side_mask],
                ga[side_mask],
                pts[side_mask],
                # Queried only at this team's matches at THIS venue.
                side_dates,
            )

            venue = pd.DataFrame({
                "match_id": group["match_id"].to_numpy()[side_mask],
                "side": side,
                "venue_matches_before": venue_state["matches_before"],
                "venue_points_before": venue_state["points_before"],
                "venue_gf_before": venue_state["gf_before"],
                "venue_ga_before": venue_state["ga_before"],
            })

            venue_frames.append(venue)

    overall_state = pd.concat(overall_frames, ignore_index=True)
    venue_state = pd.concat(venue_frames, ignore_index=True)

    side_state = overall_state.merge(venue_state, on=["match_id", "side"])

    side_state["gd_before"] = side_state["gf_before"] - side_state["ga_before"]

    # ---- previous-season prior, attached per team-side
    key_frame = sides[["match_id", "side", "season", "team"]]

    side_state = side_state.merge(key_frame, on=["match_id", "side"])

    priors = [
        previous_season_lookup[(season, team)]
        for season, team in zip(side_state["season"], side_state["team"])
    ]

    side_state["has_previous_season"] = [p["has_previous_season"] for p in priors]
    side_state["previous_season_status"] = [p["status"] for p in priors]

    for column, key in [
        ("previous_season_points", "points"),
        ("previous_season_points_from_results", "points_from_results"),
        ("previous_season_gf", "gf"),
        ("previous_season_ga", "ga"),
        ("previous_season_gd", "gd"),
        ("previous_season_matches", "matches"),
    ]:
        side_state[column] = [p.get(key, np.nan) for p in priors]

    side_state["previous_season_source"] = [
        p.get("source_season", "") for p in priors
    ]

    return side_state


def assemble_output(matches, side_state):
    """Pivot the team-side state back into one row per match."""

    carried = [
        "matches_before", "points_before", "gf_before", "ga_before",
        "last5_points_before", "last5_matches_used",
        "previous_match_points_before", "previous_match_date",
        "gd_before",
        "venue_matches_before", "venue_points_before",
        "venue_gf_before", "venue_ga_before",
        "has_previous_season", "previous_season_status",
        "previous_season_points", "previous_season_points_from_results",
        "previous_season_gf", "previous_season_ga",
        "previous_season_gd", "previous_season_matches",
    ]

    output = matches[[
        "match_id", "season", "date", "matchweek", "home_team", "away_team",
    ]].copy()

    for side in ("home", "away"):

        side_rows = side_state[side_state["side"] == side]

        renamed = {column: f"{side}_{column}" for column in carried}

        output = output.merge(
            side_rows[["match_id"] + carried].rename(columns=renamed),
            on="match_id",
            how="left",
        )

    return output


def build_everything(matches):
    """One call that produces the full match-level state table."""

    previous_season_lookup = build_previous_season_lookup(matches)

    side_state = build_state(matches, previous_season_lookup)

    output = assemble_output(matches, side_state)

    return output, side_state, previous_season_lookup


# ============================================================
# VALIDATION
# ============================================================

def validate_t1_strict_boundary(matches, side_state, audit):
    """
    T1 - every historical match used has date < current match date.

    Verified by INDEPENDENT brute force, not by re-running searchsorted. The
    production path and the checking path must not share the bug.
    """

    sides = to_team_sides(matches)

    by_team = {
        key: group.sort_values("date")
        for key, group in sides.groupby(["season", "team"], sort=False)
    }

    date_by_match_side = {
        (row.match_id, row.side): row.date
        for row in sides.itertuples()
    }

    violations = []
    same_day_contributions = 0
    count_mismatches = 0

    for row in side_state.itertuples():

        current_date = date_by_match_side[(row.match_id, row.side)]

        group = by_team[(row.season, row.team)]

        # Explicit comparison, one match at a time.
        eligible = [
            match_date for match_date in group["date"]
            if match_date < current_date
        ]

        same_day = [
            match_date for match_date in group["date"]
            if match_date == current_date
        ]

        if len(eligible) != row.matches_before:
            count_mismatches += 1

            if len(violations) < 5:
                violations.append(
                    f"{row.season} {row.team}: brute force {len(eligible)}, "
                    f"state {row.matches_before}"
                )

        if eligible and max(eligible) >= current_date:
            violations.append(f"{row.season} {row.team}: history reaches current date")

        # A same-day match contributing would push the count above the
        # strictly-earlier count.
        if row.matches_before > len(eligible):
            same_day_contributions += 1

    audit.record(
        "T1",
        "Every historical match used has date < current match date",
        f"0 violations across {len(side_state)} team-sides",
        f"{count_mismatches} count mismatches",
        count_mismatches == 0 and not violations,
        "; ".join(violations[:5]),
    )

    return same_day_contributions


def validate_t2_perturbation(matches, baseline_output, audit):
    """
    T2 - a match's own score cannot reach its own historical features.

    Every one of the 1,900 matches is rewritten to 9-0 and the entire state
    layer rebuilt. See the module docstring for why whole (season, matchweek)
    groups are safe to perturb together and why seasons are kept separate.
    """

    baseline = baseline_output.set_index("match_id")

    groups = matches.groupby(["season", "matchweek"], sort=True)

    perturbed_matches_total = 0
    changed_values_total = 0
    changed_rows_total = 0
    rebuild_count = 0
    examples = []

    for (season, matchweek), group in groups:

        target_ids = list(group["match_id"])

        perturbed = matches.copy()

        target_mask = perturbed["match_id"].isin(target_ids)

        perturbed.loc[target_mask, "home_goals"] = PERTURBED_HOME_GOALS
        perturbed.loc[target_mask, "away_goals"] = PERTURBED_AWAY_GOALS
        perturbed.loc[target_mask, "result"] = "H"
        perturbed.loc[target_mask, "home_points_from_result"] = 3
        perturbed.loc[target_mask, "away_points_from_result"] = 0

        rebuilt, _, _ = build_everything(perturbed)

        rebuild_count += 1
        perturbed_matches_total += len(target_ids)

        rebuilt = rebuilt.set_index("match_id")

        before = baseline.loc[target_ids, PERTURBATION_COMPARE_COLUMNS]
        after = rebuilt.loc[target_ids, PERTURBATION_COMPARE_COLUMNS]

        differs = ~(
            (before == after)
            | (before.isna() & after.isna())
        )

        changed_values = int(differs.to_numpy().sum())
        changed_rows = int(differs.any(axis=1).sum())

        changed_values_total += changed_values
        changed_rows_total += changed_rows

        if changed_values and len(examples) < 5:
            changed_columns = list(differs.columns[differs.any(axis=0)])
            examples.append(f"{season} MW{matchweek}: {changed_columns}")

    audit.record(
        "T2",
        "Perturbing a match to 9-0 does not change its own historical state",
        "0 changed feature values",
        f"{changed_values_total} changed values in {changed_rows_total} rows",
        changed_values_total == 0,
        f"{perturbed_matches_total} matches perturbed across {rebuild_count} "
        f"full rebuilds; " + "; ".join(examples),
    )

    return changed_values_total, perturbed_matches_total, rebuild_count


def validate_t3_same_day(matches, side_state, audit, same_day_contributions):
    """
    T3 - no same-day match contributes.

    Also measures the size of the effect, rather than assuming it, by
    recomputing under the NON-strict rule (date <= T) and counting what
    changes.
    """

    audit.record(
        "T3a",
        "No same-day match contributes to historical state",
        0, same_day_contributions,
        same_day_contributions == 0,
    )

    sides = to_team_sides(matches)

    # How often does a team have another of its own matches on the same date?
    own_same_day = int(
        sides.duplicated(subset=["season", "team", "date"]).sum()
    )

    audit.record(
        "T3b",
        "A team never has a second match on the same date",
        0, own_same_day,
        own_same_day == 0,
        "Confirms the only same-day match a team can have is the current one",
    )

    # Measure the effect of strictness: under date <= T every team-side would
    # absorb its own current match.
    changed_sides = 0

    for (_, _), group in sides.groupby(["season", "team"], sort=False):

        dates = group.sort_values("date")["date"].to_numpy()

        strict = np.searchsorted(dates, dates, side="left")
        non_strict = np.searchsorted(dates, dates, side="right")

        changed_sides += int((strict != non_strict).sum())

    audit.measure(
        "T3c",
        "Team-sides whose state would change under a non-strict (<=) rule",
        changed_sides,
        "Each side would absorb its own current match - this is the leak the "
        "strict rule prevents",
    )

    # League-level context: how many matches share their date with another
    # match. This does NOT affect team history (a team plays once per date)
    # but it is why the strict rule is expensive.
    date_counts = matches.groupby(["season", "date"])["match_id"].transform("size")

    shared_date_matches = int((date_counts > 1).sum())

    audit.measure(
        "T3d",
        "Matches sharing their date with at least one other match",
        shared_date_matches,
        "League-level context only; team history is unaffected because no "
        "team plays twice on a date",
    )

    return changed_sides, shared_date_matches


def validate_t4_cold_starts(matches, side_state, audit):
    """T4 - the first match of each team-season carries zero current-season history."""

    sides = to_team_sides(matches)

    first_dates = (
        sides.groupby(["season", "team"], as_index=False)["date"].min()
        .rename(columns={"date": "first_date"})
    )

    date_by_match_side = {
        (row.match_id, row.side): row.date
        for row in sides.itertuples()
    }

    state = side_state.merge(first_dates, on=["season", "team"], how="left")

    state["own_date"] = [
        date_by_match_side[(row.match_id, row.side)]
        for row in state.itertuples()
    ]

    is_first = state["own_date"] == state["first_date"]

    first_sides = state[is_first]

    bad = first_sides[
        (first_sides["matches_before"] != 0)
        | (first_sides["points_before"] != 0)
        | (first_sides["gf_before"] != 0)
        | (first_sides["ga_before"] != 0)
        | first_sides["last5_points_before"].notna()
        | first_sides["previous_match_points_before"].notna()
    ]

    audit.record(
        "T4a",
        "First match of each team-season has zero current-season history",
        f"0 violations across {len(first_sides)} first-match sides",
        f"{len(bad)} violations",
        len(bad) == 0,
        "; ".join(
            f"{r.season} {r.team}" for r in bad.head(5).itertuples()
        ),
    )

    # Every zero-history side must BE a first match - nothing else may be empty.
    zero_history = state[state["matches_before"] == 0]

    unexpected = zero_history[~(zero_history["own_date"] == zero_history["first_date"])]

    audit.record(
        "T4b",
        "Zero-history sides are exactly the first-match sides",
        0, len(unexpected),
        len(unexpected) == 0,
        "; ".join(
            f"{r.season} {r.team}" for r in unexpected.head(5).itertuples()
        ),
    )

    # ---- measurements (spec section 4: measure, do not assume)
    audit.measure(
        "T4c",
        "Team-sides with zero current-season history",
        len(zero_history),
        "Expected to equal 5 seasons x 20 teams = 100 team-match sides",
    )

    cold_ids = set(zero_history["match_id"])

    cold_side_counts = zero_history.groupby("match_id").size()

    both_cold = int((cold_side_counts == 2).sum())
    one_cold = int((cold_side_counts == 1).sum())

    audit.measure(
        "T4d",
        "Matches containing at least one cold-start side",
        len(cold_ids),
        f"{both_cold} with both sides cold, {one_cold} with one side cold",
    )

    cold_matches = matches[matches["match_id"].isin(cold_ids)]

    weeks = sorted(cold_matches["matchweek"].unique())

    audit.measure(
        "T4e",
        "Matchweeks in which cold starts occur",
        ", ".join(f"MW{week}" for week in weeks),
        "A first match outside MW1 would mean a rearranged opening fixture",
    )

    return len(zero_history), len(cold_ids), both_cold, one_cold, weeks


def validate_t5_history_bounds(matches, side_state, audit):
    """T5 - history never exceeds the number of previously completed matches."""

    sides = to_team_sides(matches)

    played_counts = (
        sides.groupby(["season", "team"], as_index=False)
        .size()
        .rename(columns={"size": "season_total"})
    )

    state = side_state.merge(played_counts, on=["season", "team"], how="left")

    # A side can have at most (season_total - 1) earlier matches.
    over = state[state["matches_before"] > state["season_total"] - 1]

    audit.record(
        "T5a",
        "matches_before never exceeds the team's earlier completed matches",
        0, len(over),
        len(over) == 0,
        "; ".join(f"{r.season} {r.team}" for r in over.head(5).itertuples()),
    )

    # Points can never exceed 3 per completed match, goals can never be
    # negative, and totals must be internally consistent.
    impossible = state[
        (state["points_before"] > 3 * state["matches_before"])
        | (state["points_before"] < 0)
        | (state["gf_before"] < 0)
        | (state["ga_before"] < 0)
        | (state["gd_before"] != state["gf_before"] - state["ga_before"])
    ]

    audit.record(
        "T5b",
        "Accumulated state is arithmetically possible",
        0, len(impossible),
        len(impossible) == 0,
        "; ".join(f"{r.season} {r.team}" for r in impossible.head(5).itertuples()),
    )

    return len(over) + len(impossible)


def validate_t6_previous_match(matches, side_state, audit):
    """T6 - the previous-match feature comes from the immediately preceding date."""

    sides = to_team_sides(matches)

    by_team = {
        key: group.sort_values("date")
        for key, group in sides.groupby(["season", "team"], sort=False)
    }

    date_by_match_side = {
        (row.match_id, row.side): row.date
        for row in sides.itertuples()
    }

    violations = []

    for row in side_state.itertuples():

        current_date = date_by_match_side[(row.match_id, row.side)]

        group = by_team[(row.season, row.team)]

        earlier = group[group["date"] < current_date]

        if earlier.empty:

            if pd.notna(row.previous_match_points_before):
                violations.append(
                    f"{row.season} {row.team}: value present with no earlier match"
                )

            continue

        expected_date = earlier["date"].max()
        expected_points = earlier.loc[earlier["date"].idxmax(), "pts"]

        if pd.Timestamp(row.previous_match_date) != expected_date:
            violations.append(
                f"{row.season} {row.team}: date {row.previous_match_date} "
                f"!= {expected_date}"
            )

        elif row.previous_match_points_before != expected_points:
            violations.append(
                f"{row.season} {row.team}: points "
                f"{row.previous_match_points_before} != {expected_points}"
            )

    audit.record(
        "T6",
        "Previous-match feature is the immediately preceding earlier match",
        f"0 violations across {len(side_state)} team-sides",
        f"{len(violations)} violations",
        not violations,
        "; ".join(violations[:5]),
    )

    return len(violations)


def validate_t7_last_five(side_state, audit):
    """T7 - last-5 never draws on more than five historical matches."""

    over = side_state[side_state["last5_matches_used"] > LAST_N]

    audit.record(
        "T7a",
        "Last-5 uses at most five historical matches",
        0, len(over),
        len(over) == 0,
    )

    # It must also use exactly min(5, matches_before) - no back-filling, no
    # silently dropping an available match.
    expected_used = np.minimum(side_state["matches_before"], LAST_N)

    wrong = side_state[side_state["last5_matches_used"] != expected_used]

    audit.record(
        "T7b",
        "Last-5 uses exactly min(5, matches_before) - no back-fill, no drop",
        0, len(wrong),
        len(wrong) == 0,
        "; ".join(f"{r.season} {r.team}" for r in wrong.head(5).itertuples()),
    )

    # Points must be possible for the number of matches actually used.
    with_history = side_state[side_state["matches_before"] > 0]

    impossible = with_history[
        (with_history["last5_points_before"] < 0)
        | (with_history["last5_points_before"]
           > 3 * with_history["last5_matches_used"])
    ]

    audit.record(
        "T7c",
        "Last-5 points are possible for the matches used",
        0, len(impossible),
        len(impossible) == 0,
    )

    # Short histories are real and must be visible, not padded to five.
    short = side_state[
        (side_state["matches_before"] > 0)
        & (side_state["matches_before"] < LAST_N)
    ]

    audit.measure(
        "T7d",
        "Team-sides with a genuine short (1-4 match) last-5 window",
        len(short),
        "These are carried short, never padded to five",
    )

    return len(over) + len(wrong) + len(impossible), len(short)


def validate_t8_venue(matches, side_state, audit):
    """T8 - venue history contains only matches at the relevant venue."""

    sides = to_team_sides(matches)

    by_team_side = {
        key: group.sort_values("date")
        for key, group in sides.groupby(["season", "team", "side"], sort=False)
    }

    date_by_match_side = {
        (row.match_id, row.side): row.date
        for row in sides.itertuples()
    }

    violations = []

    for row in side_state.itertuples():

        current_date = date_by_match_side[(row.match_id, row.side)]

        group = by_team_side[(row.season, row.team, row.side)]

        earlier = group[group["date"] < current_date]

        if len(earlier) != row.venue_matches_before:
            violations.append(
                f"{row.season} {row.team} {row.side}: "
                f"{len(earlier)} != {row.venue_matches_before}"
            )

        elif (
            earlier["pts"].sum() != row.venue_points_before
            or earlier["gf"].sum() != row.venue_gf_before
            or earlier["ga"].sum() != row.venue_ga_before
        ):
            violations.append(
                f"{row.season} {row.team} {row.side}: venue totals disagree"
            )

    audit.record(
        "T8a",
        "Venue history is rebuilt only from matches at that venue",
        f"0 violations across {len(side_state)} team-sides",
        f"{len(violations)} violations",
        not violations,
        "; ".join(violations[:5]),
    )

    # Venue state is a subset of overall state, so it can never exceed it.
    exceeds = side_state[
        (side_state["venue_matches_before"] > side_state["matches_before"])
        | (side_state["venue_points_before"] > side_state["points_before"])
        | (side_state["venue_gf_before"] > side_state["gf_before"])
        | (side_state["venue_ga_before"] > side_state["ga_before"])
    ]

    audit.record(
        "T8b",
        "Venue state never exceeds overall state",
        0, len(exceeds),
        len(exceeds) == 0,
        "; ".join(f"{r.season} {r.team}" for r in exceeds.head(5).itertuples()),
    )

    # A team can have played at most 19 matches at either venue.
    over_cap = side_state[side_state["venue_matches_before"] > 19]

    audit.record(
        "T8c",
        "Venue matches never exceed 19",
        0, len(over_cap),
        len(over_cap) == 0,
    )

    return len(violations) + len(exceeds) + len(over_cap)


def validate_t9_previous_season(matches, side_state, previous_season_lookup, audit):
    """T9 - the previous-season prior never comes from the current season."""

    seasons = sorted(matches["season"].unique())

    previous_of = {
        current: previous
        for previous, current in zip(seasons, seasons[1:])
    }

    wrong_source = 0
    examples = []

    for row in side_state.itertuples():

        if not row.has_previous_season:
            continue

        expected_source = previous_of.get(row.season)

        if row.previous_season_source != expected_source:
            wrong_source += 1

            if len(examples) < 5:
                examples.append(
                    f"{row.season} {row.team}: source "
                    f"{row.previous_season_source!r}"
                )

    audit.record(
        "T9a",
        "Previous-season prior is sourced from season N-1, never season N",
        0, wrong_source,
        wrong_source == 0,
        "; ".join(examples),
    )

    # Cross-check the from-scratch derivation against Instrument 1's summary.
    summary = pd.read_csv(SUMMARY_INPUT)

    summary_lookup = {
        (row.season, row.team): row
        for row in summary.itertuples()
    }

    mismatches = []

    for (season, team), prior in previous_season_lookup.items():

        if not prior["has_previous_season"]:
            continue

        reference = summary_lookup.get((prior["source_season"], team))

        if reference is None:
            mismatches.append(f"{season} {team}: absent from summary")
            continue

        if (
            prior["points"] != reference.Pts_after_sanction
            or prior["points_from_results"] != reference.Pts_from_results
            or prior["gf"] != reference.GF
            or prior["ga"] != reference.GA
            or prior["gd"] != reference.GD
            or prior["matches"] != reference.MP
        ):
            mismatches.append(f"{season} {team}: disagrees with summary")

    audit.record(
        "T9b",
        "Prior agrees with Instrument 1's independently built summary",
        0, len(mismatches),
        not mismatches,
        "; ".join(mismatches[:5]),
    )

    # A complete prior must be a full 38-match season.
    with_prior = side_state[side_state["has_previous_season"]]

    incomplete = with_prior[with_prior["previous_season_matches"] != 38]

    audit.record(
        "T9c",
        "Every previous-season prior covers a full 38-match season",
        0, len(incomplete),
        len(incomplete) == 0,
    )

    # The sanction must be visible in the prior, and only where it belongs.
    sanction_rows = []

    for (season, team), value in sorted(SANCTION_REGISTRY.items()):

        consumers = with_prior[
            (with_prior["previous_season_source"] == season)
            & (with_prior["team"] == team)
        ]

        if consumers.empty:
            sanction_rows.append(f"{season} {team}: no consuming season")
            continue

        after = set(consumers["previous_season_points"])
        raw = set(consumers["previous_season_points_from_results"])

        # points_after = points_from_results + sanction (sanction is negative)
        if after != {r + value for r in raw}:
            sanction_rows.append(
                f"{season} {team}: prior {after} vs raw {raw} and {value}"
            )

    audit.record(
        "T9d",
        "Sanctioned prior points carry Pts_after_sanction, with raw preserved",
        f"0 problems across {len(SANCTION_REGISTRY)} sanctions",
        f"{len(sanction_rows)} problems",
        not sanction_rows,
        "; ".join(sanction_rows),
    )

    return wrong_source + len(mismatches) + len(incomplete) + len(sanction_rows)


def validate_t10_promoted(matches, side_state, audit):
    """T10 - teams with no previous PL season stay explicitly marked."""

    seasons = sorted(matches["season"].unique())

    previous_of = {
        current: previous
        for previous, current in zip(seasons, seasons[1:])
    }

    teams_by_season = {
        season: set(group["home_team"]) | set(group["away_team"])
        for season, group in matches.groupby("season")
    }

    wrong_flag = 0
    fabricated = 0
    wrong_status = 0
    examples = []

    prior_columns = [
        "previous_season_points",
        "previous_season_points_from_results",
        "previous_season_gf",
        "previous_season_ga",
        "previous_season_gd",
        "previous_season_matches",
    ]

    for row in side_state.itertuples():

        previous_season = previous_of.get(row.season)

        if previous_season is None:
            expected_has = False
            expected_status = PREVIOUS_SEASON_STATUS_NO_PRIOR_SEASON
        elif row.team in teams_by_season[previous_season]:
            expected_has = True
            expected_status = PREVIOUS_SEASON_STATUS_AVAILABLE
        else:
            expected_has = False
            expected_status = PREVIOUS_SEASON_STATUS_ABSENT

        if bool(row.has_previous_season) != expected_has:
            wrong_flag += 1

            if len(examples) < 5:
                examples.append(f"{row.season} {row.team}: flag")

        if row.previous_season_status != expected_status:
            wrong_status += 1

        if not expected_has:

            values = [getattr(row, column) for column in prior_columns]

            if any(pd.notna(value) for value in values):
                fabricated += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {row.team}: fabricated prior")

    audit.record(
        "T10a",
        "has_previous_season matches actual presence in season N-1",
        0, wrong_flag,
        wrong_flag == 0,
        "; ".join(examples[:5]),
    )

    audit.record(
        "T10b",
        "Teams without a prior carry NO previous-season values at all",
        0, fabricated,
        fabricated == 0,
        "Nothing is imputed for a promoted side",
    )

    audit.record(
        "T10c",
        "Absence reason is recorded (promoted vs dataset boundary)",
        0, wrong_status,
        wrong_status == 0,
    )

    without = side_state[~side_state["has_previous_season"]]

    status_counts = without["previous_season_status"].value_counts().to_dict()

    audit.measure(
        "T10d",
        "Team-sides without a previous-season prior",
        len(without),
        str(status_counts),
    )

    return wrong_flag + fabricated + wrong_status, status_counts


def validate_structure(matches, output, audit):
    """Shape and schema of the delivered table."""

    audit.record(
        "S1",
        "One row per match",
        EXPECTED_TOTAL_MATCHES, len(output),
        len(output) == EXPECTED_TOTAL_MATCHES,
    )

    missing = [column for column in REQUIRED_COLUMNS if column not in output.columns]

    audit.record(
        "S2",
        "All spec-required columns present",
        "0 missing", f"{len(missing)} missing",
        not missing,
        ", ".join(missing),
    )

    # Counters and accumulators are always defined; only the genuinely
    # optional features may be empty.
    always_present = [
        "home_matches_before", "home_points_before", "home_gf_before",
        "home_ga_before", "away_matches_before", "away_points_before",
        "away_gf_before", "away_ga_before",
        "home_venue_matches_before", "home_venue_points_before",
        "home_venue_gf_before", "home_venue_ga_before",
        "away_venue_matches_before", "away_venue_points_before",
        "away_venue_gf_before", "away_venue_ga_before",
        "home_has_previous_season", "away_has_previous_season",
    ]

    unexpected_nulls = int(output[always_present].isna().sum().sum())

    audit.record(
        "S3",
        "No nulls in always-defined columns",
        0, unexpected_nulls,
        unexpected_nulls == 0,
    )

    # Identity columns must survive the round trip untouched.
    rejoined = output[["season", "date", "home_team", "away_team"]]

    reference = matches[["season", "date", "home_team", "away_team"]]

    identity_intact = rejoined.reset_index(drop=True).equals(
        reference.reset_index(drop=True)
    )

    audit.record(
        "S4",
        "Match identity is preserved and correctly aligned",
        "identical", "identical" if identity_intact else "DIVERGED",
        identity_intact,
    )

    # Nothing was dropped: every input match appears exactly once.
    dropped = set(matches["match_id"]) - set(output["match_id"])

    audit.record(
        "S5",
        "No match silently dropped",
        0, len(dropped),
        len(dropped) == 0,
    )


# ============================================================
# REPORT
# ============================================================

def line(label, value, verdict=None):

    if verdict is None:
        print(f"  {label:<34}{value}")
    else:
        print(f"  {label:<34}{value:<32}{verdict}")


def status_text(passed):
    return "PASS" if passed else "FAIL"


def print_test_table(audit):

    print()
    print("=" * 79)
    print("VALIDATION DETAIL")
    print("=" * 79)
    print()

    for row in audit.frame().itertuples():

        marker = {
            "PASS": "PASS",
            "FAIL": "FAIL",
            "MEASURED": "----",
        }[row.status]

        print(f"  {marker}  {row.test_id:<5} {row.test}")
        print(f"              expected: {row.expected}")
        print(f"              observed: {row.observed}")

        if row.detail:
            print(f"              {row.detail}")


def print_prior_report(side_state, status_counts):

    print()
    print("=" * 79)
    print("PREVIOUS-SEASON PRIOR")
    print("=" * 79)
    print()
    print("  Prior points use Pts_after_sanction. Pts_from_results is carried")
    print("  alongside so the distinction is never lost.")
    print()

    with_prior = side_state[side_state["has_previous_season"]]

    print(f"    team-sides with a prior      : {len(with_prior)}")
    print(f"    team-sides without a prior   : {len(side_state) - len(with_prior)}")

    for status, count in sorted(status_counts.items()):
        print(f"        {status:<28}{count}")

    print()

    sanctioned = with_prior[
        with_prior["previous_season_points"]
        != with_prior["previous_season_points_from_results"]
    ]

    if sanctioned.empty:
        print("    No sanctioned prior in use.")
        return

    print("    Sanctioned priors in use:")
    print()
    print(
        f"      {'Consuming':<12}{'Team':<18}{'Prior from':<12}"
        f"{'Raw':>5}{'Used':>6}"
    )

    seen = set()

    for row in sanctioned.itertuples():

        key = (row.season, row.team)

        if key in seen:
            continue

        seen.add(key)

        print(
            f"      {row.season:<12}{row.team:<18}"
            f"{row.previous_season_source:<12}"
            f"{int(row.previous_season_points_from_results):>5}"
            f"{int(row.previous_season_points):>6}"
        )


def print_promoted_report(side_state):

    print()
    print("=" * 79)
    print("TEAMS WITHOUT A PREVIOUS-SEASON PRIOR")
    print("=" * 79)
    print()
    print("  Explicitly marked, never imputed.")
    print()

    without = side_state[~side_state["has_previous_season"]]

    grouped = (
        without[["season", "team", "previous_season_status"]]
        .drop_duplicates()
        .sort_values(["season", "team"])
    )

    for season, group in grouped.groupby("season"):

        status = ", ".join(sorted(set(group["previous_season_status"])))

        teams = ", ".join(group["team"])

        print(f"    {season}  ({status})")
        print(f"      {teams}")


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 2: HISTORICAL TEAM STATE")
    print("=" * 79)
    print()
    print(f"  Source     : {MATCHES_INPUT.relative_to(PROJECT_ROOT)}")
    print("  Rule       : historical_date < current_date  (STRICT)")
    print("  Mechanism  : np.searchsorted(dates, T, side='left')")
    print("  Not used   : row position, kickoff time, FBref aggregates")
    print("  Scope      : no models, no Elo, no imputation, no dropping")

    for required in (MATCHES_INPUT, SUMMARY_INPUT):

        if not required.exists():
            print()
            print(f"  FAIL - missing input: {required}")
            print()
            print("PHASE 1 - INSTRUMENT 2")
            print()
            print("STATUS: FAIL")
            return 1

    matches = load_matches()

    if len(matches) != EXPECTED_TOTAL_MATCHES:
        print()
        print(f"  FAIL - expected {EXPECTED_TOTAL_MATCHES} matches, "
              f"found {len(matches)}")
        print()
        print("STATUS: FAIL")
        return 1

    audit = Audit()

    print()
    print("  Building historical state ...")

    output, side_state, previous_season_lookup = build_everything(matches)

    print(f"  Built {len(side_state)} team-side states "
          f"across {len(output)} matches.")

    # ---- structure
    validate_structure(matches, output, audit)

    # ---- T1
    print("  T1  strict boundary (independent brute force) ...")
    same_day_contributions = validate_t1_strict_boundary(matches, side_state, audit)

    # ---- T3 (needs T1's measurement)
    print("  T3  same-day exclusion ...")
    non_strict_changed, shared_date_matches = validate_t3_same_day(
        matches, side_state, audit, same_day_contributions
    )

    # ---- T4
    print("  T4  cold starts ...")
    (
        cold_sides, cold_matches, both_cold, one_cold, cold_weeks
    ) = validate_t4_cold_starts(matches, side_state, audit)

    # ---- T5
    print("  T5  history bounds ...")
    bound_violations = validate_t5_history_bounds(matches, side_state, audit)

    # ---- T6
    print("  T6  previous match ...")
    previous_violations = validate_t6_previous_match(matches, side_state, audit)

    # ---- T7
    print("  T7  last-5 window ...")
    last5_violations, short_windows = validate_t7_last_five(side_state, audit)

    # ---- T8
    print("  T8  venue history ...")
    venue_violations = validate_t8_venue(matches, side_state, audit)

    # ---- T9
    print("  T9  previous-season prior ...")
    prior_violations = validate_t9_previous_season(
        matches, side_state, previous_season_lookup, audit
    )

    # ---- T10
    print("  T10 promoted teams ...")
    promoted_violations, status_counts = validate_t10_promoted(
        matches, side_state, audit
    )

    # ---- T2 last: it rebuilds the world 190 times
    print("  T2  perturbation - rewriting every match to 9-0 ...")
    (
        perturbation_changes,
        perturbed_matches,
        rebuild_count,
    ) = validate_t2_perturbation(matches, output, audit)

    # ---- reports
    print_test_table(audit)
    print_prior_report(side_state, status_counts)
    print_promoted_report(side_state)

    # ---- outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    written = output.copy()

    written["date"] = written["date"].dt.strftime("%Y-%m-%d")

    for side in ("home", "away"):

        column = f"{side}_previous_match_date"

        written[column] = pd.to_datetime(written[column]).dt.strftime("%Y-%m-%d")

    written = written[OUTPUT_COLUMNS]

    written.to_csv(STATE_OUTPUT, index=False, encoding="utf-8")

    audit_frame = audit.frame()
    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    leakage_violations = same_day_contributions

    passed = audit.all_passed()

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {STATE_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(written)} rows, {len(written.columns)} columns)")
    print(f"  {AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(audit_frame)} entries)")

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 2")
    print("=" * 79)
    print()

    line("Total matches:", f"{len(output)}")
    line("Total team-side states:", f"{len(side_state)}")
    line(
        "Current-season cold starts:",
        f"{cold_sides} sides in {cold_matches} matches "
        f"({both_cold} both, {one_cold} one)",
    )
    line(
        "Same-day exclusions:",
        f"{non_strict_changed} sides would absorb their own match under <=",
    )
    line(
        "Historical leakage violations:",
        f"{leakage_violations}",
        status_text(leakage_violations == 0),
    )
    line(
        "Previous-season prior violations:",
        f"{prior_violations + promoted_violations}",
        status_text(prior_violations + promoted_violations == 0),
    )
    line(
        "Venue-history violations:",
        f"{venue_violations}",
        status_text(venue_violations == 0),
    )
    line(
        "Perturbation violations:",
        f"{perturbation_changes} "
        f"({perturbed_matches} matches, {rebuild_count} rebuilds)",
        status_text(perturbation_changes == 0),
    )

    print()

    other_violations = (
        bound_violations + previous_violations + last5_violations
    )

    line(
        "Other test violations:",
        f"{other_violations}",
        status_text(other_violations == 0),
    )
    line("Matches sharing a date:", f"{shared_date_matches} of {len(output)}")
    line("Short last-5 windows:", f"{short_windows}")

    failures = audit.failures()

    if failures:
        print()
        print("  FAILURES:")

        for failure in failures:
            print(
                f"    {failure['test_id']} {failure['test']}: "
                f"expected {failure['expected']}, got {failure['observed']} "
                f"{failure['detail']}".rstrip()
            )

    total_tests = len([r for r in audit.rows if r["status"] != "MEASURED"])

    print()
    print(f"  Tests run          : {total_tests}")
    print(f"  Tests passed       : {total_tests - len(failures)}")
    print(f"  Tests failed       : {len(failures)}")
    print()
    print(f"STATUS: {status_text(passed)}")
    print()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
