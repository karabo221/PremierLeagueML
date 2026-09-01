"""
===============================================================================
PHASE 1 - INSTRUMENT 3
TEAM STRENGTH FEATURES
===============================================================================

PURPOSE
    Turn the proven historical state into the team-strength features later
    phases will actually model on:

        - pre-match current-season form
        - venue-specific form (home form, away form)
        - previous-season prior
        - home-vs-away relative / difference features

    Raw team-side values are preserved alongside every derived rate and
    difference. A relative feature that cannot be reconstructed from the raw
    columns sitting next to it is not auditable, and this instrument exists
    to be auditable.

INPUTS - exactly two, both validated upstream
    outputs/phase1_matches.csv                 (Instrument 1)
    outputs/phase1_historical_team_state.csv   (Instrument 2)

    No FBref season aggregate is read. data/raw/ is never opened - and that
    is not merely asserted here, it is MEASURED by a runtime audit hook that
    records every file this process opens (test T10).

THE TWO-PATH DESIGN
    Instrument 2 built the raw state with np.searchsorted. This instrument
    rebuilds the same quantities by a deliberately DIFFERENT method - a
    pairwise date-comparison matrix - and requires the two to agree exactly
    before any feature is derived.

        mask[i, j] = date[j] < date[i]

    That matrix IS the strict boundary, written out in full. Row i's history
    is exactly the matches marked True in row i, and a same-day match gives
    False on both sides of the diagonal.

    Two independent implementations agreeing is evidence. One implementation
    checked against itself is not. If they disagree the run FAILS - it does
    not pick a winner, average them, or repair the difference.

EMPTY STATE IS NaN, NEVER ZERO
    A rate with no matches behind it is UNDEFINED, not zero. Zero points per
    match is a claim about a bad team; an absent value is a claim about
    absent information. Conflating them is how a cold start silently becomes
    a prediction that the team is terrible.

    The same applies to differences: a home-minus-away difference where
    either side is unavailable is NaN, never 0. A zero difference asserts the
    teams are evenly matched, which is exactly what is not known.

WHAT IS NOT DONE HERE
    no models, no training, no Elo, no FBref aggregates, no imputation of
    promoted-team priors, no silent repair. Every failure stops the run.
===============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_match_foundation import SANCTION_REGISTRY


# ============================================================
# FILE-ACCESS RECORDER  (evidence for T10)
# ============================================================
#
# A runtime audit hook records every path this process opens. T10 then checks
# the record rather than taking the script's word for it. Installed before any
# data is read; audit hooks cannot be uninstalled, which is the point.

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
STATE_INPUT = OUTPUTS_DIR / "phase1_historical_team_state.csv"

FEATURES_OUTPUT = OUTPUTS_DIR / "phase1_team_strength_features.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase1_team_strength_audit.csv"
INVENTORY_OUTPUT = OUTPUTS_DIR / "phase1_team_strength_feature_inventory.csv"

DECLARED_INPUTS = {MATCHES_INPUT.resolve(), STATE_INPUT.resolve()}

RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

EXPECTED_TOTAL_MATCHES = 1900
EXPECTED_TEAM_SIDES = 3800

LAST_N = 5
FULL_SEASON_MATCHES = 38

PERTURBED_HOME_GOALS = 9
PERTURBED_AWAY_GOALS = 0

# Teams whose reconstruction is printed in full for human inspection (T9).
RECONSTRUCTION_SPOT_CHECKS = [
    ("2021-2022", "Manchester City"),
    ("2023-2024", "Everton"),
    ("2024-2025", "Nottingham"),
    ("2025-2026", "Sunderland"),
]

RECONSTRUCTION_ROWS_SHOWN = 8


# ---- raw per-side columns, rebuilt and cross-checked
RAW_SIDE_COLUMNS = [
    "mp_before",
    "pts_before",
    "gf_before",
    "ga_before",
    "gd_before",
    "last5_pts_before",
    "last5_mp_before",
    "prev_match_pts_before",

    "venue_mp_before",
    "venue_pts_before",
    "venue_gf_before",
    "venue_ga_before",
    "venue_gd_before",

    "prev_season_available",
    "prev_season_pts",
    "prev_season_pts_raw",
    "prev_season_sanction",
    "prev_season_gf",
    "prev_season_ga",
    "prev_season_gd",
    "prev_season_mp",
    "prev_season_source",
    "prev_season_status",
]

# ---- derived per-side rates
RATE_SIDE_COLUMNS = [
    "ppm_before",
    "gfpm_before",
    "gapm_before",
    "gdpm_before",
    "last5_ppm_before",
    "form_delta_ppm",

    "venue_ppm_before",
    "venue_gfpm_before",
    "venue_gapm_before",
    "venue_gdpm_before",

    "prev_season_ppm",
    "prev_season_gdpm",
]

# ---- home-minus-away relative features
RELATIVE_COLUMNS = [
    "rel_ppm_diff",
    "rel_gfpm_diff",
    "rel_gapm_diff",
    "rel_gdpm_diff",
    "rel_last5_ppm_diff",
    "rel_prev_match_pts_diff",
    "rel_mp_diff",

    "rel_venue_ppm_diff",
    "rel_venue_gfpm_diff",
    "rel_venue_gapm_diff",
    "rel_venue_gdpm_diff",

    "rel_prev_season_ppm_diff",
    "rel_prev_season_gdpm_diff",

    "rel_form_available",
    "rel_venue_form_available",
    "rel_prev_season_available",
]

IDENTITY_COLUMNS = ["season", "date", "matchweek", "home_team", "away_team"]


# Mapping from Instrument 2's column names to this instrument's raw names.
STATE_COLUMN_MAP = {
    "matches_before": "mp_before",
    "points_before": "pts_before",
    "gf_before": "gf_before",
    "ga_before": "ga_before",
    "gd_before": "gd_before",
    "last5_points_before": "last5_pts_before",
    "last5_matches_used": "last5_mp_before",
    "previous_match_points_before": "prev_match_pts_before",
    "venue_matches_before": "venue_mp_before",
    "venue_points_before": "venue_pts_before",
    "venue_gf_before": "venue_gf_before",
    "venue_ga_before": "venue_ga_before",
    "has_previous_season": "prev_season_available",
    "previous_season_points": "prev_season_pts",
    "previous_season_points_from_results": "prev_season_pts_raw",
    "previous_season_gf": "prev_season_gf",
    "previous_season_ga": "prev_season_ga",
    "previous_season_gd": "prev_season_gd",
    "previous_season_matches": "prev_season_mp",
    "previous_season_status": "prev_season_status",
}

# Columns Instrument 2 also carries, compared directly in T9a.
CROSS_CHECK_COLUMNS = [
    "mp_before", "pts_before", "gf_before", "ga_before", "gd_before",
    "last5_pts_before", "last5_mp_before", "prev_match_pts_before",
    "venue_mp_before", "venue_pts_before", "venue_gf_before", "venue_ga_before",
    "prev_season_available", "prev_season_pts", "prev_season_pts_raw",
    "prev_season_gf", "prev_season_ga", "prev_season_gd", "prev_season_mp",
    "prev_season_status",
]


PREV_SEASON_STATUS_AVAILABLE = "available"
PREV_SEASON_STATUS_NO_PRIOR = "no_prior_season_in_dataset"
PREV_SEASON_STATUS_ABSENT = "absent_from_previous_season"


def output_columns():

    columns = list(IDENTITY_COLUMNS)

    for side in ("home", "away"):
        columns += [f"{side}_{name}" for name in RAW_SIDE_COLUMNS]

    for side in ("home", "away"):
        columns += [f"{side}_{name}" for name in RATE_SIDE_COLUMNS]

    columns += RELATIVE_COLUMNS

    return columns


OUTPUT_COLUMNS = output_columns()

# Columns compared by the perturbation tests: everything except identity.
PERTURBATION_COMPARE_COLUMNS = [
    column for column in OUTPUT_COLUMNS if column not in IDENTITY_COLUMNS
]

VENUE_FEATURE_COLUMNS = {
    side: [
        f"{side}_venue_mp_before", f"{side}_venue_pts_before",
        f"{side}_venue_gf_before", f"{side}_venue_ga_before",
        f"{side}_venue_gd_before", f"{side}_venue_ppm_before",
        f"{side}_venue_gfpm_before", f"{side}_venue_gapm_before",
        f"{side}_venue_gdpm_before",
    ]
    for side in ("home", "away")
}

PREV_SEASON_FEATURE_COLUMNS = [
    f"{side}_{name}"
    for side in ("home", "away")
    for name in [
        "prev_season_available", "prev_season_pts", "prev_season_pts_raw",
        "prev_season_sanction", "prev_season_gf", "prev_season_ga",
        "prev_season_gd", "prev_season_mp", "prev_season_ppm",
        "prev_season_gdpm",
    ]
]


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
# HELPERS
# ============================================================

def safe_divide(numerator, denominator):
    """
    Rate with an explicit empty state.

    A zero denominator yields NaN, never 0. See the module docstring.
    """

    numerator = np.asarray(numerator, dtype="float64")
    denominator = np.asarray(denominator, dtype="float64")

    result = np.full(numerator.shape, np.nan, dtype="float64")

    usable = denominator > 0

    np.divide(numerator, denominator, out=result, where=usable)

    result[~usable] = np.nan

    return result


def frames_match(left, right, columns):
    """Element-wise equality treating NaN as equal to NaN."""

    differs = ~(
        (left[columns] == right[columns])
        | (left[columns].isna() & right[columns].isna())
    )

    return differs


# ============================================================
# INPUT
# ============================================================

def load_matches():

    matches = pd.read_csv(MATCHES_INPUT)

    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")

    matches = matches.sort_values(
        ["season", "date", "home_team", "away_team"]
    ).reset_index(drop=True)

    matches["match_id"] = matches.index

    return matches


def load_instrument2_state():

    state = pd.read_csv(STATE_INPUT)

    state["date"] = pd.to_datetime(state["date"], format="%Y-%m-%d")

    state = state.sort_values(
        ["season", "date", "home_team", "away_team"]
    ).reset_index(drop=True)

    state["match_id"] = state.index

    renamed = state[["match_id"] + IDENTITY_COLUMNS].copy()

    for side in ("home", "away"):
        for source, target in STATE_COLUMN_MAP.items():
            renamed[f"{side}_{target}"] = state[f"{side}_{source}"]

    return renamed


def to_team_sides(matches):

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
# INDEPENDENT RECONSTRUCTION
# ============================================================

def history_mask(dates):
    """
    The strict boundary written out in full.

    mask[i, j] is True when match j is strictly earlier than match i, so row i
    IS match i's admissible history. A same-day pair gives False in both
    directions, and the diagonal is False, so a match can never see itself.

    This is deliberately a different mechanism from Instrument 2's
    searchsorted. Agreement between the two is the evidence.
    """

    return dates[:, None] > dates[None, :]


def reconstruct_side_state(dates, gf, ga, pts):
    """Rebuild one team's within-season history at each of its own matches."""

    mask = history_mask(dates)

    counts = mask.sum(axis=1)

    points_before = mask @ pts
    gf_before = mask @ gf
    ga_before = mask @ ga

    n = len(dates)

    last_n_points = np.full(n, np.nan, dtype="float64")
    last_n_used = np.zeros(n, dtype="int64")
    previous_points = np.full(n, np.nan, dtype="float64")

    for i in range(n):

        eligible = np.flatnonzero(mask[i])

        if eligible.size == 0:
            continue

        # Eligible matches in date order; the tail is the recent form window.
        ordered = eligible[np.argsort(dates[eligible], kind="stable")]

        window = ordered[-LAST_N:]

        last_n_points[i] = pts[window].sum()
        last_n_used[i] = window.size

        previous_points[i] = pts[ordered[-1]]

    return {
        "mp_before": counts.astype("int64"),
        "pts_before": points_before.astype("int64"),
        "gf_before": gf_before.astype("int64"),
        "ga_before": ga_before.astype("int64"),
        "last5_pts_before": last_n_points,
        "last5_mp_before": last_n_used,
        "prev_match_pts_before": previous_points,
    }


def reconstruct_previous_season(matches):
    """Season-end totals, and the mapping from a season to its predecessor."""

    sides = to_team_sides(matches)

    totals = sides.groupby(["season", "team"], as_index=False).agg(
        mp=("match_id", "size"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
        pts_raw=("pts", "sum"),
    )

    totals["gd"] = totals["gf"] - totals["ga"]

    totals["sanction"] = [
        SANCTION_REGISTRY.get((season, team), 0)
        for season, team in zip(totals["season"], totals["team"])
    ]

    totals["pts"] = totals["pts_raw"] + totals["sanction"]

    seasons = sorted(matches["season"].unique())

    previous_of = {
        current: previous for previous, current in zip(seasons, seasons[1:])
    }

    by_key = {(row.season, row.team): row for row in totals.itertuples()}

    teams_by_season = {
        season: set(group["team"]) for season, group in totals.groupby("season")
    }

    lookup = {}

    for season in seasons:

        previous_season = previous_of.get(season)

        for team in teams_by_season[season]:

            if previous_season is None:
                lookup[(season, team)] = {
                    "prev_season_available": False,
                    "prev_season_status": PREV_SEASON_STATUS_NO_PRIOR,
                    "prev_season_source": np.nan,
                }
                continue

            prior = by_key.get((previous_season, team))

            if prior is None:
                lookup[(season, team)] = {
                    "prev_season_available": False,
                    "prev_season_status": PREV_SEASON_STATUS_ABSENT,
                    "prev_season_source": np.nan,
                }
                continue

            lookup[(season, team)] = {
                "prev_season_available": True,
                "prev_season_status": PREV_SEASON_STATUS_AVAILABLE,
                "prev_season_source": previous_season,
                "prev_season_pts": float(prior.pts),
                "prev_season_pts_raw": float(prior.pts_raw),
                "prev_season_sanction": float(prior.sanction),
                "prev_season_gf": float(prior.gf),
                "prev_season_ga": float(prior.ga),
                "prev_season_gd": float(prior.gd),
                "prev_season_mp": float(prior.mp),
            }

    return lookup


def reconstruct_state(matches):
    """Full independent rebuild of the raw pre-match state, per team-side."""

    sides = to_team_sides(matches)

    sides = sides.sort_values(["season", "team", "date"]).reset_index(drop=True)

    previous_season = reconstruct_previous_season(matches)

    frames = []

    for (season, team), group in sides.groupby(["season", "team"], sort=False):

        dates = group["date"].to_numpy()
        gf = group["gf"].to_numpy()
        ga = group["ga"].to_numpy()
        pts = group["pts"].to_numpy()
        side_labels = group["side"].to_numpy()

        overall = reconstruct_side_state(dates, gf, ga, pts)

        block = pd.DataFrame(overall)
        block["match_id"] = group["match_id"].to_numpy()
        block["side"] = side_labels
        block["season"] = season
        block["team"] = team

        # ---- venue state: home form from home matches only, away from away
        for column in ["venue_mp_before", "venue_pts_before",
                       "venue_gf_before", "venue_ga_before"]:
            block[column] = 0

        for venue in ("home", "away"):

            venue_mask = side_labels == venue

            if not venue_mask.any():
                continue

            venue_state = reconstruct_side_state(
                dates[venue_mask],
                gf[venue_mask],
                ga[venue_mask],
                pts[venue_mask],
            )

            positions = np.flatnonzero(venue_mask)

            block.loc[positions, "venue_mp_before"] = venue_state["mp_before"]
            block.loc[positions, "venue_pts_before"] = venue_state["pts_before"]
            block.loc[positions, "venue_gf_before"] = venue_state["gf_before"]
            block.loc[positions, "venue_ga_before"] = venue_state["ga_before"]

        prior = previous_season[(season, team)]

        for key in ["prev_season_available", "prev_season_status",
                    "prev_season_source"]:
            block[key] = prior[key]

        for key in ["prev_season_pts", "prev_season_pts_raw",
                    "prev_season_sanction", "prev_season_gf",
                    "prev_season_ga", "prev_season_gd", "prev_season_mp"]:
            block[key] = prior.get(key, np.nan)

        frames.append(block)

    state = pd.concat(frames, ignore_index=True)

    state["gd_before"] = state["gf_before"] - state["ga_before"]
    state["venue_gd_before"] = state["venue_gf_before"] - state["venue_ga_before"]

    return state


def pivot_to_matches(matches, side_state):
    """Team-side rows back to one row per match."""

    output = matches[["match_id"] + IDENTITY_COLUMNS].copy()

    for side in ("home", "away"):

        rows = side_state[side_state["side"] == side]

        renamed = {name: f"{side}_{name}" for name in RAW_SIDE_COLUMNS}

        output = output.merge(
            rows[["match_id"] + RAW_SIDE_COLUMNS].rename(columns=renamed),
            on="match_id",
            how="left",
        )

    return output


# ============================================================
# FEATURE DERIVATION
# ============================================================

def add_rate_features(features):
    """
    Per-side rates. Every one is NaN when its denominator is zero - see the
    module docstring on empty state.
    """

    for side in ("home", "away"):

        mp = features[f"{side}_mp_before"]

        features[f"{side}_ppm_before"] = safe_divide(features[f"{side}_pts_before"], mp)
        features[f"{side}_gfpm_before"] = safe_divide(features[f"{side}_gf_before"], mp)
        features[f"{side}_gapm_before"] = safe_divide(features[f"{side}_ga_before"], mp)
        features[f"{side}_gdpm_before"] = safe_divide(features[f"{side}_gd_before"], mp)

        features[f"{side}_last5_ppm_before"] = safe_divide(
            features[f"{side}_last5_pts_before"],
            features[f"{side}_last5_mp_before"],
        )

        # Recent form relative to the season so far: positive means the team
        # is running above its own season average.
        features[f"{side}_form_delta_ppm"] = (
            features[f"{side}_last5_ppm_before"] - features[f"{side}_ppm_before"]
        )

        venue_mp = features[f"{side}_venue_mp_before"]

        features[f"{side}_venue_ppm_before"] = safe_divide(
            features[f"{side}_venue_pts_before"], venue_mp)
        features[f"{side}_venue_gfpm_before"] = safe_divide(
            features[f"{side}_venue_gf_before"], venue_mp)
        features[f"{side}_venue_gapm_before"] = safe_divide(
            features[f"{side}_venue_ga_before"], venue_mp)
        features[f"{side}_venue_gdpm_before"] = safe_divide(
            features[f"{side}_venue_gd_before"], venue_mp)

        prev_mp = features[f"{side}_prev_season_mp"].fillna(0)

        features[f"{side}_prev_season_ppm"] = safe_divide(
            features[f"{side}_prev_season_pts"], prev_mp)
        features[f"{side}_prev_season_gdpm"] = safe_divide(
            features[f"{side}_prev_season_gd"], prev_mp)

    return features


def add_relative_features(features):
    """
    Home-minus-away differences.

    A difference is NaN whenever either side is unavailable. Subtracting NaN
    propagates NaN naturally, which is exactly the wanted behaviour - a zero
    would assert the teams are level, which is the one thing not known.
    """

    pairs = [
        ("rel_ppm_diff", "ppm_before"),
        ("rel_gfpm_diff", "gfpm_before"),
        ("rel_gapm_diff", "gapm_before"),
        ("rel_gdpm_diff", "gdpm_before"),
        ("rel_last5_ppm_diff", "last5_ppm_before"),
        ("rel_prev_match_pts_diff", "prev_match_pts_before"),
        ("rel_mp_diff", "mp_before"),
        ("rel_prev_season_ppm_diff", "prev_season_ppm"),
        ("rel_prev_season_gdpm_diff", "prev_season_gdpm"),
    ]

    for target, source in pairs:
        features[target] = features[f"home_{source}"] - features[f"away_{source}"]

    # Venue-relative features deliberately compare the home team's HOME form
    # against the away team's AWAY form. That is the comparison the fixture
    # actually poses.
    venue_pairs = [
        ("rel_venue_ppm_diff", "venue_ppm_before"),
        ("rel_venue_gfpm_diff", "venue_gfpm_before"),
        ("rel_venue_gapm_diff", "venue_gapm_before"),
        ("rel_venue_gdpm_diff", "venue_gdpm_before"),
    ]

    for target, source in venue_pairs:
        features[target] = features[f"home_{source}"] - features[f"away_{source}"]

    features["rel_form_available"] = (
        (features["home_mp_before"] > 0) & (features["away_mp_before"] > 0)
    )

    features["rel_venue_form_available"] = (
        (features["home_venue_mp_before"] > 0)
        & (features["away_venue_mp_before"] > 0)
    )

    features["rel_prev_season_available"] = (
        features["home_prev_season_available"].astype(bool)
        & features["away_prev_season_available"].astype(bool)
    )

    return features


def build_features(matches):
    """Full pipeline: reconstruct raw state, then derive every feature."""

    side_state = reconstruct_state(matches)

    features = pivot_to_matches(matches, side_state)

    features = add_rate_features(features)
    features = add_relative_features(features)

    return features, side_state


# ============================================================
# TESTS
# ============================================================

def test_t1_temporal_integrity(matches, features, audit):
    """
    T1 - every feature value is derivable exclusively from matches before T.

    Checked structurally: the raw counters cannot exceed what was actually
    completed earlier, and every rate must equal the ratio of the raw columns
    sitting beside it. If a rate carried information the raw columns do not,
    that identity would break.
    """

    sides = to_team_sides(matches)

    earlier_counts = {}

    for (season, team), group in sides.groupby(["season", "team"], sort=False):

        dates = list(group["date"])

        for row in group.itertuples():

            earlier = sum(1 for other in dates if other < row.date)

            earlier_counts[(row.match_id, row.side)] = earlier

    violations = 0
    examples = []

    for row in features.itertuples():

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            expected = earlier_counts[(row.match_id, side)]
            observed = getattr(row, f"{side}_mp_before")

            if observed != expected:
                violations += 1

                if len(examples) < 5:
                    examples.append(
                        f"{row.season} {team} {row.date.date()}: "
                        f"{observed} != {expected}"
                    )

    audit.record(
        "T1a",
        "mp_before equals the count of that team's strictly earlier matches",
        f"0 violations across {len(features) * 2} team-sides",
        f"{violations} violations",
        violations == 0,
        "; ".join(examples),
    )

    # Every rate must be exactly reproducible from its raw numerator and
    # denominator - no rate may carry information the raw columns do not.
    identity_breaks = []

    rate_identities = [
        ("ppm_before", "pts_before", "mp_before"),
        ("gfpm_before", "gf_before", "mp_before"),
        ("gapm_before", "ga_before", "mp_before"),
        ("gdpm_before", "gd_before", "mp_before"),
        ("last5_ppm_before", "last5_pts_before", "last5_mp_before"),
        ("venue_ppm_before", "venue_pts_before", "venue_mp_before"),
        ("venue_gfpm_before", "venue_gf_before", "venue_mp_before"),
        ("venue_gapm_before", "venue_ga_before", "venue_mp_before"),
        ("venue_gdpm_before", "venue_gd_before", "venue_mp_before"),
        ("prev_season_ppm", "prev_season_pts", "prev_season_mp"),
        ("prev_season_gdpm", "prev_season_gd", "prev_season_mp"),
    ]

    for side in ("home", "away"):
        for rate, numerator, denominator in rate_identities:

            expected = safe_divide(
                features[f"{side}_{numerator}"].fillna(0),
                features[f"{side}_{denominator}"].fillna(0),
            )

            actual = features[f"{side}_{rate}"].to_numpy(dtype="float64")

            differs = ~(
                np.isclose(expected, actual, equal_nan=True)
                | (np.isnan(expected) & np.isnan(actual))
            )

            if differs.any():
                identity_breaks.append(f"{side}_{rate}: {int(differs.sum())} rows")

    audit.record(
        "T1b",
        "Every rate equals numerator/denominator of its own raw columns",
        "0 broken identities",
        f"{len(identity_breaks)} broken identities",
        not identity_breaks,
        "; ".join(identity_breaks[:5]),
    )

    # Differences must be exactly home minus away, and NaN where either is.
    diff_breaks = []

    diff_identities = [
        ("rel_ppm_diff", "ppm_before"),
        ("rel_gfpm_diff", "gfpm_before"),
        ("rel_gapm_diff", "gapm_before"),
        ("rel_gdpm_diff", "gdpm_before"),
        ("rel_last5_ppm_diff", "last5_ppm_before"),
        ("rel_prev_match_pts_diff", "prev_match_pts_before"),
        ("rel_mp_diff", "mp_before"),
        ("rel_venue_ppm_diff", "venue_ppm_before"),
        ("rel_venue_gfpm_diff", "venue_gfpm_before"),
        ("rel_venue_gapm_diff", "venue_gapm_before"),
        ("rel_venue_gdpm_diff", "venue_gdpm_before"),
        ("rel_prev_season_ppm_diff", "prev_season_ppm"),
        ("rel_prev_season_gdpm_diff", "prev_season_gdpm"),
    ]

    for target, source in diff_identities:

        expected = (
            features[f"home_{source}"].to_numpy(dtype="float64")
            - features[f"away_{source}"].to_numpy(dtype="float64")
        )

        actual = features[target].to_numpy(dtype="float64")

        differs = ~np.isclose(expected, actual, equal_nan=True)

        if differs.any():
            diff_breaks.append(f"{target}: {int(differs.sum())} rows")

    audit.record(
        "T1c",
        "Every relative feature equals home minus away exactly",
        "0 broken identities",
        f"{len(diff_breaks)} broken identities",
        not diff_breaks,
        "; ".join(diff_breaks[:5]),
    )


def test_t2_current_season_isolation(matches, features, audit):
    """T2 - current-season state never carries across a season boundary."""

    sides = to_team_sides(matches)

    first_dates = (
        sides.groupby(["season", "team"], as_index=False)["date"].min()
        .rename(columns={"date": "first_date"})
    )

    first_lookup = {
        (row.season, row.team): row.first_date
        for row in first_dates.itertuples()
    }

    carried = 0
    examples = []

    for row in features.itertuples():

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            if row.date != first_lookup[(row.season, team)]:
                continue

            # A team's opening match of a season must be blank even when it
            # played 38 matches the season before.
            values = [
                getattr(row, f"{side}_mp_before"),
                getattr(row, f"{side}_pts_before"),
                getattr(row, f"{side}_gf_before"),
                getattr(row, f"{side}_ga_before"),
                getattr(row, f"{side}_venue_mp_before"),
                getattr(row, f"{side}_venue_pts_before"),
            ]

            if any(value != 0 for value in values):
                carried += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}")

    audit.record(
        "T2a",
        "Current-season state does not carry across a season boundary",
        0, carried,
        carried == 0,
        "; ".join(examples),
    )

    # A returning team is the sharp case: it has PL history, but not in the
    # season it just rejoined.
    seasons = sorted(matches["season"].unique())

    returning = []

    teams_by_season = {
        season: set(group["home_team"]) | set(group["away_team"])
        for season, group in matches.groupby("season")
    }

    for index, season in enumerate(seasons):

        if index < 2:
            continue

        previous = teams_by_season[seasons[index - 1]]

        for team in teams_by_season[season]:

            if team in previous:
                continue

            if any(team in teams_by_season[s] for s in seasons[:index - 1]):
                returning.append((season, team))

    audit.measure(
        "T2b",
        "Returning teams (played in the PL before, but not last season)",
        len(returning),
        "; ".join(f"{season} {team}" for season, team in sorted(returning)),
    )


def test_t3_venue_isolation(matches, features, audit):
    """T3 - home form uses only home matches, away form only away matches."""

    sides = to_team_sides(matches)

    by_team_venue = {
        key: group.sort_values("date")
        for key, group in sides.groupby(["season", "team", "side"], sort=False)
    }

    violations = 0
    examples = []

    for row in features.itertuples():

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            group = by_team_venue[(row.season, team, side)]

            earlier = group[group["date"] < row.date]

            expected = (
                len(earlier),
                earlier["pts"].sum(),
                earlier["gf"].sum(),
                earlier["ga"].sum(),
            )

            observed = (
                getattr(row, f"{side}_venue_mp_before"),
                getattr(row, f"{side}_venue_pts_before"),
                getattr(row, f"{side}_venue_gf_before"),
                getattr(row, f"{side}_venue_ga_before"),
            )

            if expected != observed:
                violations += 1

                if len(examples) < 5:
                    examples.append(
                        f"{row.season} {team} {side}: {observed} != {expected}"
                    )

    audit.record(
        "T3a",
        "Venue form rebuilt only from that team's matches at that venue",
        f"0 violations across {len(features) * 2} team-sides",
        f"{violations} violations",
        violations == 0,
        "; ".join(examples),
    )

    # Structural containment: venue state is a strict subset of overall state.
    exceeds = 0

    for side in ("home", "away"):
        exceeds += int((
            (features[f"{side}_venue_mp_before"] > features[f"{side}_mp_before"])
            | (features[f"{side}_venue_pts_before"] > features[f"{side}_pts_before"])
            | (features[f"{side}_venue_gf_before"] > features[f"{side}_gf_before"])
            | (features[f"{side}_venue_ga_before"] > features[f"{side}_ga_before"])
            | (features[f"{side}_venue_mp_before"] > 19)
        ).sum())

    audit.record(
        "T3b",
        "Venue state never exceeds overall state, and never exceeds 19 matches",
        0, exceeds,
        exceeds == 0,
    )


def test_t4_previous_season_isolation(matches, features, audit):
    """T4 - previous-season values come only from the completed prior season."""

    seasons = sorted(matches["season"].unique())

    previous_of = {
        current: previous for previous, current in zip(seasons, seasons[1:])
    }

    sides = to_team_sides(matches)

    totals = sides.groupby(["season", "team"], as_index=False).agg(
        mp=("match_id", "size"),
        gf=("gf", "sum"),
        ga=("ga", "sum"),
        pts_raw=("pts", "sum"),
    )

    totals["gd"] = totals["gf"] - totals["ga"]

    reference = {(row.season, row.team): row for row in totals.itertuples()}

    violations = 0
    incomplete = 0
    examples = []

    for row in features.itertuples():

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            if not getattr(row, f"{side}_prev_season_available"):
                continue

            source = getattr(row, f"{side}_prev_season_source")

            if source != previous_of.get(row.season):
                violations += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: source {source!r}")

                continue

            prior = reference[(source, team)]

            sanction = SANCTION_REGISTRY.get((source, team), 0)

            expected = (
                float(prior.pts_raw + sanction),
                float(prior.pts_raw),
                float(prior.gf),
                float(prior.ga),
                float(prior.gd),
                float(prior.mp),
            )

            observed = tuple(
                float(getattr(row, f"{side}_prev_season_{name}"))
                for name in ["pts", "pts_raw", "gf", "ga", "gd", "mp"]
            )

            if expected != observed:
                violations += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: {observed} != {expected}")

            if prior.mp != FULL_SEASON_MATCHES:
                incomplete += 1

    audit.record(
        "T4a",
        "Previous-season values come only from the completed season N-1",
        0, violations,
        violations == 0,
        "; ".join(examples),
    )

    audit.record(
        "T4b",
        "Every previous-season prior covers a complete 38-match season",
        0, incomplete,
        incomplete == 0,
    )


def test_t5_promoted_teams(matches, features, audit):
    """T5 - a team with no prior stays explicitly unavailable, never fabricated."""

    seasons = sorted(matches["season"].unique())

    previous_of = {
        current: previous for previous, current in zip(seasons, seasons[1:])
    }

    teams_by_season = {
        season: set(group["home_team"]) | set(group["away_team"])
        for season, group in matches.groupby("season")
    }

    value_columns = [
        "prev_season_pts", "prev_season_pts_raw", "prev_season_sanction",
        "prev_season_gf", "prev_season_ga", "prev_season_gd",
        "prev_season_mp", "prev_season_ppm", "prev_season_gdpm",
    ]

    wrong_flag = 0
    fabricated = 0
    examples = []

    for row in features.itertuples():

        previous_season = previous_of.get(row.season)

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            expected_available = (
                previous_season is not None
                and team in teams_by_season[previous_season]
            )

            observed_available = bool(getattr(row, f"{side}_prev_season_available"))

            if observed_available != expected_available:
                wrong_flag += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: flag")

            if expected_available:
                continue

            present = [
                name for name in value_columns
                if pd.notna(getattr(row, f"{side}_{name}"))
            ]

            if present:
                fabricated += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: {present}")

    audit.record(
        "T5a",
        "prev_season_available reflects actual presence in season N-1",
        0, wrong_flag,
        wrong_flag == 0,
        "; ".join(examples[:5]),
    )

    audit.record(
        "T5b",
        "Teams without a prior carry NaN in every previous-season column",
        0, fabricated,
        fabricated == 0,
        "Nothing imputed, no zero substituted for absent information",
    )

    # The relative feature must also refuse to invent a comparison.
    unavailable = features[~features["rel_prev_season_available"]]

    leaked = int(
        unavailable["rel_prev_season_ppm_diff"].notna().sum()
        + unavailable["rel_prev_season_gdpm_diff"].notna().sum()
    )

    audit.record(
        "T5c",
        "Previous-season difference is NaN when either side lacks a prior",
        0, leaked,
        leaked == 0,
        "A zero difference would assert the teams are level - it is not known",
    )

    without = int(
        (~features["home_prev_season_available"].astype(bool)).sum()
        + (~features["away_prev_season_available"].astype(bool)).sum()
    )

    audit.measure(
        "T5d",
        "Team-sides carrying no previous-season prior",
        without,
    )


def test_t6_sanctions(features, audit):
    """T6 - the prior uses sanctioned points; raw points stay available."""

    problems = []
    verified = []

    for (season, team), value in sorted(SANCTION_REGISTRY.items()):

        consumers = features[
            (
                (features["home_team"] == team)
                & (features["home_prev_season_source"] == season)
            )
        ]

        if consumers.empty:
            problems.append(f"{season} {team}: no consuming season found")
            continue

        used = set(consumers["home_prev_season_pts"].dropna())
        raw = set(consumers["home_prev_season_pts_raw"].dropna())
        sanction = set(consumers["home_prev_season_sanction"].dropna())

        if len(used) != 1 or len(raw) != 1 or len(sanction) != 1:
            problems.append(f"{season} {team}: inconsistent prior across matches")
            continue

        used_value = used.pop()
        raw_value = raw.pop()
        sanction_value = sanction.pop()

        if used_value != raw_value + value or sanction_value != value:
            problems.append(
                f"{season} {team}: used {used_value}, raw {raw_value}, "
                f"sanction {sanction_value}, expected {value}"
            )
            continue

        # The rate must be built from the sanctioned figure, not the raw one.
        expected_ppm = used_value / FULL_SEASON_MATCHES

        observed_ppm = set(consumers["home_prev_season_ppm"].dropna())

        if len(observed_ppm) != 1 or not np.isclose(observed_ppm.pop(), expected_ppm):
            problems.append(f"{season} {team}: ppm not built from sanctioned points")
            continue

        verified.append(
            f"{season} {team}: raw {raw_value:.0f} {value:+.0f} "
            f"-> {used_value:.0f} used"
        )

    audit.record(
        "T6a",
        "Prior points use the sanctioned value, raw preserved alongside",
        f"0 problems across {len(SANCTION_REGISTRY)} sanctions",
        f"{len(problems)} problems",
        not problems,
        "; ".join(problems) if problems else "; ".join(verified),
    )

    # Everywhere else the two must be identical - a sanction must not leak
    # into a team that never received one.
    with_prior = features[features["home_prev_season_available"].astype(bool)]

    unsanctioned = with_prior[with_prior["home_prev_season_sanction"] == 0]

    drifted = int(
        (
            unsanctioned["home_prev_season_pts"]
            != unsanctioned["home_prev_season_pts_raw"]
        ).sum()
    )

    audit.record(
        "T6b",
        "Unsanctioned priors have identical used and raw points",
        0, drifted,
        drifted == 0,
    )


def test_t7_cold_starts(matches, features, audit):
    """T7 - the opening match of a team-season is genuinely cold."""

    sides = to_team_sides(matches)

    first_lookup = {
        (row.season, row.team): row.date
        for row in (
            sides.groupby(["season", "team"], as_index=False)["date"].min()
        ).itertuples()
    }

    violations = 0
    cold_sides = 0
    rate_violations = 0
    examples = []

    for row in features.itertuples():

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            if row.date != first_lookup[(row.season, team)]:
                continue

            cold_sides += 1

            counters = {
                "mp_before": getattr(row, f"{side}_mp_before"),
                "pts_before": getattr(row, f"{side}_pts_before"),
                "gf_before": getattr(row, f"{side}_gf_before"),
                "ga_before": getattr(row, f"{side}_ga_before"),
                "gd_before": getattr(row, f"{side}_gd_before"),
                "venue_mp_before": getattr(row, f"{side}_venue_mp_before"),
                "venue_pts_before": getattr(row, f"{side}_venue_pts_before"),
                "last5_mp_before": getattr(row, f"{side}_last5_mp_before"),
            }

            bad = {name: value for name, value in counters.items() if value != 0}

            if bad:
                violations += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: {bad}")

            # Empty state, explicitly: counters are 0, rates are NaN.
            rates = [
                "ppm_before", "gfpm_before", "gapm_before", "gdpm_before",
                "last5_ppm_before", "venue_ppm_before", "venue_gdpm_before",
                "form_delta_ppm",
            ]

            defined = [
                name for name in rates
                if pd.notna(getattr(row, f"{side}_{name}"))
            ]

            if defined or pd.notna(getattr(row, f"{side}_last5_pts_before")):
                rate_violations += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: rates {defined}")

    audit.record(
        "T7a",
        "Opening match of a team-season has all counters at zero",
        f"0 violations across {cold_sides} cold-start sides",
        f"{violations} violations",
        violations == 0,
        "; ".join(examples[:5]),
    )

    audit.record(
        "T7b",
        "Cold-start rates are the explicit empty state (NaN), never zero",
        0, rate_violations,
        rate_violations == 0,
        "Zero points per match would assert a bad team, not absent information",
    )

    # And the cold start must not quietly acquire a previous-season prior.
    promoted_cold = 0

    seasons = sorted(matches["season"].unique())

    previous_of = {
        current: previous for previous, current in zip(seasons, seasons[1:])
    }

    teams_by_season = {
        season: set(group["home_team"]) | set(group["away_team"])
        for season, group in matches.groupby("season")
    }

    for row in features.itertuples():

        previous_season = previous_of.get(row.season)

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            if row.date != first_lookup[(row.season, team)]:
                continue

            has_real_prior = (
                previous_season is not None
                and team in teams_by_season[previous_season]
            )

            if has_real_prior:
                continue

            if pd.notna(getattr(row, f"{side}_prev_season_pts")):
                promoted_cold += 1

    audit.record(
        "T7c",
        "No previous-season value appears for a promoted team at its cold start",
        0, promoted_cold,
        promoted_cold == 0,
    )

    audit.measure("T7d", "Cold-start team-sides measured", cold_sides)


def test_t8_perturbation(matches, baseline, audit):
    """
    T8 - a FUTURE result cannot reach an EARLIER feature row.

    Every (season, matchweek) group is rewritten to 9-0 and the whole feature
    layer rebuilt: 190 rebuilds covering all 1,900 matches. Three separate
    claims are checked on each rebuild.

        (a) temporal   - a row whose two teams have no perturbed match
                         strictly earlier must be byte-identical
        (b) venue      - a row's HOME venue features must be unchanged when
                         the home team's perturbed matches were all AWAY
                         matches (and symmetrically), which is venue
                         isolation stated causally rather than structurally
        (c) prior      - previous-season features within the perturbed season
                         must never move, since the prior is drawn from an
                         untouched earlier season

    Groups are perturbed one season at a time. Perturbing season S legitimately
    changes season S+1's prior, so mixing seasons would flag correct behaviour
    as a violation.
    """

    baseline = baseline.set_index("match_id")

    sides = to_team_sides(matches)

    side_lookup = {}

    for row in sides.itertuples():
        side_lookup.setdefault((row.season, row.team), []).append(
            (row.date, row.side, row.match_id)
        )

    temporal_violations = 0
    venue_violations = 0
    prior_violations = 0

    rows_checked = 0
    venue_rows_checked = 0
    rebuilds = 0
    perturbed_total = 0

    examples = []

    for (season, matchweek), group in matches.groupby(["season", "matchweek"]):

        target_ids = set(group["match_id"])

        perturbed = matches.copy()

        mask = perturbed["match_id"].isin(target_ids)

        perturbed.loc[mask, "home_goals"] = PERTURBED_HOME_GOALS
        perturbed.loc[mask, "away_goals"] = PERTURBED_AWAY_GOALS
        perturbed.loc[mask, "result"] = "H"
        perturbed.loc[mask, "home_points_from_result"] = 3
        perturbed.loc[mask, "away_points_from_result"] = 0

        rebuilt, _ = build_features(perturbed)

        rebuilds += 1
        perturbed_total += len(target_ids)

        rebuilt = rebuilt.set_index("match_id")

        # ---- which perturbed matches sit before each team's own match
        perturbed_by_team = {}

        for row in group.itertuples():
            for side, team in (("home", row.home_team), ("away", row.away_team)):
                perturbed_by_team.setdefault((season, team), []).append(
                    (row.date, side)
                )

        season_rows = matches[matches["season"] == season]

        unaffected_ids = []
        venue_safe = {"home": [], "away": []}

        for row in season_rows.itertuples():

            touched = False

            for side, team in (("home", row.home_team), ("away", row.away_team)):

                earlier_perturbed = [
                    entry for entry in perturbed_by_team.get((season, team), [])
                    if entry[0] < row.date
                ]

                if earlier_perturbed:
                    touched = True

                # Venue claim: this side's venue features depend only on this
                # team's earlier matches AT THIS VENUE.
                earlier_same_venue = [
                    entry for entry in earlier_perturbed if entry[1] == side
                ]

                if not earlier_same_venue:
                    venue_safe[side].append(row.match_id)

            if not touched:
                unaffected_ids.append(row.match_id)

        # ---- (a) temporal
        if unaffected_ids:

            before = baseline.loc[unaffected_ids, PERTURBATION_COMPARE_COLUMNS]
            after = rebuilt.loc[unaffected_ids, PERTURBATION_COMPARE_COLUMNS]

            differs = frames_match(before, after, PERTURBATION_COMPARE_COLUMNS)

            changed = int(differs.to_numpy().sum())

            temporal_violations += changed
            rows_checked += len(unaffected_ids)

            if changed and len(examples) < 5:
                examples.append(
                    f"{season} MW{matchweek} temporal: "
                    f"{list(differs.columns[differs.any(axis=0)])}"
                )

        # ---- (b) venue isolation
        for side in ("home", "away"):

            safe_ids = venue_safe[side]

            if not safe_ids:
                continue

            columns = VENUE_FEATURE_COLUMNS[side]

            before = baseline.loc[safe_ids, columns]
            after = rebuilt.loc[safe_ids, columns]

            differs = frames_match(before, after, columns)

            changed = int(differs.to_numpy().sum())

            venue_violations += changed
            venue_rows_checked += len(safe_ids)

            if changed and len(examples) < 5:
                examples.append(
                    f"{season} MW{matchweek} venue {side}: "
                    f"{list(differs.columns[differs.any(axis=0)])}"
                )

        # ---- (c) previous-season prior within the perturbed season
        season_ids = list(season_rows["match_id"])

        before = baseline.loc[season_ids, PREV_SEASON_FEATURE_COLUMNS]
        after = rebuilt.loc[season_ids, PREV_SEASON_FEATURE_COLUMNS]

        differs = frames_match(before, after, PREV_SEASON_FEATURE_COLUMNS)

        changed = int(differs.to_numpy().sum())

        prior_violations += changed

        if changed and len(examples) < 5:
            examples.append(
                f"{season} MW{matchweek} prior: "
                f"{list(differs.columns[differs.any(axis=0)])}"
            )

    audit.record(
        "T8a",
        "A future result never changes an earlier feature row",
        "0 changed values",
        f"{temporal_violations} changed values",
        temporal_violations == 0,
        f"{rows_checked} unaffected rows compared across {rebuilds} rebuilds "
        f"({perturbed_total} matches perturbed); " + "; ".join(examples[:3]),
    )

    audit.record(
        "T8b",
        "Perturbing away matches never moves home venue form (and vice versa)",
        "0 changed values",
        f"{venue_violations} changed values",
        venue_violations == 0,
        f"{venue_rows_checked} venue-safe rows compared",
    )

    audit.record(
        "T8c",
        "Perturbing a season never moves that season's previous-season prior",
        "0 changed values",
        f"{prior_violations} changed values",
        prior_violations == 0,
    )

    return temporal_violations, venue_violations, prior_violations, rebuilds


def test_t9_reconstruction(matches, features, instrument2, audit):
    """
    T9 - independent reconstruction agrees with the generated feature state.

    Two comparisons:
        a) against Instrument 2's separately built and separately validated
           state file (searchsorted vs pairwise mask)
        b) against a from-scratch per-team walk written a third way, for a
           set of named team-seasons printed in full for human inspection
    """

    merged = features.merge(
        instrument2,
        on="match_id",
        suffixes=("", "_i2"),
    )

    mismatched_columns = []

    for side in ("home", "away"):
        for name in CROSS_CHECK_COLUMNS:

            column = f"{side}_{name}"

            left = merged[column]
            right = merged[f"{column}_i2"]

            if left.dtype == bool or right.dtype == bool:
                differs = left.astype(bool) != right.astype(bool)
            else:
                differs = ~((left == right) | (left.isna() & right.isna()))

            if differs.any():
                mismatched_columns.append(f"{column}: {int(differs.sum())} rows")

    audit.record(
        "T9a",
        "Pairwise-mask reconstruction agrees with Instrument 2's searchsorted state",
        "0 mismatched columns",
        f"{len(mismatched_columns)} mismatched columns",
        not mismatched_columns,
        "; ".join(mismatched_columns[:5]),
    )

    # ---- third implementation: a plain running walk over sorted matches
    sides = to_team_sides(matches)

    walk_violations = 0
    examples = []
    spot_check_tables = {}

    for (season, team), group in sides.groupby(["season", "team"], sort=False):

        ordered = group.sort_values("date")

        running = {
            "mp": 0, "pts": 0, "gf": 0, "ga": 0,
            "home_mp": 0, "home_pts": 0, "away_mp": 0, "away_pts": 0,
        }

        recent = []

        rows = []

        for entry in ordered.itertuples():

            # State BEFORE this match is whatever the walk has accumulated.
            expected = {
                "mp_before": running["mp"],
                "pts_before": running["pts"],
                "gf_before": running["gf"],
                "ga_before": running["ga"],
                "venue_mp_before": running[f"{entry.side}_mp"],
                "venue_pts_before": running[f"{entry.side}_pts"],
                "last5_pts_before": sum(recent[-LAST_N:]) if recent else None,
            }

            feature_row = features[features["match_id"] == entry.match_id].iloc[0]

            observed = {
                "mp_before": feature_row[f"{entry.side}_mp_before"],
                "pts_before": feature_row[f"{entry.side}_pts_before"],
                "gf_before": feature_row[f"{entry.side}_gf_before"],
                "ga_before": feature_row[f"{entry.side}_ga_before"],
                "venue_mp_before": feature_row[f"{entry.side}_venue_mp_before"],
                "venue_pts_before": feature_row[f"{entry.side}_venue_pts_before"],
                "last5_pts_before": (
                    feature_row[f"{entry.side}_last5_pts_before"]
                    if pd.notna(feature_row[f"{entry.side}_last5_pts_before"])
                    else None
                ),
            }

            if expected != observed:
                walk_violations += 1

                if len(examples) < 5:
                    examples.append(
                        f"{season} {team} {entry.date.date()}: "
                        f"{observed} != {expected}"
                    )

            if (season, team) in RECONSTRUCTION_SPOT_CHECKS:
                rows.append({
                    "date": entry.date,
                    "side": entry.side,
                    "mp": expected["mp_before"],
                    "pts": expected["pts_before"],
                    "venue_mp": expected["venue_mp_before"],
                    "venue_pts": expected["venue_pts_before"],
                    "ppm": feature_row[f"{entry.side}_ppm_before"],
                    "venue_ppm": feature_row[f"{entry.side}_venue_ppm_before"],
                    "agrees": expected == observed,
                })

            # Now fold this match in, for the NEXT iteration only.
            running["mp"] += 1
            running["pts"] += entry.pts
            running["gf"] += entry.gf
            running["ga"] += entry.ga
            running[f"{entry.side}_mp"] += 1
            running[f"{entry.side}_pts"] += entry.pts

            recent.append(entry.pts)

        if (season, team) in RECONSTRUCTION_SPOT_CHECKS:
            spot_check_tables[(season, team)] = pd.DataFrame(rows)

    audit.record(
        "T9b",
        "A third, running-walk reconstruction agrees with every feature row",
        f"0 violations across {len(sides)} team-sides",
        f"{walk_violations} violations",
        walk_violations == 0,
        "; ".join(examples),
    )

    return spot_check_tables


def test_t10_no_fbref(audit):
    """
    T10 - the instrument operates from the validated foundation only.

    Evidence, not assertion: every path opened by this process was recorded by
    a runtime audit hook. Data files opened must be exactly the two declared
    inputs, and nothing under data/raw/ may appear at all.
    """

    opened = []

    for path in _OPENED_PATHS:

        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            continue

        opened.append(resolved)

    raw_touches = [
        str(path) for path in opened
        if RAW_DIR == path or RAW_DIR in path.parents
    ]

    audit.record(
        "T10a",
        "No file under data/raw/ was opened at any point",
        0, len(raw_touches),
        not raw_touches,
        "; ".join(sorted(set(raw_touches))[:5]),
    )

    project_data_files = {
        path for path in opened
        if path.suffix.lower() in {".csv", ".xls", ".xlsx", ".json"}
        and PROJECT_ROOT in path.parents
    }

    unexpected_reads = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in project_data_files - DECLARED_INPUTS
        # Outputs this instrument writes are legitimately opened for writing.
        if path not in {
            FEATURES_OUTPUT.resolve(),
            AUDIT_OUTPUT.resolve(),
            INVENTORY_OUTPUT.resolve(),
        }
    )

    audit.record(
        "T10b",
        "Only the two declared inputs were read",
        "0 unexpected data files",
        f"{len(unexpected_reads)} unexpected",
        not unexpected_reads,
        "; ".join(unexpected_reads[:5]),
    )

    # The FBref readers are never called. Tokens are assembled from fragments
    # so that this check cannot match its own source text.
    source = Path(__file__).read_text(encoding="utf-8")

    forbidden = ["read_" + "html(", "read_" + "excel("]

    found = [token for token in forbidden if token in source]

    audit.record(
        "T10c",
        "No FBref HTML/Excel table reader appears in this source",
        "0 occurrences", f"{len(found)} occurrences",
        not found,
        "; ".join(found),
    )

    audit.measure(
        "T10d",
        "Declared inputs",
        ", ".join(sorted(
            str(path.relative_to(PROJECT_ROOT)) for path in DECLARED_INPUTS
        )),
    )


def test_structure(features, audit):
    """Shape, schema and empty-state discipline of the delivered table."""

    audit.record(
        "S1", "One row per match",
        EXPECTED_TOTAL_MATCHES, len(features),
        len(features) == EXPECTED_TOTAL_MATCHES,
    )

    missing = [column for column in OUTPUT_COLUMNS if column not in features.columns]

    audit.record(
        "S2", "All declared feature columns present",
        "0 missing", f"{len(missing)} missing",
        not missing,
        ", ".join(missing[:10]),
    )

    always_defined = []

    for side in ("home", "away"):
        always_defined += [
            f"{side}_mp_before", f"{side}_pts_before", f"{side}_gf_before",
            f"{side}_ga_before", f"{side}_gd_before",
            f"{side}_venue_mp_before", f"{side}_venue_pts_before",
            f"{side}_venue_gf_before", f"{side}_venue_ga_before",
            f"{side}_venue_gd_before", f"{side}_prev_season_available",
        ]

    unexpected_nulls = int(features[always_defined].isna().sum().sum())

    audit.record(
        "S3", "Counters and availability flags are never null",
        0, unexpected_nulls,
        unexpected_nulls == 0,
    )

    # A rate must be defined exactly when its denominator is positive.
    mismatches = []

    rate_denominators = [
        ("ppm_before", "mp_before"),
        ("gfpm_before", "mp_before"),
        ("gapm_before", "mp_before"),
        ("gdpm_before", "mp_before"),
        ("last5_ppm_before", "last5_mp_before"),
        ("venue_ppm_before", "venue_mp_before"),
        ("venue_gdpm_before", "venue_mp_before"),
    ]

    for side in ("home", "away"):
        for rate, denominator in rate_denominators:

            defined = features[f"{side}_{rate}"].notna()
            usable = features[f"{side}_{denominator}"] > 0

            differs = int((defined != usable).sum())

            if differs:
                mismatches.append(f"{side}_{rate}: {differs} rows")

    audit.record(
        "S4", "A rate is defined exactly when its denominator is positive",
        "0 mismatches", f"{len(mismatches)} mismatches",
        not mismatches,
        "; ".join(mismatches[:5]),
    )

    # No infinities anywhere - they are the classic divide-by-zero survivor.
    numeric = features.select_dtypes(include=[np.number])

    infinite = int(np.isinf(numeric.to_numpy(dtype="float64")).sum())

    audit.record(
        "S5", "No infinite values in any numeric feature",
        0, infinite,
        infinite == 0,
    )


# ============================================================
# INVENTORY
# ============================================================

def build_inventory(features):
    """Per-column availability, so later phases can see what is actually there."""

    rows = []

    for column in OUTPUT_COLUMNS:

        series = features[column]

        if column in IDENTITY_COLUMNS:
            kind = "identity"
        elif column in RELATIVE_COLUMNS:
            kind = "relative"
        elif any(column.endswith(name) for name in RATE_SIDE_COLUMNS):
            kind = "rate"
        else:
            kind = "raw"

        available = int(series.notna().sum())

        row = {
            "column": column,
            "kind": kind,
            "dtype": str(series.dtype),
            "available": available,
            "missing": int(series.isna().sum()),
            "available_pct": round(100 * available / len(features), 2),
            "min": "",
            "max": "",
            "mean": "",
        }

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            if available:
                row["min"] = round(float(series.min()), 4)
                row["max"] = round(float(series.max()), 4)
                row["mean"] = round(float(series.mean()), 4)

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# REPORT
# ============================================================

def status_text(passed):
    return "PASS" if passed else "FAIL"


def line(label, value, verdict=None):

    if verdict is None:
        print(f"  {label:<36}{value}")
    else:
        print(f"  {label:<36}{value:<30}{verdict}")


def print_test_table(audit):

    print()
    print("=" * 79)
    print("VALIDATION DETAIL")
    print("=" * 79)
    print()

    for row in audit.frame().itertuples():

        marker = {"PASS": "PASS", "FAIL": "FAIL", "MEASURED": "----"}[row.status]

        print(f"  {marker}  {row.test_id:<5} {row.test}")
        print(f"              expected: {row.expected}")
        print(f"              observed: {row.observed}")

        if row.detail:
            print(f"              {row.detail}")


def print_spot_checks(spot_check_tables):

    print()
    print("=" * 79)
    print("INDEPENDENT RECONSTRUCTION - WORKED EXAMPLES")
    print("=" * 79)
    print()
    print("  Running walk over each team's matches in date order, compared")
    print("  row by row with the generated features. First "
          f"{RECONSTRUCTION_ROWS_SHOWN} matches shown.")

    for (season, team), table in spot_check_tables.items():

        print()
        print(f"  {season}  {team}")
        print(
            f"    {'Date':<12}{'V':<6}{'MP':>3}{'Pts':>5}{'PPM':>7}"
            f"{'VenMP':>7}{'VenPts':>8}{'VenPPM':>8}   agrees"
        )

        for row in table.head(RECONSTRUCTION_ROWS_SHOWN).itertuples():

            ppm = "" if pd.isna(row.ppm) else f"{row.ppm:.3f}"
            venue_ppm = "" if pd.isna(row.venue_ppm) else f"{row.venue_ppm:.3f}"

            print(
                f"    {row.date.date()!s:<12}{row.side:<6}{row.mp:>3}"
                f"{row.pts:>5}{ppm:>7}{row.venue_mp:>7}{row.venue_pts:>8}"
                f"{venue_ppm:>8}   {'yes' if row.agrees else 'NO'}"
            )

        disagreements = int((~table["agrees"]).sum())

        print(f"    full season: {len(table)} matches, "
              f"{disagreements} disagreements")


def print_inventory(inventory):

    print()
    print("=" * 79)
    print("FEATURE INVENTORY")
    print("=" * 79)
    print()

    counts = inventory["kind"].value_counts()

    for kind in ["identity", "raw", "rate", "relative"]:
        print(f"    {kind:<12}{counts.get(kind, 0):>4} columns")

    print(f"    {'TOTAL':<12}{len(inventory):>4} columns")

    print()
    print("  Columns with missing values (the explicit empty state):")
    print()
    print(f"    {'Column':<38}{'Available':>10}{'Missing':>9}{'Pct':>8}")

    incomplete = inventory[inventory["missing"] > 0].sort_values(
        "missing", ascending=False
    )

    for row in incomplete.itertuples():
        print(
            f"    {row.column:<38}{row.available:>10}{row.missing:>9}"
            f"{row.available_pct:>7}%"
        )

    if incomplete.empty:
        print("    (none)")


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 3: TEAM STRENGTH FEATURES")
    print("=" * 79)
    print()
    print(f"  Inputs     : {MATCHES_INPUT.relative_to(PROJECT_ROOT)}")
    print(f"               {STATE_INPUT.relative_to(PROJECT_ROOT)}")
    print("  Rule       : historical_date < current_date  (STRICT)")
    print("  Mechanism  : pairwise date-comparison mask, cross-checked")
    print("               against Instrument 2's independent searchsorted")
    print("  Empty state: NaN, never zero - for rates and differences alike")
    print("  Scope      : no models, no Elo, no FBref, no imputation")

    for required in (MATCHES_INPUT, STATE_INPUT):

        if not required.exists():
            print()
            print(f"  FAIL - missing input: {required}")
            print()
            print("STATUS: FAIL")
            return 1

    matches = load_matches()
    instrument2 = load_instrument2_state()

    if len(matches) != EXPECTED_TOTAL_MATCHES:
        print(f"\n  FAIL - expected {EXPECTED_TOTAL_MATCHES} matches, "
              f"found {len(matches)}\n\nSTATUS: FAIL")
        return 1

    audit = Audit()

    print()
    print("  Reconstructing state and deriving features ...")

    features, side_state = build_features(matches)

    print(f"  Built {len(features)} match rows, "
          f"{len(OUTPUT_COLUMNS)} columns, from {len(side_state)} team-sides.")

    print("  T9  independent reconstruction ...")
    spot_check_tables = test_t9_reconstruction(
        matches, features, instrument2, audit
    )

    print("  S   structure and empty-state discipline ...")
    test_structure(features, audit)

    print("  T1  temporal integrity ...")
    test_t1_temporal_integrity(matches, features, audit)

    print("  T2  current-season isolation ...")
    test_t2_current_season_isolation(matches, features, audit)

    print("  T3  venue isolation ...")
    test_t3_venue_isolation(matches, features, audit)

    print("  T4  previous-season isolation ...")
    test_t4_previous_season_isolation(matches, features, audit)

    print("  T5  promoted teams ...")
    test_t5_promoted_teams(matches, features, audit)

    print("  T6  sanction handling ...")
    test_t6_sanctions(features, audit)

    print("  T7  cold starts ...")
    test_t7_cold_starts(matches, features, audit)

    print("  T8  perturbation - future results against earlier rows ...")
    (
        temporal_violations,
        venue_violations,
        prior_violations,
        rebuilds,
    ) = test_t8_perturbation(matches, features, audit)

    print("  T10 input provenance ...")
    test_t10_no_fbref(audit)

    inventory = build_inventory(features)

    # ---- reports
    print_test_table(audit)
    print_spot_checks(spot_check_tables)
    print_inventory(inventory)

    # ---- outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    written = features[OUTPUT_COLUMNS].copy()
    written["date"] = written["date"].dt.strftime("%Y-%m-%d")

    written.to_csv(
        FEATURES_OUTPUT, index=False, encoding="utf-8", float_format="%.17g"
    )

    audit_frame = audit.frame()
    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    inventory.to_csv(INVENTORY_OUTPUT, index=False, encoding="utf-8")

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {FEATURES_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(written)} rows, {len(written.columns)} columns)")
    print(f"  {AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(audit_frame)} entries)")
    print(f"  {INVENTORY_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(inventory)} columns profiled)")

    passed = audit.all_passed()

    failures = audit.failures()

    def outcome(test_id):
        rows = [r for r in audit.rows if r["test_id"].startswith(test_id)]
        return status_text(all(r["status"] != "FAIL" for r in rows))

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 3")
    print("=" * 79)
    print()

    line("Total matches:", f"{len(features)}")
    line("Total team-side states:", f"{len(side_state)}")
    line("Feature columns:", f"{len(OUTPUT_COLUMNS)}")
    line("T1 temporal integrity:", "values only from before T", outcome("T1"))
    line("T2 current-season isolation:", "no carry across seasons", outcome("T2"))
    line("T3 venue isolation:", "home form home-only", outcome("T3"))
    line("T4 previous-season isolation:", "season N-1 only", outcome("T4"))
    line("T5 promoted teams:", "explicitly unavailable", outcome("T5"))
    line("T6 sanction handling:", "sanctioned used, raw kept", outcome("T6"))
    line("T7 cold starts:", "genuinely cold", outcome("T7"))
    line(
        "T8 perturbation:",
        f"{temporal_violations} temporal, {venue_violations} venue, "
        f"{prior_violations} prior",
        outcome("T8"),
    )
    line("T9 reconstruction:", "3 independent methods agree", outcome("T9"))
    line("T10 no FBref aggregates:", "file access recorded", outcome("T10"))

    print()
    line("Perturbation rebuilds:", f"{rebuilds}")

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
