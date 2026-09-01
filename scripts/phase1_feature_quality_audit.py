"""
===============================================================================
PHASE 1 - INSTRUMENT 5
FEATURE QUALITY AND AVAILABILITY GATE
===============================================================================

THE QUESTION
    Instruments 1-4 established that the foundation is trustworthy, the
    history is leakage-free, the features are correctly derived and the
    identities are coherent.

    This one asks the last question before Phase 1 freezes:

        do these features have usable distributions, sensible missingness,
        no impossible values and no accidental duplication?

THIS IS AN AUDIT, NOT FEATURE SELECTION
    Nothing is deleted, dropped, filtered or "cleaned". A constant column, a
    duplicate column and a sparse column are all REPORTED, never removed.
    Which features deserve to enter the modelling dataset is a separate
    decision, and it belongs to baseline experiments rather than to this
    instrument's intuition.

MISSINGNESS CARRIES MEANING
    A NaN here is not an absence of data quality - it is a fact about what
    was knowable before kickoff. So T3 does not merely count NaNs; it
    demands that every single one be EXPLAINED by a documented cause, and
    that every explained cause actually produce a NaN. Both directions:

        an unexplained NaN                     -> FAIL
        an explanation that produces no NaN     -> FAIL

    That two-way check is what makes the missingness pattern evidence
    rather than a tally.

INPUTS - four, all validated upstream
    outputs/phase1_matches.csv                    (Instrument 1)
    outputs/phase1_historical_team_state.csv      (Instrument 2)
    outputs/phase1_team_strength_features.csv     (Instrument 3 - the audit
                                                   target)
    outputs/phase1_team_transition_summary.csv    (Instrument 4)

EXIT CODES
    0  PASS    every test passed
    2  FAIL    an audit test failed - investigate, do not patch
    1  FATAL   the audit could not be run at all

WHAT IS NOT DONE HERE
    no models, no Elo, no XGBoost, no FBref aggregates, no feature removal,
    no imputation, no modelling dataset, no writes to data/raw/.
===============================================================================
"""

from pathlib import Path
import sys
import traceback

import numpy as np
import pandas as pd


# ============================================================
# FILE-ACCESS RECORDER  (evidence for T14)
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
STATE_INPUT = OUTPUTS_DIR / "phase1_historical_team_state.csv"
FEATURES_INPUT = OUTPUTS_DIR / "phase1_team_strength_features.csv"
TRANSITIONS_INPUT = OUTPUTS_DIR / "phase1_team_transition_summary.csv"

AUDIT_OUTPUT = OUTPUTS_DIR / "phase1_feature_quality_audit.csv"
AVAILABILITY_OUTPUT = OUTPUTS_DIR / "phase1_feature_availability.csv"
PROVENANCE_OUTPUT = OUTPUTS_DIR / "phase1_feature_provenance.csv"
MISSINGNESS_OUTPUT = OUTPUTS_DIR / "phase1_feature_missingness_reasons.csv"

DECLARED_INPUTS = {
    MATCHES_INPUT.resolve(), STATE_INPUT.resolve(),
    FEATURES_INPUT.resolve(), TRANSITIONS_INPUT.resolve(),
}

RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

EXPECTED_TOTAL_MATCHES = 1900
EXPECTED_TEAM_SIDES = 3800
EXPECTED_SEASONS = 5
EXPECTED_MATCHWEEKS = 38
FULL_SEASON_MATCHES = 38
LAST_N = 5
MAX_LAST_N_POINTS = 3 * LAST_N
MAX_VENUE_MATCHES = 19

EXIT_PASS = 0
EXIT_FATAL = 1
EXIT_FAIL = 2

PERTURBED_HOME_GOALS = 9
PERTURBED_AWAY_GOALS = 0

# Two floats are treated as equal within this tolerance throughout.
TOLERANCE = 1e-9

# Correlation at or above this is reported as mathematically equivalent.
EQUIVALENCE_CORRELATION = 1.0 - 1e-9


IDENTITY_COLUMNS = ["season", "date", "matchweek", "home_team", "away_team"]


# ---- documented missingness causes
REASON_PRESENT = "present"
REASON_COLD_START = "cold_start_no_current_season_history"
REASON_VENUE = "venue_history_unavailable"
REASON_NO_PREVIOUS_SEASON = "no_previous_season_in_dataset"
REASON_ABSENT_PREVIOUS = "absent_from_previous_season"
REASON_EITHER_SIDE_COLD = "relative_either_side_cold_start"
REASON_EITHER_SIDE_VENUE = "relative_either_side_venue_unavailable"
REASON_EITHER_SIDE_NO_PRIOR = "relative_either_side_no_prior"


# ---- provenance categories (T13)
PROV_IDENTITY = "identity"
PROV_CURRENT_HISTORY = "current_match_history"
PROV_CURRENT_VENUE = "current_venue_history"
PROV_PREVIOUS_SEASON = "previous_season"
PROV_RELATIVE = "relative"
PROV_AVAILABILITY = "availability_flag"

# Every provenance category must be derivable from the match foundation
# alone. None of them may touch an end-of-season FBref aggregate.
ALLOWED_PROVENANCE = {
    PROV_IDENTITY, PROV_CURRENT_HISTORY, PROV_CURRENT_VENUE,
    PROV_PREVIOUS_SEASON, PROV_RELATIVE, PROV_AVAILABILITY,
}

# Column-name fragments that would betray an FBref season aggregate having
# crept into the schema (T12).
FORBIDDEN_NAME_FRAGMENTS = [
    "xg", "npxg", "shooting", "goalkeeping", "keeper", "playing_time",
    "misc", "possession", "passing", "defensive", "final_position",
    "league_position", "season_total", "final_season", "attendance",
    "referee", "sca", "gca", "progressive", "aerial", "tackle",
]


# ============================================================
# ERRORS
# ============================================================

class FatalError(Exception):
    """The audit could not be run at all. Exit 1, not exit 2."""


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

    def record(self, test_id, test, scope, expected, observed, passed, detail=""):

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

        return bool(passed)

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

    def failures(self):
        return [row for row in self.rows if row["status"] == "FAIL"]

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

    for required in (MATCHES_INPUT, STATE_INPUT, FEATURES_INPUT,
                     TRANSITIONS_INPUT):
        if not required.exists():
            raise FatalError(f"missing required input: {required}")

    try:
        # float_precision="round_trip" is REQUIRED, not stylistic. pandas'
        # default C parser does not round-trip float64: a cell written as
        # 2.3333333333333335 comes back as 2.333333333333333, one ULP out.
        # Auditing a value the writer never produced is worse than useless.
        matches = pd.read_csv(MATCHES_INPUT, float_precision="round_trip")
        state = pd.read_csv(STATE_INPUT, float_precision="round_trip")
        features = pd.read_csv(FEATURES_INPUT, float_precision="round_trip")
        transitions = pd.read_csv(
            TRANSITIONS_INPUT, float_precision="round_trip"
        )
    except Exception as error:
        raise FatalError(f"input could not be parsed: {error}") from error

    for frame, name in ((matches, "matches"), (state, "state"),
                        (features, "features")):
        if len(frame) != EXPECTED_TOTAL_MATCHES:
            raise FatalError(
                f"{name} has {len(frame)} rows, expected {EXPECTED_TOTAL_MATCHES}"
            )

    for frame in (matches, state, features):
        frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d")

    for frame in (matches, state, features):
        frame.sort_values(
            ["season", "date", "home_team", "away_team"], inplace=True
        )
        frame.reset_index(drop=True, inplace=True)
        frame["match_id"] = frame.index

    return matches, state, features, transitions


def feature_columns(features):

    return [c for c in features.columns
            if c not in IDENTITY_COLUMNS and c != "match_id"]


# ============================================================
# PROVENANCE  (T13)
# ============================================================

def classify_provenance(column):
    """
    Every feature must trace to a documented source category.

    Order matters: venue is checked before general current history, because
    a venue column also contains the current-history substrings.
    """

    if column in IDENTITY_COLUMNS:
        return PROV_IDENTITY

    if column.startswith("rel_"):
        if column.endswith("_available"):
            return PROV_AVAILABILITY
        return PROV_RELATIVE

    body = column
    for prefix in ("home_", "away_"):
        if body.startswith(prefix):
            body = body[len(prefix):]
            break

    if body == "prev_season_available":
        return PROV_AVAILABILITY

    if body.startswith("prev_season"):
        return PROV_PREVIOUS_SEASON

    if body.startswith("venue_"):
        return PROV_CURRENT_VENUE

    return PROV_CURRENT_HISTORY


def side_of(column):

    if column.startswith("home_"):
        return "home"

    if column.startswith("away_"):
        return "away"

    return "both" if column.startswith("rel_") else "n/a"


# ============================================================
# MISSINGNESS MODEL  (T3)
# ============================================================

def expected_missing_mask(features, column):
    """
    The documented cause of this column's NaNs, as a boolean mask.

    Returns (mask, reason) or (None, None) when the column is never expected
    to be missing.
    """

    side = side_of(column)

    if side in ("home", "away"):

        body = column[len(side) + 1:]

        # ---- previous-season block
        #
        # `prev_season_status` is EXCLUDED: it is the metadata column that
        # explains why a prior is absent, so it is always populated - a NaN
        # there would mean the absence had no recorded reason.
        #
        # `prev_season_source` IS included: it holds the empty string when no
        # prior exists, which the CSV round-trip presents as NaN.
        if (
            body.startswith("prev_season")
            and body not in {"prev_season_available", "prev_season_status"}
        ):
            return (
                ~features[f"{side}_prev_season_available"].astype(bool),
                "prev_season_split",
            )

        # ---- venue rates: undefined until a match at that venue exists
        if body in {
            "venue_ppm_before", "venue_gfpm_before",
            "venue_gapm_before", "venue_gdpm_before",
        }:
            return features[f"{side}_venue_mp_before"] <= 0, REASON_VENUE

        # ---- last-5 block: undefined until any match exists
        if body in {"last5_pts_before", "last5_ppm_before"}:
            return features[f"{side}_last5_mp_before"] <= 0, REASON_COLD_START

        # ---- current-season rates and the previous-match feature
        if body in {
            "ppm_before", "gfpm_before", "gapm_before", "gdpm_before",
            "form_delta_ppm", "prev_match_pts_before",
        }:
            return features[f"{side}_mp_before"] <= 0, REASON_COLD_START

        return None, None

    if column.startswith("rel_"):

        if column.endswith("_available"):
            return None, None

        if "prev_season" in column:
            return (
                ~(
                    features["home_prev_season_available"].astype(bool)
                    & features["away_prev_season_available"].astype(bool)
                ),
                REASON_EITHER_SIDE_NO_PRIOR,
            )

        if "venue" in column:
            return (
                (features["home_venue_mp_before"] <= 0)
                | (features["away_venue_mp_before"] <= 0),
                REASON_EITHER_SIDE_VENUE,
            )

        if column == "rel_mp_diff":
            return None, None

        return (
            (features["home_mp_before"] <= 0)
            | (features["away_mp_before"] <= 0),
            REASON_EITHER_SIDE_COLD,
        )

    return None, None


def previous_season_reason(features, side):
    """Split an absent prior into its two genuinely different causes."""

    status = features[f"{side}_prev_season_status"]

    return status.map({
        "no_prior_season_in_dataset": REASON_NO_PREVIOUS_SEASON,
        "absent_from_previous_season": REASON_ABSENT_PREVIOUS,
    })


# ============================================================
# TESTS
# ============================================================

def test_t1_row_integrity(matches, state, features, audit):

    audit.record(
        "T1a", "Exactly 1,900 matches in the feature table",
        "feature table", EXPECTED_TOTAL_MATCHES, len(features),
        len(features) == EXPECTED_TOTAL_MATCHES,
    )

    duplicate_ids = int(features["match_id"].duplicated().sum())

    audit.record(
        "T1b", "No duplicate match identifiers",
        "1,900 rows", 0, duplicate_ids,
        duplicate_ids == 0,
    )

    # Identity must be unique on the natural key too, not merely on the index.
    natural_duplicates = int(
        features.duplicated(
            subset=["season", "date", "home_team", "away_team"]
        ).sum()
    )

    audit.record(
        "T1c", "No duplicate season+date+home+away",
        "1,900 rows", 0, natural_duplicates,
        natural_duplicates == 0,
    )

    # Exactly two team-sides per match, and no team facing itself.
    self_matches = int((features["home_team"] == features["away_team"]).sum())

    sides = pd.concat([
        features[["match_id", "home_team"]].rename(
            columns={"home_team": "team"}),
        features[["match_id", "away_team"]].rename(
            columns={"away_team": "team"}),
    ])

    per_match = sides.groupby("match_id")["team"].nunique()

    wrong_sides = int((per_match != 2).sum())

    audit.record(
        "T1d", "Exactly two distinct team-sides per match",
        "1,900 matches",
        f"2 x {len(features)} = {EXPECTED_TEAM_SIDES}",
        f"{len(sides)} sides, {wrong_sides} matches not at 2, "
        f"{self_matches} self-matches",
        wrong_sides == 0 and self_matches == 0
        and len(sides) == EXPECTED_TEAM_SIDES,
    )

    # The three upstream tables must describe the same 1,900 matches.
    aligned = (
        matches[["season", "date", "home_team", "away_team"]]
        .reset_index(drop=True)
        .equals(
            features[["season", "date", "home_team", "away_team"]]
            .reset_index(drop=True)
        )
        and matches[["season", "date", "home_team", "away_team"]]
        .reset_index(drop=True)
        .equals(
            state[["season", "date", "home_team", "away_team"]]
            .reset_index(drop=True)
        )
    )

    audit.record(
        "T1e", "Foundation, state and feature tables describe the same matches",
        "Instruments 1, 2 and 3",
        "identical", "identical" if aligned else "DIVERGED",
        aligned,
    )


def test_t2_availability(features, columns, audit):
    """Availability overall and by season, matchweek and side."""

    rows = []

    total = len(features)

    for column in columns:

        available = int(features[column].notna().sum())

        rows.append({
            "scope_type": "overall",
            "scope_value": "all",
            "column": column,
            "side": side_of(column),
            "provenance": classify_provenance(column),
            "rows": total,
            "available_count": available,
            "missing_count": total - available,
            "availability_rate": round(available / total, 6),
        })

    for scope_type, scope_column in (("season", "season"),
                                     ("matchweek", "matchweek")):

        for scope_value, group in features.groupby(scope_column):

            for column in columns:

                available = int(group[column].notna().sum())

                rows.append({
                    "scope_type": scope_type,
                    "scope_value": str(scope_value),
                    "column": column,
                    "side": side_of(column),
                    "provenance": classify_provenance(column),
                    "rows": len(group),
                    "available_count": available,
                    "missing_count": len(group) - available,
                    "availability_rate": round(available / len(group), 6),
                })

    # Side breakdown: the home and away versions of the same feature compared
    # directly, so an asymmetry cannot hide.
    paired = sorted({
        column[5:] for column in columns
        if column.startswith(("home_", "away_"))
    })

    for body in paired:

        home_column = f"home_{body}"
        away_column = f"away_{body}"

        if home_column not in features or away_column not in features:
            continue

        for side, column in (("home", home_column), ("away", away_column)):

            available = int(features[column].notna().sum())

            rows.append({
                "scope_type": "side",
                "scope_value": side,
                "column": body,
                "side": side,
                "provenance": classify_provenance(column),
                "rows": total,
                "available_count": available,
                "missing_count": total - available,
                "availability_rate": round(available / total, 6),
            })

    availability = pd.DataFrame(rows)

    audit.record(
        "T2a", "Availability computed for every feature across every scope",
        f"{len(columns)} features",
        f"{len(columns)} features x 4 scope types",
        f"{availability['column'].nunique()} features, {len(availability)} rows",
        availability["column"].nunique() >= len(columns),
    )

    # An availability rate outside [0, 1] would mean the counting is broken.
    out_of_range = availability[
        (availability["availability_rate"] < 0)
        | (availability["availability_rate"] > 1)
    ]

    audit.record(
        "T2b", "Every availability rate lies in [0, 1]",
        f"{len(availability)} scope rows",
        0, len(out_of_range),
        out_of_range.empty,
    )

    # Counts must reconcile with the row totals exactly.
    broken = availability[
        availability["available_count"] + availability["missing_count"]
        != availability["rows"]
    ]

    audit.record(
        "T2c", "available + missing equals the row count in every scope",
        f"{len(availability)} scope rows",
        0, len(broken),
        broken.empty,
    )

    fully_available = availability[
        (availability["scope_type"] == "overall")
        & (availability["availability_rate"] == 1.0)
    ]

    audit.measure(
        "T2d", "Features available on every row",
        "overall scope",
        f"{len(fully_available)} of {len(columns)}",
        "The remainder carry meaningful, explained absence",
    )

    return availability


def test_t3_missingness_reasons(features, columns, audit):
    """
    Every NaN must be explained, and every explanation must produce a NaN.

    A one-way check would let an unexplained NaN hide behind a broad rule.
    """

    rows = []
    unexplained = []
    over_explained = []

    total = len(features)

    for column in columns:

        actual_missing = features[column].isna()

        mask, reason = expected_missing_mask(features, column)

        if mask is None:

            if actual_missing.any():
                unexplained.append(
                    f"{column}: {int(actual_missing.sum())} NaN with no "
                    f"documented cause"
                )

            rows.append({
                "column": column,
                "provenance": classify_provenance(column),
                "reason": REASON_PRESENT,
                "count": total,
                "share": 1.0,
            })

            continue

        mask = mask.fillna(False).astype(bool)

        # Direction 1: an actual NaN the rule does not account for.
        missing_unexplained = int((actual_missing & ~mask).sum())

        if missing_unexplained:
            unexplained.append(f"{column}: {missing_unexplained} unexplained NaN")

        # Direction 2: the rule predicts a NaN and the value is present.
        predicted_present = int((~actual_missing & mask).sum())

        if predicted_present:
            over_explained.append(
                f"{column}: {predicted_present} rows predicted missing but present"
            )

        if reason == "prev_season_split":

            side = side_of(column)

            reasons = previous_season_reason(features, side).where(
                mask, REASON_PRESENT
            )

        else:
            reasons = pd.Series(
                np.where(mask, reason, REASON_PRESENT), index=features.index
            )

        for label, count in reasons.value_counts().items():
            rows.append({
                "column": column,
                "provenance": classify_provenance(column),
                "reason": label,
                "count": int(count),
                "share": round(int(count) / total, 6),
            })

    audit.record(
        "T3a", "Every missing value is explained by a documented cause",
        f"{len(columns)} features",
        "0 unexplained NaN", f"{len(unexplained)} columns with unexplained NaN",
        not unexplained,
        "; ".join(unexplained[:5]),
    )

    audit.record(
        "T3b", "Every documented cause actually produces a missing value",
        f"{len(columns)} features",
        "0 over-explanations", f"{len(over_explained)} columns",
        not over_explained,
        "; ".join(over_explained[:5]),
    )

    missingness = pd.DataFrame(rows)

    # The two prior-absence causes are genuinely different and must both exist.
    prior_reasons = set(
        missingness[missingness["reason"].isin(
            [REASON_NO_PREVIOUS_SEASON, REASON_ABSENT_PREVIOUS]
        )]["reason"]
    )

    audit.record(
        "T3c", "Prior absence is split into its two distinct causes",
        "previous-season features",
        f"{REASON_NO_PREVIOUS_SEASON} and {REASON_ABSENT_PREVIOUS}",
        str(sorted(prior_reasons)),
        prior_reasons == {REASON_NO_PREVIOUS_SEASON, REASON_ABSENT_PREVIOUS},
        "Dataset boundary and mid-dataset absence are not the same fact",
    )

    summary = (
        missingness[missingness["reason"] != REASON_PRESENT]
        .groupby("reason")["count"].sum().sort_values(ascending=False)
    )

    for reason, count in summary.items():
        audit.measure(
            "T3d", "Missing values by documented cause", reason, int(count),
        )

    return missingness


def test_t4_impossible_values(features, audit):
    """Constraints that no correct value can violate."""

    checks = []

    for side in ("home", "away"):

        checks += [
            (f"{side}_mp_before < 0", features[f"{side}_mp_before"] < 0),
            (f"{side}_pts_before < 0", features[f"{side}_pts_before"] < 0),
            (f"{side}_gf_before < 0", features[f"{side}_gf_before"] < 0),
            (f"{side}_ga_before < 0", features[f"{side}_ga_before"] < 0),
            (
                f"{side}_pts_before > 3 x mp",
                features[f"{side}_pts_before"]
                > 3 * features[f"{side}_mp_before"],
            ),
            (
                f"{side}_gd_before != gf - ga",
                features[f"{side}_gd_before"]
                != features[f"{side}_gf_before"] - features[f"{side}_ga_before"],
            ),
            (
                f"{side}_mp_before > 37",
                features[f"{side}_mp_before"] > FULL_SEASON_MATCHES - 1,
            ),
            (
                f"{side}_ppm_before outside [0, 3]",
                (features[f"{side}_ppm_before"] < 0)
                | (features[f"{side}_ppm_before"] > 3),
            ),
            (
                f"{side}_last5_pts_before > {MAX_LAST_N_POINTS}",
                features[f"{side}_last5_pts_before"] > MAX_LAST_N_POINTS,
            ),
            (
                f"{side}_last5_pts_before < 0",
                features[f"{side}_last5_pts_before"] < 0,
            ),
            (
                f"{side}_last5_mp_before > {LAST_N}",
                features[f"{side}_last5_mp_before"] > LAST_N,
            ),
            (
                f"{side}_last5_pts_before > 3 x last5_mp",
                features[f"{side}_last5_pts_before"]
                > 3 * features[f"{side}_last5_mp_before"],
            ),
            (
                f"{side}_venue_mp_before < 0",
                features[f"{side}_venue_mp_before"] < 0,
            ),
            (
                f"{side}_venue_pts_before > 3 x venue_mp",
                features[f"{side}_venue_pts_before"]
                > 3 * features[f"{side}_venue_mp_before"],
            ),
            (
                f"{side}_venue_gd_before != venue_gf - venue_ga",
                features[f"{side}_venue_gd_before"]
                != features[f"{side}_venue_gf_before"]
                - features[f"{side}_venue_ga_before"],
            ),
            (
                f"{side}_venue_mp_before > {MAX_VENUE_MATCHES}",
                features[f"{side}_venue_mp_before"] > MAX_VENUE_MATCHES,
            ),
            (
                f"{side}_venue_mp_before > mp_before",
                features[f"{side}_venue_mp_before"]
                > features[f"{side}_mp_before"],
            ),
            (
                f"{side}_venue_ppm_before outside [0, 3]",
                (features[f"{side}_venue_ppm_before"] < 0)
                | (features[f"{side}_venue_ppm_before"] > 3),
            ),
            (
                f"{side}_prev_match_pts_before not in (0, 1, 3)",
                features[f"{side}_prev_match_pts_before"].notna()
                & ~features[f"{side}_prev_match_pts_before"].isin([0, 1, 3]),
            ),
            (
                f"{side}_prev_season_mp != 38 when present",
                features[f"{side}_prev_season_mp"].notna()
                & (features[f"{side}_prev_season_mp"] != FULL_SEASON_MATCHES),
            ),
            (
                f"{side}_prev_season_gd != gf - ga",
                features[f"{side}_prev_season_gd"].notna()
                & (
                    features[f"{side}_prev_season_gd"]
                    != features[f"{side}_prev_season_gf"]
                    - features[f"{side}_prev_season_ga"]
                ),
            ),
            (
                f"{side}_prev_season_pts outside [0, 114]",
                features[f"{side}_prev_season_pts"].notna()
                & (
                    (features[f"{side}_prev_season_pts"] < 0)
                    | (features[f"{side}_prev_season_pts"] > 114)
                ),
            ),
            (
                f"{side}_prev_season_gf < 0",
                features[f"{side}_prev_season_gf"] < 0,
            ),
        ]

    violations = []

    for label, mask in checks:

        count = int(mask.fillna(False).sum())

        if count:
            violations.append(f"{label}: {count} rows")

    audit.record(
        "T4a", "No impossible value in any feature",
        f"{len(checks)} constraints",
        "0 violated constraints", f"{len(violations)} violated",
        not violations,
        "; ".join(violations[:8]),
    )

    numeric = features.select_dtypes(include=[np.number])

    infinite = int(np.isinf(numeric.to_numpy(dtype="float64")).sum())

    audit.record(
        "T4b", "No infinite value in any numeric feature",
        f"{numeric.shape[1]} numeric columns",
        0, infinite,
        infinite == 0,
    )

    audit.measure(
        "T4c", "Impossible-value constraints evaluated",
        "all features", len(checks),
    )


def test_t5_internal_identities(features, audit):
    """Identities that must hold by construction."""

    broken = []

    identities = [
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

        for rate, numerator, denominator in identities:

            usable = features[f"{side}_{denominator}"].fillna(0) > 0

            expected = (
                features.loc[usable, f"{side}_{numerator}"]
                / features.loc[usable, f"{side}_{denominator}"]
            )

            actual = features.loc[usable, f"{side}_{rate}"]

            differs = int((~np.isclose(expected, actual, atol=TOLERANCE)).sum())

            if differs:
                broken.append(f"{side}_{rate}: {differs} rows")

        # GD identities on every historical state
        for gd, gf, ga in [
            ("gd_before", "gf_before", "ga_before"),
            ("venue_gd_before", "venue_gf_before", "venue_ga_before"),
            ("prev_season_gd", "prev_season_gf", "prev_season_ga"),
        ]:
            difference = (
                features[f"{side}_{gd}"]
                - (features[f"{side}_{gf}"] - features[f"{side}_{ga}"])
            )

            differs = int((difference.fillna(0).abs() > TOLERANCE).sum())

            if differs:
                broken.append(f"{side}_{gd}: {differs} rows")

        # form_delta is defined as last-5 rate minus season rate
        expected_delta = (
            features[f"{side}_last5_ppm_before"] - features[f"{side}_ppm_before"]
        )

        differs = int((
            ~(
                np.isclose(
                    expected_delta, features[f"{side}_form_delta_ppm"],
                    atol=TOLERANCE, equal_nan=True,
                )
            )
        ).sum())

        if differs:
            broken.append(f"{side}_form_delta_ppm: {differs} rows")

        # used = raw + sanction, wherever a prior exists
        with_prior = features[f"{side}_prev_season_pts"].notna()

        difference = (
            features.loc[with_prior, f"{side}_prev_season_pts"]
            - features.loc[with_prior, f"{side}_prev_season_pts_raw"]
            - features.loc[with_prior, f"{side}_prev_season_sanction"]
        )

        differs = int((difference.abs() > TOLERANCE).sum())

        if differs:
            broken.append(f"{side}_prev_season_pts identity: {differs} rows")

    audit.record(
        "T5a", "Every rate, GD and sanction identity holds",
        "all features",
        "0 broken identities", f"{len(broken)} broken",
        not broken,
        "; ".join(broken[:8]),
    )

    # last5_mp must be exactly min(5, mp_before) - never padded, never short.
    wrong = 0

    for side in ("home", "away"):
        expected_used = np.minimum(features[f"{side}_mp_before"], LAST_N)
        wrong += int((features[f"{side}_last5_mp_before"] != expected_used).sum())

    audit.record(
        "T5b", "last5_mp equals min(5, mp_before) on every row",
        "3,800 team-sides", 0, wrong,
        wrong == 0,
    )


def test_t6_relative_features(features, state, audit):
    """
    T6 - every difference verified from the RAW team-side columns.

    Crucially, the raw columns are taken from Instrument 2's state file, not
    from Instrument 3's own raw columns. A wrong-side wiring bug that was
    consistent within Instrument 3 would survive a self-check.
    """

    rebuilt = pd.DataFrame(index=features.index)

    def rate(numerator, denominator):
        result = np.full(len(state), np.nan)
        usable = denominator.to_numpy() > 0
        np.divide(
            numerator.to_numpy(dtype="float64"),
            denominator.to_numpy(dtype="float64"),
            out=result, where=usable,
        )
        result[~usable] = np.nan
        return result

    for side in ("home", "away"):

        mp = state[f"{side}_matches_before"]
        venue_mp = state[f"{side}_venue_matches_before"]

        rebuilt[f"{side}_ppm"] = rate(state[f"{side}_points_before"], mp)
        rebuilt[f"{side}_gfpm"] = rate(state[f"{side}_gf_before"], mp)
        rebuilt[f"{side}_gapm"] = rate(state[f"{side}_ga_before"], mp)
        rebuilt[f"{side}_gdpm"] = rate(state[f"{side}_gd_before"], mp)

        rebuilt[f"{side}_last5_ppm"] = rate(
            state[f"{side}_last5_points_before"],
            state[f"{side}_last5_matches_used"],
        )

        rebuilt[f"{side}_venue_ppm"] = rate(
            state[f"{side}_venue_points_before"], venue_mp)
        rebuilt[f"{side}_venue_gfpm"] = rate(
            state[f"{side}_venue_gf_before"], venue_mp)
        rebuilt[f"{side}_venue_gapm"] = rate(
            state[f"{side}_venue_ga_before"], venue_mp)
        rebuilt[f"{side}_venue_gdpm"] = rate(
            state[f"{side}_venue_gf_before"] - state[f"{side}_venue_ga_before"],
            venue_mp,
        )

        rebuilt[f"{side}_mp"] = mp
        rebuilt[f"{side}_prev_match_pts"] = state[
            f"{side}_previous_match_points_before"]

        prev_mp = state[f"{side}_previous_season_matches"].fillna(0)

        rebuilt[f"{side}_prev_season_ppm"] = rate(
            state[f"{side}_previous_season_points"].fillna(0), prev_mp)
        rebuilt[f"{side}_prev_season_gdpm"] = rate(
            state[f"{side}_previous_season_gd"].fillna(0), prev_mp)

    pairs = [
        ("rel_ppm_diff", "ppm"),
        ("rel_gfpm_diff", "gfpm"),
        ("rel_gapm_diff", "gapm"),
        ("rel_gdpm_diff", "gdpm"),
        ("rel_last5_ppm_diff", "last5_ppm"),
        ("rel_prev_match_pts_diff", "prev_match_pts"),
        ("rel_mp_diff", "mp"),
        ("rel_venue_ppm_diff", "venue_ppm"),
        ("rel_venue_gfpm_diff", "venue_gfpm"),
        ("rel_venue_gapm_diff", "venue_gapm"),
        ("rel_venue_gdpm_diff", "venue_gdpm"),
        ("rel_prev_season_ppm_diff", "prev_season_ppm"),
        ("rel_prev_season_gdpm_diff", "prev_season_gdpm"),
    ]

    broken = []

    for target, body in pairs:

        expected = (
            rebuilt[f"home_{body}"].to_numpy(dtype="float64")
            - rebuilt[f"away_{body}"].to_numpy(dtype="float64")
        )

        actual = features[target].to_numpy(dtype="float64")

        differs = int((
            ~np.isclose(expected, actual, atol=TOLERANCE, equal_nan=True)
        ).sum())

        if differs:
            broken.append(f"{target}: {differs} rows")

    audit.record(
        "T6a", "Every difference reproduced from Instrument 2's raw state",
        f"{len(pairs)} relative features",
        "0 disagreements", f"{len(broken)} disagreeing",
        not broken,
        "; ".join(broken[:5]),
    )

    # A wrong-side bug: the difference must NOT match away-minus-home, nor a
    # same-side subtraction, unless it is identically zero everywhere.
    suspicious = []

    for target, body in pairs:

        actual = features[target].to_numpy(dtype="float64")

        reversed_expected = (
            rebuilt[f"away_{body}"].to_numpy(dtype="float64")
            - rebuilt[f"home_{body}"].to_numpy(dtype="float64")
        )

        finite = np.isfinite(actual) & np.isfinite(reversed_expected)

        if not finite.any():
            continue

        # If the feature equals the reversed difference everywhere AND is not
        # trivially zero, the sides are wired backwards.
        matches_reversed = np.allclose(
            actual[finite], reversed_expected[finite], atol=TOLERANCE
        )

        not_all_zero = np.any(np.abs(actual[finite]) > TOLERANCE)

        if matches_reversed and not_all_zero:
            suspicious.append(target)

    audit.record(
        "T6b", "No relative feature is wired away-minus-home",
        f"{len(pairs)} relative features",
        0, len(suspicious),
        not suspicious,
        "; ".join(suspicious),
    )

    # Availability flags must agree with the underlying availability.
    flag_problems = []

    expected_flags = {
        "rel_form_available": (
            (features["home_mp_before"] > 0) & (features["away_mp_before"] > 0)
        ),
        "rel_venue_form_available": (
            (features["home_venue_mp_before"] > 0)
            & (features["away_venue_mp_before"] > 0)
        ),
        "rel_prev_season_available": (
            features["home_prev_season_available"].astype(bool)
            & features["away_prev_season_available"].astype(bool)
        ),
    }

    for flag, expected in expected_flags.items():

        differs = int((features[flag].astype(bool) != expected).sum())

        if differs:
            flag_problems.append(f"{flag}: {differs} rows")

    audit.record(
        "T6c", "Relative availability flags match the underlying availability",
        f"{len(expected_flags)} flags",
        "0 disagreements", f"{len(flag_problems)} disagreeing",
        not flag_problems,
        "; ".join(flag_problems),
    )


def test_t7_venue_isolation(matches, features, audit):
    """T7 - venue history recomputed from the match foundation, from scratch."""

    home = matches[[
        "match_id", "season", "date", "home_team", "home_goals", "away_goals",
        "home_points_from_result",
    ]].rename(columns={
        "home_team": "team", "home_goals": "gf", "away_goals": "ga",
        "home_points_from_result": "pts",
    })
    home["side"] = "home"

    away = matches[[
        "match_id", "season", "date", "away_team", "away_goals", "home_goals",
        "away_points_from_result",
    ]].rename(columns={
        "away_team": "team", "away_goals": "gf", "home_goals": "ga",
        "away_points_from_result": "pts",
    })
    away["side"] = "away"

    sides = pd.concat([home, away], ignore_index=True)

    by_team_venue = {
        key: group.sort_values("date")
        for key, group in sides.groupby(["season", "team", "side"], sort=False)
    }

    violations = 0
    contamination = 0
    examples = []

    feature_lookup = features.set_index("match_id")

    for row in sides.itertuples():

        group = by_team_venue[(row.season, row.team, row.side)]

        earlier = group[group["date"] < row.date]

        target = feature_lookup.loc[row.match_id]

        expected = (
            len(earlier), earlier["pts"].sum(),
            earlier["gf"].sum(), earlier["ga"].sum(),
        )

        observed = (
            target[f"{row.side}_venue_mp_before"],
            target[f"{row.side}_venue_pts_before"],
            target[f"{row.side}_venue_gf_before"],
            target[f"{row.side}_venue_ga_before"],
        )

        if expected != observed:
            violations += 1

            if len(examples) < 5:
                examples.append(
                    f"{row.season} {row.team} {row.side}: {observed} != {expected}"
                )

        # Contamination check: the OTHER venue's matches must not be in there.
        other = "away" if row.side == "home" else "home"

        other_group = by_team_venue[(row.season, row.team, other)]

        other_earlier = other_group[other_group["date"] < row.date]

        combined = len(earlier) + len(other_earlier)

        if target[f"{row.side}_venue_mp_before"] == combined and combined > len(earlier):
            contamination += 1

    audit.record(
        "T7a", "Venue history rebuilt only from matches at that venue",
        f"{len(sides)} team-sides",
        0, violations,
        violations == 0,
        "; ".join(examples),
    )

    audit.record(
        "T7b", "Venue history never absorbs the other venue's matches",
        f"{len(sides)} team-sides",
        0, contamination,
        contamination == 0,
    )

    # Home venue matches + away venue matches must equal overall matches.
    mismatch = 0

    for side in ("home", "away"):
        mismatch += int((
            features[f"{side}_venue_mp_before"] > features[f"{side}_mp_before"]
        ).sum())

    audit.record(
        "T7c", "Venue matches never exceed total matches",
        "1,900 matches x 2 sides", 0, mismatch,
        mismatch == 0,
    )


def test_t8_redundancy(features, columns, audit):
    """
    T8 - inventory of duplicate and mathematically equivalent columns.

    NOTHING IS REMOVED. This is evidence for a later, separate decision.
    """

    numeric_columns = [
        column for column in columns
        if pd.api.types.is_numeric_dtype(features[column])
        and not pd.api.types.is_bool_dtype(features[column])
    ]

    exact_duplicates = []

    seen = {}

    for column in columns:

        series = features[column]

        signature = (
            str(series.dtype),
            tuple(series.fillna("__NA__").astype(str)),
        )

        if signature in seen:
            exact_duplicates.append((seen[signature], column))
        else:
            seen[signature] = column

    audit.measure(
        "T8a", "Exact duplicate column pairs",
        f"{len(columns)} features",
        len(exact_duplicates),
        "; ".join(f"{a} == {b}" for a, b in exact_duplicates[:10])
        if exact_duplicates else "none",
    )

    # Mathematically equivalent: perfectly (anti-)correlated on shared rows,
    # which captures any affine relationship y = a*x + b with a != 0.
    equivalent = []

    frame = features[numeric_columns]

    for i, left in enumerate(numeric_columns):

        left_series = frame[left]

        if left_series.notna().sum() < 3:
            continue

        for right in numeric_columns[i + 1:]:

            right_series = frame[right]

            shared = left_series.notna() & right_series.notna()

            if shared.sum() < 3:
                continue

            a = left_series[shared]
            b = right_series[shared]

            if a.nunique() < 2 or b.nunique() < 2:
                continue

            correlation = a.corr(b)

            if pd.isna(correlation):
                continue

            if abs(correlation) >= EQUIVALENCE_CORRELATION:
                equivalent.append((left, right, round(float(correlation), 12)))

    audit.measure(
        "T8b", "Mathematically equivalent column pairs (|r| = 1)",
        f"{len(numeric_columns)} numeric features",
        len(equivalent),
        "; ".join(f"{a} ~ {b} (r={r})" for a, b, r in equivalent[:12])
        if equivalent else "none",
    )

    # The audit must not have removed anything.
    audit.record(
        "T8c", "No feature was removed by this instrument",
        "audit, not feature selection",
        f"{len(columns)} features in",
        f"{len(columns)} features out",
        True,
        "Redundancy is inventoried for a later, separate decision",
    )

    return exact_duplicates, equivalent


def test_t9_constant_features(features, columns, audit):
    """T9 - columns with a single distinct value. Reported, never removed."""

    constants = []
    near_constants = []

    for column in columns:

        series = features[column]
        present = series.dropna()

        if present.empty:
            constants.append((column, "all missing", 0))
            continue

        distinct = present.nunique()

        if distinct == 1:
            constants.append((column, repr(present.iloc[0]), int(len(present))))
        elif distinct == 2 and len(present) > 100:
            counts = present.value_counts()
            if counts.iloc[1] / len(present) < 0.01:
                near_constants.append(
                    (column, f"{counts.index[1]!r} in {counts.iloc[1]} rows")
                )

    audit.measure(
        "T9a", "Constant features (one distinct value among present rows)",
        f"{len(columns)} features",
        len(constants),
        "; ".join(f"{c} = {v} ({n} rows)" for c, v, n in constants)
        if constants else "none",
    )

    audit.measure(
        "T9b", "Near-constant features (second value under 1% of rows)",
        f"{len(columns)} features",
        len(near_constants),
        "; ".join(f"{c}: {v}" for c, v in near_constants)
        if near_constants else "none",
    )

    audit.record(
        "T9c", "Constant features are reported, not removed",
        "audit, not feature selection",
        f"{len(constants)} reported",
        f"{len(constants)} reported, 0 removed",
        True,
    )

    return constants, near_constants


def test_t10_distributions(features, columns, audit):
    """T10 - ranges and basic statistics, with sanity bounds per family."""

    rows = []

    for column in columns:

        series = features[column]
        present = series.dropna()

        record = {
            "column": column,
            "provenance": classify_provenance(column),
            "side": side_of(column),
            "dtype": str(series.dtype),
            "count": int(len(present)),
            "distinct": int(present.nunique()) if not present.empty else 0,
        }

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
            and not present.empty
        ):
            record.update({
                "min": round(float(present.min()), 6),
                "p05": round(float(present.quantile(0.05)), 6),
                "median": round(float(present.median()), 6),
                "mean": round(float(present.mean()), 6),
                "p95": round(float(present.quantile(0.95)), 6),
                "max": round(float(present.max()), 6),
                "std": round(float(present.std()), 6) if len(present) > 1 else 0.0,
            })
        else:
            for key in ["min", "p05", "median", "mean", "p95", "max", "std"]:
                record[key] = ""

        rows.append(record)

    distributions = pd.DataFrame(rows)

    # Family-level sanity bounds. A value outside these is not a matter of
    # taste - it is arithmetically impossible for the quantity involved.
    bounds = {
        "ppm_before": (0, 3),
        "venue_ppm_before": (0, 3),
        "last5_ppm_before": (0, 3),
        "prev_season_ppm": (0, 3),
        "gfpm_before": (0, 10),
        "gapm_before": (0, 10),
        "gdpm_before": (-10, 10),
        "last5_pts_before": (0, MAX_LAST_N_POINTS),
        "mp_before": (0, FULL_SEASON_MATCHES - 1),
        "venue_mp_before": (0, MAX_VENUE_MATCHES),
        "prev_season_mp": (FULL_SEASON_MATCHES, FULL_SEASON_MATCHES),
        "prev_season_pts": (0, 114),
    }

    violations = []

    for side in ("home", "away"):
        for body, (low, high) in bounds.items():

            column = f"{side}_{body}"

            if column not in features:
                continue

            present = features[column].dropna()

            if present.empty:
                continue

            if present.min() < low - TOLERANCE or present.max() > high + TOLERANCE:
                violations.append(
                    f"{column}: [{present.min()}, {present.max()}] "
                    f"outside [{low}, {high}]"
                )

    audit.record(
        "T10a", "Every numeric feature lies inside its arithmetic bounds",
        f"{len(bounds)} feature families x 2 sides",
        "0 out-of-range families", f"{len(violations)} out of range",
        not violations,
        "; ".join(violations[:5]),
    )

    # Differences of bounded quantities are themselves bounded.
    diff_bounds = {
        "rel_ppm_diff": 3, "rel_venue_ppm_diff": 3,
        "rel_last5_ppm_diff": 3, "rel_prev_season_ppm_diff": 3,
        "rel_prev_match_pts_diff": 3,
    }

    diff_violations = []

    for column, limit in diff_bounds.items():

        present = features[column].dropna()

        if present.empty:
            continue

        if present.abs().max() > limit + TOLERANCE:
            diff_violations.append(
                f"{column}: max |{present.abs().max()}| > {limit}"
            )

    audit.record(
        "T10b", "Relative features stay inside the bounds their inputs imply",
        f"{len(diff_bounds)} differences",
        "0 out of range", f"{len(diff_violations)} out of range",
        not diff_violations,
        "; ".join(diff_violations),
    )

    # Degenerate spread: zero variance among present values.
    numeric = distributions[distributions["std"] != ""]

    zero_variance = numeric[numeric["std"] == 0.0]

    audit.measure(
        "T10c", "Numeric features with zero variance",
        f"{len(numeric)} numeric features",
        len(zero_variance),
        ", ".join(zero_variance["column"]) if not zero_variance.empty else "none",
    )

    return distributions


def test_t11_availability_pattern(features, transitions, audit):
    """
    T11 - the cold-start pattern must be exactly as logic requires.

    Anything becoming available EARLIER than it logically can is a FAIL. The
    rules are stated directly rather than inferred from the shape of a curve.
    """

    sides = pd.concat([
        features[["season", "date", "home_team"]].rename(
            columns={"home_team": "team"}),
        features[["season", "date", "away_team"]].rename(
            columns={"away_team": "team"}),
    ])

    first_match = sides.groupby(["season", "team"])["date"].min()

    early = 0
    late = 0
    examples = []

    for row in features.itertuples():

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            mp = getattr(row, f"{side}_mp_before")
            venue_mp = getattr(row, f"{side}_venue_mp_before")

            is_first = row.date == first_match[(row.season, team)]

            rate_defined = pd.notna(getattr(row, f"{side}_ppm_before"))
            venue_defined = pd.notna(getattr(row, f"{side}_venue_ppm_before"))

            # Available before it can be
            if is_first and (rate_defined or venue_defined or mp > 0):
                early += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: live at first match")

            if venue_mp == 0 and venue_defined:
                early += 1

            if mp == 0 and rate_defined:
                early += 1

            # Unavailable when it should be there
            if mp > 0 and not rate_defined:
                late += 1

                if len(examples) < 5:
                    examples.append(f"{row.season} {team}: mp>0 but no rate")

            if venue_mp > 0 and not venue_defined:
                late += 1

    audit.record(
        "T11a", "Nothing becomes available before it logically can",
        "3,800 team-sides",
        0, early,
        early == 0,
        "; ".join(examples[:5]),
    )

    audit.record(
        "T11b", "Nothing stays unavailable once its history exists",
        "3,800 team-sides",
        0, late,
        late == 0,
    )

    # Current-season form must be 0% available in matchweek 1.
    matchweek_1 = features[features["matchweek"] == 1]

    mw1_available = int(
        matchweek_1["home_ppm_before"].notna().sum()
        + matchweek_1["away_ppm_before"].notna().sum()
    )

    audit.record(
        "T11c", "Current-season form is unavailable throughout matchweek 1",
        f"{len(matchweek_1)} matchweek-1 matches",
        0, mw1_available,
        mw1_available == 0,
    )

    # Venue form availability must never decrease as a season progresses.
    regressions = []

    for season, group in features.groupby("season"):

        by_week = group.groupby("matchweek").apply(
            lambda block: int(
                block["home_venue_ppm_before"].notna().sum()
                + block["away_venue_ppm_before"].notna().sum()
            ),
            include_groups=False,
        )

        drops = by_week.diff().dropna()

        # A drop is only suspicious if it goes below an already-reached floor
        # of full availability; partial weeks vary in size, so compare rates.
        rates = group.groupby("matchweek").apply(
            lambda block: (
                block["home_venue_ppm_before"].notna().sum()
                + block["away_venue_ppm_before"].notna().sum()
            ) / (2 * len(block)),
            include_groups=False,
        )

        if rates.iloc[0] != 0.0:
            regressions.append(f"{season}: MW1 venue availability {rates.iloc[0]}")

    audit.record(
        "T11d", "Venue form is unavailable in matchweek 1 of every season",
        f"{EXPECTED_SEASONS} seasons",
        0, len(regressions),
        not regressions,
        "; ".join(regressions),
    )

    # Previous-season availability must match Instrument 4 exactly.
    expected_available = {
        (row.season, row.team): bool(row.has_previous_season)
        for row in transitions.itertuples()
    }

    prior_mismatch = 0

    for row in features.itertuples():

        for side, team in (("home", row.home_team), ("away", row.away_team)):

            observed = bool(getattr(row, f"{side}_prev_season_available"))

            if observed != expected_available.get((row.season, team)):
                prior_mismatch += 1

    audit.record(
        "T11e", "Previous-season availability matches Instrument 4's transitions",
        "3,800 team-sides",
        0, prior_mismatch,
        prior_mismatch == 0,
    )

    # Availability curve, reported for inspection.
    curve = features.groupby("matchweek").apply(
        lambda block: pd.Series({
            "current_form": round((
                block["home_ppm_before"].notna().sum()
                + block["away_ppm_before"].notna().sum()
            ) / (2 * len(block)), 4),
            "venue_form": round((
                block["home_venue_ppm_before"].notna().sum()
                + block["away_venue_ppm_before"].notna().sum()
            ) / (2 * len(block)), 4),
        }),
        include_groups=False,
    )

    for matchweek in [1, 2, 3, 4, 5, 10, 38]:

        if matchweek not in curve.index:
            continue

        audit.measure(
            "T11f", "Availability rate by matchweek", f"MW{matchweek}",
            f"current {curve.loc[matchweek, 'current_form']:.0%}, "
            f"venue {curve.loc[matchweek, 'venue_form']:.0%}",
        )

    return curve


def test_t12_no_final_season(matches, features, columns, audit):
    """
    T12 - no feature depends on end-of-season information.

    Two kinds of evidence:

    (a) SCHEMA - no column name betrays an FBref season aggregate.

    (b) BEHAVIOUR - the decisive one. The final matchweek of a season is,
        by definition, end-of-season information. Rewriting all of it to 9-0
        must leave EVERY feature row in that season untouched.

        A positive control accompanies it: the same perturbation MUST change
        the following season's prior. Without that, a rebuild that silently
        ignored the perturbation would pass by doing nothing.
    """

    suspicious = [
        column for column in columns
        if any(fragment in column.casefold()
               for fragment in FORBIDDEN_NAME_FRAGMENTS)
    ]

    audit.record(
        "T12a", "No column name indicates an FBref season aggregate",
        f"{len(columns)} features",
        "0 suspicious names", f"{len(suspicious)} suspicious",
        not suspicious,
        "; ".join(suspicious),
    )

    # ---- behavioural test
    #
    # BASELINE DISCIPLINE: the baseline is rebuilt IN MEMORY, not loaded from
    # the CSV. An earlier version of this test compared the CSV-loaded table
    # against an in-memory rebuild and reported 12,822 "changes" with no
    # perturbation applied at all - every one an artefact of the CSV round
    # trip (float64 written to 16 significant digits, and "" returning as
    # NaN). Comparing like with like is what makes the result mean anything.
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    try:
        from phase1_team_strength_features import build_features
    except Exception as error:
        raise FatalError(
            f"cannot import the feature builder for the T12 rebuild: {error}"
        ) from error

    compare_columns = [c for c in columns if c in features.columns]

    baseline_frame, _ = build_features(matches)
    baseline = baseline_frame.set_index("match_id")

    def differences(before, after, subset):
        """Element-wise inequality, NaN-aware, tolerant on floats."""

        differs = pd.DataFrame(False, index=before.index, columns=subset)

        for column in subset:

            left = before[column]
            right = after[column]

            both_missing = left.isna() & right.isna()

            if pd.api.types.is_float_dtype(left) and pd.api.types.is_float_dtype(right):
                close = np.isclose(
                    left.to_numpy(dtype="float64"),
                    right.to_numpy(dtype="float64"),
                    rtol=0, atol=TOLERANCE, equal_nan=True,
                )
                differs[column] = ~close
            else:
                differs[column] = ~((left == right) | both_missing)

        return differs

    # ---- null control: the builder must be deterministic
    repeat_frame, _ = build_features(matches)

    determinism = int(
        differences(
            baseline, repeat_frame.set_index("match_id"), compare_columns
        ).to_numpy().sum()
    )

    audit.record(
        "T12d",
        "Null control: rebuilding without perturbation changes nothing",
        "two consecutive unperturbed rebuilds",
        "0 changed values", f"{determinism} changed",
        determinism == 0,
        "Without this, a builder that ignored its input would pass T12b",
    )

    seasons = sorted(matches["season"].unique())

    in_season_changes = 0
    control_changes = 0
    rebuilds = 0
    examples = []

    for index, season in enumerate(seasons):

        final_week = matches[
            (matches["season"] == season)
            & (matches["matchweek"] == EXPECTED_MATCHWEEKS)
        ]

        perturbed = matches.copy()

        mask = perturbed["match_id"].isin(set(final_week["match_id"]))

        perturbed.loc[mask, "home_goals"] = PERTURBED_HOME_GOALS
        perturbed.loc[mask, "away_goals"] = PERTURBED_AWAY_GOALS
        perturbed.loc[mask, "result"] = "H"
        perturbed.loc[mask, "home_points_from_result"] = 3
        perturbed.loc[mask, "away_points_from_result"] = 0

        rebuilt_frame, _ = build_features(perturbed)

        rebuilds += 1

        rebuilt = rebuilt_frame.set_index("match_id")

        season_ids = list(matches[matches["season"] == season]["match_id"])

        differs = differences(
            baseline.loc[season_ids], rebuilt.loc[season_ids], compare_columns
        )

        changed = int(differs.to_numpy().sum())

        in_season_changes += changed

        if changed and len(examples) < 5:
            examples.append(
                f"{season}: {list(differs.columns[differs.any(axis=0)])[:4]}"
            )

        # Positive control: the NEXT season's prior MUST move.
        if index + 1 < len(seasons):

            next_ids = list(
                matches[matches["season"] == seasons[index + 1]]["match_id"]
            )

            prior_columns = [c for c in compare_columns if "prev_season" in c]

            moved = differences(
                baseline.loc[next_ids], rebuilt.loc[next_ids], prior_columns
            )

            control_changes += int(moved.to_numpy().sum())

    audit.record(
        "T12b",
        "Rewriting a season's final matchweek changes nothing within that season",
        f"{rebuilds} full rebuilds, all 5 seasons",
        "0 changed values", f"{in_season_changes} changed",
        in_season_changes == 0,
        "End-of-season results are unreachable from that season's features; "
        + "; ".join(examples[:3]),
    )

    audit.record(
        "T12c",
        "Positive control: the same perturbation DOES move the next season's prior",
        f"{rebuilds - 1} season boundaries",
        "> 0 changed values", f"{control_changes} changed",
        control_changes > 0,
        "Proves the rebuild responds to perturbation, so T12b is not vacuous",
    )

    # ---- T15: how faithfully does the delivered CSV round-trip?
    #
    # Not a defect in the features, but a property of the delivered artefact
    # that anyone re-deriving from the CSV needs to know.
    csv_baseline = features.set_index("match_id")

    float_columns = [
        c for c in compare_columns
        if pd.api.types.is_float_dtype(csv_baseline[c])
        and pd.api.types.is_float_dtype(baseline[c])
    ]

    worst = 0.0

    for column in float_columns:

        left = csv_baseline[column].to_numpy(dtype="float64")
        right = baseline[column].to_numpy(dtype="float64")

        both = np.isfinite(left) & np.isfinite(right)

        if both.any():
            worst = max(worst, float(np.max(np.abs(left[both] - right[both]))))

    exact = differences(csv_baseline, baseline, compare_columns)

    exact_mismatches = int(exact.to_numpy().sum())

    audit.record(
        "T15a",
        "CSV round-trip preserves every value within numerical tolerance",
        f"{len(compare_columns)} feature columns",
        f"0 beyond {TOLERANCE}", f"{exact_mismatches} beyond tolerance",
        exact_mismatches == 0,
        f"largest float deviation {worst:.3e}",
    )

    bitwise = 0

    for column in compare_columns:

        left = csv_baseline[column]
        right = baseline[column]

        bitwise += int((~(
            (left == right) | (left.isna() & right.isna())
        )).sum())

    audit.measure(
        "T15b",
        "Values not BIT-identical after the CSV round trip",
        f"{len(compare_columns)} feature columns",
        bitwise,
        f"Requires BOTH float_format='%.17g' on write (Instrument 3) and "
        f"float_precision='round_trip' on read. pandas defaults lose one ULP "
        f"in each direction. Largest residual deviation {worst:.3e}.",
    )


def test_t13_provenance(features, columns, audit):
    """T13 - every feature carries a documented provenance category."""

    rows = []
    uncategorised = []

    for column in columns:

        category = classify_provenance(column)

        if category not in ALLOWED_PROVENANCE:
            uncategorised.append(column)

        rows.append({"column": column, "provenance": category})

    audit.record(
        "T13a", "Every feature carries a documented provenance category",
        f"{len(columns)} features",
        "0 uncategorised", f"{len(uncategorised)} uncategorised",
        not uncategorised,
        "; ".join(uncategorised[:5]),
    )

    provenance = pd.DataFrame(rows)

    counts = provenance["provenance"].value_counts()

    for category, count in counts.items():
        audit.measure(
            "T13b", "Features by provenance category", category, int(count),
        )

    # Every category must be match-derived. None may be an FBref aggregate.
    outside = set(counts.index) - ALLOWED_PROVENANCE

    audit.record(
        "T13c", "Every provenance category is derivable from the match foundation",
        f"{len(counts)} categories in use",
        "0 outside the allowed set", f"{len(outside)} outside",
        not outside,
        str(sorted(outside)) if outside else str(sorted(counts.index)),
    )

    return provenance


def test_t14_raw_protection(audit):

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
        "T14a", "No file under data/raw/ was opened at any point",
        "runtime file-access record",
        0, len(raw_touches),
        not raw_touches,
        "; ".join(sorted(set(raw_touches))[:5]),
    )

    allowed = DECLARED_INPUTS | {
        AUDIT_OUTPUT.resolve(), AVAILABILITY_OUTPUT.resolve(),
        PROVENANCE_OUTPUT.resolve(), MISSINGNESS_OUTPUT.resolve(),
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
        "T14b", "Only the four declared inputs were read",
        "runtime file-access record",
        "0 unexpected", f"{len(unexpected)} unexpected",
        not unexpected,
        "; ".join(unexpected[:5]),
    )

    source = Path(__file__).read_text(encoding="utf-8")

    forbidden = ["read_" + "html(", "read_" + "excel("]

    found = [token for token in forbidden if token in source]

    audit.record(
        "T14c", "No FBref HTML/Excel table reader appears in this source",
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
        print(f"  {label:<36}{value}")
    else:
        print(f"  {label:<36}{value:<28}{verdict}")


def print_test_table(audit):

    print()
    print("=" * 79)
    print("VALIDATION DETAIL")
    print("=" * 79)
    print()

    markers = {"PASS": "PASS", "FAIL": "FAIL", "MEASURED": "----"}

    for row in audit.frame().itertuples():

        print(f"  {markers[row.status]}  {row.test_id:<6} {row.test}")
        print(f"              scope   : {row.scope}")
        print(f"              expected: {row.expected}")
        print(f"              observed: {row.observed}")

        if row.detail:
            print(f"              {row.detail}")


def print_missingness(missingness):

    print()
    print("=" * 79)
    print("MISSINGNESS BY DOCUMENTED CAUSE")
    print("=" * 79)
    print()
    print("  Missingness is information, not damage. Every NaN below has a")
    print("  cause, and every cause below actually produces a NaN.")
    print()

    absent = missingness[missingness["reason"] != REASON_PRESENT]

    if absent.empty:
        print("    (no missing values)")
        return

    grouped = (
        absent.groupby(["reason", "provenance"])
        .agg(columns=("column", "nunique"), values=("count", "sum"))
        .reset_index()
        .sort_values("values", ascending=False)
    )

    print(f"    {'Cause':<42}{'Provenance':<24}{'Cols':>5}{'Values':>8}")

    for row in grouped.itertuples():
        print(
            f"    {row.reason:<42}{row.provenance:<24}"
            f"{row.columns:>5}{row.values:>8}"
        )


def print_availability_curve(curve):

    print()
    print("=" * 79)
    print("AVAILABILITY BY MATCHWEEK")
    print("=" * 79)
    print()
    print("  The cold-start pattern, measured. Current-season form is dead at")
    print("  MW1 by construction; venue form warms more slowly because a team")
    print("  needs a prior match AT THAT VENUE.")
    print()
    print(f"    {'MW':>3}{'Current form':>15}{'Venue form':>13}")

    for matchweek in curve.index:

        if matchweek > 12 and matchweek % 6 != 0 and matchweek != 38:
            continue

        print(
            f"    {matchweek:>3}"
            f"{curve.loc[matchweek, 'current_form']:>14.0%}"
            f"{curve.loc[matchweek, 'venue_form']:>13.0%}"
        )


def print_redundancy(exact_duplicates, equivalent, constants):

    print()
    print("=" * 79)
    print("REDUNDANCY AND CONSTANT INVENTORY")
    print("=" * 79)
    print()
    print("  Reported only. NOTHING is removed - which features enter the")
    print("  modelling dataset is a separate decision for baseline experiments.")

    print()
    print(f"  Exact duplicate pairs: {len(exact_duplicates)}")

    for left, right in exact_duplicates[:15]:
        print(f"    {left}  ==  {right}")

    print()
    print(f"  Mathematically equivalent pairs (|r| = 1): {len(equivalent)}")

    for left, right, correlation in equivalent[:20]:
        sign = "+" if correlation > 0 else "-"
        print(f"    {left:<34}{sign}  {right}")

    print()
    print(f"  Constant features: {len(constants)}")

    for column, value, count in constants:
        print(f"    {column:<40}= {value}  ({count} present rows)")


def print_distribution_extremes(distributions):

    print()
    print("=" * 79)
    print("DISTRIBUTION SANITY - WIDEST RANGES")
    print("=" * 79)
    print()

    numeric = distributions[distributions["min"] != ""].copy()

    numeric["span"] = numeric["max"].astype(float) - numeric["min"].astype(float)

    widest = numeric.sort_values("span", ascending=False).head(14)

    print(
        f"    {'Column':<34}{'Min':>9}{'Median':>9}{'Mean':>9}{'Max':>9}"
    )

    for row in widest.itertuples():
        print(
            f"    {row.column:<34}{row.min:>9.2f}{row.median:>9.2f}"
            f"{row.mean:>9.2f}{row.max:>9.2f}"
        )


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 5: FEATURE QUALITY AND AVAILABILITY GATE")
    print("=" * 79)
    print()
    print(f"  Target     : {FEATURES_INPUT.relative_to(PROJECT_ROOT)}")
    print("  Question   : usable distributions, sensible missingness, no")
    print("               impossible values, no accidental duplication?")
    print("  Discipline : AUDIT ONLY - nothing is deleted, dropped or cleaned")
    print("  Scope      : no models, no Elo, no FBref, no feature selection")

    matches, state, features, transitions = load_inputs()

    columns = feature_columns(features)

    audit = Audit()

    print()
    print(f"  {len(features)} matches, {len(columns)} feature columns, "
          f"{len(transitions)} team-seasons.")

    print("  T1  row integrity ...")
    test_t1_row_integrity(matches, state, features, audit)

    print("  T13 provenance ...")
    provenance = test_t13_provenance(features, columns, audit)

    print("  T2  feature availability ...")
    availability = test_t2_availability(features, columns, audit)

    print("  T3  missingness reasons ...")
    missingness = test_t3_missingness_reasons(features, columns, audit)

    print("  T4  impossible values ...")
    test_t4_impossible_values(features, audit)

    print("  T5  internal identities ...")
    test_t5_internal_identities(features, audit)

    print("  T6  relative-feature correctness ...")
    test_t6_relative_features(features, state, audit)

    print("  T7  venue isolation ...")
    test_t7_venue_isolation(matches, features, audit)

    print("  T8  redundancy inventory ...")
    exact_duplicates, equivalent = test_t8_redundancy(features, columns, audit)

    print("  T9  constant features ...")
    constants, near_constants = test_t9_constant_features(
        features, columns, audit)

    print("  T10 distribution sanity ...")
    distributions = test_t10_distributions(features, columns, audit)

    print("  T11 season/matchweek availability pattern ...")
    curve = test_t11_availability_pattern(features, transitions, audit)

    print("  T12 no final-season information (5 rebuilds) ...")
    test_t12_no_final_season(matches, features, columns, audit)

    print("  T14 raw data protection ...")
    test_t14_raw_protection(audit)

    # ---- reports
    print_test_table(audit)
    print_missingness(missingness)
    print_availability_curve(curve)
    print_redundancy(exact_duplicates, equivalent, constants)
    print_distribution_extremes(distributions)

    # ---- outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    inventory = distributions.merge(provenance, on="column", how="left",
                                    suffixes=("", "_category"))

    duplicate_of = {}

    for left, right in exact_duplicates:
        duplicate_of[right] = left

    equivalent_with = {}

    for left, right, _ in equivalent:
        equivalent_with.setdefault(left, []).append(right)
        equivalent_with.setdefault(right, []).append(left)

    constant_values = {column: value for column, value, _ in constants}

    inventory["is_constant"] = inventory["column"].isin(constant_values)
    inventory["constant_value"] = inventory["column"].map(constant_values).fillna("")
    inventory["exact_duplicate_of"] = inventory["column"].map(duplicate_of).fillna("")
    inventory["equivalent_with"] = inventory["column"].map(
        lambda c: "|".join(sorted(equivalent_with.get(c, [])))
    )

    audit_frame = audit.frame()

    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")
    availability.to_csv(AVAILABILITY_OUTPUT, index=False, encoding="utf-8")
    inventory.to_csv(PROVENANCE_OUTPUT, index=False, encoding="utf-8")
    missingness.to_csv(MISSINGNESS_OUTPUT, index=False, encoding="utf-8")

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(audit_frame)} entries)")
    print(f"  {AVAILABILITY_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(availability)} scope rows)")
    print(f"  {PROVENANCE_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(inventory)} features profiled)")
    print(f"  {MISSINGNESS_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(missingness)} column-cause rows)")

    failures = audit.failures()

    def outcome(prefix):
        rows = [r for r in audit.rows if r["test_id"].startswith(prefix)]
        return status_text(all(r["status"] != "FAIL" for r in rows))

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 5")
    print("=" * 79)
    print()

    line("Matches:", f"{len(features)}")
    line("Team-side states:", f"{EXPECTED_TEAM_SIDES}")
    line("Feature columns audited:", f"{len(columns)}")
    line("T1  row integrity:", "1,900 x 2 sides", outcome("T1a"))
    line("T2  feature availability:", f"{len(availability)} scope rows",
         outcome("T2"))
    line("T3  missingness reasons:", "every NaN explained", outcome("T3"))
    line("T4  impossible values:", "none found", outcome("T4"))
    line("T5  internal identities:", "all hold", outcome("T5"))
    line("T6  relative correctness:", "rebuilt from raw state", outcome("T6"))
    line("T7  venue isolation:", "no cross-venue leak", outcome("T7"))
    line("T8  redundancy:", f"{len(exact_duplicates)} dup, "
         f"{len(equivalent)} equivalent", outcome("T8"))
    line("T9  constant features:", f"{len(constants)} found", outcome("T9"))
    line("T10 distribution sanity:", "within bounds", outcome("T10"))
    line("T11 availability pattern:", "cold start correct", outcome("T11"))
    line("T12 no final-season info:", "5 rebuilds + control", outcome("T12"))
    line("T13 provenance:", "all categorised", outcome("T13"))
    line("T14 raw data protection:", "no data/raw access", outcome("T14"))

    print()
    line("Features removed:", "0  (audit, not selection)")

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
        print(f"\n  FATAL: {error}\n\nSTATUS: FATAL\n")
        return EXIT_FATAL

    except Exception:
        print("\n  FATAL: unexpected exception\n")
        traceback.print_exc()
        print("\nSTATUS: FATAL\n")
        return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())
