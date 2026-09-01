"""
===============================================================================
PHASE 3 - INSTRUMENT 2
FEATURE BUILDER
===============================================================================

PURPOSE
    Build the modelling dataset the Phase 3 specification designed, and build
    nothing else:

        86  Phase 1 result-derived features   READ, never recomputed
      + 12  Block C  contextual               fixture list only, no new data
      + 24  Block X  lag-1 composite priors   FBref season S-1, z-scored
      ---
       122  shipped features

    Plus the metadata the specification names under Block X but does not
    count in its 24: three fbref-availability flags, two status columns and
    two source-season columns. The arithmetic is stated in the report rather
    than quietly absorbed.

INPUTS - four, all read-only
    outputs/phase1_team_strength_features.csv   the 86-feature backbone
    outputs/phase1_historical_team_state.csv    previous-match dates
    data/raw/Fixtures/*.xls                     kickoff time and weekday
    data/raw/<season>/*.xls                     FBref season aggregates

    data/raw/ is READ and never written. That is not merely asserted here, it
    is MEASURED by a runtime audit hook that records every path this process
    opens - Instrument 3's design, reused. The tests file reads the record
    for L10.

THE TWO RULES THIS INSTRUMENT EXISTS TO OBEY

    1. STRICT DATE BOUNDARY. Every historical quantity uses information whose
       date is strictly before the match date. Block C INHERITS the boundary
       rather than re-implementing it: rest days are the difference between
       the match date and Phase 1's home/away_previous_match_date, which
       Phase 1 already proved. There is deliberately no second date-boundary
       implementation in this file - two implementations of one rule is how
       the two drift apart.

    2. LAG-1 ONLY. For a match in season S the FBref aggregate for S-1 may be
       used; the aggregate for S may not. Instrument 1 measured the gap at
       75-89 days over four season pairs, checked per match, 0 violations.

WITHIN-SEASON Z-SCORES, AND WHY THE SCOPE IS THE WHOLE POINT
    Every Block P quantity is standardised across the twenty teams of ONE
    prior season and no others. A z computed over pooled seasons would carry
    the league's regime level into the feature - which, under whole-season
    test folds, is a fold fingerprint rather than a fact about a team.
    Sixteen of the raw columns drift more between seasons than teams differ
    within one (Instrument 1, T8); the within-season z is what neutralises
    them, and L11 is the probe that checks it worked.

    ddof = 0. The twenty teams are not a sample of the league that season -
    they are the league that season. sd(z) is then exactly 1, so L3's
    tolerance is not doing any silent work.

EMPTY STATE IS NaN, NEVER ZERO
    A promoted team has no prior Premier League vector. Zero is the league
    mean in z-space, so imputing zero asserts the promoted side was exactly
    average, which is the one thing it demonstrably was not. The absence is
    carried as NaN plus an explicit status, and imputation is left to the
    modelling phase where it can be fitted on training folds only.

EXTENSIBILITY - xG WITHOUT A REDESIGN
    Block P is a declarative table of (name, table, perspective, column) rows
    and Block X a declarative table of (composite, signed components). A
    lag-1 xG block is added by appending rows to BLOCK_P_QUANTITIES and
    BLOCK_X_COMPOSITES. No build function below knows any quantity by name.

WHAT IS NOT DONE HERE
    No model is trained. No Phase 1 dataset is rebuilt or rewritten. No raw
    file is written. phase0_evaluation_harness.py is not imported, not read
    and not modified. No ablation is run. This instrument proves the feature
    representation is correctly constructed - it makes no claim that any
    feature in it is useful.
===============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase1_match_foundation import (  # noqa: E402
    FIXTURES_DIR,
    SEASON_FILES,
    clean_name,
    read_fixture_table,
)
from phase3_raw_feature_audit import (  # noqa: E402
    PREV_SEASON,
    SEASON_ORDER,
    load_raw_tables,
    numeric,
)


# ============================================================
# FILE-ACCESS RECORDER  (evidence for L10)
# ============================================================
#
# Installed before any data is read. Audit hooks cannot be uninstalled, which
# is exactly the property that makes the record admissible.

_OPEN_EVENTS = []

# CPython raises the "open" audit event as (path, mode, flags). io.open fills
# in the mode string; os.open leaves mode None and fills in the flag bits.
# Both are recorded, because a write can arrive by either route and a hook
# that only understands one of them would miss the other.
_WRITE_FLAG_BITS = 0

for _flag_name in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC"):
    _WRITE_FLAG_BITS |= getattr(__import__("os"), _flag_name, 0)


def _record_open(event, args):

    if event != "open":
        return

    target = args[0]

    if not isinstance(target, (str, bytes, Path)):
        return

    mode = args[1] if len(args) > 1 else None
    flags = args[2] if len(args) > 2 else None

    _OPEN_EVENTS.append((str(target), mode, flags))


sys.addaudithook(_record_open)


def opened_events():
    """Every (path, mode, flags) this process has opened. Read by L10."""

    return list(_OPEN_EVENTS)


def opened_paths():
    """Every path this process has opened so far."""

    return [path for path, _mode, _flags in _OPEN_EVENTS]


def is_write_open(mode, flags):
    """True when an open event asked for write access, by either route."""

    if isinstance(mode, str) and any(char in mode for char in "wax+"):
        return True

    if isinstance(flags, int) and flags & _WRITE_FLAG_BITS:
        return True

    return False


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

BACKBONE_INPUT = OUTPUTS_DIR / "phase1_team_strength_features.csv"
STATE_INPUT = OUTPUTS_DIR / "phase1_historical_team_state.csv"
MATCHES_INPUT = OUTPUTS_DIR / "phase1_matches.csv"

FEATURES_OUTPUT = OUTPUTS_DIR / "phase3_features.csv"
INVENTORY_OUTPUT = OUTPUTS_DIR / "phase3_feature_inventory.csv"
BLOCK_P_OUTPUT = OUTPUTS_DIR / "phase3_block_p_priors.csv"
MISSINGNESS_OUTPUT = OUTPUTS_DIR / "phase3_missingness_reasons.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase3_feature_builder_audit.csv"

EXPECTED_MATCHES = 1900
EXPECTED_SEASON_OPENERS = 50
EXPECTED_PHASE1_FEATURES = 86
FULL_SEASON_MATCHES = 38
VENUE_MATCHES = 19

CONGESTION_WINDOW_DAYS = 14

IDENTITY_COLUMNS = ["season", "date", "matchweek", "home_team", "away_team"]

# Phase 1's vocabulary, reused verbatim. Not re-spelled, not extended.
STATUS_AVAILABLE = "available"
STATUS_ABSENT = "absent_from_previous_season"
STATUS_NO_PRIOR = "no_prior_season_in_dataset"

TARGET_SEASONS = [season for season in SEASON_ORDER if PREV_SEASON.get(season)]
SOURCE_SEASONS = sorted({PREV_SEASON[season] for season in TARGET_SEASONS})


# ============================================================
# BLOCK P - LAG-1 QUANTITIES.  Declarative, so xG is an append.
# ============================================================
#
# (name, table_type, perspective, rendered column, group)
#
# Every entry is a season aggregate for the PRIOR season. Nothing here reads
# the target season. The rendered column names are the flattened FBref
# MultiIndex paths produced by Instrument 1's loader, which types every file
# from its content and never from its filename.

BLOCK_P_QUANTITIES = [
    ("sh90",     "Shooting",      "Squad",    "Standard | Sh/90",          "attack"),
    ("sot90",    "Shooting",      "Squad",    "Standard | SoT/90",         "attack"),
    ("sot_pct",  "Shooting",      "Squad",    "Standard | SoT%",           "attack"),
    ("gsh",      "Shooting",      "Squad",    "Standard | G/Sh",           "attack"),
    ("poss",     "Standard",      "Squad",    "Unnamed: 3_level_0 | Poss", "attack"),

    ("sh90_ag",  "Shooting",      "Opponent", "Standard | Sh/90",          "defence"),
    ("sot90_ag", "Shooting",      "Opponent", "Standard | SoT/90",         "defence"),
    ("ga90",     "Goalkeeping",   "Squad",    "Performance | GA90",        "defence"),
    ("save_pct", "Goalkeeping",   "Squad",    "Performance | Save%",       "defence"),
    ("cs_pct",   "Goalkeeping",   "Squad",    "Performance | CS%",         "defence"),

    ("fls",      "Miscellaneous", "Squad",    "Performance | Fls",         "style"),
    ("crdy",     "Miscellaneous", "Squad",    "Performance | CrdY",        "style"),
    ("crdr",     "Miscellaneous", "Squad",    "Performance | CrdR",        "style"),
    ("crs",      "Miscellaneous", "Squad",    "Performance | Crs",         "style"),
    ("int",      "Miscellaneous", "Squad",    "Performance | Int",         "style"),

    ("age",      "Playing Time",  "Squad",    "Unnamed: 2_level_0 | Age",  "squad"),
    ("npl",      "Playing Time",  "Squad",    "Unnamed: 1_level_0 | # Pl", "squad"),
    ("subs",     "Playing Time",  "Squad",    "Subs | Subs",               "squad"),
    ("compl",    "Playing Time",  "Squad",    "Starts | Compl",            "squad"),
]

# gf90 is a rate over the Overall table, and the four venue quantities are
# rebuilt from the Home/Away table's W/D/L. All five are computed rather than
# read, so they are declared apart from the straight column reads above.
DERIVED_QUANTITIES = ["gf90", "home_ppm", "away_ppm", "home_gdpm", "away_gdpm"]

BLOCK_P_NAMES = [name for name, *_ in BLOCK_P_QUANTITIES] + DERIVED_QUANTITIES

# The only columns the venue quantities may be built from. Overall Pts is
# deliberately absent: it carries the points deduction and this table does
# not, and mixing the two is Phase 0 Finding 3 (re-measured as Instrument 1
# T12). L8 asserts this list is the whole venue source.
HOME_AWAY_SOURCE_COLUMNS = [
    "Home | W", "Home | D", "Home | GF", "Home | GA", "Home | MP",
    "Away | W", "Away | D", "Away | GF", "Away | GA", "Away | MP",
]


# ============================================================
# BLOCK X - COMPOSITE INDICES.  Fixed weights, declared, never fitted.
# ============================================================
#
# composite -> [(sign, block-P name), ...].  The value is the MEAN of the
# signed z-scores, so every component carries equal weight by construction.
# No weight is fitted, therefore no weight can be fitted on test data,
# therefore this block has no free parameter to leak through.

BLOCK_X_COMPOSITES = {
    "prior_attack":     [(+1, "sh90"), (+1, "sot90"), (+1, "gf90")],
    "prior_finishing":  [(+1, "sot_pct"), (+1, "gsh")],
    "prior_defence":    [(-1, "sh90_ag"), (-1, "sot90_ag"), (-1, "ga90")],
    "prior_keeping":    [(+1, "save_pct"), (+1, "cs_pct")],
    "prior_control":    [(+1, "poss"), (+1, "crs"), (-1, "fls")],
    "prior_discipline": [(+1, "crdy"), (+1, "crdr")],
    "prior_rotation":   [(+1, "subs"), (-1, "compl"), (+1, "npl"), (+1, "age")],
}

# prior_venue_split is not an equal-weight mean and is built separately.
# See build_block_x() for the exact implementation and the reasoning.
VENUE_SPLIT = "prior_venue_split"

BLOCK_X_NAMES = list(BLOCK_X_COMPOSITES) + [VENUE_SPLIT]


# ============================================================
# BLOCK C - CONTEXT.  Twelve columns, fixture list only.
# ============================================================

BLOCK_C_SIDE_COLUMNS = ["rest_days", "matches_last14", "is_season_opener"]

BLOCK_C_COLUMNS = (
    [f"{side}_{name}" for side in ("home", "away") for name in BLOCK_C_SIDE_COLUMNS]
    + [
        "rel_rest_days_diff",
        "rel_matches_last14_diff",
        "days_since_season_start",
        "kickoff_hour_bucket",
        "day_of_week",
        "rel_context_available",
    ]
)

# Kickoff buckets. FBref's Time is "HH:MM (HH:MM)" - the first is the local
# UK kickoff, the parenthesised one is the exporting browser's timezone. The
# local time is a fact about the match; the second is a fact about whoever
# downloaded the file, and is discarded.
KICKOFF_BUCKETS = [
    ("early", 0, 14),        # the 12:00 / 12:30 lunchtime slots
    ("afternoon", 14, 17),   # the 15:00 Saturday block
    ("evening", 17, 24),     # 17:30, and the 19:30 / 20:00 midweek slots
]


def block_x_columns():

    columns = []

    for side in ("home", "away"):
        columns += [f"{side}_{name}" for name in BLOCK_X_NAMES]

    columns += [f"rel_{name}_diff" for name in BLOCK_X_NAMES]

    return columns


BLOCK_X_COLUMNS = block_x_columns()

# Named by the specification, listed under Block X, NOT counted in its 24.
BLOCK_X_AVAILABILITY = [
    "home_prior_fbref_available",
    "away_prior_fbref_available",
    "rel_prior_fbref_available",
]

# Metadata, matching Phase 1's prev_season_status / prev_season_source pair.
BLOCK_X_METADATA = [
    "home_prior_status",
    "away_prior_status",
    "home_prior_source_season",
    "away_prior_source_season",
]

NEW_COLUMNS = (
    BLOCK_C_COLUMNS + BLOCK_X_COLUMNS + BLOCK_X_AVAILABILITY + BLOCK_X_METADATA)


# ============================================================
# OUTPUT ENCODING
# ============================================================

def configure_stdout():
    """Team names carry non-ASCII; Windows consoles default to cp1252."""

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
            "expected": "",
            "observed": observed,
            "status": "INFO",
            "detail": detail,
        })

    @property
    def failures(self):
        return [row for row in self.rows if row["status"] == "FAIL"]

    def frame(self):
        return pd.DataFrame(self.rows)

    def print_rows(self):

        for row in self.rows:
            print("  [{}] {:<6} {:<54} {}".format(
                row["test_id"], row["status"], str(row["test"])[:54],
                row["observed"]))


def banner(title):

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


# ============================================================
# LOADING
# ============================================================

def load_backbone():
    """Phase 1's 86 features. Read, never recomputed."""

    return pd.read_csv(
        BACKBONE_INPUT, float_precision="round_trip", parse_dates=["date"])


def load_state():
    """Phase 1's historical state - the previous-match dates Block C reuses."""

    return pd.read_csv(
        STATE_INPUT,
        float_precision="round_trip",
        parse_dates=["date", "home_previous_match_date", "away_previous_match_date"],
    )


def load_matches():

    return pd.read_csv(MATCHES_INPUT, parse_dates=["date"])


def parse_kickoff_hour(value):
    """
    "19:30 (20:30)" -> 19.

    The parenthesised time is the downloading browser's timezone and is not a
    fact about the match.
    """

    if pd.isna(value):
        return np.nan

    head = str(value).split("(")[0].strip()

    if ":" not in head:
        return np.nan

    hour = pd.to_numeric(head.split(":")[0], errors="coerce")

    return float(hour) if pd.notna(hour) else np.nan


def load_fixture_context():
    """
    Kickoff time and weekday for every played match, read through Phase 1's
    fixture reader. Phase 1 already solved the encoding trap in these files;
    this adds two columns to its output and does not re-implement it.
    """

    records = []

    for season, filename in SEASON_FILES.items():

        table = read_fixture_table(FIXTURES_DIR / filename)
        played = table[table["Score"].notna()]

        for _, row in played.iterrows():

            records.append({
                "season": season,
                "date": pd.to_datetime(
                    row["Date"], format="%Y-%m-%d", errors="coerce"),
                "home_team": clean_name(row["Home"]),
                "away_team": clean_name(row["Away"]),
                "kickoff_hour": parse_kickoff_hour(row.get("Time")),
                "fixture_day": str(row.get("Day")).strip(),
            })

    return pd.DataFrame(records)


# ============================================================
# BLOCK C - CONTEXT
# ============================================================

def bucket_for_hour(hour):

    if pd.isna(hour):
        return None

    for label, low, high in KICKOFF_BUCKETS:
        if low <= hour < high:
            return label

    return None


def congestion_counts(dates_by_team, window_days=CONGESTION_WINDOW_DAYS):
    """
    Matches in the half-open window [T-window, T), per team per date.

    HALF-OPEN IS THE WHOLE POINT. `right` is the first index at or after T, so
    a match played on T itself is excluded - and so is every other match that
    day. 1,706 of 1,900 matches share a date with another match, so an
    inclusive right edge would let two teams meeting today count each other.
    """

    counts = {}

    for team, played in dates_by_team.items():

        dates = np.sort(np.asarray(played, dtype="datetime64[ns]"))

        for date in np.unique(dates):

            window_start = date - np.timedelta64(window_days, "D")

            left = int(np.searchsorted(dates, window_start, side="left"))
            right = int(np.searchsorted(dates, date, side="left"))

            counts[(team, pd.Timestamp(date))] = right - left

    return counts


def build_block_c(backbone, state, fixture_context):
    """
    Twelve contextual columns, from dates the project already holds.

    REST DAYS ARE NOT RECOMPUTED. They are (date - Phase 1's
    previous_match_date), so the strict date < T boundary is inherited from an
    implementation that was already proved, rather than re-derived from a
    second one that could disagree with it.

    Phase 1 leaves that date NaT at a side's first match of the season, which
    is why rest_days is NaN for exactly the season openers. The summer gap is
    a different quantity from a rest advantage and is not allowed into the
    same column: 82 must never sit where 3 sits.
    """

    keys = ["season", "date", "home_team", "away_team"]

    merged = backbone[keys].merge(
        state[keys + ["home_previous_match_date", "away_previous_match_date"]],
        on=keys, how="left", validate="one_to_one")

    merged = merged.merge(
        fixture_context, on=keys, how="left", validate="one_to_one")

    block = merged[keys].copy()

    for side in ("home", "away"):

        previous = merged[f"{side}_previous_match_date"]

        block[f"{side}_rest_days"] = (
            (merged["date"] - previous).dt.days.astype("float64"))

        block[f"{side}_is_season_opener"] = previous.isna()

    # One long team-match table, so congestion is counted once per team-side
    # rather than twice with two chances to differ.
    long = pd.concat([
        merged[["date"]].assign(team=merged["home_team"]),
        merged[["date"]].assign(team=merged["away_team"]),
    ], ignore_index=True)

    counts = congestion_counts(
        {team: group["date"].to_numpy() for team, group in long.groupby("team")})

    for side in ("home", "away"):

        block[f"{side}_matches_last14"] = [
            counts[(team, date)]
            for team, date in zip(merged[f"{side}_team"], merged["date"])
        ]

    block["rel_rest_days_diff"] = block["home_rest_days"] - block["away_rest_days"]

    block["rel_matches_last14_diff"] = (
        block["home_matches_last14"] - block["away_matches_last14"])

    # The season's first match date is, by definition, on or before every
    # match of that season - so this uses no information from the future.
    season_start = merged.groupby("season")["date"].transform("min")

    block["days_since_season_start"] = (merged["date"] - season_start).dt.days

    block["kickoff_hour_bucket"] = [
        bucket_for_hour(hour) for hour in merged["kickoff_hour"]]

    block["day_of_week"] = merged["date"].dt.day_name().str.slice(0, 3)

    block["rel_context_available"] = (
        block["home_rest_days"].notna() & block["away_rest_days"].notna())

    # Carried for the audit only; dropped before emission.
    block["_fixture_day"] = merged["fixture_day"]
    block["_kickoff_hour"] = merged["kickoff_hour"]
    block["_season_start"] = season_start
    block["_home_previous_match_date"] = merged["home_previous_match_date"]
    block["_away_previous_match_date"] = merged["away_previous_match_date"]

    return block


# ============================================================
# BLOCK P - LAG-1 QUANTITIES, Z-SCORED WITHIN ONE PRIOR SEASON
# ============================================================

def zscore(values):
    """
    Standardise across the twenty teams of one season.

    ddof = 0: the twenty teams are the league that season, not a sample of it.
    sd(z) is then exactly 1.
    """

    array = np.asarray(values, dtype="float64")

    mean = np.nanmean(array)
    sd = np.nanstd(array, ddof=0)

    if not np.isfinite(sd) or sd == 0:
        return np.full(array.shape, np.nan)

    return (array - mean) / sd


def build_block_p(raw_tables, source_seasons=None):
    """
    One row per (source_season, team) - 20 teams x 4 usable prior seasons.

    Every quantity is read from the source season's own tables and
    standardised within that season only. No statistic computed here ever
    sees rows from more than one season; that is L4, and it is a property of
    this loop rather than of a downstream check that could be removed.
    """

    source_seasons = list(source_seasons or SOURCE_SEASONS)

    frames = []

    for season in source_seasons:

        overall = raw_tables[(season, "Overall", "League")]
        home_away = raw_tables[(season, "Home/Away", "League")]

        frame = pd.DataFrame({
            "source_season": season,
            "team": list(overall["Squad"]),
        })

        raw = {}

        for name, table_type, perspective, column, _group in BLOCK_P_QUANTITIES:

            table = raw_tables[(season, table_type, perspective)]
            series = numeric(table.set_index("Squad")[column])
            raw[name] = series.reindex(frame["team"]).to_numpy(dtype="float64")

        # gf90 - goals for per match, from the Overall table.
        overall_indexed = overall.set_index("Squad")

        goals_for = numeric(overall_indexed["GF"]).reindex(
            frame["team"]).to_numpy("float64")
        played = numeric(overall_indexed["MP"]).reindex(
            frame["team"]).to_numpy("float64")

        raw["gf90"] = goals_for / played

        # VENUE. Points are rebuilt from THIS TABLE'S W/D/L and never taken
        # from Overall Pts, which carries a sanction this table does not.
        venue_table = home_away.set_index("Squad").reindex(frame["team"])

        for venue in ("Home", "Away"):

            wins = numeric(venue_table[f"{venue} | W"]).to_numpy("float64")
            draws = numeric(venue_table[f"{venue} | D"]).to_numpy("float64")
            matches = numeric(venue_table[f"{venue} | MP"]).to_numpy("float64")
            scored = numeric(venue_table[f"{venue} | GF"]).to_numpy("float64")
            conceded = numeric(venue_table[f"{venue} | GA"]).to_numpy("float64")

            key = venue.lower()
            raw[f"{key}_ppm"] = (3.0 * wins + draws) / matches
            raw[f"{key}_gdpm"] = (scored - conceded) / matches
            raw[f"{key}_mp"] = matches

        for name in BLOCK_P_NAMES:
            frame[f"raw_{name}"] = raw[name]
            frame[f"z_{name}"] = zscore(raw[name])

        frame["raw_home_mp"] = raw["home_mp"]
        frame["raw_away_mp"] = raw["away_mp"]
        frame["raw_overall_mp"] = played

        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# BLOCK X - COMPOSITE INDICES
# ============================================================

def build_block_x(block_p):
    """
    Eight indices per team-season, from Block P's z-scores.

    Seven are equal-weight means of signed z-scores and are NOT
    re-standardised: their spread is a consequence of how correlated their
    components are, which is itself information, and flattening it would
    throw that away.

    THE VENUE SPLIT - THE EXACT IMPLEMENTATION, AS REQUIRED

        The specification writes

            prior_venue_split = z(z_home_ppm - z_away_ppm)

        which read literally is a z of a difference of z-scores. It is built
        literally, in three explicit steps, and does not silently become
        something else:

          1. raw home_ppm and away_ppm are computed from the Home/Away
             table's W/D/L                                 (build_block_p)
          2. each is standardised WITHIN the prior season, giving z_home_ppm
             and z_away_ppm
          3. their difference d = z_home_ppm - z_away_ppm is standardised
             within that same prior season, giving prior_venue_split

        THE OUTER STANDARDISATION IS A PURE RESCALE. Both inner z-scores have
        mean exactly 0 over the twenty teams, so d has mean exactly 0, so
        step 3 only divides by sd(d). No team's rank changes and no sign
        flips; it puts the index on the same unit scale as its siblings.

        It is NOT the same quantity as z(home_ppm - away_ppm) computed on the
        raw rates, which would weight the two venues by their raw spreads.
        The specification asked for the difference in z-space and that is
        what ships. The report prints the correlation between the two
        definitions and the two raw spreads, so the choice is visible rather
        than buried in a docstring.
    """

    frame = block_p[["source_season", "team"]].copy()

    for composite, components in BLOCK_X_COMPOSITES.items():

        stacked = np.vstack([
            sign * block_p[f"z_{name}"].to_numpy("float64")
            for sign, name in components
        ])

        # A NaN in any component gives NaN in the composite. A mean over a
        # subset of the components is a different index on a different scale,
        # not a partial answer to the same question.
        frame[composite] = np.where(
            np.isnan(stacked).any(axis=0), np.nan, stacked.mean(axis=0))

    difference = (
        block_p["z_home_ppm"].to_numpy("float64")
        - block_p["z_away_ppm"].to_numpy("float64"))

    seasons = block_p["source_season"].to_numpy()

    split = np.full(len(block_p), np.nan)

    for season in pd.unique(seasons):
        mask = seasons == season
        split[mask] = zscore(difference[mask])

    frame[VENUE_SPLIT] = split
    frame["_venue_split_z_difference"] = difference

    return frame


# ============================================================
# ATTACHING BLOCK X TO MATCHES
# ============================================================

def attach_block_x(backbone, block_x_seasons):
    """
    Map each side of each match onto its lag-1 prior vector.

    The season shift happens in exactly ONE place - PREV_SEASON - so an
    off-by-one would move every prior at once and L1 would see it. A shift
    applied per-column is a shift that can be applied to some columns and not
    to others, which is why it is not done that way.
    """

    lookup = block_x_seasons.set_index(["source_season", "team"])

    block = backbone[IDENTITY_COLUMNS].copy()

    prior_season = backbone["season"].map(PREV_SEASON)
    has_prior_season = prior_season.notna().to_numpy()

    for side in ("home", "away"):

        keys = list(zip(prior_season, backbone[f"{side}_team"]))
        present = np.array([key in lookup.index for key in keys])

        for name in BLOCK_X_NAMES:

            block[f"{side}_{name}"] = np.array([
                lookup.at[key, name] if key in lookup.index else np.nan
                for key in keys
            ], dtype="float64")

        block[f"{side}_prior_fbref_available"] = present

        block[f"{side}_prior_status"] = np.where(
            present,
            STATUS_AVAILABLE,
            np.where(has_prior_season, STATUS_ABSENT, STATUS_NO_PRIOR))

        # Phase 1 records the source season only where a prior exists and
        # leaves it NaN otherwise. Matched exactly, so the two columns can be
        # compared without a translation step in between.
        block[f"{side}_prior_source_season"] = [
            season if ok else None
            for season, ok in zip(prior_season, present)
        ]

    for name in BLOCK_X_NAMES:
        block[f"rel_{name}_diff"] = block[f"home_{name}"] - block[f"away_{name}"]

    block["rel_prior_fbref_available"] = (
        block["home_prior_fbref_available"] & block["away_prior_fbref_available"])

    return block


# ============================================================
# THE BUILD
# ============================================================

def build_phase3_blocks(backbone, state, fixture_context, raw_tables):
    """
    The new columns only - Block C and Block X - keyed by match identity.

    Kept separate from the backbone join so the perturbation tests can drive
    this function directly with modified inputs and diff its output. The
    Phase 1 backbone is never rebuilt, here or anywhere else in Phase 3.
    """

    block_c = build_block_c(backbone, state, fixture_context)
    block_p = build_block_p(raw_tables)
    block_x_seasons = build_block_x(block_p)
    block_x = attach_block_x(backbone, block_x_seasons)

    audit_columns = [column for column in block_c.columns if column.startswith("_")]

    new_blocks = pd.concat(
        [
            backbone[IDENTITY_COLUMNS].reset_index(drop=True),
            block_c[BLOCK_C_COLUMNS].reset_index(drop=True),
            block_x.drop(columns=IDENTITY_COLUMNS).reset_index(drop=True),
        ],
        axis=1,
    )

    return new_blocks, block_p, block_x_seasons, block_c[audit_columns]


def assemble(backbone, new_blocks):
    """86 + 12 + 24 + metadata, in a declared column order."""

    phase1_features = [
        column for column in backbone.columns if column not in IDENTITY_COLUMNS]

    ordered = (
        IDENTITY_COLUMNS
        + phase1_features
        + BLOCK_C_COLUMNS
        + BLOCK_X_COLUMNS
        + BLOCK_X_AVAILABILITY
        + BLOCK_X_METADATA
    )

    frame = pd.concat(
        [
            backbone.reset_index(drop=True),
            new_blocks.drop(columns=IDENTITY_COLUMNS).reset_index(drop=True),
        ],
        axis=1,
    )

    missing = [column for column in ordered if column not in frame.columns]

    if missing:
        raise SystemExit(f"FATAL: assembled frame is missing {missing}")

    return frame[ordered]


def build_everything():
    """Load, build, assemble. The one entry point the tests file drives."""

    backbone = load_backbone()
    state = load_state()
    matches = load_matches()
    fixture_context = load_fixture_context()
    raw_tables, manifest = load_raw_tables()

    new_blocks, block_p, block_x_seasons, block_c_audit = build_phase3_blocks(
        backbone, state, fixture_context, raw_tables)

    frame = assemble(backbone, new_blocks)

    return {
        "frame": frame,
        "backbone": backbone,
        "state": state,
        "matches": matches,
        "fixture_context": fixture_context,
        "raw_tables": raw_tables,
        "manifest": manifest,
        "new_blocks": new_blocks,
        "block_p": block_p,
        "block_x_seasons": block_x_seasons,
        "block_c_audit": block_c_audit,
    }


# ============================================================
# INVENTORY AND MISSINGNESS
# ============================================================

def block_of(column):

    if column in IDENTITY_COLUMNS:
        return "identity"
    if column in BLOCK_C_COLUMNS:
        return "C_context"
    if column in BLOCK_X_COLUMNS:
        return "X_prior_composite"
    if column in BLOCK_X_AVAILABILITY:
        return "X_availability"
    if column in BLOCK_X_METADATA:
        return "X_metadata"

    return "phase1_backbone"


def build_inventory(frame):

    rows = []

    for column in frame.columns:

        series = frame[column]
        is_numeric = pd.api.types.is_numeric_dtype(series)
        values = pd.to_numeric(series, errors="coerce")

        rows.append({
            "column": column,
            "block": block_of(column),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
            "min": float(values.min()) if is_numeric else "",
            "median": float(values.median()) if is_numeric else "",
            "mean": float(values.mean()) if is_numeric else "",
            "max": float(values.max()) if is_numeric else "",
            "std": float(values.std(ddof=0)) if is_numeric else "",
            "is_constant": bool(series.nunique(dropna=True) <= 1),
        })

    return pd.DataFrame(rows)


def missingness_reasons(frame):
    """
    One row per new column, with a documented cause for its NaNs.

    L5 reads this BOTH ways: every NaN must be attributable to a cause listed
    here, and every cause listed here must actually produce that many NaNs.
    A cause that explains nothing is as wrong as a NaN that nothing explains.
    """

    opener = {
        "home": frame["home_is_season_opener"],
        "away": frame["away_is_season_opener"],
    }

    either_opener = int((opener["home"] | opener["away"]).sum())

    either_no_prior = int(
        (frame["home_prior_status"].ne(STATUS_AVAILABLE)
         | frame["away_prior_status"].ne(STATUS_AVAILABLE)).sum())

    rows = []

    for column in NEW_COLUMNS:

        series = frame[column]
        nulls = int(series.isna().sum())

        if column in ("home_rest_days", "away_rest_days"):
            side = column.split("_")[0]
            cause = "season_opener_no_previous_match_this_season"
            explained = int(opener[side].sum())

        elif column == "rel_rest_days_diff":
            cause = "either_side_is_a_season_opener"
            explained = either_opener

        elif column in BLOCK_X_AVAILABILITY:
            cause = "never_missing_boolean_flag"
            explained = 0

        elif column.endswith("_prior_status"):
            cause = "never_missing_status_is_always_recorded"
            explained = 0

        elif column.endswith("_prior_source_season"):
            side = column.split("_")[0]
            cause = "no_prior_vector_so_no_source_season"
            explained = int(
                frame[f"{side}_prior_status"].ne(STATUS_AVAILABLE).sum())

        elif column.startswith(("home_prior", "away_prior")):
            side = column.split("_")[0]
            cause = "no_lag1_prior_vector_for_this_team_season"
            explained = int(
                frame[f"{side}_prior_status"].ne(STATUS_AVAILABLE).sum())

        elif column.startswith("rel_prior") and column.endswith("_diff"):
            cause = "either_side_lacks_a_lag1_prior_vector"
            explained = either_no_prior

        else:
            cause = "never_missing"
            explained = 0

        rows.append({
            "column": column,
            "block": block_of(column),
            "null_count": nulls,
            "documented_cause": cause,
            "cause_predicts": explained,
            "reconciled": bool(nulls == explained),
        })

    return pd.DataFrame(rows)


# ============================================================
# BUILDER SELF-AUDIT
# ============================================================

def self_audit(frame, block_p, block_x_seasons, block_c_audit, matches, audit):
    """
    Structural checks the builder runs on itself.

    The eleven leakage tests live in phase3_feature_builder_tests.py. These
    are the ones that must stop the build rather than report on it.
    """

    audit.record(
        "B1", "row count preserved from the Phase 1 backbone",
        EXPECTED_MATCHES, len(frame), len(frame) == EXPECTED_MATCHES)

    duplicated = int(frame.duplicated(subset=IDENTITY_COLUMNS).sum())
    audit.record("B2", "no duplicate match identifiers", 0, duplicated,
                 duplicated == 0)

    audit.record("B3", "Block C emits 12 columns", 12, len(BLOCK_C_COLUMNS),
                 len(BLOCK_C_COLUMNS) == 12)

    audit.record("B4", "Block X emits 24 composite columns", 24,
                 len(BLOCK_X_COLUMNS), len(BLOCK_X_COLUMNS) == 24)

    phase1_count = len(
        [column for column in frame.columns if block_of(column) == "phase1_backbone"])

    audit.record("B5", "Phase 1 backbone carried through unchanged",
                 EXPECTED_PHASE1_FEATURES, phase1_count,
                 phase1_count == EXPECTED_PHASE1_FEATURES)

    shipped = len(BLOCK_C_COLUMNS) + len(BLOCK_X_COLUMNS) + phase1_count
    audit.record("B6", "shipped feature count", 122, shipped, shipped == 122,
                 "86 Phase 1 + 12 Block C + 24 Block X")

    audit.measure(
        "B7", "metadata columns emitted beyond the 122",
        len(BLOCK_X_AVAILABILITY) + len(BLOCK_X_METADATA),
        "3 availability flags + 2 status + 2 source-season; named by the "
        "specification under Block X, not counted in its 24")

    audit.record("B8", "Block P covers 20 teams x 4 prior seasons", 80,
                 len(block_p), len(block_p) == 80)

    per_season = set(block_p.groupby("source_season").size())
    audit.record("B9", "exactly 20 teams in every prior season", {20},
                 per_season, per_season == {20})

    venue_mp = set(block_p["raw_home_mp"]) | set(block_p["raw_away_mp"])
    audit.record("B10", "every venue split is 19 + 19 matches",
                 {float(VENUE_MATCHES)}, venue_mp,
                 venue_mp == {float(VENUE_MATCHES)})

    overall_mp = set(block_p["raw_overall_mp"])
    audit.record("B11", "every prior season is a complete 38-match season",
                 {float(FULL_SEASON_MATCHES)}, overall_mp,
                 overall_mp == {float(FULL_SEASON_MATCHES)})

    raw_nan = int(
        block_p[[f"raw_{name}" for name in BLOCK_P_NAMES]].isna().sum().sum())
    audit.record("B12", "no NaN among the raw prior quantities", 0, raw_nan,
                 raw_nan == 0)

    openers = int(frame["home_is_season_opener"].sum())
    audit.record("B13", "home-side season openers", EXPECTED_SEASON_OPENERS,
                 openers, openers == EXPECTED_SEASON_OPENERS)

    # The weekday derived from the date must agree with the weekday FBref
    # printed in the fixture list - two sources for one fact.
    day_mismatch = int(
        (frame["day_of_week"].to_numpy()
         != block_c_audit["_fixture_day"].to_numpy()).sum())
    audit.record("B14", "weekday from date agrees with the fixture list Day",
                 0, day_mismatch, day_mismatch == 0)

    unbucketed = int(frame["kickoff_hour_bucket"].isna().sum())
    audit.record("B15", "every kickoff falls in a declared bucket", 0,
                 unbucketed, unbucketed == 0)

    # Congestion recounted by a deliberately different method - a brute-force
    # date filter over the match list, rather than searchsorted over sorted
    # per-team arrays. Two methods agreeing is evidence; one method checked
    # against itself is not.
    long = pd.concat([
        matches[["date"]].assign(team=matches["home_team"]),
        matches[["date"]].assign(team=matches["away_team"]),
    ], ignore_index=True)

    by_team = {team: group["date"] for team, group in long.groupby("team")}
    window = pd.Timedelta(days=CONGESTION_WINDOW_DAYS)

    mismatches = 0

    for side in ("home", "away"):
        emitted = frame[f"{side}_matches_last14"].to_numpy()
        for position, (team, date) in enumerate(
                zip(frame[f"{side}_team"], frame["date"])):
            dates = by_team[team]
            recount = int(((dates >= date - window) & (dates < date)).sum())
            mismatches += int(recount != emitted[position])

    audit.record("B16", "congestion agrees with an independent recount", 0,
                 mismatches, mismatches == 0,
                 "brute-force date filter vs searchsorted")

    # Block X availability must agree with the Phase 1 flag it sits beside.
    for side in ("home", "away"):
        disagreements = int(
            (frame[f"{side}_prior_fbref_available"]
             != frame[f"{side}_prev_season_available"]).sum())
        audit.record(
            f"B17{side[0]}",
            f"{side} FBref availability agrees with Phase 1's flag",
            0, disagreements, disagreements == 0)

    return audit


# ============================================================
# REPORT
# ============================================================

def print_coverage(frame):

    print("Team-sides by prior status, home side:")
    print()
    print(frame.groupby(["season", "home_prior_status"]).size()
          .unstack(fill_value=0).to_string())
    print()

    teams = frame[["season", "home_team", "home_prior_status"]].drop_duplicates(
        subset=["season", "home_team"])

    print("Distinct teams per season by prior status:")
    print()
    print(teams.groupby(["season", "home_prior_status"]).size()
          .unstack(fill_value=0).to_string())
    print()


def print_block_summary(frame):

    print("{:<34} {:>9} {:>7} {:>11} {:>11}".format(
        "column", "non-null", "null", "mean", "sd"))
    print("-" * 76)

    for column in BLOCK_C_COLUMNS + BLOCK_X_COLUMNS:

        values = pd.to_numeric(frame[column], errors="coerce")
        populated = int(values.notna().sum())

        print("{:<34} {:>9} {:>7} {:>11} {:>11}".format(
            column,
            int(frame[column].notna().sum()),
            int(frame[column].isna().sum()),
            "-" if not populated else "{:.4f}".format(values.mean()),
            "-" if not populated else "{:.4f}".format(values.std(ddof=0)),
        ))

    print()


def print_venue_split_report(block_p, block_x_seasons):

    alternative = np.full(len(block_p), np.nan)
    seasons = block_p["source_season"].to_numpy()

    raw_difference = (
        block_p["raw_home_ppm"].to_numpy("float64")
        - block_p["raw_away_ppm"].to_numpy("float64"))

    for season in pd.unique(seasons):
        mask = seasons == season
        alternative[mask] = zscore(raw_difference[mask])

    shipped = block_x_seasons[VENUE_SPLIT].to_numpy("float64")

    print("  shipped     : z(z_home_ppm - z_away_ppm)   [the specification]")
    print("  alternative : z(home_ppm - away_ppm)       [raw-rate difference]")
    print()
    print("  correlation between the two definitions: {:.6f}".format(
        float(np.corrcoef(shipped, alternative)[0, 1])))
    print()

    for season in pd.unique(seasons):
        mask = seasons == season
        print("  {}   sd(home_ppm) {:.4f}   sd(away_ppm) {:.4f}".format(
            season,
            float(np.std(block_p["raw_home_ppm"].to_numpy()[mask], ddof=0)),
            float(np.std(block_p["raw_away_ppm"].to_numpy()[mask], ddof=0))))

    print()
    print("  The two definitions differ exactly to the extent those two")
    print("  spreads differ. The specification asked for the difference in")
    print("  z-space, and that is what ships.")
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 3 - INSTRUMENT 2: FEATURE BUILDER")

    print("  Phase 1 backbone : {}".format(BACKBONE_INPUT))
    print("  Phase 1 state    : {}".format(STATE_INPUT))
    print("  Fixtures         : {}".format(FIXTURES_DIR))
    print("  FBref aggregates : {}".format(RAW_DIR))
    print()

    built = build_everything()

    frame = built["frame"]
    block_p = built["block_p"]
    block_x_seasons = built["block_x_seasons"]

    print("  backbone rows {}, columns {}".format(
        len(built["backbone"]), built["backbone"].shape[1]))
    print("  raw FBref tables loaded {}".format(len(built["raw_tables"])))
    print("  prior seasons used as sources: {}".format(", ".join(SOURCE_SEASONS)))
    print("  assembled columns {}".format(frame.shape[1]))
    print()

    banner("1. BUILDER SELF-AUDIT")

    audit = Audit()
    self_audit(frame, block_p, block_x_seasons, built["block_c_audit"],
               built["matches"], audit)
    audit.print_rows()

    banner("2. PRIOR-SEASON COVERAGE")
    print_coverage(frame)

    banner("3. THE NEW COLUMNS")
    print_block_summary(frame)

    banner("4. VENUE SPLIT - THE TWO DEFINITIONS")
    print_venue_split_report(block_p, block_x_seasons)

    banner("5. MISSINGNESS")

    reasons = missingness_reasons(frame)

    with_nulls = reasons[reasons["null_count"] > 0]

    print(with_nulls[["column", "null_count", "documented_cause", "reconciled"]]
          .to_string(index=False))
    print()
    print("  columns with unreconciled missingness: {}".format(
        int((~reasons["reconciled"]).sum())))
    print()

    banner("6. WRITING OUTPUTS")

    inventory = build_inventory(frame)

    frame.to_csv(FEATURES_OUTPUT, index=False, encoding="utf-8",
                 float_format="%.17g")
    inventory.to_csv(INVENTORY_OUTPUT, index=False, encoding="utf-8")
    block_p.merge(block_x_seasons, on=["source_season", "team"]).to_csv(
        BLOCK_P_OUTPUT, index=False, encoding="utf-8", float_format="%.17g")
    reasons.to_csv(MISSINGNESS_OUTPUT, index=False, encoding="utf-8")
    audit.frame().to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    for path in (FEATURES_OUTPUT, INVENTORY_OUTPUT, BLOCK_P_OUTPUT,
                 MISSINGNESS_OUTPUT, AUDIT_OUTPUT):
        print("  {}".format(path))
    print()

    banner("PHASE 3 - INSTRUMENT 2 STATUS")

    failures = audit.failures
    phase1_count = len(
        [column for column in frame.columns if block_of(column) == "phase1_backbone"])

    print("  Matches                    : {}".format(len(frame)))
    print("  Phase 1 features carried   : {}".format(phase1_count))
    print("  Block C columns            : {}".format(len(BLOCK_C_COLUMNS)))
    print("  Block X composite columns  : {}".format(len(BLOCK_X_COLUMNS)))
    print("  SHIPPED FEATURES           : {}".format(
        phase1_count + len(BLOCK_C_COLUMNS) + len(BLOCK_X_COLUMNS)))
    print("  Metadata columns           : {}".format(
        len(BLOCK_X_AVAILABILITY) + len(BLOCK_X_METADATA)))
    print("  Identity columns           : {}".format(len(IDENTITY_COLUMNS)))
    print("  Builder checks run         : {}".format(len(audit.rows)))
    print("  Builder checks failed      : {}".format(len(failures)))
    print()
    print("  {}".format("PASS" if not failures else "FAIL"))
    print()
    print("No raw file was written. No Phase 1 dataset was rebuilt or rewritten.")
    print("No model was trained. The evaluation harness was not touched.")
    print("This proves the representation is correctly constructed. It makes no")
    print("claim that any feature in it is useful - that is the B0-B6 ladder.")
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
