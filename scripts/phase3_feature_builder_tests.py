"""
===============================================================================
PHASE 3 - INSTRUMENT 2: THE ELEVEN LEAKAGE TESTS
===============================================================================

The specification (section 7) names eleven tests that must pass before any
Phase 3 feature file is frozen. This is those eleven, written so that each one
FAILS if the thing it names is wrong.

    L1   index shift - the prior is season S-1, and it predates the match
    L2   perturbation, both directions, with the null control
    L3   z-score scope - twenty teams, one season, deletion-invariant
    L4   no cross-season normalisation
    L5   missingness explained in both directions
    L6   rest days obey date < T strictly
    L7   congestion window is half-open, [T-14, T)
    L8   venue points come from W/D/L, never from Overall Pts
    L9   no constant, no duplicate - on the EMITTED schema
    L10  file-access recorder - data/raw is read, never written
    L11  fold-fingerprint probe - is season identity still recoverable?

THE HABIT THIS FILE IS BUILT AROUND

    Three pieces of code in this project have run, exited 0, and been wrong.
    Every test below therefore carries a CONTROL: a second measurement that
    would come out differently if the test were incapable of failing.

      - L2 has the null control Instrument 5 learned the hard way, plus a
        zero-perturbation rebuild, because a builder that ignored its input
        entirely would sail through the "nothing changed" half.
      - L4 computes the pooled z-score it is supposed to have avoided, and
        requires it to DIFFER from what shipped.
      - L7 recounts the congestion window with an inclusive right edge and
        requires that to differ, so the half-open edge is shown to be
        load-bearing rather than decorative.
      - L8 rebuilds venue points the wrong way and requires exactly the two
        sanctioned team-seasons to move.
      - L11 runs the same probe on the raw, un-z-scored quantities, so a
        "cannot detect the season" verdict is backed by a demonstration that
        the probe can detect the season when it is there.

WHAT THIS FILE DOES NOT DO

    It writes no feature file, trains no production model, and does not
    import, read or modify phase0_evaluation_harness.py. L11 fits a
    classifier, but it is an instrument: it never sees H/D/A, it never leaves
    this process, and its output is a verdict rather than a probability.
===============================================================================
"""

from pathlib import Path
import copy
import hashlib
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase3_feature_builder as builder  # noqa: E402

from phase3_feature_builder import (  # noqa: E402
    BLOCK_C_COLUMNS,
    BLOCK_P_NAMES,
    BLOCK_P_QUANTITIES,
    BLOCK_X_AVAILABILITY,
    BLOCK_X_COLUMNS,
    BLOCK_X_COMPOSITES,
    BLOCK_X_METADATA,
    BLOCK_X_NAMES,
    CONGESTION_WINDOW_DAYS,
    HOME_AWAY_SOURCE_COLUMNS,
    IDENTITY_COLUMNS,
    NEW_COLUMNS,
    OUTPUTS_DIR,
    PREV_SEASON,
    RAW_DIR,
    SEASON_ORDER,
    SOURCE_SEASONS,
    STATUS_AVAILABLE,
    VENUE_SPLIT,
    Audit,
    banner,
    build_block_p,
    build_block_x,
    build_everything,
    build_phase3_blocks,
    congestion_counts,
    configure_stdout,
    missingness_reasons,
    numeric,
    zscore,
)


LEAKAGE_OUTPUT = OUTPUTS_DIR / "phase3_leakage_audit.csv"
PROBE_OUTPUT = OUTPUTS_DIR / "phase3_fold_fingerprint_probe.csv"
REDUNDANCY_OUTPUT = OUTPUTS_DIR / "phase3_redundancy_findings.csv"

# Redundancies the emitted schema is KNOWN to contain, each with the reason it
# is emitted anyway. L9 fails on any redundancy that is not on this list, so
# adding a block later cannot introduce one silently.
#
# All three are consequences of one fact about the fixture list: every team's
# first match of a season falls in matchweek 1, so the 50 openers are the same
# 50 matches for both sides, and "this side has no current-season form" and
# "this side has no rest-day figure" are the same 50 rows.
DECLARED_REDUNDANCIES = {
    ("away_is_season_opener", "home_is_season_opener"):
        "a match is an opener for both sides or for neither - no season in "
        "this dataset has a team's first fixture outside matchweek 1",

    ("rel_context_available", "rel_form_available"):
        "Phase 1's rel_form_available is true exactly when neither side is a "
        "season opener, which is the same predicate rel_context_available "
        "carries; kept because Block C is specified as twelve columns and "
        "dropping one is a modelling decision for the ablation",

    ("home_is_season_opener", "rel_context_available"):
        "same predicate, negated",

    ("away_is_season_opener", "rel_context_available"):
        "same predicate, negated",

    ("away_is_season_opener", "rel_form_available"):
        "same predicate, negated",

    ("home_is_season_opener", "rel_form_available"):
        "same predicate, negated",
}

PHASE0_LEAKAGE_AUDIT = OUTPUTS_DIR / "phase0_leakage_audit.csv"

TOLERANCE = 1e-12

# Sanctioned team-seasons, from Phase 0 Finding 3 / Instrument 1 T11-T12.
SANCTIONED = {("2023-2024", "Everton"), ("2023-2024", "Nottingham")}

PERMUTATIONS = 2000
PROBE_SEED = 20260830


# ============================================================
# COMPARISON HELPERS
# ============================================================

def cell_differences(left, right, columns):
    """
    Count differing cells between two frames, treating NaN == NaN as equal.

    An absence that stays an absence is not a change; without that rule every
    perturbation test would report the 608 missing priors as differences and
    drown the signal it is looking for.
    """

    total = 0
    per_column = {}

    for column in columns:

        a = pd.Series(left[column].to_numpy())
        b = pd.Series(right[column].to_numpy())

        both_missing = a.isna().to_numpy() & b.isna().to_numpy()
        equal = both_missing | a.eq(b).to_numpy()

        differing = int((~equal).sum())

        if differing:
            per_column[column] = differing

        total += differing

    return total, per_column


def rows_changed(left, right, columns, mask=None):
    """Number of ROWS in which at least one of `columns` differs."""

    changed = np.zeros(len(left), dtype=bool)

    for column in columns:

        a = pd.Series(left[column].to_numpy())
        b = pd.Series(right[column].to_numpy())

        both_missing = a.isna().to_numpy() & b.isna().to_numpy()
        changed |= ~(both_missing | a.eq(b).to_numpy())

    if mask is not None:
        changed = changed & np.asarray(mask)

    return int(changed.sum())


def sha256(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


def hash_tree(root, pattern="**/*"):

    return {
        str(path): sha256(path)
        for path in sorted(Path(root).glob(pattern))
        if path.is_file()
    }


# ============================================================
# PERTURBATION HELPERS
# ============================================================

def perturb_season_tables(raw_tables, season, scale=1.5, shift=3.0):
    """
    Rewrite every fully-numeric stat column of one season's FBref tables.

    Match counts are left alone. Perturbing MP would change the denominators
    as well as the numerators and make a "did anything move" test pass for
    the wrong reason; leaving them fixed keeps the perturbation aimed at the
    statistics themselves.
    """

    perturbed = copy.deepcopy(raw_tables)

    protected = ("MP", "90s", "Min")

    for key, frame in list(perturbed.items()):

        if key[0] != season:
            continue

        for column in frame.columns:

            if column == "Squad":
                continue

            if column.split(" | ")[-1].strip() in protected:
                continue

            values = numeric(frame[column])

            if values.isna().any():
                continue

            perturbed[key][column] = values * scale + shift

    return perturbed


def perturb_results(backbone, matches, season, home_goals=9, away_goals=0):
    """
    Rewrite every result in one season, exactly as the specification asks.

    The backbone's result-derived columns are overwritten too, not just the
    goals - if any Phase 3 column read the Phase 1 features rather than the
    match identity, this would move it.
    """

    new_backbone = backbone.copy()
    new_matches = matches.copy()

    season_rows = new_backbone["season"] == season

    for column in new_backbone.columns:

        if column in IDENTITY_COLUMNS:
            continue

        if pd.api.types.is_bool_dtype(new_backbone[column]):
            new_backbone.loc[season_rows, column] = True

        elif pd.api.types.is_numeric_dtype(new_backbone[column]):
            new_backbone.loc[season_rows, column] = float(home_goals)

    match_rows = new_matches["season"] == season
    new_matches.loc[match_rows, "home_goals"] = home_goals
    new_matches.loc[match_rows, "away_goals"] = away_goals
    new_matches.loc[match_rows, "result"] = "H"

    return new_backbone, new_matches


def rebuild(built, backbone=None, state=None, fixture_context=None, raw_tables=None):
    """Re-run the Phase 3 blocks with any subset of the inputs replaced."""

    new_blocks, block_p, block_x_seasons, block_c_audit = build_phase3_blocks(
        backbone if backbone is not None else built["backbone"],
        state if state is not None else built["state"],
        fixture_context if fixture_context is not None else built["fixture_context"],
        raw_tables if raw_tables is not None else built["raw_tables"],
    )

    return new_blocks


# ============================================================
# L1 - INDEX SHIFT
# ============================================================

def test_l1(built, audit):

    banner("L1  INDEX SHIFT - the prior is season S-1, and it predates the match")

    frame = built["frame"]
    matches = built["matches"]

    availability = matches.groupby("season")["date"].max()

    position = {season: index for index, season in enumerate(SEASON_ORDER)}

    for side in ("home", "away"):

        source = frame[f"{side}_prior_source_season"]
        populated = source.notna()

        expected = frame["season"].map(PREV_SEASON)

        mismatched = int((source[populated] != expected[populated]).sum())

        audit.record(
            f"L1a-{side}", f"{side} prior_source_season == PREV_SEASON[match season]",
            0, mismatched, mismatched == 0,
            f"{int(populated.sum())} populated rows")

        shift = [
            position[frame["season"].iloc[index]] - position[source.iloc[index]]
            for index in np.flatnonzero(populated.to_numpy())
        ]

        wrong_shift = int(sum(1 for value in shift if value != 1))

        audit.record(
            f"L1b-{side}", f"{side} prior is exactly one season earlier",
            0, wrong_shift, wrong_shift == 0,
            "index shift measured in SEASON_ORDER, not inferred from the label")

        prior_available = source[populated].map(availability)
        match_dates = frame.loc[populated, "date"]

        late = int((match_dates <= prior_available.to_numpy()).sum())

        audit.record(
            f"L1c-{side}", f"{side} prior availability date < match date, strictly",
            0, late, late == 0,
            "availability date = last match of the prior season")

        gap = (match_dates - prior_available.to_numpy()).dt.days

        audit.measure(
            f"L1d-{side}", f"{side} smallest gap between prior and match",
            f"{int(gap.min())} days", "the tightest case in the dataset")

        # Absence must be recorded consistently: no source season exactly
        # where no prior vector exists.
        status_absent = frame[f"{side}_prior_status"].ne(STATUS_AVAILABLE)
        inconsistent = int((status_absent != source.isna()).sum())

        audit.record(
            f"L1e-{side}", f"{side} source season is NaN exactly where no prior",
            0, inconsistent, inconsistent == 0)

    return audit


# ============================================================
# L2 - PERTURBATION, BOTH DIRECTIONS
# ============================================================

def test_l2(built, audit):

    banner("L2  PERTURBATION - both directions, with the null controls")

    print("  The specification writes L2 in terms of RESULTS. For the Phase 1")
    print("  backbone that is the right medium, and Instrument 5 already ran")
    print("  it there. For the NEW blocks it is not: Block X is derived from")
    print("  the FBref season aggregate and Block C from dates, and neither")
    print("  reads a scoreline. So L2 is run in both media - results AND the")
    print("  medium that actually carries the information - and the null")
    print("  control is placed where it can actually fire. L2b as literally")
    print("  written cannot hold for Block X; L2b-fbref is the test it means.")
    print()

    frame = built["frame"]
    baseline = built["new_blocks"]

    new_columns = [c for c in NEW_COLUMNS if c in baseline.columns]

    # ---- L2 zero-perturbation control -----------------------------------
    # Data_V records Instrument 5 reporting 12,822 phantom changes because it
    # compared a CSV-loaded baseline against an in-memory rebuild. The fix is
    # permanent: rebuild from untouched inputs first, and require zero.
    untouched = rebuild(built)

    total, per_column = cell_differences(baseline, untouched, new_columns)

    audit.record(
        "L2z", "NULL CONTROL: zero perturbation gives zero differences",
        0, total, total == 0,
        "in-memory rebuild vs in-memory baseline; the trap Instrument 5 hit")

    # ---- L2a target-season FBref, must NOT move -------------------------
    total_a = 0

    for season in SEASON_ORDER:

        if PREV_SEASON.get(season) is None:
            continue

        perturbed = perturb_season_tables(built["raw_tables"], season)
        rebuilt = rebuild(built, raw_tables=perturbed)

        rows = (frame["season"] == season).to_numpy()

        moved = rows_changed(baseline, rebuilt, BLOCK_X_COLUMNS, mask=rows)
        total_a += moved

        print("    {}  target-season aggregate perturbed -> {} rows moved".format(
            season, moved))

    audit.record(
        "L2a", "target-season aggregate perturbed: lag-1 priors do NOT move",
        0, total_a, total_a == 0,
        "4 target seasons; the same-season aggregate never enters a feature")

    # ---- L2b prior-season FBref, MUST move (the null control) -----------
    print()

    failures_b = []

    for season in SEASON_ORDER:

        prior = PREV_SEASON.get(season)

        if prior is None:
            continue

        perturbed = perturb_season_tables(built["raw_tables"], prior)
        rebuilt = rebuild(built, raw_tables=perturbed)

        rows = (frame["season"] == season).to_numpy()
        available = frame["home_prior_fbref_available"].to_numpy() & rows

        moved = rows_changed(baseline, rebuilt, BLOCK_X_COLUMNS, mask=available)
        expected = int(available.sum())

        if moved != expected:
            failures_b.append((season, moved, expected))

        print("    {}  prior-season aggregate perturbed -> {} of {} rows moved"
              .format(season, moved, expected))

    audit.record(
        "L2b-fbref", "NULL CONTROL: prior-season aggregate perturbed, priors DO move",
        "every available row", "0 shortfalls" if not failures_b else failures_b,
        not failures_b,
        "without this, a builder that ignored its input would pass L2a")

    # ---- L2c results rewritten to 9-0, must NOT move --------------------
    print()

    total_c = 0

    for season in SEASON_ORDER:

        new_backbone, _new_matches = perturb_results(
            built["backbone"], built["matches"], season)

        rebuilt = rebuild(built, backbone=new_backbone)

        rows = (frame["season"] == season).to_numpy()

        moved = rows_changed(baseline, rebuilt, new_columns, mask=rows)
        total_c += moved

    audit.record(
        "L2c", "every result in season S rewritten 9-0: new blocks do NOT move",
        0, total_c, total_c == 0,
        "5 seasons; the specification's L2(a), run over Block C and Block X")

    audit.measure(
        "L2c-note", "why L2(b) on results cannot fire for the new blocks",
        "vacuous by construction",
        "no Block C or Block X column reads a scoreline: Block C is dates, "
        "Block X is the FBref aggregate. L2b-fbref and L2d are the live "
        "controls. Phase 1's own result perturbation is Instrument 5 T12b/T12d.")

    # ---- L2d Block C's own null control: move the input dates -----------
    shifted_state = built["state"].copy()

    shifted_state["home_previous_match_date"] = (
        shifted_state["home_previous_match_date"] - pd.Timedelta(days=1))

    rebuilt = rebuild(built, state=shifted_state)

    populated = baseline["home_rest_days"].notna().to_numpy()

    delta = (rebuilt["home_rest_days"].to_numpy()
             - baseline["home_rest_days"].to_numpy())

    exactly_one = int(np.nansum(delta[populated] == 1.0))

    audit.record(
        "L2d", "NULL CONTROL: previous-match date -1 day moves rest days by +1",
        int(populated.sum()), exactly_one, exactly_one == int(populated.sum()),
        "proves Block C reads its input rather than reproducing a constant")

    away_moved = rows_changed(baseline, rebuilt, ["away_rest_days"])

    audit.record(
        "L2e", "perturbing the home date leaves the away side alone",
        0, away_moved, away_moved == 0)

    return audit


# ============================================================
# L3 - Z-SCORE SCOPE
# ============================================================

def test_l3(built, audit):

    banner("L3  Z-SCORE SCOPE - twenty teams, one season, deletion-invariant")

    block_p = built["block_p"]

    z_columns = [f"z_{name}" for name in BLOCK_P_NAMES]

    bad_rows, bad_mean, bad_sd = [], [], []

    for season, group in block_p.groupby("source_season"):

        if len(group) != 20:
            bad_rows.append((season, len(group)))

        for column in z_columns:

            values = group[column].to_numpy("float64")

            if abs(np.nanmean(values)) > TOLERANCE:
                bad_mean.append((season, column, float(np.nanmean(values))))

            if abs(np.nanstd(values, ddof=0) - 1.0) > TOLERANCE:
                bad_sd.append((season, column, float(np.nanstd(values, ddof=0))))

    audit.record("L3a", "exactly 20 team rows in every prior season",
                 0, len(bad_rows), not bad_rows, str(bad_rows))

    audit.record("L3b", f"every z has mean 0 within its season (tol {TOLERANCE:g})",
                 0, len(bad_mean), not bad_mean, str(bad_mean[:3]))

    audit.record("L3c", f"every z has sd 1 within its season (tol {TOLERANCE:g})",
                 0, len(bad_sd), not bad_sd, str(bad_sd[:3]))

    # Deletion invariance: build each season on its own, and build every
    # three-season subset. If any statistic crossed a season boundary, the
    # numbers would move when a neighbour is removed.
    solo_mismatches = 0

    for season in SOURCE_SEASONS:

        solo = build_block_p(built["raw_tables"], source_seasons=[season])
        full = block_p[block_p["source_season"] == season].reset_index(drop=True)

        for column in z_columns:
            if not np.array_equal(
                    solo[column].to_numpy("float64"),
                    full[column].to_numpy("float64"),
                    equal_nan=True):
                solo_mismatches += 1

    audit.record(
        "L3d", "a season built alone is bit-identical to the same season built with all",
        0, solo_mismatches, solo_mismatches == 0,
        f"{len(SOURCE_SEASONS)} seasons x {len(z_columns)} z-columns")

    drop_mismatches = 0

    for dropped in SOURCE_SEASONS:

        kept = [season for season in SOURCE_SEASONS if season != dropped]
        subset = build_block_p(built["raw_tables"], source_seasons=kept)

        for season in kept:

            left = subset[subset["source_season"] == season].reset_index(drop=True)
            right = block_p[block_p["source_season"] == season].reset_index(drop=True)

            for column in z_columns:
                if not np.array_equal(
                        left[column].to_numpy("float64"),
                        right[column].to_numpy("float64"),
                        equal_nan=True):
                    drop_mismatches += 1

    audit.record(
        "L3e", "removing an unrelated season leaves every z bit-identical",
        0, drop_mismatches, drop_mismatches == 0,
        "the specification's deletion test, run for all four drop choices")

    return audit


# ============================================================
# L4 - NO CROSS-SEASON NORMALISATION
# ============================================================

def test_l4(built, audit):

    banner("L4  NO CROSS-SEASON NORMALISATION")

    block_p = built["block_p"]

    z_columns = [f"z_{name}" for name in BLOCK_P_NAMES]
    raw_columns = [f"raw_{name}" for name in BLOCK_P_NAMES]

    # THE CONTROL. Compute the pooled z the rule forbids, and require that it
    # is measurably DIFFERENT from what shipped. A test that cannot tell the
    # forbidden thing from the permitted one is not a test.
    pooled = {}

    for name in BLOCK_P_NAMES:
        pooled[name] = zscore(block_p[f"raw_{name}"].to_numpy("float64"))

    identical = [
        name for name in BLOCK_P_NAMES
        if np.allclose(pooled[name], block_p[f"z_{name}"].to_numpy("float64"),
                       rtol=0, atol=1e-9, equal_nan=True)
    ]

    audit.record(
        "L4a", "CONTROL: the pooled z differs from every shipped z",
        0, len(identical), not identical,
        "if these matched, L3's within-season checks would be measuring nothing")

    # A pooled z leaves per-season means away from zero; a within-season z
    # cannot. This is the same fact from the other side.
    pooled_offsets = []

    for name in BLOCK_P_NAMES:
        for season, group in block_p.groupby("source_season"):
            mask = (block_p["source_season"] == season).to_numpy()
            pooled_offsets.append(abs(float(np.nanmean(pooled[name][mask]))))

    audit.measure(
        "L4b", "largest per-season mean the pooled z would have left behind",
        "{:.4f}".format(max(pooled_offsets)),
        "the shipped z leaves 0 by construction - L3b measures it at < 1e-12")

    # Every quantity that enters a feature is read from one season's table.
    # Measured by rebuilding season-by-season, which L3d already did; here
    # the claim is stated over the RAW inputs as well as the z-scores.
    raw_mismatches = 0

    for season in SOURCE_SEASONS:

        solo = build_block_p(built["raw_tables"], source_seasons=[season])
        full = block_p[block_p["source_season"] == season].reset_index(drop=True)

        for column in raw_columns:
            if not np.array_equal(
                    solo[column].to_numpy("float64"),
                    full[column].to_numpy("float64"),
                    equal_nan=True):
                raw_mismatches += 1

    audit.record(
        "L4c", "no raw quantity is computed over rows spanning two seasons",
        0, raw_mismatches, raw_mismatches == 0,
        "season-by-season rebuild is bit-identical for the raw inputs too")

    # Block X inherits the scope: every composite is a function of same-season
    # z-scores only, so rebuilding one season alone reproduces it exactly.
    composite_mismatches = 0

    for season in SOURCE_SEASONS:

        solo = build_block_x(
            build_block_p(built["raw_tables"], source_seasons=[season]))
        full = built["block_x_seasons"]
        full = full[full["source_season"] == season].reset_index(drop=True)

        for name in BLOCK_X_NAMES:
            if not np.array_equal(
                    solo[name].to_numpy("float64"),
                    full[name].to_numpy("float64"),
                    equal_nan=True):
                composite_mismatches += 1

    audit.record(
        "L4d", "every Block X composite is season-local too",
        0, composite_mismatches, composite_mismatches == 0,
        "includes the venue split, whose outer z is also within-season")

    return audit


# ============================================================
# L5 - MISSINGNESS EXPLAINED BOTH WAYS
# ============================================================

def test_l5(built, audit):

    banner("L5  MISSINGNESS EXPLAINED IN BOTH DIRECTIONS")

    frame = built["frame"]

    reasons = missingness_reasons(frame)

    unreconciled = reasons[~reasons["reconciled"]]

    audit.record(
        "L5a", "every new column's NaN count equals its documented cause",
        0, len(unreconciled), len(unreconciled) == 0,
        str(list(unreconciled["column"])[:5]))

    # FORWARD: each NaN is attributable. Set equality, not just count
    # equality - the same number of NaNs in the wrong rows is still wrong.
    opener = {side: frame[f"{side}_is_season_opener"].to_numpy()
              for side in ("home", "away")}

    predicates = {}

    for side in ("home", "away"):
        predicates[f"{side}_rest_days"] = opener[side]
        no_prior = frame[f"{side}_prior_status"].ne(STATUS_AVAILABLE).to_numpy()
        predicates[f"{side}_prior_source_season"] = no_prior
        for name in BLOCK_X_NAMES:
            predicates[f"{side}_{name}"] = no_prior

    predicates["rel_rest_days_diff"] = opener["home"] | opener["away"]

    either_no_prior = (
        frame["home_prior_status"].ne(STATUS_AVAILABLE).to_numpy()
        | frame["away_prior_status"].ne(STATUS_AVAILABLE).to_numpy())

    for name in BLOCK_X_NAMES:
        predicates[f"rel_{name}_diff"] = either_no_prior

    wrong_rows = []

    for column, predicate in predicates.items():
        if not np.array_equal(frame[column].isna().to_numpy(), predicate):
            wrong_rows.append(column)

    audit.record(
        "L5b", "the NaN ROWS are exactly the rows the cause predicts",
        0, len(wrong_rows), not wrong_rows, str(wrong_rows[:5]))

    # BACKWARD: a cause that explains nothing is as wrong as an unexplained
    # NaN. Every documented cause must actually produce the NaNs it claims.
    idle_causes = reasons[
        (reasons["cause_predicts"] > 0) & (reasons["null_count"] == 0)]

    audit.record(
        "L5c", "no documented cause fails to produce its NaNs",
        0, len(idle_causes), len(idle_causes) == 0,
        str(list(idle_causes["column"])[:5]))

    never_missing = reasons[reasons["documented_cause"].str.startswith("never_missing")]
    violated = never_missing[never_missing["null_count"] > 0]

    audit.record(
        "L5d", "columns declared never-missing carry no NaN",
        0, len(violated), len(violated) == 0,
        str(list(violated["column"])[:5]))

    # And nothing outside the documented list carries a NaN.
    covered = set(reasons["column"])
    stray = [
        column for column in NEW_COLUMNS
        if column not in covered and frame[column].isna().any()
    ]

    audit.record(
        "L5e", "no new column escapes the missingness ledger",
        0, len(stray), not stray, str(stray))

    total_nan = int(frame[NEW_COLUMNS].isna().sum().sum())
    total_explained = int(reasons["cause_predicts"].sum())

    audit.record(
        "L5f", "total NaN across the new blocks equals total explained",
        total_explained, total_nan, total_nan == total_explained)

    return audit


# ============================================================
# L6 - REST DAYS OBEY date < T
# ============================================================

def test_l6(built, audit):

    banner("L6  REST DAYS OBEY date < T, STRICTLY")

    frame = built["frame"]
    state = built["state"]
    audit_columns = built["block_c_audit"]

    for side in ("home", "away"):

        previous = audit_columns[f"_{side}_previous_match_date"]
        populated = previous.notna().to_numpy()

        not_strict = int(
            (previous[populated].to_numpy()
             >= frame.loc[populated, "date"].to_numpy()).sum())

        audit.record(
            f"L6a-{side}", f"{side} previous_match_date < match date, strictly",
            0, not_strict, not_strict == 0,
            f"{int(populated.sum())} populated rows")

        expected = (frame.loc[populated, "date"].to_numpy()
                    - previous[populated].to_numpy())
        expected_days = pd.to_timedelta(expected).days.to_numpy().astype("float64")

        emitted = frame.loc[populated, f"{side}_rest_days"].to_numpy("float64")

        wrong = int((emitted != expected_days).sum())

        audit.record(
            f"L6b-{side}", f"{side} rest days equal (match date - previous date)",
            0, wrong, wrong == 0, "reused from Phase 1, not recomputed")

        below_one = int((emitted < 1).sum())

        audit.record(
            f"L6c-{side}", f"{side} no populated rest day is below 1", 0,
            below_one, below_one == 0)

        missing = int(frame[f"{side}_rest_days"].isna().sum())

        audit.record(
            f"L6d-{side}", f"{side} rest days missing on exactly the season openers",
            50, missing, missing == 50)

    # The 50 must be Phase 0's 50, read from Phase 0's own artefact rather
    # than re-asserted here as a literal.
    phase0 = pd.read_csv(PHASE0_LEAKAGE_AUDIT)
    cold_start = phase0[phase0["test_id"] == "T4"]

    phase0_count = int(cold_start.iloc[0]["observed"]) if len(cold_start) else -1

    audit.record(
        "L6e", "the NaN rows match Phase 0's cold-start count",
        phase0_count, int(frame["home_rest_days"].isna().sum()),
        phase0_count == int(frame["home_rest_days"].isna().sum()),
        "read from outputs/phase0_leakage_audit.csv T4, not hardcoded")

    opener_rows = frame[frame["home_rest_days"].isna()]

    matchweeks = set(opener_rows["matchweek"])
    per_season = opener_rows.groupby("season").size().to_dict()

    audit.record(
        "L6f", "every rest-day NaN is a matchweek-1 match", {1}, matchweeks,
        matchweeks == {1})

    audit.record(
        "L6g", "ten of them in each of the five seasons",
        "10 per season", per_season,
        set(per_season.values()) == {10} and len(per_season) == 5)

    both_sides = int(
        (frame["home_rest_days"].isna() == frame["away_rest_days"].isna()).sum())

    audit.record(
        "L6h", "home and away openers are the same 50 matches",
        len(frame), both_sides, both_sides == len(frame))

    return audit


# ============================================================
# L7 - CONGESTION WINDOW IS HALF-OPEN
# ============================================================

def test_l7(built, audit):

    banner("L7  CONGESTION WINDOW IS HALF-OPEN, [T-14, T)")

    frame = built["frame"]
    matches = built["matches"]

    long = pd.concat([
        matches[["date"]].assign(team=matches["home_team"]),
        matches[["date"]].assign(team=matches["away_team"]),
    ], ignore_index=True)

    by_team = {team: group["date"] for team, group in long.groupby("team")}
    window = pd.Timedelta(days=CONGESTION_WINDOW_DAYS)

    half_open_wrong = 0
    inclusive_differs = 0

    for side in ("home", "away"):

        emitted = frame[f"{side}_matches_last14"].to_numpy()

        for position, (team, date) in enumerate(
                zip(frame[f"{side}_team"], frame["date"])):

            dates = by_team[team]

            half_open = int(((dates >= date - window) & (dates < date)).sum())
            inclusive = int(((dates >= date - window) & (dates <= date)).sum())

            half_open_wrong += int(half_open != emitted[position])
            inclusive_differs += int(inclusive != emitted[position])

    audit.record(
        "L7a", "emitted count equals a [T-14, T) recount", 0, half_open_wrong,
        half_open_wrong == 0, "independent brute-force recount")

    audit.record(
        "L7b", "CONTROL: an inclusive right edge would change the answer",
        "> 0", inclusive_differs, inclusive_differs > 0,
        "the half-open edge is load-bearing, not decorative")

    # Same-day perturbation: give one team a phantom extra match on a date it
    # already plays, and require nothing to move.
    team = frame["home_team"].iloc[0]
    dates = by_team[team].to_numpy()
    target = dates[len(dates) // 2]

    baseline_counts = congestion_counts({team: dates})

    same_day = congestion_counts({team: np.append(dates, target)})

    # THE CLAIM IS ABOUT THE PERTURBED DATE ITSELF. A match played on T does
    # not count towards T's own congestion - that is what the half-open right
    # edge means. It does count towards LATER matches within fourteen days,
    # because by then it is a match that was actually played, and a test that
    # forbade that would be testing arithmetic rather than the boundary.
    at_target = int(same_day[(team, pd.Timestamp(target))]
                    - baseline_counts[(team, pd.Timestamp(target))])

    audit.record(
        "L7c", "a phantom SAME-DAY match does not change that date's own count",
        0, at_target, at_target == 0,
        f"team {team}, date {pd.Timestamp(target).date()}")

    # And every count it does move must be a strictly later date inside the
    # window - never an earlier one, which would be the future leaking back.
    illegitimate = [
        key for key, value in baseline_counts.items()
        if same_day[key] != value
        and not (pd.Timestamp(target) < key[1]
                 <= pd.Timestamp(target) + pd.Timedelta(days=CONGESTION_WINDOW_DAYS))
    ]

    audit.record(
        "L7c2", "it moves only strictly-later dates inside the window",
        0, len(illegitimate), not illegitimate, str(illegitimate[:3]))

    # And the null control: a phantom match the day BEFORE must be counted.
    day_before = target - np.timedelta64(1, "D")
    earlier = congestion_counts({team: np.append(dates, day_before)})

    moved_up = earlier[(team, pd.Timestamp(target))] - baseline_counts[
        (team, pd.Timestamp(target))]

    audit.record(
        "L7d", "NULL CONTROL: a phantom match one day earlier adds exactly 1",
        1, moved_up, moved_up == 1,
        "without this, a counter that returned a constant would pass L7c")

    # Phase 0 recorded Time as CAUTION: scheduled in advance, but never to be
    # used to order matches within a day. It is used here only as a bucket,
    # and that is measured rather than promised.
    scrambled = built["fixture_context"].copy()
    scrambled["kickoff_hour"] = (scrambled["kickoff_hour"] * 0 + 11.0)

    rebuilt = rebuild(built, fixture_context=scrambled)

    order_sensitive = [
        column for column in BLOCK_C_COLUMNS if column != "kickoff_hour_bucket"]

    moved = rows_changed(built["new_blocks"], rebuilt, order_sensitive)

    audit.record(
        "L7e", "kickoff time is never used to order or slice history",
        0, moved, moved == 0,
        "Phase 0 registry marks Time CAUTION for exactly this reason")

    bucket_moved = rows_changed(
        built["new_blocks"], rebuilt, ["kickoff_hour_bucket"])

    audit.record(
        "L7f", "CONTROL: the bucket itself does move when the time changes",
        "> 0", bucket_moved, bucket_moved > 0)

    return audit


# ============================================================
# L8 - SANCTION CONSISTENCY
# ============================================================

def test_l8(built, audit):

    banner("L8  SANCTION CONSISTENCY - venue points from W/D/L, never Overall Pts")

    tables = built["raw_tables"]
    block_p = built["block_p"]

    forbidden = [c for c in HOME_AWAY_SOURCE_COLUMNS if c.endswith("Pts")]

    audit.record(
        "L8a", "no Pts column appears in the declared venue source list",
        0, len(forbidden), not forbidden, str(HOME_AWAY_SOURCE_COLUMNS))

    # Recompute venue points independently and require an exact match.
    wrong = 0

    for season in SOURCE_SEASONS:

        table = tables[(season, "Home/Away", "League")].set_index("Squad")
        rows = block_p[block_p["source_season"] == season]

        for venue in ("home", "away"):

            label = venue.capitalize()

            wins = numeric(table[f"{label} | W"]).reindex(rows["team"]).to_numpy()
            draws = numeric(table[f"{label} | D"]).reindex(rows["team"]).to_numpy()
            played = numeric(table[f"{label} | MP"]).reindex(rows["team"]).to_numpy()

            expected = (3.0 * wins + draws) / played

            wrong += int((rows[f"raw_{venue}_ppm"].to_numpy() != expected).sum())

    audit.record(
        "L8b", "venue points per match equal (3W + D) / MP from the venue table",
        0, wrong, wrong == 0, "recomputed independently over 80 team-seasons")

    # The venue table's own Pts column agrees with W/D/L everywhere - which is
    # what makes it safe to derive from, and why the disagreement below is
    # attributable to the Overall table rather than to this one.
    venue_pts_wrong = 0
    disagreements = set()

    for season in SEASON_ORDER:

        venue_table = tables[(season, "Home/Away", "League")].set_index("Squad")
        overall = tables[(season, "Overall", "League")].set_index("Squad")

        for team in overall.index:

            home_pts = float(numeric(venue_table[f"Home | Pts"]).loc[team])
            away_pts = float(numeric(venue_table[f"Away | Pts"]).loc[team])

            derived_home = float(
                3 * numeric(venue_table["Home | W"]).loc[team]
                + numeric(venue_table["Home | D"]).loc[team])
            derived_away = float(
                3 * numeric(venue_table["Away | W"]).loc[team]
                + numeric(venue_table["Away | D"]).loc[team])

            venue_pts_wrong += int(home_pts != derived_home)
            venue_pts_wrong += int(away_pts != derived_away)

            overall_pts = float(numeric(overall["Pts"]).loc[team])

            if home_pts + away_pts != overall_pts:
                disagreements.add((season, team))

    audit.record(
        "L8c", "the venue table's own Pts equals 3W + D for all 100 team-seasons",
        0, venue_pts_wrong, venue_pts_wrong == 0,
        "so the derivation and the table agree; the sanction is elsewhere")

    audit.record(
        "L8d", "Home Pts + Away Pts disagrees with Overall Pts for exactly two",
        sorted(SANCTIONED), sorted(disagreements), disagreements == SANCTIONED,
        "Everton and Nottingham Forest, 2023-24 - the two point deductions")

    # THE CONTROL. Build the venue rate the forbidden way and require the two
    # sanctioned teams - and only them - to move.
    season = "2023-2024"

    venue_table = tables[(season, "Home/Away", "League")].set_index("Squad")
    overall = tables[(season, "Overall", "League")].set_index("Squad")
    rows = block_p[block_p["source_season"] == season]

    shipped = rows["raw_home_ppm"].to_numpy()

    home_share = numeric(venue_table["Home | Pts"]).reindex(rows["team"]).to_numpy()
    away_share = numeric(venue_table["Away | Pts"]).reindex(rows["team"]).to_numpy()
    overall_pts = numeric(overall["Pts"]).reindex(rows["team"]).to_numpy()

    # The naive mistake: scale the venue split so it reconciles with Overall.
    contaminated = (
        home_share * (overall_pts / (home_share + away_share)) / 19.0)

    moved_teams = {
        team for team, a, b in zip(rows["team"], shipped, contaminated) if a != b}

    audit.record(
        "L8e", "CONTROL: reconciling to Overall Pts moves exactly the sanctioned two",
        {team for _s, team in SANCTIONED}, moved_teams,
        moved_teams == {team for _s, team in SANCTIONED},
        "shows the test would fire if Overall Pts had leaked into the venue rate")

    return audit


# ============================================================
# L9 - NO CONSTANT, NO DUPLICATE, ON THE EMITTED SCHEMA
# ============================================================

def test_l9(built, audit):
    """
    L9 on the emitted schema, with the findings written down rather than
    acted on.

    The rule this test enforces is NOT "there are no redundancies". Running
    it honestly shows that there are, and two different kinds:

      - ones Phase 1 already found, recorded in phase1_feature_provenance.csv
        and deliberately kept. Phase 1 is frozen and out of scope here, so
        those are re-derived and checked to be EXACTLY the known set - a new
        one appearing among them would mean the backbone had changed.

      - ones Block C introduces, which are new and are declared below with
        their reason. They are emitted anyway, because the brief specifies
        twelve Block C columns and because dropping a feature is a modelling
        decision that belongs to the ablation, where its cost can be
        measured rather than assumed. Phase 1's Instrument 5 set that
        precedent: reported, nothing removed.

    So the assertion is that every redundancy present is a DECLARED one. An
    undeclared redundancy - from a future xG block, say - fails the run.
    """

    banner("L9  NO CONSTANT / NO DUPLICATE - on the EMITTED schema")

    frame = built["frame"]

    shipped = [
        column for column in frame.columns
        if builder.block_of(column) in
        ("phase1_backbone", "C_context", "X_prior_composite")
    ]

    new_shipped = [
        column for column in shipped
        if builder.block_of(column) in ("C_context", "X_prior_composite")
    ]

    # ---- constants ------------------------------------------------------
    constants = [
        column for column in shipped if frame[column].nunique(dropna=True) <= 1]

    new_constants = [column for column in constants if column in new_shipped]

    audit.record(
        "L9a", "no NEW feature is constant", 0, len(new_constants),
        not new_constants, str(new_constants))

    provenance = pd.read_csv(OUTPUTS_DIR / "phase1_feature_provenance.csv")

    known_constants = set(
        provenance.loc[provenance["is_constant"].astype(bool), "column"])

    audit.record(
        "L9b", "constants are exactly the ones Phase 1 already recorded",
        sorted(known_constants), sorted(constants),
        set(constants) == known_constants,
        "read from phase1_feature_provenance.csv; a new one would mean the "
        "frozen backbone had moved")

    # ---- duplicates and equivalences ------------------------------------
    numeric_columns = [
        column for column in shipped
        if pd.api.types.is_numeric_dtype(frame[column])
    ]

    correlations = frame[numeric_columns].astype("float64").corr()

    duplicate_pairs = []
    equivalent_pairs = []

    for index, left in enumerate(numeric_columns):
        for right in numeric_columns[index + 1:]:

            if frame[left].equals(frame[right]):
                duplicate_pairs.append((left, right))
                continue

            r = correlations.loc[left, right]

            if pd.notna(r) and abs(r) > 0.99999:
                equivalent_pairs.append((left, right, float(r)))

    known_equivalent = {
        tuple(sorted((row["column"], row["equivalent_with"])))
        for _, row in provenance.dropna(subset=["equivalent_with"]).iterrows()
    }

    backbone_equivalent = {
        tuple(sorted((left, right)))
        for left, right, _r in equivalent_pairs
        if builder.block_of(left) == builder.block_of(right) == "phase1_backbone"
    }

    audit.record(
        "L9c", "backbone equivalences are exactly the ones Phase 1 recorded",
        len(known_equivalent), len(backbone_equivalent),
        backbone_equivalent == known_equivalent,
        "4 pairs, each a column and the same column divided by 38")

    undeclared = []

    for left, right in duplicate_pairs:
        if tuple(sorted((left, right))) not in DECLARED_REDUNDANCIES:
            undeclared.append((left, right, "exact duplicate"))

    for left, right, _r in equivalent_pairs:
        pair = tuple(sorted((left, right)))
        if pair in known_equivalent or pair in DECLARED_REDUNDANCIES:
            continue
        undeclared.append((left, right, "|r| > 0.99999"))

    audit.record(
        "L9d", "every redundancy in the emitted schema is a declared one",
        0, len(undeclared), not undeclared, str(undeclared[:4]))

    # ---- the findings, written down -------------------------------------
    findings = []

    for left, right in duplicate_pairs:
        findings.append({
            "left": left, "right": right, "relation": "exact_duplicate",
            "r": 1.0,
            "left_block": builder.block_of(left),
            "right_block": builder.block_of(right),
            "origin": ("phase1_known"
                       if tuple(sorted((left, right))) in known_equivalent
                       else "phase3_new"),
            "reason": DECLARED_REDUNDANCIES.get(
                tuple(sorted((left, right))), ""),
        })

    for left, right, r in equivalent_pairs:
        pair = tuple(sorted((left, right)))
        findings.append({
            "left": left, "right": right, "relation": "equivalent",
            "r": r,
            "left_block": builder.block_of(left),
            "right_block": builder.block_of(right),
            "origin": "phase1_known" if pair in known_equivalent else "phase3_new",
            "reason": DECLARED_REDUNDANCIES.get(pair, ""),
        })

    for column in constants:
        findings.append({
            "left": column, "right": "", "relation": "constant", "r": "",
            "left_block": builder.block_of(column), "right_block": "",
            "origin": "phase1_known" if column in known_constants else "phase3_new",
            "reason": "constant by construction - every prior is a full season",
        })

    findings_frame = pd.DataFrame(findings)
    findings_frame.to_csv(REDUNDANCY_OUTPUT, index=False, encoding="utf-8")

    new_findings = findings_frame[findings_frame["origin"] == "phase3_new"]

    audit.measure(
        "L9e", "redundancies introduced by the NEW blocks", len(new_findings),
        "; ".join(f"{row.left} ~ {row.right}" if row.right else str(row.left)
                  for row in new_findings.itertuples()))

    audit.measure(
        "L9f", "availability flags duplicated between Phase 1 and Phase 3",
        sum(1 for side in ("home", "away")
            if frame[f"{side}_prior_fbref_available"].equals(
                frame[f"{side}_prev_season_available"])),
        "reported, not removed: the agreement IS the cross-check (builder B17)")

    print()
    print("  Redundancies found in the emitted schema:")
    print()
    print(findings_frame[["left", "right", "relation", "origin"]]
          .to_string(index=False))
    print()
    print("  Findings are REPORTED, not acted on. Which features earn a place")
    print("  in a model is a modelling decision, and it belongs to the")
    print("  ablation where the cost of dropping one can be measured.")
    print()
    print("  {}".format(REDUNDANCY_OUTPUT))
    print()

    return audit


# ============================================================
# L10 - FILE-ACCESS RECORDER
# ============================================================

def test_l10(built, audit, raw_before, outputs_before):

    banner("L10  FILE-ACCESS RECORDER - data/raw is read, never written")

    events = builder.opened_events()

    raw_root = str(RAW_DIR)

    raw_events = [
        (path, mode, flags) for path, mode, flags in events
        if str(Path(path).resolve()).startswith(raw_root)
    ]

    audit.record(
        "L10a", "data/raw was actually read", "> 0", len(raw_events),
        len(raw_events) > 0, "recorded by a sys.addaudithook, not asserted")

    raw_writes = [
        (path, mode, flags) for path, mode, flags in raw_events
        if builder.is_write_open(mode, flags)
    ]

    audit.record(
        "L10b", "no open under data/raw asked for write access",
        0, len(raw_writes), not raw_writes, str(raw_writes[:3]))

    raw_after = hash_tree(RAW_DIR)

    changed = [path for path in raw_before if raw_before[path] != raw_after.get(path)]
    added = [path for path in raw_after if path not in raw_before]

    audit.record(
        "L10c", "every file under data/raw is byte-identical afterwards",
        0, len(changed) + len(added), not changed and not added,
        f"{len(raw_before)} files hashed with SHA-256 before and after")

    # Phase 0 and Phase 1 artefacts are inputs to this phase and must survive
    # it untouched. Hashes, not promises.
    outputs_after = {
        path: sha256(path)
        for path in map(str, sorted(OUTPUTS_DIR.glob("phase[01]_*")))
        if Path(path).is_file()
    }

    outputs_changed = [
        path for path in outputs_before
        if outputs_before[path] != outputs_after.get(path)
    ]

    audit.record(
        "L10d", "no Phase 0 or Phase 1 output was modified",
        0, len(outputs_changed), not outputs_changed,
        f"{len(outputs_before)} artefacts hashed; {str(outputs_changed[:3])}")

    harness_touched = [
        path for path, _mode, _flags in events
        if "phase0_evaluation_harness" in path
    ]

    audit.record(
        "L10e", "phase0_evaluation_harness.py was never opened",
        0, len(harness_touched), not harness_touched)

    audit.measure(
        "L10f", "distinct paths opened by this process",
        len({path for path, _m, _f in events}),
        "hook installed before the first read; audit hooks cannot be removed")

    return audit


# ============================================================
# L11 - FOLD-FINGERPRINT PROBE
# ============================================================

def leave_one_out_nearest_centroid(features, labels):
    """
    Leave-one-out accuracy of the most trivial linear classifier there is.

    For each held-out row the class centroids are rebuilt without it, closed
    form, so there is no optimiser, no seed and no hyper-parameter to tune -
    and therefore nothing in the probe that could be tuned until it produced
    the answer someone wanted.
    """

    classes = np.unique(labels)

    sums = np.array([features[labels == c].sum(axis=0) for c in classes])
    counts = np.array([(labels == c).sum() for c in classes], dtype="float64")

    correct = 0

    for index in range(len(labels)):

        own = int(np.flatnonzero(classes == labels[index])[0])

        adjusted_sums = sums.copy()
        adjusted_counts = counts.copy()

        adjusted_sums[own] -= features[index]
        adjusted_counts[own] -= 1

        centroids = adjusted_sums / adjusted_counts[:, None]

        distance = ((centroids - features[index]) ** 2).sum(axis=1)

        correct += int(classes[int(distance.argmin())] == labels[index])

    return correct / len(labels)


def standardise(matrix):
    """Pooled standardisation - scale only, so between-season shifts survive."""

    matrix = np.asarray(matrix, dtype="float64")
    sd = matrix.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)

    return (matrix - matrix.mean(axis=0)) / sd


def test_l11(built, audit):

    banner("L11  FOLD-FINGERPRINT PROBE - is the season still recoverable?")

    print("  A leakage instrument, not a prediction model. It never sees H/D/A,")
    print("  it never touches phase0_evaluation_harness.py, nothing it fits")
    print("  leaves this process, and its output is a verdict. Reported")
    print("  explicitly because the brief said no models.")
    print()

    block_p = built["block_p"]
    block_x = built["block_x_seasons"]

    labels = block_p["source_season"].to_numpy()
    n_classes = len(np.unique(labels))
    chance = 1.0 / n_classes

    composites = standardise(block_x[BLOCK_X_NAMES].to_numpy("float64"))
    z_scores = standardise(
        block_p[[f"z_{name}" for name in BLOCK_P_NAMES]].to_numpy("float64"))
    raw = standardise(
        block_p[[f"raw_{name}" for name in BLOCK_P_NAMES]].to_numpy("float64"))

    observed = leave_one_out_nearest_centroid(composites, labels)
    z_observed = leave_one_out_nearest_centroid(z_scores, labels)
    raw_observed = leave_one_out_nearest_centroid(raw, labels)

    # The permutation null. Season labels are shuffled and the whole probe
    # re-run, so the reference distribution comes from this data and this
    # classifier rather than from an assumed chance level.
    rng = np.random.default_rng(PROBE_SEED)

    null = np.array([
        leave_one_out_nearest_centroid(composites, rng.permutation(labels))
        for _ in range(PERMUTATIONS)
    ])

    p_value = float((null >= observed).sum() + 1) / (PERMUTATIONS + 1)

    print("  rows {} team-seasons, {} classes, chance {:.3f}".format(
        len(labels), n_classes, chance))
    print()
    print("    Block X composites  (within-season z)  accuracy {:.4f}".format(observed))
    print("    Block P z-scores    (within-season z)  accuracy {:.4f}".format(z_observed))
    print("    raw quantities      (POSITIVE CONTROL) accuracy {:.4f}".format(raw_observed))
    print()
    print("    permutation null over {} shuffles: mean {:.4f}, 95th pct {:.4f}".format(
        PERMUTATIONS, float(null.mean()), float(np.percentile(null, 95))))
    print("    one-sided p for Block X: {:.4f}".format(p_value))
    print()

    # THE CONTROL COMES FIRST. A "cannot detect the season" verdict is worth
    # nothing unless the probe demonstrably can detect it when it is present.
    audit.record(
        "L11a", "CONTROL: the probe recovers the season from RAW quantities",
        f"> {chance:.3f}", "{:.4f}".format(raw_observed), raw_observed > chance,
        "the drift Instrument 1 measured at T8, seen directly")

    verdict = "CLEAN" if p_value > 0.05 else "LEAK"

    audit.record(
        "L11b", "season identity is NOT recoverable from Block X",
        "p > 0.05", "{:.4f} (accuracy {:.4f})".format(p_value, observed),
        p_value > 0.05,
        f"VERDICT: {verdict}")

    audit.record(
        "L11c", "nor from the Block P z-scores underneath it",
        f"<= {chance:.3f}", "{:.4f}".format(z_observed), z_observed <= chance)

    audit.measure(
        "L11d", "why the accuracy is 0 rather than merely at chance",
        "{:.4f}".format(observed),
        "within-season z makes every season's centroid the zero vector; under "
        "leave-one-out the held-out row's own centroid is pulled away from it, "
        "so the true class is the LEAST likely guess. Below chance here is the "
        "signature of no regime information, not of a broken probe.")

    probe = pd.DataFrame([
        {"representation": "block_x_composites", "accuracy": observed,
         "n": len(labels), "classes": n_classes, "chance": chance,
         "permutation_p": p_value, "verdict": verdict},
        {"representation": "block_p_zscores", "accuracy": z_observed,
         "n": len(labels), "classes": n_classes, "chance": chance,
         "permutation_p": "", "verdict": ""},
        {"representation": "raw_quantities_control", "accuracy": raw_observed,
         "n": len(labels), "classes": n_classes, "chance": chance,
         "permutation_p": "", "verdict": "POSITIVE CONTROL"},
    ])

    probe.to_csv(PROBE_OUTPUT, index=False, encoding="utf-8")

    print("  probe detail written to {}".format(PROBE_OUTPUT))
    print()

    return audit


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 3 - INSTRUMENT 2: THE ELEVEN LEAKAGE TESTS")

    # Hashed BEFORE anything is built, so L10 can compare.
    raw_before = hash_tree(RAW_DIR)
    outputs_before = {
        str(path): sha256(path)
        for path in sorted(OUTPUTS_DIR.glob("phase[01]_*"))
        if path.is_file()
    }

    print("  raw files hashed        : {}".format(len(raw_before)))
    print("  phase 0/1 outputs hashed: {}".format(len(outputs_before)))
    print()

    built = build_everything()

    print("  built {} rows x {} columns".format(*built["frame"].shape))

    audit = Audit()

    test_l1(built, audit)
    test_l2(built, audit)
    test_l3(built, audit)
    test_l4(built, audit)
    test_l5(built, audit)
    test_l6(built, audit)
    test_l7(built, audit)
    test_l8(built, audit)
    test_l9(built, audit)
    test_l10(built, audit, raw_before, outputs_before)
    test_l11(built, audit)

    banner("RESULTS")

    audit.print_rows()

    frame = audit.frame()
    frame.to_csv(LEAKAGE_OUTPUT, index=False, encoding="utf-8")

    tests = frame[frame["status"] != "INFO"]
    failures = audit.failures

    banner("PHASE 3 - INSTRUMENT 2 LEAKAGE STATUS")

    for label in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"):

        group = tests[tests["test_id"].str.startswith(label + "-")
                      | tests["test_id"].str.match(rf"^{label}[a-z]")
                      | tests["test_id"].eq(label)]

        passed = int((group["status"] == "PASS").sum())
        failed = int((group["status"] == "FAIL").sum())

        print("  {:<5} {:>2} assertions   {}".format(
            label, len(group), "PASS" if failed == 0 and passed else
            ("FAIL" if failed else "no assertions")))

    print()
    print("  Assertions run    : {}".format(len(tests)))
    print("  Assertions failed : {}".format(len(failures)))
    print("  Measurements      : {}".format(int((frame["status"] == "INFO").sum())))
    print()
    print("  {}".format("PASS" if not failures else "FAIL"))
    print()
    print("  {}".format(LEAKAGE_OUTPUT))
    print()
    print("This establishes that the feature representation is correctly")
    print("constructed, reproducible and leakage-safe. It establishes nothing")
    print("whatever about whether any of these features is useful.")
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
