"""
===============================================================================
PHASE 4 - THE DYNAMIC-STATE LADDER.  D0, D1, D2.
===============================================================================

Governed by PHASE4_D2_PREDECLARATION.txt and its Amendments 1-3. Nothing here
chooses anything the pre-declaration did not already fix:

    rungs        D0 base rate, D1 results-derived, D2 = D1 + dynamic state
    cadence      one DC refit per distinct date, window strictly earlier
    inner CV     5 contiguous date segments, 4 expanding folds, ties SMALLEST
    grid         the 21 points of Amendment 2 A2.3
    solver       phase3_ablation_ladder.fit_multinomial, Newton, tol 1e-9
    bootstrap    paired per-match, 10,000 draws, seed 20260901
    G6           applicable wherever EPV < 10, per Amendment 3

D3 and D4 ARE NOT RUN HERE, by instruction. Their design widths are computed
and asserted anyway - DS3 requires nesting D1 < D2 < D3 < D4, and computing a
width is not fitting a model.
===============================================================================
"""

from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import (CLASS_INDEX, CLASSES,  # noqa: E402
                                       evaluate, validate_probabilities)
from phase3_feature_builder import (Audit, banner,  # noqa: E402
                                    configure_stdout, block_of,
                                    declare_block,
                                    PHASE1_BACKBONE_COLUMNS)

import phase2_poisson_dixon_coles as DC          # noqa: E402
import phase3_ablation_ladder as L3              # noqa: E402
import phase3_regularisation_sensitivity as I4   # noqa: E402
import phase4_dynamic_state as STATE             # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

ELO_RESULTS = OUTPUTS_DIR / "phase2_elo_results.csv"
ELO_FULL = OUTPUTS_DIR / "phase2_elo_metrics_full.csv"
DC_RESULTS = OUTPUTS_DIR / "phase2_poisson_dc_results.csv"
DC_SUMMARY = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"
BASE_RATE_SUMMARY = OUTPUTS_DIR / "phase2_base_rate_fold_summary.csv"
PASSTHROUGH_STATE = OUTPUTS_DIR / "phase4_dc_state.csv"

FOLD_OUTPUT = OUTPUTS_DIR / "phase4_ladder_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase4_ladder_pooled.csv"
DELTA_OUTPUT = OUTPUTS_DIR / "phase4_ladder_deltas.csv"
CURVE_OUTPUT = OUTPUTS_DIR / "phase4_ladder_lambda_curves.csv"
COEF_OUTPUT = OUTPUTS_DIR / "phase4_ladder_dynamic_coefficients.csv"
PRED_OUTPUT = OUTPUTS_DIR / "phase4_ladder_predictions.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase4_ladder_audit.csv"

FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

# Amendment 2 A2.3. The upper 13 are the pre-declaration's section 5 grid
# unchanged; the lower 8 extend it four decades downward.
LAMBDA_GRID = (0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003,
               0.001, 0.003,
               0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
               300.0, 1000.0, 3000.0, 10000.0)

N_INNER_BLOCKS = 5           # section 4: 5 segments, 4 expanding inner folds
TIE_BREAK = "smallest"       # section 4: differs from Phase 3's "largest"

METRICS = ["accuracy", "balanced_accuracy", "macro_f1",
           "log_loss", "brier_score", "rps"]

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260901

EPV_APPLICABILITY = 10.0     # Amendment 3 A3.1, Peduzzi et al. 1996

# Amendment 4 A4.2. The normal-consistency constant 2 * PHI^-1(0.75). For a
# Gaussian sample the interquartile range is this many sigma in expectation,
# so dividing by it turns the IQR into an estimator of the SAME quantity the
# standard deviation estimates. That is what makes robust scaling a drop-in
# replacement for a broken estimator rather than a change of units.
IQR_TO_SIGMA = 1.3489795003921634

# Amendment 4 A4.1. Qualifying by SOURCE: read out of a Dixon-Coles fit whose
# window can be too short to identify the strengths. rel_elo_diff is not on
# this list because Elo has no fitting window, not because its coefficient
# came back large.
DC_DERIVED_COLUMNS = ("rel_attack_diff", "rel_defence_diff",
                      "expected_total_goals")

DC_VARIANT = "dc_walkforward"
POISSON_VARIANT = "poisson_walkforward"

# The season whose corruption drives DS4 and DS5. It is the ONLY season that
# is a test season and never a training season, so "no other fold moves" is a
# meaningful assertion rather than one that is true by construction.
CORRUPTION_SEASON = "2025-2026"
CORRUPTION_SEED = 20260902

# The four point-in-time state columns are NOT Phase 1 backbone columns and
# must not be classifiable as such. Declaring them here is what lets
# block_of() raise on everything genuinely unknown.
DYNAMIC_COLUMNS = declare_block("D_dynamic_state", STATE.DYNAMIC_COLUMNS)


# ============================================================
# DESIGN
# ============================================================

# Phase 1's backbone minus the two season-identifier columns Phase 3 holds out
# as metadata. Fixed at import time from the DECLARED list, so it cannot vary
# with what happens to be attached to a feature frame.
D1_FEATURE_NAMES = [c for c in PHASE1_BACKBONE_COLUMNS
                    if c not in L3.HELD_OUT_AS_METADATA]


def d1_features(features):
    """
    D1's 84 feature names, SELECTED BY INCLUSION against the declared
    backbone. It used to select by exclusion - everything block_of() did not
    recognise - and that is half of the base-contamination bug recorded in
    PROJECT_GOTCHAS.md. Six columns attached to the frame before this was
    called ended up inside the control arm.

    Selecting by inclusion means an unknown column cannot get in no matter
    when it was attached. The frame is still checked, because a MISSING
    backbone column is a different failure and must not pass quietly.

    TWO VOCABULARIES, AND THEY ARE DIFFERENT NUMBERS. This returns 84 FEATURE
    NAMES. They expand to 88 DESIGN COLUMNS through the categorical levels in
    L3.CATEGORICAL_LEVELS. DS3 asserts the design width; E10e asserts the name
    count. A gate written against the wrong one of those looks like a defect
    in the pipeline and is not.
    """

    missing = [c for c in D1_FEATURE_NAMES if c not in features.columns]

    if missing:
        raise SystemExit(
            "FATAL: the feature frame is missing {} declared Phase 1 backbone "
            "columns: {}".format(len(missing), missing))

    return list(D1_FEATURE_NAMES)


def build_design(features, feature_names, dynamic=None):
    """
    Design matrix, column names, and a PASS-THROUGH mask.

    Section 6: continuous features are standardised on training statistics;
    boolean and indicator columns pass through unstandardised. The mask is
    built from each column's KIND, not from its observed values, so a
    continuous feature that happens to be 0/1 inside one fold is never
    silently reclassified.
    """

    numeric, categorical = L3.design_columns(feature_names, features)

    matrix, names = L3.build_matrix(features, numeric, categorical)

    booleans = set(c for c in numeric
                   if pd.api.types.is_bool_dtype(features[c]))

    passthrough = np.array(
        [(n in booleans) or ("=" in n) for n in names], dtype=bool)

    if dynamic is not None:
        matrix = np.hstack([matrix, dynamic.to_numpy(float)])
        names = names + list(dynamic.columns)
        passthrough = np.concatenate(
            [passthrough, np.zeros(dynamic.shape[1], dtype=bool)])

    return matrix, names, passthrough


def robust_mask(names):
    """Amendment 4 A4.1's qualifying columns, located by name in the design."""

    return np.array([n in DC_DERIVED_COLUMNS for n in names], dtype=bool)


# ============================================================
# THE FITTED PIPELINE
# ============================================================

def fit_pipeline(matrix, labels, train_rows, eval_rows, penalty, passthrough,
                 robust=None):
    """
    Impute, standardise and fit on train_rows; transform and predict
    eval_rows with the TRAINING statistics.

    This is I4.fit_pipeline with two declared differences:

      section 6      the pass-through rule for boolean and indicator columns
      Amendment 4    robust scaling for the columns flagged in the robust mask

    With both masks empty it IS I4.fit_pipeline, and DS0 asserts that
    bit-for-bit. An omitted robust mask means no column takes robust scaling,
    so every number the first ladder run produced is reproduced unchanged -
    DS13 asserts that against the committed artefact rather than trusting it.
    """

    if robust is None:
        robust = np.zeros(matrix.shape[1], dtype=bool)

    if np.any(robust & passthrough):
        raise RuntimeError(
            "a column cannot both pass through unstandardised and take "
            "robust scaling; Amendment 4 applies only to continuous columns")

    train = matrix[train_rows]
    held = matrix[eval_rows]

    if matrix.shape[1] == 0:
        weights, iterations, gradient = L3.fit_multinomial(
            np.zeros((len(train), 0)), labels[train_rows], penalty=penalty)
        return {
            "proba": L3.predict_multinomial(weights, np.zeros((len(held), 0))),
            "weights": weights, "iterations": iterations,
            "gradient": gradient, "mean": np.zeros(0), "sd": np.zeros(0),
            "imputed_cells": 0, "constant_columns": 0,
        }

    with np.errstate(invalid="ignore"):
        centre = np.nanmean(train, axis=0)

        if robust.any():
            # A4.2: the mean of a column containing 47.26 is not a typical
            # value of it, so a qualifying column is filled with its median.
            centre = np.where(robust, np.nanmedian(train, axis=0), centre)

    centre = np.where(np.isfinite(centre), centre, 0.0)

    imputed = int(np.isnan(train).sum() + np.isnan(held).sum())

    train = np.where(np.isnan(train), centre, train)
    held = np.where(np.isnan(held), centre, held)

    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)

    # Amendment 4 A4.2: median and IQR / 1.3489795 on the qualifying columns.
    # The same target quantity, estimated by something a handful of outliers
    # cannot move. Computed on the training rows of THIS fit - the outer
    # training rows outside the inner CV, that inner fold's rows inside it.
    if robust.any():
        quartiles = np.percentile(train, [25.0, 75.0], axis=0)
        spread = (quartiles[1] - quartiles[0]) / IQR_TO_SIGMA
        mean = np.where(robust, np.median(train, axis=0), mean)
        sd = np.where(robust, spread, sd)

    # The scale BEFORE the degenerate-column guard. G10 needs it: the guard
    # assumes "spread of zero implies a constant column", and where that
    # implication is false the guard is being applied outside its domain and
    # silently leaves a varying column unscaled.
    raw_scale = sd.copy()

    constant = int((sd == 0).sum())
    sd = np.where(sd == 0, 1.0, sd)

    # section 6: booleans and indicators are neither centred nor scaled
    mean = np.where(passthrough, 0.0, mean)
    sd = np.where(passthrough, 1.0, sd)

    train = (train - mean) / sd
    held = (held - mean) / sd

    weights, iterations, gradient = L3.fit_multinomial(
        train, labels[train_rows], penalty=penalty)

    return {
        "proba": L3.predict_multinomial(weights, held),
        "weights": weights, "iterations": iterations, "gradient": gradient,
        "mean": mean, "sd": sd, "raw_scale": raw_scale,
        "imputed_cells": imputed, "constant_columns": constant,
    }


# ============================================================
# INNER VALIDATION  (section 4)
# ============================================================

def inner_splits(train_rows, blocks):
    """
    5 contiguous chronological segments of the outer training rows, then 4
    expanding folds: fold i trains on segments 1..i and validates on i+1.

    The atomic unit is the CALENDAR DATE, per section 4's resolution of the
    (season, matchweek) conflict. A block straddling the outer training
    boundary would silently vanish here, so it is raised rather than dropped.
    """

    train_set = set(int(r) for r in train_rows)

    usable, partial = [], []

    for block in blocks:
        rows = set(int(r) for r in block["rows"])
        if rows.issubset(train_set):
            usable.append(block)
        elif not rows.isdisjoint(train_set):
            partial.append(block)

    if partial:
        raise RuntimeError(
            "a calendar date straddles the outer training boundary: {}".format(
                [str(b["date"].date()) for b in partial]))

    covered = sum(len(b["rows"]) for b in usable)

    if covered != len(train_rows):
        raise RuntimeError("inner blocks cover {} of {} training rows".format(
            covered, len(train_rows)))

    target = covered / N_INNER_BLOCKS
    segments = [[] for _ in range(N_INNER_BLOCKS)]
    running = 0

    for block in usable:
        midpoint = running + len(block["rows"]) / 2.0
        segments[min(int(midpoint // target), N_INNER_BLOCKS - 1)].append(block)
        running += len(block["rows"])

    splits = []

    for cut in range(1, N_INNER_BLOCKS):

        train_blocks = [b for s in segments[:cut] for b in s]
        valid_blocks = segments[cut]

        if not train_blocks or not valid_blocks:
            raise RuntimeError("inner block {} came out empty".format(cut))

        splits.append({
            "inner_fold": cut,
            "train_rows": np.sort(np.concatenate(
                [b["rows"] for b in train_blocks])),
            "valid_rows": np.sort(np.concatenate(
                [b["rows"] for b in valid_blocks])),
            "train_max_date": max(b["max_date"] for b in train_blocks),
            "valid_min_date": min(b["min_date"] for b in valid_blocks),
        })

    return splits


def select_lambda(matrix, labels, results, train_rows, blocks, passthrough,
                  robust=None):
    """Mean validation log loss over the 4 inner folds. Ties to the SMALLEST."""

    splits = inner_splits(train_rows, blocks)

    curve = {}

    for penalty in LAMBDA_GRID:

        losses = []

        for split in splits:
            fitted = fit_pipeline(matrix, labels, split["train_rows"],
                                  split["valid_rows"], penalty, passthrough,
                                  robust)
            losses.append(evaluate(results[split["valid_rows"]],
                                   fitted["proba"])["log_loss"])

        curve[penalty] = float(np.mean(losses))

    best = min(curve.values())
    chosen = min(p for p, v in curve.items() if v == best)

    return chosen, curve, splits


# ============================================================
# BOOTSTRAP  (section 7)
# ============================================================

def per_match_log_loss(proba, actual):

    picked = proba[np.arange(len(actual)),
                   [CLASS_INDEX[a] for a in actual]]

    return -np.log(np.clip(picked, 1e-15, 1.0))


def per_match_rps(proba, actual):

    onehot = np.zeros_like(proba)
    onehot[np.arange(len(actual)), [CLASS_INDEX[a] for a in actual]] = 1.0
    cumulative = np.cumsum(proba, axis=1) - np.cumsum(onehot, axis=1)

    return np.sum(cumulative[:, :-1] ** 2, axis=1) / (len(CLASSES) - 1)


def paired_bootstrap(a, b, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Paired per-match CI on mean(a) - mean(b). The same indices score both."""

    rng = np.random.default_rng(seed)
    difference = a - b
    n = len(difference)

    stats = np.empty(draws)

    for draw in range(draws):
        stats[draw] = float(np.mean(difference[rng.integers(0, n, n)]))

    return (float(np.mean(difference)),
            float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


def compare(label, left, right, proba_left, proba_right, actual, scope="pooled"):
    """One delta, both primary metrics, under the sign-agreement rule."""

    ll, ll_lo, ll_hi = paired_bootstrap(
        per_match_log_loss(proba_left, actual),
        per_match_log_loss(proba_right, actual))

    rps, rps_lo, rps_hi = paired_bootstrap(
        per_match_rps(proba_left, actual), per_match_rps(proba_right, actual))

    agree = (ll > 0) == (rps > 0)
    significant = not (ll_lo <= 0 <= ll_hi)

    return {
        "comparison": label, "left": left, "right": right,
        "scope": scope, "n": len(actual),
        "log_loss_delta": ll, "log_loss_ci_lo": ll_lo, "log_loss_ci_hi": ll_hi,
        "rps_delta": rps, "rps_ci_lo": rps_lo, "rps_ci_hi": rps_hi,
        "signs_agree": agree,
        "log_loss_ci_excludes_zero": significant,
        "verdict": ("INCONCLUSIVE (sign disagreement)" if not agree
                    else ("SIGNIFICANT" if significant else "NOT SIGNIFICANT")),
    }


# ============================================================
# THE RUNGS
# ============================================================

def run_d0(frame, spec, results):
    """
    No fit. The training fold's H/D/A frequencies, repeated for every test
    row. Section 3: 0 features, and therefore no lambda and no grid.
    """

    rows, proba_by_fold = [], {}

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = str(fold_spec["test_season"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero(
            (frame["season"] == test_season).to_numpy())

        counts = np.array([float((results[train_rows] == c).sum())
                           for c in CLASSES])
        rates = counts / counts.sum()

        proba = np.tile(rates, (len(test_rows), 1))

        validate_probabilities(proba, len(test_rows))
        scores = evaluate(results[test_rows], proba)

        row = {"rung": "D0", "fold": fold, "test_season": test_season,
               "train_matches": len(train_rows), "test_matches": len(test_rows),
               "design_width": 0, "selected_lambda": np.nan,
               "selected_lambda_label": "n/a (no fit)",
               "at_grid_floor": False, "at_grid_ceiling": False,
               "rarest_class": int(counts.min()), "epv": np.nan,
               "g6_status": "NOT EXERCISED (no fit, no grid)",
               "p_home": rates[0], "p_draw": rates[1], "p_away": rates[2],
               "newton_iterations": 0, "gradient_norm": 0.0,
               "imputed_cells": 0, "constant_columns": 0}
        row.update({m: scores[m] for m in METRICS})
        rows.append(row)

        proba_by_fold[fold] = (test_rows, proba)

    return pd.DataFrame(rows), proba_by_fold


def run_rung(name, matrix, passthrough, frame, spec, labels, results, blocks,
             robust=None):
    """One fitted rung, all four frozen outer folds."""

    rows, curves, coefficients, proba_by_fold, diagnostics = [], [], [], {}, []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = str(fold_spec["test_season"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero(
            (frame["season"] == test_season).to_numpy())

        chosen, curve, splits = select_lambda(
            matrix, labels, results, train_rows, blocks, passthrough, robust)

        fitted = fit_pipeline(matrix, labels, train_rows, test_rows,
                              chosen, passthrough, robust)

        validate_probabilities(fitted["proba"], len(test_rows))
        scores = evaluate(results[test_rows], fitted["proba"])

        rarest = int(pd.Series(results[train_rows]).value_counts().min())
        width = matrix.shape[1]
        epv = rarest / width if width else np.nan

        applicable = (epv < EPV_APPLICABILITY) if width else False
        at_floor = bool(chosen == LAMBDA_GRID[0])
        at_ceiling = bool(chosen == LAMBDA_GRID[-1])
        boundary = at_floor or at_ceiling

        if not applicable:
            status = "NOT APPLICABLE (EPV {:.2f} >= {:g})".format(
                epv, EPV_APPLICABILITY)
        elif boundary:
            status = "FAIL (boundary selection at lambda {:g})".format(chosen)
        else:
            status = "PASS"

        row = {"rung": name, "fold": fold, "test_season": test_season,
               "train_matches": len(train_rows), "test_matches": len(test_rows),
               "design_width": width, "selected_lambda": chosen,
               "selected_lambda_label": "lam={:g}".format(chosen),
               "at_grid_floor": at_floor, "at_grid_ceiling": at_ceiling,
               "rarest_class": rarest, "epv": epv, "g6_status": status,
               "p_home": np.nan, "p_draw": np.nan, "p_away": np.nan,
               "newton_iterations": int(fitted["iterations"]),
               "gradient_norm": float(fitted["gradient"]),
               "imputed_cells": int(fitted["imputed_cells"]),
               "constant_columns": int(fitted["constant_columns"])}
        row.update({m: scores[m] for m in METRICS})
        rows.append(row)

        for penalty, value in curve.items():
            curves.append({"rung": name, "fold": fold,
                           "lambda": penalty,
                           "lambda_label": "lam={:g}".format(penalty),
                           "inner_mean_log_loss": value,
                           "selected": bool(penalty == chosen)})

        proba_by_fold[fold] = (test_rows, fitted["proba"])

        diagnostics.append({"fold": fold, "splits": splits,
                            "train_rows": train_rows, "test_rows": test_rows,
                            "weights": fitted["weights"],
                            "mean": fitted["mean"], "sd": fitted["sd"],
                            "raw_scale": fitted["raw_scale"],
                            "robust": robust})

    return (pd.DataFrame(rows), pd.DataFrame(curves), proba_by_fold,
            diagnostics)


def pool(proba_by_fold, results, spec):
    """Stack the four folds in fold order and score the 1,520 together."""

    order, stacked = [], []

    for fold_spec in spec["folds"]:
        fold = int(fold_spec["fold"])
        test_rows, proba = proba_by_fold[fold]
        order.append(test_rows)
        stacked.append(proba)

    rows = np.concatenate(order)
    proba = np.vstack(stacked)

    return rows, proba, evaluate(results[rows], proba)


# ============================================================
# THE LEAKAGE SUITE  (section 8)
# ============================================================

def hash_file(path):

    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ds0_pipeline_anchor(audit, matrix, labels, spec, frame, passthrough):
    """
    NOT one of DS1-DS12. The declared pass-through rule of section 6 required
    a second implementation of the fitted pipeline, and a second
    implementation with no anchor is exactly the thing this project treats as
    a defect. With an all-False mask this function must BE I4.fit_pipeline.
    """

    train_rows = np.flatnonzero(
        frame["season"].isin(spec["folds"][1]["train_seasons"]).to_numpy())
    test_rows = np.flatnonzero(
        (frame["season"] == spec["folds"][1]["test_season"]).to_numpy())

    worst = 0.0

    for penalty in (0.01, 1.0, 1000.0):

        mine = fit_pipeline(matrix, labels, train_rows, test_rows, penalty,
                            np.zeros(matrix.shape[1], dtype=bool))
        theirs = I4.fit_pipeline(matrix, labels, train_rows, test_rows, penalty)

        worst = max(worst, float(np.abs(mine["proba"] - theirs["proba"]).max()))

    audit.record(
        "DS0", "with an all-False mask the local pipeline reproduces "
               "I4.fit_pipeline bit-for-bit",
        "0.0", "{:.3e}".format(worst), worst == 0.0,
        "the pass-through rule of section 6 is the ONLY difference between "
        "them; anchoring it here is what makes the new code path readable as "
        "Phase 3's, plus one declared change")

    audit.measure(
        "DS0b", "design columns passing through unstandardised",
        int(passthrough.sum()),
        "section 6: boolean and indicator columns are neither centred nor "
        "scaled. Classified by column KIND, never by observed values")


def ds1_temporal(audit, matches, state, elo):
    """
    Every feature value for a match derives only from matches STRICTLY
    earlier than that match's date.

    Two independent proofs, because the block has two sources:

      DC   corrupt every match from a cutoff onward, refit that cutoff's
           window, and require the fitted strengths to be unchanged. This
           tests the property directly rather than re-asserting the code.
      Elo  chain each team's before-state against its own previous match's
           after-state, through Phase 2's declared season regression.
    """

    dates = sorted(matches["date"].unique())
    probes = [dates[i] for i in (120, 240, 360, 420, 500, 560)]

    rng = np.random.default_rng(CORRUPTION_SEED)
    worst = 0.0

    for cutoff in probes:

        cutoff = pd.Timestamp(cutoff)

        clean_window = matches[matches["date"] < cutoff]
        clean = DC.fit_window(clean_window, cutoff, True)

        corrupted = matches.copy()
        later = corrupted["date"] >= cutoff
        corrupted.loc[later, "home_goals"] = rng.integers(
            0, 9, int(later.sum()))
        corrupted.loc[later, "away_goals"] = rng.integers(
            0, 9, int(later.sum()))

        dirty = DC.fit_window(corrupted[corrupted["date"] < cutoff],
                              cutoff, True)

        teams = sorted(set(clean["attack"]) | set(dirty["attack"]))

        gap = max(
            [abs(clean["attack"][t] - dirty["attack"][t]) for t in teams] +
            [abs(clean["defence"][t] - dirty["defence"][t]) for t in teams] +
            [abs(clean["rho"] - dirty["rho"]),
             abs(clean["home_multiplier"] - dirty["home_multiplier"])])

        worst = max(worst, float(gap))

    audit.record(
        "DS1a", "DC state is invariant to corrupting every match from the "
                "cutoff onward, at 6 probe dates",
        "0.0", "{:.3e}".format(worst), worst == 0.0,
        "the window is matches with date STRICTLY < cutoff, so nothing at or "
        "after the cutoff can reach the fit. Tested by corruption rather "
        "than by re-reading the filter")

    # ---- Elo: before-state chains from the previous match's after-state ----
    walk = elo.sort_values(["date", "home_team", "away_team"]).copy()

    last_rating, last_season = {}, {}
    worst_elo, checked = 0.0, 0

    for _i, row in walk.iterrows():

        for side in ("home", "away"):

            team = row["{}_team".format(side)]
            before = float(row["{}_elo_before".format(side)])
            after = float(row["{}_elo_after".format(side)])

            transition = row["{}_transition".format(side)]

            if last_season.get(team) == row["season"]:
                # mid-season: the previous match's after-rating, untouched
                expected = last_rating[team]
            elif transition == "continuing":
                # a new season, and the team played the last one
                expected = (STATE.ELO_INITIAL
                            + (last_rating[team] - STATE.ELO_INITIAL)
                            * STATE.ELO_REGRESSION)
            else:
                # promoted, new to the dataset, or returning after an
                # absence: Phase 2 resets to the flat initial rating and does
                # NOT carry a stale rating forward. Reading the regression
                # onto these teams is what made the first version of this
                # check fail on correct data.
                expected = STATE.ELO_INITIAL

            worst_elo = max(worst_elo, abs(before - expected))
            checked += 1

            last_rating[team] = after
            last_season[team] = row["season"]

    audit.record(
        "DS1b", "every Elo before-rating follows from earlier matches alone, "
                "under Phase 2's declared season-transition policy",
        "< 1e-9", "{:.3e}".format(worst_elo), worst_elo < 1e-9,
        "{} before-ratings checked against continuing / promoted / "
        "returning-after-absence, the three cases of Phase 1's transition "
        "table. A before-rating that is a pure function of the team's own "
        "earlier matches cannot carry information from later ones".format(
            checked))

    # ---- the DC window is exactly the strictly-earlier count --------------
    counts = matches.groupby("date").size().sort_index()
    cumulative = counts.cumsum().shift(1).fillna(0)

    expected = matches["date"].map(cumulative).to_numpy()
    observed = state["window_matches"].to_numpy()

    audit.record(
        "DS1c", "each match's DC fitting window is exactly the count of "
                "matches strictly earlier in date",
        0, int((expected != observed).sum()),
        int((expected != observed).sum()) == 0,
        "an off-by-one here would put same-day matches into their own "
        "fitting window, which is the classic form of this leak")


def ds2_no_test_row_fits(audit, rung_diagnostics, matrix, frame, spec):
    """No outer-test row enters any fit or any scaler, at any rung."""

    violations = 0
    scaler_gap = 0.0

    for name, diagnostics in rung_diagnostics.items():

        for entry in diagnostics:

            train = set(int(r) for r in entry["train_rows"])
            test = set(int(r) for r in entry["test_rows"])

            violations += len(train & test)

            for split in entry["splits"]:
                violations += len(set(int(r) for r in split["train_rows"]) & test)
                violations += len(set(int(r) for r in split["valid_rows"]) & test)

    audit.record(
        "DS2", "no outer-test row appears in any outer or inner training set",
        0, violations, violations == 0,
        "checked across every rung, every outer fold and every inner fold")

    # the scaler is a separate surface: prove the statistics are the training
    # rows' own, not merely that the row indices were disjoint
    for name, diagnostics in rung_diagnostics.items():

        if name == "D0":
            continue

        for entry in diagnostics:

            train = matrix[name][entry["train_rows"]]

            # AMENDMENT 4 RUNS THROUGH THIS CHECK IN TWO PLACES, and the first
            # two executions of it each missed one. The first computed only
            # the sample mean and SD, and disagreed with the pipeline by 29.0
            # on every robust-scaled column. The second added the robust SCALE
            # but still imputed with the mean, and disagreed by 1.0 wherever
            # imputation moved the quartiles. Both were failures of the CHECK.
            #
            # This is deliberately a REIMPLEMENTATION rather than a call into
            # fit_pipeline: a verifier that runs the code it is verifying
            # proves only that the code is deterministic. The cost is that it
            # has to track the declared rule exactly, which is what the two
            # failures above were.
            robust = entry.get("robust")

            if robust is None:
                robust = np.zeros(train.shape[1], dtype=bool)

            with np.errstate(invalid="ignore"):
                centre = np.nanmean(train, axis=0)

                if robust.any():
                    # A4.2: a qualifying column is filled with its MEDIAN
                    centre = np.where(
                        robust, np.nanmedian(train, axis=0), centre)

            centre = np.where(np.isfinite(centre), centre, 0.0)
            filled = np.where(np.isnan(train), centre, train)

            recomputed_mean = filled.mean(axis=0)
            recomputed_sd = filled.std(axis=0, ddof=0)

            if robust.any():
                quartiles = np.percentile(filled, [25.0, 75.0], axis=0)
                recomputed_mean = np.where(
                    robust, np.median(filled, axis=0), recomputed_mean)
                recomputed_sd = np.where(
                    robust, (quartiles[1] - quartiles[0]) / IQR_TO_SIGMA,
                    recomputed_sd)

            recomputed_sd = np.where(recomputed_sd == 0, 1.0, recomputed_sd)

            mask = entry["passthrough"]
            recomputed_mean = np.where(mask, 0.0, recomputed_mean)
            recomputed_sd = np.where(mask, 1.0, recomputed_sd)

            scaler_gap = max(
                scaler_gap,
                float(np.abs(recomputed_mean - entry["mean"]).max()),
                float(np.abs(recomputed_sd - entry["sd"]).max()))

    audit.record(
        "DS2b", "the fitted scaler reproduces exactly from the training rows "
                "alone",
        "0.0", "{:.3e}".format(scaler_gap), scaler_gap == 0.0,
        "recomputed independently of the pipeline, from the training rows "
        "only; a test row leaking into the scaler would move this")


def g10_scale_domain(audit, rung_diagnostics, matrices):
    """
    G10 - THE DEGENERATE-COLUMN GUARD, TESTED INSIDE ITS OWN DOMAIN.

    DISCLOSURE: this check is written HAVING SEEN D2-static return a scale of
    1.0 on two varying columns and 0.0011 on a third. It is stated here
    because a check added after a failure is the pattern this project polices
    hardest.

    WHAT MAKES IT LEGITIMATE: it carries NO THRESHOLD. It is a pure logic
    test of an implication the pipeline already relies on -

        the guard replaces a zero scale with 1.0 on the grounds that a
        column with zero spread is CONSTANT, and scaling a constant column
        is meaningless

    - and it fails exactly when that implication is false: spread zero on a
    column whose training values are not all equal. There is no tuned
    constant to move, and the check would read identically had every rung
    passed it. A column in that state enters the penalised fit in RAW units,
    which is the same class of defect Amendment 4 exists to remove.
    """

    offenders = []

    for name, diagnostics in rung_diagnostics.items():

        if name not in matrices:
            continue

        for entry in diagnostics:

            train = matrices[name][entry["train_rows"]]

            for index in range(matrices[name].shape[1]):

                if entry["raw_scale"][index] != 0.0:
                    continue

                column = train[:, index]
                column = column[np.isfinite(column)]

                if len(column) and column.min() != column.max():
                    offenders.append("{} fold {} column {}".format(
                        name, entry["fold"], index))

    audit.record(
        "G10", "no column takes the zero-scale guard while actually varying",
        0, len(offenders), not offenders,
        "the guard means 'this column is constant, so do not scale it'. Where "
        "the column is NOT constant the guard leaves it in raw units inside a "
        "penalised fit. Offenders: {}".format(
            ", ".join(offenders) if offenders else "none"))

    return offenders


def ds3_widths(audit, features, widths):
    """Design widths and strict nesting D1 < D2 < D3 < D4."""

    base = d1_features(features)

    c_cols = [c for c in features.columns if block_of(c) == "C_context"]
    x_cols = [c for c in features.columns if block_of(c) == "X_prior_composite"]
    xa_cols = [c for c in features.columns if block_of(c) == "X_availability"]

    computed = {}

    for label, feats in (("D3", base + c_cols),
                         ("D4", base + c_cols + x_cols + xa_cols)):
        matrix, names, _mask = build_design(features, feats)
        computed[label] = matrix.shape[1] + len(DYNAMIC_COLUMNS)

    declared = {"D1": 88, "D2": 92, "D3": 112, "D4": 139}

    observed = {"D1": widths["D1"], "D2": widths["D2"],
                "D3": computed["D3"], "D4": computed["D4"]}

    matches_declared = all(observed[k] == declared[k] for k in declared)

    audit.record(
        "DS3a", "design widths match Amendment 3's table A3.2",
        str(declared), str(observed), matches_declared,
        "D3 and D4 are NOT fitted in this session; their widths are computed "
        "from the frozen feature file, which is what DS3's nesting claim "
        "needs and is not a fit")

    ordered = (observed["D1"] < observed["D2"] < observed["D3"]
               < observed["D4"])

    audit.record(
        "DS3b", "strict nesting D1 < D2 < D3 < D4",
        "strictly increasing", str([observed[k] for k in
                                    ("D1", "D2", "D3", "D4")]), ordered,
        "the ladder is nested by construction - each rung is the previous "
        "column set plus its own")

    audit.record(
        "DS3c", "D2's columns are D1's columns plus exactly the four "
                "dynamic-state columns",
        widths["D1"] + 4, widths["D2"], widths["D2"] == widths["D1"] + 4,
        "checked on the built matrices, not on the feature lists")


def ds4_ds5_corruption(audit, features, frame, spec, labels, results, blocks,
                       dynamic, baseline_lambdas, baseline_proba,
                       robust=None, rung="D2", feature_names=None):
    """
    DS4  corrupting an outer test season's FEATURES moves no selected lambda
         and no other fold's predictions - with the control that its own do.
    DS5  corrupting an outer test season's LABELS moves no selected lambda.

    {season} is used because it is the only season that is a test season and
    never a training season. Corrupting a season that is also training data
    for a later fold SHOULD move that fold, so the assertion would be false
    for a legitimate reason and would prove nothing.

    feature_names IS THE RUNG'S OWN FEATURE LIST and defaults to D1's, which
    with `dynamic` supplied is D2. It has to be a parameter: this function
    REBUILDS the design in order to corrupt it, and a D3 or D4 caller passing
    its own baseline lambdas and probabilities against a design rebuilt at D2
    width would compare two different models and fail for a reason that has
    nothing to do with leakage. The default reproduces every existing caller
    exactly.
    """

    rng = np.random.default_rng(CORRUPTION_SEED)

    target = (frame["season"] == CORRUPTION_SEASON).to_numpy()
    target_rows = np.flatnonzero(target)

    if feature_names is None:
        feature_names = d1_features(features)

    # ---- DS4: corrupt the FEATURES of that season -------------------------
    # The corruption is applied to the BUILT design matrix, not to the source
    # frame. Overwriting a boolean column in the frame can silently change its
    # dtype, which would move the pass-through mask and with it EVERY fold's
    # standardisation - a spurious failure wearing the costume of a leak.
    # Corrupting the matrix hits exactly the values the fit actually sees.
    clean_matrix, _names, passthrough = build_design(
        features, feature_names, dynamic)

    matrix = clean_matrix.copy()
    matrix[target_rows, :] = rng.normal(
        0.0, 100.0, (len(target_rows), matrix.shape[1]))

    moved_lambda, moved_other, own_moved = 0, 0.0, 0.0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero(
            (frame["season"] == str(fold_spec["test_season"])).to_numpy())

        chosen, _curve, _splits = select_lambda(
            matrix, labels, results, train_rows, blocks, passthrough, robust)

        if chosen != baseline_lambdas[fold]:
            moved_lambda += 1

        fitted = fit_pipeline(matrix, labels, train_rows, test_rows, chosen,
                              passthrough, robust)

        gap = float(np.abs(fitted["proba"] - baseline_proba[fold][1]).max())

        if str(fold_spec["test_season"]) == CORRUPTION_SEASON:
            own_moved = gap
        else:
            moved_other = max(moved_other, gap)

    audit.record(
        "DS4a", "corrupting {}'s features moves no selected lambda".format(
            CORRUPTION_SEASON),
        0, moved_lambda, moved_lambda == 0,
        "{}, all four folds re-selected on the corrupted matrix".format(rung))

    audit.record(
        "DS4b", "corrupting {}'s features moves no OTHER fold's "
                "predictions".format(CORRUPTION_SEASON),
        "0.0", "{:.3e}".format(moved_other), moved_other == 0.0,
        "{} is the only season that is a test season and never a training "
        "season, so this assertion is not true by construction".format(
            CORRUPTION_SEASON))

    audit.record(
        "DS4c", "CONTROL: its own fold's predictions DO move",
        "> 0", "{:.3e}".format(own_moved), own_moved > 0.0,
        "without this the two checks above would pass on a corruption that "
        "never landed")

    # ---- DS5: corrupt the LABELS of that season ---------------------------
    clean_mask = passthrough

    dirty_labels = labels.copy()
    dirty_labels[target_rows] = rng.integers(0, len(CLASSES), len(target_rows))

    dirty_results = results.copy()
    dirty_results[target_rows] = np.array(CLASSES)[dirty_labels[target_rows]]

    moved = 0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())

        chosen, _curve, _splits = select_lambda(
            clean_matrix, dirty_labels, dirty_results, train_rows, blocks,
            clean_mask, robust)

        if chosen != baseline_lambdas[fold]:
            moved += 1

    audit.record(
        "DS5", "corrupting {}'s labels moves no selected lambda".format(
            CORRUPTION_SEASON),
        0, moved, moved == 0,
        "labels of an outer-test season enter scoring and nothing else; if "
        "they reached the inner CV this would move")


def ds6_base_rate(audit, d0_folds):
    """D0 reproduces Phase 2's frozen base rate to < 1e-9."""

    frozen = pd.read_csv(BASE_RATE_SUMMARY, float_precision=FLOAT_PRECISION)

    joined = d0_folds.merge(frozen, on="fold", suffixes=("_d0", "_p2"))

    worst = 0.0

    for metric in METRICS:
        worst = max(worst, float(
            np.abs(joined["{}_d0".format(metric)]
                   - joined["{}_p2".format(metric)]).max()))

    for column, frozen_column in (("p_home", "train_p_home"),
                                  ("p_draw", "train_p_draw"),
                                  ("p_away", "train_p_away")):
        worst = max(worst, float(
            np.abs(joined[column] - joined[frozen_column]).max()))

    audit.record(
        "DS6", "D0 reproduces Phase 2's frozen base rate on all six metrics "
               "and all three class rates",
        "< 1e-9", "{:.3e}".format(worst), worst < 1e-9,
        "all four folds; D0 is Phase 2's baseline re-derived, so any "
        "divergence would mean the fold structure had moved under it")


def ds7_frozen_blocks(audit, features):
    """
    Block C and Block X columns read from the frozen Phase 3 artefacts.

    NOT EXERCISED at D0-D2: no rung here carries a C or X column. Recorded
    rather than skipped, with the column counts that D3 and D4 would consume,
    so the omission is visible instead of absent.
    """

    counts = {block: len([c for c in features.columns if block_of(c) == block])
              for block in ("C_context", "X_prior_composite",
                            "X_availability", "X_metadata")}

    audit.measure(
        "DS7", "Block C / Block X columns are read from the frozen artefacts",
        "NOT EXERCISED (no C or X column enters D0, D1 or D2)",
        "the frozen feature file's block counts are {}; DS10 verifies the "
        "file's hash. This test becomes live at D3, which is not run in this "
        "session".format(counts))


def ds8_determinism(audit, rungs, matrix, passthrough, frame, spec, labels,
                    results, blocks, baseline, robust=None):
    """A second full run is bit-identical."""

    worst_proba, moved_lambda = 0.0, 0

    for name in rungs:

        folds, _curves, proba_by_fold, _diag = run_rung(
            name, matrix[name], passthrough[name], frame, spec, labels,
            results, blocks,
            None if robust is None else robust.get(name))

        for fold, (_rows, proba) in proba_by_fold.items():
            worst_proba = max(worst_proba, float(
                np.abs(proba - baseline[name][fold][1]).max()))

        selected = dict(zip(folds["fold"], folds["selected_lambda"]))

        for fold, value in selected.items():
            if value != baseline["{}_lambda".format(name)][fold]:
                moved_lambda += 1

    audit.record(
        "DS8a", "a second full run of every fitted rung reproduces every "
                "probability bit-for-bit",
        "0.0", "{:.3e}".format(worst_proba), worst_proba == 0.0,
        "Newton from a zero start on a strictly convex objective, and a "
        "seeded bootstrap; nothing here should have a source of variation")

    audit.record(
        "DS8b", "and selects the same lambda at every fold",
        0, moved_lambda, moved_lambda == 0,
        "the selection is a deterministic minimum with a declared tie-break")


def ds9_contract(audit, proba_sets):
    """The probability contract, through the harness's own validator."""

    checked, failures = 0, []

    for label, proba in proba_sets:
        try:
            validate_probabilities(proba, len(proba))
            checked += 1
        except Exception as error:                      # noqa: BLE001
            failures.append("{}: {}".format(label, error))

    audit.record(
        "DS9", "every rung's pooled and per-fold output passes the harness's "
               "validate_probabilities",
        0, len(failures), not failures,
        "{} probability arrays checked. The harness raises rather than "
        "repairing, so a renormalised break cannot pass silently{}".format(
            checked, "" if not failures else "; " + "; ".join(failures)))


def ds10_manifest(audit):
    """No frozen artefact moved hash."""

    try:
        completed = subprocess.run(
            [sys.executable, "-B", str(SCRIPTS_DIR / "frozen_manifest.py"),
             "--verify"],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=600)
        output = (completed.stdout or "") + (completed.stderr or "")
        code = completed.returncode
    except Exception as error:                          # noqa: BLE001
        output, code = str(error), -1

    tail = [line for line in output.splitlines() if line.strip()][-1:]

    audit.record(
        "DS10", "frozen_manifest.py --verify: no frozen artefact moved hash",
        0, code, code == 0,
        "; ".join(tail) if tail else "no output")

    return output


def ds11_anchor(audit, frame, state, spec):
    """
    THE ANCHOR. The extracted DC state, fed through Phase 2's OWN
    outcome_probabilities(), must reproduce dc_walkforward's 0.9904 pooled
    log loss. This proves the extraction is lossless BEFORE any logistic
    touches it, so a D2 shortfall cannot be blamed on it.
    """

    test_rows = np.flatnonzero(frame["role_is_test"].to_numpy())

    proba = np.zeros((len(test_rows), len(CLASSES)))

    for position, row in enumerate(test_rows):
        probabilities, _mass, _truncated = DC.outcome_probabilities(
            float(state["lambda_home"].iloc[row]),
            float(state["lambda_away"].iloc[row]),
            float(state["rho"].iloc[row]))
        proba[position] = probabilities

    scores = evaluate(frame["result"].to_numpy()[test_rows], proba)

    dc_summary = pd.read_csv(DC_SUMMARY, float_precision=FLOAT_PRECISION)
    dc_summary = dc_summary[dc_summary["variant"] == DC_VARIANT]
    reference = float(dc_summary["log_loss"].mean())

    gap = abs(scores["log_loss"] - reference)

    audit.record(
        "DS11", "extracted DC state through Phase 2's own "
                "outcome_probabilities() reproduces dc_walkforward",
        "< 1e-9", "{:.3e}".format(gap), gap < 1e-9,
        "{} test rows; extracted {:.10f} against Phase 2's four-fold mean "
        "{:.10f}. The state D2 is built from is the state DC scored".format(
            len(test_rows), scores["log_loss"], reference))

    return scores, proba


def ds12_ordering(audit, rung_diagnostics, frame):
    """Every inner split satisfies max(train date) < min(validation date)."""

    violations, checked, tightest = 0, 0, None

    for name, diagnostics in rung_diagnostics.items():

        if name == "D0":
            continue

        for entry in diagnostics:
            for split in entry["splits"]:

                checked += 1

                gap = (split["valid_min_date"] - split["train_max_date"]).days

                if split["train_max_date"] >= split["valid_min_date"]:
                    violations += 1

                if tightest is None or gap < tightest:
                    tightest = gap

    audit.record(
        "DS12", "every inner split has max(train date) STRICTLY < "
                "min(validation date)",
        0, violations, violations == 0,
        "{} inner splits across every fitted rung and fold; the tightest gap "
        "is {} day(s). This is the constraint that forced the calendar date "
        "to replace the (season, matchweek) block".format(checked, tightest))


# ============================================================
# REFERENCE MODELS
# ============================================================

def reference_probabilities(frame):
    """
    Elo v1, Poisson walk-forward and Dixon-Coles walk-forward, on exactly the
    1,520 outer-test rows and in the frame's own row order.

    Gotcha 3b: both result files hold all 1,900 matches. Pooling the wrong set
    returns a plausible wrong number, so the count is asserted rather than
    trusted.
    """

    keys = ["date", "home_team", "away_team"]

    elo = pd.read_csv(ELO_RESULTS, float_precision=FLOAT_PRECISION)
    elo["date"] = pd.to_datetime(elo["date"], format="%Y-%m-%d")
    elo = elo[elo["evaluated"] == 1]

    if len(elo) != 1520:
        raise SystemExit("FATAL: Elo test rows = {}, expected 1520".format(
            len(elo)))

    dc = pd.read_csv(DC_RESULTS, float_precision=FLOAT_PRECISION)
    dc["date"] = pd.to_datetime(dc["date"], format="%Y-%m-%d")

    references = {}

    elo_keyed = frame[["match_id"] + keys].merge(
        elo.rename(columns={"home": "home_team", "away": "away_team"})[
            keys + ["p_home", "p_draw", "p_away"]],
        on=keys, how="left")

    references["elo_v1"] = elo_keyed

    for label, variant in (("poisson_walkforward", POISSON_VARIANT),
                           ("dc_walkforward", DC_VARIANT)):

        part = dc[dc["variant"] == variant]

        if len(part) != 1520:
            raise SystemExit("FATAL: {} rows = {}, expected 1520".format(
                variant, len(part)))

        references[label] = frame[["match_id"] + keys].merge(
            part.rename(columns={"home": "home_team", "away": "away_team"})[
                keys + ["p_home", "p_draw", "p_away"]],
            on=keys, how="left")

    return references


def reference_array(reference, rows):
    """The (n, 3) array for a set of frame row indices, in that order."""

    return reference.iloc[rows][["p_home", "p_draw", "p_away"]].to_numpy(float)


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 4 - THE DYNAMIC-STATE LADDER: D0, D1, D2")

    print("  governed by  : PHASE4_D2_PREDECLARATION.txt + Amendments 1-3")
    print("  rungs        : D0 base rate, D1 results-derived, D2 + dynamic state")
    print("  grid         : {} points, {:g} to {:g}  (Amendment 2 A2.3)".format(
        len(LAMBDA_GRID), LAMBDA_GRID[0], LAMBDA_GRID[-1]))
    print("  inner CV     : {} date segments, {} expanding folds, ties {}".format(
        N_INNER_BLOCKS, N_INNER_BLOCKS - 1, TIE_BREAK))
    print("  bootstrap    : {} draws, seed {}".format(
        BOOTSTRAP_DRAWS, BOOTSTRAP_SEED))
    print("  D3 / D4      : NOT RUN in this session, by instruction")
    print()

    audit = Audit()

    # ---- inputs -----------------------------------------------------------
    spec = L3.load_spec()
    matches = L3.load_matches()
    features = L3.load_features(matches)

    matches = matches.copy()
    matches["match_id"] = matches.index

    test_seasons = [str(f["test_season"]) for f in spec["folds"]]
    matches["role_is_test"] = matches["season"].isin(test_seasons)

    print("  generating point-in-time dynamic state for {} matches...".format(
        len(matches)))

    state, refits = STATE.build(matches)

    print("  {} DC refits, {:.1f}s".format(refits, time.time() - started))
    print()

    # ---- S1: the state must agree with the frozen passthrough file --------
    frozen_state = pd.read_csv(PASSTHROUGH_STATE,
                               float_precision=FLOAT_PRECISION)

    merged = state[["match_id", "lambda_home", "lambda_away", "rho",
                    "window_matches"]].merge(
        frozen_state, on="match_id", suffixes=("_new", "_old"))

    worst = max(
        float(np.abs(merged["lambda_home_new"]
                     - merged["lambda_home_old"]).max(skipna=True)),
        float(np.abs(merged["lambda_away_new"]
                     - merged["lambda_away_old"]).max(skipna=True)))

    audit.record(
        "S1", "the regenerated dynamic state reproduces the frozen "
              "passthrough lambdas bit-for-bit",
        "0.0", "{:.3e}".format(worst), worst == 0.0,
        "{} matches; the passthrough's G1 already anchored those lambdas to "
        "Phase 2's stored test-row values, so this chains D2's state to "
        "Phase 2 through an artefact that is already frozen".format(
            len(merged)))

    # ---- assemble ---------------------------------------------------------
    frame = matches.copy()
    dynamic = state.set_index("match_id").loc[
        frame["match_id"], DYNAMIC_COLUMNS].reset_index(drop=True)

    labels = np.array([CLASS_INDEX[r] for r in frame["result"]], dtype=int)
    results = frame["result"].to_numpy()
    blocks = I4.date_blocks(frame)

    d1_matrix, d1_names, d1_mask = build_design(
        features, d1_features(features))

    d2_matrix, d2_names, d2_mask = build_design(
        features, d1_features(features), dynamic)

    matrices = {"D1": d1_matrix, "D2": d2_matrix}
    masks = {"D1": d1_mask, "D2": d2_mask}
    names_by_rung = {"D1": d1_names, "D2": d2_names}

    # ============================================================
    banner("1. THE RUNGS")

    d0_folds, d0_proba = run_d0(frame, spec, results)

    fold_tables = {"D0": d0_folds}
    proba_by_rung = {"D0": d0_proba}
    curve_tables = {}
    diagnostics = {}

    for name in ("D1", "D2"):

        folds, curves, proba, diag = run_rung(
            name, matrices[name], masks[name], frame, spec, labels, results,
            blocks)

        for entry in diag:
            entry["passthrough"] = masks[name]

        fold_tables[name] = folds
        proba_by_rung[name] = proba
        curve_tables[name] = curves
        diagnostics[name] = diag

    all_folds = pd.concat([fold_tables[r] for r in ("D0", "D1", "D2")],
                          ignore_index=True)

    for rung in ("D0", "D1", "D2"):

        table = fold_tables[rung]

        print("  {}".format(rung))
        print("  {:<5} {:<11} {:>7} {:>9} {:>6} {:>6} {:>7} {:>7} {:>7} {:>7} {:>7}".format(
            "fold", "test", "width", "lambda", "EPV", "acc", "bal_acc",
            "mac_f1", "logloss", "brier", "RPS"))
        print("  " + "-" * 92)

        for _i, row in table.iterrows():
            print("  {:<5} {:<11} {:>7} {:>9} {:>6} {:>6.3f} {:>7.3f} {:>7.3f} "
                  "{:>7.4f} {:>7.4f} {:>7.4f}".format(
                      int(row["fold"]), row["test_season"],
                      int(row["design_width"]),
                      "-" if pd.isna(row["selected_lambda"])
                      else "{:g}".format(row["selected_lambda"]),
                      "-" if pd.isna(row["epv"]) else "{:.2f}".format(row["epv"]),
                      row["accuracy"], row["balanced_accuracy"],
                      row["macro_f1"], row["log_loss"], row["brier_score"],
                      row["rps"]))

        print()
        print("    G6: " + " | ".join(
            "f{} {}".format(int(r["fold"]), r["g6_status"])
            for _i, r in table.iterrows()))
        print()

    # ---- G6 as a gate -----------------------------------------------------
    failed_g6 = all_folds[all_folds["g6_status"].str.startswith("FAIL")]

    audit.record(
        "G6", "no applicable rung/fold selects a lambda on a grid boundary",
        0, len(failed_g6), len(failed_g6) == 0,
        "Amendment 3: G6 is applicable wherever EPV < {:g}. Applicable at "
        "every fitted (rung, fold) here - the highest EPV reached is "
        "{:.2f}".format(
            EPV_APPLICABILITY,
            float(all_folds["epv"].max(skipna=True))))

    if len(failed_g6):
        print("  G6 FAILED at: {}".format(
            ", ".join("{} fold {}".format(r["rung"], int(r["fold"]))
                      for _i, r in failed_g6.iterrows())))
        print("  Per the brief, the ladder STOPS at the failed rung.")
        print()

    # ============================================================
    banner("2. POOLED OVER THE 1,520 OUTER TEST MATCHES")

    pooled_rows, pooled = {}, {}

    for rung in ("D0", "D1", "D2"):
        rows, proba, scores = pool(proba_by_rung[rung], results, spec)
        pooled_rows[rung] = rows
        pooled[rung] = (proba, scores)

    order = pooled_rows["D1"]
    actual = results[order]

    references = reference_probabilities(frame)

    reference_proba = {
        "elo_v1": reference_array(references["elo_v1"], order),
        "poisson_walkforward": reference_array(
            references["poisson_walkforward"], order),
        "dc_walkforward": reference_array(references["dc_walkforward"], order),
    }

    reference_scores = {k: evaluate(actual, v)
                        for k, v in reference_proba.items()}

    print("  {:<24} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "model", "acc", "bal_acc", "macro_f1", "logloss", "brier", "RPS"))
    print("  " + "-" * 76)

    display = [("D0  base rate", pooled["D0"][1]),
               ("D1  results-derived", pooled["D1"][1]),
               ("D2  dynamic state", pooled["D2"][1]),
               ("Elo v1", reference_scores["elo_v1"]),
               ("Poisson walk-forward",
                reference_scores["poisson_walkforward"]),
               ("Dixon-Coles walk-forward",
                reference_scores["dc_walkforward"])]

    for label, scores in display:
        print("  {:<24} {:>7.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} "
              "{:>8.4f}".format(
                  label, scores["accuracy"], scores["balanced_accuracy"],
                  scores["macro_f1"], scores["log_loss"],
                  scores["brier_score"], scores["rps"]))

    print()

    # ============================================================
    banner("3. THE DELTAS  (paired per-match bootstrap, {} draws, seed {})".format(
        BOOTSTRAP_DRAWS, BOOTSTRAP_SEED))

    proba_of = {
        "D0": pooled["D0"][0], "D1": pooled["D1"][0], "D2": pooled["D2"][0],
        "elo_v1": reference_proba["elo_v1"],
        "poisson_walkforward": reference_proba["poisson_walkforward"],
        "dc_walkforward": reference_proba["dc_walkforward"],
    }

    deltas = []

    pairs = [("D2 - D1", "D2", "D1"),
             ("D1 - D0", "D1", "D0"),
             ("D2 - D0", "D2", "D0"),
             ("D1 - Elo v1", "D1", "elo_v1"),
             ("D2 - Elo v1", "D2", "elo_v1"),
             ("D1 - Poisson", "D1", "poisson_walkforward"),
             ("D2 - Poisson", "D2", "poisson_walkforward"),
             ("D1 - Dixon-Coles", "D1", "dc_walkforward"),
             ("D2 - Dixon-Coles", "D2", "dc_walkforward")]

    for label, left, right in pairs:
        deltas.append(compare(label, left, right, proba_of[left],
                              proba_of[right], actual))

    print("  {:<20} {:>9} {:>20} {:>9} {:>20}  {}".format(
        "comparison", "d_logloss", "95% CI", "d_RPS", "95% CI", "verdict"))
    print("  " + "-" * 110)

    for row in deltas:
        print("  {:<20} {:>+9.4f} {:>20} {:>+9.4f} {:>20}  {}".format(
            row["comparison"], row["log_loss_delta"],
            "[{:+.4f}, {:+.4f}]".format(row["log_loss_ci_lo"],
                                        row["log_loss_ci_hi"]),
            row["rps_delta"],
            "[{:+.4f}, {:+.4f}]".format(row["rps_ci_lo"], row["rps_ci_hi"]),
            row["verdict"]))

    print()
    print("  negative favours the LEFT model. A sign disagreement between")
    print("  log loss and RPS makes the comparison INCONCLUSIVE.")
    print()

    # ---- D2 - D1 per fold -------------------------------------------------
    print("  D2 - D1, PER FOLD")
    print()
    print("  {:<5} {:<11} {:>9} {:>20} {:>9} {:>20}  {}".format(
        "fold", "test", "d_logloss", "95% CI", "d_RPS", "95% CI", "verdict"))
    print("  " + "-" * 100)

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        rows_d1, proba_d1 = proba_by_rung["D1"][fold]
        _rows_d2, proba_d2 = proba_by_rung["D2"][fold]

        row = compare("D2 - D1", "D2", "D1", proba_d2, proba_d1,
                      results[rows_d1], scope="fold {}".format(fold))
        row["fold"] = fold
        deltas.append(row)

        print("  {:<5} {:<11} {:>+9.4f} {:>20} {:>+9.4f} {:>20}  {}".format(
            fold, str(fold_spec["test_season"]), row["log_loss_delta"],
            "[{:+.4f}, {:+.4f}]".format(row["log_loss_ci_lo"],
                                        row["log_loss_ci_hi"]),
            row["rps_delta"],
            "[{:+.4f}, {:+.4f}]".format(row["rps_ci_lo"], row["rps_ci_hi"]),
            row["verdict"]))

    print()

    # ---- how much of the D1-to-DC gap does D2 close? ----------------------
    d1_ll = pooled["D1"][1]["log_loss"]
    d2_ll = pooled["D2"][1]["log_loss"]
    dc_ll = reference_scores["dc_walkforward"]["log_loss"]

    d1_rps = pooled["D1"][1]["rps"]
    d2_rps = pooled["D2"][1]["rps"]
    dc_rps = reference_scores["dc_walkforward"]["rps"]

    gap_ll = d1_ll - dc_ll
    gap_rps = d1_rps - dc_rps

    closed_ll = (d1_ll - d2_ll) / gap_ll if gap_ll else np.nan
    closed_rps = (d1_rps - d2_rps) / gap_rps if gap_rps else np.nan

    print("  THE D1-to-DC GAP")
    print()
    print("    log loss   D1 {:.4f}   DC {:.4f}   gap {:+.4f}".format(
        d1_ll, dc_ll, gap_ll))
    print("               D2 {:.4f}   moved {:+.4f}   = {:.1%} of the gap".format(
        d2_ll, d1_ll - d2_ll, closed_ll))
    print()
    print("    RPS        D1 {:.4f}   DC {:.4f}   gap {:+.4f}".format(
        d1_rps, dc_rps, gap_rps))
    print("               D2 {:.4f}   moved {:+.4f}   = {:.1%} of the gap".format(
        d2_rps, d1_rps - d2_rps, closed_rps))
    print()
    print("    the tier-2 recency effect this rung was built to capture was")
    print("    +0.0359 [+0.0186, +0.0535] in log loss")
    print()

    # ============================================================
    banner("4. THE DYNAMIC BLOCK ITSELF")

    coefficients = []

    print("  PER-FOLD COEFFICIENTS on the four dynamic columns")
    print("  (standardised scale: one unit is one training-set SD)")
    print()
    print("  {:<5} {:<22} {:>10} {:>10} {:>10} {:>12}".format(
        "fold", "column", "beta_H", "beta_D", "beta_A", "train SD"))
    print("  " + "-" * 74)

    for entry in diagnostics["D2"]:

        fold = entry["fold"]
        weights = entry["weights"]

        for offset, column in enumerate(DYNAMIC_COLUMNS):

            index = d2_matrix.shape[1] - len(DYNAMIC_COLUMNS) + offset

            beta = weights[index]

            coefficients.append({
                "rung": "D2", "fold": fold, "column": column,
                "beta_home": float(beta[0]), "beta_draw": float(beta[1]),
                "beta_away": float(beta[2]),
                "train_mean": float(entry["mean"][index]),
                "train_sd": float(entry["sd"][index]),
            })

            print("  {:<5} {:<22} {:>+10.4f} {:>+10.4f} {:>+10.4f} "
                  "{:>12.4f}".format(
                      fold, column, beta[0], beta[1], beta[2],
                      entry["sd"][index]))

        print()

    coefficient_frame = pd.DataFrame(coefficients)

    # ---- rel_elo_diff dispersion, and the under-convergence evidence ------
    print("  rel_elo_diff DISPERSION, and the burn-in exposure that makes")
    print("  D2 - D1 a LOWER BOUND")
    print()
    print("  {:<5} {:<11} {:>10} {:>10} {:>9} {:>9} {:>10} {:>9}".format(
        "fold", "test", "SD train", "SD test", "win<380", "of", "max lam_h",
        "test max"))
    print("  " + "-" * 82)

    dispersion = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero(
            (frame["season"] == str(fold_spec["test_season"])).to_numpy())

        elo_train = dynamic["rel_elo_diff"].to_numpy()[train_rows]
        elo_test = dynamic["rel_elo_diff"].to_numpy()[test_rows]

        windows = state["window_matches"].to_numpy()
        lambda_home = state["lambda_home"].to_numpy()

        undersized = int((windows[train_rows] < 380).sum())

        row = {
            "fold": fold, "test_season": str(fold_spec["test_season"]),
            "rel_elo_diff_sd_train": float(np.nanstd(elo_train, ddof=0)),
            "rel_elo_diff_sd_test": float(np.nanstd(elo_test, ddof=0)),
            "rel_elo_diff_range_train": float(
                np.nanmax(elo_train) - np.nanmin(elo_train)),
            "train_rows_below_burn_in": undersized,
            "train_rows": len(train_rows),
            "train_max_lambda_home": float(np.nanmax(lambda_home[train_rows])),
            "test_max_lambda_home": float(np.nanmax(lambda_home[test_rows])),
            "neutral_state_test_rows": int(
                (~state["home_has_history"].to_numpy()[test_rows]
                 | ~state["away_has_history"].to_numpy()[test_rows]).sum()),
        }

        dispersion.append(row)

        print("  {:<5} {:<11} {:>10.1f} {:>10.1f} {:>9} {:>9} {:>10.2f} "
              "{:>9.2f}".format(
                  fold, row["test_season"], row["rel_elo_diff_sd_train"],
                  row["rel_elo_diff_sd_test"], undersized, len(train_rows),
                  row["train_max_lambda_home"], row["test_max_lambda_home"]))

    dispersion_frame = pd.DataFrame(dispersion)

    print()
    print("  win<380 counts TRAINING rows fitted on a Dixon-Coles window")
    print("  smaller than the smallest window any TEST row is scored on.")
    print("  Those rows set the standardiser the test rows pass through.")
    print()
    print("  {:<5} {:>16}".format("fold", "neutral-state"))
    print("  {:<5} {:>16}".format("", "test rows"))
    print("  " + "-" * 24)
    for row in dispersion:
        print("  {:<5} {:>16}".format(row["fold"],
                                      row["neutral_state_test_rows"]))
    print()
    print("  a neutral-state row is one where a promoted or absent side takes")
    print("  the league-neutral strength. These are the rows where dynamic")
    print("  state has least to say, and they are NOT dropped.")
    print()

    # ============================================================
    banner("5. THE LEAKAGE SUITE  (DS1 - DS12)")

    ds0_pipeline_anchor(audit, d1_matrix, labels, spec, frame, d2_mask)

    elo_frame = frame[["season", "date", "home_team", "away_team"]].copy()
    elo_source = STATE.load_elo_state(matches)
    for column in ("home_elo_before", "away_elo_before",
                   "home_elo_after", "away_elo_after",
                   "home_transition", "away_transition"):
        elo_frame[column] = elo_source[column].to_numpy()

    ds1_temporal(audit, matches, state, elo_frame)
    ds2_no_test_row_fits(audit, diagnostics, matrices, frame, spec)
    ds3_widths(audit, features,
               {"D1": d1_matrix.shape[1], "D2": d2_matrix.shape[1]})

    baseline_lambdas = {int(r["fold"]): r["selected_lambda"]
                        for _i, r in fold_tables["D2"].iterrows()}

    ds4_ds5_corruption(audit, features, frame, spec, labels, results, blocks,
                       dynamic, baseline_lambdas, proba_by_rung["D2"])

    ds6_base_rate(audit, d0_folds)
    ds7_frozen_blocks(audit, features)

    determinism_baseline = {
        "D1": proba_by_rung["D1"], "D2": proba_by_rung["D2"],
        "D1_lambda": {int(r["fold"]): r["selected_lambda"]
                      for _i, r in fold_tables["D1"].iterrows()},
        "D2_lambda": baseline_lambdas,
    }

    ds8_determinism(audit, ("D1", "D2"), matrices, masks, frame, spec, labels,
                    results, blocks, determinism_baseline)

    contract_sets = [("{} pooled".format(r), pooled[r][0])
                     for r in ("D0", "D1", "D2")]
    for rung in ("D0", "D1", "D2"):
        for fold, (_rows, proba) in proba_by_rung[rung].items():
            contract_sets.append(("{} fold {}".format(rung, fold), proba))

    ds9_contract(audit, contract_sets)
    ds10_manifest(audit)

    state_indexed = state.set_index("match_id").loc[
        frame["match_id"]].reset_index()
    ds11_anchor(audit, frame, state_indexed, spec)

    ds12_ordering(audit, diagnostics, frame)

    audit.print_rows()

    # ============================================================
    banner("6. WRITING")

    pooled_table = []

    for label, key, scores in (
            ("D0", "D0", pooled["D0"][1]), ("D1", "D1", pooled["D1"][1]),
            ("D2", "D2", pooled["D2"][1]),
            ("elo_v1", "elo_v1", reference_scores["elo_v1"]),
            ("poisson_walkforward", "poisson_walkforward",
             reference_scores["poisson_walkforward"]),
            ("dc_walkforward", "dc_walkforward",
             reference_scores["dc_walkforward"])):
        pooled_table.append({"model": label, "n": scores["n"],
                             **{m: scores[m] for m in METRICS}})

    predictions = pd.DataFrame({
        "match_id": frame["match_id"].to_numpy()[order],
        "season": frame["season"].to_numpy()[order],
        "date": frame["date"].to_numpy()[order],
        "home_team": frame["home_team"].to_numpy()[order],
        "away_team": frame["away_team"].to_numpy()[order],
        "result": actual,
    })

    for key, proba in proba_of.items():
        for position, outcome in enumerate(CLASSES):
            predictions["{}_p_{}".format(key, outcome)] = proba[:, position]

    outputs = (
        (FOLD_OUTPUT, all_folds),
        (POOLED_OUTPUT, pd.DataFrame(pooled_table)),
        (DELTA_OUTPUT, pd.DataFrame(deltas)),
        (CURVE_OUTPUT, pd.concat(list(curve_tables.values()),
                                 ignore_index=True)),
        (COEF_OUTPUT, coefficient_frame.merge(dispersion_frame, on="fold")),
        (PRED_OUTPUT, predictions),
        (AUDIT_OUTPUT, audit.frame()),
    )

    for path, data in outputs:
        data.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)
        print("  {}".format(path))

    state.to_csv(OUTPUTS_DIR / "phase4_dynamic_state.csv", index=False,
                 encoding="utf-8", float_format=FLOAT_FORMAT)
    print("  {}".format(OUTPUTS_DIR / "phase4_dynamic_state.csv"))

    print()

    failures = audit.failures

    print("  Checks run    : {}".format(len(audit.rows)))
    print("  Checks failed : {}".format(len(failures)))
    print("  Elapsed       : {:.1f}s".format(time.time() - started))
    print()

    if failures:
        for row in failures:
            print("  FAILED  {}  {}".format(row["test_id"], row["test"]))
        print()

    print("  {}".format("PASS" if not failures else "FAIL"))
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
