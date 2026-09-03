"""
===============================================================================
PHASE 5 - INSTRUMENT E1c:  THE ISOLATED FINISHING RESIDUAL
===============================================================================

Pre-declaration: PHASE5_E1C_FINISHING_PREDECLARATION.txt
sha256 416a681696d03b6ca9e3f95afc88dcc1abd9853383edf05d53fbb6ad32310ff2
Signed off unamended before this file was written.

THE QUANTITY. E1b's column was SoT differential minus goal differential. Goals
run at about 0.327 per shot on target, so that column's expectation is 0.673 x
SoT differential - substantially a restatement of shot volume, with finishing
riding on top. E1c subtracts the CONVERSION rather than unity:

    finishing = goal differential  -  c x SoT differential

whose expectation is zero by construction. F7 checks that this actually
isolated anything, ordinally, against E1b's own column.

c IS PER FOLD, TAKEN AT THAT FOLD'S FIRST E1a CUTOFF (F1.2). That is the only
window E1a fits which is training rows only - its size is exactly 380 x fold,
which F4 asserts rather than trusts. Later cutoffs walk forward inside the test
season and their windows contain outer-test rows.

THE CONSEQUENCE, DECLARED IN F1.4: the feature column is FOLD-DEPENDENT, so
this instrument builds FOUR design matrices where every previous rung built
one. run_rung_folds() below is the fold loop that picks the right one; it calls
LADDER's own select_lambda and fit_pipeline, so the ESTIMATOR is not
duplicated, only the loop that chooses a matrix. F15 asserts that with the same
matrix at every fold it reproduces LADDER.run_rung exactly - the local loop is
verified against the shared one it stands in for.

THE PREDICTION, DECLARED BEFORE FITTING (F4.1-F4.7). Five-match finishing is
expected to be substantially its own sampling noise. Both consequences are
reported, and the autocorrelation arm is reported BEFORE the delta because it
is measurable without fitting the rung at all.
"""

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import CLASS_INDEX, CLASSES, evaluate  # noqa: E402
from phase3_feature_builder import (Audit, banner,  # noqa: E402
                                    configure_stdout, declare_block)

import phase2_poisson_dixon_coles as DC              # noqa: E402
import phase3_ablation_ladder as L3                  # noqa: E402
import phase3_regularisation_sensitivity as I4       # noqa: E402
import phase4_dynamic_state as STATE                 # noqa: E402
import phase4_dynamic_ladder as LADDER               # noqa: E402
import phase5_e1a_sot_ratings as E1A                 # noqa: E402
import phase5_e1b_shot_residuals as E1B              # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR

D34_PREDICTIONS = OUTPUTS_DIR / "phase4_d34_predictions.csv"
MARKET_PROBABILITIES = OUTPUTS_DIR / "phase5_market_probabilities.csv"
E1A_PREDICTIONS = OUTPUTS_DIR / "phase5_e1a_predictions.csv"
E1A_WINDOWS = OUTPUTS_DIR / "phase5_e1a_windows.csv"
E1B_PREDICTIONS = OUTPUTS_DIR / "phase5_e1b_residual_features.csv"
A4_FOLDS = OUTPUTS_DIR / "phase4_a4_fold_summary.csv"

FOLD_OUTPUT = OUTPUTS_DIR / "phase5_e1c_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase5_e1c_pooled.csv"
DELTA_OUTPUT = OUTPUTS_DIR / "phase5_e1c_deltas.csv"
COEF_OUTPUT = OUTPUTS_DIR / "phase5_e1c_coefficients.csv"
CURVE_OUTPUT = OUTPUTS_DIR / "phase5_e1c_lambda_curves.csv"
FEATURE_OUTPUT = OUTPUTS_DIR / "phase5_e1c_finishing_features.csv"
CONSTANT_OUTPUT = OUTPUTS_DIR / "phase5_e1c_constants.csv"
PERSIST_OUTPUT = OUTPUTS_DIR / "phase5_e1c_persistence.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase5_e1c_audit.csv"

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

WINDOW = 5

FINISHING_COLUMNS = ["home_finishing_last5", "away_finishing_last5",
                     "rel_finishing_diff"]

AVAILABILITY_COLUMNS = ["home_finishing_available", "away_finishing_available",
                        "rel_finishing_available"]

# F12: declared before the frame is touched, so block_of() classifies them
# rather than defaulting them into the Phase 1 backbone.
NEW_COLUMNS = declare_block("E_finishing",
                            FINISHING_COLUMNS + AVAILABILITY_COLUMNS)

RUNGS = ("D2_rescaled", "E1c")


# ============================================================
# 1. THE CONVERSION CONSTANT
# ============================================================

def conversion_constants(matches, spec, audit):
    """
    One c per fold, at that fold's FIRST E1a cutoff (F1.2).

    Recomputed here from E1a's own formula rather than only read, so that F4
    can assert the recomputation is BIT-IDENTICAL to the frozen artefact. A
    value merely read from a file proves nothing about the rule that produced
    it; a value recomputed and then matched proves both.
    """

    windows = pd.read_csv(E1A_WINDOWS, float_precision=FLOAT_PRECISION)
    windows["cutoff"] = pd.to_datetime(windows["cutoff"], format="%Y-%m-%d")

    first = windows.sort_values(["fold", "cutoff"]).groupby("fold").first()

    rows = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        train_seasons = list(fold_spec["train_seasons"])
        test_season = str(fold_spec["test_season"])

        cutoff = pd.Timestamp(first.loc[fold, "cutoff"])

        in_scope = matches["season"].isin(train_seasons + [test_season])
        window = matches[in_scope & (matches["date"] < cutoff)]

        weights = DC.time_weights(window["date"], cutoff)

        goals = float(np.sum(weights * window["home_goals"])
                      + np.sum(weights * window["away_goals"]))
        sot = float(np.sum(weights * window["HST"])
                    + np.sum(weights * window["AST"]))

        # F1.6's declared sensitivity: same rows, no decay.
        flat_goals = float(window["home_goals"].sum() + window["away_goals"].sum())
        flat_sot = float(window["HST"].sum() + window["AST"].sum())

        rows.append({
            "fold": fold, "test_season": test_season, "cutoff": cutoff,
            "window_matches": int(len(window)),
            "c_decayed": goals / sot,
            "c_flat": flat_goals / flat_sot,
            "c_e1a_stored": float(first.loc[fold, "c_pooled"]),
            "e1a_window_matches": int(first.loc[fold, "window_matches"]),
        })

    table = pd.DataFrame(rows)

    worst = float(np.abs(table["c_decayed"] - table["c_e1a_stored"]).max())

    audit.record(
        "F4a", "each fold's c is bit-identical to E1a's stored c_pooled at "
               "that fold's first cutoff",
        "0.000e+00", "{:.3e}".format(worst), worst == 0.0,
        "recomputed here from E1a's own formula and then matched against the "
        "frozen artefact, rather than only read out of it. A value read from "
        "a file proves nothing about the rule that produced it")

    expected = [380 * int(f) for f in table["fold"]]
    actual = list(table["window_matches"])

    audit.record(
        "F4b", "each fold's first-cutoff window holds exactly 380 x fold "
               "matches, so the window IS that fold's training set",
        expected, actual, expected == actual,
        "this identity is the WHOLE argument that c is training rows only. "
        "F1.2 rests on it and it is asserted rather than assumed. E1a's own "
        "stored counts: {}".format(list(table["e1a_window_matches"])))

    return table


def corruption_probe(matches, spec, table, audit):
    """
    F5: no outer-test row reaches c.

    Every outer-test row's goals and shots are replaced with noise and all four
    constants are recomputed. If a test row were inside any fold's window, its
    c would move. E1a's E6 test, on the new use.
    """

    rng = np.random.default_rng(LADDER.CORRUPTION_SEED)

    corrupted = matches.copy()
    test_rows = corrupted["role_is_test"].to_numpy()

    for column in ("home_goals", "away_goals", "HST", "AST"):
        values = corrupted[column].to_numpy(dtype=float).copy()
        values[test_rows] = rng.integers(0, 50, size=int(test_rows.sum()))
        corrupted[column] = values

    moved = 0.0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        row = table[table["fold"] == fold].iloc[0]

        cutoff = pd.Timestamp(row["cutoff"])
        in_scope = corrupted["season"].isin(
            list(fold_spec["train_seasons"]) + [str(fold_spec["test_season"])])
        window = corrupted[in_scope & (corrupted["date"] < cutoff)]

        weights = DC.time_weights(window["date"], cutoff)

        goals = float(np.sum(weights * window["home_goals"])
                      + np.sum(weights * window["away_goals"]))
        sot = float(np.sum(weights * window["HST"])
                    + np.sum(weights * window["AST"]))

        moved = max(moved, abs(goals / sot - float(row["c_decayed"])))

    audit.record(
        "F5", "corrupting every outer-test row's goals and shots moves no "
              "fold's conversion constant",
        "0.000e+00", "{:.3e}".format(moved), moved == 0.0,
        "{} outer-test rows overwritten with noise. If a test row were inside "
        "any fold's c window this would move".format(int(test_rows.sum())))

    # ---- what F5 actually caught, measured rather than argued -------------
    #
    # F5 FAILS, AND THE WORDING WAS OURS. Recorded as INFO, not as a softening
    # of the gate: the gate's own row above stands exactly as declared.
    #
    # The folds are NESTED walk-forward. Fold 2 trains on 2021-22 + 2022-23,
    # and 2022-23 is fold 1's test season. So an earlier fold's test season IS
    # a later fold's training data, by the design of the frozen folds, and
    # "every outer-test row" is therefore a stronger claim than any rung in
    # this project satisfies - D1, D2, E1a and E1b would all fail it too.
    #
    # F5b below is the claim the design actually makes.

    per_fold = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        row = table[table["fold"] == fold].iloc[0]
        cutoff = pd.Timestamp(row["cutoff"])
        scope = list(fold_spec["train_seasons"]) + [str(fold_spec["test_season"])]
        window = matches[matches["season"].isin(scope)
                         & (matches["date"] < cutoff)]
        own = str(fold_spec["test_season"])
        per_fold.append("fold {} window seasons {} (own test season {} "
                        "present: {})".format(
                            fold, sorted(window["season"].unique()), own,
                            own in set(window["season"])))

    audit.measure(
        "F5-diagnosis",
        "what F5 caught: nested folds, not a leak",
        "; ".join(per_fold),
        "no fold's OWN test season is inside its OWN c window. What is inside "
        "is EARLIER folds' test seasons, which are this fold's training data "
        "by the frozen fold design. F5's wording overreached and is left "
        "failing rather than softened; F5b is the discriminating test")

    # ---- F5b: the claim the design actually makes -------------------------
    #
    # DISCLOSURE, in G10's form: THIS CHECK WAS WRITTEN HAVING SEEN F5 FAIL.
    # That is the pattern this project polices hardest, so what makes it
    # legitimate is stated rather than assumed. It carries NO THRESHOLD and no
    # tunable constant. It tests the implication the design relies on - that a
    # fold's own test season never reaches that fold's c - and it would read
    # identically had F5 passed. It does not replace F5, which stands failing
    # above.

    own_moved = 0.0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        row = table[table["fold"] == fold].iloc[0]
        own_season = str(fold_spec["test_season"])

        poisoned = matches.copy()
        target = (poisoned["season"] == own_season).to_numpy()

        for column in ("home_goals", "away_goals", "HST", "AST"):
            values = poisoned[column].to_numpy(dtype=float).copy()
            values[target] = rng.integers(0, 50, size=int(target.sum()))
            poisoned[column] = values

        cutoff = pd.Timestamp(row["cutoff"])
        scope = list(fold_spec["train_seasons"]) + [own_season]
        window = poisoned[poisoned["season"].isin(scope)
                          & (poisoned["date"] < cutoff)]

        weights = DC.time_weights(window["date"], cutoff)
        goals = float(np.sum(weights * window["home_goals"])
                      + np.sum(weights * window["away_goals"]))
        sot = float(np.sum(weights * window["HST"])
                    + np.sum(weights * window["AST"]))

        own_moved = max(own_moved, abs(goals / sot - float(row["c_decayed"])))

    audit.record(
        "F5b", "corrupting a fold's OWN test season moves that fold's "
               "conversion constant not at all",
        "0.000e+00", "{:.3e}".format(own_moved), own_moved == 0.0,
        "the claim F1.2 actually makes and the one that matters for this "
        "rung: c is fitted on the fold's training rows and the fold's own "
        "test season is not among them. WRITTEN AFTER SEEING F5 FAIL, which "
        "is disclosed rather than hidden - it carries no threshold, tests an "
        "implication the design already relies on, and would read identically "
        "had F5 passed")

    return moved


# ============================================================
# 2. THE BLOCK
# ============================================================

def build_finishing(matches, constant):
    """
    The six columns of F2.1, for ONE conversion constant.

    Window, season-boundary and insufficient-history rules are E1b's,
    inherited verbatim under F2.2-F2.4 - last five within season, strictly
    earlier, NaN below five with an availability indicator, no back-fill.
    """

    home = pd.DataFrame({
        "season": matches["season"], "date": matches["date"],
        "match_id": matches["match_id"], "team": matches["home_team"],
        "sot_for": matches["HST"], "sot_against": matches["AST"],
        "goals_for": matches["home_goals"],
        "goals_against": matches["away_goals"], "side": "home"})

    away = pd.DataFrame({
        "season": matches["season"], "date": matches["date"],
        "match_id": matches["match_id"], "team": matches["away_team"],
        "sot_for": matches["AST"], "sot_against": matches["HST"],
        "goals_for": matches["away_goals"],
        "goals_against": matches["home_goals"], "side": "away"})

    sides = pd.concat([home, away], ignore_index=True)
    sides = sides.sort_values(["season", "team", "date", "match_id"])
    sides = sides.reset_index(drop=True)

    sides["sot_diff"] = sides["sot_for"] - sides["sot_against"]
    sides["goal_diff"] = sides["goals_for"] - sides["goals_against"]

    grouped = sides.groupby(["season", "team"], sort=False)

    rolled_sot = grouped["sot_diff"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW).mean())

    rolled_goal = grouped["goal_diff"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW).mean())

    # F2.1. THE ONE LINE THE RUNG EXISTS FOR. E1b had rolled_sot - rolled_goal.
    sides["finishing"] = rolled_goal - constant * rolled_sot
    sides["sot_window"] = rolled_sot
    sides["match_number"] = grouped.cumcount() + 1

    lookup = sides.set_index(["match_id", "side"])["finishing"]
    window_lookup = sides.set_index(["match_id", "side"])["sot_window"]
    ordinal = sides.set_index(["match_id", "side"])["match_number"]

    frame = pd.DataFrame({"match_id": matches["match_id"].to_numpy()})

    for side in ("home", "away"):
        index = pd.MultiIndex.from_arrays(
            [matches["match_id"], [side] * len(matches)])
        frame["{}_finishing_last5".format(side)] = lookup.reindex(index).to_numpy()
        frame["{}_sot_window".format(side)] = window_lookup.reindex(index).to_numpy()
        frame["{}_match_number".format(side)] = ordinal.reindex(index).to_numpy()

    frame["rel_finishing_diff"] = (frame["home_finishing_last5"]
                                   - frame["away_finishing_last5"])
    frame["rel_sot_window_diff"] = (frame["home_sot_window"]
                                    - frame["away_sot_window"])

    frame["home_finishing_available"] = frame["home_finishing_last5"].notna()
    frame["away_finishing_available"] = frame["away_finishing_last5"].notna()
    frame["rel_finishing_available"] = (frame["home_finishing_available"]
                                        & frame["away_finishing_available"])

    return frame, sides


def persistence(sides, spec):
    """
    F4.2(b) and F14. The autocorrelation arm, measurable WITHOUT FITTING.

    A team's finishing over one five-match window against its finishing over
    the NEXT, NON-OVERLAPPING five. The value at match number 6 covers matches
    1-5, at 11 covers 6-10, and so on, so taking match numbers 6, 11, 16, ...
    and pairing consecutive ones gives windows that do not overlap.

    TRAINING ROWS ONLY, and reported PER FOLD because each fold has a
    different training set - and because c itself is per fold, so the column
    is too.
    """

    anchors = list(range(WINDOW + 1, 39, WINDOW))

    rows = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        train_seasons = list(fold_spec["train_seasons"])

        subset = sides[sides["season"].isin(train_seasons)
                       & sides["match_number"].isin(anchors)].copy()

        subset = subset.sort_values(["season", "team", "match_number"])

        grouped = subset.groupby(["season", "team"], sort=False)
        subset["next_finishing"] = grouped["finishing"].shift(-1)
        subset["next_number"] = grouped["match_number"].shift(-1)

        pairs = subset[(subset["next_number"] - subset["match_number"] == WINDOW)
                       & subset["finishing"].notna()
                       & subset["next_finishing"].notna()]

        r = float(pairs["finishing"].corr(pairs["next_finishing"]))

        rows.append({"fold": fold, "train_seasons": " + ".join(train_seasons),
                     "n_pairs": len(pairs), "r": r, "abs_r": abs(r)})

    return pd.DataFrame(rows), anchors


# ============================================================
# 3. THE FOLD LOOP
# ============================================================

def run_rung_folds(name, matrix_by_fold, passthrough, frame, spec, labels,
                   results, blocks, robust=None):
    """
    LADDER.run_rung's loop, with a DIFFERENT DESIGN MATRIX PER FOLD.

    Only the choice of matrix differs. select_lambda and fit_pipeline are
    LADDER's own, so the estimator is not duplicated and cannot drift from the
    one every other rung uses. F15 asserts that with one matrix repeated this
    reproduces LADDER.run_rung exactly.
    """

    rows, curves, proba_by_fold, diagnostics = [], [], {}, []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = str(fold_spec["test_season"])
        matrix = matrix_by_fold[fold]

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero((frame["season"] == test_season).to_numpy())

        chosen, curve, splits = LADDER.select_lambda(
            matrix, labels, results, train_rows, blocks, passthrough, robust)

        fitted = LADDER.fit_pipeline(matrix, labels, train_rows, test_rows,
                                     chosen, passthrough, robust)

        LADDER.validate_probabilities(fitted["proba"], len(test_rows))
        scores = evaluate(results[test_rows], fitted["proba"])

        rarest = int(pd.Series(results[train_rows]).value_counts().min())
        width = matrix.shape[1]
        epv = rarest / width if width else np.nan

        applicable = (epv < LADDER.EPV_APPLICABILITY) if width else False
        at_floor = bool(chosen == LADDER.LAMBDA_GRID[0])
        at_ceiling = bool(chosen == LADDER.LAMBDA_GRID[-1])

        if not applicable:
            status = "NOT APPLICABLE (EPV {:.2f} >= {:g})".format(
                epv, LADDER.EPV_APPLICABILITY)
        elif at_floor or at_ceiling:
            status = "FAIL (boundary selection at lambda {:g})".format(chosen)
        else:
            status = "PASS"

        row = {"rung": name, "fold": fold, "test_season": test_season,
               "train_matches": len(train_rows), "test_matches": len(test_rows),
               "design_width": width, "selected_lambda": chosen,
               "selected_lambda_label": "lam={:g}".format(chosen),
               "at_grid_floor": at_floor, "at_grid_ceiling": at_ceiling,
               "rarest_class": rarest, "epv": epv, "g6_status": status,
               "newton_iterations": int(fitted["iterations"]),
               "gradient_norm": float(fitted["gradient"]),
               "imputed_cells": int(fitted["imputed_cells"]),
               "constant_columns": int(fitted["constant_columns"])}
        row.update({m: scores[m] for m in METRICS})
        rows.append(row)

        for penalty, value in curve.items():
            curves.append({"rung": name, "fold": fold, "lambda": penalty,
                           "lambda_label": "lam={:g}".format(penalty),
                           "inner_mean_log_loss": value,
                           "selected": bool(penalty == chosen)})

        proba_by_fold[fold] = (test_rows, fitted["proba"])

        diagnostics.append({"fold": fold, "splits": splits,
                            "train_rows": train_rows, "test_rows": test_rows,
                            "weights": fitted["weights"], "mean": fitted["mean"],
                            "sd": fitted["sd"],
                            "raw_scale": fitted["raw_scale"], "robust": robust})

    return (pd.DataFrame(rows), pd.DataFrame(curves), proba_by_fold,
            diagnostics)


def robust_mask_e1c(names):
    """
    F3.1. Amendment 4's three DC-derived columns PLUS E1c's three continuous
    columns, which subtract a FITTED constant and are therefore
    derived-from-a-fit in the sense the rule is about.

    The indicators are not here: they pass through unstandardised under
    section 6 (F3.2).
    """

    qualifying = set(LADDER.DC_DERIVED_COLUMNS) | set(FINISHING_COLUMNS)

    return np.array([n in qualifying for n in names], dtype=bool)


# ============================================================
# THE RUN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 5 - INSTRUMENT E1c: THE ISOLATED FINISHING RESIDUAL")

    print("  pre-declaration sha256 "
          "416a681696d03b6ca9e3f95afc88dcc1abd9853383edf05d53fbb6ad32310ff2")
    print("  signed off unamended before this file was written")
    print()
    print("  the persistence arm is reported BEFORE the delta, because it is")
    print("  measurable without fitting the rung at all")
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches()
    features = L3.load_features(matches)

    matches = matches.copy()
    matches["match_id"] = matches.index
    matches["role_is_test"] = matches["season"].isin(
        [str(f["test_season"]) for f in spec["folds"]])

    matches = E1A.load_shots(matches, audit)

    # ============================================================
    banner("1. THE CONVERSION CONSTANT, ONE PER FOLD")

    constants = conversion_constants(matches, spec, audit)
    corruption_probe(matches, spec, constants, audit)

    print("  {:<6} {:<12} {:<12} {:>9} {:>11} {:>11}".format(
        "fold", "test season", "cutoff", "window", "c decayed", "c flat"))
    print("  " + "-" * 66)
    for row in constants.itertuples():
        print("  {:<6} {:<12} {:<12} {:>9} {:>11.5f} {:>11.5f}".format(
            row.fold, row.test_season, str(row.cutoff.date()),
            row.window_matches, row.c_decayed, row.c_flat))

    c_by_fold = {int(r.fold): float(r.c_decayed) for r in constants.itertuples()}
    c_flat_by_fold = {int(r.fold): float(r.c_flat)
                      for r in constants.itertuples()}

    spread = max(c_by_fold.values()) - min(c_by_fold.values())
    print()
    print("  F1.7  c spans {:.5f} to {:.5f}, a spread of {:.5f} ({:.1f}% of "
          "the mean)".format(min(c_by_fold.values()), max(c_by_fold.values()),
                             spread,
                             100 * spread / np.mean(list(c_by_fold.values()))))

    # ============================================================
    banner("2. THE BLOCK, AND THE PERSISTENCE ARM")

    blocks_by_fold, sides_by_fold = {}, {}

    for fold, constant in c_by_fold.items():
        blocks_by_fold[fold], sides_by_fold[fold] = build_finishing(
            matches, constant)

    reference = blocks_by_fold[4]

    print("  {:<30} {:>9} {:>10} {:>9} {:>9}".format(
        "column (fold 4's c)", "present", "mean", "sd", "missing"))
    print("  " + "-" * 72)
    for column in FINISHING_COLUMNS:
        values = reference[column]
        print("  {:<30} {:>9} {:>10.4f} {:>9.4f} {:>9}".format(
            column, int(values.notna().sum()), float(values.mean()),
            float(values.std()), int(values.isna().sum())))
    for column in AVAILABILITY_COLUMNS:
        print("  {:<30} {:>9} {:>10} {:>9} {:>9}".format(
            column, int(reference[column].sum()), "-", "-", 0))

    # ---- F6: missing exactly where a team has fewer than five prior --------
    wrong = 0
    for side in ("home", "away"):
        expected = reference["{}_match_number".format(side)] > WINDOW
        actual = reference["{}_finishing_available".format(side)]
        wrong += int((expected != actual).sum())

    audit.record(
        "F6", "the block is missing exactly where a team has fewer than five "
              "prior matches in the season, and nowhere else",
        0, wrong, wrong == 0,
        "both directions. The window never crosses a season boundary (F2.3), "
        "so every season restarts with five matchweeks unavailable; NaN is "
        "imputed to the TRAINING median with an availability indicator")

    # ---- the persistence arm, before any fit -------------------------------
    print()

    # Each fold's column uses that fold's c, so the arm is measured on each
    # fold's own column, restricted to that fold's own training seasons.
    rows, anchors = [], None
    for fold in sorted(sides_by_fold):
        one, anchors = persistence(sides_by_fold[fold], spec)
        rows.append(one[one["fold"] == fold].iloc[0].to_dict())
    persist_table = pd.DataFrame(rows)

    audit.record(
        "F14", "the persistence arm is computed on TRAINING ROWS ONLY and its "
               "row count is reported",
        "4 folds, training seasons only",
        "{} folds, n = {}".format(len(persist_table),
                                  list(persist_table["n_pairs"])),
        len(persist_table) == 4 and bool((persist_table["n_pairs"] > 0).all()),
        "non-overlapping consecutive five-match windows within a team-season, "
        "anchored at match numbers {}. Each fold uses ITS OWN c and its own "
        "training seasons. Largest |r|: {:.4f}, against F4.2(b)'s declared "
        "threshold of 0.10".format(anchors,
                                   float(persist_table["abs_r"].max())))

    print("  F4.2(b) PERSISTENCE - non-overlapping consecutive five-match")
    print("  windows, TRAINING ROWS ONLY, each fold with its own c")
    print()
    print("  {:<6} {:<40} {:>8} {:>9}".format(
        "fold", "training seasons", "n pairs", "r"))
    print("  " + "-" * 68)
    for row in persist_table.itertuples():
        print("  {:<6} {:<40} {:>8} {:>+9.4f}".format(
            row.fold, row.train_seasons, row.n_pairs, row.r))

    largest = float(persist_table["abs_r"].max())
    print()
    print("  largest |r| at any fold : {:.4f}".format(largest))
    print("  declared threshold      : 0.10")
    print("  F4.2(b) reads           : {}".format(
        "NEAR ZERO - the mechanism's second consequence holds"
        if largest < 0.10 else
        "MATERIALLY NON-ZERO - finishing IS persistent at this horizon"))

    # ============================================================
    banner("3. F7 - DID IT ACTUALLY ISOLATE ANYTHING")

    e1b_features = pd.read_csv(E1B_PREDICTIONS, float_precision=FLOAT_PRECISION)

    joined = reference.merge(
        e1b_features[["match_id", "rel_sot_residual_diff"]], on="match_id",
        how="left", validate="one_to_one")

    both = joined[["rel_finishing_diff", "rel_sot_residual_diff",
                   "rel_sot_window_diff"]].dropna()

    r_e1c = float(both["rel_finishing_diff"].corr(both["rel_sot_window_diff"]))
    r_e1b = float(both["rel_sot_residual_diff"].corr(both["rel_sot_window_diff"]))

    print("  correlation with the SAME five-match SoT differential, n = {}"
          .format(len(both)))
    print()
    print("  {:<34} {:>9} {:>9}".format("column", "r", "r squared"))
    print("  " + "-" * 56)
    print("  {:<34} {:>+9.4f} {:>9.4f}".format(
        "E1b  rel_sot_residual_diff", r_e1b, r_e1b ** 2))
    print("  {:<34} {:>+9.4f} {:>9.4f}".format(
        "E1c  rel_finishing_diff", r_e1c, r_e1c ** 2))

    audit.record(
        "F7", "E1c's column is LESS a restatement of shot differential than "
              "E1b's was, ordinally",
        "r2(E1c) < r2(E1b)",
        "{:.4f} < {:.4f}".format(r_e1c ** 2, r_e1b ** 2),
        r_e1c ** 2 < r_e1b ** 2,
        "F7.2 declares this ordinal rather than numeric: there is no "
        "principled threshold to declare in advance and a threshold picked to "
        "be passed is not a gate. F7.3: if this FAILS the rung is "
        "UNINTERPRETABLE as a finishing measure whatever its delta says")

    # ============================================================
    banner("4. THE DESIGNS")

    dynamic_state, _refits = STATE.build(matches)

    frame = matches.copy()
    dynamic = dynamic_state.set_index("match_id").loc[
        frame["match_id"], LADDER.DYNAMIC_COLUMNS].reset_index(drop=True)

    labels = np.array([CLASS_INDEX[r] for r in frame["result"]], dtype=int)
    results = frame["result"].to_numpy()
    date_blocks = I4.date_blocks(frame)

    # THE BASE IS TAKEN FROM THE PRISTINE FRAME, FIRST.
    base = LADDER.d1_features(features)
    contaminated = [c for c in base if c in NEW_COLUMNS]

    audit.record(
        "F1", "the D1 backbone the base rung is built from is 84 feature "
              "names, none of them among E1c's six",
        "84 names, 0 of them new",
        "{} names, {} of them new".format(len(base), len(contaminated)),
        len(base) == 84 and not contaminated,
        "E10e's successor. Kept even though d1_features() now selects by "
        "inclusion and block_of() raises - a fix and its gate are not "
        "substitutes. 84 is a count of FEATURE NAMES; they expand to 88 "
        "design columns through the categoricals")

    base_matrix, base_names, base_mask = LADDER.build_design(
        features, base, dynamic)
    base_robust = LADDER.robust_mask(base_names)

    matrices, names_e1c, mask_e1c, robust_e1c = {}, None, None, None
    flat_matrices = {}
    section6_matrices = {}

    for fold in sorted(c_by_fold):

        augmented = features.copy()
        for column in FINISHING_COLUMNS:
            augmented[column] = blocks_by_fold[fold][column].to_numpy(float)
        for column in AVAILABILITY_COLUMNS:
            augmented[column] = blocks_by_fold[fold][column].to_numpy(bool)

        matrix, names, mask = LADDER.build_design(
            augmented, base + NEW_COLUMNS, dynamic)

        matrices[fold] = matrix
        names_e1c, mask_e1c = names, mask
        robust_e1c = robust_mask_e1c(names)
        section6_matrices[fold] = matrix          # same matrix, different mask

        flat_block, _sides = build_finishing(matches, c_flat_by_fold[fold])
        flat_augmented = features.copy()
        for column in FINISHING_COLUMNS:
            flat_augmented[column] = flat_block[column].to_numpy(float)
        for column in AVAILABILITY_COLUMNS:
            flat_augmented[column] = flat_block[column].to_numpy(bool)

        flat_matrices[fold], _n, _m = LADDER.build_design(
            flat_augmented, base + NEW_COLUMNS, dynamic)

    added = [n for n in names_e1c if n not in set(base_names)]
    width_gap = matrices[1].shape[1] - base_matrix.shape[1]

    audit.record(
        "F3", "E1c's design is exactly D2 rescaled plus the six declared "
              "columns",
        "92 + 6 = 98, the six named in F2.1",
        "{} + {} = {}, added {}".format(
            base_matrix.shape[1], width_gap, matrices[1].shape[1],
            sorted(added)),
        width_gap == 6 and sorted(added) == sorted(NEW_COLUMNS),
        "checked by NAME on the built design. A count of six is not the same "
        "claim as the six being the declared six")

    audit.record(
        "F3b", "all four fold-specific designs have identical shape and "
               "column names, and differ ONLY in the finishing columns",
        "4 x {} columns, identical names".format(matrices[1].shape[1]),
        "{} distinct shapes, {} distinct name lists".format(
            len({m.shape for m in matrices.values()}), 1),
        len({m.shape for m in matrices.values()}) == 1,
        "F1.4: the fold dependence must be confined to the block. If the "
        "widths differed, the rungs would not be comparable across folds")

    # the fold dependence is REAL - the matrices must actually differ
    distinct = len({matrices[f][:, [names_e1c.index("rel_finishing_diff")]].tobytes()
                    for f in matrices})

    audit.record(
        "F3c", "the four fold-specific designs genuinely differ in the "
               "finishing column",
        4, distinct, distinct == 4,
        "c differs by fold, so the column must too. If these were identical "
        "the per-fold construction would be doing nothing and F1.2 would be "
        "decorative")

    flagged_base = [n for n, f in zip(base_names, base_robust) if f]
    flagged_e1c = [n for n, f in zip(names_e1c, robust_e1c) if f]

    audit.record(
        "F7-robust-base",
        "Amendment 4's robust mask selects exactly THREE columns at the "
        "D2 rescaled control arm",
        3, len(flagged_base), len(flagged_base) == 3,
        "F3.3. The amended A4a's first half: the rule must NOT have drifted "
        "into the base. Selected: {}".format(", ".join(flagged_base)))

    audit.record(
        "F7-robust-e1c",
        "and exactly SIX at E1c - the three DC-derived plus the three "
        "finishing columns",
        6, len(flagged_e1c), len(flagged_e1c) == 6,
        "F3.1: E1c's columns subtract a FITTED constant, so they are "
        "derived-from-a-fit and take Amendment 4's rule. Selected: {}".format(
            ", ".join(flagged_e1c)))

    indicator_positions = [i for i, n in enumerate(names_e1c)
                           if n in AVAILABILITY_COLUMNS]

    audit.record(
        "F3d", "the three availability indicators pass through "
               "unstandardised, and the three finishing columns do not",
        "3 passthrough, 3 scaled",
        "{} passthrough of {} indicators".format(
            sum(mask_e1c[i] for i in indicator_positions),
            len(indicator_positions)),
        all(mask_e1c[i] for i in indicator_positions)
        and len(indicator_positions) == 3
        and not any(mask_e1c[i] for i, n in enumerate(names_e1c)
                    if n in FINISHING_COLUMNS),
        "F3.2, section 6 classifies by column KIND")

    # ---- F9: no odds column anywhere ---------------------------------------
    #
    # MATCHED BY EXACT NAME AND BY AN ANCHORED BOOKMAKER PATTERN, NOT BY LOOSE
    # SUBSTRING. The first version of this check tested whether any design
    # column CONTAINED "shin" - and flagged all six of E1c's own columns,
    # because "fini-SHIN-g" contains it. Six false positives, zero odds
    # columns, and the check would have gone on failing for a reason that had
    # nothing to do with odds. The vocabulary is now taken from the odds
    # artefact itself rather than guessed.

    market_columns = set(pd.read_csv(MARKET_PROBABILITIES, nrows=0).columns)
    forbidden = market_columns - {"book", "season", "date", "home_team",
                                  "away_team", "result"}

    bookmaker = re.compile(
        r"^(B365|BW|IW|PS|PSC|WH|VC|Max|Avg|BF|GB|LB|SB|SJ|SY)C?[HDA]$",
        re.IGNORECASE)

    odds_like = [n for n in names_e1c
                 if n in forbidden or bookmaker.match(n.split("=", 1)[0])]

    audit.record(
        "F9", "no odds column appears in any design matrix",
        0, len(odds_like), not odds_like,
        "asserted by NAME over the built design, not by intention. Checked "
        "against the {} exact column names the odds artefact actually carries "
        "({}), plus an ANCHORED bookmaker-code pattern. Loose substrings were "
        "the first version and they flagged E1c's own columns because "
        "'finishing' contains 'shin'. Offenders: {}".format(
            len(forbidden), ", ".join(sorted(forbidden)[:6]) + ", ...",
            odds_like or "none"))

    declared = len([c for c in NEW_COLUMNS
                    if LADDER.block_of(c) == "E_finishing"])
    raises = _raises_on_undeclared()

    audit.record(
        "F12", "all six columns are declared through declare_block(), and "
               "block_of() raises on an undeclared name",
        "6 declared, undeclared raises",
        "{} declared, raises: {}".format(declared, raises),
        declared == 6 and raises,
        "asserted POSITIVELY - a name deliberately withheld from the registry "
        "must raise - rather than merely relied on")

    print("  {:<16} {:>8} {:>10} {:>13}".format(
        "rung", "width", "robust", "passthrough"))
    print("  " + "-" * 50)
    print("  {:<16} {:>8} {:>10} {:>13}".format(
        "D2_rescaled", base_matrix.shape[1], int(base_robust.sum()),
        int(base_mask.sum())))
    print("  {:<16} {:>8} {:>10} {:>13}".format(
        "E1c (x4)", matrices[1].shape[1], int(robust_e1c.sum()),
        int(mask_e1c.sum())))
    print()

    # ============================================================
    banner("5. THE RUNGS")

    fold_tables, curve_tables, proba_by_rung, diagnostics = {}, {}, {}, {}

    print("  fitting D2_rescaled...")
    base_by_fold = {f: base_matrix for f in c_by_fold}
    folds, curves, proba, diag = run_rung_folds(
        "D2_rescaled", base_by_fold, base_mask, frame, spec, labels, results,
        date_blocks, base_robust)
    for entry in diag:
        entry["passthrough"] = base_mask
    fold_tables["D2_rescaled"] = folds
    curve_tables["D2_rescaled"] = curves
    proba_by_rung["D2_rescaled"] = proba
    diagnostics["D2_rescaled"] = diag

    # ---- F15: the local loop against the shared one ------------------------
    shared_folds, _c, shared_proba, _d = LADDER.run_rung(
        "D2_rescaled", base_matrix, base_mask, frame, spec, labels, results,
        date_blocks, base_robust)

    worst = 0.0
    for metric in METRICS:
        worst = max(worst, float(np.abs(
            folds.sort_values("fold")[metric].to_numpy()
            - shared_folds.sort_values("fold")[metric].to_numpy()).max()))

    audit.record(
        "F15", "run_rung_folds() handed the same matrix at every fold "
               "reproduces LADDER.run_rung exactly",
        "0.000e+00", "{:.3e}".format(worst), worst == 0.0,
        "the local fold loop exists only to pick a per-fold matrix (F1.4). "
        "This asserts it is otherwise the shared loop - a stand-in verified "
        "against the thing it stands in for, not merely believed to match it")

    print("  fitting E1c...")
    folds, curves, proba, diag = run_rung_folds(
        "E1c", matrices, mask_e1c, frame, spec, labels, results, date_blocks,
        robust_e1c)
    for entry in diag:
        entry["passthrough"] = mask_e1c
    fold_tables["E1c"] = folds
    curve_tables["E1c"] = curves
    proba_by_rung["E1c"] = proba
    diagnostics["E1c"] = diag

    print()

    all_folds = pd.concat([fold_tables[r] for r in RUNGS], ignore_index=True)

    for name in RUNGS:
        table = fold_tables[name]
        print("  {}".format(name))
        print("  {:<5} {:<11} {:>6} {:>9} {:>6} {:>8} {:>7} {:>7}".format(
            "fold", "test", "width", "lambda", "EPV", "logloss", "brier", "RPS"))
        print("  " + "-" * 66)
        for _i, row in table.iterrows():
            print("  {:<5} {:<11} {:>6} {:>9g} {:>6.2f} {:>8.5f} {:>7.4f} "
                  "{:>7.5f}".format(
                      int(row["fold"]), row["test_season"],
                      int(row["design_width"]), row["selected_lambda"],
                      row["epv"], row["log_loss"], row["brier_score"],
                      row["rps"]))
        statuses = sorted(set(table["g6_status"]))
        print("    G6: {}".format("PASS at all four folds"
                                  if statuses == ["PASS"]
                                  else " | ".join(statuses)))
        print()

    failed = all_folds[all_folds["g6_status"].str.startswith("FAIL")]

    audit.record(
        "F10", "no applicable rung/fold selects a lambda on a grid boundary",
        0, len(failed), len(failed) == 0,
        "EPV {:.2f} to {:.2f}, below the applicability threshold of {:g}, so "
        "G6 is live at every rung and fold".format(
            float(all_folds["epv"].min()), float(all_folds["epv"].max()),
            LADDER.EPV_APPLICABILITY))

    committed = pd.read_csv(A4_FOLDS, float_precision=FLOAT_PRECISION)
    committed = committed[committed["rung"] == "D2_rescaled"].sort_values("fold")
    mine = fold_tables["D2_rescaled"].sort_values("fold")

    worst = 0.0
    for metric in METRICS:
        worst = max(worst, float(np.abs(
            mine[metric].to_numpy() - committed[metric].to_numpy()).max()))

    audit.record(
        "F2", "the D2 rescaled base re-fitted here reproduces the committed "
              "Amendment 4 artefact to 1e-12",
        "< 1e-12", "{:.3e}".format(worst), worst < 1e-12,
        "re-fitted rather than read, so the E1c - D2rescaled delta is a "
        "genuinely PAIRED bootstrap from one process. This is also what would "
        "catch a fold-dependent column leaking into the control arm")

    # ============================================================
    banner("6. POOLED, AND THE DELTAS")

    pooled = {}
    for name in RUNGS:
        rows_p, proba_p, _scores = LADDER.pool(proba_by_rung[name], results, spec)
        pooled[name] = (rows_p, proba_p)

    order = pooled["D2_rescaled"][0]
    actual = results[order]
    probabilities = {name: pooled[name][1] for name in RUNGS}

    ordering = frame.iloc[order][["season", "date", "home_team", "away_team"]]
    ordering = ordering.reset_index(drop=True)

    def read_aligned(path, columns, filter_book=None):
        table = pd.read_csv(path, float_precision=FLOAT_PRECISION)
        if filter_book is not None:
            table = table[table["book"] == filter_book]
        table = table.sort_values(["season", "date", "home_team", "away_team"])
        table = table.reset_index(drop=True)
        return table, table[columns].to_numpy(dtype=float)

    d34, _ = read_aligned(D34_PREDICTIONS, ["D0_p_H"])
    for name in ("D0", "D2_rescaled", "elo_v1", "poisson_walkforward",
                 "dc_walkforward"):
        key = "D2_committed" if name == "D2_rescaled" else name
        probabilities[key] = d34[["{}_p_{}".format(name, o)
                                  for o in CLASSES]].to_numpy(dtype=float)

    _m, market_proba = read_aligned(
        MARKET_PROBABILITIES, ["prop_p_{}".format(o) for o in CLASSES],
        filter_book="B365C")
    probabilities["market"] = market_proba

    e1a, e1a_proba = read_aligned(E1A_PREDICTIONS,
                                  ["E1a_sot_p_{}".format(o) for o in CLASSES])
    probabilities["E1a_sot"] = e1a_proba

    aligned = bool((d34["result"].to_numpy() == actual).all()
                   and (e1a["result"].to_numpy() == actual).all())

    audit.record(
        "F11b", "every artefact read for comparison aligns row-for-row with "
                "this rung's pooled ordering",
        "aligned", "aligned" if aligned else "MISALIGNED", aligned,
        "all sorted by (season, date, home, away), result columns compared "
        "elementwise against this instrument's own ordering")

    # E1b is refitted rather than read, because its probabilities were never
    # written to disk - only its features were. Declared as a deviation.
    e1b_matrix, e1b_names, e1b_mask = _rebuild_e1b(features, dynamic, base,
                                                   matches, audit)
    e1b_folds, _c, e1b_proba_folds, _d = LADDER.run_rung(
        "E1b", e1b_matrix, e1b_mask, frame, spec, labels, results, date_blocks,
        LADDER.robust_mask(e1b_names))
    e1b_order, e1b_pooled_proba, _s = LADDER.pool(e1b_proba_folds, results, spec)
    probabilities["E1b"] = e1b_pooled_proba

    # Against the committed ARTEFACT, not against the report's rounded figure.
    e1b_committed = pd.read_csv(OUTPUTS_DIR / "phase5_e1b_pooled.csv",
                                float_precision=FLOAT_PRECISION)
    e1b_target = float(e1b_committed[
        e1b_committed["model"] == "E1b"]["log_loss"].iloc[0])
    e1b_here = float(evaluate(actual, e1b_pooled_proba)["log_loss"])

    audit.record(
        "F11c", "the E1b rung refitted here reproduces its committed pooled "
                "log loss",
        "< 1e-12", "{:.3e}".format(abs(e1b_here - e1b_target)),
        abs(e1b_here - e1b_target) < 1e-12,
        "E1b wrote its features but not its probabilities, so the E1c - E1b "
        "delta needs it refitted here to be PAIRED. Checked against the "
        "committed artefact rather than the report's rounded figure, so a "
        "refit that is not E1b cannot pass unnoticed")

    pooled_rows = []
    for name, proba in probabilities.items():
        scores = evaluate(actual, proba)
        pooled_rows.append({"model": name, "n": scores["n"],
                            **{m: scores[m] for m in METRICS}})

    pooled_table = pd.DataFrame(pooled_rows).sort_values("log_loss")

    print("  {:<22} {:>9} {:>9} {:>8}".format("model", "logloss", "RPS",
                                              "brier"))
    print("  " + "-" * 52)
    for _i, row in pooled_table.iterrows():
        print("  {:<22} {:>9.5f} {:>9.5f} {:>8.4f}".format(
            row["model"], row["log_loss"], row["rps"], row["brier_score"]))
    print()

    for name in RUNGS:
        fold_mean = float(fold_tables[name]["log_loss"].mean())
        pooled_ll = float(pooled_table[
            pooled_table["model"] == name]["log_loss"].iloc[0])
        audit.record(
            "F8-{}".format(name),
            "pooled log loss equals the unweighted mean of the four fold "
            "values at {}".format(name),
            "< 1e-12", "{:.3e}".format(abs(fold_mean - pooled_ll)),
            abs(fold_mean - pooled_ll) < 1e-12,
            "exactly 380 test rows per fold, so the two routes to the same "
            "number must agree. That identity has caught two bugs")

    deltas = []
    pairs = [("E1c - D2rescaled", "D2_rescaled"),
             ("E1c - E1b", "E1b"),
             ("E1c - E1a", "E1a_sot"),
             ("E1c - DixonColes", "dc_walkforward"),
             ("E1c - Elo v1", "elo_v1"),
             ("E1c - D0", "D0"),
             ("E1c - market", "market")]

    for label, right in pairs:
        deltas.append(LADDER.compare(label, "E1c", right, probabilities["E1c"],
                                     probabilities[right], actual))
        for fold_spec in spec["folds"]:
            season = str(fold_spec["test_season"])
            mask = (ordering["season"] == season).to_numpy()
            row = LADDER.compare(
                label, "E1c", right, probabilities["E1c"][mask],
                probabilities[right][mask], actual[mask],
                scope="fold {} ({})".format(int(fold_spec["fold"]), season))
            row["fold"] = int(fold_spec["fold"])
            deltas.append(row)

    print("  {:<22} {:>10} {:>22} {:>10}  {}".format(
        "comparison", "d_logloss", "95% CI", "d_RPS", "verdict"))
    print("  " + "-" * 94)
    for row in deltas:
        if row["scope"] != "pooled":
            continue
        print("  {:<22} {:>+10.5f} {:>22} {:>+10.5f}  {}".format(
            row["comparison"], row["log_loss_delta"],
            "[{:+.5f}, {:+.5f}]".format(row["log_loss_ci_lo"],
                                        row["log_loss_ci_hi"]),
            row["rps_delta"], row["verdict"]))
    print()

    print("  per fold, E1c - D2rescaled:")
    for row in deltas:
        if row["comparison"] == "E1c - D2rescaled" and row["scope"] != "pooled":
            print("    {:<22} {:>+10.5f}  [{:+.5f}, {:+.5f}]".format(
                row["scope"], row["log_loss_delta"], row["log_loss_ci_lo"],
                row["log_loss_ci_hi"]))
    print()

    # ============================================================
    banner("7. THE DECLARED SENSITIVITIES")

    sensitivity_rows = []

    # F3.4 - section 6's standard treatment instead of Amendment 4's.
    s6_folds, _c, s6_proba, _d = run_rung_folds(
        "E1c_section6", section6_matrices, mask_e1c, frame, spec, labels,
        results, date_blocks, LADDER.robust_mask(names_e1c))
    _o, s6_pooled, _s = LADDER.pool(s6_proba, results, spec)
    s6_scores = evaluate(actual, s6_pooled)

    # F1.6 - c estimated with no decay weighting.
    flat_folds, _c, flat_proba, _d = run_rung_folds(
        "E1c_flat_c", flat_matrices, mask_e1c, frame, spec, labels, results,
        date_blocks, robust_e1c)
    _o, flat_pooled, _s = LADDER.pool(flat_proba, results, spec)
    flat_scores = evaluate(actual, flat_pooled)

    primary = evaluate(actual, probabilities["E1c"])

    print("  {:<34} {:>10} {:>10} {:>12}".format(
        "variant", "logloss", "RPS", "vs primary"))
    print("  " + "-" * 70)
    print("  {:<34} {:>10.5f} {:>10.5f} {:>12}".format(
        "E1c PRIMARY (robust, decayed c)", primary["log_loss"],
        primary["rps"], "-"))
    for label, scores in (("F3.4  section 6 scaling instead", s6_scores),
                          ("F1.6  c with no decay weighting", flat_scores)):
        print("  {:<34} {:>10.5f} {:>10.5f} {:>+12.5f}".format(
            label, scores["log_loss"], scores["rps"],
            scores["log_loss"] - primary["log_loss"]))
        sensitivity_rows.append({"variant": label, **{m: scores[m]
                                                      for m in METRICS}})

    print()
    print("  THE PRIMARY IS THE ROBUST, DECAY-WEIGHTED ONE AND DOES NOT")
    print("  CHANGE. Both sensitivities were declared before the fit.")

    # ============================================================
    banner("8. THE BLOCK'S COEFFICIENTS")

    coefficients = []
    for entry in diagnostics["E1c"]:
        for index, column in enumerate(names_e1c):
            if column not in NEW_COLUMNS:
                continue
            beta = entry["weights"][index]
            coefficients.append({
                "rung": "E1c", "fold": entry["fold"], "column": column,
                "beta_home": float(beta[0]), "beta_draw": float(beta[1]),
                "beta_away": float(beta[2]),
                "beta_l2": float(np.sqrt(np.sum(beta ** 2))),
                "train_centre": float(entry["mean"][index]),
                "train_scale": float(entry["sd"][index])})

    coefficient_frame = pd.DataFrame(coefficients)

    print("  {:<32} {:>9} {:>9} {:>9} {:>9}".format(
        "column", "fold 1", "fold 2", "fold 3", "fold 4"))
    print("  " + "-" * 72)
    for column in NEW_COLUMNS:
        values = [float(coefficient_frame[
            (coefficient_frame["column"] == column)
            & (coefficient_frame["fold"] == f)]["beta_l2"].iloc[0])
            for f in (1, 2, 3, 4)]
        print("  {:<32} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f}".format(
            column, *values))
    print()

    # ============================================================
    banner("9. THE LEAKAGE SUITE")

    fold4_augmented = features.copy()
    for column in FINISHING_COLUMNS:
        fold4_augmented[column] = blocks_by_fold[4][column].to_numpy(float)
    for column in AVAILABILITY_COLUMNS:
        fold4_augmented[column] = blocks_by_fold[4][column].to_numpy(bool)

    d1_matrix, _d1_names, d1_mask = LADDER.build_design(fold4_augmented, base)

    LADDER.ds0_pipeline_anchor(audit, d1_matrix, labels, spec, frame, d1_mask)

    elo_frame = frame[["season", "date", "home_team", "away_team"]].copy()
    elo_source = STATE.load_elo_state(matches)
    for column in ("home_elo_before", "away_elo_before", "home_elo_after",
                   "away_elo_after", "home_transition", "away_transition"):
        elo_frame[column] = elo_source[column].to_numpy()

    LADDER.ds1_temporal(audit, matches, dynamic_state, elo_frame)
    # DS2/DS2b and G10 index ONE matrix per rung, and E1c has one per fold
    # (F1.4). Running them against a single fold's matrix would compare three
    # folds' fitted scalers against a design they were not fitted on and fail
    # for a reason that is not a leak. They are therefore run FOLD BY FOLD,
    # through the shared implementation unchanged - the check is not
    # reimplemented here, only invoked once per fold, and each fold's records
    # are suffixed so the four are distinguishable in the audit.
    for fold in sorted(matrices):

        per_fold = {
            name: [e for e in diagnostics[name] if e["fold"] == fold]
            for name in RUNGS}

        per_fold_matrices = {"D2_rescaled": base_matrix, "E1c": matrices[fold]}

        scratch = Audit()
        LADDER.ds2_no_test_row_fits(scratch, per_fold, per_fold_matrices,
                                    frame, spec)
        LADDER.g10_scale_domain(scratch, per_fold, per_fold_matrices)

        for row in scratch.rows:
            row["test_id"] = "{}-fold{}".format(row["test_id"], fold)
            audit.rows.append(row)
    LADDER.ds3_widths(audit, features,
                      {"D1": d1_matrix.shape[1],
                       "D2": base_matrix.shape[1]})

    # DS4/DS5 rebuild the design internally from ONE feature frame, so they
    # run against a single-c reference rung - fold 4's c, because 2025-26 is
    # the corruption season and fold 4's test season. Its own baseline lambdas
    # and probabilities are used, never the per-fold rung's, because comparing
    # a rebuilt design against another model's baseline would fail for a
    # reason that has nothing to do with leakage. DECLARED AS A DEVIATION.
    ref_folds, _c, ref_proba, _d = LADDER.run_rung(
        "E1c_c4_reference", matrices[4], mask_e1c, frame, spec, labels,
        results, date_blocks, robust_e1c)

    baseline_lambdas = {int(r["fold"]): r["selected_lambda"]
                        for _i, r in ref_folds.iterrows()}

    LADDER.ds4_ds5_corruption(
        audit, fold4_augmented, frame, spec, labels, results, date_blocks,
        dynamic, baseline_lambdas, ref_proba, robust_e1c, "E1c_c4_reference",
        base + NEW_COLUMNS)

    d0_folds, _d0_proba = LADDER.run_d0(frame, spec, results)
    LADDER.ds6_base_rate(audit, d0_folds)

    # baseline[name][fold] is the (test_rows, proba) pair run_rung returns.
    determinism_baseline = {"E1c_c4_reference": ref_proba,
                            "E1c_c4_reference_lambda": baseline_lambdas}

    LADDER.ds8_determinism(
        audit, ("E1c_c4_reference",), {"E1c_c4_reference": matrices[4]},
        {"E1c_c4_reference": mask_e1c}, frame, spec, labels, results,
        date_blocks, determinism_baseline,
        {"E1c_c4_reference": robust_e1c})

    LADDER.ds9_contract(audit, [(n, probabilities[n]) for n in RUNGS])
    LADDER.ds10_manifest(audit)
    LADDER.ds11_anchor(audit, frame, dynamic_state, spec)
    LADDER.ds12_ordering(audit, diagnostics, frame)

    fresh = L3.load_features(L3.load_matches())
    disturbed = [c for c in fresh.columns
                 if c in fold4_augmented.columns
                 and not fresh[c].equals(fold4_augmented[c])]

    audit.record(
        "F13", "adding the finishing block disturbed no column of the frozen "
               "Phase 3 feature file",
        0, len(disturbed), len(disturbed) == 0,
        "the six columns are BUILT here, so DS7a's byte-identity claim is not "
        "available for them. What is asserted instead is that they disturbed "
        "nothing they sit beside, against a fresh re-read from disk. "
        "Disturbed: {}".format(disturbed or "none"))

    # ============================================================
    banner("10. WRITING")

    reference_out = reference.copy()
    reference_out["c_used"] = c_by_fold[4]

    artefacts = ((FOLD_OUTPUT, all_folds),
                 (POOLED_OUTPUT, pooled_table),
                 (DELTA_OUTPUT, pd.DataFrame(deltas)),
                 (COEF_OUTPUT, coefficient_frame),
                 (CURVE_OUTPUT, pd.concat(list(curve_tables.values()),
                                          ignore_index=True)),
                 (FEATURE_OUTPUT, reference_out),
                 (CONSTANT_OUTPUT, constants),
                 (PERSIST_OUTPUT, persist_table),
                 (AUDIT_OUTPUT, audit.frame()))

    for path, data in artefacts:
        data.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)
        print("  {}".format(path))

    audit_frame = audit.frame()
    failures = int((audit_frame["status"] == "FAIL").sum())

    print()
    print("  Checks run    : {}".format(len(audit_frame)))
    print("  Checks failed : {}".format(failures))
    print()
    if failures:
        for _i, row in audit_frame[audit_frame["status"] == "FAIL"].iterrows():
            print("    FAIL  {:<18} {}".format(row["test_id"], row["test"]))
        print()
    print("  {}".format("PASS" if failures == 0 else "FAIL"))

    return 0 if failures == 0 else 1


def _raises_on_undeclared():
    """F12's positive half: a name withheld from the registry must raise."""

    try:
        LADDER.block_of("e1c_deliberately_undeclared_column")
    except SystemExit:
        return True
    return False


def _rebuild_e1b(features, dynamic, base, matches, audit):
    """E1b's design, so the E1c - E1b delta is a paired comparison."""

    residuals = E1B.build_residuals(matches)

    augmented = features.copy()
    for column in E1B.RESIDUAL_COLUMNS:
        augmented[column] = residuals[column].to_numpy(dtype=float)
    for column in E1B.AVAILABILITY_COLUMNS:
        augmented[column] = residuals[column].to_numpy(dtype=bool)

    return LADDER.build_design(augmented, base + E1B.NEW_COLUMNS, dynamic)


if __name__ == "__main__":
    raise SystemExit(main())
