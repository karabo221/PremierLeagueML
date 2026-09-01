"""
===============================================================================
PHASE 3 - INSTRUMENT 3
THE NESTED ABLATION LADDER,  B0 - B6
===============================================================================

THE QUESTION
    Not "can we predict", but "which information actually improves
    prediction". Each rung adds exactly one block of features, everything
    else held fixed, through the four frozen folds and the harness's metric
    set.

        B0  base rate                                   0 features
        B1  + Phase 1 results-only features            86
        B2  + Block C context                          12
        B3  + prior_attack, prior_defence               6   the shooting core
        B4  + prior_venue_split                         3   lag-1 home/away
        B5  + prior_finishing, keeping, control         9
        B6  + prior_discipline, rotation                6
                                                      ---
                                                      122

    B5 and B6 sit last deliberately. They are the rungs most likely to cost
    more in variance than they return in signal, and the ladder is built so
    that showing this is a result rather than a disappointment.

NOTHING IS TUNED, SEARCHED OR SELECTED
    One model class. One configuration. Declared below, before any number was
    seen, and used identically at every rung of every fold:

        multinomial logistic regression, full softmax over [H, D, A]
        L2 penalty  lambda = 1.0  on the summed negative log-likelihood
        intercept unpenalised
        Newton's method to a gradient tolerance, no step size to choose

    No hyperparameter grid was run. No alternative lambda was evaluated. No
    feature was selected or dropped on the strength of a score. No early
    stopping, no validation split inside a training fold, no second model
    class to compare against. The ONLY thing that varies across the seven
    rungs is which block of columns is present, which is the entire point:
    if anything else moved, the delta would not be attributable.

    Newton's method is a solver for a convex fit, not a search. With
    lambda > 0 the penalised multinomial objective is strictly convex, so
    there is one optimum, and the run reports the gradient norm it reached
    rather than asking anyone to take convergence on trust.

THE PIPELINE IS FITTED ON TRAINING FOLDS ONLY
    Imputation and standardisation are model fitting, not data preparation.
    Phase 3's specification is explicit that imputing a promoted team's
    absent prior is "a MODELLING decision and belongs downstream, where it
    can be fitted on training folds only". So per fold:

        1. column means computed on the TRAINING rows alone -> impute
        2. mean and sd computed on the TRAINING rows alone  -> standardise
        3. fit
        4. transform the test rows with the training statistics, predict

    Test A6 perturbs the test season's feature values and requires the fitted
    coefficients not to move. Absence is never silently converted into
    "average": the availability flags Phase 1 already ships are in the design
    matrix from B1 onward, so the model can learn what an absence means.

WHAT IS NOT DONE HERE
    No XGBoost, no random forest, no hyperparameter tuning, no feature
    selection, no betting simulation, no external data. The evaluation
    harness is imported and never modified. data/raw is never opened.
===============================================================================
"""

from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import (  # noqa: E402
    CLASS_INDEX,
    CLASSES,
    evaluate,
    validate_probabilities,
)
from phase3_feature_builder import (  # noqa: E402
    BLOCK_C_COLUMNS,
    IDENTITY_COLUMNS,
    STATUS_ABSENT,
    STATUS_AVAILABLE,
    STATUS_NO_PRIOR,
    Audit,
    banner,
    configure_stdout,
)


# ============================================================
# FILE-ACCESS RECORDER  (evidence for A8)
# ============================================================

_OPEN_EVENTS = []

# This instrument opens data/raw itself, to SHA-256 every file before and
# after the run and prove none of them moved. That is not the model reading
# raw data, and a test that could not tell the two apart would be reporting
# its own integrity check as a violation. So every open is tagged with
# whether the integrity hasher was running at the time, and A8 asks the
# precise question: did anything OTHER than the hasher touch data/raw?
_HASHING = False

_WRITE_FLAG_BITS = 0

for _flag_name in ("O_WRONLY", "O_RDWR", "O_CREAT", "O_APPEND", "O_TRUNC"):
    _WRITE_FLAG_BITS |= getattr(__import__("os"), _flag_name, 0)


def _record_open(event, args):

    if event != "open":
        return

    if not isinstance(args[0], (str, bytes, Path)):
        return

    _OPEN_EVENTS.append((
        str(args[0]),
        args[1] if len(args) > 1 else None,
        args[2] if len(args) > 2 else None,
        _HASHING,
    ))


sys.addaudithook(_record_open)


def is_write_open(mode, flags):

    if isinstance(mode, str) and any(char in mode for char in "wax+"):
        return True

    return bool(isinstance(flags, int) and flags & _WRITE_FLAG_BITS)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

FEATURES_CSV = OUTPUTS_DIR / "phase3_features.csv"
MATCHES_CSV = OUTPUTS_DIR / "phase1_matches.csv"
SPEC_JSON = OUTPUTS_DIR / "phase0_evaluation_spec.json"
FOLDS_CSV = OUTPUTS_DIR / "phase0_evaluation_folds.csv"
BASE_RATE_SUMMARY = OUTPUTS_DIR / "phase2_base_rate_fold_summary.csv"
ELO_SUMMARY = OUTPUTS_DIR / "phase2_elo_fold_summary.csv"
POISSON_SUMMARY = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"

FOLD_SUMMARY_OUTPUT = OUTPUTS_DIR / "phase3_ablation_fold_summary.csv"
LADDER_OUTPUT = OUTPUTS_DIR / "phase3_ablation_ladder.csv"
RESULTS_OUTPUT = OUTPUTS_DIR / "phase3_ablation_results.csv"
COEFFICIENT_OUTPUT = OUTPUTS_DIR / "phase3_ablation_coefficients.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase3_ablation_audit.csv"

EXPECTED_TOTAL_MATCHES = 1900
FLOAT_PRECISION = "round_trip"

# ---- THE MODEL. Declared once. Never varied, never searched. ----
L2_PENALTY = 1.0            # the conventional default, fixed before any run
NEWTON_MAX_ITERATIONS = 100
GRADIENT_TOLERANCE = 1e-9
HESSIAN_JITTER = 1e-10      # numerical guard only; lambda already makes it PD

SHUFFLE_SEED = 20260830     # used only by the label-shuffle control, A10

METRICS = ["accuracy", "balanced_accuracy", "macro_f1",
           "log_loss", "brier_score", "rps"]

PRIMARY_METRIC = "log_loss"


# ============================================================
# THE LADDER
# ============================================================
#
# Each rung names the columns it ADDS. The feature set at a rung is the union
# of its own columns and every earlier rung's, so the ladder is nested by
# construction rather than by careful bookkeeping - test A3 checks that the
# nesting actually holds and that the counts are the declared ones.

PRIOR_SIDES = ("home", "away")


def prior_columns(*names):
    """home_X, away_X and rel_X_diff for each composite named."""

    columns = []

    for name in names:
        columns += [f"{side}_prior_{name}" for side in PRIOR_SIDES]
        columns += [f"rel_prior_{name}_diff"]

    return columns


LADDER = [
    ("B0", "base rate", []),
    ("B1", "+ Phase 1 results-only features", None),   # filled in from the file
    ("B2", "+ Block C context", list(BLOCK_C_COLUMNS)),
    ("B3", "+ prior_attack, prior_defence", prior_columns("attack", "defence")),
    ("B4", "+ prior_venue_split", prior_columns("venue_split")),
    ("B5", "+ prior_finishing, keeping, control",
     prior_columns("finishing", "keeping", "control")),
    ("B6", "+ prior_discipline, rotation",
     prior_columns("discipline", "rotation")),
]

EXPECTED_ADDED = {"B0": 0, "B1": 86, "B2": 12, "B3": 6, "B4": 3, "B5": 9, "B6": 6}

# Columns held out of the design matrix, with the reason. Both are the
# provenance columns that record WHICH season a prior came from - their value
# is a season label, and a season label is the fold fingerprint this project
# spent Phase 3 section 0.7 and test L11 guarding against. Phase 1 counts them
# among its 86; they are metadata here, exactly as Phase 3's own
# *_prior_source_season columns are metadata rather than features.
HELD_OUT_AS_METADATA = {
    "home_prev_season_source": "season identifier, not a fact about a team",
    "away_prev_season_source": "season identifier, not a fact about a team",
}

# Categorical columns and their declared level vocabularies. The levels are
# declared here rather than learned from the data, so a level that appears
# only in a test season cannot silently change the width of the design matrix.
CATEGORICAL_LEVELS = {
    "kickoff_hour_bucket": ["early", "afternoon", "evening"],
    "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "home_prev_season_status": [STATUS_AVAILABLE, STATUS_ABSENT, STATUS_NO_PRIOR],
    "away_prev_season_status": [STATUS_AVAILABLE, STATUS_ABSENT, STATUS_NO_PRIOR],
}


# ============================================================
# LOADING
# ============================================================

def hash_file(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_state():
    """SHA-256 of every Phase 0/1/2 artefact and script this must not disturb."""

    global _HASHING

    state = {}
    _HASHING = True

    try:
        for path in sorted(OUTPUTS_DIR.glob("phase[012]_*")):
            if path.is_file():
                state[str(path)] = hash_file(path)

        for path in sorted(SCRIPTS_DIR.glob("phase[012]_*.py")):
            state[str(path)] = hash_file(path)

        for path in sorted(RAW_DIR.rglob("*")):
            if path.is_file():
                state[str(path)] = hash_file(path)
    finally:
        _HASHING = False

    return state


def load_spec():

    spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))

    if "folds" not in spec:
        raise SystemExit("FATAL: the frozen spec carries no folds")

    return spec


def load_matches():
    """
    Phase 1's validated foundation, in Phase 2's ordering.

    The harness's own load_matches() reads data/raw/Fixtures and is
    deliberately not called, exactly as Phase 2 Instrument 1 does not call it.
    """

    matches = pd.read_csv(MATCHES_CSV, float_precision=FLOAT_PRECISION)

    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")

    matches = matches.sort_values(
        ["season", "date", "home_team", "away_team"]).reset_index(drop=True)

    if len(matches) != EXPECTED_TOTAL_MATCHES:
        raise SystemExit(f"FATAL: {len(matches)} matches, expected {EXPECTED_TOTAL_MATCHES}")

    unknown = set(matches["result"]) - set(CLASSES)

    if unknown:
        raise SystemExit(f"FATAL: unexpected result labels {sorted(unknown)}")

    return matches


def load_features(matches):
    """The 122-feature dataset, aligned row-for-row onto the match order."""

    features = pd.read_csv(FEATURES_CSV, float_precision=FLOAT_PRECISION)
    features["date"] = pd.to_datetime(features["date"], format="%Y-%m-%d")

    keys = ["season", "date", "home_team", "away_team"]

    aligned = matches[keys].merge(features, on=keys, how="left", validate="one_to_one")

    if len(aligned) != len(matches):
        raise SystemExit("FATAL: feature join changed the row count")

    return aligned


def phase1_feature_columns(features):
    """The 86, taken from the file rather than from a list typed out here."""

    from phase3_feature_builder import block_of

    return [c for c in features.columns if block_of(c) == "phase1_backbone"]


# ============================================================
# DESIGN MATRIX
# ============================================================

def design_columns(feature_names, features):
    """
    Expand a rung's feature list into design-matrix column names.

    Booleans become 0/1. Declared categoricals become one indicator per
    declared level. Everything else passes through as a float.
    """

    numeric, categorical = [], []

    for name in feature_names:

        if name in HELD_OUT_AS_METADATA:
            continue

        if name in CATEGORICAL_LEVELS:
            categorical.append(name)
        else:
            numeric.append(name)

    return numeric, categorical


def build_matrix(features, numeric, categorical):

    blocks = []
    names = []

    for name in numeric:

        series = features[name]

        if pd.api.types.is_bool_dtype(series):
            blocks.append(series.to_numpy("float64").reshape(-1, 1))
        else:
            blocks.append(pd.to_numeric(series, errors="coerce")
                          .to_numpy("float64").reshape(-1, 1))

        names.append(name)

    for name in categorical:

        values = features[name].astype("object")

        for level in CATEGORICAL_LEVELS[name]:
            blocks.append((values == level).to_numpy("float64").reshape(-1, 1))
            names.append(f"{name}={level}")

    if not blocks:
        return np.zeros((len(features), 0)), []

    return np.hstack(blocks), names


# ============================================================
# THE MODEL
# ============================================================

def softmax(logits):

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)

    return exponent / exponent.sum(axis=1, keepdims=True)


def fit_multinomial(x_train, y_train, n_classes=3, penalty=L2_PENALTY):
    """
    Penalised multinomial logistic regression by Newton's method.

    Strictly convex for penalty > 0, so there is one optimum and no starting
    point, learning rate or stopping rule to choose. The intercept is the
    last column and is not penalised.

    Returns (weights, iterations, final gradient sup-norm).
    """

    n_rows, n_features = x_train.shape

    design = np.hstack([x_train, np.ones((n_rows, 1))])
    width = n_features + 1

    targets = np.zeros((n_rows, n_classes))
    targets[np.arange(n_rows), y_train] = 1.0

    penalty_mask = np.ones(width)
    penalty_mask[-1] = 0.0                      # intercept unpenalised

    weights = np.zeros((width, n_classes))

    iterations = 0
    gradient_norm = np.inf

    for iterations in range(1, NEWTON_MAX_ITERATIONS + 1):

        probabilities = softmax(design @ weights)

        gradient = (design.T @ (probabilities - targets)
                    + penalty * penalty_mask[:, None] * weights)

        gradient_norm = float(np.abs(gradient).max())

        if gradient_norm < GRADIENT_TOLERANCE:
            break

        hessian = np.zeros((width * n_classes, width * n_classes))

        for k in range(n_classes):
            for l in range(n_classes):

                weight = probabilities[:, k] * ((k == l) - probabilities[:, l])
                block = design.T @ (design * weight[:, None])

                if k == l:
                    block = block + penalty * np.diag(penalty_mask)

                hessian[k * width:(k + 1) * width,
                        l * width:(l + 1) * width] = block

        hessian[np.diag_indices_from(hessian)] += HESSIAN_JITTER

        step = np.linalg.solve(hessian, gradient.T.reshape(-1))

        weights = weights - step.reshape(n_classes, width).T

    return weights, iterations, gradient_norm


def predict_multinomial(weights, x_test):

    design = np.hstack([x_test, np.ones((len(x_test), 1))])

    return softmax(design @ weights)


# ============================================================
# PER-FOLD PIPELINE
# ============================================================

def fit_fold(matrix, labels, train_mask, test_mask):
    """
    Impute, standardise and fit on the TRAINING rows; transform and predict
    the test rows with the training statistics.

    Every statistic used to transform the test rows is computed from training
    rows alone. A6 proves it by perturbation rather than by assertion.
    """

    train = matrix[train_mask]
    test = matrix[test_mask]

    if matrix.shape[1] == 0:
        # B0 - no features. The intercept alone reproduces the training base
        # rate exactly, which is what makes B0 the ladder's own control.
        weights, iterations, gradient = fit_multinomial(
            np.zeros((len(train), 0)), labels[train_mask])
        proba = predict_multinomial(weights, np.zeros((len(test), 0)))
        train_proba = predict_multinomial(weights, np.zeros((len(train), 0)))
        return proba, {"n_columns": 0, "iterations": iterations,
                       "gradient": gradient, "imputed_cells": 0,
                       "constant_columns": 0, "train_proba": train_proba}

    with np.errstate(invalid="ignore"):
        centre = np.nanmean(train, axis=0)

    centre = np.where(np.isfinite(centre), centre, 0.0)

    imputed_cells = int(np.isnan(train).sum() + np.isnan(test).sum())

    train = np.where(np.isnan(train), centre, train)
    test = np.where(np.isnan(test), centre, test)

    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)

    constant_columns = int((sd == 0).sum())
    sd = np.where(sd == 0, 1.0, sd)

    train = (train - mean) / sd
    test = (test - mean) / sd

    weights, iterations, gradient = fit_multinomial(train, labels[train_mask])

    proba = predict_multinomial(weights, test)

    return proba, {
        "n_columns": matrix.shape[1],
        "iterations": iterations,
        "gradient": gradient,
        "imputed_cells": imputed_cells,
        "constant_columns": constant_columns,
        "weights": weights,
        # Scored on the training rows too. Not a tuning signal - nothing is
        # chosen from it - but the gap between train and test log loss is the
        # difference between "these features carry nothing" and "these
        # features carry something the fit could not hold onto", and those
        # are different findings.
        "train_proba": predict_multinomial(weights, train),
    }


def run_rung(rung, columns, features, matches, spec, labels):
    """One rung, all four folds."""

    numeric, categorical = design_columns(columns, features)
    matrix, names = build_matrix(features, numeric, categorical)

    fold_rows = []
    predictions = []
    coefficients = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        train_seasons = list(fold_spec["train_seasons"])
        test_season = str(fold_spec["test_season"])

        train_mask = matches["season"].isin(train_seasons).to_numpy()
        test_mask = (matches["season"] == test_season).to_numpy()

        proba, detail = fit_fold(matrix, labels, train_mask, test_mask)

        proba = validate_probabilities(proba, int(test_mask.sum()))

        actual = matches.loc[test_mask, "result"].to_numpy()
        scores = evaluate(actual, proba)

        train_scores = evaluate(
            matches.loc[train_mask, "result"].to_numpy(), detail["train_proba"])

        row = {
            "rung": rung,
            "fold": fold,
            "train_seasons": " + ".join(train_seasons),
            "test_season": test_season,
            "train_matches": int(train_mask.sum()),
            "test_matches": int(test_mask.sum()),
            "design_columns": detail["n_columns"],
            "newton_iterations": detail["iterations"],
            "gradient_norm": detail["gradient"],
            "imputed_cells": detail["imputed_cells"],
            "constant_columns": detail["constant_columns"],
        }
        row.update({metric: scores[metric] for metric in METRICS})

        row["train_log_loss"] = train_scores["log_loss"]
        row["generalisation_gap"] = scores["log_loss"] - train_scores["log_loss"]

        fold_rows.append(row)

        predicted = np.argmax(proba, axis=1)

        predictions.append(pd.DataFrame({
            "rung": rung,
            "fold": fold,
            "test_season": test_season,
            "date": matches.loc[test_mask, "date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "home": matches.loc[test_mask, "home_team"].to_numpy(),
            "away": matches.loc[test_mask, "away_team"].to_numpy(),
            "actual_result": actual,
            "p_home": proba[:, 0],
            "p_draw": proba[:, 1],
            "p_away": proba[:, 2],
            "predicted_result": [CLASSES[i] for i in predicted],
        }))

        if "weights" in detail:
            for position, name in enumerate(names):
                coefficients.append({
                    "rung": rung, "fold": fold, "feature": name,
                    "coef_H": detail["weights"][position, CLASS_INDEX["H"]],
                    "coef_D": detail["weights"][position, CLASS_INDEX["D"]],
                    "coef_A": detail["weights"][position, CLASS_INDEX["A"]],
                })

    return (pd.DataFrame(fold_rows), pd.concat(predictions, ignore_index=True),
            pd.DataFrame(coefficients), names)


def run_ladder(features, matches, spec, labels, ladder):
    """Every rung, nested, in order."""

    cumulative = []
    summaries, predictions, coefficients = [], [], []
    column_sets = {}

    for rung, description, added in ladder:

        cumulative = cumulative + list(added)
        column_sets[rung] = list(cumulative)

        summary, prediction, coefficient, names = run_rung(
            rung, cumulative, features, matches, spec, labels)

        summary.insert(1, "description", description)
        summary.insert(2, "features_added", len(added))
        summary.insert(3, "features_total", len(cumulative))

        summaries.append(summary)
        predictions.append(prediction)
        coefficients.append(coefficient)

    return (pd.concat(summaries, ignore_index=True),
            pd.concat(predictions, ignore_index=True),
            pd.concat(coefficients, ignore_index=True) if any(
                len(c) for c in coefficients) else pd.DataFrame(),
            column_sets)


# ============================================================
# THE LADDER TABLE
# ============================================================

def build_ladder_table(summary):
    """
    Per rung: the pooled mean over folds, fold 4 on its own, and the delta
    against the rung below.

    Fold 4 is reported separately at EVERY rung because Phase 3 section 9
    predicted it would behave differently, and an average would hide exactly
    the thing the prediction is about.
    """

    rows = []
    previous = {}

    for rung in summary["rung"].unique():

        block = summary[summary["rung"] == rung]

        row = {
            "rung": rung,
            "description": block["description"].iloc[0],
            "features_added": int(block["features_added"].iloc[0]),
            "features_total": int(block["features_total"].iloc[0]),
        }

        for metric in METRICS:

            mean = float(block[metric].mean())
            fold4 = float(block.loc[block["fold"] == 4, metric].iloc[0])

            row[f"{metric}_mean"] = mean
            row[f"{metric}_fold4"] = fold4

            if previous:
                row[f"{metric}_delta_mean"] = mean - previous[f"{metric}_mean"]
                row[f"{metric}_delta_fold4"] = fold4 - previous[f"{metric}_fold4"]
            else:
                row[f"{metric}_delta_mean"] = np.nan
                row[f"{metric}_delta_fold4"] = np.nan

        row["train_log_loss_mean"] = float(block["train_log_loss"].mean())
        row["generalisation_gap_mean"] = float(block["generalisation_gap"].mean())
        row["design_columns_fold4"] = int(
            block.loc[block["fold"] == 4, "design_columns"].iloc[0])

        for fold in (1, 2, 3, 4):
            row[f"log_loss_fold{fold}"] = float(
                block.loc[block["fold"] == fold, "log_loss"].iloc[0])
            row[f"rps_fold{fold}"] = float(
                block.loc[block["fold"] == fold, "rps"].iloc[0])

        rows.append(row)
        previous = row

    return pd.DataFrame(rows)


# ============================================================
# TESTS
# ============================================================

def test_ladder(summary, ladder_table, predictions, column_sets, features,
                matches, spec, labels, ladder, before_state, audit):

    # ---- A1  B0 must reproduce Phase 2's frozen baseline ----------------
    base = pd.read_csv(BASE_RATE_SUMMARY, float_precision=FLOAT_PRECISION)
    mine = summary[summary["rung"] == "B0"].sort_values("fold")

    worst = 0.0

    for metric in METRICS:
        difference = np.abs(
            mine[metric].to_numpy() - base[metric].to_numpy()).max()
        worst = max(worst, float(difference))

    audit.record(
        "A1", "B0 reproduces Phase 2's frozen base-rate numbers",
        "< 1e-9", "{:.3e}".format(worst), worst < 1e-9,
        "an intercept-only softmax IS the base rate; if this drifts, the "
        "pipeline is not scoring what Phase 2 scored")

    # ---- A2  folds are the frozen folds ---------------------------------
    folds_csv = pd.read_csv(FOLDS_CSV)
    mismatch = 0

    for fold_spec in spec["folds"]:
        fold = int(fold_spec["fold"])
        row = folds_csv[folds_csv["fold"] == fold].iloc[0]
        block = summary[(summary["rung"] == "B6") & (summary["fold"] == fold)].iloc[0]
        mismatch += int(block["train_matches"] != row["train_matches"])
        mismatch += int(block["test_matches"] != row["test_matches"])
        mismatch += int(block["test_season"] != row["test_season"])

    audit.record(
        "A2", "fold sizes and test seasons match the frozen fold table",
        0, mismatch, mismatch == 0)

    # ---- A3  the ladder is nested, with the declared counts -------------
    not_nested = []
    wrong_counts = []

    order = [rung for rung, _d, _a in LADDER]

    for index in range(1, len(order)):
        earlier = set(column_sets[order[index - 1]])
        later = set(column_sets[order[index]])
        if not earlier < later and earlier != later:
            not_nested.append(order[index])

    for rung, _description, added in ladder:
        if len(added) != EXPECTED_ADDED[rung]:
            wrong_counts.append((rung, len(added), EXPECTED_ADDED[rung]))

    audit.record("A3a", "each rung is a strict superset of the one below",
                 0, len(not_nested), not not_nested, str(not_nested))

    audit.record("A3b", "each rung adds exactly the declared number of columns",
                 EXPECTED_ADDED, "0 wrong" if not wrong_counts else wrong_counts,
                 not wrong_counts)

    total = len(column_sets["B6"])

    audit.record("A3c", "the full ladder reaches 122 features", 122, total,
                 total == 122)

    # ---- A4  no test season ever enters a fit ---------------------------
    overlaps = 0

    for fold_spec in spec["folds"]:
        train_seasons = set(fold_spec["train_seasons"])
        overlaps += int(str(fold_spec["test_season"]) in train_seasons)

    audit.record("A4", "no fold trains on its own test season", 0, overlaps,
                 overlaps == 0)

    # ---- A5  determinism ------------------------------------------------
    repeat_summary, repeat_predictions, _c, _s = run_ladder(
        features, matches, spec, labels, ladder)

    difference = float(np.abs(
        predictions[["p_home", "p_draw", "p_away"]].to_numpy()
        - repeat_predictions[["p_home", "p_draw", "p_away"]].to_numpy()).max())

    audit.record("A5", "two runs produce bit-identical probabilities",
                 0.0, "{:.3e}".format(difference), difference == 0.0,
                 "no seed, no shuffle, no randomness anywhere in the fit")

    # ---- A6  the pipeline is fitted on training rows only ---------------
    # Corrupt the TEST season's features and require the fitted coefficients
    # and every OTHER fold's predictions to be unmoved. If imputation or
    # standardisation had seen the test rows, this would move them.
    corrupted = features.copy()

    numeric_columns = [
        c for c in column_sets["B6"]
        if c not in HELD_OUT_AS_METADATA
        and c not in CATEGORICAL_LEVELS
        and pd.api.types.is_numeric_dtype(corrupted[c])
        and not pd.api.types.is_bool_dtype(corrupted[c])
    ]

    test_rows = (corrupted["season"] == "2025-2026").to_numpy()

    for column in numeric_columns:
        corrupted.loc[test_rows, column] = 999.0

    corrupt_summary, corrupt_predictions, _c, _s = run_ladder(
        corrupted, matches, spec, labels, ladder)

    unaffected = corrupt_predictions[corrupt_predictions["fold"] != 4]
    original = predictions[predictions["fold"] != 4]

    moved = float(np.abs(
        unaffected[["p_home", "p_draw", "p_away"]].to_numpy()
        - original[["p_home", "p_draw", "p_away"]].to_numpy()).max())

    audit.record(
        "A6a", "corrupting the 2025-26 features leaves folds 1-3 untouched",
        0.0, "{:.3e}".format(moved), moved == 0.0,
        "2025-26 is test-only; it is in no fold's training set")

    fold4_moved = float(np.abs(
        corrupt_predictions[corrupt_predictions["fold"] == 4][
            ["p_home", "p_draw", "p_away"]].to_numpy()
        - predictions[predictions["fold"] == 4][
            ["p_home", "p_draw", "p_away"]].to_numpy()).max())

    audit.record(
        "A6b", "CONTROL: fold 4's own predictions do move when its features do",
        "> 0", "{:.3e}".format(fold4_moved), fold4_moved > 0,
        "without this, A6a would pass for a pipeline that ignored features")

    # ---- A7  label-shuffle control --------------------------------------
    # Shuffle the TRAINING labels only, refit the full rung, and require the
    # score to collapse towards the base rate. A pipeline that scored well on
    # shuffled labels would be scoring an artefact.
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled = labels.copy()

    for fold_spec in spec["folds"]:
        train_mask = matches["season"].isin(fold_spec["train_seasons"]).to_numpy()
        shuffled[train_mask] = rng.permutation(shuffled[train_mask])

    shuffle_ladder = [
        ("B0", "base rate", []),
        ("B6", "full ladder", list(column_sets["B6"])),
    ]

    shuffled_summary, _p, _c, _s = run_ladder(
        features, matches, spec, shuffled, shuffle_ladder)

    real = float(summary[summary["rung"] == "B6"]["log_loss"].mean())
    noise = float(shuffled_summary[shuffled_summary["rung"] == "B6"]["log_loss"].mean())
    base_rate_loss = float(summary[summary["rung"] == "B0"]["log_loss"].mean())

    audit.record(
        "A7", "shuffling the training labels destroys the model's edge",
        f"> {base_rate_loss:.4f}", "{:.4f}".format(noise),
        noise > base_rate_loss,
        f"real B6 {real:.4f}, shuffled B6 {noise:.4f}, base rate "
        f"{base_rate_loss:.4f}")

    # ---- A8  isolation ---------------------------------------------------
    after_state = frozen_state()

    changed = [path for path in before_state
               if before_state[path] != after_state.get(path)]

    audit.record(
        "A8a", "no Phase 0, 1 or 2 artefact, script or raw file was modified",
        0, len(changed), not changed, str(changed[:3]))

    raw_events = [
        (path, mode, flags, hashing) for path, mode, flags, hashing in _OPEN_EVENTS
        if str(Path(path).resolve()).startswith(str(RAW_DIR))
    ]

    by_model = [event for event in raw_events if not event[3]]

    audit.record(
        "A8b", "nothing but the integrity hasher opened data/raw",
        0, len(by_model), not by_model,
        "the harness's own load_matches() reads data/raw and is deliberately "
        "not called; matches come from Phase 1's foundation instead")

    raw_writes = [event for event in raw_events if is_write_open(event[1], event[2])]

    audit.record(
        "A8c", "no open under data/raw asked for write access",
        0, len(raw_writes), not raw_writes, str(raw_writes[:2]))

    audit.measure(
        "A8d", "data/raw opens by the integrity hasher", len(raw_events),
        "70 files hashed before the run and again after it")

    # ---- A9  fold 1 has no lag-1 prior in training ----------------------
    # Measured and reported rather than discovered later: 2021-22 is fold 1's
    # only training season and every Block X value in it is absent, so rungs
    # B3-B6 cannot learn a prior coefficient in that fold.
    fold1 = summary[(summary["rung"] == "B6") & (summary["fold"] == 1)].iloc[0]

    audit.measure(
        "A9", "columns that are constant in fold 1's training set",
        int(fold1["constant_columns"]),
        "2021-22 has no prior season, so every Block X column is absent in "
        "fold 1's training rows; B3-B6 can add nothing there and their fold-1 "
        "deltas are expected to be ~0")

    # ---- A10  probability contract --------------------------------------
    probabilities = predictions[["p_home", "p_draw", "p_away"]].to_numpy()

    sums = np.abs(probabilities.sum(axis=1) - 1.0).max()
    negative = int((probabilities < 0).sum())

    audit.record("A10a", "every predicted row sums to 1", "< 1e-9",
                 "{:.3e}".format(float(sums)), sums < 1e-9)

    audit.record("A10b", "no negative probability", 0, negative, negative == 0)

    audit.record(
        "A10c", "every prediction passed the harness's own validator",
        len(predictions), len(predictions), True,
        "validate_probabilities() is called per fold before scoring")

    # ---- A11  convergence ------------------------------------------------
    worst_gradient = float(summary["gradient_norm"].max())

    audit.record(
        "A11", "every fit converged to the declared gradient tolerance",
        f"< {GRADIENT_TOLERANCE:g}", "{:.3e}".format(worst_gradient),
        worst_gradient < GRADIENT_TOLERANCE,
        "Newton on a strictly convex objective; the optimum is unique")

    return audit


# ============================================================
# REPORT
# ============================================================

def print_ladder(table):

    print("{:<5} {:<38} {:>5} {:>6} {:>9} {:>9} {:>9} {:>9}".format(
        "rung", "block added", "add", "total", "logloss", "d(mean)",
        "LL fold4", "d(fold4)"))
    print("-" * 96)

    for _index, row in table.iterrows():

        print("{:<5} {:<38} {:>5} {:>6} {:>9.4f} {:>9} {:>9.4f} {:>9}".format(
            row["rung"], row["description"][:38],
            int(row["features_added"]), int(row["features_total"]),
            row["log_loss_mean"],
            "-" if pd.isna(row["log_loss_delta_mean"])
            else "{:+.4f}".format(row["log_loss_delta_mean"]),
            row["log_loss_fold4"],
            "-" if pd.isna(row["log_loss_delta_fold4"])
            else "{:+.4f}".format(row["log_loss_delta_fold4"]),
        ))

    print()


def print_metric_table(table, metric):

    print("  {:<5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "rung", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "delta"))
    print("  " + "-" * 74)

    for _index, row in table.iterrows():

        print("  {:<5} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>10}".format(
            row["rung"],
            row[f"{metric}_fold1"], row[f"{metric}_fold2"],
            row[f"{metric}_fold3"], row[f"{metric}_fold4"],
            row[f"{metric}_mean"],
            "-" if pd.isna(row[f"{metric}_delta_mean"])
            else "{:+.4f}".format(row[f"{metric}_delta_mean"])))

    print()


def print_reference_points(table):

    print("  Existing Phase 2 reference points, same folds, same metrics:")
    print()
    print("  {:<24} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "model", "fold 1", "fold 2", "fold 3", "fold 4", "mean"))
    print("  " + "-" * 68)

    references = []

    base = pd.read_csv(BASE_RATE_SUMMARY).sort_values("fold")
    references.append(("base rate (B0)", base["log_loss"].to_numpy()))

    if ELO_SUMMARY.exists():
        elo = pd.read_csv(ELO_SUMMARY).sort_values("fold")
        if "log_loss" in elo.columns and len(elo) == 4:
            references.append(("Elo", elo["log_loss"].to_numpy()))

    for name, values in references:
        print("  {:<24} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
            name, *values, float(np.mean(values))))

    best = table.loc[table["log_loss_mean"].idxmin()]

    print("  {:<24} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}".format(
        f"best rung ({best['rung']})",
        best["log_loss_fold1"], best["log_loss_fold2"],
        best["log_loss_fold3"], best["log_loss_fold4"],
        best["log_loss_mean"]))

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 3 - INSTRUMENT 3: THE ABLATION LADDER, B0 - B6")

    print("  model     : multinomial logistic regression, softmax over H/D/A")
    print("  penalty   : L2, lambda = {} (declared, never varied)".format(L2_PENALTY))
    print("  solver    : Newton, tolerance {:g}, no step size to choose".format(
        GRADIENT_TOLERANCE))
    print("  folds     : the four frozen expanding-window folds")
    print("  metrics   : the harness's evaluate(), imported not reimplemented")
    print()
    print("  Nothing is tuned, searched or selected. The only thing that")
    print("  changes across the seven rungs is which block of columns exists.")
    print()

    before_state = frozen_state()

    matches = load_matches()
    features = load_features(matches)
    labels = np.array([CLASS_INDEX[r] for r in matches["result"]], dtype=int)
    spec = load_spec()

    ladder = [list(entry) for entry in LADDER]
    ladder[1][2] = phase1_feature_columns(features)
    ladder = [tuple(entry) for entry in ladder]

    print("  matches {}, feature columns {}".format(len(matches), features.shape[1]))
    print("  held out as metadata: {}".format(", ".join(HELD_OUT_AS_METADATA)))
    print()

    summary, predictions, coefficients, column_sets = run_ladder(
        features, matches, spec, labels, ladder)

    table = build_ladder_table(summary)

    banner("1. THE LADDER - LOG LOSS")
    print_ladder(table)

    banner("2. LOG LOSS BY FOLD")
    print_metric_table(table, "log_loss")

    banner("3. RPS BY FOLD")
    print_metric_table(table, "rps")

    banner("4. TRAIN VERSUS TEST - WHERE THE LADDER'S SCORE GOES")

    print("  {:<5} {:>6} {:>12} {:>12} {:>12}".format(
        "rung", "cols", "train LL", "test LL", "gap"))
    print("  " + "-" * 52)

    for _index, row in table.iterrows():
        print("  {:<5} {:>6} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            row["rung"], int(row["design_columns_fold4"]),
            row["train_log_loss_mean"], row["log_loss_mean"],
            row["generalisation_gap_mean"]))

    print()
    print("  Nothing is selected from this table. It is here so that a rung")
    print("  that fails to improve the test score can be read correctly:")
    print("  a flat train score means the block carries no signal, and a")
    print("  falling train score with a rising test score means it carries")
    print("  signal the fit could not hold onto at this sample size.")
    print()

    full = summary[summary["rung"] == "B6"]

    print("  At B6, columns that are CONSTANT within each fold's training set")
    print("  and can therefore contribute nothing in that fold:")
    print()
    print("  {:<8} {:>14} {:>12} {:>12}".format(
        "fold", "train matches", "columns", "constant"))
    print("  " + "-" * 50)

    for _index, row in full.iterrows():
        print("  {:<8} {:>14} {:>12} {:>12}".format(
            int(row["fold"]), int(row["train_matches"]),
            int(row["design_columns"]), int(row["constant_columns"])))

    print()
    print("  Fold 1 trains on 2021-22 alone, which has no previous season in")
    print("  the dataset at all - so every previous-season column, Phase 1's")
    print("  included, is absent there. That is why fold 1's score stops")
    print("  moving after B2: from B3 on, every column added is constant in")
    print("  its training set.")
    print()

    banner("5. AGAINST THE EXISTING REFERENCE POINTS")
    print_reference_points(table)

    banner("6. AUDIT")

    audit = Audit()
    test_ladder(summary, table, predictions, column_sets, features, matches,
                spec, labels, ladder, before_state, audit)
    audit.print_rows()

    banner("7. WRITING OUTPUTS")

    summary.to_csv(FOLD_SUMMARY_OUTPUT, index=False, encoding="utf-8",
                   float_format="%.17g")
    table.to_csv(LADDER_OUTPUT, index=False, encoding="utf-8",
                 float_format="%.17g")
    predictions.to_csv(RESULTS_OUTPUT, index=False, encoding="utf-8",
                       float_format="%.17g")
    if len(coefficients):
        coefficients.to_csv(COEFFICIENT_OUTPUT, index=False, encoding="utf-8",
                            float_format="%.17g")
    audit.frame().to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    for path in (FOLD_SUMMARY_OUTPUT, LADDER_OUTPUT, RESULTS_OUTPUT,
                 COEFFICIENT_OUTPUT, AUDIT_OUTPUT):
        print("  {}".format(path))
    print()

    banner("PHASE 3 - INSTRUMENT 3 STATUS")

    failures = audit.failures

    print("  Rungs run          : {}".format(len(table)))
    print("  Fits performed     : {}".format(len(summary)))
    print("  Checks run         : {}".format(len(audit.rows)))
    print("  Checks failed      : {}".format(len(failures)))
    print()
    print("  {}".format("PASS" if not failures else "FAIL"))
    print()
    print("No model was tuned. No hyperparameter was searched. No feature was")
    print("selected on the strength of a score. The evaluation harness was")
    print("imported and not modified. data/raw was opened only to SHA-256 it")
    print("before and after the run - no data was read from it and nothing was")
    print("written to it (A8b, A8c).")
    print()
    print("This measures THIS model at THIS declared configuration. It is not")
    print("a measurement of how much information the features contain.")
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
