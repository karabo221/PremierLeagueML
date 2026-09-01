"""
===============================================================================
PHASE 3 - INSTRUMENT 4
REGULARISATION SENSITIVITY OF THE B0 - B6 LADDER
===============================================================================

THE QUESTION

    Instrument 3 measured the ladder at one fixed penalty, lambda = 1.0, and
    closed by saying so:

        "This measures THIS model at THIS declared configuration. It is not a
         measurement of how much information the features contain."

    That is the confound this instrument removes. The frozen ladder's shape -
    B1 best, B2 through B6 monotonically worse, the generalisation gap widening
    from 0.007 at B0 to 0.300 at B6 - is equally consistent with two very
    different findings:

        (a) the FBref prior-season blocks carry no usable signal
        (b) lambda = 1 on 132 standardised columns against 380-1520 training
            rows could not hold onto the signal they do carry

    Both predict the same table. Only moving lambda separates them.

WHAT THIS IS NOT

    Not a new model. Not a feature change. Not a search for a better number.
    The feature builder, the evaluation harness, the folds, the metrics, the
    model class and the solver are all imported unchanged. The ONLY thing that
    varies is the L2 penalty, over a grid fixed in a committed file before the
    first fit was run:

        PHASE3_REGULARISATION_PREDECLARATION.txt

    That file declares the grid, the inner selection procedure, the tie-break,
    the primary metric, and - critically - what result would count as changing
    the original conclusion. Test R6 asserts that the grid used here is
    byte-identical to the grid declared there, so an edit to either cannot
    quietly license a result the other did not authorise.

THE SELECTION IS LEAKAGE-SAFE BY CONSTRUCTION, AND THEN PROVED

    Lambda is chosen per (rung, outer fold) by an expanding-window inner CV
    over calendar-date blocks, cut from that fold's TRAINING rows alone.
    The outer test season is never scored, never standardised against, and
    never reachable from the selector: a runtime tripwire (R4) raises if a
    test-season row index ever enters the inner procedure, and perturbation
    tests (R2, R3) corrupt the outer test season's features and labels and
    require every selected lambda to be unmoved.

    A full outer-test surface - every rung x fold x lambda - is also computed,
    because "does any lambda rescue B2-B6" deserves a direct answer. It is
    labelled ORACLE everywhere it appears and is never read by the selector.
    An oracle number bounds what selection could have achieved. It is not a
    result and is not permitted into the headline table.

WHAT IS WRITTEN, AND WHAT IS NOT

    Writes outputs/phase3_reg_*.csv and nothing else. The frozen lambda = 1
    reference (outputs/phase3_ablation_*.csv), every Phase 0/1/2 artefact,
    every script, and data/raw are SHA-256'd before and after and must not
    move (R8).
===============================================================================
"""

from pathlib import Path
import hashlib
import json
import sys
import time

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
    Audit,
    banner,
    block_of,
    configure_stdout,
)

# Instrument 3 is imported, not copied and not modified. The model, the
# design-matrix construction, the ladder itself and the categorical
# vocabularies all come from there, so a change to Instrument 3 changes this
# instrument too rather than silently diverging from it.
import phase3_ablation_ladder as L3  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

PREDECLARATION = PROJECT_ROOT / "PHASE3_REGULARISATION_PREDECLARATION.txt"

FROZEN_FOLD_SUMMARY = OUTPUTS_DIR / "phase3_ablation_fold_summary.csv"
FROZEN_LADDER = OUTPUTS_DIR / "phase3_ablation_ladder.csv"

CURVES_OUTPUT = OUTPUTS_DIR / "phase3_reg_lambda_curves.csv"
SELECTED_OUTPUT = OUTPUTS_DIR / "phase3_reg_selected.csv"
SURFACE_OUTPUT = OUTPUTS_DIR / "phase3_reg_surface.csv"
BLOCK_NORM_OUTPUT = OUTPUTS_DIR / "phase3_reg_block_norms.csv"
LADDER_OUTPUT = OUTPUTS_DIR / "phase3_reg_ladder.csv"
VERDICT_OUTPUT = OUTPUTS_DIR / "phase3_reg_verdict.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase3_reg_audit.csv"

FLOAT_PRECISION = "round_trip"

# ---- THE GRID.  Declared in the pre-declaration, asserted by R6. ----
LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

FROZEN_LAMBDA = 1.0             # Instrument 3's configuration, a grid point

N_INNER_SEGMENTS = 4            # Q1..Q4 -> three expanding inner folds
N_INNER_FOLDS = N_INNER_SEGMENTS - 1

METRICS = L3.METRICS
PRIMARY_METRIC = L3.PRIMARY_METRIC       # "log_loss"

# Thresholds sharpening the pre-declaration's REFRAMED prose ("systematically
# far from 1", "the gap collapses"). Pinned here before the first fit. REFRAMED
# is a DESCRIPTIVE annotation on top of the verdict; it cannot change the
# OVERTURNED / WEAKENED / UPHELD trichotomy, which is fully pre-specified.
REFRAMED_LAMBDA_FLOOR = 10.0
REFRAMED_GAP_SHRINK = 0.5

BLOCK_ORDER = ["phase1_backbone", "C_context", "X_prior_composite",
               "X_availability", "X_metadata", "identity"]


class LeakageError(RuntimeError):
    """Raised by the R4 tripwire the instant a test row reaches the selector."""


# ============================================================
# FROZEN-STATE HASHING
# ============================================================

def hash_file(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_state():
    """
    SHA-256 of everything this instrument must not disturb.

    Wider than Instrument 3's own: Phase 3's artefacts are frozen here too,
    because the lambda = 1 ladder is this experiment's reference and a run
    that overwrote it would destroy the thing it is measured against.
    """

    L3._HASHING = True
    state = {}

    try:
        for pattern in ("phase[012]_*", "phase3_*"):
            for path in sorted(OUTPUTS_DIR.glob(pattern)):
                if path.is_file() and not path.name.startswith("phase3_reg_"):
                    state[str(path)] = hash_file(path)

        for path in sorted(SCRIPTS_DIR.glob("phase*.py")):
            state[str(path)] = hash_file(path)

        if PREDECLARATION.exists():
            state[str(PREDECLARATION)] = hash_file(PREDECLARATION)

        if RAW_DIR.exists():
            for path in sorted(RAW_DIR.rglob("*")):
                if path.is_file():
                    state[str(path)] = hash_file(path)
    finally:
        L3._HASHING = False

    return state


# ============================================================
# THE INNER SPLIT
# ============================================================

def date_blocks(matches):
    """
    Calendar dates in chronological order. The atomic unit of the inner split.

    AMENDMENT 1 (see the pre-declaration). This was declared as the
    (season, matchweek) block and had to be changed, because a Premier League
    matchweek is not a contiguous stretch of time: postponed fixtures are
    replayed months later under their ORIGINAL matchweek number. 31 of the 190
    matchweeks span more than a week, and 2022-23 matchweek 8 spans 200 days,
    from 16 September to 4 April.

    Ordering those blocks by their earliest date and cutting between them
    produced inner folds that trained later in time than they validated -
    outer fold 1's second inner split trained to 2022-05-19 and validated from
    2021-12-28. The declared unit could not deliver the property it was
    declared for, so it was replaced with the thing it was a proxy for.

    The matchweek block existed to stop two CONTEMPORANEOUS matches landing on
    opposite sides of a boundary. The calendar date captures exactly that, and
    a matchweek divided by a boundary is divided precisely because its halves
    were played months apart and were never contemporaneous. R5c measures how
    often that happens rather than forbidding it.
    """

    frame = matches[["season", "matchweek", "date"]].copy()
    frame["row"] = np.arange(len(frame))

    blocks = []

    for date, part in frame.groupby("date", sort=True):
        blocks.append({
            "date": date,
            "min_date": date,
            "max_date": date,
            "rows": np.sort(part["row"].to_numpy()),
        })

    blocks.sort(key=lambda b: b["date"])

    return blocks


def inner_splits(train_rows, blocks):
    """
    Expanding-window inner folds over the outer fold's TRAINING rows only.

    The training blocks, chronologically, are cut into four contiguous
    segments of near-equal row count, giving three inner folds:

        train Q1        validate Q2
        train Q1+Q2     validate Q3
        train Q1+Q2+Q3  validate Q4

    Expanding, because that is the shape of the outer evaluation. A rolling or
    random inner scheme would select lambda for a regime the model is never
    used in. Identical for all four outer folds, including fold 1, which trains
    on a single season and admits no season-level inner scheme at all.
    """

    train_set = set(int(r) for r in train_rows)

    usable = [b for b in blocks
              if set(int(r) for r in b["rows"]).issubset(train_set)]

    partial = [b for b in blocks
               if not set(int(r) for r in b["rows"]).isdisjoint(train_set)
               and not set(int(r) for r in b["rows"]).issubset(train_set)]

    if partial:
        raise LeakageError(
            "a calendar date straddles the outer training boundary: {}".format(
                [str(b["date"].date()) for b in partial]))

    total = sum(len(b["rows"]) for b in usable)
    target = total / N_INNER_SEGMENTS

    segments = [[] for _ in range(N_INNER_SEGMENTS)]
    running = 0

    for block in usable:
        # Place the block in the segment its MIDPOINT falls into, so a block
        # is never divided and the segments stay contiguous in time.
        midpoint = running + len(block["rows"]) / 2.0
        index = min(int(midpoint // target), N_INNER_SEGMENTS - 1)
        segments[index].append(block)
        running += len(block["rows"])

    splits = []

    for cut in range(1, N_INNER_SEGMENTS):

        train_blocks = [b for segment in segments[:cut] for b in segment]
        valid_blocks = segments[cut]

        if not train_blocks or not valid_blocks:
            raise RuntimeError("inner split {} came out empty".format(cut))

        splits.append({
            "inner_fold": cut,
            "train_rows": np.sort(np.concatenate([b["rows"] for b in train_blocks])),
            "valid_rows": np.sort(np.concatenate([b["rows"] for b in valid_blocks])),
            "train_max_date": max(b["max_date"] for b in train_blocks),
            "valid_min_date": min(b["min_date"] for b in valid_blocks),
        })

    return splits


# ============================================================
# THE FITTED PIPELINE  (index-based; arithmetic identical to Instrument 3)
# ============================================================

def fit_pipeline(matrix, labels, train_rows, eval_rows, penalty):
    """
    Impute, standardise and fit on `train_rows`; transform and predict
    `eval_rows` with the training statistics.

    This is Instrument 3's fit_fold() taking index arrays instead of boolean
    masks, so that the same code can serve an INNER training set as well as an
    outer one. Nothing else about it differs - R7 proves that by reproducing
    Instrument 3's frozen numbers to < 1e-9 at lambda = 1.
    """

    train = matrix[train_rows]
    evaluate_on = matrix[eval_rows]

    if matrix.shape[1] == 0:
        weights, iterations, gradient = L3.fit_multinomial(
            np.zeros((len(train), 0)), labels[train_rows], penalty=penalty)
        return {
            "proba": L3.predict_multinomial(weights, np.zeros((len(evaluate_on), 0))),
            "train_proba": L3.predict_multinomial(weights, np.zeros((len(train), 0))),
            "weights": weights,
            "n_columns": 0,
            "iterations": iterations,
            "gradient": gradient,
            "imputed_cells": 0,
            "constant_columns": 0,
        }

    with np.errstate(invalid="ignore"):
        centre = np.nanmean(train, axis=0)

    centre = np.where(np.isfinite(centre), centre, 0.0)

    imputed_cells = int(np.isnan(train).sum() + np.isnan(evaluate_on).sum())

    train = np.where(np.isnan(train), centre, train)
    evaluate_on = np.where(np.isnan(evaluate_on), centre, evaluate_on)

    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)

    constant_columns = int((sd == 0).sum())
    sd = np.where(sd == 0, 1.0, sd)

    train = (train - mean) / sd
    evaluate_on = (evaluate_on - mean) / sd

    weights, iterations, gradient = L3.fit_multinomial(
        train, labels[train_rows], penalty=penalty)

    return {
        "proba": L3.predict_multinomial(weights, evaluate_on),
        "train_proba": L3.predict_multinomial(weights, train),
        "weights": weights,
        "n_columns": matrix.shape[1],
        "iterations": iterations,
        "gradient": gradient,
        "imputed_cells": imputed_cells,
        "constant_columns": constant_columns,
    }


# ============================================================
# COEFFICIENT NORMS
# ============================================================

def feature_of(design_name):
    """Design column -> source feature. 'day_of_week=Sat' -> 'day_of_week'."""

    return design_name.split("=", 1)[0]


def coefficient_norms(weights, names):
    """
    Total L2 norm of the fitted coefficients, and the same split by block.

    The intercept is excluded: it is unpenalised, it is not a feature, and
    including it would let the base rate dominate a norm that is supposed to
    describe how hard the fit is leaning on the columns.

    Columns are standardised, so a norm is comparable across blocks and
    across lambda without further scaling.
    """

    if not names:
        return 0.0, {}

    body = weights[:len(names), :]

    total = float(np.sqrt((body ** 2).sum()))

    per_block = {}

    for position, name in enumerate(names):
        block = block_of(feature_of(name))
        per_block.setdefault(block, {"sq": 0.0, "columns": 0})
        per_block[block]["sq"] += float((body[position] ** 2).sum())
        per_block[block]["columns"] += 1

    for block in per_block:
        per_block[block]["norm"] = float(np.sqrt(per_block[block]["sq"]))

    return total, per_block


# ============================================================
# ONE RUN
# ============================================================

class Run:
    """
    One complete pass: select lambda per (rung, fold), refit, score.

    Instantiated fresh for the real run and for each perturbation test, so a
    corrupted feature frame can never reuse a cached fit from the clean one.
    """

    def __init__(self, features, matches, labels, spec, ladder, blocks,
                 compute_surface=True, folds=None, quiet=False, grid=None):

        self.features = features
        self.matches = matches
        self.labels = labels
        self.spec = spec
        self.ladder = ladder
        self.blocks = blocks
        self.compute_surface = compute_surface
        self.quiet = quiet

        # Instrument 5 supplies its own grid; omitted, this is the grid
        # declared in this instrument's own pre-declaration and asserted by
        # R6, so Instrument 4 behaves exactly as it did before the argument
        # existed. Instrument 5's C1 proves that claim rather than asserting it.
        self.grid = tuple(grid) if grid is not None else LAMBDA_GRID

        self.folds = [f for f in spec["folds"]
                      if folds is None or int(f["fold"]) in folds]

        self.curves = []          # inner-CV, per rung x fold x inner x lambda
        self.selected = []        # one row per rung x fold
        self.surface = []         # ORACLE, per rung x fold x lambda
        self.block_norms = []
        self.predictions = []
        self.column_sets = {}

        # R4's tripwire. Set per outer fold, consulted inside the selector.
        self._forbidden = None
        self.tripwire_checks = 0

    # ---- the tripwire -------------------------------------------------

    def guard(self, rows, where):

        self.tripwire_checks += 1

        if self._forbidden is None:
            raise LeakageError("selector ran with no forbidden set armed")

        offending = np.intersect1d(np.asarray(rows), self._forbidden)

        if offending.size:
            raise LeakageError(
                "{}: {} outer-TEST row(s) reached the selector, first {}".format(
                    where, offending.size, offending[:5].tolist()))

    # ---- the selector -------------------------------------------------

    def select_lambda(self, rung, matrix, train_rows, fold):
        """
        Inner CV over the outer fold's training rows. Training data only.

        Every row index this touches passes the R4 tripwire first. The outer
        test season is not an argument to this function and is not reachable
        from it.
        """

        self.guard(train_rows, "outer training set handed to selector")

        splits = inner_splits(train_rows, self.blocks)

        curve = {}

        for penalty in self.grid:

            losses = []

            for split in splits:

                self.guard(split["train_rows"], "inner train")
                self.guard(split["valid_rows"], "inner validation")

                fitted = fit_pipeline(
                    matrix, self.labels, split["train_rows"],
                    split["valid_rows"], penalty)

                actual = self.matches["result"].to_numpy()[split["valid_rows"]]
                scores = evaluate(actual, fitted["proba"])

                losses.append(scores[PRIMARY_METRIC])

                self.curves.append({
                    "rung": rung,
                    "fold": fold,
                    "inner_fold": split["inner_fold"],
                    "lambda": penalty,
                    "inner_train_matches": int(len(split["train_rows"])),
                    "inner_valid_matches": int(len(split["valid_rows"])),
                    "inner_train_max_date": split["train_max_date"].strftime("%Y-%m-%d"),
                    "inner_valid_min_date": split["valid_min_date"].strftime("%Y-%m-%d"),
                    "inner_val_log_loss": scores["log_loss"],
                    "inner_val_rps": scores["rps"],
                    "inner_val_accuracy": scores["accuracy"],
                    "constant_columns": fitted["constant_columns"],
                })

            curve[penalty] = float(np.mean(losses))

        best = min(curve.values())

        # Declared tie-break: exact ties go to the LARGER lambda.
        chosen = max(p for p, value in curve.items() if value == best)

        return chosen, curve, splits

    # ---- one rung -----------------------------------------------------

    def run_rung(self, rung, columns):

        numeric, categorical = L3.design_columns(columns, self.features)
        matrix, names = L3.build_matrix(self.features, numeric, categorical)

        for fold_spec in self.folds:

            fold = int(fold_spec["fold"])
            test_season = str(fold_spec["test_season"])

            train_rows = np.flatnonzero(
                self.matches["season"].isin(fold_spec["train_seasons"]).to_numpy())
            test_rows = np.flatnonzero(
                (self.matches["season"] == test_season).to_numpy())

            self._forbidden = test_rows

            chosen, curve, splits = self.select_lambda(rung, matrix, train_rows, fold)

            self._forbidden = None

            actual = self.matches["result"].to_numpy()[test_rows]
            train_actual = self.matches["result"].to_numpy()[train_rows]

            grid = self.grid if self.compute_surface else (chosen,)

            for penalty in grid:

                fitted = fit_pipeline(
                    matrix, self.labels, train_rows, test_rows, penalty)

                proba = validate_probabilities(fitted["proba"], len(test_rows))
                scores = evaluate(actual, proba)
                train_scores = evaluate(train_actual, fitted["train_proba"])

                total_norm, per_block = coefficient_norms(fitted["weights"], names)

                row = {
                    "rung": rung,
                    "fold": fold,
                    "test_season": test_season,
                    "lambda": penalty,
                    "is_selected": bool(penalty == chosen),
                    "is_frozen_lambda": bool(penalty == FROZEN_LAMBDA),
                    "design_columns": fitted["n_columns"],
                    "train_matches": int(len(train_rows)),
                    "test_matches": int(len(test_rows)),
                    "newton_iterations": fitted["iterations"],
                    "gradient_norm": fitted["gradient"],
                    "constant_columns": fitted["constant_columns"],
                    "coef_l2_norm": total_norm,
                }
                row.update({metric: scores[metric] for metric in METRICS})
                row["train_log_loss"] = train_scores["log_loss"]
                row["generalisation_gap"] = scores["log_loss"] - train_scores["log_loss"]

                self.surface.append(row)

                for block, values in per_block.items():
                    self.block_norms.append({
                        "rung": rung, "fold": fold, "lambda": penalty,
                        "is_selected": bool(penalty == chosen),
                        "block": block,
                        "design_columns": values["columns"],
                        "block_l2_norm": values["norm"],
                        "block_l2_norm_per_column": (
                            values["norm"] / np.sqrt(values["columns"])),
                    })

                if penalty == chosen:

                    predicted = np.argmax(proba, axis=1)

                    self.predictions.append(pd.DataFrame({
                        "rung": rung, "fold": fold, "test_season": test_season,
                        "date": self.matches["date"].dt.strftime("%Y-%m-%d")
                                    .to_numpy()[test_rows],
                        "home": self.matches["home_team"].to_numpy()[test_rows],
                        "away": self.matches["away_team"].to_numpy()[test_rows],
                        "actual_result": actual,
                        "p_home": proba[:, 0],
                        "p_draw": proba[:, 1],
                        "p_away": proba[:, 2],
                        "predicted_result": [CLASSES[i] for i in predicted],
                    }))

                    selected_row = {
                        "rung": rung,
                        "fold": fold,
                        "test_season": test_season,
                        "selected_lambda": chosen,
                        "at_grid_floor": bool(chosen == self.grid[0]),
                        "at_grid_ceiling": bool(chosen == self.grid[-1]),
                        "inner_best_log_loss": curve[chosen],
                        "inner_log_loss_at_frozen": curve.get(FROZEN_LAMBDA, float("nan")),
                        "inner_folds": len(splits),
                        "design_columns": fitted["n_columns"],
                        "coef_l2_norm": total_norm,
                    }
                    selected_row.update(
                        {"test_" + m: scores[m] for m in METRICS})
                    selected_row["train_log_loss"] = train_scores["log_loss"]
                    selected_row["generalisation_gap"] = (
                        scores["log_loss"] - train_scores["log_loss"])

                    for penalty_value in self.grid:
                        selected_row["inner_ll_lam_{:g}".format(penalty_value)] = \
                            curve[penalty_value]

                    self.selected.append(selected_row)

            if not self.quiet:
                # The lambda = 1 column only means something when 1 is on the
                # grid. Instrument 5's grid starts at 100, so it reports the
                # inner score at its own floor instead of crashing on a key
                # that this run never evaluated.
                reference = (FROZEN_LAMBDA if FROZEN_LAMBDA in curve
                             else self.grid[0])
                print("    {:<3} fold {}  lambda* = {:<7g} test LL {:.4f}"
                      "   (inner LL at lambda={:g}: {:.4f})".format(
                          rung, fold, chosen,
                          [r for r in self.selected
                           if r["rung"] == rung and r["fold"] == fold
                           ][0]["test_log_loss"],
                          reference, curve[reference]))

    def execute(self):

        cumulative = []

        for rung, _description, added in self.ladder:
            cumulative = cumulative + list(added)
            self.column_sets[rung] = list(cumulative)
            self.run_rung(rung, cumulative)

        return self

    # ---- frames -------------------------------------------------------

    def selected_frame(self):
        return pd.DataFrame(self.selected).sort_values(
            ["rung", "fold"]).reset_index(drop=True)

    def surface_frame(self):
        return pd.DataFrame(self.surface).sort_values(
            ["rung", "fold", "lambda"]).reset_index(drop=True)

    def curves_frame(self):
        return pd.DataFrame(self.curves)

    def block_norm_frame(self):
        return pd.DataFrame(self.block_norms)

    def prediction_frame(self):
        return pd.concat(self.predictions, ignore_index=True)

    def lambda_vector(self):
        """(rung, fold) -> selected lambda. The object R2/R3 compare."""
        return {(r["rung"], r["fold"]): r["selected_lambda"] for r in self.selected}


# ============================================================
# THE HEADLINE TABLE
# ============================================================

def build_ladder_table(selected, frozen_ladder):
    """
    The ladder at SELECTED lambda, with the frozen lambda = 1 column beside it.

    The frozen column is read from Instrument 3's output, never recomputed
    into this table - the reference has to be the artefact on disk or it is
    not a reference.
    """

    rows = []
    previous = None

    for rung in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:

        block = selected[selected["rung"] == rung]

        if not len(block):
            continue

        frozen = frozen_ladder[frozen_ladder["rung"] == rung]

        row = {
            "rung": rung,
            "description": frozen["description"].iloc[0] if len(frozen) else "",
            "features_total": int(frozen["features_total"].iloc[0]) if len(frozen) else -1,
            "design_columns": int(block["design_columns"].max()),
            "lambda_selected_min": float(block["selected_lambda"].min()),
            "lambda_selected_max": float(block["selected_lambda"].max()),
            "lambda_selected_median": float(block["selected_lambda"].median()),
        }

        for fold in (1, 2, 3, 4):
            part = block[block["fold"] == fold]
            if len(part):
                row["lambda_fold{}".format(fold)] = float(
                    part["selected_lambda"].iloc[0])
                row["log_loss_fold{}".format(fold)] = float(
                    part["test_log_loss"].iloc[0])

        for metric in METRICS:
            row["{}_mean".format(metric)] = float(block["test_" + metric].mean())

        row["train_log_loss_mean"] = float(block["train_log_loss"].mean())
        row["generalisation_gap_mean"] = float(block["generalisation_gap"].mean())
        row["coef_l2_norm_mean"] = float(block["coef_l2_norm"].mean())

        row["frozen_log_loss_mean"] = (
            float(frozen["log_loss_mean"].iloc[0]) if len(frozen) else np.nan)
        row["frozen_generalisation_gap_mean"] = (
            float(frozen["generalisation_gap_mean"].iloc[0]) if len(frozen) else np.nan)
        row["log_loss_mean_vs_frozen"] = (
            row["log_loss_mean"] - row["frozen_log_loss_mean"])

        if previous is not None:
            row["log_loss_delta_mean"] = row["log_loss_mean"] - previous["log_loss_mean"]
        else:
            row["log_loss_delta_mean"] = np.nan

        rows.append(row)
        previous = row

    return pd.DataFrame(rows)


def build_verdict(selected, table, frozen_ladder):
    """
    The pre-declared decision rule, applied.

    D(rung)  = mean over folds of [LL(rung) - LL(B1)], paired fold by fold
    SE(rung) = sd(D over folds, ddof=1) / sqrt(4)

    OVERTURNED  argmin lies in B2..B6 and D + SE < 0
    WEAKENED    argmin lies in B2..B6 and D + SE >= 0
    UPHELD      argmin is B0 or B1
    """

    pivot = selected.pivot_table(
        index="fold", columns="rung", values="test_log_loss")

    reference = pivot["B1"]

    rows = []

    for rung in pivot.columns:

        paired = (pivot[rung] - reference).to_numpy()

        mean_difference = float(np.mean(paired))
        standard_error = float(np.std(paired, ddof=1) / np.sqrt(len(paired)))

        rows.append({
            "rung": rung,
            "mean_log_loss": float(pivot[rung].mean()),
            "D_vs_B1": mean_difference,
            "SE_of_D": standard_error,
            "D_plus_SE": mean_difference + standard_error,
            "beats_B1_by_1SE": bool(mean_difference + standard_error < 0),
            "folds": int(len(paired)),
        })

    frame = pd.DataFrame(rows).sort_values("rung").reset_index(drop=True)

    winner = frame.loc[frame["mean_log_loss"].idxmin(), "rung"]

    if winner in ("B0", "B1"):
        verdict = "UPHELD"
    elif bool(frame.loc[frame["rung"] == winner, "beats_B1_by_1SE"].iloc[0]):
        verdict = "OVERTURNED"
    else:
        verdict = "WEAKENED"

    # The descriptive REFRAMED annotation, on top of the verdict.
    median_lambda = float(selected["selected_lambda"].median())

    gap_now = (float(table.loc[table["rung"] == "B6", "log_loss_mean"].iloc[0])
               - float(table.loc[table["rung"] == "B1", "log_loss_mean"].iloc[0]))
    gap_frozen = (float(frozen_ladder.loc[frozen_ladder["rung"] == "B6",
                                          "log_loss_mean"].iloc[0])
                  - float(frozen_ladder.loc[frozen_ladder["rung"] == "B1",
                                            "log_loss_mean"].iloc[0]))

    reframed = bool(
        verdict == "UPHELD"
        and median_lambda >= REFRAMED_LAMBDA_FLOOR
        and gap_now < REFRAMED_GAP_SHRINK * gap_frozen)

    frame.attrs["verdict"] = verdict
    frame.attrs["winner"] = winner
    frame.attrs["reframed"] = reframed
    frame.attrs["median_lambda"] = median_lambda
    frame.attrs["gap_now"] = gap_now
    frame.attrs["gap_frozen"] = gap_frozen

    return frame


# ============================================================
# TESTS
# ============================================================

def declared_grid(predeclaration=None):
    """The grid as written in the pre-declaration governing this run."""

    predeclaration = predeclaration or PREDECLARATION

    if not predeclaration.exists():
        return None

    for line in predeclaration.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("LAMBDA_GRID"):
            values = stripped.split("=", 1)[1]
            return tuple(float(v.strip()) for v in values.split(","))

    return None


def test_everything(run, features, matches, labels, spec, ladder, blocks,
                    selected, table, before_state, audit,
                    grid=None, predeclaration=None):

    # Instrument 5 re-runs this whole battery against its own grid and its own
    # pre-declaration. Omitted, both default to Instrument 4's, so the call
    # made by this module's main() is unchanged.
    grid = tuple(grid) if grid is not None else LAMBDA_GRID
    predeclaration = predeclaration or PREDECLARATION

    # ---- R6  the grid is the declared grid -------------------------------
    declared = declared_grid(predeclaration)

    audit.record(
        "R6a", "the pre-declaration file exists and is readable",
        "present", "present" if declared is not None else "MISSING",
        declared is not None,
        str(predeclaration))

    audit.record(
        "R6b", "the grid used is the grid declared before the run",
        str(list(grid)), str(list(declared)) if declared else "n/a",
        declared is not None and tuple(declared) == grid,
        "a grid chosen after seeing results would make this a search, not a test")

    audit.measure(
        "R6c", "SHA-256 of the pre-declaration",
        hash_file(predeclaration)[:16] + "...",
        "recorded so a later edit cannot retroactively license this result")

    # ---- R1  no inner row is ever an outer-test row ----------------------
    violations = 0
    inner_row_count = 0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train_rows = np.flatnonzero(
            matches["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero(
            (matches["season"] == str(fold_spec["test_season"])).to_numpy())

        for split in inner_splits(train_rows, blocks):
            used = np.concatenate([split["train_rows"], split["valid_rows"]])
            inner_row_count += len(used)
            violations += int(np.intersect1d(used, test_rows).size)

    audit.record(
        "R1", "no inner-CV row is an outer-test row, any fold",
        0, violations, violations == 0,
        "{} inner row-uses checked across {} folds".format(
            inner_row_count, len(spec["folds"])))

    # ---- R4  the tripwire actually fired, and actually works -------------
    audit.measure(
        "R4a", "tripwire guard invocations during the real run",
        run.tripwire_checks,
        "every row set entering the selector was checked against the outer "
        "test rows before use")

    # A guard that never raises is not a guard. Prove it raises.
    prover = Run(features, matches, labels, spec, ladder, blocks,
                 compute_surface=False, quiet=True, grid=grid)
    prover._forbidden = np.array([0, 1, 2])

    try:
        prover.guard(np.array([2, 5, 9]), "deliberate")
        tripwire_fires = False
    except LeakageError:
        tripwire_fires = True

    audit.record(
        "R4b", "CONTROL: the tripwire raises when handed a forbidden row",
        "raises", "raised" if tripwire_fires else "SILENT", tripwire_fires,
        "without this, R4a would pass for a guard that could never fail")

    # ---- R5  every inner split is time-respecting ------------------------
    # Amended: the atomic unit is the calendar date, not (season, matchweek).
    # See Amendment 1 in the pre-declaration - a matchweek can span 200 days.
    out_of_order = 0
    dates_straddling = 0
    matchweeks_divided = set()
    matches_in_divided = 0
    checked = 0

    date_of = {}
    matchweek_of = {}

    for block in blocks:
        for row in block["rows"]:
            date_of[int(row)] = block["date"]

    for position, row in enumerate(matches.itertuples(index=False)):
        matchweek_of[position] = (row.season, int(row.matchweek))

    for fold_spec in spec["folds"]:

        train_rows = np.flatnonzero(
            matches["season"].isin(fold_spec["train_seasons"]).to_numpy())

        for split in inner_splits(train_rows, blocks):

            checked += 1

            if split["train_max_date"] >= split["valid_min_date"]:
                out_of_order += 1

            train_dates = {date_of[int(r)] for r in split["train_rows"]}
            valid_dates = {date_of[int(r)] for r in split["valid_rows"]}

            dates_straddling += len(train_dates & valid_dates)

            train_weeks = {matchweek_of[int(r)] for r in split["train_rows"]}
            valid_weeks = {matchweek_of[int(r)] for r in split["valid_rows"]}

            for week in train_weeks & valid_weeks:
                matchweeks_divided.add((int(fold_spec["fold"]),
                                        split["inner_fold"], week))
                matches_in_divided += sum(
                    1 for r in split["valid_rows"] if matchweek_of[int(r)] == week)

    audit.record(
        "R5a", "every inner split has max(train date) < min(validation date)",
        0, out_of_order, out_of_order == 0,
        "{} inner splits checked; strict, not >=".format(checked))

    audit.record(
        "R5b", "no calendar date straddles an inner boundary",
        0, dates_straddling, dates_straddling == 0,
        "the date is the atomic unit; two matches played the same afternoon "
        "may never sit on opposite sides of a split")

    audit.measure(
        "R5c", "(season, matchweek) blocks divided by an inner boundary",
        "{} across {} splits, {} matches".format(
            len(matchweeks_divided), checked, matches_in_divided),
        "measured, not forbidden: a matchweek is divided here only because "
        "postponement put its halves months apart, so they were never "
        "contemporaneous - which is what the boundary is protecting")

    # ---- R7  lambda = 1 reproduces Instrument 3 exactly -------------------
    # Applies only when lambda = 1 is a point of the grid being run.
    # Instrument 5's grid starts at 100, and replaces this with its own
    # anchor test C1 against Instrument 4's stored lambda = 100 column.
    surface = run.surface_frame()

    if FROZEN_LAMBDA not in grid:

        audit.measure(
            "R7", "reproduce Instrument 3 at lambda = 1",
            "not applicable",
            "lambda = 1 is not in this grid ({:g} .. {:g}); the anchoring "
            "test for this run is declared separately".format(
                grid[0], grid[-1]))

        return _finish_battery(run, features, matches, labels, spec, ladder,
                               blocks, selected, surface, before_state, audit,
                               grid)

    frozen = pd.read_csv(FROZEN_FOLD_SUMMARY, float_precision=FLOAT_PRECISION)

    mine = surface[surface["lambda"] == FROZEN_LAMBDA]

    merged = frozen.merge(
        mine, on=["rung", "fold"], suffixes=("_frozen", "_mine"), how="inner")

    worst = 0.0
    worst_metric = ""

    for metric in METRICS:
        difference = float(np.abs(
            merged[metric + "_frozen"].to_numpy()
            - merged[metric + "_mine"].to_numpy()).max())
        if difference > worst:
            worst, worst_metric = difference, metric

    audit.record(
        "R7a", "at lambda = 1 this reproduces Instrument 3's frozen numbers",
        "< 1e-9", "{:.3e} ({})".format(worst, worst_metric), worst < 1e-9,
        "{} rung x fold pairs, all six metrics; if this drifts, the "
        "re-indexed pipeline is not the pipeline it claims to be".format(len(merged)))

    audit.record(
        "R7b", "every frozen rung x fold pair was matched and compared",
        len(frozen), len(merged), len(merged) == len(frozen))

    train_difference = float(np.abs(
        merged["train_log_loss_frozen"].to_numpy()
        - merged["train_log_loss_mine"].to_numpy()).max())

    audit.record(
        "R7c", "training-set log loss also reproduces at lambda = 1",
        "< 1e-9", "{:.3e}".format(train_difference), train_difference < 1e-9)

    return _finish_battery(run, features, matches, labels, spec, ladder,
                           blocks, selected, surface, before_state, audit, grid)


def _finish_battery(run, features, matches, labels, spec, ladder, blocks,
                    selected, surface, before_state, audit, grid):
    """Everything in the battery that does not depend on lambda = 1."""

    # ---- R10  B0 is invariant across the grid ---------------------------
    b0 = surface[surface["rung"] == "B0"]

    spread = float(b0.groupby("fold")["log_loss"].apply(
        lambda s: s.max() - s.min()).max())

    audit.record(
        "R10", "B0's fit is identical at every lambda",
        0.0, "{:.3e}".format(spread), spread == 0.0,
        "no features, unpenalised intercept - the penalty has nothing to act on")

    # ---- R2  corrupting outer-TEST FEATURES moves no selected lambda -----
    # R2a is Instrument 3's A6 shape: 2025-26 is test-only, so it appears in
    # no fold's training set and NOTHING may move except fold 4's own output.
    corrupted = features.copy()

    numeric_columns = [
        c for c in run.column_sets["B6"]
        if c not in L3.HELD_OUT_AS_METADATA
        and c not in L3.CATEGORICAL_LEVELS
        and pd.api.types.is_numeric_dtype(corrupted[c])
        and not pd.api.types.is_bool_dtype(corrupted[c])
    ]

    test_only = (corrupted["season"] == "2025-2026").to_numpy()

    for column in numeric_columns:
        corrupted.loc[test_only, column] = 999.0

    print("    R2a: refitting the whole selection on corrupted 2025-26 features...")

    corrupt_run = Run(corrupted, matches, labels, spec, ladder, blocks,
                      compute_surface=False, quiet=True, grid=grid).execute()

    lambda_moves = sum(
        1 for key, value in run.lambda_vector().items()
        if corrupt_run.lambda_vector().get(key) != value)

    audit.record(
        "R2a", "corrupting 2025-26 features moves no selected lambda anywhere",
        0, lambda_moves, lambda_moves == 0,
        "2025-26 is test-only; it is in no fold's training set, so no "
        "selector may have seen it")

    original_predictions = run.prediction_frame()
    corrupt_predictions = corrupt_run.prediction_frame()

    keys = ["rung", "fold", "date", "home", "away"]
    joined = original_predictions.merge(
        corrupt_predictions, on=keys, suffixes=("_a", "_b"), how="inner")

    early = joined[joined["fold"] != 4]

    moved = float(np.abs(
        early[["p_home_a", "p_draw_a", "p_away_a"]].to_numpy()
        - early[["p_home_b", "p_draw_b", "p_away_b"]].to_numpy()).max())

    audit.record(
        "R2b", "and leaves every fold-1-to-3 prediction bit-identical",
        0.0, "{:.3e}".format(moved), moved == 0.0,
        "{} predictions compared".format(len(early)))

    late = joined[joined["fold"] == 4]

    fold4_moved = float(np.abs(
        late[["p_home_a", "p_draw_a", "p_away_a"]].to_numpy()
        - late[["p_home_b", "p_draw_b", "p_away_b"]].to_numpy()).max())

    audit.record(
        "R2c", "CONTROL: fold 4's own predictions DO move when its features do",
        "> 0", "{:.3e}".format(fold4_moved), fold4_moved > 0,
        "without this, R2a and R2b would pass for a pipeline ignoring features")

    # R2d is the stronger, per-fold version. For outer fold k, corrupt ONLY
    # season(test_k) and require fold k's own lambdas to be unmoved. Folds
    # other than k legitimately move for k < 4, because their training sets
    # contain that season - so only fold k is asserted on.
    per_fold_moves = 0
    per_fold_checked = 0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = str(fold_spec["test_season"])

        print("    R2d: fold {} - corrupting {} features and reselecting..."
              .format(fold, test_season))

        local = features.copy()
        rows = (local["season"] == test_season).to_numpy()

        for column in numeric_columns:
            local.loc[rows, column] = 999.0

        local_run = Run(local, matches, labels, spec, ladder, blocks,
                        compute_surface=False, folds=[fold], quiet=True,
                        grid=grid).execute()

        for (rung, this_fold), value in local_run.lambda_vector().items():
            per_fold_checked += 1
            if run.lambda_vector()[(rung, this_fold)] != value:
                per_fold_moves += 1

    audit.record(
        "R2d", "per fold: corrupting THAT fold's test features moves no lambda",
        0, per_fold_moves, per_fold_moves == 0,
        "{} (rung, fold) selections re-run, one corrupted season each; this "
        "covers folds 1-3, whose test seasons R2a cannot touch".format(
            per_fold_checked))

    # ---- R3  corrupting outer-TEST LABELS moves no selected lambda -------
    per_fold_label_moves = 0
    per_fold_label_checked = 0

    rng = np.random.default_rng(L3.SHUFFLE_SEED)

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = str(fold_spec["test_season"])

        print("    R3: fold {} - shuffling {} labels and reselecting..."
              .format(fold, test_season))

        rows = np.flatnonzero((matches["season"] == test_season).to_numpy())

        scrambled = labels.copy()
        scrambled[rows] = rng.permutation(scrambled[rows])

        scrambled_matches = matches.copy()
        scrambled_matches.loc[rows, "result"] = [
            CLASSES[i] for i in scrambled[rows]]

        local_run = Run(features, scrambled_matches, scrambled, spec, ladder,
                        blocks, compute_surface=False, folds=[fold],
                        quiet=True, grid=grid).execute()

        for (rung, this_fold), value in local_run.lambda_vector().items():
            per_fold_label_checked += 1
            if run.lambda_vector()[(rung, this_fold)] != value:
                per_fold_label_moves += 1

    audit.record(
        "R3", "per fold: shuffling THAT fold's test LABELS moves no lambda",
        0, per_fold_label_moves, per_fold_label_moves == 0,
        "{} selections re-run; the outer test outcome cannot reach the "
        "selector even through the label column".format(per_fold_label_checked))

    # ---- R9  reproducibility --------------------------------------------
    print("    R9: repeating the entire selection from scratch...")

    repeat = Run(features, matches, labels, spec, ladder, blocks,
                 compute_surface=False, quiet=True, grid=grid).execute()

    lambda_differences = sum(
        1 for key, value in run.lambda_vector().items()
        if repeat.lambda_vector().get(key) != value)

    audit.record(
        "R9a", "a second independent run selects the identical lambdas",
        0, lambda_differences, lambda_differences == 0,
        "{} (rung, fold) selections".format(len(run.lambda_vector())))

    repeat_predictions = repeat.prediction_frame()

    rejoined = original_predictions.merge(
        repeat_predictions, on=keys, suffixes=("_a", "_b"), how="inner")

    repeat_moved = float(np.abs(
        rejoined[["p_home_a", "p_draw_a", "p_away_a"]].to_numpy()
        - rejoined[["p_home_b", "p_draw_b", "p_away_b"]].to_numpy()).max())

    audit.record(
        "R9b", "and produces bit-identical probabilities",
        0.0, "{:.3e}".format(repeat_moved), repeat_moved == 0.0,
        "{} predictions; no seed, no shuffle, no randomness in the fit or "
        "the split".format(len(rejoined)))

    audit.record(
        "R9c", "every prediction row sums to 1 and none is negative",
        "< 1e-9",
        "{:.3e}".format(float(np.abs(
            original_predictions[["p_home", "p_draw", "p_away"]]
            .to_numpy().sum(axis=1) - 1.0).max())),
        float(np.abs(original_predictions[["p_home", "p_draw", "p_away"]]
                     .to_numpy().sum(axis=1) - 1.0).max()) < 1e-9
        and int((original_predictions[["p_home", "p_draw", "p_away"]]
                 .to_numpy() < 0).sum()) == 0)

    # ---- R11  convergence and grid boundaries ----------------------------
    worst_gradient = float(surface["gradient_norm"].max())

    audit.record(
        "R11a", "every fit converged to the declared gradient tolerance",
        "< {:g}".format(L3.GRADIENT_TOLERANCE),
        "{:.3e}".format(worst_gradient),
        worst_gradient < L3.GRADIENT_TOLERANCE,
        "Newton on a strictly convex objective at every lambda in the grid")

    floor_hits = int(selected["at_grid_floor"].sum())
    ceiling_hits = int(selected["at_grid_ceiling"].sum())

    audit.measure(
        "R11b", "selections landing on the grid FLOOR (lambda = {:g})".format(
            grid[0]),
        "{} of {}".format(floor_hits, len(selected)),
        "the grid is not extended; a floor hit means the inner CV wanted even "
        "less penalty than the grid offers")

    audit.measure(
        "R11c", "selections landing on the grid CEILING (lambda = {:g})".format(
            grid[-1]),
        "{} of {}".format(ceiling_hits, len(selected)),
        "a ceiling hit means the inner CV wanted even MORE penalty than the "
        "grid offers, which would itself be the finding")

    # ---- R8  isolation ---------------------------------------------------
    after_state = frozen_state()

    changed = [path for path in before_state
               if before_state[path] != after_state.get(path)]

    audit.record(
        "R8a", "no frozen artefact, script, pre-declaration or raw file moved",
        0, len(changed), not changed, str(changed[:3]))

    audit.measure(
        "R8b", "files SHA-256'd before and after the run", len(before_state),
        "includes the lambda = 1 reference this experiment is measured against")

    raw_events = [
        (path, mode, flags, hashing) for path, mode, flags, hashing in L3._OPEN_EVENTS
        if str(Path(path).resolve()).startswith(str(RAW_DIR))
    ]

    by_model = [event for event in raw_events if not event[3]]

    audit.record(
        "R8c", "nothing but the integrity hasher opened data/raw",
        0, len(by_model), not by_model,
        "matches and features come from Phase 1 and Phase 3 outputs, never "
        "from raw")

    raw_writes = [event for event in raw_events
                  if L3.is_write_open(event[1], event[2])]

    audit.record(
        "R8d", "no open under data/raw asked for write access",
        0, len(raw_writes), not raw_writes, str(raw_writes[:2]))

    return audit


# ============================================================
# REPORT
# ============================================================

def print_selected_lambdas(selected):

    print("  Selected lambda, per rung x fold. Chosen by inner CV on training")
    print("  rows only; the outer test season never entered the choice.")
    print()
    print("  {:<5} {:>10} {:>10} {:>10} {:>10} {:>12}".format(
        "rung", "fold 1", "fold 2", "fold 3", "fold 4", "median"))
    print("  " + "-" * 62)

    for rung in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:

        block = selected[selected["rung"] == rung]

        if not len(block):
            continue

        values = []

        for fold in (1, 2, 3, 4):
            part = block[block["fold"] == fold]
            values.append(float(part["selected_lambda"].iloc[0]) if len(part)
                          else np.nan)

        print("  {:<5} {:>10g} {:>10g} {:>10g} {:>10g} {:>12g}".format(
            rung, *values, float(np.median(values))))

    print()


def print_ladder(table):

    print("  {:<5} {:<32} {:>6} {:>9} {:>9} {:>9} {:>9}".format(
        "rung", "block added", "lam*", "test LL", "d(mean)", "frozen LL", "change"))
    print("  " + "-" * 88)

    for _index, row in table.iterrows():

        print("  {:<5} {:<32} {:>6g} {:>9.4f} {:>9} {:>9.4f} {:>+9.4f}".format(
            row["rung"], str(row["description"])[:32],
            row["lambda_selected_median"],
            row["log_loss_mean"],
            "-" if pd.isna(row["log_loss_delta_mean"])
            else "{:+.4f}".format(row["log_loss_delta_mean"]),
            row["frozen_log_loss_mean"],
            row["log_loss_mean_vs_frozen"]))

    print()


def print_surface(surface, metric="log_loss", grid=None):

    grid = tuple(grid) if grid is not None else LAMBDA_GRID

    print("  ORACLE SURFACE - outer-test {} at every lambda.".format(metric))
    print("  Mean over the four folds. NOT used to select anything; printed")
    print("  because 'does any lambda rescue B2-B6' deserves a direct answer.")
    print()

    header = "  {:<5}".format("rung") + "".join(
        "{:>9g}".format(p) for p in grid)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rung in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:

        block = surface[surface["rung"] == rung]

        if not len(block):
            continue

        means = [float(block[block["lambda"] == p][metric].mean())
                 for p in grid]

        best = int(np.argmin(means))

        cells = "".join(
            ("{:>8.4f}*".format(v) if i == best else "{:>9.4f}".format(v))
            for i, v in enumerate(means))

        print("  {:<5}{}".format(rung, cells))

    print()
    print("  * = best lambda for that rung on the OUTER TEST. Oracle only.")
    print()


def print_generalisation(surface, grid=None):

    grid = tuple(grid) if grid is not None else LAMBDA_GRID

    print("  Generalisation gap (test LL - train LL), mean over folds.")
    print("  This is the quantity that diagnoses whether lambda = 1 was the")
    print("  problem: if the gap at B6 collapses as lambda rises, the frozen")
    print("  ladder was measuring overfitting rather than absent signal.")
    print()

    header = "  {:<5}".format("rung") + "".join(
        "{:>9g}".format(p) for p in grid)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rung in ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]:

        block = surface[surface["rung"] == rung]

        if not len(block):
            continue

        print("  {:<5}{}".format(rung, "".join(
            "{:>9.4f}".format(float(
                block[block["lambda"] == p]["generalisation_gap"].mean()))
            for p in grid)))

    print()


def print_block_norms(block_norms, selected):

    print("  Coefficient L2 norm by feature block, at the SELECTED lambda,")
    print("  mean over folds. Columns are standardised, so these are")
    print("  comparable across blocks. Per-column figures in brackets.")
    print()

    chosen = block_norms[block_norms["is_selected"]]

    blocks_present = [b for b in BLOCK_ORDER
                      if b in set(chosen["block"].unique())]

    header = "  {:<5}".format("rung") + "".join(
        "{:>26}".format(b[:26]) for b in blocks_present)
    print(header)
    print("  " + "-" * min(len(header) - 2, 140))

    for rung in ["B1", "B2", "B3", "B4", "B5", "B6"]:

        part = chosen[chosen["rung"] == rung]

        if not len(part):
            continue

        cells = ""

        for block in blocks_present:
            values = part[part["block"] == block]
            if len(values):
                cells += "{:>15.3f} [{:>6.3f}]".format(
                    float(values["block_l2_norm"].mean()),
                    float(values["block_l2_norm_per_column"].mean()))
            else:
                cells += "{:>26}".format("-")

        print("  {:<5}{}".format(rung, cells))

    print()


def print_verdict(verdict, table, selected):

    print("  D(rung)  = mean over the four folds of [LL(rung) - LL(B1)], paired")
    print("  SE(rung) = sd(D, ddof=1) / sqrt(4)")
    print()
    print("  {:<6} {:>12} {:>12} {:>10} {:>12} {:>10}".format(
        "rung", "mean LL", "D vs B1", "SE", "D + SE", "beats B1"))
    print("  " + "-" * 68)

    for _index, row in verdict.iterrows():
        print("  {:<6} {:>12.4f} {:>+12.4f} {:>10.4f} {:>+12.4f} {:>10}".format(
            row["rung"], row["mean_log_loss"], row["D_vs_B1"],
            row["SE_of_D"], row["D_plus_SE"],
            "yes" if row["beats_B1_by_1SE"] else "no"))

    print()
    print("  Best rung by mean outer-test log loss : {}".format(
        verdict.attrs["winner"]))
    print()
    print("  PRE-DECLARED VERDICT : {}".format(verdict.attrs["verdict"]))
    print()

    if verdict.attrs["reframed"]:
        print("  Additionally REFRAMED: the winner is unchanged, but the median")
        print("  selected lambda is {:g} (not 1) and the B1-to-B6 spread fell".format(
            verdict.attrs["median_lambda"]))
        print("  from {:.4f} to {:.4f}. The frozen ladder overstated the harm.".format(
            verdict.attrs["gap_frozen"], verdict.attrs["gap_now"]))
        print()


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 3 - INSTRUMENT 4: REGULARISATION SENSITIVITY, B0 - B6")

    print("  question  : is 'B2-B6 do not help' a fact about the features,")
    print("              or a fact about lambda = 1?")
    print()
    print("  model     : unchanged - imported from Instrument 3, not copied")
    print("  grid      : {}".format(", ".join("{:g}".format(p) for p in LAMBDA_GRID)))
    print("  selection : expanding-window inner CV over calendar-date")
    print("              blocks, cut from TRAINING rows only, {} inner folds".format(
        N_INNER_FOLDS))
    print("  criterion : mean inner-validation {}, ties to the LARGER lambda".format(
        PRIMARY_METRIC))
    print("  declared  : {}".format(PREDECLARATION.name))
    print("              sha256 {}".format(hash_file(PREDECLARATION)[:32]))
    print()
    print("  The grid, the inner procedure, the tie-break and the criterion")
    print("  for 'the conclusion changed' were all fixed in that file before")
    print("  the first fit. R6 asserts the grid still matches it.")
    print()

    before_state = frozen_state()

    matches = L3.load_matches()
    features = L3.load_features(matches)
    labels = np.array([CLASS_INDEX[r] for r in matches["result"]], dtype=int)
    spec = L3.load_spec()

    ladder = [list(entry) for entry in L3.LADDER]
    ladder[1][2] = L3.phase1_feature_columns(features)
    ladder = [tuple(entry) for entry in ladder]

    blocks = date_blocks(matches)

    print("  matches {}, feature columns {}, calendar-date blocks {}".format(
        len(matches), features.shape[1], len(blocks)))
    print()

    banner("1. SELECTING LAMBDA, AND THE ORACLE SURFACE")

    print("  {} rungs x {} folds x {} lambdas. Inner CV first, then the full".format(
        len(ladder), len(spec["folds"]), len(LAMBDA_GRID)))
    print("  outer-test surface. This takes a few minutes.")
    print()

    run = Run(features, matches, labels, spec, ladder, blocks,
              compute_surface=True).execute()

    selected = run.selected_frame()
    surface = run.surface_frame()
    curves = run.curves_frame()
    block_norms = run.block_norm_frame()

    frozen_ladder = pd.read_csv(FROZEN_LADDER, float_precision=FLOAT_PRECISION)

    table = build_ladder_table(selected, frozen_ladder)
    verdict = build_verdict(selected, table, frozen_ladder)

    print()
    banner("2. THE SELECTED LAMBDAS")
    print_selected_lambdas(selected)

    banner("3. THE LADDER AT SELECTED LAMBDA, AGAINST THE FROZEN LADDER")
    print_ladder(table)

    banner("4. THE ORACLE SURFACE - EVERY LAMBDA, OUTER TEST")
    print_surface(surface, "log_loss")

    banner("5. WHERE THE SCORE WENT - GENERALISATION GAP BY LAMBDA")
    print_generalisation(surface)

    banner("6. COEFFICIENT NORMS BY BLOCK, AT SELECTED LAMBDA")
    print_block_norms(block_norms, selected)

    banner("7. THE PRE-DECLARED VERDICT")
    print_verdict(verdict, table, selected)

    banner("8. AUDIT")

    audit = Audit()
    test_everything(run, features, matches, labels, spec, ladder, blocks,
                    selected, table, before_state, audit)
    print()
    audit.print_rows()

    banner("9. WRITING OUTPUTS")

    verdict_frame = verdict.copy()
    verdict_frame["verdict"] = verdict.attrs["verdict"]
    verdict_frame["winner"] = verdict.attrs["winner"]
    verdict_frame["reframed"] = verdict.attrs["reframed"]
    verdict_frame["median_selected_lambda"] = verdict.attrs["median_lambda"]
    verdict_frame["b1_to_b6_gap_selected"] = verdict.attrs["gap_now"]
    verdict_frame["b1_to_b6_gap_frozen"] = verdict.attrs["gap_frozen"]
    verdict_frame["predeclaration_sha256"] = hash_file(PREDECLARATION)

    writes = [
        (CURVES_OUTPUT, curves),
        (SELECTED_OUTPUT, selected),
        (SURFACE_OUTPUT, surface),
        (BLOCK_NORM_OUTPUT, block_norms),
        (LADDER_OUTPUT, table),
        (VERDICT_OUTPUT, verdict_frame),
        (AUDIT_OUTPUT, audit.frame()),
    ]

    for path, frame in writes:
        frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
        print("  {}".format(path))

    print()

    banner("PHASE 3 - INSTRUMENT 4 STATUS")

    failures = audit.failures

    print("  Lambdas in grid    : {}".format(len(LAMBDA_GRID)))
    print("  Selections made    : {}".format(len(selected)))
    print("  Outer fits scored  : {}".format(len(surface)))
    print("  Inner fits scored  : {}".format(len(curves)))
    print("  Checks run         : {}".format(len(audit.rows)))
    print("  Checks failed      : {}".format(len(failures)))
    print("  Elapsed            : {:.1f} min".format((time.time() - started) / 60.0))
    print()
    print("  {}".format("PASS" if not failures else "FAIL"))
    print()
    print("  Verdict: {}{}".format(
        verdict.attrs["verdict"],
        " (REFRAMED)" if verdict.attrs["reframed"] else ""))
    print()
    print("Lambda was selected on training data alone, by a procedure declared")
    print("before the first fit. The outer test season is proved unreachable by")
    print("the selector twice over: a runtime tripwire, and perturbation of both")
    print("its features and its labels. The oracle surface is reported and was")
    print("used for nothing. The lambda = 1 reference was read, never rewritten.")
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
