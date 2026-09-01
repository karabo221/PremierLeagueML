"""
===============================================================================
PHASE 2 - INSTRUMENT 1
BASE-RATE PROBABILITY BASELINE
===============================================================================

OBJECTIVE
    Establish the simplest legitimate probabilistic baseline for Premier
    League match-result prediction, and the number every later model must
    beat.

        P(H) = home wins in training / training matches
        P(D) = draws in training     / training matches
        P(A) = away wins in training / training matches

    Every test match in a fold receives that one vector. No team identity,
    no features, no current-season information, no previous-season
    information. That is the point: it measures what league-wide H/D/A
    frequency alone buys you.

WALK-FORWARD, NOT POOLED
    The tempting shortcut is to compute one distribution from all 1,900
    matches. That would let 2025-26 outcomes inform the 2022-23 prediction,
    which is the exact leak Phase 0 was built to prevent.

    So four separate distributions are estimated - each from that fold's
    training seasons alone:

        Fold 1   train 2021-2022                     ->  test 2022-2023
        Fold 2   train 2021-2022 .. 2022-2023        ->  test 2023-2024
        Fold 3   train 2021-2022 .. 2023-2024        ->  test 2024-2025
        Fold 4   train 2021-2022 .. 2024-2025        ->  test 2025-2026

METRICS ARE IMPORTED, NOT REIMPLEMENTED
    accuracy, balanced accuracy, macro F1, log loss and Brier all come from
    scripts/phase0_evaluation_harness.py via `evaluate`. A baseline scored by
    its own private metric implementation is not comparable to anything, and
    comparability is the entire purpose of a baseline.

    The harness also owns `validate_probabilities`, the gate every model
    output must pass. This baseline passes through it like any other model.

SOURCES - all frozen, all read-only
    outputs/phase0_evaluation_spec.json    the frozen fold definition
    outputs/phase0_evaluation_folds.csv    cross-check of the same
    outputs/phase1_matches.csv             the validated match foundation

    NOTE: the Phase 0 harness's own load_matches() reads data/raw/Fixtures.
    It is deliberately NOT called. Match data comes from Phase 1's validated
    foundation instead, so this instrument never touches raw data (T18).

    Phase 0 and Phase 1 outputs and scripts are SHA-256 hashed before and
    after the run and required to be identical (T19).

EXIT CODES
    0  PASS    every test passed
    2  FAIL    a test failed - investigate, do not weaken the test
    1  FATAL   the instrument could not be run at all

WHAT IS NOT BUILT HERE
    no Elo, no Poisson, no Dixon-Coles, no logistic regression, no random
    forest, no XGBoost, no feature selection, no hyperparameter tuning, and
    no new evaluation methodology.
===============================================================================
"""

from pathlib import Path
import hashlib
import json
import sys
import traceback

import numpy as np
import pandas as pd


# ============================================================
# FILE-ACCESS RECORDER  (evidence for T18)
# ============================================================

_OPENED_PATHS = []


def _record_open(event, args):

    if event != "open":
        return

    target = args[0]

    if isinstance(target, (str, bytes, Path)):
        _OPENED_PATHS.append(str(target))


sys.addaudithook(_record_open)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

sys.path.insert(0, str(SCRIPTS_DIR))

# The single scoring function, and the probability contract, both owned by
# Phase 0. Importing the module does not execute it - it is __main__ guarded.
from phase0_evaluation_harness import (  # noqa: E402
    CLASSES,
    CLASS_INDEX,
    PROBABILITY_TOLERANCE,
    ProbabilityError,
    accuracy_score,
    balanced_accuracy_score,
    brier_score,
    encode_labels,
    evaluate,
    log_loss_score,
    macro_f1_score,
    rps_score,
    validate_probabilities,
)


# ============================================================
# CONFIGURATION
# ============================================================

SPEC_JSON = OUTPUTS_DIR / "phase0_evaluation_spec.json"
FOLDS_CSV = OUTPUTS_DIR / "phase0_evaluation_folds.csv"
MATCHES_CSV = OUTPUTS_DIR / "phase1_matches.csv"

RESULTS_OUTPUT = OUTPUTS_DIR / "phase2_base_rate_results.csv"
SUMMARY_OUTPUT = OUTPUTS_DIR / "phase2_base_rate_fold_summary.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase2_base_rate_audit.csv"

DECLARED_INPUTS = {
    SPEC_JSON.resolve(), FOLDS_CSV.resolve(), MATCHES_CSV.resolve(),
}

OWN_OUTPUTS = {
    RESULTS_OUTPUT.resolve(), SUMMARY_OUTPUT.resolve(), AUDIT_OUTPUT.resolve(),
}

EXPECTED_FOLDS = 4
EXPECTED_TRAIN_SIZES = [380, 760, 1140, 1520]
EXPECTED_TEST_SIZE = 380
EXPECTED_TOTAL_TEST = 1520
EXPECTED_TOTAL_MATCHES = 1900

EXIT_PASS = 0
EXIT_FATAL = 1
EXIT_FAIL = 2

# float64 needs 17 significant digits to round-trip; pandas writes 16 by
# default and its C reader loses a ULP on the way back. Phase 1 Instrument 5
# established both halves of this - carry them forward so T16 and T17 compare
# real values rather than rounding artefacts.
FLOAT_FORMAT = "%.17g"
FLOAT_PRECISION = "round_trip"

RESULTS_COLUMNS = [
    "fold", "test_season", "date", "home", "away", "actual_result",
    "p_home", "p_draw", "p_away", "predicted_result",
]

SUMMARY_COLUMNS = [
    "fold", "train_seasons", "test_season",
    "train_matches", "test_matches",
    "train_home_wins", "train_draws", "train_away_wins",
    "train_p_home", "train_p_draw", "train_p_away",
    "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score",
    "rps",
]

METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score",
    "rps",
]


class FatalError(Exception):
    """The instrument could not be run at all. Exit 1, not exit 2."""


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
# FROZEN-STATE GUARD  (T19)
# ============================================================

def hash_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def frozen_state():
    """SHA-256 of every Phase 0 and Phase 1 script and output."""

    tracked = {}

    for directory, patterns in (
        (SCRIPTS_DIR, ["phase0_*.py", "phase1_*.py"]),
        (OUTPUTS_DIR, ["phase0_*.csv", "phase0_*.json", "phase1_*.csv"]),
    ):
        for pattern in patterns:
            for path in sorted(directory.glob(pattern)):
                tracked[str(path.relative_to(PROJECT_ROOT))] = hash_file(path)

    return tracked


# ============================================================
# INPUT
# ============================================================

def load_spec():

    for required in (SPEC_JSON, FOLDS_CSV, MATCHES_CSV):
        if not required.exists():
            raise FatalError(f"missing required input: {required}")

    try:
        spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
    except Exception as error:
        raise FatalError(f"frozen spec could not be parsed: {error}") from error

    if "folds" not in spec:
        raise FatalError("frozen spec carries no folds")

    return spec


def load_matches():
    """
    Match data from Phase 1's validated foundation.

    Deliberately NOT the Phase 0 harness's own load_matches(), which reads
    data/raw/Fixtures - see the module docstring.
    """

    matches = pd.read_csv(MATCHES_CSV, float_precision=FLOAT_PRECISION)

    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")

    matches = matches.sort_values(
        ["season", "date", "home_team", "away_team"]
    ).reset_index(drop=True)

    matches["match_id"] = matches.index

    if len(matches) != EXPECTED_TOTAL_MATCHES:
        raise FatalError(
            f"foundation has {len(matches)} matches, "
            f"expected {EXPECTED_TOTAL_MATCHES}"
        )

    unknown = set(matches["result"]) - set(CLASSES)

    if unknown:
        raise FatalError(f"unexpected result labels: {sorted(unknown)}")

    return matches


# ============================================================
# THE BASELINE
# ============================================================

def base_rate(training_results):
    """
    The whole model.

    Returns [P(H), P(D), P(A)] in the frozen class order. The order is taken
    from the harness's CLASSES rather than hard-coded, so the vector cannot
    drift out of step with the contract it is scored against.
    """

    counts = np.array(
        [int((training_results == label).sum()) for label in CLASSES],
        dtype=float,
    )

    total = counts.sum()

    if total <= 0:
        raise FatalError("cannot estimate a base rate from zero training matches")

    return counts, counts / total


def run_baseline(matches, spec):
    """
    Fit and predict for all four folds.

    Returns (results, summaries). Called more than once - by the determinism
    check (T17) and by the perturbation control (T20) - so it must be a pure
    function of its arguments.
    """

    results = []
    summaries = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        train_seasons = list(fold_spec["train_seasons"])
        test_season = str(fold_spec["test_season"])

        train = matches[matches["season"].isin(train_seasons)]
        test = matches[matches["season"] == test_season]

        # ---- fit: training results only, nothing else enters
        counts, probabilities = base_rate(train["result"])

        # ---- predict: the same vector for every test match
        proba = np.tile(probabilities, (len(test), 1))

        # The harness's own gate, applied to this model like any other.
        proba = validate_probabilities(proba, len(test))

        predicted_index = np.argmax(proba, axis=1)

        fold_results = pd.DataFrame({
            "fold": fold,
            "test_season": test_season,
            "date": test["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "home": test["home_team"].to_numpy(),
            "away": test["away_team"].to_numpy(),
            "actual_result": test["result"].to_numpy(),
            "p_home": proba[:, 0],
            "p_draw": proba[:, 1],
            "p_away": proba[:, 2],
            "predicted_result": [CLASSES[i] for i in predicted_index],
        })

        results.append(fold_results)

        scores = evaluate(test["result"].to_numpy(), proba)

        summaries.append({
            "fold": fold,
            "train_seasons": " + ".join(train_seasons),
            "test_season": test_season,
            "train_matches": int(len(train)),
            "test_matches": int(len(test)),
            "train_home_wins": int(counts[CLASS_INDEX["H"]]),
            "train_draws": int(counts[CLASS_INDEX["D"]]),
            "train_away_wins": int(counts[CLASS_INDEX["A"]]),
            "train_p_home": float(probabilities[CLASS_INDEX["H"]]),
            "train_p_draw": float(probabilities[CLASS_INDEX["D"]]),
            "train_p_away": float(probabilities[CLASS_INDEX["A"]]),
            "accuracy": scores["accuracy"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "macro_f1": scores["macro_f1"],
            "log_loss": scores["log_loss"],
            "brier_score": scores["brier_score"],
            "rps": scores["rps"],
        })

    return (
        pd.concat(results, ignore_index=True)[RESULTS_COLUMNS],
        pd.DataFrame(summaries)[SUMMARY_COLUMNS],
    )


# ============================================================
# TESTS
# ============================================================

def test_fold_structure(matches, spec, summary, audit):

    folds = spec["folds"]

    audit.record(
        "T1", "Exactly four folds exist",
        "frozen Phase 0 spec",
        EXPECTED_FOLDS, len(folds),
        len(folds) == EXPECTED_FOLDS,
    )

    # ---- T2: folds match the frozen spec, cross-checked against the CSV too
    folds_csv = pd.read_csv(FOLDS_CSV, float_precision=FLOAT_PRECISION)

    disagreements = []

    for fold_spec in folds:

        fold = int(fold_spec["fold"])

        row = folds_csv[folds_csv["fold"] == fold]

        if row.empty:
            disagreements.append(f"fold {fold} absent from folds CSV")
            continue

        row = row.iloc[0]

        expected_train = " + ".join(fold_spec["train_seasons"])

        if str(row["train_seasons"]) != expected_train:
            disagreements.append(
                f"fold {fold} train {row['train_seasons']!r} != {expected_train!r}"
            )

        if str(row["test_season"]) != str(fold_spec["test_season"]):
            disagreements.append(f"fold {fold} test season disagrees")

        used = summary[summary["fold"] == fold].iloc[0]

        if used["train_seasons"] != expected_train:
            disagreements.append(f"fold {fold} baseline used different train seasons")

        if used["test_season"] != str(fold_spec["test_season"]):
            disagreements.append(f"fold {fold} baseline used a different test season")

    audit.record(
        "T2", "Fold train/test seasons match the frozen Phase 0 specification",
        "spec JSON, folds CSV and the baseline itself",
        "0 disagreements", f"{len(disagreements)} disagreements",
        not disagreements,
        "; ".join(disagreements[:5]),
    )

    # ---- T3: training sizes
    observed_train = list(summary.sort_values("fold")["train_matches"])

    audit.record(
        "T3", "Training sizes are 380 / 760 / 1140 / 1520",
        f"{EXPECTED_FOLDS} folds",
        str(EXPECTED_TRAIN_SIZES), str(observed_train),
        observed_train == EXPECTED_TRAIN_SIZES,
    )

    # ---- T4: test sizes
    observed_test = list(summary.sort_values("fold")["test_matches"])

    audit.record(
        "T4", "Every test set contains exactly 380 matches",
        f"{EXPECTED_FOLDS} folds",
        f"{EXPECTED_TEST_SIZE} x {EXPECTED_FOLDS}", str(observed_test),
        observed_test == [EXPECTED_TEST_SIZE] * EXPECTED_FOLDS,
    )

    # ---- T5: no test match appears in training
    overlaps = []

    for fold_spec in folds:

        fold = int(fold_spec["fold"])

        train = matches[matches["season"].isin(fold_spec["train_seasons"])]
        test = matches[matches["season"] == fold_spec["test_season"]]

        key = ["season", "date", "home_team", "away_team"]

        train_keys = set(map(tuple, train[key].astype(str).to_numpy()))
        test_keys = set(map(tuple, test[key].astype(str).to_numpy()))

        shared = train_keys & test_keys

        if shared:
            overlaps.append(f"fold {fold}: {len(shared)} shared matches")

        # Identity overlap too - the same match_id must never be on both sides
        if set(train["match_id"]) & set(test["match_id"]):
            overlaps.append(f"fold {fold}: shared match_id")

    audit.record(
        "T5", "No test match occurs in training data",
        f"{EXPECTED_FOLDS} folds",
        0, len(overlaps),
        not overlaps,
        "; ".join(overlaps),
    )

    # ---- T6: every training date strictly earlier than every test date
    boundary = []

    for fold_spec in folds:

        fold = int(fold_spec["fold"])

        train = matches[matches["season"].isin(fold_spec["train_seasons"])]
        test = matches[matches["season"] == fold_spec["test_season"]]

        max_train = train["date"].max()
        min_test = test["date"].min()

        if not max_train < min_test:
            boundary.append(
                f"fold {fold}: max train {max_train.date()} "
                f"not before min test {min_test.date()}"
            )

        audit.measure(
            "T6b", "Temporal gap between training and test",
            f"fold {fold}",
            f"{(min_test - max_train).days} days",
            f"train ends {max_train.date()}, test starts {min_test.date()}",
        )

    audit.record(
        "T6", "Every training date is earlier than every test date",
        f"{EXPECTED_FOLDS} folds",
        0, len(boundary),
        not boundary,
        "; ".join(boundary),
    )


def test_probability_estimation(matches, spec, summary, audit):

    # ---- T7: probabilities recomputed independently from the training slice
    mismatches = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train = matches[matches["season"].isin(fold_spec["train_seasons"])]

        row = summary[summary["fold"] == fold].iloc[0]

        # Independent recount, deliberately not via base_rate()
        home = int((train["result"] == "H").sum())
        draw = int((train["result"] == "D").sum())
        away = int((train["result"] == "A").sum())
        total = len(train)

        expected = {
            "train_home_wins": home,
            "train_draws": draw,
            "train_away_wins": away,
            "train_p_home": home / total,
            "train_p_draw": draw / total,
            "train_p_away": away / total,
        }

        for field, value in expected.items():

            observed = row[field]

            if isinstance(value, int):
                if int(observed) != value:
                    mismatches.append(f"fold {fold} {field}: {observed} != {value}")
            elif not np.isclose(float(observed), value, atol=1e-15):
                mismatches.append(f"fold {fold} {field}: {observed} != {value}")

        if home + draw + away != total:
            mismatches.append(f"fold {fold}: counts do not sum to training size")

    audit.record(
        "T7", "Base-rate probabilities are calculated ONLY from training data",
        f"{EXPECTED_FOLDS} folds",
        "0 mismatches", f"{len(mismatches)} mismatches",
        not mismatches,
        "Recounted independently from the training slice",
    )

    # ---- T8 / T20: test outcomes cannot influence the probabilities
    #
    # PER-FOLD, and that detail is the whole test.
    #
    # An earlier version rewrote every test season at once and reported 18
    # changed values. It was measuring the wrong thing: 2022-2023 is fold 1's
    # TEST season and folds 2-4's TRAINING season, so a global rewrite moves
    # folds 2-4 legitimately. That is walk-forward working, not leaking.
    #
    # The real question is narrower: does fold k's own test season reach
    # fold k's own estimator? So perturb one fold's test season and inspect
    # only that fold's row.
    perturbation_changes = []
    perturbation_checks = 0

    probability_fields = [
        "train_home_wins", "train_draws", "train_away_wins",
        "train_p_home", "train_p_draw", "train_p_away",
    ]

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = fold_spec["test_season"]

        for forced in ("H", "D", "A"):

            perturbed = matches.copy()

            perturbed.loc[
                perturbed["season"] == test_season, "result"
            ] = forced

            _, perturbed_summary = run_baseline(perturbed, spec)

            before = summary[summary["fold"] == fold].iloc[0]
            after = perturbed_summary[perturbed_summary["fold"] == fold].iloc[0]

            for field in probability_fields:

                perturbation_checks += 1

                if float(before[field]) != float(after[field]):
                    perturbation_changes.append(
                        f"fold {fold} forced {forced}: {field} "
                        f"{before[field]} -> {after[field]}"
                    )

    audit.record(
        "T8", "No test outcome influences the calculated probabilities",
        f"{perturbation_checks} field checks "
        f"({EXPECTED_FOLDS} folds x 3 forced outcomes x {len(probability_fields)} fields)",
        "0 changed values", f"{len(perturbation_changes)} changed",
        not perturbation_changes,
        "Each fold's own test season rewritten, that fold's estimate inspected",
    )

    audit.record(
        "T20", "Perturbation control: rewriting test results leaves the baseline fixed",
        f"{EXPECTED_FOLDS * 3} refits, 380 outcomes rewritten each",
        "0 changed probabilities", f"{len(perturbation_changes)} changed",
        not perturbation_changes,
        "; ".join(perturbation_changes[:5]) if perturbation_changes
        else "4,560 test outcomes rewritten in total; no estimate moved",
    )

    # Positive control: perturbing TRAINING data MUST move the distribution,
    # otherwise the two tests above pass by doing nothing at all.
    trained_on = spec["folds"][-1]["train_seasons"]

    control = matches.copy()
    control.loc[control["season"].isin(trained_on), "result"] = "D"

    _, control_summary = run_baseline(control, spec)

    moved = int((
        control_summary.sort_values("fold")["train_p_draw"].to_numpy()
        != summary.sort_values("fold")["train_p_draw"].to_numpy()
    ).sum())

    audit.record(
        "T8b", "Positive control: perturbing TRAINING data does move the baseline",
        f"{EXPECTED_FOLDS} folds",
        "> 0 folds change", f"{moved} folds changed",
        moved > 0,
        "Proves T8 and T20 are not vacuous",
    )


def test_prediction_contract(results, summary, audit):

    # ---- T9: one vector per fold
    varying = []

    for fold, group in results.groupby("fold"):

        distinct = group[["p_home", "p_draw", "p_away"]].drop_duplicates()

        if len(distinct) != 1:
            varying.append(f"fold {fold}: {len(distinct)} distinct vectors")

    audit.record(
        "T9", "Every test match in a fold receives the same probability vector",
        f"{EXPECTED_FOLDS} folds",
        "1 distinct vector per fold", f"{len(varying)} folds with more",
        not varying,
        "; ".join(varying),
    )

    # ---- T10: order is [P(H), P(D), P(A)]
    order_problems = []

    if list(CLASSES) != ["H", "D", "A"]:
        order_problems.append(f"harness class order is {CLASSES}")

    if RESULTS_COLUMNS[6:9] != ["p_home", "p_draw", "p_away"]:
        order_problems.append("results columns are not p_home, p_draw, p_away")

    for fold, group in results.groupby("fold"):

        row = summary[summary["fold"] == fold].iloc[0]

        first = group.iloc[0]

        # The column labelled p_home must carry the HOME-win frequency, and so on.
        if not np.isclose(first["p_home"], row["train_p_home"], atol=1e-15):
            order_problems.append(f"fold {fold}: p_home is not the H frequency")

        if not np.isclose(first["p_draw"], row["train_p_draw"], atol=1e-15):
            order_problems.append(f"fold {fold}: p_draw is not the D frequency")

        if not np.isclose(first["p_away"], row["train_p_away"], atol=1e-15):
            order_problems.append(f"fold {fold}: p_away is not the A frequency")

    audit.record(
        "T10", "Probability vectors are ordered [P(H), P(D), P(A)]",
        "results table and harness contract",
        "0 order problems", f"{len(order_problems)} problems",
        not order_problems,
        f"harness CLASSES = {CLASSES}",
    )

    # ---- T11: rows sum to 1
    sums = results[["p_home", "p_draw", "p_away"]].sum(axis=1)

    off = int((np.abs(sums - 1.0) > PROBABILITY_TOLERANCE).sum())

    audit.record(
        "T11", "Every probability vector sums to 1 within 1e-6",
        f"{len(results)} test matches",
        0, off,
        off == 0,
        f"max deviation {float(np.abs(sums - 1.0).max()):.3e}",
    )

    # ---- T12: no NaN or infinity, and every value inside [0, 1]
    block = results[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

    bad = int((~np.isfinite(block)).sum())

    out_of_range = int(((block < 0.0) | (block > 1.0)).sum())

    audit.record(
        "T12", "No NaN or infinity exists in the probability output",
        f"{block.size} probability values",
        0, bad,
        bad == 0,
    )

    audit.record(
        "T12b", "Every probability lies in [0, 1]",
        f"{block.size} probability values",
        0, out_of_range,
        out_of_range == 0,
    )

    # ---- T13: predicted_result is argmax
    expected_prediction = [CLASSES[i] for i in np.argmax(block, axis=1)]

    wrong = int((results["predicted_result"].to_numpy() != np.array(
        expected_prediction)).sum())

    audit.record(
        "T13", "predicted_result is argmax([p_home, p_draw, p_away])",
        f"{len(results)} test matches",
        0, wrong,
        wrong == 0,
    )

    # Ties would make argmax order-dependent. Record whether any exist.
    ties = int((
        np.sum(block == block.max(axis=1, keepdims=True), axis=1) > 1
    ).sum())

    audit.record(
        "T13b", "No probability tie makes the argmax ambiguous",
        f"{len(results)} test matches",
        0, ties,
        ties == 0,
        "A tie would make predicted_result depend on column order",
    )

    # ---- T14 / T15: coverage
    audit.record(
        "T14", "Results contain exactly 1,520 test matches",
        "4 folds x 380",
        EXPECTED_TOTAL_TEST, len(results),
        len(results) == EXPECTED_TOTAL_TEST,
    )

    duplicated = int(results.duplicated(
        subset=["test_season", "date", "home", "away"]
    ).sum())

    audit.record(
        "T15", "Each test match appears exactly once",
        f"{len(results)} rows",
        0, duplicated,
        duplicated == 0,
    )

    seasons_tested = sorted(results["test_season"].unique())

    audit.record(
        "T15b", "Each test season appears exactly once, and 2021-2022 never",
        "4 test seasons",
        "['2022-2023', '2023-2024', '2024-2025', '2025-2026']",
        str(seasons_tested),
        seasons_tested == ["2022-2023", "2023-2024", "2024-2025", "2025-2026"],
        "2021-2022 can only ever be training",
    )


def test_metric_reproduction(results, summary, audit):
    """T16 - fold metrics must be reproducible from the results table alone."""

    mismatches = []

    for fold, group in results.groupby("fold"):

        proba = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

        recomputed = evaluate(group["actual_result"].to_numpy(), proba)

        row = summary[summary["fold"] == fold].iloc[0]

        for metric in METRIC_NAMES:

            if not np.isclose(float(row[metric]), recomputed[metric], atol=1e-12):
                mismatches.append(
                    f"fold {fold} {metric}: {row[metric]} != {recomputed[metric]}"
                )

    audit.record(
        "T16", "Fold-summary metrics reproduce from the results table",
        f"{EXPECTED_FOLDS} folds x {len(METRIC_NAMES)} metrics",
        "0 mismatches", f"{len(mismatches)} mismatches",
        not mismatches,
        "; ".join(mismatches[:5]),
    )

    # The metrics come from the Phase 0 harness, not from a local copy.
    imported_from_harness = all(
        function.__module__ == "phase0_evaluation_harness"
        for function in (
            evaluate, accuracy_score, balanced_accuracy_score,
            macro_f1_score, log_loss_score, brier_score, rps_score,
            validate_probabilities,
        )
    )

    audit.record(
        "T16b", "Every metric is imported from the Phase 0 harness, not reimplemented",
        "8 scoring functions",
        "all from phase0_evaluation_harness",
        evaluate.__module__,
        imported_from_harness,
        "A baseline scored by a private metric copy is not comparable",
    )


def test_determinism(matches, spec, results, summary, audit):
    """T17 - running the pipeline again must produce identical output."""

    repeat_results, repeat_summary = run_baseline(matches, spec)

    results_identical = results.equals(repeat_results)
    summary_identical = summary.equals(repeat_summary)

    audit.record(
        "T17", "Running the script twice produces identical outputs",
        "full pipeline re-executed in-process",
        "identical",
        f"results {'identical' if results_identical else 'DIFFER'}, "
        f"summary {'identical' if summary_identical else 'DIFFER'}",
        results_identical and summary_identical,
    )

    # And the written artefact must round-trip exactly, so a later instrument
    # comparing against the CSV compares real values.
    if RESULTS_OUTPUT.exists():

        reloaded = pd.read_csv(RESULTS_OUTPUT, float_precision=FLOAT_PRECISION)

        columns = ["p_home", "p_draw", "p_away"]

        deviation = float(np.max(np.abs(
            reloaded[columns].to_numpy(dtype=float)
            - results[columns].to_numpy(dtype=float)
        ))) if len(reloaded) == len(results) else float("inf")

        audit.record(
            "T17b", "Written probabilities round-trip through CSV exactly",
            f"{len(results)} rows x 3 probabilities",
            "0.0 deviation", f"{deviation:.3e}",
            deviation == 0.0,
            f"float_format={FLOAT_FORMAT}, float_precision={FLOAT_PRECISION}",
        )


def test_isolation(before_state, audit):
    """T18 and T19 - raw data untouched, frozen phases untouched."""

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
        "T18", "The script does not read data/raw/",
        "runtime file-access record",
        0, len(raw_touches),
        not raw_touches,
        "The Phase 0 harness's own load_matches() reads data/raw and is "
        "deliberately never called",
    )

    data_files = {
        path for path in opened
        if path.suffix.lower() in {".csv", ".json", ".xls", ".xlsx"}
        and PROJECT_ROOT in path.parents
    }

    # The T19 guard OPENS every frozen Phase 0 and Phase 1 file in order to
    # hash it. That is an integrity read, not a modelling input, so it is
    # allowed here - but only for files the guard actually tracks. Anything
    # else, an FBref aggregate above all, still fails.
    hashed_for_guard = {
        (PROJECT_ROOT / name).resolve() for name in before_state
    }

    unexpected = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in data_files - DECLARED_INPUTS - OWN_OUTPUTS - hashed_for_guard
    )

    audit.record(
        "T18b", "Only declared inputs and hash-guarded frozen files were read",
        "runtime file-access record",
        "0 unexpected", f"{len(unexpected)} unexpected",
        not unexpected,
        f"{len(DECLARED_INPUTS)} declared inputs, "
        f"{len(hashed_for_guard)} files opened solely to hash for T19",
    )

    after_state = frozen_state()

    changed = sorted(
        name for name in before_state
        if before_state[name] != after_state.get(name)
    )

    missing = sorted(set(before_state) - set(after_state))

    added = sorted(set(after_state) - set(before_state))

    audit.record(
        "T19", "The script does not modify Phase 0 or Phase 1 outputs or scripts",
        f"{len(before_state)} tracked files, SHA-256 before and after",
        "0 changed", f"{len(changed) + len(missing) + len(added)} changed",
        not changed and not missing and not added,
        f"changed: {changed[:3]}; removed: {missing[:3]}; added: {added[:3]}"
        if (changed or missing or added) else "every hash identical",
    )

    audit.measure(
        "T19b", "Frozen Phase 0 and Phase 1 files under hash guard",
        "scripts and outputs", len(before_state),
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

    markers = {"PASS": "PASS", "FAIL": "FAIL", "MEASURED": "----"}

    for row in audit.frame().itertuples():

        print(f"  {markers[row.status]}  {row.test_id:<5} {row.test}")
        print(f"              scope   : {row.scope}")
        print(f"              expected: {row.expected}")
        print(f"              observed: {row.observed}")

        if row.detail:
            print(f"              {row.detail}")


def print_distributions(summary):

    print()
    print("=" * 79)
    print("FOUR FOLD PROBABILITY DISTRIBUTIONS")
    print("=" * 79)
    print()
    print("  Each estimated from that fold's TRAINING seasons alone. The")
    print("  distribution shifts fold to fold because the training window grows.")
    print()
    print(
        f"    {'Fold':<5}{'Train seasons':<44}{'N':>5}"
        f"{'P(H)':>8}{'P(D)':>8}{'P(A)':>8}"
    )

    for row in summary.itertuples():

        seasons = row.train_seasons

        if len(seasons) > 42:
            seasons = seasons[:39] + "..."

        print(
            f"    {row.fold:<5}{seasons:<44}{row.train_matches:>5}"
            f"{row.train_p_home:>8.4f}{row.train_p_draw:>8.4f}"
            f"{row.train_p_away:>8.4f}"
        )

    print()
    print(
        f"    {'Fold':<5}{'Train counts H / D / A':<44}"
        f"{'Test season':>14}"
    )

    for row in summary.itertuples():
        print(
            f"    {row.fold:<5}"
            f"{f'{row.train_home_wins} / {row.train_draws} / {row.train_away_wins}':<44}"
            f"{row.test_season:>14}"
        )


def print_metrics(summary, overall, results):

    print()
    print("=" * 79)
    print("SIX METRICS PER FOLD")
    print("=" * 79)
    print()
    print("  Metrics computed by the Phase 0 harness's evaluate().")
    print()
    print(
        f"    {'Fold':<5}{'Test season':<13}{'N':>5}{'Acc':>9}{'BalAcc':>9}"
        f"{'MacroF1':>9}{'LogLoss':>9}{'Brier':>9}{'RPS':>9}"
    )

    for row in summary.itertuples():
        print(
            f"    {row.fold:<5}{row.test_season:<13}{row.test_matches:>5}"
            f"{row.accuracy:>9.4f}{row.balanced_accuracy:>9.4f}"
            f"{row.macro_f1:>9.4f}{row.log_loss:>9.4f}{row.brier_score:>9.4f}"
            f"{row.rps:>9.4f}"
        )

    print()
    print(f"    {'ALL':<5}{'1,520 matches':<13}{overall['n']:>5}"
          f"{overall['accuracy']:>9.4f}{overall['balanced_accuracy']:>9.4f}"
          f"{overall['macro_f1']:>9.4f}{overall['log_loss']:>9.4f}"
          f"{overall['brier_score']:>9.4f}{overall['rps']:>9.4f}")

    print()
    print("  Reading these honestly:")
    print()

    predicted = sorted(results["predicted_result"].unique())

    print(f"    The baseline only ever predicts {predicted} - the argmax of a")
    print("    fixed vector cannot vary. Accuracy therefore just measures how")
    print("    often the test season's most common class occurs.")
    print(f"    Balanced accuracy {overall['balanced_accuracy']:.4f} is the honest")
    print("    number: one class fully recalled, the other two never predicted.")
    print("    Log loss, Brier and RPS are the ones a real model must beat,")
    print("    because they score the whole distribution rather than the argmax.")
    print(f"    RPS {overall['rps']:.4f} is the ordered score: this baseline puts the")
    print("    same vector on every match, so it never misplaces mass towards the")
    print("    draw on purpose - it simply cannot know which way a match leans.")


def print_reference_points(overall, results):

    print()
    print("=" * 79)
    print("REFERENCE POINTS")
    print("=" * 79)
    print()

    uniform_log_loss = float(np.log(3.0))

    # Uniform log loss is ln(3) and uniform Brier is 2/3 whatever the outcomes
    # are. Uniform RPS is NOT constant: per match it is 5/18 for H or A and 1/9
    # for D, so the mean depends on the draw share of the test set. It is
    # therefore measured against the real outcomes, not asserted.
    uniform_rps = rps_score(
        encode_labels(results["actual_result"].to_numpy()),
        np.full((len(results), len(CLASSES)), 1.0 / len(CLASSES)),
    )

    print("  Two anchors this baseline sits between:")
    print()
    print(f"    uniform 1/3 guess     log loss {uniform_log_loss:.4f}   "
          f"brier {2.0 / 3.0:.4f}   rps {uniform_rps:.4f}")
    print(f"    this base rate        log loss {overall['log_loss']:.4f}   "
          f"brier {overall['brier_score']:.4f}   rps {overall['rps']:.4f}")
    print(f"    perfect prediction    log loss {0.0:.4f}   brier {0.0:.4f}   "
          f"rps {0.0:.4f}")
    print()
    print(f"  Knowing the league-wide H/D/A frequency is worth "
          f"{uniform_log_loss - overall['log_loss']:.4f} log loss and "
          f"{uniform_rps - overall['rps']:.4f} RPS")
    print("  over guessing uniformly. That is the entire value of this model.")


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 79)
    print("PHASE 2 - INSTRUMENT 1: BASE-RATE PROBABILITY BASELINE")
    print("=" * 79)
    print()
    print(f"  Folds      : {FOLDS_CSV.relative_to(PROJECT_ROOT)} (frozen)")
    print(f"  Matches    : {MATCHES_CSV.relative_to(PROJECT_ROOT)} (frozen)")
    print("  Metrics    : imported from phase0_evaluation_harness")
    print("  Model      : P(H), P(D), P(A) from training frequency, nothing else")
    print("  Class order: [P(H), P(D), P(A)]")
    print("  Not built  : Elo, Poisson, Dixon-Coles, LR, RF, XGBoost, tuning")

    before_state = frozen_state()

    spec = load_spec()
    matches = load_matches()

    audit = Audit()

    print()
    print(f"  {len(matches)} matches, {len(spec['folds'])} folds.")
    print("  Fitting four walk-forward base rates ...")

    results, summary = run_baseline(matches, spec)

    print(f"  {len(results)} test predictions produced.")

    print("  T1-T6   fold structure and temporal boundary ...")
    test_fold_structure(matches, spec, summary, audit)

    print("  T7-T8   probability estimation and perturbation control ...")
    test_probability_estimation(matches, spec, summary, audit)

    print("  T9-T15  prediction contract and coverage ...")
    test_prediction_contract(results, summary, audit)

    print("  T16     metric reproduction ...")
    test_metric_reproduction(results, summary, audit)

    # ---- overall metrics across every test match
    overall = evaluate(
        results["actual_result"].to_numpy(),
        results[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
    )

    # ---- outputs written before the determinism round-trip check
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        RESULTS_OUTPUT, index=False, encoding="utf-8", float_format=FLOAT_FORMAT
    )
    summary.to_csv(
        SUMMARY_OUTPUT, index=False, encoding="utf-8", float_format=FLOAT_FORMAT
    )

    print("  T17     determinism ...")
    test_determinism(matches, spec, results, summary, audit)

    print("  T18-T19 isolation and frozen-state guard ...")
    test_isolation(before_state, audit)

    audit_frame = audit.frame()
    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    # ---- reports
    print_test_table(audit)
    print_distributions(summary)
    print_metrics(summary, overall, results)
    print_reference_points(overall, results)

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {RESULTS_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(results)} test matches)")
    print(f"  {SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(summary)} folds)")
    print(f"  {AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(audit_frame)} entries)")

    failures = audit.failures()

    def outcome(prefix):
        rows = [r for r in audit.rows if r["test_id"].startswith(prefix)]
        return status_text(all(r["status"] != "FAIL" for r in rows))

    print()
    print("=" * 79)
    print("PHASE 2 - INSTRUMENT 1 STATUS")
    print("=" * 79)
    print()

    line("Folds:", f"{len(summary)}")
    line("Test matches:", f"{len(results)}")
    line("T1  four folds:", "4", outcome("T1"))
    line("T2  matches frozen spec:", "spec + CSV + baseline", outcome("T2"))
    line("T3  train sizes:", "380/760/1140/1520", outcome("T3"))
    line("T4  test sizes:", "380 each", outcome("T4"))
    line("T5  no train/test overlap:", "0 shared", outcome("T5"))
    line("T6  temporal boundary:", "train before test", outcome("T6"))
    line("T7  training-only estimate:", "recounted independently", outcome("T7"))
    line("T8  no test influence:", "3 forced refits", outcome("T8"))
    line("T9  one vector per fold:", "1 distinct each", outcome("T9"))
    line("T10 order [H,D,A]:", "verified against harness", outcome("T10"))
    line("T11 rows sum to 1:", "within 1e-6", outcome("T11"))
    line("T12 no NaN/inf:", "all finite, in [0,1]", outcome("T12"))
    line("T13 argmax prediction:", "no ties", outcome("T13"))
    line("T14 1,520 test matches:", f"{len(results)}", outcome("T14"))
    line("T15 each match once:", "0 duplicates", outcome("T15"))
    line("T16 metrics reproduce:", "from results CSV", outcome("T16"))
    line("T17 determinism:", "identical on rerun", outcome("T17"))
    line("T18 no data/raw read:", "file access recorded", outcome("T18"))
    line("T19 frozen phases intact:", f"{len(before_state)} hashes", outcome("T19"))
    line("T20 perturbation control:", "test rewrites ignored", outcome("T20"))

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
        print(f"{total_tests - len(failures)}/{total_tests} tests passed")
        print()
        print("STATUS: FAIL / INVESTIGATE")
        print()
        return EXIT_FAIL

    print(f"{total_tests}/{total_tests} tests passed")
    print()
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

    except ProbabilityError as error:
        print(f"\n  FATAL: probability contract violated: {error}\n")
        print("STATUS: FATAL\n")
        return EXIT_FATAL

    except Exception:
        print("\n  FATAL: unexpected exception\n")
        traceback.print_exc()
        print("\nSTATUS: FATAL\n")
        return EXIT_FATAL


if __name__ == "__main__":
    sys.exit(main())
