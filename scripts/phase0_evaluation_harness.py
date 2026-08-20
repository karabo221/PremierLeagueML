"""
PHASE 0 - INSTRUMENT 6: EVALUATION HARNESS

Defines and tests the evaluation framework every future Premier League model
must be judged by. It trains nothing, builds no dataset, and engineers no
features. It is the referee, not a player.

TIME-AWARE BY CONSTRUCTION

  Evaluation is out-of-time, season-based and expanding-window. There is no
  random split anywhere in this file, and the harness actively proves it: every
  fold's test set must be exactly one whole season, and building the folds twice
  must produce byte-identical results. A random split satisfies neither.

      Fold 1  train 2021-2022                                  test 2022-2023
      Fold 2  train 2021-2022 .. 2022-2023                     test 2023-2024
      Fold 3  train 2021-2022 .. 2023-2024                     test 2024-2025
      Fold 4  train 2021-2022 .. 2024-2025                     test 2025-2026

MODEL-INDEPENDENT

  Nothing here knows what a model is. A model is anything that can produce, for
  n test matches, an (n, 3) array of probabilities in the fixed column order
  [P(H), P(D), P(A)]. Logistic regression, Elo, Dixon-Coles, XGBoost and the
  eventual base-rate baseline all enter through that one door and are scored by
  the identical function on the identical folds.

WHAT IS TESTED HERE

  The harness tests ITSELF. Metric implementations are checked against cases
  with known closed-form answers, and the probability validator is fed
  deliberately malformed input to confirm it rejects what it must. A checker
  that has never been shown a failing case is not a checker.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_statistical_integrity import decode, parse_tables  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIXTURES_DIR = RAW_DIR / "Fixtures"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

FOLDS_PATH = OUTPUT_DIR / "phase0_evaluation_folds.csv"
SPEC_PATH = OUTPUT_DIR / "phase0_evaluation_spec.csv"
SPEC_JSON_PATH = OUTPUT_DIR / "phase0_evaluation_spec.json"

SCORE_SEPARATORS = ["–", "—", "-"]

# Fixed, ordered, and never to be reordered: every probability array in this
# project is indexed [P(H), P(D), P(A)].
CLASSES = ["H", "D", "A"]
CLASS_INDEX = {label: i for i, label in enumerate(CLASSES)}

PROBABILITY_TOLERANCE = 1e-6
LOG_LOSS_EPSILON = 1e-15

FOLD_FIELDS = [
    "fold",
    "train_seasons",
    "test_season",
    "train_matches",
    "test_matches",
    "max_train_date",
    "min_test_date",
    "temporal_order_valid",
    "overlap_valid",
]

SPEC_FIELDS = [
    "metric",
    "type",
    "direction",
    "purpose",
    "required_for_model",
]


# --------------------------------------------------------------------------
# loading  (validated fixtures only - Instruments 2 and 4 cleared these)
# --------------------------------------------------------------------------


def parse_score(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "":
        return None
    for separator in SCORE_SEPARATORS:
        if separator in text:
            parts = text.split(separator)
            if len(parts) != 2:
                return None
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                return None
    return None


def load_matches():
    """Every played match, chronologically ordered, with its H/D/A label."""
    rows = []
    files = sorted(
        [p for p in FIXTURES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".xls"],
        key=lambda p: p.name.lower(),
    )

    for path in files:
        raw = path.read_bytes()
        text, _ = decode(raw)
        tables, _, _ = parse_tables(text)
        if tables is None:
            continue
        table = max(tables, key=lambda t: t.shape[0] * t.shape[1])
        season = path.stem.split(" PL")[0].strip()

        for _, row in table.iterrows():
            if row.isna().all():
                continue
            home, away = row.get("Home"), row.get("Away")
            if pd.isna(home) or pd.isna(away):
                continue
            score = parse_score(row.get("Score"))
            if score is None:
                continue
            date = pd.to_datetime(row.get("Date"), errors="coerce")
            if pd.isna(date):
                continue

            home_goals, away_goals = score
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"

            rows.append({
                "season": season,
                "date": date.normalize(),
                "home": str(home).strip(),
                "away": str(away).strip(),
                "result": result,
            })

    matches = pd.DataFrame(rows)
    matches["match_id"] = (
        matches["season"] + "|" + matches["date"].dt.strftime("%Y-%m-%d")
        + "|" + matches["home"] + "|" + matches["away"]
    )
    return matches.sort_values(["date", "home", "away"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# fold construction  -  deterministic, season-aligned, expanding window
# --------------------------------------------------------------------------


def build_folds(matches):
    """Expanding-window out-of-time folds, one per season after the first.

    There is no shuffling, no sampling and no random_state, because there is no
    randomness: a fold is a pure function of the season ordering. Fold k trains
    on every season before test season k and tests on the whole of it.
    """
    seasons = sorted(matches["season"].unique())
    folds = []
    for position in range(1, len(seasons)):
        train_seasons = seasons[:position]
        test_season = seasons[position]
        folds.append({
            "fold": position,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "train_ids": tuple(
                matches.loc[matches["season"].isin(train_seasons), "match_id"]
            ),
            "test_ids": tuple(
                matches.loc[matches["season"] == test_season, "match_id"]
            ),
        })
    return folds, seasons


# --------------------------------------------------------------------------
# probability contract
# --------------------------------------------------------------------------


class ProbabilityError(ValueError):
    """Raised when a model's output violates the probability contract."""


def validate_probabilities(proba, n_expected):
    """Gate every model output must pass before it is scored.

    Checks, in order: shape, finiteness, lower bound, upper bound, row sums, and
    prediction count against the test fold. Raises rather than repairing - a
    harness that renormalises a broken model silently hides the break.
    """
    array = np.asarray(proba, dtype=float)

    if array.ndim != 2:
        raise ProbabilityError(
            "expected a 2-D array, got {} dimension(s)".format(array.ndim)
        )
    if array.shape[1] != len(CLASSES):
        raise ProbabilityError(
            "expected {} columns [P(H), P(D), P(A)], got {}".format(
                len(CLASSES), array.shape[1]
            )
        )
    if array.shape[0] != n_expected:
        raise ProbabilityError(
            "expected {} predictions to match the test fold, got {}".format(
                n_expected, array.shape[0]
            )
        )
    if not np.all(np.isfinite(array)):
        raise ProbabilityError("array contains NaN or infinite values")
    if np.any(array < 0.0):
        raise ProbabilityError(
            "negative probability at row(s) {}".format(
                np.unique(np.where(array < 0.0)[0])[:5].tolist()
            )
        )
    if np.any(array > 1.0):
        raise ProbabilityError(
            "probability above 1 at row(s) {}".format(
                np.unique(np.where(array > 1.0)[0])[:5].tolist()
            )
        )

    sums = array.sum(axis=1)
    off = np.where(np.abs(sums - 1.0) > PROBABILITY_TOLERANCE)[0]
    if off.size:
        raise ProbabilityError(
            "row probabilities must sum to 1 within {}; {} row(s) violate this, "
            "first at row {} summing to {:.6f}".format(
                PROBABILITY_TOLERANCE, off.size, int(off[0]), sums[off[0]]
            )
        )
    return array


# --------------------------------------------------------------------------
# metrics  -  implemented here so the harness carries no model dependency
# --------------------------------------------------------------------------


def encode_labels(labels):
    return np.array([CLASS_INDEX[str(label).strip()] for label in labels], dtype=int)


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def balanced_accuracy_score(y_true, y_pred):
    """Mean per-class recall. Classes absent from y_true are skipped."""
    recalls = []
    for index in range(len(CLASSES)):
        actual = y_true == index
        if not actual.any():
            continue
        recalls.append(np.mean(y_pred[actual] == index))
    return float(np.mean(recalls)) if recalls else float("nan")


def macro_f1_score(y_true, y_pred):
    scores = []
    for index in range(len(CLASSES)):
        true_positive = np.sum((y_pred == index) & (y_true == index))
        predicted = np.sum(y_pred == index)
        actual = np.sum(y_true == index)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def log_loss_score(y_true, proba):
    clipped = np.clip(proba, LOG_LOSS_EPSILON, 1.0)
    picked = clipped[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(picked)))


def brier_score(y_true, proba):
    """Multiclass Brier: mean squared error against the one-hot outcome.

    Ranges 0 (perfect) to 2 (confidently wrong on every match).
    """
    onehot = np.zeros_like(proba)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate(y_true_labels, proba):
    """The single scoring function. Every model and the baseline pass through it.

    Takes true H/D/A labels and an (n, 3) probability array; returns the metric
    set every model is required to report.
    """
    y_true = encode_labels(y_true_labels)
    array = validate_probabilities(proba, len(y_true))
    y_pred = np.argmax(array, axis=1)
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1_score(y_true, y_pred),
        "log_loss": log_loss_score(y_true, array),
        "brier_score": brier_score(y_true, array),
    }


# --------------------------------------------------------------------------
# the model interface  -  defined, deliberately not implemented
# --------------------------------------------------------------------------


MODEL_INTERFACE = {
    "contract": "predict_proba(test_matches) -> ndarray of shape (n, 3)",
    "column_order": ["P(H)", "P(D)", "P(A)"],
    "constraints": [
        "every value >= 0",
        "every value <= 1",
        "each row sums to 1 within {}".format(PROBABILITY_TOLERANCE),
        "exactly one row per test match, in test-fold order",
        "no finite-value violations (NaN, inf)",
    ],
    "scored_by": "evaluate(y_true_labels, proba)",
    "folds": "the four expanding-window folds in phase0_evaluation_folds.csv",
    "note": (
        "The base-rate baseline is NOT implemented here. When it is built it "
        "enters through this same interface and is scored by this same "
        "function on these same folds, so its numbers are directly comparable "
        "to any model. Defining it now and fitting it later is deliberate."
    ),
}


# --------------------------------------------------------------------------
# self-tests
# --------------------------------------------------------------------------


def run_metric_selftests():
    """Check the metric implementations against closed-form known answers."""
    results = []
    labels = ["H", "D", "A", "H", "D", "A"]
    y_true = encode_labels(labels)

    # 1. perfect, fully confident predictions
    perfect = np.zeros((len(labels), 3))
    perfect[np.arange(len(labels)), y_true] = 1.0
    scores = evaluate(labels, perfect)
    results.append(("perfect predictions -> accuracy 1.0",
                    abs(scores["accuracy"] - 1.0) < 1e-12, scores["accuracy"], 1.0))
    results.append(("perfect predictions -> macro F1 1.0",
                    abs(scores["macro_f1"] - 1.0) < 1e-12, scores["macro_f1"], 1.0))
    results.append(("perfect predictions -> Brier 0.0",
                    abs(scores["brier_score"]) < 1e-12, scores["brier_score"], 0.0))
    results.append(("perfect predictions -> log loss ~0",
                    scores["log_loss"] < 1e-9, scores["log_loss"], 0.0))

    # 2. uniform predictions have exact closed forms
    uniform = np.full((len(labels), 3), 1.0 / 3.0)
    scores = evaluate(labels, uniform)
    expected_log_loss = float(np.log(3.0))
    expected_brier = (2.0 / 3.0)
    results.append(("uniform -> log loss ln(3)",
                    abs(scores["log_loss"] - expected_log_loss) < 1e-9,
                    scores["log_loss"], expected_log_loss))
    results.append(("uniform -> Brier 2/3",
                    abs(scores["brier_score"] - expected_brier) < 1e-9,
                    scores["brier_score"], expected_brier))

    # 3. a constant rule scores 1/3 balanced accuracy on a balanced set
    constant = np.tile(np.array([0.8, 0.1, 0.1]), (len(labels), 1))
    scores = evaluate(labels, constant)
    results.append(("constant 'always H' -> balanced accuracy 1/3",
                    abs(scores["balanced_accuracy"] - 1.0 / 3.0) < 1e-12,
                    scores["balanced_accuracy"], 1.0 / 3.0))
    return results


def run_probability_selftests():
    """Feed the validator malformed input and confirm every case is rejected."""
    n = 4
    valid = np.full((n, 3), 1.0 / 3.0)

    cases = [
        ("well-formed array is accepted", valid, n, True),
        ("negative probability rejected",
         np.array([[-0.1, 0.6, 0.5]] * n), n, False),
        # no negative values here, so this genuinely exercises the upper bound
        # rather than being caught by the negative check first
        ("probability above 1 rejected",
         np.array([[1.4, 0.3, 0.3]] * n), n, False),
        ("rows summing to 0.9 rejected",
         np.array([[0.3, 0.3, 0.3]] * n), n, False),
        ("rows summing to 1.2 rejected",
         np.array([[0.4, 0.4, 0.4]] * n), n, False),
        ("wrong prediction count rejected", valid, n + 1, False),
        ("wrong column count rejected", np.full((n, 2), 0.5), n, False),
        ("NaN rejected",
         np.array([[np.nan, 0.5, 0.5]] * n), n, False),
        ("1-D array rejected", np.full(n, 1.0), n, False),
    ]

    results = []
    for name, array, expected_n, should_pass in cases:
        try:
            validate_probabilities(array, expected_n)
            accepted = True
            message = ""
        except ProbabilityError as exc:
            accepted = False
            message = str(exc)
        results.append((name, accepted == should_pass, accepted, message))
    return results


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 6: EVALUATION HARNESS")
    print("=" * 78)
    print("Project root : {}".format(PROJECT_ROOT))
    print("Mode         : READ-ONLY; no dataset, no features, no model training")
    print("Split        : chronological, season-based, expanding window")
    print("Random split : NOT USED - no shuffle, no sampling, no random_state")
    print("Classes      : {} (fixed column order for every probability array)".format(
        ", ".join(CLASSES)))
    print()

    matches = load_matches()
    if not len(matches):
        print("FATAL: no fixtures loaded")
        return 1

    folds, seasons = build_folds(matches)

    print("  matches loaded : {}".format(len(matches)))
    print("  seasons        : {}".format(", ".join(seasons)))
    print("  folds defined  : {}".format(len(folds)))
    print()

    failures = []

    # ------------------------------------------------------------------
    # fold validation
    # ------------------------------------------------------------------

    print("=" * 78)
    print("FOLD VALIDATION")
    print("=" * 78)
    print()

    fold_rows = []
    for fold in folds:
        train_ids = set(fold["train_ids"])
        test_ids = set(fold["test_ids"])

        train = matches[matches["match_id"].isin(train_ids)]
        test = matches[matches["match_id"].isin(test_ids)]

        max_train_date = train["date"].max()
        min_test_date = test["date"].min()

        temporal_ok = bool(max_train_date < min_test_date)
        overlap = train_ids & test_ids
        overlap_ok = len(overlap) == 0

        # a test season must be whole - a random subset would fail this
        season_total = int((matches["season"] == fold["test_season"]).sum())
        whole_season = len(test_ids) == season_total

        # no training row may carry a test season's label
        no_test_season_in_train = fold["test_season"] not in set(train["season"])

        print("  FOLD {}".format(fold["fold"]))
        print("    train seasons : {}".format(", ".join(fold["train_seasons"])))
        print("    test season   : {}".format(fold["test_season"]))
        print("    train matches : {:>4}".format(len(train)))
        print("    test matches  : {:>4}".format(len(test)))
        print("    max train date: {}".format(str(max_train_date.date())))
        print("    min test date : {}".format(str(min_test_date.date())))
        print("    gap           : {} days".format((min_test_date - max_train_date).days))
        print("    [{}] every training date is earlier than every test date".format(
            "OK" if temporal_ok else "FAIL"))
        print("    [{}] no test match appears in training ({} shared)".format(
            "OK" if overlap_ok else "FAIL", len(overlap)))
        print("    [{}] test season is untouched by training".format(
            "OK" if no_test_season_in_train else "FAIL"))
        print("    [{}] test set is a WHOLE season ({}/{}) - not a random subset".format(
            "OK" if whole_season else "FAIL", len(test_ids), season_total))
        print()

        if not temporal_ok:
            failures.append("fold {}: training dates reach into the test period".format(fold["fold"]))
        if not overlap_ok:
            failures.append("fold {}: {} test matches appear in training".format(
                fold["fold"], len(overlap)))
        if not no_test_season_in_train:
            failures.append("fold {}: test season present in training".format(fold["fold"]))
        if not whole_season:
            failures.append("fold {}: test set is not a whole season".format(fold["fold"]))

        fold_rows.append({
            "fold": fold["fold"],
            "train_seasons": " + ".join(fold["train_seasons"]),
            "test_season": fold["test_season"],
            "train_matches": len(train),
            "test_matches": len(test),
            "max_train_date": str(max_train_date.date()),
            "min_test_date": str(min_test_date.date()),
            "temporal_order_valid": temporal_ok,
            "overlap_valid": overlap_ok,
        })

    # ------------------------------------------------------------------
    # partition and determinism
    # ------------------------------------------------------------------

    print("=" * 78)
    print("PARTITION AND DETERMINISM")
    print("=" * 78)

    tested_counts = {}
    for fold in folds:
        for match_id in fold["test_ids"]:
            tested_counts[match_id] = tested_counts.get(match_id, 0) + 1

    multiply_tested = [m for m, c in tested_counts.items() if c > 1]
    never_tested = set(matches["match_id"]) - set(tested_counts)

    print("  Matches tested exactly once : {}".format(
        sum(1 for c in tested_counts.values() if c == 1)))
    print("  Matches tested more than once: {}".format(len(multiply_tested)))
    print("  Matches never in a test fold : {} (all of {}, the first season -".format(
        len(never_tested), seasons[0]))
    print("                                 it can only ever be training)")

    partition_ok = len(multiply_tested) == 0
    first_season_only = {
        matches.loc[matches["match_id"] == m, "season"].iloc[0] for m in never_tested
    } if never_tested else set()
    coverage_ok = first_season_only in ({seasons[0]}, set())

    print("  [{}] every match is a test match in at most one fold".format(
        "OK" if partition_ok else "FAIL"))
    print("  [{}] the only never-tested matches belong to {}".format(
        "OK" if coverage_ok else "FAIL", seasons[0]))

    if not partition_ok:
        failures.append("a match is tested in more than one fold")
    if not coverage_ok:
        failures.append("matches outside the first season are never tested")

    rebuilt, _ = build_folds(matches)
    deterministic = all(
        a["train_ids"] == b["train_ids"] and a["test_ids"] == b["test_ids"]
        and a["fold"] == b["fold"] and a["test_season"] == b["test_season"]
        for a, b in zip(folds, rebuilt)
    ) and len(folds) == len(rebuilt)

    print("  [{}] rebuilding the folds reproduces them exactly".format(
        "OK" if deterministic else "FAIL"))
    print("       (a random split could not survive this test)")
    if not deterministic:
        failures.append("fold construction is not deterministic")
    print()

    # ------------------------------------------------------------------
    # target definition
    # ------------------------------------------------------------------

    print("=" * 78)
    print("PRIMARY TARGET")
    print("=" * 78)
    print("  Result: H = home win, D = draw, A = away win")
    print("  Probability arrays are indexed [P(H), P(D), P(A)] - fixed, never reordered.")
    print()
    print("  Observed class distribution (label frequency only - no model involved):")
    overall_counts = matches["result"].value_counts()
    for label in CLASSES:
        count = int(overall_counts.get(label, 0))
        print("    {}  {:>5}  {:>6.2%}".format(label, count, count / len(matches)))
    print()
    print("  Per test season:")
    for fold in folds:
        season_matches = matches[matches["season"] == fold["test_season"]]
        counts = season_matches["result"].value_counts()
        parts = " ".join(
            "{} {:>5.1%}".format(label, counts.get(label, 0) / len(season_matches))
            for label in CLASSES
        )
        print("    {:<12} {}".format(fold["test_season"], parts))
    print()

    # ------------------------------------------------------------------
    # why accuracy alone is insufficient
    # ------------------------------------------------------------------

    print("=" * 78)
    print("WHY ACCURACY ALONE IS INSUFFICIENT")
    print("=" * 78)
    print("  Arithmetic on the observed label distribution. No model is fitted,")
    print("  and this is NOT the baseline being evaluated - it is the reason the")
    print("  metric set has five entries instead of one.")
    print()

    home_rate = float((matches["result"] == "H").mean())
    print("  A rule that always says 'home win' would score:")
    print("    accuracy          {:>6.2%}   <- looks respectable".format(home_rate))
    print("    balanced accuracy {:>6.2%}   <- recall is 100% on H, 0% on D and A".format(1 / 3))
    print("    macro F1          low       <- two of three classes score zero")
    print("    log loss          unbounded <- a confident wrong call is punished")
    print("    Brier score       poor      <- squared error against the outcome")
    print()
    print("  Accuracy rewards collapsing onto the majority class. Draws are the")
    print("  hardest and least frequent outcome, and a model that never predicts one")
    print("  can still look accurate. Balanced accuracy and macro F1 expose that;")
    print("  log loss and Brier score judge the PROBABILITIES rather than the")
    print("  argmax, which is what a forecast is actually for. RPS belongs here too")
    print("  once ordered outcomes matter - noted in the spec, not yet required.")
    print()

    # ------------------------------------------------------------------
    # self-tests
    # ------------------------------------------------------------------

    print("=" * 78)
    print("HARNESS SELF-TESTS: METRICS")
    print("=" * 78)
    print("  Closed-form cases. If these drift, every number the project ever")
    print("  reports is wrong, so they are checked on each run.")
    print()
    for name, ok, observed, expected in run_metric_selftests():
        print("  [{}] {:<48} got {:.6f} want {:.6f}".format(
            "OK" if ok else "FAIL", name, observed, expected))
        if not ok:
            failures.append("metric self-test failed: {}".format(name))
    print()

    print("=" * 78)
    print("HARNESS SELF-TESTS: PROBABILITY CONTRACT")
    print("=" * 78)
    print("  Deliberately malformed inputs. Each must be REJECTED; a validator")
    print("  never shown a failing case is not a validator.")
    print()
    for name, ok, accepted, message in run_probability_selftests():
        print("  [{}] {:<42} {}".format(
            "OK" if ok else "FAIL", name, "accepted" if accepted else "rejected"))
        if message and not accepted:
            print("           reason: {}".format(message[:88]))
        if not ok:
            failures.append("probability self-test failed: {}".format(name))
    print()

    # ------------------------------------------------------------------
    # baseline interface (defined, not trained)
    # ------------------------------------------------------------------

    print("=" * 78)
    print("MODEL AND BASELINE INTERFACE (defined, deliberately not implemented)")
    print("=" * 78)
    print("  contract     : {}".format(MODEL_INTERFACE["contract"]))
    print("  column order : {}".format(", ".join(MODEL_INTERFACE["column_order"])))
    print("  scored by    : {}".format(MODEL_INTERFACE["scored_by"]))
    print("  constraints  :")
    for constraint in MODEL_INTERFACE["constraints"]:
        print("      - {}".format(constraint))
    print()
    print("  {}".format(MODEL_INTERFACE["note"]))
    print()
    print("  No model was trained. No baseline was fitted. No probabilities were")
    print("  produced from data - the only arrays scored above are synthetic")
    print("  fixtures used to test the harness against itself.")
    print()

    # ------------------------------------------------------------------
    # write specs
    # ------------------------------------------------------------------

    metrics_spec = [
        {
            "metric": "Accuracy",
            "type": "classification (argmax)",
            "direction": "higher is better",
            "purpose": "share of matches whose predicted class is correct",
            "required_for_model": "YES - but never reported alone",
        },
        {
            "metric": "Balanced Accuracy",
            "type": "classification (argmax)",
            "direction": "higher is better",
            "purpose": "mean per-class recall; exposes a model that ignores draws",
            "required_for_model": "YES",
        },
        {
            "metric": "Macro F1",
            "type": "classification (argmax)",
            "direction": "higher is better",
            "purpose": "unweighted mean F1 over H, D, A; every class counts equally",
            "required_for_model": "YES",
        },
        {
            "metric": "Log Loss",
            "type": "probabilistic",
            "direction": "lower is better",
            "purpose": "penalises confident wrong probabilities; the primary "
                       "probabilistic score",
            "required_for_model": "YES",
        },
        {
            "metric": "Brier Score",
            "type": "probabilistic",
            "direction": "lower is better",
            "purpose": "multiclass squared error against the one-hot outcome; "
                       "0 perfect, 2 worst",
            "required_for_model": "YES",
        },
        {
            "metric": "RPS",
            "type": "probabilistic (ordered)",
            "direction": "lower is better",
            "purpose": "ranked probability score; respects H > D > A ordering and "
                       "is the standard in football forecasting",
            "required_for_model": "NOT YET - named in Direction_2.txt; add when "
                                  "ordered outcomes are modelled",
        },
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    folds_frame = pd.DataFrame(fold_rows, columns=FOLD_FIELDS)
    spec_frame = pd.DataFrame(metrics_spec, columns=SPEC_FIELDS)
    folds_frame.to_csv(FOLDS_PATH, index=False, encoding="utf-8")
    spec_frame.to_csv(SPEC_PATH, index=False, encoding="utf-8")

    machine_spec = {
        "instrument": "phase0_evaluation_harness",
        "evaluation_type": "out-of-time, season-based, expanding window",
        "random_split_used": False,
        "classes": CLASSES,
        "probability_column_order": ["P(H)", "P(D)", "P(A)"],
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "seasons": seasons,
        "folds": [
            {
                "fold": row["fold"],
                "train_seasons": row["train_seasons"].split(" + "),
                "test_season": row["test_season"],
                "train_matches": row["train_matches"],
                "test_matches": row["test_matches"],
                "max_train_date": row["max_train_date"],
                "min_test_date": row["min_test_date"],
            }
            for row in fold_rows
        ],
        "metrics": metrics_spec,
        "model_interface": MODEL_INTERFACE,
    }
    SPEC_JSON_PATH.write_text(
        json.dumps(machine_spec, indent=2), encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    print("=" * 78)
    print("FOLD SUMMARY")
    print("=" * 78)
    header = "  {:<6}{:<48}{:<14}{:>8}{:>8}".format(
        "fold", "train seasons", "test season", "train", "test")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in fold_rows:
        print("  {:<6}{:<48}{:<14}{:>8}{:>8}".format(
            row["fold"], row["train_seasons"], row["test_season"],
            row["train_matches"], row["test_matches"]))
    print()

    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 6 STATUS")
    print("=" * 78)
    print()
    if not failures:
        print("  PASS")
        print()
        print("  The chronological evaluation framework is valid and executable.")
        print("  {} out-of-time folds, {} test matches in total, no fold leaking a".format(
            len(folds), sum(r["test_matches"] for r in fold_rows)))
        print("  single day of the future into its training window. The metric")
        print("  implementations agree with their closed forms, and the probability")
        print("  contract rejects every malformed case it was shown.")
    else:
        print("  FAIL / INVESTIGATE")
        print()
        for failure in failures:
            print("    - {}".format(failure))
    print()
    print("  Reports written:")
    print("    {}".format(FOLDS_PATH))
    print("    {}".format(SPEC_PATH))
    print("    {}".format(SPEC_JSON_PATH))
    print()
    print("No source data was modified.")
    print("No ML dataset was built.")
    print("No feature engineering was performed.")
    print("No model was trained.")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
