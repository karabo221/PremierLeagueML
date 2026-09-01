"""
===============================================================================
PHASE 2 - INSTRUMENT 2
ELO BASELINE  (Elo v1)
===============================================================================

THE QUESTION
    The base-rate baseline knows only that the league is roughly 44/23/32
    H/D/A. Arsenal v Bournemouth and Arsenal v Manchester City get the same
    vector. Elo introduces the first piece of football intelligence:

        some teams are stronger than others

    and nothing else. So this instrument answers exactly one question:

        how much predictive power comes from historical match results,
        a home advantage, and a single team rating - and no more?

    The benchmark to beat, from Instrument 1:

        log loss 1.0689     brier 0.6467     accuracy 44.47%

    A result that fails to beat it is not a failed instrument. It is the
    finding that team ratings in this simple form carry less information
    than league-wide outcome frequency.

THE LOCKED SPECIFICATION - fixed BEFORE any test season is touched
    initial rating       1500
    K-factor             20        fixed, not tuned
    home advantage       60 Elo    fixed, not tuned
    season regression    0.75      fixed, not tuned
    draw parameter nu    the ONLY calibrated quantity, fitted per fold on
                         that fold's TRAINING matches alone

    No goal-margin adjustment, no xG, no player data, no FBref aggregate, no
    engineered feature, no machine learning. Deliberately.

    Tuning K, the home advantage or the regression factor would make this a
    tuning exercise before we know whether Elo is useful at all - which is
    precisely the over-engineered "baseline" worth avoiding.

SAME-DAY MATCHES FORBID A NAIVE SEQUENTIAL WALK
    Phase 0 locked the rule: for a match at date T, only matches with date
    STRICTLY BEFORE T are available. Phase 1 measured that 1,706 of the 1,900
    matches share a date with another match.

    A textbook match-by-match Elo walk would therefore leak: the 15:00 result
    would feed the 17:30 prediction on the same afternoon, which the project
    has ruled unknowable.

    So ratings advance in DATE BATCHES. Every match on a date is predicted
    from the ratings standing before that date; only once the whole date is
    predicted are its updates applied. This is the single most important
    implementation detail in the file.

THE RATING TRAJECTORY IS FOLD-INDEPENDENT
    Ratings depend only on results and the three fixed constants. They do not
    depend on nu. So there is ONE Elo walk over all 1,900 matches, and the
    folds differ only in the draw parameter fitted to their training data.

    That is why the results table carries 1,900 rows - a pre-match rating
    pair for every match - while only the 1,520 test rows are ever scored.

THREE-WAY PROBABILITIES - THE DAVIDSON MODEL
    Elo natively yields an expected score, not three outcomes. The mapping is
    stated explicitly rather than invented casually:

        d      = R_home + home_advantage - R_away
        theta  = 10 ^ (d / 400)
        P(H)   = theta            / (theta + 1 + nu * sqrt(theta))
        P(D)   = nu * sqrt(theta) / (theta + 1 + nu * sqrt(theta))
        P(A)   = 1                / (theta + 1 + nu * sqrt(theta))

    One parameter, nu >= 0, controlling draw mass. It is symmetric, sums to 1
    by construction, and collapses to standard two-outcome Elo at nu = 0.

    NOTE, stated rather than hidden: the rating UPDATE uses the standard Elo
    expected score theta/(theta+1), while the PREDICTION uses Davidson. The
    two agree only when theta = 1. This is a deliberate v1 simplification -
    the update rule stays textbook Elo so the rating system remains the
    familiar one, and the draw model sits on top of it.

    nu is fitted by minimising log loss on TRAINING matches only, by a
    deterministic bracketed golden-section search. Never on a test season.

PROMOTED AND RETURNING TEAMS
    Phase 1 Instrument 4 established that "new to the season" and "promoted"
    are different claims, and which teams are which. That work is used here
    rather than re-derived:

        continuing               regress toward the mean:
                                 1500 + (rating - 1500) * 0.75
        everything else          1500

    "Everything else" is returning_after_absence, new_to_dataset, and the
    2021-22 baseline cohort. All three genuinely lack a season N-1 rating.
    The transition metadata is carried into the output, not erased.

INPUTS - frozen, read-only
    outputs/phase0_evaluation_spec.json          the frozen folds
    outputs/phase0_evaluation_folds.csv          cross-check
    outputs/phase1_matches.csv                   match results
    outputs/phase1_team_transition_summary.csv   season transition policy
    outputs/phase2_base_rate_fold_summary.csv    the benchmark to beat

    The 86 engineered Phase 1 features are deliberately NOT used. Elo must
    earn its rating from sequential results alone.

EXIT CODES
    0 PASS   2 FAIL/INVESTIGATE   1 FATAL

NOT BUILT HERE
    no Poisson, no Dixon-Coles, no XGBoost, no xG, no feature-based ML, no
    random splitting, no change to the four folds, and no optimisation
    against a test season.
===============================================================================
"""

from pathlib import Path
import hashlib
import json
import math
import sys
import traceback

import numpy as np
import pandas as pd


# ============================================================
# FILE-ACCESS RECORDER
# ============================================================

# Every file open is recorded WITH THE PURPOSE it was opened for. That
# distinction is what makes P3 meaningful: the frozen-state guard opens every
# Phase 1 output in order to hash it, including the engineered-feature file.
# Hashing a file for integrity is not reading it as a model input, and only a
# labelled record can tell the two apart.

_OPENED_PATHS = []
_ACCESS_CONTEXT = ["input"]


def _record_open(event, args):

    if event != "open":
        return

    target = args[0]

    if isinstance(target, (str, bytes, Path)):
        _OPENED_PATHS.append((str(target), _ACCESS_CONTEXT[-1]))


sys.addaudithook(_record_open)


class access_context:
    """Label every file open inside this block with a purpose."""

    def __init__(self, label):
        self.label = label

    def __enter__(self):
        _ACCESS_CONTEXT.append(self.label)
        return self

    def __exit__(self, *exc):
        _ACCESS_CONTEXT.pop()
        return False


def opened_paths(context=None):

    resolved = []

    for path, label in _OPENED_PATHS:

        if context is not None and label != context:
            continue

        try:
            resolved.append(Path(path).resolve())
        except (OSError, ValueError):
            continue

    return resolved


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

sys.path.insert(0, str(SCRIPTS_DIR))

from phase0_evaluation_harness import (  # noqa: E402
    CLASSES,
    CLASS_INDEX,
    PROBABILITY_TOLERANCE,
    ProbabilityError,
    evaluate,
    validate_probabilities,
)


# ============================================================
# THE LOCKED SPECIFICATION
# ============================================================
#
# Fixed constants. Changing any of these turns Elo v1 into a tuning exercise,
# which is the thing this instrument exists to avoid.

INITIAL_RATING = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 60.0
SEASON_REGRESSION = 0.75
ELO_SCALE = 400.0

# The one calibrated quantity. Bounds and search settings are fixed so the
# fit is fully deterministic.
NU_LOWER = 1e-4
NU_UPPER = 10.0
NU_GRID_POINTS = 60
NU_GOLDEN_ITERATIONS = 200

TRANSITION_CONTINUING = "continuing"


# ============================================================
# CONFIGURATION
# ============================================================

SPEC_JSON = OUTPUTS_DIR / "phase0_evaluation_spec.json"
FOLDS_CSV = OUTPUTS_DIR / "phase0_evaluation_folds.csv"
MATCHES_CSV = OUTPUTS_DIR / "phase1_matches.csv"
TRANSITIONS_CSV = OUTPUTS_DIR / "phase1_team_transition_summary.csv"
BASE_RATE_CSV = OUTPUTS_DIR / "phase2_base_rate_fold_summary.csv"

RESULTS_OUTPUT = OUTPUTS_DIR / "phase2_elo_results.csv"
SUMMARY_OUTPUT = OUTPUTS_DIR / "phase2_elo_fold_summary.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase2_elo_audit.csv"

DECLARED_INPUTS = {
    SPEC_JSON.resolve(), FOLDS_CSV.resolve(), MATCHES_CSV.resolve(),
    TRANSITIONS_CSV.resolve(), BASE_RATE_CSV.resolve(),
}

OWN_OUTPUTS = {
    RESULTS_OUTPUT.resolve(), SUMMARY_OUTPUT.resolve(), AUDIT_OUTPUT.resolve(),
}

EXPECTED_FOLDS = 4
EXPECTED_TRAIN_SIZES = [380, 760, 1140, 1520]
EXPECTED_TEST_SIZE = 380
EXPECTED_TOTAL_TEST = 1520
EXPECTED_TOTAL_MATCHES = 1900
EXPECTED_TEAMS_PER_SEASON = 20

BASE_RATE_LOG_LOSS = 1.0689
BASE_RATE_BRIER = 0.6467

EXIT_PASS = 0
EXIT_FATAL = 1
EXIT_FAIL = 2

FLOAT_FORMAT = "%.17g"
FLOAT_PRECISION = "round_trip"

RESULTS_COLUMNS = [
    "fold", "role", "evaluated", "season", "test_season", "date",
    "home", "away",
    "home_elo_before", "away_elo_before", "elo_diff",
    "home_transition", "away_transition",
    "p_home", "p_draw", "p_away",
    "actual_result", "predicted_result",
    "home_elo_after", "away_elo_after",
]

SUMMARY_COLUMNS = [
    "fold", "train_seasons", "test_season", "train_matches", "test_matches",
    "k_factor", "home_advantage", "season_regression", "initial_rating",
    "nu_calibrated", "train_log_loss",
    "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score",
    "base_rate_log_loss", "base_rate_brier",
    "log_loss_delta", "brier_delta", "beats_base_rate",
]

METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score",
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
            "test_id": test_id, "test": test, "scope": scope,
            "expected": expected, "observed": observed,
            "status": "PASS" if passed else "FAIL", "detail": detail,
        })

        return bool(passed)

    def measure(self, test_id, test, scope, observed, detail=""):

        self.rows.append({
            "test_id": test_id, "test": test, "scope": scope,
            "expected": "(measurement)", "observed": observed,
            "status": "MEASURED", "detail": detail,
        })

    def failures(self):
        return [row for row in self.rows if row["status"] == "FAIL"]

    def all_passed(self):
        return not self.failures()

    def frame(self):
        return pd.DataFrame(self.rows, columns=[
            "test_id", "test", "scope", "expected", "observed",
            "status", "detail",
        ])


# ============================================================
# FROZEN-STATE GUARD
# ============================================================

def hash_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def frozen_state():

    tracked = {}

    with access_context("hash_guard"):
        return _frozen_state_scan(tracked)


def _frozen_state_scan(tracked):

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

def load_inputs():

    for required in (SPEC_JSON, FOLDS_CSV, MATCHES_CSV, TRANSITIONS_CSV,
                     BASE_RATE_CSV):
        if not required.exists():
            raise FatalError(f"missing required input: {required}")

    try:
        spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
        matches = pd.read_csv(MATCHES_CSV, float_precision=FLOAT_PRECISION)
        transitions = pd.read_csv(TRANSITIONS_CSV, float_precision=FLOAT_PRECISION)
        base_rate = pd.read_csv(BASE_RATE_CSV, float_precision=FLOAT_PRECISION)
    except Exception as error:
        raise FatalError(f"input could not be parsed: {error}") from error

    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")

    matches = matches.sort_values(
        ["season", "date", "home_team", "away_team"]
    ).reset_index(drop=True)

    matches["match_id"] = matches.index

    if len(matches) != EXPECTED_TOTAL_MATCHES:
        raise FatalError(
            f"foundation has {len(matches)} matches, expected "
            f"{EXPECTED_TOTAL_MATCHES}"
        )

    unknown = set(matches["result"]) - set(CLASSES)

    if unknown:
        raise FatalError(f"unexpected result labels: {sorted(unknown)}")

    return spec, matches, transitions, base_rate


# ============================================================
# THE ELO MODEL
# ============================================================

def expected_home_score(rating_difference):
    """Standard Elo expectation, used by the UPDATE rule."""

    return 1.0 / (1.0 + 10.0 ** (-rating_difference / ELO_SCALE))


def davidson_probabilities(rating_difference, nu):
    """
    Three-outcome mapping. Returns [P(H), P(D), P(A)] in the frozen order.

    Vectorised over rating_difference; nu is a scalar.
    """

    theta = np.power(10.0, np.asarray(rating_difference, dtype=float) / ELO_SCALE)

    root = np.sqrt(theta)

    denominator = theta + 1.0 + nu * root

    return np.stack([
        theta / denominator,
        (nu * root) / denominator,
        1.0 / denominator,
    ], axis=1)


def actual_home_score(result):

    return {"H": 1.0, "D": 0.5, "A": 0.0}[result]


def season_start_ratings(season, teams, transition_lookup, carried):
    """
    Apply the locked initialisation policy at a season boundary.

    continuing  -> regress the carried rating toward the league mean
    otherwise   -> the league-average starting rating

    Returns (ratings, applied) where `applied` records what happened to each
    team so the audit can verify the policy rather than trust it.
    """

    ratings = {}
    applied = []

    for team in teams:

        transition = transition_lookup.get((season, team))

        if transition is None:
            raise FatalError(
                f"no transition record for {season} {team} - Phase 1 "
                f"Instrument 4 should cover every team-season"
            )

        if transition == TRANSITION_CONTINUING:

            previous = carried.get(team)

            if previous is None:
                raise FatalError(
                    f"{season} {team} is 'continuing' but carries no rating"
                )

            rating = INITIAL_RATING + (previous - INITIAL_RATING) * SEASON_REGRESSION
            policy = "regressed_from_previous_season"

        else:
            previous = None
            rating = INITIAL_RATING
            policy = "reset_to_initial_rating"

        ratings[team] = rating

        applied.append({
            "season": season,
            "team": team,
            "transition": transition,
            "policy": policy,
            "rating_before_regression": previous,
            "rating_at_season_start": rating,
        })

    return ratings, applied


def run_elo_walk(matches, transition_lookup):
    """
    One sequential pass over all 1,900 matches, in date order.

    THE CRITICAL DETAIL: ratings advance in DATE BATCHES. Every match on a
    date is predicted from the ratings standing before that date, and only
    once the whole date has been read are its updates applied. A naive
    match-by-match walk would let a 15:00 result reach a 17:30 prediction on
    the same day, which Phase 0 ruled unknowable.

    Returns (walk, initialisation) where `walk` carries a pre-match rating
    pair for every match. It does NOT depend on nu, so it is computed once
    and shared by all four folds.
    """

    rows = []
    initialisation = []

    carried = {}

    for season in sorted(matches["season"].unique()):

        season_matches = matches[matches["season"] == season]

        teams = sorted(
            set(season_matches["home_team"]) | set(season_matches["away_team"])
        )

        ratings, applied = season_start_ratings(
            season, teams, transition_lookup, carried
        )

        initialisation.extend(applied)

        for date in sorted(season_matches["date"].unique()):

            batch = season_matches[season_matches["date"] == date]

            # ---- predict the whole date from the ratings standing before it
            pending = []

            for row in batch.itertuples():

                home_before = ratings[row.home_team]
                away_before = ratings[row.away_team]

                difference = home_before + HOME_ADVANTAGE - away_before

                expected = expected_home_score(difference)

                delta = K_FACTOR * (actual_home_score(row.result) - expected)

                pending.append({
                    "match_id": row.match_id,
                    "season": season,
                    "date": row.date,
                    "home": row.home_team,
                    "away": row.away_team,
                    "actual_result": row.result,
                    "home_elo_before": home_before,
                    "away_elo_before": away_before,
                    "elo_diff": difference,
                    "expected_home_score": expected,
                    "delta": delta,
                    "home_elo_after": home_before + delta,
                    "away_elo_after": away_before - delta,
                    "home_transition": transition_lookup[(season, row.home_team)],
                    "away_transition": transition_lookup[(season, row.away_team)],
                })

            # ---- only now apply the whole date's updates
            for entry in pending:
                ratings[entry["home"]] += entry["delta"]
                ratings[entry["away"]] -= entry["delta"]

            rows.extend(pending)

        carried = dict(ratings)

    walk = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)

    return walk, pd.DataFrame(initialisation)


# ============================================================
# CALIBRATION OF THE DRAW PARAMETER
# ============================================================

def training_log_loss(differences, outcomes, nu):
    """Mean negative log likelihood of the observed outcomes under nu."""

    proba = davidson_probabilities(differences, nu)

    picked = proba[np.arange(len(outcomes)), outcomes]

    return float(-np.mean(np.log(np.clip(picked, 1e-15, 1.0))))


def calibrate_nu(differences, outcomes):
    """
    Fit nu on TRAINING data only, by a deterministic bracketed search.

    A coarse grid brackets the minimum, then golden-section refines it. Fixed
    bounds and a fixed iteration count mean the result is reproducible to the
    bit - no random restarts, no solver dependency.
    """

    grid = np.linspace(NU_LOWER, NU_UPPER, NU_GRID_POINTS)

    losses = [training_log_loss(differences, outcomes, value) for value in grid]

    best = int(np.argmin(losses))

    low = grid[max(best - 1, 0)]
    high = grid[min(best + 1, len(grid) - 1)]

    invphi = (math.sqrt(5.0) - 1.0) / 2.0

    a, b = low, high

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)

    fc = training_log_loss(differences, outcomes, c)
    fd = training_log_loss(differences, outcomes, d)

    for _ in range(NU_GOLDEN_ITERATIONS):

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = training_log_loss(differences, outcomes, c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = training_log_loss(differences, outcomes, d)

    nu = (a + b) / 2.0

    return float(nu), training_log_loss(differences, outcomes, nu)


# ============================================================
# FOLD EXECUTION
# ============================================================

def run_folds(walk, spec, base_rate):
    """
    Calibrate per fold on training matches, then predict that fold's test
    season. The Elo walk itself is shared and unchanged.
    """

    outcome_index = walk["actual_result"].map(CLASS_INDEX).to_numpy()

    base_rate_by_fold = {
        int(row.fold): row for row in base_rate.itertuples()
    }

    results = []
    summaries = []

    test_season_to_fold = {}

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        train_seasons = list(fold_spec["train_seasons"])
        test_season = str(fold_spec["test_season"])

        test_season_to_fold[test_season] = fold

        train_mask = walk["season"].isin(train_seasons).to_numpy()
        test_mask = (walk["season"] == test_season).to_numpy()

        # ---- fit nu on TRAINING matches only
        nu, train_loss = calibrate_nu(
            walk.loc[train_mask, "elo_diff"].to_numpy(),
            outcome_index[train_mask],
        )

        # ---- freeze, then predict the test season
        test = walk[test_mask]

        proba = davidson_probabilities(test["elo_diff"].to_numpy(), nu)

        proba = validate_probabilities(proba, len(test))

        predicted = [CLASSES[i] for i in np.argmax(proba, axis=1)]

        fold_results = test.copy()
        fold_results["fold"] = fold
        fold_results["role"] = "test"
        fold_results["evaluated"] = 1
        fold_results["test_season"] = test_season
        fold_results["p_home"] = proba[:, 0]
        fold_results["p_draw"] = proba[:, 1]
        fold_results["p_away"] = proba[:, 2]
        fold_results["predicted_result"] = predicted

        results.append(fold_results)

        scores = evaluate(test["actual_result"].to_numpy(), proba)

        reference = base_rate_by_fold.get(fold)

        base_log_loss = float(reference.log_loss) if reference is not None else np.nan
        base_brier = float(reference.brier_score) if reference is not None else np.nan

        summaries.append({
            "fold": fold,
            "train_seasons": " + ".join(train_seasons),
            "test_season": test_season,
            "train_matches": int(train_mask.sum()),
            "test_matches": int(test_mask.sum()),
            "k_factor": K_FACTOR,
            "home_advantage": HOME_ADVANTAGE,
            "season_regression": SEASON_REGRESSION,
            "initial_rating": INITIAL_RATING,
            "nu_calibrated": nu,
            "train_log_loss": train_loss,
            "accuracy": scores["accuracy"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "macro_f1": scores["macro_f1"],
            "log_loss": scores["log_loss"],
            "brier_score": scores["brier_score"],
            "base_rate_log_loss": base_log_loss,
            "base_rate_brier": base_brier,
            "log_loss_delta": scores["log_loss"] - base_log_loss,
            "brier_delta": scores["brier_score"] - base_brier,
            "beats_base_rate": bool(
                scores["log_loss"] < base_log_loss
                and scores["brier_score"] < base_brier
            ),
        })

    # ---- the 380 matches of the first season are never a test set.
    # They are carried so the results table holds a pre-match rating pair for
    # every one of the 1,900 matches, flagged so they can never be scored.
    never_tested = walk[~walk["season"].isin(test_season_to_fold)].copy()

    if len(never_tested):

        first_fold_nu = summaries[0]["nu_calibrated"]

        proba = davidson_probabilities(
            never_tested["elo_diff"].to_numpy(), first_fold_nu
        )

        never_tested["fold"] = 0
        never_tested["role"] = "never_a_test_season"
        never_tested["evaluated"] = 0
        never_tested["test_season"] = ""
        never_tested["p_home"] = proba[:, 0]
        never_tested["p_draw"] = proba[:, 1]
        never_tested["p_away"] = proba[:, 2]
        never_tested["predicted_result"] = [
            CLASSES[i] for i in np.argmax(proba, axis=1)
        ]

        results.append(never_tested)

    combined = pd.concat(results, ignore_index=True)

    combined = combined.sort_values(
        ["evaluated", "fold", "date", "home"], ascending=[False, True, True, True]
    ).reset_index(drop=True)

    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")

    return combined[RESULTS_COLUMNS], pd.DataFrame(summaries)[SUMMARY_COLUMNS]


def build_everything(matches, transition_lookup, spec, base_rate):
    """One call producing the full pipeline - re-used by every control test."""

    walk, initialisation = run_elo_walk(matches, transition_lookup)

    results, summary = run_folds(walk, spec, base_rate)

    return walk, initialisation, results, summary


# ============================================================
# TESTS - DATA INTEGRITY
# ============================================================

def test_data_integrity(matches, walk, results, audit):

    audit.record(
        "D1", "Exactly 1,900 matches consumed",
        "match foundation", EXPECTED_TOTAL_MATCHES, len(matches),
        len(matches) == EXPECTED_TOTAL_MATCHES,
    )

    audit.record(
        "D2", "Exactly 1,900 Elo prediction rows",
        "one pre-match rating pair per match",
        EXPECTED_TOTAL_MATCHES, len(results),
        len(results) == EXPECTED_TOTAL_MATCHES,
        "1,520 evaluated as test rows, 380 from the never-tested first season",
    )

    missing_teams = int(
        results["home"].isna().sum() + results["away"].isna().sum()
    )

    audit.record(
        "D3", "No missing teams",
        f"{len(results)} rows x 2 sides", 0, missing_teams,
        missing_teams == 0,
    )

    duplicates = int(results.duplicated(
        subset=["season", "date", "home", "away"]
    ).sum())

    audit.record(
        "D4", "No duplicate matches",
        f"{len(results)} rows", 0, duplicates,
        duplicates == 0,
    )

    block = results[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

    non_finite = int((~np.isfinite(block)).sum())
    out_of_range = int(((block < 0.0) | (block > 1.0)).sum())

    sums = block.sum(axis=1)
    off = int((np.abs(sums - 1.0) > PROBABILITY_TOLERANCE).sum())

    audit.record(
        "D5", "No NaN or infinite probability",
        f"{block.size} values", 0, non_finite,
        non_finite == 0,
    )

    audit.record(
        "D6", "Every probability lies in [0, 1]",
        f"{block.size} values", 0, out_of_range,
        out_of_range == 0,
    )

    audit.record(
        "D7", "Every probability vector sums to 1 within 1e-6",
        f"{len(results)} rows", 0, off,
        off == 0,
        f"max deviation {float(np.abs(sums - 1.0).max()):.3e}",
    )

    # The walk must cover every match exactly once.
    audit.record(
        "D8", "The Elo walk covers every match exactly once",
        f"{EXPECTED_TOTAL_MATCHES} matches",
        f"{EXPECTED_TOTAL_MATCHES} unique match ids",
        f"{walk['match_id'].nunique()} unique of {len(walk)} rows",
        walk["match_id"].nunique() == len(walk) == EXPECTED_TOTAL_MATCHES,
    )

    finite_ratings = int((
        ~np.isfinite(walk[["home_elo_before", "away_elo_before"]].to_numpy())
    ).sum())

    audit.record(
        "D9", "Every pre-match rating is finite",
        f"{len(walk)} matches x 2 sides", 0, finite_ratings,
        finite_ratings == 0,
    )


# ============================================================
# TESTS - ELO MECHANICS
# ============================================================

def test_elo_mechanics(matches, walk, initialisation, transitions,
                       transition_lookup, audit):

    seasons = sorted(matches["season"].unique())

    # ---- E1: the first season starts everyone at 1500
    first = initialisation[initialisation["season"] == seasons[0]]

    wrong_start = int((first["rating_at_season_start"] != INITIAL_RATING).sum())

    audit.record(
        "E1", "Every team starts the first season at the initial rating",
        f"{len(first)} teams in {seasons[0]}",
        f"{INITIAL_RATING} x {len(first)}", f"{len(first) - wrong_start} correct",
        wrong_start == 0,
    )

    # ---- E2: Elo is zero-sum within a match
    imbalance = np.abs(
        (walk["home_elo_after"] - walk["home_elo_before"])
        + (walk["away_elo_after"] - walk["away_elo_before"])
    )

    worst = float(imbalance.max())

    audit.record(
        "E2", "Every update is zero-sum: the home gain equals the away loss",
        f"{len(walk)} matches",
        "0 imbalance", f"max {worst:.3e}",
        worst < 1e-9,
    )

    # ---- E3: the season transition policy matches Phase 1 Instrument 4
    transition_lookup = {
        (row.season, row.team): row.transition_type
        for row in transitions.itertuples()
    }

    policy_errors = []

    carried_lookup = {}

    for season in seasons:

        block = initialisation[initialisation["season"] == season]

        for row in block.itertuples():

            expected_transition = transition_lookup[(season, row.team)]

            if row.transition != expected_transition:
                policy_errors.append(f"{season} {row.team}: transition mismatch")
                continue

            if expected_transition == TRANSITION_CONTINUING:

                if row.policy != "regressed_from_previous_season":
                    policy_errors.append(f"{season} {row.team}: not regressed")
                    continue

                expected_rating = (
                    INITIAL_RATING
                    + (row.rating_before_regression - INITIAL_RATING)
                    * SEASON_REGRESSION
                )

                if abs(row.rating_at_season_start - expected_rating) > 1e-9:
                    policy_errors.append(f"{season} {row.team}: regression wrong")

            else:

                if row.policy != "reset_to_initial_rating":
                    policy_errors.append(f"{season} {row.team}: not reset")

                elif row.rating_at_season_start != INITIAL_RATING:
                    policy_errors.append(f"{season} {row.team}: not at 1500")

        carried_lookup[season] = block

    audit.record(
        "E3", "Season initialisation follows the locked policy exactly",
        f"{len(initialisation)} team-seasons",
        "0 policy errors", f"{len(policy_errors)} errors",
        not policy_errors,
        "; ".join(policy_errors[:5]),
    )

    # Teams lacking a season N-1 rating must be reset, never carried.
    reset_rows = initialisation[
        initialisation["policy"] == "reset_to_initial_rating"
    ]

    carried_in_error = int(reset_rows["rating_before_regression"].notna().sum())

    audit.record(
        "E4", "Teams without a season N-1 rating carry nothing across the gap",
        f"{len(reset_rows)} team-seasons reset",
        0, carried_in_error,
        carried_in_error == 0,
        str(reset_rows["transition"].value_counts().to_dict()),
    )

    # ---- E5: SAME-DAY ISOLATION.
    #
    # The decisive mechanical test. Every match on a date must have been
    # predicted from the ratings standing before that date, so two matches
    # on the same date sharing no team must be mutually invisible - and a
    # team's rating must be identical across every match it plays on a date
    # (Phase 1 proved a team never plays twice on one date, so this checks
    # that no OTHER match on that date moved it).
    same_day_violations = []

    for (season, date), batch in walk.groupby(["season", "date"]):

        if len(batch) < 2:
            continue

        # Reconstruct the ratings standing before this date from the previous
        # appearance of each team, and confirm the batch used exactly those.
        for row in batch.itertuples():

            earlier = walk[
                (walk["season"] == season)
                & (walk["date"] < date)
                & ((walk["home"] == row.home) | (walk["away"] == row.home))
            ]

            if earlier.empty:
                continue

            last = earlier.sort_values("date").iloc[-1]

            expected = (
                last["home_elo_after"] if last["home"] == row.home
                else last["away_elo_after"]
            )

            if abs(row.home_elo_before - expected) > 1e-9:
                same_day_violations.append(
                    f"{season} {date.date()} {row.home}: "
                    f"{row.home_elo_before:.6f} != {expected:.6f}"
                )

    audit.record(
        "E5", "Same-day matches never contribute to each other's ratings",
        f"{int((walk.groupby(['season', 'date']).size() > 1).sum())} "
        f"multi-match dates",
        "0 violations", f"{len(same_day_violations)} violations",
        not same_day_violations,
        "Ratings advance in date batches, never match by match; "
        + "; ".join(same_day_violations[:3]),
    )

    # ---- IS THE DATE-BATCHING LOAD-BEARING?
    #
    # Measured, not assumed. A naive match-by-match walk is run alongside the
    # batched one and the two rating trajectories compared.
    #
    # They agree exactly on this dataset, and the reason is structural rather
    # than lucky: Phase 1 proved no team ever plays twice on one date, so
    # matches sharing a date involve disjoint teams and cannot reach each
    # other's ratings. The batching is therefore correct and defensive, but
    # not load-bearing HERE. It would become load-bearing the moment a
    # rescheduled double-header appeared, which is exactly why it stays.
    naive = {}
    naive_diffs = []

    for season in sorted(matches["season"].unique()):

        season_matches = matches[matches["season"] == season]
        season_teams = sorted(
            set(season_matches["home_team"]) | set(season_matches["away_team"])
        )

        ratings, _ = season_start_ratings(
            season, season_teams, transition_lookup, naive
        )

        for date in sorted(season_matches["date"].unique()):

            for row in season_matches[season_matches["date"] == date].itertuples():

                difference = (
                    ratings[row.home_team] + HOME_ADVANTAGE
                    - ratings[row.away_team]
                )

                naive_diffs.append((row.match_id, difference))

                delta = K_FACTOR * (
                    actual_home_score(row.result)
                    - expected_home_score(difference)
                )

                # The leak a naive walk would allow: applied immediately.
                ratings[row.home_team] += delta
                ratings[row.away_team] -= delta

        naive = dict(ratings)

    naive_frame = pd.DataFrame(
        naive_diffs, columns=["match_id", "naive_diff"]
    ).set_index("match_id")

    aligned = walk.set_index("match_id")["elo_diff"]

    gap = np.abs(aligned - naive_frame["naive_diff"].reindex(aligned.index))

    differing = int((gap > 1e-9).sum())

    audit.measure(
        "E7", "Date-batching versus a naive match-by-match walk",
        f"{len(walk)} matches",
        f"{differing} differing rating differences, max gap {gap.max():.6f}",
        "Identical because no team plays twice on one date (Phase 1 CHK17), "
        "so same-day matches involve disjoint teams. The batching is correct "
        "and defensive, not load-bearing on this dataset.",
    )

    # ---- rating spread, reported for inspection
    audit.measure(
        "E6", "Pre-match rating range across the whole walk",
        f"{len(walk)} matches",
        f"{walk['home_elo_before'].min():.1f} to "
        f"{walk['home_elo_before'].max():.1f}",
        f"mean home-away Elo difference "
        f"{float(walk['elo_diff'].mean()):.2f} (includes +{HOME_ADVANTAGE:.0f} "
        f"home advantage)",
    )


# ============================================================
# TESTS - TEMPORAL INTEGRITY AND PERTURBATION
# ============================================================

def test_temporal_integrity(matches, transition_lookup, spec, base_rate,
                            walk, results, summary, audit):
    """
    The distinction that matters, stated by the brief and tested directly:

        perturbing a match must NOT change that match's own prediction
        perturbing a match SHOULD change later predictions

    A test that only checked the first half could be passed by an Elo that
    never updates at all. Both halves are therefore required.
    """

    baseline_walk = walk.set_index("match_id")

    # Perturb the last matchweek group of each season's opening third, plus a
    # mid-season and a late group, so the control spans the calendar.
    sample_ids = []

    for season in sorted(matches["season"].unique()):

        season_matches = matches[matches["season"] == season]

        for matchweek in (5, 19, 33):

            block = season_matches[season_matches["matchweek"] == matchweek]

            sample_ids.extend(list(block["match_id"]))

    own_changed = 0
    earlier_changed = 0
    later_changed = 0
    checks = 0

    rating_columns = ["home_elo_before", "away_elo_before", "elo_diff"]

    for match_id in sample_ids:

        perturbed = matches.copy()

        target = perturbed[perturbed["match_id"] == match_id].iloc[0]

        # Arsenal 3-0 Chelsea becomes Chelsea 9-0 Arsenal: flip the outcome
        # to its opposite and blow out the scoreline.
        flipped = {"H": "A", "A": "H", "D": "H"}[target["result"]]

        perturbed.loc[perturbed["match_id"] == match_id, "home_goals"] = (
            0 if flipped == "A" else 9
        )
        perturbed.loc[perturbed["match_id"] == match_id, "away_goals"] = (
            9 if flipped == "A" else 0
        )
        perturbed.loc[perturbed["match_id"] == match_id, "result"] = flipped

        rebuilt_walk, _ = run_elo_walk(perturbed, transition_lookup)

        rebuilt = rebuilt_walk.set_index("match_id")

        target_date = target["date"]
        target_season = target["season"]

        # ---- the perturbed match's OWN pre-match state must be untouched
        before = baseline_walk.loc[match_id, rating_columns].to_numpy(dtype=float)
        after = rebuilt.loc[match_id, rating_columns].to_numpy(dtype=float)

        if not np.array_equal(before, after):
            own_changed += 1

        checks += 1

        # ---- every match at or before that date must be untouched too
        at_or_before = baseline_walk[
            (baseline_walk["season"] == target_season)
            & (baseline_walk["date"] <= target_date)
        ].index

        left = baseline_walk.loc[at_or_before, rating_columns].to_numpy(dtype=float)
        right = rebuilt.loc[at_or_before, rating_columns].to_numpy(dtype=float)

        earlier_changed += int((left != right).sum())

        # ---- POSITIVE CONTROL: later matches involving those teams must move
        later = baseline_walk[
            (baseline_walk["season"] == target_season)
            & (baseline_walk["date"] > target_date)
            & (
                (baseline_walk["home"] == target["home_team"])
                | (baseline_walk["away"] == target["home_team"])
            )
        ].index

        if len(later):

            left = baseline_walk.loc[later, rating_columns].to_numpy(dtype=float)
            right = rebuilt.loc[later, rating_columns].to_numpy(dtype=float)

            if (left != right).any():
                later_changed += 1

    audit.record(
        "T1", "A match's own pre-match Elo carries no information from that match",
        f"{checks} matches perturbed to a flipped 9-0 scoreline",
        0, own_changed,
        own_changed == 0,
    )

    audit.record(
        "T2", "No prediction at or before the perturbed date changes",
        f"{checks} perturbations",
        0, earlier_changed,
        earlier_changed == 0,
    )

    audit.record(
        "T3", "Positive control: LATER matches for those teams do change",
        f"{checks} perturbations",
        f"> 0 of {checks}", f"{later_changed} of {checks} moved",
        later_changed > 0,
        "The result legitimately updates the rating; an Elo that never "
        "updated would fail here",
    )

    # ---- fold isolation: a fold's calibration must not see its test season
    nu_changed = []
    nu_train_changed = 0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = str(fold_spec["test_season"])

        for forced in ("H", "D", "A"):

            perturbed = matches.copy()

            perturbed.loc[perturbed["season"] == test_season, "result"] = forced

            _, _, _, perturbed_summary = build_everything(
                perturbed, transition_lookup, spec, base_rate
            )

            before = float(
                summary[summary["fold"] == fold]["nu_calibrated"].iloc[0])
            after = float(
                perturbed_summary[perturbed_summary["fold"] == fold][
                    "nu_calibrated"].iloc[0])

            if before != after:
                nu_changed.append(
                    f"fold {fold} forced {forced}: nu {before} -> {after}"
                )

    audit.record(
        "T4", "A fold's draw parameter is calibrated on training data alone",
        f"{EXPECTED_FOLDS} folds x 3 forced test outcomes",
        "0 changed", f"{len(nu_changed)} changed",
        not nu_changed,
        "Rewriting an entire test season leaves that fold's nu untouched; "
        + "; ".join(nu_changed[:3]),
    )

    # ---- POSITIVE CONTROL for calibration
    control = matches.copy()

    control.loc[control["season"] == "2021-2022", "result"] = "D"

    _, _, _, control_summary = build_everything(
        control, transition_lookup, spec, base_rate
    )

    nu_train_changed = int((
        control_summary["nu_calibrated"].to_numpy()
        != summary["nu_calibrated"].to_numpy()
    ).sum())

    audit.record(
        "T5", "Positive control: perturbing TRAINING data does move the calibration",
        f"{EXPECTED_FOLDS} folds",
        f"> 0 of {EXPECTED_FOLDS}", f"{nu_train_changed} folds moved",
        nu_train_changed > 0,
        "Proves T4 is not vacuous",
    )


# ============================================================
# TESTS - FOLD STRUCTURE
# ============================================================

def test_fold_structure(matches, spec, summary, results, audit):

    audit.record(
        "F1", "Exactly four folds, matching the frozen Phase 0 specification",
        "spec JSON", EXPECTED_FOLDS, len(spec["folds"]),
        len(spec["folds"]) == EXPECTED_FOLDS,
    )

    folds_csv = pd.read_csv(FOLDS_CSV, float_precision=FLOAT_PRECISION)

    disagreements = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        row = folds_csv[folds_csv["fold"] == fold]

        expected_train = " + ".join(fold_spec["train_seasons"])

        if row.empty or str(row.iloc[0]["train_seasons"]) != expected_train:
            disagreements.append(f"fold {fold} train seasons")

        used = summary[summary["fold"] == fold].iloc[0]

        if used["train_seasons"] != expected_train:
            disagreements.append(f"fold {fold} Elo used different train seasons")

        if used["test_season"] != str(fold_spec["test_season"]):
            disagreements.append(f"fold {fold} Elo used a different test season")

    audit.record(
        "F2", "Fold definitions are unchanged from Phase 0",
        "spec JSON, folds CSV and the Elo run",
        "0 disagreements", f"{len(disagreements)} disagreements",
        not disagreements,
        "; ".join(disagreements[:5]),
    )

    observed_train = list(summary.sort_values("fold")["train_matches"])
    observed_test = list(summary.sort_values("fold")["test_matches"])

    audit.record(
        "F3", "Training sizes are 380 / 760 / 1140 / 1520",
        f"{EXPECTED_FOLDS} folds",
        str(EXPECTED_TRAIN_SIZES), str(observed_train),
        observed_train == EXPECTED_TRAIN_SIZES,
    )

    audit.record(
        "F4", "Every test set contains exactly 380 matches",
        f"{EXPECTED_FOLDS} folds",
        f"{EXPECTED_TEST_SIZE} x {EXPECTED_FOLDS}", str(observed_test),
        observed_test == [EXPECTED_TEST_SIZE] * EXPECTED_FOLDS,
    )

    overlaps = 0
    boundary = []

    for fold_spec in spec["folds"]:

        train = matches[matches["season"].isin(fold_spec["train_seasons"])]
        test = matches[matches["season"] == fold_spec["test_season"]]

        overlaps += len(set(train["match_id"]) & set(test["match_id"]))

        if not train["date"].max() < test["date"].min():
            boundary.append(f"fold {fold_spec['fold']}")

    audit.record(
        "F5", "No test match occurs in training data",
        f"{EXPECTED_FOLDS} folds", 0, overlaps,
        overlaps == 0,
    )

    audit.record(
        "F6", "Every training date is earlier than every test date",
        f"{EXPECTED_FOLDS} folds", 0, len(boundary),
        not boundary,
        "; ".join(boundary),
    )

    evaluated = results[results["evaluated"] == 1]

    audit.record(
        "F7", "Exactly 1,520 evaluated test predictions",
        "4 folds x 380", EXPECTED_TOTAL_TEST, len(evaluated),
        len(evaluated) == EXPECTED_TOTAL_TEST,
    )

    duplicated = int(evaluated.duplicated(
        subset=["test_season", "date", "home", "away"]
    ).sum())

    audit.record(
        "F8", "Each test match appears exactly once",
        f"{len(evaluated)} rows", 0, duplicated,
        duplicated == 0,
    )

    seasons_tested = sorted(evaluated["test_season"].unique())

    audit.record(
        "F9", "The four test seasons are exactly those in the frozen spec",
        "4 test seasons",
        "['2022-2023', '2023-2024', '2024-2025', '2025-2026']",
        str(seasons_tested),
        seasons_tested == ["2022-2023", "2023-2024", "2024-2025", "2025-2026"],
    )


# ============================================================
# TESTS - OUTPUT CONTRACT
# ============================================================

def test_output_contract(results, summary, audit):

    block = results[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

    expected_prediction = [CLASSES[i] for i in np.argmax(block, axis=1)]

    wrong = int((results["predicted_result"].to_numpy()
                 != np.array(expected_prediction)).sum())

    audit.record(
        "O1", "predicted_result is argmax([p_home, p_draw, p_away])",
        f"{len(results)} rows", 0, wrong,
        wrong == 0,
    )

    audit.record(
        "O2", "Probability columns are ordered [P(H), P(D), P(A)]",
        "results schema and harness contract",
        "['H', 'D', 'A']", str(list(CLASSES)),
        list(CLASSES) == ["H", "D", "A"]
        and RESULTS_COLUMNS[13:16] == ["p_home", "p_draw", "p_away"],
    )

    # ---- metrics must reproduce from the results table alone
    mismatches = []

    evaluated = results[results["evaluated"] == 1]

    for fold, group in evaluated.groupby("fold"):

        proba = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

        recomputed = evaluate(group["actual_result"].to_numpy(), proba)

        row = summary[summary["fold"] == fold].iloc[0]

        for metric in METRIC_NAMES:
            if not np.isclose(float(row[metric]), recomputed[metric], atol=1e-12):
                mismatches.append(f"fold {fold} {metric}")

    audit.record(
        "O3", "Fold-summary metrics reproduce from the results table",
        f"{EXPECTED_FOLDS} folds x {len(METRIC_NAMES)} metrics",
        "0 mismatches", f"{len(mismatches)} mismatches",
        not mismatches,
        "; ".join(mismatches[:5]),
    )

    audit.record(
        "O4", "Metrics come from the Phase 0 harness, not a private copy",
        "scoring functions", "phase0_evaluation_harness", evaluate.__module__,
        evaluate.__module__ == "phase0_evaluation_harness",
    )

    # ---- the never-tested rows must never enter a metric
    never = results[results["evaluated"] == 0]

    audit.record(
        "O5", "The never-tested first season is excluded from every metric",
        f"{len(never)} rows flagged evaluated=0",
        "380 excluded", f"{len(never)} excluded",
        len(never) == EXPECTED_TOTAL_MATCHES - EXPECTED_TOTAL_TEST
        and set(never["season"]) == {"2021-2022"},
        "They carry Elo state for inspection only",
    )

    # ---- the locked specification must be recorded, identically, per fold
    locked = summary[[
        "k_factor", "home_advantage", "season_regression", "initial_rating"
    ]].drop_duplicates()

    audit.record(
        "O6", "The locked Elo constants are identical across all four folds",
        f"{EXPECTED_FOLDS} folds",
        "1 distinct specification", f"{len(locked)} distinct",
        len(locked) == 1,
        f"K={K_FACTOR}, HA={HOME_ADVANTAGE}, "
        f"regression={SEASON_REGRESSION}, initial={INITIAL_RATING}",
    )


def test_determinism(matches, transition_lookup, spec, base_rate,
                     results, summary, audit):

    _, _, repeat_results, repeat_summary = build_everything(
        matches, transition_lookup, spec, base_rate
    )

    results_identical = results.equals(repeat_results)
    summary_identical = summary.equals(repeat_summary)

    audit.record(
        "O7", "Running the pipeline twice produces identical outputs",
        "full rebuild in-process",
        "identical",
        f"results {'identical' if results_identical else 'DIFFER'}, "
        f"summary {'identical' if summary_identical else 'DIFFER'}",
        results_identical and summary_identical,
        "The nu search is a fixed bracketed golden section - no randomness",
    )

    if RESULTS_OUTPUT.exists():

        reloaded = pd.read_csv(RESULTS_OUTPUT, float_precision=FLOAT_PRECISION)

        columns = ["p_home", "p_draw", "p_away",
                   "home_elo_before", "away_elo_before"]

        deviation = float(np.max(np.abs(
            reloaded[columns].to_numpy(dtype=float)
            - results[columns].to_numpy(dtype=float)
        ))) if len(reloaded) == len(results) else float("inf")

        audit.record(
            "O8", "Written values round-trip through CSV exactly",
            f"{len(results)} rows x {len(columns)} columns",
            "0.0 deviation", f"{deviation:.3e}",
            deviation == 0.0,
        )


def test_isolation(before_state, audit):

    every_open = opened_paths()
    input_opens = opened_paths("input")

    raw_touches = [
        str(path) for path in every_open
        if RAW_DIR == path or RAW_DIR in path.parents
    ]

    audit.record(
        "P1", "No file under data/raw/ was opened for any purpose",
        "labelled runtime file-access record", 0, len(raw_touches),
        not raw_touches,
        "The Phase 0 harness's own load_matches() reads data/raw and is "
        "never called",
    )

    input_data_files = {
        path for path in input_opens
        if path.suffix.lower() in {".csv", ".json", ".xls", ".xlsx"}
        and PROJECT_ROOT in path.parents
    }

    unexpected = sorted(
        str(path.relative_to(PROJECT_ROOT))
        for path in input_data_files - DECLARED_INPUTS - OWN_OUTPUTS
    )

    audit.record(
        "P2", "Only the declared frozen inputs were read as inputs",
        "opens labelled 'input'",
        "0 unexpected", f"{len(unexpected)} unexpected",
        not unexpected,
        f"{len(DECLARED_INPUTS)} declared inputs; "
        f"{len(input_data_files)} data files opened as input",
    )

    # ---- P3: the engineered features were never read AS AN INPUT.
    #
    # The file IS opened, by the frozen-state guard, to hash it. That is an
    # integrity read. The labelled record separates the two, so this test
    # says what it means rather than what is convenient.
    features_path = (OUTPUTS_DIR / "phase1_team_strength_features.csv").resolve()

    read_as_input = features_path in input_data_files

    hashed_only = [
        label for path, label in
        ((Path(p).resolve(), lab) for p, lab in _OPENED_PATHS)
        if path == features_path
    ]

    audit.record(
        "P3", "The 86 engineered features were never read as a model input",
        "phase1_team_strength_features.csv",
        "0 input reads", f"{int(read_as_input)} input reads",
        not read_as_input,
        f"opened {len(hashed_only)} time(s), all under "
        f"{sorted(set(hashed_only))} - Elo derives its rating from "
        f"sequential match results alone",
    )

    audit.record(
        "P4", "No engineered feature column reaches the Elo output",
        f"{len(RESULTS_COLUMNS)} output columns",
        "0 engineered feature names",
        "0 engineered feature names",
        not ({"ppm_before", "venue_ppm_before", "prev_season_ppm",
              "last5_ppm_before", "form_delta_ppm"} & set(RESULTS_COLUMNS)),
        "Elo output carries ratings and probabilities only",
    )

    after_state = frozen_state()

    changed = sorted(
        name for name in before_state
        if before_state[name] != after_state.get(name)
    )

    audit.record(
        "P5", "Phase 0 and Phase 1 outputs and scripts are unmodified",
        f"{len(before_state)} tracked files, SHA-256 before and after",
        "0 changed", f"{len(changed)} changed",
        not changed and set(before_state) == set(after_state),
        "; ".join(changed[:3]) if changed else "every hash identical",
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


def print_specification(summary):

    print()
    print("=" * 79)
    print("THE LOCKED SPECIFICATION")
    print("=" * 79)
    print()
    print("  Fixed before any test season was touched, identical in all folds:")
    print()
    print(f"    initial rating     {INITIAL_RATING:.0f}")
    print(f"    K-factor           {K_FACTOR:.0f}")
    print(f"    home advantage     {HOME_ADVANTAGE:.0f} Elo points")
    print(f"    season regression  {SEASON_REGRESSION}")
    print()
    print("  The ONLY calibrated quantity is the Davidson draw parameter nu,")
    print("  fitted per fold on that fold's training matches alone:")
    print()
    print(f"    {'Fold':<6}{'Train seasons':<44}{'nu':>8}{'train logloss':>15}")

    for row in summary.itertuples():

        seasons = row.train_seasons

        if len(seasons) > 42:
            seasons = seasons[:39] + "..."

        print(f"    {row.fold:<6}{seasons:<44}{row.nu_calibrated:>8.4f}"
              f"{row.train_log_loss:>15.4f}")


def print_ratings(walk, initialisation, matches):

    print()
    print("=" * 79)
    print("ELO RATINGS")
    print("=" * 79)
    print()

    final_season = sorted(matches["season"].unique())[-1]

    season_walk = walk[walk["season"] == final_season]

    final = {}

    for row in season_walk.sort_values("date").itertuples():
        final[row.home] = row.home_elo_after
        final[row.away] = row.away_elo_after

    ranked = sorted(final.items(), key=lambda item: item[1], reverse=True)

    print(f"  End-of-{final_season} ratings, top and bottom five:")
    print()

    for team, rating in ranked[:5]:
        print(f"    {team:<20}{rating:>8.1f}")

    print("    ...")

    for team, rating in ranked[-5:]:
        print(f"    {team:<20}{rating:>8.1f}")

    print()
    print("  Season initialisation policy applied:")
    print()

    counts = initialisation.groupby(["policy", "transition"]).size()

    for (policy, transition), count in counts.items():
        print(f"    {policy:<34}{transition:<26}{count:>4}")


def print_metrics(summary, overall, base_rate, results):

    print()
    print("=" * 79)
    print("FIVE METRICS PER FOLD")
    print("=" * 79)
    print()
    print(
        f"    {'Fold':<5}{'Test season':<13}{'N':>5}{'Acc':>9}{'BalAcc':>9}"
        f"{'MacroF1':>9}{'LogLoss':>9}{'Brier':>9}"
    )

    for row in summary.itertuples():
        print(
            f"    {row.fold:<5}{row.test_season:<13}{row.test_matches:>5}"
            f"{row.accuracy:>9.4f}{row.balanced_accuracy:>9.4f}"
            f"{row.macro_f1:>9.4f}{row.log_loss:>9.4f}{row.brier_score:>9.4f}"
        )

    print()
    print(f"    {'ALL':<5}{'1,520 matches':<13}{overall['n']:>5}"
          f"{overall['accuracy']:>9.4f}{overall['balanced_accuracy']:>9.4f}"
          f"{overall['macro_f1']:>9.4f}{overall['log_loss']:>9.4f}"
          f"{overall['brier_score']:>9.4f}")

    print()
    print("=" * 79)
    print("HEAD TO HEAD AGAINST THE BASE RATE - PER FOLD")
    print("=" * 79)
    print()
    print(
        f"    {'Fold':<6}{'Test season':<13}{'LogLoss':>10}{'base':>9}"
        f"{'delta':>9}{'Brier':>10}{'base':>9}{'delta':>9}   beats"
    )

    for row in summary.itertuples():
        print(
            f"    {row.fold:<6}{row.test_season:<13}"
            f"{row.log_loss:>10.4f}{row.base_rate_log_loss:>9.4f}"
            f"{row.log_loss_delta:>+9.4f}"
            f"{row.brier_score:>10.4f}{row.base_rate_brier:>9.4f}"
            f"{row.brier_delta:>+9.4f}   {'yes' if row.beats_base_rate else 'NO'}"
        )

    print()
    print("  Negative delta = Elo is better. Both metrics are losses.")


def print_comparison(overall, base_rate, results):

    uniform_log_loss = float(np.log(3.0))

    base_overall_n = int(base_rate["test_matches"].sum())

    weights = base_rate["test_matches"].to_numpy(dtype=float)

    base_overall = {
        metric: float(np.average(
            base_rate[metric].to_numpy(dtype=float), weights=weights))
        for metric in METRIC_NAMES
    }

    print()
    print("=" * 79)
    print("MODEL COMPARISON - 1,520 UNTOUCHED TEST MATCHES")
    print("=" * 79)
    print()
    print(
        f"    {'Model':<14}{'Accuracy':>10}{'BalAcc':>9}{'MacroF1':>9}"
        f"{'LogLoss':>10}{'Brier':>9}"
    )

    print(
        f"    {'Uniform':<14}{1 / 3:>10.4f}{1 / 3:>9.4f}{'':>9}"
        f"{uniform_log_loss:>10.4f}{2 / 3:>9.4f}"
    )

    print(
        f"    {'Base Rate':<14}{base_overall['accuracy']:>10.4f}"
        f"{base_overall['balanced_accuracy']:>9.4f}"
        f"{base_overall['macro_f1']:>9.4f}"
        f"{base_overall['log_loss']:>10.4f}{base_overall['brier_score']:>9.4f}"
    )

    print(
        f"    {'Elo v1':<14}{overall['accuracy']:>10.4f}"
        f"{overall['balanced_accuracy']:>9.4f}{overall['macro_f1']:>9.4f}"
        f"{overall['log_loss']:>10.4f}{overall['brier_score']:>9.4f}"
    )

    print()

    log_loss_gain = base_overall["log_loss"] - overall["log_loss"]
    brier_gain = base_overall["brier_score"] - overall["brier_score"]

    print(f"    Elo vs Base Rate    log loss {log_loss_gain:+.4f}   "
          f"brier {brier_gain:+.4f}")
    print("    (positive = Elo improves on the base rate)")

    print()

    if log_loss_gain > 0 and brier_gain > 0:
        print("  VERDICT: Elo beats the base rate on both probabilistic metrics.")
        print("  Team ratings carry information the league-wide frequency does not.")
    elif log_loss_gain > 0 or brier_gain > 0:
        print("  VERDICT: Elo improves one probabilistic metric but not both.")
        print("  A partial result - worth understanding before adding complexity.")
    else:
        print("  VERDICT: Elo does NOT beat the base rate.")
        print("  That is a real finding, not a bug: team ratings in this simple")
        print("  form carry less information than league-wide outcome frequency.")

    predicted = sorted(results[results["evaluated"] == 1][
        "predicted_result"].unique())

    print()
    print(f"  Classes Elo ever predicts: {predicted}")

    if predicted == ["H"]:
        print("  Elo still never predicts a draw or an away win. Under the")
        print("  Davidson model a draw needs P(D) to be the argmax, which")
        print("  requires nu > theta + 1/sqrt(theta) - rare at these ratings.")


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 79)
    print("PHASE 2 - INSTRUMENT 2: ELO BASELINE (v1)")
    print("=" * 79)
    print()
    print(f"  Matches    : {MATCHES_CSV.relative_to(PROJECT_ROOT)} (frozen)")
    print(f"  Transitions: {TRANSITIONS_CSV.relative_to(PROJECT_ROOT)} (frozen)")
    print("  Metrics    : imported from phase0_evaluation_harness")
    print("  Model      : sequential Elo from results, + fixed home advantage")
    print("  Update     : DATE BATCHES - same-day matches cannot see each other")
    print("  Calibrated : draw parameter nu only, on training data only")
    print("  Not used   : the 86 engineered features, xG, FBref aggregates")

    before_state = frozen_state()

    spec, matches, transitions, base_rate = load_inputs()

    transition_lookup = {
        (row.season, row.team): row.transition_type
        for row in transitions.itertuples()
    }

    audit = Audit()

    print()
    print(f"  {len(matches)} matches, {len(spec['folds'])} folds, "
          f"{len(transitions)} team-seasons.")
    print("  Running the Elo walk in date batches ...")

    walk, initialisation, results, summary = build_everything(
        matches, transition_lookup, spec, base_rate
    )

    multi_day = int((matches.groupby(["season", "date"]).size() > 1).sum())

    print(f"  {len(walk)} pre-match rating pairs across "
          f"{matches['date'].nunique()} dates ({multi_day} carrying more "
          f"than one match).")

    print("  D   data integrity ...")
    test_data_integrity(matches, walk, results, audit)

    print("  E   Elo mechanics and same-day isolation ...")
    test_elo_mechanics(matches, walk, initialisation, transitions,
                       transition_lookup, audit)

    print("  F   fold structure ...")
    test_fold_structure(matches, spec, summary, results, audit)

    print("  O   output contract ...")
    test_output_contract(results, summary, audit)

    evaluated = results[results["evaluated"] == 1]

    overall = evaluate(
        evaluated["actual_result"].to_numpy(),
        evaluated[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(RESULTS_OUTPUT, index=False, encoding="utf-8",
                   float_format=FLOAT_FORMAT)
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8",
                   float_format=FLOAT_FORMAT)

    print("  O7  determinism ...")
    test_determinism(matches, transition_lookup, spec, base_rate,
                     results, summary, audit)

    print("  T   temporal integrity and perturbation controls ...")
    test_temporal_integrity(matches, transition_lookup, spec, base_rate,
                            walk, results, summary, audit)

    print("  P   provenance and frozen-state guard ...")
    test_isolation(before_state, audit)

    audit_frame = audit.frame()
    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    print_test_table(audit)
    print_specification(summary)
    print_ratings(walk, initialisation, matches)
    print_metrics(summary, overall, base_rate, results)
    print_comparison(overall, base_rate, results)

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {RESULTS_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(results)} rows, {len(evaluated)} evaluated)")
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
    print("PHASE 2 - INSTRUMENT 2 STATUS")
    print("=" * 79)
    print()

    line("Matches:", f"{len(matches)}")
    line("Elo prediction rows:", f"{len(results)}")
    line("Evaluated test matches:", f"{len(evaluated)}")
    line("D  data integrity:", "1,900 rows, no NaN", outcome("D"))
    line("E  Elo mechanics:", "zero-sum, policy, same-day", outcome("E"))
    line("F  fold structure:", "frozen Phase 0 folds", outcome("F"))
    line("O  output contract:", "argmax, metrics, determinism", outcome("O"))
    line("T  temporal integrity:", "own-match + positive control", outcome("T"))
    line("P  provenance:", "no raw, no features", outcome("P"))

    if failures:
        print()
        print("  FAILURES:")

        for failure in failures:
            print(
                f"    {failure['test_id']} {failure['test']}: "
                f"expected {failure['expected']}, got {failure['observed']} "
                f"{failure['detail']}".rstrip()
            )

    total_tests = len([r for r in audit.rows if r["status"] in {"PASS", "FAIL"}])

    print()
    print(f"  Tests run          : {total_tests}")
    print(f"  Tests passed       : {total_tests - len(failures)}")
    print(f"  Tests failed       : {len(failures)}")
    print()
    print(f"{total_tests - len(failures)}/{total_tests} tests passed")
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
