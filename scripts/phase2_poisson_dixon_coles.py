"""
===============================================================================
PHASE 2 - INSTRUMENT 3
POISSON / DIXON-COLES  (v1)
===============================================================================

THE QUESTION
    Elo asks which team is stronger. This asks how many goals each team
    should score, builds a full score matrix, and sums it into H/D/A.

        lambda_home = A_home * D_away * H
        lambda_away = A_away * D_home

        P(x, y) = tau(x, y) * Poisson(x; lambda_home) * Poisson(y; lambda_away)

        P(H) = sum over x > y      P(D) = sum over x == y      P(A) = x < y

    So the real question is not whether accuracy beats Elo. It is whether
    modelling GOALS produces better-CALIBRATED probabilities than modelling
    team strength alone - which is what log loss and Brier measure.

    Benchmarks to beat, on the same 1,520 untouched test matches:

        Base Rate   log loss 1.0689   brier 0.6467
        Elo v1      log loss 0.9994   brier 0.5958

FOUR VARIANTS, VARYING EXACTLY TWO THINGS
    Two questions are being asked, and each needs its own controlled contrast.

    (1) HOW MUCH HISTORY may the fit see? The brief contains a real tension:

        "No test-season information enters parameter fitting"
        "Strict historical cutoff is respected"
        "No match can influence its own prediction"

    Elo used every match with date < T, INCLUDING earlier matches of the test
    season. A Poisson fitted once on the training seasons would predict May
    2026 with parameters frozen in May 2025, and would lose to Elo on
    information access rather than on modelling. So both are built:

        *_static         fitted ONCE per fold on the training seasons only.
                         Satisfies the first guarantee literally. No test
                         season match ever touches the fit.

        *_walkforward    refitted before every distinct test date on all
                         matches with date < T. Satisfies the strict cutoff,
                         and gives Poisson exactly the information Elo had -
                         the only fair head-to-head.

    (2) Does the DIXON-COLES CORRECTION earn its place? That cannot be
    answered by a model that always applies it, so the plain model is run
    beside it:

        poisson_*        rho pinned to 0. tau is then identically 1 and the
                         two scorelines are independent Poisson draws.

        dc_*             rho fitted by profile likelihood on the same window.

    Estimation is two-stage - attack/defence/home first, rho afterwards - so
    a pair shares IDENTICAL expected goals and differs in the correction and
    nothing else. Tests C7/C8 enforce exactly that, which is what makes
    "Dixon-Coles minus Poisson" a measurement rather than a slogan.

        variant                plain Poisson    Dixon-Coles
        training seasons only  poisson_static   dc_static
        strict cutoff          poisson_wf       dc_walkforward

    All four are audited identically and all four are reported. None is
    hidden, and none is selected after seeing the test seasons.

THE LADDER THIS INSTRUMENT REPORTS
        base rate  ->  Poisson  ->  Dixon-Coles
    Each rung is judged against the rung directly below it. Beating the base
    rate is a floor, not a result.

THE LOCKED SPECIFICATION - fixed before any test season was touched
    max goals per side        25 (score matrix truncated, then renormalised)
    time-decay half-life      107 days
    rho search bounds         [-0.5, 0.5]

    The half-life is NOT tuned here. It is the value implied by Dixon and
    Coles's own published xi = 0.0065 per day (ln 2 / 0.0065 = 106.6 days),
    taken from the literature exactly as Elo's K=20 and 60-point home
    advantage were taken as conventional constants. Tuning it against these
    test seasons is precisely what this instrument must not do.

    Fitted per window, from that window's matches only:
        A_i   attack strength for every team
        D_i   defence strength for every team
        H     home advantage multiplier
        rho   the Dixon-Coles low-score correction

ESTIMATION - NO SCIPY, FULLY DETERMINISTIC
    The attack/defence/home parameters are fitted by multiplicative iterative
    scaling, which is the exact closed-form coordinate solution of the
    weighted Poisson likelihood:

        A_i = (weighted goals scored by i) / (weighted expected, given D, H)
        D_i = (weighted goals conceded by i) / (weighted expected, given A, H)
        H   = (weighted home goals) / (weighted expected, given A, D)

    rho is then profiled by the same bracketed golden-section search used for
    Elo's draw parameter. Two-stage rather than joint MLE - a v1
    simplification, stated rather than hidden.

    No random initialisation, no solver dependency, no restarts. The fit is
    reproducible to the bit.

TEAMS WITH NO HISTORY IN THE WINDOW
    A promoted side has no matches in the fitting window, so it has no attack
    or defence parameter. It is given the league-neutral value (A = D = 1,
    i.e. exactly average), the direct analogue of Elo's reset to 1500.
    Nothing is fabricated, and every occurrence is counted and reported.

INPUTS - frozen, read-only
    outputs/phase0_evaluation_spec.json          the frozen folds
    outputs/phase0_evaluation_folds.csv          cross-check
    outputs/phase1_matches.csv                   match results and scores
    outputs/phase2_base_rate_fold_summary.csv    benchmark
    outputs/phase2_elo_fold_summary.csv          benchmark

    The 86 engineered Phase 1 features are deliberately NOT used, exactly as
    in Elo v1. This is a historical-results-only baseline.

EXIT CODES
    0 PASS   2 FAIL/INVESTIGATE   1 FATAL

NOT BUILT HERE
    no feature-based ML, no XGBoost, no xG, no FBref aggregate, no random
    splitting, no change to the four folds, no tuning against a test season.
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
# PURPOSE-LABELLED FILE-ACCESS RECORDER
# ============================================================

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
    """Label every file open inside this block with its purpose."""

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
    PROBABILITY_TOLERANCE,
    ProbabilityError,
    evaluate,
    validate_probabilities,
)


# ============================================================
# THE LOCKED SPECIFICATION
# ============================================================

MAX_GOALS = 25

# ln(2) / 0.0065 = 106.6 days, from Dixon & Coles (1997). Taken from the
# literature, not fitted here.
TIME_DECAY_HALF_LIFE_DAYS = 107.0

RHO_LOWER = -0.5
RHO_UPPER = 0.5
RHO_GRID_POINTS = 60
RHO_GOLDEN_ITERATIONS = 60

SCALING_MAX_ITERATIONS = 200
SCALING_TOLERANCE = 1e-10

NEUTRAL_STRENGTH = 1.0
MIN_RATE = 1e-6

VARIANT_POISSON_STATIC = "poisson_static"
VARIANT_POISSON_WALKFORWARD = "poisson_walkforward"
VARIANT_STATIC = "dc_static"
VARIANT_WALKFORWARD = "dc_walkforward"
PRIMARY_VARIANT = VARIANT_WALKFORWARD

# The 2 x 2 design. Information access varies down the pairs, the Dixon-Coles
# correction varies across them, so each comparison moves exactly one thing.
#
#                      plain Poisson (rho = 0)      Dixon-Coles (rho fitted)
#   training only      poisson_static               dc_static
#   strict cutoff      poisson_walkforward          dc_walkforward
VARIANTS = (
    VARIANT_POISSON_STATIC, VARIANT_STATIC,
    VARIANT_POISSON_WALKFORWARD, VARIANT_WALKFORWARD,
)

STATIC_VARIANTS = (VARIANT_POISSON_STATIC, VARIANT_STATIC)
WALKFORWARD_VARIANTS = (VARIANT_POISSON_WALKFORWARD, VARIANT_WALKFORWARD)
DIXON_COLES_VARIANTS = (VARIANT_STATIC, VARIANT_WALKFORWARD)

# Each plain-Poisson variant and its Dixon-Coles partner, same information set.
DC_PAIRS = (
    (VARIANT_POISSON_STATIC, VARIANT_STATIC),
    (VARIANT_POISSON_WALKFORWARD, VARIANT_WALKFORWARD),
)


def is_static(variant):
    """True if the variant fits once per fold on the training seasons only."""

    return variant in STATIC_VARIANTS


def uses_dixon_coles(variant):
    """True if rho is fitted; False pins rho to 0 and leaves plain Poisson."""

    return variant in DIXON_COLES_VARIANTS


# ============================================================
# CONFIGURATION
# ============================================================

SPEC_JSON = OUTPUTS_DIR / "phase0_evaluation_spec.json"
FOLDS_CSV = OUTPUTS_DIR / "phase0_evaluation_folds.csv"
MATCHES_CSV = OUTPUTS_DIR / "phase1_matches.csv"
BASE_RATE_CSV = OUTPUTS_DIR / "phase2_base_rate_fold_summary.csv"
ELO_CSV = OUTPUTS_DIR / "phase2_elo_fold_summary.csv"

RESULTS_OUTPUT = OUTPUTS_DIR / "phase2_poisson_dc_results.csv"
SUMMARY_OUTPUT = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase2_poisson_dc_audit.csv"

DECLARED_INPUTS = {
    SPEC_JSON.resolve(), FOLDS_CSV.resolve(), MATCHES_CSV.resolve(),
    BASE_RATE_CSV.resolve(), ELO_CSV.resolve(),
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

FLOAT_FORMAT = "%.17g"
FLOAT_PRECISION = "round_trip"

RESULTS_COLUMNS = [
    "variant", "fold", "test_season", "date", "home", "away",
    "lambda_home", "lambda_away", "rho", "fit_matches", "fit_cutoff_date",
    "home_has_history", "away_has_history",
    "score_matrix_mass", "truncated_mass",
    "p_home", "p_draw", "p_away",
    "actual_result", "predicted_result",
]

SUMMARY_COLUMNS = [
    "variant", "fold", "train_seasons", "test_season",
    "train_matches", "test_matches", "refits",
    "max_goals", "half_life_days", "rho_first_fit", "rho_last_fit",
    "home_advantage_multiplier", "neutral_team_predictions",
    "degenerate_parameters",
    "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score",
    "rps",
    "base_rate_log_loss", "base_rate_brier", "base_rate_rps",
    "elo_log_loss", "elo_brier",
    "log_loss_vs_base", "log_loss_vs_elo",
    "brier_vs_base", "brier_vs_elo",
    "rps_vs_base",
    "beats_base_rate", "beats_elo",
]

# The full metric set the project requires. RPS is included because the
# harness computes it and the base rate already reports it; a model scored on
# a subset of the agreed metrics is not comparable to the baseline.
METRIC_NAMES = [
    "accuracy", "balanced_accuracy", "macro_f1", "log_loss", "brier_score",
    "rps",
]

# Elo v1's frozen summary predates RPS and has no such column. It is compared
# on the metrics it actually recorded rather than back-filled with a guess.
ELO_METRIC_NAMES = [
    metric for metric in METRIC_NAMES if metric != "rps"
]

# Precomputed log factorials for the Poisson pmf - no scipy.
_LOG_FACTORIAL = np.array(
    [math.lgamma(k + 1) for k in range(MAX_GOALS + 1)], dtype=float
)

_GOALS = np.arange(MAX_GOALS + 1)


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

    with access_context("hash_guard"):

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

def load_inputs():

    for required in (SPEC_JSON, FOLDS_CSV, MATCHES_CSV, BASE_RATE_CSV, ELO_CSV):
        if not required.exists():
            raise FatalError(f"missing required input: {required}")

    try:
        spec = json.loads(SPEC_JSON.read_text(encoding="utf-8"))
        matches = pd.read_csv(MATCHES_CSV, float_precision=FLOAT_PRECISION)
        base_rate = pd.read_csv(BASE_RATE_CSV, float_precision=FLOAT_PRECISION)
        elo = pd.read_csv(ELO_CSV, float_precision=FLOAT_PRECISION)
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

    if (matches["home_goals"] > MAX_GOALS).any() or (
            matches["away_goals"] > MAX_GOALS).any():
        raise FatalError(
            f"a score exceeds the {MAX_GOALS}-goal score matrix truncation"
        )

    return spec, matches, base_rate, elo


# ============================================================
# THE MODEL
# ============================================================

def time_weights(dates, reference_date):
    """
    Exponential decay toward the cutoff. Locked half-life, never fitted.

    A match played `half_life` days before the cutoff counts half as much as
    one played on the cutoff.
    """

    age_days = (reference_date - dates).dt.days.to_numpy(dtype=float)

    return np.power(0.5, age_days / TIME_DECAY_HALF_LIFE_DAYS)


def fit_attack_defence(window, weights):
    """
    Multiplicative iterative scaling - the closed-form coordinate solution of
    the weighted Poisson likelihood. Deterministic, no solver dependency.

    Returns (attack, defence, home_multiplier, iterations, converged).
    """

    teams = sorted(set(window["home_team"]) | set(window["away_team"]))

    index = {team: position for position, team in enumerate(teams)}

    home = window["home_team"].map(index).to_numpy()
    away = window["away_team"].map(index).to_numpy()

    home_goals = window["home_goals"].to_numpy(dtype=float)
    away_goals = window["away_goals"].to_numpy(dtype=float)

    n_teams = len(teams)

    # Weighted goals scored / conceded, the numerators of every update.
    scored = (
        np.bincount(home, weights=weights * home_goals, minlength=n_teams)
        + np.bincount(away, weights=weights * away_goals, minlength=n_teams)
    )

    conceded = (
        np.bincount(home, weights=weights * away_goals, minlength=n_teams)
        + np.bincount(away, weights=weights * home_goals, minlength=n_teams)
    )

    attack = np.ones(n_teams, dtype=float)
    defence = np.ones(n_teams, dtype=float)
    home_multiplier = 1.0

    total_home_goals = float(np.sum(weights * home_goals))

    converged = False
    iterations = 0

    for iterations in range(1, SCALING_MAX_ITERATIONS + 1):

        previous = np.concatenate([attack, defence, [home_multiplier]])

        # ---- attack
        denominator = (
            np.bincount(
                home, weights=weights * defence[away] * home_multiplier,
                minlength=n_teams)
            + np.bincount(away, weights=weights * defence[home],
                          minlength=n_teams)
        )

        attack = np.where(denominator > 0, scored / np.maximum(denominator, MIN_RATE),
                          NEUTRAL_STRENGTH)

        # ---- defence
        denominator = (
            np.bincount(
                away, weights=weights * attack[home] * home_multiplier,
                minlength=n_teams)
            + np.bincount(home, weights=weights * attack[away],
                          minlength=n_teams)
        )

        defence = np.where(denominator > 0, conceded / np.maximum(denominator, MIN_RATE),
                           NEUTRAL_STRENGTH)

        # ---- home advantage
        expected_home = float(np.sum(weights * attack[home] * defence[away]))

        if expected_home > 0:
            home_multiplier = total_home_goals / expected_home

        # ---- fix the scale invariance: A -> cA, D -> D/c leaves both rates
        # unchanged, so pin the geometric mean of attack at 1.
        positive = attack > 0

        if positive.any():
            scale = float(np.exp(np.mean(np.log(attack[positive]))))

            if scale > 0:
                attack = attack / scale
                defence = defence * scale

        current = np.concatenate([attack, defence, [home_multiplier]])

        if np.max(np.abs(current - previous)) < SCALING_TOLERANCE:
            converged = True
            break

    # DEGENERATE BOUNDARY MLE.
    #
    # A team that has scored no goals in the window gets attack exactly 0;
    # one that has conceded none gets defence exactly 0. The likelihood is
    # then maximised on the boundary and the MLE does not exist in the
    # interior - the model would be claiming the team CANNOT score, or
    # CANNOT concede, on the strength of one or two matches.
    #
    # Bournemouth won their 2022-23 debut 2-0, which made their fitted
    # defence 0 and told the model Manchester City would certainly fail to
    # score against them. City won 4-0.
    #
    # Such a team has no usable estimate, so it takes the league-neutral
    # value - exactly the rule already applied to a team with no history at
    # all. This introduces no tunable constant.
    degenerate_attack = attack <= 0.0
    degenerate_defence = defence <= 0.0

    attack = np.where(degenerate_attack, NEUTRAL_STRENGTH, attack)
    defence = np.where(degenerate_defence, NEUTRAL_STRENGTH, defence)

    return (
        {team: float(attack[index[team]]) for team in teams},
        {team: float(defence[index[team]]) for team in teams},
        float(home_multiplier),
        iterations,
        converged,
        int(degenerate_attack.sum() + degenerate_defence.sum()),
    )


def poisson_pmf_row(rate):
    """Poisson pmf over 0..MAX_GOALS for one rate. No scipy."""

    rate = max(float(rate), MIN_RATE)

    return np.exp(_GOALS * math.log(rate) - rate - _LOG_FACTORIAL)


def dixon_coles_tau(lambda_home, lambda_away, rho):
    """
    The low-score correction, as published. Returns the 2x2 block that
    multiplies scores where both sides scored at most one.
    """

    return np.array([
        [1.0 - lambda_home * lambda_away * rho, 1.0 + lambda_home * rho],
        [1.0 + lambda_away * rho, 1.0 - rho],
    ], dtype=float)


def score_matrix(lambda_home, lambda_away, rho):
    """
    Full joint distribution over scores 0..MAX_GOALS, Dixon-Coles corrected.

    Returned UNNORMALISED so the caller can measure the truncated tail mass
    before renormalising - a distribution silently rescaled is a distribution
    whose error you never see.
    """

    matrix = np.outer(
        poisson_pmf_row(lambda_home), poisson_pmf_row(lambda_away)
    )

    tau = dixon_coles_tau(lambda_home, lambda_away, rho)

    matrix[:2, :2] = matrix[:2, :2] * tau

    return np.maximum(matrix, 0.0)


def outcome_probabilities(lambda_home, lambda_away, rho):
    """
    Collapse the score matrix into [P(H), P(D), P(A)].

    Returns (probabilities, matrix_mass, truncated_mass).
    """

    matrix = score_matrix(lambda_home, lambda_away, rho)

    mass = float(matrix.sum())

    if mass <= 0:
        raise FatalError("score matrix carries no mass")

    home_win = float(np.tril(matrix, -1).sum())   # x > y
    draw = float(np.trace(matrix))                # x == y
    away_win = float(np.triu(matrix, 1).sum())    # x < y

    probabilities = np.array([home_win, draw, away_win]) / mass

    return probabilities, mass, 1.0 - mass


def match_rates(window, attack, defence, home_multiplier):
    """
    Expected goals for every match in a window. Vectorised.

    A team absent from the window has no fitted strength and takes the
    league-neutral value - the analogue of Elo's reset to 1500.
    """

    home_attack = window["home_team"].map(attack).fillna(NEUTRAL_STRENGTH)
    away_defence = window["away_team"].map(defence).fillna(NEUTRAL_STRENGTH)
    away_attack = window["away_team"].map(attack).fillna(NEUTRAL_STRENGTH)
    home_defence = window["home_team"].map(defence).fillna(NEUTRAL_STRENGTH)

    home_rates = np.maximum(
        (home_attack * away_defence * home_multiplier).to_numpy(dtype=float),
        MIN_RATE,
    )

    away_rates = np.maximum(
        (away_attack * home_defence).to_numpy(dtype=float), MIN_RATE
    )

    return home_rates, away_rates


def fit_rho(window, weights, attack, defence, home_multiplier):
    """
    Profile rho by deterministic bracketed golden section.

    PERFORMANCE, and why it is written this way: the expected-goals rates do
    not depend on rho, and the part of the log-likelihood that excludes tau
    is therefore constant across the search. Both are computed ONCE. Each
    objective evaluation then touches only the matches where both sides
    scored at most one - the only cells tau modifies.

    Dropping the rho-independent term is safe because argmin is unchanged by
    an additive constant.
    """

    home_rates, away_rates = match_rates(
        window, attack, defence, home_multiplier
    )

    home_goals = window["home_goals"].to_numpy()
    away_goals = window["away_goals"].to_numpy()

    low = (home_goals <= 1) & (away_goals <= 1)

    if not low.any():
        return 0.0

    low_home_rate = home_rates[low]
    low_away_rate = away_rates[low]
    low_weights = weights[low]

    both_zero = (home_goals[low] == 0) & (away_goals[low] == 0)
    home_zero = (home_goals[low] == 0) & (away_goals[low] == 1)
    away_zero = (home_goals[low] == 1) & (away_goals[low] == 0)
    both_one = (home_goals[low] == 1) & (away_goals[low] == 1)

    def objective(rho):

        tau = np.ones(low.sum(), dtype=float)

        tau[both_zero] = 1.0 - low_home_rate[both_zero] * low_away_rate[both_zero] * rho
        tau[home_zero] = 1.0 + low_home_rate[home_zero] * rho
        tau[away_zero] = 1.0 + low_away_rate[away_zero] * rho
        tau[both_one] = 1.0 - rho

        if np.any(tau <= 0):
            return np.inf

        return float(-np.sum(low_weights * np.log(tau)))

    grid = np.linspace(RHO_LOWER, RHO_UPPER, RHO_GRID_POINTS)

    losses = [objective(value) for value in grid]

    best = int(np.argmin(losses))

    if not np.isfinite(losses[best]):
        return 0.0

    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, len(grid) - 1)]

    invphi = (math.sqrt(5.0) - 1.0) / 2.0

    c = b - invphi * (b - a)
    d = a + invphi * (b - a)

    fc, fd = objective(c), objective(d)

    for _ in range(RHO_GOLDEN_ITERATIONS):

        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = objective(d)

    return float((a + b) / 2.0)


def fit_window(window, cutoff_date, dixon_coles=True):
    """
    Fit the complete model on one window of historical matches.

    dixon_coles=False pins rho to 0, which makes tau identically 1 and leaves
    the plain independent-Poisson model. Because estimation is two-stage -
    attack/defence/home first, rho profiled afterwards - the rates are
    IDENTICAL either way, and the two variants differ in exactly one thing:
    the low-score correction. That is what makes the comparison clean.
    """

    weights = time_weights(window["date"], cutoff_date)

    attack, defence, home_multiplier, iterations, converged, degenerate = (
        fit_attack_defence(window, weights)
    )

    rho = (
        fit_rho(window, weights, attack, defence, home_multiplier)
        if dixon_coles else 0.0
    )

    return {
        "attack": attack,
        "defence": defence,
        "home_multiplier": home_multiplier,
        "rho": rho,
        "matches": int(len(window)),
        "cutoff_date": cutoff_date,
        "iterations": iterations,
        "converged": converged,
        "degenerate_parameters": degenerate,
    }


def predict_matches(block, model):
    """Predict a block of matches from one fitted model."""

    attack = model["attack"]
    defence = model["defence"]

    home_rates, away_rates = match_rates(
        block, attack, defence, model["home_multiplier"]
    )

    rows = []

    for position, row in enumerate(block.itertuples()):

        home_known = row.home_team in attack
        away_known = row.away_team in attack

        lambda_home = float(home_rates[position])
        lambda_away = float(away_rates[position])

        probabilities, mass, truncated = outcome_probabilities(
            lambda_home, lambda_away, model["rho"]
        )

        rows.append({
            "match_id": row.match_id,
            "season": row.season,
            "date": row.date,
            "home": row.home_team,
            "away": row.away_team,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "rho": model["rho"],
            "fit_matches": model["matches"],
            "fit_cutoff_date": model["cutoff_date"],
            "home_has_history": bool(home_known),
            "away_has_history": bool(away_known),
            "score_matrix_mass": mass,
            "truncated_mass": truncated,
            "p_home": probabilities[0],
            "p_draw": probabilities[1],
            "p_away": probabilities[2],
            "actual_result": row.result,
        })

    return rows


# ============================================================
# FOLD EXECUTION
# ============================================================

def run_fold(matches, fold_spec, variant):
    """
    Produce every test prediction for one fold under one variant.

    *_static         one fit on the training seasons, used for the whole
                     test season. No test match ever enters the fit.

    *_walkforward    one fit per distinct test date, on every match with
                     date STRICTLY BEFORE that date. Same information set
                     Elo had.

    poisson_*        rho pinned to 0 (plain independent Poisson).
    dc_*             rho fitted (Dixon-Coles low-score correction).
    """

    train_seasons = list(fold_spec["train_seasons"])
    test_season = str(fold_spec["test_season"])

    train = matches[matches["season"].isin(train_seasons)]
    test = matches[matches["season"] == test_season]

    dixon_coles = uses_dixon_coles(variant)

    rows = []
    fits = []

    if is_static(variant):

        model = fit_window(train, train["date"].max(), dixon_coles)

        fits.append(model)

        rows.extend(predict_matches(test, model))

    else:

        for cutoff in sorted(test["date"].unique()):

            cutoff = pd.Timestamp(cutoff)

            # STRICT: date < cutoff. Same-day matches are never in the window.
            window = matches[
                (matches["season"].isin(train_seasons + [test_season]))
                & (matches["date"] < cutoff)
            ]

            model = fit_window(window, cutoff, dixon_coles)

            fits.append(model)

            rows.extend(predict_matches(test[test["date"] == cutoff], model))

    frame = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)

    return frame, fits


def run_variant(matches, spec, variant, base_rate, elo):

    base_by_fold = {int(row.fold): row for row in base_rate.itertuples()}
    elo_by_fold = {int(row.fold): row for row in elo.itertuples()}

    results = []
    summaries = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        frame, fits = run_fold(matches, fold_spec, variant)

        proba = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

        proba = validate_probabilities(proba, len(frame))

        frame["variant"] = variant
        frame["fold"] = fold
        frame["test_season"] = str(fold_spec["test_season"])
        frame["predicted_result"] = [CLASSES[i] for i in np.argmax(proba, axis=1)]

        results.append(frame)

        scores = evaluate(frame["actual_result"].to_numpy(), proba)

        base = base_by_fold[fold]
        reference = elo_by_fold[fold]

        neutral = int((
            ~frame["home_has_history"] | ~frame["away_has_history"]
        ).sum())

        summaries.append({
            "variant": variant,
            "fold": fold,
            "train_seasons": " + ".join(train_seasons_of(fold_spec)),
            "test_season": str(fold_spec["test_season"]),
            "train_matches": int(
                matches["season"].isin(fold_spec["train_seasons"]).sum()),
            "test_matches": int(len(frame)),
            "refits": len(fits),
            "max_goals": MAX_GOALS,
            "half_life_days": TIME_DECAY_HALF_LIFE_DAYS,
            "rho_first_fit": fits[0]["rho"],
            "rho_last_fit": fits[-1]["rho"],
            "home_advantage_multiplier": fits[-1]["home_multiplier"],
            "neutral_team_predictions": neutral,
            "degenerate_parameters": int(sum(f["degenerate_parameters"] for f in fits)),
            "accuracy": scores["accuracy"],
            "balanced_accuracy": scores["balanced_accuracy"],
            "macro_f1": scores["macro_f1"],
            "log_loss": scores["log_loss"],
            "brier_score": scores["brier_score"],
            "rps": scores["rps"],
            "base_rate_log_loss": float(base.log_loss),
            "base_rate_brier": float(base.brier_score),
            "base_rate_rps": float(base.rps),
            "elo_log_loss": float(reference.log_loss),
            "elo_brier": float(reference.brier_score),
            "log_loss_vs_base": scores["log_loss"] - float(base.log_loss),
            "log_loss_vs_elo": scores["log_loss"] - float(reference.log_loss),
            "brier_vs_base": scores["brier_score"] - float(base.brier_score),
            "brier_vs_elo": scores["brier_score"] - float(reference.brier_score),
            "rps_vs_base": scores["rps"] - float(base.rps),
            "beats_base_rate": bool(
                scores["log_loss"] < float(base.log_loss)
                and scores["brier_score"] < float(base.brier_score)),
            "beats_elo": bool(
                scores["log_loss"] < float(reference.log_loss)
                and scores["brier_score"] < float(reference.brier_score)),
        })

    return pd.concat(results, ignore_index=True), pd.DataFrame(summaries)


def train_seasons_of(fold_spec):
    return list(fold_spec["train_seasons"])


def build_everything(matches, spec, base_rate, elo):

    frames = []
    summaries = []

    for variant in VARIANTS:

        frame, summary = run_variant(matches, spec, variant, base_rate, elo)

        frames.append(frame)
        summaries.append(summary)

    results = pd.concat(frames, ignore_index=True)

    results = results.sort_values(
        ["variant", "fold", "date", "home"]
    ).reset_index(drop=True)

    results["date"] = pd.to_datetime(results["date"]).dt.strftime("%Y-%m-%d")
    results["fit_cutoff_date"] = pd.to_datetime(
        results["fit_cutoff_date"]).dt.strftime("%Y-%m-%d")

    return (
        results[RESULTS_COLUMNS],
        pd.concat(summaries, ignore_index=True)[SUMMARY_COLUMNS],
    )


# ============================================================
# TESTS
# ============================================================

def test_data_integrity(matches, results, audit):

    audit.record(
        "D1", "Exactly 1,900 matches consumed",
        "match foundation", EXPECTED_TOTAL_MATCHES, len(matches),
        len(matches) == EXPECTED_TOTAL_MATCHES,
    )

    for variant in VARIANTS:

        block = results[results["variant"] == variant]

        audit.record(
            f"D2.{variant}", "Exactly 1,520 test predictions",
            variant, EXPECTED_TOTAL_TEST, len(block),
            len(block) == EXPECTED_TOTAL_TEST,
        )

        duplicated = int(block.duplicated(
            subset=["test_season", "date", "home", "away"]).sum())

        audit.record(
            f"D3.{variant}", "Each test match appears exactly once",
            variant, 0, duplicated,
            duplicated == 0,
        )

    missing = int(results["home"].isna().sum() + results["away"].isna().sum())

    audit.record(
        "D4", "No missing teams",
        f"{len(results)} rows", 0, missing,
        missing == 0,
    )

    block = results[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

    non_finite = int((~np.isfinite(block)).sum())
    out_of_range = int(((block < 0.0) | (block > 1.0)).sum())

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

    # Every variant must evaluate on the SAME 1,520 matches as Elo did.
    identical = True

    reference = None

    for variant in VARIANTS:

        keys = set(map(tuple, results[results["variant"] == variant][
            ["test_season", "date", "home", "away"]].to_numpy()))

        if reference is None:
            reference = keys
        elif keys != reference:
            identical = False

    expected_keys = set(map(tuple, matches[
        matches["season"] != sorted(matches["season"].unique())[0]
    ].assign(date=lambda f: f["date"].dt.strftime("%Y-%m-%d"))[
        ["season", "date", "home_team", "away_team"]].to_numpy()))

    audit.record(
        "D7", "Every variant evaluates the identical 1,520 test matches",
        "the four frozen test seasons",
        "identical sets", "identical" if identical else "DIVERGED",
        identical and reference == expected_keys,
        "The same matches Elo and the base rate were scored on",
    )


def test_score_matrix(results, audit):
    """The distributional guarantees the brief asks for, checked directly."""

    # ---- P(H) + P(D) + P(A) == 1
    sums = results[["p_home", "p_draw", "p_away"]].sum(axis=1)

    off = int((np.abs(sums - 1.0) > PROBABILITY_TOLERANCE).sum())

    audit.record(
        "M1", "Every outcome vector sums to 1 within 1e-6",
        f"{len(results)} predictions", 0, off,
        off == 0,
        f"max deviation {float(np.abs(sums - 1.0).max()):.3e}",
    )

    # ---- the score matrix itself is a valid distribution, and H/D/A is
    # correctly read off it. Rebuilt independently from the stored lambdas.
    rebuild_errors = []
    worst_truncation = 0.0

    sample = results.sample(
        n=min(400, len(results)), random_state=0
    ) if len(results) > 400 else results

    for row in sample.itertuples():

        matrix = score_matrix(row.lambda_home, row.lambda_away, row.rho)

        if np.any(matrix < 0):
            rebuild_errors.append(f"{row.date} {row.home}: negative cell")
            continue

        mass = matrix.sum()

        normalised = matrix / mass

        if abs(normalised.sum() - 1.0) > 1e-12:
            rebuild_errors.append(f"{row.date} {row.home}: matrix != 1")
            continue

        # H/D/A read off the matrix by explicit comparison, not by tril/triu.
        home_win = sum(
            normalised[x, y]
            for x in range(MAX_GOALS + 1) for y in range(MAX_GOALS + 1) if x > y
        )
        draw = sum(normalised[x, x] for x in range(MAX_GOALS + 1))
        away_win = sum(
            normalised[x, y]
            for x in range(MAX_GOALS + 1) for y in range(MAX_GOALS + 1) if x < y
        )

        if (
            abs(home_win - row.p_home) > 1e-10
            or abs(draw - row.p_draw) > 1e-10
            or abs(away_win - row.p_away) > 1e-10
        ):
            rebuild_errors.append(f"{row.date} {row.home}: H/D/A mismatch")

        worst_truncation = max(worst_truncation, float(row.truncated_mass))

    audit.record(
        "M2", "Score matrix is a valid distribution and H/D/A reads off it",
        f"{len(sample)} predictions rebuilt independently",
        "0 errors", f"{len(rebuild_errors)} errors",
        not rebuild_errors,
        "; ".join(rebuild_errors[:5]),
    )

    audit.record(
        "M3", "Score-matrix truncation loses negligible mass",
        f"{len(results)} predictions, scores 0..{MAX_GOALS}",
        "< 1e-6 truncated",
        f"max {float(results['truncated_mass'].max()):.3e}",
        float(results["truncated_mass"].max()) < 1e-6,
        "Mass beyond the truncation is renormalised away, and measured first",
    )

    # ---- rates must be positive and plausible
    bad_rates = int((
        (results["lambda_home"] <= 0) | (results["lambda_away"] <= 0)
        | (results["lambda_home"] > 10) | (results["lambda_away"] > 10)
    ).sum())

    audit.record(
        "M4", "Every expected-goals rate is positive and plausible",
        f"{len(results)} predictions x 2 sides", 0, bad_rates,
        bad_rates == 0,
        f"home {results['lambda_home'].min():.3f}-"
        f"{results['lambda_home'].max():.3f}, away "
        f"{results['lambda_away'].min():.3f}-{results['lambda_away'].max():.3f}",
    )

    # ---- the model must give the home side an advantage on average
    audit.measure(
        "M5", "Mean expected goals",
        f"{len(results)} predictions",
        f"home {results['lambda_home'].mean():.3f}, "
        f"away {results['lambda_away'].mean():.3f}",
        "Home advantage appears as a multiplicative rate, not an Elo bonus",
    )

    audit.measure(
        "M6", "Fitted Dixon-Coles rho",
        "all fits",
        f"{results['rho'].min():.4f} to {results['rho'].max():.4f}",
        "Negative rho lifts 0-0 and 1-1 and suppresses 1-0 and 0-1, the "
        "low-score correction the model exists for",
    )


def test_dixon_coles(results, audit):
    """
    What the Dixon-Coles correction is allowed to do, checked directly.

    The point of running plain Poisson beside Dixon-Coles is that the two
    differ in ONE thing. These tests are what make that claim true rather
    than assumed.
    """

    plain = results[~results["variant"].isin(DIXON_COLES_VARIANTS)]
    corrected = results[results["variant"].isin(DIXON_COLES_VARIANTS)]

    # ---- C1: the plain-Poisson variants carry no correction at all
    nonzero = int((plain["rho"] != 0.0).sum())

    audit.record(
        "C1", "The plain-Poisson variants pin rho to exactly zero",
        f"{len(plain)} predictions", 0, nonzero,
        nonzero == 0,
        "tau is then identically 1 and the model is independent Poisson",
    )

    # ---- C2: the Dixon-Coles variants actually fit one, inside its bounds
    fitted = corrected["rho"].to_numpy(dtype=float)

    out_of_bounds = int(((fitted < RHO_LOWER) | (fitted > RHO_UPPER)).sum())

    audit.record(
        "C2", "The fitted rho is non-degenerate and inside its search bounds",
        f"{len(corrected)} predictions",
        f"non-zero, within [{RHO_LOWER}, {RHO_UPPER}]",
        f"{int((fitted != 0.0).sum())} non-zero, {out_of_bounds} out of bounds",
        bool((fitted != 0.0).any()) and out_of_bounds == 0,
        f"range {fitted.min():.4f} to {fitted.max():.4f}",
    )

    # ---- C3: rho = 0 must reproduce the independent Poisson product exactly
    product_errors = []

    sample = corrected.sample(
        n=min(200, len(corrected)), random_state=0
    ) if len(corrected) > 200 else corrected

    for row in sample.itertuples():

        independent = np.outer(
            poisson_pmf_row(row.lambda_home), poisson_pmf_row(row.lambda_away)
        )

        if not np.allclose(
            score_matrix(row.lambda_home, row.lambda_away, 0.0),
            independent, rtol=0, atol=1e-15,
        ):
            product_errors.append(f"{row.date} {row.home}")

    audit.record(
        "C3", "At rho = 0 the score matrix is exactly the Poisson product",
        f"{len(sample)} score matrices rebuilt",
        "0 errors", f"{len(product_errors)} errors",
        not product_errors,
        "; ".join(product_errors[:5]),
    )

    # ---- C4: the correction is confined to the four low-score cells
    spill = []
    moved_inside = 0

    for row in sample.itertuples():

        uncorrected = score_matrix(row.lambda_home, row.lambda_away, 0.0)
        adjusted = score_matrix(row.lambda_home, row.lambda_away, row.rho)

        difference = np.abs(adjusted - uncorrected)

        outside = difference.copy()
        outside[:2, :2] = 0.0

        if outside.max() > 1e-15:
            spill.append(f"{row.date} {row.home}: {outside.max():.3e}")

        if row.rho != 0.0 and difference[:2, :2].max() > 0:
            moved_inside += 1

    audit.record(
        "C4", "Dixon-Coles touches ONLY scores where both sides scored 0 or 1",
        f"{len(sample)} score matrices, cells outside the 2x2 block",
        "0 cells moved", f"{len(spill)} matrices moved",
        not spill,
        "; ".join(spill[:5]),
    )

    audit.record(
        "C5", "Positive control: it does move the four low-score cells",
        f"{len(sample)} score matrices",
        f"> 0 of {len(sample)}", f"{moved_inside} of {len(sample)} moved",
        moved_inside > 0,
        "Proves C4 is not vacuous - a correction that did nothing would pass "
        "C4 trivially",
    )

    # ---- C6: tau redistributes mass, it does not create or destroy it
    mass_errors = []

    for row in sample.itertuples():

        before = float(score_matrix(row.lambda_home, row.lambda_away, 0.0).sum())
        after = float(
            score_matrix(row.lambda_home, row.lambda_away, row.rho).sum())

        if abs(after - before) > 1e-12:
            mass_errors.append(f"{row.date} {row.home}: {after - before:.3e}")

    audit.record(
        "C6", "The correction conserves total probability mass",
        f"{len(sample)} score matrices",
        "0 mass changes", f"{len(mass_errors)} changes",
        not mass_errors,
        "tau is constructed so the four adjustments cancel exactly; the "
        "renormalisation therefore corrects truncation only, never tau",
    )

    # ---- C7: within a pair, the expected goals are identical. This is what
    # makes "does Dixon-Coles help" a question about the correction alone.
    rate_errors = []

    keys = ["fold", "test_season", "date", "home", "away"]

    for plain_variant, dc_variant in DC_PAIRS:

        left = results[results["variant"] == plain_variant].set_index(keys)
        right = results[results["variant"] == dc_variant].set_index(keys)

        if not left.index.equals(right.index):
            rate_errors.append(f"{plain_variant}/{dc_variant}: match sets differ")
            continue

        for column in ("lambda_home", "lambda_away", "fit_matches"):

            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                rtol=0, atol=1e-12,
            ):
                rate_errors.append(f"{plain_variant}/{dc_variant}: {column}")

    audit.record(
        "C7", "Each Poisson/Dixon-Coles pair shares identical expected goals",
        f"{len(DC_PAIRS)} pairs x {EXPECTED_TOTAL_TEST} matches",
        "0 differences", f"{len(rate_errors)} differences",
        not rate_errors,
        "Two-stage estimation fits attack/defence/home before rho, so the "
        "pair differs in the low-score correction and nothing else",
    )

    # ---- C8: and the outcome probabilities therefore DO differ
    changed_pairs = 0

    for plain_variant, dc_variant in DC_PAIRS:

        left = results[results["variant"] == plain_variant].set_index(keys)
        right = results[results["variant"] == dc_variant].set_index(keys)

        columns = ["p_home", "p_draw", "p_away"]

        if not np.allclose(
            left[columns].to_numpy(dtype=float),
            right[columns].to_numpy(dtype=float),
            rtol=0, atol=1e-12,
        ):
            changed_pairs += 1

    audit.record(
        "C8", "Positive control: the correction changes the H/D/A output",
        f"{len(DC_PAIRS)} pairs",
        f"{len(DC_PAIRS)} of {len(DC_PAIRS)} differ",
        f"{changed_pairs} of {len(DC_PAIRS)} differ",
        changed_pairs == len(DC_PAIRS),
        "Identical probabilities would mean the comparison measures nothing",
    )


def test_temporal_integrity(matches, spec, results, summary, base_rate, elo,
                            audit):
    """
    The strict cutoff, checked structurally on all 1,520, and causally on a
    sample by rebuilding.
    """

    # ---- M1: every fit used only strictly-earlier matches
    violations = []

    for row in results.itertuples():

        cutoff = pd.Timestamp(row.fit_cutoff_date)
        match_date = pd.Timestamp(row.date)

        if row.variant in WALKFORWARD_VARIANTS:

            if cutoff != match_date:
                violations.append(
                    f"{row.variant} {row.date} {row.home}: cutoff {cutoff.date()}"
                )

        else:

            if cutoff >= match_date:
                violations.append(
                    f"{row.variant} {row.date} {row.home}: cutoff not before"
                )

    audit.record(
        "T1", "Every prediction's fitting window ends at or before its own date",
        f"{len(results)} predictions",
        "0 violations", f"{len(violations)} violations",
        not violations,
        "Walk-forward fits at the match date on matches STRICTLY before it; "
        "static fits on the training seasons only",
    )

    # ---- the static variants must never see a test-season match
    static = results[results["variant"].isin(STATIC_VARIANTS)]

    static_leaks = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        block = static[static["fold"] == fold]

        train_end = matches[
            matches["season"].isin(fold_spec["train_seasons"])]["date"].max()

        cutoffs = set(pd.to_datetime(block["fit_cutoff_date"]))

        if cutoffs != {train_end}:
            static_leaks.append(f"fold {fold}: cutoffs {sorted(cutoffs)}")

        sizes = set(block["fit_matches"])

        if sizes != {EXPECTED_TRAIN_SIZES[fold - 1]}:
            static_leaks.append(f"fold {fold}: fit sizes {sorted(sizes)}")

    audit.record(
        "T2", "Both static variants fit on the training seasons alone",
        f"{EXPECTED_FOLDS} folds x {len(STATIC_VARIANTS)} static variants",
        "0 leaks", f"{len(static_leaks)} leaks",
        not static_leaks,
        "No test-season match enters the fit; window size equals the "
        "training size exactly",
    )

    # ---- walk-forward window sizes must grow, and never include the day
    walk = results[results["variant"].isin(WALKFORWARD_VARIANTS)]

    window_errors = []

    for row in walk.itertuples():

        cutoff = pd.Timestamp(row.fit_cutoff_date)

        expected = int((
            matches["date"] < cutoff
        ).sum() - (matches["date"] < cutoff).sum() + len(matches[
            (matches["date"] < cutoff)
            & (matches["season"] <= row.test_season)
        ]))

        if row.fit_matches != expected:
            window_errors.append(
                f"{row.date} {row.home}: {row.fit_matches} != {expected}"
            )

    audit.record(
        "T3", "Walk-forward windows hold exactly the strictly-earlier matches",
        f"{len(walk)} predictions",
        "0 errors", f"{len(window_errors)} errors",
        not window_errors,
        "; ".join(window_errors[:5]),
    )

    # ---- causal test: a match cannot influence its own prediction.
    # Run against the primary variant alone: the perturbation rebuild is the
    # expensive part of the instrument, and every variant shares one fitting
    # window, so rebuilding all four would repeat the same evidence.
    walk_primary = results[results["variant"] == PRIMARY_VARIANT]

    baseline = results.set_index(["variant", "fold", "date", "home"])

    sample_targets = []

    for fold_spec in spec["folds"]:

        test = matches[matches["season"] == fold_spec["test_season"]]

        for matchweek in (28,):

            block = test[test["matchweek"] == matchweek]

            if len(block):
                sample_targets.append(
                    (int(fold_spec["fold"]), fold_spec, block.iloc[0])
                )

    own_changed = 0
    later_changed = 0
    checks = 0

    for fold, fold_spec, target in sample_targets:

        perturbed = matches.copy()

        flipped = {"H": "A", "A": "H", "D": "H"}[target["result"]]

        mask = perturbed["match_id"] == target["match_id"]

        perturbed.loc[mask, "home_goals"] = 0 if flipped == "A" else 9
        perturbed.loc[mask, "away_goals"] = 9 if flipped == "A" else 0
        perturbed.loc[mask, "result"] = flipped

        frame, _ = run_fold(perturbed, fold_spec, VARIANT_WALKFORWARD)

        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")

        original = walk_primary[
            walk_primary["fold"] == fold].set_index(["date", "home"])
        rebuilt = frame.set_index(["date", "home"])

        target_date = target["date"].strftime("%Y-%m-%d")

        columns = ["lambda_home", "lambda_away", "p_home", "p_draw", "p_away"]

        key = (target_date, target["home_team"])

        checks += 1

        if not np.allclose(
            original.loc[key, columns].to_numpy(dtype=float),
            rebuilt.loc[key, columns].to_numpy(dtype=float),
            rtol=0, atol=1e-12,
        ):
            own_changed += 1

        # Everything strictly earlier must also be untouched.
        earlier = original.index.get_level_values("date") < target_date

        if earlier.any():
            if not np.allclose(
                original[earlier][columns].to_numpy(dtype=float),
                rebuilt.loc[original[earlier].index][columns].to_numpy(dtype=float),
                rtol=0, atol=1e-12,
            ):
                own_changed += 1

        # POSITIVE CONTROL: later predictions must move.
        later = original.index.get_level_values("date") > target_date

        if later.any():
            if not np.allclose(
                original[later][columns].to_numpy(dtype=float),
                rebuilt.loc[original[later].index][columns].to_numpy(dtype=float),
                rtol=0, atol=1e-12,
            ):
                later_changed += 1

    audit.record(
        "T4", "A match cannot influence its own prediction, nor any earlier one",
        f"{checks} matches perturbed to a flipped 9-0 scoreline",
        0, own_changed,
        own_changed == 0,
    )

    audit.record(
        "T5", "Positive control: LATER predictions do move",
        f"{checks} perturbations",
        f"> 0 of {checks}", f"{later_changed} of {checks} moved",
        later_changed > 0,
        "The result legitimately enters later fitting windows; a model that "
        "ignored its input would fail here",
    )

    # ---- fold isolation for the static variant: rewriting an entire test
    # season must leave its parameters untouched.
    static_changed = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        for forced_home, forced_away in ((9, 0), (0, 9)):

            perturbed = matches.copy()

            mask = perturbed["season"] == fold_spec["test_season"]

            perturbed.loc[mask, "home_goals"] = forced_home
            perturbed.loc[mask, "away_goals"] = forced_away
            perturbed.loc[mask, "result"] = "H" if forced_home > forced_away else "A"

            frame, fits = run_fold(perturbed, fold_spec, VARIANT_STATIC)

            original = summary[
                (summary["variant"] == VARIANT_STATIC)
                & (summary["fold"] == fold)
            ].iloc[0]

            if abs(fits[0]["rho"] - float(original["rho_first_fit"])) > 1e-12:
                static_changed.append(f"fold {fold}: rho moved")

            if abs(
                fits[0]["home_multiplier"]
                - float(original["home_advantage_multiplier"])
            ) > 1e-12:
                static_changed.append(f"fold {fold}: home multiplier moved")

    audit.record(
        "T6", "Perturbing a test season leaves dc_static's parameters untouched",
        f"{EXPECTED_FOLDS} folds x 2 rewrites",
        "0 changed", f"{len(static_changed)} changed",
        not static_changed,
        "; ".join(static_changed[:5]),
    )

    # ---- POSITIVE CONTROL for the static fit
    control = matches.copy()

    control.loc[control["season"] == "2021-2022", "home_goals"] = 5
    control.loc[control["season"] == "2021-2022", "away_goals"] = 0
    control.loc[control["season"] == "2021-2022", "result"] = "H"

    _, control_fits = run_fold(control, spec["folds"][0], VARIANT_STATIC)

    original = summary[
        (summary["variant"] == VARIANT_STATIC) & (summary["fold"] == 1)
    ].iloc[0]

    moved = abs(
        control_fits[0]["home_multiplier"]
        - float(original["home_advantage_multiplier"])
    ) > 1e-9

    audit.record(
        "T7", "Positive control: perturbing TRAINING data does move the fit",
        "fold 1 training season rewritten",
        "parameters move", "moved" if moved else "UNCHANGED",
        bool(moved),
        "Proves T6 is not vacuous",
    )


def test_fold_structure(matches, spec, summary, results, audit):

    audit.record(
        "F1", "Exactly four folds from the frozen Phase 0 specification",
        "spec JSON", EXPECTED_FOLDS, len(spec["folds"]),
        len(spec["folds"]) == EXPECTED_FOLDS,
    )

    folds_csv = pd.read_csv(FOLDS_CSV, float_precision=FLOAT_PRECISION)

    disagreements = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        expected_train = " + ".join(fold_spec["train_seasons"])

        row = folds_csv[folds_csv["fold"] == fold]

        if row.empty or str(row.iloc[0]["train_seasons"]) != expected_train:
            disagreements.append(f"fold {fold} spec/CSV")

        for variant in VARIANTS:

            used = summary[
                (summary["variant"] == variant) & (summary["fold"] == fold)
            ].iloc[0]

            if used["train_seasons"] != expected_train:
                disagreements.append(f"{variant} fold {fold} train")

            if used["test_season"] != str(fold_spec["test_season"]):
                disagreements.append(f"{variant} fold {fold} test")

    audit.record(
        "F2", "Fold definitions unchanged from Phase 0",
        f"spec, folds CSV and all {len(VARIANTS)} variants",
        "0 disagreements", f"{len(disagreements)} disagreements",
        not disagreements,
        "; ".join(disagreements[:5]),
    )

    for variant in VARIANTS:

        block = summary[summary["variant"] == variant].sort_values("fold")

        audit.record(
            f"F3.{variant}", "Training sizes are 380 / 760 / 1140 / 1520",
            variant, str(EXPECTED_TRAIN_SIZES),
            str(list(block["train_matches"])),
            list(block["train_matches"]) == EXPECTED_TRAIN_SIZES,
        )

        audit.record(
            f"F4.{variant}", "Every test set holds exactly 380 matches",
            variant, f"{EXPECTED_TEST_SIZE} x {EXPECTED_FOLDS}",
            str(list(block["test_matches"])),
            list(block["test_matches"]) == [EXPECTED_TEST_SIZE] * EXPECTED_FOLDS,
        )

    seasons_tested = sorted(results["test_season"].unique())

    audit.record(
        "F5", "The four test seasons are exactly those in the frozen spec",
        f"all {len(VARIANTS)} variants",
        "['2022-2023', '2023-2024', '2024-2025', '2025-2026']",
        str(seasons_tested),
        seasons_tested == ["2022-2023", "2023-2024", "2024-2025", "2025-2026"],
    )


def test_output_contract(results, summary, audit):

    block = results[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

    expected = [CLASSES[i] for i in np.argmax(block, axis=1)]

    wrong = int((results["predicted_result"].to_numpy() != np.array(expected)).sum())

    audit.record(
        "O1", "predicted_result is argmax([p_home, p_draw, p_away])",
        f"{len(results)} predictions", 0, wrong,
        wrong == 0,
    )

    mismatches = []

    for (variant, fold), group in results.groupby(["variant", "fold"]):

        proba = group[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)

        recomputed = evaluate(group["actual_result"].to_numpy(), proba)

        row = summary[
            (summary["variant"] == variant) & (summary["fold"] == fold)
        ].iloc[0]

        for metric in METRIC_NAMES:
            if not np.isclose(float(row[metric]), recomputed[metric], atol=1e-12):
                mismatches.append(f"{variant} fold {fold} {metric}")

    audit.record(
        "O2", "Fold metrics reproduce from the results table",
        f"{len(VARIANTS)} variants x {EXPECTED_FOLDS} folds x {len(METRIC_NAMES)} metrics",
        "0 mismatches", f"{len(mismatches)} mismatches",
        not mismatches,
        "; ".join(mismatches[:5]),
    )

    audit.record(
        "O3", "Metrics come from the Phase 0 harness, not a private copy",
        "scoring", "phase0_evaluation_harness", evaluate.__module__,
        evaluate.__module__ == "phase0_evaluation_harness",
    )


def test_determinism(matches, spec, base_rate, elo, results, summary, audit):

    repeat_results, repeat_summary = build_everything(
        matches, spec, base_rate, elo
    )

    results_identical = results.equals(repeat_results)
    summary_identical = summary.equals(repeat_summary)

    audit.record(
        "O4", "Running the pipeline twice produces identical outputs",
        "full rebuild in-process", "identical",
        f"results {'identical' if results_identical else 'DIFFER'}, "
        f"summary {'identical' if summary_identical else 'DIFFER'}",
        results_identical and summary_identical,
        "Iterative scaling and the rho search are both fully deterministic",
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
        "; ".join(unexpected[:5]),
    )

    features_path = (OUTPUTS_DIR / "phase1_team_strength_features.csv").resolve()

    read_as_input = features_path in input_data_files

    audit.record(
        "P3", "The 86 engineered features were never read as a model input",
        "phase1_team_strength_features.csv",
        "0 input reads", f"{int(read_as_input)} input reads",
        not read_as_input,
        "Poisson/DC derives its rates from historical scorelines alone",
    )

    after_state = frozen_state()

    changed = sorted(
        name for name in before_state
        if before_state[name] != after_state.get(name)
    )

    audit.record(
        "P4", "Phase 0 and Phase 1 outputs and scripts are unmodified",
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

        print(f"  {markers[row.status]}  {row.test_id:<22} {row.test}")
        print(f"              scope   : {row.scope}")
        print(f"              expected: {row.expected}")
        print(f"              observed: {row.observed}")

        if row.detail:
            print(f"              {row.detail}")


def print_specification(summary, results):

    print()
    print("=" * 79)
    print("MODEL SPECIFICATION")
    print("=" * 79)
    print()
    print("  lambda_home = A_home * D_away * H      lambda_away = A_away * D_home")
    print("  P(x,y) = tau(x,y) * Poisson(x; lambda_home) * Poisson(y; lambda_away)")
    print()
    print("  Locked, not tuned against any test season:")
    print(f"    max goals per side     {MAX_GOALS}")
    print(f"    time-decay half-life   {TIME_DECAY_HALF_LIFE_DAYS:.0f} days"
          f"   (Dixon & Coles xi = 0.0065/day)")
    print(f"    rho search bounds      [{RHO_LOWER}, {RHO_UPPER}]")
    print()
    print("  Fitted per window by multiplicative iterative scaling, then rho")
    print("  profiled by bracketed golden section. No scipy, no randomness.")
    print()
    print(
        f"    {'Variant':<22}{'Fold':<6}{'Refits':>7}{'Fit size':>10}"
        f"{'rho':>9}{'H mult':>9}{'neutral':>9}"
    )

    for row in summary.itertuples():

        fits = results[
            (results["variant"] == row.variant) & (results["fold"] == row.fold)
        ]

        print(
            f"    {row.variant:<22}{row.fold:<6}{row.refits:>7}"
            f"{int(fits['fit_matches'].max()):>10}"
            f"{row.rho_last_fit:>9.4f}{row.home_advantage_multiplier:>9.4f}"
            f"{row.neutral_team_predictions:>9}"
        )

    print()
    print("  rho = 0.0000 on the poisson_* rows is pinned, not fitted: those")
    print("  variants are plain independent Poisson, and exist so that the")
    print("  Dixon-Coles correction can be judged on its own contribution.")

    print()
    print("  'neutral' counts predictions where a side had no history in the")
    print("  fitting window and was given league-average strength - the direct")
    print("  analogue of Elo's reset to 1500. Nothing is fabricated.")


def print_metrics(summary, overall_by_variant, base_overall, elo_overall):

    print()
    print("=" * 79)
    print("SIX METRICS PER FOLD")
    print("=" * 79)

    for variant in VARIANTS:

        block = summary[summary["variant"] == variant]

        print()
        print(f"  {variant}")
        print(
            f"    {'Fold':<5}{'Test season':<13}{'N':>5}{'Acc':>9}{'BalAcc':>9}"
            f"{'MacroF1':>9}{'LogLoss':>9}{'Brier':>9}{'RPS':>9}"
        )

        for row in block.itertuples():
            print(
                f"    {row.fold:<5}{row.test_season:<13}{row.test_matches:>5}"
                f"{row.accuracy:>9.4f}{row.balanced_accuracy:>9.4f}"
                f"{row.macro_f1:>9.4f}{row.log_loss:>9.4f}"
                f"{row.brier_score:>9.4f}{row.rps:>9.4f}"
            )

        overall = overall_by_variant[variant]

        print(
            f"    {'ALL':<5}{'1,520':<13}{overall['n']:>5}"
            f"{overall['accuracy']:>9.4f}{overall['balanced_accuracy']:>9.4f}"
            f"{overall['macro_f1']:>9.4f}{overall['log_loss']:>9.4f}"
            f"{overall['brier_score']:>9.4f}{overall['rps']:>9.4f}"
        )

    print()
    print("=" * 79)
    print("DOES THE DIXON-COLES CORRECTION EARN ITS PLACE?")
    print("=" * 79)
    print()
    print("  Same rates, same fitting window, same test matches. The only")
    print("  difference is tau. Negative delta = the correction helped.")
    print()
    print(
        f"    {'Pair':<24}{'Fold':<6}{'LogLoss':>10}{'delta':>9}"
        f"{'Brier':>10}{'delta':>9}{'RPS':>10}{'delta':>9}"
    )

    for plain_variant, dc_variant in DC_PAIRS:

        plain_block = summary[summary["variant"] == plain_variant
                              ].set_index("fold")
        dc_block = summary[summary["variant"] == dc_variant].set_index("fold")

        label = dc_variant.replace("dc_", "")

        for fold in sorted(dc_block.index):

            before = plain_block.loc[fold]
            after = dc_block.loc[fold]

            print(
                f"    {label:<24}{fold:<6}"
                f"{after['log_loss']:>10.4f}"
                f"{after['log_loss'] - before['log_loss']:>+9.4f}"
                f"{after['brier_score']:>10.4f}"
                f"{after['brier_score'] - before['brier_score']:>+9.4f}"
                f"{after['rps']:>10.4f}"
                f"{after['rps'] - before['rps']:>+9.4f}"
            )

        plain_overall = overall_by_variant[plain_variant]
        dc_overall = overall_by_variant[dc_variant]

        print(
            f"    {label:<24}{'ALL':<6}"
            f"{dc_overall['log_loss']:>10.4f}"
            f"{dc_overall['log_loss'] - plain_overall['log_loss']:>+9.4f}"
            f"{dc_overall['brier_score']:>10.4f}"
            f"{dc_overall['brier_score'] - plain_overall['brier_score']:>+9.4f}"
            f"{dc_overall['rps']:>10.4f}"
            f"{dc_overall['rps'] - plain_overall['rps']:>+9.4f}"
        )
        print()

    print()
    print("=" * 79)
    print("HEAD TO HEAD PER FOLD - versus the base rate and Elo v1")
    print("=" * 79)
    print()
    print(
        f"    {'Variant':<22}{'Fold':<6}{'LogLoss':>10}{'Base':>9}{'delta':>9}"
        f"{'Elo':>9}{'delta':>9}   beats base   beats Elo"
    )

    for row in summary.itertuples():
        print(
            f"    {row.variant:<22}{row.fold:<6}{row.log_loss:>10.4f}"
            f"{row.base_rate_log_loss:>9.4f}{row.log_loss_vs_base:>+9.4f}"
            f"{row.elo_log_loss:>9.4f}{row.log_loss_vs_elo:>+9.4f}"
            f"   {'yes' if row.beats_base_rate else 'NO':<11}"
            f"{'yes' if row.beats_elo else 'NO'}"
        )

    print()
    print("  Negative delta = the goal model is better. Both metrics are")
    print("  losses. 'beats' requires BOTH log loss and Brier to improve.")


def print_comparison(overall_by_variant, base_overall, elo_overall, results):

    uniform = float(np.log(3.0))

    print()
    print("=" * 79)
    print("MODEL COMPARISON - 1,520 UNTOUCHED TEST MATCHES")
    print("=" * 79)
    print()
    print(
        f"    {'Model':<24}{'Accuracy':>10}{'BalAcc':>9}{'MacroF1':>9}"
        f"{'LogLoss':>10}{'Brier':>9}{'RPS':>9}"
    )

    print(f"    {'Uniform':<24}{1/3:>10.4f}{1/3:>9.4f}{'':>9}"
          f"{uniform:>10.4f}{2/3:>9.4f}{2/9:>9.4f}")

    print(f"    {'Base Rate':<24}{base_overall['accuracy']:>10.4f}"
          f"{base_overall['balanced_accuracy']:>9.4f}"
          f"{base_overall['macro_f1']:>9.4f}"
          f"{base_overall['log_loss']:>10.4f}"
          f"{base_overall['brier_score']:>9.4f}"
          f"{base_overall['rps']:>9.4f}")

    print(f"    {'Elo v1':<24}{elo_overall['accuracy']:>10.4f}"
          f"{elo_overall['balanced_accuracy']:>9.4f}"
          f"{elo_overall['macro_f1']:>9.4f}"
          f"{elo_overall['log_loss']:>10.4f}"
          f"{elo_overall['brier_score']:>9.4f}"
          f"{'n/a':>9}")

    for variant in VARIANTS:

        overall = overall_by_variant[variant]

        label = (
            "Poisson " if not uses_dixon_coles(variant) else "Dixon-Coles "
        ) + variant.split("_", 1)[1]

        print(f"    {label:<24}{overall['accuracy']:>10.4f}"
              f"{overall['balanced_accuracy']:>9.4f}"
              f"{overall['macro_f1']:>9.4f}{overall['log_loss']:>10.4f}"
              f"{overall['brier_score']:>9.4f}{overall['rps']:>9.4f}")

    print()
    print("  Elo v1's frozen summary predates RPS and is not back-filled.")

    # ---- the ladder the brief actually asks about: base rate -> Poisson ->
    # Dixon-Coles. Each rung is judged against the rung below it, not against
    # the best number on the page.
    print()

    for plain_variant, dc_variant in DC_PAIRS:

        plain_overall = overall_by_variant[plain_variant]
        dc_overall = overall_by_variant[dc_variant]

        rung_one = base_overall["log_loss"] - plain_overall["log_loss"]
        rung_two = plain_overall["log_loss"] - dc_overall["log_loss"]

        print(f"    {plain_variant.split('_', 1)[1]}:")
        print(f"      base rate -> Poisson       log loss {rung_one:+.4f}")
        print(f"      Poisson   -> Dixon-Coles   log loss {rung_two:+.4f}")

    print()
    print("    (positive = the higher rung improves on the one below it)")

    primary = overall_by_variant[PRIMARY_VARIANT]

    log_loss_gain = elo_overall["log_loss"] - primary["log_loss"]
    brier_gain = elo_overall["brier_score"] - primary["brier_score"]

    print()
    print(f"    {PRIMARY_VARIANT} vs Elo v1   log loss {log_loss_gain:+.4f}   "
          f"brier {brier_gain:+.4f}")
    print("    (positive = Poisson/DC improves on Elo)")

    print()

    if log_loss_gain > 0 and brier_gain > 0:
        print("  VERDICT: modelling goals produces better-calibrated probabilities")
        print("  than modelling team strength alone.")
    elif log_loss_gain > 0 or brier_gain > 0:
        print("  VERDICT: modelling goals improves one probabilistic metric but")
        print("  not both. A partial result worth understanding before moving on.")
    else:
        print("  VERDICT: modelling goals does NOT beat Elo on calibration.")
        print("  A real finding: the extra machinery of a full score matrix does")
        print("  not, in this simple form, out-predict a single team rating.")

    predicted = sorted(results[results["variant"] == PRIMARY_VARIANT][
        "predicted_result"].unique())

    print()
    print(f"  Classes {PRIMARY_VARIANT} ever predicts: {predicted}")

    draws = int((results[results["variant"] == PRIMARY_VARIANT][
        "predicted_result"] == "D").sum())

    print(f"  Draw predictions: {draws} of {EXPECTED_TOTAL_TEST}")

    if draws == 0:
        print("  Still no draw is ever the argmax - in football the draw is")
        print("  rarely the single most likely outcome, even when well modelled.")


# ============================================================
# MAIN
# ============================================================

def run():

    print()
    print("=" * 79)
    print("PHASE 2 - INSTRUMENT 3: POISSON / DIXON-COLES (v1)")
    print("=" * 79)
    print()
    print(f"  Matches    : {MATCHES_CSV.relative_to(PROJECT_ROOT)} (frozen)")
    print("  Metrics    : imported from phase0_evaluation_harness")
    print("  Model      : attack/defence/home Poisson + Dixon-Coles low-score rho")
    print("  Variants   : poisson_static / dc_static  (training seasons only)")
    print("               poisson_walkforward / dc_walkforward")
    print("               (refit before every test date)")
    print("  Ladder     : base rate -> Poisson -> Dixon-Coles")
    print("  Estimation : multiplicative iterative scaling + golden-section rho")
    print("  Not used   : the 86 engineered features, xG, FBref aggregates")

    before_state = frozen_state()

    spec, matches, base_rate, elo = load_inputs()

    audit = Audit()

    print()
    print(f"  {len(matches)} matches, {len(spec['folds'])} folds, "
          f"{len(VARIANTS)} variants.")
    print("  Fitting ... (walk-forward refits once per test date)")

    results, summary = build_everything(matches, spec, base_rate, elo)

    print(f"  {len(results)} predictions across all variants "
          f"({int(summary['refits'].sum())} model fits).")

    print("  D   data integrity ...")
    test_data_integrity(matches, results, audit)

    print("  M   score matrix and distributional guarantees ...")
    test_score_matrix(results, audit)

    print("  C   Dixon-Coles correction behaviour ...")
    test_dixon_coles(results, audit)

    print("  F   fold structure ...")
    test_fold_structure(matches, spec, summary, results, audit)

    print("  O   output contract ...")
    test_output_contract(results, summary, audit)

    print("  T   temporal integrity and perturbation controls ...")
    test_temporal_integrity(matches, spec, results, summary, base_rate, elo,
                            audit)

    print("  O4  determinism ...")
    test_determinism(matches, spec, base_rate, elo, results, summary, audit)

    print("  P   provenance and frozen-state guard ...")
    test_isolation(before_state, audit)

    # ---- overall metrics
    overall_by_variant = {}

    for variant in VARIANTS:

        block = results[results["variant"] == variant]

        overall_by_variant[variant] = evaluate(
            block["actual_result"].to_numpy(),
            block[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float),
        )

    weights = base_rate["test_matches"].to_numpy(dtype=float)

    base_overall = {
        metric: float(np.average(
            base_rate[metric].to_numpy(dtype=float), weights=weights))
        for metric in METRIC_NAMES
    }

    elo_overall = {
        metric: float(np.average(
            elo[metric].to_numpy(dtype=float), weights=weights))
        for metric in ELO_METRIC_NAMES
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(RESULTS_OUTPUT, index=False, encoding="utf-8",
                   float_format=FLOAT_FORMAT)
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8",
                   float_format=FLOAT_FORMAT)

    audit_frame = audit.frame()
    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    print_test_table(audit)
    print_specification(summary, results)
    print_metrics(summary, overall_by_variant, base_overall, elo_overall)
    print_comparison(overall_by_variant, base_overall, elo_overall, results)

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {RESULTS_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(results)} predictions, {len(VARIANTS)} variants)")
    print(f"  {SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(summary)} variant-folds)")
    print(f"  {AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(audit_frame)} entries)")

    failures = audit.failures()

    def outcome(prefix):
        rows = [r for r in audit.rows if r["test_id"].startswith(prefix)]
        return status_text(all(r["status"] != "FAIL" for r in rows))

    print()
    print("=" * 79)
    print("PHASE 2 - INSTRUMENT 3 STATUS")
    print("=" * 79)
    print()

    line("Matches:", f"{len(matches)}")
    line("Predictions:",
         f"{len(results)}  ({EXPECTED_TOTAL_TEST} x {len(VARIANTS)} variants)")
    line("Model fits:", f"{int(summary['refits'].sum())}")
    line("D  data integrity:", "same 1,520 test matches", outcome("D"))
    line("M  score matrix:", "valid distribution", outcome("M"))
    line("C  Dixon-Coles:", "confined + mass-conserving", outcome("C"))
    line("F  fold structure:", "frozen Phase 0 folds", outcome("F"))
    line("O  output contract:", "argmax, metrics, determinism", outcome("O"))
    line("T  temporal integrity:", "cutoff + both controls", outcome("T"))
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
