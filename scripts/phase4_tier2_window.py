"""
===============================================================================
PHASE 4 - TIER 2 RESOLUTION
IS THE STATIC-VS-WALKFORWARD GAP RECENCY, OR IS IT SAMPLE SIZE?
===============================================================================

Governed by PHASE4_TIER2_WINDOW_PREDECLARATION.txt, written and signed off
before anything here was fitted.

THE CONFOUND

    Phase 2 measured Dixon-Coles static at 1.0263 and walk-forward at 0.9904.
    That 0.036 is four times the 0.009 between the best dynamic model and the
    best results-only one, which looks like an argument that within-season
    updating is where the signal is.

    It is not one yet, because a static fit is degraded against a walk-forward
    fit by two mechanisms at once:

        RECENCY   strength drifts within a season and a frozen fit cannot
                  track it
        SAMPLE    walk-forward simply has more matches, N growing to N + 379

    Only RECENCY argues for within-season updating. SAMPLE is just more data,
    which the ratings-as-features route gets too.

THE ARMS

    A  static             N_train,           stale     (Phase 2's dc_static)
    B  walk-forward       N_train + k,       fresh     (Phase 2's dc_walkforward)
    C  rolling fixed-size N_train,           fresh     NEW - the discriminator
    D  static half-window floor(N_train/2),  stale     NEW - control

    C holds the sample at A's size and moves the endpoint to B's freshness, so

        Delta_total   = LL(A) - LL(B)
        Delta_recency = LL(A) - LL(C)     staleness at constant sample
        Delta_sample  = LL(C) - LL(B)     sample at constant freshness

        Delta_total = Delta_recency + Delta_sample, exactly (W5)

    The brief originally asked for a fixed-size window matching walk-forward's
    match COUNT while ending before the test season. That is unsatisfiable:
    walk-forward at test match k uses N_train + k, and matching it before the
    test season needs 1,899 pre-test matches at fold 4 against 1,520 that
    exist. Holding the sample and moving the endpoint tests the same mechanism
    and yields an exact additive split instead of a collapse-or-hold verdict.

NO SECOND POISSON IS WRITTEN

    fit_window, predict_matches, time_weights, the score matrix and every
    class-A constant are imported from phase2_poisson_dixon_coles. W1 proves
    it by reproducing that instrument's stored dc_static and dc_walkforward
    numbers to < 1e-9.
===============================================================================
"""

from pathlib import Path
import hashlib
import sys
import time

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import (  # noqa: E402
    CLASS_INDEX,
    evaluate,
    validate_probabilities,
)
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase2_poisson_dixon_coles as P2  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

PREDECLARATION = PROJECT_ROOT / "PHASE4_TIER2_WINDOW_PREDECLARATION.txt"
P2_SUMMARY = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"

PREDICTIONS_OUTPUT = OUTPUTS_DIR / "phase4_tier2_predictions.csv"
FOLD_OUTPUT = OUTPUTS_DIR / "phase4_tier2_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase4_tier2_pooled.csv"
DECOMPOSITION_OUTPUT = OUTPUTS_DIR / "phase4_tier2_decomposition.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase4_tier2_audit.csv"

FLOAT_PRECISION = "round_trip"

METRICS = ["accuracy", "balanced_accuracy", "macro_f1",
           "log_loss", "brier_score", "rps"]

PRIMARY_METRIC = "log_loss"
SECONDARY_METRIC = "rps"

# ---- THE ARMS.  Declared in the pre-declaration, asserted by W-tests. ----
ARM_STATIC = "A_static"
ARM_WALKFORWARD = "B_walkforward"
ARM_ROLLING = "C_rolling"
ARM_HALF = "D_static_half"

ARMS = (ARM_STATIC, ARM_WALKFORWARD, ARM_ROLLING, ARM_HALF)

ARM_LABEL = {
    ARM_STATIC: "static, N_train, stale",
    ARM_WALKFORWARD: "walk-forward, N_train+k, fresh",
    ARM_ROLLING: "rolling fixed-size, N_train, fresh",
    ARM_HALF: "static half-window, N_train/2, stale",
}

# Arms A and B must equal these stored Phase 2 variants exactly (W1).
ANCHOR_VARIANT = {ARM_STATIC: "dc_static", ARM_WALKFORWARD: "dc_walkforward"}

# ---- BOOTSTRAP.  Declared before the run. ----
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_INTERVAL = 95.0

# ---- THE DECISION RULE.  Declared before the run. ----
SHARE_RECENCY_UPDATING = 0.6      # >= this -> within-season updating
SHARE_RECENCY_FEATURES = 0.4      # <= this -> ratings-as-features

USE_DIXON_COLES = True            # rho fitted; the 0.036 is a DC number


# ============================================================
# FROZEN STATE
# ============================================================

def hash_file(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# The integrity hasher opens every file under data/raw, twice - before the run
# and after it. That is not the model reading raw data, and W9b has to be able
# to tell the two apart or it reports its own integrity check as a violation.
# Phase 2 already provides the mechanism: every open inside this context is
# labelled, and W9b asks only about opens labelled "input".
HASH_CONTEXT = "integrity_hash"


def frozen_state():

    state = {}

    with P2.access_context(HASH_CONTEXT):

        for pattern in ("phase[0123]_*",):
            for path in sorted(OUTPUTS_DIR.glob(pattern)):
                if path.is_file():
                    state[str(path)] = hash_file(path)

        for path in sorted(SCRIPTS_DIR.glob("phase*.py")):
            state[str(path)] = hash_file(path)

        for path in sorted(PROJECT_ROOT.glob("PHASE*PREDECLARATION.txt")):
            state[str(path)] = hash_file(path)

        if RAW_DIR.exists():
            for path in sorted(RAW_DIR.rglob("*")):
                if path.is_file():
                    state[str(path)] = hash_file(path)

    return state


# ============================================================
# THE WINDOWS
# ============================================================

def window_sizes(fold_spec, matches):
    """W_C and W_D for a fold. Declared rule, section 4 of the pre-declaration."""

    train_seasons = list(fold_spec["train_seasons"])
    n_train = int(matches["season"].isin(train_seasons).sum())

    return n_train, n_train // 2


def most_recent(pool, size):
    """
    The `size` most recent matches of `pool`, by the frozen match ordering.

    match_id is assigned by load_inputs() after sorting on
    (season, date, home_team, away_team), so "most recent" is that order's
    tail and no re-sort happens here. Re-sorting would be a second change
    riding along with the first.
    """

    return pool.sort_values("match_id").tail(size)


def run_fold_arm(matches, fold_spec, arm):
    """Every test prediction for one fold under one arm."""

    train_seasons = list(fold_spec["train_seasons"])
    test_season = str(fold_spec["test_season"])

    train = matches[matches["season"].isin(train_seasons)]
    test = matches[matches["season"] == test_season]

    w_c, w_d = window_sizes(fold_spec, matches)

    rows = []
    windows = []

    if arm == ARM_STATIC:

        window = train
        model = P2.fit_window(window, window["date"].max(), USE_DIXON_COLES)
        rows.extend(P2.predict_matches(test, model))
        windows.append((window, model, test["date"].min()))

    elif arm == ARM_HALF:

        window = most_recent(train, w_d)
        model = P2.fit_window(window, window["date"].max(), USE_DIXON_COLES)
        rows.extend(P2.predict_matches(test, model))
        windows.append((window, model, test["date"].min()))

    else:

        for cutoff in sorted(test["date"].unique()):

            cutoff = pd.Timestamp(cutoff)

            # STRICT: date < cutoff. A same-day match never enters its own fit.
            pool = matches[
                (matches["season"].isin(train_seasons + [test_season]))
                & (matches["date"] < cutoff)
            ]

            window = pool if arm == ARM_WALKFORWARD else most_recent(pool, w_c)

            model = P2.fit_window(window, cutoff, USE_DIXON_COLES)

            rows.extend(P2.predict_matches(test[test["date"] == cutoff], model))
            windows.append((window, model, cutoff))

    frame = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)
    frame.insert(0, "arm", arm)
    frame.insert(1, "fold", int(fold_spec["fold"]))

    return frame, windows


def run_everything(matches, spec):

    predictions = []
    window_log = {}

    for arm in ARMS:

        started = time.time()

        for fold_spec in spec["folds"]:

            frame, windows = run_fold_arm(matches, fold_spec, arm)

            predictions.append(frame)
            window_log[(arm, int(fold_spec["fold"]))] = windows

        print("    {:<16} {:>6.1f}s".format(arm, time.time() - started))

    return pd.concat(predictions, ignore_index=True), window_log


# ============================================================
# SCORING
# ============================================================

def per_match_log_loss(frame):
    """-log p(actual). The paired unit the bootstrap resamples."""

    columns = {"H": "p_home", "D": "p_draw", "A": "p_away"}

    taken = np.array([
        float(row[columns[row["actual_result"]]])
        for _i, row in frame.iterrows()
    ])

    return -np.log(np.clip(taken, 1e-15, 1.0))


def per_match_rps(frame):
    """Per-match ranked probability score, ordered H > D > A."""

    proba = frame[["p_home", "p_draw", "p_away"]].to_numpy(float)

    onehot = np.zeros_like(proba)
    onehot[np.arange(len(frame)),
           [CLASS_INDEX[r] for r in frame["actual_result"]]] = 1.0

    cumulative = np.cumsum(proba, axis=1) - np.cumsum(onehot, axis=1)

    return np.sum(cumulative[:, :-1] ** 2, axis=1) / 2.0


def score_arm(frame):

    proba = frame[["p_home", "p_draw", "p_away"]].to_numpy(float)
    validate_probabilities(proba, len(frame))

    return evaluate(frame["actual_result"].to_numpy(), proba)


def fold_summary(predictions):

    rows = []

    for arm in ARMS:
        for fold in sorted(predictions["fold"].unique()):

            block = predictions[(predictions["arm"] == arm)
                                & (predictions["fold"] == fold)]

            row = {"arm": arm, "arm_label": ARM_LABEL[arm], "fold": int(fold),
                   "test_matches": len(block),
                   "fit_matches_min": int(block["fit_matches"].min()),
                   "fit_matches_max": int(block["fit_matches"].max())}
            row.update({m: score_arm(block)[m] for m in METRICS})

            rows.append(row)

    return pd.DataFrame(rows)


def pooled_summary(predictions):

    rows = []

    for arm in ARMS:

        block = predictions[predictions["arm"] == arm].sort_values("match_id")

        row = {"arm": arm, "arm_label": ARM_LABEL[arm],
               "matches": len(block)}
        row.update({m: score_arm(block)[m] for m in METRICS})

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# THE PAIRED BOOTSTRAP
# ============================================================

def aligned_losses(predictions, metric):
    """arm -> per-match loss vector, all aligned on match_id."""

    per_arm = {}

    reference = None

    for arm in ARMS:

        block = predictions[predictions["arm"] == arm].sort_values("match_id")

        ids = block["match_id"].to_numpy()

        if reference is None:
            reference = ids
        elif not np.array_equal(ids, reference):
            raise SystemExit("FATAL: arm {} is not aligned on match_id".format(arm))

        per_arm[arm] = (per_match_log_loss(block) if metric == "log_loss"
                        else per_match_rps(block))

    return per_arm, reference


def paired_bootstrap(losses, left, right, rng):
    """
    Mean of (left - right) per match, with a paired percentile interval.

    The same resampled match indices score both arms, so the interval is on
    the DIFFERENCE and not on two independently noisy means.
    """

    difference = losses[left] - losses[right]

    point = float(np.mean(difference))

    n = len(difference)
    draws = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    means = difference[draws].mean(axis=1)

    low = float(np.percentile(means, (100.0 - BOOTSTRAP_INTERVAL) / 2.0))
    high = float(np.percentile(means, 100.0 - (100.0 - BOOTSTRAP_INTERVAL) / 2.0))

    return {"point": point, "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0.0 or high < 0.0)}


def decompose(predictions):

    results = {}

    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):

        losses, _ids = aligned_losses(predictions, metric)

        rng = np.random.default_rng(BOOTSTRAP_SEED)

        gaps = {
            "delta_total": paired_bootstrap(losses, ARM_STATIC, ARM_WALKFORWARD, rng),
            "delta_recency": paired_bootstrap(losses, ARM_STATIC, ARM_ROLLING, rng),
            "delta_sample": paired_bootstrap(losses, ARM_ROLLING, ARM_WALKFORWARD, rng),
            "delta_half_minus_static": paired_bootstrap(losses, ARM_HALF, ARM_STATIC, rng),
        }

        total = gaps["delta_total"]["point"]

        gaps["share_recency"] = (gaps["delta_recency"]["point"] / total
                                 if total != 0 else float("nan"))

        results[metric] = gaps

    return results


def verdict_of(decomposition):

    primary = decomposition[PRIMARY_METRIC]
    secondary = decomposition[SECONDARY_METRIC]

    # ---- the gate ----
    if not primary["delta_total"]["excludes_zero"]:
        return "GATE FAILED", (
            "the 95% CI for Delta_total includes 0, so the 0.036 has not "
            "replicated and there is nothing to decompose")

    # ---- sign agreement between log loss and RPS ----
    for name in ("delta_total", "delta_recency", "delta_sample"):
        if np.sign(primary[name]["point"]) != np.sign(secondary[name]["point"]):
            return "INCONCLUSIVE", (
                "log loss and RPS disagree in sign on {}, which the "
                "pre-declaration makes a forcing condition".format(name))

    share = primary["share_recency"]

    if not primary["delta_recency"]["excludes_zero"]:
        return "RATINGS-AS-FEATURES", (
            "the CI for Delta_recency includes 0: at constant sample size, "
            "moving the endpoint forward bought nothing measurable")

    if share >= SHARE_RECENCY_UPDATING:
        return "WITHIN-SEASON UPDATING", (
            "recency accounts for {:.0%} of the gap and its CI excludes "
            "0".format(share))

    if share <= SHARE_RECENCY_FEATURES:
        return "RATINGS-AS-FEATURES", (
            "recency accounts for only {:.0%} of the gap; most of it was "
            "sample size".format(share))

    return "INCONCLUSIVE", (
        "recency accounts for {:.0%}, between the declared 40% and 60% "
        "boundaries".format(share))


# ============================================================
# TESTS
# ============================================================

def test_everything(matches, spec, predictions, fold_frame, decomposition,
                    window_log, before_state, audit):

    # ---- W1  ANCHOR ------------------------------------------------------
    stored = pd.read_csv(P2_SUMMARY, float_precision=FLOAT_PRECISION)

    worst = 0.0
    worst_where = ""
    compared = 0

    for arm, variant in ANCHOR_VARIANT.items():

        theirs = stored[stored["variant"] == variant].sort_values("fold")
        mine = fold_frame[fold_frame["arm"] == arm].sort_values("fold")

        for metric in METRICS:

            if metric not in theirs.columns:
                continue

            difference = float(np.abs(
                theirs[metric].to_numpy() - mine[metric].to_numpy()).max())

            compared += 1

            if difference > worst:
                worst, worst_where = difference, "{}/{}".format(arm, metric)

    audit.record(
        "W1", "arms A and B reproduce Phase 2's stored Dixon-Coles numbers",
        "< 1e-9", "{:.3e} ({})".format(worst, worst_where), worst < 1e-9,
        "{} arm x metric comparisons over 4 folds; this is what makes "
        "'reuse, do not rewrite' checkable".format(compared))

    # ---- W2  window sizes ------------------------------------------------
    wrong = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        w_c, w_d = window_sizes(fold_spec, matches)

        for arm, expected in ((ARM_ROLLING, w_c), (ARM_HALF, w_d)):
            for window, _model, _cutoff in window_log[(arm, fold)]:
                if len(window) != expected:
                    wrong.append((arm, fold, len(window), expected))

    audit.record(
        "W2", "arms C and D fit exactly their declared window size, always",
        0, len(wrong), not wrong, str(wrong[:3]))

    # ---- W3  STRICT PRIORITY ---------------------------------------------
    violations = 0
    checked = 0

    for arm in ARMS:
        for fold_spec in spec["folds"]:

            fold = int(fold_spec["fold"])

            for window, _model, cutoff in window_log[(arm, fold)]:
                checked += 1
                if (window["date"] >= pd.Timestamp(cutoff)).any():
                    violations += 1

    audit.record(
        "W3", "every fitting window is strictly earlier than what it predicts",
        0, violations, violations == 0,
        "{} windows checked; for the static arms the cutoff is the first test "
        "date, so a violation would mean a test match entered its own "
        "fit".format(checked))

    # ---- W4  arm D ends before the test season ---------------------------
    late = 0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        first_test = matches[
            matches["season"] == str(fold_spec["test_season"])]["date"].min()

        for window, _model, _cutoff in window_log[(ARM_HALF, fold)]:
            if (window["date"] >= first_test).any():
                late += 1

    audit.record(
        "W4", "arm D's window ends strictly before its test season",
        0, late, late == 0)

    # ---- W5  the decomposition identity ----------------------------------
    worst_residual = 0.0

    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):
        gaps = decomposition[metric]
        residual = abs(gaps["delta_total"]["point"]
                       - (gaps["delta_recency"]["point"]
                          + gaps["delta_sample"]["point"]))
        worst_residual = max(worst_residual, residual)

    audit.record(
        "W5", "Delta_total equals Delta_recency + Delta_sample exactly",
        "< 1e-12", "{:.3e}".format(worst_residual), worst_residual < 1e-12,
        "if this drifts the three numbers are not measuring one split")

    # ---- W6  A and C differ in WHICH matches, not how many ---------------
    mismatched_size = 0
    identical_content = 0
    compared_windows = 0

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        static_window = window_log[(ARM_STATIC, fold)][0][0]
        static_ids = set(static_window["match_id"].tolist())

        for window, _model, _cutoff in window_log[(ARM_ROLLING, fold)]:

            compared_windows += 1

            if len(window) != len(static_window):
                mismatched_size += 1

            if set(window["match_id"].tolist()) == static_ids:
                identical_content += 1

    audit.record(
        "W6a", "arm C's windows are the same SIZE as arm A's, every time",
        0, mismatched_size, mismatched_size == 0,
        "{} windows compared".format(compared_windows))

    audit.measure(
        "W6b", "arm C windows whose CONTENT equals arm A's",
        "{} of {}".format(identical_content, compared_windows),
        "expected to be small and to occur only at the very start of a test "
        "season, before the rolling window has dropped anything; if it were "
        "all of them, C would just be A and the experiment would be empty")

    # ---- W7  the inherited constants -------------------------------------
    expected_constants = {
        "MAX_GOALS": 25,
        "TIME_DECAY_HALF_LIFE_DAYS": 107.0,
        "NEUTRAL_STRENGTH": 1.0,
        "RHO_LOWER": -0.5,
        "RHO_UPPER": 0.5,
        "RHO_GRID_POINTS": 60,
        "RHO_GOLDEN_ITERATIONS": 60,
        "SCALING_MAX_ITERATIONS": 200,
        "SCALING_TOLERANCE": 1e-10,
    }

    drifted = [name for name, value in expected_constants.items()
               if getattr(P2, name) != value]

    audit.record(
        "W7", "every class-A constant is the one the pre-declaration names",
        0, len(drifted), not drifted,
        "read from phase2_poisson_dixon_coles at runtime: {}".format(
            drifted or "all match"))

    # ---- W10  probability contract ---------------------------------------
    proba = predictions[["p_home", "p_draw", "p_away"]].to_numpy(float)

    sums = float(np.abs(proba.sum(axis=1) - 1.0).max())
    negative = int((proba < 0).sum())

    audit.record("W10a", "every predicted row sums to 1", "< 1e-9",
                 "{:.3e}".format(sums), sums < 1e-9)

    audit.record("W10b", "no negative probability", 0, negative, negative == 0)

    audit.record(
        "W10c", "every arm passed the harness's own validator",
        len(ARMS) * 4, len(ARMS) * 4, True,
        "validate_probabilities() is called per arm per fold before scoring")

    # ---- W9  isolation ---------------------------------------------------
    after_state = frozen_state()

    changed = [path for path in before_state
               if before_state[path] != after_state.get(path)]

    audit.record(
        "W9a", "no frozen artefact, script, pre-declaration or raw file moved",
        0, len(changed), not changed, str(changed[:3]))

    # Phase 2 records opens through opened_paths()/access_context, not through
    # the (path, mode, flags, hashing) tuples Instruments 4 and 5 use. Using
    # its real API rather than guessing: a hasattr() guard on the wrong name
    # would have made this test pass by finding nothing to check.
    raw_opens = [path for path in P2.opened_paths(context="input")
                 if str(path).startswith(str(RAW_DIR))]

    hasher_opens = [path for path in P2.opened_paths(context=HASH_CONTEXT)
                    if str(path).startswith(str(RAW_DIR))]

    audit.record(
        "W9b", "nothing but the integrity hasher opened data/raw",
        0, len(raw_opens), not raw_opens,
        "matches come from Phase 1's foundation via P2.load_inputs(); the "
        "hasher's {} opens are labelled and excluded".format(len(hasher_opens)))

    audit.measure(
        "W9c", "data/raw opens by the integrity hasher", len(hasher_opens),
        "70 files hashed before the run and again after it; a W9b that could "
        "not tell these from a real read would flag its own check")

    return audit


def test_determinism(matches, spec, predictions, audit):
    """W8, run separately because it repeats the whole fit."""

    print("    W8: repeating every arm from scratch...")

    repeat, _windows = run_everything(matches, spec)

    keys = ["arm", "fold", "match_id"]

    joined = predictions.merge(repeat, on=keys, suffixes=("_a", "_b"))

    moved = float(np.abs(
        joined[["p_home_a", "p_draw_a", "p_away_a"]].to_numpy(float)
        - joined[["p_home_b", "p_draw_b", "p_away_b"]].to_numpy(float)).max())

    audit.record(
        "W8a", "a second full run reproduces every probability bit-identically",
        0.0, "{:.3e}".format(moved), moved == 0.0,
        "{} predictions compared".format(len(joined)))

    rng_one = np.random.default_rng(BOOTSTRAP_SEED)
    rng_two = np.random.default_rng(BOOTSTRAP_SEED)

    losses, _ids = aligned_losses(predictions, PRIMARY_METRIC)

    first = paired_bootstrap(losses, ARM_STATIC, ARM_ROLLING, rng_one)
    second = paired_bootstrap(losses, ARM_STATIC, ARM_ROLLING, rng_two)

    audit.record(
        "W8b", "the bootstrap interval is reproducible from its declared seed",
        "identical",
        "identical" if first == second else "DIFFERS",
        first == second,
        "seed {}, {} draws".format(BOOTSTRAP_SEED, BOOTSTRAP_DRAWS))


# ============================================================
# REPORT
# ============================================================

def print_pooled(pooled):

    print("  Pooled over all 1,520 outer test matches.")
    print()
    print("  {:<16} {:<34} {:>10} {:>10} {:>10}".format(
        "arm", "what it is", "log loss", "RPS", "accuracy"))
    print("  " + "-" * 84)

    for _i, row in pooled.iterrows():
        print("  {:<16} {:<34} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            row["arm"], row["arm_label"], row["log_loss"], row["rps"],
            row["accuracy"]))

    print()


def print_gate(decomposition):

    gate = decomposition[PRIMARY_METRIC]["delta_total"]

    print("  GATE - does the 0.036 replicate?")
    print()
    print("    Delta_total = LL(A static) - LL(B walk-forward)")
    print()
    print("      point estimate   {:+.4f}".format(gate["point"]))
    print("      95% CI           [{:+.4f}, {:+.4f}]".format(
        gate["ci_low"], gate["ci_high"]))
    print("      excludes zero    {}".format(
        "YES" if gate["excludes_zero"] else "NO"))
    print()

    if not gate["excludes_zero"]:
        print("    The gate FAILS. There is nothing to decompose, and the")
        print("    pre-declaration says to stop here rather than read a split")
        print("    out of an effect that is not there.")
        print()

    return gate["excludes_zero"]


def print_decomposition(decomposition):

    for metric in (PRIMARY_METRIC, SECONDARY_METRIC):

        gaps = decomposition[metric]

        print("  {} ({}):".format(
            metric, "PRIMARY" if metric == PRIMARY_METRIC else "secondary"))
        print()
        print("    {:<34} {:>10} {:>22} {:>9}".format(
            "", "point", "95% CI", "excl. 0"))
        print("    " + "-" * 78)

        for name, label in (
                ("delta_total", "Delta_total     A - B"),
                ("delta_recency", "Delta_recency   A - C"),
                ("delta_sample", "Delta_sample    C - B"),
                ("delta_half_minus_static", "control         D - A")):

            gap = gaps[name]

            print("    {:<34} {:>+10.4f} {:>22} {:>9}".format(
                label, gap["point"],
                "[{:+.4f}, {:+.4f}]".format(gap["ci_low"], gap["ci_high"]),
                "yes" if gap["excludes_zero"] else "no"))

        print()
        print("    share_recency = {:.1%}".format(gaps["share_recency"]))
        print()


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 4 - TIER 2: RECENCY OR SAMPLE?")

    print("  question  : the static-vs-walkforward gap is 0.036. Is that")
    print("              drift the model must track, or just more data?")
    print()
    print("  arms      : A static / B walk-forward / C rolling fixed-size")
    print("              / D static half-window")
    print("  model     : Dixon-Coles, imported from Phase 2, rho fitted")
    print("  folds     : the four frozen expanding-window folds")
    print("  bootstrap : paired per-match, {} draws, seed {}".format(
        BOOTSTRAP_DRAWS, BOOTSTRAP_SEED))
    print("  declared  : {}".format(PREDECLARATION.name))
    print("              sha256 {}".format(hash_file(PREDECLARATION)[:32]))
    print()

    before_state = frozen_state()

    spec, matches, _base_rate, _elo = P2.load_inputs()

    print("  matches {}, folds {}".format(len(matches), len(spec["folds"])))
    print()

    for fold_spec in spec["folds"]:
        w_c, w_d = window_sizes(fold_spec, matches)
        print("    fold {}  N_train {:>5}   W_C {:>5}   W_D {:>5}".format(
            int(fold_spec["fold"]),
            int(matches["season"].isin(fold_spec["train_seasons"]).sum()),
            w_c, w_d))

    print()

    banner("1. FITTING THE FOUR ARMS")

    predictions, window_log = run_everything(matches, spec)

    fold_frame = fold_summary(predictions)
    pooled = pooled_summary(predictions)
    decomposition = decompose(predictions)

    print()
    banner("2. POOLED RESULTS")
    print_pooled(pooled)

    banner("3. THE GATE")
    passed_gate = print_gate(decomposition)

    banner("4. THE DECOMPOSITION")

    if passed_gate:
        print_decomposition(decomposition)
    else:
        print("  Not reported - the gate failed.")
        print()

    banner("5. VERDICT")

    verdict, reason = verdict_of(decomposition)

    print("  PRE-DECLARED VERDICT : {}".format(verdict))
    print()
    print("  {}".format(reason))
    print()

    banner("6. AUDIT")

    audit = Audit()

    test_everything(matches, spec, predictions, fold_frame, decomposition,
                    window_log, before_state, audit)
    test_determinism(matches, spec, predictions, audit)

    print()
    audit.print_rows()

    banner("7. WRITING OUTPUTS")

    decomposition_rows = []

    for metric, gaps in decomposition.items():
        for name, gap in gaps.items():
            if name == "share_recency":
                decomposition_rows.append({
                    "metric": metric, "quantity": name,
                    "point": gap, "ci_low": np.nan, "ci_high": np.nan,
                    "excludes_zero": ""})
            else:
                decomposition_rows.append({
                    "metric": metric, "quantity": name, **gap})

    decomposition_frame = pd.DataFrame(decomposition_rows)
    decomposition_frame["verdict"] = verdict
    decomposition_frame["reason"] = reason

    writes = [
        (PREDICTIONS_OUTPUT, predictions),
        (FOLD_OUTPUT, fold_frame),
        (POOLED_OUTPUT, pooled),
        (DECOMPOSITION_OUTPUT, decomposition_frame),
        (AUDIT_OUTPUT, audit.frame()),
    ]

    for path, frame in writes:
        frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
        print("  {}".format(path))

    print()

    banner("PHASE 4 - TIER 2 STATUS")

    failures = audit.failures

    print("  Arms               : {}".format(len(ARMS)))
    print("  Predictions         : {}".format(len(predictions)))
    print("  Checks run          : {}".format(len(audit.rows)))
    print("  Checks failed       : {}".format(len(failures)))
    print("  Elapsed             : {:.1f} min".format(
        (time.time() - started) / 60.0))
    print()
    print("  {}".format("PASS" if not failures else "FAIL"))
    print()
    print("  VERDICT: {}".format(verdict))
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
