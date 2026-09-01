"""
===============================================================================
PHASE 4 - DIAGNOSTIC: DOES THE LOGISTIC LINK PRESERVE DIXON-COLES?
===============================================================================

WHY THIS RUNS BEFORE D2

    D2 puts dynamic state into a multinomial logistic regression. Dixon-Coles
    does not use a logistic link at all - it builds a score matrix over
    (home goals, away goals) and sums the cells into H/D/A. If the link
    discards structure that the score matrix carries, then every D2-vs-DC
    comparison in the ladder is confounded: a D2 shortfall could be feature
    loss OR link loss, and the ladder cannot tell which.

    So: fit the standard Phase 4 logistic on exactly two features, DC's own
    lambda_home and lambda_away, generated point-in-time under the same rules
    D2 will use. Nothing else. If that reproduces DC's 0.9904, the link is
    sound and any D2 shortfall is feature loss. If it does not, the gap is the
    link's, and it is subtracted from every later comparison rather than
    silently attributed to features.

WHAT HAD TO BE BUILT FIRST, AND WHY IT IS NOT A SHORTCUT

    phase2_poisson_dc_results.csv already carries lambda_home and lambda_away
    per match - but only for the 1,520 TEST matches. Fitting a logistic needs
    the same features on the TRAINING rows, and those were never written out.

    So this generates point-in-time DC state for all 1,900 matches by the
    identical rule Phase 2 uses:

        one refit per distinct DATE, on every match with date STRICTLY
        before that date

    That cadence is read out of phase2_poisson_dixon_coles.run_fold() rather
    than reinvented, and fit_window / match_rates are IMPORTED from it. G1
    asserts the generated state reproduces Phase 2's stored test-row lambdas
    exactly, which is what makes "same rules as D2" checkable instead of
    claimed.

    The generated state is written out, because D2 needs exactly this.
===============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import CLASS_INDEX, CLASSES, evaluate  # noqa: E402
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase2_poisson_dixon_coles as DC  # noqa: E402
import phase3_regularisation_sensitivity as I4  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DC_RESULTS = OUTPUTS_DIR / "phase2_poisson_dc_results.csv"
DC_SUMMARY = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"

STATE_OUTPUT = OUTPUTS_DIR / "phase4_dc_state.csv"
RESULT_OUTPUT = OUTPUTS_DIR / "phase4_passthrough_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase4_passthrough_pooled.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase4_passthrough_audit.csv"

FLOAT_PRECISION = "round_trip"

# D2 pre-declaration Amendment 2 section A2.3. The upper 13 are Phase 4
# section 8's grid unchanged; the lower 8 extend it four decades downward
# because two folds selected the old floor of 0.01. Declared before the re-run.
LAMBDA_GRID = (0.000001, 0.000003, 0.00001, 0.00003, 0.0001, 0.0003,
               0.001, 0.003,
               0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
               300.0, 1000.0, 3000.0, 10000.0)

# D2 pre-declaration Amendment 2 section A2.1. Derived from the fold structure,
# not from any observed value: every test season is preceded by at least one
# complete season, so no test row is ever scored on a window below 380.
MIN_TRAIN_WINDOW = 380

# A2.2: fold 1 trains on 2021-22 alone, every match of which falls below the
# burn-in. It has no training set under the gate, so this DIAGNOSTIC runs on
# folds 2-4. This does NOT propagate to D2, where all four folds remain frozen.
DIAGNOSTIC_FOLDS = (2, 3, 4)

# Phase 4 pre-declaration section 7: 5 contiguous blocks, 4 expanding inner
# folds. Tie-break on lambda is the SMALLEST among ties - note this DIFFERS
# from Phase 3's "largest", and Phase 4's rule governs here.
N_INNER_BLOCKS = 5
TIE_BREAK = "smallest"

METRICS = ["accuracy", "balanced_accuracy", "macro_f1",
           "log_loss", "brier_score", "rps"]

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260901

DC_VARIANT = "dc_walkforward"
DC_REFERENCE = 0.9904

# D2 pre-declaration Amendment 3: the events-per-variable threshold below which
# an unpenalised logistic fit is unreliable. Peduzzi et al. 1996 - imported,
# not fitted, and not tuned to put anything on either side of the line.
EPV_APPLICABILITY = 10.0


# ============================================================
# POINT-IN-TIME DC STATE  (the same rule Phase 2 walks forward on)
# ============================================================

def generate_state(matches):
    """
    lambda_home / lambda_away for every one of the 1,900 matches, each fitted
    on matches STRICTLY earlier than its own date.

    One refit per distinct date - Phase 2's cadence, inherited not invented.
    The earliest dates have little or no history; those rows are flagged
    rather than silently defaulted, because D2 will meet the same rows.
    """

    rows = []
    refits = 0

    for cutoff in sorted(matches["date"].unique()):

        cutoff = pd.Timestamp(cutoff)

        window = matches[matches["date"] < cutoff]

        block = matches[matches["date"] == cutoff]

        if len(window) == 0:
            for _i, match in block.iterrows():
                rows.append({
                    "match_id": int(match["match_id"]),
                    "lambda_home": np.nan, "lambda_away": np.nan,
                    "rho": np.nan, "window_matches": 0, "has_state": False,
                })
            continue

        model = DC.fit_window(window, cutoff, True)
        refits += 1

        # match_rates returns two parallel arrays, not a list of pairs.
        home_rates, away_rates = DC.match_rates(
            block, model["attack"], model["defence"], model["home_multiplier"])

        for (_i, match), lambda_home, lambda_away in zip(
                block.iterrows(), home_rates, away_rates):

            rows.append({
                "match_id": int(match["match_id"]),
                "lambda_home": float(lambda_home),
                "lambda_away": float(lambda_away),
                "rho": float(model["rho"]),
                "window_matches": int(len(window)),
                "has_state": True,
            })

    frame = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)

    return frame, refits


# ============================================================
# INNER VALIDATION  (Phase 4 section 7)
# ============================================================

def contiguous_blocks(train_rows, blocks, n_blocks):
    """
    Split the outer training rows into n_blocks contiguous chronological
    segments of approximately equal match count, then build expanding inner
    folds: fold i trains on 1..i and validates on i+1.
    """

    train_set = set(int(r) for r in train_rows)

    usable = [b for b in blocks
              if set(int(r) for r in b["rows"]).issubset(train_set)]

    total = sum(len(b["rows"]) for b in usable)
    target = total / n_blocks

    segments = [[] for _ in range(n_blocks)]
    running = 0

    for block in usable:
        midpoint = running + len(block["rows"]) / 2.0
        index = min(int(midpoint // target), n_blocks - 1)
        segments[index].append(block)
        running += len(block["rows"])

    splits = []

    for cut in range(1, n_blocks):

        train_blocks = [b for s in segments[:cut] for b in s]
        valid_blocks = segments[cut]

        if not train_blocks or not valid_blocks:
            raise RuntimeError("inner block {} came out empty".format(cut))

        splits.append({
            "inner_fold": cut,
            "train_rows": np.sort(np.concatenate([b["rows"] for b in train_blocks])),
            "valid_rows": np.sort(np.concatenate([b["rows"] for b in valid_blocks])),
            "train_max_date": max(b["max_date"] for b in train_blocks),
            "valid_min_date": min(b["min_date"] for b in valid_blocks),
        })

    return splits


def select_lambda(matrix, labels, results, train_rows, blocks):
    """Mean validation log loss over the 4 inner folds. Ties to the SMALLEST."""

    splits = contiguous_blocks(train_rows, blocks, N_INNER_BLOCKS)

    curve = {}

    for penalty in LAMBDA_GRID:

        losses = []

        for split in splits:

            fitted = I4.fit_pipeline(matrix, labels, split["train_rows"],
                                     split["valid_rows"], penalty)

            actual = results[split["valid_rows"]]
            losses.append(evaluate(actual, fitted["proba"])["log_loss"])

        curve[penalty] = float(np.mean(losses))

    best = min(curve.values())

    chosen = min(p for p, v in curve.items() if v == best)

    return chosen, curve, splits


# ============================================================
# BOOTSTRAP
# ============================================================

def per_match_log_loss(proba, actual):

    index = np.array([CLASS_INDEX[a] for a in actual])
    picked = proba[np.arange(len(actual)), index]

    return -np.log(np.clip(picked, 1e-15, 1.0))


def per_match_rps(proba, actual):

    onehot = np.zeros_like(proba)
    onehot[np.arange(len(actual)), [CLASS_INDEX[a] for a in actual]] = 1.0
    cumulative = np.cumsum(proba, axis=1) - np.cumsum(onehot, axis=1)

    return np.sum(cumulative[:, :-1] ** 2, axis=1) / (len(CLASSES) - 1)


def paired_bootstrap(a, b, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """Paired per-match CI on mean(a) - mean(b). Same indices score both."""

    rng = np.random.default_rng(seed)
    n = len(a)

    difference = a - b

    stats = np.empty(draws)

    for draw in range(draws):
        idx = rng.integers(0, n, n)
        stats[draw] = float(np.mean(difference[idx]))

    return (float(np.mean(difference)),
            float(np.percentile(stats, 2.5)),
            float(np.percentile(stats, 97.5)))


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 4 DIAGNOSTIC - DC PASSTHROUGH THROUGH THE LOGISTIC LINK")

    print("  features  : lambda_home, lambda_away  (nothing else)")
    print("  cadence   : one DC refit per distinct date, window date < cutoff")
    print("              inherited from phase2_poisson_dixon_coles.run_fold()")
    print("  lambda    : per fold, {} inner blocks, ties to the {}".format(
        N_INNER_BLOCKS, TIE_BREAK))
    print("  reference : dc_walkforward, RECOMPUTED on folds 2-4 (4-fold was {:.4f})".format(DC_REFERENCE))
    print()

    spec, matches, _base, _elo = DC.load_inputs()[:4]

    print("  generating point-in-time DC state for all {} matches...".format(
        len(matches)))

    state, refits = generate_state(matches)

    print("  {} refits, {:.1f}s".format(refits, time.time() - started))
    print()

    audit = Audit()

    # ---- G1  the generated state must reproduce Phase 2's stored lambdas --
    stored = pd.read_csv(DC_RESULTS, float_precision=FLOAT_PRECISION)
    stored = stored[stored["variant"] == DC_VARIANT].copy()
    # dates arrive as strings from CSV; the match frame carries datetime64
    stored["date"] = pd.to_datetime(stored["date"], format="%Y-%m-%d")

    keyed = matches[["match_id", "date", "home_team", "away_team"]].merge(
        state, on="match_id", how="left")

    joined = stored.merge(
        keyed, left_on=["date", "home", "away"],
        right_on=["date", "home_team", "away_team"],
        suffixes=("_p2", "_new"), how="inner")

    worst = max(
        float(np.abs(joined["lambda_home_p2"] - joined["lambda_home_new"]).max()),
        float(np.abs(joined["lambda_away_p2"] - joined["lambda_away_new"]).max()))

    audit.record(
        "G1", "generated state reproduces Phase 2's stored test-row lambdas",
        "< 1e-9", "{:.3e}".format(worst), worst < 1e-9,
        "{} test rows matched; this is what makes 'the same rules as D2' "
        "checkable rather than asserted".format(len(joined)))

    audit.measure(
        "G2", "matches with no prior history and therefore no DC state",
        int((~state["has_state"]).sum()),
        "the first date of 2021-22 has nothing before it; flagged, never "
        "silently defaulted to a neutral strength")

    # ---- the design matrix ------------------------------------------------
    frame = matches.merge(state, on="match_id", how="left")

    matrix = frame[["lambda_home", "lambda_away"]].to_numpy(float)
    labels = np.array([CLASS_INDEX[r] for r in frame["result"]], dtype=int)
    results = frame["result"].to_numpy()

    blocks = I4.date_blocks(frame)

    # ---- gameweek vs date blocking, measured not assumed ------------------
    gw_violations = 0

    for fold_spec in spec["folds"]:
        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        gw = frame.iloc[train_rows].groupby(["season", "matchweek"])["date"]
        spans = (gw.max() - gw.min()).dt.days
        gw_violations += int((spans > 7).sum())

    audit.measure(
        "G3", "(season, matchweek) training blocks spanning over 7 days",
        gw_violations,
        "Phase 4 section 7 specifies gameweek blocking AND strict date "
        "ordering; Instrument 4 Amendment 1 established those are "
        "incompatible here - postponed fixtures keep their original "
        "matchweek. Calendar-date blocking is used, as Amendment 1 resolved")

    banner("1. PER FOLD")

    print("  {:<6} {:<12} {:>9} {:>10} {:>10} {:>10}".format(
        "fold", "season", "lambda*", "passthru", "DC walkfwd", "gap"))
    print("  " + "-" * 62)

    dc_summary = pd.read_csv(DC_SUMMARY, float_precision=FLOAT_PRECISION)
    dc_summary = dc_summary[dc_summary["variant"] == DC_VARIANT]

    rows = []
    all_pass_proba, all_dc_proba, all_actual = [], [], []
    used_train_rows, used_test_rows = [], []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        test_season = str(fold_spec["test_season"])

        if fold not in DIAGNOSTIC_FOLDS:
            continue

        # THE BURN-IN GATE. Training rows must come from a fitting window of at
        # least MIN_TRAIN_WINDOW, so the model is only ever taught in the regime
        # its test rows are drawn from. Test rows are never filtered - every one
        # of them is scored.
        eligible = (frame["season"].isin(fold_spec["train_seasons"])
                    & (frame["window_matches"] >= MIN_TRAIN_WINDOW)).to_numpy()

        train_rows = np.flatnonzero(eligible)
        test_rows = np.flatnonzero((frame["season"] == test_season).to_numpy())

        chosen, curve, splits = select_lambda(
            matrix, labels, results, train_rows, blocks)

        fitted = I4.fit_pipeline(matrix, labels, train_rows, test_rows, chosen)

        actual = results[test_rows]
        scores = evaluate(actual, fitted["proba"])

        dc_rows = stored[stored["test_season"] == test_season].sort_values(
            ["date", "home", "away"])
        order = frame.iloc[test_rows].sort_values(["date", "home_team", "away_team"])

        dc_proba = dc_rows[["p_home", "p_draw", "p_away"]].to_numpy(float)

        # align passthrough rows to the DC row order by (date, home, away)
        key = frame.iloc[test_rows][["date", "home_team", "away_team"]].copy()
        key["_i"] = np.arange(len(test_rows))
        key = key.sort_values(["date", "home_team", "away_team"])
        pass_proba = fitted["proba"][key["_i"].to_numpy()]
        aligned_actual = actual[key["_i"].to_numpy()]

        dc_ll = float(dc_summary[dc_summary["fold"] == fold]["log_loss"].iloc[0])

        all_pass_proba.append(pass_proba)
        all_dc_proba.append(dc_proba)
        all_actual.append(aligned_actual)
        used_train_rows.append(train_rows)
        used_test_rows.append(test_rows)

        rarest = int(pd.Series(results[train_rows]).value_counts().min())

        row = {"fold": fold, "test_season": test_season,
               "selected_lambda": chosen,
               "at_grid_floor": bool(chosen == LAMBDA_GRID[0]),
               "at_grid_ceiling": bool(chosen == LAMBDA_GRID[-1]),
               "train_matches": len(train_rows),
               "rarest_class": rarest,
               "design_width": matrix.shape[1],
               "epv": rarest / matrix.shape[1],
               "dc_log_loss": dc_ll}
        row.update({m: scores[m] for m in METRICS})
        row["gap_vs_dc"] = scores["log_loss"] - dc_ll
        rows.append(row)

        print("  {:<6} {:<12} {:>9g} {:>10.4f} {:>10.4f} {:>+10.4f}".format(
            fold, test_season, chosen, scores["log_loss"], dc_ll,
            scores["log_loss"] - dc_ll))

    print()

    fold_frame = pd.DataFrame(rows)

    # ---- pooled -----------------------------------------------------------
    pass_proba = np.vstack(all_pass_proba)
    dc_proba = np.vstack(all_dc_proba)
    actual = np.concatenate(all_actual)

    pass_scores = evaluate(actual, pass_proba)
    dc_scores = evaluate(actual, dc_proba)

    banner("2. POOLED OVER {} OUTER TEST MATCHES (folds {})".format(
        len(actual), ", ".join(str(f) for f in DIAGNOSTIC_FOLDS)))

    print("  {:<22} {:>10} {:>10} {:>10}".format(
        "metric", "passthru", "DC walkfwd", "gap"))
    print("  " + "-" * 56)

    for metric in METRICS:
        print("  {:<22} {:>10.4f} {:>10.4f} {:>+10.4f}".format(
            metric, pass_scores[metric], dc_scores[metric],
            pass_scores[metric] - dc_scores[metric]))

    print()

    ll_gap, ll_lo, ll_hi = paired_bootstrap(
        per_match_log_loss(pass_proba, actual),
        per_match_log_loss(dc_proba, actual))

    rps_gap, rps_lo, rps_hi = paired_bootstrap(
        per_match_rps(pass_proba, actual), per_match_rps(dc_proba, actual))

    print("  PAIRED PER-MATCH BOOTSTRAP, {} draws, seed {}".format(
        BOOTSTRAP_DRAWS, BOOTSTRAP_SEED))
    print()
    print("    log loss   passthrough - DC = {:+.4f}  95% CI [{:+.4f}, {:+.4f}]".format(
        ll_gap, ll_lo, ll_hi))
    print("    RPS        passthrough - DC = {:+.4f}  95% CI [{:+.4f}, {:+.4f}]".format(
        rps_gap, rps_lo, rps_hi))
    print()

    signs_agree = (ll_gap > 0) == (rps_gap > 0)
    ll_excludes_zero = not (ll_lo <= 0 <= ll_hi)

    audit.record(
        "G4", "log loss and RPS agree in sign on the passthrough gap",
        "agree", "agree" if signs_agree else "DISAGREE", signs_agree,
        "the Phase 4 rule: a sign disagreement makes the comparison "
        "INCONCLUSIVE")

    audit.measure(
        "G5", "does the 95% CI for the log-loss gap exclude zero",
        "yes" if ll_excludes_zero else "no",
        "if it includes zero, the link is preserving DC within the "
        "resolution these 1,520 matches provide")

    # ---- G6, as amended (D2 pre-declaration Amendment 3) -----------------
    # G6 catches a MIS-SPECIFIED grid: the data wanting a smaller penalty than
    # the grid offers. That presupposes the penalty is doing work. Where the
    # unpenalised fit is already well identified, lambda -> 0 is the correct
    # answer rather than an artefact, and no extension of the grid can stop
    # the floor binding because there is no interior optimum to find.
    #
    # Applicability is events-per-variable, the standard logistic-regression
    # criterion (Peduzzi et al. 1996), imported not fitted:
    #     EPV = rarest training class count / design width
    boundary = int(fold_frame["at_grid_floor"].sum()
                   + fold_frame["at_grid_ceiling"].sum())

    min_epv = float(fold_frame["epv"].min())
    applicable = min_epv < EPV_APPLICABILITY

    if applicable:
        audit.record(
            "G6b", "no selected lambda sits on a grid boundary",
            0, boundary, boundary == 0,
            "EPV falls below {} for at least one fold (min {:.2f}), so the "
            "penalty is doing work here and a boundary selection is a "
            "mis-specified grid".format(EPV_APPLICABILITY, min_epv))
    else:
        audit.measure(
            "G6b", "boundary selections, gate NOT APPLICABLE",
            "{} of {} folds, min EPV {:.2f}".format(
                boundary, len(fold_frame), min_epv),
            "every fold has EPV >= {} ({:.2f} at worst), so the unpenalised "
            "fit is well identified and lambda -> 0 is the correct answer, "
            "not a grid artefact. Selected: {}".format(
                EPV_APPLICABILITY, min_epv,
                ", ".join("fold {} lambda {:g}".format(int(r["fold"]),
                                                       r["selected_lambda"])
                          for _i, r in fold_frame.iterrows()
                          if r["at_grid_floor"] or r["at_grid_ceiling"])))

    audit.measure(
        "G6a", "G6 applicability (EPV < {} means applicable)".format(
            EPV_APPLICABILITY),
        "APPLICABLE" if applicable else "NOT APPLICABLE",
        "per fold: " + ", ".join(
            "f{} EPV {:.2f} (n {} rarest {} width {})".format(
                int(r["fold"]), r["epv"], int(r["train_matches"]),
                int(r["rarest_class"]), int(r["design_width"]))
            for _i, r in fold_frame.iterrows()))

    # ---- G7  REGIME GATE, checked before any read of the gap -------------
    # Phase 2's walk-forward only ever fits windows of at least one full
    # training season, and only ever predicts test-season matches. Extending
    # that rule to TRAINING rows enters a regime Phase 2 never did: the
    # earliest 2021-22 matches are fitted on a handful of games, where DC
    # strengths are not yet identified.
    #
    # The gate is structural, not a tuned threshold: every training row must
    # come from a window at least as large as the SMALLEST window any test row
    # was scored on. A model cannot be diagnosed on features drawn from a
    # regime it is never used in.
    train_used = np.unique(np.concatenate(used_train_rows))
    test_used = np.unique(np.concatenate(used_test_rows))

    smallest_test_window = int(frame.iloc[test_used]["window_matches"].min())

    # G7 asserts the DECLARED rule (Amendment 2: burn-in 380), not a threshold
    # re-derived from whichever folds happen to be running. Re-deriving would
    # make the gate a moving target: dropping fold 1 raises the smallest test
    # window to 760, which would silently tighten an approved threshold. The
    # subset-specific gap is reported separately as G7c rather than enforced.
    undersized = int(
        (frame.iloc[train_used]["window_matches"] < MIN_TRAIN_WINDOW).sum())

    train_lambda_max = float(frame.iloc[train_used]["lambda_home"].max())
    test_lambda_max = float(frame.iloc[test_used]["lambda_home"].max())

    audit.record(
        "G7", "every training row actually used is fitted on a window no "
              "smaller than the smallest test window",
        0, undersized, undersized == 0,
        "burn-in {} (derived: every test season follows a complete season); "
        "smallest test window {}; {} of {} used training rows below it; "
        "training lambda_home reaches {:.2f} against {:.2f} across the test "
        "rows".format(MIN_TRAIN_WINDOW, smallest_test_window, undersized,
                      len(train_used), train_lambda_max, test_lambda_max))

    audit.measure(
        "G7b", "training rows excluded by the burn-in gate",
        int((frame["season"].isin(
            [s for f in spec["folds"] if int(f["fold"]) in DIAGNOSTIC_FOLDS
             for s in f["train_seasons"]])
            & (frame["window_matches"] < MIN_TRAIN_WINDOW)).sum()),
        "all of 2021-22; fold 1 therefore has no training set and this "
        "diagnostic runs on folds {} only. Does NOT propagate to D2".format(
            ", ".join(str(f) for f in DIAGNOSTIC_FOLDS)))

    below_subset = int(
        (frame.iloc[train_used]["window_matches"] < smallest_test_window).sum())

    audit.measure(
        "G7c", "used training rows below the SMALLEST WINDOW OF THIS SUBSET's "
               "test rows ({})".format(smallest_test_window),
        below_subset,
        "the declared burn-in of {} comes from the full four-fold structure, "
        "where fold 1's test season follows one complete season. Dropping "
        "fold 1 leaves test rows whose windows all start at {}, so these "
        "rows sit in a window regime this three-fold subset does not itself "
        "contain. Reported, NOT enforced - tightening an approved threshold "
        "because the subset changed would be exactly the moving target the "
        "amendment forbids".format(MIN_TRAIN_WINDOW, smallest_test_window))

    regime_ok = undersized == 0

    banner("3. THE READ")

    if not regime_ok:
        verdict = "INVALID - REGIME VIOLATION"
        print("  INVALID. This diagnostic cannot be read yet, and the gap above")
        print("  must NOT be reported as a link measurement.")
        print()
        print("  {} of the {} training rows actually used are fitted on".format(
            undersized, len(train_used)))
        print("  windows below the declared burn-in of {}. In that regime".format(
            MIN_TRAIN_WINDOW))
        print("  Dixon-Coles strengths are not identified, and the resulting")
        print("  outliers sit in TRAINING rows, distorting the standardiser")
        print("  the test rows are then mapped through.")
        print()
        print("  training lambda_home reaches {:.2f}; test rows reach {:.2f}".format(
            train_lambda_max, test_lambda_max))
        print()
        print("  The gate is not doing its job. STOPPING.")
    elif not ll_excludes_zero:
        verdict = "LINK SOUND"
        print("  LINK SOUND. The passthrough gap's CI includes zero, so at the")
        print("  resolution of {} matches the logistic link preserves what".format(len(actual)))
        print("  Dixon-Coles carries. Any D2 shortfall against DC is FEATURE")
        print("  LOSS and may be read as such.")
    elif ll_gap < 0:
        verdict = "LINK SOUND (passthrough ahead)"
        print("  The passthrough is AHEAD of DC. The link is not discarding")
        print("  information; a D2 shortfall would be feature loss.")
    else:
        verdict = "CONFOUNDED"
        print("  CONFOUNDED. The passthrough loses {:.4f} [{:+.4f}, {:+.4f}]".format(
            ll_gap, ll_lo, ll_hi))
        print("  against DC on the SAME two quantities. That gap is the link's,")
        print("  not the features'. Every D2-vs-DC comparison in the ladder")
        print("  carries it, and D2 must be read against the passthrough")
        print("  ({:.4f}), not against DC ({:.4f}).".format(
            pass_scores["log_loss"], dc_scores["log_loss"]))
        print()
        print("  STOPPING for a decision, per the brief.")

    print()

    banner("4. AUDIT")
    audit.print_rows()

    banner("5. WRITING")

    pooled = pd.DataFrame([
        {"model": "passthrough_lambda_only", **{m: pass_scores[m] for m in METRICS}},
        {"model": "dc_walkforward", **{m: dc_scores[m] for m in METRICS}},
    ])
    pooled["log_loss_gap"] = [ll_gap, 0.0]
    pooled["log_loss_ci_lo"] = [ll_lo, np.nan]
    pooled["log_loss_ci_hi"] = [ll_hi, np.nan]
    pooled["rps_gap"] = [rps_gap, 0.0]
    pooled["rps_ci_lo"] = [rps_lo, np.nan]
    pooled["rps_ci_hi"] = [rps_hi, np.nan]
    pooled["verdict"] = verdict

    for path, data in ((STATE_OUTPUT, state), (RESULT_OUTPUT, fold_frame),
                       (POOLED_OUTPUT, pooled), (AUDIT_OUTPUT, audit.frame())):
        data.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
        print("  {}".format(path))

    print()

    failures = audit.failures

    print("  Checks run    : {}".format(len(audit.rows)))
    print("  Checks failed : {}".format(len(failures)))
    print("  Elapsed       : {:.1f}s".format(time.time() - started))
    print()
    print("  {}   verdict: {}".format(
        "PASS" if not failures else "FAIL", verdict))
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
