"""
===============================================================================
PHASE 4 - STEP 1:  IS expected_total_goals REDUNDANT GIVEN D1?
===============================================================================

A DIAGNOSTIC, NOT A GATE. It fits no model of the outcome, selects no lambda,
and moves no reported number. D3 and D4 proceed whatever it says.

THE QUESTION IT SEPARATES

    A4.6 established that expected_total_goals carries little in D2. That is
    consistent with two different worlds, and the ladder cannot tell them
    apart on its own:

        REDUNDANT GIVEN D1     the column is informative about the scoring
                               environment, but D1's per-game goal columns
                               already say the same thing. Dixon-Coles needs
                               it because DC has nothing else; D2 does not.

        UNINFORMATIVE          the column is not tracking the scoring
                               environment at all. A cleaner finding, and a
                               harsher one about the DC state extraction.

    Regressing expected_total_goals on D1's scoring-environment columns, on
    TRAINING ROWS ONLY, per fold, separates them. High R-squared is the first
    world; low R-squared is the second.

THE COLUMNS, AND ONE THING THE BRIEF ASSUMED THAT IS NOT TRUE

    The brief asked for "home and away GF/game and GA/game, plus any last-5
    goal columns". The first four exist. THE LAST-5 GOAL COLUMNS DO NOT.

    Phase 1's last-5 window is POINTS-ONLY - home_last5_pts_before,
    home_last5_mp_before and the derived home_last5_ppm_before, and their
    away counterparts. There is no last-5 goals-for or goals-against column
    anywhere in the 134. The brief's "any" is doing real work, and the answer
    is none. That is recorded rather than quietly satisfied with a points
    column renamed in the reporting.

    So the specifications below are nested, and all three are reported,
    because a single arbitrary column set would make the R-squared a choice
    rather than a measurement:

        NAMED       the four the brief names - home/away GF/game, GA/game
        VENUE       + the four venue-split equivalents
        FULL        + last-5 form (points, the only last-5 there is) and the
                    two relative goal-rate differences

    Fit by least squares with an intercept, on the fold's training rows, with
    the same median imputation the pipeline uses so a missing cell is handled
    the way the ladder handles it rather than by dropping the row.
===============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase3_ablation_ladder as L3              # noqa: E402
import phase4_dynamic_state as STATE             # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR

R2_OUTPUT = OUTPUTS_DIR / "phase4_etg_redundancy.csv"
CORR_OUTPUT = OUTPUTS_DIR / "phase4_etg_correlations.csv"

FLOAT_FORMAT = "%.17g"

TARGET = "expected_total_goals"

NAMED = ["home_gfpm_before", "home_gapm_before",
         "away_gfpm_before", "away_gapm_before"]

VENUE = NAMED + ["home_venue_gfpm_before", "home_venue_gapm_before",
                 "away_venue_gfpm_before", "away_venue_gapm_before"]

FULL = VENUE + ["home_last5_ppm_before", "away_last5_ppm_before",
                "rel_gfpm_diff", "rel_gapm_diff"]

SPECS = (("NAMED", NAMED), ("VENUE", VENUE), ("FULL", FULL))


def r_squared(design, target):
    """
    Ordinary least squares with an intercept, solved by lstsq.

    Returns 1 - SSE/SST. SST is computed about the target's own mean on the
    SAME rows, so the number is the share of that fold's training variance
    the columns explain and nothing else.
    """

    ones = np.ones((len(design), 1))
    full = np.hstack([ones, design])

    beta, *_rest = np.linalg.lstsq(full, target, rcond=None)

    residual = target - full @ beta
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((target - target.mean()) ** 2))

    if sst <= 0:
        return np.nan, np.nan

    r2 = 1.0 - sse / sst

    # In-sample R-squared rises with the regressor count for free, and fold 1
    # fits 12 of them on 380 rows. The adjusted figure is reported beside it
    # so a low R-squared cannot be dismissed as an artefact of the opposite
    # problem - and so the smallest fold, which scores HIGHEST here, is not
    # read as the strongest evidence when it has the least data per column.
    n, p = len(design), design.shape[1]
    adjusted = (1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
                if n - p - 1 > 0 else np.nan)

    return r2, adjusted


def imputed(values):
    """Median imputation on the rows given, matching the pipeline's rule."""

    out = np.array(values, dtype=float)

    if np.isnan(out).any():
        out[np.isnan(out)] = np.nanmedian(out)

    return out


def main():

    configure_stdout()

    banner("PHASE 4 - STEP 1:  expected_total_goals REDUNDANCY GIVEN D1")

    print("  a DIAGNOSTIC. No fit of the outcome, no lambda, no gate.")
    print("  training rows only, per fold, median-imputed as the pipeline does.")
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches()
    features = L3.load_features(matches)

    matches = matches.copy()
    matches["match_id"] = matches.index

    dynamic_state, _refits = STATE.build(matches)

    dynamic = dynamic_state.set_index("match_id").loc[
        matches["match_id"], LADDER.DYNAMIC_COLUMNS].reset_index(drop=True)

    frame = matches.copy()

    # ---- the brief's assumption, checked rather than assumed --------------
    last5_columns = [c for c in features.columns if "last5" in c]
    last5_goal_columns = [c for c in last5_columns
                          if "gf" in c or "ga" in c or "gd" in c]

    audit.record(
        "E1", "the brief's 'any last-5 goal columns' - how many exist",
        "0 or more", len(last5_goal_columns), True,
        "Phase 1's last-5 window is points-only: {}. There is no last-5 "
        "goals-for or goals-against column, so the FULL specification uses "
        "last-5 POINTS per game and says so, rather than presenting a points "
        "column as a goal column".format(", ".join(last5_columns)))

    missing = [c for _n, cols in SPECS for c in cols
               if c not in features.columns]

    audit.record(
        "E2", "every regressor named by the three specifications exists in "
              "the frozen feature file",
        0, len(missing), len(missing) == 0,
        "missing: {}".format(missing) if missing else
        "NAMED {}, VENUE {}, FULL {} columns".format(
            len(NAMED), len(VENUE), len(FULL)))

    target_all = dynamic[TARGET].to_numpy(float)

    # ============================================================
    banner("1. R-SQUARED, PER FOLD, TRAINING ROWS ONLY")

    print("  expected_total_goals regressed on D1's scoring-environment")
    print("  columns. Higher = more of the column is already in D1.")
    print()
    print("  {:<5} {:<13} {:>8} {:>10} {:>10} {:>10} {:>12}".format(
        "fold", "train thru", "n_train", "NAMED", "VENUE", "FULL",
        "FULL (adj)"))
    print("  " + "-" * 76)

    rows = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())

        target = imputed(target_all[train_rows])

        values, adjusted = {}, {}

        for label, columns in SPECS:

            design = np.column_stack([
                imputed(pd.to_numeric(features[c], errors="coerce")
                        .to_numpy(float)[train_rows]) for c in columns])

            values[label], adjusted[label] = r_squared(design, target)

        rows.append({"fold": fold, "test_season": str(fold_spec["test_season"]),
                     "train_seasons": " ".join(fold_spec["train_seasons"]),
                     "n_train": len(train_rows),
                     "r2_named": values["NAMED"], "r2_venue": values["VENUE"],
                     "r2_full": values["FULL"],
                     "r2_named_adjusted": adjusted["NAMED"],
                     "r2_venue_adjusted": adjusted["VENUE"],
                     "r2_full_adjusted": adjusted["FULL"],
                     "n_regressors_named": len(NAMED),
                     "n_regressors_venue": len(VENUE),
                     "n_regressors_full": len(FULL)})

        print("  {:<5} {:<13} {:>8} {:>10.4f} {:>10.4f} {:>10.4f} "
              "{:>12.4f}".format(
                  fold, str(fold_spec["train_seasons"][-1]), len(train_rows),
                  values["NAMED"], values["VENUE"], values["FULL"],
                  adjusted["FULL"]))

    print()

    r2_frame = pd.DataFrame(rows)

    # ============================================================
    banner("2. SIMPLE CORRELATIONS, PER FOLD, TRAINING ROWS ONLY")

    print("  Pearson r between expected_total_goals and each column.")
    print()

    correlations = []

    print(("  {:<30}" + "{:>10}" * 4).format(
        "column", "fold 1", "fold 2", "fold 3", "fold 4"))
    print("  " + "-" * 72)

    by_column = {}

    for column in FULL:

        per_fold = []

        for fold_spec in spec["folds"]:

            fold = int(fold_spec["fold"])

            train_rows = np.flatnonzero(
                frame["season"].isin(fold_spec["train_seasons"]).to_numpy())

            left = imputed(target_all[train_rows])
            right = imputed(pd.to_numeric(features[column], errors="coerce")
                            .to_numpy(float)[train_rows])

            if np.std(left) == 0 or np.std(right) == 0:
                r = np.nan
            else:
                r = float(np.corrcoef(left, right)[0, 1])

            per_fold.append(r)

            correlations.append({
                "fold": fold, "test_season": str(fold_spec["test_season"]),
                "column": column, "pearson_r": r, "r_squared": r * r,
                "n_train": len(train_rows)})

        by_column[column] = per_fold

        print(("  {:<30}" + "{:>+10.4f}" * 4).format(column, *per_fold))

    print()

    correlation_frame = pd.DataFrame(correlations)

    # ============================================================
    banner("3. THE READ")

    worst = float(r2_frame["r2_full"].min())
    best = float(r2_frame["r2_full"].max())
    named_worst = float(r2_frame["r2_named"].min())
    named_best = float(r2_frame["r2_named"].max())

    strongest = max(by_column, key=lambda c: max(abs(v) for v in by_column[c]))
    strongest_r = max(abs(v) for v in by_column[strongest])

    audit.measure(
        "E3", "R-squared of expected_total_goals on D1's scoring-environment "
              "columns, across the four folds",
        "NAMED {:.4f} to {:.4f} | FULL {:.4f} to {:.4f} | FULL adjusted "
        "{:.4f} to {:.4f}".format(
            named_worst, named_best, worst, best,
            float(r2_frame["r2_full_adjusted"].min()),
            float(r2_frame["r2_full_adjusted"].max())),
        "the diagnostic's whole content. High means D1 already carries the "
        "column's information and DC needs it only because DC has nothing "
        "else; low means the column is not tracking the scoring environment "
        "at all")

    audit.measure(
        "E4", "strongest single correlation with expected_total_goals",
        "{} at |r| = {:.4f}".format(strongest, strongest_r),
        "reported so the R-squared cannot be carried by one column without "
        "that being visible")

    if worst >= 0.5:
        read = ("REDUNDANT GIVEN D1. The FULL specification explains "
                "{:.0%} to {:.0%} of the column's training variance, so most "
                "of what it carries is already in D1's per-game goal rates. "
                "Dixon-Coles needs the column because DC has nothing else to "
                "say about the scoring environment; D2 does not.".format(
                    worst, best))
    elif best < 0.25:
        read = ("GENUINELY UNINFORMATIVE, not merely redundant. D1's "
                "scoring-environment columns explain only {:.0%} to {:.0%} "
                "of the column's training variance, so the two are not "
                "saying the same thing - and A4.6 already showed the column "
                "adds little on its own. The cleaner finding, and the "
                "harsher one about what the DC state extraction "
                "carries.".format(worst, best))
    else:
        read = ("PARTIAL. The FULL specification explains {:.0%} to {:.0%} "
                "of the column's training variance - too much to call the "
                "column independent of D1, too little to call it a "
                "restatement of D1. Reported as the intermediate case rather "
                "than rounded to whichever branch reads better.".format(
                    worst, best))

    for sentence in read.split(". "):
        print("  {}".format(sentence.rstrip(".") + "."))

    print()

    audit.print_rows()

    banner("4. WRITING")

    for path, data in ((R2_OUTPUT, r2_frame), (CORR_OUTPUT, correlation_frame)):
        data.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)
        print("  {}".format(path))

    print()
    print("  Checks run    : {}".format(len(audit.rows)))
    print("  Checks failed : {}".format(len(audit.failures)))
    print()
    print("  DIAGNOSTIC - not a gate. D3 and D4 proceed regardless.")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
