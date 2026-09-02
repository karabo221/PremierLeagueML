"""
===============================================================================
PHASE 4 - TWO DIAGNOSTICS ON THE D0/D1/D2 RUN
===============================================================================

Neither is a rung. Neither changes a reported number. Both exist because a
result with an unmeasured degree of freedom behind it is not a finished
result.

1.  THE STANDARDISATION DEVIATION, MEASURED

    Section 6 of the pre-declaration says boolean and indicator columns pass
    through unstandardised. Phase 3's inherited pipeline standardises every
    column. The ladder follows the pre-declaration, which means it runs a
    code path Phase 3's frozen numbers do not anchor.

    DS0 already proves the two pipelines are the same function when the mask
    is empty. This measures the other half: how much the mask itself moves
    the answer. The pre-declared rule stays primary whichever way this comes
    out - it is reported so the deviation has a size rather than a shrug.

2.  WHAT THE BURN-IN ROWS COST THE STANDARDISER

    Amendment 2 established that Dixon-Coles strengths are not identified on
    a fortnight of football, and that 380 training rows - all of 2021-22 -
    are fitted on windows below the smallest any test row is scored on. A2.2
    ruled that this does NOT propagate to D2, so those rows are training data
    here.

    They are still in the standardiser. This measures the distortion directly:
    the SD the fit actually used, against the SD of the same column on rows
    that clear the burn-in, against the SD on the test rows the column is
    ultimately mapped through.

    It does NOT refit anything with those rows removed. That would be a new
    rung under an undeclared design change, which A2.2 explicitly reserves for
    its own declaration.
===============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import CLASS_INDEX, evaluate  # noqa: E402
from phase3_feature_builder import banner, configure_stdout  # noqa: E402

import phase3_ablation_ladder as L3              # noqa: E402
import phase3_regularisation_sensitivity as I4   # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402
import phase4_dynamic_state as STATE             # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR

SENSITIVITY_OUTPUT = OUTPUTS_DIR / "phase4_ladder_standardisation_check.csv"
BURN_IN_OUTPUT = OUTPUTS_DIR / "phase4_ladder_burn_in_exposure.csv"

BURN_IN = 380       # Amendment 2 A2.1


def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 4 - DIAGNOSTICS ON THE D0/D1/D2 RUN")

    spec = L3.load_spec()
    matches = L3.load_matches()
    features = L3.load_features(matches)

    matches = matches.copy()
    matches["match_id"] = matches.index

    state, _refits = STATE.build(matches)

    frame = matches.copy()
    dynamic = state.set_index("match_id").loc[
        frame["match_id"], LADDER.DYNAMIC_COLUMNS].reset_index(drop=True)

    labels = np.array([CLASS_INDEX[r] for r in frame["result"]], dtype=int)
    results = frame["result"].to_numpy()
    blocks = I4.date_blocks(frame)

    base = LADDER.d1_features(features)

    designs = {
        "D1": LADDER.build_design(features, base),
        "D2": LADDER.build_design(features, base, dynamic),
    }

    # ============================================================
    banner("1. THE SECTION-6 PASS-THROUGH RULE, PRICED")

    frozen = pd.read_csv(LADDER.FOLD_OUTPUT, float_precision="round_trip")

    rows = []

    for rung in ("D1", "D2"):

        matrix, _names, declared_mask = designs[rung]

        for variant, mask in (
                ("declared (section 6 pass-through)", declared_mask),
                ("all columns standardised (Phase 3)",
                 np.zeros(matrix.shape[1], dtype=bool))):

            proba_by_fold = {}

            for fold_spec in spec["folds"]:

                fold = int(fold_spec["fold"])

                train_rows = np.flatnonzero(
                    frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
                test_rows = np.flatnonzero(
                    (frame["season"] == str(fold_spec["test_season"])).to_numpy())

                chosen, _curve, _splits = LADDER.select_lambda(
                    matrix, labels, results, train_rows, blocks, mask)

                fitted = LADDER.fit_pipeline(
                    matrix, labels, train_rows, test_rows, chosen, mask)

                proba_by_fold[fold] = (test_rows, fitted["proba"])

                rows.append({"rung": rung, "variant": variant, "fold": fold,
                             "selected_lambda": chosen,
                             "log_loss": evaluate(results[test_rows],
                                                  fitted["proba"])["log_loss"]})

            order, proba, scores = LADDER.pool(proba_by_fold, results, spec)

            rows.append({"rung": rung, "variant": variant, "fold": -1,
                         "selected_lambda": np.nan,
                         "log_loss": scores["log_loss"],
                         "rps": scores["rps"], "n": scores["n"]})

    sensitivity = pd.DataFrame(rows)

    pooled = sensitivity[sensitivity["fold"] == -1]

    print("  {:<5} {:<36} {:>10} {:>10}".format(
        "rung", "variant", "log loss", "RPS"))
    print("  " + "-" * 66)

    for _i, row in pooled.iterrows():
        print("  {:<5} {:<36} {:>10.4f} {:>10.4f}".format(
            row["rung"], row["variant"], row["log_loss"], row["rps"]))

    print()

    for rung in ("D1", "D2"):
        part = pooled[pooled["rung"] == rung]
        declared = float(part.iloc[0]["log_loss"])
        other = float(part.iloc[1]["log_loss"])
        print("  {}  the rule is worth {:+.4f} log loss".format(
            rung, other - declared))

    d2_d1_declared = (float(pooled[(pooled["rung"] == "D2")].iloc[0]["log_loss"])
                      - float(pooled[(pooled["rung"] == "D1")].iloc[0]["log_loss"]))
    d2_d1_other = (float(pooled[(pooled["rung"] == "D2")].iloc[1]["log_loss"])
                   - float(pooled[(pooled["rung"] == "D1")].iloc[1]["log_loss"]))

    print()
    print("  D2 - D1 under the declared rule       {:+.4f}".format(d2_d1_declared))
    print("  D2 - D1 with everything standardised  {:+.4f}".format(d2_d1_other))
    print()
    print("  the primary answer is the declared rule's, whichever is larger.")
    print()

    # ---- does the frozen run reproduce here? ------------------------------
    check = sensitivity[(sensitivity["fold"] > 0)
                        & (sensitivity["variant"].str.startswith("declared"))]
    joined = check.merge(frozen[["rung", "fold", "log_loss",
                                 "selected_lambda"]],
                         on=["rung", "fold"], suffixes=("_here", "_frozen"))

    worst = float(np.abs(joined["log_loss_here"]
                         - joined["log_loss_frozen"]).max())
    lambda_moves = int((joined["selected_lambda_here"]
                        != joined["selected_lambda_frozen"]).sum())

    print("  reproduces the ladder run: worst log-loss gap {:.3e}, "
          "{} lambda moves".format(worst, lambda_moves))
    print()

    # ============================================================
    banner("2. THE BURN-IN ROWS INSIDE THE STANDARDISER")

    windows = state["window_matches"].to_numpy()

    print("  Per fold and column: the SD the fit USED (all training rows),")
    print("  the SD of the training rows that CLEAR the burn-in, and the SD")
    print("  of the test rows the column is finally mapped through.")
    print()
    print("  {:<5} {:<22} {:>11} {:>11} {:>11} {:>9} {:>9}".format(
        "fold", "column", "SD used", "SD >=380", "SD test", "inflation",
        "test span"))
    print("  " + "-" * 86)

    exposure = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero(
            (frame["season"] == str(fold_spec["test_season"])).to_numpy())

        clean_rows = train_rows[windows[train_rows] >= BURN_IN]

        for column in LADDER.DYNAMIC_COLUMNS:

            values = dynamic[column].to_numpy()

            used = float(np.nanstd(values[train_rows], ddof=0))
            clean = float(np.nanstd(values[clean_rows], ddof=0)) \
                if len(clean_rows) else np.nan
            test = float(np.nanstd(values[test_rows], ddof=0))

            inflation = used / clean if clean and np.isfinite(clean) else np.nan

            exposure.append({
                "fold": fold, "test_season": str(fold_spec["test_season"]),
                "column": column,
                "train_rows": len(train_rows),
                "train_rows_clearing_burn_in": len(clean_rows),
                "sd_used": used, "sd_clearing_burn_in": clean,
                "sd_test": test, "sd_inflation": inflation,
                "test_range_in_used_sd": float(
                    (np.nanmax(values[test_rows]) - np.nanmin(values[test_rows]))
                    / used) if used else np.nan,
            })

            span = exposure[-1]["test_range_in_used_sd"]

            print("  {:<5} {:<22} {:>11.4f} {:>11} {:>11.4f} {:>9} "
                  "{:>9}".format(
                      fold, column, used,
                      "-" if not np.isfinite(clean) else "{:.4f}".format(clean),
                      test,
                      "-" if not np.isfinite(inflation)
                      else "{:.2f}x".format(inflation),
                      "{:.2f} SD".format(span)))

        print()

    exposure_frame = pd.DataFrame(exposure)

    print("  SD >=380 is blank at fold 1: its training set is 2021-22 alone,")
    print("  every row of which is below the burn-in. That is A2.2's finding,")
    print("  re-measured on D2's own training rows rather than inherited.")
    print()
    print("  'test span' is the full range of the column across the test")
    print("  season, expressed in the standardised units the fit works in.")
    print("  A column whose entire test season spans a fifth of one SD is a")
    print("  near-constant to the model however much signal it carries. The")
    print("  penalty then shrinks a feature that arrived already compressed,")
    print("  which is the mechanism by which D2 - D1 is a LOWER bound rather")
    print("  than a point estimate.")
    print()

    for path, data in ((SENSITIVITY_OUTPUT, sensitivity),
                       (BURN_IN_OUTPUT, exposure_frame)):
        data.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
        print("  {}".format(path))

    print()
    print("  Elapsed: {:.1f}s".format(time.time() - started))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
