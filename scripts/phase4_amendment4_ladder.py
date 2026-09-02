"""
===============================================================================
PHASE 4 - AMENDMENTS 4 AND 5:  D2 RESCALED, AND D2-STATIC
===============================================================================

TWO CHANGES, DECLARED BEFORE ANYTHING HERE WAS FITTED, AND NOTHING ELSE MOVES.

    AMENDMENT 4   the three Dixon-Coles-derived columns take robust scaling -
                  median, and IQR / 1.3489795. Same target quantity, an
                  estimator a handful of 47.26s cannot move. Selected by
                  SOURCE (read out of a DC fit whose window can be too short
                  to identify the strengths), not by which coefficients came
                  back zero. rel_elo_diff does not qualify: Elo has no
                  fitting window.

    AMENDMENT 5   D2-static. Identical features, folds, grid, family and
                  scaling; the state is frozen at season start rather than
                  refit per calendar date. D2 - D2static is the recency
                  contribution in the ladder's own units, with none of the
                  cross-family caveat the 0.036 comparison carried.

WHAT IS NOT RE-FITTED

    D1 carries no qualifying column, so Amendment 4 cannot touch it. It is
    re-run here anyway and DS13 asserts it reproduces the COMMITTED artefact
    bit for bit - which is what proves the pipeline edit changed nothing it
    was not supposed to. The original D2 is re-run under the same assertion.

    Section 6 is NOT changed. Its alternative scores better and that is
    precisely why it stays. See A4.4.

    D3 and D4 are NOT run.

OUTPUTS ARE WRITTEN TO THEIR OWN FILENAMES. The first run's artefacts are
under the manifest and are not overwritten, so the amendment can be given a
size rather than replacing the thing it changed.
===============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import (CLASS_INDEX, CLASSES,  # noqa: E402
                                       evaluate, validate_probabilities)
from phase3_feature_builder import Audit, banner, configure_stdout, block_of  # noqa: E402

import phase2_poisson_dixon_coles as DC          # noqa: E402
import phase3_ablation_ladder as L3              # noqa: E402
import phase3_regularisation_sensitivity as I4   # noqa: E402
import phase4_dynamic_state as STATE             # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR

FOLD_OUTPUT = OUTPUTS_DIR / "phase4_a4_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase4_a4_pooled.csv"
DELTA_OUTPUT = OUTPUTS_DIR / "phase4_a4_deltas.csv"
CURVE_OUTPUT = OUTPUTS_DIR / "phase4_a4_lambda_curves.csv"
COEF_OUTPUT = OUTPUTS_DIR / "phase4_a4_dynamic_coefficients.csv"
SCALE_OUTPUT = OUTPUTS_DIR / "phase4_a4_contamination.csv"
PRED_OUTPUT = OUTPUTS_DIR / "phase4_a4_predictions.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase4_a4_audit.csv"
STATIC_STATE_OUTPUT = OUTPUTS_DIR / "phase4_static_state.csv"

COMMITTED_FOLDS = LADDER.FOLD_OUTPUT

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

BURN_IN = 380            # Amendment 2 A2.1


# ============================================================
# RUNGS
# ============================================================

RUNGS = ("D1", "D2_original", "D2_rescaled", "D2_static")

RUNG_LABEL = {
    "D1": "results-derived, 88 columns",
    "D2_original": "dynamic state, SD scaling (the first run)",
    "D2_rescaled": "dynamic state, Amendment 4 robust scaling",
    "D2_static": "state frozen at season start, robust scaling",
}


def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 4 - AMENDMENT 4 (ROBUST SCALING) AND AMENDMENT 5 (D2-STATIC)")

    print("  robust columns : {}".format(", ".join(LADDER.DC_DERIVED_COLUMNS)))
    print("  robust scale   : median, IQR / {:.16f}".format(
        LADDER.IQR_TO_SIGMA))
    print("  held fixed     : features, folds, grid, inner CV, solver, seed")
    print("  section 6      : UNCHANGED (A4.4)")
    print("  D3 / D4        : NOT RUN")
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches()
    features = L3.load_features(matches)

    matches = matches.copy()
    matches["match_id"] = matches.index
    matches["role_is_test"] = matches["season"].isin(
        [str(f["test_season"]) for f in spec["folds"]])

    print("  generating state...")

    dynamic_state, refits = STATE.build(matches)
    static_state, static_fits = STATE.build_static(matches)

    print("  {} per-date refits, {} per-season fits, {:.1f}s".format(
        refits, static_fits, time.time() - started))
    print()

    frame = matches.copy()

    dynamic = dynamic_state.set_index("match_id").loc[
        frame["match_id"], LADDER.DYNAMIC_COLUMNS].reset_index(drop=True)
    static = static_state.set_index("match_id").loc[
        frame["match_id"], LADDER.DYNAMIC_COLUMNS].reset_index(drop=True)

    labels = np.array([CLASS_INDEX[r] for r in frame["result"]], dtype=int)
    results = frame["result"].to_numpy()
    blocks = I4.date_blocks(frame)

    base = LADDER.d1_features(features)

    d1_matrix, d1_names, d1_mask = LADDER.build_design(features, base)
    d2_matrix, d2_names, d2_mask = LADDER.build_design(features, base, dynamic)
    ds_matrix, ds_names, ds_mask = LADDER.build_design(features, base, static)

    matrices = {"D1": d1_matrix, "D2_original": d2_matrix,
                "D2_rescaled": d2_matrix, "D2_static": ds_matrix}
    masks = {"D1": d1_mask, "D2_original": d2_mask,
             "D2_rescaled": d2_mask, "D2_static": ds_mask}

    robust = {
        "D1": None,
        "D2_original": None,
        "D2_rescaled": LADDER.robust_mask(d2_names),
        "D2_static": LADDER.robust_mask(ds_names),
    }

    audit.record(
        "A4a", "the robust mask selects exactly Amendment 4's three columns",
        3, int(robust["D2_rescaled"].sum()),
        int(robust["D2_rescaled"].sum()) == 3,
        "flagged: {} | NOT flagged, on the source criterion of A4.1: "
        "rel_elo_diff".format(
            ", ".join(n for n, f in zip(d2_names, robust["D2_rescaled"]) if f)))

    audit.record(
        "A4b", "no column both passes through unstandardised and takes "
               "robust scaling",
        0, int((robust["D2_rescaled"] & d2_mask).sum()),
        not (robust["D2_rescaled"] & d2_mask).any(),
        "A4.4: Amendment 4 applies only to continuous columns, so section 6's "
        "rule and the robust rule can never contend for the same column")

    # ============================================================
    banner("1. THE RUNGS")

    fold_tables, curve_tables, proba_by_rung, diagnostics = {}, {}, {}, {}

    for name in RUNGS:

        folds, curves, proba, diag = LADDER.run_rung(
            name, matrices[name], masks[name], frame, spec, labels, results,
            blocks, robust[name])

        for entry in diag:
            entry["passthrough"] = masks[name]

        fold_tables[name] = folds
        curve_tables[name] = curves
        proba_by_rung[name] = proba
        diagnostics[name] = diag

    all_folds = pd.concat([fold_tables[r] for r in RUNGS], ignore_index=True)

    for name in RUNGS:

        table = fold_tables[name]

        print("  {}  -  {}".format(name, RUNG_LABEL[name]))
        print("  {:<5} {:<11} {:>6} {:>8} {:>6} {:>6} {:>7} {:>7} {:>8} {:>7} {:>7}".format(
            "fold", "test", "width", "lambda", "EPV", "acc", "bal_acc",
            "mac_f1", "logloss", "brier", "RPS"))
        print("  " + "-" * 92)

        for _i, row in table.iterrows():
            print("  {:<5} {:<11} {:>6} {:>8g} {:>6.2f} {:>6.3f} {:>7.3f} "
                  "{:>7.3f} {:>8.4f} {:>7.4f} {:>7.4f}".format(
                      int(row["fold"]), row["test_season"],
                      int(row["design_width"]), row["selected_lambda"],
                      row["epv"], row["accuracy"], row["balanced_accuracy"],
                      row["macro_f1"], row["log_loss"], row["brier_score"],
                      row["rps"]))

        statuses = sorted(set(table["g6_status"]))
        print()
        print("    G6: {}".format(
            "PASS at all four folds" if statuses == ["PASS"]
            else " | ".join("f{} {}".format(int(r["fold"]), r["g6_status"])
                            for _i, r in table.iterrows())))
        print()

    failed = all_folds[all_folds["g6_status"].str.startswith("FAIL")]

    audit.record(
        "G6", "no applicable rung/fold selects a lambda on a grid boundary",
        0, len(failed), len(failed) == 0,
        "EPV {:.2f} to {:.2f} across the fitted rungs, so the gate is live "
        "everywhere".format(float(all_folds["epv"].min()),
                            float(all_folds["epv"].max())))

    # ---- DS13: the pipeline edit changed nothing it should not have ------
    committed = pd.read_csv(COMMITTED_FOLDS, float_precision=FLOAT_PRECISION)

    worst, moved = 0.0, 0

    for here, there in (("D1", "D1"), ("D2_original", "D2")):

        mine = fold_tables[here].sort_values("fold")
        theirs = committed[committed["rung"] == there].sort_values("fold")

        for metric in METRICS:
            worst = max(worst, float(np.abs(
                mine[metric].to_numpy() - theirs[metric].to_numpy()).max()))

        moved += int((mine["selected_lambda"].to_numpy()
                      != theirs["selected_lambda"].to_numpy()).sum())

    audit.record(
        "DS13", "D1 and the original D2 reproduce the COMMITTED artefact "
                "after the pipeline edit",
        "< 1e-12 and 0 lambda moves",
        "{:.3e}, {} moves".format(worst, moved),
        worst < 1e-12 and moved == 0,
        "the robust path is opt-in and defaults to off. Verified against "
        "outputs/phase4_ladder_fold_summary.csv as committed at c2528c5, not "
        "against an in-memory copy of it")

    # ============================================================
    banner("2. POOLED, AND THE REFERENCES")

    pooled = {}

    for name in RUNGS:
        rows, proba, scores = LADDER.pool(proba_by_rung[name], results, spec)
        pooled[name] = (rows, proba, scores)

    order = pooled["D1"][0]
    actual = results[order]

    d0_folds, d0_proba = LADDER.run_d0(frame, spec, results)
    _rows0, d0_pooled_proba, d0_scores = LADDER.pool(d0_proba, results, spec)

    references = LADDER.reference_probabilities(frame)
    reference_proba = {k: LADDER.reference_array(references[k], order)
                       for k in references}
    reference_scores = {k: evaluate(actual, v)
                        for k, v in reference_proba.items()}

    display = ([("D0  base rate", d0_scores)]
               + [("{:<12} {}".format(n, ""), pooled[n][2]) for n in RUNGS]
               + [("Elo v1", reference_scores["elo_v1"]),
                  ("Poisson walk-forward",
                   reference_scores["poisson_walkforward"]),
                  ("Dixon-Coles walk-forward",
                   reference_scores["dc_walkforward"])])

    print("  {:<26} {:>7} {:>8} {:>8} {:>9} {:>8} {:>8}".format(
        "model", "acc", "bal_acc", "macro_f1", "logloss", "brier", "RPS"))
    print("  " + "-" * 80)

    for label, scores in display:
        print("  {:<26} {:>7.4f} {:>8.4f} {:>8.4f} {:>9.5f} {:>8.4f} "
              "{:>8.5f}".format(
                  label.strip(), scores["accuracy"],
                  scores["balanced_accuracy"], scores["macro_f1"],
                  scores["log_loss"], scores["brier_score"], scores["rps"]))

    print()

    # ============================================================
    banner("3. THE DELTAS")

    proba_of = {n: pooled[n][1] for n in RUNGS}
    proba_of["D0"] = d0_pooled_proba
    proba_of.update(reference_proba)

    deltas = []

    pairs = [
        ("D2rescaled - D1", "D2_rescaled", "D1"),
        ("D2rescaled - D2orig", "D2_rescaled", "D2_original"),
        ("D2rescaled - D2static", "D2_rescaled", "D2_static"),
        ("D2static - D1", "D2_static", "D1"),
        ("D2rescaled - D0", "D2_rescaled", "D0"),
        ("D2rescaled - Elo v1", "D2_rescaled", "elo_v1"),
        ("D2rescaled - Poisson", "D2_rescaled", "poisson_walkforward"),
        ("D2rescaled - DixonColes", "D2_rescaled", "dc_walkforward"),
    ]

    for label, left, right in pairs:
        deltas.append(LADDER.compare(label, left, right, proba_of[left],
                                     proba_of[right], actual))

    def show(rows, title):
        print("  {}".format(title))
        print()
        print("  {:<24} {:>9} {:>20} {:>9} {:>20}  {}".format(
            "comparison", "d_logloss", "95% CI", "d_RPS", "95% CI", "verdict"))
        print("  " + "-" * 114)
        for row in rows:
            print("  {:<24} {:>+9.5f} {:>20} {:>+9.5f} {:>20}  {}".format(
                row["comparison"] if row["scope"] == "pooled"
                else "  {}".format(row["scope"]),
                row["log_loss_delta"],
                "[{:+.5f}, {:+.5f}]".format(row["log_loss_ci_lo"],
                                            row["log_loss_ci_hi"]),
                row["rps_delta"],
                "[{:+.5f}, {:+.5f}]".format(row["rps_ci_lo"],
                                            row["rps_ci_hi"]),
                row["verdict"]))
        print()

    show(deltas, "POOLED OVER 1,520 MATCHES  (negative favours the LEFT model)")

    # ---- per fold, for the two comparisons that carry the session --------
    per_fold = []

    for label, left, right in (("D2rescaled - D1", "D2_rescaled", "D1"),
                               ("D2rescaled - D2static", "D2_rescaled",
                                "D2_static")):

        for fold_spec in spec["folds"]:

            fold = int(fold_spec["fold"])
            rows_left, proba_left = proba_by_rung[left][fold]
            _r, proba_right = proba_by_rung[right][fold]

            row = LADDER.compare(label, left, right, proba_left, proba_right,
                                 results[rows_left],
                                 scope="fold {} ({})".format(
                                     fold, fold_spec["test_season"]))
            row["fold"] = fold
            per_fold.append(row)

    show([r for r in per_fold if r["comparison"] == "D2rescaled - D1"],
         "D2rescaled - D1, PER FOLD")
    show([r for r in per_fold if r["comparison"] == "D2rescaled - D2static"],
         "D2rescaled - D2static, PER FOLD")

    deltas.extend(per_fold)

    # ---- A5.2's declared folds 2-4 pooling -------------------------------
    later = np.concatenate([proba_by_rung["D1"][f][0] for f in (2, 3, 4)])
    later_actual = results[later]

    subset = []

    for label, left, right in (("D2rescaled - D2static", "D2_rescaled",
                                "D2_static"),
                               ("D2rescaled - D1", "D2_rescaled", "D1")):
        left_proba = np.vstack([proba_by_rung[left][f][1] for f in (2, 3, 4)])
        right_proba = np.vstack([proba_by_rung[right][f][1] for f in (2, 3, 4)])
        subset.append(LADDER.compare(label, left, right, left_proba,
                                     right_proba, later_actual,
                                     scope="folds 2-4"))

    show(subset, "FOLDS 2-4 ONLY  (declared in A5.2: the folds whose TRAINING "
                 "rows carry frozen DC state at all)")

    deltas.extend(subset)

    # ---- G9: A5.2's prediction ------------------------------------------
    gap_f1 = float(np.abs(proba_by_rung["D2_static"][1][1]
                          - proba_by_rung["D1"][1][1]).max())
    lambda_f1_static = float(
        fold_tables["D2_static"].set_index("fold").loc[1, "selected_lambda"])
    lambda_f1_d1 = float(
        fold_tables["D1"].set_index("fold").loc[1, "selected_lambda"])

    audit.record(
        "G9", "A5.2's prediction: D2-static at fold 1 equals D1 at fold 1, "
              "bit for bit, and selects the same lambda",
        "0.0 and equal", "{:.3e}, {:g} vs {:g}".format(
            gap_f1, lambda_f1_static, lambda_f1_d1),
        gap_f1 == 0.0 and lambda_f1_static == lambda_f1_d1,
        "2021-22 is the first season, so it has no frozen DC state and every "
        "season-start rating in it is 1500 flat. Fold 1 trains on it alone, "
        "so all four dynamic columns are constant there and the penalty "
        "drives them to zero. Declared in advance and asserted, not excused")

    # ---- G9b/G9c: if G9 fails, say WHETHER the substance survived ----------
    # A5.2 predicted two things - that the block goes inert at fold 1, and
    # that the two rungs therefore agree exactly. The first is checkable on
    # its own terms and is checked here, so a failure of the strict form is
    # not reported as though the mechanism had failed too.
    static_entry = [e for e in diagnostics["D2_static"] if e["fold"] == 1][0]
    d1_entry = [e for e in diagnostics["D1"] if e["fold"] == 1][0]

    dynamic_weights = static_entry["weights"][
        ds_matrix.shape[1] - len(LADDER.DYNAMIC_COLUMNS):ds_matrix.shape[1]]

    audit.record(
        "G9b", "the MECHANISM A5.2 predicted: at fold 1 every dynamic-state "
               "coefficient is exactly zero",
        "0.0", "{:.3e}".format(float(np.abs(dynamic_weights).max())),
        float(np.abs(dynamic_weights).max()) == 0.0,
        "the block is inert at fold 1 exactly as declared - 2021-22 supplies "
        "no frozen state and no Elo spread, so the four columns are constant "
        "and the penalty zeroes them")

    # the two designs are different SIZES, so the Newton solve runs on a
    # 279x279 Hessian rather than 267x267 and the SHARED coefficients round
    # differently in their last bits. Measured rather than asserted.
    shared_gap = float(np.abs(
        static_entry["weights"][:d1_matrix.shape[1]]
        - d1_entry["weights"][:d1_matrix.shape[1]]).max())

    audit.measure(
        "G9c", "largest disagreement in the 88 SHARED coefficients at fold 1",
        "{:.3e}".format(shared_gap),
        "the two designs differ in width, so fit_multinomial solves a "
        "{}x{} system in one case and {}x{} in the other. The shared "
        "coefficients therefore round differently in their last bits, which "
        "is where G9's probability gap comes from - not from the dynamic "
        "columns, which G9b shows are exactly zero. This is a MEASUREMENT of "
        "the explanation, not the explanation on its own".format(
            (d1_matrix.shape[1] + 1) * 3, (d1_matrix.shape[1] + 1) * 3,
            (ds_matrix.shape[1] + 1) * 3, (ds_matrix.shape[1] + 1) * 3))

    # ============================================================
    banner("4. THE BLOCK, RESCALED")

    coefficients = []

    for name in ("D2_original", "D2_rescaled", "D2_static"):

        width = matrices[name].shape[1]

        for entry in diagnostics[name]:

            for offset, column in enumerate(LADDER.DYNAMIC_COLUMNS):

                index = width - len(LADDER.DYNAMIC_COLUMNS) + offset
                beta = entry["weights"][index]

                coefficients.append({
                    "rung": name, "fold": entry["fold"], "column": column,
                    "beta_home": float(beta[0]), "beta_draw": float(beta[1]),
                    "beta_away": float(beta[2]),
                    "beta_l2": float(np.sqrt(np.sum(beta ** 2))),
                    "train_centre": float(entry["mean"][index]),
                    "train_scale": float(entry["sd"][index]),
                    "robust_scaled": bool(robust[name] is not None
                                          and robust[name][index]),
                })

    coefficient_frame = pd.DataFrame(coefficients)

    print("  COEFFICIENT L2 NORM per dynamic column, original vs rescaled")
    print("  (sqrt of the sum of the three class coefficients squared)")
    print()
    print("  {:<22} {:>8} {:>10} {:>10} {:>9}".format(
        "column", "fold", "original", "rescaled", "ratio"))
    print("  " + "-" * 64)

    for column in LADDER.DYNAMIC_COLUMNS:
        for fold in (1, 2, 3, 4):
            original = float(coefficient_frame[
                (coefficient_frame["rung"] == "D2_original")
                & (coefficient_frame["fold"] == fold)
                & (coefficient_frame["column"] == column)]["beta_l2"].iloc[0])
            rescaled = float(coefficient_frame[
                (coefficient_frame["rung"] == "D2_rescaled")
                & (coefficient_frame["fold"] == fold)
                & (coefficient_frame["column"] == column)]["beta_l2"].iloc[0])
            print("  {:<22} {:>8} {:>10.5f} {:>10.5f} {:>9}".format(
                column if fold == 1 else "", fold, original, rescaled,
                "-" if original == 0 else "{:.1f}x".format(rescaled / original)))
        print()

    print("  FULL PER-FOLD COEFFICIENTS, D2 rescaled")
    print()
    print("  {:<5} {:<22} {:>10} {:>10} {:>10} {:>11}".format(
        "fold", "column", "beta_H", "beta_D", "beta_A", "train scale"))
    print("  " + "-" * 74)

    for _i, row in coefficient_frame[
            coefficient_frame["rung"] == "D2_rescaled"].iterrows():
        print("  {:<5} {:<22} {:>+10.4f} {:>+10.4f} {:>+10.4f} "
              "{:>11.4f}".format(
                  int(row["fold"]), row["column"], row["beta_home"],
                  row["beta_draw"], row["beta_away"], row["train_scale"]))

    print()

    print("  FULL PER-FOLD COEFFICIENTS, D2-static")
    print()
    print("  {:<5} {:<22} {:>10} {:>10} {:>10} {:>11}".format(
        "fold", "column", "beta_H", "beta_D", "beta_A", "train scale"))
    print("  " + "-" * 74)

    for _i, row in coefficient_frame[
            coefficient_frame["rung"] == "D2_static"].iterrows():
        print("  {:<5} {:<22} {:>+10.4f} {:>+10.4f} {:>+10.4f} "
              "{:>11.4f}".format(
                  int(row["fold"]), row["column"], row["beta_home"],
                  row["beta_draw"], row["beta_away"], row["train_scale"]))

    print()

    # ---- the contamination table, recomputed -----------------------------
    banner("5. THE CONTAMINATION TABLE, RECOMPUTED")

    windows = dynamic_state["window_matches"].to_numpy()

    print("  'scale used' is what the fit divided by. Original = the sample")
    print("  SD; rescaled = median/IQR per A4.2. 'clean' is the SD of the")
    print("  training rows clearing the 380 burn-in.")
    print()
    print("  {:<5} {:<22} {:>10} {:>10} {:>8} {:>8} {:>9} {:>9}".format(
        "fold", "column", "orig scale", "A4 scale", "clean", "test SD",
        "orig span", "A4 span"))
    print("  " + "-" * 88)

    contamination = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])

        train_rows = np.flatnonzero(
            frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
        test_rows = np.flatnonzero(
            (frame["season"] == str(fold_spec["test_season"])).to_numpy())

        clean_rows = train_rows[windows[train_rows] >= BURN_IN]

        original_entry = [e for e in diagnostics["D2_original"]
                          if e["fold"] == fold][0]
        rescaled_entry = [e for e in diagnostics["D2_rescaled"]
                          if e["fold"] == fold][0]

        for offset, column in enumerate(LADDER.DYNAMIC_COLUMNS):

            index = d2_matrix.shape[1] - len(LADDER.DYNAMIC_COLUMNS) + offset

            values = dynamic[column].to_numpy()

            original_scale = float(original_entry["sd"][index])
            rescaled_scale = float(rescaled_entry["sd"][index])
            clean = (float(np.nanstd(values[clean_rows], ddof=0))
                     if len(clean_rows) else np.nan)
            test_sd = float(np.nanstd(values[test_rows], ddof=0))
            span = float(np.nanmax(values[test_rows])
                         - np.nanmin(values[test_rows]))

            row = {
                "fold": fold, "test_season": str(fold_spec["test_season"]),
                "column": column,
                "robust_scaled": bool(robust["D2_rescaled"][index]),
                "scale_original": original_scale,
                "scale_amendment4": rescaled_scale,
                "sd_clearing_burn_in": clean, "sd_test": test_sd,
                "test_span_original_units": span / original_scale,
                "test_span_amendment4_units": span / rescaled_scale,
                "scale_shrunk_by": original_scale / rescaled_scale,
            }
            contamination.append(row)

            print("  {:<5} {:<22} {:>10.4f} {:>10.4f} {:>8} {:>8.4f} "
                  "{:>9.2f} {:>9.2f}".format(
                      fold, column, original_scale, rescaled_scale,
                      "-" if not np.isfinite(clean) else "{:.4f}".format(clean),
                      test_sd, row["test_span_original_units"],
                      row["test_span_amendment4_units"]))

        print()

    contamination_frame = pd.DataFrame(contamination)

    # ============================================================
    banner("6. THE LEAKAGE SUITE ON THE NEW RUNGS")

    LADDER.ds0_pipeline_anchor(audit, d1_matrix, labels, spec, frame, d2_mask)

    elo_frame = frame[["season", "date", "home_team", "away_team"]].copy()
    elo_source = STATE.load_elo_state(matches)
    for column in ("home_elo_before", "away_elo_before", "home_elo_after",
                   "away_elo_after", "home_transition", "away_transition"):
        elo_frame[column] = elo_source[column].to_numpy()

    LADDER.ds1_temporal(audit, matches, dynamic_state, elo_frame)

    # DS1d: the same property for the STATIC state, whose window rule differs
    season_first = matches.groupby("season")["date"].min()
    expected = matches["season"].map(
        lambda s: int((matches["date"] < season_first[s]).sum())).to_numpy()

    audit.record(
        "DS1d", "each match's STATIC fitting window is exactly the count of "
                "matches strictly before its season's first date",
        0, int((expected != static_state["window_matches"].to_numpy()).sum()),
        int((expected != static_state["window_matches"].to_numpy()).sum()) == 0,
        "the freezing rule of A5.1, checked against the data rather than "
        "against the code that implements it")

    LADDER.ds2_no_test_row_fits(audit, diagnostics, matrices, frame, spec)

    offenders = LADDER.g10_scale_domain(audit, diagnostics, matrices)

    # The imputed mass, measured per rung and fold, because it is what drives
    # G10 and it is not visible from the scale alone.
    for name, state in (("D2_rescaled", dynamic_state),
                        ("D2_static", static_state)):

        missing = ~state.set_index("match_id").loc[
            frame["match_id"], "has_state"].to_numpy()

        shares = []

        for fold_spec in spec["folds"]:
            train_rows = np.flatnonzero(
                frame["season"].isin(fold_spec["train_seasons"]).to_numpy())
            shares.append("f{} {:.0%}".format(
                int(fold_spec["fold"]), missing[train_rows].mean()))

        audit.measure(
            "G10b-{}".format(name),
            "share of {}'s TRAINING rows with no DC state, imputed to the "
            "training median".format(name),
            ", ".join(shares),
            "at 50% the imputed value necessarily occupies both quartiles and "
            "the interquartile range is identically zero; below that it still "
            "compresses them. This is the quantity G10 is downstream of")
    LADDER.ds3_widths(audit, features,
                      {"D1": d1_matrix.shape[1], "D2": d2_matrix.shape[1]})

    audit.record(
        "DS3d", "D2-static has the same design width as D2",
        d2_matrix.shape[1], ds_matrix.shape[1],
        ds_matrix.shape[1] == d2_matrix.shape[1],
        "the rungs differ in WHEN the state stopped updating and in nothing "
        "else, so a width difference would mean the comparison was not the "
        "one A5.3 declared")

    baseline_lambdas = {int(r["fold"]): r["selected_lambda"]
                        for _i, r in fold_tables["D2_rescaled"].iterrows()}

    LADDER.ds4_ds5_corruption(
        audit, features, frame, spec, labels, results, blocks, dynamic,
        baseline_lambdas, proba_by_rung["D2_rescaled"],
        robust["D2_rescaled"], "D2 rescaled")

    LADDER.ds6_base_rate(audit, d0_folds)
    LADDER.ds7_frozen_blocks(audit, features)

    determinism_baseline = {
        name: proba_by_rung[name] for name in
        ("D2_rescaled", "D2_static")}
    determinism_baseline.update({
        "D2_rescaled_lambda": baseline_lambdas,
        "D2_static_lambda": {int(r["fold"]): r["selected_lambda"]
                             for _i, r in fold_tables["D2_static"].iterrows()},
    })

    LADDER.ds8_determinism(
        audit, ("D2_rescaled", "D2_static"), matrices, masks, frame, spec,
        labels, results, blocks, determinism_baseline, robust)

    contract_sets = [("{} pooled".format(n), pooled[n][1]) for n in RUNGS]
    for name in RUNGS:
        for fold, (_rows, proba) in proba_by_rung[name].items():
            contract_sets.append(("{} fold {}".format(name, fold), proba))

    LADDER.ds9_contract(audit, contract_sets)
    LADDER.ds10_manifest(audit)

    dynamic_indexed = dynamic_state.set_index("match_id").loc[
        frame["match_id"]].reset_index()
    LADDER.ds11_anchor(audit, frame, dynamic_indexed, spec)

    # ---- G8: the static state must BE arm A / dc_static on the test rows --
    static_indexed = static_state.set_index("match_id").loc[
        frame["match_id"]].reset_index()

    stored = pd.read_csv(LADDER.DC_RESULTS, float_precision=FLOAT_PRECISION)
    stored["date"] = pd.to_datetime(stored["date"], format="%Y-%m-%d")
    stored = stored[stored["variant"] == "dc_static"]

    keyed = frame[["match_id", "date", "home_team", "away_team"]].merge(
        static_indexed[["match_id", "lambda_home", "lambda_away"]],
        on="match_id")

    joined = stored.merge(
        keyed, left_on=["date", "home", "away"],
        right_on=["date", "home_team", "away_team"],
        suffixes=("_p2", "_new"))

    g8 = max(
        float(np.abs(joined["lambda_home_p2"]
                     - joined["lambda_home_new"]).max()),
        float(np.abs(joined["lambda_away_p2"]
                     - joined["lambda_away_new"]).max()))

    audit.record(
        "G8", "the generated STATIC state reproduces Phase 2's stored "
              "dc_static lambdas on every test row",
        "< 1e-9", "{:.3e}".format(g8), g8 < 1e-9,
        "{} test rows. A5.1 claims the freezing rule coincides with tier 2's "
        "ARM_STATIC by construction; this is the assertion of it. Without it "
        "D2-static would be a rung named after an arm it might not "
        "match".format(len(joined)))

    LADDER.ds12_ordering(audit, diagnostics, frame)

    audit.print_rows()

    # ============================================================
    banner("7. WRITING")

    pooled_table = [{"model": "D0", "n": d0_scores["n"],
                     **{m: d0_scores[m] for m in METRICS}}]

    for name in RUNGS:
        pooled_table.append({"model": name, "n": pooled[name][2]["n"],
                             **{m: pooled[name][2][m] for m in METRICS}})

    for key, scores in reference_scores.items():
        pooled_table.append({"model": key, "n": scores["n"],
                             **{m: scores[m] for m in METRICS}})

    predictions = pd.DataFrame({
        "match_id": frame["match_id"].to_numpy()[order],
        "season": frame["season"].to_numpy()[order],
        "date": frame["date"].to_numpy()[order],
        "home_team": frame["home_team"].to_numpy()[order],
        "away_team": frame["away_team"].to_numpy()[order],
        "result": actual,
    })

    for key, proba in proba_of.items():
        for position, outcome in enumerate(CLASSES):
            predictions["{}_p_{}".format(key, outcome)] = proba[:, position]

    for path, data in ((FOLD_OUTPUT, all_folds),
                       (POOLED_OUTPUT, pd.DataFrame(pooled_table)),
                       (DELTA_OUTPUT, pd.DataFrame(deltas)),
                       (CURVE_OUTPUT, pd.concat(list(curve_tables.values()),
                                                ignore_index=True)),
                       (COEF_OUTPUT, coefficient_frame),
                       (SCALE_OUTPUT, contamination_frame),
                       (PRED_OUTPUT, predictions),
                       (AUDIT_OUTPUT, audit.frame()),
                       (STATIC_STATE_OUTPUT, static_state)):
        data.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)
        print("  {}".format(path))

    print()

    failures = audit.failures

    print("  Checks run    : {}".format(len(audit.rows)))
    print("  Checks failed : {}".format(len(failures)))
    print("  Elapsed       : {:.1f}s".format(time.time() - started))
    print()

    for row in failures:
        print("  FAILED  {}  {}".format(row["test_id"], row["test"]))

    if failures:
        print()

    print("  {}".format("PASS" if not failures else "FAIL"))
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
