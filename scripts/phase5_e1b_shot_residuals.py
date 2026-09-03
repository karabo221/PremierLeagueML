"""
===============================================================================
PHASE 5 - INSTRUMENT E1b:  ROLLING SHOT RESIDUALS AS A LADDER BLOCK
===============================================================================

Pre-declaration: PHASE5_E1_SHOT_PREDECLARATION.txt section 6
sha256 d385bfd4d081f40e4d88a96939fde005db9d539559d21b234ebcaca5eff6eca4

THE BASE IS D2 RESCALED, AND THAT WAS NOT A CHOICE MADE HERE. E7.1 committed
the gate order before E1a ran: E1a first, then E1b on whichever base E1a leaves
standing - D2 rescaled if E1a is null or inconclusive. E1a - Dixon-Coles came
back NOT SIGNIFICANT, so the base is D2 rescaled. Written down in advance
precisely so it could not be picked afterwards.

THE SIGNAL. Who has been finishing above or below their chances: a team's SoT
differential over its last five matches minus its goal differential over the
same five. Orthogonal to a rating by construction, because it is the DIFFERENCE
between what the shots implied and what the scoreline recorded.

WINDOW: LAST FIVE, WITHIN SEASON. Not a chosen parameter - it is the project's
existing convention, matching D1's last5_pts_before and last5_mp_before. A new
window length would be a tuned one, and the tuning would happen on these
outer-test rows.

SCALING: SECTION 6's STANDARD TREATMENT, NOT AMENDMENT 4's ROBUST RULE. The
robust rule applies to DC-DERIVED columns and selects three of them BY NAME.
These residuals come from a rolling arithmetic window and no fitted model, so
they are not in that class. A4a continues to assert the robust mask still finds
exactly three columns, which is what would catch this drifting.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import CLASS_INDEX, CLASSES, evaluate  # noqa: E402
from phase3_feature_builder import (Audit, banner,  # noqa: E402
                                    configure_stdout, declare_block)

import phase3_ablation_ladder as L3              # noqa: E402
import phase3_regularisation_sensitivity as I4   # noqa: E402
import phase4_dynamic_state as STATE             # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402
import phase5_e1a_sot_ratings as E1A             # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR

D34_PREDICTIONS = OUTPUTS_DIR / "phase4_d34_predictions.csv"
MARKET_PROBABILITIES = OUTPUTS_DIR / "phase5_market_probabilities.csv"
E1A_PREDICTIONS = OUTPUTS_DIR / "phase5_e1a_predictions.csv"
A4_FOLDS = OUTPUTS_DIR / "phase4_a4_fold_summary.csv"

FOLD_OUTPUT = OUTPUTS_DIR / "phase5_e1b_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase5_e1b_pooled.csv"
DELTA_OUTPUT = OUTPUTS_DIR / "phase5_e1b_deltas.csv"
COEF_OUTPUT = OUTPUTS_DIR / "phase5_e1b_coefficients.csv"
CURVE_OUTPUT = OUTPUTS_DIR / "phase5_e1b_lambda_curves.csv"
RESIDUAL_OUTPUT = OUTPUTS_DIR / "phase5_e1b_residual_features.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase5_e1b_audit.csv"

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

WINDOW = 5

RESIDUAL_COLUMNS = ["home_sot_residual_last5", "away_sot_residual_last5",
                    "rel_sot_residual_diff"]

AVAILABILITY_COLUMNS = ["home_sot_residual_available",
                        "away_sot_residual_available",
                        "rel_sot_residual_available"]

# DECLARED, so block_of() classifies them instead of defaulting them into the
# Phase 1 backbone. That default, plus d1_features() selecting by exclusion,
# is what put these six columns inside the control arm on the first run.
NEW_COLUMNS = declare_block("E_shot_residual",
                            RESIDUAL_COLUMNS + AVAILABILITY_COLUMNS)

RUNGS = ("D2_rescaled", "E1b")


# ============================================================
# THE BLOCK
# ============================================================

def build_residuals(matches):
    """The six columns of E6.5, built strictly from earlier matches.

    For each team, at each of its own matches, the window is that team's
    PREVIOUS FIVE matches IN THE SAME SEASON. The match itself is excluded and
    the window never crosses a season boundary - Phase 1's T2 rule, which this
    block inherits rather than re-invents.

    A team with fewer than five prior matches in the season gets NaN and an
    availability indicator of False. NaN is then imputed to the TRAINING median
    by the pipeline's own declared rule, the same rule every other column
    takes. Nothing is back-filled from the prior season; that would cross the
    boundary T2 exists to prevent.
    """

    # One row per team per match, so a team's history is a simple time series.
    home = pd.DataFrame({
        "season": matches["season"], "date": matches["date"],
        "match_id": matches["match_id"], "team": matches["home_team"],
        "sot_for": matches["HST"], "sot_against": matches["AST"],
        "goals_for": matches["home_goals"],
        "goals_against": matches["away_goals"], "side": "home"})

    away = pd.DataFrame({
        "season": matches["season"], "date": matches["date"],
        "match_id": matches["match_id"], "team": matches["away_team"],
        "sot_for": matches["AST"], "sot_against": matches["HST"],
        "goals_for": matches["away_goals"],
        "goals_against": matches["home_goals"], "side": "away"})

    sides = pd.concat([home, away], ignore_index=True)
    sides = sides.sort_values(["season", "team", "date", "match_id"])
    sides = sides.reset_index(drop=True)

    sides["sot_diff"] = sides["sot_for"] - sides["sot_against"]
    sides["goal_diff"] = sides["goals_for"] - sides["goals_against"]

    grouped = sides.groupby(["season", "team"], sort=False)

    # shift(1) is what makes it STRICTLY EARLIER: the current match is not in
    # its own window. min_periods=WINDOW is what makes "fewer than five" a NaN
    # rather than a mean of two.
    rolled_sot = grouped["sot_diff"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW).mean())

    rolled_goal = grouped["goal_diff"].transform(
        lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW).mean())

    sides["residual"] = rolled_sot - rolled_goal
    sides["available"] = sides["residual"].notna()

    # The team's own match number within the season, 1-based. Not a feature -
    # it exists so E10b can assert WHERE the block is missing rather than
    # merely that it is missing somewhere.
    sides["match_number"] = grouped.cumcount() + 1

    lookup = sides.set_index(["match_id", "side"])["residual"]
    ordinal = sides.set_index(["match_id", "side"])["match_number"]

    frame = pd.DataFrame({"match_id": matches["match_id"].to_numpy()})

    frame["home_sot_residual_last5"] = lookup.reindex(
        pd.MultiIndex.from_arrays(
            [matches["match_id"], ["home"] * len(matches)])).to_numpy()

    frame["away_sot_residual_last5"] = lookup.reindex(
        pd.MultiIndex.from_arrays(
            [matches["match_id"], ["away"] * len(matches)])).to_numpy()

    frame["rel_sot_residual_diff"] = (frame["home_sot_residual_last5"]
                                      - frame["away_sot_residual_last5"])

    frame["home_sot_residual_available"] = frame[
        "home_sot_residual_last5"].notna()
    frame["away_sot_residual_available"] = frame[
        "away_sot_residual_last5"].notna()
    frame["rel_sot_residual_available"] = (
        frame["home_sot_residual_available"]
        & frame["away_sot_residual_available"])

    for side in ("home", "away"):
        frame["{}_match_number".format(side)] = ordinal.reindex(
            pd.MultiIndex.from_arrays(
                [matches["match_id"], [side] * len(matches)])).to_numpy()

    return frame


# ============================================================
# THE RUN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 5 - INSTRUMENT E1b: ROLLING SHOT RESIDUALS")

    print("  base: D2 rescaled, fixed by E7.1 BEFORE E1a ran")
    print("  E1a - Dixon-Coles was NOT SIGNIFICANT, so the base is D2")
    print("  rescaled. The gate order was committed in advance.")
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches()
    features = L3.load_features(matches)

    matches = matches.copy()
    matches["match_id"] = matches.index
    matches["role_is_test"] = matches["season"].isin(
        [str(f["test_season"]) for f in spec["folds"]])

    matches = E1A.load_shots(matches, audit)

    # ============================================================
    banner("1. THE BLOCK")

    residuals = build_residuals(matches)

    print("  {} columns built".format(len(NEW_COLUMNS)))
    print()
    print("  {:<32} {:>8} {:>10} {:>9} {:>9}".format(
        "column", "present", "mean", "sd", "missing"))
    print("  " + "-" * 74)

    for column in RESIDUAL_COLUMNS:
        values = residuals[column]
        print("  {:<32} {:>8} {:>10.4f} {:>9.4f} {:>9}".format(
            column, int(values.notna().sum()), float(values.mean()),
            float(values.std()), int(values.isna().sum())))

    for column in AVAILABILITY_COLUMNS:
        print("  {:<32} {:>8} {:>10} {:>9} {:>9}".format(
            column, int(residuals[column].sum()), "-", "-", 0))

    print()

    # The unavailable rows are the first five matchweeks of each season, and
    # that is a structural fact worth asserting rather than assuming.
    per_season = matches.groupby("season").apply(
        lambda g: int(residuals.loc[g.index, "rel_sot_residual_available"]
                      .sum()), include_groups=False)

    print("  available rows by season: {}".format(per_season.to_dict()))
    print()

    # EXACT: a side's residual exists if and only if that side has already
    # played more than five matches in this season. Both directions checked.
    wrong = 0
    for side in ("home", "away"):
        expected = residuals["{}_match_number".format(side)] > WINDOW
        actual_available = residuals["{}_sot_residual_available".format(side)]
        wrong += int((expected != actual_available).sum())

    audit.record(
        "E10b", "the residual block is missing exactly where a team has fewer "
                "than five prior matches in the season, and nowhere else",
        0, wrong, wrong == 0,
        "the window never crosses a season boundary (E6.3), so every season "
        "restarts with five matchweeks of unavailable rows. NaN is imputed to "
        "the TRAINING median by the pipeline's declared rule and carries an "
        "availability indicator, exactly as the project's other optional "
        "columns do")

    # ---- attach to the feature frame --------------------------------------
    #
    # THE BASE IS TAKEN FIRST, FROM THE PRISTINE FRAME, AND THAT IS NOT
    # cosmetic. block_of() returns "phase1_backbone" for any name it does not
    # recognise, and d1_features() selects the backbone BY EXCLUSION - so six
    # columns attached before the base is read are silently swept into the
    # base itself. That made D2 rescaled 98 columns wide, stopped it
    # reproducing the committed artefact, and left E1b "adding" nothing.
    # E10e is the assertion that catches it.
    base = LADDER.d1_features(features)

    augmented = features.copy()

    for column in RESIDUAL_COLUMNS:
        augmented[column] = residuals[column].to_numpy(dtype=float)

    for column in AVAILABILITY_COLUMNS:
        augmented[column] = residuals[column].to_numpy(dtype=bool)

    contaminated = [c for c in base if c in NEW_COLUMNS]

    audit.record(
        "E10e", "the D1 backbone the base rung is built from contains none of "
                "the six new columns",
        "84 names, 0 of them new",
        "{} names, {} of them new".format(len(base), len(contaminated)),
        len(base) == 84 and not contaminated,
        "block_of() classifies an unrecognised name as phase1_backbone and "
        "d1_features() selects the backbone by exclusion, so attaching before "
        "reading the base puts the new block INSIDE the control arm. This "
        "check exists because that is exactly what happened on the first run. "
        "84 is a count of FEATURE NAMES; they expand to 88 design columns "
        "through the categoricals, which is what DS3a asserts separately")

    # ============================================================
    banner("2. THE DESIGNS")

    dynamic_state, _refits = STATE.build(matches)

    frame = matches.copy()

    dynamic = dynamic_state.set_index("match_id").loc[
        frame["match_id"], LADDER.DYNAMIC_COLUMNS].reset_index(drop=True)

    labels = np.array([CLASS_INDEX[r] for r in frame["result"]], dtype=int)
    results = frame["result"].to_numpy()
    blocks = I4.date_blocks(frame)

    feature_lists = {"D2_rescaled": base, "E1b": base + NEW_COLUMNS}

    matrices, names_by_rung, masks, robust = {}, {}, {}, {}

    for name in RUNGS:
        matrix, names, mask = LADDER.build_design(
            augmented, feature_lists[name], dynamic)
        matrices[name] = matrix
        names_by_rung[name] = names
        masks[name] = mask
        robust[name] = LADDER.robust_mask(names)

    print("  {:<14} {:>8} {:>10} {:>12}".format(
        "rung", "width", "robust", "passthrough"))
    print("  " + "-" * 48)
    for name in RUNGS:
        print("  {:<14} {:>8} {:>10} {:>12}".format(
            name, matrices[name].shape[1], int(robust[name].sum()),
            int(masks[name].sum())))
    print()

    width_gap = matrices["E1b"].shape[1] - matrices["D2_rescaled"].shape[1]

    added = [n for n in names_by_rung["E1b"]
             if n not in set(names_by_rung["D2_rescaled"])]

    audit.record(
        "E10", "E1b's design is exactly D2 rescaled plus the six declared "
               "columns",
        "92 + 6 = 98, the six named in E6.5",
        "{} + {} = {}, added {}".format(
            matrices["D2_rescaled"].shape[1], width_gap,
            matrices["E1b"].shape[1], sorted(added)),
        width_gap == 6 and sorted(added) == sorted(NEW_COLUMNS),
        "checked by NAME on the built design, not by counting. A count of six "
        "is not the same claim as the six being the declared six")

    for name in RUNGS:
        flagged = [n for n, f in zip(names_by_rung[name], robust[name]) if f]
        audit.record(
            "A4a-{}".format(name),
            "Amendment 4's robust mask still selects exactly its three "
            "columns at {}".format(name),
            3, len(flagged), len(flagged) == 3,
            "E6.6: the residuals are NOT DC-derived and must not take robust "
            "scaling. This is the check that would catch them drifting into "
            "it: {}".format(", ".join(flagged)))

    # The indicators must pass through unstandardised under section 6.
    indicator_positions = [i for i, n in enumerate(names_by_rung["E1b"])
                           if n in AVAILABILITY_COLUMNS]

    all_passthrough = all(masks["E1b"][i] for i in indicator_positions)

    audit.record(
        "E10c", "the three availability indicators pass through "
                "unstandardised, and the three residuals do not",
        "3 passthrough, 3 standardised",
        "{} passthrough of {} indicators".format(
            sum(masks["E1b"][i] for i in indicator_positions),
            len(indicator_positions)),
        all_passthrough and len(indicator_positions) == 3
        and not any(masks["E1b"][i] for i, n
                    in enumerate(names_by_rung["E1b"])
                    if n in RESIDUAL_COLUMNS),
        "section 6 classifies by column KIND, so the indicators were built as "
        "bool dtype rather than as 0/1 floats. This asserts the classifier "
        "agreed")

    # ============================================================
    banner("3. THE RUNGS")

    fold_tables, curve_tables, proba_by_rung, diagnostics = {}, {}, {}, {}

    for name in RUNGS:

        print("  fitting {}...".format(name))

        folds, curves, proba, diag = LADDER.run_rung(
            name, matrices[name], masks[name], frame, spec, labels, results,
            blocks, robust[name])

        for entry in diag:
            entry["passthrough"] = masks[name]

        fold_tables[name] = folds
        curve_tables[name] = curves
        proba_by_rung[name] = proba
        diagnostics[name] = diag

    print()

    all_folds = pd.concat([fold_tables[r] for r in RUNGS], ignore_index=True)

    for name in RUNGS:
        table = fold_tables[name]
        print("  {}".format(name))
        print("  {:<5} {:<11} {:>6} {:>9} {:>6} {:>8} {:>7} {:>7}".format(
            "fold", "test", "width", "lambda", "EPV", "logloss", "brier",
            "RPS"))
        print("  " + "-" * 66)
        for _i, row in table.iterrows():
            print("  {:<5} {:<11} {:>6} {:>9g} {:>6.2f} {:>8.5f} {:>7.4f} "
                  "{:>7.5f}".format(
                      int(row["fold"]), row["test_season"],
                      int(row["design_width"]), row["selected_lambda"],
                      row["epv"], row["log_loss"], row["brier_score"],
                      row["rps"]))
        statuses = sorted(set(table["g6_status"]))
        print("    G6: {}".format("PASS at all four folds"
                                  if statuses == ["PASS"]
                                  else " | ".join(statuses)))
        print()

    failed = all_folds[all_folds["g6_status"].str.startswith("FAIL")]

    audit.record(
        "G6", "no applicable rung/fold selects a lambda on a grid boundary",
        0, len(failed), len(failed) == 0,
        "EPV {:.2f} to {:.2f}, far below the applicability threshold of {:g}, "
        "so the gate is live at every rung and fold. Boundary selections: "
        "{}".format(float(all_folds["epv"].min()),
                    float(all_folds["epv"].max()),
                    LADDER.EPV_APPLICABILITY,
                    "none" if failed.empty else "see table"))

    # ---- the base rung IS the committed one --------------------------------
    committed = pd.read_csv(A4_FOLDS, float_precision=FLOAT_PRECISION)
    committed = committed[committed["rung"] == "D2_rescaled"].sort_values("fold")

    mine = fold_tables["D2_rescaled"].sort_values("fold")

    worst = 0.0
    for metric in METRICS:
        worst = max(worst, float(np.abs(
            mine[metric].to_numpy() - committed[metric].to_numpy()).max()))

    audit.record(
        "E1b-A1", "the D2 rescaled base re-fitted here reproduces the "
                  "committed Amendment 4 artefact bit for bit",
        "< 1e-12", "{:.3e}".format(worst), worst < 1e-12,
        "re-fitted rather than read so the E1b - D2rescaled delta is a "
        "genuinely PAIRED bootstrap from one process")

    # ============================================================
    banner("4. POOLED, AND THE DELTAS")

    pooled = {}
    for name in RUNGS:
        rows, proba, scores = LADDER.pool(proba_by_rung[name], results, spec)
        pooled[name] = (rows, proba, scores)

    order = pooled["D2_rescaled"][0]
    actual = results[order]

    probabilities = {name: pooled[name][1] for name in RUNGS}

    ordering = frame.iloc[order][["season", "date", "home_team", "away_team"]]
    ordering = ordering.reset_index(drop=True)

    def read_aligned(path, columns, filter_book=None):
        table = pd.read_csv(path, float_precision=FLOAT_PRECISION)
        if filter_book is not None:
            table = table[table["book"] == filter_book]
        table = table.sort_values(["season", "date", "home_team", "away_team"])
        table = table.reset_index(drop=True)
        return table, table[columns].to_numpy(dtype=float)

    d34, _ = read_aligned(D34_PREDICTIONS, ["D0_p_H"])
    for name in ("D0", "D2_rescaled", "D4", "elo_v1", "poisson_walkforward",
                 "dc_walkforward"):
        probabilities[name if name != "D2_rescaled" else "D2_committed"] = (
            d34[["{}_p_{}".format(name, o) for o in CLASSES]]
            .to_numpy(dtype=float))

    _m, market_proba = read_aligned(
        MARKET_PROBABILITIES, ["prop_p_{}".format(o) for o in CLASSES],
        filter_book="B365C")
    probabilities["market"] = market_proba

    e1a, e1a_proba = read_aligned(
        E1A_PREDICTIONS, ["E1a_sot_p_{}".format(o) for o in CLASSES])
    probabilities["E1a_sot"] = e1a_proba

    aligned = bool((d34["result"].to_numpy() == actual).all()
                   and (e1a["result"].to_numpy() == actual).all())

    audit.record(
        "E10d", "every artefact read for comparison aligns row-for-row with "
                "this rung's pooled ordering",
        "aligned", "aligned" if aligned else "MISALIGNED", aligned,
        "all sorted by (season, date, home, away) and their result columns "
        "compared elementwise against this instrument's own ordering")

    pooled_rows = []
    for name, proba in probabilities.items():
        scores = evaluate(actual, proba)
        pooled_rows.append({"model": name, "n": scores["n"],
                            **{m: scores[m] for m in METRICS}})

    pooled_table = pd.DataFrame(pooled_rows).sort_values("log_loss")

    print("  {:<22} {:>9} {:>9} {:>8}".format(
        "model", "logloss", "RPS", "brier"))
    print("  " + "-" * 52)
    for _i, row in pooled_table.iterrows():
        print("  {:<22} {:>9.5f} {:>9.5f} {:>8.4f}".format(
            row["model"], row["log_loss"], row["rps"], row["brier_score"]))
    print()

    deltas = []

    pairs = [("E1b - D2rescaled", "D2_rescaled"),
             ("E1b - E1a", "E1a_sot"),
             ("E1b - DixonColes", "dc_walkforward"),
             ("E1b - Elo v1", "elo_v1"),
             ("E1b - Poisson", "poisson_walkforward"),
             ("E1b - D0", "D0"),
             ("E1b - market", "market")]

    for label, right in pairs:
        deltas.append(LADDER.compare(label, "E1b", right,
                                     probabilities["E1b"],
                                     probabilities[right], actual))

        for fold_spec in spec["folds"]:
            season = str(fold_spec["test_season"])
            mask = (ordering["season"] == season).to_numpy()
            row = LADDER.compare(
                label, "E1b", right, probabilities["E1b"][mask],
                probabilities[right][mask], actual[mask],
                scope="fold {} ({})".format(int(fold_spec["fold"]), season))
            row["fold"] = int(fold_spec["fold"])
            deltas.append(row)

    print("  {:<22} {:>10} {:>22} {:>10}  {}".format(
        "comparison", "d_logloss", "95% CI", "d_RPS", "verdict"))
    print("  " + "-" * 94)
    for row in deltas:
        if row["scope"] != "pooled":
            continue
        print("  {:<22} {:>+10.5f} {:>22} {:>+10.5f}  {}".format(
            row["comparison"], row["log_loss_delta"],
            "[{:+.5f}, {:+.5f}]".format(row["log_loss_ci_lo"],
                                        row["log_loss_ci_hi"]),
            row["rps_delta"], row["verdict"]))
    print()

    # ============================================================
    banner("5. THE BLOCK'S COEFFICIENTS")

    coefficients = []

    names = names_by_rung["E1b"]

    for entry in diagnostics["E1b"]:
        for index, column in enumerate(names):
            if column not in NEW_COLUMNS:
                continue
            beta = entry["weights"][index]
            coefficients.append({
                "rung": "E1b", "fold": entry["fold"], "column": column,
                "beta_home": float(beta[0]), "beta_draw": float(beta[1]),
                "beta_away": float(beta[2]),
                "beta_l2": float(np.sqrt(np.sum(beta ** 2))),
                "train_centre": float(entry["mean"][index]),
                "train_scale": float(entry["sd"][index])})

    coefficient_frame = pd.DataFrame(coefficients)

    print("  {:<32} {:>9} {:>9} {:>9} {:>9}".format(
        "column", "fold 1", "fold 2", "fold 3", "fold 4"))
    print("  " + "-" * 72)
    for column in NEW_COLUMNS:
        values = [float(coefficient_frame[
            (coefficient_frame["column"] == column)
            & (coefficient_frame["fold"] == f)]["beta_l2"].iloc[0])
            for f in (1, 2, 3, 4)]
        print("  {:<32} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f}".format(
            column, *values))
    print()

    # ============================================================
    banner("6. THE LEAKAGE SUITE")

    d1_matrix, _d1_names, d1_mask = LADDER.build_design(augmented, base)

    LADDER.ds0_pipeline_anchor(audit, d1_matrix, labels, spec, frame, d1_mask)

    elo_frame = frame[["season", "date", "home_team", "away_team"]].copy()
    elo_source = STATE.load_elo_state(matches)
    for column in ("home_elo_before", "away_elo_before", "home_elo_after",
                   "away_elo_after", "home_transition", "away_transition"):
        elo_frame[column] = elo_source[column].to_numpy()

    LADDER.ds1_temporal(audit, matches, dynamic_state, elo_frame)
    LADDER.ds2_no_test_row_fits(audit, diagnostics, matrices, frame, spec)
    LADDER.g10_scale_domain(audit, diagnostics, matrices)
    # DS3 computes D3/D4 widths by block from the feature file, so it must see
    # the PRISTINE one - the six new columns would otherwise be counted into
    # every rung's declared width.
    LADDER.ds3_widths(audit, features,
                      {"D1": d1_matrix.shape[1],
                       "D2": matrices["D2_rescaled"].shape[1]})

    baseline_lambdas = {int(r["fold"]): r["selected_lambda"]
                        for _i, r in fold_tables["E1b"].iterrows()}

    LADDER.ds4_ds5_corruption(
        audit, augmented, frame, spec, labels, results, blocks, dynamic,
        baseline_lambdas, proba_by_rung["E1b"], robust["E1b"], "E1b",
        feature_lists["E1b"])

    d0_folds, _d0_proba = LADDER.run_d0(frame, spec, results)
    LADDER.ds6_base_rate(audit, d0_folds)

    determinism_baseline = {name: proba_by_rung[name] for name in RUNGS}
    determinism_baseline.update({
        "{}_lambda".format(name): {int(r["fold"]): r["selected_lambda"]
                                   for _i, r in fold_tables[name].iterrows()}
        for name in RUNGS})

    LADDER.ds8_determinism(
        audit, RUNGS, matrices, masks, frame, spec, labels, results,
        blocks, determinism_baseline, robust)
    LADDER.ds9_contract(audit, [(n, probabilities[n]) for n in RUNGS])
    LADDER.ds10_manifest(audit)
    LADDER.ds11_anchor(audit, frame, dynamic_state, spec)
    LADDER.ds12_ordering(audit, diagnostics, frame)

    # DS7: the residual columns are NOT frozen Phase 3 artefacts - they are
    # built here. Asserting them against the frozen file would be asserting
    # something false, so what is asserted instead is that the block did not
    # DISTURB the frozen columns it sits beside.
    fresh = L3.load_features(L3.load_matches())

    disturbed = [c for c in fresh.columns
                 if c in augmented.columns
                 and not fresh[c].equals(augmented[c])]

    audit.record(
        "DS7-E1b", "adding the residual block disturbed no column of the "
                   "frozen Phase 3 feature file",
        0, len(disturbed), len(disturbed) == 0,
        "the six new columns are BUILT here, not read from the frozen file, "
        "so DS7a's byte-identity claim does not apply to them. What does "
        "apply is that they must not have changed anything they sit beside, "
        "checked against a fresh re-read from disk. Disturbed: {}".format(
            disturbed or "none"))

    # ============================================================
    banner("7. WRITING")

    artefacts = ((FOLD_OUTPUT, all_folds),
                 (POOLED_OUTPUT, pooled_table),
                 (DELTA_OUTPUT, pd.DataFrame(deltas)),
                 (COEF_OUTPUT, coefficient_frame),
                 (CURVE_OUTPUT, pd.concat(list(curve_tables.values()),
                                          ignore_index=True)),
                 (RESIDUAL_OUTPUT, residuals),
                 (AUDIT_OUTPUT, audit.frame()))

    for path, data in artefacts:
        data.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)
        print("  {}".format(path))

    audit_frame = audit.frame()
    failures = int((audit_frame["status"] == "FAIL").sum())

    print()
    print("  Checks run    : {}".format(len(audit_frame)))
    print("  Checks failed : {}".format(failures))
    print()
    print("  {}".format("PASS" if failures == 0 else "FAIL"))

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
