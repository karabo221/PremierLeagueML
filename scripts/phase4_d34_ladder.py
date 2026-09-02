"""
===============================================================================
PHASE 4 - D3 AND D4:  BLOCK C AND BLOCK X ON TOP OF D2 RESCALED
===============================================================================

THE DECLARED LADDER, FINISHED. Section 3 of the D2 pre-declaration:

    D3   D2 rescaled + frozen Phase 3 C_context (12 features)      112 columns
    D4   D3 + X_prior_composite (24) + X_availability (3)          139 columns

    X_metadata's 4 columns are NOT added. Section 3 excludes them and they
    stay excluded here.

WHAT IS RE-RUN AND WHY

    D2 RESCALED IS RE-FITTED IN THIS PROCESS. It is not read out of the
    Amendment 4 artefact. Every delta in this file is a PAIRED per-match
    bootstrap, and pairing requires both arms to be scored on the same rows
    in the same order from the same run. A1 asserts the re-fit reproduces
    outputs/phase4_a4_fold_summary.csv bit for bit, so re-running it costs
    nothing in credibility and buys the pairing.

    D1, the original D2 and D2-static are NOT re-run. D2 rescaled is the
    governing D2 and is the base the brief names.

    D2-STATIC IS NOT FIXED HERE, and its numbers are quoted nowhere in this
    instrument. See the amendment in PHASE4_AMENDMENT6_D2STATIC.txt.

THE INTERPRETATION RULE, BINDING, WRITTEN BEFORE THE FIT

    Phase 3 concluded that Block C and Block X add nothing. If D3 or D4
    contradict that, PHASE 4 WINS and Phase 3's conclusion is amended rather
    than defended. Phase 3's lambda was selected under a rule that was
    rewritten after its result was known; these rungs select per rung and per
    fold on blocked inner CV over training rows only, declared in advance.

    In the other direction the rule is equally binding and is the one that
    bites more often here: A NON-SIGNIFICANT DIFFERENCE IS NOT EQUALITY, and
    no block is declared useful on accuracy. Accuracy is reported because the
    Phase 0 metric set is reported whole, and it decides nothing.

EPV, AND WHY A LARGE LAMBDA AT D4 IS NOT A DEFECT

    Events per variable runs 0.79 to 3.12 at D3 and 0.63 to 2.52 at D4, all
    of it far below the applicability threshold of 10, so G6 is live at every
    rung and every fold. D4 fits 139 columns on as few as 88 events of the
    rarest class. A high selected lambda there is the penalty doing exactly
    the job the thin design requires, and it is read that way. What would be
    a defect is a lambda at the GRID BOUNDARY, and that is what G6 tests.

DS7 GOES LIVE FOR THE FIRST TIME

    No C or X column entered D0, D1 or D2, so DS7 has been recorded "NOT
    EXERCISED" in every run to date. D3 is the first rung that consumes one.
    Its implementation in the ladder module MEASURES block counts; that is
    not the declared test and it is not reused. The live DS7 is written here
    and it ASSERTS - see ds7_live() below.
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

import phase3_ablation_ladder as L3              # noqa: E402
import phase3_regularisation_sensitivity as I4   # noqa: E402
import phase4_dynamic_state as STATE             # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR

FOLD_OUTPUT = OUTPUTS_DIR / "phase4_d34_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase4_d34_pooled.csv"
DELTA_OUTPUT = OUTPUTS_DIR / "phase4_d34_deltas.csv"
CURVE_OUTPUT = OUTPUTS_DIR / "phase4_d34_lambda_curves.csv"
COEF_OUTPUT = OUTPUTS_DIR / "phase4_d34_block_coefficients.csv"
PRED_OUTPUT = OUTPUTS_DIR / "phase4_d34_predictions.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase4_d34_audit.csv"

# Deliberately outside outputs/. See the note above section 5.
SCRATCH_DIR = OUTPUTS_DIR.parent / ".d34_inprogress"

A4_FOLDS = OUTPUTS_DIR / "phase4_a4_fold_summary.csv"
FEATURES_CSV = L3.FEATURES_CSV
INVENTORY_CSV = OUTPUTS_DIR / "phase3_feature_inventory.csv"

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

RUNGS = ("D2_rescaled", "D3", "D4")

RUNG_LABEL = {
    "D2_rescaled": "dynamic state, robust scaling - the governing D2",
    "D3": "D2 rescaled + Block C context",
    "D4": "D3 + Block X prior-season FBref",
}

BLOCKS_ADDED = {
    "D2_rescaled": (),
    "D3": ("C_context",),
    "D4": ("C_context", "X_prior_composite", "X_availability"),
}


# ============================================================
# DS7, LIVE
# ============================================================

def ds7_live(audit, features, matrices, names_by_rung):
    """
    DS7, AS DECLARED: Block C and Block X columns are read from the frozen
    Phase 3 artefacts and are BYTE-IDENTICAL to them.

    The ladder module's ds7_frozen_blocks() counts columns and records "NOT
    EXERCISED". At D0-D2 that was honest, because no C or X column entered
    any design. At D3 it would be a measurement standing in for an assertion,
    so it is not called. This is.

    THREE THINGS ARE ASSERTED, and they fail independently:

      DS7a  the values sitting in the D3 and D4 DESIGN MATRICES are exactly
            the frozen file's values. The file is re-read from disk here,
            fresh, and re-aligned onto the match order by the join keys -
            NOT taken from the already-loaded frame. A check that reads the
            same in-memory object the design was built from cannot detect a
            file that changed; it can only detect a bug in numpy.

            Equality is exact, NaN-for-NaN, not a tolerance. These columns
            are copied, never computed, so anything but exact equality is a
            defect however small it is.

      DS7b  the frozen file agrees with phase3_feature_inventory.csv, which
            the feature builder wrote in the same run and which records
            per-column non-null, distinct, min, median, mean, max and std.
            That is an INDEPENDENT description of the same bytes, so this
            catches a feature file that was regenerated with the inventory
            left behind - the exact failure a self-comparison cannot see.

      DS7c  both files are under the frozen manifest. Until this session they
            were not, and DS7's own note in the ladder module claimed "DS10
            verifies the file's hash" when DS10 could not: the file was not
            listed. The note was false, and it was only findable by trying to
            make DS7 do its job.
    """

    # ---- DS7a: the design matrix against a FRESH read of the file ---------
    matches = L3.load_matches()

    frozen = pd.read_csv(FEATURES_CSV, float_precision=FLOAT_PRECISION)
    frozen["date"] = pd.to_datetime(frozen["date"], format="%Y-%m-%d")

    keys = ["season", "date", "home_team", "away_team"]
    aligned = matches[keys].merge(frozen, on=keys, how="left",
                                  validate="one_to_one")

    block_columns = [c for c in features.columns
                     if block_of(c) in ("C_context", "X_prior_composite",
                                        "X_availability")]

    mismatched, compared = [], 0

    for rung in ("D3", "D4"):

        names = names_by_rung[rung]
        matrix = matrices[rung]

        for column in block_columns:

            # Categorical block columns expand to indicator levels and are
            # checked through those levels rather than by name.
            if column in names:
                expected = aligned[column]

                if pd.api.types.is_bool_dtype(expected):
                    expected = expected.to_numpy("float64")
                else:
                    expected = pd.to_numeric(
                        expected, errors="coerce").to_numpy("float64")

                observed = matrix[:, names.index(column)]
                compared += 1

                if not np.array_equal(expected, observed, equal_nan=True):
                    mismatched.append("{}:{}".format(rung, column))

            else:
                levels = [n for n in names if n.startswith(column + "=")]

                for level_name in levels:
                    level = level_name.split("=", 1)[1]
                    expected = (aligned[column].astype("object")
                                == level).to_numpy("float64")
                    observed = matrix[:, names.index(level_name)]
                    compared += 1

                    if not np.array_equal(expected, observed, equal_nan=True):
                        mismatched.append("{}:{}".format(rung, level_name))

    audit.record(
        "DS7a", "every Block C and Block X column in the D3 and D4 designs is "
                "byte-identical to a FRESH read of the frozen feature file",
        0, len(mismatched), not mismatched,
        "{} design columns compared across the two rungs, exact equality "
        "NaN-for-NaN, against outputs/phase3_features.csv re-read from disk "
        "and re-joined - not against the in-memory frame the design was built "
        "from. Offenders: {}".format(
            compared, ", ".join(mismatched) if mismatched else "none"))

    # ---- DS7b: the file against the inventory written beside it -----------
    inventory = pd.read_csv(INVENTORY_CSV, float_precision=FLOAT_PRECISION)
    inventory = inventory.set_index("column")

    drifted, checked = [], 0

    for column in block_columns:

        if column not in inventory.index:
            drifted.append("{} (absent from inventory)".format(column))
            continue

        recorded = inventory.loc[column]
        series = frozen[column]

        observed = {
            "non_null": int(series.notna().sum()),
            "null": int(series.isna().sum()),
            "distinct": int(series.nunique(dropna=True)),
        }

        for field, value in observed.items():
            checked += 1
            if int(recorded[field]) != value:
                drifted.append("{}.{} {} vs {}".format(
                    column, field, recorded[field], value))

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            for field, value in (("min", numeric.min()),
                                 ("median", numeric.median()),
                                 ("mean", numeric.mean()),
                                 ("max", numeric.max()),
                                 ("std", numeric.std(ddof=0))):
                checked += 1
                if not np.isclose(float(recorded[field]), float(value),
                                  rtol=0, atol=1e-12, equal_nan=True):
                    drifted.append("{}.{} {} vs {}".format(
                        column, field, recorded[field], value))

    audit.record(
        "DS7b", "the frozen feature file agrees with the inventory the "
                "feature builder wrote beside it, on every C and X column",
        0, len(drifted), not drifted,
        "{} recorded statistics checked over {} columns. The inventory is an "
        "INDEPENDENT description of the same bytes, so this is the check that "
        "would catch a regenerated feature file - which comparing the file "
        "against itself never could. Drifted: {}".format(
            checked, len(block_columns),
            ", ".join(drifted) if drifted else "none"))

    # ---- DS7c: both files are actually frozen -----------------------------
    manifest = (LADDER.PROJECT_ROOT / "FROZEN_MANIFEST.txt").read_text(
        encoding="utf-8")

    listed = [name for name in ("outputs/phase3_features.csv",
                                "outputs/phase3_feature_inventory.csv")
              if name in manifest]

    audit.record(
        "DS7c", "the feature file and its inventory are under the frozen "
                "manifest",
        2, len(listed), len(listed) == 2,
        "DS7's declared wording is 'byte-identical to the FROZEN Phase 3 "
        "artefacts'. Neither file was in the manifest before this session, so "
        "the word 'frozen' was unbacked and the ladder module's own note that "
        "'DS10 verifies the file's hash' was false. Listed now, so DS10 "
        "carries them and DS7a compares against something that cannot move "
        "silently. Present: {}".format(", ".join(listed) if listed else "none"))



# ============================================================
# THE D4 FOLD 1 IDENTITY
# ============================================================

D4_F1_NOTE = ("D4 f1 == D3 f1 by construction: all 27 Block X columns are "
              "constant over 2021-22 training rows - see X1a-X1d")


def d4_fold1_identity(audit, all_folds, deltas, matrices, names_by_rung,
                      diagnostics, coefficient_frame):
    """D4 fold 1 reproduces D3 fold 1 exactly, and that is STRUCTURAL.

    Fold 1 trains on 2021-22, the FIRST season in the data, so no team has a
    prior season to describe and every Block X column is constant over the
    training rows. A constant column takes scale 1.0 and a coefficient of
    exactly zero, so D4's design differs from D3's by 27 columns that cannot
    move a single prediction.

    The consequence is a fold summary row that looks DUPLICATED and a
    per-fold delta of 5e-18 that looks like a bug. Both are correct. This is
    written as four assertions rather than as a footnote because a structural
    identity recorded only in prose is one that a future reader has to take
    on trust - and the coefficient file, which is the only place the evidence
    currently lives, is not the file anybody opens first.

    Returns the flag text, which is stamped onto the two tables where the
    number is actually read.
    """

    d3_names = list(names_by_rung["D3"])
    d4_names = list(names_by_rung["D4"])

    d3_set = set(d3_names)
    added = [n for n in d4_names if n not in d3_set]

    # ---- X1a: the arithmetic on the fold summary itself -------------------
    def constants(rung):
        row = all_folds[(all_folds["rung"] == rung) & (all_folds["fold"] == 1)]
        return int(row["constant_columns"].iloc[0])

    gap = constants("D4") - constants("D3")

    audit.record(
        "X1a", "at fold 1, D4 carries exactly 27 more constant columns than "
               "D3 - one for every column D4 adds",
        27, gap, gap == 27,
        "D3 {} constant of 112, D4 {} of 139. The whole of the width "
        "difference is inert, which is what makes the two rungs identical "
        "there. Read on the fold summary, the table the number is read "
        "from".format(constants("D3"), constants("D4")))

    # ---- X1b: and those 27 are Block X, not 27 arbitrary columns ----------
    foreign = [n for n in added
               if block_of(n.split("=", 1)[0]) not in ("X_prior_composite",
                                                       "X_availability")]

    audit.record(
        "X1b", "the 27 columns D4 adds are all Block X",
        "0 foreign of 27", "{} foreign of {}".format(len(foreign), len(added)),
        len(foreign) == 0 and len(added) == 27,
        "counting to 27 twice is not the same claim as the 27 being the same "
        "27. Checked by BLOCK MEMBERSHIP on the built design: {}".format(
            "none" if not foreign else ", ".join(foreign)))

    # ---- X1c: constant ON THE TRAINING ROWS, measured, not inferred -------
    matrix = np.asarray(matrices["D4"], dtype=float)
    train_rows = [d["train_rows"] for d in diagnostics["D4"]
                  if int(d["fold"]) == 1][0]

    index_of = {n: i for i, n in enumerate(d4_names)}

    varying, all_missing = [], 0

    for column in added:
        values = matrix[np.ix_(train_rows, [index_of[column]])].ravel()
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            all_missing += 1
        elif np.unique(finite).size > 1:
            varying.append(column)

    audit.record(
        "X1c", "every one of those 27 columns is constant over fold 1's "
               "training rows",
        0, len(varying), len(varying) == 0,
        "2021-22 is the first season in the data, so {} of the 27 are wholly "
        "MISSING there and the remaining {} hold one value. Measured on the "
        "built design over the {} training rows, not inferred from the "
        "season list. Varying: {}".format(
            all_missing, 27 - all_missing, len(train_rows),
            "none" if not varying else ", ".join(varying)))

    # ---- X1d: which is why the prediction delta is floating-point zero ----
    subset = coefficient_frame[(coefficient_frame["rung"] == "D4")
                               & (coefficient_frame["fold"] == 1)
                               & (coefficient_frame["block"].isin(
                                   ("X_prior_composite", "X_availability")))]

    worst_beta = float(np.abs(subset["beta_l2"].to_numpy()).max())

    entry = [r for r in deltas
             if r["comparison"] == "D4 - D3" and r.get("fold") == 1]

    observed = abs(float(entry[0]["log_loss_delta"])) if entry else float("nan")

    audit.record(
        "X1d", "with 27 inert columns the D4 - D3 fold 1 log loss delta is "
               "floating-point zero, and every added coefficient is exactly 0",
        "< 1e-15 and beta 0", "{:.3e}, max |beta| {:.3e}".format(
            observed, worst_beta),
        observed < 1e-15 and worst_beta == 0.0,
        "this is the row that reads as a bug. It is the ridge doing the only "
        "thing it can with a constant column. NOT a tolerance on a "
        "difference that should be zero - the coefficients are exactly zero "
        "and the delta is what floating point leaves behind")

    print("  D4 fold 1 == D3 fold 1, structurally.")
    print("    constant columns  D3 {} of 112  ->  D4 {} of 139   (+{})".format(
        constants("D3"), constants("D4"), gap))
    print("    of the 27 added Block X columns, {} are wholly missing over "
          "2021-22".format(all_missing))
    print("    max |beta| over those 27 at fold 1: {:.3e}".format(worst_beta))
    print("    d_logloss(D4 - D3) at fold 1:       {:.3e}".format(observed))
    print()
    print("  flagged on the fold summary and the delta table as "
          "structural_note.")
    print()

    return D4_F1_NOTE

# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 4 - D3 AND D4:  BLOCK C AND BLOCK X")

    print("  base           : D2 rescaled (Amendment 4 robust scaling)")
    print("  D3             : + C_context")
    print("  D4             : + X_prior_composite and X_availability")
    print("  X_metadata     : NOT added (section 3)")
    print("  held fixed     : folds, grid, inner CV, solver, seed, section 6")
    print("  bootstrap      : {} draws, seed {}".format(
        LADDER.BOOTSTRAP_DRAWS, LADDER.BOOTSTRAP_SEED))
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches()
    features = L3.load_features(matches)

    matches = matches.copy()
    matches["match_id"] = matches.index

    # DS11 keys off role_is_test. Omitting it cost a complete 35-minute fit
    # cycle: every rung, every delta and every coefficient computed, then a
    # KeyError in the leakage suite before a single artefact was written. The
    # pre-flight below exists so that class of mistake costs seconds.
    matches["role_is_test"] = matches["season"].isin(
        [str(f["test_season"]) for f in spec["folds"]])

    required = ("season", "date", "home_team", "away_team", "result",
                "match_id", "role_is_test")
    absent = [c for c in required if c not in matches.columns]

    if absent:
        raise SystemExit(
            "FATAL: the frame is missing {} - the leakage suite reads these "
            "AFTER every rung is fitted, so this is checked before any fit "
            "rather than discovered forty minutes in".format(absent))

    print("  generating state...")

    dynamic_state, refits = STATE.build(matches)

    print("  {} per-date refits, {:.1f}s".format(refits, time.time() - started))
    print()

    frame = matches.copy()

    dynamic = dynamic_state.set_index("match_id").loc[
        frame["match_id"], LADDER.DYNAMIC_COLUMNS].reset_index(drop=True)

    labels = np.array([CLASS_INDEX[r] for r in frame["result"]], dtype=int)
    results = frame["result"].to_numpy()
    blocks = I4.date_blocks(frame)

    # ---- the feature lists -----------------------------------------------
    base = LADDER.d1_features(features)

    c_cols = [c for c in features.columns if block_of(c) == "C_context"]
    x_cols = [c for c in features.columns
              if block_of(c) == "X_prior_composite"]
    xa_cols = [c for c in features.columns if block_of(c) == "X_availability"]

    feature_lists = {
        "D2_rescaled": base,
        "D3": base + c_cols,
        "D4": base + c_cols + x_cols + xa_cols,
    }

    matrices, names_by_rung, masks, robust = {}, {}, {}, {}

    for name in RUNGS:
        matrix, names, mask = LADDER.build_design(
            features, feature_lists[name], dynamic)
        matrices[name] = matrix
        names_by_rung[name] = names
        masks[name] = mask
        robust[name] = LADDER.robust_mask(names)

    d1_matrix, _d1_names, d1_mask = LADDER.build_design(features, base)

    print("  {:<14} {:>8} {:>10} {:>12} {:>12}".format(
        "rung", "width", "robust", "passthrough", "blocks added"))
    print("  " + "-" * 62)
    for name in RUNGS:
        print("  {:<14} {:>8} {:>10} {:>12} {:>12}".format(
            name, matrices[name].shape[1], int(robust[name].sum()),
            int(masks[name].sum()), len(BLOCKS_ADDED[name])))
    print()

    # Amendment 4's mask must still find its three columns after the design
    # grew by 20 and 47 columns. It locates them by NAME, so a width change
    # cannot move it - but that is the claim, and this is the assertion.
    for name in RUNGS:
        flagged = [n for n, f in zip(names_by_rung[name], robust[name]) if f]
        audit.record(
            "A4a-{}".format(name),
            "Amendment 4's robust mask still selects exactly its three "
            "columns at {}".format(name),
            3, len(flagged), len(flagged) == 3,
            "the mask is by NAME, so it carries into a wider design "
            "unchanged: {}".format(", ".join(flagged)))

        audit.record(
            "A4b-{}".format(name),
            "no {} column both passes through unstandardised and takes "
            "robust scaling".format(name),
            0, int((robust[name] & masks[name]).sum()),
            not (robust[name] & masks[name]).any(),
            "A4.4 - the robust rule applies to continuous columns only, so it "
            "and section 6 can never contend for the same column")

    # ============================================================
    banner("1. THE RUNGS")

    fold_tables, curve_tables, proba_by_rung, diagnostics = {}, {}, {}, {}

    for name in RUNGS:

        print("  fitting {}...".format(name))

        folds, curves, proba, diag = LADDER.run_rung(
            name, matrices[name], masks[name], frame, spec, labels, results,
            blocks, robust[name])

        # DS2b reimplements the scaler and needs the rung's pass-through mask
        # to zero the columns section 6 leaves alone. run_rung does not carry
        # it, so it is attached here rather than inferred there.
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

        print("  {}  -  {}".format(name, RUNG_LABEL[name]))
        print("  {:<5} {:<11} {:>6} {:>9} {:>6} {:>6} {:>7} {:>7} {:>8} "
              "{:>7} {:>7}".format(
                  "fold", "test", "width", "lambda", "EPV", "acc", "bal_acc",
                  "mac_f1", "logloss", "brier", "RPS"))
        print("  " + "-" * 93)

        for _i, row in table.iterrows():
            print("  {:<5} {:<11} {:>6} {:>9g} {:>6.2f} {:>6.3f} {:>7.3f} "
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
        "EPV {:.2f} to {:.2f} across the three rungs, all far below the "
        "applicability threshold of {:g}, so the gate is live at every rung "
        "and every fold. Boundary selections: {}".format(
            float(all_folds["epv"].min()), float(all_folds["epv"].max()),
            LADDER.EPV_APPLICABILITY,
            "none" if not len(failed) else " | ".join(
                "{} f{}".format(r["rung"], int(r["fold"]))
                for _i, r in failed.iterrows())))

    # ---- A1: the re-fitted D2 rescaled IS the committed one ---------------
    committed = pd.read_csv(A4_FOLDS, float_precision=FLOAT_PRECISION)
    committed = committed[committed["rung"] == "D2_rescaled"].sort_values("fold")

    mine = fold_tables["D2_rescaled"].sort_values("fold")

    worst = 0.0
    for metric in METRICS:
        worst = max(worst, float(np.abs(
            mine[metric].to_numpy() - committed[metric].to_numpy()).max()))

    moved = int((mine["selected_lambda"].to_numpy()
                 != committed["selected_lambda"].to_numpy()).sum())

    audit.record(
        "A1", "the D2 rescaled re-fitted here reproduces the COMMITTED "
              "Amendment 4 artefact bit for bit",
        "< 1e-12 and 0 lambda moves", "{:.3e}, {} moves".format(worst, moved),
        worst < 1e-12 and moved == 0,
        "the base rung is re-fitted rather than read so that every delta "
        "below is a genuinely PAIRED bootstrap - both arms scored on the same "
        "rows in the same order from one process. Verified against "
        "outputs/phase4_a4_fold_summary.csv on disk, not an in-memory copy")

    if failed.empty:
        pass
    else:
        print("  G6 FAILED. The brief stops the ladder at the failing rung.")
        print()

    # ============================================================
    banner("2. POOLED, AND THE REFERENCES")

    pooled = {}

    for name in RUNGS:
        rows, proba, scores = LADDER.pool(proba_by_rung[name], results, spec)
        pooled[name] = (rows, proba, scores)

    order = pooled["D2_rescaled"][0]
    actual = results[order]

    d0_folds, d0_proba = LADDER.run_d0(frame, spec, results)
    _rows0, d0_pooled_proba, d0_scores = LADDER.pool(d0_proba, results, spec)

    references = LADDER.reference_probabilities(frame)
    reference_proba = {k: LADDER.reference_array(references[k], order)
                       for k in references}
    reference_scores = {k: evaluate(actual, v)
                        for k, v in reference_proba.items()}

    display = ([("D0  base rate", d0_scores)]
               + [(n, pooled[n][2]) for n in RUNGS]
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
                  label, scores["accuracy"], scores["balanced_accuracy"],
                  scores["macro_f1"], scores["log_loss"],
                  scores["brier_score"], scores["rps"]))

    print()

    # The identity that has caught two bugs: at 380 test rows per fold the
    # pooled log loss must equal the unweighted mean of the four fold values.
    for name in RUNGS:
        fold_mean = float(fold_tables[name]["log_loss"].mean())
        gap = abs(fold_mean - float(pooled[name][2]["log_loss"]))
        audit.record(
            "P1-{}".format(name),
            "{}'s pooled log loss equals the unweighted mean of its four "
            "fold values".format(name),
            "< 1e-12", "{:.3e}".format(gap), gap < 1e-12,
            "true only because every fold tests exactly 380 rows. It is "
            "cheap and it has caught two bugs, so it is asserted rather "
            "than assumed")

    # ============================================================
    banner("3. THE DELTAS")

    proba_of = {n: pooled[n][1] for n in RUNGS}
    proba_of["D0"] = d0_pooled_proba
    proba_of.update(reference_proba)

    deltas = []

    pairs = [
        ("D3 - D2rescaled", "D3", "D2_rescaled"),
        ("D4 - D3", "D4", "D3"),
        ("D4 - D2rescaled", "D4", "D2_rescaled"),
        ("D3 - D0", "D3", "D0"),
        ("D3 - Elo v1", "D3", "elo_v1"),
        ("D3 - Poisson", "D3", "poisson_walkforward"),
        ("D3 - DixonColes", "D3", "dc_walkforward"),
        ("D4 - D0", "D4", "D0"),
        ("D4 - Elo v1", "D4", "elo_v1"),
        ("D4 - Poisson", "D4", "poisson_walkforward"),
        ("D4 - DixonColes", "D4", "dc_walkforward"),
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

    # ---- per fold, for the three ladder steps ----------------------------
    per_fold = []

    for label, left, right in (("D3 - D2rescaled", "D3", "D2_rescaled"),
                               ("D4 - D3", "D4", "D3"),
                               ("D4 - D2rescaled", "D4", "D2_rescaled")):

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

    for label in ("D3 - D2rescaled", "D4 - D3", "D4 - D2rescaled"):
        show([r for r in per_fold if r["comparison"] == label],
             "{}, PER FOLD".format(label))

    deltas.extend(per_fold)

    # ---- the same four references, per fold, for D3 and D4 ---------------
    reference_per_fold = []

    for rung in ("D3", "D4"):
        for key, label in (("D0", "D0"), ("elo_v1", "Elo v1"),
                           ("poisson_walkforward", "Poisson"),
                           ("dc_walkforward", "DixonColes")):
            for fold_spec in spec["folds"]:

                fold = int(fold_spec["fold"])
                rows_left, proba_left = proba_by_rung[rung][fold]

                if key == "D0":
                    _r, right_proba = d0_proba[fold]
                else:
                    right_proba = LADDER.reference_array(references[key],
                                                         rows_left)

                row = LADDER.compare(
                    "{} - {}".format(rung, label), rung, key, proba_left,
                    right_proba, results[rows_left],
                    scope="fold {} ({})".format(fold,
                                                fold_spec["test_season"]))
                row["fold"] = fold
                reference_per_fold.append(row)

    deltas.extend(reference_per_fold)

    print("  The 32 per-fold reference comparisons (D3 and D4 against D0, Elo")
    print("  v1, Poisson and Dixon-Coles at each of the four folds) are")
    print("  written to the delta artefact rather than printed. The pooled")
    print("  form of each is in the table above.")
    print()

    # ============================================================
    banner("4. THE BLOCKS THEMSELVES")

    coefficients = []

    for name in ("D3", "D4"):

        names = names_by_rung[name]

        for entry in diagnostics[name]:

            for index, column in enumerate(names):

                block = block_of(column.split("=", 1)[0])

                if block not in ("C_context", "X_prior_composite",
                                 "X_availability"):
                    continue

                beta = entry["weights"][index]

                coefficients.append({
                    "rung": name, "fold": entry["fold"], "column": column,
                    "block": block,
                    "beta_home": float(beta[0]), "beta_draw": float(beta[1]),
                    "beta_away": float(beta[2]),
                    "beta_l2": float(np.sqrt(np.sum(beta ** 2))),
                    "train_centre": float(entry["mean"][index]),
                    "train_scale": float(entry["sd"][index]),
                    "selected_lambda": float(
                        fold_tables[name].set_index("fold").loc[
                            entry["fold"], "selected_lambda"]),
                })

    coefficient_frame = pd.DataFrame(coefficients)

    for name in ("D3", "D4"):

        subset = coefficient_frame[coefficient_frame["rung"] == name]

        print("  {} - L2 NORM of each added column's three class "
              "coefficients".format(name))
        print()
        print("  {:<34} {:>9} {:>9} {:>9} {:>9}".format(
            "column", "fold 1", "fold 2", "fold 3", "fold 4"))
        print("  " + "-" * 74)

        for column in sorted(set(subset["column"])):
            values = [float(subset[(subset["column"] == column)
                                   & (subset["fold"] == f)]["beta_l2"].iloc[0])
                      for f in (1, 2, 3, 4)]
            print("  {:<34} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f}".format(
                column, *values))

        print()

        for block in ("C_context", "X_prior_composite", "X_availability"):

            block_subset = subset[subset["block"] == block]

            if block_subset.empty:
                continue

            print("    {} block L2 norm by fold: {}".format(
                block, "  ".join(
                    "f{} {:.5f}".format(f, float(np.sqrt(np.sum(
                        block_subset[block_subset["fold"] == f]["beta_l2"]
                        .to_numpy() ** 2))))
                    for f in (1, 2, 3, 4))))

        print()

    # ============================================================
    banner("4b. THE D4 FOLD 1 IDENTITY")

    note = d4_fold1_identity(audit, all_folds, deltas, matrices,
                             names_by_rung, diagnostics, coefficient_frame)

    # THE FLAG GOES WHERE THE NUMBER IS READ. X1a-X1d prove the identity, but
    # a reader who opens the fold summary and sees two rungs with identical
    # metrics will not go looking for an audit row to explain it - they will
    # assume a duplicated write. Same for the 5e-18 in the delta table.
    all_folds["structural_note"] = ""
    all_folds.loc[(all_folds["rung"] == "D4") & (all_folds["fold"] == 1),
                  "structural_note"] = note

    for row in deltas:
        row.setdefault("structural_note", "")

    for row in deltas:
        if row.get("fold") == 1 and row["comparison"] in ("D4 - D3",
                                                          "D4 - D2rescaled"):
            row["structural_note"] = note

    # ============================================================
    # A CRASH-SAFETY COPY, WRITTEN BEFORE THE LEAKAGE SUITE.
    #
    # Everything above this line is the expensive part - three rungs, twelve
    # outer fits, over a thousand inner fits and 55 bootstraps. The first
    # attempt at this instrument crashed in the suite below and threw all of
    # it away unwritten.
    #
    # IT GOES TO A SCRATCH DIRECTORY, NOT TO outputs/. DS10 re-runs
    # frozen_manifest.py --verify inside the suite, and a file matching a
    # FROZEN_PATTERNS glob that is on disk but not yet in the manifest is
    # exactly what --verify is built to report. Writing the real artefacts
    # here would fail DS10 every first run - a gate failing because of the
    # instrument's own bookkeeping, which is the worst kind of false alarm.
    # The real artefacts are written after the suite, under section 7.
    #
    # The audit is not copied here at all. It is not complete until the
    # suite has run, and a half-written audit that looks like a full one is
    # worse than no audit.
    # ============================================================
    banner("5. THE CRASH-SAFETY COPY")

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

    artefacts = ((FOLD_OUTPUT, all_folds),
                 (POOLED_OUTPUT, pd.DataFrame(pooled_table)),
                 (DELTA_OUTPUT, pd.DataFrame(deltas)),
                 (CURVE_OUTPUT, pd.concat(list(curve_tables.values()),
                                          ignore_index=True)),
                 (COEF_OUTPUT, coefficient_frame),
                 (PRED_OUTPUT, predictions))

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    for path, data in artefacts:
        data.to_csv(SCRATCH_DIR / path.name, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)

    print("  {} tables -> {}".format(len(artefacts), SCRATCH_DIR))
    print("  not the reported artefacts; those are written under section 7")
    print()

    # ============================================================
    banner("6. THE LEAKAGE SUITE")

    LADDER.ds0_pipeline_anchor(audit, d1_matrix, labels, spec, frame, d1_mask)

    elo_frame = frame[["season", "date", "home_team", "away_team"]].copy()
    elo_source = STATE.load_elo_state(matches)
    for column in ("home_elo_before", "away_elo_before", "home_elo_after",
                   "away_elo_after", "home_transition", "away_transition"):
        elo_frame[column] = elo_source[column].to_numpy()

    LADDER.ds1_temporal(audit, matches, dynamic_state, elo_frame)

    LADDER.ds2_no_test_row_fits(audit, diagnostics, matrices, frame, spec)

    offenders = LADDER.g10_scale_domain(audit, diagnostics, matrices)

    # G10 is a LOGIC test over the rungs it is handed. Passing it here says
    # nothing about the standing failure, which is on D2-static and is not in
    # this instrument. Recorded so a clean G10 row cannot be read as the open
    # failure having been resolved.
    audit.measure(
        "G10-scope", "what G10 was evaluated over in THIS instrument",
        "{} - offenders: {}".format(", ".join(RUNGS),
                                    len(offenders) if offenders else 0),
        "the STANDING G10 FAILURE IS ON D2-STATIC AND IS UNCHANGED. D2-static "
        "is not run here, so it cannot appear in this row. G10 stays failing "
        "at the project level until D2-static is rebuilt on the restricted "
        "training set - see PHASE4_AMENDMENT6_D2STATIC.txt")

    LADDER.ds3_widths(audit, features,
                      {"D1": d1_matrix.shape[1],
                       "D2": matrices["D2_rescaled"].shape[1]})

    # DS3a computes D3 and D4's widths from the feature file. They are now
    # FITTED, so the fitted width is asserted against the declared one too -
    # a computed width and a fitted width are different claims.
    audit.record(
        "DS3e", "the FITTED D3 and D4 design widths are the declared 112 "
                "and 139",
        "112, 139", "{}, {}".format(matrices["D3"].shape[1],
                                    matrices["D4"].shape[1]),
        matrices["D3"].shape[1] == 112 and matrices["D4"].shape[1] == 139,
        "DS3a computes these from the frozen feature file without fitting. "
        "This asserts the matrices that were actually fitted, which is a "
        "different claim and the one the reported numbers rest on")

    audit.record(
        "DS3f", "D3 adds only C_context columns to D2, and D4 adds only "
                "X_prior_composite and X_availability to D3",
        "0 foreign columns", "{} foreign".format(
            len([c for r in ("D3", "D4") for c in names_by_rung[r]
                 if c not in names_by_rung["D2_rescaled"]
                 and block_of(c.split("=", 1)[0]) not in
                 ("C_context", "X_prior_composite", "X_availability")])),
        not [c for r in ("D3", "D4") for c in names_by_rung[r]
             if c not in names_by_rung["D2_rescaled"]
             and block_of(c.split("=", 1)[0]) not in
             ("C_context", "X_prior_composite", "X_availability")],
        "nesting checked by BLOCK MEMBERSHIP of every added design column, "
        "not by counting. X_metadata's four columns must not appear and this "
        "is what would catch them")

    metadata_present = [c for r in RUNGS for c in names_by_rung[r]
                        if block_of(c.split("=", 1)[0]) == "X_metadata"]

    audit.record(
        "DS3g", "X_metadata enters no rung",
        0, len(metadata_present), not metadata_present,
        "section 3 excludes X_metadata's four columns from D4. Two of them "
        "are categorical and would expand to several indicator levels, so "
        "their absence is asserted on the built designs rather than on the "
        "feature list: {}".format(
            ", ".join(metadata_present) if metadata_present else "none"))

    baseline_lambdas = {int(r["fold"]): r["selected_lambda"]
                        for _i, r in fold_tables["D4"].iterrows()}

    LADDER.ds4_ds5_corruption(
        audit, features, frame, spec, labels, results, blocks, dynamic,
        baseline_lambdas, proba_by_rung["D4"], robust["D4"], "D4",
        feature_lists["D4"])

    LADDER.ds6_base_rate(audit, d0_folds)

    ds7_live(audit, features, matrices, names_by_rung)

    determinism_baseline = {name: proba_by_rung[name] for name in ("D3", "D4")}
    determinism_baseline.update({
        "D3_lambda": {int(r["fold"]): r["selected_lambda"]
                      for _i, r in fold_tables["D3"].iterrows()},
        "D4_lambda": baseline_lambdas,
    })

    LADDER.ds8_determinism(
        audit, ("D3", "D4"), matrices, masks, frame, spec, labels, results,
        blocks, determinism_baseline, robust)

    contract_sets = [("{} pooled".format(n), pooled[n][1]) for n in RUNGS]
    for name in RUNGS:
        for fold, (_rows, proba) in proba_by_rung[name].items():
            contract_sets.append(("{} fold {}".format(name, fold), proba))

    LADDER.ds9_contract(audit, contract_sets)
    LADDER.ds10_manifest(audit)

    dynamic_indexed = dynamic_state.set_index("match_id").loc[
        frame["match_id"]].reset_index()
    LADDER.ds11_anchor(audit, frame, dynamic_indexed, spec)

    LADDER.ds12_ordering(audit, diagnostics, frame)

    audit.print_rows()

    # ============================================================
    banner("7. WRITING")

    for path, data in artefacts:
        data.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)
        print("  {}".format(path))

    audit.frame().to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8",
                         float_format=FLOAT_FORMAT)
    print("  {}".format(AUDIT_OUTPUT))

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
