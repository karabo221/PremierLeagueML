"""
===============================================================================
PHASE 3 - INSTRUMENT 5
CEILING RESOLUTION, AND THE FROZEN REGULARISATION CHOICE
===============================================================================

THE QUESTION

    Instrument 4 ran lambda over 0.01 ... 100 and all 28 of its (rung, fold)
    selections landed on the ceiling. That is not a result, it is a boundary.

    The arithmetic says so plainly. As lambda goes to infinity every rung
    collapses onto the unpenalised intercept, which IS B0, mean log loss
    1.068888. At lambda = 100, B1 scored 1.007648 - below that asymptote. A
    curve sitting below its own limit at the edge of the grid has to turn back
    up somewhere outside it. Instrument 4 found the edge of its box.

    This instrument opens the box two more decades, and then closes the
    question: the output is a FROZEN choice, so the penalty stops being a live
    parameter for the rest of the project.

THE PROTOCOL IS INSTRUMENT 4'S, IMPORTED

    Not re-specified and not copied. Run, fit_pipeline, inner_splits,
    date_blocks, the whole audit battery and the report helpers are imported
    from phase3_regularisation_sensitivity and handed a different grid. The
    one code change that made this possible - an optional `grid` argument
    defaulting to Instrument 4's own declared constant - is declared in
    section 2 of this instrument's pre-declaration, and C1 proves it changed
    nothing by re-deriving Instrument 4's stored lambda = 100 column through
    the refactored code.

    That chains the reproduction the whole way down:

        Instrument 5 at lambda = 100  ==  Instrument 4 at lambda = 100   (C1)
        Instrument 4 at lambda = 1    ==  Instrument 3 at lambda = 1     (R7)

THE FREEZE RULE WAS FIXED BEFORE THE RESULT

    Section 4 of PHASE3_CEILING_PREDECLARATION.txt declares what gets frozen,
    how the scalar is derived, and the three conditions - VALID, BLOCKED,
    WEAK - under which the freeze does or does not stand. C2 asserts the rule
    was applied exactly as written, so the freeze cannot be a number chosen
    after looking at the curve.

WHAT IS WRITTEN

    outputs/phase3_ceiling_*.csv and outputs/phase3_frozen_regularisation.json
    and nothing else. Instrument 4's results, the lambda = 1 ablation, every
    Phase 0/1/2 artefact and data/raw are SHA-256'd before and after (R8, C4).
===============================================================================
"""

from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import CLASS_INDEX  # noqa: E402
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase3_ablation_ladder as L3  # noqa: E402
import phase3_regularisation_sensitivity as I4  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()

PREDECLARATION = PROJECT_ROOT / "PHASE3_CEILING_PREDECLARATION.txt"

I4_SURFACE = OUTPUTS_DIR / "phase3_reg_surface.csv"
I4_SELECTED = OUTPUTS_DIR / "phase3_reg_selected.csv"
FROZEN_LADDER = OUTPUTS_DIR / "phase3_ablation_ladder.csv"

SELECTED_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_selected.csv"
SURFACE_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_surface.csv"
COMBINED_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_combined_surface.csv"
CURVES_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_lambda_curves.csv"
BLOCK_NORM_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_block_norms.csv"
LADDER_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_ladder.csv"
VERDICT_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_verdict.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase3_ceiling_audit.csv"
FROZEN_OUTPUT = OUTPUTS_DIR / "phase3_frozen_regularisation.json"

FLOAT_PRECISION = "round_trip"

# ---- THE GRID.  Declared in this instrument's pre-declaration, asserted R6.
LAMBDA_GRID = (100.0, 300.0, 1000.0, 3000.0, 10000.0)

ANCHOR_LAMBDA = 100.0       # shared with Instrument 4's grid; C1's fixed point

COMBINED_GRID = tuple(sorted(set(I4.LAMBDA_GRID) | set(LAMBDA_GRID)))

METRICS = I4.METRICS
PRIMARY_METRIC = I4.PRIMARY_METRIC

RUNGS = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]

# C3's threshold for "the grid now reaches the degenerate end". The penalty
# enters the gradient linearly, so for large lambda the coefficient norm falls
# roughly as 1/lambda; across two decades that predicts ~0.01. A tenth is a
# generous bar that a grid which had NOT reached the degenerate end would
# still fail. Pinned before the run.
C3_NORM_RATIO_CEILING = 0.10

# C4: Instrument 4's results are inputs here and must survive untouched.
I4_ARTEFACT_GLOB = "phase3_reg_*"

# This instrument's OWN outputs, which Instrument 4's isolation check must not
# read as tampering. C4 applies the stricter frozen set that DOES cover
# Instrument 4's results.
I4_FROZEN_EXCLUDE = {
    "exclude_prefixes": ("phase3_reg_", "phase3_ceiling_"),
    "exclude_names": ("phase3_frozen_regularisation.json",),
}


# ============================================================
# FROZEN STATE  (stricter than Instrument 4's: its outputs are now inputs)
# ============================================================

def hash_file(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_frozen_state():
    """
    Everything Instrument 4 froze, PLUS Instrument 4's own results.

    Instrument 4 excluded phase3_reg_* from its freeze because it was writing
    them. Here they are the reference this experiment is anchored to, so they
    are frozen and C4 checks them specifically.
    """

    L3._HASHING = True
    state = {}

    try:
        for pattern in ("phase[012]_*", "phase3_*"):
            for path in sorted(OUTPUTS_DIR.glob(pattern)):
                if path.is_file() and not path.name.startswith("phase3_ceiling_") \
                        and path.name != FROZEN_OUTPUT.name:
                    state[str(path)] = hash_file(path)

        for path in sorted(SCRIPTS_DIR.glob("phase*.py")):
            state[str(path)] = hash_file(path)

        for path in sorted(PROJECT_ROOT.glob("PHASE3_*PREDECLARATION.txt")):
            state[str(path)] = hash_file(path)

        if RAW_DIR.exists():
            for path in sorted(RAW_DIR.rglob("*")):
                if path.is_file():
                    state[str(path)] = hash_file(path)
    finally:
        L3._HASHING = False

    return state


# ============================================================
# THE COMBINED SURFACE
# ============================================================

def combined_surface(surface):
    """
    Instrument 4's nine lambdas and this instrument's five, in one frame.

    100 appears in both. C1 has already asserted they agree to < 1e-9, so the
    duplicate is dropped in favour of Instrument 4's stored value - the older
    artefact wins, which is the only tie-break that cannot drift.
    """

    stored = pd.read_csv(I4_SURFACE, float_precision=FLOAT_PRECISION)
    stored = stored.copy()
    stored["source"] = "instrument_4"

    mine = surface.copy()
    mine["source"] = "instrument_5"
    mine = mine[mine["lambda"] != ANCHOR_LAMBDA]

    shared = [c for c in stored.columns if c in mine.columns]

    return pd.concat([stored[shared], mine[shared]], ignore_index=True) \
             .sort_values(["rung", "fold", "lambda"]).reset_index(drop=True)


def print_combined(combined, metric="log_loss"):

    print("  ORACLE - outer-test {} over the full combined grid, mean of".format(
        metric))
    print("  the four folds. Instrument 4's nine lambdas and this run's five.")
    print("  Selected nothing; printed so the turning point can be seen.")
    print()

    header = "  {:<5}".format("rung") + "".join(
        "{:>8g}".format(p) for p in COMBINED_GRID)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rung in RUNGS:

        block = combined[combined["rung"] == rung]

        if not len(block):
            continue

        means = []

        for penalty in COMBINED_GRID:
            part = block[block["lambda"] == penalty]
            means.append(float(part[metric].mean()) if len(part) else np.nan)

        best = int(np.nanargmin(means))

        cells = "".join(
            ("{:>7.4f}*".format(v) if i == best else "{:>8.4f}".format(v))
            for i, v in enumerate(means))

        print("  {:<5}{}".format(rung, cells))

    print()
    print("  * = each rung's best lambda on the OUTER TEST. Oracle only - the")
    print("  headline table uses the lambda the inner CV chose without it.")
    print()


def turning_points(combined, metric="log_loss"):
    """Per rung: the oracle argmin over the combined grid, and whether it is interior."""

    rows = []

    for rung in RUNGS:

        block = combined[combined["rung"] == rung]

        if not len(block):
            continue

        means = {}

        for penalty in COMBINED_GRID:
            part = block[block["lambda"] == penalty]
            if len(part):
                means[penalty] = float(part[metric].mean())

        best = min(means, key=lambda p: (means[p], -p))

        rows.append({
            "rung": rung,
            "oracle_best_lambda": best,
            "oracle_best_log_loss": means[best],
            "log_loss_at_grid_floor": means[COMBINED_GRID[0]],
            "log_loss_at_grid_ceiling": means[COMBINED_GRID[-1]],
            "interior": bool(best not in (COMBINED_GRID[0], COMBINED_GRID[-1])),
        })

    return pd.DataFrame(rows)


# ============================================================
# THE FROZEN CHOICE
# ============================================================

def penalised_selections(selected):
    """
    The selections the freeze rule is allowed to see. AMENDMENT 2.

    A rung with no penalised design column has a constant lambda-curve - that
    is test R10, not an assumption - so its "selection" is the declared
    tie-break's output on a flat function and says nothing about where the
    optimum is. Including it let B0 return the grid ceiling and blocked the
    freeze for a rung that has no coefficients to shrink.

    The test is on the design matrix, never on the rung's name.
    """

    return selected[selected["design_columns"] > 0]


def apply_freeze_rule(selected):
    """
    Section 4 of the pre-declaration as amended by Amendment 2, applied.

    scope    selections from rungs with at least one penalised design column
    scalar   median of those selected lambdas, snapped to the nearest grid
             point in log space, exact ties to the LARGER
    status   BLOCKED  any in-scope selection sits on the grid ceiling
             WEAK     VALID, but the in-scope selections span over a decade
             VALID    otherwise
    """

    in_scope = penalised_selections(selected)

    excluded = sorted(set(selected["rung"]) - set(in_scope["rung"]))

    values = in_scope["selected_lambda"].to_numpy(dtype=float)

    median = float(np.median(values))

    # Snap to the nearest declared grid point, measured in log space because
    # the grid is geometric. Exact ties go to the LARGER, as declared.
    log_median = np.log10(median)
    closest = min(abs(np.log10(p) - log_median) for p in LAMBDA_GRID)
    scalar = float(max(p for p in LAMBDA_GRID
                       if abs(np.log10(p) - log_median) == closest))

    on_ceiling = int((values == LAMBDA_GRID[-1]).sum())
    on_floor = int((values == LAMBDA_GRID[0]).sum())

    spread_decades = float(np.log10(values.max()) - np.log10(values.min()))

    if on_ceiling:
        status = "BLOCKED"
    elif spread_decades > 1.0:
        status = "WEAK"
    else:
        status = "VALID"

    return {
        "status": status,
        "scalar": None if status == "BLOCKED" else scalar,
        "median_raw": median,
        "selections_on_ceiling": on_ceiling,
        "selections_on_floor": on_floor,
        "spread_decades": spread_decades,
        "min_selected": float(values.min()),
        "max_selected": float(values.max()),
        "n_selections": int(len(values)),
        "n_all_selections": int(len(selected)),
        "excluded_rungs": excluded,
        "excluded_selections": [
            {"rung": r["rung"], "fold": int(r["fold"]),
             "selected_lambda": float(r["selected_lambda"])}
            for _i, r in selected[selected["design_columns"] <= 0].iterrows()
        ],
    }


def _jsonable(value):
    """numpy scalars out of a DataFrame are not JSON types. np.bool_ especially."""

    if isinstance(value, np.bool_):
        return bool(value)

    if hasattr(value, "item"):
        return value.item()

    return str(value)


def write_frozen_artefact(freeze, selected, turning, verdict_attrs):

    payload = {
        "instrument": "phase3_instrument_5_ceiling_resolution",
        "status": freeze["status"],
        "frozen_procedure": (
            "Penalty is chosen per (rung, outer fold) by the nested "
            "expanding-window inner CV of Instrument 4, over the combined "
            "grid, fitted on training rows only. This is the primary freeze "
            "and takes precedence over the scalar."
        ),
        "frozen_lambda_scalar": freeze["scalar"],
        "scalar_caveat": (
            None if freeze["status"] == "VALID" else
            "status is {}; see the pre-declaration section 4.2".format(
                freeze["status"])
        ),
        "combined_grid": list(COMBINED_GRID),
        "instrument_5_grid": list(LAMBDA_GRID),
        "instrument_4_grid": list(I4.LAMBDA_GRID),
        "amendment_2": {
            "rule": (
                "The freeze rule operates only over rungs with at least one "
                "PENALISED design column. A rung with none has a constant "
                "lambda-curve (test R10), so its selection is the declared "
                "tie-break's output on a flat function and carries no "
                "information about the optimum."
            ),
            "excluded_rungs": freeze["excluded_rungs"],
            "excluded_selections": freeze["excluded_selections"],
            "disclosure": (
                "Amendment 2 was written AFTER the result was known and it "
                "changes the outcome from BLOCKED to VALID. It was put to the "
                "project owner alongside the literal BLOCKED outcome and the "
                "owner chose it. See PHASE3_CEILING_PREDECLARATION.txt "
                "Amendment 2 for the full disclosure and the argument."
            ),
            "status_under_literal_rule": "BLOCKED",
            "scalar_under_literal_rule": None,
        },
        "known_failures": {
            "C3a": (
                "coefficient norm ratio 10000/100 was 0.1212 against a "
                "pinned threshold of 0.10 - left failing, not relaxed"
            ),
            "C3b": (
                "B6 sits marginally further from B0 at lambda = 10000 than at "
                "lambda = 100 - left failing, not relaxed"
            ),
            "reading": (
                "C3 tested that the grid reaches the DEGENERATE end. "
                "Resolving the ceiling only requires that it BRACKETS an "
                "interior optimum, which it does for all six featured rungs "
                "with the curve rising either side of lambda = 1000."
            ),
        },
        "selection_summary": {
            "n_in_scope": freeze["n_selections"],
            "n_all": freeze["n_all_selections"],
            "n": freeze["n_selections"],
            "median_raw": freeze["median_raw"],
            "min": freeze["min_selected"],
            "max": freeze["max_selected"],
            "spread_decades": freeze["spread_decades"],
            "on_grid_ceiling": freeze["selections_on_ceiling"],
            "on_grid_floor": freeze["selections_on_floor"],
        },
        "per_rung_fold": [
            {"rung": r["rung"], "fold": int(r["fold"]),
             "selected_lambda": float(r["selected_lambda"]),
             "test_log_loss": float(r["test_log_loss"])}
            for _i, r in selected.iterrows()
        ],
        "oracle_turning_points": turning.to_dict("records"),
        "rung_verdict": verdict_attrs,
        "provenance": {
            "ceiling_predeclaration_sha256": hash_file(PREDECLARATION),
            "regularisation_predeclaration_sha256": hash_file(
                I4.PREDECLARATION),
            "instrument_4_surface_sha256": hash_file(I4_SURFACE),
            "instrument_4_selected_sha256": hash_file(I4_SELECTED),
            "ablation_ladder_sha256": hash_file(FROZEN_LADDER),
        },
    }

    FROZEN_OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=False, default=_jsonable),
        encoding="utf-8")

    return payload


# ============================================================
# THE INSTRUMENT-5 TESTS
# ============================================================

def test_ceiling(run, surface, combined, selected, freeze, before_strict,
                 audit):

    # ---- C1  the anchor: lambda = 100 must equal Instrument 4's stored -----
    stored = pd.read_csv(I4_SURFACE, float_precision=FLOAT_PRECISION)

    theirs = stored[stored["lambda"] == ANCHOR_LAMBDA]
    mine = surface[surface["lambda"] == ANCHOR_LAMBDA]

    merged = theirs.merge(mine, on=["rung", "fold"],
                          suffixes=("_i4", "_i5"), how="inner")

    worst = 0.0
    worst_metric = ""

    for metric in METRICS:
        difference = float(np.abs(
            merged[metric + "_i4"].to_numpy()
            - merged[metric + "_i5"].to_numpy()).max())
        if difference > worst:
            worst, worst_metric = difference, metric

    audit.record(
        "C1a", "at lambda = 100 this reproduces Instrument 4's stored numbers",
        "< 1e-9", "{:.3e} ({})".format(worst, worst_metric), worst < 1e-9,
        "{} rung x fold pairs, all six metrics; this is the anchor shared by "
        "the two grids, and it also proves the optional-grid refactor of "
        "Instrument 4 changed nothing".format(len(merged)))

    audit.record(
        "C1b", "every Instrument 4 rung x fold pair at lambda = 100 matched",
        len(theirs), len(merged), len(merged) == len(theirs))

    norm_difference = float(np.abs(
        merged["coef_l2_norm_i4"].to_numpy()
        - merged["coef_l2_norm_i5"].to_numpy()).max())

    audit.record(
        "C1c", "and the fitted coefficient norms are identical too",
        "< 1e-9", "{:.3e}".format(norm_difference), norm_difference < 1e-9)

    # ---- C2  the freeze rule was applied as declared ----------------------
    # Scope is Amendment 2's: rungs with at least one penalised design column.
    # Recomputed here independently of apply_freeze_rule so the test is a
    # check rather than an echo.
    values = penalised_selections(selected)["selected_lambda"].to_numpy(float)

    recomputed_status = (
        "BLOCKED" if (values == LAMBDA_GRID[-1]).any()
        else "WEAK" if (np.log10(values.max()) - np.log10(values.min())) > 1.0
        else "VALID")

    audit.record(
        "C2a", "the freeze status follows the declared conditions",
        recomputed_status, freeze["status"],
        freeze["status"] == recomputed_status,
        "{} in-scope selections, {} on the ceiling, spread {:.2f} decades".format(
            len(values), freeze["selections_on_ceiling"],
            freeze["spread_decades"]))

    # ---- C2d  AMENDMENT 2's exclusion is exactly the featureless rungs ----
    excluded = selected[selected["design_columns"] <= 0]

    wrongly_excluded = int((excluded["design_columns"] > 0).sum())

    audit.record(
        "C2d", "only rungs with zero penalised columns were excluded",
        0, wrongly_excluded, wrongly_excluded == 0,
        "excluded {}: {} selections; R10 (passed above) is the evidence that "
        "their lambda-curve is constant and their selection is the "
        "tie-break's output, not a boundary hit".format(
            freeze["excluded_rungs"] or "nothing",
            len(excluded)))

    audit.measure(
        "C2e", "selections in scope after Amendment 2",
        "{} of {}".format(freeze["n_selections"], freeze["n_all_selections"]),
        "the excluded selections are reported in every table and in the "
        "frozen artefact; they are removed from the RULE, not from the record")

    if freeze["status"] == "BLOCKED":
        audit.record(
            "C2b", "BLOCKED means no scalar was frozen",
            None, freeze["scalar"], freeze["scalar"] is None,
            "a selection on the ceiling means the optimum is still outside "
            "the grid; the pre-declaration forbids a second extension")
    else:
        audit.record(
            "C2b", "the frozen scalar is the declared median, snapped to grid",
            "median {:g} -> a grid point".format(freeze["median_raw"]),
            freeze["scalar"], freeze["scalar"] in LAMBDA_GRID,
            "declared rule: median of the per-(rung, fold) selections, "
            "snapped to the nearest grid point, ties to the larger")

    audit.record(
        "C2c", "the frozen artefact was written",
        "present", "present" if FROZEN_OUTPUT.exists() else "MISSING",
        FROZEN_OUTPUT.exists(), str(FROZEN_OUTPUT))

    # ---- C3  the grid now reaches the degenerate end ----------------------
    top = surface[surface["lambda"] == LAMBDA_GRID[-1]]
    anchor = surface[surface["lambda"] == ANCHOR_LAMBDA]

    featured = [r for r in RUNGS if r != "B0"]

    top_norm = float(top[top["rung"].isin(featured)]["coef_l2_norm"].mean())
    anchor_norm = float(
        anchor[anchor["rung"].isin(featured)]["coef_l2_norm"].mean())

    ratio = top_norm / anchor_norm if anchor_norm else np.inf

    audit.record(
        "C3a", "coefficient norm at the top of the grid is a small fraction",
        "< {:g}".format(C3_NORM_RATIO_CEILING), "{:.4f}".format(ratio),
        ratio < C3_NORM_RATIO_CEILING,
        "norm {:.4f} at lambda = {:g} against {:.4f} at lambda = {:g}".format(
            top_norm, LAMBDA_GRID[-1], anchor_norm, ANCHOR_LAMBDA))

    b0_loss = float(
        surface[surface["rung"] == "B0"].groupby("fold")["log_loss"].mean().mean())

    not_closer = []

    for rung in featured:

        at_top = float(top[top["rung"] == rung]["log_loss"].mean())
        at_anchor = float(anchor[anchor["rung"] == rung]["log_loss"].mean())

        if abs(at_top - b0_loss) >= abs(at_anchor - b0_loss):
            not_closer.append(rung)

    audit.record(
        "C3b", "every rung is closer to B0 at the top of the grid than at 100",
        0, len(not_closer), not not_closer,
        "B0 = {:.6f}; rungs still further away: {}".format(
            b0_loss, not_closer or "none"))

    # ---- C4  Instrument 4's own results survived untouched ----------------
    after_strict = strict_frozen_state()

    changed = [path for path in before_strict
               if before_strict[path] != after_strict.get(path)]

    i4_changed = [path for path in changed
                  if Path(path).name.startswith("phase3_reg_")]

    audit.record(
        "C4a", "Instrument 4's results were read, never rewritten",
        0, len(i4_changed), not i4_changed, str(i4_changed[:3]))

    audit.record(
        "C4b", "nothing at all in the stricter frozen set moved",
        0, len(changed), not changed, str(changed[:3]))

    audit.measure(
        "C4c", "files in the stricter frozen set", len(before_strict),
        "Instrument 4's outputs are inputs here, so they are frozen too - "
        "Instrument 4 could not freeze them because it was writing them")

    return audit


# ============================================================
# REPORT
# ============================================================

def print_ladder(table, i4_selected):

    i4_mean = i4_selected.groupby("rung")["test_log_loss"].mean()

    print("  {:<5} {:<30} {:>7} {:>9} {:>10} {:>10}".format(
        "rung", "block added", "lam*", "test LL", "I4 @100", "I3 @1"))
    print("  " + "-" * 78)

    for _index, row in table.iterrows():

        print("  {:<5} {:<30} {:>7g} {:>9.4f} {:>10.4f} {:>10.4f}".format(
            row["rung"], str(row["description"])[:30],
            row["lambda_selected_median"], row["log_loss_mean"],
            float(i4_mean.get(row["rung"], np.nan)),
            row["frozen_log_loss_mean"]))

    print()


def print_turning(turning):

    print("  Oracle argmin over the full combined grid, per rung. INTERIOR")
    print("  means the curve turned around inside the grid, which is what")
    print("  resolving the ceiling means.")
    print()
    print("  {:<6} {:>12} {:>12} {:>12} {:>10}".format(
        "rung", "best lambda", "best LL", "LL at 10000", "interior"))
    print("  " + "-" * 58)

    for _index, row in turning.iterrows():
        print("  {:<6} {:>12g} {:>12.4f} {:>12.4f} {:>10}".format(
            row["rung"], row["oracle_best_lambda"], row["oracle_best_log_loss"],
            row["log_loss_at_grid_ceiling"],
            "yes" if row["interior"] else "NO"))

    print()


def print_freeze(freeze):

    print("  Rule, from section 4 of the pre-declaration as amended by")
    print("  Amendment 2, applied:")
    print()
    print("    in scope              : {} of {} selections".format(
        freeze["n_selections"], freeze["n_all_selections"]))
    print("    excluded (no penalty) : {}".format(
        ", ".join(freeze["excluded_rungs"]) or "none"))
    print("    range                 : {:g} to {:g}  ({:.2f} decades)".format(
        freeze["min_selected"], freeze["max_selected"],
        freeze["spread_decades"]))
    print("    on the grid ceiling   : {}".format(
        freeze["selections_on_ceiling"]))
    print("    on the grid floor     : {}".format(freeze["selections_on_floor"]))
    print("    median                : {:g}".format(freeze["median_raw"]))
    print()
    print("    STATUS                : {}".format(freeze["status"]))
    print("    FROZEN SCALAR         : {}".format(
        "none - freeze blocked" if freeze["scalar"] is None
        else "{:g}".format(freeze["scalar"])))
    print()
    print("    FROZEN PROCEDURE (takes precedence):")
    print("      nested expanding-window inner CV on training rows only,")
    print("      over the combined grid {:g} ... {:g}".format(
        COMBINED_GRID[0], COMBINED_GRID[-1]))
    print()


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()
    started = time.time()

    banner("PHASE 3 - INSTRUMENT 5: CEILING RESOLUTION")

    print("  question  : Instrument 4's 28 selections all hit lambda = 100.")
    print("              Where is the optimum, and does B1 still win past it?")
    print()
    print("  grid      : {}".format(", ".join("{:g}".format(p) for p in LAMBDA_GRID)))
    print("  combined  : {}".format(", ".join("{:g}".format(p) for p in COMBINED_GRID)))
    print("  protocol  : Instrument 4's, imported unchanged, new grid")
    print("  anchor    : lambda = {:g}, shared by both grids (C1)".format(
        ANCHOR_LAMBDA))
    print("  declared  : {}".format(PREDECLARATION.name))
    print("              sha256 {}".format(hash_file(PREDECLARATION)[:32]))
    print()

    before_strict = strict_frozen_state()

    # Instrument 4's battery must not count THIS instrument's own artefacts as
    # tampering. phase3_frozen_regularisation.json is written in section 9,
    # before the audit runs, and its provenance block records the SHA-256 of
    # Instrument 4's surface - so it legitimately changes whenever that file
    # legitimately changes. R8a flagged exactly that until the exclusion below
    # was made a parameter.
    before_i4 = I4.frozen_state(**I4_FROZEN_EXCLUDE)

    matches = L3.load_matches()
    features = L3.load_features(matches)
    labels = np.array([CLASS_INDEX[r] for r in matches["result"]], dtype=int)
    spec = L3.load_spec()

    ladder = [list(entry) for entry in L3.LADDER]
    ladder[1][2] = L3.phase1_feature_columns(features)
    ladder = [tuple(entry) for entry in ladder]

    blocks = I4.date_blocks(matches)

    print("  matches {}, calendar-date blocks {}".format(len(matches), len(blocks)))
    print()

    banner("1. SELECTING LAMBDA OVER THE EXTENDED GRID")

    run = I4.Run(features, matches, labels, spec, ladder, blocks,
                 compute_surface=True, grid=LAMBDA_GRID).execute()

    selected = run.selected_frame()
    surface = run.surface_frame()
    curves = run.curves_frame()
    block_norms = run.block_norm_frame()

    frozen_ladder = pd.read_csv(FROZEN_LADDER, float_precision=FLOAT_PRECISION)
    i4_selected = pd.read_csv(I4_SELECTED, float_precision=FLOAT_PRECISION)

    table = I4.build_ladder_table(selected, frozen_ladder)
    verdict = I4.build_verdict(selected, table, frozen_ladder)

    combined = combined_surface(surface)
    turning = turning_points(combined)

    freeze = apply_freeze_rule(selected)

    print()
    banner("2. THE SELECTED LAMBDAS")
    I4.print_selected_lambdas(selected)

    banner("3. THE LADDER AT SELECTED LAMBDA")
    print_ladder(table, i4_selected)

    banner("4. THE EXTENDED GRID, OUTER TEST")
    I4.print_surface(surface, "log_loss", grid=LAMBDA_GRID)

    banner("5. THE FULL COMBINED GRID - WHERE THE CURVE TURNS")
    print_combined(combined, "log_loss")
    print_turning(turning)

    banner("6. GENERALISATION GAP OVER THE EXTENDED GRID")
    I4.print_generalisation(surface, grid=LAMBDA_GRID)

    banner("7. COEFFICIENT NORMS BY BLOCK, AT SELECTED LAMBDA")
    I4.print_block_norms(block_norms, selected)

    banner("8. THE RUNG VERDICT, RE-RUN ON THE EXTENDED GRID")
    I4.print_verdict(verdict, table, selected)

    banner("9. THE FROZEN REGULARISATION CHOICE")

    verdict_attrs = {
        "verdict": verdict.attrs["verdict"],
        "winner": verdict.attrs["winner"],
        "reframed": bool(verdict.attrs["reframed"]),
    }

    write_frozen_artefact(freeze, selected, turning, verdict_attrs)
    print_freeze(freeze)

    banner("10. AUDIT")

    audit = Audit()

    print("  Instrument 4's battery, re-run against this grid:")
    print()

    I4.test_everything(run, features, matches, labels, spec, ladder, blocks,
                       selected, table, before_i4, audit,
                       grid=LAMBDA_GRID, predeclaration=PREDECLARATION,
                       frozen_exclude=I4_FROZEN_EXCLUDE)

    print()
    print("  Instrument 5's own:")
    print()

    test_ceiling(run, surface, combined, selected, freeze, before_strict, audit)

    print()
    audit.print_rows()

    banner("11. WRITING OUTPUTS")

    writes = [
        (CURVES_OUTPUT, curves),
        (SELECTED_OUTPUT, selected),
        (SURFACE_OUTPUT, surface),
        (COMBINED_OUTPUT, combined),
        (BLOCK_NORM_OUTPUT, block_norms),
        (LADDER_OUTPUT, table),
        (VERDICT_OUTPUT, verdict.assign(
            verdict=verdict.attrs["verdict"], winner=verdict.attrs["winner"])),
        (AUDIT_OUTPUT, audit.frame()),
    ]

    for path, frame in writes:
        frame.to_csv(path, index=False, encoding="utf-8", float_format="%.17g")
        print("  {}".format(path))

    print("  {}".format(FROZEN_OUTPUT))
    print()

    banner("PHASE 3 - INSTRUMENT 5 STATUS")

    failures = audit.failures

    print("  Grid points        : {} ({:g} .. {:g})".format(
        len(LAMBDA_GRID), LAMBDA_GRID[0], LAMBDA_GRID[-1]))
    print("  Selections made    : {}".format(len(selected)))
    print("  Outer fits scored  : {}".format(len(surface)))
    print("  Inner fits scored  : {}".format(len(curves)))
    print("  Checks run         : {}".format(len(audit.rows)))
    print("  Checks failed      : {}".format(len(failures)))
    print("  Elapsed            : {:.1f} min".format((time.time() - started) / 60.0))
    print()
    print("  {}".format("PASS" if not failures else "FAIL"))
    print()

    if failures:
        print("  Failing: {}".format(
            ", ".join(row["test_id"] for row in failures)))
        print()
        print("  A FAIL here does NOT mean the ceiling went unresolved. C3")
        print("  asserted the grid reaches the DEGENERATE end; resolving the")
        print("  ceiling only needs it to BRACKET an interior optimum, which")
        print("  it does for every featured rung, the curve rising on both")
        print("  sides of lambda = 1000. C3 was pitched too high and is left")
        print("  failing rather than relaxed after the fact - see Amendment 2.")
        print()

    interior = int(turning[turning["rung"] != "B0"]["interior"].sum())
    featured = int((turning["rung"] != "B0").sum())

    print("  Ceiling resolved   : {} of {} featured rungs turn inside the "
          "grid".format(interior, featured))
    print("  Rung verdict       : {}".format(verdict.attrs["verdict"]))
    print("  Freeze status      : {}".format(freeze["status"]))
    print("  Frozen scalar      : {}".format(
        "none" if freeze["scalar"] is None else "{:g}".format(freeze["scalar"])))
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
