"""
===============================================================================
ELO v1 - THE FULL PHASE 0 METRIC SET, INCLUDING RPS
===============================================================================

WHY THIS EXISTS AND WHY IT IS NOT A RE-RUN

    Elo v1 ran 2026-08-21. The harness gained RPS on 2026-08-24. So Elo's fold
    summary carries five metrics where every other model in the ladder carries
    six, and the Phase 4 pre-declaration carries "Elo v1 0.9994" forward as a
    reference anchor - a line that would be log-loss-only while every other
    line is dual-metric, which breaks the sign-agreement rule for that
    comparison.

    Re-running phase2_elo_baseline.py does NOT fix this, and it is worth being
    precise about why. That script already calls the harness's evaluate(),
    which now returns rps. It then builds its summary row by naming five
    metrics explicitly:

        "accuracy": scores["accuracy"], ... "brier_score": scores["brier_score"]

    rps is computed and dropped on the floor. Re-running unchanged reproduces
    the same five columns - verified: all three Elo artefacts came back
    BYTE-IDENTICAL, sha256 for sha256.

    So the choice was between editing a frozen Phase 2 instrument, or deriving
    the missing metric from what that instrument already published. This takes
    the second route: phase2_elo_results.csv contains every per-match
    probability, which is everything the metric needs.

THE RECOMPUTATION PATH IS VALIDATED BEFORE IT IS TRUSTED

    E1 recomputes the FIVE metrics Elo v1 already published, from the stored
    probabilities, through the same harness evaluate(), and requires them to
    match the stored summary to < 1e-9.

    That matters. If the recomputation path could not reproduce the metrics we
    already have, its rps would be worthless. Deriving a new number through an
    unvalidated path and trusting it because it looks plausible is exactly the
    failure PROJECT_GOTCHAS.md warns about - a verification that shares an
    assumption with the thing it verifies.

WHAT IS WRITTEN

    outputs/phase2_elo_metrics_full.csv - per fold and pooled, six metrics.
    The v1 artefacts are READ and never rewritten.
===============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import evaluate  # noqa: E402
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESULTS_CSV = OUTPUTS_DIR / "phase2_elo_results.csv"
SUMMARY_CSV = OUTPUTS_DIR / "phase2_elo_fold_summary.csv"

OUTPUT = OUTPUTS_DIR / "phase2_elo_metrics_full.csv"

FLOAT_PRECISION = "round_trip"

METRICS = ["accuracy", "balanced_accuracy", "macro_f1",
           "log_loss", "brier_score", "rps"]

# The five Elo v1 published. These must reproduce, or the path is not trusted.
PUBLISHED = ["accuracy", "balanced_accuracy", "macro_f1",
             "log_loss", "brier_score"]

EXPECTED_TEST_MATCHES = 1520


def load():

    results = pd.read_csv(RESULTS_CSV, float_precision=FLOAT_PRECISION)
    summary = pd.read_csv(SUMMARY_CSV, float_precision=FLOAT_PRECISION)

    return results, summary.sort_values("fold").reset_index(drop=True)


def evaluated_rows(results):
    """
    The 1,520 outer test matches.

    phase2_elo_results.csv carries all 1,900 matches - Elo predicts through the
    training seasons too, because that is how the ratings get built - and marks
    which ones were scored. Pooling over all 1,900 would silently produce a
    different number from the fold mean, which is the trap that made an early
    pooled figure read 1.0027 against the fold mean's 0.9994.
    """

    if "evaluated" in results.columns:
        mask = results["evaluated"].astype(str).str.lower().isin(["true", "1"])
        block = results[mask]
    else:
        block = results[results["role"].astype(str).str.lower() == "test"]

    return block.reset_index(drop=True)


def score(block):

    proba = block[["p_home", "p_draw", "p_away"]].to_numpy(float)

    return evaluate(block["actual_result"].to_numpy(), proba)


def build(results, summary, audit):

    test = evaluated_rows(results)

    audit.record(
        "E0", "the evaluated set is the 1,520 outer test matches",
        EXPECTED_TEST_MATCHES, len(test), len(test) == EXPECTED_TEST_MATCHES,
        "phase2_elo_results.csv holds all 1,900; pooling over the wrong "
        "subset is how a pooled log loss of 1.0027 appears where the fold "
        "mean is 0.9994")

    rows = []

    worst = 0.0
    worst_where = ""

    for fold in sorted(test["fold"].unique()):

        block = test[test["fold"] == fold]
        scores = score(block)

        stored = summary[summary["fold"] == fold].iloc[0]

        for metric in PUBLISHED:
            difference = abs(float(stored[metric]) - scores[metric])
            if difference > worst:
                worst, worst_where = difference, "fold {} {}".format(fold, metric)

        row = {"scope": "fold", "fold": int(fold),
               "test_season": block["test_season"].iloc[0],
               "matches": len(block)}
        row.update({m: scores[m] for m in METRICS})
        rows.append(row)

    audit.record(
        "E1", "the five metrics Elo v1 published reproduce from its own "
              "stored probabilities",
        "< 1e-9", "{:.3e} ({})".format(worst, worst_where or "-"), worst < 1e-9,
        "this is what makes the sixth metric trustworthy; an unvalidated "
        "derivation path would make its rps a guess")

    # ---- pooled over all 1,520, and the fold mean beside it -------------
    pooled = score(test)

    row = {"scope": "pooled", "fold": -1, "test_season": "all",
           "matches": len(test)}
    row.update({m: pooled[m] for m in METRICS})
    rows.append(row)

    frame = pd.DataFrame(rows)

    fold_rows = frame[frame["scope"] == "fold"]

    for metric in METRICS:
        difference = abs(float(fold_rows[metric].mean()) - pooled[metric])
        if metric == "log_loss":
            audit.record(
                "E2", "pooled log loss equals the unweighted fold mean",
                "< 1e-9", "{:.3e}".format(difference), difference < 1e-9,
                "the four folds hold exactly 380 test matches each, so these "
                "must agree; if they diverge the evaluated subset is wrong")

    audit.record(
        "E3", "the reference value 0.9994 is reproduced",
        "0.9994", "{:.4f}".format(pooled["log_loss"]),
        abs(pooled["log_loss"] - 0.9994) < 5e-5,
        "the figure the Phase 4 pre-declaration carries forward as the Elo "
        "anchor")

    audit.record(
        "E4", "rps is present and in range for every row",
        "0 <= rps <= 1",
        "[{:.4f}, {:.4f}]".format(frame["rps"].min(), frame["rps"].max()),
        bool((frame["rps"] >= 0).all() and (frame["rps"] <= 1).all()))

    return frame


def main():

    configure_stdout()

    banner("ELO v1 - FULL PHASE 0 METRIC SET")

    print("  Elo v1 ran before the harness had RPS. Its own script computes")
    print("  the metric today but names five columns when building its summary")
    print("  row, so a re-run reproduces byte-identical five-metric output.")
    print()
    print("  This derives the sixth from the per-match probabilities Elo v1")
    print("  already published, and validates the derivation by reproducing")
    print("  the five it published (E1) before trusting the one it did not.")
    print()

    results, summary = load()

    audit = Audit()
    frame = build(results, summary, audit)

    banner("1. THE FULL METRIC SET")

    print("  {:<8} {:<12} {:>8} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "scope", "season", "n", "acc", "logloss", "brier", "RPS", "macroF1"))
    print("  " + "-" * 82)

    for _i, row in frame.iterrows():
        print("  {:<8} {:<12} {:>8} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} "
              "{:>9.4f}".format(
                  row["scope"], str(row["test_season"])[:12], int(row["matches"]),
                  row["accuracy"], row["log_loss"], row["brier_score"],
                  row["rps"], row["macro_f1"]))

    print()

    banner("2. AUDIT")
    audit.print_rows()

    banner("3. WRITING")

    frame.to_csv(OUTPUT, index=False, encoding="utf-8", float_format="%.17g")
    print("  {}".format(OUTPUT))
    print()
    print("  The v1 artefacts were read and not rewritten.")
    print()

    failures = audit.failures

    print("  Checks run    : {}".format(len(audit.rows)))
    print("  Checks failed : {}".format(len(failures)))
    print()
    print("  {}".format("PASS" if not failures else "FAIL"))
    print()

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
