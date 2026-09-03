"""
===============================================================================
PHASE 6 - TWENTY-MATCH FINISHING PERSISTENCE
===============================================================================

E1c's F8 left the longer horizon explicitly open:

    "It cannot establish that finishing skill does not exist. It measures a
     five-match window, which F4.1 argues in advance is noise-dominated. A
     longer window might carry a persistent signal, and this instrument
     would not see it."

This discharges that caveat or fails to, and says which.

NO FIT, NO GATE, NO MODEL. It is E1c's own autocorrelation arm at a different
window length. Nothing here enters a design matrix and no rung is built from
it. It cannot and does not reopen the freeze - the freeze is committed, its
hash is in the manifest, and H5.1 forbids changing the frozen model because of
anything measured afterwards.

-------------------------------------------------------------------------------
DECLARED BEFORE RUNNING
-------------------------------------------------------------------------------

W1  THE QUANTITY is E1c's, unchanged: a team's finishing over a window is

        (goal differential per match)  -  c x (SoT differential per match)

    with c the per-fold conversion constant from that fold's first E1a
    cutoff, exactly as PHASE5_E1C_FINISHING_PREDECLARATION.txt F1.2 fixes it.
    Nothing is re-derived and nothing is re-tuned.

W2  THE WINDOW is TWENTY matches, non-overlapping, within season, training
    rows only. The only parameter that changes from E1c.

W3  THE THRESHOLD IS |r| < 0.10, THE SAME ONE E1c DECLARED. Reused rather
    than re-chosen, because a threshold picked for a second horizon after
    seeing the first horizon's result is a threshold picked to be passed.

W4  THE POWER QUESTION IS ASKED FIRST AND ANSWERED BEFORE THE ESTIMATE IS
    READ. Twenty-match non-overlapping windows within a 38-match season admit
    at most ONE pair per team-season, against seven anchors at five matches.
    That is roughly a quarter of E1c's pair count at best.

    The standard error of a correlation near zero is about 1/sqrt(n-3). To
    resolve |r| = 0.10 from zero at 95% - that is, for a sample r of 0.10 to
    have a confidence interval excluding zero - needs roughly

        1.96 / sqrt(n - 3)  <  0.10     =>     n  >  387

    DECLARED NOW: if the pair count falls materially below that, THE RUNG IS
    UNDERPOWERED AND IS REPORTED AS UNDERPOWERED. It is not reported as a
    null. An estimate near zero from a sample that could not have detected
    0.10 is not evidence of absence, and F8's caveat would remain open.

W5  THE THREE READINGS, fixed in advance:

    |r| well above 0.10, and powered
        finishing IS persistent at this horizon. Chance quality has something
        real to measure and the xG question REOPENS as a post-freeze research
        item. The freeze is NOT reopened.

    |r| near zero, and powered
        finishing is not persistent at either horizon. The xG question closes
        on evidence rather than on cost, and F8's caveat is discharged.

    underpowered
        reported as such and read NEITHER way. F8's caveat stays open and the
        xG question stays where E1c left it.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase3_ablation_ladder as L3              # noqa: E402
import phase5_e1a_sot_ratings as E1A             # noqa: E402
import phase5_e1c_finishing as E1C               # noqa: E402


OUTPUTS_DIR = E1C.OUTPUTS_DIR

PERSIST_OUTPUT = OUTPUTS_DIR / "phase6_persistence20.csv"
FEASIBILITY_OUTPUT = OUTPUTS_DIR / "phase6_persistence20_feasibility.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase6_persistence20_audit.csv"

FLOAT_FORMAT = "%.17g"

WINDOW = 20
THRESHOLD = 0.10

# W4. n > 387 for a sample r of 0.10 to exclude zero at 95%.
REQUIRED_PAIRS = int(np.ceil((1.96 / THRESHOLD) ** 2 + 3))


def pairs_at(sides, spec, window):
    """
    Non-overlapping consecutive windows, within a team-season, training rows
    only. E1c's persistence() at a different window length.

    The value at match number `window + 1` covers matches 1..window, at
    `2*window + 1` covers window+1..2*window, and so on - so anchors spaced
    `window` apart give windows that do not overlap.
    """

    anchors = list(range(window + 1, 39, window))

    rows = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        train_seasons = list(fold_spec["train_seasons"])

        subset = sides[sides["season"].isin(train_seasons)
                       & sides["match_number"].isin(anchors)].copy()

        subset = subset.sort_values(["season", "team", "match_number"])

        grouped = subset.groupby(["season", "team"], sort=False)
        subset["next_finishing"] = grouped["finishing"].shift(-1)
        subset["next_number"] = grouped["match_number"].shift(-1)

        pairs = subset[(subset["next_number"] - subset["match_number"] == window)
                       & subset["finishing"].notna()
                       & subset["next_finishing"].notna()]

        n = len(pairs)
        r = float(pairs["finishing"].corr(pairs["next_finishing"])) if n > 2 \
            else float("nan")

        se = 1.0 / np.sqrt(n - 3) if n > 3 else float("nan")

        rows.append({
            "window": window, "fold": fold,
            "train_seasons": " + ".join(train_seasons),
            "anchors": len(anchors), "n_pairs": n, "r": r,
            "se_approx": se,
            "resolves_0.10": bool(n > REQUIRED_PAIRS),
        })

    return pd.DataFrame(rows), anchors


def main():

    configure_stdout()

    banner("PHASE 6 - TWENTY-MATCH FINISHING PERSISTENCE")

    print("  Declared before running: window 20, threshold |r| < 0.10 (E1c's")
    print("  own, reused rather than re-chosen), power reported FIRST.")
    print("  No fit, no gate, no model. The freeze is not reopened.")
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches().copy()
    matches["match_id"] = matches.index
    matches["role_is_test"] = matches["season"].isin(
        [str(f["test_season"]) for f in spec["folds"]])
    matches = E1A.load_shots(matches, Audit())

    constants = E1C.conversion_constants(matches, spec, Audit())
    c_by_fold = {int(r.fold): float(r.c_decayed) for r in constants.itertuples()}

    print("  c per fold, unchanged from E1c: {}".format(
        {k: round(v, 5) for k, v in c_by_fold.items()}))
    print()

    # ============================================================
    banner("1. POWER, BEFORE THE ESTIMATE IS READ")

    print("  W4: to resolve |r| = {:.2f} from zero at 95% needs roughly".format(
        THRESHOLD))
    print("      1.96 / sqrt(n - 3) < {:.2f}   =>   n > {}".format(
        THRESHOLD, REQUIRED_PAIRS))
    print()

    tables = {}

    for window in (5, WINDOW):
        rows = []
        for fold in sorted(c_by_fold):
            _block, sides = E1C.build_finishing(matches, c_by_fold[fold])
            table, anchors = pairs_at(sides, spec, window)
            rows.append(table[table["fold"] == fold].iloc[0].to_dict())
        tables[window] = pd.DataFrame(rows)

    five, twenty = tables[5], tables[WINDOW]

    print("  {:<8} {:<40} {:>8} {:>9}".format(
        "window", "training seasons", "anchors", "n pairs"))
    print("  " + "-" * 70)
    for w, table in ((5, five), (WINDOW, twenty)):
        for row in table.itertuples():
            print("  {:<8} {:<40} {:>8} {:>9}".format(
                w if row.fold == 1 else "", row.train_seasons, row.anchors,
                row.n_pairs))
        print()

    total20 = int(twenty["n_pairs"].max())
    powered = bool(total20 > REQUIRED_PAIRS)

    print("  largest twenty-match pair count at any fold : {}".format(total20))
    print("  required to resolve |r| = 0.10              : {}".format(
        REQUIRED_PAIRS))
    print("  shortfall                                   : {}".format(
        REQUIRED_PAIRS - total20))
    print("  approximate SE at that n                    : {:.4f}".format(
        1.0 / np.sqrt(total20 - 3) if total20 > 3 else float("nan")))
    print()
    print("  POWERED TO RESOLVE THE THRESHOLD: {}".format(powered))

    # ---- WHY IT IS ZERO, AND WHAT THAT MEANS FOR EVERY LONGER WINDOW -----
    #
    # Not "few pairs". ZERO, and structurally so: two non-overlapping
    # twenty-match windows need FORTY matches and a Premier League season has
    # thirty-eight. The declared rung cannot be run at any sample size.
    #
    # So the honest follow-up is not "try a slightly shorter window and see" -
    # it is to ask whether ANY window longer than E1c's can be powered on this
    # dataset. That is pure combinatorics with no correlation in it, so it
    # cannot be fishing.

    print()
    print("  WHY ZERO, AND WHETHER ANY LONGER WINDOW COULD WORK")
    print()
    print("  Two non-overlapping windows of length w need 2w matches, and a")
    print("  season has 38. Pairs per team-season is the number of consecutive")
    print("  anchors w apart that both fit. No correlation is computed here.")
    print()
    print("  {:>4} {:>9} {:>22} {:>16} {:>10}".format(
        "w", "anchors", "pairs / team-season", "max pairs f4", "powered"))
    print("  " + "-" * 68)

    feasibility = []

    for w in (5, 6, 8, 10, 12, 15, 18, 19, 20, 25):
        anchors_w = [a for a in range(w + 1, 39, w)]
        per_season = max(0, len(anchors_w) - 1)
        # 20 teams x 4 training seasons is fold 4's ceiling
        ceiling = per_season * 20 * 4
        feasibility.append({"window": w, "anchors": len(anchors_w),
                            "pairs_per_team_season": per_season,
                            "max_pairs_fold4": ceiling,
                            "powered": bool(ceiling > REQUIRED_PAIRS)})
        print("  {:>4} {:>9} {:>22} {:>16} {:>10}".format(
            w, len(anchors_w), per_season, ceiling,
            "yes" if ceiling > REQUIRED_PAIRS else "no"))

    feasible = [f["window"] for f in feasibility if f["powered"]]

    print()
    print("  ONLY w in {} clears n > {} on this dataset.".format(
        feasible, REQUIRED_PAIRS))
    print("  The longest window that yields ANY pair at all is 18, and it")
    print("  gives 80 - a fifth of what the threshold needs.")

    audit.record(
        "W4b", "some window longer than E1c's five can resolve |r| = 0.10 on "
               "this dataset",
        "at least one w > 5", "powered windows: {}".format(feasible),
        any(w > 5 for w in feasible),
        "PURE COMBINATORICS, no correlation computed, so this cannot be "
        "fishing. 20 teams x 4 training seasons is fold 4's ceiling. The "
        "five-season dataset does not contain a longer horizon that could be "
        "read, whatever the answer at that horizon would have been")

    audit.record(
        "W4", "the twenty-match rung can resolve |r| = 0.10 from zero at 95%",
        "n > {}".format(REQUIRED_PAIRS), "n = {}".format(total20), powered,
        "declared BEFORE the estimate was read. A twenty-match "
        "non-overlapping window admits at most one pair per team-season "
        "inside a 38-match season, against seven anchors at five matches. "
        "SE ~ 1/sqrt(n-3) = {:.4f}".format(
            1.0 / np.sqrt(total20 - 3) if total20 > 3 else float("nan")))

    # ============================================================
    banner("2. THE ESTIMATE")

    print("  {:<8} {:<40} {:>8} {:>9} {:>9}".format(
        "window", "training seasons", "n pairs", "r", "SE"))
    print("  " + "-" * 80)
    for w, table in ((5, five), (WINDOW, twenty)):
        for row in table.itertuples():
            print("  {:<8} {:<40} {:>8} {:>+9.4f} {:>9}".format(
                w if row.fold == 1 else "", row.train_seasons, row.n_pairs,
                row.r,
                "{:.4f}".format(row.se_approx)
                if np.isfinite(row.se_approx) else "-"))
        print()

    largest = float(twenty["r"].abs().max())

    print("  largest |r| at twenty matches : {:.4f}".format(largest))
    print("  declared threshold            : {:.2f}".format(THRESHOLD))
    print()

    if total20 == 0:
        reading = ("UNRUNNABLE, not merely underpowered - two non-overlapping "
                   "twenty-match windows need 40 matches and a season has 38. "
                   "Read NEITHER way. F8's caveat stays OPEN, and the dataset "
                   "cannot close it at any window longer than about six.")
    elif not powered:
        reading = ("UNDERPOWERED - reported as such and read NEITHER way. "
                   "F8's caveat stays OPEN.")
    elif largest >= THRESHOLD:
        reading = ("PERSISTENT at this horizon - a POST-FREEZE research item. "
                   "The freeze is NOT reopened.")
    else:
        reading = ("NEAR ZERO and powered - finishing is not persistent at "
                   "either horizon. F8's caveat is DISCHARGED.")

    print("  W5 READING: {}".format(reading))

    audit.measure(
        "W5", "the declared reading", reading,
        "fixed in advance, so the branch that fired cannot be presented as "
        "the one that was expected")

    audit.record(
        "W2", "the twenty-match windows do not overlap and stay within season",
        "anchors {} apart".format(WINDOW),
        "anchors {}".format(list(range(WINDOW + 1, 39, WINDOW))),
        True,
        "a pair is kept only where the next anchor is exactly {} matches "
        "later, so a team-season with a gap contributes nothing rather than "
        "an overlapping pair".format(WINDOW))

    # ============================================================
    banner("3. WRITING")

    combined = pd.concat([five, twenty], ignore_index=True)
    combined.to_csv(PERSIST_OUTPUT, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)

    pd.DataFrame(feasibility).to_csv(
        FEASIBILITY_OUTPUT, index=False, encoding="utf-8",
        float_format=FLOAT_FORMAT)

    frame = audit.frame()
    frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8",
                 float_format=FLOAT_FORMAT)

    for path in (PERSIST_OUTPUT, FEASIBILITY_OUTPUT, AUDIT_OUTPUT):
        print("  {}".format(path))

    failures = int((frame["status"] == "FAIL").sum())

    print()
    print("  Checks run    : {}".format(len(frame)))
    print("  Checks failed : {}".format(failures))
    print()
    print("  {}".format("PASS" if failures == 0 else
                        "FAIL (see W4 - underpowered is a RESULT, not a bug)"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
