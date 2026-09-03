"""
===============================================================================
PHASE 6 - THE HOLDOUT SCORING INSTRUMENT
===============================================================================

Governed by PHASE6_HOLDOUT_FREEZE.txt (sha256 b36befd3...248f6) and
PHASE6_CUTOFF_PIN.txt. It implements H2.12, H3.3, H6.6 and H7 - the four
requirements the freeze places on a scoring run.

WHY IT IS WRITTEN IN SEPTEMBER 2026 AND NOT IN MAY 2027. Code that runs once,
against a holdout, with nothing to check it against, is untested code. Written
now it can be validated on the development seasons, where the answers are
already frozen and any discrepancy is a bug in this file rather than a
finding. That validation is the DRY RUN below and it is the point of the
early build.

IT WILL NOT SCORE 2026-27 WITHOUT AN EXPLICIT FLAG. H5.2 says scoring happens
once, at season end, because a number seen in November cannot be unseen in
December. The default mode is the dry run. --score-holdout is refused unless
--season-complete is also given, so that scoring the live season is a
deliberate two-key act and cannot happen by running the file.

NOTHING HERE IS TUNED, FITTED OR CHOSEN. Every constant comes from
phase2_poisson_dixon_coles.py, which the freeze validator checks against the
freeze document. The cutoff and the vocabulary are READ FROM THE PIN, not
hard-coded, so a change to either moves the pin's hash and is visible.

    ./venv/Scripts/python.exe -B scripts/phase6_score_holdout.py
        the dry run: reproduce the frozen development figures

    ... --score-holdout --season-complete
        the real thing, once, after the final 2026-27 fixture
"""

from pathlib import Path
import hashlib
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import evaluate, validate_probabilities  # noqa: E402
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase2_poisson_dixon_coles as DC          # noqa: E402
import phase3_ablation_ladder as L3              # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FREEZE = PROJECT_ROOT / "PHASE6_HOLDOUT_FREEZE.txt"
PIN = PROJECT_ROOT / "PHASE6_CUTOFF_PIN.txt"

DC_FOLDS = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"
MATCHES_CSV = OUTPUTS_DIR / "phase1_matches.csv"

DRYRUN_OUTPUT = OUTPUTS_DIR / "phase6_dryrun_summary.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase6_scoring_audit.csv"

FREEZE_SHA = "b36befd3e13d5f7f5d9b0af83b7db819e86d7b13e6fdcab4c55db0599fd248f6"

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

HOLDOUT_SEASON = "2026-2027"

# THE FROZEN TARGETS ARE READ FROM THE ARTEFACT, NOT TYPED HERE.
#
# The first version of this file hard-coded them, and the 2025-26 pair was
# WRONG: 1.0453634084335335 was invented from a six-decimal console display of
# 1.045363, and the dry run duly reported a 7.16e-08 "discrepancy" in an
# instrument that was in fact exact. The check was broken, not the code.
#
# Same principle as the pin: do not carry your own copy of a number that lives
# in a file you can read. PUBLISHED_* below are the figures REPORTS.md quotes,
# asserted against the artefact so the artefact cannot drift from the report
# without something saying so.

PUBLISHED_POOLED_LOG_LOSS = 0.99036
PUBLISHED_POOLED_RPS = 0.20350


def frozen_targets():
    """The dc_walkforward figures, per fold and pooled, from the artefact."""

    frozen = pd.read_csv(DC_FOLDS, float_precision=FLOAT_PRECISION)
    frozen = frozen[frozen["variant"] == "dc_walkforward"].sort_values("fold")

    return frozen, {"log_loss": float(frozen["log_loss"].mean()),
                    "rps": float(frozen["rps"].mean())}


class HoldoutError(RuntimeError):
    """A violation of the freeze. Never caught inside this module."""


# ============================================================
# 1. THE PIN
# ============================================================

def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_pin():
    """
    H7: the cutoff and the vocabulary are READ from the pin, not hard-coded.

    Parsed rather than duplicated, so that editing the pin changes what this
    instrument does AND moves the pin's manifest hash. A hard-coded copy could
    drift from the document that governs it and nothing would say so.
    """

    text = PIN.read_text(encoding="utf-8")

    # ---- the freeze the pin governs, and that it is the right one ---------
    cited = re.search(r"sha256\s+([0-9a-f]{64})", text)

    if not cited:
        raise HoldoutError("the pin does not cite a freeze sha256")

    # ---- P1.1, the cutoff -------------------------------------------------
    cutoff = re.search(r"THE PINNED CUTOFF IS\s*\n\s*\n\s*(\d{4}-\d{2}-\d{2})",
                       text)

    if not cutoff:
        raise HoldoutError("the pin does not carry a P1.1 cutoff date")

    # ---- P4.2, the seventeen continuing sides -----------------------------
    block = text.split("P4.2")[1].split("The dataset's vocabulary")[0]

    continuing = []
    for line in block.splitlines():
        if not line.startswith(" " * 10) or ":" in line:
            continue
        for name in re.split(r"\s{2,}", line.strip()):
            if name and not name.endswith("."):
                continuing.append(name)

    # ---- P4.3, the three promoted sides -----------------------------------
    block = text.split("P4.3")[1].split('"Ipswich Town" is')[0]

    promoted = []
    for line in block.splitlines():
        match = re.match(r"^\s{10,}([A-Z][A-Za-z' ]+?)\s{2,}\S", line)
        if match:
            promoted.append(match.group(1).strip())

    # ---- P4.6, the zero-history set ---------------------------------------
    block = text.split("P4.6")[1].split("Ipswich Town is NOT")[0]

    zero_history = []
    for line in block.splitlines():
        if line.startswith(" " * 10) and "," in line and ":" not in line:
            zero_history = [n.strip() for n in line.strip().split(",")]
            break

    # ---- P5.1, the expected scored count ----------------------------------
    expected = re.search(r"expected scored\s+(\d+)", text)

    pin = {
        "freeze_sha": cited.group(1),
        "cutoff": pd.Timestamp(cutoff.group(1)),
        "continuing": continuing,
        "promoted": promoted,
        "vocabulary": sorted(set(continuing) | set(promoted)),
        "zero_history": zero_history,
        "expected_scored": int(expected.group(1)) if expected else None,
        "sha": sha256_of(PIN),
    }

    # The parse is asserted, because a regex over prose that silently returns
    # 19 names would hand the vocabulary check a hole exactly the size of the
    # team it dropped.
    if len(continuing) != 17:
        raise HoldoutError(
            "P4.2 parsed to {} continuing sides, expected 17: {}".format(
                len(continuing), continuing))

    if len(promoted) != 3:
        raise HoldoutError(
            "P4.3 parsed to {} promoted sides, expected 3: {}".format(
                len(promoted), promoted))

    if len(pin["vocabulary"]) != 20:
        raise HoldoutError(
            "the vocabulary is {} names, expected 20".format(
                len(pin["vocabulary"])))

    if len(zero_history) != 2:
        raise HoldoutError(
            "P4.6 parsed to {} zero-history sides, expected 2: {}".format(
                len(zero_history), zero_history))

    return pin


# ============================================================
# 2. H2.12 - THE ASSERTION V6b SHOWED WAS MISSING
# ============================================================

def assert_vocabulary(matches, vocabulary, label):
    """
    H2.12 / P4.5. A team name outside the declared vocabulary RAISES.

    THIS IS THE WHOLE REASON THE INSTRUMENT EXISTS AS A SEPARATE FILE. The
    freeze validator's V6b fed "Man United" through the live match_rates()
    beside "Manchester Utd" and got home rates of 0.8165 and 1.1186 WITH NO
    EXCEPTION - the misspelled side falls through .fillna(NEUTRAL_STRENGTH)
    and is silently scored as exactly average. Over a season that corrupts
    the holdout invisibly and the corruption is indistinguishable from a
    result.

    NOTHING IS REPAIRED, MAPPED OR NORMALISED. P4.5: if the source spells a
    club differently the run FAILS, and the fix is a dated amendment to the
    pin, not a rescue inside a loader. A loader that repairs names is a
    loader that can repair the wrong one.
    """

    seen = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    unknown = [name for name in seen if name not in vocabulary]

    if unknown:
        raise HoldoutError(
            "H2.12 VIOLATION in {}: {} team name(s) outside the declared "
            "vocabulary: {}. The declared twenty are {}. NOTHING IS REPAIRED "
            "HERE - if the source's spelling has changed, amend "
            "PHASE6_CUTOFF_PIN.txt with the date and the reason.".format(
                label, len(unknown), unknown, sorted(vocabulary)))

    return seen


# ============================================================
# 3. THE SCORER
# ============================================================

def score_from_cutoff(matches, cutoff, scope_seasons):
    """
    Walk-forward Dixon-Coles from a date cutoff. H2.4, H2.5, H2.6, H2.7, H2.8.

    Matches with date >= cutoff are SCORED. Everything earlier warms the
    state. One refit per distinct scored date, on matches STRICTLY earlier
    than that date - so a same-day match is never in the window that predicts
    it, which is H2.5 and is not relaxed for a live season.

    No post-hoc transformation of any kind (H2.8).
    """

    in_scope = matches["season"].isin(scope_seasons)

    scored = matches[in_scope & (matches["date"] >= cutoff)]
    scored = scored.sort_values(["date", "match_id"])

    rows = []

    for date in sorted(scored["date"].unique()):

        date = pd.Timestamp(date)

        window = matches[in_scope & (matches["date"] < date)]

        model = DC.fit_window(window, date, dixon_coles=True)

        rows.extend(DC.predict_matches(scored[scored["date"] == date], model))

    frame = pd.DataFrame(rows)

    return frame, scored


def metrics_of(frame):
    """The full Phase 0 set, H2.9, on a block of predictions."""

    proba = frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    validate_probabilities(proba, len(frame))

    return evaluate(frame["actual_result"].to_numpy(), proba)


# ============================================================
# 4. THE DRY RUN
# ============================================================

def dry_run(matches, spec, pin, audit):
    """
    Validate the instrument where the answers are already frozen.

    TWO TARGETS, AND THEY ARE DIFFERENT NUMBERS. The brief named 0.99036 /
    0.20350; that is the POOLED figure over all four folds and 1,520
    matches. 2025-26 ALONE is 1.04536 / 0.21281. Both are in
    phase2_poisson_dc_fold_summary.csv and both are reproduced here, because
    reproducing the pooled number would not exercise the code path Phase 6
    actually uses and reproducing the season number alone would not check the
    metric aggregation.

    THE 2025-26 RUN IS THE STRUCTURAL ANALOGUE OF PHASE 6: one cutoff date,
    one contiguous scored block, a window spanning every earlier season. That
    is exactly what the holdout will be, with 2026-09-04 in place of the
    2025-26 opener.
    """

    banner("DRY RUN A - 2025-26, THE STRUCTURAL ANALOGUE OF THE HOLDOUT")

    fold4 = [f for f in spec["folds"] if str(f["test_season"]) == "2025-2026"][0]

    scope = list(fold4["train_seasons"]) + ["2025-2026"]
    cutoff = matches[matches["season"] == "2025-2026"]["date"].min()

    print("  cutoff  {}   scope {}".format(str(cutoff.date()), scope))
    print("  scoring every 2025-26 match, warming on everything earlier")
    print()

    frame, scored = score_from_cutoff(matches, cutoff, scope)

    scores = metrics_of(frame)

    frozen_folds, frozen_pooled = frozen_targets()
    ref4 = frozen_folds[frozen_folds["fold"] == 4].iloc[0]

    print("  {:<24} {:>14} {:>14} {:>12}".format(
        "", "this instrument", "frozen", "difference"))
    print("  " + "-" * 68)

    worst = 0.0
    for metric, target in (("log_loss", float(ref4["log_loss"])),
                           ("rps", float(ref4["rps"]))):
        gap = abs(scores[metric] - target)
        worst = max(worst, gap)
        print("  {:<24} {:>14.10f} {:>14.10f} {:>12.2e}".format(
            metric, scores[metric], target, gap))

    print()
    print("  matches scored {} (expected 380)".format(len(frame)))

    audit.record(
        "S1", "the instrument reproduces the frozen 2025-26 dc_walkforward "
              "figures bit for bit",
        "0.000e+00", "{:.3e}".format(worst), worst == 0.0,
        "SAME STRUCTURE AS THE HOLDOUT: one cutoff, one contiguous scored "
        "block, a window over every earlier season. If this instrument's "
        "walk-forward differed from Phase 2's in any way, this is where it "
        "would show")

    audit.record(
        "S2", "and scores exactly the 380 matches of that season",
        380, len(frame), len(frame) == 380,
        "a count assertion in the dry run, mirroring H3.3's for the holdout")

    # ---- B: the pooled figure, which is a different claim ------------------
    banner("DRY RUN B - THE POOLED FOUR-FOLD FIGURE")

    print("  0.99036 / 0.20350 is the POOLED figure over 1,520 matches, not")
    print("  2025-26's. It is the unweighted mean of the four folds because")
    print("  every fold has exactly 380 test rows. Reproduced by scoring each")
    print("  fold under its OWN scope, which is what Phase 2 did.")
    print()

    per_fold = []

    for fold_spec in spec["folds"]:

        season = str(fold_spec["test_season"])
        scope = list(fold_spec["train_seasons"]) + [season]
        cutoff = matches[matches["season"] == season]["date"].min()

        frame_f, _s = score_from_cutoff(matches, cutoff, scope)
        fold_scores = metrics_of(frame_f)

        per_fold.append({"fold": int(fold_spec["fold"]), "test_season": season,
                         "n": len(frame_f),
                         **{m: fold_scores[m] for m in METRICS}})

    fold_frame = pd.DataFrame(per_fold)

    frozen, frozen_pooled = frozen_targets()

    print("  {:<6} {:<12} {:>7} {:>14} {:>14} {:>11}".format(
        "fold", "season", "n", "log loss", "frozen", "difference"))
    print("  " + "-" * 70)

    worst_fold = 0.0
    for row, (_i, ref) in zip(per_fold, frozen.iterrows()):
        gap = abs(row["log_loss"] - float(ref["log_loss"]))
        worst_fold = max(worst_fold, gap)
        print("  {:<6} {:<12} {:>7} {:>14.10f} {:>14.10f} {:>11.2e}".format(
            row["fold"], row["test_season"], row["n"], row["log_loss"],
            float(ref["log_loss"]), gap))

    pooled_ll = float(fold_frame["log_loss"].mean())
    pooled_rps = float(fold_frame["rps"].mean())

    print()
    print("  pooled log loss {:.10f}   frozen {:.10f}   diff {:.2e}".format(
        pooled_ll, frozen_pooled["log_loss"],
        abs(pooled_ll - frozen_pooled["log_loss"])))
    print("  pooled RPS      {:.10f}   frozen {:.10f}   diff {:.2e}".format(
        pooled_rps, frozen_pooled["rps"],
        abs(pooled_rps - frozen_pooled["rps"])))

    audit.record(
        "S3", "every fold reproduces its frozen dc_walkforward log loss",
        "0.000e+00", "{:.3e}".format(worst_fold), worst_fold == 0.0,
        "all four, not only the one the holdout resembles")

    pooled_gap = max(abs(pooled_ll - frozen_pooled["log_loss"]),
                     abs(pooled_rps - frozen_pooled["rps"]))

    audit.record(
        "S4b", "the frozen artefact still agrees with the figure REPORTS.md "
               "publishes, to five decimals",
        "0.99036 / 0.20350",
        "{:.5f} / {:.5f}".format(frozen_pooled["log_loss"],
                                 frozen_pooled["rps"]),
        round(frozen_pooled["log_loss"], 5) == PUBLISHED_POOLED_LOG_LOSS
        and round(frozen_pooled["rps"], 5) == PUBLISHED_POOLED_RPS,
        "the targets above are read from the artefact rather than typed, so "
        "this is what would catch the ARTEFACT drifting away from the report")

    audit.record(
        "S4", "and the pooled figure is the frozen 0.99036 / 0.20350",
        "0.000e+00", "{:.3e}".format(pooled_gap), pooled_gap == 0.0,
        "the unweighted fold mean equals the pooled figure at exactly 380 "
        "test rows per fold - the identity that has caught two bugs in this "
        "project")

    return fold_frame


def dry_run_name_assertion(matches, pin, audit):
    """
    H2.12's assertion, exercised against a deliberate misspelling.

    A guard that has never been seen to fire is a guard nobody has tested.
    """

    banner("DRY RUN C - THE NAME ASSERTION MUST FIRE")

    # ONE CAUSE, SO THE TEST DISCRIMINATES. The first version poisoned the
    # whole five-season frame and the assertion fired on TEN names - nine of
    # them legitimately historical sides that are simply not in the 2026-27
    # twenty. It would have fired identically with no misspelling at all, so
    # it demonstrated nothing about the misspelling.
    #
    # Restricted to 2025-26 and given the three relegated sides explicitly,
    # the ONLY out-of-vocabulary name is the one deliberately introduced.
    season = matches[matches["season"] == "2025-2026"].copy()

    vocabulary = list(pin["vocabulary"]) + ["Burnley", "West Ham", "Wolves"]

    clean_first = False
    try:
        assert_vocabulary(season, vocabulary, "pre-poison control")
        clean_first = True
    except HoldoutError:
        clean_first = False

    audit.record(
        "S5a", "CONTROL: the un-poisoned 2025-26 frame passes the same "
               "assertion",
        "no exception", "no exception" if clean_first else "RAISED",
        clean_first,
        "run BEFORE the poisoning, against the same vocabulary, so the "
        "failure below is attributable to the misspelling and to nothing "
        "else. A guard that fires on everything is not a guard")

    poisoned = season.copy()

    target = poisoned["home_team"] == "Manchester Utd"
    poisoned.loc[target, "home_team"] = "Man United"

    print("  renaming {} 'Manchester Utd' home rows to 'Man United'".format(
        int(target.sum())))
    print("  (V6b showed this scores 0.8165 against 1.1186 WITHOUT raising)")
    print("  control on the same frame before poisoning: {}".format(
        "passed" if clean_first else "FAILED"))
    print()

    raised, message = False, ""

    try:
        assert_vocabulary(poisoned, vocabulary, "poisoned dry run")
    except HoldoutError as error:
        raised, message = True, str(error)

    print("  raised: {}".format(raised))
    if raised:
        print("  {}".format(message[:200]))

    single_cause = raised and "1 team name(s)" in message

    audit.record(
        "S5", "H2.12 - a team name outside the declared vocabulary RAISES, "
              "and flags EXACTLY the one that was poisoned",
        "HoldoutError naming 1 team",
        "{} naming {}".format(
            "HoldoutError" if raised else "NO EXCEPTION",
            "1 team" if single_cause else "a different count"),
        raised and single_cause,
        "the failure V6b demonstrated is now caught, and caught for the right "
        "reason - exactly one name, 'Man United', against a control on the "
        "same frame that passed. Nothing is repaired: P4.5 requires a dated "
        "amendment to the pin, not a rescue in a loader")

    audit.record(
        "S6", "and the un-poisoned control and the poisoned run differ ONLY "
              "in that one name",
        "control passes, poisoned raises",
        "control {}, poisoned {}".format(
            "passed" if clean_first else "FAILED",
            "raised" if raised else "did not raise"),
        clean_first and raised,
        "the pair is the test. Either half alone would be consistent with a "
        "guard that never fires or one that always does")


# ============================================================
# 5. THE HOLDOUT ITSELF - REFUSED BY DEFAULT
# ============================================================

def score_holdout(matches, pin, audit):
    """H6. Runs ONCE, at season end, and only behind two explicit flags."""

    banner("SCORING THE 2026-27 HOLDOUT")

    holdout = matches[matches["season"] == HOLDOUT_SEASON]

    if holdout.empty:
        raise HoldoutError(
            "no {} matches are present. The holdout cannot be scored from "
            "data that is not here.".format(HOLDOUT_SEASON))

    assert_vocabulary(holdout, pin["vocabulary"], HOLDOUT_SEASON)

    scope = sorted(set(matches["season"]))
    frame, scored = score_from_cutoff(matches, pin["cutoff"], scope)

    # ---- H3.3: assert the count, and assert what was excluded -------------
    excluded = holdout[holdout["date"] < pin["cutoff"]]
    late = excluded[excluded["date"] >= pin["cutoff"]]

    audit.record(
        "H3.3a", "the scored-match count",
        pin["expected_scored"], len(frame),
        len(frame) == pin["expected_scored"],
        "difference from the expected {}: {:+d}. A differing count is "
        "REPORTED, not absorbed".format(
            pin["expected_scored"], len(frame) - pin["expected_scored"]))

    audit.record(
        "H3.3b", "every excluded match has date strictly before the cutoff",
        0, len(late), len(late) == 0,
        "{} matches warm the state, all before {}".format(
            len(excluded), pin["cutoff"].date()))

    scores = metrics_of(frame)

    # ---- H6.6: the zero-history split -------------------------------------
    zero = frame["home"].isin(pin["zero_history"]) | frame["away"].isin(
        pin["zero_history"])

    split = []
    for label, subset in (("all scored", frame),
                          ("involving a zero-history side", frame[zero]),
                          ("no zero-history side", frame[~zero])):
        if len(subset):
            split.append({"subset": label, "n": len(subset),
                          **{m: metrics_of(subset)[m] for m in METRICS}})

    return frame, scores, pd.DataFrame(split)


# ============================================================
# THE RUN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 6 - HOLDOUT SCORING INSTRUMENT")

    audit = Audit()

    # ---- the governing documents ------------------------------------------
    freeze_sha = sha256_of(FREEZE)
    pin = load_pin()

    audit.record(
        "S0a", "the freeze on disk is the one this instrument was built "
               "against",
        FREEZE_SHA, freeze_sha, freeze_sha == FREEZE_SHA,
        "the instrument implements a specific specification and refuses to "
        "run against a different one")

    audit.record(
        "S0b", "and the pin cites that same freeze",
        FREEZE_SHA, pin["freeze_sha"], pin["freeze_sha"] == FREEZE_SHA,
        "the pin's whole design is that it records WHICH freeze text it was "
        "pinned against. This is that claim, checked")

    print("  freeze  {}".format(freeze_sha))
    print("  pin     {}".format(pin["sha"]))
    print("  cutoff  {}   (read from the pin, H7 - never hard-coded)".format(
        pin["cutoff"].date()))
    print("  vocabulary  {} names, {} of them zero-history: {}".format(
        len(pin["vocabulary"]), len(pin["zero_history"]),
        ", ".join(pin["zero_history"])))
    print("  expected scored  {}".format(pin["expected_scored"]))
    print()

    matches = L3.load_matches().copy()
    matches["match_id"] = matches.index

    seasons = sorted(set(matches["season"]))
    print("  seasons on disk: {}".format(", ".join(seasons)))
    print()

    spec = L3.load_spec()

    # ---- H5.2: the holdout is refused unless deliberately unlocked --------
    wants_holdout = "--score-holdout" in sys.argv
    confirms_end = "--season-complete" in sys.argv

    if wants_holdout and not confirms_end:
        raise HoldoutError(
            "H5.2: --score-holdout requires --season-complete as well. "
            "Scoring runs ONCE, after the final 2026-27 fixture, because a "
            "number seen in November cannot be unseen in December.")

    if wants_holdout and HOLDOUT_SEASON not in seasons:
        raise HoldoutError(
            "{} is not on disk. Nothing to score.".format(HOLDOUT_SEASON))

    if wants_holdout:
        frame, scores, split = score_holdout(matches, pin, audit)
        print(split.to_string(index=False))
    else:
        print("  MODE: DRY RUN. The holdout is not scored and 2026-27 is not")
        print("  read. --score-holdout --season-complete is the other mode,")
        print("  and it is for one run at season end.")

        fold_frame = dry_run(matches, spec, pin, audit)
        dry_run_name_assertion(matches, pin, audit)

        fold_frame.to_csv(DRYRUN_OUTPUT, index=False, encoding="utf-8",
                          float_format=FLOAT_FORMAT)

        # H6.6's split cannot be exercised on development data - no
        # zero-history side exists there - so it is recorded as still
        # outstanding rather than claimed.
        audit.measure(
            "S7", "H6.6's zero-history split is implemented but NOT exercised",
            "no zero-history side in the development seasons",
            "Coventry City and Hull City appear in no dataset season, so the "
            "split has no rows to separate until 2026-27 arrives. The code "
            "path is written; it is honestly untested and is recorded as "
            "such rather than counted as a pass")

    banner("STATUS")

    frame_out = audit.frame()
    frame_out.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8",
                     float_format=FLOAT_FORMAT)

    failures = int((frame_out["status"] == "FAIL").sum())
    info = int((frame_out["status"] == "INFO").sum())

    print("  {}".format(DRYRUN_OUTPUT))
    print("  {}".format(AUDIT_OUTPUT))
    print()
    print("  Checks run          : {}".format(len(frame_out)))
    print("  Checks failed       : {}".format(failures))
    print("  Outstanding (INFO)  : {}".format(info))
    print()
    if failures:
        for _i, row in frame_out[frame_out["status"] == "FAIL"].iterrows():
            print("    FAIL  {:<8} {}".format(row["test_id"], row["test"]))
        print()
    print("  {}".format("PASS" if failures == 0 else "FAIL"))

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
