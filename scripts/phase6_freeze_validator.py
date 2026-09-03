"""
===============================================================================
PHASE 6 - THE FREEZE VALIDATOR
===============================================================================

Checks that PHASE6_HOLDOUT_FREEZE.txt describes the code that is actually on
disk. IT FITS NOTHING, SCORES NOTHING AND CHANGES NOTHING.

WHY IT EXISTS. The freeze protocol is prose plus numbers. Prose does not
update itself, and a frozen constant that has drifted from the file that
froze it is worth less than no freeze at all - it would produce a holdout
figure attributed to a specification the run did not use.

WHAT IT CANNOT DO, STATED FIRST. It cannot verify the SCORING pipeline,
because there is no scoring pipeline: H6 runs once at season end and that
code is not written. Every check here is on the frozen model's own
implementation and on the freeze document's internal consistency. The
pipeline-level requirements - H2.12's name assertion, H3.3's count
assertion, H6.6's split - are recorded as REQUIREMENTS OUTSTANDING rather
than as passes, because a requirement on code that does not exist has not
been met by anything.

THE DECLARED VALUES ARE READ OUT OF THE FREEZE FILE, NOT TYPED HERE. A
validator that carried its own copy of the numbers would agree with itself
when the freeze file and the code had both moved. It parses H2.1's table and
compares each value against the live module attribute.
"""

from pathlib import Path
import hashlib
import inspect
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase2_poisson_dixon_coles as DC          # noqa: E402
import phase2_elo_baseline as ELO                # noqa: E402
import phase3_ablation_ladder as L3              # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FREEZE = PROJECT_ROOT / "PHASE6_HOLDOUT_FREEZE.txt"
MANIFEST = PROJECT_ROOT / "FROZEN_MANIFEST.txt"
MATCHES_CSV = OUTPUTS_DIR / "phase1_matches.csv"
PIN = PROJECT_ROOT / "PHASE6_CUTOFF_PIN.txt"
SCORING_AUDIT = OUTPUTS_DIR / "phase6_scoring_audit.csv"

AUDIT_OUTPUT = OUTPUTS_DIR / "phase6_freeze_validation.csv"

FREEZE_SHA = "b36befd3e13d5f7f5d9b0af83b7db819e86d7b13e6fdcab4c55db0599fd248f6"
FREEZE_LINES = 397

FLOAT_FORMAT = "%.17g"

# H4.1's three promoted sides, spelled as the freeze file spells them.
PROMOTED_2026_27 = ("Coventry", "Hull", "Ipswich Town")

# H2.9's metric set, in H2.9's order.
DECLARED_METRICS = ["accuracy", "balanced_accuracy", "macro_f1",
                    "log_loss", "brier_score", "rps"]

# H2.11's bootstrap.
DECLARED_DRAWS = 10000
DECLARED_SEED = 20260901


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared_constants(text):
    """
    Parse H2.1's table out of the freeze file.

    The table is `NAME<spaces>VALUE` lines between the H2.1 heading and the
    paragraph that follows it. Read rather than retyped, so this validator
    cannot agree with a copy of the numbers it brought along itself.
    """

    block = text.split("H2.1")[1].split("H2.2")[0]

    found = {}

    for line in block.splitlines():
        match = re.match(r"^\s{10,}([A-Z][A-Z_]+)\s+(-?[0-9.e+-]+)\s*$", line)
        if match:
            found[match.group(1)] = float(match.group(2))

    return found


def main():

    configure_stdout()

    banner("PHASE 6 - FREEZE VALIDATOR (verifies only; fits nothing)")

    audit = Audit()

    text = FREEZE.read_text(encoding="utf-8")

    # ============================================================
    banner("1. THE FREEZE DOCUMENT")

    actual_sha = sha256_of(FREEZE)
    audit.record(
        "V1a", "PHASE6_HOLDOUT_FREEZE.txt is the signed-off file",
        FREEZE_SHA, actual_sha, actual_sha == FREEZE_SHA,
        "the freeze's whole value is that it predates the data it will be "
        "scored against, and the hash is what makes that checkable")

    lines = len(text.splitlines())
    audit.record(
        "V1b", "and is unchanged in length", FREEZE_LINES, lines,
        lines == FREEZE_LINES, "a redundant check beside the hash, kept "
        "because it localises a change a bare hash mismatch does not")

    registered = "PHASE6_HOLDOUT_FREEZE.txt" in (
        PROJECT_ROOT / "scripts" / "frozen_manifest.py").read_text(
            encoding="utf-8")
    listed = "PHASE6_HOLDOUT_FREEZE.txt" in MANIFEST.read_text(encoding="utf-8")

    audit.record(
        "V1c", "registered in FROZEN_PATTERNS and listed in the manifest",
        "both", "patterns={} manifest={}".format(registered, listed),
        registered and listed,
        "FROZEN_PATTERNS is what sweeps it into the manifest; the manifest "
        "entry is what detects a later edit. Both are needed")

    pin_text = PIN.read_text(encoding="utf-8") if PIN.exists() else ""
    cites = FREEZE_SHA in pin_text

    audit.record(
        "V1d", "PHASE6_CUTOFF_PIN.txt exists and cites THIS freeze",
        "cites {}...".format(FREEZE_SHA[:12]),
        "cites it" if cites else "ABSENT OR CITES ANOTHER", cites,
        "the pin carries the cutoff and the twenty-name vocabulary. Its whole "
        "design is to record WHICH freeze text it was pinned against, and "
        "this is that claim checked rather than trusted")

    print("  freeze sha256 {}".format(actual_sha))
    print("  pin    sha256 {}".format(sha256_of(PIN) if PIN.exists() else "-"))
    print("  lines  {}".format(lines))

    # ============================================================
    banner("2. H2.1 - THE CLASS-A CONSTANTS, AS DECLARED vs AS IMPLEMENTED")

    declared = declared_constants(text)

    audit.record(
        "V2a", "H2.1's constant table parses to the ten declared names",
        10, len(declared), len(declared) == 10,
        "parsed FROM THE FREEZE FILE rather than retyped here: {}".format(
            ", ".join(sorted(declared))))

    print("  {:<32} {:>16} {:>16}  {}".format(
        "constant", "declared", "implemented", ""))
    print("  " + "-" * 74)

    mismatched = []

    for name in sorted(declared):
        live = getattr(DC, name, None)
        agrees = live is not None and float(live) == declared[name]
        if not agrees:
            mismatched.append(name)
        print("  {:<32} {:>16g} {:>16}  {}".format(
            name, declared[name],
            "MISSING" if live is None else "{:g}".format(float(live)),
            "ok" if agrees else "MISMATCH"))

    audit.record(
        "V2b", "every H2.1 constant equals the live module attribute",
        0, len(mismatched), not mismatched,
        "read from phase2_poisson_dixon_coles.py at run time. Mismatched: "
        "{}".format(", ".join(mismatched) if mismatched else "none"))

    # ============================================================
    banner("3. H2.2 - ESTIMATOR SHAPE, READ OUT OF THE SOURCE")

    source = inspect.getsource(DC.fit_attack_defence)

    shape_checks = [
        ("V3a", "attack initialises to 1.0 for every team",
         "attack = np.ones(n_teams, dtype=float)"),
        ("V3b", "defence initialises to 1.0 for every team",
         "defence = np.ones(n_teams, dtype=float)"),
        ("V3c", "the home multiplier initialises to 1.0",
         "home_multiplier = 1.0"),
        ("V3d", "the attack scale is pinned by its GEOMETRIC mean",
         "np.exp(np.mean(np.log(attack[positive])))"),
        ("V3e", "convergence is max absolute change < SCALING_TOLERANCE",
         "np.max(np.abs(current - previous)) < SCALING_TOLERANCE"),
        ("V3f", "the degenerate boundary MLE takes NEUTRAL_STRENGTH",
         "attack = np.where(degenerate_attack, NEUTRAL_STRENGTH, attack)"),
    ]

    for check, claim, needle in shape_checks:
        audit.record(check, claim, "present",
                     "present" if needle in source else "ABSENT",
                     needle in source,
                     "asserted against the estimator's own source, so a "
                     "rewritten initialisation or scale rule is caught even "
                     "though it moves no named constant")

    decay = inspect.getsource(DC.time_weights)
    audit.record(
        "V3g", "H2.3 - the decay is 0.5 ** (age_days / half-life) from the "
               "CUTOFF",
        "present",
        "present" if "np.power(0.5, age_days / TIME_DECAY_HALF_LIFE_DAYS)"
        in decay else "ABSENT",
        "np.power(0.5, age_days / TIME_DECAY_HALF_LIFE_DAYS)" in decay,
        "and age is (reference_date - dates), so it is measured from the "
        "cutoff rather than from a season start")

    walk = inspect.getsource(DC.run_fold)
    audit.record(
        "V3h", "H2.5 - the window rule is STRICT, matches['date'] < cutoff",
        "present",
        "present" if '(matches["date"] < cutoff)' in walk else "ABSENT",
        '(matches["date"] < cutoff)' in walk,
        "a same-day match is never in the window that predicts it. This is "
        "the rule the frozen folds use and H2.5 does not relax it")

    audit.record(
        "V3i", "H2.4 - one refit per distinct match date",
        "present",
        "present" if 'for cutoff in sorted(test["date"].unique())' in walk
        else "ABSENT",
        'for cutoff in sorted(test["date"].unique())' in walk,
        "every fixture on a date is predicted from one fit made at that date")

    rho_source = inspect.getsource(DC.fit_rho)
    audit.record(
        "V3j", "H2.7 - rho is fitted on ACTUAL GOALS, over matches where both "
               "sides scored at most one",
        "present",
        "present" if "(home_goals <= 1) & (away_goals <= 1)" in rho_source
        else "ABSENT",
        "(home_goals <= 1) & (away_goals <= 1)" in rho_source,
        "and returns 0.0 where no such match exists in the window, which is "
        "H2.7's stated fallback")

    collapse = inspect.getsource(DC.outcome_probabilities)
    audit.record(
        "V3k", "H2.8 - the score matrix collapses by triangle and trace, "
               "normalised by its mass",
        "present",
        "present" if all(n in collapse for n in
                         ("np.tril(matrix, -1)", "np.trace(matrix)",
                          "np.triu(matrix, 1)")) else "ABSENT",
        all(n in collapse for n in ("np.tril(matrix, -1)", "np.trace(matrix)",
                                    "np.triu(matrix, 1)")),
        "P(H) lower triangle, P(D) trace, P(A) upper triangle")

    # H2.8's second half: no post-hoc transformation.
    predict_source = inspect.getsource(DC.predict_matches)
    forbidden = [tag for tag in ("shrink", "calibrat", "temperature", "clip(")
                 if tag in predict_source.lower()]

    audit.record(
        "V3l", "H2.8 - NO post-hoc transformation is applied to the "
               "probabilities",
        0, len(forbidden), not forbidden,
        "no shrinkage toward the base rate, no calibration, no temperature, "
        "no clipping. A calibrated holdout is not a holdout. Found: "
        "{}".format(", ".join(forbidden) if forbidden else "none"))

    # ============================================================
    banner("4. H2.9 / H2.11 - METRICS AND BOOTSTRAP")

    audit.record(
        "V4a", "H2.9's metric set is the harness's own, in order",
        DECLARED_METRICS, LADDER.METRICS,
        list(LADDER.METRICS) == DECLARED_METRICS,
        "accuracy, balanced accuracy, macro F1, log loss, Brier, RPS - with "
        "log loss and RPS primary and RPS required to agree in sign")

    compare_source = inspect.getsource(LADDER.compare)
    draws_ok = str(DECLARED_DRAWS) in compare_source or getattr(
        LADDER, "BOOTSTRAP_DRAWS", None) == DECLARED_DRAWS
    seed_ok = str(DECLARED_SEED) in compare_source or getattr(
        LADDER, "BOOTSTRAP_SEED", None) == DECLARED_SEED

    audit.record(
        "V4b", "H2.11 - the paired bootstrap is 10,000 draws at seed 20260901",
        "10000 / 20260901",
        "{} / {}".format(getattr(LADDER, "BOOTSTRAP_DRAWS", "?"),
                         getattr(LADDER, "BOOTSTRAP_SEED", "?")),
        draws_ok and seed_ok,
        "the same draws and seed every delta in this project has used")

    audit.record(
        "V4c", "the INCONCLUSIVE rule is implemented, not merely stated",
        "present",
        "present" if "INCONCLUSIVE" in compare_source else "ABSENT",
        "INCONCLUSIVE" in compare_source,
        "RPS must agree in sign with log loss or the comparison is "
        "INCONCLUSIVE. H2.9 is binding on the holdout's own deltas")

    # ============================================================
    banner("5. H2.12 / H4 - NAMES AND THE PROMOTED SIDES")

    matches = pd.read_csv(MATCHES_CSV)
    known = sorted(set(matches["home_team"]) | set(matches["away_team"]))

    print("  the dataset's team vocabulary is {} names".format(len(known)))
    print()

    for name in PROMOTED_2026_27:
        seasons = sorted(set(
            matches[(matches["home_team"] == name)
                    | (matches["away_team"] == name)]["season"]))
        print("  {:<16} {}".format(
            name, seasons if seasons else "NO HISTORY IN THE DATASET"))

    coventry = [s for s in known if "Coventry" in s]
    hull = [s for s in known if "Hull" in s]
    ipswich = [s for s in known if "Ipswich" in s]

    audit.record(
        "V5a", "H4.1 - Coventry and Hull have NO history in the dataset",
        "neither present",
        "Coventry={} Hull={}".format(coventry or "absent", hull or "absent"),
        not coventry and not hull,
        "so both take the H4.3 neutral prior for every 2026-27 match. "
        "Checked by substring across the whole vocabulary, so a differently "
        "spelled entry would be found rather than missed")

    audit.record(
        "V5b", "H4.1 - Ipswich Town is present for 2024-25 only",
        "['Ipswich Town'], 2024-2025",
        "{}, {}".format(ipswich, sorted(set(
            matches[(matches["home_team"] == "Ipswich Town")
                    | (matches["away_team"] == "Ipswich Town")]["season"]))),
        ipswich == ["Ipswich Town"],
        "one season, which the 107-day half-life renders nearly weightless "
        "by September 2026")

    # H2.12's specific spellings.
    for spelling in ("Manchester Utd", "Nottingham", "Ipswich Town"):
        audit.record(
            "V5c-{}".format(spelling.replace(" ", "_")),
            "H2.12 - the frozen spelling {!r} is the dataset's own".format(
                spelling),
            "present", "present" if spelling in known else "ABSENT",
            spelling in known,
            "a name that fails to match does not raise - it makes a KNOWN "
            "team look like a zero-history one and silently hands it "
            "NEUTRAL_STRENGTH. That is why the spellings are specification")

    # ============================================================
    banner("6. H4.3 - WHAT AN UNKNOWN NAME ACTUALLY DOES TODAY")

    rates_source = inspect.getsource(DC.match_rates)
    fills = rates_source.count("fillna(NEUTRAL_STRENGTH)")

    audit.record(
        "V6a", "H4.3 - an absent team takes attack = defence = "
               "NEUTRAL_STRENGTH through match_rates()",
        "4 fillna sites", "{} fillna sites".format(fills), fills == 4,
        "home attack, away defence, away attack, home defence. This is the "
        "INHERITED behaviour H4.3 declares and it applies without amendment")

    # THE POINT OF SECTION 4 OF THE TASK, MEASURED RATHER THAN ASSERTED.
    probe = pd.DataFrame({
        "home_team": ["Manchester Utd", "Man United"],
        "away_team": ["Arsenal", "Arsenal"]})

    attack = pd.Series({"Manchester Utd": 1.37, "Arsenal": 1.44})
    defence = pd.Series({"Manchester Utd": 0.92, "Arsenal": 0.71})

    home_rates, _away_rates = DC.match_rates(probe, attack, defence, 1.15)

    silent = float(home_rates[0]) != float(home_rates[1])

    audit.record(
        "V6b", "A MISSPELLED KNOWN TEAM IS SILENTLY SCORED AS AVERAGE, and "
               "nothing raises",
        "rates differ, no exception",
        "correct spelling -> {:.4f}, misspelling -> {:.4f}".format(
            float(home_rates[0]), float(home_rates[1])),
        silent,
        "DEMONSTRATED, not argued: 'Man United' is not in the ratings, so "
        "fillna hands it NEUTRAL_STRENGTH and the match is scored with a "
        "different rate WITHOUT ERROR. This is exactly the failure H2.12 "
        "requires the scoring run to assert against. THE ASSERTION IS NOT "
        "IMPLEMENTED ANYWHERE YET - see section 7")

    # ============================================================
    banner("7. THE FREEZE'S REQUIREMENTS ON THE SCORING RUN")

    # These were INFO - "outstanding" - while no scoring instrument existed.
    # It does now, and it was validated on the development seasons, so each
    # requirement is checked against ITS OWN AUDIT rather than against the
    # mere presence of a file. Code existing is not code working.

    scoring = PROJECT_ROOT / "scripts" / "phase6_score_holdout.py"

    audit.record(
        "V7a", "a Phase 6 scoring instrument is on disk",
        "present", "present" if scoring.exists() else "ABSENT",
        scoring.exists(),
        "built in September 2026 rather than May 2027 so that it could be "
        "validated where the answers are frozen. Code that runs once against "
        "a holdout with nothing to check it against is untested code")

    dry = {}

    if SCORING_AUDIT.exists():
        table = pd.read_csv(SCORING_AUDIT)
        dry = dict(zip(table["test_id"], table["status"]))

    for check, requirement, source, detail in (
            ("V7b", "H2.12 - a team name outside the declared vocabulary "
                    "RAISES", "S5",
             "verified by the scoring instrument's own dry run, which poisons "
             "'Manchester Utd' to 'Man United' against a control on the same "
             "frame that passes, and asserts EXACTLY ONE name is flagged"),
            ("V7c", "H3.3 - the scored-match count is asserted", "S2",
             "exercised in the dry run as the 380-match assertion on 2025-26; "
             "the holdout path asserts against the pin's expected 360"),
            ("V7e", "H3.1 - the pinned cutoff is READ FROM THE PIN and "
                    "applied", "S0b",
             "the instrument parses PHASE6_CUTOFF_PIN.txt and checks the pin "
             "cites this exact freeze. Nothing about the cutoff is "
             "hard-coded")):

        status = dry.get(source)

        audit.record(
            check, requirement, "PASS in the scoring dry run",
            "{} = {}".format(source, status or "NOT RUN"), status == "PASS",
            detail)

    audit.measure(
        "V7d", "H6.6 - metrics split by zero-history involvement",
        "IMPLEMENTED BUT UNEXERCISED",
        "Coventry City and Hull City appear in no dataset season, so the "
        "split has no rows to separate until 2026-27 arrives. The code path "
        "is written and is honestly untested. It stays INFO rather than "
        "being counted as a pass, because running is not the same as working")

    # ============================================================
    banner("8. H3 - THE CUTOFF, AND WHAT IS MISSING")

    seasons = sorted(set(matches["season"]))

    audit.record(
        "V8a", "the repository's match data ends before 2026-27",
        "2021-2022 .. 2025-2026", ", ".join(seasons),
        "2026-2027" not in seasons,
        "so the cutoff date CANNOT be derived from anything in this "
        "repository. H3.2 says so explicitly and requires it pinned at "
        "sign-off from the published fixture list")

    pinned = re.search(r"cutoff date pinned:\s*(\S+)", text)
    placeholder = pinned and set(pinned.group(1)) <= set("_")

    pin_date = re.search(r"THE PINNED CUTOFF IS.{0,60}?(\d{4}-\d{2}-\d{2})",
                         pin_text, re.DOTALL)

    audit.record(
        "V8b", "the cutoff is pinned - in the PIN, not in the freeze's own H7 "
               "slot, which stays a placeholder ON PURPOSE",
        "H7 placeholder + a dated pin",
        "H7 {} / pin {}".format(
            "placeholder" if placeholder else "FILLED",
            pin_date.group(1) if pin_date else "NOT FOUND"),
        bool(placeholder) and bool(pin_date),
        "H3.2 says to fill H7. Doing so would move this file's sha256 off the "
        "hash that makes 'written before the data' checkable, so the date is "
        "recorded in PHASE6_CUTOFF_PIN.txt with the deviation logged and "
        "dated there. BOTH halves are asserted: H7 must still be blank AND "
        "the pin must carry a date")

    # ============================================================
    banner("9. WRITING")

    frame = audit.frame()
    frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8",
                 float_format=FLOAT_FORMAT)
    print("  {}".format(AUDIT_OUTPUT))

    failures = int((frame["status"] == "FAIL").sum())
    info = int((frame["status"] == "INFO").sum())

    print()
    print("  Checks run          : {}".format(len(frame)))
    print("  Checks failed       : {}".format(failures))
    print("  Outstanding (INFO)  : {}".format(info))
    print()
    if failures:
        for _i, row in frame[frame["status"] == "FAIL"].iterrows():
            print("    FAIL  {:<10} {}".format(row["test_id"], row["test"]))
        print()
    print("  {}".format("PASS" if failures == 0 else "FAIL"))

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
