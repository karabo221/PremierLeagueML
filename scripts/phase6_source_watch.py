"""
===============================================================================
PHASE 6 - THE SOURCE-INTEGRITY WATCH
===============================================================================

RUNS MONTHLY, FROM SEPTEMBER 2026 UNTIL THE 2026-27 SEASON ENDS.

WHAT IT IS FOR. The holdout is scored once, in May 2027. If between now and
then the source renames a column, drops a bookmaker, or spells a promoted club
differently from PHASE6_CUTOFF_PIN.txt, the scoring run fails - in May, with
the season over and nothing that can be done about it. This instrument asks
the only question that can be asked safely in the meantime:

    WILL THE 2026-27 DATA ARRIVE IN THE SHAPE THE SCORING INSTRUMENT EXPECTS?

WHAT IT MUST NOT DO, AND THE CONSTRAINT IS STRUCTURAL RATHER THAN PROMISED.
It fits nothing, scores nothing, and reports no result - no probability, no
metric, no gap, not a single scoreline. H5.2 forbids looking, not only
changing, because a number seen in November cannot be unseen in December.

    IF IT CAN READ A SCORELINE IT CAN LEAK.

So it does not read one. Column PRESENCE is checked on the header alone
(nrows=0), and the only columns ever loaded as VALUES are

    Div, Date, HomeTeam, AwayTeam

declared in SAFE_COLUMNS and passed to read_csv as usecols, with FORBIDDEN
asserted absent from the loaded frame afterwards. FTHG, FTAG, FTR, HTHG,
HTAG, HTR, HxG, AxG and every price column are never in memory. That is a
checkable property of this file, not an undertaking by its author.

THE ASSERTIONS ARE ON SCHEMA AND NAMES, NOT ON VALUES. A row count and a
match-date block structure are schedule facts, available from any fixture
list, and carry no outcome.

WHERE ITS CONSTANTS COME FROM. Nothing here is retyped. The vocabulary and
the cutoff are read from the pin THROUGH THE SCORING INSTRUMENT'S OWN PARSER,
so this watch checks the names that run will actually assert on. The twelve
match-detail columns come from E1a and the book prefixes from Phase 5B, by
import. A copy would drift.

THE SNAPSHOT IS NOT A SOURCE. The fetched file lands in data/watch/, which is
gitignored and which no scoring code reads. data/raw/ is untouched: acquiring
the real E0_2627.csv into the frozen source tree is a separate, deliberate act
at season end, and a re-download that moves a hash is a NEW SOURCE.

    ./venv/Scripts/python.exe -B scripts/phase6_source_watch.py
        fetch the live file and check it

    ... --offline
        check the newest snapshot already in data/watch/, no network

DISCLOSURE, in the project's own style. MATCHWEEK_GAP_DAYS and ROW_TOLERANCE
below were fixed AFTER the first snapshot's dates were seen, in the same
session that wrote this file. They are structural - a matchweek is a block of
dates - and they place no threshold on any result. Recorded here rather than
presented as a pre-declaration, because they are not one.
===============================================================================
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase5_e1a_sot_ratings as E1A                # noqa: E402
import phase5_market_benchmark as MKT               # noqa: E402
import phase6_score_holdout as H6                   # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WATCH_DIR = PROJECT_ROOT / "data" / "watch"
RAW_ODDS_DIR = PROJECT_ROOT / "data" / "raw" / "Odds"
FIXTURES_DIR = PROJECT_ROOT / "data" / "raw" / "Fixtures"

SOURCE_URL = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
SNAPSHOT_STEM = "E0_2627"

# The last frozen season, for the header diff. Its hash is in the manifest, so
# it is a fixed point to compare against rather than another moving file.
REFERENCE_ODDS = RAW_ODDS_DIR / "E0_2526.csv"

# The export the pin's section 7 REPLACED as the spine, kept as a row so the
# cross-check it used to provide is not quietly forgotten. See SW10/SW10b.
HOLDOUT_FIXTURES = FIXTURES_DIR / "2026-2027 PL Season.xls"

FLOAT_FORMAT = "%.17g"

EXPECTED_DIV = "E0"
FULL_SEASON_MATCHES = 380
MATCHES_PER_MATCHWEEK = 10
MATCHWEEKS_PER_SEASON = 38
MATCHES_PER_TEAM = 38

# A matchweek is a block of match dates separated from the next by at least
# this many days. See the disclosure in the module docstring.
MATCHWEEK_GAP_DAYS = 4
ROW_TOLERANCE = 2

# ---------------------------------------------------------------------------
# THE ONLY COLUMNS THIS FILE EVER LOADS AS VALUES.
# ---------------------------------------------------------------------------
SAFE_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam"]

# Asserted absent from the loaded frame. Presence of these in the SOURCE is
# checked on the header; presence in MEMORY is a defect.
FORBIDDEN = ["FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "HxG", "AxG"]

# The spine the join and the market cross-check need. Their PRESENCE is
# required; their contents are not read here.
SPINE_COLUMNS = ["Div", "Date", "HomeTeam", "AwayTeam",
                 "FTHG", "FTAG", "FTR"]

DATE_FORMAT = "%d/%m/%Y"


# ============================================================
# THE SNAPSHOT
# ============================================================

def snapshot_path(stamp):
    return WATCH_DIR / "{}_{}.csv".format(SNAPSHOT_STEM, stamp)


def newest_snapshot():

    if not WATCH_DIR.exists():
        return None

    found = sorted(WATCH_DIR.glob("{}_*.csv".format(SNAPSHOT_STEM)))

    return found[-1] if found else None


def fetch(stamp, audit):
    """Fetch the live file into data/watch/. data/raw/ is never written."""

    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    target = snapshot_path(stamp)

    request = urllib.request.Request(
        SOURCE_URL, headers={"User-Agent": "PremierLeagueML-source-watch/1"})

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read()
            modified = response.headers.get("Last-Modified", "unknown")
            status = response.status

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        audit.record(
            "SW1", "the 2026-27 E0 file is obtainable from the source",
            "HTTP 200", "FETCH FAILED: {}".format(exc), False,
            "the holdout's spine, its odds arm and its shot columns ALL come "
            "from this file since the pin's section 7. Unobtainable in "
            "September is a problem with eight months to solve it; "
            "unobtainable in May is not")
        return None

    target.write_bytes(payload)

    audit.record(
        "SW1", "the 2026-27 E0 file is obtainable from the source",
        "HTTP 200", "HTTP {}, {} bytes".format(status, len(payload)),
        status == 200 and len(payload) > 0,
        "{} - source last modified {}. Written to data/watch/, which is "
        "gitignored and which no scoring code reads. data/raw/ is "
        "untouched".format(SOURCE_URL, modified))

    return target


# ============================================================
# THE CHECKS
# ============================================================

def check_parse(path, audit):
    """SW2. The header, read WITHOUT any row. Nothing else is on it."""

    try:
        header = list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:                       # noqa: BLE001
        audit.record("SW2", "the file parses as CSV", "a header row",
                     "PARSE FAILED: {}".format(exc), False,
                     "read with nrows=0, so a parse failure here is the "
                     "header's and not a row's")
        return None

    audit.record(
        "SW2", "the file parses as CSV and carries a header",
        "> 0 columns", "{} columns".format(len(header)), len(header) > 0,
        "the column COUNT is not asserted against a fixed number on purpose: "
        "football-data adds and drops bookmakers between seasons and always "
        "has. What matters is the columns this project names, below")

    return header


def check_columns(header, audit):
    """SW3-SW5. Every column the project actually consumes, by import."""

    # ---- SW3, the twelve match-detail columns, from E1a --------------------
    missing = [c for c in E1A.SHOT_COLUMNS if c not in header]

    audit.record(
        "SW3", "the {} match-detail columns the project uses are present "
               "under the same names".format(len(E1A.SHOT_COLUMNS)),
        "all present",
        "missing: {}".format(", ".join(missing) if missing else "none"),
        not missing,
        "imported from phase5_e1a_sot_ratings.SHOT_COLUMNS rather than "
        "retyped, so this checks the list the code actually uses: {}".format(
            ", ".join(E1A.SHOT_COLUMNS)))

    # ---- SW4, the spine ----------------------------------------------------
    missing_spine = [c for c in SPINE_COLUMNS if c not in header]

    audit.record(
        "SW4", "the spine columns the join and the market cross-check need",
        "all present",
        "missing: {}".format(", ".join(missing_spine)
                             if missing_spine else "none"),
        not missing_spine,
        "PRESENCE only. FTHG, FTAG and FTR are named here and are never "
        "loaded - see FORBIDDEN and SW2b")

    # ---- SW5, the books, from Phase 5B ------------------------------------
    for prefix, label, role, _complete in MKT.BOOKS:

        columns = ["{}{}".format(prefix, s) for s in ("H", "D", "A")]
        absent = [c for c in columns if c not in header]

        primary = role == "PRIMARY"

        audit.record(
            "SW5-{}".format(prefix),
            "{} ({}) - {}".format(label, role, ", ".join(columns)),
            "all present",
            "missing: {}".format(", ".join(absent) if absent else "none"),
            not absent,
            "the PRIMARY book is the one H6.2(c) freezes. Losing it would "
            "cost the holdout its market control, which is the whole point "
            "of scoring the market on the same matches"
            if primary else
            "declared a SENSITIVITY in PHASE5_MARKET_PREDECLARATION.txt. If "
            "it is gone, the sensitivity cannot be reproduced on 2026-27 and "
            "that is reported as a loss of scope, NEVER repaired by "
            "promoting another book into its place")


def check_names(frame, pin, audit):
    """SW6. The twenty names, through the project's own map."""

    agree = MKT.TEAM_MAP == E1A.TEAM_MAP

    audit.record(
        "SW6a", "the project's two copies of TEAM_MAP agree",
        "identical",
        "identical, {} entries".format(len(MKT.TEAM_MAP)) if agree
        else "THEY DIFFER",
        agree,
        "phase5_market_benchmark and phase5_e1a_sot_ratings each carry one. "
        "A watch that checked the wrong copy would pass while the scoring "
        "run failed")

    raw = sorted(set(frame["HomeTeam"]) | set(frame["AwayTeam"]))
    mapped = sorted({MKT.TEAM_MAP.get(name, name) for name in raw})

    unknown = [name for name in mapped if name not in pin["vocabulary"]]

    audit.record(
        "SW6", "every 2026-27 team name maps into the pin's twenty",
        "0 outside the vocabulary",
        "{} outside: {}".format(len(unknown),
                                ", ".join(unknown) if unknown else "none"),
        not unknown,
        "THE MAPPING IS THE PROJECT'S OWN DICTIONARY, applied whole and never "
        "by partial match. A name outside the vocabulary is exactly what "
        "P4.5 says must stop the scoring run, and the fix is a DATED "
        "AMENDMENT to the pin recording the source's actual spelling - never "
        "a normalisation inside a loader. Source names seen: {}".format(
            ", ".join(raw)))

    audit.measure(
        "SW6b", "distinct team names in the source file so far",
        "{} of the pin's {}".format(len(mapped), len(pin["vocabulary"])),
        "fewer than twenty early in a season is not a defect - it means not "
        "every club has appeared yet. Only names that fail to map are")


def matchweek_blocks(dates):
    """Blocks of match dates separated by >= MATCHWEEK_GAP_DAYS."""

    unique = sorted(set(dates))
    blocks = 1 if unique else 0

    for earlier, later in zip(unique, unique[1:]):
        if (later - earlier).days >= MATCHWEEK_GAP_DAYS:
            blocks += 1

    return blocks


def check_counts(frame, audit, today):
    """SW7-SW8. Schedule facts. No outcome is touched."""

    rows = len(frame)

    divisions = sorted(set(frame["Div"]))
    audit.record(
        "SW7a", "every row is the Premier League", EXPECTED_DIV,
        ", ".join(divisions), divisions == [EXPECTED_DIV],
        "a division column carrying anything else means the wrong file")

    detail = ""

    try:
        dates = pd.to_datetime(frame["Date"], format=DATE_FORMAT)
        parsed = True
    except (ValueError, TypeError) as exc:
        dates = None
        parsed = False
        detail = str(exc)

    audit.record(
        "SW7b", "every date parses as {}".format(DATE_FORMAT),
        "all {} rows".format(rows),
        "parsed" if parsed else "FAILED: {}".format(detail[:120]),
        parsed,
        "the format is the one load_odds() uses. A source that switched to "
        "ISO would raise inside the scoring run instead")

    if not parsed:
        return

    audit.record(
        "SW8a", "no match date is in the future",
        "max <= {}".format(today.date()), str(dates.max().date()),
        dates.max().date() <= today.date(),
        "a fixture list with results attached would carry unplayed dates")

    blocks = matchweek_blocks(dates.dt.date)
    expected = blocks * MATCHES_PER_MATCHWEEK

    audit.record(
        "SW8b", "the row count is consistent with the matchweeks played",
        "{} +/- {} ({} matchweek blocks x {})".format(
            expected, ROW_TOLERANCE, blocks, MATCHES_PER_MATCHWEEK),
        "{} rows".format(rows),
        abs(rows - expected) <= ROW_TOLERANCE,
        "a matchweek is a block of match dates separated from the next by at "
        "least {} days, derived from the file's OWN dates rather than from a "
        "calendar this repository does not hold. The tolerance absorbs "
        "postponements; a larger drift means rounds are being split or "
        "merged and the count needs looking at".format(MATCHWEEK_GAP_DAYS))

    audit.record(
        "SW8c", "the season has not overrun its structural bounds",
        "rows <= {} and blocks <= {}".format(
            FULL_SEASON_MATCHES, MATCHWEEKS_PER_SEASON),
        "{} rows, {} blocks".format(rows, blocks),
        rows <= FULL_SEASON_MATCHES and blocks <= MATCHWEEKS_PER_SEASON,
        "the hard bound, as opposed to SW8b's tolerance. Breaching it means "
        "the file is not one Premier League season")

    appearances = pd.concat([frame["HomeTeam"], frame["AwayTeam"]])
    counts = appearances.value_counts()

    audit.record(
        "SW8d", "no club has played more than {} matches".format(
            MATCHES_PER_TEAM),
        "max <= {}".format(MATCHES_PER_TEAM),
        "max {}".format(int(counts.max())),
        int(counts.max()) <= MATCHES_PER_TEAM,
        "a count above 38 means duplicated rows, which is the failure mode "
        "M1a caught on the development files")

    audit.measure(
        "SW8e", "matches played per club",
        "min {}, max {}".format(int(counts.min()), int(counts.max())),
        "a spread wider than a match or two mid-season means postponements, "
        "which move rows across the cutoff under H3.4 and change the 360")


def check_drift(header, audit):
    """SW9. What changed against the last frozen season, whether or not this
    project uses it. The columns a project does NOT use are where a source
    change is first visible."""

    if not REFERENCE_ODDS.exists():
        audit.measure("SW9", "header drift against {}".format(
            REFERENCE_ODDS.name), "REFERENCE MISSING",
            "no comparison was made")
        return

    reference = list(pd.read_csv(REFERENCE_ODDS, nrows=0).columns)

    removed = [c for c in reference if c not in header]
    added = [c for c in header if c not in reference]

    audit.measure(
        "SW9", "columns present in {} and gone from 2026-27".format(
            REFERENCE_ODDS.name),
        "{}: {}".format(len(removed), ", ".join(removed) if removed
                        else "none"),
        "REPORTED, NOT ASSERTED. A dropped column this project does not use "
        "costs nothing; the row exists so that a source narrowing is visible "
        "in September rather than inferred in May. SW5 is the assertion")

    audit.measure(
        "SW9b", "columns new in 2026-27",
        "{}: {}".format(len(added), ", ".join(added) if added else "none"),
        "A NEW COLUMN IS NOT AN INVITATION. H5.1 forbids any feature decision "
        "made because of something observed in 2026-27, and that includes a "
        "column that appears mid-holdout. Recorded for after the holdout is "
        "scored, and not acted on")


def check_spine(audit):
    """SW10. The pin declares which file the 2026-27 spine is built from, and
    this watch is only complete if that file is the one it checks."""

    text = H6.PIN.read_text(encoding="utf-8")

    declared = "THE 2026-27 SPINE IS THE FOOTBALL-DATA.CO.UK E0 FILE" in text

    audit.record(
        "SW10", "the pin declares the E0 file as the 2026-27 spine",
        "declared at P7.2",
        "declared" if declared else "NOT DECLARED",
        declared,
        "AMENDED 2026-09-03. This row used to report a gap: the scoring "
        "instrument read a spine built from an FBref export in "
        "data/raw/Fixtures/, which arrives only at season end and could not "
        "be checked in advance by anything. Section 7 of the pin declares the "
        "E0 file instead, on the strength of M1b/M2a/M2b/M3 of "
        "phase5_market_audit.csv - two independently sourced spines agreeing "
        "on every date and every scoreline across all 1,900 development "
        "matches. Every input the scoring run has is now the file this watch "
        "checks monthly")

    audit.measure(
        "SW10b", "the FBref export the declaration replaces",
        "present" if HOLDOUT_FIXTURES.exists() else "not on disk",
        "kept as a row because P7.6 says what is lost: the development "
        "seasons had TWO sources agreeing, and a single-source spine has no "
        "such cross-check. If {} appears at season end it is to be reconciled "
        "against the E0 spine as a CHECK and never used as the spine - a run "
        "that can silently pick either source is a run whose vocabulary and "
        "row count depend on which one it picked".format(
            HOLDOUT_FIXTURES.name))


def check_cadence(audit, today):
    """SW11. The watch is monthly, so it says when it last ran."""

    previous = sorted(OUTPUTS_DIR.glob("phase6_source_watch_*.csv"))
    stamps = [p.stem.replace("phase6_source_watch_", "") for p in previous]
    earlier = [s for s in stamps if s < today.strftime("%Y-%m-%d")]

    if earlier:
        last = datetime.strptime(earlier[-1], "%Y-%m-%d").date()
        observed = "{}, {} days ago".format(last, (today.date() - last).days)
    else:
        observed = "no earlier run - this is the first"

    audit.measure(
        "SW11", "the previous watch run", observed,
        "monthly is the declared cadence, from the freeze to the final "
        "2026-27 fixture. Each run writes its own dated artefact rather than "
        "overwriting one, so the record is a series of checks at dates and "
        "not a single mutable file")


# ============================================================
# MAIN
# ============================================================

def main(argv):

    configure_stdout()

    offline = "--offline" in argv
    today = datetime.now(timezone.utc)
    stamp = today.strftime("%Y-%m-%d")

    banner("PHASE 6 - SOURCE-INTEGRITY WATCH")
    print("  run              {}".format(stamp))
    print("  mode             {}".format("OFFLINE" if offline else "fetch"))
    print()
    print("  THIS INSTRUMENT SCORES NOTHING. It loads {} as values".format(
        ", ".join(SAFE_COLUMNS)))
    print("  and nothing else. No result, probability, metric or gap is "
          "computed here.")

    audit = Audit()

    # ---- the freeze and the pin -------------------------------------------
    banner("0. THE DOCUMENTS THIS WATCH IS GOVERNED BY")

    on_disk = H6.sha256_of(H6.FREEZE)

    audit.record(
        "SW0a", "the freeze on disk is the one Phase 6 was built against",
        H6.FREEZE_SHA, on_disk, on_disk == H6.FREEZE_SHA,
        "the same assertion the scoring instrument makes at S0a. A watch "
        "governed by a different freeze is watching the wrong thing")

    try:
        pin = H6.load_pin()
    except H6.HoldoutError as exc:
        audit.record("SW0b", "the pin parses", "20 names and a cutoff",
                     "PIN ERROR: {}".format(exc), False,
                     "parsed by phase6_score_holdout.load_pin() itself, so "
                     "this watch cannot check a vocabulary the scoring run "
                     "would not use")
        pin = None
    else:
        audit.record(
            "SW0b", "the pin parses to the twenty names and the cutoff",
            "20 names + a date",
            "{} names, cutoff {}".format(len(pin["vocabulary"]),
                                         pin["cutoff"].date()),
            len(pin["vocabulary"]) == 20,
            "read THROUGH THE SCORING INSTRUMENT'S OWN PARSER, so this watch "
            "checks the names that run will actually assert on")

    # ---- the source --------------------------------------------------------
    banner("1. THE SOURCE FILE")

    if offline:
        path = newest_snapshot()
        audit.record(
            "SW1", "a snapshot is available to check", "a file in data/watch/",
            path.name if path else "NONE", path is not None,
            "--offline was given, so nothing was fetched and this run says "
            "only what the last snapshot said")
    else:
        path = fetch(stamp, audit)

    if path is None:
        return finish(audit, stamp)

    print("  snapshot         {}".format(path.relative_to(PROJECT_ROOT)))

    header = check_parse(path, audit)

    if header is None:
        return finish(audit, stamp)

    # ---- schema ------------------------------------------------------------
    banner("2. THE COLUMNS THE PROJECT CONSUMES")
    check_columns(header, audit)

    # ---- values, and only the four safe ones -------------------------------
    banner("3. NAMES AND COUNTS")

    frame = pd.read_csv(path, usecols=SAFE_COLUMNS)

    leaked = [c for c in FORBIDDEN if c in frame.columns]
    audit.record(
        "SW2b", "no result column is loaded into memory",
        "none of {}".format(", ".join(FORBIDDEN)),
        "loaded: {}".format(", ".join(leaked) if leaked else "none"),
        not leaked,
        "usecols=SAFE_COLUMNS is what makes 'this instrument cannot leak' a "
        "property of the file rather than a promise in its header. Asserted "
        "rather than assumed, because the promise is the part that rots")

    if pin is not None:
        check_names(frame, pin, audit)

    check_counts(frame, audit, today)

    # ---- drift and cover ---------------------------------------------------
    banner("4. DRIFT, COVER AND CADENCE")
    check_drift(header, audit)
    check_spine(audit)
    check_cadence(audit, today)

    return finish(audit, stamp)


def finish(audit, stamp):

    banner("WRITING")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUTS_DIR / "phase6_source_watch_{}.csv".format(stamp)

    frame = audit.frame()
    frame.to_csv(output, index=False, encoding="utf-8",
                 float_format=FLOAT_FORMAT)
    print("  {}".format(output))

    failures = int((frame["status"] == "FAIL").sum())
    info = int((frame["status"] == "INFO").sum())

    print()
    print("  Checks run          : {}".format(len(frame)))
    print("  Checks failed       : {}".format(failures))
    print("  Reported (INFO)     : {}".format(info))
    print()

    if failures:
        for _i, row in frame[frame["status"] == "FAIL"].iterrows():
            print("    FAIL  {:<10} {}".format(row["test_id"], row["test"]))
            print("          {}".format(row["observed"]))
        print()

    print("  {}".format("PASS" if failures == 0 else "FAIL"))
    print()
    print("  NOTHING WAS SCORED. No 2026-27 result was read.")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
