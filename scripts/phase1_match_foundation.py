"""
===============================================================================
PHASE 1 - INSTRUMENT 1
MATCH FOUNDATION
===============================================================================

PURPOSE
    Build and validate the canonical match-level foundation for the whole
    project, from the fixture files ONLY.

    This table is the spine every later phase hangs off. Nothing is trusted
    downstream that is not proven here.

SCOPE - what this instrument does
    - reads the five fixture files
    - derives the canonical match fields
    - derives result and points from the actual score
    - runs the 15 required structural validations per season, plus combined
      and supplementary checks
    - independently reconstructs the season-end league statistics from the
      fixture results alone
    - applies an explicit, declared points-sanction registry

SCOPE - what this instrument deliberately does NOT do
    - it does not read the FBref statistical aggregate files
      (Phase 0 constraint 7: those are END-OF-SEASON information)
    - it does not build rolling features
    - it does not calculate Elo
    - it does not build a modelling dataset
    - it does not train anything
    - it does not compare against FBref; Phase 0 Instrument 4 already
      performed that cross-source reconciliation

READ-ONLY GUARANTEE
    data/raw/ is never renamed, moved, deleted or modified. Every source file
    is SHA-256 hashed before reading and re-hashed after processing; a
    mismatch is a hard FAIL.

TWO TRAPS THIS SCRIPT HANDLES EXPLICITLY

    TRAP 1 - THE SCORE SEPARATOR IS AN EN DASH
        Scores are "2-0" written with U+2013 EN DASH, not a hyphen-minus.
        Splitting on "-" alone silently yields nothing.

    TRAP 2 - THE FILES DECLARE NO CHARACTER SET
        The .xls files are FBref HTML exports carrying no <meta charset>. The
        bytes are UTF-8, but pandas.read_html left to guess decodes them as
        cp1252, and the en dash arrives as the mojibake sequence
        'a<80><93>'. The score then fails to parse for a reason that looks
        like a data problem and is actually an encoding problem.

        This script therefore reads the bytes and decodes UTF-8 explicitly.

        Parsing then REQUIRES a true en dash. There is no hyphen fallback, on
        purpose: a tolerant parser would silently paper over any future
        encoding regression, and catching exactly that is what this
        instrument is for. A score that does not match is reported with its
        raw repr so the cause is diagnosable at a glance.
===============================================================================
"""

from pathlib import Path
import hashlib
import io
import re
import sys

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "raw" / "Fixtures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MATCHES_OUTPUT = OUTPUTS_DIR / "phase1_matches.csv"
SUMMARY_OUTPUT = OUTPUTS_DIR / "phase1_season_team_summary.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase1_match_foundation_audit.csv"


# The season label convention is inherited from Phase 0 so that
# outputs/phase0_evaluation_folds.csv joins to this table without translation.
SEASON_FILES = {
    "2021-2022": "2021-2022 PL Season.xls",
    "2022-2023": "2022-2023 PL Season.xls",
    "2023-2024": "2023-2024 PL Season.xls",
    "2024-2025": "2024-2025 PL Season.xls",
    "2025-2026": "2025-2026 PL Season.xls",
}


EXPECTED_MATCHES_PER_SEASON = 380
EXPECTED_TEAMS_PER_SEASON = 20
EXPECTED_MATCHES_PER_TEAM = 38
EXPECTED_HOME_PER_TEAM = 19
EXPECTED_AWAY_PER_TEAM = 19
EXPECTED_TOTAL_MATCHES = 1900
EXPECTED_MATCHWEEKS = 38


EN_DASH = "–"

# Anchored: the ENTIRE cell must be "<digits><en dash><digits>".
SCORE_PATTERN = re.compile(r"^\s*(\d+)\s*" + EN_DASH + r"\s*(\d+)\s*$")


CANONICAL_MATCH_COLUMNS = [
    "season",
    "date",
    "matchweek",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "home_points_from_result",
    "away_points_from_result",
]

CANONICAL_SUMMARY_COLUMNS = [
    "season",
    "team",
    "MP",
    "W",
    "D",
    "L",
    "GF",
    "GA",
    "GD",
    "Pts_from_results",
    "sanction",
    "Pts_after_sanction",
]


# ============================================================
# POINTS SANCTION REGISTRY
# ============================================================
#
# Documented Premier League PSR points deductions.
#
# These are administrative sanctions applied to the league table. They are NOT
# match results. The underlying fixture results are never altered - the
# distinction between what was won on the pitch (Pts_from_results) and what
# stood in the table (Pts_after_sanction) is preserved end to end.
#
# NAMING: the registry is keyed on the team name EXACTLY as it appears in the
# fixture files. FBref renders Nottingham Forest as "Nottingham", and Phase 0
# constraint 11 established that fixture and FBref names already reconcile
# 20/20 in every season. No fuzzy matching is introduced here. Every key is
# verified to resolve to a real team in that season, and an unresolved key is
# a hard FAIL rather than a silently ignored zero.

SANCTION_REGISTRY = {
    ("2023-2024", "Everton"): -8,
    ("2023-2024", "Nottingham"): -4,
}

SANCTION_NOTES = {
    ("2023-2024", "Everton"): "8-point deduction (PSR breach)",
    ("2023-2024", "Nottingham"): "4-point deduction (PSR); Nottingham Forest",
}


# ============================================================
# OUTPUT ENCODING
# ============================================================

def configure_stdout():
    """The report prints team names and dashes; Windows consoles default to cp1252."""

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# ============================================================
# AUDIT LEDGER
# ============================================================

class Audit:
    """Every validation writes one row here. The ledger is the deliverable."""

    def __init__(self):
        self.rows = []

    def record(self, season, check_id, check, expected, observed, passed, detail=""):

        self.rows.append({
            "season": season,
            "check_id": check_id,
            "check": check,
            "expected": expected,
            "observed": observed,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        })

        return passed

    def failures(self, season=None):

        return [
            row for row in self.rows
            if row["status"] == "FAIL"
            and (season is None or row["season"] == season)
        ]

    def season_passed(self, season):
        return not self.failures(season)

    def all_passed(self):
        return not self.failures()

    def frame(self):

        return pd.DataFrame(self.rows, columns=[
            "season", "check_id", "check",
            "expected", "observed", "status", "detail",
        ])


# ============================================================
# SOURCE INTEGRITY
# ============================================================

def hash_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()


def hash_sources():

    return {
        season: hash_file(FIXTURES_DIR / filename)
        for season, filename in SEASON_FILES.items()
    }


# ============================================================
# LOADING
# ============================================================

def read_fixture_table(path):
    """
    Read one fixture file as HTML with an EXPLICIT UTF-8 decode.

    See TRAP 2 in the module docstring. Never hand the path straight to
    read_html - it guesses the encoding and mangles the en dash.
    """

    raw_bytes = path.read_bytes()

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} is not valid UTF-8: {error}") from error

    tables = pd.read_html(io.StringIO(text))

    if not tables:
        raise ValueError(f"{path.name} contains no HTML tables.")

    # Select the fixture table by structure, never by position or filename.
    for table in tables:

        columns = {str(col).strip() for col in table.columns}

        if {"Date", "Score", "Home", "Away"} <= columns:
            return table

    raise ValueError(
        f"{path.name} has no table carrying Date/Score/Home/Away. "
        f"Columns seen: {[list(t.columns) for t in tables]}"
    )


def parse_score(value):
    """
    Return (home_goals, away_goals, error).

    EN DASH ONLY. A hyphen is not accepted; see TRAP 2.
    """

    if pd.isna(value):
        return None, None, "missing score"

    text = str(value)

    match = SCORE_PATTERN.match(text)

    if not match:
        return None, None, f"unparseable score {text!r}"

    return int(match.group(1)), int(match.group(2)), None


def result_from_goals(home_goals, away_goals):

    if home_goals > away_goals:
        return "H"

    if home_goals == away_goals:
        return "D"

    return "A"


def points_from_result(result):
    """Returns (home_points, away_points)."""

    return {
        "H": (3, 0),
        "D": (1, 1),
        "A": (0, 3),
    }[result]


def clean_name(value):

    if pd.isna(value):
        return None

    name = re.sub(r"\s+", " ", str(value)).strip()

    return name or None


def load_season(season, audit):
    """
    Build the canonical match table for one season.

    Returns (DataFrame, list_of_load_problems).
    """

    path = FIXTURES_DIR / SEASON_FILES[season]

    table = read_fixture_table(path)

    # FBref pads the export with blank spacer rows between matchweeks.
    # A row is a match if and only if it carries a score cell.
    played = table[table["Score"].notna()].copy()

    problems = []
    records = []

    for position, (_, row) in enumerate(played.iterrows()):

        home_goals, away_goals, score_error = parse_score(row["Score"])

        home_team = clean_name(row["Home"])
        away_team = clean_name(row["Away"])
        raw_date = row["Date"]

        if score_error:
            problems.append(f"row {position}: {score_error}")

        if home_team is None:
            problems.append(f"row {position}: missing home team")

        if away_team is None:
            problems.append(f"row {position}: missing away team")

        date = pd.to_datetime(raw_date, format="%Y-%m-%d", errors="coerce")

        if pd.isna(date):
            problems.append(f"row {position}: unparseable date {raw_date!r}")

        matchweek = row["Wk"]

        if pd.isna(matchweek):
            problems.append(f"row {position}: missing matchweek")
            matchweek = None
        else:
            matchweek = int(matchweek)

        if score_error is None:
            result = result_from_goals(home_goals, away_goals)
            home_points, away_points = points_from_result(result)
        else:
            result = None
            home_points = None
            away_points = None

        records.append({
            "season": season,
            "date": date,
            "matchweek": matchweek,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result,
            "home_points_from_result": home_points,
            "away_points_from_result": away_points,
            "source_row_order": position,
        })

    matches = pd.DataFrame(records)

    audit.record(
        season, "CHK00", "Source file parsed",
        "0 load problems", f"{len(problems)} load problems",
        not problems,
        "; ".join(problems[:5]),
    )

    return matches, problems


# ============================================================
# SEASON VALIDATION - the 15 required checks (14 per-season, 1 combined)
# ============================================================

def validate_season(season, matches, audit):

    checks = {}

    # ---- 1. exactly 380 matches
    count = len(matches)

    checks["matches"] = audit.record(
        season, "CHK01", "Match count",
        EXPECTED_MATCHES_PER_SEASON, count,
        count == EXPECTED_MATCHES_PER_SEASON,
    )

    teams = sorted(
        set(matches["home_team"].dropna()) | set(matches["away_team"].dropna())
    )

    # ---- 2. exactly 20 unique teams
    checks["teams"] = audit.record(
        season, "CHK02", "Unique teams",
        EXPECTED_TEAMS_PER_SEASON, len(teams),
        len(teams) == EXPECTED_TEAMS_PER_SEASON,
    )

    home_counts = matches["home_team"].value_counts()
    away_counts = matches["away_team"].value_counts()

    played_counts = {
        team: int(home_counts.get(team, 0) + away_counts.get(team, 0))
        for team in teams
    }

    # ---- 3. every team plays exactly 38 matches
    wrong_total = {
        team: n for team, n in played_counts.items()
        if n != EXPECTED_MATCHES_PER_TEAM
    }

    checks["played_38"] = audit.record(
        season, "CHK03", "Matches per team",
        f"38 for all {EXPECTED_TEAMS_PER_SEASON}",
        f"{len(teams) - len(wrong_total)}/{len(teams)} correct",
        not wrong_total,
        str(wrong_total) if wrong_total else "",
    )

    # ---- 4. every team plays exactly 19 home matches
    wrong_home = {
        team: int(home_counts.get(team, 0)) for team in teams
        if int(home_counts.get(team, 0)) != EXPECTED_HOME_PER_TEAM
    }

    checks["home_19"] = audit.record(
        season, "CHK04", "Home matches per team",
        f"19 for all {EXPECTED_TEAMS_PER_SEASON}",
        f"{len(teams) - len(wrong_home)}/{len(teams)} correct",
        not wrong_home,
        str(wrong_home) if wrong_home else "",
    )

    # ---- 5. every team plays exactly 19 away matches
    wrong_away = {
        team: int(away_counts.get(team, 0)) for team in teams
        if int(away_counts.get(team, 0)) != EXPECTED_AWAY_PER_TEAM
    }

    checks["away_19"] = audit.record(
        season, "CHK05", "Away matches per team",
        f"19 for all {EXPECTED_TEAMS_PER_SEASON}",
        f"{len(teams) - len(wrong_away)}/{len(teams)} correct",
        not wrong_away,
        str(wrong_away) if wrong_away else "",
    )

    # ---- 6. no team plays itself
    self_matches = matches[matches["home_team"] == matches["away_team"]]

    checks["no_self"] = audit.record(
        season, "CHK06", "No team plays itself",
        0, len(self_matches),
        len(self_matches) == 0,
    )

    # ---- 7. no duplicate season + date + home_team + away_team
    duplicate_mask = matches.duplicated(
        subset=["season", "date", "home_team", "away_team"], keep=False
    )

    duplicates = matches[duplicate_mask]

    checks["no_duplicates"] = audit.record(
        season, "CHK07", "Duplicate season+date+home+away",
        0, len(duplicates),
        len(duplicates) == 0,
        "; ".join(
            f"{r.date} {r.home_team} v {r.away_team}"
            for r in duplicates.head(5).itertuples()
        ),
    )

    # ---- 8. no missing dates
    missing_dates = int(matches["date"].isna().sum())

    checks["dates"] = audit.record(
        season, "CHK08", "Missing dates",
        0, missing_dates,
        missing_dates == 0,
    )

    # ---- 9. no missing teams
    missing_teams = int(
        matches["home_team"].isna().sum() + matches["away_team"].isna().sum()
    )

    checks["team_names"] = audit.record(
        season, "CHK09", "Missing team names",
        0, missing_teams,
        missing_teams == 0,
    )

    # ---- 10. no missing scores
    missing_scores = int(
        matches["home_goals"].isna().sum() + matches["away_goals"].isna().sum()
    )

    checks["scores_present"] = audit.record(
        season, "CHK10", "Missing scores",
        0, missing_scores,
        missing_scores == 0,
    )

    # ---- 11. no negative goals
    scored = matches.dropna(subset=["home_goals", "away_goals"])

    negative = scored[(scored["home_goals"] < 0) | (scored["away_goals"] < 0)]

    checks["no_negative"] = audit.record(
        season, "CHK11", "Negative goals",
        0, len(negative),
        len(negative) == 0,
    )

    # ---- 12. every result agrees with the score
    recomputed_result = scored.apply(
        lambda row: result_from_goals(row["home_goals"], row["away_goals"]),
        axis=1,
    )

    result_mismatch = scored[scored["result"] != recomputed_result]

    checks["result_valid"] = audit.record(
        season, "CHK12", "Result agrees with score",
        f"{len(scored)}/{len(scored)}",
        f"{len(scored) - len(result_mismatch)}/{len(scored)}",
        len(result_mismatch) == 0,
        "; ".join(
            f"{r.home_team} {r.home_goals}-{r.away_goals} {r.away_team} -> {r.result}"
            for r in result_mismatch.head(5).itertuples()
        ),
    )

    # ---- 13. every points value agrees with the result
    expected_points = scored["result"].map(points_from_result)

    points_mismatch = scored[
        (scored["home_points_from_result"] != expected_points.str[0])
        | (scored["away_points_from_result"] != expected_points.str[1])
    ]

    total_points_values = len(scored) * 2

    checks["points_valid"] = audit.record(
        season, "CHK13", "Points agree with result",
        f"{total_points_values}/{total_points_values}",
        f"{total_points_values - len(points_mismatch) * 2}/{total_points_values}",
        len(points_mismatch) == 0,
        "; ".join(
            f"{r.home_team} v {r.away_team} {r.result}"
            for r in points_mismatch.head(5).itertuples()
        ),
    )

    # ---- 14. chronological within the season
    #
    # Checked in SOURCE ORDER. The output is stably sorted by date afterwards,
    # so this reports whether the source itself arrived ordered rather than
    # whether the sort worked.
    in_source_order = matches.sort_values("source_row_order")

    date_steps = in_source_order["date"].diff().dropna()

    out_of_order = int((date_steps < pd.Timedelta(0)).sum())

    checks["chronological"] = audit.record(
        season, "CHK14", "Chronological in source order",
        "0 backward steps", f"{out_of_order} backward steps",
        out_of_order == 0,
    )

    # ---- supplementary: matchweek integrity
    missing_weeks = int(matches["matchweek"].isna().sum())

    week_values = matches["matchweek"].dropna()

    bad_weeks = int(((week_values < 1) | (week_values > EXPECTED_MATCHWEEKS)).sum())

    distinct_weeks = week_values.nunique()

    checks["matchweek"] = audit.record(
        season, "CHK16", "Matchweek present and in 1..38",
        "0 missing, 0 out of range, 38 distinct",
        f"{missing_weeks} missing, {bad_weeks} out of range, "
        f"{distinct_weeks} distinct",
        missing_weeks == 0
        and bad_weeks == 0
        and distinct_weeks == EXPECTED_MATCHWEEKS,
    )

    # ---- supplementary: a team is never in two places on one date
    stacked = pd.concat([
        matches[["date", "home_team"]].rename(columns={"home_team": "team"}),
        matches[["date", "away_team"]].rename(columns={"away_team": "team"}),
    ])

    same_day = int(stacked.duplicated(subset=["date", "team"]).sum())

    checks["one_match_per_day"] = audit.record(
        season, "CHK17", "No team appears twice on one date",
        0, same_day,
        same_day == 0,
    )

    return checks, teams


# ============================================================
# SEASON-END RECONSTRUCTION
# ============================================================

def build_team_summary(matches):
    """
    Independently rebuild the season-end league statistics for every team
    from the fixture results alone.

    Nothing here is read from FBref. Phase 0 Instrument 4 already performed
    the cross-source reconciliation; this is a from-scratch derivation.
    """

    home_side = matches.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_goals": "GF",
        "away_goals": "GA",
        "home_points_from_result": "Pts",
    })[["season", "team", "opponent", "GF", "GA", "Pts", "result"]].copy()

    home_side["W"] = (home_side["result"] == "H").astype(int)
    home_side["D"] = (home_side["result"] == "D").astype(int)
    home_side["L"] = (home_side["result"] == "A").astype(int)

    away_side = matches.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_goals": "GF",
        "home_goals": "GA",
        "away_points_from_result": "Pts",
    })[["season", "team", "opponent", "GF", "GA", "Pts", "result"]].copy()

    away_side["W"] = (away_side["result"] == "A").astype(int)
    away_side["D"] = (away_side["result"] == "D").astype(int)
    away_side["L"] = (away_side["result"] == "H").astype(int)

    team_matches = pd.concat([home_side, away_side], ignore_index=True)

    summary = team_matches.groupby(["season", "team"], as_index=False).agg(
        MP=("team", "size"),
        W=("W", "sum"),
        D=("D", "sum"),
        L=("L", "sum"),
        GF=("GF", "sum"),
        GA=("GA", "sum"),
        Pts_from_results=("Pts", "sum"),
    )

    summary["GD"] = summary["GF"] - summary["GA"]

    summary["sanction"] = [
        SANCTION_REGISTRY.get((season, team), 0)
        for season, team in zip(summary["season"], summary["team"])
    ]

    summary["Pts_after_sanction"] = (
        summary["Pts_from_results"] + summary["sanction"]
    )

    for column in ["MP", "W", "D", "L", "GF", "GA", "GD",
                   "Pts_from_results", "sanction", "Pts_after_sanction"]:
        summary[column] = summary[column].astype(int)

    summary = summary.sort_values(
        ["season", "Pts_after_sanction", "GD", "GF", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)

    return summary[CANONICAL_SUMMARY_COLUMNS]


def validate_summary(season, summary, matches, audit):

    season_rows = summary[summary["season"] == season]

    total = len(season_rows)

    # ---- MP = W + D + L
    mp_bad = season_rows[
        season_rows["MP"] != season_rows["W"] + season_rows["D"] + season_rows["L"]
    ]

    audit.record(
        season, "CHK18", "MP = W + D + L",
        f"{total}/{total}", f"{total - len(mp_bad)}/{total}",
        len(mp_bad) == 0,
        "; ".join(mp_bad["team"].head(5)),
    )

    # ---- GD = GF - GA
    gd_bad = season_rows[season_rows["GD"] != season_rows["GF"] - season_rows["GA"]]

    audit.record(
        season, "CHK19", "GD = GF - GA",
        f"{total}/{total}", f"{total - len(gd_bad)}/{total}",
        len(gd_bad) == 0,
        "; ".join(gd_bad["team"].head(5)),
    )

    # ---- Pts_from_results = 3W + D
    pts_bad = season_rows[
        season_rows["Pts_from_results"] != 3 * season_rows["W"] + season_rows["D"]
    ]

    audit.record(
        season, "CHK20", "Pts_from_results = 3W + D",
        f"{total}/{total}", f"{total - len(pts_bad)}/{total}",
        len(pts_bad) == 0,
        "; ".join(pts_bad["team"].head(5)),
    )

    # ---- MP = 38 for every team
    mp38_bad = season_rows[season_rows["MP"] != EXPECTED_MATCHES_PER_TEAM]

    audit.record(
        season, "CHK21", "MP = 38 for every team",
        f"38 x {total}", f"{total - len(mp38_bad)}/{total} at 38",
        len(mp38_bad) == 0,
        "; ".join(mp38_bad["team"].head(5)),
    )

    # ---- league-wide closure: every goal scored is a goal conceded
    total_gf = int(season_rows["GF"].sum())
    total_ga = int(season_rows["GA"].sum())
    match_goals = int(matches["home_goals"].sum() + matches["away_goals"].sum())

    audit.record(
        season, "CHK22", "League GF = GA = goals in matches",
        match_goals, f"GF {total_gf} / GA {total_ga}",
        total_gf == total_ga == match_goals,
    )

    # ---- league-wide closure: GD sums to zero
    total_gd = int(season_rows["GD"].sum())

    audit.record(
        season, "CHK23", "League GD sums to zero",
        0, total_gd,
        total_gd == 0,
    )

    # ---- league-wide closure: total points = 3*decisive + 2*draws
    draws = int((matches["result"] == "D").sum())
    decisive = len(matches) - draws

    expected_total_points = 3 * decisive + 2 * draws
    observed_total_points = int(season_rows["Pts_from_results"].sum())

    audit.record(
        season, "CHK24", "League points = 3*decisive + 2*draws",
        expected_total_points, observed_total_points,
        expected_total_points == observed_total_points,
    )

    # ---- the sanction registry resolves against real teams
    season_teams = set(season_rows["team"])

    unresolved = [
        team for (registry_season, team) in SANCTION_REGISTRY
        if registry_season == season and team not in season_teams
    ]

    expected_sanctioned = sum(
        1 for (registry_season, _) in SANCTION_REGISTRY
        if registry_season == season
    )

    applied = season_rows[season_rows["sanction"] != 0]

    audit.record(
        season, "CHK25", "Sanction registry resolves to real teams",
        f"{expected_sanctioned} sanctioned, 0 unresolved",
        f"{len(applied)} sanctioned, {len(unresolved)} unresolved",
        not unresolved and len(applied) == expected_sanctioned,
        "unresolved: " + ", ".join(unresolved) if unresolved else "",
    )

    # ---- sanction arithmetic
    sanction_bad = season_rows[
        season_rows["Pts_after_sanction"]
        != season_rows["Pts_from_results"] + season_rows["sanction"]
    ]

    audit.record(
        season, "CHK26", "Pts_after_sanction = Pts_from_results + sanction",
        f"{total}/{total}", f"{total - len(sanction_bad)}/{total}",
        len(sanction_bad) == 0,
        "; ".join(sanction_bad["team"].head(5)),
    )

    # ---- unsanctioned teams must carry two identical points columns
    unsanctioned = season_rows[season_rows["sanction"] == 0]

    drifted = unsanctioned[
        unsanctioned["Pts_after_sanction"] != unsanctioned["Pts_from_results"]
    ]

    audit.record(
        season, "CHK27", "Unsanctioned teams: Pts columns identical",
        f"{len(unsanctioned)}/{len(unsanctioned)}",
        f"{len(unsanctioned) - len(drifted)}/{len(unsanctioned)}",
        len(drifted) == 0,
        "; ".join(drifted["team"].head(5)),
    )


# ============================================================
# COMBINED VALIDATION
# ============================================================

def validate_combined(matches, summary, audit, source_hashes_before):

    # ---- 15. the complete five-season dataset holds exactly 1,900 matches
    audit.record(
        "ALL", "CHK15", "Five-season match count",
        EXPECTED_TOTAL_MATCHES, len(matches),
        len(matches) == EXPECTED_TOTAL_MATCHES,
    )

    audit.record(
        "ALL", "CHK28", "Seasons present",
        len(SEASON_FILES), matches["season"].nunique(),
        matches["season"].nunique() == len(SEASON_FILES),
        ", ".join(sorted(matches["season"].unique())),
    )

    cross_duplicates = int(
        matches.duplicated(
            subset=["season", "date", "home_team", "away_team"]
        ).sum()
    )

    audit.record(
        "ALL", "CHK29", "Duplicates across full dataset",
        0, cross_duplicates,
        cross_duplicates == 0,
    )

    audit.record(
        "ALL", "CHK30", "Team-season rows",
        len(SEASON_FILES) * EXPECTED_TEAMS_PER_SEASON, len(summary),
        len(summary) == len(SEASON_FILES) * EXPECTED_TEAMS_PER_SEASON,
    )

    # ---- result domain is exactly {H, D, A}
    observed_results = sorted(matches["result"].dropna().unique())

    audit.record(
        "ALL", "CHK31", "Result domain",
        "['A', 'D', 'H']", str(observed_results),
        observed_results == ["A", "D", "H"],
    )

    # ---- no nulls anywhere in the canonical table
    null_counts = matches[CANONICAL_MATCH_COLUMNS].isna().sum()

    total_nulls = int(null_counts.sum())

    audit.record(
        "ALL", "CHK32", "Nulls in canonical match table",
        0, total_nulls,
        total_nulls == 0,
        str(null_counts[null_counts > 0].to_dict()) if total_nulls else "",
    )

    # ---- the whole registry landed, and nothing else did
    applied = summary[summary["sanction"] != 0]

    expected_applied = pd.DataFrame(
        [
            {"season": season, "team": team, "sanction": value}
            for (season, team), value in sorted(SANCTION_REGISTRY.items())
        ]
    )

    matched = (
        applied[["season", "team", "sanction"]]
        .sort_values(["season", "team"])
        .reset_index(drop=True)
        .equals(
            expected_applied
            .sort_values(["season", "team"])
            .reset_index(drop=True)
        )
    )

    audit.record(
        "ALL", "CHK33", "Sanctions applied match registry exactly",
        f"{len(SANCTION_REGISTRY)} entries", f"{len(applied)} applied",
        matched,
        "; ".join(
            f"{r.season} {r.team} {r.sanction}" for r in applied.itertuples()
        ),
    )

    # ---- promoted / relegated turnover is represented, not invented
    turnover_detail = []

    seasons = sorted(matches["season"].unique())

    for previous, current in zip(seasons, seasons[1:]):

        previous_teams = set(summary[summary["season"] == previous]["team"])
        current_teams = set(summary[summary["season"] == current]["team"])

        promoted = sorted(current_teams - previous_teams)
        relegated = sorted(previous_teams - current_teams)

        turnover_detail.append(
            f"{current}: in [{', '.join(promoted)}] out [{', '.join(relegated)}]"
        )

        audit.record(
            current, "CHK34", "Promotion/relegation turnover is balanced",
            "equal counts in and out",
            f"{len(promoted)} in / {len(relegated)} out",
            len(promoted) == len(relegated),
            turnover_detail[-1],
        )

    # ---- READ-ONLY guarantee
    source_hashes_after = hash_sources()

    unchanged = source_hashes_before == source_hashes_after

    changed = [
        season for season in source_hashes_before
        if source_hashes_before[season] != source_hashes_after.get(season)
    ]

    audit.record(
        "ALL", "CHK35", "Source files unmodified (SHA-256)",
        "all identical", "identical" if unchanged else f"CHANGED: {changed}",
        unchanged,
    )

    return turnover_detail


# ============================================================
# REPORT
# ============================================================

def status_text(passed):
    return "PASS" if passed else "FAIL"


def line(label, value, verdict=None):

    if verdict is None:
        print(f"  {label:<22}{value}")
    else:
        print(f"  {label:<22}{value:<44}{verdict}")


def print_season_report(season, matches, checks, teams, audit):

    scored = matches.dropna(subset=["home_goals", "away_goals"])

    print()
    print("-" * 79)

    line("Season:", season)

    line(
        "Matches:",
        f"{len(matches)} / {EXPECTED_MATCHES_PER_SEASON}",
        status_text(checks["matches"]),
    )

    line(
        "Teams:",
        f"{len(teams)} unique, every one playing 38",
        status_text(checks["teams"] and checks["played_38"]),
    )

    line(
        "Home/away balance:",
        "19 home / 19 away for every team",
        status_text(
            checks["home_19"] and checks["away_19"] and checks["no_self"]
        ),
    )

    line(
        "Duplicate matches:",
        "0 on season+date+home+away",
        status_text(checks["no_duplicates"]),
    )

    line(
        "Score validity:",
        f"{len(scored)}/{len(matches)} parsed on en dash, none negative",
        status_text(
            checks["scores_present"]
            and checks["no_negative"]
            and checks["dates"]
            and checks["team_names"]
        ),
    )

    line(
        "Result validity:",
        f"{len(scored)}/{len(scored)} agree with the score",
        status_text(checks["result_valid"]),
    )

    line(
        "Points validity:",
        f"{len(scored) * 2}/{len(scored) * 2} agree with the result",
        status_text(checks["points_valid"]),
    )

    line(
        "Chronological order:",
        "no backward date step in source order",
        status_text(checks["chronological"]),
    )

    line("Status:", status_text(audit.season_passed(season)))

    for failure in audit.failures(season):
        print(
            f"      FAIL {failure['check_id']} {failure['check']}: "
            f"expected {failure['expected']}, got {failure['observed']} "
            f"{failure['detail']}".rstrip()
        )


def print_summary_tables(summary):

    print()
    print("=" * 79)
    print("SEASON-END RECONSTRUCTION FROM FIXTURE RESULTS")
    print("=" * 79)
    print()
    print("  Rebuilt from match results alone. No FBref file was read.")
    print("  Identities verified: MP = W+D+L, GD = GF-GA, Pts = 3W+D.")

    for season in sorted(summary["season"].unique()):

        rows = summary[summary["season"] == season]

        print()
        print(f"  {season}")
        print(
            f"    {'#':>2}  {'Team':<18}{'MP':>3}{'W':>4}{'D':>3}{'L':>3}"
            f"{'GF':>5}{'GA':>4}{'GD':>5}{'Pts':>5}{'Sanc':>6}{'Final':>7}"
        )

        for position, row in enumerate(rows.itertuples(), start=1):

            sanction = "" if row.sanction == 0 else str(row.sanction)

            print(
                f"    {position:>2}  {row.team:<18}{row.MP:>3}{row.W:>4}"
                f"{row.D:>3}{row.L:>3}{row.GF:>5}{row.GA:>4}{row.GD:>5}"
                f"{row.Pts_from_results:>5}{sanction:>6}"
                f"{row.Pts_after_sanction:>7}"
            )


def print_sanction_report(summary):

    print()
    print("=" * 79)
    print("POINTS SANCTION REGISTRY")
    print("=" * 79)
    print()
    print("  Match results are NOT altered. Both points columns are carried")
    print("  forward so the distinction survives into every later phase.")
    print()

    applied = summary[summary["sanction"] != 0]

    if applied.empty:
        print("  No sanctions applied.")
        return

    print(
        f"    {'Season':<12}{'Team':<18}{'From results':>13}"
        f"{'Sanction':>10}{'After':>8}   Note"
    )

    for row in applied.itertuples():

        note = SANCTION_NOTES.get((row.season, row.team), "")

        print(
            f"    {row.season:<12}{row.team:<18}{row.Pts_from_results:>13}"
            f"{row.sanction:>10}{row.Pts_after_sanction:>8}   {note}"
        )

    unsanctioned = len(summary) - len(applied)

    print()
    print(f"  All other {unsanctioned} team-seasons carry sanction = 0.")


def print_turnover_report(turnover_detail):

    print()
    print("=" * 79)
    print("PROMOTED TEAMS")
    print("=" * 79)
    print()
    print("  Identities only. No previous-season Premier League statistics are")
    print("  invented for a promoted side - they simply have no prior row.")
    print()

    for detail in turnover_detail:
        print(f"    {detail}")


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()

    print()
    print("=" * 79)
    print("PHASE 1 - MATCH FOUNDATION")
    print("=" * 79)
    print()
    print(f"  Source     : {FIXTURES_DIR}")
    print("  Mode       : READ-ONLY (SHA-256 verified before and after)")
    print("  Scope      : fixture files only - no FBref aggregates, no rolling")
    print("               features, no Elo, no modelling dataset")
    print("  Separator  : U+2013 EN DASH, files decoded UTF-8 explicitly")

    missing = [
        filename for filename in SEASON_FILES.values()
        if not (FIXTURES_DIR / filename).exists()
    ]

    if missing:
        print()
        print("  FAIL - missing source files:")

        for filename in missing:
            print(f"    {filename}")

        print()
        print("=" * 79)
        print("PHASE 1 - INSTRUMENT 1 STATUS")
        print("=" * 79)
        print()
        print("FAIL")
        print()

        return 1

    source_hashes_before = hash_sources()

    audit = Audit()

    season_frames = []
    season_state = {}

    print()
    print("=" * 79)
    print("PER-SEASON VALIDATION")
    print("=" * 79)

    for season in SEASON_FILES:

        matches, _ = load_season(season, audit)

        checks, teams = validate_season(season, matches, audit)

        season_state[season] = (checks, teams)

        season_frames.append(matches)

    all_matches = pd.concat(season_frames, ignore_index=True)

    # Stable sort. Source order already runs chronologically, so this fixes a
    # deterministic within-day order without disturbing the date sequence.
    all_matches = all_matches.sort_values(
        ["season", "date", "source_row_order"]
    ).reset_index(drop=True)

    summary = build_team_summary(all_matches)

    for season in SEASON_FILES:

        season_matches = all_matches[all_matches["season"] == season]

        validate_summary(season, summary, season_matches, audit)

    turnover_detail = validate_combined(
        all_matches, summary, audit, source_hashes_before
    )

    for season in SEASON_FILES:

        checks, teams = season_state[season]

        print_season_report(
            season,
            all_matches[all_matches["season"] == season],
            checks,
            teams,
            audit,
        )

    print_summary_tables(summary)
    print_sanction_report(summary)
    print_turnover_report(turnover_detail)

    combined_failures = audit.failures("ALL")

    print()
    print("=" * 79)
    print("FIVE-SEASON COMBINED")
    print("=" * 79)
    print()

    line("Seasons:", f"{all_matches['season'].nunique()} / {len(SEASON_FILES)}")

    line(
        "Matches:",
        f"{len(all_matches)} / {EXPECTED_TOTAL_MATCHES}",
        status_text(len(all_matches) == EXPECTED_TOTAL_MATCHES),
    )

    line(
        "Team-seasons:",
        f"{len(summary)} / {len(SEASON_FILES) * EXPECTED_TEAMS_PER_SEASON}",
        status_text(len(summary) == len(SEASON_FILES) * EXPECTED_TEAMS_PER_SEASON),
    )

    line(
        "Duplicate matches:",
        "0 across all five seasons",
        status_text(
            not [f for f in combined_failures if f["check_id"] == "CHK29"]
        ),
    )

    line(
        "Result distribution:",
        "  ".join(
            f"{value} {count} ({count / len(all_matches):.1%})"
            for value, count
            in all_matches["result"].value_counts().reindex(["H", "D", "A"]).items()
        ),
    )

    line(
        "Sanctions:",
        f"{len(SANCTION_REGISTRY)} applied, match results untouched",
        status_text(
            not [f for f in combined_failures if f["check_id"] == "CHK33"]
        ),
    )

    line(
        "Source integrity:",
        "SHA-256 identical before and after",
        status_text(
            not [f for f in combined_failures if f["check_id"] == "CHK35"]
        ),
    )

    line("Status:", status_text(not combined_failures))

    for failure in combined_failures:
        print(
            f"      FAIL {failure['check_id']} {failure['check']}: "
            f"expected {failure['expected']}, got {failure['observed']} "
            f"{failure['detail']}".rstrip()
        )

    # ---- outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_matches = all_matches.copy()
    output_matches["date"] = output_matches["date"].dt.strftime("%Y-%m-%d")

    output_matches = output_matches[CANONICAL_MATCH_COLUMNS]

    output_matches.to_csv(MATCHES_OUTPUT, index=False, encoding="utf-8")
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8")

    audit_frame = audit.frame()
    audit_frame.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8")

    print()
    print("=" * 79)
    print("OUTPUTS")
    print("=" * 79)
    print()
    print(f"  {MATCHES_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(output_matches)} rows)")
    print(f"  {SUMMARY_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(summary)} rows)")
    print(f"  {AUDIT_OUTPUT.relative_to(PROJECT_ROOT)}"
          f"  ({len(audit_frame)} checks)")

    passed = audit.all_passed()

    total_checks = len(audit_frame)
    failed_checks = len(audit.failures())

    print()
    print("=" * 79)
    print("PHASE 1 - INSTRUMENT 1 STATUS")
    print("=" * 79)
    print()
    print(f"  Validations run    : {total_checks}")
    print(f"  Validations passed : {total_checks - failed_checks}")
    print(f"  Validations failed : {failed_checks}")
    print()
    print(status_text(passed))
    print()

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
