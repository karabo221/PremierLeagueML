"""
PHASE 0 - INSTRUMENT 4: CROSS-SOURCE RECONCILIATION

Rebuilds each season's final league table from the FIXTURE RESULTS alone, then
checks that reconstruction against FBref's published Overall and Home/Away
tables. Two independent sources describing the same 380 matches must agree; if
they do not, one of them is wrong and the modelling layer must not be built on
either until it is understood.

Scope: Fixtures + Overall + Home/Away only. The 2026 Shooting-Opponent table is
known to be missing (Instrument 3) and is irrelevant here - this instrument never
touches the per-statistic tables, so that gap does not block it.

STRICTLY READ-ONLY. Nothing is renamed, moved, deleted, modified, merged or
repaired. No ML features are built. No discrepancy is corrected, smoothed, or
hidden: a mismatch is reported with the exact team, metric, and both values.

Two rules drive the design:

  1. TABLES ARE FOUND BY CONTENT, NOT FILENAME. The Overall and Home/Away tables
     are located with the same schema classifier Instrument 3 uses, imported
     directly so the two instruments cannot drift apart. Filenames in this
     dataset are known to lie.

  2. NOTHING IS COERCED TO MAKE IT AGREE. An FBref cell that is not cleanly
     integral is reported UNKNOWN, not forced into a number. A team that cannot
     be matched between sources is reported MAPPING ERROR, not fuzzy-matched.
     Silence is the failure mode this instrument exists to prevent.

MultiIndex columns are never flattened. Home/Away values are read by their full
hierarchy path - ('Home', 'MP') and ('Away', 'MP') are distinct columns whose
leaf names collide, so leaf-level access would silently return the wrong one.
"""

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reused so this instrument and Instrument 3 classify tables identically.
from phase0_statistical_integrity import (  # noqa: E402
    classify_table_type,
    column_path,
    decode,
    find_squad_column,
    parse_tables,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIXTURES_DIR = RAW_DIR / "Fixtures"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OVERALL_PATH = OUTPUT_DIR / "phase0_overall_reconciliation.csv"
HOMEAWAY_PATH = OUTPUT_DIR / "phase0_homeaway_reconciliation.csv"
SUMMARY_PATH = OUTPUT_DIR / "phase0_reconciliation_summary.csv"

POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0

# FBref writes scorelines with an EN DASH, not a hyphen.
SCORE_SEPARATORS = ["–", "—", "-"]

OVERALL_METRICS = ["MP", "W", "D", "L", "GF", "GA", "GD", "Pts"]

RECON_FIELDS = [
    "season",
    "team",
    "metric",
    "fixture_value",
    "fbref_value",
    "difference",
    "status",
    "fbref_note",
]

SUMMARY_FIELDS = [
    "season",
    "fixture_file",
    "stats_folder",
    "overall_source",
    "homeaway_source",
    "fixtures_used",
    "teams",
    "overall_metrics_checked",
    "overall_mismatches",
    "overall_unmapped",
    "homeaway_metrics_checked",
    "homeaway_mismatches",
    "homeaway_unmapped",
    "status",
]

MAX_EXAMPLES = 12


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def read_table(path):
    """Parse the single HTML table out of an FBref .xls. Read-only."""
    raw = path.read_bytes()
    text, _ = decode(raw)
    tables, _, error = parse_tables(text)
    if tables is None:
        return None, error
    table = max(tables, key=lambda t: t.shape[0] * t.shape[1])
    return table, None


def strict_int(value):
    """Return (int, True) only for a cleanly integral value.

    Anything else returns (None, False) and is reported UNKNOWN. Nothing is
    rounded, truncated, or defaulted to zero to make a comparison succeed.
    """
    if value is None:
        return None, False
    if isinstance(value, float) and pd.isna(value):
        return None, False

    text = str(value).strip()
    if text == "" or text.lower() in ("nan", "none"):
        return None, False
    if text.startswith("+"):  # FBref sometimes signs goal difference
        text = text[1:]

    try:
        number = float(text)
    except ValueError:
        return None, False
    if number != int(number):
        return None, False
    return int(number), True


# --------------------------------------------------------------------------
# locating source tables BY CONTENT
# --------------------------------------------------------------------------


def find_table_by_type(folder, wanted_type):
    """Find the table of a given type in a season folder, classified by schema.

    Returns (path, table, note). If several files carry the type, identical
    content is reported and the first used; genuinely different content is a
    mapping error rather than an arbitrary pick.
    """
    if not folder.is_dir():
        return None, None, "stats folder not found: {}".format(folder)

    matches = []
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".xls":
            continue
        table, error = read_table(path)
        if table is None:
            continue
        if classify_table_type(table) == wanted_type:
            matches.append((path, table))

    if not matches:
        return None, None, "no table in {} classifies as {}".format(folder.name, wanted_type)

    if len(matches) > 1:
        signatures = {t.to_csv(index=False) for _, t in matches}
        names = ", ".join(p.name for p, _ in matches)
        if len(signatures) > 1:
            return None, None, (
                "{} files classify as {} with DIFFERENT content ({}) - "
                "cannot choose".format(len(matches), wanted_type, names)
            )
        path, table = matches[0]
        return path, table, "{} identical copies present ({})".format(len(matches), names)

    path, table = matches[0]
    return path, table, ""


def build_column_index(table):
    """Map full hierarchy path -> column object. MultiIndex is never flattened."""
    return {column_path(col): col for col in table.columns}


def squad_index(table, squad_col):
    """Map team name -> row position. Exact match only, no fuzzy matching."""
    index = {}
    for position, value in enumerate(table[squad_col]):
        if pd.isna(value):
            continue
        index[str(value).strip()] = position
    return index


# --------------------------------------------------------------------------
# reconstruction from fixtures
# --------------------------------------------------------------------------


def parse_score(value):
    """Return (home_goals, away_goals) or None. Never guesses at a bad score."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if text == "":
        return None
    for separator in SCORE_SEPARATORS:
        if separator in text:
            parts = text.split(separator)
            if len(parts) != 2:
                return None
            home, home_ok = strict_int(parts[0])
            away, away_ok = strict_int(parts[1])
            if home_ok and away_ok and home >= 0 and away >= 0:
                return home, away
            return None
    return None


def blank_record():
    return {
        "MP": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0,
        "Home_MP": 0, "Home_W": 0, "Home_D": 0, "Home_L": 0,
        "Home_GF": 0, "Home_GA": 0, "Home_Pts": 0,
        "Away_MP": 0, "Away_W": 0, "Away_D": 0, "Away_L": 0,
        "Away_GF": 0, "Away_GA": 0, "Away_Pts": 0,
    }


def reconstruct(fixtures):
    """Rebuild overall, home and away records from played fixtures alone.

    Returns (records, stats). Only rows with two named teams and a parseable
    score contribute; everything skipped is counted and reported.
    """
    stats = {
        "rows_total": len(fixtures),
        "rows_blank": 0,
        "rows_used": 0,
        "rows_no_score": 0,
        "rows_bad_score": 0,
        "rows_missing_team": 0,
    }

    records = {}

    def record_for(team):
        if team not in records:
            records[team] = blank_record()
        return records[team]

    for _, row in fixtures.iterrows():
        if row.isna().all():
            stats["rows_blank"] += 1
            continue

        home_team = row.get("Home")
        away_team = row.get("Away")
        home_blank = pd.isna(home_team) or str(home_team).strip() == ""
        away_blank = pd.isna(away_team) or str(away_team).strip() == ""

        raw_score = row.get("Score")
        score_blank = pd.isna(raw_score) or str(raw_score).strip() == ""

        # A row with no teams and no score is a spacer, not a fixture.
        if home_blank and away_blank and score_blank:
            stats["rows_blank"] += 1
            continue

        if home_blank or away_blank:
            stats["rows_missing_team"] += 1
            continue

        if score_blank:
            stats["rows_no_score"] += 1
            continue

        score = parse_score(raw_score)
        if score is None:
            stats["rows_bad_score"] += 1
            continue

        home_goals, away_goals = score
        home_team = str(home_team).strip()
        away_team = str(away_team).strip()

        home = record_for(home_team)
        away = record_for(away_team)

        home["MP"] += 1
        away["MP"] += 1
        home["Home_MP"] += 1
        away["Away_MP"] += 1

        home["GF"] += home_goals
        home["GA"] += away_goals
        away["GF"] += away_goals
        away["GA"] += home_goals

        home["Home_GF"] += home_goals
        home["Home_GA"] += away_goals
        away["Away_GF"] += away_goals
        away["Away_GA"] += home_goals

        if home_goals > away_goals:
            home["W"] += 1
            home["Home_W"] += 1
            home["Pts"] += POINTS_WIN
            home["Home_Pts"] += POINTS_WIN
            away["L"] += 1
            away["Away_L"] += 1
            away["Pts"] += POINTS_LOSS
            away["Away_Pts"] += POINTS_LOSS
        elif home_goals < away_goals:
            away["W"] += 1
            away["Away_W"] += 1
            away["Pts"] += POINTS_WIN
            away["Away_Pts"] += POINTS_WIN
            home["L"] += 1
            home["Home_L"] += 1
            home["Pts"] += POINTS_LOSS
            home["Home_Pts"] += POINTS_LOSS
        else:
            home["D"] += 1
            home["Home_D"] += 1
            home["Pts"] += POINTS_DRAW
            home["Home_Pts"] += POINTS_DRAW
            away["D"] += 1
            away["Away_D"] += 1
            away["Pts"] += POINTS_DRAW
            away["Away_Pts"] += POINTS_DRAW

        stats["rows_used"] += 1

    # Goal difference is derived, never read from a source.
    for record in records.values():
        record["GD"] = record["GF"] - record["GA"]
        record["Home_GD"] = record["Home_GF"] - record["Home_GA"]
        record["Away_GD"] = record["Away_GF"] - record["Away_GA"]

    return records, stats


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


def compare_metric(season, team, metric, fixture_value, fbref_cell, note=""):
    """One comparison row. Never coerces a value to force agreement.

    `note` carries whatever the source itself says about this team (FBref's
    Notes column). It is evidence attached to a discrepancy, never a licence to
    downgrade one: a mismatch stays a mismatch whatever the note says.
    """
    parsed, ok = strict_int(fbref_cell)
    if not ok:
        shown = "" if fbref_cell is None else str(fbref_cell).strip()
        return {
            "season": season,
            "team": team,
            "metric": metric,
            "fixture_value": fixture_value,
            "fbref_value": shown,
            "difference": "",
            "status": "UNKNOWN",
            "fbref_note": note,
        }

    difference = fixture_value - parsed
    return {
        "season": season,
        "team": team,
        "metric": metric,
        "fixture_value": fixture_value,
        "fbref_value": parsed,
        "difference": difference,
        "status": "MATCH" if difference == 0 else "MISMATCH",
        "fbref_note": note,
    }


def extract_notes(table):
    """Team -> the source's own Notes text, when the table carries one."""
    columns = build_column_index(table)
    squad_col = find_squad_column(table)
    note_col = None
    for path, col in columns.items():
        if path[-1].strip().lower() == "notes":
            note_col = col
            break
    if note_col is None or squad_col is None:
        return {}

    notes = {}
    for team, note in zip(table[squad_col], table[note_col]):
        if pd.isna(team):
            continue
        notes[str(team).strip()] = "" if pd.isna(note) else str(note).strip()
    return notes


def reconcile(season, records, table, squad_col, metric_paths, notes=None):
    """Compare every reconstructed metric against the FBref table.

    metric_paths maps metric name -> full column hierarchy path.
    """
    rows = []
    notes = notes or {}
    columns = build_column_index(table)
    teams = squad_index(table, squad_col)

    fixture_teams = set(records)
    fbref_teams = set(teams)

    for team in sorted(fixture_teams - fbref_teams):
        rows.append({
            "season": season, "team": team, "metric": "*",
            "fixture_value": "", "fbref_value": "",
            "difference": "", "status": "MAPPING ERROR (absent from FBref table)",
            "fbref_note": "",
        })
    for team in sorted(fbref_teams - fixture_teams):
        rows.append({
            "season": season, "team": team, "metric": "*",
            "fixture_value": "", "fbref_value": "",
            "difference": "", "status": "MAPPING ERROR (absent from fixtures)",
            "fbref_note": notes.get(team, ""),
        })

    for team in sorted(fixture_teams & fbref_teams):
        position = teams[team]
        for metric, path in metric_paths.items():
            column = columns.get(path)
            if column is None:
                rows.append({
                    "season": season, "team": team, "metric": metric,
                    "fixture_value": records[team][metric], "fbref_value": "",
                    "difference": "",
                    "status": "MAPPING ERROR (no column {})".format(" | ".join(path)),
                    "fbref_note": notes.get(team, ""),
                })
                continue
            rows.append(
                compare_metric(
                    season, team, metric,
                    records[team][metric],
                    table[column].iloc[position],
                    notes.get(team, ""),
                )
            )
    return rows


def check_internal_consistency(overall_table, ha_table):
    """Additional check: does FBref's own Home/Away split sum to its Overall row?

    Not requested, but it is the same two tables already loaded and it separates
    'the fixtures disagree with FBref' from 'FBref disagrees with itself'. A
    points deduction applied to one table and not the other shows up only here.
    """
    findings = []
    overall_columns = build_column_index(overall_table)
    ha_columns = build_column_index(ha_table)

    overall_squad = find_squad_column(overall_table)
    ha_squad = find_squad_column(ha_table)
    if overall_squad is None or ha_squad is None:
        return findings

    ha_positions = squad_index(ha_table, ha_squad)

    for position, team_value in enumerate(overall_table[overall_squad]):
        if pd.isna(team_value):
            continue
        team = str(team_value).strip()
        if team not in ha_positions:
            continue
        ha_position = ha_positions[team]

        for metric in ("MP", "W", "D", "L", "GF", "GA", "Pts"):
            overall_column = overall_columns.get((metric,))
            home_column = ha_columns.get(("Home", metric))
            away_column = ha_columns.get(("Away", metric))
            if overall_column is None or home_column is None or away_column is None:
                continue

            total, total_ok = strict_int(overall_table[overall_column].iloc[position])
            home, home_ok = strict_int(ha_table[home_column].iloc[ha_position])
            away, away_ok = strict_int(ha_table[away_column].iloc[ha_position])
            if not (total_ok and home_ok and away_ok):
                continue

            if home + away != total:
                findings.append({
                    "team": team,
                    "metric": metric,
                    "home": home,
                    "away": away,
                    "sum": home + away,
                    "overall": total,
                    "difference": (home + away) - total,
                })
    return findings


# --------------------------------------------------------------------------
# season handling
# --------------------------------------------------------------------------


def season_from_fixture_name(name):
    """'2021-2022 PL Season.xls' -> ('2021-2022', '2022 PL Season')."""
    match = re.search(r"(\d{4})\s*-\s*(\d{4})", name)
    if not match:
        return None, None
    return "{}-{}".format(match.group(1), match.group(2)), "{} PL Season".format(match.group(2))


def count_status(rows, *statuses):
    return sum(1 for r in rows if r["status"] in statuses)


def count_prefix(rows, prefix):
    return sum(1 for r in rows if str(r["status"]).startswith(prefix))


def print_examples(rows, label):
    bad = [
        r for r in rows
        if r["status"] != "MATCH"
    ]
    if not bad:
        return
    print("      {} ({}):".format(label, len(bad)))
    for r in bad[:MAX_EXAMPLES]:
        print(
            "        {:<22} {:<10} fixtures={:<8} fbref={:<8} diff={:<6} {}".format(
                str(r["team"])[:22], str(r["metric"]),
                str(r["fixture_value"]), str(r["fbref_value"]),
                str(r["difference"]), r["status"],
            )
        )
    if len(bad) > MAX_EXAMPLES:
        print("        ... and {} more (see the CSV)".format(len(bad) - MAX_EXAMPLES))


def main():
    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 4: CROSS-SOURCE RECONCILIATION")
    print("=" * 78)
    print("Project root : {}".format(PROJECT_ROOT))
    print("Fixtures     : {}".format(FIXTURES_DIR))
    print("Mode         : READ-ONLY (no rename, move, delete, merge, repair)")
    print("Scope        : Fixtures + Overall + Home/Away only")
    print("Points       : win {}, draw {}, loss {}".format(POINTS_WIN, POINTS_DRAW, POINTS_LOSS))
    print("Tables found : by CONTENT/SCHEMA (shared classifier), never by filename")
    print("MultiIndex   : preserved; Home/Away read via full hierarchy paths")
    print()
    print("Note: the 2026 Shooting-Opponent gap from Instrument 3 is out of scope")
    print("      here and does not block this instrument.")
    print()

    if not FIXTURES_DIR.is_dir():
        print("FATAL: fixtures directory not found: {}".format(FIXTURES_DIR))
        return 1

    fixture_files = sorted(
        [p for p in FIXTURES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".xls"],
        key=lambda p: p.name.lower(),
    )
    if not fixture_files:
        print("FATAL: no .xls fixture files in {}".format(FIXTURES_DIR))
        return 1

    overall_rows = []
    homeaway_rows = []
    summary_rows = []
    internal_findings = []

    for fixture_path in fixture_files:
        season, folder_name = season_from_fixture_name(fixture_path.name)
        print("=" * 78)
        print("SEASON {}".format(season or "UNKNOWN"))
        print("=" * 78)
        print("  fixtures    : {}".format(fixture_path.name))

        summary = {field: "" for field in SUMMARY_FIELDS}
        summary["season"] = season or fixture_path.name
        summary["fixture_file"] = fixture_path.name
        summary["stats_folder"] = folder_name or ""

        if season is None:
            print("  MAPPING ERROR: cannot read a season from this filename")
            summary["status"] = "MAPPING ERROR"
            summary_rows.append(summary)
            print()
            continue

        stats_folder = RAW_DIR / folder_name
        print("  stats folder: {}".format(folder_name))

        fixtures, error = read_table(fixture_path)
        if fixtures is None:
            print("  MAPPING ERROR: fixture file unreadable - {}".format(error))
            summary["status"] = "MAPPING ERROR"
            summary_rows.append(summary)
            print()
            continue

        # Requirement 2: drop completely blank rows before anything else.
        fixtures = fixtures.dropna(how="all")

        records, fixture_stats = reconstruct(fixtures)
        summary["fixtures_used"] = fixture_stats["rows_used"]
        summary["teams"] = len(records)

        print(
            "  rows        : {} kept after blank-drop | {} fixtures used | "
            "{} unscored | {} unparseable score | {} missing team".format(
                len(fixtures), fixture_stats["rows_used"],
                fixture_stats["rows_no_score"], fixture_stats["rows_bad_score"],
                fixture_stats["rows_missing_team"],
            )
        )
        print("  teams rebuilt: {}".format(len(records)))
        print()

        # ---------------- Overall ----------------

        print("  OVERALL RECONCILIATION")
        overall_path, overall_table, overall_note = find_table_by_type(stats_folder, "Overall")
        season_overall = []
        season_notes = {}
        if overall_table is None:
            print("      MAPPING ERROR: {}".format(overall_note))
            summary["overall_metrics_checked"] = 0
            summary["overall_mismatches"] = ""
            summary["overall_unmapped"] = "MAPPING ERROR"
            summary["overall_source"] = "NOT FOUND"
        else:
            summary["overall_source"] = overall_path.name
            print("      source: {} (content-classified)".format(overall_path.name))
            if overall_note:
                print("      note  : {}".format(overall_note))

            squad_col = find_squad_column(overall_table)
            if squad_col is None:
                print("      MAPPING ERROR: no Squad column in the Overall table")
                summary["overall_unmapped"] = "MAPPING ERROR"
            else:
                # Overall is a flat table: each metric is its own single-level column.
                paths = {m: (m,) for m in OVERALL_METRICS}
                season_notes = extract_notes(overall_table)
                season_overall = reconcile(
                    season, records, overall_table, squad_col, paths, season_notes
                )
                overall_rows.extend(season_overall)

                checked = len(season_overall)
                mismatches = count_status(season_overall, "MISMATCH")
                unknown = count_status(season_overall, "UNKNOWN")
                mapping = count_prefix(season_overall, "MAPPING ERROR")
                matches = count_status(season_overall, "MATCH")

                summary["overall_metrics_checked"] = checked
                summary["overall_mismatches"] = mismatches
                summary["overall_unmapped"] = unknown + mapping

                print(
                    "      comparisons {} | matches {} | mismatches {} | "
                    "unknown {} | mapping errors {}".format(
                        checked, matches, mismatches, unknown, mapping
                    )
                )
                print_examples(season_overall, "discrepancies")
        print()

        # ---------------- Home/Away ----------------

        print("  HOME/AWAY RECONCILIATION")
        ha_path, ha_table, ha_note = find_table_by_type(stats_folder, "Home/Away")
        season_ha = []
        if ha_table is None:
            print("      MAPPING ERROR: {}".format(ha_note))
            summary["homeaway_metrics_checked"] = 0
            summary["homeaway_mismatches"] = ""
            summary["homeaway_unmapped"] = "MAPPING ERROR"
            summary["homeaway_source"] = "NOT FOUND"
        else:
            summary["homeaway_source"] = ha_path.name
            print("      source: {} (content-classified)".format(ha_path.name))
            if ha_note:
                print("      note  : {}".format(ha_note))

            squad_col = find_squad_column(ha_table)
            if squad_col is None:
                print("      MAPPING ERROR: no Squad column in the Home/Away table")
                summary["homeaway_unmapped"] = "MAPPING ERROR"
            else:
                # Full hierarchy paths. ('Home','MP') and ('Away','MP') share a
                # leaf name - flattening would collapse them into one column.
                paths = {}
                for venue in ("Home", "Away"):
                    for metric in ("MP", "W", "D", "L", "GF", "GA", "GD", "Pts"):
                        paths["{}_{}".format(venue, metric)] = (venue, metric)

                season_ha = reconcile(
                    season, records, ha_table, squad_col, paths, season_notes
                )
                homeaway_rows.extend(season_ha)

                checked = len(season_ha)
                mismatches = count_status(season_ha, "MISMATCH")
                unknown = count_status(season_ha, "UNKNOWN")
                mapping = count_prefix(season_ha, "MAPPING ERROR")
                matches = count_status(season_ha, "MATCH")

                summary["homeaway_metrics_checked"] = checked
                summary["homeaway_mismatches"] = mismatches
                summary["homeaway_unmapped"] = unknown + mapping

                print(
                    "      comparisons {} | matches {} | mismatches {} | "
                    "unknown {} | mapping errors {}".format(
                        checked, matches, mismatches, unknown, mapping
                    )
                )
                print_examples(season_ha, "discrepancies")
        print()

        # ---------------- FBref vs itself ----------------

        if overall_table is not None and ha_table is not None:
            internal = check_internal_consistency(overall_table, ha_table)
            print("  FBREF INTERNAL CONSISTENCY (Home + Away vs Overall)")
            if not internal:
                print("      consistent: every Home+Away total equals its Overall row")
            else:
                print(
                    "      {} value(s) where FBref's own two tables disagree:".format(
                        len(internal)
                    )
                )
                for item in internal:
                    print(
                        "        {:<22} {:<5} home {} + away {} = {} but Overall = {} "
                        "(diff {:+d})".format(
                            item["team"][:22], item["metric"], item["home"],
                            item["away"], item["sum"], item["overall"],
                            item["difference"],
                        )
                    )
                internal_findings.append((season, internal))
            print()

        combined = season_overall + season_ha
        clean = (
            len(combined) > 0
            and all(r["status"] == "MATCH" for r in combined)
            and overall_table is not None
            and ha_table is not None
        )
        summary["status"] = "PASS" if clean else "FAIL"
        summary_rows.append(summary)

        print("  SEASON {} -> {}".format(season, summary["status"]))
        print()

    # ---------------- write reports ----------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(overall_rows, columns=RECON_FIELDS).to_csv(
        OVERALL_PATH, index=False, encoding="utf-8"
    )
    pd.DataFrame(homeaway_rows, columns=RECON_FIELDS).to_csv(
        HOMEAWAY_PATH, index=False, encoding="utf-8"
    )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_FIELDS)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")

    # ---------------- summary ----------------

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    header = "  {:<12}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
        "season", "ovr chk", "ovr mis", "h/a chk", "h/a mis", "status"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in summary.iterrows():
        print(
            "  {:<12}{:>10}{:>10}{:>10}{:>10}{:>10}".format(
                str(r["season"]),
                str(r["overall_metrics_checked"]),
                str(r["overall_mismatches"]),
                str(r["homeaway_metrics_checked"]),
                str(r["homeaway_mismatches"]),
                str(r["status"]),
            )
        )
    print()

    all_rows = overall_rows + homeaway_rows
    total = len(all_rows)
    mismatches = count_status(all_rows, "MISMATCH")
    unknown = count_status(all_rows, "UNKNOWN")
    mapping = count_prefix(all_rows, "MAPPING ERROR")

    print("  Total comparisons : {}".format(total))
    print("  Agreements        : {}".format(count_status(all_rows, "MATCH")))
    print("  Mismatches        : {}".format(mismatches))
    print("  Unknown values    : {}".format(unknown))
    print("  Mapping errors    : {}".format(mapping))
    print()

    if mismatches or unknown or mapping:
        print("-" * 78)
        print("EVERY DISCREPANCY")
        print("-" * 78)
        for r in all_rows:
            if r["status"] != "MATCH":
                print(
                    "  [{}] {:<22} {:<10} fixtures={:<8} fbref={:<8} diff={:<6} {}".format(
                        r["season"], str(r["team"])[:22], str(r["metric"]),
                        str(r["fixture_value"]), str(r["fbref_value"]),
                        str(r["difference"]), r["status"],
                    )
                )
                if r["fbref_note"]:
                    print("        source note: {}".format(r["fbref_note"]))
        print()

        print("-" * 78)
        print("DISCREPANCY CONTEXT")
        print("-" * 78)
        print("  Reported, not resolved. The status above is unchanged by anything here.")
        print()
        for r in all_rows:
            if r["status"] == "MISMATCH" and r["fbref_note"]:
                print(
                    "  [{}] {} {}: fixtures give {}, FBref publishes {}.".format(
                        r["season"], r["team"], r["metric"],
                        r["fixture_value"], r["fbref_value"],
                    )
                )
                print(
                    "      FBref's own Notes column for this team reads: \"{}\"".format(
                        r["fbref_note"]
                    )
                )
                print(
                    "      The reconstruction counts points earned on the pitch; the"
                )
                print(
                    "      published total is after that sanction. Both are correct at"
                )
                print(
                    "      what they measure. Nothing here was adjusted."
                )
                print()

    if internal_findings:
        print("-" * 78)
        print("FBREF INTERNAL INCONSISTENCY")
        print("-" * 78)
        print("  Where FBref's Home/Away split does not sum to its own Overall row.")
        print("  This is a disagreement inside one source, not between two sources.")
        print()
        for season, items in internal_findings:
            for item in items:
                print(
                    "  [{}] {} {}: home {} + away {} = {}, Overall says {} (diff {:+d})".format(
                        season, item["team"], item["metric"], item["home"],
                        item["away"], item["sum"], item["overall"], item["difference"],
                    )
                )
        print()
        print("  Consequence for Phase 1: a per-venue points feature built from the")
        print("  Home/Away table will not add up to a season points feature read from")
        print("  the Overall table for these teams. Derive points from results and")
        print("  handle sanctions explicitly, or the two features will silently conflict.")
        print()

    passed = list(summary["status"]).count("PASS")
    overall_pass = passed == len(summary) and len(summary) > 0

    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 4 STATUS")
    print("=" * 78)
    print()
    if overall_pass:
        print("  PASS")
        print()
        print("  All {} seasons reconcile completely.".format(len(summary)))
        print("  Every league-table value FBref publishes is reproducible from the")
        print("  fixture results alone, for both the overall and the home/away splits.")
    else:
        print("  FAIL / INVESTIGATE")
        print()
        print("  {} of {} seasons reconciled.".format(passed, len(summary)))
        print("  Discrepancies are listed above with team, metric and both values.")
        print("  Nothing was corrected.")
    print()
    print("  Reports written:")
    print("    {}".format(OVERALL_PATH))
    print("    {}".format(HOMEAWAY_PATH))
    print("    {}".format(SUMMARY_PATH))
    print()
    print("No source data was modified.")
    print("No files were renamed.")
    print("No files were deleted.")
    print("No discrepancy was repaired, smoothed, or hidden.")
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
