from pathlib import Path
import pandas as pd
import re


# ============================================================
# PHASE 0 — FIXTURE INTEGRITY VALIDATOR
# Instrument 2.1
#
# Purpose:
#   Validate the actual match-level fixture data.
#
# IMPORTANT:
#   This script NEVER modifies the source files.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = PROJECT_ROOT / "data" / "raw" / "Fixtures"


EXPECTED_MATCHES = 380
EXPECTED_TEAMS = 20
EXPECTED_MATCHES_PER_TEAM = 38
EXPECTED_HOME_MATCHES = 19
EXPECTED_AWAY_MATCHES = 19


def section(title):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def normalize_column(column):
    column = str(column)
    column = column.replace("\n", " ")
    column = column.strip().lower()
    column = re.sub(r"\s+", " ", column)
    return column


def find_column(columns, possible_names):

    normalized = {
        normalize_column(column): column
        for column in columns
    }

    for name in possible_names:

        name = normalize_column(name)

        if name in normalized:
            return normalized[name]

    return None


def load_fixture_table(file_path):

    tables = pd.read_html(file_path)

    if not tables:
        raise ValueError("No HTML tables found.")

    for table in tables:

        columns = [
            normalize_column(col)
            for col in table.columns
        ]

        has_date = any(
            col == "date" or col.startswith("date ")
            for col in columns
        )

        has_score = any(
            col == "score"
            for col in columns
        )

        if has_date and has_score:
            return table

    return max(tables, key=len)


def clean_team_name(name):

    if pd.isna(name):
        return None

    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)

    return name


def parse_score(score):

    if pd.isna(score):
        return None, None

    score = str(score).strip()

    # Normal hyphen
    match = re.search(
        r"(\d+)\s*-\s*(\d+)",
        score
    )

    if match:
        return int(match.group(1)), int(match.group(2))

    # En dash
    match = re.search(
        r"(\d+)\s*–\s*(\d+)",
        score
    )

    if match:
        return int(match.group(1)), int(match.group(2))

    # Em dash
    match = re.search(
        r"(\d+)\s*—\s*(\d+)",
        score
    )

    if match:
        return int(match.group(1)), int(match.group(2))

    # Common mojibake produced when UTF-8 score separator
    # is incorrectly decoded.
    match = re.search(
        r"(\d+)\s*â.{0,3}(\d+)",
        score
    )

    if match:
        return int(match.group(1)), int(match.group(2))

    return None, None


def is_blank_fixture_row(row, date_col, home_col, score_col, away_col):

    values = [
        row[date_col],
        row[home_col],
        row[score_col],
        row[away_col],
    ]

    return all(pd.isna(value) for value in values)


# ============================================================
# START
# ============================================================

section("PHASE 0 — FIXTURE INTEGRITY VALIDATION")

print(f"Fixtures directory:")
print(FIXTURES_DIR)


if not FIXTURES_DIR.exists():

    raise FileNotFoundError(
        f"Fixtures directory does not exist:\n{FIXTURES_DIR}"
    )


fixture_files = sorted(
    file
    for file in FIXTURES_DIR.iterdir()
    if file.is_file()
    and file.suffix.lower() == ".xls"
)


print(f"\nFixture files found: {len(fixture_files)}")


season_results = []


# ============================================================
# EACH SEASON
# ============================================================

for file_path in fixture_files:

    section(f"VALIDATING: {file_path.name}")

    season_match = re.search(
        r"(20\d{2}-20\d{2})",
        file_path.stem
    )

    season = (
        season_match.group(1)
        if season_match
        else "UNKNOWN"
    )

    print(f"Season: {season}")

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    try:

        df = load_fixture_table(file_path)

    except Exception as error:

        print("✗ FAILED TO READ FILE")
        print(f"  {error}")

        season_results.append({
            "season": season,
            "rows": None,
            "blank_rows": None,
            "fixtures": None,
            "teams": None,
            "status": "READ ERROR"
        })

        continue


    rows_read = len(df)

    print(f"Rows read: {rows_read}")


    # --------------------------------------------------------
    # COLUMNS
    # --------------------------------------------------------

    date_col = find_column(
        df.columns,
        ["Date"]
    )

    score_col = find_column(
        df.columns,
        ["Score"]
    )

    home_col = find_column(
        df.columns,
        ["Home", "Home Team"]
    )

    away_col = find_column(
        df.columns,
        ["Away", "Away Team"]
    )


    print("\nDetected columns:")
    print(f"  Date:  {date_col}")
    print(f"  Score: {score_col}")
    print(f"  Home:  {home_col}")
    print(f"  Away:  {away_col}")


    required_columns = [
        date_col,
        score_col,
        home_col,
        away_col
    ]


    if any(column is None for column in required_columns):

        print("\n✗ SCHEMA ERROR")

        season_results.append({
            "season": season,
            "rows": rows_read,
            "blank_rows": None,
            "fixtures": None,
            "teams": None,
            "status": "SCHEMA ERROR"
        })

        continue


    # ========================================================
    # REMOVE BLANK ROWS FOR VALIDATION ONLY
    # ========================================================

    blank_mask = df.apply(
        lambda row:
            is_blank_fixture_row(
                row,
                date_col,
                home_col,
                score_col,
                away_col
            ),
        axis=1
    )

    blank_rows = int(blank_mask.sum())

    matches = df.loc[~blank_mask].copy()

    actual_matches = len(matches)


    print("\n1. ROW CLASSIFICATION")

    print(f"Rows read:       {rows_read}")
    print(f"Blank rows:      {blank_rows}")
    print(f"Actual fixtures: {actual_matches}")
    print(f"Expected:        {EXPECTED_MATCHES}")


    if actual_matches == EXPECTED_MATCHES:
        print("✓ MATCH COUNT PASS")
    else:
        print("✗ MATCH COUNT FAIL")


    # ========================================================
    # DATE VALIDATION
    # ========================================================

    print("\n2. DATE VALIDATION")

    dates = pd.to_datetime(
        matches[date_col],
        errors="coerce"
    )

    invalid_dates = int(dates.isna().sum())

    print(f"Invalid/missing dates: {invalid_dates}")

    if invalid_dates == 0:
        print("✓ PASS")
    else:
        print("✗ FAIL")


    # ========================================================
    # CHRONOLOGICAL ORDER
    # ========================================================

    print("\n3. CHRONOLOGICAL ORDER")

    valid_dates = dates.dropna()

    chronological = (
        valid_dates.is_monotonic_increasing
        if len(valid_dates) > 1
        else False
    )

    print(f"Chronological: {chronological}")

    if chronological:
        print("✓ PASS")
    else:
        print("✗ FAIL")


    # ========================================================
    # TEAM COMPLETENESS
    # ========================================================

    print("\n4. TEAM COMPLETENESS")

    missing_home = int(
        matches[home_col].isna().sum()
    )

    missing_away = int(
        matches[away_col].isna().sum()
    )

    print(f"Missing home teams: {missing_home}")
    print(f"Missing away teams: {missing_away}")


    # ========================================================
    # TEAM COUNT
    # ========================================================

    home_teams = (
        matches[home_col]
        .map(clean_team_name)
    )

    away_teams = (
        matches[away_col]
        .map(clean_team_name)
    )

    teams = sorted(
        set(home_teams.dropna()) |
        set(away_teams.dropna())
    )

    print(f"\nUnique teams: {len(teams)}")
    print(f"Expected teams: {EXPECTED_TEAMS}")

    if len(teams) == EXPECTED_TEAMS:
        print("✓ PASS")
    else:
        print("✗ FAIL")


    # ========================================================
    # DUPLICATE FIXTURES
    # ========================================================

    print("\n5. DUPLICATE FIXTURES")

    duplicate_mask = matches.duplicated(
        subset=[
            date_col,
            home_col,
            away_col
        ],
        keep=False
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    print(
        f"Duplicate fixture rows: "
        f"{duplicate_rows}"
    )

    if duplicate_rows == 0:

        print("✓ PASS")

    else:

        print("✗ FAIL")

        print(
            matches.loc[
                duplicate_mask,
                [
                    date_col,
                    home_col,
                    score_col,
                    away_col
                ]
            ].to_string(index=False)
        )


    # ========================================================
    # HOME / AWAY BALANCE
    # ========================================================

    print("\n6. HOME/AWAY BALANCE")

    home_counts = home_teams.value_counts()
    away_counts = away_teams.value_counts()

    total_counts = (
        home_counts
        .add(away_counts, fill_value=0)
    )

    home_problems = home_counts[
        home_counts != EXPECTED_HOME_MATCHES
    ]

    away_problems = away_counts[
        away_counts != EXPECTED_AWAY_MATCHES
    ]

    total_problems = total_counts[
        total_counts != EXPECTED_MATCHES_PER_TEAM
    ]


    print(
        f"Teams with incorrect total matches: "
        f"{len(total_problems)}"
    )

    print(
        f"Teams with incorrect home matches: "
        f"{len(home_problems)}"
    )

    print(
        f"Teams with incorrect away matches: "
        f"{len(away_problems)}"
    )


    balance_pass = (
        len(total_problems) == 0
        and len(home_problems) == 0
        and len(away_problems) == 0
    )


    print(
        "✓ PASS"
        if balance_pass
        else "✗ FAIL"
    )


    # ========================================================
    # SCORE VALIDATION
    # ========================================================

    print("\n7. SCORE VALIDATION")

    parsed_scores = []

    invalid_scores = []

    for index, score in matches[score_col].items():

        home_goals, away_goals = parse_score(score)

        if (
            home_goals is None
            or away_goals is None
        ):

            invalid_scores.append(
                (index, score)
            )

        else:

            parsed_scores.append(
                (index, home_goals, away_goals)
            )


    print(
        f"Invalid/missing scores: "
        f"{len(invalid_scores)}"
    )

    if len(invalid_scores) == 0:

        print("✓ PASS")

    else:

        print("✗ FAIL")

        print("\nExamples:")

        for index, score in invalid_scores[:10]:

            print(
                f"  Row {index}: {score}"
            )


    # ========================================================
    # GOAL SANITY
    # ========================================================

    print("\n8. GOAL SANITY")

    negative_goals = []

    for index, home_goals, away_goals in parsed_scores:

        if home_goals < 0 or away_goals < 0:

            negative_goals.append(
                (index, home_goals, away_goals)
            )


    print(
        f"Negative goal values: "
        f"{len(negative_goals)}"
    )

    if not negative_goals:
        print("✓ PASS")
    else:
        print("✗ FAIL")


    # ========================================================
    # FINAL SEASON STATUS
    # ========================================================

    season_pass = (
        actual_matches == EXPECTED_MATCHES
        and invalid_dates == 0
        and chronological
        and missing_home == 0
        and missing_away == 0
        and len(teams) == EXPECTED_TEAMS
        and duplicate_rows == 0
        and balance_pass
        and len(invalid_scores) == 0
        and len(negative_goals) == 0
    )


    status = (
        "PASS"
        if season_pass
        else "FAIL / INVESTIGATE"
    )


    print(f"\nSEASON STATUS: {status}")


    season_results.append({
        "season": season,
        "rows": rows_read,
        "blank_rows": blank_rows,
        "fixtures": actual_matches,
        "teams": len(teams),
        "invalid_dates": invalid_dates,
        "duplicate_rows": duplicate_rows,
        "invalid_scores": len(invalid_scores),
        "status": status
    })


# ============================================================
# FINAL SUMMARY
# ============================================================

section("FINAL FIXTURE VALIDATION SUMMARY")

summary_df = pd.DataFrame(
    season_results
)

if not summary_df.empty:

    print(
        summary_df.to_string(
            index=False
        )
    )


# ============================================================
# FINAL STATUS
# ============================================================

section("PHASE 0 — INSTRUMENT 2 STATUS")

if (
    len(summary_df) == 5
    and all(
        summary_df["status"] == "PASS"
    )
):

    print("✓ ALL FIVE FIXTURE DATASETS PASSED")

else:

    print(
        "⚠ FIXTURE VALIDATION "
        "REQUIRES INVESTIGATION"
    )


print("\nNo source data was modified.")
print("No files were renamed.")
print("No files were deleted.")

print("\nNext instrument:")
print("→ Cross-source reconciliation")