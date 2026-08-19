from pathlib import Path


# ============================================================
# PHASE 0 — DATA AUDIT
# Instrument 1: Project Structure & Data Coverage
#
# IMPORTANT:
# This script ONLY audits.
# It does NOT modify, rename, move, delete, or clean data.
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FIXTURES_DIR = RAW_DIR / "Fixtures"


# ------------------------------------------------------------
# 2. EXPECTED DATA
# ------------------------------------------------------------

EXPECTED_RAW_SEASONS = {
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
}

EXPECTED_FIXTURE_SEASONS = {
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
}


# ------------------------------------------------------------
# 3. HELPER
# ------------------------------------------------------------

def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def check_directory(path):
    if path.exists() and path.is_dir():
        print(f"✓ Found: {path}")
        return True

    print(f"✗ MISSING: {path}")
    return False


# ============================================================
# START AUDIT
# ============================================================

section("PHASE 0 — PREMIER LEAGUE ML DATA AUDIT")

print(f"Project root: {PROJECT_ROOT}")


# ============================================================
# 1. DIRECTORY STRUCTURE
# ============================================================

section("1. DIRECTORY STRUCTURE")

data_exists = check_directory(DATA_DIR)
raw_exists = check_directory(RAW_DIR)
fixtures_exists = check_directory(FIXTURES_DIR)


# ============================================================
# 2. RAW STATISTICAL SEASON FOLDERS
# ============================================================

section("2. RAW STATISTICAL SEASONS")

raw_season_folders = []

if raw_exists:

    for item in sorted(RAW_DIR.iterdir()):

        # Only directories that look like season folders
        if item.is_dir() and item.name.endswith(" PL Season"):

            raw_season_folders.append(item)

            print(f"\n{item.name}")

            files = [
                file for file in item.iterdir()
                if file.is_file()
            ]

            print(f"  Files: {len(files)}")

            for file in sorted(files):
                print(f"    - {file.name}")


actual_raw_seasons = {
    folder.name.replace(" PL Season", "")
    for folder in raw_season_folders
}


# ============================================================
# 3. RAW SEASON COVERAGE
# ============================================================

section("3. RAW SEASON COVERAGE")

print("Expected:")
for season in sorted(EXPECTED_RAW_SEASONS):
    print(f"  - {season}")

print("\nFound:")
for season in sorted(actual_raw_seasons):
    print(f"  - {season}")


missing_raw = EXPECTED_RAW_SEASONS - actual_raw_seasons
unexpected_raw = actual_raw_seasons - EXPECTED_RAW_SEASONS


if not missing_raw and not unexpected_raw:

    print("\n✓ Raw season coverage PASS")

else:

    if missing_raw:
        print("\n✗ Missing raw seasons:")
        for season in sorted(missing_raw):
            print(f"    - {season}")

    if unexpected_raw:
        print("\n⚠ Unexpected raw seasons:")
        for season in sorted(unexpected_raw):
            print(f"    - {season}")


# ============================================================
# 4. FIXTURE FILES
# ============================================================

section("4. FIXTURE DATASETS")

fixture_files = []

if fixtures_exists:

    fixture_files = [
        file
        for file in sorted(FIXTURES_DIR.iterdir())
        if file.is_file()
    ]

    for file in fixture_files:
        print(f"- {file.name}")

print(f"\nFixture datasets found: {len(fixture_files)}")


# ============================================================
# 5. FIXTURE SEASON COVERAGE
# ============================================================

section("5. FIXTURE SEASON COVERAGE")

actual_fixture_seasons = set()

for file in fixture_files:

    name = file.stem

    if " PL Season" in name:

        season = name.replace(" PL Season", "")
        actual_fixture_seasons.add(season)


print("Expected:")
for season in sorted(EXPECTED_FIXTURE_SEASONS):
    print(f"  - {season}")

print("\nFound:")
for season in sorted(actual_fixture_seasons):
    print(f"  - {season}")


missing_fixture = EXPECTED_FIXTURE_SEASONS - actual_fixture_seasons
unexpected_fixture = actual_fixture_seasons - EXPECTED_FIXTURE_SEASONS


if not missing_fixture and not unexpected_fixture:

    print("\n✓ Fixture season coverage PASS")

else:

    if missing_fixture:
        print("\n✗ Missing fixture seasons:")
        for season in sorted(missing_fixture):
            print(f"    - {season}")

    if unexpected_fixture:
        print("\n⚠ Unexpected fixture seasons:")
        for season in sorted(unexpected_fixture):
            print(f"    - {season}")


# ============================================================
# 6. FIXTURE COPIES OUTSIDE FIXTURES FOLDER
# ============================================================

section("6. POSSIBLE DUPLICATE FIXTURE DATASETS")

root_fixture_candidates = []

if raw_exists:

    for file in RAW_DIR.iterdir():

        if file.is_file() and file.suffix.lower() == ".xls":

            if " PL Season" in file.stem:
                root_fixture_candidates.append(file)


if root_fixture_candidates:

    print("⚠ Fixture-looking files found directly inside data/raw:")

    for file in root_fixture_candidates:
        print(f"  - {file.relative_to(PROJECT_ROOT)}")

    print("\nThese are NOT deleted or modified.")
    print("They need to be investigated before we decide whether")
    print("they are duplicate copies or separate source data.")

else:

    print("✓ No fixture-looking files found outside data/raw/Fixtures.")


# ============================================================
# 7. FILE COUNT PER STATISTICAL SEASON
# ============================================================

section("7. STATISTICAL FILE COUNTS")

for folder in raw_season_folders:

    files = [
        file
        for file in folder.iterdir()
        if file.is_file()
    ]

    print(
        f"{folder.name:<25} "
        f"{len(files):>3} files"
    )


# ============================================================
# 8. BASIC FILE EXISTENCE CHECK
# ============================================================

section("8. EMPTY FILE CHECK")

all_data_files = []

if raw_exists:
    all_data_files.extend(
        file
        for file in RAW_DIR.rglob("*")
        if file.is_file()
    )


empty_files = []

for file in all_data_files:

    if file.stat().st_size == 0:
        empty_files.append(file)


if empty_files:

    print("✗ Empty files found:")

    for file in empty_files:
        print(f"  - {file.relative_to(PROJECT_ROOT)}")

else:

    print("✓ No empty files found.")


# ============================================================
# 9. COVERAGE MATRIX
# ============================================================

section("9. DATA COVERAGE MATRIX")

print(
    f"{'Season':<12}"
    f"{'Fixtures':<12}"
    f"{'Stats':<12}"
)

print("-" * 36)

coverage = [
    ("2021-2022", "2021-2022", "2021"),
    ("2022-2023", "2022-2023", "2022"),
    ("2023-2024", "2023-2024", "2023"),
    ("2024-2025", "2024-2025", "2024"),
    ("2025-2026", "2025-2026", "2025"),
]

for display_season, fixture_season, raw_season in coverage:

    fixture_status = (
        "YES"
        if fixture_season in actual_fixture_seasons
        else "NO"
    )

    stats_status = (
        "YES"
        if raw_season in actual_raw_seasons
        else "NO"
    )

    print(
        f"{display_season:<12}"
        f"{fixture_status:<12}"
        f"{stats_status:<12}"
    )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

section("10. PHASE 0 — INSTRUMENT 1 SUMMARY")

print(f"Raw statistical seasons: {len(actual_raw_seasons)}")
print(f"Fixture seasons:         {len(actual_fixture_seasons)}")
print(f"Fixture datasets:        {len(fixture_files)}")
print(f"Empty files:             {len(empty_files)}")
print(
    f"Fixture copies outside "
    f"Fixtures folder:        {len(root_fixture_candidates)}"
)

print("\nSTATUS:")

if (
    not missing_raw
    and not missing_fixture
    and not empty_files
):

    print("✓ BASIC STRUCTURE PASS")

else:

    print("⚠ INVESTIGATION REQUIRED")


print("\nNo data was modified.")
print("No files were renamed.")
print("No files were deleted.")

print("\nNext instrument:")
print("→ Fixture integrity validation")