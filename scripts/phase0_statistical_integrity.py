"""
PHASE 0 - INSTRUMENT 3: STATISTICAL SOURCE INTEGRITY

Validates the statistical source layer in data/raw/<season>/ without touching it.

STRICTLY READ-ONLY. This instrument does not rename, move, delete, modify, merge,
clean, or repair anything. It does not flatten MultiIndex columns, does not infer
missing data, and does not substitute values. Findings are reported, never fixed.

Two rules drive the design:

  1. FILENAMES ARE NOT EVIDENCE. Table type is decided from the parsed column
     hierarchy alone. Two files in this dataset are known to be named after the
     wrong statistic, so any filename-driven classification would be wrong.

  2. SQUAD vs OPPONENT IS DECIDED FROM THE DATA. FBref writes opponent tables
     with every squad value prefixed "vs " ("vs Arsenal"). Squad tables carry the
     bare club name. That prefix is read off the rows themselves. If a table is
     neither wholly prefixed nor wholly bare, it is labelled UNKNOWN rather than
     guessed at.

The filename IS parsed, but only to compare against the content-derived answer so
mismatches can be reported. It never decides anything.
"""

import hashlib
import io
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTEGRITY_PATH = PROJECT_ROOT / "outputs" / "phase0_statistical_integrity.csv"
COVERAGE_PATH = PROJECT_ROOT / "outputs" / "phase0_statistical_coverage.csv"

SEASON_FOLDERS = [
    "2022 PL Season",
    "2023 PL Season",
    "2024 PL Season",
    "2025 PL Season",
    "2026 PL Season",
]

EXPECTED_ROWS = 20
EXPECTED_SQUADS = 20

ENCODINGS = ["utf-8", "cp1252", "latin-1"]
FLAVORS = ["lxml", "bs4"]

# Coverage slots, in report order. Each season is expected to supply all twelve.
COVERAGE_SLOTS = [
    "Overall",
    "Home_Away",
    "Standard_Squad",
    "Standard_Opponent",
    "Shooting_Squad",
    "Shooting_Opponent",
    "Goalkeeping_Squad",
    "Goalkeeping_Opponent",
    "PlayingTime_Squad",
    "PlayingTime_Opponent",
    "Miscellaneous_Squad",
    "Miscellaneous_Opponent",
]

# (table_type, perspective) -> coverage slot
SLOT_BY_TYPE = {
    ("Overall", "League"): "Overall",
    ("Home/Away", "League"): "Home_Away",
    ("Standard", "Squad"): "Standard_Squad",
    ("Standard", "Opponent"): "Standard_Opponent",
    ("Shooting", "Squad"): "Shooting_Squad",
    ("Shooting", "Opponent"): "Shooting_Opponent",
    ("Goalkeeping", "Squad"): "Goalkeeping_Squad",
    ("Goalkeeping", "Opponent"): "Goalkeeping_Opponent",
    ("Playing Time", "Squad"): "PlayingTime_Squad",
    ("Playing Time", "Opponent"): "PlayingTime_Opponent",
    ("Miscellaneous", "Squad"): "Miscellaneous_Squad",
    ("Miscellaneous", "Opponent"): "Miscellaneous_Opponent",
}

# Table types that describe the league as a whole and have no Squad/Opponent pair.
LEAGUE_LEVEL_TYPES = ("Overall", "Home/Away")

OPPONENT_PREFIX = "vs "

FIELDS = [
    "season",
    "filename",
    "relative_path",
    "table_type",
    "perspective",
    "coverage_slot",
    "rows",
    "cols",
    "unique_squads",
    "missing_squads",
    "duplicate_squads",
    "n_tables_found",
    "columns_are_multiindex",
    "n_header_levels",
    "schema_label",
    "schema_signature",
    "schema_columns",
    "content_md5",
    "empty_cells",
    "filename_suggests",
    "filename_content_mismatch",
    "status",
    "findings",
]


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


def decode(raw_bytes):
    """Return (text, encoding). Reading only - the file on disk is never rewritten."""
    for enc in ENCODINGS:
        try:
            return raw_bytes.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("latin-1", errors="replace"), "latin-1(replace)"


def parse_tables(text):
    """Return (tables, flavor, error). Tries lxml then bs4; never edits the file."""
    last_error = None
    for flavor in FLAVORS:
        try:
            tables = pd.read_html(io.StringIO(text), flavor=flavor)
        except Exception as exc:
            last_error = "{}: {}".format(type(exc).__name__, exc)
            continue
        if len(tables) == 0:
            last_error = "read_html returned 0 tables (flavor={})".format(flavor)
            continue
        return tables, flavor, None
    return None, None, last_error


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


def column_path(col):
    """Full hierarchy of one column as a tuple of level strings. Never flattened
    to a single level - 'Performance | Gls' and 'Per 90 Minutes | Gls' must stay
    distinct, and in several schemas their leaf names collide."""
    if isinstance(col, tuple):
        return tuple(str(x) for x in col)
    return (str(col),)


def column_paths(table):
    return [column_path(c) for c in table.columns]


def render_path(path):
    return " | ".join(path)


def schema_signature(table):
    """Canonical signature over the FULL column hierarchy, order included."""
    payload = json.dumps([list(p) for p in column_paths(table)], ensure_ascii=False)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16], payload


def content_hash(table):
    """Fingerprint of the parsed table as-read, for duplicate detection."""
    return hashlib.md5(table.to_csv(index=False).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# classification - from CONTENT ONLY
# --------------------------------------------------------------------------


def classify_table_type(table):
    """Decide the table type from the column hierarchy alone.

    Every rule below is a structural marker unique to one FBref table. If nothing
    matches, the answer is UNKNOWN - the instrument does not guess.
    """
    paths = column_paths(table)
    rendered = set(render_path(p) for p in paths)
    top_levels = set(p[0] for p in paths if len(p) > 1)
    leaves = set(p[-1] for p in paths)

    # Home/Away: the only schema that groups identical stat blocks under Home and Away.
    if {"Home", "Away"}.issubset(top_levels):
        return "Home/Away"

    # Overall: the flat league table. Attendance and Top Team Scorer appear nowhere else.
    if table.columns.nlevels == 1 and {"Attendance", "Top Team Scorer", "Pts"}.issubset(leaves):
        return "Overall"

    # Goalkeeping: shots-against and saves, under a Penalty Kicks block.
    if {"Performance | SoTA", "Performance | Saves", "Performance | CS"}.issubset(rendered):
        return "Goalkeeping"

    # Shooting: the Standard block carrying shot volume and conversion.
    if {"Standard | Sh", "Standard | SoT", "Standard | G/Sh"}.issubset(rendered):
        return "Shooting"

    # Standard: the only schema with a Per 90 Minutes block alongside Poss.
    if "Per 90 Minutes" in top_levels and "Performance | Gls" in rendered:
        return "Standard"

    # Playing Time: rotation and on-pitch team success.
    if {"Subs", "Team Success"}.issubset(top_levels):
        return "Playing Time"

    # Miscellaneous: discipline and duels, with no Playing Time block.
    if {"Performance | Fls", "Performance | Crs", "Performance | TklW"}.issubset(rendered):
        return "Miscellaneous"

    return "UNKNOWN"


def find_squad_column(table):
    """Locate the existing Squad/Team column. Detection only - nothing renamed."""
    for col in table.columns:
        for level in column_path(col):
            if level.strip().lower() in ("squad", "team"):
                return col
    return None


def classify_perspective(table, squad_col, table_type):
    """Squad or Opponent, read off the ROWS, not the filename.

    FBref prefixes every row of an opponent table with 'vs '. All-prefixed means
    Opponent, none-prefixed means Squad, anything in between means UNKNOWN.
    """
    if table_type in LEAGUE_LEVEL_TYPES:
        return "League"
    if table_type == "UNKNOWN" or squad_col is None:
        return "UNKNOWN"

    values = table[squad_col].dropna().astype(str).str.strip()
    if len(values) == 0:
        return "UNKNOWN"

    prefixed = values.str.startswith(OPPONENT_PREFIX).sum()
    if prefixed == len(values):
        return "Opponent"
    if prefixed == 0:
        return "Squad"
    return "UNKNOWN"


# --------------------------------------------------------------------------
# filename reading - for MISMATCH REPORTING ONLY, never for classification
# --------------------------------------------------------------------------


def read_filename_hint(filename):
    """What the filename claims. Used only to compare against the content answer."""
    name = filename.lower()

    if "overall" in name:
        kind = "Overall"
    elif "home" in name and "away" in name:
        kind = "Home/Away"
    elif "goalkeep" in name:
        kind = "Goalkeeping"
    elif "shoot" in name or "sthoot" in name:  # 'Sthooting' typo exists in 2023
        kind = "Shooting"
    elif "standard" in name:
        kind = "Standard"
    elif "playing time" in name:
        kind = "Playing Time"
    elif "misc" in name:
        kind = "Miscellaneous"
    else:
        return "UNKNOWN", "UNKNOWN"

    if kind in LEAGUE_LEVEL_TYPES:
        return kind, "League"

    # 'oppon' catches Opponent, Opponents and the 'Opponet' typo in 2024.
    if "oppon" in name:
        return kind, "Opponent"

    # 2022 abbreviates the pair as a trailing ' O' / ' S' instead of a full word.
    stem = Path(name).stem.strip()
    if stem.endswith(" o"):
        return kind, "Opponent"
    if stem.endswith(" s"):
        return kind, "Squad"

    return kind, "Squad"


# --------------------------------------------------------------------------
# per-file inspection
# --------------------------------------------------------------------------


def blank_row():
    return {field: "" for field in FIELDS}


def inspect_file(season, path):
    row = blank_row()
    row["season"] = season
    row["filename"] = path.name
    row["relative_path"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")

    findings = []

    try:
        raw = path.read_bytes()
    except Exception as exc:
        row["status"] = "UNREADABLE"
        row["findings"] = "read failed: {}: {}".format(type(exc).__name__, exc)
        return row, None

    text, _ = decode(raw)
    tables, _, error = parse_tables(text)
    if tables is None:
        row["status"] = "UNREADABLE"
        row["findings"] = "parse failed: {}".format(error)
        return row, None

    row["n_tables_found"] = len(tables)
    if len(tables) != 1:
        findings.append(
            "expected exactly 1 HTML table, found {}".format(len(tables))
        )

    # With more than one table the largest is examined, and the count above records
    # that the file was not the single-table shape the layer is supposed to have.
    table = max(tables, key=lambda t: t.shape[0] * t.shape[1])

    row["rows"] = table.shape[0]
    row["cols"] = table.shape[1]
    row["columns_are_multiindex"] = isinstance(table.columns, pd.MultiIndex)
    row["n_header_levels"] = table.columns.nlevels

    signature, payload = schema_signature(table)
    row["schema_signature"] = signature
    row["schema_columns"] = payload
    row["content_md5"] = content_hash(table)
    row["empty_cells"] = int(table.isna().sum().sum())

    if table.shape[0] != EXPECTED_ROWS:
        findings.append(
            "expected {} rows, found {}".format(EXPECTED_ROWS, table.shape[0])
        )

    squad_col = find_squad_column(table)
    if squad_col is None:
        findings.append("no Squad/Team column present")
        row["unique_squads"] = "N/A"
        row["missing_squads"] = "N/A"
        row["duplicate_squads"] = "N/A"
    else:
        values = table[squad_col]
        missing = int(values.isna().sum() + (values.astype(str).str.strip() == "").sum())
        present = values.dropna().astype(str).str.strip()
        present = present[present != ""]
        unique = int(present.nunique())
        duplicates = int(len(present) - unique)

        row["unique_squads"] = unique
        row["missing_squads"] = missing
        row["duplicate_squads"] = duplicates

        if missing:
            findings.append("{} missing squad name(s)".format(missing))
        if duplicates:
            dupe_names = sorted(present[present.duplicated()].unique())
            findings.append(
                "{} duplicate squad name(s): {}".format(duplicates, ", ".join(dupe_names))
            )
        if unique != EXPECTED_SQUADS:
            findings.append(
                "expected {} unique squads, found {}".format(EXPECTED_SQUADS, unique)
            )

    table_type = classify_table_type(table)
    perspective = classify_perspective(table, squad_col, table_type)
    row["table_type"] = table_type
    row["perspective"] = perspective

    if table_type == "UNKNOWN":
        findings.append("schema matches no known table type")
    if perspective == "UNKNOWN" and table_type != "UNKNOWN":
        findings.append(
            "squad values are neither wholly 'vs '-prefixed nor wholly bare; "
            "perspective not determinable from content"
        )

    slot = SLOT_BY_TYPE.get((table_type, perspective), "UNKNOWN")
    row["coverage_slot"] = slot

    hint_type, hint_perspective = read_filename_hint(path.name)
    row["filename_suggests"] = "{} / {}".format(hint_type, hint_perspective)
    mismatch = (hint_type, hint_perspective) != (table_type, perspective)
    row["filename_content_mismatch"] = mismatch
    if mismatch:
        findings.append(
            "filename suggests {} / {} but content is {} / {}".format(
                hint_type, hint_perspective, table_type, perspective
            )
        )

    # A filename/content mismatch is a finding, not a failure: the data is intact,
    # only the label is wrong. Structural problems are what fail a file.
    blocking = [
        f
        for f in findings
        if not f.startswith("filename suggests")
    ]
    row["status"] = "PASS" if not blocking else "FAIL"
    row["findings"] = " | ".join(findings)

    return row, table


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def build_coverage(frame):
    """season x table-type matrix: PASS / MISSING / DUPLICATE / UNKNOWN."""
    coverage_rows = []
    detail = {}

    for season in SEASON_FOLDERS:
        sub = frame[(frame["season"] == season) & (frame["status"] != "UNREADABLE")]
        record = {"season": season}
        for slot in COVERAGE_SLOTS:
            matches = sub[sub["coverage_slot"] == slot]
            if len(matches) == 0:
                record[slot] = "MISSING"
            elif len(matches) == 1:
                record[slot] = "PASS"
            else:
                identical = matches["content_md5"].nunique() == 1
                record[slot] = "DUPLICATE" if identical else "UNKNOWN"
            detail[(season, slot)] = list(matches["filename"])

        coverage_rows.append(record)

    return pd.DataFrame(coverage_rows, columns=["season"] + COVERAGE_SLOTS), detail


def main():
    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 3: STATISTICAL SOURCE INTEGRITY")
    print("=" * 78)
    print("Project root : {}".format(PROJECT_ROOT))
    print("Raw directory: {}".format(RAW_DIR))
    print("Mode         : READ-ONLY (no rename, move, delete, merge, clean, repair)")
    print("Classifier   : table type from COLUMN HIERARCHY; Squad/Opponent from ROW VALUES")
    print("MultiIndex   : preserved; signatures use the full hierarchy, never leaves")
    print()

    if not RAW_DIR.exists():
        print("FATAL: raw directory not found: {}".format(RAW_DIR))
        return 1

    rows = []
    missing_folders = []

    for season in SEASON_FOLDERS:
        folder = RAW_DIR / season
        print("=" * 78)
        print("SEASON: {}".format(season))
        print("=" * 78)

        if not folder.is_dir():
            print("  MISSING FOLDER - not inspected")
            print()
            missing_folders.append(season)
            continue

        files = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".xls"],
            key=lambda p: p.name.lower(),
        )
        print("  .xls files: {}".format(len(files)))
        print()

        for path in files:
            row, _ = inspect_file(season, path)
            rows.append(row)

            print("  {:<62} {}".format(path.name[:62], row["status"]))
            print(
                "      content -> {} / {}   [slot: {}]".format(
                    row["table_type"], row["perspective"], row["coverage_slot"]
                )
            )
            print(
                "      rows {}  squads {}  missing {}  duplicate {}  schema {}  md5 {}".format(
                    row["rows"],
                    row["unique_squads"],
                    row["missing_squads"],
                    row["duplicate_squads"],
                    row["schema_signature"][:8],
                    str(row["content_md5"])[:8],
                )
            )
            if row["findings"]:
                for finding in str(row["findings"]).split(" | "):
                    print("      ! {}".format(finding))
            print()

    frame = pd.DataFrame(rows, columns=FIELDS)

    # Stable, readable labels for each distinct schema, ordered by first appearance.
    labels = {}
    for signature in frame["schema_signature"]:
        if signature and signature not in labels:
            labels[signature] = "SIG-{}".format(len(labels) + 1)
    frame["schema_label"] = frame["schema_signature"].map(
        lambda s: labels.get(s, "")
    )

    coverage, coverage_detail = build_coverage(frame)

    INTEGRITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(INTEGRITY_PATH, index=False, encoding="utf-8")
    coverage.to_csv(COVERAGE_PATH, index=False, encoding="utf-8")

    readable = frame[frame["status"] != "UNREADABLE"]

    # ---------------- findings ----------------

    print("=" * 78)
    print("FINDINGS")
    print("=" * 78)
    print()

    # 1. duplicate content within a season
    print("-" * 78)
    print("1. DUPLICATE CONTENT WITHIN A SEASON")
    print("-" * 78)
    intra_dupes = []
    for season in SEASON_FOLDERS:
        sub = readable[readable["season"] == season]
        for content, group in sub.groupby("content_md5"):
            if len(group) > 1:
                intra_dupes.append((season, content, list(group["filename"])))
    if not intra_dupes:
        print("  None.")
    else:
        for season, content, names in intra_dupes:
            print("  [{}] md5 {} shared by {} files:".format(season, content[:12], len(names)))
            for name in sorted(names):
                print("      {}".format(name))
            print("      -> one statistic is stored twice; another is therefore absent.")
    print()

    # 2. duplicate content ACROSS seasons (a season folder holding another's data)
    print("-" * 78)
    print("2. DUPLICATE CONTENT ACROSS SEASONS")
    print("-" * 78)
    cross = []
    for content, group in readable.groupby("content_md5"):
        if group["season"].nunique() > 1:
            cross.append((content, group))
    if not cross:
        print("  None - no season folder repeats another season's table.")
    else:
        for content, group in cross:
            print("  md5 {} appears in {} seasons:".format(content[:12], group["season"].nunique()))
            for _, r in group.iterrows():
                print("      [{}] {}".format(r["season"], r["filename"]))
    print()

    # 3. missing table types
    print("-" * 78)
    print("3. MISSING TABLE TYPES")
    print("-" * 78)
    missing_slots = []
    for _, r in coverage.iterrows():
        for slot in COVERAGE_SLOTS:
            if r[slot] == "MISSING":
                missing_slots.append((r["season"], slot))
    if not missing_slots:
        print("  None - all twelve expected tables present in every season.")
    else:
        for season, slot in missing_slots:
            print("  [{}] {} - NO FILE IN THE FOLDER CARRIES THIS DATA".format(season, slot))
    print()

    # 4. unexpected table types
    print("-" * 78)
    print("4. UNEXPECTED TABLE TYPES")
    print("-" * 78)
    unexpected = readable[readable["coverage_slot"] == "UNKNOWN"]
    if not len(unexpected):
        print("  None - every table classified into an expected slot.")
    else:
        for _, r in unexpected.iterrows():
            print(
                "  [{}] {} -> {} / {}".format(
                    r["season"], r["filename"], r["table_type"], r["perspective"]
                )
            )
    print()

    # 5. filename / content mismatches
    print("-" * 78)
    print("5. FILENAME / CONTENT MISMATCHES  (finding, not failure)")
    print("-" * 78)
    mismatches = readable[readable["filename_content_mismatch"] == True]
    if not len(mismatches):
        print("  None.")
    else:
        for _, r in mismatches.iterrows():
            print("  [{}] {}".format(r["season"], r["filename"]))
            print(
                "      filename says {}  ->  content is {} / {}".format(
                    r["filename_suggests"], r["table_type"], r["perspective"]
                )
            )
        print()
        print("  The data in these files is intact. Only the label is wrong.")
        print("  Nothing was renamed.")
    print()

    # 6. schema mismatches
    print("-" * 78)
    print("6. SCHEMA MISMATCHES")
    print("-" * 78)
    schema_by_type = defaultdict(set)
    for _, r in readable.iterrows():
        if r["table_type"] != "UNKNOWN":
            schema_by_type[r["table_type"]].add(r["schema_signature"])
    inconsistent = {t: s for t, s in schema_by_type.items() if len(s) > 1}
    if not inconsistent:
        print("  None - each table type has exactly one schema across all seasons.")
    else:
        for table_type, signatures in sorted(inconsistent.items()):
            print("  {}: {} different schemas".format(table_type, len(signatures)))
            for signature in sorted(signatures):
                sub = readable[readable["schema_signature"] == signature]
                print("      {} ({} files, {} cols)".format(
                    labels.get(signature, signature), len(sub), sub["cols"].iloc[0]
                ))
                for _, r in sub.iterrows():
                    print("          [{}] {}".format(r["season"], r["filename"]))
    print()

    # 7. squad problems
    print("-" * 78)
    print("7. SQUAD NAME PROBLEMS")
    print("-" * 78)
    squad_issues = readable[
        (readable["missing_squads"].astype(str) != "0")
        | (readable["duplicate_squads"].astype(str) != "0")
        | (readable["unique_squads"].astype(str) != str(EXPECTED_SQUADS))
    ]
    if not len(squad_issues):
        print(
            "  None - every table carries exactly {} unique squads, "
            "none missing, none duplicated.".format(EXPECTED_SQUADS)
        )
    else:
        for _, r in squad_issues.iterrows():
            print(
                "  [{}] {}: unique {}, missing {}, duplicate {}".format(
                    r["season"], r["filename"], r["unique_squads"],
                    r["missing_squads"], r["duplicate_squads"],
                )
            )
    print()

    # ---------------- schema catalogue ----------------

    print("=" * 78)
    print("SCHEMA CATALOGUE (full hierarchy preserved, leaves never flattened)")
    print("=" * 78)
    for signature, label in labels.items():
        sub = readable[readable["schema_signature"] == signature]
        if not len(sub):
            continue
        paths = json.loads(sub["schema_columns"].iloc[0])
        types = sorted(set(sub["table_type"]))
        print(
            "{}  {} file(s), {} column(s), {} header level(s) -> {}".format(
                label, len(sub), len(paths), sub["n_header_levels"].iloc[0], ", ".join(types)
            )
        )
        print("    {}".format(" ; ".join(" | ".join(p) for p in paths)))
        print()

    # ---------------- coverage matrix ----------------

    print("=" * 78)
    print("SEASON x TABLE-TYPE COVERAGE")
    print("=" * 78)
    short = {
        "Overall": "Ovr", "Home_Away": "H/A",
        "Standard_Squad": "Std-S", "Standard_Opponent": "Std-O",
        "Shooting_Squad": "Sht-S", "Shooting_Opponent": "Sht-O",
        "Goalkeeping_Squad": "GK-S", "Goalkeeping_Opponent": "GK-O",
        "PlayingTime_Squad": "PT-S", "PlayingTime_Opponent": "PT-O",
        "Miscellaneous_Squad": "Msc-S", "Miscellaneous_Opponent": "Msc-O",
    }
    mark = {"PASS": "ok", "MISSING": "MISS", "DUPLICATE": "DUP", "UNKNOWN": "UNK"}

    header = "  {:<16}".format("season") + "".join(
        "{:>7}".format(short[s]) for s in COVERAGE_SLOTS
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in coverage.iterrows():
        line = "  {:<16}".format(r["season"].split()[0]) + "".join(
            "{:>7}".format(mark[r[s]]) for s in COVERAGE_SLOTS
        )
        print(line)
    print()
    print("  ok = exactly one table   MISS = no table carries this data")
    print("  DUP = two files, identical content   UNK = two files, different content")
    print()

    for (season, slot), names in sorted(coverage_detail.items()):
        if len(names) > 1:
            print("  [{}] {} is served by {} files:".format(season, slot, len(names)))
            for name in sorted(names):
                print("      {}".format(name))
    print()

    # ---------------- status ----------------

    unreadable = frame[frame["status"] == "UNREADABLE"]
    bad_squads = len(squad_issues)
    unexpected_schema = len(unexpected) + len(inconsistent)
    unaccounted = len(missing_slots)

    checks = [
        ("every file is readable", len(unreadable) == 0 and not missing_folders),
        ("every file has {} unique squads".format(EXPECTED_SQUADS), bad_squads == 0),
        ("no unexpected schema appears", unexpected_schema == 0),
        ("all expected table types accounted for", unaccounted == 0),
        ("duplicate/missing content reported, not hidden", True),
    ]

    print("=" * 78)
    print("PASS CRITERIA")
    print("=" * 78)
    for label, ok in checks:
        print("  [{}] {}".format("PASS" if ok else "FAIL", label))
    print()

    overall = all(ok for _, ok in checks)

    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 3 STATUS")
    print("=" * 78)
    print()
    print("  {}".format("PASS" if overall else "FAIL"))
    print()

    if not overall:
        print("  Reason(s):")
        for label, ok in checks:
            if not ok:
                print("    - {}".format(label))
        print()
        if missing_slots:
            for season, slot in missing_slots:
                print("    {} has no {} table. The season folder holds 12 files,".format(season, slot))
                print("    but two of them carry the same statistic, so this one is absent.")
                print("    This is a genuine gap in the source data, not a parsing failure.")
        print()

    print("  Files inspected     : {}".format(len(frame)))
    print("  Readable            : {}".format(len(readable)))
    print("  Structurally clean  : {}".format(int((frame['status'] == 'PASS').sum())))
    print("  Filename mismatches : {} (reported, not corrected)".format(len(mismatches)))
    print()
    print("  Reports written:")
    print("    {}".format(INTEGRITY_PATH))
    print("    {}".format(COVERAGE_PATH))
    print()
    print("No source data was modified.")
    print("No files were renamed.")
    print("No files were deleted.")
    print("No missing data was inferred or replaced.")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
