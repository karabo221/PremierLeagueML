"""
PHASE 0 - INSTRUMENT 3: STATISTICAL SOURCE INVENTORY

Reports the ACTUAL schema of every statistical .xls file in data/raw/<season>/.

This instrument is strictly read-only. It does not merge, clean, rename,
reshape, interpret, or write to any source file. Every value in the report is
observed, not inferred.

Table selection rule (recorded explicitly per file, never chosen silently):
  - exactly one table parsed -> that table
  - more than one            -> the table with the most cells (rows x cols),
                                ties broken by lowest index
Both the selected index and the shapes of ALL parsed tables are reported, so a
file with multiple tables is visible as such rather than being quietly reduced.
"""

import hashlib
import io
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "outputs" / "phase0_statistical_source_inventory.csv"

SEASON_FOLDERS = [
    "2022 PL Season",
    "2023 PL Season",
    "2024 PL Season",
    "2025 PL Season",
    "2026 PL Season",
]

ENCODINGS = ["utf-8", "cp1252", "latin-1"]
FLAVORS = ["lxml", "bs4"]

FIELDS = [
    "season_folder",
    "filename",
    "relative_path",
    "file_size_bytes",
    "status",
    "error",
    "encoding_used",
    "parser_flavor",
    "n_tables_found",
    "all_table_shapes",
    "selected_table_index",
    "selected_table_shape",
    "content_md5",
    "n_rows",
    "n_cols",
    "columns_are_multiindex",
    "n_header_levels",
    "column_names",
    "squad_column_detected",
    "n_unique_squads",
    "first_3_rows",
]


def blank_row():
    return {field: "" for field in FIELDS}


def flatten(col):
    """Flattened display form of a column label. Display only - nothing is renamed."""
    if isinstance(col, tuple):
        return " | ".join(str(x) for x in col)
    return str(col)


def decode(raw_bytes):
    """Return (text, encoding_used). Reading only - the file on disk is never rewritten."""
    for enc in ENCODINGS:
        try:
            return raw_bytes.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("latin-1", errors="replace"), "latin-1(replace)"


def parse_tables(text):
    """Return (tables, flavor_used, error). Tries lxml then bs4; never edits the file."""
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


def select_table(tables):
    """Largest by cell count, ties to lowest index. The index is always reported."""
    if len(tables) == 1:
        return 0
    sizes = [t.shape[0] * t.shape[1] for t in tables]
    return int(max(range(len(tables)), key=lambda i: (sizes[i], -i)))


def find_squad_column(table):
    """Locate an existing Squad/Team column. Detection only - nothing renamed or created."""
    for col in table.columns:
        levels = col if isinstance(col, tuple) else (col,)
        for level in levels:
            if str(level).strip().lower() in ("squad", "team"):
                return col
    return None


def inspect_file(season, path):
    row = blank_row()
    row["season_folder"] = season
    row["filename"] = path.name
    row["relative_path"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    row["file_size_bytes"] = path.stat().st_size

    try:
        raw = path.read_bytes()
    except Exception as exc:
        row["status"] = "READ_ERROR"
        row["error"] = "{}: {}".format(type(exc).__name__, exc)
        return row, None

    text, enc = decode(raw)
    row["encoding_used"] = enc

    tables, flavor, error = parse_tables(text)
    if tables is None:
        row["status"] = "PARSE_ERROR"
        row["error"] = error or "unknown parse failure"
        return row, None

    idx = select_table(tables)
    table = tables[idx]

    row["status"] = "PARSED"
    row["parser_flavor"] = flavor
    row["n_tables_found"] = len(tables)
    row["all_table_shapes"] = json.dumps([list(t.shape) for t in tables])
    row["selected_table_index"] = idx
    row["selected_table_shape"] = "{}x{}".format(table.shape[0], table.shape[1])
    # Fingerprint of the parsed table as-read. Lets two files carrying identical
    # data be spotted without comparing anything by hand. Observation only.
    row["content_md5"] = hashlib.md5(
        table.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    row["n_rows"] = table.shape[0]
    row["n_cols"] = table.shape[1]
    row["columns_are_multiindex"] = isinstance(table.columns, pd.MultiIndex)
    row["n_header_levels"] = table.columns.nlevels
    row["column_names"] = json.dumps(
        [flatten(c) for c in table.columns], ensure_ascii=False
    )

    squad_col = find_squad_column(table)
    if squad_col is not None:
        row["squad_column_detected"] = flatten(squad_col)
        row["n_unique_squads"] = int(table[squad_col].dropna().nunique())
    else:
        row["squad_column_detected"] = "NONE"
        row["n_unique_squads"] = "N/A"

    head = table.head(3)
    flat_cols = [flatten(c) for c in head.columns]
    row["first_3_rows"] = json.dumps(
        [
            {
                col: ("" if pd.isna(val) else str(val))
                for col, val in zip(flat_cols, record)
            }
            for record in head.itertuples(index=False, name=None)
        ],
        ensure_ascii=False,
    )

    return row, table


def value_counts_sorted(frame, column):
    counts = frame[column].value_counts()
    return sorted(counts.items(), key=lambda kv: str(kv[0]))


def main():
    print("=" * 75)
    print("PHASE 0 - INSTRUMENT 3: STATISTICAL SOURCE INVENTORY")
    print("=" * 75)
    print("Project root : {}".format(PROJECT_ROOT))
    print("Raw directory: {}".format(RAW_DIR))
    print("Mode         : READ-ONLY (no merge, no clean, no rename, no inference)")
    print()

    if not RAW_DIR.exists():
        print("FATAL: raw directory not found: {}".format(RAW_DIR))
        return 1

    rows = []
    for season in SEASON_FOLDERS:
        folder = RAW_DIR / season
        print("=" * 75)
        print("SEASON FOLDER: {}".format(season))
        print("=" * 75)

        if not folder.is_dir():
            print("  MISSING FOLDER - not inspected")
            print()
            row = blank_row()
            row["season_folder"] = season
            row["relative_path"] = "data/raw/{}".format(season)
            row["status"] = "MISSING_FOLDER"
            row["error"] = "season folder not found"
            rows.append(row)
            continue

        files = sorted(
            [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".xls"],
            key=lambda p: p.name.lower(),
        )
        print("  .xls files found: {}  (expected 12)".format(len(files)))
        if len(files) != 12:
            print("  ! FILE COUNT DIFFERS FROM EXPECTED 12 - reported, not corrected")
        print()

        for path in files:
            row, table = inspect_file(season, path)
            rows.append(row)

            print("  FILE: {}".format(path.name))
            if row["status"] != "PARSED":
                print("    {}: {}".format(row["status"], row["error"]))
                print()
                continue

            print(
                "    tables found    : {}  (all shapes {}, selected index {})".format(
                    row["n_tables_found"],
                    row["all_table_shapes"],
                    row["selected_table_index"],
                )
            )
            print(
                "    selected shape  : {}  [{} rows x {} cols]".format(
                    row["selected_table_shape"], row["n_rows"], row["n_cols"]
                )
            )
            print(
                "    MultiIndex cols : {}  (header levels: {})".format(
                    row["columns_are_multiindex"], row["n_header_levels"]
                )
            )
            print(
                "    squad column    : {}   unique squads: {}".format(
                    row["squad_column_detected"], row["n_unique_squads"]
                )
            )
            print(
                "    encoding/parser : {} / {}".format(
                    row["encoding_used"], row["parser_flavor"]
                )
            )

            cols = json.loads(row["column_names"])
            print("    columns ({}):".format(len(cols)))
            for i, col in enumerate(cols):
                print("      [{:>2}] {}".format(i, col))

            print("    first 3 rows:")
            for line in table.head(3).to_string(max_colwidth=16).splitlines():
                print("      {}".format(line))
            print()

    frame = pd.DataFrame(rows, columns=FIELDS)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False, encoding="utf-8")

    parsed = frame[frame["status"] == "PARSED"]
    failed = frame[frame["status"] != "PARSED"]

    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print("Entries in report : {}".format(len(frame)))
    print("Parsed            : {}".format(len(parsed)))
    print("Not parsed        : {}".format(len(failed)))
    print()

    if len(parsed):
        print("Per season:")
        for season in SEASON_FOLDERS:
            sub = parsed[parsed["season_folder"] == season]
            if not len(sub):
                print("  {:<18} 0 parsed".format(season))
                continue
            print(
                "  {:<18} {:>2} parsed | MultiIndex {:>2}/{} | rows {}-{} | cols {}-{}".format(
                    season,
                    len(sub),
                    int(sub["columns_are_multiindex"].sum()),
                    len(sub),
                    sub["n_rows"].min(),
                    sub["n_rows"].max(),
                    sub["n_cols"].min(),
                    sub["n_cols"].max(),
                )
            )
        print()

        print("Header levels across parsed files:")
        for levels, count in value_counts_sorted(parsed, "n_header_levels"):
            print("  {} header level(s): {} file(s)".format(levels, count))
        print()

        print("Row counts observed (raw, uncleaned):")
        for n_rows, count in value_counts_sorted(parsed, "n_rows"):
            print("  {:>4} rows: {} file(s)".format(n_rows, count))
        print()

        print("Unique squad counts observed:")
        for n_uniq, count in value_counts_sorted(parsed, "n_unique_squads"):
            print("  {:>4} unique: {} file(s)".format(str(n_uniq), count))
        print()

        print("Tables per file:")
        for n_tab, count in value_counts_sorted(parsed, "n_tables_found"):
            print("  {} table(s): {} file(s)".format(n_tab, count))
        print()

        print(
            "Distinct column signatures: {} across {} parsed files".format(
                parsed["column_names"].nunique(), len(parsed)
            )
        )
        print()

        print("=" * 75)
        print("SCHEMA SIGNATURE GROUPS (files sharing an identical column list)")
        print("=" * 75)
        signature_groups = sorted(
            parsed.groupby("column_names"), key=lambda kv: -len(kv[1])
        )
        for i, (signature, group) in enumerate(signature_groups, 1):
            cols = json.loads(signature)
            seasons = sorted(s.split()[0] for s in group["season_folder"].unique())
            print(
                "SIGNATURE {}: {} file(s), {} column(s), {} header level(s)".format(
                    i, len(group), len(cols), group["n_header_levels"].iloc[0]
                )
            )
            print("  seasons present: {}".format(", ".join(seasons)))
            print("  columns: {}".format(" ; ".join(cols)))
            print("  files:")
            for _, r in group.sort_values(["season_folder", "filename"]).iterrows():
                print(
                    "    [{}] {}".format(r["season_folder"].split()[0], r["filename"])
                )
            print()

        print("=" * 75)
        print("IDENTICAL CONTENT (same parsed table under more than one filename)")
        print("=" * 75)
        dupes = parsed[parsed.duplicated("content_md5", keep=False)]
        if not len(dupes):
            print("  None - every parsed table is distinct.")
        else:
            for content_hash, group in dupes.groupby("content_md5"):
                print("  content_md5 {}: {} files".format(content_hash[:12], len(group)))
                for _, r in group.sort_values(["season_folder", "filename"]).iterrows():
                    print(
                        "    [{}] {}".format(
                            r["season_folder"].split()[0], r["filename"]
                        )
                    )
                print()
            print("  Reported only. Nothing renamed, deduplicated, or removed.")
        print()

    if len(failed):
        print("NOT PARSED (reported, not repaired):")
        for _, r in failed.iterrows():
            print(
                "  [{}] {}: {} - {}".format(
                    r["season_folder"], r["filename"], r["status"], r["error"]
                )
            )
        print()

    print("Report written: {}".format(OUT_PATH))
    print()
    print("No source data was modified.")
    print("No files were renamed.")
    print("No files were deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
