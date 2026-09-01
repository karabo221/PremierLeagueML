"""
===============================================================================
MIGRATION - ADD A STRING LABEL BESIDE EVERY LAMBDA KEY COLUMN
===============================================================================

WHY

    Instrument 5 found that a float column used as a KEY is not safe to read
    back from CSV without float_precision="round_trip". The failure is quiet
    and it is not what it first looks like:

        the values do NOT parse to NaN. They parse one ULP away.
        29 of 32 output CSVs differ between default and round_trip parsing,
        by at most 2.27e-13 - harmless for arithmetic, fatal for equality.

    So `surface[surface["lambda"] == 0.03]` silently matches zero rows under
    default parsing, and the NaN appears downstream in whatever mean() is
    taken over the empty selection. Nothing raises.

    Setting float_precision at all 14 call sites protects nothing outside this
    repo and regresses silently the moment someone adds a fifteenth. The fix
    is to stop using a float as a key: every lambda column gains a string
    label beside it, and consumers key on the label.

WHAT THIS DOES AND DOES NOT TOUCH

    Adds `lambda_label` after `lambda`, and `selected_lambda_label` after
    `selected_lambda`. Format is "{:g}" - 0.03, 0.3, 1, 100, 10000 - which is
    exactly how both grids are printed already.

    It adds a column and changes NOTHING else. Every pre-existing column is
    asserted byte-identical after the rewrite, read back with round_trip, or
    the file is restored and the migration fails. These are frozen evidence
    artefacts; a migration that could quietly perturb one would be worse than
    the defect it fixes.
===============================================================================
"""

from pathlib import Path
import hashlib
import shutil
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

# column -> the label column it gains
KEY_COLUMNS = {
    "lambda": "lambda_label",
    "selected_lambda": "selected_lambda_label",
}

TARGETS = sorted(
    set(OUTPUTS_DIR.glob("phase3_reg_*.csv"))
    | set(OUTPUTS_DIR.glob("phase3_ceiling_*.csv")))


def sha256(path):

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


sys.path.insert(0, str(Path(__file__).resolve().parent))

# ONE definition of the label, imported rather than restated. A migration that
# spelled the key its own way would drift from the instruments that write it,
# and the whole point of the label is that everyone agrees on it.
from phase3_regularisation_sensitivity import (  # noqa: E402
    LAMBDA_LABEL_PREFIX as LABEL_PREFIX,
    lambda_label as label_of,
)


def _label_rationale():
    """
    0.03 -> 'lam=0.03', 10000.0 -> 'lam=10000'.

    THE PREFIX IS NOT DECORATION AND MUST NOT BE DROPPED.

    The first version of this migration wrote the bare "{:g}" text - '0.03',
    '1', '10000' - and its own verification rejected it. A CSV column of
    numeric-looking strings is type-inferred straight back to float64 on read,
    so '1' returns as 1.0 and str() of it is '1.0'. The column would have
    looked like a string key while being exactly the float key it was meant to
    replace, which is worse than the original defect because it also looks
    fixed.

    'lam=' cannot be coerced to a number by any parser, so the column is a
    string for every consumer, including one that reads the file with no
    arguments at all. That is the entire point: the fix has to work for the
    reader who does not know to be careful.
    """

    raise NotImplementedError("documentation only; see lambda_label()")


def migrate(path):
    """Returns (changed, before_sha, after_sha, columns_added)."""

    before = pd.read_csv(path, float_precision=FLOAT_PRECISION)

    present = [c for c in KEY_COLUMNS if c in before.columns]

    if not present:
        return None, sha256(path), sha256(path), []

    before_sha = sha256(path)

    frame = before.copy()

    added = []
    correct = 0

    for source in present:

        target = KEY_COLUMNS[source]
        labels = frame[source].map(label_of)

        if target in frame.columns:
            # Present already - but it may carry the bare numeric-looking text
            # the first attempt wrote, which is not a string key at all. Rewrite
            # it rather than trusting its existence.
            if frame[target].astype(str).equals(labels.astype(str)):
                correct += 1
                continue
            frame[target] = labels
            added.append(target + " (repaired)")
            continue

        frame.insert(frame.columns.get_loc(source) + 1, target, labels)
        added.append(target)

    if correct == len(present):
        return False, before_sha, before_sha, []

    backup = path.with_suffix(".csv.premigration")
    shutil.copy2(path, backup)

    frame.to_csv(path, index=False, encoding="utf-8", float_format=FLOAT_FORMAT)

    # ---- prove nothing but the new column moved -------------------------
    after = pd.read_csv(path, float_precision=FLOAT_PRECISION)

    problems = []

    if len(after) != len(before):
        problems.append("row count {} -> {}".format(len(before), len(after)))

    for column in before.columns:

        if column not in after.columns:
            problems.append("column {} vanished".format(column))
            continue

        # The label columns are what this migration writes. A pre-existing one
        # carrying the bare numeric text of the first attempt is SUPPOSED to
        # change, so exempt them here - their correctness is asserted directly
        # below instead, against the values they are derived from.
        if column in KEY_COLUMNS.values():
            continue

        a, b = before[column], after[column]

        if pd.api.types.is_float_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            mask = ~(np.isnan(x) & np.isnan(y))
            if not np.array_equal(x[mask], y[mask]):
                worst = float(np.nanmax(np.abs(x[mask] - y[mask])))
                problems.append("{} moved by {:.3e}".format(column, worst))
        else:
            if not a.astype(str).equals(b.astype(str)):
                problems.append("{} changed".format(column))

    # ---- and prove the label actually round-trips as a key --------------
    for source in present:

        target = KEY_COLUMNS[source]

        recomputed = after[source].map(label_of)

        if not recomputed.astype(str).equals(after[target].astype(str)):
            problems.append("{} does not match its own values".format(target))

        plain = pd.read_csv(path)      # deliberately WITHOUT round_trip

        if not plain[target].astype(str).equals(after[target].astype(str)):
            problems.append("{} is not stable under default parsing".format(target))

    if problems:
        shutil.copy2(backup, path)
        backup.unlink()
        raise SystemExit(
            "FATAL: migration of {} perturbed the file, restored: {}".format(
                path.name, "; ".join(problems)))

    backup.unlink()

    return True, before_sha, sha256(path), added


def main():

    print()
    print("=" * 78)
    print("MIGRATION: STRING LABELS BESIDE EVERY LAMBDA KEY COLUMN")
    print("=" * 78)
    print()

    print("  {:<44} {:<26} {}".format("file", "added", "sha256 (after)"))
    print("  " + "-" * 92)

    changed = 0

    for path in TARGETS:

        did, before_sha, after_sha, added = migrate(path)

        if did:
            changed += 1

        note = (",".join(added) if added
                else "- no lambda key -" if did is None
                else "- already correct -")

        print("  {:<44} {:<26} {}".format(path.name, note, after_sha[:16] + "..."))

    print()

    # ---- the defect this migration exists to close, demonstrated --------
    probe = OUTPUTS_DIR / "phase3_ceiling_combined_surface.csv"

    if probe.exists():

        default = pd.read_csv(probe)
        strict = pd.read_csv(probe, float_precision=FLOAT_PRECISION)

        print("  The defect, on {}:".format(probe.name))
        print()
        print("    {:<10} {:>14} {:>14} {:>16}".format(
            "lambda", "float ==, dflt", "float ==, rt", "label ==, dflt"))
        print("    " + "-" * 58)

        for value in (0.03, 0.3, 1.0, 100.0):
            print("    {:<10g} {:>14} {:>14} {:>16}".format(
                value,
                int((default["lambda"] == value).sum()),
                int((strict["lambda"] == value).sum()),
                int((default["lambda_label"] == label_of(value)).sum())))

        print()
        print("  The middle column is what every consumer must remember to do.")
        print("  The right-hand column is what they get for free from now on.")
        print()

    print("  Files migrated: {} of {}".format(changed, len(TARGETS)))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
