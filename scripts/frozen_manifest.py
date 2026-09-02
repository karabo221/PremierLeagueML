"""
===============================================================================
THE FROZEN MANIFEST
===============================================================================

WHY

    Until now "frozen" meant "nobody has overwritten it yet", enforced by a
    file timestamp and good intentions. Instrument 5's C1 check anchors
    against phase3_reg_surface.csv and Instrument 4's R7 against
    phase3_ablation_fold_summary.csv; both sat in a gitignored directory where
    a silent regeneration would leave no diff to detect.

    The .gitignore exceptions put those files under version control. This
    manifest makes tampering detectable INDEPENDENTLY of git - if the repo is
    copied to a stick, or someone regenerates a file and commits it, the
    hashes still disagree with what the pre-declarations were written against.

USAGE

    python scripts/frozen_manifest.py            write / refresh the manifest
    python scripts/frozen_manifest.py --verify   check, exit 1 on any drift

    --verify is the mode that matters. Run it before trusting any instrument
    that anchors against a frozen artefact.
===============================================================================
"""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "FROZEN_MANIFEST.txt"

# Kept deliberately in step with the .gitignore exception block. A file that is
# version-controlled as evidence but absent here would be half-protected.
FROZEN_PATTERNS = [
    "PHASE3_REGULARISATION_PREDECLARATION.txt",
    "PHASE3_CEILING_PREDECLARATION.txt",
    "PHASE4_TIER2_WINDOW_PREDECLARATION.txt",
    # The D2 pre-declaration governs the whole of Phase 4 and was tracked in
    # git from the start, but was never listed here - so its hash could move
    # under an amendment with nothing but a diff to say so. Listed now.
    "PHASE4_D2_PREDECLARATION.txt",
    "outputs/phase4_dynamic_state_predeclaration.txt",
    "outputs/phase0_evaluation_folds.csv",
    "outputs/phase0_evaluation_spec.csv",
    "outputs/phase0_evaluation_spec.json",
    "outputs/phase2_base_rate_fold_summary.csv",
    "outputs/phase2_elo_fold_summary.csv",
    # Elo v1's five published metrics. The sixth, RPS, is derived from its own
    # per-match probabilities in phase2_elo_metrics_full.csv, because Elo ran
    # three days before the harness had the metric and its summary row names
    # five columns explicitly - so re-running it reproduces byte-identical
    # five-metric output rather than gaining the sixth.
    "outputs/phase2_elo_metrics_full.csv",
    "outputs/phase2_poisson_dc_fold_summary.csv",
    "outputs/phase3_ablation_fold_summary.csv",
    "outputs/phase3_ablation_ladder.csv",
    "outputs/phase3_reg_*.csv",
    "outputs/phase3_ceiling_*.csv",
    "outputs/phase3_frozen_regularisation.json",
    "outputs/phase4_tier2_*.csv",
    # The passthrough diagnostic and the state it wrote. The ladder's S1
    # anchors against phase4_dc_state.csv.
    "outputs/phase4_dc_state.csv",
    "outputs/phase4_passthrough_*.csv",
    # The D0/D1/D2 ladder. Amendment 4's rescaled rung writes to its own
    # filenames rather than over these.
    "outputs/phase4_dynamic_state.csv",
    "outputs/phase4_ladder_*.csv",
]

HEADER = """\
===============================================================================
FROZEN MANIFEST
SHA-256 of every artefact the instruments anchor against.
===============================================================================

These files are EVIDENCE, not regenerable output. Later instruments assert
equivalence against them:

    Instrument 4  R7  reproduces phase3_ablation_fold_summary.csv at lambda=1
    Instrument 5  C1  reproduces phase3_reg_surface.csv at lambda=100

and the pre-declarations were written against exactly these bytes.

Verify with:  python scripts/frozen_manifest.py --verify

A hash that has moved is not automatically wrong - an instrument may have been
legitimately re-run. It means the artefact is no longer the one the
pre-declaration was written against, and that has to be reconciled explicitly
rather than noticed later.

"""


def sha256(path):

    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files():

    seen = {}

    for pattern in FROZEN_PATTERNS:
        for path in sorted(PROJECT_ROOT.glob(pattern)):
            if path.is_file():
                seen[path.relative_to(PROJECT_ROOT).as_posix()] = path

    return dict(sorted(seen.items()))


def parse_manifest():

    if not MANIFEST.exists():
        return {}

    recorded = {}

    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and len(parts[0]) == 64:
            recorded[parts[1]] = parts[0]

    return recorded


def write_manifest(files):

    lines = [HEADER]
    lines.append("generated {}\n".format(
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))
    lines.append("{} files\n\n".format(len(files)))

    for name, path in files.items():
        lines.append("{}  {}\n".format(sha256(path), name))

    lines.append("\n" + "=" * 79 + "\nEND OF MANIFEST\n")

    MANIFEST.write_text("".join(lines), encoding="utf-8")


def verify(files):

    recorded = parse_manifest()

    if not recorded:
        print("  no manifest to verify against - run without --verify first")
        return 1

    moved, missing, unlisted = [], [], []

    for name, path in files.items():
        if name not in recorded:
            unlisted.append(name)
        elif recorded[name] != sha256(path):
            moved.append(name)

    for name in recorded:
        if name not in files:
            missing.append(name)

    print("  {:<44} {}".format("files listed in the manifest", len(recorded)))
    print("  {:<44} {}".format("files found on disk", len(files)))
    print("  {:<44} {}".format("hashes that MOVED", len(moved)))
    print("  {:<44} {}".format("listed but MISSING from disk", len(missing)))
    print("  {:<44} {}".format("on disk but NOT listed", len(unlisted)))
    print()

    for name in moved:
        print("    MOVED    {}".format(name))
    for name in missing:
        print("    MISSING  {}".format(name))
    for name in unlisted:
        print("    UNLISTED {}".format(name))

    if moved or missing:
        print()
        print("  FAIL - a frozen artefact is not the one the pre-declarations")
        print("  were written against. Reconcile this before trusting any")
        print("  instrument that anchors to it.")
        return 1

    if unlisted:
        print()
        print("  WARN - new frozen-pattern files exist that the manifest does")
        print("  not cover. Re-run without --verify to adopt them.")
        return 0

    print("  PASS - every frozen artefact matches the manifest.")
    return 0


def main():

    files = frozen_files()

    print()
    print("=" * 78)
    print("FROZEN MANIFEST")
    print("=" * 78)
    print()

    if "--verify" in sys.argv:
        return verify(files)

    write_manifest(files)

    print("  {:<58} {}".format("file", "sha256"))
    print("  " + "-" * 76)

    for name, path in files.items():
        print("  {:<58} {}".format(name[:58], sha256(path)[:16] + "..."))

    print()
    print("  {} files -> {}".format(len(files), MANIFEST.name))
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
