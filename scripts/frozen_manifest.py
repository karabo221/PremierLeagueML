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
    # Amendment 6 records a defect, a framing error and a deferral. An
    # amendment that can be edited after the fact is worth as little as a
    # pre-declaration that can, so it is frozen on the same terms.
    "PHASE4_AMENDMENT6_D2STATIC.txt",
    # PHASE 5. The market pre-declaration fixes the bookmaker, the de-vig and
    # the calibration bins BEFORE any score existed; a pre-declaration whose
    # hash can move is not one.
    "PHASE5_MARKET_PREDECLARATION.txt",
    # Declared before the xG source existed. That is the only time a
    # pre-declaration is worth anything, so its hash is frozen from now.
    "PHASE5_XG_PREDECLARATION.txt",
    # E1 sizes the xG arm. Its design was fixed before any fit, and its
    # section 2 records source-error COUNTS that gate E4 - so a hash that
    # moves is either a rewritten declaration or a changed source.
    "PHASE5_E1_SHOT_PREDECLARATION.txt",
    # E1c is the isolated finishing residual, and it is declared BEFORE the
    # section 3 gap diagnostic runs on the outer-test rows. That ordering is
    # the only thing that makes it a pre-declaration, and a hash is what makes
    # the ordering checkable.
    "PHASE5_E1C_FINISHING_PREDECLARATION.txt",
    # PHASE 6. The holdout freeze is the one declaration in this project
    # whose value depends entirely on predating the data it will be scored
    # against. Its hash is what makes "written before any 2026-27 match was
    # scored" checkable rather than asserted.
    "PHASE6_HOLDOUT_FREEZE.txt",
    # The cutoff pin. It exists so the freeze file need not change: it cites
    # the freeze hash, so it records not only WHAT the cutoff is but WHICH
    # freeze text it was pinned against. It also carries the twenty-name
    # 2026-27 vocabulary the scoring run asserts on, so a name quietly added
    # to that list would move this hash.
    "PHASE6_CUTOFF_PIN.txt",
    "outputs/phase6_freeze_validation.csv",
    # THE ODDS SOURCE ITSELF. data/raw is immutable by project rule, but that
    # rule is prose and this is the first source that arrived over the network
    # rather than on the stick. A re-download that moves a hash is a NEW
    # SOURCE, not a refresh, and this is what makes that checkable.
    "data/raw/Odds/E0_*.csv",
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
    # THE FEATURE FILE ITSELF, and the inventory that describes it.
    #
    # This was the largest hole in the manifest and it survived three phases.
    # Every rung of every ladder builds its design by reading
    # outputs/phase3_features.csv - D0 through D4, the ablation, the ceiling,
    # the regularisation surface. It was untracked and unhashed, so the one
    # file the whole of Phase 3 and Phase 4 rests on could have been
    # regenerated with nothing anywhere to detect it.
    #
    # It was found because DS7 could not be made to do its declared job. DS7
    # asserts that Block C and Block X are "byte-identical to the frozen
    # Phase 3 artefacts", and its own note claimed "DS10 verifies the file's
    # hash". DS10 did not, because the file was not listed here. The note was
    # false in a way that only became visible when DS7 went live at D3.
    #
    # phase3_feature_inventory.csv is listed beside it deliberately. It
    # records per-column non-null, distinct, min, median, mean, max and std,
    # written by the feature builder in the same run, so it is an INDEPENDENT
    # description of the same bytes. DS7 checks the feature file against that
    # record rather than only against itself.
    "outputs/phase3_features.csv",
    "outputs/phase3_feature_inventory.csv",
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
    # Amendments 4 and 5. Written to their own filenames so the first run's
    # artefacts above are not overwritten by the rung that supersedes them.
    "outputs/phase4_a4_*.csv",
    "outputs/phase4_static_state.csv",
    # D3 and D4, and the expected_total_goals redundancy diagnostic. Again
    # their own filenames, so the rungs they are compared against are not
    # overwritten by the rungs that extend them.
    "outputs/phase4_d34_*.csv",
    "outputs/phase5_*.csv",
    "outputs/phase4_etg_*.csv",
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
