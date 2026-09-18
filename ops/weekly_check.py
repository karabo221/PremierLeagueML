"""
===============================================================================
SHOULD THE WEEKLY ROUTINE WRITE A ROUND RIGHT NOW?  READS ONLY.
===============================================================================

Called by ops/weekly.ps1 before phase6_generate_predictions.py --round, so that
running the routine on the wrong day is a clean skip rather than a FAIL row in
live_log/phase6_live_log_run_audit.csv.

IT DECIDES NOTHING THE PROTOCOL DECIDES. The fixture reader, the round plan
(L10.2), the match key (L3.3) and the team mapping are the generator's own,
imported rather than copied (L8.1). This file asks plan_rounds() the same
question the generator will, and reports the answer.

Since amendment 1 (2026-09-18) the fixture feed lists the whole season, so
"not published yet" no longer happens. What can happen instead is a round
waiting on the E0 results file, which must hold every match played before the
round's state cutoff before that round is written.

    exit 0    at least one round is due: the next one, or a late one
    exit 11   every due round is already in the log
    exit 12   a round mixes written and unwritten matches. The generator
              would refuse it (L1.4, L3.1), and a person has to look.
    exit 13   the next round is waiting on the E0 results file

Run as a FILE, never through python -c: the generator imports the Phase 0
harness, which reads __main__.__file__ (PROJECT_GOTCHAS.md section 1).
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import phase6_generate_predictions as GEN  # noqa: E402


READY, ALREADY_WRITTEN, MIXED, WAITING = 0, 11, 12, 13


def main():

    GEN.configure_stdout()

    pin = GEN.load_pin()
    now = pd.Timestamp.now(tz="UTC")

    fixtures = GEN.fixtures_from_source(GEN.fetch(GEN.FIXTURES_URL))

    live = GEN.spine_from_e0(GEN.fetch(GEN.SOURCE_URL))
    live["match_id"] = [
        GEN.match_key(s, h, a) for s, h, a in
        zip(live["season"], live["home_team"], live["away_team"])]

    log = GEN.read_log()
    written = set(log["match_id"]) if len(log) else set()

    due, waiting, mixed = GEN.plan_rounds(
        fixtures, written, live, pin["cutoff"], now)

    for entry in due:
        print("  round {:>2}  {}  {} matches from {}".format(
            entry["round_number"],
            "LATE - kicked off, will be flagged" if entry["late"]
            else "next - before kickoff          ",
            len(entry["block"]), str(entry["block"]["date"].min().date())))

    for entry in waiting:
        print("  round {:>2}  waiting: E0 lacks {} played match(es) before {}"
              .format(entry["round_number"], len(entry["lagging"]),
                      str(entry["state_cutoff"].date())))

    for entry in mixed:
        print("  round {:>2}  MIXED: {} of {} already written".format(
            entry["round_number"], entry["written"], len(entry["block"])))

    if mixed:
        print("  The generator refuses a round that repeats a match (L1.4, "
              "L3.1). Look before running it.")
        return MIXED

    if due:
        return READY

    if waiting:
        return WAITING

    print("  every due round is already in the log. Nothing to write.")
    return ALREADY_WRITTEN


if __name__ == "__main__":
    raise SystemExit(main())
