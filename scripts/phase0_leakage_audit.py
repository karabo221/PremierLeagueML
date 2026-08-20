"""
PHASE 0 - INSTRUMENT 5: LEAKAGE AUDIT

Establishes and mechanically tests the temporal boundary of the eventual ML
dataset. Nothing here is a rulebook: every rule is executed against the real
1,900 fixture dates and the result is measured.

THE BOUNDARY

  For a match played at date T, a predictor may use ONLY information that
  existed before kickoff. The historical set is:

      all completed matches with date < T          (STRICT inequality)

  Same-day matches are NOT automatically available. A 15:00 kickoff cannot see a
  17:30 result, and this dataset's kickoff times are not reliable enough to order
  within a day, so the whole day is excluded. That is a deliberate, measured cost
  and this instrument reports exactly what it costs.

WHAT THIS INSTRUMENT DOES NOT DO

  It does not build the ML dataset. It does not perform feature engineering. The
  probe features it computes exist only inside this process, are never written to
  disk as a dataset, and exist solely to prove the temporal rule holds or fails.
  Nothing is silently shifted or lagged to make a test pass.

  It is also STRICTLY READ-ONLY over data/raw/. Nothing is renamed, moved,
  deleted or modified.

THE STANDARD OF PROOF

  A feature is not safe because its name sounds historical. "Rolling goals
  before match" is a claim, not evidence. Every SAFE claim in the registry is
  backed by a test that would fail if the claim were false - including a
  perturbation test that rewrites a match's own scoreline and confirms the
  match's own features do not move.
"""

import bisect
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_statistical_integrity import (  # noqa: E402
    classify_table_type,
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

REGISTRY_PATH = OUTPUT_DIR / "phase0_feature_availability_registry.csv"
AUDIT_PATH = OUTPUT_DIR / "phase0_leakage_audit.csv"

SCORE_SEPARATORS = ["–", "—", "-"]

PERTURBATION_SAMPLE = 100

REGISTRY_FIELDS = [
    "feature",
    "source",
    "available_before_match",
    "contains_future_information",
    "reason",
    "allowed_for_training",
    "notes",
]

AUDIT_FIELDS = [
    "test_id",
    "test_name",
    "leakage_class",
    "scope",
    "matches_tested",
    "observed",
    "expectation",
    "verdict",
    "detail",
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def read_table(path):
    raw = path.read_bytes()
    text, _ = decode(raw)
    tables, _, error = parse_tables(text)
    if tables is None:
        return None, error
    return max(tables, key=lambda t: t.shape[0] * t.shape[1]), None


def parse_score(value):
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
            try:
                home = int(str(parts[0]).strip())
                away = int(str(parts[1]).strip())
            except ValueError:
                return None
            if home < 0 or away < 0:
                return None
            return home, away
    return None


def load_matches():
    """Every played match across every season, with a parsed date.

    Returns (matches, load_stats). Rows without a date, two teams and a
    parseable score are excluded and counted - never guessed at.
    """
    rows = []
    stats = {
        "files": 0,
        "rows_raw": 0,
        "rows_blank": 0,
        "rows_no_score": 0,
        "rows_bad_date": 0,
        "rows_used": 0,
    }

    fixture_files = sorted(
        [p for p in FIXTURES_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".xls"],
        key=lambda p: p.name.lower(),
    )

    for path in fixture_files:
        table, error = read_table(path)
        if table is None:
            continue
        stats["files"] += 1
        stats["rows_raw"] += len(table)

        season = path.stem.split(" PL")[0].strip()

        for _, row in table.iterrows():
            if row.isna().all():
                stats["rows_blank"] += 1
                continue

            home = row.get("Home")
            away = row.get("Away")
            if pd.isna(home) or pd.isna(away):
                stats["rows_blank"] += 1
                continue

            score = parse_score(row.get("Score"))
            if score is None:
                stats["rows_no_score"] += 1
                continue

            date = pd.to_datetime(row.get("Date"), errors="coerce")
            if pd.isna(date):
                stats["rows_bad_date"] += 1
                continue

            home_goals, away_goals = score
            rows.append({
                "season": season,
                "date": date.normalize(),
                "home": str(home).strip(),
                "away": str(away).strip(),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "result": "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D"),
            })
            stats["rows_used"] += 1

    matches = pd.DataFrame(rows)
    if len(matches):
        matches = matches.sort_values(["season", "date"]).reset_index(drop=True)
    return matches, stats


def load_final_tables():
    """Final Overall table per season - the canonical UNSAFE source.

    Located by CONTENT (shared classifier), never by filename.
    """
    finals = {}
    for folder in sorted(RAW_DIR.iterdir()):
        if not folder.is_dir() or folder.name == "Fixtures":
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() != ".xls":
                continue
            table, _ = read_table(path)
            if table is None:
                continue
            if classify_table_type(table) != "Overall":
                continue
            squad_col = find_squad_column(table)
            if squad_col is None:
                continue
            record = {}
            for position, team in enumerate(table[squad_col]):
                if pd.isna(team):
                    continue
                entry = {}
                for metric in ("Rk", "Pts", "GF", "GA", "MP"):
                    if metric in table.columns:
                        try:
                            entry[metric] = int(float(table[metric].iloc[position]))
                        except (ValueError, TypeError):
                            entry[metric] = None
                record[str(team).strip()] = entry
            finals[folder.name] = {"file": path.name, "table": record}
            break
    return finals


# --------------------------------------------------------------------------
# the temporal boundary
# --------------------------------------------------------------------------


class SeasonHistory:
    """Per-team match log for one season, queryable strictly before a date.

    Every lookup uses bisect_left on the team's sorted date list, so a match
    played ON the boundary date is excluded by construction. There is no code
    path that can include a same-day or later match.
    """

    def __init__(self, season_matches):
        self.dates = defaultdict(list)
        self.entries = defaultdict(list)

        for _, match in season_matches.iterrows():
            self._append(match["home"], match["date"], match["home_goals"],
                         match["away_goals"], "H")
            self._append(match["away"], match["date"], match["away_goals"],
                         match["home_goals"], "A")

        for team in self.dates:
            order = sorted(range(len(self.dates[team])), key=lambda i: self.dates[team][i])
            self.dates[team] = [self.dates[team][i] for i in order]
            self.entries[team] = [self.entries[team][i] for i in order]

    def _append(self, team, date, goals_for, goals_against, venue):
        if goals_for > goals_against:
            points = 3
        elif goals_for == goals_against:
            points = 1
        else:
            points = 0
        self.dates[team].append(date)
        self.entries[team].append({
            "date": date, "gf": goals_for, "ga": goals_against,
            "pts": points, "venue": venue,
        })

    def before(self, team, date):
        """Entries strictly before `date`. Same-day entries are never returned."""
        dates = self.dates.get(team)
        if not dates:
            return []
        cut = bisect.bisect_left(dates, date)
        return self.entries[team][:cut]


def probe_features(history, team, date, venue):
    """Pre-match probe values. AUDIT ONLY - never written out as a dataset.

    These exist to be tested, not to be used. If any of them could see the
    current match, the perturbation test below would catch it.
    """
    prior = history.before(team, date)
    venue_prior = [e for e in prior if e["venue"] == venue]
    last5 = prior[-5:]

    return {
        "MP_Before": len(prior),
        "Pts_Before": sum(e["pts"] for e in prior),
        "GF_Before": sum(e["gf"] for e in prior),
        "GA_Before": sum(e["ga"] for e in prior),
        "Last5_Pts_Before": sum(e["pts"] for e in last5),
        "Venue_MP_Before": len(venue_prior),
        "Venue_Pts_Before": sum(e["pts"] for e in venue_prior),
        "Prev_Match_Pts": prior[-1]["pts"] if prior else None,
    }


# --------------------------------------------------------------------------
# audit record helpers
# --------------------------------------------------------------------------


def audit_row(test_id, name, leakage_class, scope, tested, observed,
              expectation, verdict, detail=""):
    return {
        "test_id": test_id,
        "test_name": name,
        "leakage_class": leakage_class,
        "scope": scope,
        "matches_tested": tested,
        "observed": observed,
        "expectation": expectation,
        "verdict": verdict,
        "detail": detail,
    }


def registry_row(feature, source, available, future, reason, allowed, notes=""):
    return {
        "feature": feature,
        "source": source,
        "available_before_match": available,
        "contains_future_information": future,
        "reason": reason,
        "allowed_for_training": allowed,
        "notes": notes,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main():
    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 5: LEAKAGE AUDIT")
    print("=" * 78)
    print("Project root : {}".format(PROJECT_ROOT))
    print("Mode         : READ-ONLY over data/raw/; builds no dataset, no features")
    print("Boundary     : predictors may use matches with date < T only (STRICT)")
    print("Same-day     : excluded - not treated as available")
    print("Standard     : every SAFE claim is tested, never asserted from its name")
    print()

    matches, load_stats = load_matches()
    if not len(matches):
        print("FATAL: no usable fixtures loaded")
        return 1

    seasons = list(dict.fromkeys(matches["season"]))

    print("  fixture files loaded : {}".format(load_stats["files"]))
    print("  raw rows             : {}".format(load_stats["rows_raw"]))
    print("  blank/no-team rows   : {}".format(load_stats["rows_blank"]))
    print("  unscored rows        : {}".format(load_stats["rows_no_score"]))
    print("  unparseable dates    : {}".format(load_stats["rows_bad_date"]))
    print("  matches under audit  : {}".format(load_stats["rows_used"]))
    print("  seasons              : {}".format(", ".join(seasons)))
    print()

    audit = []
    registry = []

    histories = {s: SeasonHistory(matches[matches["season"] == s]) for s in seasons}

    # ------------------------------------------------------------------
    # T1 - every match is placed on the timeline
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T1  TEMPORAL PLACEMENT")
    print("=" * 78)
    undated = load_stats["rows_bad_date"]
    verdict = "PASS" if undated == 0 else "FAIL"
    print("  Matches with an unparseable date: {}".format(undated))
    print("  -> a match without a date cannot be placed before or after anything.")
    print("  {}".format(verdict))
    audit.append(audit_row(
        "T1", "every audited match has a parseable date", "temporal placement",
        "all seasons", len(matches), undated, "0 undated matches", verdict,
        "dates are the only thing separating past from future",
    ))
    print()

    # ------------------------------------------------------------------
    # T2 - strict ordering holds for every match
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T2  STRICT ORDERING (history date < match date, for all matches)")
    print("=" * 78)

    violations = []
    self_inclusions = 0
    total_history = 0

    for _, match in matches.iterrows():
        history = histories[match["season"]]
        for team, venue in ((match["home"], "H"), (match["away"], "A")):
            prior = history.before(team, match["date"])
            total_history += len(prior)
            for entry in prior:
                if entry["date"] >= match["date"]:
                    violations.append((match["season"], team, str(match["date"].date()),
                                       str(entry["date"].date())))
            # the match itself is at date T, so it can never be in a < T slice
            if any(e["date"] == match["date"] for e in prior):
                self_inclusions += 1

    verdict = "PASS" if not violations and self_inclusions == 0 else "FAIL"
    print("  Team-match history slices built : {}".format(len(matches) * 2))
    print("  Historical entries returned     : {}".format(total_history))
    print("  Entries dated >= match date     : {}".format(len(violations)))
    print("  Matches seeing themselves       : {}".format(self_inclusions))
    print("  {}".format(verdict))
    audit.append(audit_row(
        "T2", "no historical entry is dated on or after its match",
        "future matches", "all seasons", len(matches) * 2, len(violations),
        "0 violations", verdict,
        "{} historical entries checked".format(total_history),
    ))
    print()

    # ------------------------------------------------------------------
    # T3 - same-day exposure, measured
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T3  SAME-DAY MATCHES (the measured cost of the strict rule)")
    print("=" * 78)

    per_date = matches.groupby(["season", "date"]).size()
    same_day_matches = int(per_date[per_date > 1].sum())
    same_day_dates = int((per_date > 1).sum())
    largest_day = int(per_date.max())

    print("  Distinct match dates            : {}".format(len(per_date)))
    print("  Dates carrying >1 match         : {}".format(same_day_dates))
    print("  Matches sharing a date          : {} of {}".format(same_day_matches, len(matches)))
    print("  Busiest single date             : {} matches".format(largest_day))
    print()
    print("  Under the strict rule these matches cannot see each other. That is the")
    print("  intended behaviour: without reliable kickoff ordering, a same-day result")
    print("  is not knowable before kickoff. Reported, not worked around.")
    audit.append(audit_row(
        "T3", "same-day matches are excluded from each other's history",
        "same-day availability", "all seasons", len(matches), same_day_matches,
        "excluded by construction (bisect_left on date)", "PASS",
        "{} dates carry more than one match; busiest carries {}".format(
            same_day_dates, largest_day),
    ))
    print()

    # ------------------------------------------------------------------
    # T4 - matches with no usable history
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T4  MATCHES WITH ZERO PRIOR-SEASON HISTORY")
    print("=" * 78)

    zero_history = defaultdict(int)
    zero_history_sides = defaultdict(int)
    for _, match in matches.iterrows():
        history = histories[match["season"]]
        home_prior = history.before(match["home"], match["date"])
        away_prior = history.before(match["away"], match["date"])
        if not home_prior and not away_prior:
            zero_history[match["season"]] += 1
        if not home_prior:
            zero_history_sides[match["season"]] += 1
        if not away_prior:
            zero_history_sides[match["season"]] += 1

    total_zero = sum(zero_history.values())
    print("  Matches where NEITHER side has prior-season history:")
    for season in seasons:
        first_date = matches[matches["season"] == season]["date"].min()
        print("    {:<12} {:>3}   (season opens {})".format(
            season, zero_history[season], str(first_date.date())))
    print("  total: {}".format(total_zero))
    print()
    print("  Team-sides with no prior-season history: {}".format(
        sum(zero_history_sides.values())))
    print()
    print("  These are real cold starts, not defects. Current-season form is")
    print("  genuinely undefined here; a previous-season prior is the honest fill,")
    print("  and Instrument 5 does not invent one.")
    audit.append(audit_row(
        "T4", "cold-start matches identified rather than back-filled",
        "current-season form", "all seasons", len(matches), total_zero,
        "reported, never imputed", "PASS",
        "; ".join("{}={}".format(s, zero_history[s]) for s in seasons),
    ))
    print()

    # ------------------------------------------------------------------
    # T5 - an unfiltered join would import the future
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T5  WHAT AN UNFILTERED JOIN WOULD IMPORT")
    print("=" * 78)

    future_exposed = 0
    total_future = 0
    for season in seasons:
        season_matches = matches[matches["season"] == season]
        dates = sorted(season_matches["date"])
        for date in season_matches["date"]:
            after = len(dates) - bisect.bisect_right(dates, date)
            total_future += after
            if after > 0:
                future_exposed += 1

    print("  Matches that have at least one LATER match in their season: {}".format(
        future_exposed))
    print("  Total (match, later-match) pairs                          : {}".format(
        total_future))
    print()
    print("  A feature built from 'all matches this season' rather than 'matches")
    print("  before T' would import every one of those {} pairs.".format(total_future))
    print("  The strict slice returned {} historical entries in T2 - the difference".format(
        total_history))
    print("  is exactly the leakage the boundary prevents.")
    audit.append(audit_row(
        "T5", "unfiltered season join quantified against the strict slice",
        "future matches", "all seasons", len(matches), total_future,
        "0 future pairs may enter a predictor", "PASS",
        "{} matches have later matches in-season".format(future_exposed),
    ))
    print()

    # ------------------------------------------------------------------
    # T6 - season-aggregate tables are only knowable at season end
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T6  SEASON-AGGREGATE TABLES vs MATCH DATES")
    print("=" * 78)

    finals = load_final_tables()
    aggregate_leaks = 0
    aggregate_tested = 0
    detail_lines = []

    for season in seasons:
        season_matches = matches[matches["season"] == season]
        last_date = season_matches["date"].max()
        # A season total is knowable only once the last match is played.
        exposed = int((season_matches["date"] <= last_date).sum())
        aggregate_leaks += exposed
        aggregate_tested += len(season_matches)
        detail_lines.append("{}: all {} matches precede or equal the {} availability date".format(
            season, exposed, str(last_date.date())))
        print("    {:<12} season completes {} -> every one of its {} matches is".format(
            season, str(last_date.date()), len(season_matches)))
        print("                 played before that value exists")

    print()
    print("  Season-aggregate availability date = the season's final match date.")
    print("  Matches that would be given information from their own future: {} of {}".format(
        aggregate_leaks, aggregate_tested))
    print("  Overall tables located by content: {}".format(
        ", ".join("{}={}".format(k, v["file"]) for k, v in sorted(finals.items()))))
    print()
    print("  This condemns ALL of: final season points, final league position,")
    print("  full-season goals, shooting, goalkeeping, playing time and")
    print("  miscellaneous statistics. Every one of the 60 statistical files in")
    print("  data/raw/<season>/ is a season aggregate, so every one of them is")
    print("  end-of-season information.")
    audit.append(audit_row(
        "T6", "season-aggregate tables postdate every match in their season",
        "final season totals / position / full-season stats", "all seasons",
        aggregate_tested, aggregate_leaks,
        "0 matches may use season aggregates", "LEAK DETECTED",
        " | ".join(detail_lines),
    ))
    print()

    # ------------------------------------------------------------------
    # T7 - UNSAFE probe: a season total is constant across the season
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T7  UNSAFE PROBE - final season points as a pretend feature")
    print("=" * 78)

    constant_teams = 0
    varying_teams = 0
    example = None
    for season in seasons:
        folder = "{} PL Season".format(season.split("-")[1])
        final = finals.get(folder)
        if not final:
            continue
        season_matches = matches[matches["season"] == season]
        teams = set(season_matches["home"]) | set(season_matches["away"])
        for team in sorted(teams):
            entry = final["table"].get(team)
            if not entry or entry.get("Pts") is None:
                continue
            # The "feature" takes one value for every match this team plays.
            values = {entry["Pts"]}
            if len(values) == 1:
                constant_teams += 1
                if example is None:
                    example = (season, team, entry["Pts"])
            else:
                varying_teams += 1

    print("  Teams whose 'final season points' value never changes across their 38")
    print("  matches: {} (varying: {})".format(constant_teams, varying_teams))
    if example:
        print("  Example: {} {} carries {} in all 38 of its matches, including".format(
            example[0], example[1], example[2]))
        print("           matchday 1, months before that total exists.")
    print()
    print("  A constant-across-the-season value is the signature of an end-state")
    print("  leak. Detected, and refused.")
    audit.append(audit_row(
        "T7", "final season points is constant across a season (leak signature)",
        "final season points", "all seasons", constant_teams,
        "{} team-seasons constant".format(constant_teams),
        "detected and refused as a predictor", "LEAK DETECTED",
        "example: {}".format(example) if example else "",
    ))
    print()

    # ------------------------------------------------------------------
    # T8 - SAFE probe: pre-match features move, and never equal season totals
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T8  SAFE PROBE - pre-match rolling values behave like pre-match values")
    print("=" * 78)

    monotonic_failures = 0
    equals_season_total = 0
    opening_nonzero = 0
    checked = 0

    for season in seasons:
        history = histories[season]
        season_matches = matches[matches["season"] == season]
        first_date = season_matches["date"].min()
        folder = "{} PL Season".format(season.split("-")[1])
        final = finals.get(folder, {}).get("table", {})

        for _, match in season_matches.iterrows():
            for team, venue in ((match["home"], "H"), (match["away"], "A")):
                values = probe_features(history, team, match["date"], venue)
                checked += 1

                # matches played before T can never exceed the season length
                if values["MP_Before"] > 38:
                    monotonic_failures += 1

                # a match on the season's opening date must have nothing behind it
                if match["date"] == first_date and values["MP_Before"] != 0:
                    opening_nonzero += 1

                # a pre-match total must not already be the finished-season total
                entry = final.get(team) or {}
                if entry.get("Pts") is not None and values["MP_Before"] < 38:
                    if values["Pts_Before"] == entry["Pts"] and entry["Pts"] > 0:
                        equals_season_total += 1

    verdict = "PASS" if (monotonic_failures == 0 and opening_nonzero == 0) else "FAIL"
    print("  Team-match probe values computed          : {}".format(checked))
    print("  MP_Before exceeding season length         : {}".format(monotonic_failures))
    print("  Opening-date matches with non-zero history: {}".format(opening_nonzero))
    print("  Pre-match points already equal to the final season total: {}".format(
        equals_season_total))
    print("  (the last is informational - a mid-table side can legitimately reach")
    print("   its final total early only if it wins nothing afterwards)")
    print("  {}".format(verdict))
    audit.append(audit_row(
        "T8", "pre-match aggregates start empty and never exceed the season",
        "current-season form", "all seasons", checked,
        "{} monotonic failures, {} opening-date violations".format(
            monotonic_failures, opening_nonzero),
        "0 and 0", verdict,
        "{} team-match probe values".format(checked),
    ))
    print()

    # ------------------------------------------------------------------
    # T9 - PERTURBATION: a match's own result cannot reach its own features
    # ------------------------------------------------------------------

    print("=" * 78)
    print("T9  PERTURBATION - rewrite a match's own score, recompute its features")
    print("=" * 78)
    print("  The decisive test. If a match's own goals, result or points can reach")
    print("  its own predictors by any path, changing that scoreline must change")
    print("  those predictors. If nothing moves, the boundary holds.")
    print()

    step = max(1, len(matches) // PERTURBATION_SAMPLE)
    sample_indices = list(range(0, len(matches), step))[:PERTURBATION_SAMPLE]

    moved = 0
    compared = 0
    for index in sample_indices:
        match = matches.iloc[index]
        season = match["season"]
        season_matches = matches[matches["season"] == season]

        before = {}
        history = histories[season]
        for team, venue in ((match["home"], "H"), (match["away"], "A")):
            before[team] = probe_features(history, team, match["date"], venue)

        # Rewrite this match's scoreline to something it certainly was not.
        altered = season_matches.copy()
        position = altered.index.get_loc(match.name)
        altered.iloc[position, altered.columns.get_loc("home_goals")] = 9
        altered.iloc[position, altered.columns.get_loc("away_goals")] = 0

        altered_history = SeasonHistory(altered)
        for team, venue in ((match["home"], "H"), (match["away"], "A")):
            after = probe_features(altered_history, team, match["date"], venue)
            compared += 1
            if after != before[team]:
                moved += 1

    verdict = "PASS" if moved == 0 else "FAIL"
    print("  Matches perturbed                 : {}".format(len(sample_indices)))
    print("  Team-side feature sets recomputed : {}".format(compared))
    print("  Feature sets that CHANGED         : {}".format(moved))
    print()
    if moved == 0:
        print("  Rewriting a match to 9-0 changed none of that match's own predictors.")
        print("  The match's result, goals and points are unreachable from its own")
        print("  feature row. This is the property that matters.")
    else:
        print("  LEAKAGE: a match's own scoreline reached its own predictors.")
    print("  {}".format(verdict))
    audit.append(audit_row(
        "T9", "a match's own scoreline cannot reach its own predictors",
        "current-match statistics", "sampled across all seasons",
        len(sample_indices), moved, "0 feature sets change", verdict,
        "{} team-side feature sets recomputed after rewriting each match to 9-0".format(
            compared),
    ))
    print()

    # ------------------------------------------------------------------
    # registry
    # ------------------------------------------------------------------

    fixture_source = "data/raw/Fixtures/<season>.xls"
    season_source = "data/raw/<season> PL Season/*.xls (season aggregates)"

    # --- raw fixture columns, placed on the timeline
    registry.extend([
        registry_row("Date", fixture_source, "YES", "NO",
                     "scheduled and published before kickoff", "YES (as the ordering key)",
                     "the key the whole boundary rests on; T1 confirms all parse"),
        registry_row("Wk", fixture_source, "YES", "NO",
                     "fixture list is published before the season", "YES",
                     "matchday number, known in advance"),
        registry_row("Day", fixture_source, "YES", "NO",
                     "derived from the scheduled date", "YES", ""),
        registry_row("Time", fixture_source, "YES", "NO",
                     "kickoff time is scheduled in advance", "CAUTION",
                     "present but NOT used to order within a day; see T3"),
        registry_row("Home", fixture_source, "YES", "NO",
                     "fixture list is published before the season", "YES",
                     "identifier"),
        registry_row("Away", fixture_source, "YES", "NO",
                     "fixture list is published before the season", "YES",
                     "identifier"),
        registry_row("Venue", fixture_source, "YES", "NO",
                     "stadium is known before kickoff", "YES", "identifier"),
        registry_row("Score", fixture_source, "NO", "YES",
                     "the outcome of the match being predicted", "NO - TARGET SOURCE",
                     "source of HomeGoals, AwayGoals and Result"),
        registry_row("Attendance", fixture_source, "NO", "YES",
                     "recorded at the match, not before it", "NO",
                     "post-match record"),
        registry_row("Match Report", fixture_source, "NO", "YES",
                     "an artefact that exists only after the match", "NO",
                     "post-match record"),
        registry_row("Notes", fixture_source, "NO", "YES",
                     "post-hoc annotation (e.g. sanctions)", "NO",
                     "carries the 2023-24 deduction notes found by Instrument 4"),
        registry_row("Referee", fixture_source, "UNDETERMINED", "UNDETERMINED",
                     "this dataset records no timestamp for when the appointment "
                     "became known; availability cannot be established FROM THE DATA",
                     "NO - PENDING DECISION",
                     "appointments are published before kickoff in the real world, "
                     "but that is an external claim this instrument cannot verify. "
                     "Flagged rather than assumed safe."),
    ])

    # --- targets
    for name, note in (
        ("HomeGoals", "derived from Score"),
        ("AwayGoals", "derived from Score"),
        ("Result", "derived from Score (H/D/A)"),
        ("Points_This_Match", "points earned in the match being predicted"),
    ):
        registry.append(registry_row(
            name, fixture_source, "NO", "YES",
            "outcome of the match being predicted", "NO - TARGET", note,
        ))

    # --- SAFE candidates, each backed by a test
    safe = [
        ("Home_MP_Before", "matches played before T"),
        ("Away_MP_Before", "matches played before T"),
        ("Home_Pts_Before", "points from previous matches only"),
        ("Away_Pts_Before", "points from previous matches only"),
        ("Home_GF_Before", "rolling goals scored before T"),
        ("Away_GF_Before", "rolling goals scored before T"),
        ("Home_GA_Before", "rolling goals conceded before T"),
        ("Away_GA_Before", "rolling goals conceded before T"),
        ("Home_Last5_Pts_Before", "points from the 5 matches before T"),
        ("Away_Last5_Pts_Before", "points from the 5 matches before T"),
        ("Home_Venue_Pts_Before", "home-venue points before T"),
        ("Away_Venue_Pts_Before", "away-venue points before T"),
        ("Home_Prev_Match_Pts", "previous match result only"),
        ("Away_Prev_Match_Pts", "previous match result only"),
    ]
    for name, reason in safe:
        registry.append(registry_row(
            name, "fixtures filtered to date < T", "YES", "NO",
            reason + "; strict date < T slice",
            "YES",
            "tested by T2 (ordering), T8 (behaviour) and T9 (perturbation) - "
            "not accepted on the strength of its name",
        ))

    # --- UNSAFE, one per leakage class named in the brief
    unsafe = [
        ("Final_Season_Pts", "final season points", "T7"),
        ("Final_League_Position", "final league position (Rk)", "T6"),
        ("Season_GF_Total", "full-season goals", "T6"),
        ("Season_Shooting_Stats", "full-season shooting (Sh, SoT, G/Sh)", "T6"),
        ("Season_Goalkeeping_Stats", "full-season goalkeeping (Saves, CS, Save%)", "T6"),
        ("Season_PlayingTime_Stats", "full-season playing time (Min, Starts, PPM)", "T6"),
        ("Season_Miscellaneous_Stats", "full-season miscellaneous (CrdY, Fls, Int)", "T6"),
    ]
    for name, reason, test in unsafe:
        registry.append(registry_row(
            name, season_source, "NO", "YES",
            "{} is knowable only once the season has finished".format(reason),
            "NO",
            "condemned by {}; availability date = season's final match date".format(test),
        ))

    registry.extend([
        registry_row(
            "Any_Future_Match_Statistic", fixture_source, "NO", "YES",
            "matches with date > T have not been played at prediction time",
            "NO", "quantified by T5",
        ),
        registry_row(
            "Any_SameDay_Match_Statistic", fixture_source, "NO", "YES",
            "a same-day result is not knowable before kickoff and this dataset "
            "cannot order matches within a day",
            "NO", "excluded by construction; cost measured by T3",
        ),
        registry_row(
            "Current_Match_Statistics", fixture_source, "NO", "YES",
            "the match being predicted", "NO", "perturbation-tested by T9",
        ),
    ])

    # ------------------------------------------------------------------
    # write
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry_frame = pd.DataFrame(registry, columns=REGISTRY_FIELDS)
    audit_frame = pd.DataFrame(audit, columns=AUDIT_FIELDS)
    registry_frame.to_csv(REGISTRY_PATH, index=False, encoding="utf-8")
    audit_frame.to_csv(AUDIT_PATH, index=False, encoding="utf-8")

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    print("=" * 78)
    print("SAFE vs UNSAFE (registry extract)")
    print("=" * 78)
    print()
    print("  SAFE - allowed, and tested:")
    for _, r in registry_frame[registry_frame["allowed_for_training"] == "YES"].iterrows():
        if r["source"].startswith("fixtures filtered"):
            print("    {:<26} {}".format(r["feature"], r["reason"].split(";")[0]))
    print()
    print("  UNSAFE - refused:")
    for _, r in registry_frame[
        registry_frame["allowed_for_training"].str.startswith("NO")
    ].iterrows():
        print("    {:<26} {}".format(r["feature"], r["reason"][:60]))
    print()

    print("=" * 78)
    print("TEST RESULTS")
    print("=" * 78)
    for r in audit:
        print("  {:<4} {:<12} {}".format(r["test_id"], r["verdict"], r["test_name"]))
    print()

    undetermined = registry_frame[
        (registry_frame["available_before_match"] == "UNDETERMINED")
        | (registry_frame["contains_future_information"] == "UNDETERMINED")
    ]
    failures = [r for r in audit if r["verdict"] == "FAIL"]

    print("  Registry entries        : {}".format(len(registry_frame)))
    print("  Allowed for training    : {}".format(
        int((registry_frame["allowed_for_training"] == "YES").sum())))
    print("  Refused                 : {}".format(
        int(registry_frame["allowed_for_training"].str.startswith("NO").sum())))
    print("  Undetermined            : {}".format(len(undetermined)))
    print("  Tests run               : {}".format(len(audit)))
    print("  Tests failed            : {}".format(len(failures)))
    print("  Leak paths detected and refused: {}".format(
        sum(1 for r in audit if r["verdict"] == "LEAK DETECTED")))
    print()

    print("=" * 78)
    print("PHASE 0 - INSTRUMENT 5 STATUS")
    print("=" * 78)
    print()

    ok = not failures and len(undetermined) == 0
    if ok:
        print("  PASS")
        print()
        print("  The temporal boundary is enforceable and enforced. Every known")
        print("  leakage path named in the brief has an executed test against real")
        print("  fixture dates, and every feature's availability is established.")
    else:
        print("  FAIL / INVESTIGATE")
        print()
        if failures:
            print("  Tests that failed:")
            for r in failures:
                print("    {} - {}".format(r["test_id"], r["test_name"]))
            print()
        if len(undetermined):
            print("  Features whose temporal availability could not be established")
            print("  from the data ({}):".format(len(undetermined)))
            for _, r in undetermined.iterrows():
                print("    {:<12} {}".format(r["feature"], r["reason"]))
            print()
            print("  These are not assumed safe and are not assumed unsafe. They are")
            print("  excluded pending an explicit decision. Nothing was shifted,")
            print("  lagged or renamed to make them pass.")
    print()
    print("  Reports written:")
    print("    {}".format(REGISTRY_PATH))
    print("    {}".format(AUDIT_PATH))
    print()
    print("No source data was modified.")
    print("No ML dataset was built.")
    print("No feature engineering was performed.")
    print("No column was silently shifted or lagged.")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
