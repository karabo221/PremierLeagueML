"""
PHASE 3 - INSTRUMENT 1: RAW FEATURE AUDIT

Answers one question before any Phase 3 feature is written:

    Of everything FBref gives us, what can legitimately become a pre-match
    feature, and in what representation?

Phase 0 condemned all 60 season-aggregate tables as end-of-season information.
That verdict was about the CURRENT season and it stands. This instrument tests
the one door it left open: for a match in season S, the aggregate for season
S-1 was complete and public before S kicked off. That is the LAG-1 PRIOR, and
its safety is a property of dates, so it is measured against dates here.

WHAT THIS INSTRUMENT DOES

  - types all 60 files by column hierarchy (never by filename - Phase 0
    Finding 1 proved two filenames lie) and by row prefix for squad/opponent
  - verifies the folder-name -> season mapping against fixture-derived tables
    instead of trusting "2022 PL Season" to mean 2021-22
  - measures the lag-1 temporal boundary on real dates
  - catalogues every stat column into the nine feature families, and tests
    each one for constancy, derivability, squad/opponent redundancy and
    cross-season regime drift
  - reports coverage: which team-seasons actually have a lag-1 prior

WHAT THIS INSTRUMENT DOES NOT DO

  It builds no features and writes no modelling dataset. It trains nothing.
  It does not touch outputs/phase0_evaluation_*. It is STRICTLY READ-ONLY
  over data/raw/ - nothing renamed, moved, deleted or modified.

THE STANDARD OF PROOF

  A column is not usable because its name sounds like team strength. Every
  ALLOW verdict below is backed by a test that would fail if the claim were
  false; every REFUSE names the test that condemned it.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_statistical_integrity import (  # noqa: E402
    classify_perspective,
    classify_table_type,
    column_path,
    decode,
    find_squad_column,
    parse_tables,
    render_path,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs"

MATCHES_PATH = OUT_DIR / "phase1_matches.csv"
STATE_PATH = OUT_DIR / "phase1_historical_team_state.csv"

CATALOGUE_PATH = OUT_DIR / "phase3_feature_catalogue.csv"
COVERAGE_PATH = OUT_DIR / "phase3_lag1_coverage.csv"
AUDIT_PATH = OUT_DIR / "phase3_raw_feature_audit.csv"

# Folder label -> season it actually describes. ASSERTED HERE, VERIFIED BY T1.
FOLDER_SEASON = {
    "2022 PL Season": "2021-2022",
    "2023 PL Season": "2022-2023",
    "2024 PL Season": "2023-2024",
    "2025 PL Season": "2024-2025",
    "2026 PL Season": "2025-2026",
}

SEASON_ORDER = ["2021-2022", "2022-2023", "2023-2024", "2024-2025", "2025-2026"]
PREV_SEASON = {s: (SEASON_ORDER[i - 1] if i else None) for i, s in enumerate(SEASON_ORDER)}

DRIFT_LIMIT = 0.50


# ==========================================================================
# feature families
# ==========================================================================
#
# The nine families the project asked for. A column lands in exactly one, by
# its stat, then by the perspective of the table it came from. Perspective is
# what turns an attacking column into a defensive one: "vs Arsenal" shooting
# is not Arsenal shooting, it is what Arsenal conceded.

STAT_FAMILY = {
    # team form - outcome counters
    "MP": "team_form", "W": "team_form", "D": "team_form", "L": "team_form",
    "Pts": "team_form", "Pts/MP": "team_form", "PPM": "team_form", "Rk": "team_form",
    "GD": "team_form",
    # goals
    "GF": "attacking_strength", "GA": "defensive_strength",
    # shooting
    "Sh": "shooting", "SoT": "shooting", "SoT%": "shooting", "Sh/90": "shooting",
    "SoT/90": "shooting", "G/Sh": "shooting", "G/SoT": "shooting",
    # goalkeeping
    "SoTA": "goalkeeping", "Saves": "goalkeeping", "Save%": "goalkeeping",
    "CS": "goalkeeping", "CS%": "goalkeeping", "GA90": "goalkeeping",
    "PKatt": "goalkeeping", "PKA": "goalkeeping", "PKsv": "goalkeeping",
    "PKm": "goalkeeping",
    # playing time / squad management
    "Min": "playing_time", "Mn/MP": "playing_time", "Min%": "playing_time",
    "90s": "playing_time", "Starts": "playing_time", "Mn/Start": "playing_time",
    "Compl": "playing_time", "Subs": "playing_time", "Mn/Sub": "playing_time",
    "unSub": "playing_time", "Age": "playing_time", "# Pl": "playing_time",
    "onG": "playing_time", "onGA": "playing_time", "+/-": "playing_time",
    "+/-90": "playing_time",
    # miscellaneous
    "CrdY": "miscellaneous", "CrdR": "miscellaneous", "2CrdY": "miscellaneous",
    "Fls": "miscellaneous", "Fld": "miscellaneous", "Off": "miscellaneous",
    "Crs": "miscellaneous", "Int": "miscellaneous", "TklW": "miscellaneous",
    "PKwon": "miscellaneous", "PKcon": "miscellaneous", "OG": "miscellaneous",
    # standard / possession
    "Poss": "attacking_strength", "Gls": "attacking_strength",
    "Ast": "attacking_strength", "G+A": "attacking_strength",
    "G-PK": "attacking_strength", "PK": "attacking_strength",
    "G+A-PK": "attacking_strength",
    # non-features
    "Attendance": "contextual", "Top Team Scorer": "contextual",
    "Goalkeeper": "contextual", "Notes": "contextual", "Squad": "identity",
}

# Families that FLIP when read off an opponent table.
FLIP = {
    "attacking_strength": "defensive_strength",
    "defensive_strength": "attacking_strength",
    "shooting": "shooting_against",
    "goalkeeping": "goalkeeping_against",
}


def stat_leaf(rendered):
    """Last hierarchy level - the stat itself, stripped of its block."""
    return rendered.split(" | ")[-1].strip()


def family_for(rendered, perspective, table_type):
    leaf = stat_leaf(rendered)
    fam = STAT_FAMILY.get(leaf, "unclassified")
    if table_type == "Home/Away":
        return "home_away_strength"
    if perspective == "Opponent":
        return FLIP.get(fam, fam)
    return fam


# ==========================================================================
# audit record helpers
# ==========================================================================

AUDIT = []


def record(test_id, name, scope, tested, observed, expectation, verdict, detail=""):
    AUDIT.append({
        "test_id": test_id, "test_name": name, "scope": scope,
        "items_tested": tested, "observed": observed,
        "expectation": expectation, "verdict": verdict, "detail": detail,
    })
    print("  [{}] {:<6} {:<46} {}".format(test_id, verdict, name, observed))


def banner(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print()


# ==========================================================================
# loading
# ==========================================================================


def load_raw_tables():
    """Type every file from CONTENT. Returns {(season, type, perspective): frame}."""
    tables, manifest = {}, []
    for folder, season in FOLDER_SEASON.items():
        for path in sorted((RAW_DIR / folder).glob("*.xls")):
            text, _enc = decode(path.read_bytes())
            parsed, _flavor, err = parse_tables(text)
            if err:
                raise SystemExit("FATAL: cannot parse {}: {}".format(path, err))
            table = parsed[0]
            ttype = classify_table_type(table)
            squad_col = find_squad_column(table)
            persp = classify_perspective(table, squad_col, ttype)
            manifest.append({
                "season": season, "folder": folder, "filename": path.name,
                "table_type": ttype, "perspective": persp,
                "n_rows": len(table), "n_cols": table.shape[1],
            })
            key = (season, ttype, persp)
            if key in tables:  # byte-duplicates handled by T0b; first wins
                continue
            frame = table.copy()
            frame.columns = [render_path(column_path(c)) for c in frame.columns]
            squad_cols = [c for c in frame.columns
                          if stat_leaf(c).lower() in ("squad", "team")]
            if not squad_cols:
                continue
            sq = squad_cols[0]
            frame[sq] = (frame[sq].astype(str)
                         .str.replace(r"^vs\s+", "", regex=True).str.strip())
            tables[key] = frame.rename(columns={sq: "Squad"})
    return tables, pd.DataFrame(manifest)


def numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def filename_claim(name):
    lowered = name.lower()
    for key, val in (("goalkeep", "Goalkeeping"), ("shoot", "Shooting"),
                     ("sthoot", "Shooting"), ("misc", "Miscellaneous"),
                     ("playing", "Playing Time"), ("standard", "Standard"),
                     ("home", "Home/Away"), ("overall", "Overall")):
        if key in lowered:
            return val
    return "?"


# ==========================================================================
# main
# ==========================================================================


def main():
    banner("PHASE 3 - INSTRUMENT 1: RAW FEATURE AUDIT")
    print("Project root: {}".format(PROJECT_ROOT))
    print("Read-only over data/raw/. No feature dataset is written.")

    for p in (MATCHES_PATH, STATE_PATH):
        if not p.exists():
            raise SystemExit(
                "FATAL: missing upstream artefact {}. Run Phase 1 first.".format(p))

    matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"],
                          float_precision="round_trip")
    tables, manifest = load_raw_tables()

    # ------------------------------------------------------------------
    banner("1. FILE TYPING - CONTENT OVER FILENAME")
    # ------------------------------------------------------------------
    print(manifest.pivot_table(index=["table_type", "perspective"], columns="season",
                               values="filename", aggfunc="size")
          .fillna(0).astype(int).to_string())
    print()

    unknown = manifest[(manifest.table_type == "UNKNOWN")
                       | (manifest.perspective == "UNKNOWN")]
    record("T0a", "every file typed from its column hierarchy", "60 files",
           len(manifest),
           "{} typed, {} unknown".format(len(manifest) - len(unknown), len(unknown)),
           "0 unknown", "PASS" if unknown.empty else "FAIL")

    liars = [r for _, r in manifest.iterrows()
             if filename_claim(r.filename) not in ("?", r.table_type)]
    record("T0b", "filenames that misdescribe their contents", "60 files",
           len(manifest), "{} misdescribed".format(len(liars)),
           "documented, and never used for typing", "INFO",
           "; ".join("{}/{} claims {}, is {}".format(
               r.season, r.filename, filename_claim(r.filename), r.table_type)
               for r in liars))

    missing_slots = []
    for season in SEASON_ORDER:
        for ttype, persp in [("Shooting", "Squad"), ("Shooting", "Opponent"),
                             ("Goalkeeping", "Squad"), ("Goalkeeping", "Opponent"),
                             ("Standard", "Squad"), ("Standard", "Opponent"),
                             ("Miscellaneous", "Squad"), ("Miscellaneous", "Opponent"),
                             ("Playing Time", "Squad"), ("Playing Time", "Opponent"),
                             ("Overall", "League"), ("Home/Away", "League")]:
            if (season, ttype, persp) not in tables:
                missing_slots.append("{} {}/{}".format(season, ttype, persp))
    record("T0c", "season x table-type coverage", "60 slots", 60,
           "{} slots empty".format(len(missing_slots)),
           "1 - 2025-26 Shooting/Opponent (Phase 0 Finding 2)",
           "PASS" if len(missing_slots) == 1 else "FAIL", "; ".join(missing_slots))

    # ------------------------------------------------------------------
    banner("2. SEASON MAPPING - VERIFIED, NOT ASSUMED")
    # ------------------------------------------------------------------
    # "2022 PL Season" is claimed to hold 2021-22. Check the Overall table's
    # goals-for vector against the table rebuilt from that season's fixtures.
    # A wrong mapping would silently shift every prior by a year, which is the
    # single most damaging error Phase 3 could make.
    mismatches = []
    for season in SEASON_ORDER:
        overall = tables.get((season, "Overall", "League"))
        if overall is None:
            mismatches.append((season, "no Overall table"))
            continue
        d = matches[matches.season == season]
        gf = {}
        for _, r in d.iterrows():
            gf[r.home_team] = gf.get(r.home_team, 0) + r.home_goals
            gf[r.away_team] = gf.get(r.away_team, 0) + r.away_goals
        fb = dict(zip(overall["Squad"], numeric(overall["GF"])))
        if set(fb) != set(gf):
            mismatches.append((season, "squad set differs"))
            continue
        bad = [t for t in gf if int(fb[t]) != int(gf[t])]
        if bad:
            mismatches.append((season, "GF differs for {}".format(bad)))

    record("T1", "folder->season mapping reproduces fixture goals-for", "5 seasons",
           5, "{} mismatched".format(len(mismatches)), "0 mismatched",
           "PASS" if not mismatches else "FAIL", str(mismatches))

    # ------------------------------------------------------------------
    banner("3. THE LAG-1 TEMPORAL BOUNDARY")
    # ------------------------------------------------------------------
    # A season aggregate becomes available on that season's FINAL match date.
    # For the lag-1 prior to be legal, that date must precede the EARLIEST
    # match of the target season - for every season pair, with no exceptions.
    bounds = matches.groupby("season")["date"].agg(["min", "max"])
    rows, violations = [], 0
    for season in SEASON_ORDER:
        prev = PREV_SEASON[season]
        if prev is None:
            rows.append({"target_season": season, "prior_season": None,
                         "prior_available_from": pd.NaT,
                         "target_first_match": bounds.loc[season, "min"],
                         "gap_days": np.nan, "legal": None,
                         "note": "earliest season in dataset - no prior exists"})
            continue
        avail = bounds.loc[prev, "max"]
        first = bounds.loc[season, "min"]
        legal = bool(avail < first)
        violations += (not legal)
        rows.append({"target_season": season, "prior_season": prev,
                     "prior_available_from": avail, "target_first_match": first,
                     "gap_days": (first - avail).days, "legal": legal, "note": ""})
    boundary = pd.DataFrame(rows)
    print(boundary.to_string(index=False))
    print()
    record("T2", "lag-1 aggregate complete before target season starts",
           "4 season pairs", 4,
           "{} violations, min gap {} days".format(
               violations, int(boundary.gap_days.dropna().min())),
           "0 violations", "PASS" if violations == 0 else "FAIL")

    # The same aggregate must predate EVERY match of the target season, not
    # just the first. That is implied above, but implication is not
    # measurement, so it is measured.
    late = 0
    for _, r in boundary.dropna(subset=["prior_season"]).iterrows():
        d = matches[matches.season == r.target_season]
        late += int((d["date"] <= r.prior_available_from).sum())
    record("T3", "no target-season match predates its own prior", "1,520 matches",
           int((matches.season != SEASON_ORDER[0]).sum()),
           "{} matches on/before their prior's availability date".format(late),
           "0", "PASS" if late == 0 else "FAIL")

    # The condemned direction, restated as a measurement so the ALLOW above
    # cannot be mistaken for a blanket amnesty on season aggregates.
    same_season_early = 0
    for season in SEASON_ORDER:
        d = matches[matches.season == season]
        same_season_early += int((d["date"] < bounds.loc[season, "max"]).sum())
    record("T4", "same-season aggregate still condemned", "1,900 matches",
           len(matches),
           "{} matches precede their own season's aggregate".format(same_season_early),
           "the overwhelming majority - Phase 0 T6/T7 stands",
           "PASS" if same_season_early > 1800 else "FAIL",
           "re-measured, not softened")

    # ------------------------------------------------------------------
    banner("4. LAG-1 COVERAGE - WHICH TEAM-SEASONS HAVE A PRIOR")
    # ------------------------------------------------------------------
    def prior_squads(season):
        prev = PREV_SEASON[season]
        if prev is None or (prev, "Overall", "League") not in tables:
            return set()
        return set(tables[(prev, "Overall", "League")]["Squad"])

    cov_rows = []
    for season in SEASON_ORDER:
        prev = PREV_SEASON[season]
        prev_teams = prior_squads(season)
        for t in sorted(set(matches[matches.season == season].home_team)):
            has = t in prev_teams
            cov_rows.append({
                "season": season, "team": t, "prior_season": prev,
                "has_lag1_prior": has,
                "status": ("available" if has
                           else "no_prior_season_in_dataset" if prev is None
                           else "absent_from_previous_season")})
    coverage = pd.DataFrame(cov_rows)
    print(coverage.groupby(["season", "status"]).size()
          .unstack(fill_value=0).to_string())
    print()

    home_cov = sum(1 for _, r in matches.iterrows()
                   if r.home_team in prior_squads(r.season))
    away_cov = sum(1 for _, r in matches.iterrows()
                   if r.away_team in prior_squads(r.season))
    record("T5", "lag-1 prior coverage over team-sides", "3,800 team-sides",
           len(matches) * 2,
           "{} of {} covered ({:.1f}%)".format(
               home_cov + away_cov, len(matches) * 2,
               100 * (home_cov + away_cov) / (len(matches) * 2)),
           "matches Phase 1's 1,292 per side", "INFO")

    # Cross-check against what Phase 1 already established, so a divergence in
    # the prior-availability logic surfaces here rather than in modelling.
    state = pd.read_csv(STATE_PATH, float_precision="round_trip",
                        parse_dates=["date", "home_previous_match_date",
                                     "away_previous_match_date"])
    p1_home = int(state["home_previous_season_status"].eq("available").sum())
    record("T6", "coverage agrees with Phase 1 prior-availability",
           "1,900 home sides", len(matches),
           "instrument {} vs Phase 1 {}".format(home_cov, p1_home),
           "identical", "PASS" if home_cov == p1_home else "FAIL")

    # ------------------------------------------------------------------
    banner("5. COLUMN CATALOGUE AND REPRESENTATION TESTS")
    # ------------------------------------------------------------------
    cat_rows = []
    for (season, ttype, persp), frame in tables.items():
        if ttype == "UNKNOWN":
            continue
        for col in frame.columns:
            if col == "Squad":
                continue
            vals = numeric(frame[col])
            cat_rows.append({
                "season": season, "table_type": ttype, "perspective": persp,
                "column": col, "stat": stat_leaf(col),
                "family": family_for(col, persp, ttype),
                "n_numeric": int(vals.notna().sum()),
                "mean": vals.mean(), "sd": vals.std(ddof=1)})
    cat = pd.DataFrame(cat_rows)

    grp = cat.groupby(["table_type", "perspective", "column", "stat", "family"],
                      dropna=False)
    summary = grp.agg(
        seasons=("season", "nunique"),
        numeric_seasons=("n_numeric", lambda s: int((s > 0).sum())),
        within_sd=("sd", "mean"),
        mean_of_means=("mean", "mean"),
        between_season_sd=("mean", lambda s: s.std(ddof=1))).reset_index()

    summary["is_numeric"] = summary.numeric_seasons > 0
    summary["is_constant_within_season"] = (
        summary.is_numeric & (summary.within_sd.fillna(1.0).abs() < 1e-12))
    summary["drift_ratio"] = (summary.between_season_sd
                              / summary.within_sd.replace(0, np.nan))

    # T7 - constant columns carry no information.
    consts = summary[summary.is_constant_within_season]
    record("T7", "columns constant across all 20 teams", "all stat columns",
           int(summary.is_numeric.sum()), "{} constant".format(len(consts)),
           "flagged and refused", "INFO",
           "; ".join("{}/{}:{}".format(r.table_type, r.perspective, r.column)
                     for _, r in consts.iterrows()))

    # T8 - regime drift. A column whose league mean moves more between seasons
    # than teams differ within one is measuring the season, not the team. Under
    # whole-season test folds that is a fold fingerprint, so such a column must
    # never enter as a raw level.
    drifty = (summary[summary.drift_ratio > DRIFT_LIMIT]
              .sort_values("drift_ratio", ascending=False))
    record("T8", "cross-season regime drift vs within-season spread",
           "all stat columns", int(summary.drift_ratio.notna().sum()),
           "{} columns with drift ratio > {}".format(len(drifty), DRIFT_LIMIT),
           "all must be league-relative, never raw", "INFO")
    if len(drifty):
        print()
        print("  Columns whose season-to-season drift rivals team-to-team spread:")
        print("  {:<14} {:<9} {:<26} {:>9} {:>10} {:>7}".format(
            "TABLE", "SIDE", "COLUMN", "WITHIN", "BETWEEN", "RATIO"))
        for _, r in drifty.iterrows():
            print("  {:<14} {:<9} {:<26} {:>9.3f} {:>10.3f} {:>7.3f}".format(
                r.table_type, r.perspective, r.column[:26],
                r.within_sd, r.between_season_sd, r.drift_ratio))
        print()

    # T9 - squad/opponent redundancy. Possession is zero-sum: the opponent
    # column is 100 - the squad column and adds nothing.
    poss_sums = []
    for season in SEASON_ORDER:
        sq = tables.get((season, "Standard", "Squad"))
        op = tables.get((season, "Standard", "Opponent"))
        if sq is None or op is None:
            continue
        cs = [c for c in sq.columns if stat_leaf(c) == "Poss"]
        co = [c for c in op.columns if stat_leaf(c) == "Poss"]
        if not cs or not co:
            continue
        j = (sq[["Squad", cs[0]]].rename(columns={cs[0]: "_poss_for"})
             .merge(op[["Squad", co[0]]].rename(columns={co[0]: "_poss_against"}),
                    on="Squad"))
        poss_sums.append(
            (season, float((numeric(j["_poss_for"]) + numeric(j["_poss_against"])).mean())))
    ok_red = bool(poss_sums) and all(abs(m - 100.0) < 0.5 for _, m in poss_sums)
    record("T9", "opponent possession is 100 - squad possession", "5 seasons",
           len(poss_sums),
           "sum = {}".format(", ".join("{:.1f}".format(m) for _, m in poss_sums)),
           "100.0 - the opponent column is redundant",
           "PASS" if ok_red else "FAIL")

    # T10/T11 - arithmetically derivable columns. Keeping both sides of an
    # identity inflates the feature count without adding information.
    pts_bad_total, gd_bad_total, sanctioned = 0, 0, []
    for season in SEASON_ORDER:
        ov = tables.get((season, "Overall", "League"))
        if ov is None:
            continue
        pts, w, d = numeric(ov["Pts"]), numeric(ov["W"]), numeric(ov["D"])
        gd, gf, ga = numeric(ov["GD"]), numeric(ov["GF"]), numeric(ov["GA"])
        gd_bad_total += int((gd != gf - ga).sum())
        off = pts != 3 * w + d
        pts_bad_total += int(off.sum())
        sanctioned += [(season, s) for s in ov.loc[off, "Squad"]]
    record("T10", "GD = GF - GA holds in every Overall table", "100 team-seasons",
           100, "{} violations".format(gd_bad_total),
           "0 - GD is derivable, refuse the duplicate",
           "PASS" if gd_bad_total == 0 else "FAIL")
    record("T11", "Pts = 3W + D except where sanctioned", "100 team-seasons",
           100, "{} violations".format(pts_bad_total),
           "2 - Everton and Nottingham Forest, 2023-24",
           "PASS" if pts_bad_total == 2 else "FAIL",
           "Phase 0 Finding 3; sanctions stay explicit, never absorbed: {}".format(
               sanctioned))

    # T12 - the Home/Away table does not carry the sanction, the Overall one
    # does. Mixing them is the trap Phase 0 Finding 3 named.
    ha_conflicts = []
    for season in SEASON_ORDER:
        ha = tables.get((season, "Home/Away", "League"))
        ov = tables.get((season, "Overall", "League"))
        if ha is None or ov is None:
            continue
        hp = [c for c in ha.columns if c == "Home | Pts"]
        ap = [c for c in ha.columns if c == "Away | Pts"]
        if not hp or not ap:
            continue
        j = ha[["Squad", hp[0], ap[0]]].merge(ov[["Squad", "Pts"]], on="Squad")
        split = numeric(j[hp[0]]) + numeric(j[ap[0]])
        ha_conflicts += [(season, s) for s in j.loc[split != numeric(j["Pts"]), "Squad"]]
    record("T12", "Home Pts + Away Pts vs Overall Pts", "100 team-seasons", 100,
           "{} disagree".format(len(ha_conflicts)),
           "2 - the sanctioned clubs; never mix the two sources",
           "PASS" if len(ha_conflicts) == 2 else "FAIL", str(ha_conflicts))

    # T13 - is xG anywhere? The project plan prioritises it, so its absence is
    # a finding, not an omission.
    all_leaves = sorted(set(cat["stat"]))
    xg_like = [l for l in all_leaves
               if any(k in l.lower() for k in ("xg", "npxg", "xag", "expected", "psxg"))]
    record("T13", "expected-goals columns present anywhere", "all 60 files",
           len(all_leaves), "{} xG-like columns found".format(len(xg_like)),
           "0 - xG must be sourced externally", "INFO", str(xg_like))

    # T14 - unclassified columns. A stat with no family is a stat nobody has
    # decided about, and undecided columns must not drift into a model.
    unclassified = summary[summary.family == "unclassified"]
    record("T14", "every stat column lands in a declared family",
           "all stat columns", len(summary),
           "{} unclassified".format(len(unclassified)), "0",
           "PASS" if unclassified.empty else "FAIL",
           "; ".join(sorted(set(unclassified["stat"]))))

    # ------------------------------------------------------------------
    banner("6. FAMILY BREAKDOWN")
    # ------------------------------------------------------------------
    fam = (summary[summary.is_numeric].groupby("family")
           .agg(columns=("column", "size"),
                constant=("is_constant_within_season", "sum"),
                high_drift=("drift_ratio", lambda s: int((s > DRIFT_LIMIT).sum())))
           .sort_values("columns", ascending=False))
    fam["survives"] = fam["columns"] - fam["constant"]
    print(fam.to_string())
    print()

    # ------------------------------------------------------------------
    banner("7. CONTEXTUAL FEATURES DERIVABLE FROM FIXTURES ALONE")
    # ------------------------------------------------------------------
    state["home_rest"] = (state.date - state.home_previous_match_date).dt.days
    state["away_rest"] = (state.date - state.away_previous_match_date).dt.days
    avail = int(state.home_rest.notna().sum())
    record("T15", "rest days derivable without new data", "1,900 matches",
           len(state),
           "{} of {} available ({:.2f}%)".format(
               avail, len(state), 100 * avail / len(state)),
           "1,850 - the 50 GW1 matches have no previous match",
           "PASS" if avail == 1850 else "FAIL",
           "reuses Phase 1's previous-match date, which already obeys date < T")

    in_season = state[(state.home_rest <= 40) & (state.away_rest <= 40)]
    diff = in_season.home_rest - in_season.away_rest
    record("T16", "rest differential has usable spread", "in-season matches",
           len(in_season),
           "sd {:.2f} days, |diff| >= 2 in {:.1f}% of matches".format(
               diff.std(), 100 * (diff.abs() >= 2).mean()),
           "non-degenerate", "INFO",
           "confounded with European participation - must be tested against a "
           "team-strength control, never alone")

    # ------------------------------------------------------------------
    banner("8. WRITING REPORTS")
    # ------------------------------------------------------------------
    OUT_DIR.mkdir(exist_ok=True)
    (summary.sort_values(["family", "table_type", "perspective", "column"])
     .to_csv(CATALOGUE_PATH, index=False, float_format="%.17g"))
    coverage.to_csv(COVERAGE_PATH, index=False)
    pd.DataFrame(AUDIT).to_csv(AUDIT_PATH, index=False)
    for p in (CATALOGUE_PATH, COVERAGE_PATH, AUDIT_PATH):
        print("  {}".format(p))

    # ------------------------------------------------------------------
    banner("PHASE 3 - INSTRUMENT 1 STATUS")
    # ------------------------------------------------------------------
    failures = [r for r in AUDIT if r["verdict"] == "FAIL"]
    print("  Files typed from content     : {}".format(len(manifest)))
    print("  Stat columns catalogued      : {}".format(int(summary.is_numeric.sum())))
    print("  Constant columns refused     : {}".format(len(consts)))
    print("  High-drift columns flagged   : {}".format(len(drifty)))
    print("  Tests run                    : {}".format(len(AUDIT)))
    print("  Tests failed                 : {}".format(len(failures)))
    print()
    if failures:
        print("  FAIL / INVESTIGATE")
        for r in failures:
            print("    {} - {}".format(r["test_id"], r["test_name"]))
    else:
        print("  PASS")
        print()
        print("  The lag-1 season prior is temporally legal and its coverage is")
        print("  measured. The representation rules it needs are established by")
        print("  measurement, not assertion.")
    print()
    print("No source data was modified.")
    print("No feature dataset was built.")
    print("No model was trained.")
    print("The evaluation harness was not touched.")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
