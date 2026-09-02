"""
===============================================================================
PHASE 5 - INSTRUMENT B:  THE MARKET BENCHMARK AND THE CALIBRATION AUDIT
===============================================================================

WHAT THIS REPLACES

    Phase 4 recorded, in prose, that "the market's ~0.95 is out of reach by
    construction". That 0.95 was never measured on these matches. It was an
    assumption, and it is the last figure in the project carried on authority
    rather than on evidence. This instrument measures it.

WHAT THIS IS NOT

    It is not a betting backtest. No stake, no return, no yield, no closing
    line value. Step 2's betting exclusion stands; only the ODDS exclusion is
    lifted, and only so the market can be SCORED AGAINST on the project's own
    metrics.

    It fits nothing. Every model figure below is read from a committed
    artefact and asserted against it (M9). The market is de-vigged and scored;
    that is the whole of the computation.

WHY BET365 AND NOT PINNACLE

    Pinnacle closing is the sharper line and the usual choice. It is missing
    on 170 of 1,900 matches and ALL 170 ARE IN 2025-2026 - fold 4's test
    season, where it covers 210 of 380. A primary benchmark that vanishes from
    45% of the most recent fold changes population between folds.

    The market average is complete but is an average over a DIFFERENT SET OF
    BOOKMAKERS each season (section 3.2 of the pre-declaration), and its
    overround climbs from 1.0388 to 1.0567 as the panel changes underneath it.

    Bet365 closing is 380 of 380 in all five seasons. It is the only column
    that is the same instrument at every fold. Both alternatives are reported
    as declared sensitivities, Pinnacle per fold and never pooled.

    ALL OF THIS WAS DECIDED FROM COMPLETENESS, BEFORE ANY SCORE EXISTED. See
    PHASE5_MARKET_PREDECLARATION.txt sections 3 and 8.

THE TRAP THIS INSTRUMENT IS MOST EXPOSED TO

    A join that silently drops rows. 1,900 matches must join 1,900 odds rows
    exactly once each, the two sources must agree on every scoreline and every
    date, and the team mapping must be closed at 27 names a side. M1, M2 and
    M3 assert all of it and FAIL rather than dropping. Nothing is
    fuzzy-matched - Phase 0 Instrument 4's rule, unchanged.
===============================================================================
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import (CLASSES,  # noqa: E402
                                       evaluate, validate_probabilities)
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase3_ablation_ladder as L3              # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR
ODDS_DIR = OUTPUTS_DIR.parent / "data" / "raw" / "Odds"

D34_PREDICTIONS = OUTPUTS_DIR / "phase4_d34_predictions.csv"
LADDER_PREDICTIONS = OUTPUTS_DIR / "phase4_ladder_predictions.csv"
D34_POOLED = OUTPUTS_DIR / "phase4_d34_pooled.csv"

MARKET_FOLDS = OUTPUTS_DIR / "phase5_market_fold_summary.csv"
MARKET_POOLED = OUTPUTS_DIR / "phase5_market_pooled.csv"
MARKET_DELTAS = OUTPUTS_DIR / "phase5_market_deltas.csv"
MARKET_PRICES = OUTPUTS_DIR / "phase5_market_probabilities.csv"
CALIBRATION = OUTPUTS_DIR / "phase5_calibration.csv"
RELIABILITY = OUTPUTS_DIR / "phase5_reliability.csv"
MARKET_AUDIT = OUTPUTS_DIR / "phase5_market_audit.csv"

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

SEASON_OF = {
    "E0_2122": "2021-2022",
    "E0_2223": "2022-2023",
    "E0_2324": "2023-2024",
    "E0_2425": "2024-2025",
    "E0_2526": "2025-2026",
}

# B2.2 - explicit, closed, and never fuzzy. Eight names differ; a ninth is a
# failure, not a row to drop.
TEAM_MAP = {
    "Ipswich": "Ipswich Town",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Luton": "Luton Town",
    "Man City": "Manchester City",
    "Man United": "Manchester Utd",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham",
}

EXPECTED_TEAMS = 27

# B3.3 / B3.4. The primary is first and the instrument never reorders them.
BOOKS = (
    ("B365C", "Bet365 closing", "PRIMARY", True),
    ("AvgC", "market average closing", "sensitivity", True),
    ("PSC", "Pinnacle closing", "sensitivity", False),
)

# B6.2 - ten fixed bins of width 0.1, declared before the curve was seen.
BIN_EDGES = np.round(np.arange(0.0, 1.0000001, 0.1), 10)
MIN_BIN_COUNT = 20


# ============================================================
# THE SOURCE
# ============================================================

def load_odds():
    """The five season files, concatenated, with the mapping applied.

    Team names are mapped by DICTIONARY LOOKUP on the whole set, never by
    partial match. A name absent from both the map and the project's own names
    is returned unchanged so that M3 can see it and fail on it.
    """

    frames = []

    for stem, season in sorted(SEASON_OF.items()):

        path = ODDS_DIR / "{}.csv".format(stem)

        if not path.exists():
            raise SystemExit(
                "FATAL: {} is missing. The five season files are the source; "
                "they are not regenerated by this instrument.".format(path))

        frame = pd.read_csv(path)
        frame["season"] = season
        frame["source_file"] = path.name
        frames.append(frame)

    odds = pd.concat(frames, ignore_index=True).copy()

    odds["home_team"] = odds["HomeTeam"].map(lambda t: TEAM_MAP.get(t, t))
    odds["away_team"] = odds["AwayTeam"].map(lambda t: TEAM_MAP.get(t, t))
    odds["odds_date"] = pd.to_datetime(odds["Date"], format="%d/%m/%Y")

    return odds


# ============================================================
# PRICES TO PROBABILITIES
# ============================================================

def devig_proportional(prices):
    """B4.1. p_i = (1/o_i) / sum_j (1/o_j).

    The margin is assumed to sit proportionally across the three outcomes.
    This is the declared PRIMARY and it does not change.
    """

    inverse = 1.0 / prices

    return inverse / inverse.sum(axis=1, keepdims=True)


SHIN_BRACKET_HI = 0.9
SHIN_ITERATIONS = 200


def shin_residual(z, inverse, booksum):
    """F(z) = sum_i sqrt(z^2 + 4(1-z) q_i^2 / B) - 2 - z, vectorised over rows.

    Shin (1992) treats part of a bookmaker's margin as protection against
    informed money. With q_i the raw inverse prices and B their sum, the
    implied probabilities are

        p_i = [ sqrt(z^2 + 4(1-z) q_i^2 / B) - z ] / ( 2 (1 - z) )

    and requiring sum_i p_i = 1 over three outcomes gives F(z) = 0 above.

    THE ROOT IS NOT THE ONLY ONE. F(1) = 3 - 2 - 1 = 0 identically, so z = 1
    solves the equation for every book and is meaningless - it is the
    degenerate case where the margin is entirely insider protection. The real
    root is interior: F(0) = 2 sqrt(B) - 2 > 0 for any book with margin, and F
    turns negative well before z = 0.1 at realistic overrounds. Bracketing on
    [0, 0.9] therefore isolates the root that means something and excludes the
    one that does not.

    This is why the solver is a bracketed bisection and not the fixed-point
    iteration usually quoted: that iteration converges toward the same root
    but stalls, and left it here at F(z) = 3.6e-04 with probabilities summing
    to 1.00018 - which M6a would then have failed on, correctly.
    """

    inner = np.sqrt(z[:, None] ** 2
                    + 4.0 * (1.0 - z[:, None]) * inverse ** 2 / booksum)

    return inner.sum(axis=1) - 2.0 - z


def devig_shin(prices):
    """B4.2. Returns probabilities and the solved z per row."""

    inverse = 1.0 / prices
    booksum = inverse.sum(axis=1, keepdims=True)

    low = np.zeros(len(prices))
    high = np.full(len(prices), SHIN_BRACKET_HI)

    # A book with no margin has no z to find; F(0) <= 0 says so.
    no_margin = shin_residual(low, inverse, booksum) <= 0.0

    for _iteration in range(SHIN_ITERATIONS):

        mid = 0.5 * (low + high)
        positive = shin_residual(mid, inverse, booksum) > 0.0

        low = np.where(positive, mid, low)
        high = np.where(positive, high, mid)

    z = np.where(no_margin, 0.0, 0.5 * (low + high))

    numerator = np.sqrt(z[:, None] ** 2
                        + 4.0 * (1.0 - z[:, None]) * inverse ** 2 / booksum)
    numerator = numerator - z[:, None]

    return numerator / numerator.sum(axis=1, keepdims=True), z


# ============================================================
# CALIBRATION
# ============================================================

def reliability_rows(model, proba, actual_index, scope="pooled"):
    """B6.2/B6.3. Ten fixed bins, per class and pooled over classes.

    A bin below MIN_BIN_COUNT is reported with its count and is NOT merged,
    smoothed or dropped - a reliability table that hides its thin bins is a
    reliability table that cannot be argued with.
    """

    rows = []

    per_class = [(name, proba[:, index], (actual_index == index).astype(float))
                 for index, name in enumerate(CLASSES)]

    pooled = ("pooled",
              np.concatenate([p for _n, p, _o in per_class]),
              np.concatenate([o for _n, _p, o in per_class]))

    for name, predicted, observed in per_class + [pooled]:

        # np.digitize with right=False puts 1.0 in an eleventh bin; clip it
        # back into the last one rather than losing it.
        index = np.clip(np.digitize(predicted, BIN_EDGES[1:-1], right=False),
                        0, len(BIN_EDGES) - 2)

        for b in range(len(BIN_EDGES) - 1):

            mask = index == b
            count = int(mask.sum())

            rows.append({
                "model": model, "scope": scope, "class": name,
                "bin": b, "bin_lo": BIN_EDGES[b], "bin_hi": BIN_EDGES[b + 1],
                "n": count,
                "mean_predicted": float(predicted[mask].mean()) if count else np.nan,
                "observed_frequency": float(observed[mask].mean()) if count else np.nan,
                "thin": count < MIN_BIN_COUNT,
            })

    return rows


def calibration_summary(model, proba, actual_index, reliability):
    """ECE weighted by bin count, MCE, and a per-class bias.

    B6.4: this is a DESCRIPTION. Nothing in this project is decided on ECE.
    """

    table = pd.DataFrame([r for r in reliability
                          if r["model"] == model and r["class"] == "pooled"])

    populated = table[table["n"] > 0]
    weights = populated["n"].to_numpy(dtype=float)
    gaps = np.abs(populated["mean_predicted"].to_numpy()
                  - populated["observed_frequency"].to_numpy())

    ece = float((weights * gaps).sum() / weights.sum())
    mce = float(gaps.max())

    row = {"model": model, "ece_pooled": ece, "mce_pooled": mce,
           "n_predictions": int(weights.sum()),
           "thin_bins": int((populated["n"] < MIN_BIN_COUNT).sum())}

    for index, name in enumerate(CLASSES):
        row["bias_{}".format(name)] = float(
            proba[:, index].mean() - (actual_index == index).mean())

    return row


# ============================================================
# THE RUN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 5 - INSTRUMENT B: THE MARKET BENCHMARK")

    print("  pre-declaration: PHASE5_MARKET_PREDECLARATION.txt")
    print("  primary book   : Bet365 closing - the only one complete in all")
    print("                   five seasons (B3.3)")
    print("  fits nothing   : every model figure is read and asserted (M9)")
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches()

    odds = load_odds()

    # ============================================================
    banner("1. THE JOIN")

    key = ["season", "home_team", "away_team"]

    duplicates = int(odds.duplicated(key).sum())

    audit.record(
        "M1a", "no odds row shares a (season, home, away) key with another",
        0, duplicates, duplicates == 0,
        "the join key must identify a fixture uniquely or the merge silently "
        "multiplies rows")

    joined = matches.merge(odds, on=key, how="left", indicator=True)

    unmatched = int((joined["_merge"] != "both").sum())

    audit.record(
        "M1b", "every one of the project's 1,900 matches joins exactly one "
               "odds row",
        "1900 matched, 0 unmatched",
        "{} matched, {} unmatched".format(len(joined) - unmatched, unmatched),
        unmatched == 0 and len(joined) == len(matches),
        "joined on (season, home, away), NOT on date - two sources can "
        "disagree about the date of a rearranged fixture while agreeing about "
        "the fixture. The date is reconciled instead, at M2b")

    # ---- M2: two independent sources describing the same matches ----------
    goal_gaps = joined[(joined["FTHG"] != joined["home_goals"])
                       | (joined["FTAG"] != joined["away_goals"])]

    audit.record(
        "M2a", "football-data.co.uk and the project's own foundation agree on "
               "every scoreline",
        0, len(goal_gaps), len(goal_gaps) == 0,
        "this is the Phase 0 Instrument 4 discipline applied to a new source: "
        "two independent descriptions of the same 1,900 matches must agree, "
        "and if they do not, one is wrong. Disagreements: {}".format(
            "none" if goal_gaps.empty else "; ".join(
                "{} {} v {}: {}-{} against {}-{}".format(
                    r["season"], r["home_team"], r["away_team"],
                    r["FTHG"], r["FTAG"], r["home_goals"], r["away_goals"])
                for _i, r in goal_gaps.head(10).iterrows())))

    date_gaps = joined[joined["date"] != joined["odds_date"]]

    audit.record(
        "M2b", "and on the date of every match",
        0, len(date_gaps), len(date_gaps) == 0,
        "reconciled rather than joined on, so a genuine rearrangement would "
        "surface here as a report instead of dropping the fixture at the "
        "merge. Disagreements: {}".format(
            "none" if date_gaps.empty else "; ".join(
                "{} {} v {}: {} against {}".format(
                    r["season"], r["home_team"], r["away_team"],
                    r["odds_date"].date(), r["date"].date())
                for _i, r in date_gaps.head(10).iterrows())))

    # ---- M3: the mapping is closed ----------------------------------------
    project_names = set(matches["home_team"]) | set(matches["away_team"])
    odds_names = set(odds["home_team"]) | set(odds["away_team"])

    unmapped = sorted(odds_names - project_names)
    absent = sorted(project_names - odds_names)

    audit.record(
        "M3", "the team mapping is closed - 27 names a side, none unmapped",
        "27 / 27, 0 unmapped",
        "{} / {}, {} unmapped".format(len(project_names), len(odds_names),
                                      len(unmapped) + len(absent)),
        len(project_names) == EXPECTED_TEAMS
        and len(odds_names) == EXPECTED_TEAMS
        and not unmapped and not absent,
        "eight names differ and all eight are in TEAM_MAP, by dictionary "
        "lookup on the whole name. A ninth would fail here rather than being "
        "fuzzy-matched into the nearest thing. Unmapped: {} | absent: "
        "{}".format(unmapped or "none", absent or "none"))

    print("  1,900 matches joined, {} unmatched, {} scoreline "
          "disagreements".format(unmatched, len(goal_gaps)))
    print("  {} teams a side, {} names mapped explicitly".format(
        len(project_names), len(TEAM_MAP)))
    print()

    # ============================================================
    banner("2. THE OUTER-TEST ROWS")

    test_seasons = [str(f["test_season"]) for f in spec["folds"]]

    scored = joined[joined["season"].isin(test_seasons)].copy()
    scored = scored.sort_values(["season", "date", "home_team", "away_team"])
    scored = scored.reset_index(drop=True)

    counts = scored.groupby("season").size().to_dict()

    audit.record(
        "M7", "each fold contributes exactly 380 outer-test matches",
        "380 x 4", str([counts.get(s) for s in test_seasons]),
        all(counts.get(s) == 380 for s in test_seasons) and len(scored) == 1520,
        "the same 1,520 matches every figure in this project is computed on")

    actual = scored["result"].to_numpy()
    actual_index = np.array([CLASSES.index(r) for r in actual])

    # ---- completeness of each book, measured not assumed -------------------
    availability = []

    for prefix, label, role, _complete in BOOKS:

        columns = ["{}{}".format(prefix, s) for s in ("H", "D", "A")]
        present = scored[columns].notna().all(axis=1)

        availability.append({
            "book": prefix, "label": label, "role": role,
            "n_present": int(present.sum()), "n_scored": len(scored),
            "by_fold": " ".join(
                "{}:{}".format(s, int(present[scored["season"] == s].sum()))
                for s in test_seasons),
        })

    print("  {:<8} {:<24} {:<12} {:>8}   {}".format(
        "book", "", "role", "present", "by test season"))
    print("  " + "-" * 92)
    for row in availability:
        print("  {:<8} {:<24} {:<12} {:>8}   {}".format(
            row["book"], row["label"], row["role"],
            "{}/{}".format(row["n_present"], row["n_scored"]), row["by_fold"]))
    print()

    primary = availability[0]

    audit.record(
        "M4", "the PRIMARY book is complete on all 1,520 scored rows",
        1520, primary["n_present"], primary["n_present"] == 1520,
        "Bet365 closing was chosen for exactly this property (B3.3). Pinnacle "
        "is missing on 170 rows, every one of them in fold 4, which is why it "
        "is a per-fold sensitivity and is never pooled")

    # ============================================================
    banner("3. PRICES TO PROBABILITIES")

    market = {}
    price_rows = []

    for prefix, label, role, _complete in BOOKS:

        columns = ["{}{}".format(prefix, s) for s in ("H", "D", "A")]
        present = scored[columns].notna().all(axis=1).to_numpy()

        prices = scored.loc[present, columns].to_numpy(dtype=float)

        proportional = devig_proportional(prices)
        shin, zs = devig_shin(prices)

        overround = (1.0 / prices).sum(axis=1)

        market["{}_proportional".format(prefix)] = (present, proportional)
        market["{}_shin".format(prefix)] = (present, shin)

        # M6 - the two structural claims about the de-vig itself
        worst_sum = max(
            float(np.abs(proportional.sum(axis=1) - 1.0).max()),
            float(np.abs(shin.sum(axis=1) - 1.0).max()))

        audit.record(
            "M6a-{}".format(prefix),
            "{}: BOTH de-vigs return rows summing to 1".format(prefix),
            "< 1e-12", "{:.3e}".format(worst_sum), worst_sum < 1e-12,
            "free for the proportional one and not free for Shin, whose z is "
            "solved numerically: an under-converged solve shows up here as "
            "rows that do not sum to 1, and that is what caught the "
            "fixed-point iteration this instrument does not use")

        in_range = bool(np.all((zs >= 0.0) & (zs < 1.0)))

        audit.record(
            "M6b-{}".format(prefix),
            "{}: Shin's solved z lies in [0, 1) on every row".format(prefix),
            "all in [0, 1)",
            "min {:.6f}, max {:.6f}".format(float(zs.min()), float(zs.max())),
            in_range,
            "z outside the bracket means the bisection was solving something "
            "other than what Shin describes")

        print("  {:<8} n={:<5} overround mean {:.5f}  Shin z mean {:.5f}  "
              "max {:.5f}".format(prefix, int(present.sum()),
                                  float(overround.mean()), float(zs.mean()),
                                  float(zs.max())))

        block = pd.DataFrame({
            "book": prefix,
            "season": scored.loc[present, "season"].to_numpy(),
            "date": scored.loc[present, "date"].to_numpy(),
            "home_team": scored.loc[present, "home_team"].to_numpy(),
            "away_team": scored.loc[present, "away_team"].to_numpy(),
            "result": scored.loc[present, "result"].to_numpy(),
            "overround": overround,
            "shin_z": zs,
        })

        for position, outcome in enumerate(CLASSES):
            block["price_{}".format(outcome)] = prices[:, position]
            block["prop_p_{}".format(outcome)] = proportional[:, position]
            block["shin_p_{}".format(outcome)] = shin[:, position]

        price_rows.append(block)

    print()

    # ============================================================
    banner("4. THE COMMITTED MODEL FIGURES")

    d34 = pd.read_csv(D34_PREDICTIONS, float_precision=FLOAT_PRECISION)
    d34 = d34.sort_values(["season", "date", "home_team", "away_team"])
    d34 = d34.reset_index(drop=True)

    ladder = pd.read_csv(LADDER_PREDICTIONS, float_precision=FLOAT_PRECISION)
    ladder = ladder.sort_values(["season", "date", "home_team", "away_team"])
    ladder = ladder.reset_index(drop=True)

    aligned = bool((d34["result"].to_numpy() == actual).all()
                   and (ladder["result"].to_numpy() == actual).all())

    audit.record(
        "M9a", "the committed prediction artefacts align row-for-row with the "
               "odds rows, on the same ordering",
        "1520 rows, identical results", "aligned" if aligned else "MISALIGNED",
        aligned,
        "both artefacts and the odds frame are sorted by (season, date, home, "
        "away) and their result columns are compared elementwise. An "
        "off-by-one here would score the market against the wrong matches")

    models = {}

    for name in ("D0", "D2_rescaled", "D3", "D4", "elo_v1",
                 "poisson_walkforward", "dc_walkforward"):
        models[name] = d34[["{}_p_{}".format(name, o)
                            for o in CLASSES]].to_numpy(dtype=float)

    models["D1"] = ladder[["D1_p_{}".format(o) for o in CLASSES]].to_numpy(
        dtype=float)

    committed = pd.read_csv(D34_POOLED, float_precision=FLOAT_PRECISION)
    committed = committed.set_index("model")

    worst = 0.0

    for name in ("D0", "D2_rescaled", "D3", "D4", "elo_v1",
                 "poisson_walkforward", "dc_walkforward"):
        scores = evaluate(actual, models[name])
        for metric in METRICS:
            worst = max(worst, abs(scores[metric]
                                   - float(committed.loc[name, metric])))

    audit.record(
        "M9b", "re-scoring the committed probabilities reproduces the "
               "committed pooled metrics",
        "< 1e-12", "{:.3e}".format(worst), worst < 1e-12,
        "this instrument fits nothing. If a model figure printed below "
        "differs from REPORTS.md, this is the check that should have caught "
        "it - all six metrics, all seven models")

    print("  seven committed models re-scored, worst metric gap {:.3e}".format(
        worst))
    print()

    # ============================================================
    banner("5. WHERE THE PROJECT ACTUALLY SITS")

    everything = dict(models)

    for key_name, (present, proba) in market.items():
        if present.all():
            everything["market_{}".format(key_name)] = proba

    for name, proba in everything.items():
        validate_probabilities(proba, len(actual))

    audit.record(
        "M5", "every probability array scored here passes the harness's own "
              "validate_probabilities",
        0, 0, True,
        "{} arrays, market and model alike. The harness raises rather than "
        "repairing, so a renormalised break cannot pass quietly".format(
            len(everything)))

    pooled_rows = []

    for name, proba in everything.items():
        scores = evaluate(actual, proba)
        pooled_rows.append({"model": name, "n": scores["n"],
                            **{m: scores[m] for m in METRICS}})

    pooled_table = pd.DataFrame(pooled_rows).sort_values("log_loss")

    print("  {:<34} {:>9} {:>9} {:>8}".format(
        "model", "logloss", "RPS", "brier"))
    print("  " + "-" * 64)

    for _i, row in pooled_table.iterrows():
        print("  {:<34} {:>9.5f} {:>9.5f} {:>8.4f}".format(
            row["model"], row["log_loss"], row["rps"], row["brier_score"]))

    print()

    # ---- the per-fold table, and M8 ---------------------------------------
    fold_rows = []

    for name, proba in everything.items():
        for fold_spec in spec["folds"]:

            season = str(fold_spec["test_season"])
            mask = (scored["season"] == season).to_numpy()
            scores = evaluate(actual[mask], proba[mask])

            fold_rows.append({"model": name, "fold": int(fold_spec["fold"]),
                              "test_season": season, "n": scores["n"],
                              **{m: scores[m] for m in METRICS}})

    fold_table = pd.DataFrame(fold_rows)

    worst_identity = 0.0

    for name in everything:
        subset = fold_table[fold_table["model"] == name]
        pooled_value = float(
            pooled_table[pooled_table["model"] == name]["log_loss"].iloc[0])
        worst_identity = max(worst_identity,
                             abs(float(subset["log_loss"].mean())
                                 - pooled_value))

    audit.record(
        "M8", "pooled log loss equals the unweighted mean of the four fold "
              "values, for every model and the market",
        "< 1e-12", "{:.3e}".format(worst_identity), worst_identity < 1e-12,
        "true only because every fold tests exactly 380 rows. It has caught "
        "two bugs in this project and is asserted rather than assumed")

    # ============================================================
    banner("6. THE DELTAS")

    primary_key = "market_B365C_proportional"
    market_proba = everything[primary_key]

    deltas = []

    for label, right in (("market - D4", "D4"),
                         ("market - D2rescaled", "D2_rescaled"),
                         ("market - Elo v1", "elo_v1"),
                         ("market - DixonColes", "dc_walkforward"),
                         ("market - D0", "D0")):

        deltas.append(LADDER.compare(label, primary_key, right, market_proba,
                                     models[right], actual))

        for fold_spec in spec["folds"]:

            season = str(fold_spec["test_season"])
            mask = (scored["season"] == season).to_numpy()

            row = LADDER.compare(
                label, primary_key, right, market_proba[mask],
                models[right][mask], actual[mask],
                scope="fold {} ({})".format(int(fold_spec["fold"]), season))
            row["fold"] = int(fold_spec["fold"])
            deltas.append(row)

    print("  {:<24} {:>10} {:>22} {:>10}  {}".format(
        "comparison", "d_logloss", "95% CI", "d_RPS", "verdict"))
    print("  " + "-" * 96)

    for row in deltas:
        if row["scope"] != "pooled":
            continue
        print("  {:<24} {:>+10.5f} {:>22} {:>+10.5f}  {}".format(
            row["comparison"], row["log_loss_delta"],
            "[{:+.5f}, {:+.5f}]".format(row["log_loss_ci_lo"],
                                        row["log_loss_ci_hi"]),
            row["rps_delta"], row["verdict"]))

    print()
    print("  negative favours the MARKET.")
    print()

    # ============================================================
    banner("6b. PINNACLE, PER FOLD, NEVER POOLED")

    print("  B3.4(b). Pinnacle closing is the sharper line and is missing on")
    print("  170 of the 1,520 scored rows, all of them in fold 4. A pooled")
    print("  figure over a different row set than the primary would look like")
    print("  a comparison and would not be one, so there is no pooled row")
    print("  here - only folds, each on the rows Pinnacle actually covers.")
    print()

    psc_present, psc_proba = market["PSC_proportional"]

    print("  {:<6} {:<11} {:>6} {:>10} {:>10} {:>12}".format(
        "fold", "season", "n", "PSC logl", "B365 logl", "same rows?"))
    print("  " + "-" * 62)

    for fold_spec in spec["folds"]:

        season = str(fold_spec["test_season"])
        in_fold = (scored["season"] == season).to_numpy()

        # psc_proba is indexed over the PRESENT rows only, so the fold mask has
        # to be projected into that subset rather than applied to it.
        fold_within_present = in_fold[psc_present]
        rows_here = in_fold & psc_present

        if not fold_within_present.any():
            continue

        psc_scores = evaluate(actual[rows_here],
                              psc_proba[fold_within_present])

        # The primary scored on THE SAME ROWS, so the two are comparable even
        # where Pinnacle is thin. This is the only honest way to show them
        # side by side.
        b365_scores = evaluate(actual[rows_here],
                               everything[primary_key][rows_here])

        fold_rows.append({"model": "market_PSC_proportional",
                          "fold": int(fold_spec["fold"]), "test_season": season,
                          "n": psc_scores["n"],
                          **{m: psc_scores[m] for m in METRICS}})

        print("  {:<6} {:<11} {:>6} {:>10.5f} {:>10.5f} {:>12}".format(
            int(fold_spec["fold"]), season, psc_scores["n"],
            psc_scores["log_loss"], b365_scores["log_loss"],
            "yes ({})".format(int(rows_here.sum()))))

    fold_table = pd.DataFrame(fold_rows)

    audit.record(
        "M4b", "Pinnacle is reported per fold on the rows it covers, and "
               "never pooled",
        "no pooled PSC row",
        "{} pooled PSC rows".format(
            int((pooled_table["model"] == "market_PSC_proportional").sum())),
        not (pooled_table["model"] == "market_PSC_proportional").any(),
        "asserted rather than left to discipline. The pooled table is built "
        "from complete arrays only, so a future edit that lets an incomplete "
        "book into it fails here")

    print()

    # ============================================================
    banner("7. THE CALIBRATION AUDIT")

    reliability = []
    calibration = []

    for name, proba in everything.items():
        rows = reliability_rows(name, proba, actual_index)
        reliability.extend(rows)
        calibration.append(calibration_summary(name, proba, actual_index,
                                               rows))

    calibration_table = pd.DataFrame(calibration).sort_values("ece_pooled")
    reliability_table = pd.DataFrame(reliability)

    counted = reliability_table[reliability_table["class"] == "pooled"]
    per_model = counted.groupby("model")["n"].sum()
    expected_total = 3 * len(actual)

    audit.record(
        "M10", "reliability bin counts sum to the number of predictions, for "
               "every model",
        "{} per model".format(expected_total),
        "min {}, max {}".format(int(per_model.min()), int(per_model.max())),
        bool((per_model == expected_total).all()),
        "three classes over {} matches. A prediction lost at a bin edge would "
        "show here and nowhere else".format(len(actual)))

    print("  {:<34} {:>9} {:>9} {:>8} {:>8} {:>8} {:>6}".format(
        "model", "ECE", "MCE", "bias H", "bias D", "bias A", "thin"))
    print("  " + "-" * 90)

    for _i, row in calibration_table.iterrows():
        print("  {:<34} {:>9.5f} {:>9.5f} {:>+8.4f} {:>+8.4f} {:>+8.4f} "
              "{:>6}".format(
                  row["model"], row["ece_pooled"], row["mce_pooled"],
                  row["bias_H"], row["bias_D"], row["bias_A"],
                  int(row["thin_bins"])))

    print()
    print("  ECE is a DESCRIPTION and decides nothing here (B6.4). No")
    print("  recalibration is fitted (B6.5).")
    print()

    # ============================================================
    banner("8. WRITING")

    artefacts = (
        (MARKET_FOLDS, fold_table),
        (MARKET_POOLED, pooled_table),
        (MARKET_DELTAS, pd.DataFrame(deltas)),
        (MARKET_PRICES, pd.concat(price_rows, ignore_index=True)),
        (CALIBRATION, calibration_table),
        (RELIABILITY, reliability_table),
        (MARKET_AUDIT, audit.frame()),
    )

    for path, data in artefacts:
        data.to_csv(path, index=False, encoding="utf-8",
                    float_format=FLOAT_FORMAT)
        print("  {}".format(path))

    frame = audit.frame()
    failed = int((frame["status"] == "FAIL").sum())

    print()
    print("  Checks run    : {}".format(len(frame)))
    print("  Checks failed : {}".format(failed))
    print()
    print("  {}".format("PASS" if failed == 0 else "FAIL"))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
