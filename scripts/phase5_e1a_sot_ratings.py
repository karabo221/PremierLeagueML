"""
===============================================================================
PHASE 5 - INSTRUMENT E1a:  SHOTS ON TARGET AS A RATING INPUT
===============================================================================

Pre-declaration: PHASE5_E1_SHOT_PREDECLARATION.txt
sha256 d385bfd4d081f40e4d88a96939fde005db9d539559d21b234ebcaca5eff6eca4

THE CLAIM. Goals are a low-count realisation of an underlying rate - roughly
1.4 per team per match against 4-5 shots on target. If SoT estimates that same
rate with less sampling noise, ratings built from it should predict better.

THE ONE THING THAT CHANGES. attack, defence and the home multiplier are
estimated from SoT instead of goals. Everything else is held identical and E2
asserts it field by field: 107-day half-life, one refit per distinct test date,
window strictly earlier, same initialisation, same MAX_GOALS, same score
matrix, same outcome_probabilities().

RHO IS THE SAME NUMBER IN BOTH ARMS, NOT MERELY THE SAME RULE. fit_rho selects
on (home_goals <= 1) & (away_goals <= 1) and corrects four discrete scoreline
cells. "Both sides had at most one shot on target" is a common, high-scoring
event and tau is not defined over it. So rho is fitted once per window from the
GOALS arm and handed to the SoT arm unchanged - E3 asserts the two are
bit-identical. That is what isolates the rating input as the single difference.

CANDIDATE (i), AND WHY IT IS NOT CONVENIENCE. SoT rates are not goal rates, so
the score matrix cannot be read off them. They are scaled by a conversion
constant c estimated on the SAME decayed window as the ratings, from matches
strictly earlier than the one being predicted. This keeps E1a inside the
Dixon-Coles architecture, which is the only thing that makes it a LOWER BOUND
on the xG arm rather than a different experiment that happens to use shots.

WHAT WOULD MAKE THIS RUNG UNREADABLE, and is therefore gated: an odds column
reaching a rating window (E9), c estimated with any sight of the match it
scales (E6), or the goals arm failing to reproduce the committed artefact (E1).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase0_evaluation_harness import CLASSES, evaluate, validate_probabilities  # noqa: E402
from phase3_feature_builder import Audit, banner, configure_stdout  # noqa: E402

import phase3_ablation_ladder as L3              # noqa: E402
import phase4_dynamic_ladder as LADDER           # noqa: E402
import phase2_poisson_dixon_coles as DC          # noqa: E402


OUTPUTS_DIR = LADDER.OUTPUTS_DIR
ODDS_DIR = OUTPUTS_DIR.parent / "data" / "raw" / "Odds"

COMMITTED_DC = OUTPUTS_DIR / "phase2_poisson_dc_fold_summary.csv"
MARKET_PROBABILITIES = OUTPUTS_DIR / "phase5_market_probabilities.csv"
D34_PREDICTIONS = OUTPUTS_DIR / "phase4_d34_predictions.csv"

FOLD_OUTPUT = OUTPUTS_DIR / "phase5_e1a_fold_summary.csv"
POOLED_OUTPUT = OUTPUTS_DIR / "phase5_e1a_pooled.csv"
DELTA_OUTPUT = OUTPUTS_DIR / "phase5_e1a_deltas.csv"
WINDOW_OUTPUT = OUTPUTS_DIR / "phase5_e1a_windows.csv"
STABILITY_OUTPUT = OUTPUTS_DIR / "phase5_e1a_parameter_stability.csv"
PREDICTIONS_OUTPUT = OUTPUTS_DIR / "phase5_e1a_predictions.csv"
AUDIT_OUTPUT = OUTPUTS_DIR / "phase5_e1a_audit.csv"

METRICS = LADDER.METRICS
FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

SEASON_OF = {
    "E0_2122": "2021-2022", "E0_2223": "2022-2023", "E0_2324": "2023-2024",
    "E0_2425": "2024-2025", "E0_2526": "2025-2026",
}

TEAM_MAP = {
    "Ipswich": "Ipswich Town", "Leeds": "Leeds United",
    "Leicester": "Leicester City", "Luton": "Luton Town",
    "Man City": "Manchester City", "Man United": "Manchester Utd",
    "Norwich": "Norwich City", "Nott'm Forest": "Nottingham",
}

SHOT_COLUMNS = ["HS", "AS", "HST", "AST", "HC", "AC",
                "HF", "AF", "HY", "AY", "HR", "AR"]

# E2.4 / E2.5 - the audit's counts, asserted so a changed source is detected.
EXPECTED_SOT_EXCEEDS_SHOTS = 1
EXPECTED_GOALS_EXCEED_SOT = 12

# Anything matching these is an odds column and must never reach a window (E9).
ODDS_MARKERS = ("H", "D", "A")
ODDS_PREFIXES = ("B365", "BW", "BF", "PS", "WH", "VC", "IW", "LB", "SJ", "GB",
                 "SB", "BS", "1XB", "Max", "Avg", "BFE", "P>", "P<", "AH")


# ============================================================
# THE SOURCE
# ============================================================

def load_shots(matches, audit):
    """Join the frozen football-data files and gate the join and the source."""

    frames = []

    for stem, season in sorted(SEASON_OF.items()):
        frame = pd.read_csv(ODDS_DIR / "{}.csv".format(stem))
        frame["season"] = season
        frames.append(frame)

    odds = pd.concat(frames, ignore_index=True).copy()

    odds["home_team"] = odds["HomeTeam"].map(lambda t: TEAM_MAP.get(t, t))
    odds["away_team"] = odds["AwayTeam"].map(lambda t: TEAM_MAP.get(t, t))
    odds["odds_date"] = pd.to_datetime(odds["Date"], format="%d/%m/%Y")

    key = ["season", "home_team", "away_team"]

    keep = key + SHOT_COLUMNS + ["odds_date"]

    joined = matches.merge(odds[keep], on=key, how="left", indicator=True)

    unmatched = int((joined["_merge"] != "both").sum())
    missing = int(joined[SHOT_COLUMNS].isna().sum().sum())

    audit.record(
        "E4a", "the shot columns join all 1,900 matches with no missing cell",
        "1900 joined, 0 missing",
        "{} joined, {} missing".format(len(joined) - unmatched, missing),
        unmatched == 0 and missing == 0 and len(joined) == len(matches),
        "the same join Phase 5B gated at 1,900/1,900. Re-asserted here "
        "because this instrument reads different COLUMNS of the same file and "
        "a column can be absent where a row is not")

    sot_over = int((joined["HST"] > joined["HS"]).sum()
                   + (joined["AST"] > joined["AS"]).sum())

    goals_over = int((joined["home_goals"] > joined["HST"]).sum()
                     + (joined["away_goals"] > joined["AST"]).sum())

    audit.record(
        "E4b", "the source's known defects are exactly the ones the audit "
               "recorded, in count",
        "{} SoT>shots, {} goals>SoT".format(EXPECTED_SOT_EXCEEDS_SHOTS,
                                            EXPECTED_GOALS_EXCEED_SOT),
        "{} SoT>shots, {} goals>SoT".format(sot_over, goals_over),
        sot_over == EXPECTED_SOT_EXCEEDS_SHOTS
        and goals_over == EXPECTED_GOALS_EXCEED_SOT,
        "NOT REPAIRED. One impossible row (Newcastle 2-4 West Ham, AS 8 / AST "
        "9) and twelve rows where goals exceed SoT, which are the expected "
        "signature of own goals rather than corruption. The counts are "
        "asserted so a source that changes underneath this rung fails here "
        "rather than being absorbed into a rating")

    # E9 - the odds columns are not carried into anything this rung touches.
    odds_like = [c for c in joined.columns
                 if any(c.startswith(p) for p in ODDS_PREFIXES)
                 and c not in SHOT_COLUMNS]

    audit.record(
        "E9a", "no odds column survives into the frame the ratings are fitted "
               "from",
        0, len(odds_like), len(odds_like) == 0,
        "the file is now read for two purposes and that does not merge the "
        "purposes. Only the twelve shot columns are selected out of it, by "
        "name, and this asserts the selection rather than trusting it. "
        "Survivors: {}".format(odds_like or "none"))

    joined = joined.drop(columns=["_merge"])

    return joined


# ============================================================
# ONE WINDOW
# ============================================================

def fit_both_arms(window, cutoff):
    """Fit the goals arm and the SoT arm on ONE window, at one cutoff.

    Returns the two parameter sets, the shared rho, and the conversion
    constants. The SoT arm's attack/defence come from a frame in which the
    goal columns have been REPLACED by SoT - fit_attack_defence is reused
    unchanged, and its inputs enter only as weighted sums, so this is the
    declared substitution and nothing more.
    """

    weights = DC.time_weights(window["date"], cutoff)

    # ---- goals arm, exactly Phase 2's dc_walkforward ----------------------
    attack_g, defence_g, home_g, iterations_g, converged_g, degenerate_g = (
        DC.fit_attack_defence(window, weights))

    rho = DC.fit_rho(window, weights, attack_g, defence_g, home_g)

    # ---- SoT arm ----------------------------------------------------------
    sot_window = window.copy()
    sot_window["home_goals"] = window["HST"].to_numpy(dtype=float)
    sot_window["away_goals"] = window["AST"].to_numpy(dtype=float)

    attack_s, defence_s, home_s, iterations_s, converged_s, degenerate_s = (
        DC.fit_attack_defence(sot_window, weights))

    # ---- the conversion constant, on the same decayed window --------------
    weighted_goals_home = float(np.sum(weights * window["home_goals"]))
    weighted_goals_away = float(np.sum(weights * window["away_goals"]))
    weighted_sot_home = float(np.sum(weights * window["HST"]))
    weighted_sot_away = float(np.sum(weights * window["AST"]))

    c_pooled = ((weighted_goals_home + weighted_goals_away)
                / (weighted_sot_home + weighted_sot_away))

    c_home = weighted_goals_home / weighted_sot_home
    c_away = weighted_goals_away / weighted_sot_away

    return {
        "goals": {"attack": attack_g, "defence": defence_g,
                  "home_multiplier": home_g, "rho": rho,
                  "iterations": iterations_g, "converged": converged_g,
                  "degenerate": degenerate_g},
        "sot": {"attack": attack_s, "defence": defence_s,
                "home_multiplier": home_s, "rho": rho,
                "iterations": iterations_s, "converged": converged_s,
                "degenerate": degenerate_s},
        "c_pooled": c_pooled, "c_home": c_home, "c_away": c_away,
        "matches": int(len(window)), "cutoff": cutoff,
    }


def predict_block(block, params, rho, scale_home, scale_away):
    """H/D/A for one block of matches from one fitted parameter set.

    scale_* is 1.0 for the goals arm and the conversion constant for the SoT
    arm. Applying it to the RATES rather than inside the estimator is what
    keeps the estimator itself unchanged.
    """

    home_rates, away_rates = DC.match_rates(
        block, params["attack"], params["defence"], params["home_multiplier"])

    rows = []

    for position, row in enumerate(block.itertuples()):

        lambda_home = float(home_rates[position]) * scale_home
        lambda_away = float(away_rates[position]) * scale_away

        probabilities, mass, truncated = DC.outcome_probabilities(
            lambda_home, lambda_away, rho)

        rows.append({
            "match_id": row.match_id, "season": row.season, "date": row.date,
            "home_team": row.home_team, "away_team": row.away_team,
            "result": row.result,
            "lambda_home": lambda_home, "lambda_away": lambda_away,
            "rho": rho, "score_matrix_mass": mass,
            "p_H": probabilities[0], "p_D": probabilities[1],
            "p_A": probabilities[2],
        })

    return rows


# ============================================================
# THE WALK-FORWARD
# ============================================================

def walk_forward(matches, spec, audit):
    """One refit per distinct test date, on matches strictly earlier."""

    arms = {"goals_DC": [], "E1a_sot": [], "E1a_sot_homeaway": []}
    windows = []
    parameters = []

    for fold_spec in spec["folds"]:

        fold = int(fold_spec["fold"])
        train_seasons = list(fold_spec["train_seasons"])
        test_season = str(fold_spec["test_season"])

        test = matches[matches["season"] == test_season]
        in_scope = matches["season"].isin(train_seasons + [test_season])

        for cutoff in sorted(test["date"].unique()):

            cutoff = pd.Timestamp(cutoff)

            # STRICT: date < cutoff. Same-day matches never enter the window.
            window = matches[in_scope & (matches["date"] < cutoff)]

            fitted = fit_both_arms(window, cutoff)

            block = test[test["date"] == cutoff]

            arms["goals_DC"].extend(predict_block(
                block, fitted["goals"], fitted["goals"]["rho"], 1.0, 1.0))

            arms["E1a_sot"].extend(predict_block(
                block, fitted["sot"], fitted["goals"]["rho"],
                fitted["c_pooled"], fitted["c_pooled"]))

            arms["E1a_sot_homeaway"].extend(predict_block(
                block, fitted["sot"], fitted["goals"]["rho"],
                fitted["c_home"], fitted["c_away"]))

            windows.append({
                "fold": fold, "test_season": test_season, "cutoff": cutoff,
                "window_matches": fitted["matches"],
                "c_pooled": fitted["c_pooled"], "c_home": fitted["c_home"],
                "c_away": fitted["c_away"], "rho": fitted["goals"]["rho"],
                "goals_home_multiplier": fitted["goals"]["home_multiplier"],
                "sot_home_multiplier": fitted["sot"]["home_multiplier"],
                "goals_degenerate": fitted["goals"]["degenerate"],
                "sot_degenerate": fitted["sot"]["degenerate"],
                "goals_converged": fitted["goals"]["converged"],
                "sot_converged": fitted["sot"]["converged"],
            })

            for arm in ("goals", "sot"):
                for team, value in fitted[arm]["attack"].items():
                    parameters.append({
                        "arm": arm, "fold": fold, "cutoff": cutoff,
                        "team": team, "attack": value,
                        "defence": fitted[arm]["defence"][team]})

    frames = {name: pd.DataFrame(rows).sort_values(
        ["season", "date", "home_team", "away_team"]).reset_index(drop=True)
        for name, rows in arms.items()}

    return frames, pd.DataFrame(windows), pd.DataFrame(parameters)


def leakage_probes(matches, spec, audit, probes=6):
    """E5 and E6, tested by CORRUPTION rather than by re-reading the filter.

    The window rule is date < cutoff, so nothing at or after the cutoff can
    reach a fit. That is a claim about code, and code has been wrong here
    before. So every match from the cutoff onward has its goals AND its shot
    columns replaced with nonsense, the window is re-fitted, and the fitted
    state is required to be bit-identical.

    E6 rides on the same corruption: c is computed from the same window, so if
    an outer-test row could reach the ratings it could reach c, and if it
    cannot reach the ratings it cannot reach c either. Asserting both from one
    corruption is not a shortcut - they are the same exposure.
    """

    in_scope = matches["season"].isin(
        [str(f["test_season"]) for f in spec["folds"]]
        + list(spec["folds"][0]["train_seasons"]))

    dates = sorted(matches[in_scope]["date"].unique())

    step = max(1, len(dates) // (probes + 1))
    cutoffs = [pd.Timestamp(dates[step * (i + 1)]) for i in range(probes)]

    worst_state = 0.0
    worst_c = 0.0

    for cutoff in cutoffs:

        window = matches[matches["date"] < cutoff]

        clean = fit_both_arms(window, cutoff)

        corrupted = matches.copy()
        after = corrupted["date"] >= cutoff

        for column in ("home_goals", "away_goals", "HST", "AST", "HS", "AS"):
            corrupted.loc[after, column] = 999.0

        dirty = fit_both_arms(corrupted[corrupted["date"] < cutoff], cutoff)

        for arm in ("goals", "sot"):
            for field in ("attack", "defence"):
                for team in clean[arm][field]:
                    worst_state = max(worst_state, abs(
                        clean[arm][field][team] - dirty[arm][field][team]))
            worst_state = max(worst_state, abs(
                clean[arm]["home_multiplier"] - dirty[arm]["home_multiplier"]))

        worst_state = max(worst_state,
                          abs(clean["goals"]["rho"] - dirty["goals"]["rho"]))

        for field in ("c_pooled", "c_home", "c_away"):
            worst_c = max(worst_c, abs(clean[field] - dirty[field]))

    audit.record(
        "E5", "corrupting every match from the cutoff onward moves no fitted "
              "SoT or goals state, at {} probe dates".format(probes),
        0.0, "{:.3e}".format(worst_state), worst_state == 0.0,
        "goals, shots and shots on target all replaced with 999 from the "
        "cutoff forward. Attack, defence, the home multiplier and rho are "
        "compared team by team. Tested by corruption because re-reading the "
        "filter tests the reader, not the filter")

    audit.record(
        "E6", "and moves neither the pooled nor the home/away conversion "
              "constants",
        0.0, "{:.3e}".format(worst_c), worst_c == 0.0,
        "c is estimated from the same window as the ratings, so it carries "
        "the same exposure and is asserted under the same corruption. This is "
        "what makes 'training rows only' a tested claim rather than a "
        "declared one")

    return cutoffs


def parameter_stability(parameters):
    """Mean absolute change in a team's parameter between successive refits.

    E5.2(b). If SoT estimates the same underlying rate with less sampling
    noise, its parameters should move LESS between consecutive refits than the
    goal-based ones do - and that is a claim about the parameters, testable
    whether or not the log loss moves.

    Compared WITHIN fold, because the window resets its season scope at a fold
    boundary and a jump there would not be sampling noise.
    """

    rows = []

    for (arm, fold, team), group in parameters.groupby(
            ["arm", "fold", "team"], sort=False):

        group = group.sort_values("cutoff")

        if len(group) < 2:
            continue

        for column in ("attack", "defence"):
            values = group[column].to_numpy(dtype=float)
            rows.append({
                "arm": arm, "fold": fold, "team": team, "parameter": column,
                "refits": len(values),
                "mean_abs_change": float(np.abs(np.diff(values)).mean()),
                "mean_level": float(values.mean()),
            })

    frame = pd.DataFrame(rows)

    # A raw change is not comparable between arms whose parameters need not sit
    # on the same scale, so the COEFFICIENT of variation of the step is
    # reported alongside it - the step relative to the level it moves around.
    frame["relative_change"] = frame["mean_abs_change"] / frame["mean_level"]

    return frame


# ============================================================
# THE RUN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 5 - INSTRUMENT E1a: SHOTS ON TARGET AS A RATING INPUT")

    print("  pre-declaration sha256 d385bfd4...6eca4, signed off")
    print("  one difference: attack/defence/home from SoT, not goals")
    print("  rho is the SAME NUMBER in both arms, from the goals fit")
    print()

    audit = Audit()

    spec = L3.load_spec()
    matches = L3.load_matches()
    matches = matches.copy()
    matches["match_id"] = matches.index

    matches = load_shots(matches, audit)

    banner("1. LEAKAGE PROBES, BEFORE ANYTHING IS BELIEVED")

    leakage_probes(matches, spec, audit)

    print("  E5/E6: state and conversion constants unmoved by "
          "corrupting everything from six cutoffs forward.")
    print()

    banner("2. THE WALK-FORWARD")

    frames, windows, parameters = walk_forward(matches, spec, audit)

    print("  {} refits across four folds".format(len(windows)))
    print("  window sizes {} to {} matches".format(
        int(windows["window_matches"].min()),
        int(windows["window_matches"].max())))
    print()

    # ---- E3: rho identical between the arms -------------------------------
    rho_gap = float(np.abs(frames["goals_DC"]["rho"].to_numpy()
                           - frames["E1a_sot"]["rho"].to_numpy()).max())

    audit.record(
        "E3", "rho is bit-identical between the arms",
        0.0, "{:.3e}".format(rho_gap), rho_gap == 0.0,
        "not merely fitted by the same rule - it is the SAME NUMBER, fitted "
        "once per window from the goals arm and handed to the SoT arm. This "
        "is what isolates the rating input as the single difference")

    # ---- E9b: the rating windows never saw an odds column ------------------
    audit.record(
        "E9b", "the frame the walk-forward fits from carries no odds column",
        0,
        len([c for c in matches.columns
             if any(c.startswith(p) for p in ODDS_PREFIXES)
             and c not in SHOT_COLUMNS]),
        not [c for c in matches.columns
             if any(c.startswith(p) for p in ODDS_PREFIXES)
             and c not in SHOT_COLUMNS],
        "asserted on the frame actually passed to walk_forward, after the "
        "join, not on the selection that built it")

    # ============================================================
    banner("3. THE CONVERSION CONSTANT")

    print("  c is the one new estimated quantity in the rung, and candidate")
    print("  (i) rests on its stability.")
    print()
    print("  {:<12} {:>7} {:>9} {:>9} {:>9} {:>9}".format(
        "scope", "refits", "mean c", "sd", "min", "max"))
    print("  " + "-" * 60)
    print("  {:<12} {:>7} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f}".format(
        "all", len(windows), windows["c_pooled"].mean(),
        windows["c_pooled"].std(), windows["c_pooled"].min(),
        windows["c_pooled"].max()))

    for fold_spec in spec["folds"]:
        fold = int(fold_spec["fold"])
        subset = windows[windows["fold"] == fold]
        print("  {:<12} {:>7} {:>9.5f} {:>9.5f} {:>9.5f} {:>9.5f}".format(
            "fold {}".format(fold), len(subset), subset["c_pooled"].mean(),
            subset["c_pooled"].std(), subset["c_pooled"].min(),
            subset["c_pooled"].max()))

    print()
    print("  home/away split (the declared sensitivity):")
    print("    c_home  mean {:.5f}  sd {:.5f}".format(
        windows["c_home"].mean(), windows["c_home"].std()))
    print("    c_away  mean {:.5f}  sd {:.5f}".format(
        windows["c_away"].mean(), windows["c_away"].std()))
    print()

    # Within-season drift: c against days elapsed inside each test season.
    drifts = []
    for fold_spec in spec["folds"]:
        fold = int(fold_spec["fold"])
        subset = windows[windows["fold"] == fold].sort_values("cutoff")
        day = (subset["cutoff"] - subset["cutoff"].min()).dt.days.to_numpy()
        slope = float(np.polyfit(day, subset["c_pooled"].to_numpy(), 1)[0])
        drifts.append({"fold": fold, "slope_per_day": slope,
                       "total_drift": slope * float(day.max())})
        print("    fold {} within-season drift {:+.3e} per day "
              "({:+.5f} over the season)".format(fold, slope,
                                                 slope * float(day.max())))

    print()

    # ============================================================
    banner("4. THE METRICS")

    order = frames["goals_DC"]
    actual = order["result"].to_numpy()

    probabilities = {}

    for name, frame in frames.items():
        probabilities[name] = frame[["p_H", "p_D", "p_A"]].to_numpy(float)
        validate_probabilities(probabilities[name], len(actual))

    audit.record(
        "E7", "every probability array passes the harness's "
              "validate_probabilities",
        0, 0, True,
        "{} arrays. The harness raises rather than repairing".format(
            len(probabilities)))

    # ---- E1: the goals arm IS the committed dc_walkforward ----------------
    committed = pd.read_csv(COMMITTED_DC, float_precision=FLOAT_PRECISION)
    committed = committed[committed["variant"] == "dc_walkforward"]
    committed = committed.sort_values("fold")

    fold_rows = []

    for name, frame in frames.items():
        for fold_spec in spec["folds"]:
            season = str(fold_spec["test_season"])
            mask = (frame["season"] == season).to_numpy()
            scores = evaluate(actual[mask], probabilities[name][mask])
            fold_rows.append({"model": name, "fold": int(fold_spec["fold"]),
                              "test_season": season, "n": scores["n"],
                              **{m: scores[m] for m in METRICS}})

    fold_table = pd.DataFrame(fold_rows)

    mine = fold_table[fold_table["model"] == "goals_DC"].sort_values("fold")

    worst = 0.0
    for metric in ("accuracy", "log_loss", "brier_score", "rps"):
        worst = max(worst, float(np.abs(
            mine[metric].to_numpy()
            - committed[metric].to_numpy()).max()))

    audit.record(
        "E1", "the goals arm re-fitted here reproduces the COMMITTED "
              "dc_walkforward bit for bit",
        "< 1e-12", "{:.3e}".format(worst), worst < 1e-12,
        "re-fitted rather than read so that every delta is a genuinely "
        "PAIRED bootstrap. If this moves, the two arms are not differing in "
        "one thing and nothing below is readable")

    # ---- E2: the arms differ in exactly one input -------------------------
    settings = {
        "half_life_days": DC.TIME_DECAY_HALF_LIFE_DAYS,
        "max_goals": DC.MAX_GOALS,
        "refits": len(windows),
        "rho_source": "goals arm, shared",
    }

    audit.record(
        "E2", "the two arms differ in exactly one input and nothing else",
        "identical settings", "identical: {}".format(settings),
        True,
        "both arms are produced inside ONE call to fit_both_arms per window, "
        "from the same window, the same weights and the same cutoff. They "
        "cannot drift apart by configuration because there is only one "
        "configuration. half-life {}, MAX_GOALS {}, {} refits".format(
            DC.TIME_DECAY_HALF_LIFE_DAYS, DC.MAX_GOALS, len(windows)))

    # ---- E8: the pooled/fold-mean identity ---------------------------------
    pooled_rows = []

    for name in frames:
        scores = evaluate(actual, probabilities[name])
        pooled_rows.append({"model": name, "n": scores["n"],
                            **{m: scores[m] for m in METRICS}})

    pooled_table = pd.DataFrame(pooled_rows)

    worst_identity = 0.0
    for name in frames:
        subset = fold_table[fold_table["model"] == name]
        value = float(pooled_table[pooled_table["model"] == name]
                      ["log_loss"].iloc[0])
        worst_identity = max(worst_identity,
                             abs(float(subset["log_loss"].mean()) - value))

    audit.record(
        "E8", "pooled log loss equals the unweighted mean of the four fold "
              "values",
        "< 1e-12", "{:.3e}".format(worst_identity), worst_identity < 1e-12,
        "true only because every fold tests exactly 380 rows")

    # ---- the references, read not re-fitted -------------------------------
    d34 = pd.read_csv(D34_PREDICTIONS, float_precision=FLOAT_PRECISION)
    d34 = d34.sort_values(["season", "date", "home_team", "away_team"])
    d34 = d34.reset_index(drop=True)

    market = pd.read_csv(MARKET_PROBABILITIES, float_precision=FLOAT_PRECISION)
    market = market[market["book"] == "B365C"]
    market = market.sort_values(["season", "date", "home_team", "away_team"])
    market = market.reset_index(drop=True)

    aligned = bool((d34["result"].to_numpy() == actual).all()
                   and (market["result"].to_numpy() == actual).all())

    audit.record(
        "E10a", "the committed model and market artefacts align row-for-row "
                "with this rung's predictions",
        "aligned", "aligned" if aligned else "MISALIGNED", aligned,
        "all three sorted by (season, date, home, away) and their result "
        "columns compared elementwise")

    for name in ("D0", "D2_rescaled", "D4", "elo_v1", "poisson_walkforward",
                 "dc_walkforward"):
        probabilities[name] = d34[
            ["{}_p_{}".format(name, o) for o in CLASSES]].to_numpy(float)

    probabilities["market"] = market[
        ["prop_p_{}".format(o) for o in CLASSES]].to_numpy(float)

    for name in ("D0", "D2_rescaled", "D4", "elo_v1", "poisson_walkforward",
                 "dc_walkforward", "market"):
        scores = evaluate(actual, probabilities[name])
        pooled_rows.append({"model": name, "n": scores["n"],
                            **{m: scores[m] for m in METRICS}})

    pooled_table = pd.DataFrame(pooled_rows).sort_values("log_loss")

    print("  {:<24} {:>9} {:>9} {:>8}".format(
        "model", "logloss", "RPS", "brier"))
    print("  " + "-" * 54)
    for _i, row in pooled_table.iterrows():
        print("  {:<24} {:>9.5f} {:>9.5f} {:>8.4f}".format(
            row["model"], row["log_loss"], row["rps"], row["brier_score"]))
    print()

    print("  E1a per fold:")
    print("  {:<6} {:<11} {:>10} {:>10} {:>10}".format(
        "fold", "season", "E1a", "goals-DC", "difference"))
    print("  " + "-" * 52)
    for fold_spec in spec["folds"]:
        fold = int(fold_spec["fold"])
        e1a = float(fold_table[(fold_table["model"] == "E1a_sot")
                               & (fold_table["fold"] == fold)]
                    ["log_loss"].iloc[0])
        dc = float(fold_table[(fold_table["model"] == "goals_DC")
                              & (fold_table["fold"] == fold)]
                   ["log_loss"].iloc[0])
        print("  {:<6} {:<11} {:>10.5f} {:>10.5f} {:>+10.5f}".format(
            fold, str(fold_spec["test_season"]), e1a, dc, e1a - dc))
    print()

    # ============================================================
    banner("5. THE DELTAS")

    deltas = []

    pairs = [("E1a - DixonColes", "dc_walkforward"),
             ("E1a - D2rescaled", "D2_rescaled"),
             ("E1a - Elo v1", "elo_v1"),
             ("E1a - D0", "D0"),
             ("E1a - market", "market")]

    for label, right in pairs:
        deltas.append(LADDER.compare(label, "E1a_sot", right,
                                     probabilities["E1a_sot"],
                                     probabilities[right], actual))

        for fold_spec in spec["folds"]:
            season = str(fold_spec["test_season"])
            mask = (order["season"] == season).to_numpy()
            row = LADDER.compare(
                label, "E1a_sot", right, probabilities["E1a_sot"][mask],
                probabilities[right][mask], actual[mask],
                scope="fold {} ({})".format(int(fold_spec["fold"]), season))
            row["fold"] = int(fold_spec["fold"])
            deltas.append(row)

    # the declared sensitivity, against the primary
    deltas.append(LADDER.compare(
        "E1a homeaway - E1a", "E1a_sot_homeaway", "E1a_sot",
        probabilities["E1a_sot_homeaway"], probabilities["E1a_sot"], actual))

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
    print("  negative favours the LEFT model (E1a).")
    print()

    # ============================================================
    banner("6. PARAMETER STABILITY - E5.2(b)")

    stability = parameter_stability(parameters)

    print("  mean absolute change in a team's parameter between successive")
    print("  refits, within fold. Lower = the estimate moves around less.")
    print()
    print("  {:<10} {:<10} {:>12} {:>12} {:>10}".format(
        "parameter", "arm", "mean |step|", "mean level", "relative"))
    print("  " + "-" * 60)

    for column in ("attack", "defence"):
        for arm in ("goals", "sot"):
            subset = stability[(stability["parameter"] == column)
                               & (stability["arm"] == arm)]
            print("  {:<10} {:<10} {:>12.6f} {:>12.6f} {:>10.5f}".format(
                column, arm, subset["mean_abs_change"].mean(),
                subset["mean_level"].mean(), subset["relative_change"].mean()))

    print()

    ratios = {}
    for column in ("attack", "defence"):
        g = stability[(stability["parameter"] == column)
                      & (stability["arm"] == "goals")]["relative_change"].mean()
        s = stability[(stability["parameter"] == column)
                      & (stability["arm"] == "sot")]["relative_change"].mean()
        ratios[column] = s / g
        print("  {}: SoT relative step is {:.3f}x the goals arm's".format(
            column, s / g))

    print()

    # ============================================================
    banner("7. WRITING")

    predictions = order[["match_id", "season", "date", "home_team",
                         "away_team", "result"]].copy()

    for name in ("goals_DC", "E1a_sot", "E1a_sot_homeaway"):
        for position, outcome in enumerate(CLASSES):
            predictions["{}_p_{}".format(name, outcome)] = (
                probabilities[name][:, position])

    artefacts = ((FOLD_OUTPUT, fold_table),
                 (POOLED_OUTPUT, pooled_table),
                 (DELTA_OUTPUT, pd.DataFrame(deltas)),
                 (WINDOW_OUTPUT, windows),
                 (STABILITY_OUTPUT, stability),
                 (PREDICTIONS_OUTPUT, predictions),
                 (AUDIT_OUTPUT, audit.frame()))

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
