"""
===============================================================================
PHASE 4 - D2's DYNAMIC-STATE BLOCK, GENERATED POINT-IN-TIME
===============================================================================

Four columns, and nothing else (D2 pre-declaration section 2):

    rel_elo_diff           home Elo before match  -  away Elo before match
    rel_attack_diff        DC attack, refit per window, home - away
    rel_defence_diff       DC defence, refit per window, home - away
    expected_total_goals   lambda_home + lambda_away

WHY THIS IS A SEPARATE FILE FROM phase4_dc_passthrough.py

    The passthrough needed lambda_home and lambda_away. D2 needs the ATTACK
    and DEFENCE strengths those lambdas are built from, and the passthrough's
    generate_state() never wrote them out. Regenerating here rather than
    patching the diagnostic keeps the diagnostic's frozen output untouched.

    The cadence is identical and is still inherited, not invented: one refit
    per distinct calendar date, on every match with date STRICTLY earlier.
    S1 asserts the lambdas this produces are bit-identical to the ones the
    passthrough already wrote, so the two files cannot silently diverge.

    Elo needs no regeneration. phase2_elo_results.csv carries exactly one row
    per match for all 1,900, from a single continuous walk - S4 proves the
    before-state of every match is a function of strictly earlier matches by
    chaining it against the after-state of each team's previous match.
===============================================================================
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase2_poisson_dixon_coles as DC  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ELO_RESULTS = OUTPUTS_DIR / "phase2_elo_results.csv"
DC_RESULTS = OUTPUTS_DIR / "phase2_poisson_dc_results.csv"
PASSTHROUGH_STATE = OUTPUTS_DIR / "phase4_dc_state.csv"

STATE_OUTPUT = OUTPUTS_DIR / "phase4_dynamic_state.csv"

FLOAT_PRECISION = "round_trip"
DC_VARIANT = "dc_walkforward"

# Phase 2's own, inherited unchanged (D2 pre-declaration section 6).
ELO_INITIAL = 1500.0
ELO_REGRESSION = 0.75

DYNAMIC_COLUMNS = ["rel_elo_diff", "rel_attack_diff", "rel_defence_diff",
                   "expected_total_goals"]


def generate_dc_state(matches):
    """
    Point-in-time Dixon-Coles state for every one of the 1,900 matches.

    One refit per distinct date; the fitting window is every match with date
    STRICTLY before that date. Read out of run_fold(), not reinvented.

    A team absent from the window takes NEUTRAL_STRENGTH, exactly as
    match_rates() does - and the row is FLAGGED so the count of such rows can
    be reported rather than silently absorbed.
    """

    rows = []
    refits = 0

    for cutoff in sorted(matches["date"].unique()):

        cutoff = pd.Timestamp(cutoff)

        window = matches[matches["date"] < cutoff]
        block = matches[matches["date"] == cutoff]

        if len(window) == 0:
            for _i, match in block.iterrows():
                rows.append({
                    "match_id": int(match["match_id"]),
                    "lambda_home": np.nan, "lambda_away": np.nan,
                    "attack_home": np.nan, "attack_away": np.nan,
                    "defence_home": np.nan, "defence_away": np.nan,
                    "rho": np.nan, "window_matches": 0,
                    "has_state": False,
                    "home_has_history": False, "away_has_history": False,
                })
            continue

        model = DC.fit_window(window, cutoff, True)
        refits += 1

        attack = model["attack"]
        defence = model["defence"]

        home_rates, away_rates = DC.match_rates(
            block, attack, defence, model["home_multiplier"])

        for (_i, match), lambda_home, lambda_away in zip(
                block.iterrows(), home_rates, away_rates):

            home, away = match["home_team"], match["away_team"]

            rows.append({
                "match_id": int(match["match_id"]),
                "lambda_home": float(lambda_home),
                "lambda_away": float(lambda_away),
                "attack_home": float(attack.get(home, DC.NEUTRAL_STRENGTH)),
                "attack_away": float(attack.get(away, DC.NEUTRAL_STRENGTH)),
                "defence_home": float(defence.get(home, DC.NEUTRAL_STRENGTH)),
                "defence_away": float(defence.get(away, DC.NEUTRAL_STRENGTH)),
                "rho": float(model["rho"]),
                "window_matches": int(len(window)),
                "has_state": True,
                "home_has_history": bool(home in attack),
                "away_has_history": bool(away in attack),
            })

    frame = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)

    return frame, refits


def generate_static_dc_state(matches):
    """
    AMENDMENT 5 A5.1. Dixon-Coles state FROZEN AT SEASON START.

    One fit per season, in place of one fit per distinct date:

        cutoff  the maximum date in the window - Phase 2's own convention
                for a static fit, and the reference the 107-day decay is
                measured back from
        window  every match with date STRICTLY EARLIER than the first date
                of the season
        state   used unchanged for every match in that season

    Read out of phase2_poisson_dixon_coles.run_fold()'s static branch,
    which fits `train` at `train["date"].max()`, and out of tier 2's
    ARM_STATIC, which does the same. Not a cadence invented for this rung.

    On a fold's TEST rows this coincides with arm A by construction: a test
    season is preceded by exactly that fold's training seasons. G8 asserts it
    against Phase 2's stored dc_static lambdas rather than claiming it.

    2021-22 has an EMPTY window and therefore no state at all. A5.2 declares
    that in advance and turns it into a test rather than a caveat.
    """

    rows = []
    fits = 0

    for season in sorted(matches["season"].unique()):

        block = matches[matches["season"] == season]

        window = matches[matches["date"] < block["date"].min()]

        if len(window) == 0:
            for _i, match in block.iterrows():
                rows.append({
                    "match_id": int(match["match_id"]),
                    "lambda_home": np.nan, "lambda_away": np.nan,
                    "attack_home": np.nan, "attack_away": np.nan,
                    "defence_home": np.nan, "defence_away": np.nan,
                    "rho": np.nan, "window_matches": 0, "has_state": False,
                    "home_has_history": False, "away_has_history": False,
                })
            continue

        model = DC.fit_window(window, window["date"].max(), True)
        fits += 1

        attack = model["attack"]
        defence = model["defence"]

        home_rates, away_rates = DC.match_rates(
            block, attack, defence, model["home_multiplier"])

        for (_i, match), lambda_home, lambda_away in zip(
                block.iterrows(), home_rates, away_rates):

            home, away = match["home_team"], match["away_team"]

            rows.append({
                "match_id": int(match["match_id"]),
                "lambda_home": float(lambda_home),
                "lambda_away": float(lambda_away),
                "attack_home": float(attack.get(home, DC.NEUTRAL_STRENGTH)),
                "attack_away": float(attack.get(away, DC.NEUTRAL_STRENGTH)),
                "defence_home": float(defence.get(home, DC.NEUTRAL_STRENGTH)),
                "defence_away": float(defence.get(away, DC.NEUTRAL_STRENGTH)),
                "rho": float(model["rho"]),
                "window_matches": int(len(window)),
                "has_state": True,
                "home_has_history": bool(home in attack),
                "away_has_history": bool(away in attack),
            })

    frame = pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)

    return frame, fits


def season_start_ratings(elo_state):
    """
    AMENDMENT 5 A5.1. Each team's Elo AT THE START OF ITS SEASON.

    Phase 2's season_start_ratings() verbatim - continuing teams regress to
    1500 + (r - 1500) * 0.75, everything else starts flat at 1500 - but READ
    out of the frozen Elo artefact rather than recomputed, as the team's
    elo_before at its first match of that season. Within a season it never
    moves.
    """

    sides = []

    for side in ("home", "away"):
        part = elo_state[["season", "date", "{}_team".format(side),
                          "{}_elo_before".format(side)]].copy()
        part.columns = ["season", "date", "team", "rating"]
        sides.append(part)

    long = pd.concat(sides, ignore_index=True).sort_values(["date", "team"])

    first = long.groupby(["season", "team"], sort=True)["rating"].first()

    return first


def load_elo_state(matches):
    """One row per match, from Phase 2's single continuous Elo walk."""

    elo = pd.read_csv(ELO_RESULTS, float_precision=FLOAT_PRECISION)
    elo["date"] = pd.to_datetime(elo["date"], format="%Y-%m-%d")

    keyed = matches[["match_id", "season", "date",
                     "home_team", "away_team"]].merge(
        elo[["date", "home", "away", "home_elo_before", "away_elo_before",
             "home_elo_after", "away_elo_after",
             "home_transition", "away_transition"]],
        left_on=["date", "home_team", "away_team"],
        right_on=["date", "home", "away"],
        how="left", validate="one_to_one")

    if keyed["home_elo_before"].isna().any():
        raise SystemExit("FATAL: Elo state missing for at least one match")

    return keyed


def _assemble(elo_state, dc_state, home_rating, away_rating):

    frame = elo_state.merge(dc_state, on="match_id", how="left",
                            validate="one_to_one")

    frame["rel_elo_diff"] = home_rating - away_rating
    frame["rel_attack_diff"] = frame["attack_home"] - frame["attack_away"]
    frame["rel_defence_diff"] = frame["defence_home"] - frame["defence_away"]
    frame["expected_total_goals"] = frame["lambda_home"] + frame["lambda_away"]

    return frame


def build(matches):
    """The full dynamic-state frame, keyed on match_id. Refit per date."""

    dc_state, refits = generate_dc_state(matches)
    elo_state = load_elo_state(matches)

    frame = _assemble(elo_state, dc_state,
                      elo_state["home_elo_before"].to_numpy(),
                      elo_state["away_elo_before"].to_numpy())

    return frame, refits


def build_static(matches):
    """
    AMENDMENT 5. The same four columns, frozen at season start.

    Identical assembly, identical column names, so a rung built on this is
    the same design matrix with the same widths - the only thing that differs
    is when the state stopped updating.
    """

    dc_state, fits = generate_static_dc_state(matches)
    elo_state = load_elo_state(matches)

    starts = season_start_ratings(elo_state)

    home = pd.MultiIndex.from_arrays(
        [elo_state["season"], elo_state["home_team"]])
    away = pd.MultiIndex.from_arrays(
        [elo_state["season"], elo_state["away_team"]])

    home_rating = starts.reindex(home).to_numpy(float)
    away_rating = starts.reindex(away).to_numpy(float)

    if np.isnan(home_rating).any() or np.isnan(away_rating).any():
        raise SystemExit("FATAL: a team has no season-start rating")

    frame = _assemble(elo_state, dc_state, home_rating, away_rating)

    return frame, fits
