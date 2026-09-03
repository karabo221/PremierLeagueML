"""
===============================================================================
PHASE 5 - GAP DIAGNOSTIC:  WHERE DOES THE 0.02979 TO THE MARKET LIVE?
===============================================================================

EXPLORATORY. NO FITTING, NO GATE, NO KEEP/DROP. Nothing here enters a model
and nothing here licenses a design decision. It is a description of a gap that
three declared rungs have failed to close.

WHY IT IS BEING LOOKED AT AT ALL. E1a put shot information inside the rating
and did not close the gap. E1b put it beside the rating and did not close it.
E1a's third branch fired: SoT estimates attack and defence about 40% more
precisely between refits and none of that converts to accuracy. So the residual
gap is not a rating-precision problem, and the question of what it IS has no
answer yet.

WHAT THIS CANNOT ESTABLISH, STATED BEFORE THE NUMBERS

  It cannot identify a cause. Every split below is a description of where the
  gap SITS, and a gap that sits somewhere is consistent with many mechanisms.

  It cannot be used to design a rung. A split chosen after seeing which one
  showed something, turned into a feature, would be fitted on these very
  outer-test rows. THE E1c DECLARATION WAS HASHED BEFORE THIS RAN, which is
  the only reason E1c is unaffected by it.

  It has no multiplicity control. Section 5 correlates the per-match gap
  against every numeric column on disk. At n=1520 the two-sided 5% critical
  value for a single correlation is |r| ~ 0.050, so across ~130 columns about
  SIX are expected to clear it with nothing behind them. The full ranked list
  is printed, and the count examined is printed beside it, so that nothing
  here can be quoted as a selected finding.

  It conditions on the outcome in section 4b. "The favourite lost" is not
  available before kick-off. That split describes; it cannot be used.

EVERY SPLIT LOOKED AT IS REPORTED, INCLUDING THE NULL ONES. That is the whole
discipline of this file: the ones that show something must not be selected
from a larger set silently.

THE MODEL SIDE IS BOTH DIXON-COLES AND E1a, per the brief. If they disagree
about where the gap lives, that is itself informative and is reported as such.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase3_feature_builder import banner, configure_stdout  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

E1A_PREDICTIONS = OUTPUTS_DIR / "phase5_e1a_predictions.csv"
MARKET_PROBABILITIES = OUTPUTS_DIR / "phase5_market_probabilities.csv"
FEATURES_CSV = OUTPUTS_DIR / "phase3_features.csv"
DYNAMIC_CSV = OUTPUTS_DIR / "phase4_dynamic_state.csv"
RESIDUAL_CSV = OUTPUTS_DIR / "phase5_e1b_residual_features.csv"

SPLIT_OUTPUT = OUTPUTS_DIR / "phase5_gap_splits.csv"
CORR_OUTPUT = OUTPUTS_DIR / "phase5_gap_correlations.csv"
MATCH_OUTPUT = OUTPUTS_DIR / "phase5_gap_per_match.csv"

FLOAT_PRECISION = "round_trip"
FLOAT_FORMAT = "%.17g"

# B4.1 of PHASE5_MARKET_PREDECLARATION.txt. Bet365 closing, proportional
# de-vig, which is the DECLARED PRIMARY. Shin is the declared sensitivity and
# is reported alongside so that no split rests on the de-vig choice.
PRIMARY_BOOK = "B365C"
PRIMARY_DEVIG = "prop"
SENSITIVITY_DEVIG = "shin"

CLASSES = ["H", "D", "A"]

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20260901

# The favourite-probability bins. Fixed edges, declared here rather than
# chosen from the data's quantiles, so the bin boundaries cannot be moved to
# make a bin look different.
FAVOURITE_BINS = [(0.00, 0.40), (0.40, 0.50), (0.50, 0.60),
                  (0.60, 0.70), (0.70, 1.01)]

# Matchweek buckets. Early season is where a rating has least information and
# where a promoted side is least characterised, so the first bucket is the one
# the lineup hypothesis would not especially predict and the rating-warm-up
# story would.
MATCHWEEK_BUCKETS = [(1, 6), (7, 19), (20, 31), (32, 38)]

STATUS_ABSENT = "absent_from_previous_season"


# ============================================================
# LOADING
# ============================================================

def load_market():
    """The primary benchmark and its declared sensitivity, on the 1,520."""

    market = pd.read_csv(MARKET_PROBABILITIES, float_precision=FLOAT_PRECISION)
    market = market[market["book"] == PRIMARY_BOOK].copy()
    market["date"] = pd.to_datetime(market["date"], format="%Y-%m-%d")

    keep = ["season", "date", "home_team", "away_team"]

    for devig in (PRIMARY_DEVIG, SENSITIVITY_DEVIG):
        for cls in CLASSES:
            keep.append("{}_p_{}".format(devig, cls))

    return market[keep]


def load_models():
    """Dixon-Coles and E1a, both already scored on the frozen outer folds."""

    preds = pd.read_csv(E1A_PREDICTIONS, float_precision=FLOAT_PRECISION)
    preds["date"] = pd.to_datetime(preds["date"], format="%Y-%m-%d")

    return preds


def logloss_of(frame, prefix, actual):
    """Per-match log loss for one probability triple."""

    picked = np.array([frame["{}_p_{}".format(prefix, cls)].to_numpy()
                       for cls in CLASSES]).T

    index = np.array([CLASSES.index(r) for r in actual], dtype=int)

    return -np.log(picked[np.arange(len(index)), index])


def rps_of(frame, prefix, actual):
    """Per-match ranked probability score, on the H/D/A ordering."""

    probs = np.array([frame["{}_p_{}".format(prefix, cls)].to_numpy()
                      for cls in CLASSES]).T

    index = np.array([CLASSES.index(r) for r in actual], dtype=int)

    outcomes = np.zeros_like(probs)
    outcomes[np.arange(len(index)), index] = 1.0

    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(outcomes, axis=1)

    return ((cum_p - cum_o) ** 2)[:, :-1].sum(axis=1) / (len(CLASSES) - 1)


# ============================================================
# THE INTERVAL
# ============================================================

def mean_ci(values, rng):
    """
    Percentile bootstrap on the mean of a per-match gap.

    Not a paired model-vs-model test - the pairing is already inside each
    value, which is one match's model loss minus that same match's market
    loss. This interval says how well the MEAN of a subset is pinned down by
    the number of matches in it, which is the thing that stops a six-match
    bin being read as a finding.
    """

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    draws = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[draws].mean(axis=1)

    return (float(values.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def split_rows(frame, label, group_column, rng, rows):
    """One reported split. EVERY level is emitted, including empty-ish ones."""

    print()
    print("  {}".format(label))
    print("  {:<26} {:>5} {:>10} {:>22} {:>10} {:>10}".format(
        "level", "n", "DC gap", "95% CI", "E1a gap", "d_RPS DC"))
    print("  " + "-" * 88)

    for level, chunk in frame.groupby(group_column, sort=False, observed=True):

        dc_mean, dc_lo, dc_hi = mean_ci(chunk["gap_dc"], rng)
        e1_mean, _e1_lo, _e1_hi = mean_ci(chunk["gap_e1a"], rng)
        rps_mean = float(chunk["gap_rps_dc"].mean())

        print("  {:<26} {:>5} {:>+10.5f} {:>22} {:>+10.5f} {:>+10.5f}".format(
            str(level), len(chunk), dc_mean,
            "[{:+.5f}, {:+.5f}]".format(dc_lo, dc_hi), e1_mean, rps_mean))

        rows.append({
            "split": label, "level": str(level), "n": len(chunk),
            "gap_dc": dc_mean, "gap_dc_lo": dc_lo, "gap_dc_hi": dc_hi,
            "gap_e1a": e1_mean, "gap_rps_dc": rps_mean,
            "market_logloss": float(chunk["ll_market"].mean()),
            "dc_logloss": float(chunk["ll_dc"].mean()),
            "e1a_logloss": float(chunk["ll_e1a"].mean()),
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    configure_stdout()

    banner("PHASE 5 - GAP DIAGNOSTIC (EXPLORATORY, NO GATE)")

    print("  Every split looked at is reported, including the null ones.")
    print("  Nothing here enters a model. E1c was hashed before this ran.")
    print()

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # ---- assemble -------------------------------------------------------
    preds = load_models()
    market = load_market()

    keys = ["season", "date", "home_team", "away_team"]

    frame = preds.merge(market, on=keys, how="left", validate="one_to_one")

    missing = int(frame["{}_p_H".format(PRIMARY_DEVIG)].isna().sum())

    if len(frame) != 1520 or missing:
        raise SystemExit(
            "FATAL: expected 1520 joined rows with a complete primary book, "
            "got {} rows and {} missing".format(len(frame), missing))

    actual = frame["result"].to_numpy()

    frame["ll_market"] = logloss_of(frame, PRIMARY_DEVIG, actual)
    frame["ll_market_shin"] = logloss_of(frame, SENSITIVITY_DEVIG, actual)
    frame["ll_dc"] = logloss_of(frame, "goals_DC", actual)
    frame["ll_e1a"] = logloss_of(frame, "E1a_sot", actual)

    frame["gap_dc"] = frame["ll_dc"] - frame["ll_market"]
    frame["gap_e1a"] = frame["ll_e1a"] - frame["ll_market"]
    frame["gap_dc_shin"] = frame["ll_dc"] - frame["ll_market_shin"]

    frame["rps_market"] = rps_of(frame, PRIMARY_DEVIG, actual)
    frame["rps_dc"] = rps_of(frame, "goals_DC", actual)
    frame["rps_e1a"] = rps_of(frame, "E1a_sot", actual)
    frame["gap_rps_dc"] = frame["rps_dc"] - frame["rps_market"]
    frame["gap_rps_e1a"] = frame["rps_e1a"] - frame["rps_market"]

    # ---- the headline the splits decompose ------------------------------
    banner("0. THE QUANTITY BEING DECOMPOSED")

    print("  {:<28} {:>12} {:>12}".format("", "log loss", "RPS"))
    print("  " + "-" * 54)
    for label, ll, rps in (
            ("market (B365C, prop)", frame["ll_market"], frame["rps_market"]),
            ("market (B365C, shin)", frame["ll_market_shin"], None),
            ("Dixon-Coles", frame["ll_dc"], frame["rps_dc"]),
            ("E1a SoT ratings", frame["ll_e1a"], frame["rps_e1a"])):
        print("  {:<28} {:>12.5f} {:>12.5f}".format(
            label, float(ll.mean()),
            float(rps.mean()) if rps is not None else float("nan")))

    print()
    mean, lo, hi = mean_ci(frame["gap_dc"], rng)
    print("  DC  - market : {:+.5f}  [{:+.5f}, {:+.5f}]".format(mean, lo, hi))
    mean_e, lo_e, hi_e = mean_ci(frame["gap_e1a"], rng)
    print("  E1a - market : {:+.5f}  [{:+.5f}, {:+.5f}]".format(
        mean_e, lo_e, hi_e))
    mean_s, _, _ = mean_ci(frame["gap_dc_shin"], rng)
    print("  DC  - market under the SHIN sensitivity : {:+.5f}".format(mean_s))
    print()
    print("  The per-match gap is not a small number that is small everywhere.")
    print("  sd {:.5f}, min {:+.5f}, max {:+.5f} - it is a DIFFERENCE OF TWO".format(
        float(frame["gap_dc"].std(ddof=0)), float(frame["gap_dc"].min()),
        float(frame["gap_dc"].max())))
    print("  LOG LOSSES on single matches, so its spread dwarfs its mean and")
    print("  every subset mean below needs its interval read with it.")

    # ---- the splits ------------------------------------------------------
    banner("1. BY MARKET-IMPLIED FAVOURITE PROBABILITY")

    fav_p = frame[["{}_p_{}".format(PRIMARY_DEVIG, c)
                   for c in CLASSES]].max(axis=1)
    frame["favourite_p"] = fav_p

    labels = []
    for lo_edge, hi_edge in FAVOURITE_BINS:
        labels.append("{:.2f}-{:.2f}".format(lo_edge, min(hi_edge, 1.0)))

    frame["favourite_bin"] = pd.cut(
        fav_p, bins=[b[0] for b in FAVOURITE_BINS] + [FAVOURITE_BINS[-1][1]],
        labels=labels, right=False)

    frame = frame.sort_values("favourite_p").reset_index(drop=True)

    rows = []
    rows = split_rows(frame, "favourite probability", "favourite_bin",
                      rng, rows)

    banner("2. BY MATCHWEEK, AND BY SEASON")

    features = pd.read_csv(FEATURES_CSV, float_precision=FLOAT_PRECISION)
    features["date"] = pd.to_datetime(features["date"], format="%Y-%m-%d")

    context = features[keys + ["matchweek", "home_prev_season_status",
                               "away_prev_season_status"]]

    frame = frame.merge(context, on=keys, how="left", validate="one_to_one")

    if int(frame["matchweek"].isna().sum()):
        raise SystemExit("FATAL: matchweek did not join on every row")

    def bucket(week):
        for lo_w, hi_w in MATCHWEEK_BUCKETS:
            if lo_w <= week <= hi_w:
                return "MW {}-{}".format(lo_w, hi_w)
        return "MW ?"

    frame["matchweek_bucket"] = frame["matchweek"].map(bucket)

    rows = split_rows(frame, "matchweek bucket", "matchweek_bucket", rng, rows)
    rows = split_rows(frame, "season", "season", rng, rows)

    banner("3. BY PROMOTED SIDE")

    promoted = ((frame["home_prev_season_status"] == STATUS_ABSENT)
                | (frame["away_prev_season_status"] == STATUS_ABSENT))

    frame["promoted_involved"] = np.where(promoted, "promoted side", "neither")

    both = ((frame["home_prev_season_status"] == STATUS_ABSENT)
            & (frame["away_prev_season_status"] == STATUS_ABSENT))

    frame["promoted_detail"] = np.select(
        [both,
         frame["home_prev_season_status"] == STATUS_ABSENT,
         frame["away_prev_season_status"] == STATUS_ABSENT],
        ["both promoted", "home promoted", "away promoted"],
        default="neither promoted")

    rows = split_rows(frame, "promoted involved", "promoted_involved",
                      rng, rows)
    rows = split_rows(frame, "promoted detail", "promoted_detail", rng, rows)

    banner("4. PER OUTCOME, AND THE FAVOURITE'S FATE")

    frame["actual"] = frame["result"]
    rows = split_rows(frame, "actual outcome", "actual", rng, rows)

    market_pick = np.array(CLASSES)[
        frame[["{}_p_{}".format(PRIMARY_DEVIG, c)
               for c in CLASSES]].to_numpy().argmax(axis=1)]

    frame["market_pick"] = market_pick
    frame["favourite_correct"] = np.where(
        market_pick == frame["result"].to_numpy(),
        "favourite delivered", "favourite did not")

    rows = split_rows(frame, "market pick", "market_pick", rng, rows)
    rows = split_rows(frame, "did the favourite deliver", "favourite_correct",
                      rng, rows)

    print()
    print("  CONDITIONING ON THE OUTCOME. 'The favourite did not deliver' is")
    print("  not known before kick-off. This split DESCRIBES the gap; it can")
    print("  never be a feature, and no rung may be designed from it.")

    banner("4b. STRONG FAVOURITE x WHETHER IT DELIVERED")

    strong = frame[frame["favourite_p"] >= 0.60].copy()
    strong["cell"] = strong["favourite_correct"]

    print()
    print("  The hypothesis under test: the gap concentrates where a strong")
    print("  side underperforms its rating. Restricted to favourite_p >= 0.60,")
    print("  n = {}.".format(len(strong)))

    rows = split_rows(strong, "strong favourite (p>=0.60)", "cell", rng, rows)

    banner("4c. SHARPNESS - TURNING THE 4b READING INTO A MEASUREMENT")

    print()
    print("  4b says each model is worse than the market when the favourite")
    print("  delivers and better when it does not. That is the signature of")
    print("  being LESS CONFIDENT than the market, and confidence is directly")
    print("  measurable rather than inferable. Reported so the reading is a")
    print("  measurement and not a story about one.")
    print()
    print("  {:<24} {:>12} {:>16} {:>12}".format(
        "", "mean max p", "p on mkt's pick", "mean p(D)"))
    print("  " + "-" * 68)

    for label, prefix in (("market (B365C, prop)", PRIMARY_DEVIG),
                          ("Dixon-Coles", "goals_DC"),
                          ("E1a SoT ratings", "E1a_sot")):

        probs = np.array([frame["{}_p_{}".format(prefix, c)].to_numpy()
                          for c in CLASSES]).T

        pick_index = np.array([CLASSES.index(p)
                               for p in frame["market_pick"]], dtype=int)

        print("  {:<24} {:>12.4f} {:>16.4f} {:>12.4f}".format(
            label, float(probs.max(axis=1).mean()),
            float(probs[np.arange(len(probs)), pick_index].mean()),
            float(probs[:, CLASSES.index("D")].mean())))

    print()
    print("  observed base rates on the 1,520: H {:.4f}  D {:.4f}  A {:.4f}".format(
        float((frame["result"] == "H").mean()),
        float((frame["result"] == "D").mean()),
        float((frame["result"] == "A").mean())))

    banner("5. CORRELATION WITH EVERY NUMERIC COLUMN ON DISK")

    dynamic = pd.read_csv(DYNAMIC_CSV, float_precision=FLOAT_PRECISION)
    residual = pd.read_csv(RESIDUAL_CSV, float_precision=FLOAT_PRECISION)

    pool = features.merge(
        preds[keys + ["match_id"]], on=keys, how="inner",
        validate="one_to_one")

    # Both carry their own copies of the identity columns. Merging those in
    # would suffix them and silently break the join that follows, which is
    # exactly what happened on the first run.
    for extra in (dynamic, residual):
        shared = [c for c in extra.columns
                  if c != "match_id" and c not in pool.columns]
        pool = pool.merge(extra[["match_id"] + shared], on="match_id",
                          how="left", validate="one_to_one")

    pool = pool.merge(
        frame[keys + ["gap_dc", "gap_e1a", "favourite_p"]], on=keys,
        how="left", validate="one_to_one")

    numeric = [c for c in pool.columns
               if c not in ("gap_dc", "gap_e1a", "match_id", "matchweek")
               and pd.api.types.is_numeric_dtype(pool[c])
               and not pd.api.types.is_bool_dtype(pool[c])
               and pool[c].nunique(dropna=True) > 1]

    corr_rows = []

    for column in numeric:
        pair = pool[[column, "gap_dc", "gap_e1a"]].dropna()
        if len(pair) < 100 or pair[column].nunique() <= 1:
            continue
        corr_rows.append({
            "column": column, "n": len(pair),
            "r_gap_dc": float(pair[column].corr(pair["gap_dc"])),
            "r_gap_e1a": float(pair[column].corr(pair["gap_e1a"])),
        })

    corr = pd.DataFrame(corr_rows)
    corr["abs_r"] = corr["r_gap_dc"].abs()
    corr = corr.sort_values("abs_r", ascending=False).reset_index(drop=True)

    critical = 1.96 / np.sqrt(1520)

    print()
    print("  columns examined            : {}".format(len(corr)))
    print("  two-sided 5% critical |r|   : {:.4f}  (single test, n=1520)".format(
        critical))
    print("  expected to clear by chance : {:.1f}".format(
        0.05 * len(corr)))
    print("  actually clearing           : {}".format(
        int((corr["abs_r"] > critical).sum())))
    print("  largest |r| anywhere        : {:.4f}".format(
        float(corr["abs_r"].max())))
    print()
    print("  top 20 by |r| against the Dixon-Coles gap:")
    print("  {:<38} {:>9} {:>10}".format("column", "r vs DC", "r vs E1a"))
    print("  " + "-" * 60)
    for row in corr.head(20).itertuples():
        print("  {:<38} {:>+9.4f} {:>+10.4f}".format(
            row.column, row.r_gap_dc, row.r_gap_e1a))

    # ---- write -----------------------------------------------------------
    banner("6. WRITING")

    splits = pd.DataFrame(rows)
    splits.to_csv(SPLIT_OUTPUT, index=False, float_format=FLOAT_FORMAT)
    corr.to_csv(CORR_OUTPUT, index=False, float_format=FLOAT_FORMAT)

    keep = keys + ["result", "ll_market", "ll_dc", "ll_e1a", "gap_dc",
                   "gap_e1a", "gap_rps_dc", "favourite_p", "favourite_bin",
                   "matchweek", "matchweek_bucket", "promoted_detail",
                   "market_pick", "favourite_correct"]
    frame[keep].to_csv(MATCH_OUTPUT, index=False, float_format=FLOAT_FORMAT)

    for path in (SPLIT_OUTPUT, CORR_OUTPUT, MATCH_OUTPUT):
        print("  {}".format(path))

    print()
    print("  These match outputs/phase5_*.csv, so they are tracked and hashed")
    print("  like every other Phase 5 artefact. THAT DOES NOT PROMOTE THEM.")
    print("  The manifest records WHAT WAS PRODUCED; it does not turn a")
    print("  description into evidence. Freezing them is in fact the useful")
    print("  half - an exploratory number that can be quietly rewritten later")
    print("  is worse than one that cannot. No instrument anchors to them and")
    print("  no gate reads them.")
    print()


if __name__ == "__main__":
    main()
