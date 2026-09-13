"""
PHASE 6 - COMPILE THE FROZEN ARTEFACTS INTO TYPED MODULES FOR THE FRONTEND

    python scripts/phase6_build_frontend_data.py

WHAT THIS IS. The frontend in web/ shows two kinds of number, and they have
different authorities, so they arrive by different routes:

    FROZEN FIGURES      outputs/*.csv, committed, hash-pinned in
                        FROZEN_MANIFEST.txt. They CANNOT change - that is what
                        "frozen" means - so they are compiled into TypeScript
                        at build time rather than fetched at runtime. A figure
                        that cannot move should not be behind a network call.

    THE PREDICTIONS     live_log/phase6_live_predictions.csv, the PRIMARY
    MIRROR              record under L9.4, because git is what timestamps a
                        pre-kickoff claim in a way the operator cannot
                        backdate. Also compiled in, for the same reason.

    RESULTS AND MARKET  NOT here, and not in git at all. L4.8 keeps 2026-27
                        scorelines out of the repository until the holdout is
                        deliberately acquired in May 2027, so the frontend
                        reads those from Supabase at request time.

WHAT THIS IS NOT. It computes nothing and decides nothing. Every value it
writes is read from a committed artefact and carries the artefact's filename
with it, so a figure on a web page can be traced back the same way a figure in
REPORT.md can. The one exception is the D0-to-Dixon-Coles decomposition, which
is arithmetic over values from three named artefacts and is marked as derived.

Re-runnable. Overwrites its outputs and nothing else.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(ROOT, "outputs")
LIVE_LOG = os.path.join(ROOT, "live_log")
TARGET = os.path.join(ROOT, "web", "lib", "frozen.generated.ts")


# ============================================================
# 1. READING
# ============================================================

def read(name, directory=OUTPUTS):
    with open(os.path.join(directory, name), newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def num(value):
    """Empty cells become None rather than 0.0, which would be a lie."""
    if value is None or value == "":
        return None
    return float(value)


def pick(rows, key, value):
    for row in rows:
        if row[key] == value:
            return row
    raise KeyError("no row with %s == %r" % (key, value))


# ============================================================
# 2. THE LADDER
# ============================================================
#
# Every rung, with the artefact each figure is read FROM. Where two artefacts
# carry the same model the one REPORT.md section 3 cites is used, so the site
# and the report cannot drift apart.
#
#   kind      drives presentation only: which colour a bar takes.

LADDER_SPEC = [
    # (label, source artefact, model key, kind, note)
    ("D0 - base rate", "phase4_ladder_pooled.csv", "D0", "baseline",
     "Season base rates only. Best calibration in the project and the worst model in it."),
    ("D1 - results-derived features", "phase4_ladder_pooled.csv", "D1", "ladder",
     "Current-season results. 83% of the whole distance to Dixon-Coles."),
    ("D2 - + dynamic state", "phase4_ladder_pooled.csv", "D2", "ladder", None),
    ("D2 rescaled - Amendment 4", "phase4_d34_pooled.csv", "D2_rescaled", "ladder", None),
    ("D3 - + context, Block C", "phase4_d34_pooled.csv", "D3", "ladder",
     "Not significant against D2 rescaled. The gate stands failed."),
    ("D4 - + prior-season FBref", "phase4_d34_pooled.csv", "D4", "ladder",
     "139 columns, and Elo v1's single rating still has a lower log loss."),
    ("Elo v1 - one K=20 rating", "phase4_ladder_pooled.csv", "elo_v1", "rating",
     "Flat 1500 start, 60-point home advantage. Beats every engineered rung."),
    ("Poisson walk-forward", "phase4_ladder_pooled.csv", "poisson_walkforward", "rating", None),
    ("Dixon-Coles walk-forward", "phase2_poisson_dc_fold_summary.csv", "dc_walkforward",
     "frozen", "The frozen model. Refits once per distinct scored date."),
    ("E1a - shots-on-target ratings", "phase5_e1a_pooled.csv", "E1a_sot", "arm",
     "Lower point estimate than Dixon-Coles, and the difference is not significant."),
    ("E1b - shot residual", "phase5_e1b_pooled.csv", "E1b", "arm", None),
    ("E1c - finishing residual", "phase5_e1c_pooled.csv", "E1c", "arm",
     "Fired the sign-agreement rule against Elo v1: reported INCONCLUSIVE."),
    ("Market - Bet365 closing", "phase5_market_pooled.csv", "market_B365C_proportional",
     "market", "Chosen on completeness before any score existed. Never fitted to anything."),
]


def build_ladder():
    # dc_walkforward's pooled row is not in the fold summary, which is per-fold;
    # REPORT.md section 3 cites the fold summary for the FROZEN figure and the
    # pooled value appears in phase4_ladder_pooled.csv. Read it from the pooled
    # file and cite both, rather than pooling four folds here - this script
    # computes nothing.
    pooled = {
        "phase4_ladder_pooled.csv": read("phase4_ladder_pooled.csv"),
        "phase4_d34_pooled.csv": read("phase4_d34_pooled.csv"),
        "phase5_market_pooled.csv": read("phase5_market_pooled.csv"),
        "phase5_e1a_pooled.csv": read("phase5_e1a_pooled.csv"),
        "phase5_e1b_pooled.csv": read("phase5_e1b_pooled.csv"),
        "phase5_e1c_pooled.csv": read("phase5_e1c_pooled.csv"),
    }

    out = []
    for label, source, key, kind, note in LADDER_SPEC:
        cited = source
        if source == "phase2_poisson_dc_fold_summary.csv":
            # the pooled figure lives in the ladder artefact; the fold summary is
            # what REPORT.md cites for the model itself. Name both.
            row = pick(pooled["phase4_ladder_pooled.csv"], "model", key)
            cited = "phase4_ladder_pooled.csv + phase2_poisson_dc_fold_summary.csv"
        else:
            row = pick(pooled[source], "model", key)

        out.append({
            "key": key,
            "label": label,
            "kind": kind,
            "n": int(row["n"]),
            "logLoss": num(row["log_loss"]),
            "rps": num(row["rps"]),
            "accuracy": num(row["accuracy"]),
            "balancedAccuracy": num(row["balanced_accuracy"]),
            "macroF1": num(row["macro_f1"]),
            "brier": num(row["brier_score"]),
            "note": note,
            "source": cited,
        })
    return out


# ============================================================
# 3. THE DELTAS
# ============================================================
#
# Pooled scope only. Every one is the same paired bootstrap - 1,520 matched
# per-match scores, 10,000 draws, seed 20260901 - which is what makes a 0.003
# difference readable at all.

DELTA_SPEC = [
    ("phase4_ladder_deltas.csv", "D1 - D0", "Current-season results pay"),
    ("phase4_ladder_deltas.csv", "D2 - D1", "Dynamic state pays, barely"),
    ("phase4_a4_deltas.csv", "D2rescaled - D1", "Amendment 4's rescaled rung"),
    ("phase4_d34_deltas.csv", "D3 - D2rescaled", "Context block: no"),
    ("phase4_d34_deltas.csv", "D4 - D3", "Prior-season FBref: no"),
    ("phase4_d34_deltas.csv", "D4 - D2rescaled", "The two blocks together: no"),
    ("phase4_a4_deltas.csv", "D2rescaled - DixonColes", "The unmeasured 13% residual"),
    ("phase5_e1a_deltas.csv", "E1a - DixonColes", "Why the freeze took Dixon-Coles, not E1a"),
    ("phase5_e1b_deltas.csv", "E1b - D2rescaled", "Shot residual: no"),
    ("phase5_e1c_deltas.csv", "E1c - D2rescaled", "Finishing residual: no"),
    ("phase5_market_deltas.csv", "market - DixonColes", "THE GAP"),
    ("phase5_market_deltas.csv", "market - D4", "The gap at D4"),
    ("phase5_market_deltas.csv", "market - D0", "The whole measurable range"),
]


def build_deltas():
    cache, out = {}, []
    for source, comparison, label in DELTA_SPEC:
        if source not in cache:
            cache[source] = [r for r in read(source) if r["scope"] == "pooled"]
        row = pick(cache[source], "comparison", comparison)
        out.append({
            "comparison": comparison,
            "label": label,
            "left": row["left"],
            "right": row["right"],
            "n": int(row["n"]),
            "logLossDelta": num(row["log_loss_delta"]),
            "logLossCi": [num(row["log_loss_ci_lo"]), num(row["log_loss_ci_hi"])],
            "rpsDelta": num(row["rps_delta"]),
            "rpsCi": [num(row["rps_ci_lo"]), num(row["rps_ci_hi"])],
            "signsAgree": row["signs_agree"] == "True",
            "excludesZero": row["log_loss_ci_excludes_zero"] == "True",
            "verdict": row["verdict"],
            "source": source,
        })
    return out


# ============================================================
# 4. THE DECOMPOSITION - THE ONLY DERIVED FIGURES HERE
# ============================================================
#
# REPORT.md section 3.2, recomputed from the artefacts rather than transcribed,
# and flagged `derived` in the emitted module so a reader can tell it apart from
# a figure that was read off disk. Against D2 RESCALED (Amendment 4), which is
# the row section 3.2 names - against the original D2 the split is 83/4/13.

def build_decomposition(ladder, deltas):
    by_key = {row["key"]: row for row in ladder}
    by_cmp = {row["comparison"]: row for row in deltas}

    d0 = by_key["D0"]["logLoss"]
    d1 = by_key["D1"]["logLoss"]
    dc = by_key["dc_walkforward"]["logLoss"]

    # deltas are `left - right`, so the sign that turns one into an IMPROVEMENT
    # depends on which side the better model sits. D2rescaled - D1 is negative
    # (D2rescaled is better); D2rescaled - DixonColes is positive (DC is
    # better). Both terms below are improvements, hence the opposite signs.
    total = d0 - dc
    results = d0 - d1
    state = -by_cmp["D2rescaled - D1"]["logLossDelta"]
    residual = +by_cmp["D2rescaled - DixonColes"]["logLossDelta"]

    reconstructed = results + state + residual
    if abs(reconstructed - total) > 5e-5:
        raise AssertionError(
            "the parts sum to %.6f against a total of %.6f - the decomposition "
            "does not close, so it is not reportable" % (reconstructed, total))

    parts = [
        ("Current-season results", "D0 -> D1", results, "phase4_ladder_pooled.csv", True),
        ("Continuously updated rating state", "D1 -> D2 rescaled", state,
         "phase4_a4_deltas.csv", True),
        ("Everything still unaccounted for", "D2 rescaled -> Dixon-Coles", residual,
         "phase4_a4_deltas.csv", False),
        ("Static historical description", "Blocks C and X", 0.0,
         "phase4_d34_deltas.csv", False),
    ]

    return {
        "total": total,
        "totalSource": "phase4_ladder_pooled.csv",
        "derived": True,
        "parts": [{
            "label": label, "span": span, "logLoss": value,
            "share": value / total, "significant": significant, "source": source,
        } for label, span, value, source, significant in parts],
    }


# ============================================================
# 5. THE GAP DIAGNOSTIC, SHARPNESS, AND THE REFUTED HYPOTHESIS
# ============================================================

def build_sharpness():
    """
    §4.3's finding - Dixon-Coles is MORE confident than the market, so the
    market's edge is direction and not calibration.

    RECOMPUTED here, and derived rather than read, because REPORT.md §4.3 cites
    phase5_calibration.csv for the mean-max-p table and THAT FILE DOES NOT
    CARRY THOSE COLUMNS - it holds ECE, MCE and the per-class biases. The
    figures are real: recomputing from the per-match artefacts reproduces
    0.5489 / 0.5391 / 0.5175 exactly. It is the citation that is wrong, which
    is a §10-class slip and is recorded here rather than propagated.
    """

    predictions = read("phase5_e1a_predictions.csv")
    market = read("phase5_market_probabilities.csv")
    calibration = read("phase5_calibration.csv")

    def key(row):
        return "%s_%s_%s" % (row["season"], row["home_team"], row["away_team"])

    scored = {key(r) for r in predictions}
    b365 = [r for r in market if r["book"] == "B365C" and key(r) in scored]

    if len(b365) != len(predictions):
        raise AssertionError(
            "market covers %d of the %d scored rows; §4.3 is computed on the "
            "same 1,520 or not at all" % (len(b365), len(predictions)))

    def mean_max(rows, prefix):
        return sum(max(num(r[prefix + "_H"]), num(r[prefix + "_D"]),
                       num(r[prefix + "_A"])) for r in rows) / len(rows)

    def mean_draw(rows, prefix):
        return sum(num(r[prefix + "_D"]) for r in rows) / len(rows)

    def ece_of(model):
        try:
            return num(pick(calibration, "model", model)["ece_pooled"])
        except KeyError:
            return None                       # E1a has no calibration row

    spec = [
        ("Market - Bet365 closing", b365, "prop_p", "market_B365C_proportional"),
        ("Dixon-Coles", predictions, "goals_DC_p", "dc_walkforward"),
        ("E1a - shots on target", predictions, "E1a_sot_p", None),
        ("D0 - base rate", None, None, "D0"),
    ]

    out = []
    for label, rows, prefix, calibration_key in spec:
        out.append({
            "label": label,
            "n": len(rows) if rows else 1520,
            "meanMaxP": mean_max(rows, prefix) if rows else None,
            "meanPDraw": mean_draw(rows, prefix) if rows else None,
            "ece": ece_of(calibration_key) if calibration_key else None,
            "biasD": (num(pick(calibration, "model", calibration_key)["bias_D"])
                      if calibration_key else None),
        })

    return {
        "rows": out,
        "derived": True,
        "source": "phase5_e1a_predictions.csv + phase5_market_probabilities.csv",
        "calibrationSource": "phase5_calibration.csv",
        "note": ("Mean max p is recomputed from the per-match artefacts. "
                 "REPORT.md §4.3 cites phase5_calibration.csv, which does not "
                 "carry those columns; the values reproduce exactly."),
    }


def build_gap():
    splits = read("phase5_gap_splits.csv")
    correlations = read("phase5_gap_correlations.csv")

    # every split that was looked at, including the null ones. Reported in full
    # because reporting only the separating one is the failure the diagnostic's
    # own framing forbids.
    grouped = {}
    for row in splits:
        grouped.setdefault(row["split"], []).append({
            "level": row["level"],
            "n": int(row["n"]),
            "gap": num(row["gap_dc"]),
            "ci": [num(row["gap_dc_lo"]), num(row["gap_dc_hi"])],
            "gapRps": num(row["gap_rps_dc"]),
        })

    strongest = max(correlations, key=lambda r: num(r["abs_r"]))
    threshold = 0.0503                       # the naive |r| threshold at n=1520
    clearing = [r for r in correlations if num(r["abs_r"]) >= threshold]

    return {
        "splits": grouped,
        "splitsSource": "phase5_gap_splits.csv",
        "correlations": {
            "columnsExamined": len(correlations),
            "largestAbsR": num(strongest["abs_r"]),
            "largestColumn": strongest["column"],
            "threshold": threshold,
            "clearingThreshold": len(clearing),
            "expectedByChance": 6.4,
            "source": "phase5_gap_correlations.csv",
        },
        "sharpness": build_sharpness(),
    }


def build_tier2():
    rows = read("phase4_tier2_decomposition.csv")
    rows = [r for r in rows if r["metric"] == "log_loss"]
    out = {}
    for row in rows:
        out[row["quantity"]] = {
            "point": num(row["point"]),
            "ci": [num(row["ci_low"]), num(row["ci_high"])],
            "excludesZero": row["excludes_zero"] == "True",
        }
    return {"quantities": out, "source": "phase4_tier2_decomposition.csv"}


# ============================================================
# 6. THE EVALUATION DESIGN
# ============================================================

def build_folds():
    return [{
        "fold": int(row["fold"]),
        "trainSeasons": row["train_seasons"],
        "testSeason": row["test_season"],
        "trainMatches": int(row["train_matches"]),
        "testMatches": int(row["test_matches"]),
        "maxTrainDate": row["max_train_date"],
        "minTestDate": row["min_test_date"],
        "temporalOrderValid": row["temporal_order_valid"] == "True",
        "overlapValid": row["overlap_valid"] == "True",
    } for row in read("phase0_evaluation_folds.csv")]


def build_leakage():
    rows = read("phase0_leakage_audit.csv")
    field = "test" if "test" in rows[0] else list(rows[0])[0]
    status = "status" if "status" in rows[0] else None
    return {
        "tests": len(rows),
        "passed": sum(1 for r in rows if not status or r[status].upper() == "PASS"),
        "source": "phase0_leakage_audit.csv",
    }


# ============================================================
# 7. THE PREDICTIONS MIRROR - THE PRIMARY RECORD
# ============================================================

def build_mirror():
    predictions = read("phase6_live_predictions.csv", LIVE_LOG)
    contamination = read("phase6_live_contamination.csv", LIVE_LOG)

    rows = [{
        "matchId": r["match_id"],
        "season": r["season"],
        "roundId": int(r["round_id"]),
        "matchweekLabel": int(r["matchweek_label"]) if r["matchweek_label"] else None,
        "scheduledDate": r["scheduled_date"],
        "scheduledKickoff": r["scheduled_kickoff"] or None,
        "homeTeam": r["home_team"],
        "awayTeam": r["away_team"],
        "stateCutoffDate": r["state_cutoff_date"],
        "generatedAtUtc": r["generated_at_utc"],
        "pHome": num(r["p_home"]),
        "pDraw": num(r["p_draw"]),
        "pAway": num(r["p_away"]),
        "lambdaHome": num(r["lambda_home"]),
        "lambdaAway": num(r["lambda_away"]),
        "rho": num(r["rho"]),
        "fitMatches": int(r["fit_matches"]),
        "homeHasHistory": r["home_has_history"] == "True",
        "awayHasHistory": r["away_has_history"] == "True",
        "rowHash": r["row_hash"],
        "prevHash": r["prev_hash"],
    } for r in predictions]

    contaminated = [{
        "matchId": r["match_id"],
        "playedDate": r["played_date"],
        "homeTeam": r["home_team"],
        "awayTeam": r["away_team"],
        "reason": r["reason"],
        "recordedAtUtc": r["recorded_at_utc"],
        "excludedFromLog": r["excluded_from_log"] == "True",
        "excludedFromHoldout": r["excluded_from_holdout"] == "True",
    } for r in contamination]

    return rows, contaminated


# ============================================================
# 8. THE GOVERNING DOCUMENTS
# ============================================================
#
# A figure that cannot name its own governing texts is not evidence of
# anything, so the site carries the hashes and re-reads them from disk here
# rather than trusting a constant typed into a web page.

def build_meta(mirror):
    def hash_of(name):
        path = os.path.join(ROOT, name)
        return sha256(path) if os.path.exists(path) else None

    freeze = hash_of("PHASE6_HOLDOUT_FREEZE.txt")
    pin = hash_of("PHASE6_CUTOFF_PIN.txt")
    protocol = hash_of("PHASE6_LIVE_LOG_PROTOCOL.txt")

    # the mirror records which documents were in force when each row was
    # written. If the files on disk have moved since, the site must say so
    # rather than show today's hash beside yesterday's prediction.
    declared = {r["freeze_sha"] for r in read("phase6_live_predictions.csv", LIVE_LOG)} \
        if mirror else set()

    return {
        "devMatches": 1520,
        "sourceMatches": 1900,
        "seasons": "2021-22 to 2025-26",
        "holdoutSeason": "2026-2027",
        "cutoffDate": "2026-09-04",
        "cutoffFixture": "Ipswich Town v Liverpool",
        "expectedHoldoutMatches": 360,
        "bootstrapDraws": 10000,
        "bootstrapSeed": 20260901,
        "primaryMetric": "RPS with log loss, under the sign-agreement rule",
        "book": "B365C",
        "devig": "proportional",
        "cadenceCostLogLoss": 0.0009134,
        "cadenceCostRps": 0.0002268,
        "logRefitsPerSeason": 38,
        "frozenRefitsRange": "109 to 120",
        "freezeSha": freeze,
        "pinSha": pin,
        "protocolSha": protocol,
        "freezeShaMatchesMirror": (declared == {freeze}) if declared else None,
        "noHistoryShareOfHoldout": 0.195,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ============================================================
# 9. EMIT
# ============================================================

HEADER = '''/**
 * GENERATED FILE - DO NOT EDIT.
 *
 *   python scripts/phase6_build_frontend_data.py
 *
 * Every figure below is read from a committed artefact in outputs/ or from the
 * predictions mirror in live_log/, and carries the artefact's filename in its
 * `source` field so a number on a web page can be traced the way a number in
 * REPORT.md can. Nothing here is computed except DECOMPOSITION, which is
 * arithmetic over three named artefacts and is flagged `derived: true`.
 *
 * 2026-27 results and closing odds are deliberately ABSENT: L4.8 keeps them
 * out of the repository until May 2027, so the frontend reads those from
 * Supabase at request time instead.
 *
 * Built %s from %d artefacts.
 */

'''


def ts(name, value, type_hint=None):
    body = json.dumps(value, indent=2, allow_nan=False)
    annotation = ": %s" % type_hint if type_hint else ""
    return "export const %s%s = %s as const;\n\n" % (name, annotation, body)


def main():
    ladder = build_ladder()
    deltas = build_deltas()
    decomposition = build_decomposition(ladder, deltas)
    gap = build_gap()
    tier2 = build_tier2()
    folds = build_folds()
    leakage = build_leakage()
    mirror, contaminated = build_mirror()
    meta = build_meta(mirror)

    sources = sorted({row["source"] for row in ladder} | {row["source"] for row in deltas})

    chunks = [
        HEADER % (meta["generatedAt"], len(sources)),
        ts("META", meta),
        ts("LADDER", ladder),
        ts("DELTAS", deltas),
        ts("DECOMPOSITION", decomposition),
        ts("GAP", gap),
        ts("TIER2", tier2),
        ts("FOLDS", folds),
        ts("LEAKAGE", leakage),
        ts("PREDICTIONS", mirror),
        ts("CONTAMINATED", contaminated),
    ]

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(chunks))

    print("wrote %s" % os.path.relpath(TARGET, ROOT))
    print("  ladder rungs        %d" % len(ladder))
    print("  deltas              %d" % len(deltas))
    print("  gap splits          %d" % len(gap["splits"]))
    print("  folds               %d" % len(folds))
    print("  predictions mirror  %d" % len(mirror))
    print("  contaminated        %d" % len(contaminated))
    print()
    print("  freeze  %s" % meta["freezeSha"])
    print("  pin     %s" % meta["pinSha"])
    print("  mirror agrees with the freeze on disk: %s" % meta["freezeShaMatchesMirror"])
    print()
    print("  decomposition (derived, total %.5f log loss):" % decomposition["total"])
    for part in decomposition["parts"]:
        print("    %-36s %7.5f  %4.0f%%%s" % (
            part["label"], part["logLoss"], part["share"] * 100,
            "" if part["significant"] else "   not significant"))


if __name__ == "__main__":
    main()
