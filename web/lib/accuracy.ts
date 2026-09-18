/**
 * THE RUNNING SCORECARD.
 *
 * This is the only place in web/ that computes a figure from the live log
 * rather than reading one. Two things follow from that, and both are rules
 * rather than preferences:
 *
 *   1. EVERY NUMBER HERE IS INFORMAL (L5.1). The official 2026-27 result is
 *      phase6_score_holdout.py, run once after the final fixture. A scorecard
 *      is exactly the kind of instrument that tempts a change to the model,
 *      and any such change marks the holdout compromised - so the board that
 *      renders this carries the informal notice, and nothing here feeds back.
 *
 *   2. AN EMPTY LOG RENDERS AS ABSENCE, NEVER AS ZERO. Supabase has never been
 *      written as of this writing; `settled` is 0 and every rate is null. A
 *      null renders as "nothing settled yet". A 0% would read as a measurement
 *      of a model that had been wrong every time, which is the opposite of
 *      what an empty database means.
 *
 * A tally carries `expected` beside `right` on purpose. A bare hit rate cannot
 * be read: 64% is good if the model expected 50% and poor if it expected 80%.
 * `expected` is just the model's own probabilities for the matches it called,
 * added up, so the pair says whether the model is calibrated as well as whether
 * it is lucky.
 */

import type { ResultRow } from "@/lib/supabase";
import type { Prediction } from "@/lib/types";
import { lean } from "@/lib/format";

/**
 * The chance that BOTH sides score, under the same Dixon-Coles distribution
 * the probabilities on the card came from.
 *
 * The model gives three numbers per match: the goals it expects from the home
 * side (lambda), the goals it expects from the away side (mu), and the
 * low-score dependence parameter (rho) that adjusts the four cells 0-0, 0-1,
 * 1-0 and 1-1.
 *
 *     P(both score) = 1 - P(home blanks) - P(away blanks) + P(0-0)
 *
 * The tau correction leaves the MARGINALS untouched - summing the corrected
 * 0-0 and 0-1 cells gives back exp(-lambda) * exp(-mu) * (1 + mu), the plain
 * Poisson value - so P(home blanks) is exactly exp(-lambda) and P(away blanks)
 * exactly exp(-mu). Only the joint cell needs the correction:
 *
 *     P(0-0) = (1 - lambda * mu * rho) * exp(-(lambda + mu))
 *
 * which is why this is a closed form and not a summed score matrix.
 *
 * Worth stating because it is the trap the obvious shortcut walks into: two
 * sides on ONE expected goal each comes out at 40%, not a coin toss. A rule of
 * "call it when both are near 1.0" would have us calling yes on matches the
 * model actually leans against.
 */
export function pBothScore(
  lambdaHome: number,
  lambdaAway: number,
  rho: number
): number {
  const homeBlanks = Math.exp(-lambdaHome);
  const awayBlanks = Math.exp(-lambdaAway);
  const nilNil =
    (1 - lambdaHome * lambdaAway * rho) * Math.exp(-(lambdaHome + lambdaAway));
  return 1 - homeBlanks - awayBlanks + nilNil;
}

/** Above this, the card says "both to score"; at or below it, it does not. */
export const BTTS_CALL = 0.5;

export type TallyKey = "home" | "draw" | "away" | "btts";

export interface Tally {
  key: TallyKey;
  /** What the box is called on the page. */
  label: string;
  /** The sentence under the figure, written where the counts are known. */
  called: number;
  right: number;
  /** The model's own probabilities over the called matches, summed. */
  expected: number;
  /** right / called, or null when nothing has been called yet. */
  rate: number | null;
}

export interface Scorecard {
  /** Matches with both a prediction and a captured result, contamination out. */
  settled: number;
  tallies: Tally[];
}

const LABEL: Record<TallyKey, string> = {
  home: "Home wins",
  draw: "Draws",
  away: "Away wins",
  btts: "Both teams to score",
};

/**
 * Join the predictions to the captured results and count.
 *
 * TWO TALLIES, NEVER ONE (protocol L10.2 C). "pre-kickoff" is the tally the
 * site has always shown: rows written before their own kickoff, contaminated
 * results out. "late" counts only rows written AFTER kickoff - the same numbers
 * a timely run would have produced, since each was fitted on matches before its
 * round and nothing later, but without the timestamp proof. It answers "how
 * would it have done", and it is never added into the pre-kickoff figure.
 *
 * A result with no matching prediction is skipped rather than counted as a miss.
 */
export function scorecard(
  predictions: readonly Prediction[],
  results: readonly ResultRow[],
  which: "pre-kickoff" | "late" = "pre-kickoff"
): Scorecard {
  const byMatch = new Map(predictions.map((p) => [p.matchId, p]));

  const counts: Record<TallyKey, { called: number; right: number; expected: number }> = {
    home: { called: 0, right: 0, expected: 0 },
    draw: { called: 0, right: 0, expected: 0 },
    away: { called: 0, right: 0, expected: 0 },
    btts: { called: 0, right: 0, expected: 0 },
  };

  let settled = 0;

  for (const result of results) {
    const p = byMatch.get(result.matchId);
    if (!p) continue;
    const late = !p.writtenPreKickoff || result.contaminated;
    if (which === "late" ? !late : late) continue;

    settled += 1;

    // The outcome the model rated highest is the one it "called". Where it
    // called home, the probability it gave home is what it expected to score.
    const call = lean(p.pHome, p.pDraw, p.pAway);
    const outcome: Record<"H" | "D" | "A", { key: TallyKey; p: number }> = {
      H: { key: "home", p: p.pHome },
      D: { key: "draw", p: p.pDraw },
      A: { key: "away", p: p.pAway },
    };
    const { key, p: pCalled } = outcome[call];
    counts[key].called += 1;
    counts[key].expected += pCalled;
    if (result.result === call) counts[key].right += 1;

    // Both to score is a separate claim on the same match, so it is counted
    // independently of which side the model leaned to.
    const pBoth = pBothScore(p.lambdaHome, p.lambdaAway, p.rho);
    if (pBoth > BTTS_CALL) {
      counts.btts.called += 1;
      counts.btts.expected += pBoth;
      if (result.homeGoals > 0 && result.awayGoals > 0) counts.btts.right += 1;
    }
  }

  const order: TallyKey[] = ["home", "draw", "away", "btts"];

  return {
    settled,
    tallies: order.map((key) => {
      const c = counts[key];
      return {
        key,
        label: LABEL[key],
        called: c.called,
        right: c.right,
        expected: c.expected,
        rate: c.called === 0 ? null : c.right / c.called,
      };
    }),
  };
}

/**
 * How a fixture card says what the goal rates mean, without the Greek.
 * "about 2.0 goals to 0.8" - one decimal, because the third is noise to a
 * reader and the exact figures stay in the log.
 */
export const goalsPhrase = (lambdaHome: number, lambdaAway: number): string =>
  `about ${lambdaHome.toFixed(1)} goals to ${lambdaAway.toFixed(1)}`;
