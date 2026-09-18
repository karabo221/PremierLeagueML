import { Crest } from "@/components/Crest";
import { goalsPhrase, pBothScore, BTTS_CALL } from "@/lib/accuracy";
import { displayName } from "@/lib/crests";
import { dayShort, dateShort, lean, pct, pctShort } from "@/lib/format";
import type { ResultRow } from "@/lib/supabase";
import type { Prediction } from "@/lib/types";

export type Fixture = Prediction;

const OUTCOME = { H: "home", D: "draw", A: "away" } as const;

/**
 * One match, set as a ruled row rather than a card - the way a results column
 * has been set since long before anyone had a grid to put it in.
 *
 * The probability triad is still the row's whole point and keeps the largest
 * type on it. What changed is underneath: lambda and rho used to sit there as
 * two Greek letters on every row, which told a reader who already knew them
 * nothing new and a reader who did not, nothing at all. They are now a
 * sentence - "expects about 2.0 goals to 0.8" - plus the chance both sides
 * score, which is computed from those same three numbers and is the thing
 * people actually ask about. The exact figures are still in the log, which is
 * where a reader who wants three decimal places should be reading them.
 *
 * A captured result is shown WITH the prediction rather than replacing it. The
 * prediction is the evidence; hiding it once the answer is known is how a log
 * stops being one.
 */
export function FixtureCard({
  fixture,
  result,
}: {
  fixture: Fixture;
  result?: ResultRow;
}) {
  const f = fixture;
  const top = lean(f.pHome, f.pDraw, f.pAway);
  const coldStart = !f.homeHasHistory || !f.awayHasHistory;
  const pBoth = pBothScore(f.lambdaHome, f.lambdaAway, f.rho);

  // Did the model's leading outcome happen? Stated plainly, never scored -
  // L5.1 keeps the official figure in phase6_score_holdout.py.
  const called = result ? result.result === top : null;
  const bothScored = result ? result.homeGoals > 0 && result.awayGoals > 0 : null;
  const saidBoth = pBoth > BTTS_CALL;

  const leaning =
    top === "H"
      ? displayName(f.homeTeam)
      : top === "A"
        ? displayName(f.awayTeam)
        : "a draw";

  return (
    <article className={`fx${result ? " is-settled" : ""}`}>
      <div className="fx-when">
        <span>{dayShort(f.scheduledDate)}</span>
        <span>{f.scheduledKickoff ?? dateShort(f.scheduledDate)}</span>
        {f.writtenPreKickoff ? null : (
          <span
            className="fx-cold"
            title="Written after this match kicked off, from only the matches played before its round. Kept out of the pre-kickoff scorecard."
          >
            after kickoff
          </span>
        )}
      </div>

      <div className="fx-mid">
        <div className="fx-tm">
          <Crest team={f.homeTeam} size={26} />
          <span className="fx-nm">{displayName(f.homeTeam)}</span>
          {result ? (
            <span className="fx-score">
              {result.homeGoals}&ndash;{result.awayGoals}
            </span>
          ) : (
            <span className="fx-v">v</span>
          )}
          <Crest team={f.awayTeam} size={26} />
          <span className="fx-nm">{displayName(f.awayTeam)}</span>
        </div>

        <p className="fx-sub">
          {result ? (
            <>
              <span className={`fx-tick${called ? " is-hit" : " is-miss"}`}>
                {called ? "Called it" : "Missed"}
              </span>{" "}
              &middot; we said {leaning}
              {saidBoth ? (
                <>
                  , and said both would score &mdash;{" "}
                  {bothScored ? "both did" : "they did not"}
                </>
              ) : null}
              .
            </>
          ) : (
            <>
              {top === "D" ? (
                <>
                  Model leans towards <strong>a draw</strong>.
                </>
              ) : (
                <>
                  Model leans <strong>{leaning}</strong>.
                </>
              )}{" "}
              Expects {goalsPhrase(f.lambdaHome, f.lambdaAway)}.
              <span className="fx-btts"> &middot; both to score {pctShort(pBoth)}</span>
              {coldStart ? (
                <span className="fx-cold" title="No history in the dataset for one side">
                  {" "}
                  &middot; one side is new to the data
                </span>
              ) : null}
            </>
          )}
        </p>
      </div>

      <div className="fx-odds">
        {(
          [
            ["H", f.pHome, displayName(f.homeTeam)],
            ["D", f.pDraw, "Draw"],
            ["A", f.pAway, displayName(f.awayTeam)],
          ] as const
        ).map(([key, p, label]) => (
          <div
            key={key}
            className={`fx-prob is-${OUTCOME[key]}${top === key ? " is-top" : ""}${
              result && result.result === key ? " is-actual" : ""
            }`}
          >
            <span className="fx-p">{pctShort(p)}</span>
            <span className="fx-l">{label}</span>
          </div>
        ))}
      </div>

      <span className="fx-sr">
        Home {pct(f.pHome)}, draw {pct(f.pDraw)}, away {pct(f.pAway)}.
      </span>
    </article>
  );
}
