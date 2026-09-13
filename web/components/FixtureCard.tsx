import { Crest } from "@/components/Crest";
import { displayName } from "@/lib/crests";
import { dayShort, dateShort, lean, pct, pctShort } from "@/lib/format";
import type { ResultRow } from "@/lib/supabase";
import type { Prediction } from "@/lib/types";

// The fixture shape lives in lib/types.ts, shared with the database read.
export type Fixture = Prediction;

const OUTCOME = { H: "home", D: "draw", A: "away" } as const;

/**
 * One match. The probability triad is the card's whole point, so it gets the
 * largest type on it; lambda and rho sit underneath because they are what the
 * probabilities were computed FROM and a reader who wants them wants them
 * exactly, not rounded.
 *
 * A captured result, when one exists, is shown WITH the prediction rather than
 * replacing it. The prediction is the evidence; hiding it once the answer is
 * known is how a log stops being one.
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

  // Did the model's leading outcome happen? Stated plainly, never scored -
  // L5.1 keeps the official figure in phase6_score_holdout.py.
  const called = result ? result.result === top : null;

  return (
    <article className={`fx${result ? " is-settled" : ""}`}>
      <header className="fx-top">
        <span className="mono">
          {dayShort(f.scheduledDate)} {dateShort(f.scheduledDate)}
          {f.scheduledKickoff ? ` · ${f.scheduledKickoff}` : ""}
        </span>
        {result ? (
          <span className={`fx-called${called ? " is-hit" : " is-miss"}`}>
            {called ? "called" : "missed"}
          </span>
        ) : (
          <span className="mono fx-round">R{f.roundId}</span>
        )}
      </header>

      <div className="fx-teams">
        <div className="fx-team">
          <Crest team={f.homeTeam} />
          <span className="fx-nm">{displayName(f.homeTeam)}</span>
        </div>

        {result ? (
          <span className="fx-score figure">
            {result.homeGoals}&ndash;{result.awayGoals}
          </span>
        ) : (
          <span className="fx-v mono">v</span>
        )}

        <div className="fx-team is-away">
          <span className="fx-nm">{displayName(f.awayTeam)}</span>
          <Crest team={f.awayTeam} />
        </div>
      </div>

      <div className="fx-probs">
        {(
          [
            ["H", f.pHome, "Home"],
            ["D", f.pDraw, "Draw"],
            ["A", f.pAway, "Away"],
          ] as const
        ).map(([key, p, label]) => (
          <div
            key={key}
            className={`fx-prob is-${OUTCOME[key]}${top === key ? " is-top" : ""}${
              result && result.result === key ? " is-actual" : ""
            }`}
          >
            <span className="fx-p figure">{pctShort(p)}</span>
            <span className="fx-l mono">{label}</span>
          </div>
        ))}
      </div>

      <div
        className="fx-bar"
        role="img"
        aria-label={
          `Home ${pct(f.pHome)}, draw ${pct(f.pDraw)}, away ${pct(f.pAway)}`
        }
      >
        <i className="is-home" style={{ width: `${f.pHome * 100}%` }} />
        <i className="is-draw" style={{ width: `${f.pDraw * 100}%` }} />
        <i className="is-away" style={{ width: `${f.pAway * 100}%` }} />
      </div>

      <footer className="fx-foot mono">
        <span>
          &lambda; {f.lambdaHome.toFixed(3)} &mdash; {f.lambdaAway.toFixed(3)}
        </span>
        {coldStart ? (
          <span className="fx-cold" title="No history in the dataset for one side">
            cold start
          </span>
        ) : (
          <span>&rho; {f.rho.toFixed(3)}</span>
        )}
      </footer>
    </article>
  );
}
