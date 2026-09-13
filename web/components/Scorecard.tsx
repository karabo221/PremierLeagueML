import type { Scorecard as ScorecardData, Tally } from "@/lib/accuracy";

/**
 * The running scorecard, in the place the old fact strip stood.
 *
 * Every box carries three things and needs all three: the rate, the counts it
 * came from, and what the model EXPECTED to get right. The last one is what
 * makes the first readable - 64% is a good week if the model expected 50% and
 * a poor one if it expected 80% - and it is the reason this is a scorecard
 * rather than a boast.
 *
 * The empty state is deliberate and is not a zero. Supabase has never been
 * written, so the honest render is "nothing settled yet", not four boxes
 * reading 0% which would say the model has been wrong every time.
 */

function rateLabel(rate: number | null): string {
  return rate === null ? "—" : `${Math.round(rate * 100)}%`;
}

function TallyBox({ tally }: { tally: Tally }) {
  const { label, called, right, expected, rate } = tally;

  if (called === 0) {
    return (
      <div className="sc-col is-quiet">
        <span className="sc-k">{label}</span>
        <span className="sc-n">—</span>
        <p className="sc-d">Not called yet this season.</p>
      </div>
    );
  }

  // Behind its own expectation is worth marking, but only as a quiet tint:
  // on a handful of matches it is noise, and the caption says so.
  const behind = right < expected;

  return (
    <div className="sc-col">
      <span className="sc-k">{label}</span>
      <span className="sc-n">{rateLabel(rate)}</span>
      <p className="sc-d">
        {tally.key === "btts" ? (
          <>
            We said both would score in <strong>{called}</strong>{" "}
            {called === 1 ? "match" : "matches"} and both did in{" "}
            <strong>{right}</strong>.
          </>
        ) : (
          <>
            We picked {tally.key === "draw" ? "the draw" : `the ${tally.key} side`} in{" "}
            <strong>{called}</strong> {called === 1 ? "match" : "matches"} and were right
            in <strong>{right}</strong>.
          </>
        )}
      </p>
      <span className="sc-exp">
        the model expected about {expected.toFixed(1)}
      </span>
      <span className="sc-bar" aria-hidden="true">
        <i
          className={behind ? "is-behind" : undefined}
          style={{ width: `${Math.round((rate ?? 0) * 100)}%` }}
        />
      </span>
    </div>
  );
}

export function Scorecard({
  data,
  rounds,
  unavailable,
}: {
  data: ScorecardData;
  rounds: number;
  /** Why there is nothing to show, when the database could not be read. */
  unavailable?: string;
}) {
  return (
    <section className="sc" aria-label="How the predictions have done so far">
      <div className="sc-hd">
        <h2 className="sc-h">How we&apos;re doing so far</h2>
        <span className="sc-count">
          {data.settled === 0
            ? "no matches settled yet"
            : `${data.settled} ${data.settled === 1 ? "match" : "matches"} settled · 2026-27 season to date`}
        </span>
      </div>

      <div className="sc-cols">
        <div className="sc-col is-rounds">
          <span className="sc-k">Rounds on record</span>
          <span className="sc-n">{rounds}</span>
          <p className="sc-d">
            Each one written and published <strong>before</strong> its first kickoff.
          </p>
        </div>

        {data.settled === 0 ? (
          <div className="sc-col is-wide is-quiet">
            <span className="sc-k">Home wins · draws · away wins · both to score</span>
            <span className="sc-n">—</span>
            <p className="sc-d">
              {unavailable ??
                "No result has been captured yet, so there is nothing to score. " +
                  "The boxes fill in on their own once the first matchweek is played " +
                  "and its scorelines are collected."}
            </p>
          </div>
        ) : (
          data.tallies.map((tally) => <TallyBox key={tally.key} tally={tally} />)
        )}
      </div>

      <p className="sc-ft">
        Every box carries what the model <strong>expected</strong> to get right as well
        as what it did, because a percentage on its own cannot tell you whether it is
        good. These are a running tally kept for interest &mdash; not the season&apos;s
        result, and not something the model is allowed to learn from.
      </p>
    </section>
  );
}
