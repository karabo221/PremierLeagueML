import Link from "next/link";

import { FixtureCard } from "@/components/FixtureCard";
import { Scorecard } from "@/components/Scorecard";
import { Crest } from "@/components/Crest";
import { scorecard } from "@/lib/accuracy";
import { displayName } from "@/lib/crests";
import { CONTAMINATED, LADDER, META } from "@/lib/frozen.generated";
import { readLiveLog, readPredictions } from "@/lib/supabase";

// Re-read every 10 minutes. That is what makes a new matchweek appear without
// anyone redeploying: the generator writes the round to the database, and the
// next revalidation picks it up. Results arrive at a matchweek's cadence, not a
// second's, so a tighter window would spend invocations to show the same rows.
export const revalidate = 600;

const frozenModel = LADDER.find((row) => row.kind === "frozen")!;
const market = LADDER.find((row) => row.kind === "market")!;

/** How far along the road from a blind guess to the market the model got. */
const blindGuess = LADDER.find((row) => row.kind === "baseline")!;
const roadCovered =
  (blindGuess.logLoss! - frozenModel.logLoss!) /
  (blindGuess.logLoss! - market.logLoss!);

export default async function FixturesPage() {
  const [live, feed] = await Promise.all([readLiveLog(), readPredictions()]);
  const resultsByMatch = new Map(live.results.map((r) => [r.matchId, r]));
  const predictions = feed.rows;

  // Rounds newest first. The round boundary is derived, never labelled - L1.3 -
  // so the id is what groups fixtures, not a matchweek number.
  const rounds = [...new Set(predictions.map((p) => p.roundId))].sort((a, b) => b - a);
  const current = rounds[0];
  const currentFixtures = predictions.filter((p) => p.roundId === current);
  const earlier = predictions.filter((p) => p.roundId !== current);

  const card = scorecard(predictions, live.results);
  const cutoff = currentFixtures[0]?.stateCutoffDate;

  return (
    <>
      {/* ── the welcome ──────────────────────────────────────────────── */}
      <section className="hero">
        <div className="shell">
          <p className="eyebrow">
            Round {current} &middot; using only matches played up to {cutoff}
          </p>

          <h1 className="hero-h display">
            This week&apos;s predictions <span className="hero-em">look like this</span>
          </h1>

          <p className="prose hero-p">
            Welcome. Every week, before a ball is kicked, a computer reads the
            season so far and writes down what it thinks will happen &mdash; as{" "}
            <strong>chances, not scorelines</strong>. It has never seen a result it
            is predicting, and it has never watched a football match.{" "}
            <Link href="/how-it-works" className="inline-link">
              Here is how it does that
            </Link>
            , in plain English.
          </p>

          {/*
            The raw status detail names environment variables, which is the right
            level of explanation on the record page and the wrong one here. A
            reader on the front page needs to know the boxes are empty and that
            nothing is broken; whoever needs the variable names is already on
            /log looking for them.
          */}
          <Scorecard
            data={card}
            rounds={rounds.length}
            unavailable={
              live.status.state === "error" || live.status.state === "unconfigured"
                ? "The results are not reaching this page at the moment, so there is " +
                  "nothing to score yet. The predictions below are unaffected — they " +
                  "come from the committed copy, which is the real record either way."
                : undefined
            }
          />
        </div>
      </section>

      <div className="shell stack stack-lg page-body">
        {/* ── the round ─────────────────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Round {current}</h2>
            <span className="pill" title={feed.detail}>
              <i className={`pill-dot${feed.source === "database" ? "" : " is-idle"}`} />
              {feed.source === "database" ? "live from the log" : "committed copy"}
            </span>
          </div>

          <div className="fx-list">
            {currentFixtures.map((f) => (
              <FixtureCard
                key={f.matchId}
                fixture={f}
                result={resultsByMatch.get(f.matchId)}
              />
            ))}
          </div>

          <p className="legend">
            <span>
              <i className="sw is-home" />
              home win
            </span>
            <span>
              <i className="sw is-draw" />
              draw
            </span>
            <span>
              <i className="sw is-away" />
              away win
            </span>
            <span className="legend-note">
              The highlighted box is the outcome the model rates highest. &ldquo;Both
              to score&rdquo; is worked out from the goals it expects each side to
              score, not guessed from the percentages.
            </span>
          </p>
        </section>

        {earlier.length > 0 && (
          <section className="stack stack-md">
            <div className="sec-hd">
              <h2 className="sec-h">Earlier rounds</h2>
            </div>
            <div className="fx-list">
              {earlier.map((f) => (
                <FixtureCard
                  key={f.matchId}
                  fixture={f}
                  result={resultsByMatch.get(f.matchId)}
                />
              ))}
            </div>
          </section>
        )}

        {/* ── coverage, which no figure may appear without (L2.4) ──── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">What this record does not cover</h2>
            <Link href="/log" className="pill pill-link">
              The full record
            </Link>
          </div>

          <div className="cov-grid">
            <div className="cov-stat">
              <span className="cov-n">{predictions.length}</span>
              <span className="cov-l">predictions written</span>
              <p className="cov-d">
                Each one appended once, before its matchweek&apos;s first kickoff,
                and never regenerated.
              </p>
            </div>

            <div className="cov-stat is-bad">
              <span className="cov-n">{CONTAMINATED.length}</span>
              <span className="cov-l">matches left out</span>
              <p className="cov-d">
                Played before the log began, so a prediction for one is not a
                pre-kickoff claim. Excluded from every figure here &mdash; and{" "}
                <strong>not</strong> excluded from the season&apos;s real
                measurement, which is a date rule fixed in advance.
              </p>
            </div>

            <div className="cov-stat">
              <span className="cov-n">
                {live.coverage ? live.coverage.resultsCaptured : "—"}
              </span>
              <span className="cov-l">results collected</span>
              <p className="cov-d">
                {live.status.state === "connected"
                  ? "Read live from the results database."
                  : live.status.detail}
              </p>
            </div>

            <div className="cov-stat">
              <span className="cov-n">~{META.expectedHoldoutMatches}</span>
              <span className="cov-l">matches to come</span>
              <p className="cov-d">
                Every 2026-27 match from {META.cutoffDate} &mdash;{" "}
                {META.cutoffFixture} &mdash; onward.
              </p>
            </div>
          </div>

          <details className="drawer">
            <summary>
              The {CONTAMINATED.length} matches that were left out, named
            </summary>
            <div className="scroll-x">
              <table className="grid-table">
                <thead>
                  <tr>
                    <th>match</th>
                    <th>played</th>
                    <th>in this record</th>
                    <th>in the season&apos;s answer</th>
                  </tr>
                </thead>
                <tbody>
                  {CONTAMINATED.map((c) => (
                    <tr key={c.matchId}>
                      <td>
                        <span className="cont-match">
                          <Crest team={c.homeTeam} size={20} />
                          {displayName(c.homeTeam)}
                          <span className="fx-v">v</span>
                          {displayName(c.awayTeam)}
                          <Crest team={c.awayTeam} size={20} />
                        </span>
                      </td>
                      <td className="mono">{c.playedDate}</td>
                      <td>
                        <span className="verdict is-null">LEFT OUT</span>
                      </td>
                      <td>
                        <span className="verdict is-sig">KEPT</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
                <caption>
                  A gap is a record, not a silence. No late row was written for any
                  of these: a prediction worked out after the result already exists
                  is not evidence, whatever label it carries.
                </caption>
              </table>
            </div>
          </details>
        </section>

        <div className="informal">
          <span className="informal-t">Running tally &mdash; not the final answer</span>
          <p>
            The percentages on this page are a scorecard we keep for interest. The
            season&apos;s real measurement happens <strong>once</strong>, after the
            final fixture, against rules written down and sealed in September 2026.
            Nothing seen here may change the model, and if it did, the season would
            stop counting.
          </p>
          <p>
            For context on what &ldquo;good&rdquo; looks like: on five past seasons
            this model covered about{" "}
            <strong>{Math.round(roadCovered * 100)}% of the distance</strong> from a
            blind guess to a professional bookmaker.{" "}
            <Link href="/evidence" className="inline-link">
              The working is here
            </Link>
            .
          </p>
        </div>
      </div>
    </>
  );
}
