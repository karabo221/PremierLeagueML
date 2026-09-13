import Link from "next/link";

import { FixtureCard } from "@/components/FixtureCard";
import { Crest } from "@/components/Crest";
import { displayName } from "@/lib/crests";
import { CONTAMINATED, LADDER, META } from "@/lib/frozen.generated";
import { dateLong, fixed, stampShort } from "@/lib/format";
import { readLiveLog, readPredictions } from "@/lib/supabase";

// Re-read every 10 minutes. That is what makes a new matchweek appear without
// anyone redeploying: the generator writes the round to the database, and the
// next revalidation picks it up. Results arrive at a matchweek's cadence, not a
// second's, so a tighter window would spend invocations to show the same rows.
export const revalidate = 600;

const frozenModel = LADDER.find((row) => row.kind === "frozen")!;
const market = LADDER.find((row) => row.kind === "market")!;

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

  const dates = currentFixtures.map((f) => f.scheduledDate).sort();
  const written = currentFixtures[0]?.generatedAtUtc;
  const cutoff = currentFixtures[0]?.stateCutoffDate;
  const fitMatches = currentFixtures[0]?.fitMatches;

  return (
    <>
      {/* ── hero ─────────────────────────────────────────────────────── */}
      <section className="hero">
        <div className="shell">
          <p className="eyebrow">
            2026-27 &middot; round {current} &middot; state cutoff {cutoff}
          </p>

          <h1 className="hero-h display">
            {currentFixtures.length} fixtures, priced before
            <br />
            <span className="hero-em">anyone had kicked a ball</span>
          </h1>

          <p className="prose hero-p">
            A frozen Dixon-Coles model, fitted on every completed match strictly
            before this round&apos;s first fixture, and written down where the
            timestamp cannot be moved afterwards. On four seasons of held-out
            development matches it reached{" "}
            <strong>{fixed(frozenModel.logLoss!)} log loss</strong> against a
            closing-market benchmark of {fixed(market.logLoss!)} &mdash; and{" "}
            <Link href="/evidence" className="inline-link">
              no convincing explanation was found for the difference
            </Link>
            .
          </p>

          <div className="hero-facts">
            <div className="hero-fact">
              <span className="eyebrow">Written</span>
              <span className="figure">{written ? stampShort(written) : "—"}</span>
            </div>
            <div className="hero-fact">
              <span className="eyebrow">First kickoff</span>
              <span className="figure">{dateLong(dates[0])}</span>
            </div>
            <div className="hero-fact">
              <span className="eyebrow">Fit window</span>
              <span className="figure">{fitMatches?.toLocaleString("en-GB")} matches</span>
            </div>
            <div className="hero-fact">
              <span className="eyebrow">Rounds logged</span>
              <span className="figure">{rounds.length}</span>
            </div>
          </div>
        </div>
      </section>

      <div className="shell stack stack-lg page-body">
        {/* ── the round ─────────────────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Round {current}</h2>
            <span className="pill" title={feed.detail}>
              <i className={`pill-dot${feed.source === "database" ? "" : " is-idle"}`} />
              {feed.source === "database" ? "live from the log" : "committed mirror"}
            </span>
          </div>
          <p className="sec-sub">{feed.detail}</p>

          <div className="fx-grid">
            {currentFixtures.map((f) => (
              <FixtureCard
                key={f.matchId}
                fixture={f}
                result={resultsByMatch.get(f.matchId)}
              />
            ))}
          </div>

          <p className="legend mono">
            <span><i className="sw is-home" />home win</span>
            <span><i className="sw is-draw" />draw</span>
            <span><i className="sw is-away" />away win</span>
            <span className="legend-note">
              &lambda; is the fitted goal rate each side was given; &rho; is the
              low-score dependence parameter, shared across the round.
            </span>
          </p>
        </section>

        {/* ── coverage, which no figure may appear without (L2.4) ──── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">What this log does not cover</h2>
            <Link href="/log" className="pill pill-link">
              Full integrity view
            </Link>
          </div>

          <div className="cov-grid">
            <div className="cov-stat">
              <span className="cov-n figure">{predictions.length}</span>
              <span className="cov-l">predictions written</span>
              <p className="cov-d">
                Each one appended once, before its matchweek&apos;s first kickoff,
                and never regenerated.
              </p>
            </div>

            <div className="cov-stat is-bad">
              <span className="cov-n figure">{CONTAMINATED.length}</span>
              <span className="cov-l">matches contaminated</span>
              <p className="cov-d">
                Played before the log began. Excluded from every figure here
                &mdash; and <strong>not</strong> excluded from the holdout, which
                is a date rule in a frozen document.
              </p>
            </div>

            <div className="cov-stat">
              <span className="cov-n figure">
                {live.coverage ? live.coverage.resultsCaptured : "—"}
              </span>
              <span className="cov-l">results captured</span>
              <p className="cov-d">
                {live.status.state === "connected"
                  ? "Read live from the dashboard's copy."
                  : live.status.detail}
              </p>
            </div>

            <div className="cov-stat">
              <span className="cov-n figure">~{META.expectedHoldoutMatches}</span>
              <span className="cov-l">holdout matches expected</span>
              <p className="cov-d">
                Every 2026-27 match from {META.cutoffDate} &mdash;{" "}
                {META.cutoffFixture} &mdash; onward.
              </p>
            </div>
          </div>

          <details className="drawer">
            <summary>
              The {CONTAMINATED.length} contaminated matches, named
            </summary>
            <div className="scroll-x">
              <table className="grid-table">
                <thead>
                  <tr>
                    <th>match</th>
                    <th>played</th>
                    <th>in the log</th>
                    <th>in the holdout</th>
                  </tr>
                </thead>
                <tbody>
                  {CONTAMINATED.map((c) => (
                    <tr key={c.matchId}>
                      <td>
                        <span className="cont-match">
                          <Crest team={c.homeTeam} size={18} />
                          {displayName(c.homeTeam)}
                          <span className="fx-v mono">v</span>
                          {displayName(c.awayTeam)}
                          <Crest team={c.awayTeam} size={18} />
                        </span>
                      </td>
                      <td className="mono">{c.playedDate}</td>
                      <td>
                        <span className="verdict is-null">EXCLUDED</span>
                      </td>
                      <td>
                        <span className="verdict is-sig">KEPT</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
                <caption>
                  live_log/phase6_live_contamination.csv &mdash; a gap is a
                  record, not a silence. No late row was written for any of
                  these: a prediction computed after the result exists is not
                  evidence, whatever flag it carries.
                </caption>
              </table>
            </div>
          </details>
        </section>

        {earlier.length > 0 && (
          <section className="stack stack-md">
            <h2 className="sec-h">Earlier rounds</h2>
            <div className="fx-grid">
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

        <div className="informal">
          <span className="informal-t">Informal &middot; not a result</span>
          <p>
            No figure on this site is the official 2026-27 measurement, and none
            may be reported as one. The official figure is{" "}
            <span className="mono">phase6_score_holdout.py</span>, run once after
            the final fixture behind two explicit flags. This log refits{" "}
            {META.logRefitsPerSeason} times a season where the frozen model refits{" "}
            {META.frozenRefitsRange} &mdash; a strictly smaller information set,
            pre-declared at a cost of {fixed(META.cadenceCostLogLoss, 7)} log loss
            &mdash; so the two are not comparable even at season end.
          </p>
        </div>
      </div>
    </>
  );
}
