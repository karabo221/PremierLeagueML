import type { Metadata } from "next";
import Link from "next/link";

import { Crest } from "@/components/Crest";
import { displayName, MISSING_CRESTS } from "@/lib/crests";
import { CONTAMINATED, META, PREDICTIONS } from "@/lib/frozen.generated";
import { dateShort, fixed, shortHash, stampShort } from "@/lib/format";
import { readLiveLog, type LiveStatus } from "@/lib/supabase";

export const revalidate = 600;

export const metadata: Metadata = {
  title: "The log",
  description:
    "The integrity view of the live prediction log: the row-hash chain, " +
    "coverage against the holdout, the contaminated matchweek, and where each " +
    "half of the record actually lives.",
};

function statusPill(status: LiveStatus) {
  const map = {
    connected: { dot: "", label: "connected" },
    empty: { dot: " is-idle", label: "connected · no results yet" },
    unconfigured: { dot: " is-idle", label: "not configured" },
    error: { dot: " is-bad", label: "unreachable" },
  } as const;
  const { dot, label } = map[status.state];
  return (
    <span className="pill">
      <i className={`pill-dot${dot}`} />
      {label}
    </span>
  );
}

export default async function LogPage() {
  const live = await readLiveLog();

  // L4.2: prev_hash of the first row is sixty-four zeroes, and each row's
  // prev_hash is the previous row's row_hash. Checked here as a display claim
  // only - the authoritative verifier is phase6_generate_predictions.py
  // --verify, which recomputes every hash from the row's own fields.
  const genesis = "0".repeat(64);
  const chainLinks = PREDICTIONS.every((row, index) =>
    index === 0 ? row.prevHash === genesis : row.prevHash === PREDICTIONS[index - 1].rowHash
  );

  const mirrorCount = PREDICTIONS.length;
  const dbCount = live.predictionsInDatabase;
  const mirrorsAgree = dbCount === null ? null : dbCount === mirrorCount;

  return (
    <>
      <section className="hero">
        <div className="shell">
          <p className="eyebrow">
            Governed by PHASE6_LIVE_LOG_PROTOCOL.txt &middot; {shortHash(META.protocolSha)}
          </p>
          <h1 className="hero-h display">
            The record, and{" "}
            <span className="hero-em">what makes it hard to fake</span>
          </h1>
          <p className="prose hero-p">
            A season-end batch run asserts that the freeze held. A pre-kickoff
            timestamped log <strong>proves what was said before the result existed</strong>.
            The second is the stronger evidence, because it does not depend on
            trusting the operator&apos;s account of the order in which things
            happened. This page is how you check it.
          </p>
        </div>
      </section>

      <div className="shell stack stack-lg page-body">
        {/* ── where each half lives ─────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Two records, on purpose</h2>
            {statusPill(live.status)}
          </div>

          <div className="split-2">
            <div className="panel">
              <div className="panel-hd">
                <span className="eyebrow">Primary &middot; git</span>
                <span className="verdict is-sig">AUTHORITATIVE</span>
              </div>
              <div className="panel-bd stack stack-sm">
                <p className="prose">
                  <span className="mono">live_log/phase6_live_predictions.csv</span> and
                  the contamination register. Committed, and committing them is the
                  entire point: git timestamps &ldquo;this was said before kickoff&rdquo;
                  in a way the operator cannot backdate. These carry no scoreline.
                </p>
                <dl className="kv">
                  <dt>predictions</dt>
                  <dd className="figure">{mirrorCount}</dd>
                  <dt>contaminated</dt>
                  <dd className="figure">{CONTAMINATED.length}</dd>
                  <dt>chain links</dt>
                  <dd>
                    <span className={`verdict ${chainLinks ? "is-sig" : "is-null"}`}>
                      {chainLinks ? "CONTIGUOUS" : "BROKEN"}
                    </span>
                  </dd>
                </dl>
              </div>
            </div>

            <div className="panel">
              <div className="panel-hd">
                <span className="eyebrow">Dashboard&apos;s copy &middot; Supabase</span>
                <span className="verdict is-null">DERIVED</span>
              </div>
              <div className="panel-bd stack stack-sm">
                <p className="prose">
                  Results and closing odds live here <strong>only</strong>. The 2026-27
                  scorelines stay out of the repository until the holdout is deliberately
                  acquired in May 2027, so this page reads them over the network. If this
                  database were lost, the evidence would not be.
                </p>
                <dl className="kv">
                  <dt>results captured</dt>
                  <dd className="figure">
                    {live.coverage ? live.coverage.resultsCaptured : "—"}
                  </dd>
                  <dt>predictions mirrored</dt>
                  <dd className="figure">
                    {dbCount === null ? "—" : `${dbCount} of ${mirrorCount}`}
                  </dd>
                  <dt>agreement</dt>
                  <dd>
                    <span
                      className={`verdict ${
                        mirrorsAgree === null
                          ? "is-null"
                          : mirrorsAgree
                            ? "is-sig"
                            : "is-inconclusive"
                      }`}
                    >
                      {mirrorsAgree === null
                        ? "UNKNOWN"
                        : mirrorsAgree
                          ? "AGREES"
                          : "BEHIND"}
                    </span>
                  </dd>
                </dl>
              </div>
            </div>
          </div>

          {live.status.state !== "connected" && (
            <div className="informal">
              <span className="informal-t">
                {live.status.state === "error"
                  ? "The dashboard's copy could not be read"
                  : live.status.state === "unconfigured"
                    ? "The dashboard's copy is not configured"
                    : "Connected, nothing captured yet"}
              </span>
              <p>{live.status.detail}</p>
              <p>
                This degrades the page, not the record. The predictions above come from
                the committed mirror, which is the primary record either way &mdash; a run
                without credentials writes the mirror, reports that Supabase was not
                written, and is a degraded run rather than a failed one.
              </p>
            </div>
          )}
        </section>

        {/* ── the chain ─────────────────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">The row-hash chain</h2>
            <span className="mono sec-cite">L4.2 &middot; {PREDICTIONS.length} rows</span>
          </div>
          <p className="sec-sub">
            Each row hashes its own fields together with the previous row&apos;s hash, so
            an edit, a deletion or a reorder is detectable after the fact by anyone
            holding the file. This matters because RLS does not cover the one case that
            would matter most: <strong>the service role bypasses it entirely</strong>, so
            the log&apos;s immutability rests on the chain and the git mirror, not on
            database permissions.
          </p>

          <div className="chain scroll-x">
            <span className="chain-node is-genesis mono" title={genesis}>
              genesis
            </span>
            {PREDICTIONS.map((row) => (
              <span key={row.matchId} className="chain-step">
                <i className="chain-arrow" aria-hidden="true" />
                <span className="chain-node mono" title={row.rowHash}>
                  {row.rowHash.slice(0, 8)}
                </span>
              </span>
            ))}
            <span className="chain-step">
              <i className="chain-arrow" aria-hidden="true" />
              <span className="chain-node is-head mono">head</span>
            </span>
          </div>

          <div className="scroll-x">
            <table className="grid-table">
              <thead>
                <tr>
                  <th>match</th>
                  <th>written</th>
                  <th>cutoff</th>
                  <th>row hash</th>
                  <th>links to</th>
                </tr>
              </thead>
              <tbody>
                {PREDICTIONS.map((row, index) => {
                  const expected = index === 0 ? genesis : PREDICTIONS[index - 1].rowHash;
                  const linked = row.prevHash === expected;
                  return (
                    <tr key={row.matchId}>
                      <td>
                        <span className="cont-match">
                          <Crest team={row.homeTeam} size={18} />
                          {displayName(row.homeTeam)}
                          <span className="fx-v mono">v</span>
                          {displayName(row.awayTeam)}
                          <Crest team={row.awayTeam} size={18} />
                        </span>
                      </td>
                      <td className="mono">{stampShort(row.generatedAtUtc)}</td>
                      <td className="mono">{row.stateCutoffDate}</td>
                      <td className="mono dt-ci">{row.rowHash.slice(0, 16)}…</td>
                      <td>
                        <span className={`verdict ${linked ? "is-sig" : "is-null"}`}>
                          {index === 0 ? "GENESIS" : linked ? "OK" : "BROKEN"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
              <caption>
                The authoritative check is{" "}
                <span className="mono">
                  python scripts/phase6_generate_predictions.py --verify
                </span>
                , which recomputes every hash from each row&apos;s own fields and repairs
                nothing. This table only checks that the links are contiguous, which is
                the weaker of the two claims.
              </caption>
            </table>
          </div>
        </section>

        {/* ── results, once they exist ───────────────────────────────── */}
        {live.results.length > 0 && (
          <section className="stack stack-md">
            <div className="sec-hd">
              <h2 className="sec-h">Captured results</h2>
              <span className="mono sec-cite">Supabase &middot; read {stampShort(live.readAt)}</span>
            </div>
            <div className="scroll-x">
              <table className="grid-table">
                <thead>
                  <tr>
                    <th>match</th>
                    <th>played</th>
                    <th>score</th>
                    <th>outcome</th>
                    <th>state age</th>
                    <th>stale</th>
                  </tr>
                </thead>
                <tbody>
                  {live.results.map((r) => (
                    <tr key={r.matchId}>
                      <td className="mono">{r.matchId}</td>
                      <td className="mono">{dateShort(r.playedDate)}</td>
                      <td className="figure">
                        {r.homeGoals}&ndash;{r.awayGoals}
                      </td>
                      <td className="mono">{r.result}</td>
                      <td className="figure">{r.stateAgeDays}d</td>
                      <td>
                        {r.isStale ? (
                          <span className="verdict is-inconclusive">STALE</span>
                        ) : (
                          <span className="verdict is-null">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
                <caption>
                  State age is the days between the state cutoff the prediction was
                  written against and the date actually played, which under L3.1 may
                  differ from the scheduled date because a postponed match keeps its
                  original prediction. Stale is pre-declared at 10 days from the
                  development distribution &mdash; not chosen after seeing these rows.
                </caption>
              </table>
            </div>
          </section>
        )}

        {/* ── the known gaps ────────────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Known gaps in this page</h2>
          </div>

          <div className="split-2">
            <div className="panel">
              <div className="panel-hd">
                <span className="eyebrow">Pinnacle is absent</span>
              </div>
              <div className="panel-bd">
                <p className="prose">
                  The primary book is fixed at Bet365 closing with a proportional de-vig,
                  both chosen before any score existed. Pinnacle is not in the 2026-27
                  file and <strong>no other book is promoted to replace it</strong> &mdash;
                  choosing a sensitivity instrument after seeing which one survived is
                  selection on the very thing a sensitivity exists to test. That is a
                  reduction in scope, recorded rather than worked around.
                </p>
              </div>
            </div>

            <div className="panel">
              <div className="panel-hd">
                <span className="eyebrow">Four sides have no crest</span>
              </div>
              <div className="panel-bd stack stack-sm">
                <p className="prose">
                  The crest library covers 16 of the 20 sides. These four render as
                  monograms, which is a cosmetic gap and is listed so it does not read as
                  a data problem:
                </p>
                <div className="missing">
                  {MISSING_CRESTS.map((team) => (
                    <span key={team} className="missing-item">
                      <Crest team={team} size={22} />
                      {displayName(team)}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="informal">
          <span className="informal-t">The cadence offset, pre-declared</span>
          <p>
            This log refits {META.logRefitsPerSeason} times a season. The frozen
            walk-forward refits once per distinct scored date &mdash;{" "}
            {META.frozenRefitsRange} times &mdash; so 37 of 38 matchweeks are predicted
            without that matchweek&apos;s earlier results. Measured on the development
            seasons before any 2026-27 row was written, that costs{" "}
            <strong>{fixed(META.cadenceCostLogLoss, 7)} log loss</strong> and{" "}
            {fixed(META.cadenceCostRps, 7)} RPS. It is recorded here so it cannot be
            invented afterwards to explain a gap &mdash; and honestly, four seasons is
            four observations: the figure is a measurement, not a law, and matchweek
            cadence was actually <em>better</em> in 2022-23.
          </p>
          <p>
            <Link href="/evidence" className="inline-link">
              The development evidence
            </Link>{" "}
            is where the frozen figures live.
          </p>
        </div>
      </div>
    </>
  );
}
