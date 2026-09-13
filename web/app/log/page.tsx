import type { Metadata } from "next";
import Link from "next/link";

import { Crest } from "@/components/Crest";
import { displayName } from "@/lib/crests";
import { CONTAMINATED, META, PREDICTIONS } from "@/lib/frozen.generated";
import { dateShort, fixed, stampShort } from "@/lib/format";
import { readLiveLog, type LiveStatus } from "@/lib/supabase";

export const revalidate = 600;

export const metadata: Metadata = {
  title: "The record",
  description:
    "How to check that these predictions were really written before kickoff: " +
    "where each half of the record lives, what has been covered, and why a row " +
    "cannot be edited afterwards without it showing.",
};

function statusPill(status: LiveStatus) {
  const map = {
    connected: { dot: "", label: "connected" },
    empty: { dot: " is-idle", label: "connected · no results yet" },
    unconfigured: { dot: " is-idle", label: "not set up yet" },
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
          <p className="eyebrow">The part that makes the rest of it count</p>
          <h1 className="hero-h display">
            The record, and{" "}
            <span className="hero-em">what makes it hard to fake</span>
          </h1>
          <p className="prose hero-p">
            Anyone can say a model did well. A prediction with a timestamp nobody can
            move <strong>proves what was said before the result existed</strong>, which
            is much harder and much more useful. This page is how you check ours,
            without taking our word for any of it.
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
                <span className="eyebrow">The predictions &middot; in git</span>
                <span className="verdict is-sig">THE REAL RECORD</span>
              </div>
              <div className="panel-bd stack stack-sm">
                <p className="prose">
                  The predictions and the list of matches left out are committed to a
                  code repository, and committing them is the entire point: git stamps
                  &ldquo;this was said before kickoff&rdquo; in a way we cannot go back
                  and change. These files contain <strong>no scorelines at all</strong>.
                </p>
                <dl className="kv">
                  <dt>predictions written</dt>
                  <dd className="figure">{mirrorCount}</dd>
                  <dt>matches left out</dt>
                  <dd className="figure">{CONTAMINATED.length}</dd>
                  <dt>chain unbroken</dt>
                  <dd>
                    <span className={`verdict ${chainLinks ? "is-sig" : "is-null"}`}>
                      {chainLinks ? "YES" : "NO"}
                    </span>
                  </dd>
                </dl>
              </div>
            </div>

            <div className="panel">
              <div className="panel-hd">
                <span className="eyebrow">The results &middot; in a database</span>
                <span className="verdict is-null">A COPY</span>
              </div>
              <div className="panel-bd stack stack-sm">
                <p className="prose">
                  Scorelines and closing odds live <strong>only</strong> here. The
                  2026-27 results stay out of the repository until the season&apos;s
                  measurement is deliberately taken in May 2027, so this page reads them
                  over the network instead. If this database were lost tomorrow, the
                  evidence would not be.
                </p>
                <dl className="kv">
                  <dt>results collected</dt>
                  <dd className="figure">
                    {live.coverage ? live.coverage.resultsCaptured : "—"}
                  </dd>
                  <dt>predictions copied across</dt>
                  <dd className="figure">
                    {dbCount === null ? "—" : `${dbCount} of ${mirrorCount}`}
                  </dd>
                  <dt>the two agree</dt>
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
                          ? "YES"
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
                  ? "The results database could not be read"
                  : live.status.state === "unconfigured"
                    ? "The results database is not set up yet"
                    : "Connected, nothing collected yet"}
              </span>
              <p>{live.status.detail}</p>
              <p>
                This makes the page worse, not the record. The predictions still come
                from the committed copy, which is the real record either way &mdash; a
                run without database credentials writes the files, says so, and counts
                as a degraded run rather than a failed one.
              </p>
            </div>
          )}
        </section>

        {/* ── the chain, folded away ────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Why nobody can quietly edit these predictions</h2>
          </div>
          <p className="sec-sub">
            Every row carries a fingerprint of its own contents{" "}
            <strong>plus the fingerprint of the row before it</strong>. Change one
            number in one row and every fingerprint after it stops matching, so an
            edit, a deletion or a reordering shows up to anyone holding the file. That
            is what the record rests on &mdash; not on database permissions, which the
            administrator account can bypass entirely.
          </p>

          {/*
            This used to be a strip of truncated hashes and a row-per-prediction
            table, printed in full, above the fold. Nobody has ever verified a hash
            by reading one off a web page; the sentence above is the claim, and the
            evidence for it is one click away for whoever wants it.
          */}
          <details className="drawer">
            <summary>
              Show the chain &mdash; all {PREDICTIONS.length} fingerprints
            </summary>
            <div className="scroll-x">
              <div className="chain">
                <span className="chain-node is-genesis mono" title={genesis}>
                  start
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
                  <span className="chain-node is-head mono">now</span>
                </span>
              </div>

              <table className="grid-table">
                <thead>
                  <tr>
                    <th>match</th>
                    <th>written</th>
                    <th>knew results up to</th>
                    <th>fingerprint</th>
                    <th>links back</th>
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
                            <Crest team={row.homeTeam} size={20} />
                            {displayName(row.homeTeam)}
                            <span className="fx-v">v</span>
                            {displayName(row.awayTeam)}
                            <Crest team={row.awayTeam} size={20} />
                          </span>
                        </td>
                        <td className="mono">{stampShort(row.generatedAtUtc)}</td>
                        <td className="mono">{row.stateCutoffDate}</td>
                        <td className="mono dt-ci">{row.rowHash.slice(0, 16)}…</td>
                        <td>
                          <span className={`verdict ${linked ? "is-sig" : "is-null"}`}>
                            {index === 0 ? "FIRST" : linked ? "OK" : "BROKEN"}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
                <caption>
                  This table only checks that the links join up, which is the weaker of
                  the two claims. The real check is{" "}
                  <span className="mono">
                    python scripts/phase6_generate_predictions.py --verify
                  </span>
                  , which recalculates every fingerprint from the row&apos;s own fields
                  and repairs nothing.
                </caption>
              </table>
            </div>
          </details>
        </section>

        {/* ── results, once they exist ───────────────────────────────── */}
        {live.results.length > 0 && (
          <section className="stack stack-md">
            <div className="sec-hd">
              <h2 className="sec-h">The results as they come in</h2>
              <span className="sec-cite">read {stampShort(live.readAt)}</span>
            </div>
            <div className="scroll-x">
              <table className="grid-table">
                <thead>
                  <tr>
                    <th>match</th>
                    <th>played</th>
                    <th>score</th>
                    <th>outcome</th>
                    <th>prediction age</th>
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
                  Prediction age is the gap between the last result the model was
                  allowed to see and the day the match was actually played &mdash; which
                  can differ from the scheduled day, because a postponed match keeps its
                  original prediction rather than getting a fresh one. Ten days counts
                  as stale, and that threshold was set from the development seasons, not
                  chosen after seeing these rows.
                </caption>
              </table>
            </div>
          </section>
        )}

        {/* ── the one gap worth naming ──────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">One gap worth naming</h2>
          </div>
          <div className="panel">
            <div className="panel-hd">
              <span className="eyebrow">A second bookmaker is missing</span>
            </div>
            <div className="panel-bd">
              <p className="prose">
                The bookmaker we measure against was fixed as Bet365&apos;s closing
                price before any 2026-27 score existed. A second one, Pinnacle, was
                meant to be the sanity check and is not in this season&apos;s file
                &mdash; and <strong>no other bookmaker has been promoted to replace
                it</strong>. Picking a replacement after seeing which ones survived
                would be choosing the test by its answer, which is the exact thing a
                sanity check exists to prevent. So the scope is smaller, and it is
                written down rather than worked around.
              </p>
            </div>
          </div>
        </section>

        <div className="informal">
          <span className="informal-t">Why this log scores worse than the sealed model</span>
          <p>
            This log refits the model {META.logRefitsPerSeason} times a season &mdash;
            once a week. The sealed version refits after every single day that had
            matches on it, {META.frozenRefitsRange} times, so it goes into most
            weekends knowing results this log does not. Measured on past seasons,
            before a single 2026-27 row was written, that costs{" "}
            <strong>{fixed(META.cadenceCostLogLoss, 4)}</strong> &mdash; small, but
            real. It is recorded here so it cannot be invented later to explain away a
            bad run. Honestly, four seasons is four observations: weekly refitting was
            actually <em>better</em> in 2022-23.
          </p>
          <p>
            <Link href="/evidence" className="inline-link">
              The development evidence
            </Link>{" "}
            is where the sealed figures live, and{" "}
            <Link href="/how-it-works" className="inline-link">
              how it works
            </Link>{" "}
            is the version without the jargon.
          </p>
        </div>
      </div>
    </>
  );
}
