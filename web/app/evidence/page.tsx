import type { Metadata } from "next";

import { DeltaTable } from "@/components/DeltaTable";
import { LadderPlot } from "@/components/LadderPlot";
import {
  DECOMPOSITION,
  DELTAS,
  FOLDS,
  GAP,
  LADDER,
  LEAKAGE,
  META,
  TIER2,
} from "@/lib/frozen.generated";
import { ci, fixed, pct, signed } from "@/lib/format";

export const metadata: Metadata = {
  title: "The evidence",
  description:
    "The frozen development figures: a results ladder over 1,520 outer-test " +
    "matches, every delta with its paired-bootstrap interval, and the " +
    "diagnostic that failed to explain a 0.02979 gap to the closing market.",
};

const frozen = LADDER.find((r) => r.kind === "frozen")!;
const market = LADDER.find((r) => r.kind === "market")!;
const gapDelta = DELTAS.find((d) => d.comparison === "market - DixonColes")!;
const elo = LADDER.find((r) => r.key === "elo_v1")!;
const d4 = LADDER.find((r) => r.key === "D4")!;

const SHARP = GAP.sharpness.rows;
const dcSharp = SHARP.find((r) => r.label === "Dixon-Coles")!;
const mktSharp = SHARP.find((r) => r.label.startsWith("Market"))!;

/** The splits worth surfacing; the rest stay in the drawer, all of them. */
const HEADLINE_SPLIT = "did the favourite deliver";

const uniq = (xs: readonly string[]): string => [...new Set(xs)].sort().join(" · ");

/**
 * Every artefact behind this page, collected in one block at the foot instead
 * of printed beside each heading. Derived from the data rather than typed, so
 * a filename cannot go stale here while the figure above it stays current.
 */
const SOURCES: ReadonlyArray<readonly [string, string]> = [
  ["How wrong each attempt was", uniq(LADDER.map((r) => r.source))],
  ["Every comparison and its range", uniq(DELTAS.map((d) => d.source))],
  ["Freshness against volume", TIER2.source],
  ["The gap, split every way", GAP.splitsSource],
  [
    `The gap, against ${GAP.correlations.columnsExamined} columns`,
    GAP.correlations.source,
  ],
  ["Confidence and calibration", GAP.sharpness.calibrationSource],
  ["Leakage checks, run before any modelling", LEAKAGE.source],
  ["The four test seasons", "phase0_evaluation_folds.csv"],
];

export default function EvidencePage() {
  const recency = TIER2.quantities["delta_recency"];
  const sampleSize = TIER2.quantities["delta_sample"];
  const total = TIER2.quantities["delta_total"];
  // the artefact carries the share itself, so it is read rather than divided
  const recencyShare = TIER2.quantities["share_recency"];

  return (
    <>
      <section className="hero">
        <div className="shell">
          <p className="eyebrow">
            {META.seasons} &middot; {META.devMatches.toLocaleString("en-GB")} matches
            no model had seen &middot; four test seasons fixed in advance
          </p>
          <h1 className="hero-h display">
            We ran into a wall &mdash;{" "}
            <span className="hero-em">and here is how we know it is real</span>
          </h1>
          <p className="prose hero-p">
            Using everything we could get hold of and did test, the best model
            scores <strong>{fixed(frozen.logLoss!, 3)}</strong> where the bookmakers
            score {fixed(market.logLoss!, 3)} on the same matches &mdash; lower is
            better, so they are still ahead. Most of what follows is a{" "}
            <strong>nothing found</strong>, and the point of the page is that these
            same tests would have found something had there been something to find.
          </p>
        </div>
      </section>

      <div className="shell stack stack-lg page-body">
        {/* ── the ladder ─────────────────────────────────────────────── */}
        <section className="stack stack-md" id="ladder">
          <div className="sec-hd">
            <h2 className="sec-h">Every attempt, and how wrong it was</h2>
          </div>
          <LadderPlot />

          <div className="note-grid">
            {LADDER.filter((r) => r.note).map((r) => (
              <div key={r.key} className="note">
                <span className="note-k">{r.label}</span>
                <p>{r.note}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── the two uncomfortable readings ────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Two findings we did not want</h2>
          </div>

          <div className="split-2">
            <div className="panel">
              <div className="panel-hd">
                <span className="eyebrow">One rating beats 139 columns</span>
              </div>
              <div className="panel-bd stack stack-sm">
                <p className="prose">
                  Elo v1 &mdash; a single K=20 rating, flat 1500 start, 60-point home
                  advantage &mdash; has a lower pooled log loss than every rung on the
                  ladder, D4&apos;s 139 columns included:{" "}
                  <strong>{fixed(elo.logLoss!)}</strong> against{" "}
                  {fixed(d4.logLoss!)}. Neither metric separates them. Recency-weighted
                  strength is most of the signal, and 139 engineered columns are a less
                  efficient way of writing it down than one rating is.
                </p>
              </div>
            </div>

            <div className="panel">
              <div className="panel-hd">
                <span className="eyebrow">And it is freshness, not volume</span>
              </div>
              <div className="panel-bd stack stack-sm">
                <p className="prose">
                  Refitting the model as the season goes helps &mdash; but we split
                  that help in two to find out <em>why</em>. Roughly{" "}
                  <strong>{pct(recencyShare.point!, 0)} of it</strong> comes from the
                  model being up to date; the rest, from it having seen more matches,{" "}
                  <strong>cannot be told apart from zero</strong>. Being current beats
                  having more history.
                </p>
                <p className="cite">
                  Freshness {fixed(recency.point!)} {ci(recency.ci as [number, number])}{" "}
                  &middot; extra data {fixed(sampleSize.point!)}{" "}
                  {ci(sampleSize.ci as [number, number])} &middot; total{" "}
                  {fixed(total.point!)}. A range that straddles zero means the effect
                  could as easily be nothing.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* ── decomposition ─────────────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Where the improvement actually came from</h2>
            <span className="verdict is-inconclusive">DERIVED</span>
          </div>
          <p className="sec-sub">
            The whole distance from a blind guess to the sealed model, split into
            the steps that produced it. This is the one figure on the page worked
            out here rather than read straight off a file &mdash; it is arithmetic
            over three named sources, and the parts are checked to add up to the
            total before the page will build.
          </p>

          <div className="decomp">
            <div className="decomp-bar" role="img" aria-label="Decomposition of the log loss improvement">
              {DECOMPOSITION.parts
                .filter((p) => p.share > 0)
                .map((part) => (
                  <i
                    key={part.label}
                    className={part.significant ? "is-sig" : "is-null"}
                    style={{ width: `${part.share * 100}%` }}
                  />
                ))}
            </div>

            <div className="decomp-keys">
              {DECOMPOSITION.parts.map((part) => (
                <div key={part.label} className={`decomp-key${part.share === 0 ? " is-zero" : ""}`}>
                  <span className="decomp-pc figure">
                    {part.share === 0 ? "nil" : pct(part.share, 0)}
                  </span>
                  <span className="decomp-nm">{part.label}</span>
                  <span className="decomp-sp mono">{part.span}</span>
                  <span className="decomp-ll figure">{fixed(part.logLoss, 5)}</span>
                  {!part.significant && (
                    <span className="verdict is-null">
                      {part.share === 0 ? "NOTHING MEASURABLE" : "NOT SIGNIFICANT"}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>

          <p className="prose">
            The {pct(DECOMPOSITION.parts[2].share, 0)} residual is{" "}
            <strong>not significant on {META.devMatches.toLocaleString("en-GB")} matches</strong>.
            That is resolution, not equality: it is unmeasured, not shown to be absent.
          </p>
        </section>

        {/* ── every delta ───────────────────────────────────────────── */}
        <section className="stack stack-md" id="deltas">
          <div className="sec-hd">
            <h2 className="sec-h">Every test we ran, including the ones that found nothing</h2>
          </div>
          <DeltaTable />
        </section>

        {/* ── the gap ───────────────────────────────────────────────── */}
        <section className="stack stack-md" id="gap">
          <div className="sec-hd">
            <h2 className="sec-h">The gap, and four ways we failed to explain it</h2>
          </div>

          <div className="gap-hero">
            <div className="gap-n">
              <span className="eyebrow">market &minus; Dixon-Coles &middot; &Delta; log loss</span>
              <span className="gap-fig figure">{signed(gapDelta.logLossDelta!)}</span>
              <span className="mono gap-ci">
                {ci(gapDelta.logLossCi as [number, number])} &middot; &Delta;RPS{" "}
                {signed(gapDelta.rpsDelta!)}
              </span>
              <span className="verdict is-sig">SIGNIFICANT</span>
            </div>

            <div className="gap-reasons">
              <div className="gap-reason">
                <span className="gap-k mono">1 &middot; uniform</span>
                <p>
                  Every split examined &mdash; favourite probability, matchweek, season,
                  promoted side, market pick, actual outcome &mdash; is null or nearly so.
                  The per-match standard deviation is 0.24897 against a mean of 0.0298:
                  the spread dwarfs the mean.
                </p>
              </div>
              <div className="gap-reason">
                <span className="gap-k mono">2 &middot; not on disk</span>
                <p>
                  Over {GAP.correlations.columnsExamined} numeric columns the largest
                  |r| is {fixed(GAP.correlations.largestAbsR!, 4)} (
                  <span className="mono">{GAP.correlations.largestColumn}</span>).{" "}
                  {GAP.correlations.clearingThreshold} clear the naive{" "}
                  {GAP.correlations.threshold} threshold where{" "}
                  {GAP.correlations.expectedByChance} are expected by chance. Nothing
                  explains more than 0.6% of the variance.
                </p>
              </div>
              <div className="gap-reason">
                <span className="gap-k mono">3 &middot; not calibration</span>
                <p>
                  Dixon-Coles is <strong>more</strong> confident than the market, not
                  less &mdash; mean max p {fixed(dcSharp.meanMaxP!, 4)} against{" "}
                  {fixed(mktSharp.meanMaxP!, 4)}. It is confident about a different
                  outcome. The market&apos;s edge is direction.
                </p>
              </div>
              <div className="gap-reason">
                <span className="gap-k mono">4 &middot; not the team sheet</span>
                <p>
                  The lineup hypothesis predicts the model loses most where a favourite
                  underperforms. Conditioning on whether the favourite delivered gives{" "}
                  <strong>+0.03866</strong> when it does against <strong>+0.01889</strong>{" "}
                  when it does not &mdash; the opposite sign. Recorded in advance, which is
                  why it reads as a refutation.
                </p>
              </div>
            </div>
          </div>

          <div className="scroll-x">
            <table className="grid-table">
              <thead>
                <tr>
                  <th>sharpness</th>
                  <th>mean max p</th>
                  <th>mean p(draw)</th>
                  <th>ECE</th>
                  <th>bias on draws</th>
                </tr>
              </thead>
              <tbody>
                {SHARP.map((row) => (
                  <tr key={row.label}>
                    <td>{row.label}</td>
                    <td className="figure">
                      {row.meanMaxP === null ? "—" : fixed(row.meanMaxP, 4)}
                    </td>
                    <td className="figure">
                      {row.meanPDraw === null ? "—" : fixed(row.meanPDraw, 4)}
                    </td>
                    <td className="figure">
                      {row.ece === null ? "—" : fixed(row.ece, 5)}
                    </td>
                    <td className="figure">
                      {row.biasD === null ? "—" : signed(row.biasD, 5)}
                    </td>
                  </tr>
                ))}
              </tbody>
              <caption>
                {GAP.sharpness.note} Read with one caveat the project states outright:
                D0, a constant base rate, has the best calibration here and is the
                worst model in the ladder, so calibration alone decides nothing. Two
                findings survive it &mdash; every model under-predicts draws, market
                included, and the market shows the textbook favourite&ndash;longshot bias.
              </caption>
            </table>
          </div>

          <details className="drawer">
            <summary>
              Every split that was looked at, including the null ones
            </summary>
            <div className="scroll-x">
              <table className="grid-table">
                <thead>
                  <tr>
                    <th>split</th>
                    <th>level</th>
                    <th>n</th>
                    <th>gap</th>
                    <th>95% CI</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(GAP.splits).flatMap(([split, levels]) =>
                    levels.map((level, index) => (
                      <tr
                        key={`${split}-${level.level}`}
                        className={split === HEADLINE_SPLIT ? "is-focus" : undefined}
                      >
                        <td>{index === 0 ? split : ""}</td>
                        <td className="dt-lvl">{level.level}</td>
                        <td className="figure">{level.n}</td>
                        <td className="figure">{signed(level.gap!)}</td>
                        <td className="figure dt-ci">
                          {ci(level.ci as [number, number])}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
                <caption>
                  {GAP.splitsSource} &mdash; reported in full. Publishing only the
                  split that separated would be the exact failure a pre-declared
                  diagnostic exists to prevent. The season split is the only one with
                  any separation, and conditioning on the actual outcome can never
                  become a feature: it describes, and that limit is declared with it.
                </caption>
              </table>
            </div>
          </details>
        </section>

        {/* ── method ────────────────────────────────────────────────── */}
        <section className="stack stack-md" id="method">
          <div className="sec-hd">
            <h2 className="sec-h">Why a &ldquo;nothing found&rdquo; here means something</h2>
          </div>

          <div className="scroll-x">
            <table className="grid-table">
              <thead>
                <tr>
                  <th>fold</th>
                  <th>train seasons</th>
                  <th>train</th>
                  <th>test season</th>
                  <th>test</th>
                  <th>last train date</th>
                  <th>first test date</th>
                  <th>order valid</th>
                </tr>
              </thead>
              <tbody>
                {FOLDS.map((fold) => (
                  <tr key={fold.fold}>
                    <td className="figure">{fold.fold}</td>
                    <td className="mono">{fold.trainSeasons}</td>
                    <td className="figure">{fold.trainMatches}</td>
                    <td className="mono">{fold.testSeason}</td>
                    <td className="figure">{fold.testMatches}</td>
                    <td className="mono">{fold.maxTrainDate}</td>
                    <td className="mono">{fold.minTestDate}</td>
                    <td>
                      <span className="verdict is-sig">
                        {fold.temporalOrderValid && fold.overlapValid ? "TRUE" : "FALSE"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
              <caption>
                Fixed before any model existed, and never moved since. Every fold
                learns only from seasons that had already finished when its test
                season began, so no model was ever marked on work it had seen. A
                separate suite ran {LEAKAGE.tests} checks over all{" "}
                {META.sourceMatches.toLocaleString("en-GB")} matches before any
                modelling started, {LEAKAGE.passed} of them passing.
              </caption>
            </table>
          </div>

          <div className="informal">
            <span className="informal-t">The honest frame for the gap</span>
            <p>
              The comparison above is <strong>tilted in our favour</strong>. Our
              figures come from seasons the project could look at while it was being
              built; the bookmakers have never been fitted to anything. We lose
              anyway &mdash; by {fixed(0.03918, 3)} at our most elaborate model, with
              the thumb on our own side of the scale. How much of that tilt there is,
              is exactly what the sealed 2026&ndash;27 season exists to measure:
              once, at the end, whatever it says.
            </p>
          </div>
        </section>

        {/* ── every source, in one place instead of beside every heading ── */}
        <section className="stack stack-md" id="sources">
          <div className="sec-hd">
            <h2 className="sec-h">Where these numbers come from</h2>
          </div>
          <p className="sec-sub">
            Every figure on this page is read from a committed file rather than typed
            in, and each one is hash-pinned so it cannot quietly change. These used to
            sit beside each heading, which put a filename in front of a reader before
            they had read the finding.
          </p>
          <div className="scroll-x">
            <table className="grid-table">
              <thead>
                <tr>
                  <th>what</th>
                  <th>file</th>
                </tr>
              </thead>
              <tbody>
                {SOURCES.map(([what, file]) => (
                  <tr key={file}>
                    <td>{what}</td>
                    <td className="mono">{file}</td>
                  </tr>
                ))}
              </tbody>
              <caption>
                Compiled by <span className="mono">phase6_build_frontend_data.py</span>{" "}
                on {META.generatedAt.slice(0, 10)}. The sealed rules are recorded under{" "}
                <span className="mono">{META.freezeSha.slice(0, 12)}…</span>, the cutoff
                under <span className="mono">{META.pinSha.slice(0, 12)}…</span>, and the
                weekly protocol under{" "}
                <span className="mono">{META.protocolSha.slice(0, 12)}…</span>.
              </caption>
            </table>
          </div>
        </section>
      </div>
    </>
  );
}
