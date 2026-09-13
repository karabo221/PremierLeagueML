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
            {META.seasons} &middot; {META.devMatches.toLocaleString("en-GB")} outer-test
            matches &middot; four frozen folds
          </p>
          <h1 className="hero-h display">
            A boundary, measured &mdash; and{" "}
            <span className="hero-em">no explanation for it</span>
          </h1>
          <p className="prose hero-p">
            Within the information classes this project could access and did test,
            the best model reaches <strong>{fixed(frozen.logLoss!)} log loss</strong>{" "}
            and {fixed(frozen.rps!)} RPS against a market benchmark of{" "}
            {fixed(market.logLoss!)} and {fixed(market.rps!)} on the same matches.
            Most of what follows is a null, and the point of the page is that the
            instrument which produced those nulls would have detected the
            alternative.
          </p>
        </div>
      </section>

      <div className="shell stack stack-lg page-body">
        {/* ── the ladder ─────────────────────────────────────────────── */}
        <section className="stack stack-md" id="ladder">
          <div className="sec-hd">
            <h2 className="sec-h">The results ladder</h2>
            <span className="mono sec-cite">
              phase4_ladder_pooled.csv &middot; phase4_d34_pooled.csv &middot;
              phase5_market_pooled.csv
            </span>
          </div>
          <LadderPlot />

          <div className="note-grid">
            {LADDER.filter((r) => r.note).map((r) => (
              <div key={r.key} className="note">
                <span className="note-k mono">{r.label}</span>
                <p>{r.note}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── the two uncomfortable readings ────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Two readings that are hard on the engineering</h2>
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
                  The tier-2 instrument splits the walk-forward advantage into recency
                  and training-set size. Recency accounts for{" "}
                  <strong>{fixed(recency.point!)}</strong>{" "}
                  {ci(recency.ci as [number, number])} of a total{" "}
                  {fixed(total.point!)} &mdash; {pct(recencyShare.point!, 0)}{" "}
                  &mdash; while the sample-size term is {fixed(sampleSize.point!)}{" "}
                  {ci(sampleSize.ci as [number, number])} and does not clear zero.
                </p>
                <p className="cite mono">{TIER2.source}</p>
              </div>
            </div>
          </div>
        </section>

        {/* ── decomposition ─────────────────────────────────────────── */}
        <section className="stack stack-md">
          <div className="sec-hd">
            <h2 className="sec-h">Where the {fixed(DECOMPOSITION.total, 4)} went</h2>
            <span className="verdict is-inconclusive">DERIVED</span>
          </div>
          <p className="sec-sub">
            The whole distance from a base rate to the frozen model, decomposed
            against D2 rescaled. This is the one figure on the page computed here
            rather than read from disk &mdash; it is arithmetic over three named
            artefacts, and the parts are checked to sum to the total before the page
            will build.
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
            <h2 className="sec-h">Every comparison, nulls included</h2>
          </div>
          <DeltaTable />
        </section>

        {/* ── the gap ───────────────────────────────────────────────── */}
        <section className="stack stack-md" id="gap">
          <div className="sec-hd">
            <h2 className="sec-h">The gap, and four ways of failing to explain it</h2>
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
            <h2 className="sec-h">Why the nulls are readable</h2>
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
                phase0_evaluation_folds.csv &mdash; fixed in Phase 0 before any model
                existed and never moved. Every fold trains only on seasons that
                finished before its test season began. The leakage suite ran{" "}
                {LEAKAGE.tests} tests over all {META.sourceMatches.toLocaleString("en-GB")}{" "}
                matches before modelling, {LEAKAGE.passed} passing (
                <span className="mono">{LEAKAGE.source}</span>).
              </caption>
            </table>
          </div>

          <div className="informal">
            <span className="informal-t">The honest frame for the gap</span>
            <p>
              The comparison is <strong>biased in this project&apos;s favour</strong>.
              2025-26 was scored during Phase 3&apos;s lambda sweep, so every model
              figure on this page is a walk-forward <em>development</em> estimate. The
              market has never been fitted to anything. The project loses by 0.03918 at
              D4 with the thumb on its own side of the scale, and the size of that
              optimism is exactly what the pending holdout exists to measure &mdash; once,
              at the end of the 2026-27 season, whatever it says.
            </p>
          </div>
        </section>
      </div>
    </>
  );
}
