import { fixed, onScale } from "@/lib/format";
import { LADDER } from "@/lib/frozen.generated";

/**
 * The ladder, on ONE declared scale.
 *
 * The axis is truncated at 0.95 and the caption says so. A truncated axis is a
 * real choice and hiding it would be the chart equivalent of reporting a
 * subgroup without its interval: the whole ladder spans 0.96 to 1.07, so an
 * axis from zero would compress every difference this project spent four
 * phases measuring into three pixels. Every tick below names a value the plot
 * actually reaches.
 *
 * THE ROWS ARE NAMED TWICE. "D4 - + prior-season FBref" is what the artefact
 * calls it and what REPORT.md and every CSV call it, so that spelling has to
 * survive somewhere or a reader cannot cross-reference. It survives as the
 * row's title attribute; what is PRINTED is what the rung actually was. A
 * reader who has never opened this project reads "139 stats from last season"
 * and knows what failed; "D4" tells them only that there were at least four.
 */

const LO = 0.95;
const HI = 1.07;
const TICKS = [0.95, 0.98, 1.01, 1.04, 1.07];

const KIND_LABEL: Record<string, string> = {
  baseline: "starting point",
  ladder: "built by hand",
  rating: "a rating",
  frozen: "sealed",
  arm: "side test",
  market: "the yardstick",
};

/** What each rung actually was, for a reader who has never met the codes. */
const PLAIN: Record<string, string> = {
  D0: "A blind guess — league averages only",
  D1: "Attempt 1 — this season's results",
  D2: "Attempt 2 — plus form and rest",
  D2_rescaled: "Attempt 2, rescaled",
  D3: "Attempt 3 — plus match context",
  D4: "Attempt 4 — plus 139 stats from last season",
  elo_v1: "Attempt 5 — one strength rating per team",
  poisson_walkforward: "Attempt 6 — a goals model",
  dc_walkforward: "The sealed model — goals, with a low-score tweak",
  E1a_sot: "Side test — shots-on-target ratings",
  E1b: "Side test — shot volume",
  E1c: "Side test — finishing",
  market_B365C_proportional: "The bookmakers — Bet365 closing prices",
};

export function LadderPlot() {
  return (
    <figure className="lad">
      <div className="lad-rows">
        {LADDER.map((row) => (
          <div key={row.key} className={`lad-row is-${row.kind}`}>
            <span className="lad-nm" title={row.label}>
              {PLAIN[row.key] ?? row.label}
            </span>

            <span className="lad-track">
              <i style={{ width: `${onScale(row.logLoss!, LO, HI)}%` }} />
            </span>

            <span className="lad-v figure">{fixed(row.logLoss!)}</span>
            <span className="lad-r figure">{fixed(row.rps!)}</span>
            <span className="lad-k">{KIND_LABEL[row.kind]}</span>
          </div>
        ))}
      </div>

      <div className="lad-axis mono" aria-hidden="true">
        {TICKS.map((tick) => (
          <span key={tick} style={{ left: `${onScale(tick, LO, HI)}%` }}>
            {tick.toFixed(2)}
          </span>
        ))}
      </div>

      <figcaption>
        How wrong each attempt was, over the same 1,520 matches none of them had
        seen &mdash; <strong>lower is better</strong>, and a shorter bar is a better
        model. <strong>The axis starts at {LO.toFixed(2)}, not zero</strong> &mdash;
        the whole ladder lives between{" "}
        {fixed(LADDER[LADDER.length - 1].logLoss!, 2)} and{" "}
        {fixed(LADDER[0].logLoss!, 2)}, so a bar&apos;s length is distance above{" "}
        {LO.toFixed(2)} and not a proportion of anything. The second figure is RPS,
        a second way of scoring the same predictions; where the two disagree about
        which model won, the comparison is reported inconclusive rather than
        resolved in whichever direction suited. Hover a name for the code the
        artefacts use.
      </figcaption>
    </figure>
  );
}
