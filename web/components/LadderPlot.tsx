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
 */

const LO = 0.95;
const HI = 1.07;
const TICKS = [0.95, 0.98, 1.01, 1.04, 1.07];

const KIND_LABEL: Record<string, string> = {
  baseline: "baseline",
  ladder: "engineered",
  rating: "rating",
  frozen: "frozen",
  arm: "Phase 5 arm",
  market: "benchmark",
};

export function LadderPlot() {
  return (
    <figure className="lad">
      <div className="lad-rows">
        {LADDER.map((row) => (
          <div key={row.key} className={`lad-row is-${row.kind}`}>
            <span className="lad-nm">{row.label}</span>

            <span className="lad-track">
              <i style={{ width: `${onScale(row.logLoss!, LO, HI)}%` }} />
            </span>

            <span className="lad-v figure">{fixed(row.logLoss!)}</span>
            <span className="lad-r figure">{fixed(row.rps!)}</span>
            <span className="lad-k mono">{KIND_LABEL[row.kind]}</span>
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
        Pooled log loss over the same 1,520 outer-test matches, lower is better.
        <strong> The axis is truncated at {LO.toFixed(2)}</strong> &mdash; the whole
        ladder lives between {fixed(LADDER[LADDER.length - 1].logLoss!, 2)} and{" "}
        {fixed(LADDER[0].logLoss!, 2)}, so a bar&apos;s length is distance above{" "}
        {LO.toFixed(2)} and not a proportion of anything. RPS is the primary
        metric and is given beside each figure; a comparison where the two
        disagree in sign is reported inconclusive rather than resolved.
      </figcaption>
    </figure>
  );
}
