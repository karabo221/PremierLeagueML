import { ci, fixed, signed } from "@/lib/format";
import { DELTAS, META } from "@/lib/frozen.generated";

type Delta = (typeof DELTAS)[number];

function verdictClass(verdict: string): string {
  if (verdict === "SIGNIFICANT") return "is-sig";
  if (verdict === "INCONCLUSIVE") return "is-inconclusive";
  return "is-null";
}

/**
 * Every comparison, with its interval and its verdict - including, and
 * especially, the ones that found nothing. A table of only the significant
 * rows would misrepresent a project whose headline result is a null: the nulls
 * are the finding, and they are only worth reading because the same instrument
 * detected the alternative elsewhere in the same table.
 */
export function DeltaTable({ rows = DELTAS }: { rows?: readonly Delta[] }) {
  return (
    <div className="scroll-x">
      <table className="grid-table">
        <thead>
          <tr>
            <th>comparison</th>
            <th>&Delta; log loss</th>
            <th>95% CI</th>
            <th>&Delta; RPS</th>
            <th>verdict</th>
            <th>artefact</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.comparison} className={row.comparison.includes("market - DixonColes") ? "is-focus" : undefined}>
              <td>
                <span className="dt-cmp">{row.comparison}</span>
                <span className="dt-lbl">{row.label}</span>
              </td>
              <td className="figure">{signed(row.logLossDelta!)}</td>
              <td className="figure dt-ci">{ci(row.logLossCi as [number, number])}</td>
              <td className="figure">{signed(row.rpsDelta!)}</td>
              <td>
                <span className={`verdict ${verdictClass(row.verdict)}`}>
                  {row.verdict === "SIGNIFICANT" ? "SIGNIFICANT" : row.verdict}
                </span>
                {!row.signsAgree && (
                  <span className="dt-flag mono" title="Log loss and RPS disagree in sign">
                    signs differ
                  </span>
                )}
              </td>
              <td className="mono dt-src">{row.source}</td>
            </tr>
          ))}
        </tbody>
        <caption>
          Negative favours the left-hand model. Every interval is the same paired
          bootstrap over the same {META.devMatches.toLocaleString("en-GB")} matched
          per-match scores &mdash; {META.bootstrapDraws.toLocaleString("en-GB")}{" "}
          draws, seed {META.bootstrapSeed}, identical draws for every delta in the
          project. Pairing is what makes a {fixed(0.003, 3)} difference readable:
          the models are scored on identical rows, so the shared difficulty of a
          season cancels.
        </caption>
      </table>
    </div>
  );
}
