/**
 * Formatting. Two rules carried from the project's vocabulary:
 *
 *   1. A figure keeps the precision its artefact records it at. The deltas are
 *      read at 5 decimal places because a 0.003 difference is the whole point
 *      of the paired bootstrap; rounding one to 2dp on a web page would erase
 *      the finding.
 *
 *   2. A sign is never dropped. A delta of -0.02979 means the right-hand model
 *      is worse, and "0.02979" on its own is ambiguous about which way.
 */

export const pct = (p: number, digits = 1): string =>
  `${(p * 100).toFixed(digits)}%`;

/** Whole-percent, for a probability bar's own label where space is tight. */
export const pctShort = (p: number): string => `${Math.round(p * 100)}%`;

export const fixed = (value: number, digits = 5): string => value.toFixed(digits);

/** U+2212 MINUS SIGN, not a hyphen: these sit in tabular figures. */
export const signed = (value: number, digits = 5): string =>
  `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(digits)}`;

export const ci = (bounds: readonly [number, number], digits = 5): string =>
  `[${signed(bounds[0], digits)}, ${signed(bounds[1], digits)}]`;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

/** Dates are parsed as UTC. A local-midnight parse shifts the day westward. */
function utc(iso: string): Date {
  return new Date(`${iso.slice(0, 10)}T00:00:00Z`);
}

export const dayShort = (iso: string): string => DAYS[utc(iso).getUTCDay()];

export const dateShort = (iso: string): string => {
  const d = utc(iso);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
};

export const dateLong = (iso: string): string => {
  const d = utc(iso);
  return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
};

/** "2026-09-11T19:21:39Z" -> "11 Sep 19:21 UTC" */
export const stampShort = (iso: string): string => {
  const d = new Date(iso);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ` +
    `${String(d.getUTCHours()).padStart(2, "0")}:` +
    `${String(d.getUTCMinutes()).padStart(2, "0")} UTC`;
};

export const shortHash = (hash: string | null, head = 8): string =>
  hash ? `${hash.slice(0, head)}…${hash.slice(-5)}` : "—";

/** Which of H/D/A the model leans to, for a card's emphasis. */
export type Lean = "H" | "D" | "A";

export const lean = (pHome: number, pDraw: number, pAway: number): Lean => {
  const top = Math.max(pHome, pDraw, pAway);
  return top === pHome ? "H" : top === pDraw ? "D" : "A";
};

/**
 * Position on a fixed scale, clamped. The ladder plot declares 0.95 -> 1.07
 * once and every bar, tick and label is placed by this, so no mark can sit
 * outside the axis it is drawn against.
 */
export const onScale = (value: number, lo: number, hi: number): number =>
  Math.max(0, Math.min(100, ((value - lo) / (hi - lo)) * 100));
