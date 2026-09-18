/**
 * WAS A PREDICTION WRITTEN BEFORE ITS OWN KICKOFF?
 *
 * PHASE6_LIVE_LOG_PROTOCOL.txt L10.2 B. Since 2026-09-18 a round whose kickoff
 * has passed is written rather than ruled out, and the log keeps the two kinds
 * apart. The database carries no column for it on purpose - the schema is
 * unchanged by the amendment - so this is the same rule as
 * derived_pre_kickoff() in scripts/phase6_generate_predictions.py, applied to
 * the same three fields:
 *
 *     generated_at_utc  <  scheduled_date + scheduled_kickoff (UK), in UTC
 *
 * A row with no kickoff time is taken to kick off at 00:00 UK, so a missing
 * time can only make a row late, never early.
 */

const LONDON = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Europe/London",
  hourCycle: "h23",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});

/** London wall-clock time at a UTC instant, read back as if it were UTC. */
function londonWallAsUtc(instant: number): number {
  const part = Object.fromEntries(
    LONDON.formatToParts(new Date(instant)).map((p) => [p.type, p.value])
  );
  return Date.UTC(
    Number(part.year),
    Number(part.month) - 1,
    Number(part.day),
    Number(part.hour),
    Number(part.minute)
  );
}

/** A UK local date and "HH:MM" as a UTC instant, in milliseconds. */
export function ukKickoffUtc(date: string, clock: string | null): number {
  const [y, m, d] = date.slice(0, 10).split("-").map(Number);
  const [hh, mm] = (clock && /^\d{1,2}:\d{2}/.test(clock) ? clock : "00:00")
    .split(":")
    .map(Number);

  const wall = Date.UTC(y, m - 1, d, hh, mm);
  // Two passes settle the offset either side of a clock change.
  let guess = wall - (londonWallAsUtc(wall) - wall);
  guess = wall - (londonWallAsUtc(guess) - guess);
  return guess;
}

export function writtenBeforeKickoff(
  generatedAtUtc: string,
  scheduledDate: string,
  scheduledKickoff: string | null
): boolean {
  return Date.parse(generatedAtUtc) < ukKickoffUtc(scheduledDate, scheduledKickoff);
}
