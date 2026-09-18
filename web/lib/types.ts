/**
 * The shapes shared between the compiled mirror and the live database read.
 *
 * These are declared rather than derived. frozen.generated.ts emits everything
 * `as const`, which is right for the frozen figures - a literal type is a
 * pleasant way of saying "this number cannot change" - but it makes every
 * mirror row its own literal type, so a row fetched from Supabase at runtime is
 * not assignable to it. One written-down interface, and both sources widen to
 * it.
 */

export interface Prediction {
  matchId: string;
  season: string;
  roundId: number;
  matchweekLabel: number | null;
  scheduledDate: string;
  scheduledKickoff: string | null;
  homeTeam: string;
  awayTeam: string;
  stateCutoffDate: string;
  generatedAtUtc: string;
  pHome: number;
  pDraw: number;
  pAway: number;
  lambdaHome: number;
  lambdaAway: number;
  rho: number;
  fitMatches: number;
  homeHasHistory: boolean;
  awayHasHistory: boolean;
  rowHash: string;
  prevHash: string;
  /**
   * Derived, never stored in the database (protocol L10.2 B): written before
   * this match's own kickoff. False is a late row - the same numbers a timely
   * run gives, without the timestamp proof - and is kept out of every
   * pre-kickoff figure. See lib/kickoff.ts.
   */
  writtenPreKickoff: boolean;
}

export interface Contaminated {
  matchId: string;
  playedDate: string;
  homeTeam: string;
  awayTeam: string;
  reason: string;
  recordedAtUtc: string;
  excludedFromLog: boolean;
  excludedFromHoldout: boolean;
}
