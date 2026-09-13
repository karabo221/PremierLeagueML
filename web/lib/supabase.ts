/**
 * THE READ SIDE OF THE LIVE LOG.
 *
 * What lives here and why. The frontend needs three things that are NOT in the
 * repository, because L4.8 keeps 2026-27 scorelines and closing prices out of
 * it until the holdout is deliberately acquired in May 2027:
 *
 *     results     the scorelines captured after each matchweek
 *     market      Bet365 closing prices, de-vigged proportionally (B4.1)
 *     coverage    the live_log_coverage view, which is the only reason a
 *                 running figure may be shown at all (L2.4, L5.2)
 *
 * Predictions are read from here TOO, and the reason is operational rather than
 * evidential: reading them live means a new matchweek appears on the site
 * without anybody rebuilding or redeploying it. The git mirror is still the
 * primary record under L9.4 - git is what timestamps a pre-kickoff claim in a
 * way the operator cannot backdate - and it is still compiled into the bundle,
 * where it serves as the fallback whenever the database is behind, empty or
 * unreachable. The site says which of the two it is showing.
 *
 * READ ONLY. The anon key is used against the select-only policies in
 * ops/supabase_live_log.sql. There is no write path in web/ at all, and the
 * service key must never appear in this directory - the service role bypasses
 * RLS entirely, which section 5 of the DDL states outright.
 *
 * EVERY READ CAN FAIL SOFT. As of this writing Supabase has never been written:
 * the 2026-09-11 round ran mirror-only, recorded at L4d in
 * live_log/phase6_live_log_run_audit.csv. A missing table, absent credentials
 * or an empty database must render as a stated absence, never as a crash and
 * never as a zero that reads like a measurement.
 */

import { createClient, type SupabaseClient } from "@supabase/supabase-js";

import { PREDICTIONS } from "@/lib/frozen.generated";
import type { Prediction } from "@/lib/types";

export interface Coverage {
  predictionsWritten: number;
  resultsCaptured: number;
  contaminatedMatches: number;
  staleMatches: number;
  firstPredicted: string | null;
  lastPredicted: string | null;
  authorityNote: string;
}

export interface ResultRow {
  matchId: string;
  playedDate: string;
  homeGoals: number;
  awayGoals: number;
  result: "H" | "D" | "A";
  stateAgeDays: number;
  isStale: boolean;
  contaminated: boolean;
  capturedAtUtc: string;
}

export interface MarketRow {
  matchId: string;
  book: string;
  oddsHome: number;
  oddsDraw: number;
  oddsAway: number;
  pHome: number;
  pDraw: number;
  pAway: number;
  overround: number;
}

/** Why a read returned nothing. Rendered verbatim, so it has to be readable. */
export type LiveStatus =
  | { state: "connected"; rows: number }
  | { state: "unconfigured"; detail: string }
  | { state: "empty"; detail: string }
  | { state: "error"; detail: string };

export interface LiveLog {
  coverage: Coverage | null;
  results: ResultRow[];
  market: MarketRow[];
  predictionsInDatabase: number | null;
  status: LiveStatus;
  readAt: string;
}

function client(): SupabaseClient | null {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return null;
  return createClient(url, key, {
    auth: { persistSession: false },
    global: { headers: { "x-application-name": "premierleague-ml-web" } },
  });
}

const EMPTY = (status: LiveStatus): LiveLog => ({
  coverage: null,
  results: [],
  market: [],
  predictionsInDatabase: null,
  status,
  readAt: new Date().toISOString(),
});

export interface PredictionFeed {
  rows: readonly Prediction[];
  /** Which record is on screen, so the page can say so rather than imply it. */
  source: "database" | "mirror";
  detail: string;
}

/**
 * The round list, live where possible.
 *
 * WHY LIVE AT ALL. The generator writes every round to both the mirror and the
 * database. Reading the database means a new matchweek shows up on its own,
 * within one revalidation window, with no rebuild and no deploy. Reading only
 * the compiled mirror would mean a code push every week to publish figures that
 * were already computed - work that buys the reader nothing.
 *
 * WHY THE MIRROR IS STILL HERE. It is the primary record, and it is the honest
 * fallback: if the database is empty, behind or unreachable, the site shows the
 * rows git can prove were written before kickoff rather than showing nothing.
 * WHICHEVER IS USED IS NAMED ON THE PAGE - a fallback that looks identical to
 * the real thing is how a stale figure gets read as a current one.
 */
export async function readPredictions(): Promise<PredictionFeed> {
  const mirror = {
    rows: PREDICTIONS as readonly Prediction[],
    source: "mirror" as const,
    detail:
      `The committed mirror, ${PREDICTIONS.length} rows. This is the primary ` +
      `record either way; the database is not adding anything to it right now.`,
  };

  const supabase = client();
  if (!supabase) return mirror;

  try {
    const { data, error } = await supabase
      .from("predictions")
      .select("*")
      .order("scheduled_date", { ascending: true })
      .order("match_id", { ascending: true });

    if (error || !data || data.length === 0) return mirror;

    const rows: Prediction[] = data.map((r: Record<string, unknown>) => ({
      matchId: String(r.match_id),
      season: String(r.season),
      roundId: Number(r.round_id),
      matchweekLabel: r.matchweek_label === null ? null : Number(r.matchweek_label),
      scheduledDate: String(r.scheduled_date),
      scheduledKickoff: (r.scheduled_kickoff as string) ?? null,
      homeTeam: String(r.home_team),
      awayTeam: String(r.away_team),
      stateCutoffDate: String(r.state_cutoff_date),
      generatedAtUtc: String(r.generated_at_utc),
      pHome: Number(r.p_home),
      pDraw: Number(r.p_draw),
      pAway: Number(r.p_away),
      lambdaHome: Number(r.lambda_home),
      lambdaAway: Number(r.lambda_away),
      rho: Number(r.rho),
      fitMatches: Number(r.fit_matches),
      homeHasHistory: Boolean(r.home_has_history),
      awayHasHistory: Boolean(r.away_has_history),
      rowHash: String(r.row_hash),
      prevHash: String(r.prev_hash),
    }));

    // The database being BEHIND the mirror is a real state and not an error:
    // a round was written mirror-only. Show the longer record and say why.
    if (rows.length < PREDICTIONS.length) {
      return {
        rows: PREDICTIONS as readonly Prediction[],
        source: "mirror",
        detail:
          `The database holds ${rows.length} of the mirror's ${PREDICTIONS.length} ` +
          `predictions, so the mirror is shown. Run ` +
          `scripts/phase6_backfill_supabase.py --write to catch it up.`,
      };
    }

    return {
      rows,
      source: "database",
      detail: `${rows.length} rows, read live. A new round appears here on its own.`,
    };
  } catch {
    return mirror;
  }
}

export async function readLiveLog(): Promise<LiveLog> {
  const supabase = client();

  if (!supabase) {
    return EMPTY({
      state: "unconfigured",
      detail:
        "NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY are not set, " +
        "so the dashboard's copy cannot be read. The predictions below come " +
        "from the committed mirror, which is the primary record either way.",
    });
  }

  try {
    const [coverage, results, market, predictions] = await Promise.all([
      supabase.from("live_log_coverage").select("*").maybeSingle(),
      supabase.from("results").select("*").order("played_date", { ascending: false }),
      supabase.from("market").select("*"),
      supabase.from("predictions").select("match_id", { count: "exact", head: true }),
    ]);

    const failure = coverage.error ?? results.error ?? market.error ?? predictions.error;
    if (failure) {
      // The most likely cause by far, and worth naming rather than showing a
      // bare PostgREST string: the schema has not been applied yet.
      const missingSchema = /does not exist|schema cache|relation/i.test(failure.message);
      return EMPTY({
        state: "error",
        detail: missingSchema
          ? `The tables are not there yet - apply ops/supabase_live_log.sql. (${failure.message})`
          : failure.message,
      });
    }

    const rows = (results.data ?? []).length;
    const row = coverage.data as Record<string, unknown> | null;

    return {
      coverage: row
        ? {
            predictionsWritten: Number(row.predictions_written ?? 0),
            resultsCaptured: Number(row.results_captured ?? 0),
            contaminatedMatches: Number(row.contaminated_matches ?? 0),
            staleMatches: Number(row.stale_matches ?? 0),
            firstPredicted: (row.first_predicted as string) ?? null,
            lastPredicted: (row.last_predicted as string) ?? null,
            authorityNote: String(row.authority_note ?? ""),
          }
        : null,
      results: (results.data ?? []).map((r: Record<string, unknown>) => ({
        matchId: String(r.match_id),
        playedDate: String(r.played_date),
        homeGoals: Number(r.home_goals),
        awayGoals: Number(r.away_goals),
        result: r.result as "H" | "D" | "A",
        stateAgeDays: Number(r.state_age_days),
        isStale: Boolean(r.is_stale),
        contaminated: Boolean(r.contaminated),
        capturedAtUtc: String(r.captured_at_utc),
      })),
      market: (market.data ?? []).map((r: Record<string, unknown>) => ({
        matchId: String(r.match_id),
        book: String(r.book),
        oddsHome: Number(r.odds_home),
        oddsDraw: Number(r.odds_draw),
        oddsAway: Number(r.odds_away),
        pHome: Number(r.p_home),
        pDraw: Number(r.p_draw),
        pAway: Number(r.p_away),
        overround: Number(r.overround),
      })),
      predictionsInDatabase: predictions.count ?? 0,
      status:
        rows === 0
          ? {
              state: "empty",
              detail:
                "Connected, and no result has been captured yet. Nothing is " +
                "wrong: the first matchweek of the log has not been scored.",
            }
          : { state: "connected", rows },
      readAt: new Date().toISOString(),
    };
  } catch (error) {
    return EMPTY({
      state: "error",
      detail: error instanceof Error ? error.message : String(error),
    });
  }
}
