-- ============================================================================
-- PHASE 6 - THE LIVE PREDICTION LOG, SUPABASE SCHEMA AND RLS
-- ============================================================================
--
-- Governed by PHASE6_LIVE_LOG_PROTOCOL.txt section 9, which is governed by
-- PHASE6_HOLDOUT_FREEZE.txt (sha256 b36befd3...248f6) and
-- PHASE6_CUTOFF_PIN.txt.
--
-- THE SERVICE KEY IS NOT IN THIS FILE AND IS NOT IN THIS REPOSITORY. The
-- generator reads SUPABASE_URL and SUPABASE_SERVICE_KEY from the environment.
--
-- WHAT THIS IS NOT. Supabase is the dashboard's copy. The PRIMARY record is
-- live_log/phase6_live_predictions.csv in git, because git is what timestamps
-- a row in a way the operator cannot backdate (L9.4). If this database is
-- lost, the evidence is not.
--
-- Apply with:  psql "$SUPABASE_DB_URL" -f ops/supabase_live_log.sql
--         or:  paste into the Supabase SQL editor
--
-- Idempotent: safe to re-run.
-- ============================================================================


-- ============================================================
-- 1. predictions
-- ============================================================
--
-- One row per match, written ONCE, before that matchweek's first kickoff.
-- Never updated, never deleted, never regenerated (L1.4, L3.1).

create table if not exists public.predictions (
    -- L3.3. season + home + away. NOT date-based, because L3.1 keeps a
    -- postponed match's original prediction and the date moves underneath it.
    match_id            text primary key,

    season              text        not null default '2026-2027',
    round_id            integer     not null,
    matchweek_label     integer,                -- informational only, L1.3

    scheduled_date      date        not null,
    scheduled_kickoff   text,

    home_team           text        not null,
    away_team           text        not null,

    -- L1.2. The state window was every completed match with date < this.
    state_cutoff_date   date        not null,

    -- The claim this table exists to make: written at this instant, and the
    -- first kickoff of the round was strictly after it.
    generated_at_utc    timestamptz not null,

    p_home              double precision not null,
    p_draw              double precision not null,
    p_away              double precision not null,
    lambda_home         double precision not null,
    lambda_away         double precision not null,
    rho                 double precision not null,

    fit_matches         integer     not null,
    home_has_history    boolean     not null,
    away_has_history    boolean     not null,

    -- Which frozen documents were in force. A row that cannot name its own
    -- governing texts is not evidence of anything.
    freeze_sha          text        not null,
    pin_sha             text        not null,
    protocol_sha        text        not null,

    -- L4.2. The chain. prev_hash of the first row is sixty-four zeroes.
    prev_hash           text        not null,
    row_hash            text        not null unique,

    inserted_at         timestamptz not null default now(),

    -- Cheap invariants, asserted in the database rather than trusted from the
    -- writer. The writer already checks them; two independent checks of the
    -- same claim is the project's standing pattern.
    constraint predictions_probabilities_sum
        check (abs(p_home + p_draw + p_away - 1.0) < 1e-9),
    constraint predictions_probabilities_range
        check (p_home > 0 and p_draw > 0 and p_away > 0),
    constraint predictions_rates_positive
        check (lambda_home > 0 and lambda_away > 0),
    -- L1.2, in the schema: the state cutoff precedes the fixture it predicts.
    constraint predictions_cutoff_precedes_fixture
        check (state_cutoff_date <= scheduled_date),
    constraint predictions_hashes_are_sha256
        check (row_hash ~ '^[0-9a-f]{64}$' and prev_hash ~ '^[0-9a-f]{64}$')
);

create index if not exists predictions_round_idx
    on public.predictions (round_id, scheduled_date);


-- ============================================================
-- 2. results
-- ============================================================
--
-- Populated after a matchweek, from the E0 file the pin declares as the
-- 2026-27 spine (P7.2). Scorelines only - no metric is computed here, and
-- L5.1 keeps the official result in phase6_score_holdout.py.

create table if not exists public.results (
    match_id            text primary key
        references public.predictions (match_id) on delete restrict,

    season              text        not null default '2026-2027',

    -- The date actually PLAYED, which under L3.1 may differ from the
    -- scheduled_date the prediction was written against.
    played_date         date        not null,

    home_goals          integer     not null check (home_goals >= 0),
    away_goals          integer     not null check (away_goals >= 0),
    result              char(1)     not null check (result in ('H', 'D', 'A')),

    -- L3.4. played_date - state_cutoff_date. Populated at capture, because at
    -- generation the date played is not yet known.
    state_age_days      integer     not null,
    -- L3.5. Pre-declared at 10 days, from the development distribution.
    is_stale            boolean     not null,

    -- L2. True when no pre-kickoff prediction existed for this match. Such a
    -- match is excluded from the LOG's figures and is never excluded from the
    -- holdout, which remains a date rule in a frozen document (L2.2).
    contaminated        boolean     not null default false,

    source_sha256       text        not null,
    captured_at_utc     timestamptz not null,
    inserted_at         timestamptz not null default now(),

    constraint results_result_matches_goals check (
        (home_goals > away_goals and result = 'H') or
        (home_goals = away_goals and result = 'D') or
        (home_goals < away_goals and result = 'A')
    )
);


-- ============================================================
-- 3. market
-- ============================================================
--
-- B3.3 fixes the primary book as BET365 CLOSING. B4.1 fixes the de-vig as
-- proportional. Both were chosen in PHASE5_MARKET_PREDECLARATION.txt before
-- any score existed, and neither is reconsidered here.
--
-- PINNACLE IS ABSENT FROM THE 2026-27 FILE (SW5-PSC) AND NO OTHER BOOK IS
-- PROMOTED TO REPLACE IT. REPORT.md section 7 records that as a reduction in
-- scope. Choosing a sensitivity instrument after seeing which one survived is
-- selection on the very thing a sensitivity exists to test.

create table if not exists public.market (
    match_id            text primary key
        references public.predictions (match_id) on delete restrict,

    season              text        not null default '2026-2027',
    book                text        not null default 'B365C',

    odds_home           double precision not null check (odds_home > 1.0),
    odds_draw           double precision not null check (odds_draw > 1.0),
    odds_away           double precision not null check (odds_away > 1.0),

    -- B4.1, proportional. Computed by phase5_market_benchmark.devig_proportional,
    -- imported rather than reimplemented.
    p_home              double precision not null,
    p_draw              double precision not null,
    p_away              double precision not null,
    overround           double precision not null,

    source_sha256       text        not null,
    captured_at_utc     timestamptz not null,
    inserted_at         timestamptz not null default now(),

    constraint market_probabilities_sum
        check (abs(p_home + p_draw + p_away - 1.0) < 1e-9),
    constraint market_book_is_the_declared_primary
        check (book = 'B365C')
);


-- ============================================================
-- 4. RLS
-- ============================================================
--
-- INSERT permitted. UPDATE and DELETE have NO POLICY AT ALL, which under RLS
-- means denied - there is nothing to grant them. Stated explicitly because
-- "no policy" reads like an omission and here it is the mechanism.

alter table public.predictions enable row level security;
alter table public.results     enable row level security;
alter table public.market      enable row level security;

-- Belt as well as braces: revoke the verbs outright, so a future permissive
-- policy added by accident still has no privilege to ride on.
revoke update, delete, truncate on public.predictions from anon, authenticated;
revoke update, delete, truncate on public.results     from anon, authenticated;
revoke update, delete, truncate on public.market      from anon, authenticated;

-- ---- read: the frontend, anon key, SELECT only -----------------------------
drop policy if exists predictions_read on public.predictions;
create policy predictions_read on public.predictions
    for select to anon, authenticated using (true);

drop policy if exists results_read on public.results;
create policy results_read on public.results
    for select to anon, authenticated using (true);

drop policy if exists market_read on public.market;
create policy market_read on public.market
    for select to anon, authenticated using (true);

-- ---- append: permitted ------------------------------------------------------
drop policy if exists predictions_append on public.predictions;
create policy predictions_append on public.predictions
    for insert to anon, authenticated with check (true);

drop policy if exists results_append on public.results;
create policy results_append on public.results
    for insert to anon, authenticated with check (true);

drop policy if exists market_append on public.market;
create policy market_append on public.market
    for insert to anon, authenticated with check (true);

-- NO UPDATE POLICY AND NO DELETE POLICY ON ANY OF THE THREE TABLES.
-- This blank space is the immutability guarantee. Do not fill it in.


-- ============================================================
-- 5. WHAT RLS DOES NOT DO, STATED BECAUSE IT IS A REAL LIMIT
-- ============================================================
--
-- L9.5. THE SERVICE ROLE BYPASSES RLS ENTIRELY. Everything above protects the
-- log from the frontend and from anyone holding the anon key. It does NOT
-- protect it from the key the generator itself writes with.
--
-- So the immutability of this log rests on three things, and RLS is only one:
--
--   1. the generator's own refusal to write a match_id that already exists,
--      which raises rather than skipping quietly (L1.4);
--   2. the L4 row-hash chain, which makes an edit, a deletion or a reorder
--      detectable after the fact by anyone holding the file;
--   3. the mirror in git, which is the primary record and is timestamped by
--      something other than the operator's own database.
--
-- A reader who trusts only this file should trust (2) and (3).

-- ============================================================
-- 6. THE INTEGRITY VIEW THE DASHBOARD SHOULD READ
-- ============================================================
--
-- So that a dashboard cannot show a running figure without also being able to
-- show how much of the holdout the log actually covers (L2.4, L5.2).

create or replace view public.live_log_coverage as
select
    (select count(*) from public.predictions)                        as predictions_written,
    (select count(*) from public.results)                            as results_captured,
    (select count(*) from public.results where contaminated)         as contaminated_matches,
    (select count(*) from public.results where is_stale)             as stale_matches,
    (select min(scheduled_date) from public.predictions)             as first_predicted,
    (select max(scheduled_date) from public.predictions)             as last_predicted,
    -- Informal, and the view says so in its own column name.
    'INFORMAL - the official 2026-27 result is phase6_score_holdout.py run '
    'once at season end (L5.1)'                                      as authority_note;

grant select on public.live_log_coverage to anon, authenticated;
