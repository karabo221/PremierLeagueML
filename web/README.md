# web/ — the prediction log and the evidence

A Next.js App Router site with three routes, deployed to Vercel.

| route       | what it is                                                              | data source                       |
| ----------- | ----------------------------------------------------------------------- | --------------------------------- |
| `/`         | This round's fixtures with their pre-kickoff probabilities              | compiled mirror + Supabase        |
| `/evidence` | The frozen development figures: ladder, deltas, gap diagnostic          | compiled only, no network         |
| `/log`      | Integrity: the hash chain, coverage, contamination, the known gaps      | compiled mirror + Supabase        |

## Where the numbers come from, and why it is split

Two kinds of figure with two different authorities, so they arrive by two routes.

**Frozen figures** are compiled into `lib/frozen.generated.ts` at build time by
`scripts/phase6_build_frontend_data.py`. They are read from the committed
artefacts in `outputs/`, they are hash-pinned in `FROZEN_MANIFEST.txt`, and they
cannot change — so putting them behind a network call would only add a way for
them to be unavailable. Every value carries the filename it came from, and the
UI prints that citation, the same way `REPORT.md` does.

**The predictions and results** are read from Supabase at request time, and
the pages revalidate every 600 seconds. That is what makes a new matchweek
appear on its own: the generator writes the round to the database, and the site
picks it up within ten minutes. **No deploy, no code push.**

Results and closing odds have no other source — L4.8 keeps 2026-27 scorelines
out of the repository until the holdout is deliberately acquired in May 2027.

The predictions also exist in the repository as the committed mirror, which is
the *primary* record under L9.4: git is what timestamps a pre-kickoff claim in a
way the operator cannot backdate. The site compiles it in as the **fallback**,
shown whenever the database is empty, behind or unreachable — and it says on the
page which of the two you are looking at, because a fallback that looks
identical to live data is how a stale figure gets read as a current one.

Regenerate the compiled fallback when you want it current — not required for the
site to show a new round, only for the offline copy to match:

```bash
npm run data          # python ../scripts/phase6_build_frontend_data.py
```

## The weekly routine

Two commands, both of which fetch their own data from football-data.co.uk. No
manual entry, and nothing to deploy.

```bash
# BEFORE the matchweek's first kickoff - writes the round to the mirror and,
# with credentials set, to Supabase. Refuses to rewrite a match it already has.
python scripts/phase6_generate_predictions.py --round

# AFTER the matchweek - fetches the scorelines and closing odds, writes them to
# Supabase only.
python scripts/phase6_capture_results.py
```

The generator needs `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` in the environment
to reach the database. Without them it still writes the mirror and says so — a
degraded run, not a failed one — but the site will then be showing the fallback
until the backfill catches the database up.

Committing the mirror afterwards is worth doing, and it is an *evidence* step
rather than a publishing one: the commit is what timestamps the claim. The
website does not wait for it.

## Local

```bash
npm install
npm run data
cp .env.local.example .env.local     # then paste the anon key
npm run dev
```

`npm run dev`, `build` and `start` all go through `scripts/next-with-shim.mjs`.
That is not optional on this machine — see **FAT32** below.

## Supabase

The site is **read-only**. It uses the anon/publishable key against the
select-only RLS policies in `ops/supabase_live_log.sql`. There is no write path
in this directory, and the service key must never appear here: the service role
bypasses RLS entirely, which section 5 of the DDL states outright.

```
NEXT_PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<the anon key>
```

Every read fails soft. Absent credentials, an unapplied schema or an empty
database render as a stated absence — never a crash, and never a zero that reads
like a measurement. That matters right now, because the log's first round ran
without credentials and **Supabase has never been written**; the run recorded it
at `L4d` in `live_log/phase6_live_log_run_audit.csv`.

### Before the first result can be captured

`public.results.match_id` is a foreign key onto `public.predictions`, so the
predictions have to exist in Supabase before `phase6_capture_results.py` can
insert a single scoreline. Two steps, in order:

The simplest route is one file, pasted into the Supabase SQL editor. It creates
the schema **and** inserts the predictions already written, and it is safe to run
twice:

```
ops/supabase_bootstrap.sql
```

Regenerate it any time with `python scripts/phase6_backfill_supabase.py --sql`.

Or, with credentials in the environment:

```bash
psql "$SUPABASE_DB_URL" -f ../ops/supabase_live_log.sql
python ../scripts/phase6_backfill_supabase.py --check   # verifies, writes nothing
python ../scripts/phase6_backfill_supabase.py --write
```

## Deploying to Vercel

The app is in a subdirectory of a Python repository, so the project root has to
be set explicitly.

1. Import the repository, set **Root Directory** to `web`.
2. Framework preset: Next.js. Build and output settings need no changes.
3. Environment variables: `NEXT_PUBLIC_SUPABASE_URL` and
   `NEXT_PUBLIC_SUPABASE_ANON_KEY`.
4. Deploy.

`lib/frozen.generated.ts` is committed, so Vercel does **not** need Python —
which is the point of compiling it here rather than fetching at runtime. You do
not need to redeploy for a new matchweek; see **The weekly routine** above.

`/` and `/log` are statically prerendered and revalidate every 600 seconds.
Results arrive at a matchweek's cadence, not a second's, so a tighter window
would spend invocations to show the same rows.

## FAT32

`E:` is FAT32, and Node returns `EISDIR` instead of `EINVAL` for `readlink` on a
regular file there. webpack treats `EINVAL` as "not a symlink, carry on" and has
no case for `EISDIR`, so the build dies naming a file that is plainly a regular
file. `fat32-readlink.cjs` corrects that one error code, preloaded via
`NODE_OPTIONS=--require` by `scripts/next-with-shim.mjs` — it has to be in place
before any module loads, because Next's bundled `graceful-fs` captures
`fs.readlink` at require time and a later patch has no effect.

It probes the filesystem first and does nothing on NTFS, so the same scripts
work on Vercel's Linux builders untouched. Full write-up in
`PROJECT_GOTCHAS.md` section 10.

## Crests

16 of the 20 2026-27 sides have a crest in `public/crests/`, sourced from
`E:\automation\channels\quiz-shorts\public\img\crests`. Four do not, and render
as monograms: **Nottingham Forest, Hull City, Ipswich Town, Coventry City**.
`lib/crests.ts` maps team names exactly as the live log spells them — the log's
spelling is what the pin declares, and a lookup miss here would be silent.
