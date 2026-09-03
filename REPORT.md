# Predicting Premier League match outcomes from pre-kickoff information

**A measured boundary, and what could not be established**

*Assembled 2026-09-03 from the session record in `REPORTS.md`. No analysis was
run for this document. Every figure below is read from a committed artefact and
cited to it; where the record does not support a claim, that is said in
section 9 rather than smoothed over.*

---

## 1. Question and scope

Given only information available **before kickoff**, how well can the outcome of
a Premier League match — home win, draw, away win — be predicted, and where does
the remaining error live?

The dataset is five complete seasons, **2021-22 to 2025-26, 1,900 matches**,
built from FBref fixture exports and validated before anything was fitted
(`outputs/phase1_matches.csv`; `outputs/phase1_match_foundation_audit.csv`).
Odds and match-detail columns come from football-data.co.uk
(`data/raw/Odds/E0_*.csv`), which is the only source in the project that arrived
over the network rather than on disk.

Everything is evaluated on the **1,520 outer-test matches** of a four-fold
walk-forward design (`outputs/phase0_evaluation_folds.csv`).

Three exclusions were taken as scoping decisions at the outset and held
throughout: no lineup or injury data, no xG, no betting backtests. The first of
these is the one that matters most for reading section 8.

---

## 2. Method

The method is the part of this project most worth reporting, because most of the
results are negative and a negative result is only worth as much as the design
that produced it.

### 2.1 Walk-forward evaluation on four frozen folds

Folds were fixed in Phase 0, before any model existed, and never moved
(`outputs/phase0_evaluation_folds.csv`):

| fold | train seasons | train | test season | test |
|---|---|---|---|---|
| 1 | 2021-22 | 380 | 2022-23 | 380 |
| 2 | 2021-22 → 2022-23 | 760 | 2023-24 | 380 |
| 3 | 2021-22 → 2023-24 | 1140 | 2024-25 | 380 |
| 4 | 2021-22 → 2024-25 | 1520 | 2025-26 | 380 |

Every fold trains only on seasons that finished before its test season began;
`temporal_order_valid` and `overlap_valid` are recorded True for all four in the
artefact itself. **1,520 outer-test matches** is the population every pooled
figure in this report is computed on, and it is the same 1,520 rows for every
model, which is what makes the deltas paired.

The two continuously-updated models — Poisson and Dixon-Coles — additionally
refit **once per distinct match date** inside each fold, on a window that is
strictly `date < cutoff`, so a match is never inside the window that predicts
it, and same-day fixtures cannot see each other. Matches are weighted by
`0.5 ** (age_days / 107)` measured from the cutoff, not from a season start
(`scripts/phase2_poisson_dixon_coles.py`; the constants are asserted against the
freeze document at V2a/V2b and the estimator's own source at V3a–V3l,
`outputs/phase6_freeze_validation.csv`).

### 2.2 Pre-declared gates

Each instrument was specified in a pre-declaration written and hashed **before**
the instrument was run: the questions, the thresholds, the branches for reading
each possible outcome, and what would count as a failure. Those documents are
`PHASE3_*`, `PHASE4_*` and `PHASE5_*` in the repository root, and their hashes
are in `FROZEN_MANIFEST.txt`, so an edit after the fact is detectable rather
than a matter of trust.

The value of this is visible only in the failures. Three gates stand failed in
the record because a claim was pitched too high; none was softened after the
fact (section 7).

### 2.3 The metric set, and RPS as primary

Six metrics, fixed in Phase 0 (`outputs/phase0_evaluation_spec.csv`): accuracy,
balanced accuracy, macro F1, log loss, Brier score, and **RPS**, which respects
the H > D > A ordering. Accuracy is reported but never alone.

The primary reading is **RPS with log loss**, under a declared
**sign-agreement rule**: if the two disagree in sign, the comparison is reported
`INCONCLUSIVE` rather than resolved in favour of whichever number is convenient.
The rule is not decorative — E1c − Elo v1 fired it
(`outputs/phase5_e1c_deltas.csv`).

### 2.4 Paired per-match bootstrap

Every comparison in this report is a **paired bootstrap over the 1,520 matched
per-match scores, 10,000 draws, seed 20260901** — the same draws and the same
seed for every delta in the project (validator V4b). Pairing is what makes a
0.003 difference readable at all: the two models are scored on identical rows,
so the shared difficulty of a season cancels.

### 2.5 The leakage suite

Nine tests, run over all 1,900 matches before modelling
(`outputs/phase0_leakage_audit.csv`): every match date parses; no historical
entry is dated on or after its match; same-day matches are excluded from each
other's history (1,706 affected); cold-start matches are identified rather than
back-filled (50); an unfiltered season join is quantified to show what it would
have admitted (356,430 rows); season-aggregate tables are shown to postdate
every match in their season (1,900); final season points is constant across a
season, which is the signature of that leak (100 team-seasons); pre-match
aggregates start empty and never exceed the season, with 0 monotonic failures;
and a match's own scoreline is shown unable to reach its own predictors.

### 2.6 The frozen manifest

Every pre-declaration, every frozen artefact and the odds source itself are
hashed into `FROZEN_MANIFEST.txt` (`scripts/frozen_manifest.py`). A figure that
moves is caught by hash rather than by memory. This machinery detected a
last-ULP arithmetic difference between machines (section 7.3), which is the
level of sensitivity it operates at.

---

## 3. The results ladder

All figures pooled over the same 1,520 outer-test matches. Lower is better.

| model | log loss | RPS | source artefact |
|---|---|---|---|
| D0 — base rate | 1.06889 | 0.23185 | `phase4_ladder_pooled.csv` |
| D1 — results-derived features | 1.00393 | 0.20769 | `phase4_ladder_pooled.csv` |
| D2 — + dynamic state | 1.00086 | 0.20667 | `phase4_ladder_pooled.csv` |
| D2 rescaled (Amendment 4) | 1.00028 | 0.20646 | `phase4_d34_pooled.csv` |
| D3 — + context (Block C) | 1.00125 | 0.20662 | `phase4_d34_pooled.csv` |
| D4 — + prior-season FBref (Block X) | 0.99975 | 0.20619 | `phase4_d34_pooled.csv` |
| Elo v1 | 0.99943 | 0.20667 | `phase4_ladder_pooled.csv` |
| Poisson walk-forward | 0.99042 | 0.20355 | `phase4_ladder_pooled.csv` |
| **Dixon-Coles walk-forward** | **0.99036** | **0.20350** | `phase2_poisson_dc_fold_summary.csv` |
| E1a — shots-on-target ratings | 0.98124 | 0.20117 | `phase5_e1a_pooled.csv` |
| E1b — shot residual | 0.99758 | 0.20568 | `phase5_e1b_pooled.csv` |
| E1c — finishing residual | 0.99956 | 0.20637 | `phase5_e1c_pooled.csv` |
| market — Bet365 closing | 0.96057 | 0.19469 | `phase5_market_pooled.csv` |

The full metric set for the three that matter most
(`outputs/phase5_market_pooled.csv`):

| model | acc | bal. acc | macro F1 | log loss | Brier | RPS |
|---|---|---|---|---|---|---|
| D0 | 0.44474 | 0.33333 | 0.20522 | 1.06889 | 0.64667 | 0.23185 |
| Dixon-Coles | 0.52632 | 0.45106 | 0.39174 | 0.99036 | 0.58969 | 0.20350 |
| market | 0.55132 | 0.47245 | 0.40948 | 0.96057 | 0.57019 | 0.19469 |

### 3.1 The deltas, with intervals

| comparison | Δ log loss | 95% CI | Δ RPS | verdict | artefact |
|---|---|---|---|---|---|
| D1 − D0 | −0.06496 | [−0.08121, −0.04938] | −0.02416 | significant | `phase4_ladder_deltas.csv` |
| D2 − D1 | −0.00307 | [−0.00407, −0.00208] | −0.00102 | significant | `phase4_ladder_deltas.csv` |
| D2 rescaled − D1 | −0.00365 | [−0.00489, −0.00241] | −0.00123 | significant | `phase4_a4_deltas.csv` |
| D3 − D2 rescaled | +0.00097 | [−0.00082, +0.00277] | +0.00016 | **not significant** | `phase4_d34_deltas.csv` |
| D4 − D3 | −0.00150 | [−0.00448, +0.00148] | −0.00043 | **not significant** | `phase4_d34_deltas.csv` |
| D4 − D2 rescaled | −0.00053 | [−0.00363, +0.00259] | −0.00027 | **not significant** | `phase4_d34_deltas.csv` |
| E1a − Dixon-Coles | −0.00912 | [−0.02177, +0.00391] | −0.00233 | **not significant** | `phase5_e1a_deltas.csv` |
| E1b − D2 rescaled | −0.00270 | [−0.00536, +0.00001] | −0.00078 | **not significant** | `phase5_e1b_deltas.csv` |
| E1c − D2 rescaled | −0.00072 | [−0.00229, +0.00084] | −0.00009 | **not significant** | `phase5_e1c_deltas.csv` |
| market − Dixon-Coles | −0.02979 | [−0.04229, −0.01759] | −0.00881 | significant | `phase5_market_deltas.csv` |
| market − D0 | −0.10831 | [−0.12806, −0.08891] | −0.03716 | significant | `phase5_market_deltas.csv` |

### 3.2 The decomposition

Of the whole D0-to-Dixon-Coles distance of **0.0785** in log loss
(`REPORTS.md` Phase 4 conclusion, from `phase4_ladder_pooled.csv`,
`phase4_a4_deltas.csv` and `phase4_d34_deltas.csv`):

| component | log loss | share |
|---|---|---|
| current-season results (D0 → D1) | 0.0650 | **83%** |
| continuously updated rating state (D1 → D2 rescaled) | 0.0036 | **5%** |
| everything still between D2 rescaled and Dixon-Coles | 0.0099 | **13%** |
| static historical description (Blocks C and X) | — | **nothing measurable** |

The 13% residual is **not significant on 1,520 matches**
(D2 rescaled − Dixon-Coles = +0.00992 [−0.00330, +0.02314],
`phase4_a4_deltas.csv`). That is resolution, not equality: it is unmeasured,
not shown to be absent.

Two things follow, and both are uncomfortable for the engineering.

**Elo v1 — a single K=20 rating with a flat 1500 start and a 60-point home
advantage — has a lower pooled log loss than every rung on the ladder, D4's 139
columns included** (0.99943 against D4's 0.99975), and neither metric separates
them (D4 − Elo v1 = +0.00032 [−0.00746, +0.00782]). Recency-weighted strength is
most of the signal, and 139 engineered columns are a less efficient way of
writing it down than one rating is.

**A separate decomposition isolates why.** The tier-2 instrument
(`outputs/phase4_tier2_decomposition.csv`) splits the walk-forward advantage
into *recency* and *training-set size*: recency accounts for
0.03557 [0.01767, 0.05333] of a total 0.03590, or **99%**, while the sample-size
term is 0.00034 [−0.00071, +0.00141] and does not clear zero. It is the freshness
of the state that pays, not the amount of history behind it.

---

## 4. The market benchmark and the gap diagnostic

### 4.1 The benchmark

The market's figure was an assumption in this project for a long time — prose in
Phase 4 recorded "the market's ~0.95 is out of reach by construction", which had
never been measured on these matches. Measured, on the same 1,520 rows, it is
**0.96057 log loss, 0.19469 RPS** (`outputs/phase5_market_pooled.csv`). The
assumption was optimistic by about 0.01.

Bet365 closing was chosen as primary **before any score was computed**, on
completeness alone: it is 380 of 380 in all five seasons, where Pinnacle covers
1,350 of 1,520 with all 170 gaps in fold 4, and the market average is an average
over a different panel of bookmakers each season. It is not a claim that Bet365
is the sharpest line. It does not matter which was picked: the four
book × de-vig combinations land within 0.0006 of each other (0.96001 to 0.96057),
and Pinnacle scored per fold on exactly the rows it covers is indistinguishable
from Bet365 on those same rows.

**The comparison is biased in the project's favour**, and this is the honest
frame for the 0.02979: 2025-26 was scored during Phase 3's lambda sweep, so every
model figure here is a walk-forward *development* estimate. The market has never
been fitted to anything. The project loses by 0.03918 at D4 with the thumb on
its own side of the scale.

### 4.2 The gap is uniform

`scripts/phase5_gap_diagnostic.py`, run after E1c's pre-declaration was hashed
and committed, exploratory, no gate, nothing from it enters a model
(`outputs/phase5_gap_splits.csv`, `outputs/phase5_gap_correlations.csv`).

Every split that was looked at is reported, including the null ones:

| split | read |
|---|---|
| favourite probability (5 bins) | non-monotonic, every CI contains the pooled figure — **null** |
| matchweek (4 blocks) | +0.04255 / +0.02976 / +0.02412 / +0.02863 — **null** |
| season | +0.03627 / +0.04176 / **+0.00889** / +0.03224 — the only split with any separation |
| promoted side involved | +0.03022 vs +0.02871 — **dead null** |
| market pick | H +0.03622 vs A +0.01883 — mild |
| actual outcome | A +0.00270 [−0.02074, +0.02617] · H +0.03869 · D +0.04874 — the sharpest |
| correlation over 128 numeric on-disk columns | largest \|r\| **0.0785**; 20 clear the naive 0.0503 threshold where **6.4** are expected by chance |

Nothing on disk explains more than 0.6% of the variance of the per-match gap.
The gap's per-match standard deviation is 0.24897 against a mean of 0.0298 — its
spread dwarfs its mean, which is why every subset carries a wide interval and
why "uniform" is the reading rather than "flat everywhere by coincidence".

### 4.3 The sharpness finding

Added because the reading above is otherwise inferable and confidence is
directly measurable — and it **corrected** the first reading
(`outputs/phase5_calibration.csv`):

| | mean max p | p on market's pick | mean p(D) |
|---|---|---|---|
| market | 0.5391 | 0.5391 | 0.2366 |
| Dixon-Coles | **0.5489** | 0.5316 | 0.2296 |
| E1a | 0.5175 | 0.5047 | 0.2358 |

**Dixon-Coles is more confident than the market, not less.** It is confident
about a different outcome. The market's edge over Dixon-Coles is **direction,
not calibration**.

A calibration audit was run over ten fixed bins declared before the curve was
seen, with no recalibration fitted — Platt or isotonic on the outer-test rows
would be fitting on the rows being scored. It decides nothing, and the reason is
worth stating: **D0 has the best calibration in the project (ECE 0.00760) and is
the worst model in it.** A constant base-rate prediction is almost perfectly
calibrated by construction and carries no information. Two findings survive that
caveat: every model under-predicts draws, market included (`bias D` negative for
all twelve entries), and the market shows the textbook favourite–longshot bias,
which the Shin de-vig corrects.

---

## 5. The refuted hypothesis

The lineup story — that the model's remaining error is missing team-news
information — makes a directional prediction: **the model should lose most where
a strong side underperforms its rating.**

It does not. Conditioning on whether the favourite delivered, the Dixon-Coles
gap to the market is **+0.03866 when the favourite delivers against +0.01889
when it does not** (`outputs/phase5_gap_splits.csv`). That is the opposite sign
to the prediction. On the stronger split — a favourite at ≥0.60 — the same
reversal holds, +0.03393 against +0.01726.

**The hypothesis was recorded in advance**, in the gap diagnostic's own framing,
which is the only reason this reads as a refutation rather than as one more
subgroup. The split conditions on the outcome and can therefore never become a
feature; it describes, and that limit is declared with it.

This does not establish that lineup information is worthless. It establishes
that *this particular mechanism* — losing on games where a favourite
underperforms — is not where the measured gap sits.

---

## 6. What makes the negatives readable

Most of this project's results are nulls. A null is only worth reading if the
instrument that produced it would have detected the alternative. This section is
the evidence for that, and it is the reason the report is worth publishing.

### 6.1 Three gates stand failed because the claim was too strong

None of the three was softened, and each was left failing in a committed audit
artefact where anyone can see it.

**C3a and C3b** (`outputs/phase3_ceiling_audit.csv`). C3 asserted the
regularisation grid reaches the degenerate end: C3a required the coefficient
norm at the top of the grid to be under 0.10 of its value at λ=100, and observed
0.1212; C3b required every rung to be closer to B0 at λ=10000 than at λ=100, and
B6 was not. The threshold had been justified by an argument that ridge norms
decay as 1/λ, predicting about 0.01 over two decades. That argument does not hold
at n=1,520, where at λ=10000 the penalty is still only comparable to the Fisher
information — the fit is at the *start* of the asymptotic regime. The observed
decay is 0.4802 → 0.0582, roughly eightfold. Resolving the ceiling question only
ever required the grid to *bracket* an interior optimum, which it does for all
six featured rungs. **Lowering the threshold after seeing 0.1212 would have been
exactly the failure these documents exist to prevent.** The failures stand.

**G9** (`REPORTS.md`, Phase 4 Amendment 4). The declared form was "bit for bit",
which is too strong for a design that adds four all-zero columns and therefore
solves a 279×279 Newton system where the control solves 267×267. The measured
disagreement in the shared coefficients is 6.9e-18. **The wording was ours, and
A5.2 was not amended to soften it.**

**F5** (`outputs/phase5_e1c_audit.csv`, observed 6.984e-01). Its replacement,
F5b, tests the claim the design actually makes — corrupt a fold's *own* test
season and that fold's c must not move — and passes at 0.000e+00. F5b carries a
disclosure in its own audit row: written after seeing F5 fail, carrying no
threshold, and reading identically had F5 passed. An INFO row records the
diagnosis beside the failing gate, so the audit CSV is not misleading when read
alone.

### 6.2 The base-contamination bug

This is the single most instructive failure in the record, because the run would
have completed and reported a clean null.

`block_of()` returns `phase1_backbone` for any column name it does not
recognise. `d1_features()` selects the backbone **by exclusion**. E1b attached
six new residual columns to the feature frame before the base rung was read — so
those six were classified as backbone by default and swept into the *control
arm itself*. D2 rescaled came out 98 columns wide, stopped reproducing the
committed Amendment 4 artefact (7.906e-03 against a 1e-12 tolerance), and E1b
"added" nothing because there was nothing left for it to add.

**Four gates fired at once** — E10, E10c, E1b-A1 and DS3a — and E10e now asserts
the base directly. Without them the comparison still runs: it compares a rung
against itself, reports a null, and the null is an artefact of feature-frame
ordering. **A default-to-known classifier combined with a select-by-exclusion
rule is a silent contamination machine**, and nothing about the output would
have looked wrong.

### 6.3 The AVX-512 arithmetic difference, isolated by running a control

Six frozen hashes moved while the instrument itself reported zero failures and
every printed number was the committed one
(`PROJECT_GOTCHAS.md`). Measured on `phase5_e1b_shot_residuals.py`: log loss
`1.0060002752144963` → `…61`, pooled RPS `0.20762578612240679` → `…676`, Newton
residual `9.1325613738035827e-11` → `9.1323837381196427e-11`; λ, EPV, G6, design
width and accuracy unchanged. Last-ULP on the solutions.

The cause is that numpy dispatches kernels on CPU features, so summation order
inside a reduction differs between machines; the laptop reports
`{'baseline': ['X86_V2'], 'found': ['X86_V3']}` with AVX-512 live, and seven or
eight Newton iterations amplify a last-ULP difference into the fifth significant
figure of a residual that is itself 1e-11.

**The method of isolating it is the part that generalises.** Comparing a
modified run against the *committed* artefact cannot separate "my change" from
"this machine", because both are confounded in the same diff. The unmodified
code was run on the same machine as a control, and produced hashes identical to
the hardened run and different from committed — proving the hardening
bit-transparent and the movement environmental. A committed artefact is not a
control when the machine is one of the variables.

### 6.4 The λ-label migration, caught only by testing the round trip

The defect: `surface["lambda"] == 0.03` silently matched zero rows, because
reading the CSV without `float_precision="round_trip"` returns a value one ULP
from the literal.

The fix: write a string label beside every λ column so consumers key on a string.
**The first attempt wrote the bare text** — `"0.03"`, `"1"`, `"10000"`. It reads
correctly in the file and passes any visual inspection. It was wrong: a CSV
column of numeric-looking strings is **type-inferred straight back to float64 on
read**, so the column would have been the identical float key it was written to
replace — while *looking* fixed, which is strictly worse than the original bug.

It was caught only because the migration's verifier tested the **round trip**
(write, re-read, compare) rather than the write. The fix is now `lam=0.03`, whose
prefix is load-bearing: no parser can coerce it to a number.

**When you fix a defect, check that your verification does not depend on the
same assumption the defect did.**

### 6.5 Phase 3's Amendment 2, disclosed rather than corrected

`PHASE3_CEILING_PREDECLARATION.txt`, Amendment 2, **changes an outcome** — from
BLOCKED with no frozen scalar to VALID with λ=1000 — **and it was written after
the result was known.** That is the opposite of the ordering the
pre-declarations exist to enforce, so the amendment records, in full, what had
been seen when it was written: the complete selection table, the outer-test
surface, the 13-point grid, that every featured rung's optimum is interior at
λ=1000, the rung verdict, and both C3 failures.

It also records who decided. The amendment was **not** applied on the agent's
own judgement: it was put to the project owner with the literal BLOCKED outcome,
the option of accepting BLOCKED unchanged, and a third option of freezing
per-rung; the owner chose the amendment.

The argument offered for it being non-outcome-driven, left for the reader to
weigh: a rung with no penalised parameters has a constant λ-curve on *any* data,
so its selection is always the tie-break's output and never evidence about the
optimum — the amended rule would have been correct before any number was seen.
And it does not disarm the alarm it was written for: had B1–B6 themselves
selected 10000, the status would still be BLOCKED.

**The frozen scalar it produced is not load-bearing anywhere downstream.**
`outputs/phase3_frozen_regularisation.json` is read by no instrument other than
the one that wrote it; every ladder rung from D1 onward selects its own λ per
fold. So the amendment's practical consequence, in the end, was nil.

### 6.6 Two more checker bugs, both ours

Recorded because they are the same class and the count matters: **DS2b**
computed a sample SD where Amendment 4 declares median/IQR (off by 29.0), then
imputed with the mean where the declaration says median (off by 1.0); **F9**
flagged all six of E1c's own columns because it tested whether a design column
*contained* `"shin"` and "fini**shin**g" does — six false positives, zero odds
columns.

Neither was a pipeline defect. Both were verifiers that did not implement the
rule they were verifying. **A verifier that does not implement the declared rule
cannot verify it** — three times in this project.

---

## 7. Limitations

Stated plainly, and none of them is repaired by anything above.

**There is no clean holdout in this report.** 2025-26 was scored during Phase 3's
lambda sweep and its B0–B6 ablation. **Every figure in this document is a
walk-forward development estimate, optimistic by an unknown margin.** None of it
may be described as an out-of-sample result. Section 8 exists because of this
sentence.

**Elo is under-converged league-wide**, so D2 − D1 stands as a **lower bound** on
what dynamic state is worth, not as its value.

**No lineup or injury data.** The market's 0.96057 is out of reach **by
construction**, not by shortfall. That was a scoping decision, and the distance
to the market is not evidence that the modelling is deficient.

**D2-static is deferred with G10 failing.** `PHASE4_AMENDMENT6_D2STATIC.txt`
records the defect, the framing error and the only design that would make the
rung readable. It is not a result that is being withheld; it is a rung that was
not built.

**Resolution on 1,520 matches is roughly 0.005 in log loss.** Several results in
section 3 are "not detected", not "absent" — D3, D4, E1a, E1b and E1c all fall
in that category, and the 13% unexplained residual in the decomposition does
too. A confidence interval covering zero is the resolution of the test, not a
proof of equivalence.

**The gap diagnostic has no multiplicity control** and cannot identify a cause. A
gap that sits everywhere is consistent with many mechanisms.

**One season of the finishing question is unanswerable with this dataset.** The
twenty-match persistence rung is not underpowered but **unrunnable**: two
non-overlapping twenty-match windows need 40 matches and a season has 38, so the
rung yields zero pairs at every fold at any sample size
(`outputs/phase6_persistence20.csv`,
`outputs/phase6_persistence20_feasibility.csv`). Only windows of 5 and 6 clear
the power requirement on this dataset. That question needs more seasons, not a
different window.

---

## 8. The pending holdout

Every figure above is a development estimate. **One clean measurement of how
optimistic they are is now frozen and pending.**

`PHASE6_HOLDOUT_FREEZE.txt` (sha256 `b36befd3…248f6`, 397 lines, unmodified)
declares one model — Dixon-Coles walk-forward — with every constant that makes
it reproducible, and a cutoff *rule*. It freezes Dixon-Coles rather than E1a,
whose point estimate is lower at 0.98124, because E1a − Dixon-Coles is
−0.00912 [−0.02177, +0.00391], **not significant**: freezing E1a would mean
selecting on an unresolved difference using the very 1,520 matches whose
optimism the holdout exists to measure.

`PHASE6_CUTOFF_PIN.txt` pins the cutoff at **2026-09-04** (Ipswich Town v
Liverpool), evaluated from the published fixture list on 2026-09-03 and absolute
from that evaluation. Roughly **360 matches** are expected to be scored;
matchweeks 1 and 2 warm the state without being scored.

The scoring instrument was built in September 2026 rather than May 2027, so that
it could be validated where the answers are already frozen. Its dry run
reproduces the development figures at 0.00e+00 on every comparison
(`outputs/phase6_dryrun_summary.csv`, `outputs/phase6_scoring_audit.csv`), and
it refuses to score the live season without two explicit flags.

**This section will be completed at the end of the 2026-27 season, and its
result stands whatever it says.** If the holdout figure is materially worse than
0.99036, the development estimates in this report were optimistic and the size
of the optimism is the difference — read against the market on the same matches
as a control for how hard the season was. The freeze records in advance that
19.5% of a 2026-27 season involves a club with no history in the dataset, so
promotion churn and development optimism are **not separable by this design**;
the split is reported so the size of that confound is visible rather than argued
about afterwards.

The holdout also cannot resolve a small optimism: one season of ~360 matches
gives worse resolution than the 1,520 development matches did. It can resolve a
large one, and it is the only clean measurement this project will ever have.

---

## 9. What is not fully traceable

Three claims in the record could not be tied to a committed artefact as stated,
and are corrected here rather than repeated.

1. **"Amendment 2 was superseded by D1"** does not appear in `REPORTS.md` or the
   pre-declarations. What *is* checkable is stronger and narrower: no instrument
   other than `phase3_regularisation_ceiling.py` reads
   `outputs/phase3_frozen_regularisation.json`, and every ladder rung selects its
   own λ per fold — so the amendment's frozen scalar never became a constant
   anywhere downstream. Section 6.5 states it that way.

2. **The "eleven odds column names"** in the record
   (`REPORTS.md`, F9's checker bug) are the eleven columns of the *derived*
   market artefact — `overround`, `shin_z`, `price_*`, `prop_p_*`, `shin_p_*` —
   not eleven raw columns of the source file. The project consumes **nine** raw
   book columns: `B365CH/D/A`, `AvgCH/D/A`, `PSCH/D/A`
   (`scripts/phase5_market_benchmark.py`, `BOOKS`).

3. **The 83/5/13 decomposition** is computed against **D2 rescaled**
   (Amendment 4), not against the original D2. Against the original D2 the split
   is 83 / 4 / 13. Both rungs are in the ladder table; the row used is named in
   section 3.2.

---

## Evidence

Pre-declarations and freezes, all hashed in `FROZEN_MANIFEST.txt`:
`PHASE3_REGULARISATION_PREDECLARATION.txt`, `PHASE3_CEILING_PREDECLARATION.txt`,
`PHASE4_TIER2_WINDOW_PREDECLARATION.txt`, `PHASE4_D2_PREDECLARATION.txt`,
`PHASE4_AMENDMENT6_D2STATIC.txt`, `PHASE5_MARKET_PREDECLARATION.txt`,
`PHASE5_XG_PREDECLARATION.txt`, `PHASE5_E1_SHOT_PREDECLARATION.txt`,
`PHASE5_E1C_FINISHING_PREDECLARATION.txt`, `PHASE6_HOLDOUT_FREEZE.txt`,
`PHASE6_CUTOFF_PIN.txt`.

Result artefacts cited above: `phase0_evaluation_folds.csv`,
`phase0_evaluation_spec.csv`, `phase0_leakage_audit.csv`, `phase1_matches.csv`,
`phase1_match_foundation_audit.csv`, `phase2_poisson_dc_fold_summary.csv`,
`phase3_ceiling_audit.csv`, `phase3_ceiling_verdict.csv`,
`phase4_ladder_pooled.csv`, `phase4_ladder_deltas.csv`, `phase4_a4_deltas.csv`,
`phase4_d34_pooled.csv`, `phase4_d34_deltas.csv`,
`phase4_tier2_decomposition.csv`, `phase4_tier2_pooled.csv`,
`phase5_market_pooled.csv`, `phase5_market_deltas.csv`, `phase5_calibration.csv`,
`phase5_gap_splits.csv`, `phase5_gap_correlations.csv`, `phase5_e1a_pooled.csv`,
`phase5_e1a_deltas.csv`, `phase5_e1b_pooled.csv`, `phase5_e1b_deltas.csv`,
`phase5_e1c_pooled.csv`, `phase5_e1c_deltas.csv`, `phase5_e1c_audit.csv`,
`phase6_dryrun_summary.csv`, `phase6_scoring_audit.csv`,
`phase6_freeze_validation.csv`, `phase6_persistence20.csv`,
`phase6_persistence20_feasibility.csv`, `phase6_source_watch_2026-09-03.csv`.

Session narrative: `REPORTS.md`. Environment and failure modes:
`PROJECT_GOTCHAS.md`.

---

## The result, stated once

Within the information classes this project could access and did test — current
season results, continuously updated rating state, static historical
description, shot volume, shot-on-target-derived ratings, and an isolated
finishing residual — **the best model reaches 0.99036 log loss and 0.20350 RPS
against a market benchmark of 0.96057 and 0.19469 on the same 1,520 matches, and
no statistically convincing explanation was found for the remaining 0.02979.**

The gap is uniform across every subset examined; no variable on disk explains
more than 0.6% of its per-match variance; the model is *more* confident than the
market rather than less, so the market's edge is direction and not calibration;
and the one hypothesis recorded in advance — missing lineup information — makes a
directional prediction the data contradicts.

That is a boundary, measured. It is not a claim that the gap is unclosable, and
it is not a claim that this project succeeded or failed. Several of the nulls
above are "not detected at a resolution of 0.005", and the whole ladder is a
development estimate whose optimism has not yet been priced. Section 8 is where
that changes, once.
