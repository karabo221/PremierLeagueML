# Reports

One entry per completed instrument, newest first. Each states what was
measured, what came back, and what it does not license.

The rule this file exists for: **a number without its interval and its gate
status is not a result.** Every figure below is on the four frozen outer folds
and the same 1,520 outer-test matches unless it says otherwise.

---

## Phase 5 — Instrument E1c: the isolated finishing residual, and where the market gap lives

*Run 2026-09-03. `scripts/phase5_e1c_finishing.py`, **59 checks, 56 PASS, 2 INFO,
1 deliberate FAIL**. Pre-declaration `PHASE5_E1C_FINISHING_PREDECLARATION.txt`,
sha256 `416a6816…0ff2`, signed off unamended before the instrument was written
and before the gap diagnostic below was run. Alongside it, an exploratory
diagnostic with no gate: `scripts/phase5_gap_diagnostic.py`.*

### The rung, and what it settles

E1b's declared column was SoT differential minus goal differential. Since goals
run at ≈0.327 per shot on target, its expectation is ≈0.673 × SoT differential —
**substantially a restatement of shot volume**, which the E1 report said in
advance was its limitation. E1c subtracts the conversion instead of unity:
`goal_diff − c·SoT_diff`, whose expectation is zero by construction.

**It worked, and F7 measures how well.** Correlation with the same five-match
SoT differential:

| column | r | r² |
|---|---|---|
| E1b `rel_sot_residual_diff` | +0.9031 | **0.8155** |
| E1c `rel_finishing_diff` | +0.0876 | **0.0077** |

E1b's column was 82% shot differential. E1c's is 0.8%.

**E1c − D2 rescaled = −0.00072 [−0.00229, +0.00084], not significant.**
Per fold: +0.00086, +0.00011, −0.00152, −0.00233.

| model | log loss | RPS | Brier |
|---|---|---|---|
| market (Bet365 closing) | 0.96057 | 0.19469 | 0.5702 |
| E1a — SoT ratings | 0.98124 | 0.20117 | 0.5834 |
| Dixon-Coles | 0.99036 | 0.20350 | 0.5897 |
| E1b | 0.99758 | 0.20568 | 0.5950 |
| Elo v1 | 0.99943 | 0.20667 | 0.5958 |
| **E1c** | **0.99956** | **0.20637** | 0.5964 |
| D2 rescaled | 1.00028 | 0.20646 | 0.5970 |

| comparison | Δ log loss | 95% CI | Δ RPS | verdict |
|---|---|---|---|---|
| **E1c − D2rescaled** | **−0.00072** | **[−0.00229, +0.00084]** | −0.00009 | **not significant** |
| E1c − E1b | +0.00198 | [+0.00007, +0.00389] | +0.00069 | significant |
| E1c − E1a | +0.01832 | [+0.00552, +0.03111] | +0.00520 | significant |
| E1c − Dixon-Coles | +0.00920 | [−0.00412, +0.02236] | +0.00287 | not significant |
| E1c − Elo v1 | +0.00013 | [−0.00831, +0.00838] | −0.00030 | **INCONCLUSIVE** (signs disagree) |
| E1c − D0 | −0.06933 | [−0.08621, −0.05341] | −0.02549 | significant |
| E1c − market | +0.03899 | [+0.02770, +0.05030] | +0.01167 | significant |

### The prediction resolved on branch F4.3, as declared

F4.2 named two consequences of "five-match finishing is substantially its own
sampling noise". Both hold.

**(a) is null**, above. **(b) is near zero.** The correlation between a team's
finishing over one five-match window and its finishing over the next,
non-overlapping five — training rows only, each fold with its own c:

| fold | training seasons | n pairs | r |
|---|---|---|---|
| 1 | 2021-22 | 120 | +0.0481 |
| 2 | + 2022-23 | 240 | +0.0857 |
| 3 | + 2023-24 | 360 | −0.0060 |
| 4 | + 2024-25 | 480 | +0.0313 |

Largest |r| **0.0857 against the declared threshold of 0.10**. The mechanism's
second consequence holds and the column does not predict its own next window.

**So F4.3 fires: the column is noise, the mechanism is confirmed, and shot
information is exhausted at this resolution.** F0.4 declared E1c the last
shot-derived rung so that a null could not be followed by a fourth variant.
**The phase closes on that.**

### The decomposition this makes possible

E1b and E1c are the same block with and without its shot-volume component.

| | r² with SoT differential | Δ vs D2 rescaled |
|---|---|---|
| E1b — volume included | 0.8155 | −0.00270 |
| E1c — volume removed | 0.0077 | −0.00072 |

**Roughly three quarters of E1b's effect went with the volume.** E1b was
measuring shot differential, not finishing — which the E1 report named in
advance as the obvious reading and which is now measured rather than argued.
E1c − E1b = +0.00198 [+0.00007, +0.00389] says the same thing from the other
side: the volume version is significantly better than the finishing version.

The whole shot arm is monotone — **E1a 0.98124 < E1b 0.99758 < E1c 0.99956 <
D2 rescaled 1.00028**. The useful part of shot information is volume, and it is
worth most *inside* the rating rather than beside it.

### Both declared sensitivities are immaterial

| variant | log loss | RPS | vs primary |
|---|---|---|---|
| E1c primary — robust scaling, decayed c | 0.99956 | 0.20637 | — |
| F3.4 — section 6 scaling instead | 0.99958 | 0.20636 | +0.00002 |
| F1.6 — c with no decay weighting | 0.99960 | 0.20638 | +0.00003 |

Both move the fifth decimal. F3.1's departure from E1b's scaling rule was
declared in advance *because* it was a departure; it turns out not to be
load-bearing, which is the cleanest way for a declared departure to end.

### F5 FAILS, and the wording was ours

F5 declared: corrupt **every** outer-test row and no fold's c may move. It moves
by 0.698. **That is not a leak — the claim was too strong.**

The folds are nested walk-forward. Fold 2 trains on 2021-22 + 2022-23, and
2022-23 is fold 1's *test* season. An earlier fold's test season is a later
fold's training data by the design of the frozen folds, so **D1, D2, E1a and
E1b would every one of them fail F5 as worded.**

| fold | seasons in its c window | own test season present |
|---|---|---|
| 1 | 2021-22 | **False** |
| 2 | 2021-22, 2022-23 | **False** |
| 3 | 2021-22 … 2023-24 | **False** |
| 4 | 2021-22 … 2024-25 | **False** |

**F5 is left failing rather than softened**, exactly as G9 was when its "bit for
bit" wording proved too strong for a design that solves a larger Newton system.
`F5b` tests the claim the design actually makes — corrupt a fold's *own* test
season, that fold's c must not move — and passes at **0.000e+00**. F5b carries a
G10-style disclosure in its own detail: written after seeing F5 fail, carrying
no threshold, and reading identically had F5 passed. An INFO row records the
diagnosis beside the failing gate so the audit CSV is not misleading alone.

### One checker bug, ours again

**F9 flagged all six of E1c's own columns**, because it tested whether a design
column *contained* `"shin"` and "fini**shin**g" does. Six false positives, zero
odds columns. It now matches the eleven exact names the odds artefact carries —
`overround`, `shin_z`, `price_*`, `prop_p_*`, `shin_p_*` — plus an anchored
bookmaker-code pattern, and was verified to still catch `B365CH` and `prop_p_H`.
**A verifier that does not implement the declared rule cannot verify it**, for
the third time in this project.

### The instrument's one structural novelty

c is per fold, so **the design matrix is per fold** — four of them, where every
previous rung built one. `run_rung_folds()` is the fold loop that picks the
right matrix; it calls LADDER's own `select_lambda` and `fit_pipeline`, so the
estimator is not duplicated. **F15 asserts that handed the same matrix at every
fold it reproduces `LADDER.run_rung` at 0.000e+00** — the stand-in is verified
against the thing it stands in for rather than believed to match it. F3c
asserts the four matrices genuinely differ, so the per-fold construction is not
decorative.

λ 1000/1000/300/300, EPV 0.90–3.57, **G6 PASS at every fold**. F2: the D2
rescaled base reproduces the committed Amendment 4 artefact, which is what would
catch a fold-dependent column leaking into the control arm.

---

## Phase 5 — the gap diagnostic: the 0.02979 is uniform

*Run 2026-09-03. `scripts/phase5_gap_diagnostic.py`. **EXPLORATORY. No gate, no
KEEP/DROP, nothing here enters a model.** Run only after E1c's declaration was
hashed and committed, which is the only reason E1c is unaffected by it.*

Three declared rungs have failed to close the Dixon-Coles→market gap, and E1a's
third branch said it is not a rating-precision problem. So: where does it live?

**DC − market +0.02979 [+0.01759, +0.04229]; E1a − market +0.02067 [+0.00992,
+0.03147].** Under the Shin sensitivity the DC gap is +0.03025, so nothing below
rests on the de-vig choice. The per-match gap has sd 0.24897 against a mean of
0.0298 — its spread dwarfs its mean, which is why every subset carries an
interval.

**Every split looked at is reported, including the null ones.**

| split | levels | read |
|---|---|---|
| favourite probability | .00–.40 +0.02617 · .40–.50 +0.03893 · .50–.60 +0.02057 · .60–.70 +0.01267 · .70–1.0 +0.05157 | non-monotonic, every CI contains the pooled figure — **null** |
| matchweek | 1-6 +0.04255 · 7-19 +0.02976 · 20-31 +0.02412 · 32-38 +0.02863 | **null** |
| season | 2022-23 +0.03627 · 2023-24 +0.04176 · **2024-25 +0.00889** · 2025-26 +0.03224 | one season nearly matched; the only split with any separation |
| promoted involved | neither +0.03022 · promoted +0.02871 | **dead null on DC**; E1a disagrees (+0.01344 vs +0.03888) |
| promoted detail | home +0.00696 · away +0.03722 · both +0.14117 (n=24) | the n=24 cell is uninterpretable, CI [−0.10, +0.47] |
| actual outcome | **A +0.00270 [−0.02074, +0.02617]** · H +0.03869 · **D +0.04874** | the sharpest split; E1a's draw gap is +0.00735, so **the two models disagree about draws** |
| market pick | H (958) +0.03622 · A (562) +0.01883 | mild |
| favourite delivered | DC +0.03866 vs +0.01889 · **E1a +0.07818 vs −0.05000** | conditions on the outcome; describes only |
| strong favourite ≥0.60 | DC +0.03393 vs +0.01726 · E1a +0.12322 vs −0.16795 | see below |
| correlation, 128 numeric columns | largest \|r\| **0.0785**; 20 clear the naive 0.0503 where **6.4** are expected by chance | nothing on disk explains >0.6% of the variance |

**The hypothesis is not supported.** The lineup story predicts the model loses
most where a strong side underperforms its rating. For Dixon-Coles the gap is
**larger when the favourite delivers** (+0.03866) than when it does not
(+0.01889) — the opposite sign to the prediction.

**A sharpness measurement, added because the reading above is inferable and
confidence is directly measurable.** It corrected the first reading:

| | mean max p | p on market's pick | mean p(D) |
|---|---|---|---|
| market | 0.5391 | 0.5391 | 0.2366 |
| Dixon-Coles | **0.5489** | 0.5316 | 0.2296 |
| E1a | 0.5175 | 0.5047 | 0.2358 |

**Dixon-Coles is *more* confident than the market, not less.** It is simply
confident about a different outcome. E1a genuinely is underconfident. So the
market's edge over DC is not sharpness — it is direction.

**The read: the gap is uniform.** That is the branch the brief called "different
and more interesting". It does not sit in any identifiable subset, no variable
on disk tracks it, and the one split that conditions on the outcome points
against the lineup hypothesis rather than for it.

**What this cannot establish.** It cannot identify a cause; a gap that sits
everywhere is consistent with many mechanisms. It has no multiplicity control.
The favourite-delivered split conditions on the outcome and can never be a
feature. **No rung may be designed from any of it.**

---

## Phase 5 — Instrument E1: shots on target as a rating input

*Run 2026-09-02/03. `scripts/phase5_e1a_sot_ratings.py` (12 checks, **0 failures**)
and `scripts/phase5_e1b_shot_residuals.py` (35 checks, **0 failures**).
Pre-declaration `PHASE5_E1_SHOT_PREDECLARATION.txt`, sha256 `d385bfd4…6eca4`,
signed off before fitting. The data was already on disk — the football-data.co.uk
files frozen for the market benchmark carry the shot columns.*

### E1a — the rating-versus-rating comparison

Attack, defence and the home multiplier estimated from shots on target instead
of goals. Everything else held identical: same 107-day half-life, same 460
refits, same window rule, same MAX_GOALS, same score matrix. **ρ is the same
number in both arms** — fitted once per window from the goals arm and handed to
the SoT arm, asserted bit-identical, which is what isolates the rating input as
the single difference.

| model | log loss | RPS | Brier |
|---|---|---|---|
| market (Bet365 closing) | 0.96057 | 0.19469 | 0.5702 |
| **E1a — SoT ratings** | **0.98124** | 0.20117 | 0.5834 |
| Dixon-Coles walk-forward | 0.99036 | 0.20350 | 0.5897 |
| Poisson walk-forward | 0.99042 | 0.20355 | 0.5900 |
| E1b | 0.99758 | 0.20568 | 0.5950 |
| Elo v1 | 0.99943 | 0.20667 | 0.5958 |
| D4 | 0.99975 | 0.20619 | 0.5963 |
| D2 rescaled | 1.00028 | 0.20646 | 0.5970 |
| D0 base rate | 1.06889 | 0.23185 | 0.6467 |

| comparison | Δ log loss | 95% CI | Δ RPS | verdict |
|---|---|---|---|---|
| **E1a − Dixon-Coles** | **−0.00912** | **[−0.02177, +0.00391]** | −0.00233 | **not significant** |
| E1a − D2 rescaled | −0.01904 | [−0.03195, −0.00628] | −0.00529 | significant |
| E1a − Elo v1 | −0.01819 | [−0.02992, −0.00628] | −0.00550 | significant |
| E1a − D0 | −0.08764 | [−0.10543, −0.06969] | −0.03068 | significant |
| E1a − market | +0.02067 | [+0.00990, +0.03176] | +0.00648 | significant |

**The comparison the rung exists for does not clear the bar.** Signs agree, so
this is *not significant* rather than inconclusive. Per fold: −0.01417,
−0.00477, +0.00349, −0.02103 — three of four favour SoT, one does not.

The point estimate closes **31% of the Dixon-Coles→market gap**, and it is 2.5×
the largest effect Phase 4 ever found. It is also not distinguishable from zero
on 1,520 matches. Both of those are true and neither cancels the other.

**What the interval does rule out.** To close the 0.02979 to the market, E1a −
DC would need to reach −0.02979. The 95% interval stops at **−0.02177**. Closing
the gap with SoT-based ratings is excluded at 95% — the rung is underpowered to
confirm a small effect but not underpowered to refute the large one it was
sized against.

### The conversion constant

Candidate (i) rests on c being estimable, and E2.3 predicted it would be.

| scope | refits | mean c | sd | min | max |
|---|---|---|---|---|---|
| all | 460 | 0.32680 | 0.00703 | 0.30865 | 0.34026 |
| fold 1 | 117 | 0.32055 | 0.00688 | — | — |
| fold 2 | 120 | 0.33136 | 0.00499 | — | — |
| fold 3 | 109 | 0.32455 | 0.00498 | — | — |
| fold 4 | 114 | 0.33055 | 0.00454 | — | — |

Within-season drift is at most **±0.011 across a whole season** (−0.005, +0.011,
+0.009, −0.005 by fold) — smaller than the between-fold spread, so c is not
sliding underneath the ratings within a season. The declared home/away
sensitivity is a null: `c_home` 0.32896, `c_away` 0.32424, and the split model
differs from the pooled one by **+0.00004 [−0.00243, +0.00243]**, not
significant. One constant is enough.

### The third branch fired

E5.2 declared two consequences of the mechanism and E5.3 named the case where
only the second holds. That is what happened.

| parameter | arm | mean \|step\| | mean level | relative |
|---|---|---|---|---|
| attack | goals | 0.016764 | 1.060532 | 0.01587 |
| attack | **SoT** | 0.009454 | 1.025999 | **0.00924** |
| defence | goals | 0.019817 | 1.282353 | 0.01645 |
| defence | **SoT** | 0.038585 | 4.022815 | **0.00987** |

Mean absolute change in a team's parameter between successive refits, within
fold, relative to the level it moves around. **SoT-based parameters move 0.58×
(attack) and 0.60× (defence) as much as goal-based ones.** The raw step is
reported alongside because the two arms' parameters need not share a scale —
SoT defence sits at a level of 4.02 against goals' 1.28 — so the relative figure
is the comparable one.

**SoT estimates team strength roughly 40% more precisely, and that precision
does not convert into predictive accuracy.** The mechanism claimed in E5.1 is
real. Its consequence is not.

### E1b — rolling shot residuals

Base: **D2 rescaled**, fixed by E7.1 before E1a ran. E1a − DC came back not
significant, so the pre-committed rule put E1b on D2 rescaled. Six columns,
design width 92 → 98, λ 1000/1000/300/300, EPV 0.90–3.57, **G6 PASS at every
fold**.

| comparison | Δ log loss | 95% CI | Δ RPS | verdict |
|---|---|---|---|---|
| E1b − D2 rescaled | −0.00270 | [−0.00536, **+0.00001**] | −0.00078 | not significant |
| E1b − E1a | +0.01634 | [+0.00397, +0.02848] | +0.00451 | significant |
| E1b − Dixon-Coles | +0.00722 | [−0.00621, +0.02054] | +0.00218 | not significant |
| E1b − Elo v1 | −0.00185 | [−0.01044, +0.00665] | −0.00099 | not significant |
| E1b − D0 | −0.07131 | [−0.08825, −0.05528] | −0.02617 | significant |
| E1b − market | +0.03701 | [+0.02603, +0.04791] | +0.01099 | significant |

**The interval's upper bound is +0.00001.** It does not exclude zero and the
result is not significant — that is the rule, and a bound one part in a hundred
thousand from clearing it is still not clearing it. Recorded exactly as it came
out rather than rounded into a win.

The effect is −0.00270, against −0.00365 for D2 − D1, the largest thing Phase 4
found. The block's coefficients grow with training data — `rel_sot_residual_diff`
runs 0.0158, 0.0232, 0.0686, 0.0728 across folds — the same pattern Block X
showed, and here it converts into rather more.

**E1a beats E1b significantly.** Putting shot information into the rating is
worth more than putting it beside the rating as a feature, by 0.0163
[+0.0040, +0.0285].

**One limitation of the declared quantity, stated because it is not obvious.**
The residual is SoT differential minus goal differential, both per match. Since
goals ≈ 0.327 × SoT, its expectation is roughly 0.67 × SoT differential — so the
column is substantially a restatement of shot differential rather than a pure
finishing measure. A `goal_diff − c·SoT_diff` version would isolate finishing
properly. That was not what was declared, and it was not swapped in after seeing
this result; it is named here as the obvious follow-up, requiring its own
declaration.

### What was caught

**The base rung was contaminated on the first run, by the control arm's own
construction.** `block_of()` returns `phase1_backbone` for any name it does not
recognise, and `d1_features()` selects the backbone *by exclusion* — so the six
new columns, attached to the feature frame before the base was read, were swept
into D2 rescaled itself. D2 came out 98 columns wide, stopped reproducing the
committed Amendment 4 artefact (7.906e-03 against a 1e-12 tolerance), and E1b
"added" nothing because there was nothing left to add.

Four gates caught it at once — E10, E10c, E1b-A1 and DS3a — and E10e now asserts
the base directly. **The comparison would still have run.** It would have
compared a rung against itself and reported a null, and the null would have been
an artefact of feature-frame ordering.

### Gates

**E1a, 12 checks.** E1 (goals arm reproduces committed `dc_walkforward` at
0.000e+00), E2 (one configuration, so the arms cannot drift apart), E3 (ρ
bit-identical, 0.000e+00), E4a/E4b (join and the source's known defect counts),
E5 and E6 (**both 0.000e+00** — corrupting goals, shots and SoT from six cutoffs
forward moves neither the fitted state nor the conversion constants), E7, E8,
E9a/E9b (no odds column reaches a rating window), E10a.

**E1b, 35 checks.** E10/E10b/E10c/E10e, A4a at both rungs (the robust mask still
finds exactly its three DC-derived columns, so the residuals did not drift into
Amendment 4's rule), G6, E1b-A1 (5.551e-17), and DS0–DS12 in full.

**DS7 could not apply as written and was replaced rather than skipped.** DS7a
asserts byte-identity against the frozen Phase 3 feature file; the six residual
columns are *built* here, so that claim is not available for them. What is
asserted instead is that adding the block disturbed no column it sits beside,
checked against a fresh re-read from disk. Zero disturbed.

### The read on A, per the declared branches

**E9.2 — the null branch.** E1a − Dixon-Coles is not significant, so on the
declared reading shot information adds little over goals at this resolution and
**A is scoped accordingly: not abandoned, not expected to move much.**

The third branch sharpens that considerably, and E5.3 said in advance that it
would be the outcome that most changed how A is read:

**The remaining 0.02979 to the market is not a rating-precision problem.** SoT
estimates team strength ~40% more stably than goals do, and the accuracy did not
follow. xG's mechanism is the same mechanism — a less noisy estimate of the same
underlying rate — so the thing xG would do better is the thing that has just
been shown not to pay. Closing the gap from a better shot-derived estimate of
team strength is excluded at 95% for SoT, and xG is an improvement in degree
along that axis, not a different axis.

**The asymmetry matters and it points the same way.** Shot counts were recorded
contemporaneously by a scorer; nobody's model produced them. The xG arm carries
the untestable retro-fit leak of A4.2 — today's xG for a 2021 match may come from
a model trained through 2026 — so any xG advantage is an upper bound on what was
available in real time. **E1 is the cleaner measurement, and it came back null.**

What this does *not* establish: that xG carries nothing SoT does not. Shot
*quality* is exactly what SoT throws away, and a chance-quality signal could
differ in kind rather than degree. The declared bound is an argument about
information content and it could be wrong. But the expected value of the five
FBref downloads is now materially lower than it was before E1 ran.

### On the xG files

Five CSVs were supplied. They are **season-aggregate league tables** — 20 rows
each, team totals of xG/xGA/xPTS — not per-match xG. They cannot serve the arm:
§5.1 needs per-match home and away xG for 1,900 matches, a season total for a
team includes the very matches being predicted, and there is no per-match
resolution for a walk-forward window. Two of the five are byte-identical
duplicates of 2021-22 (sha256 `865b7313…`), so they cover four seasons, not five,
and **2024-25 is absent**. They are left in `data/` untracked and unused.

### Evidence

`outputs/phase5_e1a_{fold_summary,pooled,deltas,windows,parameter_stability,predictions,audit}.csv`,
`outputs/phase5_e1b_{fold_summary,pooled,deltas,coefficients,lambda_curves,residual_features,audit}.csv`.
All under `FROZEN_MANIFEST.txt`.

---

## Phase 5 — Instrument B: the market benchmark, measured

*Run 2026-09-02. `scripts/phase5_market_benchmark.py`, 19 checks, **0 failures**.
Pre-declaration `PHASE5_MARKET_PREDECLARATION.txt`, written before any score
existed. Fits nothing: every model figure is read from a committed artefact and
asserted against it.*

### The number that was an assumption

Phase 4 recorded, in prose, that "the market's ~0.95 is out of reach by
construction". That figure was never measured on these matches. It is now.

**The market's pooled log loss on the same 1,520 outer-test matches is
0.96057**, RPS 0.19469. The assumption was optimistic by about 0.01 — the
market is not as good as the project had been assuming it was.

| model | log loss | RPS | Brier |
|---|---|---|---|
| **market — Bet365 closing** | **0.96057** | 0.19469 | 0.5702 |
| Dixon-Coles walk-forward | 0.99036 | 0.20350 | 0.5897 |
| Poisson walk-forward | 0.99042 | 0.20355 | 0.5900 |
| Elo v1 | 0.99943 | 0.20667 | 0.5958 |
| D4 | 0.99975 | 0.20619 | 0.5963 |
| D2 rescaled | 1.00028 | 0.20646 | 0.5970 |
| D3 | 1.00125 | 0.20662 | 0.5976 |
| D1 | 1.00393 | 0.20769 | 0.5994 |
| D0 base rate | 1.06889 | 0.23185 | 0.6467 |

### The gap, and it is significant at every fold

| comparison | Δ log loss | 95% CI | Δ RPS | verdict |
|---|---|---|---|---|
| market − D4 | −0.03918 | [−0.05041, −0.02787] | −0.01150 | **significant** |
| market − D2 rescaled | −0.03971 | [−0.05106, −0.02838] | −0.01177 | **significant** |
| market − Elo v1 | −0.03886 | [−0.05037, −0.02755] | −0.01198 | **significant** |
| market − Dixon-Coles | −0.02979 | [−0.04229, −0.01759] | −0.00881 | **significant** |
| market − D0 | −0.10831 | [−0.12806, −0.08891] | −0.03716 | **significant** |

Negative favours the market. Signs agree on every row, and every interval
excludes zero. Per fold, market − D4 is −0.0320, −0.0592, −0.0397, −0.0259 —
**significant at all four**, never once inconclusive.

This is the first comparison in the project where something beats the ladder
decisively. Every internal delta from D1 upward has been under 0.004; this one
is ten times that.

**And the comparison is biased in the project's favour.** 2025-26 was scored
during Phase 3's lambda sweep and its B0–B6 ablation, so every model figure
here is a walk-forward development estimate, optimistic by an unknown margin.
The market has never been fitted to anything. The project loses by 0.039 with
the thumb on its own side of the scale.

### Where the ladder sits on the road to the market

Of the 0.10831 in log loss between the base rate and the market:

- **D0 → D4 covers 0.06914, or 64%**
- the remaining **0.03918, or 36%**, is not covered by anything in this project
- Dixon-Coles covers 0.07853 of it, 72%, and is still 0.02979 short

### Which odds, and why it is not Pinnacle

Decided from completeness before any score was computed, and this is the part
of the instrument most exposed to being chosen after the fact.

| book | present on 1,520 | by test season |
|---|---|---|
| **Bet365 closing (primary)** | **1520/1520** | 380 / 380 / 380 / 380 |
| market average closing | 1520/1520 | 380 / 380 / 380 / 380 |
| Pinnacle closing | 1350/1520 | 380 / 380 / 380 / **210** |

Pinnacle is the sharper line and the usual choice. It is missing on 170 rows
and **all 170 are in fold 4**, where it covers 210 of 380 — a primary benchmark
that changes population between folds. The market average is complete but is an
average over a *different set of bookmakers each season*: VC and WH leave, 1XB
and BFE arrive, Pinnacle drops to 210, and the overround climbs from 1.0388 to
1.0567 as the panel shifts underneath it.

Bet365 closing is 380 of 380 in all five seasons — the only column that is the
same instrument at every fold. That is the whole of the argument for it; it is
not a claim that Bet365 is sharpest.

**It does not matter.** All four book × de-vig combinations land inside 0.0006
of each other:

| variant | log loss |
|---|---|
| market average, Shin | 0.96001 |
| Bet365, Shin | 0.96011 |
| market average, proportional | 0.96031 |
| Bet365, proportional | 0.96057 |

Pinnacle, per fold and never pooled, scored on exactly the rows it covers
against Bet365 on those same rows: 0.9622 vs 0.9617, 0.8996 vs 0.8997, 0.9664
vs 0.9678, 0.9875 vs 0.9842. Indistinguishable. **The 0.039 gap is not an
artefact of which bookmaker was picked.**

### The calibration audit, and why ECE decides nothing

Ten fixed bins of width 0.1, declared before the curve was seen. Thin bins are
reported with their counts, never merged or dropped. No recalibration is fitted
— Platt or isotonic on the outer-test rows would be fitting on the rows being
scored.

| model | ECE | MCE | bias H | bias D | bias A |
|---|---|---|---|---|---|
| D0 base rate | **0.00760** | 0.01140 | +0.0025 | −0.0114 | +0.0089 |
| market avg, Shin | 0.00778 | 0.05061 | −0.0037 | −0.0032 | +0.0069 |
| market Bet365, Shin | 0.00790 | 0.09139 | +0.0008 | −0.0082 | +0.0074 |
| market Bet365, proportional | 0.01059 | 0.10230 | −0.0037 | −0.0042 | +0.0078 |
| D2 rescaled | 0.01625 | 0.11800 | +0.0107 | −0.0122 | +0.0014 |
| Dixon-Coles | 0.01677 | 0.16273 | −0.0082 | −0.0112 | +0.0194 |
| D4 | 0.01716 | 0.10010 | +0.0075 | −0.0100 | +0.0025 |
| Elo v1 | 0.02695 | 0.19316 | +0.0040 | −0.0132 | +0.0092 |

**D0 has the best calibration in the project and is the worst model in it.** A
constant base-rate prediction is almost perfectly calibrated by construction
and carries no information whatever. That is the entire argument for B6.4: ECE
is a description, and nothing here is decided on it.

Two real findings survive that caveat:

**Every model under-predicts draws.** `bias D` is negative for all twelve
entries, market included, from −0.0032 to −0.0175. The market's draw bias is
the smallest; Poisson's is the largest at −0.0175, with a matching +0.0225 on
away wins.

**The market shows the textbook favourite–longshot bias**, and Shin corrects
it. Bet365's proportional reliability curve, pooled over classes:

| bin | n | mean predicted | observed |
|---|---|---|---|
| 0.0–0.1 | 148 | 0.0764 | **0.0473** |
| 0.1–0.2 | 763 | 0.1590 | 0.1547 |
| 0.2–0.3 | 1750 | 0.2535 | 0.2474 |
| 0.3–0.4 | 581 | 0.3456 | 0.3666 |
| 0.4–0.5 | 492 | 0.4485 | 0.4370 |
| 0.5–0.6 | 378 | 0.5519 | 0.5661 |
| 0.6–0.7 | 258 | 0.6486 | 0.6550 |
| 0.7–0.8 | 142 | 0.7483 | 0.7465 |
| 0.8–0.9 | 48 | 0.8352 | **0.9375** |

Longshots over-priced, heavy favourites under-priced — the middle eight bins
sit within 0.02. Applying Shin, which exists to redistribute margin away from
longshots, drops ECE from 0.01059 to 0.00790 and MCE from 0.1023 to 0.0914.
**A declared sensitivity behaving exactly as its theory predicts is a check on
the implementation, and it passed.** The 0.8–0.9 bin holds 48 predictions and
is flagged thin; it is not evidence on its own.

### The join

1,900 project matches against 1,900 football-data.co.uk rows, joined on
(season, home, away) rather than on date — two sources can disagree about the
date of a rearranged fixture while agreeing about the fixture. The date is then
reconciled instead of joined on.

**Zero unmatched, zero duplicate keys, zero scoreline disagreements, zero date
disagreements.** Two independently-built descriptions of the same 1,900
matches agree completely — the Phase 0 Instrument 4 discipline applied to a new
source. Eight team names differ and all eight are mapped by explicit
dictionary lookup; a ninth would fail the gate rather than be fuzzy-matched.

### What this does and does not license

It **does not** establish that the market is a ceiling. It measures where the
market sits on these 1,520 matches. Whether 0.96057 is reachable without lineup
and injury data is not a question these numbers answer — and the project's own
scoping decision at Step 2 is why it has neither.

It **does** retire the 0.95 assumption. Any future claim about the distance to
the market should quote 0.96057 and this instrument, not a remembered figure.

Step 2's betting exclusion stands: no stake, no return, no yield, no closing
line value is computed anywhere in this instrument. Odds enter as a benchmark
to be scored against and never as a feature; no rung was re-fitted and no
number already reported moved. XGBoost and random forests stay excluded.

### Evidence

`outputs/phase5_{market_fold_summary,market_pooled,market_deltas,market_probabilities,calibration,reliability,market_audit}.csv`,
`data/raw/Odds/E0_{2122,2223,2324,2425,2526}.csv`. All under
`FROZEN_MANIFEST.txt`.

19 checks, 0 failures: M1a–M1b (join integrity), M2a–M2b (cross-source
reconciliation), M3 (closed mapping), M4/M4b (primary completeness, Pinnacle
never pooled), M5 (harness validation), M6a–M6b per book (de-vig identities and
Shin's bracket), M7 (380 per fold), M8 (the pooled/fold-mean identity), M9a–M9b
(committed artefacts reproduce to 0.000e+00), M10 (bin counts).

**M9b is the one worth naming.** Re-scoring the committed probabilities of all
seven models reproduced every one of the six pooled metrics to **0.000e+00**.
Nothing in this entry is a re-derivation.

---

## Phase 4 — the ladder finished: D3, D4, and the Phase 4 conclusion

*Run 2026-09-02. `scripts/phase4_d34_ladder.py`, 44 checks, **0 failures** in this
instrument. G9 and G10 stand failing at the project level on grounds outside it.
Brief `NEXT_SESSION_BRIEF.txt`; D2-static amendment sha256 `90743c9b…c4812`.*

### What the added blocks actually did

**Block X's three availability columns are exact duplicates of columns D1
already had.** `home_prior_fbref_available` is byte-identical to
`home_prev_season_available`, and the same holds for the away and the relative
pair — three of Block X's 27 columns carrying nothing the backbone was not
already carrying. The ridge does the only thing it can with perfectly collinear
columns and splits the weight evenly between them: at D4 fold 4 both members of
each pair land on **|β| = 0.026751**, equal to six figures.

So D4's 27 new columns are 24 new descriptions and 3 restatements, and that was
true before anything was fitted.

**The added columns do not dominate the design.** At D4 fold 4 the twelve
largest coefficients are all Phase 1 backbone columns, led by `rel_elo_diff`
(0.0592) and `rel_attack_diff` (0.0580) — the same two columns Amendment 4
identified as the dynamic-state block's real content. The largest Block C
column reaches 0.0236 and the largest Block X composite 0.0251, both below every
one of the top twelve.

| block | n | max \|β\| | mean \|β\| |
|---|---|---|---|
| phase1_backbone | 92 | 0.059238 | 0.015460 |
| X_availability | 3 | 0.026751 | 0.016496 |
| X_prior_composite | 24 | 0.025053 | 0.012912 |
| C_context | 20 | 0.023571 | 0.009366 |

**A correction.** An earlier reading of this session's evidence said the six
largest coefficients in the design were Block C and Block X columns. That was an
artefact of ranking within `phase4_d34_block_coefficients.csv`, which by design
holds only the added blocks; no ranking inside it can support a claim about the
design. Re-fitted over all 139 columns, **none** of the top ten is a C or X
column. The claim is withdrawn.

**What does hold** is that Block X's coefficient mass grows monotonically with
training data — summed L2 norm across its 27 columns of **0 → 0.255 → 0.320 →
0.359** over folds 1 to 4 — and converts into no measurable gain at any fold.
The penalty is not suppressing these columns to zero. They take weight, they
take more of it as evidence accumulates, and the predictions do not move.

### The null result

Every ladder-internal delta is not significant, RPS agreeing in sign with log
loss throughout.

| comparison | Δ log loss | 95% CI | Δ RPS | verdict |
|---|---|---|---|---|
| D3 − D2 rescaled | +0.00097 | [−0.00082, +0.00277] | +0.00016 | not significant |
| D4 − D3 | −0.00150 | [−0.00448, +0.00148] | −0.00043 | not significant |
| D4 − D2 rescaled | −0.00053 | [−0.00363, +0.00259] | −0.00027 | not significant |

Per fold the sign moves in both directions and no interval excludes zero.
**Phase 3's conclusion that Block C and Block X add nothing is upheld.** The
binding rule written before the fit — that Phase 4 wins a contradiction and
Phase 3 is amended rather than defended — never had to fire.

The counterpart rule bites instead: **a non-significant difference is not
equality.** These intervals are wide enough to contain effects worth having, and
1,520 matches cannot separate them from zero.

Against the references both rungs behave as D2 rescaled did: **significant
against D0** (−0.0676 at D3, −0.0691 at D4), **inconclusive against Elo v1** on
sign disagreement, and not significantly different from Poisson or Dixon-Coles.

### The ladder, D0 to D4

| model | columns | log loss | RPS |
|---|---|---|---|
| D0 base rate | 0 | 1.068888 | 0.231851 |
| D1 current-season results | 88 | 1.003929 | 0.207687 |
| D2 rescaled | 92 | 1.000283 | 0.206458 |
| D3 (+ Block C context) | 112 | 1.001250 | 0.206615 |
| D4 (+ Block X prior season) | 139 | 0.999749 | 0.206188 |
| **Elo v1 (single K=20 rating)** | **1** | **0.999431** | 0.206670 |
| Poisson walk-forward | — | 0.990415 | 0.203547 |
| Dixon-Coles walk-forward | — | 0.990364 | 0.203499 |

Accuracy is in the artefacts and decides nothing here.

### D4 fold 1 is identical to D3 fold 1, and that is structural

Fold 1 trains on 2021-22, the first season in the data, so no team has a prior
season to describe. All 27 Block X columns are constant over its training rows —
24 of them wholly missing — and a constant column takes scale 1.0 and a
coefficient of exactly zero. `constant_columns` goes 29 → 56, the difference
exactly 27, and the D4 − D3 fold 1 delta is **5.1e-18**: floating-point zero,
not a tolerance on a difference that ought to be zero. Every one of the 27
added coefficients is exactly 0.0, so the residue is summation order and
nothing else.

This is asserted rather than narrated — **X1a–X1d** in the audit ledger — and
flagged in the `structural_note` column of both the fold summary and the delta
table, where the numbers are actually read. A fold summary row that looks
duplicated and a delta of 5e-18 are otherwise exactly what a reader files as a
bug.

**The flag earns itself immediately.** That row's verdict reads *INCONCLUSIVE
(sign disagreement)* — the log loss residue is +5.1e-18 and the RPS residue
−1.8e-18, so the sign test fires on noise 15 orders of magnitude below
anything meaningful. Without the note, the delta table shows an inconclusive
verdict on a comparison that is an identity.

**These rungs were re-fitted on the E: laptop; the first run was on D:.** Every
reported figure reproduces bit-for-bit — pooled metrics to all 17 digits,
pooled deltas to 13. What moves is the residue: A1 against the committed
Amendment 4 artefact reads 5.551e-17 here against 0.000e+00 there, and this
fold 1 delta 5.113e-18 against 5.405e-18. Both are far below their gates, and
DS8a still reproduces every probability exactly **within** a process. The
lesson for anything that keys on these numbers: bit-identity holds within a
machine, not across two.

### expected_total_goals resolved on the uninformative branch

The diagnostic asked whether the column is redundant *given D1* rather than
uninformative in general. Regressed on D1's twelve scoring-environment columns,
per fold, training rows only:

| fold | adj R² (all 12) | strongest single correlation |
|---|---|---|
| 1 | 0.073 | `rel_gapm_diff` r = +0.138 |
| 2 | 0.046 | `rel_gapm_diff` r = +0.109 |
| 3 | 0.027 | `rel_gapm_diff` r = +0.086 |
| 4 | 0.019 | `rel_gapm_diff` r = +0.074 |

Falling as training grows, and low throughout. **The column is not redundant
given D1 — it is uninformative.** D1 does not contain it, nothing in D1 predicts
it, and Amendment 4 already established that it carries little once its scale is
sane. Dixon-Coles needs it because Dixon-Coles has nothing else.

The structural argument for the column was wrong, and wrong for a different
reason than was supposed when it was conceded: not that the results-derived
columns already encoded it, but that there is little there to encode.

### The Phase 4 conclusion

The four questions Phase 4 committed to, answered from the experiment.

**1. Does dynamic team-strength state explain most of the remaining gap between
the results-derived rung and Poisson/DC?** No — about a quarter of it. D2
rescaled − D1 = **−0.00365 [−0.00489, −0.00241]**, significant, against a
D1-to-Dixon-Coles gap of 0.01357. That is **26.9%**. The residual, D2 rescaled −
DC = +0.00992 [−0.00330, +0.02314], is **not significant on 1,520 matches**.
That is resolution, not equality: the remaining 73% is unmeasured, not shown to
be absent.

**2. Once dynamic state is present, does Block C add anything?** No. D3 − D2
rescaled = +0.00097 [−0.00082, +0.00277].

**3. Once dynamic state is present, does Block X add anything?** No. D4 − D3 =
−0.00150 [−0.00448, +0.00148]. Three of its 27 columns are exact duplicates of
backbone columns, and fold 1 cannot see any of it at all.

**4. Where is the strongest signal?** **Continuously updated strength — and it
does not need the engineering.** Decomposing the whole D0-to-Dixon-Coles
distance of 0.0785 in log loss:

- current-season results (D0 → D1): **0.0650, 83%**
- continuously updated rating state (D1 → D2 rescaled): 0.0036, **5%**
- everything still between D2 rescaled and DC: 0.0099, 13%
- static historical description (Blocks C and X): **nothing measurable**

But the compact form beats the engineered one. **Elo v1 — a single K=20 rating —
has a lower pooled log loss than every rung on the ladder, D4's 139 columns
included**, and neither metric separates the two. Poisson and Dixon-Coles, also
continuously updated but working from goals rather than results, beat everything
here. The answer is not "a combination": recency-weighted strength is most of
the signal, and 139 engineered columns are a less efficient way of writing it
down than one rating is.

### What this does not license

**There is no clean holdout.** 2025-26 was scored during Phase 3's lambda sweep
and its B0–B6 ablation. Every number in this entry is a walk-forward
**development estimate**, optimistic by an unknown margin. None of it may be
described as an out-of-sample holdout result.

**The market's ~0.95 is out of reach by construction**, not by shortfall. This
project has no lineup and no injury data. That is a scoping decision taken at
Step 2, and it caps what any rung here can reach; the distance to the market is
not evidence that the modelling is deficient.

### Evidence

`outputs/phase4_d34_{fold_summary,pooled,deltas,lambda_curves,block_coefficients,predictions,audit}.csv`,
`outputs/phase4_etg_{redundancy,correlations}.csv`,
`outputs/phase3_{features,feature_inventory}.csv`. All under
`FROZEN_MANIFEST.txt`.

44 checks, 0 failures: A1, A4a/b at three rungs, G6, G10, P1, X1a–X1d, and
DS0–DS12 in full.

**DS7's first live run.** It had been recorded "NOT EXERCISED" in every prior
run because no C or X column entered D0–D2. Its implementation in the ladder
module measured block *counts*, which is not the declared test, so it was
rewritten to assert rather than measure, and run before D3 was read:

- **DS7a** — 67 C and X design columns across D3 and D4 are byte-identical, NaN
  for NaN, to a fresh re-read of `outputs/phase3_features.csv` from disk. Not to
  the in-memory frame the design was built from. 0 offenders.
- **DS7b** — 302 recorded statistics over 39 columns agree with
  `phase3_feature_inventory.csv`, an independent description of the same bytes.
  This is the check that would catch a regenerated feature file; a file compared
  against itself never could. 0 drifted.
- **DS7c** — both files are now under the frozen manifest. **Neither was
  before**, so DS7's declared wording "byte-identical to the frozen Phase 3
  artefacts" was unbacked, and the ladder module's own note that "DS10 verifies
  the file's hash" was false. The manifest goes 55 → 70 files.

### Still open

- **D2-static restricted rung** — deferred. `PHASE4_AMENDMENT6_D2STATIC.txt`
  records the defect, the framing error, and the only design that would make the
  rung readable. **G10 stays failing** until it is rebuilt.
- **G9** — failing at 2.220e-16, correctly and unsoftened. A 279×279 against a
  267×267 Newton system is a real test of an over-strong claim.
- **The burn-in question** — Elo is under-converged league-wide, so D2 − D1
  stands as a lower bound on what dynamic state is worth.

Step 2's exclusions hold: no xG, no market odds, no XGBoost, no random forests,
no betting backtests.

---

## Phase 4 — Amendment 4 (robust scaling) and Amendment 5 (D2-static)

*Run 2026-09-02. `scripts/phase4_amendment4_ladder.py`, 35 checks, **2 failures**
(G9, G10) — both reported, neither fixed by moving the rule that caught them.
Pre-declaration sha256 `95265231…d2bf0ad`.*

### Amendment 4 did what it was written to do

`expected_total_goals` now enters the fit divided by 0.64–0.88 instead of
15–30. Its test season spans **4.8–5.8 standardised units instead of
0.14–0.25**; `rel_defence_diff` spans 3.7–5.7 instead of 0.45–0.81.

**D2 rescaled − D2 original = −0.00058 log loss [−0.00097, −0.00020]**, RPS
−0.00021 [−0.00034, −0.00009]. Signs agree, significant. Real, and small.

| model | log loss | RPS |
|---|---|---|
| D1 | 1.003929 | 0.207687 |
| D2 original | 1.000863 | 0.206671 |
| **D2 rescaled** | **1.000283** | **0.206458** |
| Elo v1 | 0.999431 | 0.206670 |
| Dixon-Coles walk-forward | 0.990364 | 0.203499 |

**D2 rescaled − Elo v1 is INCONCLUSIVE**: log loss +0.00085, RPS −0.00021. The
rescaled rung passes Elo on RPS and trails it on log loss; signs disagree, and
the binding rule makes that inconclusive rather than a win. **Elo still leads on
log loss.**

**The headline is therefore: neither metric separates 92 columns from a single
K=20 rating.** The earlier phrasing — "92 columns land on the same number" — was
literally true of the original D2's RPS and is not true of the rescaled rung. It
is withdrawn, not softened.

**D2 rescaled − D1 = −0.00365 [−0.00489, −0.00241]**, significant at every fold.

### A4.6's prediction resolved on its second branch

Given a sane scale and five SD of room, **`expected_total_goals` shrank**: its
coefficient norm fell to **0.1–0.3×** the original at every fold.

**The column genuinely carries little.** That is the finding, stated without a
hedge: the Amendment 4 scale sits alongside *both* the burn-in-clearing SD and
the test SD, so the compression defect is gone and the shrinkage that remains is
real. It is not "compression may still explain part of it".

`rel_attack_diff` is the column that gained (1.3–1.7×); `rel_defence_diff` moved
in neither direction consistently (0.8–1.5×). So the dynamic-state block is a
two-column block **on its merits** — `rel_elo_diff` and `rel_attack_diff` — and
the column section 2 argued was structurally load-bearing is not one of them.

### Amendment 5's rung is not readable

A5.2 declared in advance that 2021-22 has no frozen state and predicted fold 1
would go inert. **G9b confirms it exactly** — every dynamic coefficient zero,
identical λ, log loss equal to 17 significant figures.

What A5.2 did not anticipate: the same missing season is **50%, 33% and 25% of
folds 2, 3 and 4's training rows**, imputed to the median. At 50% the imputed
value necessarily occupies both quartiles, so the IQR is identically zero. Two
D2-static columns took the degenerate-column guard **while genuinely varying**
and entered the penalised fit in raw units; a third took a scale of 0.0011.

**G10 FAILS on that and is deliberately left failing.** It carries no threshold —
it tests an implication the pipeline already relies on (zero spread ⟹ constant
column) and fails where that is false.

**And the rung answers a narrower question than it was commissioned for.** D1's
84 form features — season-to-date PPG, last-5, venue form — update match by match
in *both* arms; only the rating state freezes. So D2 − D2static measures the
marginal recency of the **rating**, on top of form features that are already
fresh. It cannot approach tier 2's 0.0356 and was never going to. That framing
error was the project's, not the implementation's.

So **D2 − D2static neither answers the tier-2 question nor is readable as it
stands.** Its numbers are in the artefacts, marked NOT READABLE, and are not
quoted here. Fixing the rung needs its own declaration — see
`NEXT_SESSION_BRIEF.txt` step 3 for the only design that would make it paired.

### G9 failed as written

The strict form — "bit for bit" — was too strong for a design that adds four
all-zero columns and therefore solves a 279×279 Newton system where D1 solves
267×267. G9c measures the resulting disagreement in the shared coefficients at
6.9e-18. The wording was ours; **A5.2 is not amended to soften it.**

### Two checker bugs, both ours

DS2b computed a sample SD where Amendment 4 declares median/IQR (off by 29.0),
then still imputed with the mean where A4.2 declares the median (off by 1.0).
Neither was a pipeline defect. It now reimplements the declared rule in full and
passes at 0.0. **A verifier that does not implement the rule cannot verify it.**

DS13 confirms the pipeline edit changed nothing it should not have: D1 and the
original D2 reproduce the artefact committed at `c2528c5` to 0.0 with no λ moves.

### Still standing

Section 6 unchanged. D3 and D4 not run.

---

## Phase 4 — the dynamic-state ladder: D0, D1, D2

*Run 2026-09-02. `scripts/phase4_dynamic_ladder.py`, 24 checks, 0 failures.
Governed by `PHASE4_D2_PREDECLARATION.txt` + Amendments 1–3.*

### The result

**Ninety-two columns land on the number a single K=20 rating produced in one
pass.**

| model | design width | log loss | RPS |
|---|---|---|---|
| D2 — D1 + dynamic state | 92 | 1.00086 | **0.206671** |
| Elo v1 | 1 rating | 0.99943 | **0.206670** |

D2's pooled RPS is seven ten-millionths behind Elo's. Its log loss is +0.0014
behind, CI [−0.0070, +0.0097] — not significant. **D1 − Elo is also not
significant** (+0.0045 log loss, CI [−0.0040, +0.0130]).

That is the Phase 4 result so far. Eighty-four results-derived features, plus
four columns of point-in-time Dixon-Coles and Elo state refit once per calendar
date, reproduce what Elo v1 already had. Neither rung separates from it at the
resolution 1,520 matches provide.

Not licensed by this: that the rungs are *equal* to Elo. A CI covering zero is
the resolution of the test, not a proof of equivalence.

### The full ladder

| model | acc | bal_acc | macro_f1 | log loss | brier | RPS |
|---|---|---|---|---|---|---|
| D0 base rate | .4447 | .3333 | .2052 | 1.06889 | .64667 | .23185 |
| D1 results-derived (88 cols) | .5230 | .4394 | .3805 | 1.00393 | .59941 | .20769 |
| D2 dynamic state (92 cols) | .5250 | .4415 | .3823 | 1.00086 | .59736 | .20667 |
| Elo v1 | .5355 | .4471 | .3874 | 0.99943 | .59580 | .20667 |
| Poisson walk-forward | .5270 | .4513 | .3907 | 0.99042 | .58998 | .20355 |
| Dixon-Coles walk-forward | .5263 | .4511 | .3917 | 0.99036 | .58970 | .20350 |

λ selected per rung per fold: D1 = 1000, 1000, 300, 300; D2 = 1000, 1000, 300,
300. **G6 PASS at every applicable (rung, fold)** — EPV 0.96–3.98, so the gate
is live everywhere it is fitted, and nothing selected at either boundary.

**D2 − D1 = −0.00307 log loss, CI [−0.00407, −0.00208]; RPS −0.00102, CI
[−0.00135, −0.00068].** Signs agree, significant, and consistent in sign and
size across all four folds. It closes 22.6% of the D1→DC log-loss gap and 24.3%
of the RPS gap. D2 does not reach Dixon-Coles: +0.0105 behind, CI [−0.0028,
+0.0239].

### The 0.036 comparison, and the correction to it

The first reading of this run set D2 − D1 against tier 2's within-season
updating effect of **+0.0359 [+0.0186, +0.0535]** and called the 0.00307 an
8.5% shortfall — a twelvefold miss.

**That framing was wrong, and the error was the project's own, not the
instrument's.** Tier 2 measured *fresh DC minus stale DC*. D1 is not the stale
arm. Its 84 features include season-to-date PPG, last-5 form and venue form,
every one of which updates match by match. **D1 already carries recency**,
expressed as form rather than as a rating.

Placed against the arm it actually belongs beside:

| | log loss | share of the A→C recency gap |
|---|---|---|
| Arm A — stale DC | 1.0263 | 0% |
| **D1** | 1.00393 | **63%** |
| **D2** | 1.00086 | **71%** |
| Arm C — fresh DC | 0.9907 | 100% |

D2's increment is correctly sized against the wrong baseline. It is not a
twelvefold shortfall.

**Caveat, load-bearing:** this crosses model families — a penalised multinomial
logistic against a Dixon-Coles score matrix — so the absolute log losses are not
strictly commensurable. **It is recorded as an order-of-magnitude reframing, not
as a fourth arm of tier 2.** The in-ladder measurement that answers the question
without this caveat is D2 − D2static, which is a separate rung.

None of this rescues D2. D2 still lands on Elo.

### What the block actually did

`rel_elo_diff` carries it (β(H) rising 0.022 → 0.080 across folds);
`rel_attack_diff` is second; **`rel_defence_diff` and `expected_total_goals`
returned coefficients at zero at every fold.**

The diagnostic run (`scripts/phase4_ladder_diagnostics.py`) measured why:
`expected_total_goals` is standardised by an SD **25–33× larger** than the
regime it operates in, so the whole test season spans **0.14–0.25 of one
standardised unit**. `rel_defence_diff` is 6–10× inflated. The cause is the
2021-22 training rows, fitted on Dixon-Coles windows below the 380-match
burn-in, where λ_home reaches 47.26 against a maximum of 4.64 across every test
row — the same mechanism that made the passthrough diagnostic INVALID, allowed
into D2 by A2.2's ruling that the burn-in gate does not propagate.

So **D2 was effectively a two-column block.** D2 − D1 is a lower bound. The
evidence is consistent with the burn-in rows having cost something; it does not
isolate the cost, because a zero coefficient is jointly explained by compression
and by the column carrying little. Amendment 4 addresses the scale.

### On record, not acted on

Section 6's pass-through rule for boolean and indicator columns is **worse** than
standardising everything — by 0.0035 at D1 and 0.0027 at D2, and under
all-standardised D2 reaches 0.9981 and beats Elo. The rule is kept anyway. There
is a principled argument for standardising indicators (ridge's penalty should be
scale-invariant), but it existed before the numbers and was not made. Changing a
declared rule because its alternative scored better is the contaminated move.
The sensitivity stays on record; the ladder does not re-run under it.

### Evidence

`outputs/phase4_ladder_{fold_summary,pooled,deltas,lambda_curves,dynamic_coefficients,predictions,audit}.csv`,
`outputs/phase4_dynamic_state.csv`,
`outputs/phase4_ladder_{standardisation_check,burn_in_exposure}.csv`.
All under `FROZEN_MANIFEST.txt`.

DS1–DS12 in full, 0 failures. DS11 — extracted DC state through Phase 2's own
`outcome_probabilities()` — reproduces dc_walkforward to **1.11e-16**, so no D2
shortfall is attributable to state extraction. DS7 is the one declared test that
did not run: no Block C or Block X column enters D0, D1 or D2. It goes live at
D3.

D3 and D4 were not run.
