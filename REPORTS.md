# Reports

One entry per completed instrument, newest first. Each states what was
measured, what came back, and what it does not license.

The rule this file exists for: **a number without its interval and its gate
status is not a result.** Every figure below is on the four frozen outer folds
and the same 1,520 outer-test matches unless it says otherwise.

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
