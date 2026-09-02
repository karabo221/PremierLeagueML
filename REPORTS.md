# Reports

One entry per completed instrument, newest first. Each states what was
measured, what came back, and what it does not license.

The rule this file exists for: **a number without its interval and its gate
status is not a result.** Every figure below is on the four frozen outer folds
and the same 1,520 outer-test matches unless it says otherwise.

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
