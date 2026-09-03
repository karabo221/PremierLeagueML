# Project gotchas

Environment artefacts that have masqueraded as data defects, and the general
principles they surfaced. Read this before concluding that an artefact is
corrupt, that data has been lost, or that a fix works.

---

## THE PRINCIPLE THAT COST THE MOST TO LEARN

### A fix whose verification shares an assumption with the defect is not verified

The λ-label migration is the case study, and it is worth reading even if you
never touch that code.

**The defect:** `surface["lambda"] == 0.03` silently matched zero rows, because
reading the CSV without `float_precision="round_trip"` returns a value one ULP
away from the literal `0.03`.

**The fix:** write a string label column beside every λ column, so consumers key
on a string instead of a float.

**The first attempt wrote the bare text** — `"0.03"`, `"1"`, `"10000"`. It looks
correct. It reads correctly in the file. It would have passed any visual
inspection, any `head`, any eyeball of the CSV.

It was wrong. A CSV column of numeric-looking strings is **type-inferred straight
back to `float64` on read**. `"1"` returns as `1.0`. The column would have been
the identical float key it was written to replace — while *looking* fixed, which
is strictly worse than the original bug.

It was caught only because the migration's verifier tested the **round trip**
(write, re-read, compare) rather than the **write**. A verifier that had checked
"did we write the right characters?" would have passed and shipped the defect.

The fix is now `lam=0.03`. The prefix is load-bearing: no parser can coerce it to
a number, so the column is a string even for a reader who passes no arguments.

**Generalisation:** when you fix a defect, ask what assumption the defect
depended on, then check that your *verification* does not depend on the same
assumption. Verify through the same path the consumer uses, not through the path
you just wrote.

This is the same family as the rule in `CLAUDE.md` for the other repo — *check the
measurement against the thing it measures* — but sharper: the measurement must
not inherit the bug's blind spot.

---

## 1. `python -c` fails on these scripts. Run them as FILES.

**ONE root cause, three masks.** An earlier version of this file listed these as
three separate environment artefacts. That was wrong, and the error is worth
recording because it cost real time: three symptoms were treated as three
problems, and two of the "fixes" (clearing `__pycache__`, serialising processes)
addressed nothing.

**Symptom, wearing three different masks:**

```
AttributeError: 'bytes' object has no attribute 'co_filename'   (from numpy.rec)
TypeError: bad argument type for built-in operation              (from _write_atomic)
AttributeError: module '__main__' has no attribute '__file__'
```

**Cause, singular:** `phase0_evaluation_harness.py` reads `__main__.__file__` at
import time. Under `python -c` there is no `__main__.__file__`, so **any import
chain that reaches the harness dies.**

All three messages are that one failure surfacing at different points in the
import machinery — which is why it was misdiagnosed first as a corrupt `.pyc`
and then as a process race. **It was neither.** Clearing `__pycache__` does not
help; it only changes which mask you see. Nor does avoiding concurrent
processes.

The tell that unified them: the failures tracked *which module was being
imported*, never *how many processes were running* or *how old the cache was*.

**Do this instead:** write a file and run it.

```bash
./venv/Scripts/python.exe -B scripts/whatever.py
PYTHONDONTWRITEBYTECODE=1 ./venv/Scripts/python.exe -B /path/to/scratch.py
```

`-c` is fine for anything that does **not** import the harness — plain pandas
reads of `outputs/*.csv`, for instance. That is why some `-c` invocations work and
others do not, which made the pattern hard to see.

---

## 2. A `FileNotFoundError` on a frozen input is probably not missing

**Symptom:** Instrument 5 died with

```
FileNotFoundError: outputs\phase3_reg_surface.csv
```

while that file sat on disk, unmodified, with its original timestamp.

**Cause:** a transient under concurrent access, not deletion.

**Do this:** `ls` the file and re-run **before** believing a frozen artefact was
destroyed. Check `FROZEN_MANIFEST.txt` — if the hash still matches, nothing was
lost. Do not "repair" or regenerate an artefact on the strength of one traceback;
regenerating a frozen reference is far more destructive than the error was.

---

## 3. Default float parsing: harmless for arithmetic, fatal for keys

**Measured across all 32 output CSVs:** 29 differ between default parsing and
`float_precision="round_trip"`. Largest divergence anywhere is **2.27e-13**
(Elo ratings ≈ 1810, so ~1e-16 relative).

**This does not matter for arithmetic.** A 1e-16 perturbation on a log loss of
1.0016 is nothing.

**It is fatal for `==`, `merge(on=)`, `groupby`, `isin`.** `df["lambda"] == 0.03`
matches nothing. Nothing raises. The `NaN` appears later, in whatever `mean()` is
taken over the empty selection — far from the cause.

**Do this:**
- Never key on a float that came from a CSV. Use the `lam=` label columns.
- Where a float column must be compared, read with `float_precision="round_trip"`.
- Do **not** try to fix this by setting the parameter everywhere. It protects
  nothing outside this repo and regresses the moment someone adds a call site
  without it.

Two known divergences in `phase3_ablation_ladder.py` (lines ~1009, ~1013) are
display-only and deliberately left alone.

---

## 3b. 1,900 rows on disk, 1,520 of them evaluated — pool the wrong set and get a plausible wrong number

`phase2_elo_results.csv` and `phase2_poisson_dc_results.csv` hold **all 1,900
matches**. Elo and DC predict through the training seasons too, because that is
how the state gets built. Only **1,520** are outer-test rows and only those are
scored.

Pooling per-match log loss over all 1,900 returns **1.0027**. The correct figure
is **0.9994**.

**That is the dangerous kind of wrong number: plausible.** It is the right order
of magnitude, it sits near the base rate, and nothing raises. It was caught only
because the fold mean disagreed with it — and with exactly 380 test matches per
fold the unweighted fold mean *must* equal the pooled figure, so the disagreement
was itself the detector.

**Do this:**
- Filter on the `evaluated` column (or `role == "test"`) before pooling anything.
- Assert the count is 1,520 (or 380 × folds used) rather than trusting the filter.
  `E0` in `phase2_elo_rps_supplement.py` does this; `E2` cross-checks pooled
  against the fold mean, which is what would catch a future variant of the same
  slip.

The general pattern: **when two routes to the same quantity exist, compute both
and assert they agree.** A pooled figure and a mean-of-folds figure are the same
number under equal fold sizes, so any divergence localises the bug immediately.

---

## 4. `outputs/` is gitignored except for the evidence set

The blanket ignore is right for anything rebuilt from `data/raw`. It was wrong
for artefacts later instruments **anchor** against — Instrument 5's `C1` reads
`phase3_reg_surface.csv`, Instrument 4's `R7` reads
`phase3_ablation_fold_summary.csv` — and for pre-declarations, whose entire
function is to be unamendable after the fact.

Those now have `!` exceptions and are listed in `FROZEN_MANIFEST.txt`.

```bash
./venv/Scripts/python.exe -B scripts/frozen_manifest.py --verify
```

Run that before trusting any instrument that anchors to a frozen artefact. A
moved hash is not automatically wrong — an instrument may have been legitimately
re-run — but it means the artefact is no longer the one the pre-declaration was
written against, and that has to be reconciled explicitly rather than discovered
later.

---

## 5. An isolation test must be able to recognise its own hasher

Every instrument SHA-256s `data/raw` before and after the run to prove it did not
read or write it. That hashing **opens every file in `data/raw`**, so a naive
"did anything open `data/raw`?" test flags its own integrity check.

- Instruments 4 and 5 tag hasher opens with a `_HASHING` flag.
- Phase 2 provides `access_context("label")` and `opened_paths(context=...)`.

`W9b` in `phase4_tier2_window.py` failed exactly this way on first run (140 opens
= 70 files × 2 passes) and was fixed by labelling the opens through Phase 2's own
mechanism — **not** by subtracting the expected count. Special-casing a number
would have disarmed the test for real reads too.

Related: a check must also not treat the caller's own outputs as tampering.
`R8a` flagged Instrument 5's own `phase3_frozen_regularisation.json`, because
`frozen_state()` excluded Instrument 4's outputs but knew nothing of Instrument
5's. The exclusion is now a parameter.

---

## 6. Two machines, one stick

`PremierLeagueML` lives at `D:\PremierLeagueML` here. Real work has also happened
on another laptop at `C:\Users\karab\PremierLeagueML`.

- `git config --global --add safe.directory D:/PremierLeagueML` — D: records no
  ownership, so git refuses the repo without this.
- **`.venv/` (leading dot) is dead** and cannot be revived; it was built against
  another machine's Python and has no `Scripts/`.
- **`venv/` (no dot) is the one to use.** `./venv/Scripts/python.exe`,
  pandas 3.0.5 + numpy 2.5.2, **no scipy** — nothing needs it, the Phase 3 solver
  is hand-written Newton on numpy. Bare `python` on PATH has no pandas.

---

## 7. A default-to-known classifier plus select-by-exclusion silently absorbs anything new into the control arm

**Neither half is wrong on its own.** Together they mean that adding a column
changes the thing you are measuring *against* — and the run completes, all
metrics are finite, and it reports a null.

**The two halves, as they stood:**

```python
def block_of(column):            # phase3_feature_builder.py
    ...
    return "phase1_backbone"     # anything unrecognised

def d1_features(features):       # phase4_dynamic_ladder.py
    return [c for c in features.columns if block_of(c) == "phase1_backbone"]
```

`block_of()` defaulted an unknown name to the *known* class. `d1_features()`
then selected that class **by exclusion** — "the backbone is whatever is left".
So the base rung was defined as *everything the classifier failed to recognise*.

**The failure:** E1b attached its six shot-residual columns to the feature frame
**before** reading the base. `block_of()` called all six `phase1_backbone`;
`d1_features()` swept all six into D2 rescaled. D2 came out **98 columns wide
instead of 92**, stopped reproducing its own committed Amendment 4 artefact
(7.906e-03 against a 1e-12 tolerance), and E1b "added" nothing because there was
nothing left to add.

**The comparison would still have run.** It would have compared a rung against
itself, returned a small null, and the null would have been an artefact of
feature-frame ordering. Four gates fired at once — E10, E10c, E1b-A1, DS3a — but
every one of them is a *width* or *reproduction* check downstream of the defect.
Nothing in the selection itself objected.

**The general shape, which is what to watch for:**

> A classifier whose default is a real category, consumed by a selector that
> defines a set as *the complement* of the other categories. The default makes
> new things look old; the complement makes old things a catch-all. The set you
> think is fixed is now a function of what else happens to be in scope.

It is the same family as the λ-label defect at the top of this file: the fix
looks right, reads right, and is wrong in a way no eyeball catches.

### What was done about it

Both halves are closed, and the gate is kept as well as the fix.

* **`block_of()` RAISES on an unrecognised name.** There is no default. Phase
  3's six blocks are declared as before; later phases register their own
  columns through **`declare_block(block, columns)`** at import time —
  `D_dynamic_state` for Phase 4's four state columns, `E_shot_residual` for
  E1b's six. Re-declaring a name under a *different* block is fatal; an
  identical re-declaration is idempotent, because one process may import a
  module twice.
* **`d1_features()` selects BY INCLUSION** against `PHASE1_BACKBONE_COLUMNS`, an
  explicit 86-name list in `phase3_feature_builder.py`. A column attached to the
  frame cannot get in whatever it is called and whenever it was attached. A
  *missing* backbone column is a different failure and still raises.
* **`E10e` still asserts the base directly**, and is not removed on the strength
  of the fix. A fix and its gate are not substitutes — see the λ-label case.

`B5` continues to assert the backbone count against the *built* frame, so the
declared list is checked against reality rather than trusted.

### The two vocabularies, recorded once so the next gate is written against the right one

`E10e`'s own first version asserted **88** and had to be corrected. The numbers
are both right and they count different things:

| quantity | count | what it is | asserted by |
|---|---|---|---|
| D1 **feature names** | **84** | 86 Phase 1 backbone columns − 2 held out as metadata (`home_prev_season_source`, `away_prev_season_source`) | `E10e` |
| D1 **design columns** | **88** | the same 84 after the categoricals expand through `L3.CATEGORICAL_LEVELS` | `DS3`, `DS3a` |

The expansion is +4: `home_prev_season_status` and `away_prev_season_status`
each carry three declared levels. Everything else is one name, one column.

**Write a width gate against the design; write a membership gate against the
names.** A gate written against the wrong vocabulary fails by 4 and looks like a
pipeline defect.

---

## 8. A third machine, and `venv/` does not run on it

`venv/pyvenv.cfg` records `home = C:\Users\karab\AppData\Local\Programs\Python\
Python312`. On a machine without that user, `./venv/Scripts/python.exe` exits
with

```
No Python at '"C:\Users\karab\AppData\Local\Programs\Python\Python312\python.exe'
```

which reads like a corrupt venv and is not one. **The `Lib/site-packages` tree
is intact and machine-independent** — pandas 3.0.5, numpy 2.5.2, both cp312
win_amd64 wheels.

**Do this, and do not edit `pyvenv.cfg`:** point any Python 3.12 on PATH at the
venv's packages.

```bash
PYTHONPATH='D:\PremierLeagueML\venv\Lib\site-packages' python -B scripts/whatever.py
```

Rewriting `pyvenv.cfg` would fix this machine and break the other one — the
config is per-machine state living in a directory that travels. The environment
variable is per-invocation and leaves nothing behind.

Section 1 still applies in full: **run scripts as FILES**, `-c` dies on anything
that reaches the harness.

---

## 8b. A CLONE OF THIS REPOSITORY DISAGREED WITH THE DISK, AND THE FREEZE LOST ITS HASH

Found 2026-09-03, immediately after the first push, by cloning what had just
been pushed and hashing it.

```
PHASE6_HOLDOUT_FREEZE.txt   in the E: working tree    b36befd3...248f6
PHASE6_HOLDOUT_FREEZE.txt   in a fresh clone          8373973b...
```

**Cause: `core.autocrlf=true` and no `.gitattributes`.** Git stores LF and
converts to CRLF on checkout. Nothing on the machine the files were written on
ever showed it, because that working tree was never checked out again.

**What it would have cost.** On the D: laptop, or any clone:

* `frozen_manifest.py --verify` fails on every `.txt` and `.md`
* the freeze validator's V1a fails
* `phase6_score_holdout.py` **refuses to run**, because its S0a assertion does
  not match — and it is designed to refuse, and would have been right to

The freeze's entire evidential value is its hash. A hash that holds in one
working tree only is not evidence. **Pushing did not make the evidence
portable; it made a second copy that disagreed with the first, silently.**

**The fix is `.gitattributes`, and it is not optional.** Documents, code and
the report are `-text` (no conversion, ever). The CSV artefacts and the two
frozen JSON specs are `text eol=crlf`, because that is how pandas wrote them
on Windows and how the manifest hashed them — pinned explicitly rather than
left to depend on each machine's `core.autocrlf`.

**Two more defects surfaced only in the SECOND clone**, which is the point:

* `*.json` as `-text` was wrong — those two files are CRLF on disk, so a clone
  reported both as MOVED
* `outputs/phase2_elo_metrics_full.csv` was **listed in the manifest and never
  tracked by git at all**, because `.gitignore`'s blanket `outputs/*` had no
  exception for it. The manifest called it frozen; it existed on one laptop.
  Same defect as DS7c: what the manifest freezes must be what `.gitignore`
  lets through, in both directions.

### How to check this, and it is the only way

Reading the diff cannot show it — the diff is exactly what does not change.

```bash
git clone file://E:/PremierLeagueML /tmp/clonetest
cd /tmp/clonetest
python -B scripts/frozen_manifest.py --verify        # must PASS *here*
python -B scripts/phase1_match_foundation.py          # rebuild the spine
python -B scripts/phase6_freeze_validator.py          # must PASS *here*
```

Verified 2026-09-03: manifest 121 files PASS in the clone, the rebuilt
`phase1_matches.csv` is byte-identical to E:'s, and the freeze validator
returns 35 checks / 0 failures inside the clone. `outputs/phase1_matches.csv`
is not tracked and does not need to be — it is regenerated from `data/raw`,
which is — so a fresh clone runs Phase 1 first.

---

## 9. Frozen output hashes are MACHINE-DEPENDENT. The manifest FAIL is not always drift

**Symptom:** re-running an instrument that changes nothing, on a machine that has
never run it before, and getting

```
hashes that MOVED    6
FAIL - a frozen artefact is not the one the pre-declarations were written against.
```

while the instrument itself reports **0 failures** and every printed number is
the committed one.

**Measured, on `phase5_e1b_shot_residuals.py`:**

| | |
|---|---|
| log loss | `1.0060002752144963` → `1.0060002752144961` |
| pooled RPS | `0.20762578612240679` → `...676` |
| Newton residual | `9.1325613738035827e-11` → `9.1323837381196427e-11` |
| λ, EPV, G6, design width, accuracy | **unchanged** |

Last-ULP on the solutions; the 5th significant figure of a *residual that is
itself 1e-11*. Nothing that any gate is written against moved.

**Cause: numpy dispatches its kernels on CPU features, so the summation order
inside a reduction differs between machines.** The laptop this was found on is a
Tiger Lake with AVX-512 live (`X86_V4: True`, `AVX512_ICL: True`); numpy reports
`{'baseline': ['X86_V2'], 'found': ['X86_V3']}` and uses the widest kernels the
CPU offers. A machine without AVX-512 accumulates in a different order and lands
one ULP away. Seven or eight Newton iterations amplify that into the residual
while leaving the solution at 1e-16.

**The tell that separates this from real drift:**
`phase5_e1b_residual_features.csv` did **not** move. It is pure pandas
arithmetic with no BLAS in it. Every file that moved is downstream of the
solver. *If a non-BLAS artefact moves too, it is not this — it is a real change.*

### How to tell whether a code change caused it, and do not skip this

Comparing a modified run against the *committed* artefact cannot separate "my
change" from "this machine", because both are confounded in the same diff. **Run
the unmodified code on the same machine and compare against that.**

```bash
git stash push scripts/            # park the change
<run the instrument>               # this is the control
sha256sum outputs/<the artefacts>
git checkout -- outputs && git stash pop
```

Done for the `block_of` / `d1_features` hardening of section 7: the control on
unmodified code produced hashes **identical to the hardened run** and different
from committed — so the hardening is bit-transparent and the movement is the
machine. Six of seven artefacts were byte-identical between the two runs. The
seventh was `audit.csv`, differing only in the *stdout text* DS10 captures from
its own nested `frozen_manifest.py --verify` call, which was reporting on an
untracked file in the working tree. Its verdict was PASS in both.

This is the same principle as the top of this file: the verification must not
share an assumption with the thing it is verifying. A committed artefact is not
a control when the machine is one of the variables.

### What to do about it

**Restore the committed artefacts. Do not re-freeze.** The evidence set is the
one the pre-declarations were written against; adopting this machine's bytes
would rebase it onto this machine's arithmetic and buy nothing. The internal
gates that compare at a declared tolerance — `E1b-A1` at 1e-12, `DS13`, the
pooled-equals-fold-mean identity — are the checks that actually mean something
here, and they all pass.

### And one thing this exposed about the committed set

`E1b-A1` re-fits the D2 rescaled base and compares it against the committed
`phase4_a4_fold_summary.csv`. The committed E1b audit records **5.551e-17**. On
this machine it reads **0.000e+00** — the refit lands on the Amendment 4 numbers
exactly.

**So the two committed artefacts were not produced under the same arithmetic.**
This machine reproduces the Amendment 4 one bit for bit and the E1b one not
quite. Both pass at 1e-12 and no reported number is affected. It is recorded
because "bit for bit" appears in the wording of that gate, and across two
machines that phrase means bit for bit *at the declared tolerance*, not
literally.
