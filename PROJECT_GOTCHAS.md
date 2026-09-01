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

**Symptom, wearing two different masks:**

```
AttributeError: 'bytes' object has no attribute 'co_filename'   (from numpy.rec)
TypeError: bad argument type for built-in operation              (from _write_atomic)
AttributeError: module '__main__' has no attribute '__file__'
```

**Cause:** `phase0_evaluation_harness.py` reads `__main__.__file__` at import
time. Under `python -c` there is no `__main__.__file__`, so any import chain that
reaches the harness dies. The first two messages are the same failure surfacing
through the bytecode-cache write path, which is why it was misdiagnosed as a
corrupt `.pyc` and then as a process race. **It was neither.** Clearing
`__pycache__` does not help; it changes which mask you see.

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
