# =============================================================================
# THE WEEKLY ROUTINE, ONE COMMAND
# =============================================================================
#
#     .\ops\weekly.ps1
#
# Safe to run any day, as often as you like. In order:
#
#   1. capture   phase6_capture_results.py - scorelines and closing prices for
#                every predicted match that has been played. Nothing new to
#                capture is a clean no-op.
#   2. check     ops/weekly_check.py - is there a round to write? The fixture
#                feed (fixturedownload.com) lists the whole season, so this
#                skips only when the next round is already written, or while
#                the E0 results file is behind.
#   3. generate  phase6_generate_predictions.py --round - only when step 2 says
#                so. Writes the next round and any round that has already
#                kicked off; late rows are flagged (protocol L10.2).
#   4. verify    phase6_generate_predictions.py --verify - the chain.
#
# The site needs nothing after this: it reads Supabase and revalidates every
# ten minutes. Committing the predictions mirror is the EVIDENCE step (L9.4) -
# the command is printed at the end whenever a round was written.
#
# Credentials come from ops/set-env.ps1 when it exists (gitignored). Without it
# every write is mirror-only, which the scripts report as a degraded run.
# =============================================================================

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$envFile = Join-Path $PSScriptRoot "set-env.ps1"
if (Test-Path $envFile) {
    . $envFile
}
else {
    Write-Host "  no ops/set-env.ps1 - Supabase will NOT be written (mirror-only)" -ForegroundColor Yellow
}

# venv/ runs on the laptop it was built on and nowhere else (PROJECT_GOTCHAS.md
# section 8). Elsewhere, any Python 3.12 on PATH plus the venv's packages.
# pyvenv.cfg is per-machine state and is deliberately not edited.
$python = $null
$venvPython = Join-Path $root "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    try { & $venvPython --version 2>$null | Out-Null } catch { }
    if ($LASTEXITCODE -eq 0) { $python = $venvPython }
}
if (-not $python) {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if (-not $found) {
        Write-Host "  no working python: venv/ does not run here and none is on PATH" -ForegroundColor Red
        exit 1
    }
    $python = $found.Source
    $env:PYTHONPATH = Join-Path $root "venv\Lib\site-packages"
}
$env:PYTHONDONTWRITEBYTECODE = "1"
Write-Host "  python   $python" -ForegroundColor DarkGray

function Write-Step([string]$label) {
    Write-Host ""
    Write-Host "== $label" -ForegroundColor Cyan
}

Write-Step "1. capture results"
& $python -B scripts/phase6_capture_results.py | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "  capture FAILED - stopping before anything is generated" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Step "2. is there a round to write?"
& $python -B ops/weekly_check.py | Out-Host
$check = $LASTEXITCODE

$wrote = $false
switch ($check) {
    0 {
        Write-Step "3. write the round"
        & $python -B scripts/phase6_generate_predictions.py --round | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  generator FAILED - read its STATUS block above" -ForegroundColor Red
            exit $LASTEXITCODE
        }
        $wrote = $true
    }
    11 { Write-Host "  skipped: the next round is already written" -ForegroundColor Yellow }
    13 { Write-Host "  skipped: the results file is behind - run again once it catches up" -ForegroundColor Yellow }
    12 {
        Write-Host "  STOPPED: the next round mixes written and unwritten matches" -ForegroundColor Red
        exit 12
    }
    default {
        Write-Host "  check FAILED (exit $check)" -ForegroundColor Red
        exit $check
    }
}

Write-Step "4. verify the chain"
& $python -B scripts/phase6_generate_predictions.py --verify | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "  the chain does NOT verify - investigate before anything else" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
if ($wrote) {
    Write-Host "  Round written. Commit the mirror - the commit is what timestamps it:" -ForegroundColor Green
    Write-Host "      git add live_log/ && git commit -m 'Round N predictions, pre-kickoff'" -ForegroundColor Green
}
else {
    Write-Host "  Done. No round written this run." -ForegroundColor Green
}
