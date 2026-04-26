# release_v2.2.ps1
# Cleans up repo and creates v2.2.0 release.
# Idempotent: skips steps already done. Bails on first error.
# Run from repo directory.
#
# Optional: pass -DryRun to print what would happen without doing it.
#   .\release_v2.2.ps1 -DryRun

[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$VERSION = "v2.2.0"

function Do-It {
    param([string]$Description, [scriptblock]$Action)
    if ($DryRun) {
        Write-Host "  [DRY] $Description" -ForegroundColor Cyan
    } else {
        Write-Host "  $Description"
        & $Action
    }
}

Write-Host ""
Write-Host "=== v2.2 release ===" -ForegroundColor Green
if ($DryRun) { Write-Host "DRY RUN MODE" -ForegroundColor Yellow }
Write-Host ""

# --- Sanity: are we in the right place? ---
if (-not (Test-Path .\dyb-2026m-elm-basin.md)) {
    Write-Error "Not in repo root (dyb-2026m-elm-basin.md missing)"
    exit 1
}

# --- Step 1: Move stale v2.1 PDF to prior_versions ---
if (Test-Path .\dyb-2026m-elm-basin_v2.1.pdf) {
    if (-not (Test-Path .\prior_versions)) {
        Do-It "Create prior_versions/" { New-Item -ItemType Directory .\prior_versions | Out-Null }
    }
    Do-It "Move v2.1 PDF to prior_versions/" {
        Move-Item .\dyb-2026m-elm-basin_v2.1.pdf .\prior_versions\
    }
} else {
    Write-Host "  v2.1 PDF already moved (skip)"
}

# --- Step 2: Remove Odrzywolek PDFs from repo ---
$odrPdfs = @('2603.21852v2.pdf', '2603.21852v2-SI.pdf')
foreach ($pdf in $odrPdfs) {
    if (Test-Path ".\$pdf") {
        Do-It "git rm $pdf" {
            git rm $pdf | Out-Null
        }
    } else {
        Write-Host "  $pdf already removed (skip)"
    }
}

# --- Step 3: Add ignore pattern (only if not present) ---
$gitignore = Get-Content .\.gitignore -Raw -ErrorAction SilentlyContinue
if (-not ($gitignore -match '2603\.21852v2\*\.pdf')) {
    Do-It "Add Odrzywolek pattern to .gitignore" {
        Add-Content .\.gitignore "`n# upstream papers not to redistribute`n2603.21852v2*.pdf"
    }
} else {
    Write-Host "  .gitignore already has Odrzywolek pattern (skip)"
}

# --- Step 4: Verify v2.2 PDF exists ---
if (-not (Test-Path .\dyb-2026m-elm-basin.pdf) -and -not (Test-Path .\dyb-2026m-elm-basin_v2.2.pdf)) {
    Write-Error "No v2.2 PDF found. Re-export from Typora first."
    exit 1
}
Write-Host "  v2.2 PDF present (ok)"

# --- Step 5: Show staging state, eyeball gate ---
Write-Host ""
Write-Host "Current git status:" -ForegroundColor Yellow
git status --short

if ($DryRun) {
    Write-Host ""
    Write-Host "Dry run complete. Re-run without -DryRun to execute." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
$confirm = Read-Host "Proceed to stage all + commit + tag + push? (y/N)"
if ($confirm -ne 'y' -and $confirm -ne 'Y') {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit 0
}

# --- Step 6: Stage, commit, tag, push ---
Do-It "git add -A" { git add -A }
Do-It "git commit" {
    git commit -m "Release v2.2: review-response patches, metadata bump, repo cleanup"
}
Do-It "git tag $VERSION" {
    git tag -a $VERSION -m "v2.2.0: review-response patches, mechanism claim softened"
}
Do-It "git push origin main --tags" { git push origin main --tags }

# --- Step 7: Create GitHub release ---
Write-Host ""
$releaseNotes = @"
## v2.2.0: Review-response patches

Responds to external review feedback on the v2.1 release.

### Changed
- **Abstract and Introduction:** removed claim that prior work conflated commitment and validity. New framing: prior work reported commitment-based criteria; we add a symbolic-correctness criterion and characterize the gap systematically.
- **Section 3.4:** removed unsupported gradient-escape-routes mechanism claim. The co-occurrence of subtree-collapse with improved valid-snap rates at d=4 is now described as observation, with the mechanism flagged as not established here.
- **Conclusion:** "solves the commitment problem" → "solves the commitment problem across all tested conditions."
- **Section 3.3:** added explanation of the 0.000 variance in exp d=2 false snaps (deterministic snapped form, not numerical artifact).
- **Abstract:** "peak near representational depth" → split into accurate per-function statements (ln peaks at, exp peaks above).
- **Repo:** Odrzywolek's PDFs removed from tracking (now in .gitignore); v2.1 PDF archived to prior_versions/.

### Unchanged
- Empirical results (Table 1, Figures 1-2): no data has changed.
- v2.1 contributions (Section 4 connection to Odrzywolek SI warm-start evidence; Figure 2 rate comparison; odrzywolek_si_table_s5.csv) are retained.

### Files
- ``dyb-2026m-elm-basin.md`` (paper draft, v2.2)
- ``dyb-2026m-elm-basin.pdf`` (rendered)
- ``dyb-2026m-elm-basin_v2.2_tldr.md`` (TL;DR summaries, five perspectives)
- ``snapping_v2_final.csv`` (240 runs, this work)
- ``odrzywolek_si_table_s5.csv`` (transcribed from Odrzywolek 2026 SI Table S5)
- ``figure1_heatmap_v2.png``, ``figure2_rate_comparison.png``
- ``eml_layer_v2.py``, ``experiment_v2.py``, ``make_figure2.py``
"@

# Try gh CLI; fall back to instructions
$ghAvailable = $null -ne (Get-Command gh -ErrorAction SilentlyContinue)
if ($ghAvailable) {
    Write-Host "Creating GitHub release via gh CLI..."
    $tmpNotes = Join-Path $env:TEMP "v2.2_release_notes.md"
    $releaseNotes | Set-Content -NoNewline $tmpNotes
    gh release create $VERSION --title $VERSION --notes-file $tmpNotes
    Remove-Item $tmpNotes
    Write-Host ""
    Write-Host "Release created. Zenodo webhook will fire within 1-2 minutes." -ForegroundColor Green
    Write-Host "Check: https://zenodo.org/account/settings/github/"
} else {
    Write-Host "gh CLI not installed. Create release manually:" -ForegroundColor Yellow
    Write-Host "  1. Visit https://github.com/chokmah-me/eml-ice40-cybenko/releases/new"
    Write-Host "  2. Choose tag: $VERSION"
    Write-Host "  3. Title: $VERSION"
    Write-Host "  4. Paste these notes:"
    Write-Host ""
    Write-Host "--- BEGIN NOTES ---" -ForegroundColor DarkGray
    Write-Host $releaseNotes
    Write-Host "--- END NOTES ---" -ForegroundColor DarkGray
    Write-Host ""
    Write-Host "Or install gh: winget install GitHub.cli"
}
