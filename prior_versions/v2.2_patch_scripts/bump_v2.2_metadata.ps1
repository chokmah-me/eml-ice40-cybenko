# bump_v2.2_metadata.ps1
# Updates CITATION.cff, .zenodo.json, README.md to v2.2.0.
# Idempotent: safe to re-run. Reports what changed.
# Run from repo directory.

$VERSION = "2.2.0"
$DATE    = "2026-04-24"
$CHANGELOG_ENTRY = "- **v2.2** ($DATE): Response to review feedback. Tones down novelty framing relative to Odrzywolek (no longer claims prior work conflated commitment and validity). Removes unsupported gradient-escape-routes mechanism claim. Softens ""solves completely"" to ""solves across tested conditions"". Explains 0.000 variance in exp d=2 false snaps."

$changes = @()

# ---- CITATION.cff ----
$cffPath = ".\CITATION.cff"
if (Test-Path $cffPath) {
    $cff = Get-Content $cffPath -Raw
    $newCff = $cff `
        -replace 'version:\s*"[^"]*"',          "version: ""$VERSION""" `
        -replace "date-released:\s*`"[^`"]*`"", "date-released: ""$DATE"""
    if ($newCff -ne $cff) {
        [System.IO.File]::WriteAllText((Resolve-Path $cffPath), $newCff)
        $changes += "CITATION.cff: bumped to $VERSION / $DATE"
    } else {
        $changes += "CITATION.cff: already at $VERSION (no change)"
    }
} else {
    $changes += "CITATION.cff: NOT FOUND (skipped)"
}

# ---- .zenodo.json ----
$zenPath = ".\.zenodo.json"
if (Test-Path $zenPath) {
    $zen = Get-Content $zenPath -Raw
    # Update version field if present (formats vary; handle "version": "x.y.z")
    $newZen = $zen -replace '("version"\s*:\s*")[^"]*(")', "`${1}$VERSION`${2}"
    # Update publication_date if present
    $newZen = $newZen -replace '("publication_date"\s*:\s*")[^"]*(")', "`${1}$DATE`${2}"
    if ($newZen -ne $zen) {
        [System.IO.File]::WriteAllText((Resolve-Path $zenPath), $newZen)
        $changes += ".zenodo.json: bumped to $VERSION / $DATE"
    } else {
        $changes += ".zenodo.json: no version/date fields to update or already current"
    }
    # Validate JSON still parses
    try {
        $null = Get-Content $zenPath -Raw | ConvertFrom-Json
        $changes += ".zenodo.json: valid JSON"
    } catch {
        $changes += ".zenodo.json: WARNING - JSON parse failed after edit, restore manually"
    }
} else {
    $changes += ".zenodo.json: NOT FOUND (skipped)"
}

# ---- README.md ----
$readmePath = ".\README.md"
if (Test-Path $readmePath) {
    $readme = Get-Content $readmePath -Raw
    if ($readme -match [regex]::Escape("**v$VERSION**")) {
        $changes += "README.md: v$VERSION entry already present (no change)"
    } else {
        # Find a Changelog heading and insert under it; otherwise append
        if ($readme -match '(?ms)^(##\s*Changelog\s*\r?\n)') {
            $newReadme = $readme -replace '(?ms)^(##\s*Changelog\s*\r?\n)', "`$1`n$CHANGELOG_ENTRY`n"
            $changes += "README.md: prepended v$VERSION under Changelog heading"
        } else {
            $newReadme = $readme.TrimEnd() + "`n`n## Changelog`n`n$CHANGELOG_ENTRY`n"
            $changes += "README.md: appended new Changelog section with v$VERSION entry"
        }
        [System.IO.File]::WriteAllText((Resolve-Path $readmePath), $newReadme)
    }
} else {
    $changes += "README.md: NOT FOUND (skipped)"
}

# ---- Report ----
Write-Host ""
Write-Host "=== v2.2 metadata bump ==="
$changes | ForEach-Object { Write-Host "  $_" }
Write-Host ""
Write-Host "Review:"
Write-Host "  git diff CITATION.cff .zenodo.json README.md"
Write-Host ""
Write-Host "If happy:"
Write-Host "  git add CITATION.cff .zenodo.json README.md"
Write-Host "  git commit -m 'v2.2 metadata bump'"
Write-Host "  git push"
