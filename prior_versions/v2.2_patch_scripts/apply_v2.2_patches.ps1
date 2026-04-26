# apply_v2.2_patches.ps1
# Applies DeepSeek-review response patches to dyb-2026m-elm-basin.md
# Run from the repo directory.
#
# Safe: reads file, applies all replacements in memory, writes once.
# Uses -SimpleMatch for each replacement (no regex surprises).

$path = ".\dyb-2026m-elm-basin.md"

if (-not (Test-Path $path)) {
    Write-Error "File not found: $path"
    exit 1
}

# Backup first
$backup = ".\dyb-2026m-elm-basin.backup_before_v2.2.md"
Copy-Item $path $backup
Write-Host "Backup: $backup"

$text = Get-Content -Path $path -Raw

$patches = @(
    @{
        name = "Abstract: kill 'conflated' framing"
        find = "We show this protocol cleanly separates two problems that prior work conflated."
        repl = "Prior work reported commitment-based success criteria; we add a symbolic-correctness criterion and characterize the gap systematically."
    },
    @{
        name = "Abstract: 'near' -> split honestly"
        find = "Valid snap rates peak near the function's representational depth."
        repl = "Valid snap rates peak at the function's representational depth for ln(x); for exp(x) they peak above it."
    },
    @{
        name = "Abstract: soften 'gradient escape routes'"
        find = "Extra depth helps exp (17 of 20 recover eml(x,1) at depth 4 by routing x through one subtree and collapsing the rest) but hurts ln at depth 5, which has more ways to misplace its three required gate levels."
        repl = "Extra depth correlates with higher valid-snap rate for exp (17 of 20 recover eml(x,1) at depth 4) but with lower valid-snap rate for ln at depth 5, which has more ways to misplace its three required gate levels. The mechanism behind the exp improvement is not established here."
    },
    @{
        name = "Intro: reframe prior-work characterization"
        find = "Prior work checked only whether selector weights reached simplex vertices, not whether the resulting expression matched the target."
        repl = "Prior work reported whether selector weights reached simplex vertices but did not systematically separate that from whether the resulting expression matched the target."
    },
    @{
        name = "Section 3.3: explain 0.000 variance + drop 'basin size'"
        find = "**exp d=2** (5 valid / 15 false). All 15 false snaps land on eml(x,x) = exp(x) - ln(x), with post_snap_loss 0.688 (std 0.000 to 4 decimals). The depth-2 tree has exactly 4 possible snapped forms. Only one (eml(x,1)) is correct; one other (eml(x,x)) is a genuine approximation of exp(x) on [-2, 2] and captures 75% of seeds. Basin *size* dominates over basin *count* at this depth."
        repl = "**exp d=2** (5 valid / 15 false). All 15 false snaps land on eml(x,x) = exp(x) - ln(x), with post_snap_loss 0.688 (std 0.000 to 4 decimals). The zero variance reflects identical symbolic forms across seeds: once snapping commits every selector to the same vertices, the resulting function is deterministic and its evaluation is identical across runs. The depth-2 tree has exactly 4 possible snapped forms. Only one (eml(x,1)) is correct; one other (eml(x,x)) is a genuine approximation of exp(x) on [-2, 2] and captures 75% of seeds. At this depth, the fraction of seeds captured by the eml(x,x) form (15/20) exceeds the fraction captured by any other non-target form, which explains why extra configuration count at higher depths is compatible with improved recovery."
    },
    @{
        name = "Section 3.4: soften causal claim about gradient paths"
        find = "We flag it because Section 3.2's finding that exp improves with depth depends on this mechanism: the extra gates do not represent more of exp(x), they provide alternative gradient paths out of the eml(x,x) basin."
        repl = "The co-occurrence of this subtree-collapse pattern with improved valid-snap rates at d=4 is consistent with multiple mechanisms (more parameters to fit during phase 1, alternative gradient paths, or simply a larger basin of attraction for the correct form); distinguishing between these would require ablation experiments not conducted here."
    },
    @{
        name = "Conclusion: 'solves completely' -> 'across tested conditions'"
        find = "Temperature annealing in EML expression trees solves the commitment problem: every selector reaches a simplex vertex at every depth we tested."
        repl = "Temperature annealing in EML expression trees solves the commitment problem across all tested conditions: every selector reaches a simplex vertex at every depth we tested."
    }
)

$applied = 0
$missed = @()

foreach ($p in $patches) {
    if ($text.Contains($p.find)) {
        $text = $text.Replace($p.find, $p.repl)
        Write-Host "  APPLIED: $($p.name)"
        $applied++
    } else {
        Write-Host "  MISSED : $($p.name)" -ForegroundColor Yellow
        $missed += $p.name
    }
}

# Try v2.1 version line replacement first, fall back to v2.0.1
$v21_line = 'v2.1 April 24 2026 (adds Sect. 4 connection to Odrzywolek SI warm-start evidence; extended reference list)'
$v201_line = 'v2.0.1 April 24 2026 (Hebrew font and typos, url to arXiv:2603.21852v2)'
$v22_v21_repl = 'v2.2 April 24 2026 (v2.1: SI warm-start subsection + Figure 2. v2.2: responds to review feedback: tones down novelty framing re Odrzywolek, removes unsupported gradient-escape mechanism claim, softens "solves completely" to tested-conditions, explains 0.000 variance)'
$v22_v201_repl = 'v2.2 April 24 2026 (responds to review feedback: tones down novelty framing re Odrzywolek, removes unsupported gradient-escape mechanism claim, softens "solves completely" to tested-conditions, explains 0.000 variance)'

if ($text.Contains($v21_line)) {
    $text = $text.Replace($v21_line, $v22_v21_repl)
    Write-Host "  APPLIED: Version bump v2.1 -> v2.2"
    $applied++
} elseif ($text.Contains($v201_line)) {
    $text = $text.Replace($v201_line, $v22_v201_repl)
    Write-Host "  APPLIED: Version bump v2.0.1 -> v2.2"
    $applied++
} else {
    Write-Host "  MISSED : version line (edit by hand)" -ForegroundColor Yellow
    $missed += "version line"
}

# Write once, preserving LF line endings
[System.IO.File]::WriteAllText((Resolve-Path $path), $text)

Write-Host ""
Write-Host "Applied $applied patches."
if ($missed.Count -gt 0) {
    Write-Host "Missed $($missed.Count) patches. Fix by hand:" -ForegroundColor Yellow
    $missed | ForEach-Object { Write-Host "  - $_" }
}
Write-Host ""
Write-Host "Diff to review:"
Write-Host "  git diff --word-diff=color $path"
