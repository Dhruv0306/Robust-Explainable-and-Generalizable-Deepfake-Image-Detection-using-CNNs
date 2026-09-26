$ErrorActionPreference = "Stop"

$baselineRoot = ".\data\output\baseline_on_lomo_holdouts"
$lomoCsv = ".\data\output\lomo\lomo_seed_metrics.csv"
$outputRoot = ".\data\output\baseline_lomo_comparison"

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$metrics = @("accuracy", "precision", "recall", "f1", "roc_auc")

# Load LOMO seed-level metrics and index by holdout/model/seed.
$lomoRows = Import-Csv $lomoCsv
$lomoIndex = @{}

foreach ($row in $lomoRows) {
    $key = "$($row.holdout)|$($row.model)|$($row.seed)"
    if ($lomoIndex.ContainsKey($key)) {
        throw "Duplicate LOMO row for key: $key"
    }
    $lomoIndex[$key] = $row
}

# Read each baseline result and pair it with the matching LOMO row.
$pairedRows = @()
$baselineFiles = Get-ChildItem $baselineRoot -Recurse -Filter "test_results.json" -File

foreach ($file in $baselineFiles) {
    $relativePath = [System.IO.Path]::GetRelativePath(
        (Resolve-Path $baselineRoot).Path,
        $file.FullName
    )
    $parts = $relativePath -split '[\\/]'

    if ($parts.Count -lt 3) {
        Write-Warning "Skipping unexpected baseline path: $relativePath"
        continue
    }

    $holdout = $parts[0]
    $runFolder = $parts[1]

    if ($runFolder -notmatch '^(xception|resnet50|efficientnet_b0)_seed(\d+)$') {
        Write-Warning "Skipping unrecognized model/seed folder: $runFolder"
        continue
    }

    $model = $Matches[1]
    $seed = [int]$Matches[2]
    $key = "$holdout|$model|$seed"

    if (-not $lomoIndex.ContainsKey($key)) {
        throw "No matching LOMO metrics row for baseline run: $key"
    }

    $result = Get-Content $file.FullName -Raw | ConvertFrom-Json

    if ($null -eq $result.video_metrics.mean) {
        throw "Missing video_metrics.mean in: $($file.FullName)"
    }

    $baselineMetrics = $result.video_metrics.mean
    $lomoRow = $lomoIndex[$key]

    $out = [ordered]@{
        holdout = $holdout
        model = $model
        seed = $seed
        baseline_file = $relativePath
        lomo_run_folder = $lomoRow.run_folder
    }

    foreach ($metric in $metrics) {
        $baselineValue = [double]$baselineMetrics.$metric
        $lomoValue = [double]$lomoRow.$metric

        $out["baseline_$metric"] = $baselineValue
        $out["lomo_$metric"] = $lomoValue
        $out["delta_$metric"] = $lomoValue - $baselineValue
    }

    $pairedRows += [pscustomobject]$out
}

# Verify expected pair count and unique pair keys.
$pairKeys = $pairedRows | ForEach-Object {
    "$($_.holdout)|$($_.model)|$($_.seed)"
} | Sort-Object -Unique

if ($pairedRows.Count -ne 36 -or $pairKeys.Count -ne 36) {
    throw "Expected 36 unique paired runs, found $($pairedRows.Count) rows and $($pairKeys.Count) unique keys."
}

$pairedPath = Join-Path $outputRoot "paired_seed_metrics.csv"
$pairedRows |
    Sort-Object holdout, model, seed |
    Export-Csv $pairedPath -NoTypeInformation

# Calculate mean and sample SD for baseline, LOMO, and paired deltas.
function Get-Mean([double[]]$values) {
    return ($values | Measure-Object -Average).Average
}

function Get-SampleSd([double[]]$values) {
    if ($values.Count -lt 2) {
        return [double]::NaN
    }

    $mean = Get-Mean $values
    $sumSquares = 0.0
    foreach ($value in $values) {
        $sumSquares += [math]::Pow(($value - $mean), 2)
    }

    return [math]::Sqrt($sumSquares / ($values.Count - 1))
}

$summaryRows = @()

foreach ($group in ($pairedRows | Group-Object holdout, model)) {
    $runs = @($group.Group)
    $holdout = $runs[0].holdout
    $model = $runs[0].model

    $summary = [ordered]@{
        holdout = $holdout
        model = $model
        n_seeds = $runs.Count
    }

    foreach ($metric in $metrics) {
        $baselineValues = [double[]]@($runs | ForEach-Object { $_."baseline_$metric" })
        $lomoValues = [double[]]@($runs | ForEach-Object { $_."lomo_$metric" })
        $deltaValues = [double[]]@($runs | ForEach-Object { $_."delta_$metric" })

        $summary["baseline_${metric}_mean"] = Get-Mean $baselineValues
        $summary["baseline_${metric}_sd"] = Get-SampleSd $baselineValues
        $summary["lomo_${metric}_mean"] = Get-Mean $lomoValues
        $summary["lomo_${metric}_sd"] = Get-SampleSd $lomoValues
        $summary["delta_${metric}_mean"] = Get-Mean $deltaValues
        $summary["delta_${metric}_sd"] = Get-SampleSd $deltaValues
    }

    $summaryRows += [pscustomobject]$summary
}

$summaryPath = Join-Path $outputRoot "paired_summary_by_holdout_model.csv"
$summaryRows |
    Sort-Object holdout, model |
    Export-Csv $summaryPath -NoTypeInformation

Write-Host "Paired runs checked: $($pairedRows.Count)"
Write-Host "Seed-level output: $pairedPath"
Write-Host "Summary output: $summaryPath"