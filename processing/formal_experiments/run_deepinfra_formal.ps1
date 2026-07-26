[CmdletBinding()]
param(
    [ValidateSet("all", "kqapro", "grailqa")]
    [string]$Dataset = "all",

    [string]$Python = "C:\Users\why\AppData\Local\Programs\Python\Python312\python.exe",

    [switch]$PlanOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$runner = Join-Path $PSScriptRoot "run_formal_experiment.py"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Formal experiment runner not found: $runner"
}

if (-not $env:DEEPINFRA_TOKEN) {
    $env:DEEPINFRA_TOKEN = [Environment]::GetEnvironmentVariable(
        "DEEPINFRA_TOKEN",
        "User"
    )
}

if (-not $PlanOnly -and -not $env:DEEPINFRA_TOKEN) {
    throw (
        "DEEPINFRA_TOKEN is missing. Set it in the current process or as a " +
        "persistent user environment variable before executing the experiment."
    )
}

$datasets = if ($Dataset -eq "all") {
    @("kqapro", "grailqa")
}
else {
    @($Dataset)
}

Push-Location $projectRoot
try {
    foreach ($name in $datasets) {
        $mode = if ($PlanOnly) { "plan-only validation" } else { "formal execution" }
        Write-Output ("[{0:O}] Starting {1}: {2}" -f (Get-Date), $name, $mode)

        $arguments = @(
            $runner,
            "--dataset", $name,
            "--backend", "deepinfra"
        )
        if (-not $PlanOnly) {
            $arguments += "--execute"
        }

        & $Python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "$name failed with exit code $LASTEXITCODE"
        }

        Write-Output ("[{0:O}] Finished {1}" -f (Get-Date), $name)
    }
}
finally {
    Pop-Location
}
