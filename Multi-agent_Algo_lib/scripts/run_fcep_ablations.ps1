param(
    [int[]]$Seeds = @(321, 322, 323),
    [int]$TotalSteps = 500000,
    [string]$PythonExe = "D:\Tools\Anaconda\python.exe",
    [string]$OutputRoot = "artifacts\fcep_ablation",
    [int]$FlowWarmupSteps = 0
)

$ErrorActionPreference = "Stop"
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptRoot "..\..")).Path
$TrainingScript = Join-Path $ScriptRoot "train_masac_multi_agent_dmp.py"

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

$Profiles = @(
    @{ Name = "A_baseline"; PhaseMode = "classic"; FlowLoss = $false },
    @{ Name = "B_phase_only"; PhaseMode = "fcep"; FlowLoss = $false },
    @{ Name = "C_constraint_only"; PhaseMode = "classic"; FlowLoss = $true },
    @{ Name = "D_full"; PhaseMode = "fcep"; FlowLoss = $true }
)

Push-Location $ProjectRoot
try {
    foreach ($Seed in $Seeds) {
        foreach ($Profile in $Profiles) {
            $ProfileOutput = Join-Path $OutputRoot $Profile.Name
            $CommonArgs = @(
                $TrainingScript,
                "--seed", $Seed,
                "--total-steps", $TotalSteps,
                "--output-root", $ProfileOutput,
                "--phase-integrator", "exponential",
                "--phase-min", "1e-4",
                "--flow-zero-threshold", "1e-4",
                "--flow-consistency-weight", "0.01",
                "--minimum-flow-consistency", "0.0",
                "--flow-consistency-warmup-steps", $FlowWarmupSteps
            )
            $FlowSwitch = if ($Profile.FlowLoss) {
                "--enable-flow-consistency-loss"
            } else {
                "--no-enable-flow-consistency-loss"
            }
            & $PythonExe @CommonArgs --phase-mode $Profile.PhaseMode $FlowSwitch
            if ($LASTEXITCODE -ne 0) {
                throw "Ablation failed: profile=$($Profile.Name), seed=$Seed"
            }
        }
    }
}
finally {
    Pop-Location
}
