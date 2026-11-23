# Runs a whole training group on this PC: the coordinator and N group workers
# (group_worker_tool), on the ArrayFire CPU route.
#
#   python make_job.py <job folder> <workers> '<node type numbers as JSON>'
#   .\run_local.ps1 -JobDir <job folder> -Coordinator <cyxwiz-coordinator.exe> -Tool <group_worker_tool.exe>
#
# -Scenario plain     every worker runs to the end of the job
#           drop      the last worker is stopped hard after -AfterSeconds; the group goes on without it
#           restart   the coordinator is stopped hard after -AfterSeconds and started again; the workers join again
#
# Prints one line per worker and a summary of the coordinator's rounds. Exit code 0 when the job completed.
param(
    [Parameter(Mandatory = $true)][string]$JobDir,
    [Parameter(Mandatory = $true)][string]$Coordinator,
    [Parameter(Mandatory = $true)][string]$Tool,
    [ValidateSet('plain', 'drop', 'restart')][string]$Scenario = 'plain',
    [int]$AfterSeconds = 15,
    [int]$TimeoutSeconds = 600
)

$job = Get-Content (Join-Path $JobDir 'job.json') -Raw | ConvertFrom-Json
$workers = $job.workers.Count
$address = $job.listen -replace '^0\.0\.0\.0', '127.0.0.1'
Remove-Item (Join-Path $JobDir 'state') -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $JobDir -Filter '*.log' | Remove-Item -Force
Get-ChildItem $JobDir -Filter '*.err' | Remove-Item -Force

function Start-Coordinator([string]$suffix) {
    Start-Process -FilePath $Coordinator -ArgumentList (Join-Path $JobDir 'job.json') -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $JobDir "coordinator$suffix.log") -RedirectStandardError (Join-Path $JobDir "coordinator$suffix.err")
}

$coordinatorProcess = Start-Coordinator ''
Start-Sleep -Seconds 1
$procs = 0..($workers - 1) | ForEach-Object {
    Start-Process -FilePath $Tool -PassThru -WindowStyle Hidden -WorkingDirectory (Split-Path $Tool) `
        -ArgumentList "--coordinator=$address", "--ticket-file=$(Join-Path $JobDir "node$_.ticket")", '--cpu' `
        -RedirectStandardOutput (Join-Path $JobDir "worker$_.log") -RedirectStandardError (Join-Path $JobDir "worker$_.err")
}
$clock = [Diagnostics.Stopwatch]::StartNew()

if ($Scenario -eq 'drop') {
    Start-Sleep -Seconds $AfterSeconds
    Stop-Process -Id $procs[-1].Id -Force
    "dropped worker $($workers - 1) after $AfterSeconds s"
}
if ($Scenario -eq 'restart') {
    Start-Sleep -Seconds $AfterSeconds
    Stop-Process -Id $coordinatorProcess.Id -Force
    "stopped the coordinator after $AfterSeconds s"
    Start-Sleep -Seconds 3
    $coordinatorProcess = Start-Coordinator '2'
    'started the coordinator again'
}

foreach ($proc in $procs) {
    $left = [Math]::Max(1, $TimeoutSeconds - [int]$clock.Elapsed.TotalSeconds)
    if (-not $proc.HasExited -and -not $proc.WaitForExit($left * 1000)) {
        "worker pid $($proc.Id) did not finish in $TimeoutSeconds s: stopped"
        Stop-Process -Id $proc.Id -Force
    }
}
if (-not $coordinatorProcess.WaitForExit(20000)) { Stop-Process -Id $coordinatorProcess.Id -Force; 'the coordinator did not stop by itself: stopped' }
"elapsed $([int]$clock.Elapsed.TotalSeconds) s"

$plain = { param($path) if (Test-Path $path) { (Get-Content $path) -replace '\x1b\[[0-9;]*m', '' } }
$coordinatorLines = @(& $plain (Join-Path $JobDir 'coordinator.log')) + @(& $plain (Join-Path $JobDir 'coordinator2.log'))
$rounds = @($coordinatorLines | Select-String 'round committed')
$completed = [bool]($coordinatorLines | Select-String 'the job has ended.*completed')
"coordinator: $($rounds.Count) rounds committed; completed: $completed"
$coordinatorLines | Select-String 'going on from|out of the group|round abandoned|the job has ended|not started' |
    ForEach-Object { '  ' + ($_.Line -replace '^\S+\s+', '') }
if ($rounds.Count -gt 0) {
    $pick = @($rounds[0], $rounds[[int]($rounds.Count / 2)], $rounds[-1]) | Select-Object -Unique
    foreach ($line in $pick) { '  ' + (($line.Line -split '\s+' | Where-Object { $_ -match '^(round|used|rows|epochs|loss)=' }) -join ' ') }
}
0..($workers - 1) | ForEach-Object {
    $lines = & $plain (Join-Path $JobDir "worker$_.log")
    $steps = @($lines | Select-String 'Group worker: round \d+: (\d+) steps' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value })
    $last = $lines | Select-String '^(Group finished|Removed from the group|Could not go on|Not in a group|Joining|Training)' | Select-Object -Last 1
    $stepText = if ($steps.Count -gt 0) { "steps per round first/median/last $($steps[0])/$(($steps | Sort-Object)[[int]($steps.Count / 2)])/$($steps[-1])" } else { 'no round' }
    "worker ${_}: $($steps.Count) rounds; $stepText; $(if ($last) { $last.Line } else { 'no final line (stopped hard)' })"
}
if ($completed) { exit 0 } else { exit 1 }
