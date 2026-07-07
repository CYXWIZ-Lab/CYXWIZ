param(
    [Parameter(Position = 0)]
    [string]$Question,

    [string]$Preset,

    [string]$Endpoint = "http://127.0.0.1:8768/completion",

    [int]$Top = 1,

    [int]$TimeoutSeconds = 120,

    [string]$PythonExe = "python",

    [string]$PacketPath = "$env:TEMP\tofix42_mytest_packet.json",

    [string[]]$SourceType,

    [string[]]$PathContains,

    [string[]]$TitleContains,

    [string[]]$Tag,

    [switch]$ShowPrompt,

    [switch]$RetrievalOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..\..")).Path

$phase1a = Join-Path $scriptDir "phase1a_retrieval.py"
$phase1b = Join-Path $scriptDir "phase1b_answer.py"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-PythonJson {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    $output = & $PythonExe @Arguments
    return @{
        output = $output
        exit_code = $LASTEXITCODE
    }
}

function Add-FilterArgs {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.Generic.List[string]]$Arguments
    )

    foreach ($value in ($SourceType | Where-Object { $_ })) {
        $Arguments.Add("--source-type")
        $Arguments.Add($value)
    }
    foreach ($value in ($PathContains | Where-Object { $_ })) {
        $Arguments.Add("--path-contains")
        $Arguments.Add($value)
    }
    foreach ($value in ($TitleContains | Where-Object { $_ })) {
        $Arguments.Add("--title-contains")
        $Arguments.Add($value)
    }
    foreach ($value in ($Tag | Where-Object { $_ })) {
        $Arguments.Add("--tag")
        $Arguments.Add($value)
    }
}

function Get-SuspectReason {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Packet,

        [pscustomobject]$ProbeReport
    )

    if ($null -eq $ProbeReport) {
        return ""
    }

    $answerText = ""
    if ($null -ne $ProbeReport.answer) {
        $answerText = [string]$ProbeReport.answer
    }
    if ([string]::IsNullOrWhiteSpace($answerText)) {
        return ""
    }

    $questionText = ""
    if ($null -ne $Packet.question) {
        $questionText = [string]$Packet.question
    }
    $evidenceText = (@($Packet.evidence) | ForEach-Object {
        if ($null -ne $_.text) {
            [string]$_.text
        }
        else {
            ""
        }
    }) -join "`n"
    $topSourceType = ""
    if (@($Packet.evidence).Count -gt 0 -and $Packet.evidence[0].citation) {
        if ($null -ne $Packet.evidence[0].citation.source_type) {
            $topSourceType = [string]$Packet.evidence[0].citation.source_type
        }
    }

    $proceduralQuestion = $questionText -match '^\s*(how to|how do i|how can i|how do we|build|create|use)\b'
    $inventedCommand = (
        ($answerText -match 'cyxwiz-engine') -and
        ($evidenceText -notmatch 'cyxwiz-engine')
    )
    if ($inventedCommand) {
        return "Answer introduced a cyxwiz-engine command that does not appear in the retrieved evidence."
    }

    $inventedCliFlags = (
        ($answerText -match '--configure|--apply|--train') -and
        ($evidenceText -notmatch '--configure|--apply|--train')
    )
    if ($inventedCliFlags) {
        return "Answer introduced CLI flags or commands that are not present in the retrieved evidence."
    }

    $proceduralGuideFromSourceOnly = (
        $proceduralQuestion -and
        $topSourceType -eq "source" -and
        $answerText -match 'step-by-step|1\.|2\.|3\.' -and
        $evidenceText -notmatch 'example|usage|how to|step-by-step'
    )
    if ($proceduralGuideFromSourceOnly) {
        return "Answer looks procedural, but the retrieved evidence is implementation source rather than explicit usage documentation."
    }

    return ""
}

function Get-TestClassification {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Packet,

        [pscustomobject]$ProbeReport,

        [switch]$RetrievalOnlyMode
    )

    $evidenceCount = @($Packet.evidence).Count
    if ($evidenceCount -eq 0) {
        return @{
            classification = "answer_suspect"
            reason = "No retrieval evidence was found."
            suspicion = ""
        }
    }

    if ($RetrievalOnlyMode) {
        return @{
            classification = "retrieval_pass"
            reason = "Retrieval returned at least one evidence chunk."
            suspicion = ""
        }
    }

    if ($null -eq $ProbeReport) {
        return @{
            classification = "runtime_fail"
            reason = "Probe report was not produced."
            suspicion = ""
        }
    }

    if (-not [string]::IsNullOrWhiteSpace([string]$ProbeReport.error)) {
        return @{
            classification = "runtime_fail"
            reason = [string]$ProbeReport.error
            suspicion = ""
        }
    }

    $suspectReason = Get-SuspectReason -Packet $Packet -ProbeReport $ProbeReport

    $missing = @($ProbeReport.sections_missing)
    if (-not [bool]$ProbeReport.parsed -or $missing.Count -gt 0) {
        $detail = if ($missing.Count -gt 0) {
            "Missing sections: $($missing -join ', ')"
        }
        else {
            "Structured answer was not parsed."
        }
        return @{
            classification = "format_fail"
            reason = $detail
            suspicion = $suspectReason
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($suspectReason)) {
        return @{
            classification = "answer_suspect"
            reason = $suspectReason
            suspicion = $suspectReason
        }
    }

    if ([bool]$ProbeReport.ok) {
        return @{
            classification = "retrieval_pass"
            reason = "Probe returned all required sections from retrieved evidence."
            suspicion = ""
        }
    }

    return @{
        classification = "answer_suspect"
        reason = "Probe completed but did not satisfy the success checks."
        suspicion = $suspectReason
    }
}

Write-Host "Repo root: $repoRoot"
Write-Host "Question: $Question"
Write-Host "Preset: $Preset"
Write-Host "Packet: $PacketPath"
Write-Host ""

if ([string]::IsNullOrWhiteSpace($Question) -and [string]::IsNullOrWhiteSpace($Preset)) {
    throw "Provide -Preset or a question."
}

Push-Location $repoRoot
try {
    $scriptExitCode = 0
    Write-Host "[1/3] Building retrieval packet..."
    $packetArgs = [System.Collections.Generic.List[string]]::new()
    $null = $packetArgs.Add($phase1a)
    $null = $packetArgs.Add("packet")
    if (-not [string]::IsNullOrWhiteSpace($Question)) {
        $null = $packetArgs.Add($Question)
    }
    if (-not [string]::IsNullOrWhiteSpace($Preset)) {
        $null = $packetArgs.Add("--preset")
        $null = $packetArgs.Add($Preset)
    }
    $null = $packetArgs.Add("--top")
    $null = $packetArgs.Add([string]$Top)
    $null = $packetArgs.Add("--json")
    Add-FilterArgs -Arguments $packetArgs

    $packetJson = & $PythonExe @packetArgs
    if ($LASTEXITCODE -ne 0) {
        throw "phase1a_retrieval.py packet failed with exit code $LASTEXITCODE"
    }
    Set-Content -LiteralPath $PacketPath -Value $packetJson -Encoding utf8

    $packet = $packetJson | ConvertFrom-Json
    $topCitation = $null
    if ($packet.evidence -and $packet.evidence.Count -gt 0) {
        $topCitation = $packet.evidence[0].citation
    }

    if ($topCitation) {
        Write-Host "Top citation: $($topCitation.path):$($topCitation.line_start)-$($topCitation.line_end)"
        Write-Host "Title: $($topCitation.title)"
        Write-Host "Type: $($topCitation.source_type)"
    }
    else {
        Write-Host "Top citation: none"
    }
    Write-Host ""

    if ($ShowPrompt) {
        Write-Host "[2/3] Prompt preview..."
        Invoke-PythonStep -Arguments @($phase1b, "prompt", "--packet", $PacketPath)
        Write-Host ""
    }

    if ($RetrievalOnly) {
        $classification = Get-TestClassification -Packet $packet -ProbeReport $null -RetrievalOnlyMode
        Write-Host "Classification: $($classification.classification)"
        Write-Host "Reason: $($classification.reason)"
        if (-not [string]::IsNullOrWhiteSpace([string]$classification.suspicion)) {
            Write-Host "Suspicion: $($classification.suspicion)"
        }
        if ($classification.classification -ne "retrieval_pass") {
            $scriptExitCode = 2
        }
        Write-Host "Retrieval-only mode complete."
        exit $scriptExitCode
    }

    Write-Host "[3/3] Probing live runtime..."
    $probeResult = Invoke-PythonJson -Arguments @(
        $phase1b,
        "probe",
        "--packet", $PacketPath,
        "--endpoint", $Endpoint,
        "--timeout-seconds", $TimeoutSeconds,
        "--json"
    )
    $probeJson = [string]$probeResult.output
    try {
        $probeReport = $probeJson | ConvertFrom-Json
    }
    catch {
        Write-Host "Classification: runtime_fail"
        Write-Host "Reason: Probe returned non-JSON output."
        Write-Host ""
        if ($probeJson) {
            $probeJson
        }
        exit 2
    }
    $classification = Get-TestClassification -Packet $packet -ProbeReport $probeReport
    Write-Host "Classification: $($classification.classification)"
    Write-Host "Reason: $($classification.reason)"
    if (-not [string]::IsNullOrWhiteSpace([string]$classification.suspicion)) {
        Write-Host "Suspicion: $($classification.suspicion)"
    }
    Write-Host ""
    $probeJson
    if ($probeResult.exit_code -ne 0 -or $classification.classification -ne "retrieval_pass") {
        $scriptExitCode = 2
    }
    exit $scriptExitCode
}
finally {
    Pop-Location
}
