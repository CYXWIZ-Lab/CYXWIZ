param(
    [string]$Executable = "build/bin/Release/test_dense_training_benchmark.exe",
    [string]$Qualification = "build/bin/Release/arrayfire-route-qualification.json",
    [string]$OutputRoot = "build/ticket89-performance-final",
    [int]$WarmupBatches = 10,
    [int]$MeasuredBatches = 100
)

$ErrorActionPreference = "Stop"

$executablePath = (Resolve-Path -LiteralPath $Executable).Path
$qualificationPath = (Resolve-Path -LiteralPath $Qualification).Path
$outputRootPath = [System.IO.Path]::GetFullPath($OutputRoot)
[System.IO.Directory]::CreateDirectory($outputRootPath) | Out-Null
$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue

$routes = @(
    @{ route_id = "cpu:0"; backend = "cpu"; device_id = 0 },
    @{ route_id = "cuda:0"; backend = "cuda"; device_id = 0 },
    @{ route_id = "opencl:0"; backend = "opencl"; device_id = 0 },
    @{ route_id = "opencl:1"; backend = "opencl"; device_id = 1 },
    @{ route_id = "opencl:2"; backend = "opencl"; device_id = 2 }
)

$results = @()
foreach ($route in $routes) {
    $routeDirectory = Join-Path $outputRootPath ($route.route_id -replace ':', '-')
    [System.IO.Directory]::CreateDirectory($routeDirectory) | Out-Null
    $stdoutPath = Join-Path $routeDirectory "benchmark.stdout.txt"
    $stderrPath = Join-Path $routeDirectory "benchmark.stderr.txt"

    $arguments = @(
        "--device", $route.backend,
        "--device-id", [string]$route.device_id,
        "--warmup-batches", [string]$WarmupBatches,
        "--measured-batches", [string]$MeasuredBatches,
        "--qualification", $qualificationPath,
        "--trace-root", $routeDirectory
    )
    $process = Start-Process -FilePath $executablePath `
        -ArgumentList $arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $gpuSampleCount = 0
    $peakGpuEnginePercent = 0.0
    $peakGpuEnginePath = ""
    $nvidiaSampleCount = 0
    $peakNvidiaUtilizationPercent = 0.0
    $peakNvidiaMemoryMiB = 0.0
    $nvidiaGpuUuid = ""
    while (-not $process.HasExited) {
        try {
            $counter = Get-Counter `
                -Counter '\GPU Engine(*)\Utilization Percentage' `
                -ErrorAction Stop
            $pidMarker = "pid_$($process.Id)_"
            $pidSamples = $counter.CounterSamples | Where-Object {
                $_.Path -like "*$pidMarker*"
            }
            foreach ($sample in $pidSamples) {
                ++$gpuSampleCount
                if ($sample.CookedValue -gt $peakGpuEnginePercent) {
                    $peakGpuEnginePercent = $sample.CookedValue
                    $peakGpuEnginePath = $sample.Path
                }
            }
        } catch {
            # The benchmark output still records route and CPU/memory evidence.
        }
        if ($null -ne $nvidiaSmi) {
            try {
                $nvidiaRows = & $nvidiaSmi.Source `
                    --query-gpu=uuid,utilization.gpu,memory.used `
                    --format=csv,noheader,nounits 2>$null
                foreach ($row in $nvidiaRows) {
                    $columns = $row -split ',' | ForEach-Object { $_.Trim() }
                    if ($columns.Count -eq 3) {
                        ++$nvidiaSampleCount
                        $nvidiaGpuUuid = $columns[0]
                        $peakNvidiaUtilizationPercent = [Math]::Max(
                            $peakNvidiaUtilizationPercent,
                            [double]$columns[1])
                        $peakNvidiaMemoryMiB = [Math]::Max(
                            $peakNvidiaMemoryMiB,
                            [double]$columns[2])
                    }
                }
            } catch {
                # NVIDIA telemetry is supplementary to route-owned trace data.
            }
        }
        Start-Sleep -Milliseconds 200
        $process.Refresh()
    }
    $process.WaitForExit()

    $stdout = Get-Content -LiteralPath $stdoutPath
    $stderr = Get-Content -LiteralPath $stderrPath -Raw
    $exitCode = $process.ExitCode
    if ($null -eq $exitCode -and
        ($stdout | Select-String -Pattern '^host_sync_summary=' -Quiet) -and
        [string]::IsNullOrWhiteSpace($stderr)) {
        # Windows PowerShell can lose ExitCode after redirected Start-Process.
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        throw "Benchmark $($route.route_id) failed with exit code $exitCode`: $stderr"
    }

    $result = [ordered]@{
        route_id = $route.route_id
        process_id = $process.Id
        exit_code = $exitCode
        gpu_utilization_source = "Windows GPU Engine performance counters; maximum single engine for benchmark PID"
        gpu_counter_sample_count = $gpuSampleCount
        peak_gpu_engine_percent = [Math]::Round($peakGpuEnginePercent, 3)
        peak_gpu_engine_path = $peakGpuEnginePath
        nvidia_utilization_source = "nvidia-smi adapter-level telemetry; benchmark routes run serially with Engine stopped"
        nvidia_sample_count = $nvidiaSampleCount
        nvidia_gpu_uuid = $nvidiaGpuUuid
        peak_nvidia_utilization_percent = $peakNvidiaUtilizationPercent
        peak_nvidia_memory_mib = $peakNvidiaMemoryMiB
    }
    foreach ($line in $stdout) {
        if ($line -match '^([a-z][a-z0-9_]*)=(.*)$') {
            $result[$matches[1]] = $matches[2]
        }
    }

    $resultPath = Join-Path $routeDirectory "result.json"
    $result | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $resultPath
    $results += [pscustomobject]$result
}

$report = [ordered]@{
    schema = 1
    captured_at = (Get-Date).ToUniversalTime().ToString("o")
    executable = $executablePath
    qualification = $qualificationPath
    warmup_batches = $WarmupBatches
    measured_batches = $MeasuredBatches
    routes = $results
}
$reportPath = Join-Path $outputRootPath "ticket89-matched-route-report.json"
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $reportPath
$report | ConvertTo-Json -Depth 6
