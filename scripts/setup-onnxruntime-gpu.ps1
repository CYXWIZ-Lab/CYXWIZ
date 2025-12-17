# setup-onnxruntime-gpu.ps1
# Downloads and sets up ONNX Runtime GPU for Windows
# Run this script from the project root directory

param(
    [string]$Version = "1.21.0"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$ProjectRoot/CMakeLists.txt")) {
    $ProjectRoot = (Get-Location).Path
}

$TargetDir = Join-Path $ProjectRoot "external/onnxruntime-gpu"
$TempFile = Join-Path $env:TEMP "onnxruntime-gpu.zip"
$NuGetUrl = "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu.Windows/$Version"

Write-Host "Setting up ONNX Runtime GPU v$Version..." -ForegroundColor Cyan
Write-Host "Target directory: $TargetDir"

# Create target directory
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

# Download NuGet package
Write-Host "Downloading from NuGet..." -ForegroundColor Yellow
Invoke-WebRequest -Uri $NuGetUrl -OutFile $TempFile

# Extract
Write-Host "Extracting..." -ForegroundColor Yellow
Expand-Archive -Path $TempFile -DestinationPath $TargetDir -Force

# Organize files
Write-Host "Organizing files..." -ForegroundColor Yellow

# Create organized directories
$BinDir = Join-Path $TargetDir "bin"
$LibDir = Join-Path $TargetDir "lib"
$IncludeDir = Join-Path $TargetDir "include"

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
New-Item -ItemType Directory -Path $LibDir -Force | Out-Null
New-Item -ItemType Directory -Path $IncludeDir -Force | Out-Null

# Copy DLLs
$NativeDir = Join-Path $TargetDir "runtimes/win-x64/native"
if (Test-Path $NativeDir) {
    Copy-Item "$NativeDir/*.dll" $BinDir -Force
    Copy-Item "$NativeDir/*.lib" $LibDir -Force
}

# Copy headers
$HeaderDir = Join-Path $TargetDir "buildTransitive/native/include"
if (Test-Path $HeaderDir) {
    Copy-Item "$HeaderDir/*.h" $IncludeDir -Force
}

# Cleanup temp file
Remove-Item $TempFile -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "ONNX Runtime GPU setup complete!" -ForegroundColor Green
Write-Host "Files installed to: $TargetDir" -ForegroundColor Green
Write-Host ""
Write-Host "Contents:" -ForegroundColor Cyan
Write-Host "  bin/  - DLLs (onnxruntime.dll, onnxruntime_providers_cuda.dll, etc.)"
Write-Host "  lib/  - Import libraries (.lib files)"
Write-Host "  include/ - Header files"
Write-Host ""
Write-Host "Now run CMake to configure the project." -ForegroundColor Yellow
