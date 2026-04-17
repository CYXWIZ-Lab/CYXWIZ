@echo off
REM scripts/rebuild.cmd - safe incremental rebuild wrapper (Windows).
REM
REM Why this exists (2026-04-17 lesson):
REM   "cmake --build ... | tail -3" silently hid a C2440 compile error
REM   and we ran a zombie binary for an hour. This wrapper avoids pipes,
REM   checks cmake's real errorlevel, and flags when the .exe mtime
REM   hasn't moved forward despite a nominally-successful build.
REM
REM Usage:
REM   scripts\rebuild.cmd                 (default target: cyxwiz-engine)
REM   scripts\rebuild.cmd <target>
REM
REM Env:
REM   PRESET=Release (default) | Debug
REM   BUILD_DIR=build (default)

setlocal enableextensions enabledelayedexpansion

if "%~1"=="" (set TARGET=cyxwiz-engine) else (set TARGET=%~1)
if "%PRESET%"=="" set PRESET=Release
if "%BUILD_DIR%"=="" set BUILD_DIR=build

set BINARY=%BUILD_DIR%\bin\%PRESET%\cyxwiz-engine.exe
set LOG=%BUILD_DIR%\last-build.log

echo ==^> Building %TARGET% (%PRESET%)

REM Record binary mtime before build, for the "did anything actually
REM rebuild" sanity check below. The for-loop modifier gives a string
REM like "04/17/2026 08:24 PM" (minute resolution — fine for this).
set MTIME_BEFORE=
if exist "%BINARY%" (
    for %%F in ("%BINARY%") do set MTIME_BEFORE=%%~tF
)

REM No pipe — cmake's errorlevel propagates directly.
cmake --build "%BUILD_DIR%" --config "%PRESET%" --target "%TARGET%" > "%LOG%" 2>&1
set RC=%ERRORLEVEL%

if not %RC%==0 (
    echo ==^> BUILD FAILED ^(cmake exit %RC%^)
    echo Last 40 lines of %LOG%:
    echo ---
    powershell -NoProfile -Command "Get-Content '%LOG%' -Tail 40"
    exit /b 1
)

REM Compare binary mtime.
set MTIME_AFTER=
if exist "%BINARY%" (
    for %%F in ("%BINARY%") do set MTIME_AFTER=%%~tF
)

if defined MTIME_BEFORE (
    if "%MTIME_BEFORE%"=="%MTIME_AFTER%" (
        echo ==^> WARNING: cmake OK but binary mtime unchanged.
        echo     Either nothing needed rebuilding, or MSBuild skipped a
        echo     failing TU and reused stale .obj. Review %LOG% to confirm.
    )
)

REM Warning summary. findstr counts matching lines; /C: matches literal.
for /f %%C in ('findstr /R /C:"warning C[0-9][0-9]*:" "%LOG%" 2^>nul ^| find /c /v ""') do set WARN_COUNT=%%C
if not defined WARN_COUNT set WARN_COUNT=0

echo ==^> Warnings: %WARN_COUNT%
if %WARN_COUNT% GTR 0 (
    REM Sort -Unique + head via powershell so we don't depend on cygwin.
    powershell -NoProfile -Command ^
        "Select-String -Path '%LOG%' -Pattern 'warning C[0-9]+:' | ForEach-Object { $_.Line } | Sort-Object -Unique | Select-Object -First 10"
    if %WARN_COUNT% GTR 10 echo     ... (%WARN_COUNT% total lines; see %LOG% for full list)
)

if exist "%BINARY%" (
    for %%F in ("%BINARY%") do echo ==^> Binary: %%~zF bytes  %%~tF
)

echo ==^> OK
endlocal
exit /b 0
