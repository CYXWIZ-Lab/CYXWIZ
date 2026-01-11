@echo off
REM CyxWiz Distributed Training Launcher for Windows
REM =================================================
REM
REM Usage:
REM   launch_distributed.bat <num_gpus> <python_script> [args...]
REM
REM Examples:
REM   launch_distributed.bat 4 distributed_training_example.py
REM   launch_distributed.bat 2 train.py --epochs 100
REM

setlocal EnableDelayedExpansion

if "%~2"=="" (
    echo Usage: %0 ^<num_gpus^> ^<python_script^> [args...]
    echo.
    echo Examples:
    echo   %0 4 distributed_training_example.py
    echo   %0 2 train.py --epochs 100
    exit /b 1
)

set NUM_GPUS=%1
shift
set SCRIPT=%1
shift
set ARGS=%*

REM Default values
if "%MASTER_ADDR%"=="" set MASTER_ADDR=127.0.0.1
if "%MASTER_PORT%"=="" set MASTER_PORT=29500
if "%DISTRIBUTED_BACKEND%"=="" set DISTRIBUTED_BACKEND=cpu

echo ==============================================
echo CyxWiz Distributed Training Launcher (Windows)
echo ==============================================
echo World size:  %NUM_GPUS%
echo Master:      %MASTER_ADDR%:%MASTER_PORT%
echo Backend:     %DISTRIBUTED_BACKEND%
echo Script:      %SCRIPT%
echo ==============================================
echo.

REM Launch processes
for /L %%R in (0,1,%NUM_GPUS%) do (
    if %%R LSS %NUM_GPUS% (
        echo Launching rank %%R...
        start "Rank %%R" cmd /c "set RANK=%%R && set LOCAL_RANK=%%R && set WORLD_SIZE=%NUM_GPUS% && set MASTER_ADDR=%MASTER_ADDR% && set MASTER_PORT=%MASTER_PORT% && set DISTRIBUTED_BACKEND=%DISTRIBUTED_BACKEND% && python %SCRIPT% %ARGS% && pause"
        timeout /t 1 /nobreak >nul
    )
)

echo.
echo All %NUM_GPUS% processes launched in separate windows.
echo Close this window or press Ctrl+C to stop monitoring.
echo.

pause
