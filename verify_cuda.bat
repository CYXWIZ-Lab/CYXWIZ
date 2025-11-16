@echo off
REM CyxWiz CUDA Verification Script
REM This script checks if CUDA is properly installed and integrated

echo ========================================
echo CyxWiz CUDA Verification Script
echo ========================================
echo.

set CHECKS_PASSED=0
set CHECKS_TOTAL=7

REM Check 1: nvcc command
echo [1/7] Checking nvcc compiler...
where nvcc >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    nvcc --version | findstr "release"
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else (
    echo ✗ FAILED: nvcc not found in PATH
)
echo.

REM Check 2: CUDA_PATH environment variable
echo [2/7] Checking CUDA_PATH environment variable...
if defined CUDA_PATH (
    echo CUDA_PATH = %CUDA_PATH%
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else (
    echo ✗ FAILED: CUDA_PATH not set
)
echo.

REM Check 3: CUDA headers
echo [3/7] Checking CUDA headers...
if exist "%CUDA_PATH%\include\cuda_runtime.h" (
    echo Found: %CUDA_PATH%\include\cuda_runtime.h
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\include\cuda_runtime.h" (
    echo Found CUDA headers in standard location
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else (
    echo ✗ FAILED: cuda_runtime.h not found
)
echo.

REM Check 4: CUDA libraries
echo [4/7] Checking CUDA libraries...
if exist "%CUDA_PATH%\lib\x64\cudart.lib" (
    echo Found: %CUDA_PATH%\lib\x64\cudart.lib
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\lib\x64\cudart.lib" (
    echo Found CUDA libraries in standard location
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else (
    echo ✗ FAILED: cudart.lib not found
)
echo.

REM Check 5: CMake configuration
echo [5/7] Checking CMake configuration...
if exist build\windows-release\CMakeCache.txt (
    findstr /C:"CYXWIZ_ENABLE_CUDA:BOOL=ON" build\windows-release\CMakeCache.txt >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo CUDA enabled in CMake configuration
        set /a CHECKS_PASSED+=1
        echo ✓ PASSED
    ) else (
        echo ✗ FAILED: CUDA not enabled in CMake (run enable_cuda.bat)
    )
) else (
    echo ✗ FAILED: Project not configured (run enable_cuda.bat)
)
echo.

REM Check 6: Build output
echo [6/7] Checking build output...
if exist build\windows-release\bin\Release\cyxwiz-backend.dll (
    echo Found: build\windows-release\bin\Release\cyxwiz-backend.dll
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else if exist build\windows-release\bin\cyxwiz-backend.dll (
    echo Found: build\windows-release\bin\cyxwiz-backend.dll
    set /a CHECKS_PASSED+=1
    echo ✓ PASSED
) else (
    echo ✗ FAILED: cyxwiz-backend.dll not built (run enable_cuda.bat)
)
echo.

REM Check 7: CUDA runtime linkage
echo [7/7] Checking CUDA runtime linkage...
where dumpbin >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    if exist build\windows-release\bin\Release\cyxwiz-backend.dll (
        dumpbin /DEPENDENTS build\windows-release\bin\Release\cyxwiz-backend.dll 2>nul | findstr /i cudart >nul
        if %ERRORLEVEL% EQU 0 (
            echo CUDA runtime (cudart64_XX.dll) linked successfully
            set /a CHECKS_PASSED+=1
            echo ✓ PASSED
        ) else (
            echo ✗ FAILED: CUDA runtime not linked to cyxwiz-backend.dll
        )
    ) else if exist build\windows-release\bin\cyxwiz-backend.dll (
        dumpbin /DEPENDENTS build\windows-release\bin\cyxwiz-backend.dll 2>nul | findstr /i cudart >nul
        if %ERRORLEVEL% EQU 0 (
            echo CUDA runtime (cudart64_XX.dll) linked successfully
            set /a CHECKS_PASSED+=1
            echo ✓ PASSED
        ) else (
            echo ✗ FAILED: CUDA runtime not linked to cyxwiz-backend.dll
        )
    ) else (
        echo ✗ FAILED: cyxwiz-backend.dll not found
    )
) else (
    echo ⚠ WARNING: dumpbin not available (Visual Studio tools not in PATH)
    echo Cannot verify CUDA runtime linkage
    echo Run this from "x64 Native Tools Command Prompt for VS 2022"
)
echo.

REM Summary
echo ========================================
echo Verification Summary
echo ========================================
echo Checks passed: %CHECKS_PASSED% / %CHECKS_TOTAL%
echo.

if %CHECKS_PASSED% EQU %CHECKS_TOTAL% (
    echo ✓ ALL CHECKS PASSED!
    echo.
    echo CUDA Toolkit is properly installed and integrated.
    echo.
    echo Next steps:
    echo 1. Start the Central Server:
    echo    cd cyxwiz-central-server
    echo    cargo run --release
    echo.
    echo 2. Start the Server Node in another terminal:
    echo    cd build\windows-release\bin\Release
    echo    cyxwiz-server-node.exe
    echo.
    echo 3. Check the logs for:
    echo    "CUDA device 0: X.X GB total, Y.Y GB free"
) else if %CHECKS_PASSED% GEQ 5 (
    echo ⚠ PARTIAL SUCCESS
    echo.
    echo Most checks passed, but some issues detected.
    echo Review the failed checks above.
    echo.
    echo If CMake/build checks failed, run:
    echo    enable_cuda.bat
) else (
    echo ✗ VERIFICATION FAILED
    echo.
    echo CUDA Toolkit may not be properly installed.
    echo.
    echo Please:
    echo 1. Install CUDA Toolkit from:
    echo    https://developer.nvidia.com/cuda-downloads
    echo.
    echo 2. Open a NEW terminal after installation
    echo.
    echo 3. Run this verification script again
    echo.
    echo 4. If CUDA is installed but checks fail, run:
    echo    enable_cuda.bat
)
echo.
pause
