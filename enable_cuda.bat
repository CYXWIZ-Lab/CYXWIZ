@echo off
REM CyxWiz CUDA Enablement Script
REM This script reconfigures and rebuilds the project with CUDA support

echo ========================================
echo CyxWiz CUDA Enablement Script
echo ========================================
echo.

REM Check if CUDA is installed
echo [1/5] Checking CUDA installation...
where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CUDA Toolkit not found!
    echo.
    echo Please install CUDA Toolkit from:
    echo https://developer.nvidia.com/cuda-downloads
    echo.
    echo After installation, open a NEW terminal and run this script again.
    pause
    exit /b 1
)

REM Show CUDA version
echo CUDA Toolkit detected:
nvcc --version | findstr "release"
echo.

REM Check CUDA_PATH environment variable
if not defined CUDA_PATH (
    echo WARNING: CUDA_PATH environment variable not set!
    echo Attempting to find CUDA installation...

    REM Try to find CUDA installation
    for /f "delims=" %%i in ('dir /b /ad "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*" 2^>nul') do (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%i"
    )

    if defined CUDA_PATH (
        echo Found CUDA at: %CUDA_PATH%
    ) else (
        echo ERROR: Could not locate CUDA installation directory
        pause
        exit /b 1
    )
) else (
    echo CUDA_PATH: %CUDA_PATH%
)
echo.

REM Clean previous build
echo [2/5] Cleaning previous build...
if exist build\windows-release (
    echo Removing build\windows-release...
    rmdir /s /q build\windows-release
    echo Build directory cleaned.
) else (
    echo No previous build found.
)
echo.

REM Configure with CUDA enabled
echo [3/5] Configuring CMake with CUDA enabled...
cmake -B build/windows-release -S . ^
  -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DCYXWIZ_ENABLE_CUDA=ON ^
  -DCMAKE_BUILD_TYPE=Release

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed!
    echo.
    echo Trying with explicit CUDA path...
    cmake -B build/windows-release -S . ^
      -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ^
      -DCYXWIZ_ENABLE_CUDA=ON ^
      -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
      -DCMAKE_BUILD_TYPE=Release

    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: CMake configuration failed even with explicit CUDA path!
        echo Please check CUDA installation and try again.
        pause
        exit /b 1
    )
)
echo.

REM Build the project
echo [4/5] Building project with CUDA support...
echo This may take a few minutes...
cmake --build build/windows-release --config Release -j 8

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    echo Check the error messages above.
    pause
    exit /b 1
)
echo.

REM Verify CUDA linking
echo [5/5] Verifying CUDA integration...
if exist build\windows-release\bin\Release\cyxwiz-backend.dll (
    echo Checking DLL dependencies...
    dumpbin /DEPENDENTS build\windows-release\bin\Release\cyxwiz-backend.dll | findstr /i cudart
    if %ERRORLEVEL% EQU 0 (
        echo SUCCESS: CUDA runtime linked successfully!
    ) else (
        echo WARNING: Could not verify CUDA runtime linkage.
        echo The build may still work - test by running cyxwiz-server-node.exe
    )
) else (
    echo WARNING: Could not find cyxwiz-backend.dll
    echo Check if build completed successfully.
)
echo.

echo ========================================
echo CUDA Enablement Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run the Central Server:
echo    cd cyxwiz-central-server
echo    cargo run --release
echo.
echo 2. In another terminal, run the Server Node:
echo    cd build\windows-release\bin\Release
echo    cyxwiz-server-node.exe
echo.
echo 3. Look for "CUDA device X: Y.Y GB total, Z.Z GB free" in the logs
echo.
pause
