@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM CyxWiz Setup Script for Windows
REM ============================================================================
REM This script checks for required dependencies and sets up the build environment
REM for the CyxWiz project on Windows.
REM
REM Requirements checked:
REM   - Visual Studio 2026 (with C++ tools)
REM   - CMake 3.20+
REM   - Python 3.8+ (optional)
REM   - Rust/Cargo 1.70+
REM   - vcpkg (will be cloned and bootstrapped if missing)
REM ============================================================================

echo.
echo ============================================================================
echo CyxWiz Setup Script for Windows
echo ============================================================================
echo.

set ERROR_COUNT=0
set WARNING_COUNT=0

REM ============================================================================
REM Check for Visual Studio 2026
REM ============================================================================
echo [1/5] Checking for Visual Studio 2026...
where cl.exe >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Visual Studio 2026 not found in PATH
    echo.
    echo Please install Visual Studio 2026 with C++ Desktop Development workload:
    echo https://visualstudio.microsoft.com/downloads/
    echo.
    echo After installation, run this script from "Developer Command Prompt for VS 18 2026"
    set /a ERROR_COUNT+=1
) else (
    cl.exe 2>&1 | findstr /C:"Version" >nul
    echo [OK] Visual Studio 2026 found
)
echo.

REM ============================================================================
REM Check for CMake 3.20+
REM ============================================================================
echo [2/5] Checking for CMake 3.20+...
where cmake >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake not found in PATH
    echo.
    echo Please install CMake 3.20 or later:
    echo https://cmake.org/download/
    echo.
    echo Add CMake to your PATH during installation
    set /a ERROR_COUNT+=1
) else (
    for /f "tokens=3" %%i in ('cmake --version ^| findstr /C:"version"') do set CMAKE_VERSION=%%i
    echo [OK] CMake found: version !CMAKE_VERSION!

    REM Check version (basic check for 3.20+)
    for /f "tokens=1,2 delims=." %%a in ("!CMAKE_VERSION!") do (
        if %%a LSS 3 (
            echo [ERROR] CMake version too old. Need 3.20+, found !CMAKE_VERSION!
            set /a ERROR_COUNT+=1
        ) else if %%a EQU 3 (
            if %%b LSS 20 (
                echo [ERROR] CMake version too old. Need 3.20+, found !CMAKE_VERSION!
                set /a ERROR_COUNT+=1
            )
        )
    )
)
echo.

REM ============================================================================
REM Check for Python 3.8+ (optional)
REM ============================================================================
echo [3/5] Checking for Python 3.8+ ^(optional^)...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Python not found in PATH
    echo.
    echo Python is optional but recommended for scripting support.
    echo Download from: https://www.python.org/downloads/
    set /a WARNING_COUNT+=1
) else (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    echo [OK] Python found: version !PYTHON_VERSION!

    REM Basic version check for 3.8+
    for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
        if %%a LSS 3 (
            echo [WARNING] Python version too old. Need 3.8+, found !PYTHON_VERSION!
            set /a WARNING_COUNT+=1
        ) else if %%a EQU 3 (
            if %%b LSS 8 (
                echo [WARNING] Python version too old. Need 3.8+, found !PYTHON_VERSION!
                set /a WARNING_COUNT+=1
            )
        )
    )
)
echo.

REM ============================================================================
REM Check for Rust/Cargo 1.70+
REM ============================================================================
echo [4/5] Checking for Rust/Cargo 1.70+...
where cargo >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Rust/Cargo not found in PATH
    echo.
    echo Please install Rust from:
    echo https://rustup.rs/
    echo.
    echo After installation, restart your terminal and run this script again
    set /a ERROR_COUNT+=1
) else (
    for /f "tokens=2" %%i in ('cargo --version 2^>^&1') do set CARGO_VERSION=%%i
    echo [OK] Cargo found: version !CARGO_VERSION!

    REM Basic version check for 1.70+
    for /f "tokens=1,2 delims=." %%a in ("!CARGO_VERSION!") do (
        if %%a LSS 1 (
            echo [ERROR] Cargo version too old. Need 1.70+, found !CARGO_VERSION!
            set /a ERROR_COUNT+=1
        ) else if %%a EQU 1 (
            if %%b LSS 70 (
                echo [ERROR] Cargo version too old. Need 1.70+, found !CARGO_VERSION!
                set /a ERROR_COUNT+=1
            )
        )
    )
)
echo.

REM ============================================================================
REM Setup vcpkg
REM ============================================================================
echo [5/5] Setting up vcpkg...

if exist "vcpkg\.git" (
    echo [OK] vcpkg repository already exists
) else (
    echo [INFO] Cloning vcpkg repository...
    git clone https://github.com/microsoft/vcpkg.git
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to clone vcpkg repository
        echo.
        echo Make sure git is installed and accessible
        set /a ERROR_COUNT+=1
        goto :summary
    )
    echo [OK] vcpkg cloned successfully
)

if exist "vcpkg\vcpkg.exe" (
    echo [OK] vcpkg already bootstrapped
) else (
    echo [INFO] Bootstrapping vcpkg...
    call vcpkg\bootstrap-vcpkg.bat
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to bootstrap vcpkg
        set /a ERROR_COUNT+=1
        goto :summary
    )
    echo [OK] vcpkg bootstrapped successfully
)

echo.
echo [INFO] Installing vcpkg dependencies...
echo This may take several minutes on first run...
vcpkg\vcpkg install
if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some vcpkg packages failed to install
    echo You may need to install them manually later
    set /a WARNING_COUNT+=1
) else (
    echo [OK] vcpkg dependencies installed
)
echo.

REM ============================================================================
REM Summary
REM ============================================================================
:summary
echo ============================================================================
echo Setup Summary
echo ============================================================================
echo.

if %ERROR_COUNT% GTR 0 (
    echo [FAILED] Setup completed with %ERROR_COUNT% error^(s^) and %WARNING_COUNT% warning^(s^)
    echo.
    echo Please fix the errors above and run this script again.
    echo.
    exit /b 1
) else if %WARNING_COUNT% GTR 0 (
    echo [WARNING] Setup completed with %WARNING_COUNT% warning^(s^)
    echo.
    echo The warnings above are for optional components.
    echo You can proceed with the build, but some features may be unavailable.
    echo.
) else (
    echo [SUCCESS] All dependencies are installed!
    echo.
)

echo Next Steps:
echo   1. Build the project:
echo      build.bat
echo.
echo   2. Build specific components:
echo      build.bat --engine           ^(Build only Engine^)
echo      build.bat --server-node      ^(Build only Server Node^)
echo      build.bat --central-server   ^(Build only Central Server^)
echo.
echo   3. Build in Debug mode:
echo      build.bat --debug
echo.
echo   4. For more options:
echo      build.bat --help
echo.
echo ============================================================================

if %ERROR_COUNT% GTR 0 (
    exit /b 1
)

endlocal
