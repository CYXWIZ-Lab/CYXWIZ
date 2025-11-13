@echo off
REM ============================================================================
REM CyxWiz Build Script for Windows
REM ============================================================================
REM This script builds the CyxWiz project components with Visual Studio 2022.
REM
REM Usage: build.bat [options]
REM
REM Options:
REM   --help, -h           Show help message
REM   --debug              Build in Debug mode (default: Release)
REM   --clean              Clean build directory before building
REM   --engine             Build only Engine component
REM   --server-node        Build only Server Node component
REM   --central-server     Build only Central Server component
REM   -j N                 Use N parallel jobs (default: 8)
REM ============================================================================

setlocal enabledelayedexpansion

REM Parse command line arguments
set BUILD_TYPE=Release
set BUILD_TARGET=all
set CLEAN_BUILD=0
set PARALLEL_JOBS=8
set BUILD_ENGINE=ON
set BUILD_SERVER_NODE=ON
set BUILD_CENTRAL_SERVER=ON

:parse_args
if "%~1"=="" goto end_parse
if /i "%~1"=="--help" goto show_help
if /i "%~1"=="-h" goto show_help
if /i "%~1"=="--debug" (
    set BUILD_TYPE=Debug
    shift
    goto parse_args
)
if /i "%~1"=="--clean" (
    set CLEAN_BUILD=1
    shift
    goto parse_args
)
if /i "%~1"=="--engine" (
    set BUILD_TARGET=engine
    set BUILD_ENGINE=ON
    set BUILD_SERVER_NODE=OFF
    set BUILD_CENTRAL_SERVER=OFF
    shift
    goto parse_args
)
if /i "%~1"=="--server-node" (
    set BUILD_TARGET=server-node
    set BUILD_ENGINE=OFF
    set BUILD_SERVER_NODE=ON
    set BUILD_CENTRAL_SERVER=OFF
    shift
    goto parse_args
)
if /i "%~1"=="--central-server" (
    set BUILD_TARGET=central-server
    set BUILD_ENGINE=OFF
    set BUILD_SERVER_NODE=OFF
    set BUILD_CENTRAL_SERVER=ON
    shift
    goto parse_args
)
if /i "%~1"=="-j" (
    set PARALLEL_JOBS=%~2
    shift
    shift
    goto parse_args
)
echo [ERROR] Unknown option: %~1
echo Run 'build.bat --help' for usage
exit /b 1

:show_help
echo.
echo ============================================================================
echo CyxWiz Build Script
echo ============================================================================
echo.
echo Usage: build.bat [options]
echo.
echo Options:
echo   --help, -h           Show this help message
echo   --debug              Build in Debug mode (default: Release)
echo   --clean              Clean build directory before building
echo   --engine             Build only Engine component
echo   --server-node        Build only Server Node component
echo   --central-server     Build only Central Server component
echo   -j N                 Use N parallel jobs (default: 8)
echo.
echo Examples:
echo   build.bat                    Build all components in Release mode
echo   build.bat --debug            Build all in Debug mode
echo   build.bat --server-node      Build only Server Node
echo   build.bat --clean            Clean build and rebuild all
echo   build.bat -j 16              Build with 16 parallel jobs
echo.
echo ============================================================================
exit /b 0

:end_parse

REM Record start time
set START_TIME=%TIME%

echo.
echo ============================================================================
echo CyxWiz Build Script for Windows
echo ============================================================================
echo.

echo Configuration:
echo   Build Type:      %BUILD_TYPE%
echo   Components:      %BUILD_TARGET%
echo   Parallel Jobs:   %PARALLEL_JOBS%
echo   Clean Build:     %CLEAN_BUILD%
echo.
echo ============================================================================
echo.

REM Check if setup was run
if not exist "vcpkg\vcpkg.exe" (
    echo [ERROR] vcpkg not found!
    echo.
    echo Please run setup.bat first to install dependencies.
    echo.
    exit /b 1
)

set BUILD_DIR=build\windows-release

REM Clean build if requested
if %CLEAN_BUILD%==1 (
    echo [CLEAN] Cleaning build directory...
    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
    )
    echo [OK] Build directory cleaned
    echo.
)

REM ============================================================================
REM Step 1: Configure CMake
REM ============================================================================
echo [1/4] Configuring CMake...
set CMAKE_START=%TIME%
echo.

cmake -B %BUILD_DIR% -S . ^
    -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ^
    -DCYXWIZ_BUILD_ENGINE=%BUILD_ENGINE% ^
    -DCYXWIZ_BUILD_SERVER_NODE=%BUILD_SERVER_NODE% ^
    -DCYXWIZ_BUILD_CENTRAL_SERVER=%BUILD_CENTRAL_SERVER% ^
    -DCYXWIZ_BUILD_TESTS=ON

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo.
    echo Common fixes:
    echo   1. Run setup.bat to ensure vcpkg is installed
    echo   2. Check that Visual Studio 2022 is installed
    echo   3. Try: build.bat --clean
    echo.
    exit /b 1
)

set CMAKE_END=%TIME%
call :calculate_time "%CMAKE_START%" "%CMAKE_END%" CMAKE_DURATION
echo.
echo [OK] CMake configured successfully ^(%CMAKE_DURATION%^)
echo.

REM ============================================================================
REM Step 2: Build C++ components
REM ============================================================================
if /i "%BUILD_TARGET%"=="central-server" goto build_central_server

echo [2/4] Building C++ components...
set CPP_START=%TIME%
echo.

if /i "%BUILD_TARGET%"=="all" (
    cmake --build %BUILD_DIR% --config %BUILD_TYPE% -j %PARALLEL_JOBS%
) else (
    cmake --build %BUILD_DIR% --config %BUILD_TYPE% --target cyxwiz-%BUILD_TARGET% -j %PARALLEL_JOBS%
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] C++ build failed!
    echo.
    exit /b 1
)

set CPP_END=%TIME%
call :calculate_time "%CPP_START%" "%CPP_END%" CPP_DURATION
echo.
echo [OK] C++ build completed ^(%CPP_DURATION%^)
echo.

REM ============================================================================
REM Step 3: Build Central Server (Rust)
REM ============================================================================
:build_central_server
if /i "%BUILD_TARGET%"=="engine" goto build_summary
if /i "%BUILD_TARGET%"=="server-node" goto build_summary

echo [3/4] Building Central Server ^(Rust^)...
set RUST_START=%TIME%
echo.

cd cyxwiz-central-server

if "%BUILD_TYPE%"=="Debug" (
    cargo build
) else (
    cargo build --release
)

if %ERRORLEVEL% NEQ 0 (
    cd ..
    echo.
    echo [ERROR] Rust build failed!
    echo.
    exit /b 1
)

cd ..
set RUST_END=%TIME%
call :calculate_time "%RUST_START%" "%RUST_END%" RUST_DURATION
echo.
echo [OK] Rust build completed ^(%RUST_DURATION%^)
echo.

REM ============================================================================
REM Step 4: Build Summary
REM ============================================================================
:build_summary
set END_TIME=%TIME%
call :calculate_time "%START_TIME%" "%END_TIME%" TOTAL_DURATION

echo ============================================================================
echo [4/4] Build Summary
echo ============================================================================
echo.
echo Total Time: %TOTAL_DURATION%
echo.

if /i "%BUILD_TARGET%"=="all" (
    echo Executables:
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe" (
        echo   Engine:         %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
    )
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-node.exe" (
        echo   Server Node:    %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-node.exe
    )
    if exist "cyxwiz-central-server\target\release\cyxwiz-central-server.exe" (
        echo   Central Server: cyxwiz-central-server\target\release\cyxwiz-central-server.exe
    ) else if exist "cyxwiz-central-server\target\debug\cyxwiz-central-server.exe" (
        echo   Central Server: cyxwiz-central-server\target\debug\cyxwiz-central-server.exe
    )
) else if /i "%BUILD_TARGET%"=="engine" (
    echo Executable:
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe" (
        echo   Engine:         %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
    )
) else if /i "%BUILD_TARGET%"=="server-node" (
    echo Executable:
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-node.exe" (
        echo   Server Node:    %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-node.exe
    )
) else if /i "%BUILD_TARGET%"=="central-server" (
    echo Executable:
    if exist "cyxwiz-central-server\target\release\cyxwiz-central-server.exe" (
        echo   Central Server: cyxwiz-central-server\target\release\cyxwiz-central-server.exe
    ) else if exist "cyxwiz-central-server\target\debug\cyxwiz-central-server.exe" (
        echo   Central Server: cyxwiz-central-server\target\debug\cyxwiz-central-server.exe
    )
)

echo.
echo Next Steps:
if /i "%BUILD_TARGET%"=="all" (
    echo   - Run the Engine:         .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
    echo   - Run the Server Node:    .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-node.exe
    echo   - Run the Central Server: cd cyxwiz-central-server ^&^& cargo run --release
) else if /i "%BUILD_TARGET%"=="engine" (
    echo   - Run the Engine:         .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
) else if /i "%BUILD_TARGET%"=="server-node" (
    echo   - Run the Server Node:    .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-node.exe
) else if /i "%BUILD_TARGET%"=="central-server" (
    echo   - Run the Central Server: cd cyxwiz-central-server ^&^& cargo run --release
)

echo.
echo ============================================================================

goto :eof

REM ============================================================================
REM Helper function to calculate time difference
REM ============================================================================
:calculate_time
setlocal
set start=%~1
set end=%~2

REM Remove colons and decimals to get pure numbers
set start=%start::=%
set start=%start:.=%
set start=%start: =0%

set end=%end::=%
set end=%end:.=%
set end=%end: =0%

REM Calculate difference in centiseconds
set /a diff=%end%-%start%

if %diff% LSS 0 set /a diff=%diff%+24000000

REM Convert to seconds and minutes
set /a seconds=%diff%/100
set /a minutes=%seconds%/60
set /a seconds=%seconds%%%60

if %minutes% GTR 0 (
    endlocal & set "%~3=%minutes% min %seconds% sec"
) else (
    endlocal & set "%~3=%seconds% sec"
)
goto :eof
