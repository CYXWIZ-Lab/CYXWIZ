@echo off
REM ============================================================================
REM CyxWiz Build Script for Windows
REM ============================================================================
REM This script builds the CyxWiz project components with Visual Studio 18 2026.
REM
REM Usage: build.bat [options]
REM
REM Options:
REM   --help, -h           Show help message
REM   --debug              Build in Debug mode (default: Release)
REM   --clean              Clean build directory before building
REM   --engine             Build only Engine component
REM   --server-node        Build only Server Node component
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
    shift
    goto parse_args
)
if /i "%~1"=="--server-node" (
    set BUILD_TARGET=server-node
    set BUILD_ENGINE=OFF
    set BUILD_SERVER_NODE=ON
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

if /i "%BUILD_TYPE%"=="Debug" (
    set BUILD_DIR=build\windows-debug
) else (
    set BUILD_DIR=build\windows-release
)

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
echo [1/3] Configuring CMake...
set CMAKE_START=%TIME%
echo.

cmake -B %BUILD_DIR% -S . ^
    -G "Visual Studio 18 2026" -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake ^
    -DVCPKG_OVERLAY_PORTS=vcpkg-ports ^
    -DCYXWIZ_BUILD_ENGINE=%BUILD_ENGINE% ^
    -DCYXWIZ_BUILD_SERVER_NODE=%BUILD_SERVER_NODE% ^
    -DCYXWIZ_BUILD_TESTS=ON

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo.
    echo Common fixes:
    echo   1. Run setup.bat to ensure vcpkg is installed
    echo   2. Check that Visual Studio 2026 is installed
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
echo [2/3] Building C++ components...
set CPP_START=%TIME%
echo.

if /i "%BUILD_TARGET%"=="all" (
    cmake --build %BUILD_DIR% --config %BUILD_TYPE% -j %PARALLEL_JOBS%
) else if /i "%BUILD_TARGET%"=="server-node" (
    cmake --build %BUILD_DIR% --config %BUILD_TYPE% --target cyxwiz-server-daemon cyxwiz-server-gui -j %PARALLEL_JOBS%
) else (
    cmake --build %BUILD_DIR% --config %BUILD_TYPE% --target cyxwiz-engine -j %PARALLEL_JOBS%
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
REM Step 3: Build Summary
REM ============================================================================
:build_summary
set END_TIME=%TIME%
call :calculate_time "%START_TIME%" "%END_TIME%" TOTAL_DURATION

echo ============================================================================
echo [3/3] Build Summary
echo ============================================================================
echo.
echo Total Time: %TOTAL_DURATION%
echo.

if /i "%BUILD_TARGET%"=="all" (
    echo Executables:
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe" (
        echo   Engine:         %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
    )
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-daemon.exe" (
        echo   Server daemon:  %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-daemon.exe
    )
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-gui.exe" (
        echo   Server GUI:     %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-gui.exe
    )
) else if /i "%BUILD_TARGET%"=="engine" (
    echo Executable:
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe" (
        echo   Engine:         %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
    )
) else if /i "%BUILD_TARGET%"=="server-node" (
    echo Executables:
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-daemon.exe" (
        echo   Server daemon:  %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-daemon.exe
    )
    if exist "%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-gui.exe" (
        echo   Server GUI:     %BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-gui.exe
    )
)

echo.
echo Next Steps:
if /i "%BUILD_TARGET%"=="all" (
    echo   - Run the Engine:         .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
    echo   - Run the Server GUI:     .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-gui.exe
    echo   - Run the daemon:         .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-daemon.exe
) else if /i "%BUILD_TARGET%"=="engine" (
    echo   - Run the Engine:         .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-engine.exe
) else if /i "%BUILD_TARGET%"=="server-node" (
    echo   - Run the Server GUI:     .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-gui.exe
    echo   - Run the daemon:         .\%BUILD_DIR%\bin\%BUILD_TYPE%\cyxwiz-server-daemon.exe
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
