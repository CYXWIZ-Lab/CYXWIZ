@echo off
REM Build script for Windows

echo Building CyxWiz...

REM Check if vcpkg is installed
if not exist "vcpkg\" (
    echo Error: vcpkg not found. Please install vcpkg first.
    echo Clone from: https://github.com/microsoft/vcpkg
    exit /b 1
)

REM Install dependencies via vcpkg
echo Installing dependencies...
vcpkg\vcpkg.exe install

REM Configure CMake
echo Configuring CMake...
cmake --preset windows-release

REM Build
echo Building...
cmake --build build\windows-release --config Release

echo Build complete!
echo Executables are in: build\windows-release\bin\
pause
