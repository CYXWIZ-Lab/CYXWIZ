#!/bin/bash
# Build script for Linux/macOS

set -e

echo "Building CyxWiz..."

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PRESET="linux-release"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PRESET="macos-release"
else
    echo "Unsupported OS: $OSTYPE"
    exit 1
fi

# Check if vcpkg is installed
if [ ! -d "vcpkg" ]; then
    echo "Error: vcpkg not found. Please install vcpkg first."
    echo "Clone from: https://github.com/microsoft/vcpkg"
    exit 1
fi

# Install dependencies via vcpkg
echo "Installing dependencies..."
./vcpkg/vcpkg install

# Configure CMake
echo "Configuring CMake..."
cmake --preset $PRESET

# Build
echo "Building..."
cmake --build build/$PRESET --config Release

echo "Build complete!"
echo "Executables are in: build/$PRESET/bin/"
