#!/bin/bash
# setup-onnxruntime-gpu.sh
# Downloads and sets up ONNX Runtime GPU
# Run this script from the project root directory

VERSION="${1:-1.21.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET_DIR="$PROJECT_ROOT/external/onnxruntime-gpu"

echo "Setting up ONNX Runtime GPU v$VERSION..."
echo "Target directory: $TARGET_DIR"

# Detect platform
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="win-x64"
    NUGET_PACKAGE="Microsoft.ML.OnnxRuntime.Gpu.Windows"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux-x64"
    NUGET_PACKAGE="Microsoft.ML.OnnxRuntime.Gpu.Linux"
else
    echo "Unsupported platform: $OSTYPE"
    echo "ONNX Runtime GPU is only available for Windows and Linux"
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

# Download
TEMP_FILE="/tmp/onnxruntime-gpu.zip"
NUGET_URL="https://www.nuget.org/api/v2/package/$NUGET_PACKAGE/$VERSION"

echo "Downloading from NuGet..."
curl -L -o "$TEMP_FILE" "$NUGET_URL"

# Extract
echo "Extracting..."
unzip -o "$TEMP_FILE" -d "$TARGET_DIR"

# Organize files
echo "Organizing files..."
mkdir -p "$TARGET_DIR/bin" "$TARGET_DIR/lib" "$TARGET_DIR/include"

# Copy files based on platform
if [[ "$PLATFORM" == "win-x64" ]]; then
    cp "$TARGET_DIR/runtimes/win-x64/native/"*.dll "$TARGET_DIR/bin/" 2>/dev/null || true
    cp "$TARGET_DIR/runtimes/win-x64/native/"*.lib "$TARGET_DIR/lib/" 2>/dev/null || true
elif [[ "$PLATFORM" == "linux-x64" ]]; then
    cp "$TARGET_DIR/runtimes/linux-x64/native/"*.so* "$TARGET_DIR/lib/" 2>/dev/null || true
fi

# Copy headers
cp "$TARGET_DIR/buildTransitive/native/include/"*.h "$TARGET_DIR/include/" 2>/dev/null || true

# Cleanup
rm -f "$TEMP_FILE"

echo ""
echo "ONNX Runtime GPU setup complete!"
echo "Files installed to: $TARGET_DIR"
echo ""
echo "Now run CMake to configure the project."
