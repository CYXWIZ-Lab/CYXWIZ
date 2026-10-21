#!/bin/bash
# ============================================================================
# CyxWiz Build Script for Linux/macOS
# ============================================================================
# This script builds the CyxWiz project components with Ninja or Make.
#
# Usage: ./build.sh [options]
#
# Options:
#   --help, -h           Show help message
#   --debug              Build in Debug mode (default: Release)
#   --clean              Clean build directory before building
#   --engine             Build only Engine component
#   --server-node        Build only Server Node component
#   --build-dir PATH     Use a compatible existing or custom build tree
#   -j N                 Use N parallel jobs (default: auto-detect)
# ============================================================================

set -e  # Exit on error

# Default values
BUILD_TYPE="Release"
BUILD_TARGET="all"
CLEAN_BUILD=0
PARALLEL_JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
BUILD_ENGINE=ON
BUILD_SERVER_NODE=ON
BUILD_DIR_OVERRIDE=""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
else
    echo "[ERROR] Unsupported OS: $OSTYPE"
    exit 1
fi

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            echo ""
            echo "============================================================================"
            echo "CyxWiz Build Script"
            echo "============================================================================"
            echo ""
            echo "Usage: ./build.sh [options]"
            echo ""
            echo "Options:"
            echo "  --help, -h           Show this help message"
            echo "  --debug              Build in Debug mode (default: Release)"
            echo "  --clean              Clean build directory before building"
            echo "  --engine             Build only Engine component"
            echo "  --server-node        Build only Server Node component"
            echo "  --build-dir PATH     Use a compatible existing or custom build tree"
            echo "  -j N                 Use N parallel jobs (default: auto-detect)"
            echo ""
            echo "Examples:"
            echo "  ./build.sh                    Build all components in Release mode"
            echo "  ./build.sh --debug            Build all in Debug mode"
            echo "  ./build.sh --server-node      Build only Server Node"
            echo "  ./build.sh --debug --engine --build-dir build"
            echo "  ./build.sh --clean            Clean build and rebuild all"
            echo "  ./build.sh -j 16              Build with 16 parallel jobs"
            echo ""
            echo "============================================================================"
            exit 0
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --engine)
            BUILD_TARGET="engine"
            BUILD_ENGINE=ON
            BUILD_SERVER_NODE=OFF
            shift
            ;;
        --server-node)
            BUILD_TARGET="server-node"
            BUILD_ENGINE=OFF
            BUILD_SERVER_NODE=ON
            shift
            ;;
        --build-dir)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] --build-dir requires a path"
                exit 1
            fi
            BUILD_DIR_OVERRIDE="$2"
            shift 2
            ;;
        -j)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            echo "Run with --help for usage"
            exit 1
            ;;
    esac
done

BUILD_TYPE_DIR=$(echo "$BUILD_TYPE" | tr '[:upper:]' '[:lower:]')
if [[ -n "$BUILD_DIR_OVERRIDE" ]]; then
    BUILD_DIR="$BUILD_DIR_OVERRIDE"
else
    BUILD_DIR="build/${OS}-${BUILD_TYPE_DIR}"
fi

# Record start time
START_TIME=$(date +%s)

echo ""
echo "============================================================================"
echo "CyxWiz Build Script for $OS"
echo "============================================================================"
echo ""

echo "Configuration:"
echo "  OS:              $OS"
echo "  Build Type:      $BUILD_TYPE"
echo "  Components:      $BUILD_TARGET"
echo "  Build Directory: $BUILD_DIR"
echo "  Parallel Jobs:   $PARALLEL_JOBS"
echo "  Clean Build:     $CLEAN_BUILD"
echo ""
echo "============================================================================"
echo ""

# Check if setup was run
if [ ! -f "vcpkg/vcpkg" ]; then
    echo "[ERROR] vcpkg not found!"
    echo ""
    echo "Please run ./setup.sh first to install dependencies."
    echo ""
    exit 1
fi

# Clean build if requested
if [ $CLEAN_BUILD -eq 1 ]; then
    if [[ -z "$BUILD_DIR" || "$BUILD_DIR" == "." || "$BUILD_DIR" == "/" ]]; then
        echo "[ERROR] Refusing to clean unsafe build directory: '$BUILD_DIR'"
        exit 1
    fi
    echo "[CLEAN] Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    echo "[OK] Build directory cleaned"
    echo ""
fi

# ============================================================================
# Step 1: Configure CMake
# ============================================================================
echo "[1/3] Configuring CMake..."
CMAKE_START=$(date +%s)
echo ""

# Detect generator
GENERATOR="Ninja"
if ! command -v ninja &> /dev/null; then
    GENERATOR="Unix Makefiles"
    echo "[INFO] Ninja not found, using Unix Makefiles (slower)"
    echo ""
fi

cmake -B "$BUILD_DIR" -S . \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DCYXWIZ_BUILD_ENGINE=$BUILD_ENGINE \
    -DCYXWIZ_BUILD_SERVER_NODE=$BUILD_SERVER_NODE \
    -DCYXWIZ_BUILD_TESTS=ON

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] CMake configuration failed!"
    echo ""
    echo "Common fixes:"
    echo "  1. Run ./setup.sh to ensure vcpkg is installed"
    echo "  2. Check that C++ compiler is installed"
    echo "  3. Try: ./build.sh --clean"
    echo ""
    exit 1
fi

CMAKE_END=$(date +%s)
CMAKE_DURATION=$((CMAKE_END - CMAKE_START))
echo ""
echo "[OK] CMake configured successfully (${CMAKE_DURATION}s)"
echo ""

# ============================================================================
# Step 2: Build C++ components
# ============================================================================
echo "[2/3] Building C++ components..."
CPP_START=$(date +%s)
echo ""

if [[ "$BUILD_TARGET" == "all" ]]; then
    cmake --build "$BUILD_DIR" -j $PARALLEL_JOBS
elif [[ "$BUILD_TARGET" == "server-node" ]]; then
    cmake --build "$BUILD_DIR" --target cyxwiz-server-daemon cyxwiz-server-gui -j $PARALLEL_JOBS
else
    cmake --build "$BUILD_DIR" --target cyxwiz-engine -j $PARALLEL_JOBS
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] C++ build failed!"
    echo ""
    exit 1
fi

CPP_END=$(date +%s)
CPP_DURATION=$((CPP_END - CPP_START))
echo ""
echo "[OK] C++ build completed (${CPP_DURATION}s)"
echo ""

# ============================================================================
# Step 3: Build Summary
# ============================================================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# Format total duration
if [ $TOTAL_DURATION -ge 60 ]; then
    MINUTES=$((TOTAL_DURATION / 60))
    SECONDS=$((TOTAL_DURATION % 60))
    TOTAL_DURATION_STR="${MINUTES} min ${SECONDS} sec"
else
    TOTAL_DURATION_STR="${TOTAL_DURATION} sec"
fi

echo "============================================================================"
echo "[3/3] Build Summary"
echo "============================================================================"
echo ""
echo "Total Time: $TOTAL_DURATION_STR"
echo ""

if [[ "$BUILD_TARGET" == "all" ]]; then
    echo "Executables:"
    [ -f "$BUILD_DIR/bin/cyxwiz-engine" ] && \
        echo "  Engine:         $BUILD_DIR/bin/cyxwiz-engine"
    [ -f "$BUILD_DIR/bin/cyxwiz-server-daemon" ] && \
        echo "  Server daemon:  $BUILD_DIR/bin/cyxwiz-server-daemon"
    [ -f "$BUILD_DIR/bin/cyxwiz-server-gui" ] && \
        echo "  Server GUI:     $BUILD_DIR/bin/cyxwiz-server-gui"
elif [[ "$BUILD_TARGET" == "engine" ]]; then
    echo "Executable:"
    [ -f "$BUILD_DIR/bin/cyxwiz-engine" ] && \
        echo "  Engine:         $BUILD_DIR/bin/cyxwiz-engine"
elif [[ "$BUILD_TARGET" == "server-node" ]]; then
    echo "Executables:"
    [ -f "$BUILD_DIR/bin/cyxwiz-server-daemon" ] && \
        echo "  Server daemon:  $BUILD_DIR/bin/cyxwiz-server-daemon"
    [ -f "$BUILD_DIR/bin/cyxwiz-server-gui" ] && \
        echo "  Server GUI:     $BUILD_DIR/bin/cyxwiz-server-gui"
fi

echo ""
echo "Next Steps:"
if [[ "$BUILD_TARGET" == "all" ]]; then
    echo "  - Run the Engine:         ./$BUILD_DIR/bin/cyxwiz-engine"
    echo "  - Run the Server GUI:     ./$BUILD_DIR/bin/cyxwiz-server-gui"
    echo "  - Run the daemon:         ./$BUILD_DIR/bin/cyxwiz-server-daemon"
elif [[ "$BUILD_TARGET" == "engine" ]]; then
    echo "  - Run the Engine:         ./$BUILD_DIR/bin/cyxwiz-engine"
elif [[ "$BUILD_TARGET" == "server-node" ]]; then
    echo "  - Run the Server GUI:     ./$BUILD_DIR/bin/cyxwiz-server-gui"
    echo "  - Run the daemon:         ./$BUILD_DIR/bin/cyxwiz-server-daemon"
fi

echo ""
echo "============================================================================"
