#!/bin/bash
# ============================================================================
# CyxWiz Setup Script for Linux/macOS
# ============================================================================
# This script checks for required dependencies and sets up the build environment
# for the CyxWiz project on Linux and macOS.
#
# Requirements checked:
#   - GCC/Clang compiler
#   - CMake 3.20+
#   - Python 3.8+ (optional)
#   - Rust/Cargo 1.70+
#   - vcpkg (will be cloned and bootstrapped if missing)
# ============================================================================

set -e  # Exit on error

echo ""
echo "============================================================================"
echo "CyxWiz Setup Script for Linux/macOS"
echo "============================================================================"
echo ""

ERROR_COUNT=0
WARNING_COUNT=0

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "Detected OS: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "Detected OS: macOS"
else
    echo "[ERROR] Unsupported OS: $OSTYPE"
    exit 1
fi
echo ""

# ============================================================================
# Check for CMake 3.20+
# ============================================================================
echo "[1/5] Checking for CMake 3.20+..."
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | sed 's/cmake version //')
    echo "[OK] CMake found: version $CMAKE_VERSION"

    # Check version (basic check for 3.20+)
    MAJOR=$(echo $CMAKE_VERSION | cut -d. -f1)
    MINOR=$(echo $CMAKE_VERSION | cut -d. -f2)

    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 20 ]); then
        echo "[ERROR] CMake version too old. Need 3.20+, found $CMAKE_VERSION"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
else
    echo "[ERROR] CMake not found!"
    echo ""
    if [[ "$OS" == "linux" ]]; then
        echo "Install with: sudo apt-get install cmake"
    else
        echo "Install with: brew install cmake"
    fi
    ERROR_COUNT=$((ERROR_COUNT + 1))
fi
echo ""

# ============================================================================
# Check for C++ compiler
# ============================================================================
echo "[2/5] Checking for C++ compiler..."
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -1)
    echo "[OK] GCC found: $GCC_VERSION"
elif command -v clang++ &> /dev/null; then
    CLANG_VERSION=$(clang++ --version | head -1)
    echo "[OK] Clang found: $CLANG_VERSION"
else
    echo "[ERROR] No C++ compiler found!"
    echo ""
    if [[ "$OS" == "linux" ]]; then
        echo "Install with: sudo apt-get install build-essential"
    else
        echo "Install with: xcode-select --install"
    fi
    ERROR_COUNT=$((ERROR_COUNT + 1))
fi
echo ""

# ============================================================================
# Check for Python 3.8+ (optional)
# ============================================================================
echo "[3/5] Checking for Python 3.8+ (optional)..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | sed 's/Python //')
    echo "[OK] Python found: version $PYTHON_VERSION"

    # Check version (basic check for 3.8+)
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 8 ]); then
        echo "[WARNING] Python version too old. Need 3.8+, found $PYTHON_VERSION"
        WARNING_COUNT=$((WARNING_COUNT + 1))
    fi
else
    echo "[WARNING] Python not found"
    echo ""
    echo "Python is optional but recommended for scripting support."
    if [[ "$OS" == "linux" ]]; then
        echo "Install with: sudo apt-get install python3 python3-dev"
    else
        echo "Install with: brew install python@3.11"
    fi
    WARNING_COUNT=$((WARNING_COUNT + 1))
fi
echo ""

# ============================================================================
# Check for Rust/Cargo 1.70+
# ============================================================================
echo "[4/5] Checking for Rust/Cargo 1.70+..."
if command -v cargo &> /dev/null; then
    CARGO_VERSION=$(cargo --version | sed 's/cargo //')
    echo "[OK] Cargo found: version $CARGO_VERSION"

    # Check version (basic check for 1.70+)
    MAJOR=$(echo $CARGO_VERSION | cut -d. -f1)
    MINOR=$(echo $CARGO_VERSION | cut -d. -f2)

    if [ "$MAJOR" -lt 1 ] || ([ "$MAJOR" -eq 1 ] && [ "$MINOR" -lt 70 ]); then
        echo "[ERROR] Cargo version too old. Need 1.70+, found $CARGO_VERSION"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi
else
    echo "[ERROR] Rust/Cargo not found!"
    echo ""
    echo "Install with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  source \$HOME/.cargo/env"
    ERROR_COUNT=$((ERROR_COUNT + 1))
fi
echo ""

# ============================================================================
# Setup vcpkg
# ============================================================================
echo "[5/5] Setting up vcpkg..."

if [ -d "vcpkg/.git" ]; then
    echo "[OK] vcpkg repository already exists"
else
    echo "[INFO] Cloning vcpkg repository..."
    git clone https://github.com/microsoft/vcpkg.git
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to clone vcpkg repository"
        echo ""
        echo "Make sure git is installed and accessible"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    else
        echo "[OK] vcpkg cloned successfully"
    fi
fi

if [ -f "vcpkg/vcpkg" ]; then
    echo "[OK] vcpkg already bootstrapped"
else
    if [ -d "vcpkg" ]; then
        echo "[INFO] Bootstrapping vcpkg..."
        cd vcpkg
        ./bootstrap-vcpkg.sh
        if [ $? -ne 0 ]; then
            echo "[ERROR] vcpkg bootstrap failed"
            cd ..
            ERROR_COUNT=$((ERROR_COUNT + 1))
        else
            cd ..
            echo "[OK] vcpkg bootstrapped successfully"
        fi
    fi
fi

echo ""
echo "[INFO] Installing vcpkg dependencies..."
echo "This may take several minutes on first run..."
if [ -f "vcpkg/vcpkg" ]; then
    vcpkg/vcpkg install
    if [ $? -ne 0 ]; then
        echo "[WARNING] Some vcpkg packages failed to install"
        echo "You may need to install them manually later"
        WARNING_COUNT=$((WARNING_COUNT + 1))
    else
        echo "[OK] vcpkg dependencies installed"
    fi
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "Setup Summary"
echo "============================================================================"
echo ""

if [ $ERROR_COUNT -gt 0 ]; then
    echo "[FAILED] Setup completed with $ERROR_COUNT error(s) and $WARNING_COUNT warning(s)"
    echo ""
    echo "Please fix the errors above and run this script again."
    echo ""
    exit 1
elif [ $WARNING_COUNT -gt 0 ]; then
    echo "[WARNING] Setup completed with $WARNING_COUNT warning(s)"
    echo ""
    echo "The warnings above are for optional components."
    echo "You can proceed with the build, but some features may be unavailable."
    echo ""
else
    echo "[SUCCESS] All dependencies are installed!"
    echo ""
fi

echo "Next Steps:"
echo "  1. Build the project:"
echo "     ./build.sh"
echo ""
echo "  2. Build specific components:"
echo "     ./build.sh --engine           (Build only Engine)"
echo "     ./build.sh --server-node      (Build only Server Node)"
echo "     ./build.sh --central-server   (Build only Central Server)"
echo ""
echo "  3. Build in Debug mode:"
echo "     ./build.sh --debug"
echo ""
echo "  4. For more options:"
echo "     ./build.sh --help"
echo ""
echo "============================================================================"

if [ $ERROR_COUNT -gt 0 ]; then
    exit 1
fi
