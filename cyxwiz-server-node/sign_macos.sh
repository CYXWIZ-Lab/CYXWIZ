#!/bin/bash
# macOS Code Signing Script for CyxWiz Server Node
# Adds SMC entitlements for temperature and power monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build/macos-release/bin"
ENTITLEMENTS="${SCRIPT_DIR}/macos_entitlements.plist"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== CyxWiz macOS Code Signing Tool ===${NC}"
echo ""

# Check if entitlements file exists
if [ ! -f "$ENTITLEMENTS" ]; then
    echo -e "${RED}Error: Entitlements file not found: $ENTITLEMENTS${NC}"
    exit 1
fi

# Check if binaries exist
if [ ! -f "$BUILD_DIR/cyxwiz-server-daemon" ]; then
    echo -e "${RED}Error: cyxwiz-server-daemon not found in $BUILD_DIR${NC}"
    echo "Please build the project first with: ninja -C build/macos-release"
    exit 1
fi

echo -e "${YELLOW}Signing binaries with SMC entitlements...${NC}"
echo ""

# Sign daemon
echo "Signing cyxwiz-server-daemon..."
codesign --force --deep --sign - \
  --entitlements "$ENTITLEMENTS" \
  --timestamp=none \
  "$BUILD_DIR/cyxwiz-server-daemon"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ cyxwiz-server-daemon signed successfully${NC}"
else
    echo -e "${RED}✗ Failed to sign cyxwiz-server-daemon${NC}"
    exit 1
fi

# Sign GUI if it exists
if [ -f "$BUILD_DIR/cyxwiz-server-gui" ]; then
    echo "Signing cyxwiz-server-gui..."
    codesign --force --deep --sign - \
      --entitlements "$ENTITLEMENTS" \
      --timestamp=none \
      "$BUILD_DIR/cyxwiz-server-gui"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ cyxwiz-server-gui signed successfully${NC}"
    else
        echo -e "${RED}✗ Failed to sign cyxwiz-server-gui${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}=== Code Signing Complete ===${NC}"
echo ""
echo "Verifying signatures..."
echo ""

# Verify daemon
echo "--- cyxwiz-server-daemon ---"
codesign -d --entitlements - "$BUILD_DIR/cyxwiz-server-daemon" 2>&1 | grep -A 20 "<?xml"

if [ -f "$BUILD_DIR/cyxwiz-server-gui" ]; then
    echo ""
    echo "--- cyxwiz-server-gui ---"
    codesign -d --entitlements - "$BUILD_DIR/cyxwiz-server-gui" 2>&1 | grep -A 20 "<?xml"
fi

echo ""
echo -e "${GREEN}Done! Binaries are now signed with SMC entitlements.${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} Temperature and power monitoring should now work."
echo "Test by running: ./build/macos-release/bin/cyxwiz-server-daemon"
echo ""
echo "If SMC access still fails, see MACOS_SMC_SETUP.md for advanced options."
