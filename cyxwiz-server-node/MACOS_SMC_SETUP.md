# macOS SMC Access Setup for Temperature & Power Monitoring

## Overview

The CyxWiz Server Node uses Apple's System Management Controller (SMC) to read hardware temperature and power consumption data on macOS. Due to macOS security restrictions, SMC access requires proper code signing with entitlements.

## Current Status

✅ **Working Without Entitlements:**
- GPU detection and enumeration
- GPU VRAM usage monitoring
- GPU utilization monitoring (via IOKit)
- CPU information (name, cores, frequency)
- System memory monitoring

⚠️ **Requires Entitlements:**
- CPU/GPU temperature readings
- Power consumption data

Without proper entitlements, temperature and power readings will return `0`.

## Setup Methods

### Method 1: Ad-hoc Code Signing (Development/Testing)

For local development and testing, you can use ad-hoc code signing:

```bash
# Navigate to the build directory
cd build/macos-release/bin

# Sign the daemon with entitlements
codesign --force --deep --sign - \
  --entitlements ../../../cyxwiz-server-node/macos_entitlements.plist \
  --timestamp=none \
  cyxwiz-server-daemon

# Sign the GUI client with entitlements
codesign --force --deep --sign - \
  --entitlements ../../../cyxwiz-server-node/macos_entitlements.plist \
  --timestamp=none \
  cyxwiz-server-gui
```

**Verify the signing:**
```bash
codesign -d --entitlements - cyxwiz-server-daemon
```

### Method 2: Developer Certificate Signing (Recommended for Distribution)

If you have an Apple Developer account:

1. **Get your signing identity:**
   ```bash
   security find-identity -v -p codesigning
   ```

2. **Sign with your developer certificate:**
   ```bash
   # Replace "Developer ID Application: Your Name (TEAM_ID)" with your actual identity
   codesign --force --deep \
     --sign "Developer ID Application: Your Name (TEAM_ID)" \
     --entitlements cyxwiz-server-node/macos_entitlements.plist \
     --options runtime \
     --timestamp \
     build/macos-release/bin/cyxwiz-server-daemon

   codesign --force --deep \
     --sign "Developer ID Application: Your Name (TEAM_ID)" \
     --entitlements cyxwiz-server-node/macos_entitlements.plist \
     --options runtime \
     --timestamp \
     build/macos-release/bin/cyxwiz-server-gui
   ```

### Method 3: Automate with CMake (Advanced)

Add to `cyxwiz-server-node/CMakeLists.txt`:

```cmake
if(APPLE)
    # Define entitlements file path
    set(MACOS_ENTITLEMENTS "${CMAKE_CURRENT_SOURCE_DIR}/macos_entitlements.plist")

    # Add post-build code signing for daemon
    add_custom_command(TARGET cyxwiz-server-daemon POST_BUILD
        COMMAND codesign --force --deep --sign -
                --entitlements ${MACOS_ENTITLEMENTS}
                --timestamp=none
                $<TARGET_FILE:cyxwiz-server-daemon>
        COMMENT "Code signing daemon with SMC entitlements"
        VERBATIM
    )

    # Add post-build code signing for GUI
    add_custom_command(TARGET cyxwiz-server-gui POST_BUILD
        COMMAND codesign --force --deep --sign -
                --entitlements ${MACOS_ENTITLEMENTS}
                --timestamp=none
                $<TARGET_FILE:cyxwiz-server-gui>
        COMMENT "Code signing GUI with SMC entitlements"
        VERBATIM
    )
endif()
```

## Entitlements Explained

The `macos_entitlements.plist` file contains:

| Entitlement | Purpose |
|-------------|---------|
| `com.apple.security.device.smc` | Access to AppleSMC for temperature/power data |
| `com.apple.security.network.server` | Run gRPC/HTTP servers |
| `com.apple.security.network.client` | Connect to external services |
| `com.apple.security.files.user-selected.read-write` | File system access |
| `com.apple.security.cs.allow-dyld-environment-variables` | Load dynamic libraries (ArrayFire, etc.) |
| `com.apple.security.cs.disable-library-validation` | Allow third-party libraries |

## Testing SMC Access

After signing, test if SMC access is working:

```bash
# Run the daemon
./build/macos-release/bin/cyxwiz-server-daemon

# Check the logs for SMC messages:
# ✅ Success: "SMC: Successfully opened connection"
# ❌ Failure: "SMC: IOServiceOpen failed" or "SMC: No SMC device found"
```

## Troubleshooting

### Issue: "SMC: IOServiceOpen failed"

**Possible causes:**
1. Binary not signed with entitlements
2. macOS System Integrity Protection (SIP) blocking access
3. Binary needs to be notarized (for Hardened Runtime)

**Solutions:**
- Verify code signing: `codesign -d --entitlements - cyxwiz-server-daemon`
- Re-sign with entitlements
- For distribution, notarize the app with Apple

### Issue: Temperature still shows 0°C

**Check:**
1. Verify SMC connection logs show success
2. Ensure you're reading the correct SMC keys for your hardware
3. Intel vs Apple Silicon may have different SMC key names

**Common SMC temperature keys:**
- CPU: `TC0P` (CPU proximity), `TC0D` (CPU die), `TC0E`, `TC0F`
- GPU: `TG0P` (GPU proximity), `TG0D` (GPU die)

### Issue: App crashes on startup after signing

**Cause:** Incompatible entitlements or signing issues

**Solution:**
```bash
# Remove all code signatures
codesign --remove-signature build/macos-release/bin/cyxwiz-server-daemon

# Re-sign correctly
codesign --force --deep --sign - \
  --entitlements cyxwiz-server-node/macos_entitlements.plist \
  build/macos-release/bin/cyxwiz-server-daemon
```

## Alternative: Run with `sudo` (Not Recommended)

If code signing doesn't work, you can run with elevated privileges:

```bash
sudo ./build/macos-release/bin/cyxwiz-server-daemon
```

**⚠️ Warning:** Running as root is not recommended for security reasons and should only be used for testing.

## For Production Deployment

1. **Enroll in Apple Developer Program** ($99/year)
2. **Request App Store Distribution Certificate**
3. **Sign with hardened runtime:**
   ```bash
   codesign --force --deep \
     --sign "Developer ID Application: Your Name" \
     --entitlements macos_entitlements.plist \
     --options runtime \
     --timestamp \
     cyxwiz-server-daemon
   ```
4. **Notarize with Apple:**
   ```bash
   xcrun notarytool submit cyxwiz-server.dmg \
     --apple-id your@email.com \
     --team-id TEAM_ID \
     --password app-specific-password \
     --wait
   ```
5. **Staple the notarization:**
   ```bash
   xcrun stapler staple cyxwiz-server-daemon
   ```

## Additional Resources

- [Apple Code Signing Guide](https://developer.apple.com/library/archive/documentation/Security/Conceptual/CodeSigningGuide/)
- [Hardened Runtime](https://developer.apple.com/documentation/security/hardened_runtime)
- [Notarizing macOS Software](https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution)
- [SMC Keys Database](https://github.com/acidanthera/VirtualSMC/blob/master/Docs/SMCKeys.txt)

## Support

If SMC access still doesn't work after following this guide:

1. Check macOS Console app for security messages
2. Verify System Preferences > Security & Privacy hasn't blocked the app
3. Try creating a new entitlements file with minimal permissions
4. Consider using `powermetrics` command as a fallback (requires sudo)

---

**Note:** SMC access requirements may change with future macOS versions. Always test on your target macOS version.
