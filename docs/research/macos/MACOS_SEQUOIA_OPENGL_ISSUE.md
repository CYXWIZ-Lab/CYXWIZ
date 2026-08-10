# macOS Sequoia OpenGL Incompatibility Issue

## Problem Summary
CyxWiz Engine **cannot run on macOS Sequoia (15.x)** when using OpenCore Patcher with unsupported Intel graphics hardware.

## Affected Systems
- **macOS Version**: Sequoia 15.7+
- **Hardware**: MacBook Pro 15" (2013-2014) with Intel Iris Pro
- **Installation Method**: OpenCore Patcher (OCLP)
- **Graphics**: Intel Iris Pro (Device ID: 0x0d26, Haswell generation)

## Root Cause
1. **Apple deprecated OpenGL** starting in macOS 10.14 Mojave
2. **Apple removed full OpenGL support** in macOS Sequoia (15.x)
3. **Intel Iris Pro is not officially supported** on macOS Sequoia
4. **OpenCore Patcher** provides graphics driver patches for compatibility, but these patches have **incomplete OpenGL/NSGL support**
5. **GLFW cannot create ANY OpenGL context** via NSGL backend

## Error Message
```
[cyxwiz] [error] GLFW Error 65545: NSGL: Failed to find a suitable pixel format
[cyxwiz] [error] Failed to create GLFW window
```

## Diagnostic Test Results
All OpenGL context creation attempts fail:
- ❌ Default window (no hints)
- ❌ OpenGL 2.1 compatibility
- ❌ OpenGL 3.2 Core
- ❌ OpenGL 3.3 Core
- ❌ OpenGL 4.1 Core

**Conclusion**: NSGL (macOS's native OpenGL layer) is completely non-functional on this configuration.

## Solutions

### Option 1: Downgrade macOS ✅ RECOMMENDED
**Difficulty**: Easy
**Compatibility**: Full

Downgrade to macOS **Monterey 12.x** or **Big Sur 11.x**:
- OpenCore Patcher fully supports these versions
- OpenGL works properly on Intel Iris Pro
- Engine runs without modifications

**How to**:
1. Download macOS Monterey installer
2. Use OpenCore Patcher to create bootable USB
3. Clean install or downgrade

### Option 2: Port to Metal/SDL2 🔨 LONG-TERM
**Difficulty**: High
**Compatibility**: Native macOS Sequoia support

Replace OpenGL rendering with Metal:
- **Option A**: Use SDL2 with Metal renderer
- **Option B**: Use MoltenVK (Vulkan to Metal)
- **Option C**: Native Metal via MetalKit

**Estimated effort**: 2-4 weeks for SDL2 port

**Required changes**:
- Replace GLFW with SDL2
- Port ImGui backend to SDL2_Renderer or Metal
- Update shaders to Metal Shading Language
- Test on macOS Sonoma and Sequoia

### Option 3: Mesa3D Software Rendering 🐌 SLOW
**Difficulty**: Medium
**Compatibility**: Works but extremely slow

Use Mesa3D's software OpenGL implementation:
```bash
brew install mesa
export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH
```

**Cons**:
- 10-100x slower than GPU rendering
- Not suitable for ML workloads
- Good for testing UI only

### Option 4: Run in VM/Container 🖥️ WORKAROUND
**Difficulty**: Easy
**Compatibility**: Full Linux/Windows support

Run engine on:
- **UTM VM** with Linux and GPU passthrough
- **Docker Desktop** with X11 forwarding
- **Remote machine** (Windows/Linux) via SSH/VNC

### Option 5: Dual Boot Linux 🐧 ALTERNATIVE
**Difficulty**: Medium
**Compatibility**: Full native GPU support

Install Linux alongside macOS:
- Use rEFInd bootloader
- Install Ubuntu 22.04 or Fedora 39
- Full Intel Iris Pro OpenGL support
- Better ML performance

## Workarounds Attempted (All Failed)

### ✅ Attempted: OpenGL version fallback
```cpp
// Tried: 3.3 Core → 3.2 Core → 2.1 → Default
// Result: All failed with same NSGL error
```

### ✅ Attempted: GLFW window hints
```cpp
glfwWindowHint(GLFW_COCOA_RETINA_FRAMEBUFFER, GLFW_TRUE);
glfwWindowHint(GLFW_SAMPLES, 0);  // Disable multisampling
glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_FALSE);
// Result: No improvement
```

### ✅ Attempted: Minimal window configuration
```cpp
glfwDefaultWindowHints();
glfwCreateWindow(640, 480, "Test", nullptr, nullptr);
// Result: Still fails
```

## Why This Happens

### Apple's OpenGL Deprecation Timeline
- **macOS 10.14 Mojave (2018)**: OpenGL deprecated, Metal recommended
- **macOS 10.15 Catalina (2019)**: OpenGL marked for removal
- **macOS 11 Big Sur (2020)**: OpenGL support reduced
- **macOS 12 Monterey (2021)**: OpenGL still functional but unsupported
- **macOS 13 Ventura (2022)**: Further OpenGL degradation
- **macOS 14 Sonoma (2023)**: OpenGL barely functional
- **macOS 15 Sequoia (2024)**: **OpenGL effectively removed** on unsupported hardware

### OpenCore Patcher Limitations
OpenCore Patcher patches:
- Graphics drivers for basic display output
- Metal framework for basic UI rendering
- **Does NOT restore full NSGL/OpenGL functionality**

Intel Iris Pro (Haswell) was last officially supported in:
- macOS 10.15 Catalina (2019)
- Deprecated in Big Sur (11.x)
- Removed in Monterey (12.x) - OCLP adds support back
- **Broken OpenGL in Sequoia (15.x)** - OCLP cannot fully restore

## Recommendations by Use Case

### For Development (Need working engine NOW)
→ **Downgrade to macOS Monterey 12.x**

### For Production Deployment
→ **Port to SDL2 + Metal** (future-proof)

### For Testing UI Only
→ **Mesa3D software rendering**

### For ML Training
→ **Linux dual boot or VM**

## Prevention

### Add macOS Version Check
Add startup check in `application.cpp`:
```cpp
#ifdef __APPLE__
if (@available(macOS 15.0, *)) {
    spdlog::error("macOS Sequoia (15.x) is not supported due to OpenGL removal");
    spdlog::error("Please use macOS Monterey (12.x) or earlier");
    spdlog::error("Or wait for Metal port (planned for v0.2.0)");
    return false;
}
#endif
```

## References

- [GLFW Issue #1334](https://github.com/glfw/glfw/issues/1334) - macOS OpenGL deprecation
- [OpenCore Patcher Graphics Patches](https://github.com/dortania/OpenCore-Legacy-Patcher/blob/main/docs/GRAPHICS.md)
- [Apple Metal Transition Guide](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [SDL2 Metal Renderer](https://wiki.libsdl.org/SDL2/SDL_RENDERER_METAL)

## Status

- **Current**: Engine cannot run on macOS Sequoia with OCLP + Intel Iris Pro
- **Workaround**: Downgrade to macOS Monterey
- **Long-term**: Metal/SDL2 port planned for v0.2.0

---

**Date**: 2026-01-28
**Tested on**: macOS Sequoia 15.7, Intel Iris Pro, GLFW 3.4.0
**Diagnostic**: All OpenGL context creation fails with NSGL error 65545
**Resolution**: Use macOS Monterey or port to Metal
