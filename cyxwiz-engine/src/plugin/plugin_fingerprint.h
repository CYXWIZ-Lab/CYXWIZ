// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// Build fingerprint of a plugin (TOFIX125 P5 step 5.1, Part D6, decision
// D4). A trainable plugin node is a real cyxwiz::Module: the plugin shares
// C++ objects with the Engine, so it must be built the way the Engine was
// (compiler family, standard library, build type, pointer size, SDK and
// ArrayFire versions). A mismatch would crash; the fingerprint turns it into
// a refused load with the reason.
//
// CYXWIZ_PLUGIN_ENTRY exports the fingerprint of the plugin's own build
// (CyxwizPluginBuildFingerprint); the loader compares it with the Engine's
// before any other plugin code runs. The text is
//   "sdk=0.2;api=1.4;compiler=msvc-19;stdlib=msvc-stl;build=release;ptr=64;arrayfire=3.10"
// and is computed where this header is compiled: in the plugin for the
// plugin, in the Engine for the Engine.

#include <cyxwiz/version.h>

#include <string>
#include <utility>
#include <vector>

#if __has_include(<af/version.h>)
#include <af/version.h>
#endif

// A plugin with panels draws in the Engine's ImGui context
// (IPlugin::SetImGuiContext), so it must be built with the Engine's ImGui:
// the same version and the same configuration. The SDK ships that ImGui
// (CyxWiz::PluginImGui, plugin-api-reference.md section 9). Where <imgui.h>
// is on the include path the fingerprint names it ("imgui=" below).
#if __has_include(<imgui.h>)
#include <imgui.h>
#endif

namespace cyxwiz::plugin {

// Plugin API version this SDK provides: the same numbers as
// PLUGIN_API_VERSION_* in plugin_types.h, which checks at compile time that
// they match (plugin_types.h includes this header, so the numbers cannot be
// read from there). Raise both whenever the SDK's interfaces or the classes a
// plugin derives from change shape. A plugin whose minor is higher than the
// Engine's is refused.
inline constexpr int kFingerprintApiMajor = 1;
inline constexpr int kFingerprintApiMinor = 4;

inline std::string CurrentBuildFingerprint() {
#ifdef CYXWIZ_TEST_BUILD_FINGERPRINT
    return CYXWIZ_TEST_BUILD_FINGERPRINT;  // tests only: a plugin built "differently"
#else
    std::string compiler;
#if defined(__clang__)
    compiler = "clang-" + std::to_string(__clang_major__);
#elif defined(_MSC_VER)
    compiler = "msvc-" + std::to_string(_MSC_VER / 100);  // 19: VS 2015-2022, one binary interface
#elif defined(__GNUC__)
    compiler = "gcc-" + std::to_string(__GNUC__);
#else
    compiler = "unknown";
#endif

#if defined(_MSVC_STL_VERSION) || defined(_CPPLIB_VER)
    const char* stdlib = "msvc-stl";
#elif defined(_LIBCPP_VERSION)
    const char* stdlib = "libc++";
#elif defined(__GLIBCXX__)
    const char* stdlib = "libstdc++";
#else
    const char* stdlib = "unknown";
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
    const char* build = "debug";  // debug CRT and checked iterators change object layouts
#elif defined(_GLIBCXX_DEBUG)
    const char* build = "debug";
#else
    const char* build = "release";
#endif

#if defined(AF_VERSION_MAJOR)
    const std::string arrayfire = std::to_string(AF_VERSION_MAJOR) + "." + std::to_string(AF_VERSION_MINOR);
#else
    const std::string arrayfire = "none";  // a plugin that does not include the backend
#endif

#if defined(IMGUI_VERSION_NUM)
    // The version, docking, and the sizes of the structures a configuration
    // change alters (imconfig.h: ImDrawIdx, ImTextureID, user fields), so a
    // plugin built with another ImGui is refused instead of crashing as it draws.
#if defined(IMGUI_HAS_DOCK)
    const char* docking = "dock";
#else
    const char* docking = "nodock";
#endif
    const std::string imgui = std::to_string(IMGUI_VERSION_NUM) + "-" + docking +
                              "-io" + std::to_string(sizeof(ImGuiIO)) + "-style" + std::to_string(sizeof(ImGuiStyle)) +
                              "-vtx" + std::to_string(sizeof(ImDrawVert)) + "-idx" + std::to_string(sizeof(ImDrawIdx)) +
                              "-tex" + std::to_string(sizeof(ImTextureID));
#else
    const std::string imgui = "none";  // a plugin that does not draw
#endif

    return "sdk=" + std::to_string(CYXWIZ_VERSION_MAJOR) + "." + std::to_string(CYXWIZ_VERSION_MINOR) +
           ";api=" + std::to_string(kFingerprintApiMajor) + "." + std::to_string(kFingerprintApiMinor) +
           ";compiler=" + compiler + ";stdlib=" + stdlib + ";build=" + build +
           ";ptr=" + std::to_string(sizeof(void*) * 8) + ";arrayfire=" + arrayfire + ";imgui=" + imgui;
#endif
}

struct FingerprintDifference {
    std::string key;
    std::string plugin;  // the plugin's value ("" when missing)
    std::string engine;
};

// key=value pairs in order.
std::vector<std::pair<std::string, std::string>> ParseBuildFingerprint(const std::string& text);

// What differs in a way that matters. Equal: sdk (major.minor), compiler,
// stdlib, build, ptr, arrayfire and imgui (each when both name one), api
// major; the plugin's api minor may not be newer than the Engine's.
std::vector<FingerprintDifference> CompareBuildFingerprints(const std::string& engine, const std::string& plugin);

// "The plugin was built with build=debug, arrayfire=3.9; this Engine with
// build=release, arrayfire=3.10. Rebuild the plugin against the CyxWiz SDK
// 0.2 with the same settings."
std::string DescribeFingerprintMismatch(const std::vector<FingerprintDifference>& differences);

// The loader's message for a plugin that exports no fingerprint.
std::string MissingFingerprintMessage(const std::string& engine);

}  // namespace cyxwiz::plugin
