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
//   "sdk=0.2;api=1.2;compiler=msvc-19;stdlib=msvc-stl;build=release;ptr=64;arrayfire=3.10"
// and is computed where this header is compiled: in the plugin for the
// plugin, in the Engine for the Engine.

#include <cyxwiz/version.h>

#include <string>
#include <utility>
#include <vector>

#if __has_include(<af/version.h>)
#include <af/version.h>
#endif

namespace cyxwiz::plugin {

// Plugin API version this SDK provides (plugin_types.h keeps the same numbers
// as PLUGIN_API_VERSION_*).
inline constexpr int kFingerprintApiMajor = 1;
inline constexpr int kFingerprintApiMinor = 2;

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

    return "sdk=" + std::to_string(CYXWIZ_VERSION_MAJOR) + "." + std::to_string(CYXWIZ_VERSION_MINOR) +
           ";api=" + std::to_string(kFingerprintApiMajor) + "." + std::to_string(kFingerprintApiMinor) +
           ";compiler=" + compiler + ";stdlib=" + stdlib + ";build=" + build +
           ";ptr=" + std::to_string(sizeof(void*) * 8) + ";arrayfire=" + arrayfire;
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
// stdlib, build, ptr, arrayfire (when both name one), api major; the plugin's
// api minor may not be newer than the Engine's.
std::vector<FingerprintDifference> CompareBuildFingerprints(const std::string& engine, const std::string& plugin);

// "The plugin was built with build=debug, arrayfire=3.9; this Engine with
// build=release, arrayfire=3.10. Rebuild the plugin against the CyxWiz SDK
// 0.2 with the same settings."
std::string DescribeFingerprintMismatch(const std::vector<FingerprintDifference>& differences);

// The loader's message for a plugin that exports no fingerprint.
std::string MissingFingerprintMessage(const std::string& engine);

}  // namespace cyxwiz::plugin
