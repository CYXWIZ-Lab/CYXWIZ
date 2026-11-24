// SPDX-License-Identifier: Apache-2.0
// Part of the CyxWiz plugin SDK (see cmake/sdk/NOTICE).
#pragma once

// std::regex options that differ between standard libraries.

#include <regex>

namespace cyxwiz::portable {

// The option that anchors ^ and $ at every line of an ECMAScript pattern
// (std::regex_constants::multiline, C++17 LWG 2503). MSVC's standard library
// before VS 2026 (toolset 14.50) has no such option: its ECMAScript matching
// anchors at every line whether asked or not, so there is nothing to add.
// Newer MSVC defines _REGEX_LEGACY_MULTILINE_MODE in <regex>; older does not.
inline std::regex_constants::syntax_option_type RegexMultiline() {
#if defined(_MSVC_STL_UPDATE) && !defined(_REGEX_LEGACY_MULTILINE_MODE)
    return std::regex_constants::syntax_option_type{};
#else
    return std::regex_constants::multiline;
#endif
}

}  // namespace cyxwiz::portable
