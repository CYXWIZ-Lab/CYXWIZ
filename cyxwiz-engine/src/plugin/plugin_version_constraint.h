#pragma once

// plugin.json "dependencies": does an installed plugin's version meet the
// version a dependent asks for?

#include "plugin_types.h"

#include <cctype>
#include <sstream>
#include <string>

namespace cyxwiz::plugin {

namespace detail {

// "1", "1.2", "1.2.3" (missing parts are 0); false for anything else.
inline bool ParseConstraintVersion(const std::string& text, PluginVersion& out) {
    unsigned parts[3] = {0, 0, 0};
    int count = 0;
    size_t i = 0;
    while (i < text.size() && count < 3) {
        if (!std::isdigit(static_cast<unsigned char>(text[i]))) return false;
        unsigned value = 0;
        while (i < text.size() && std::isdigit(static_cast<unsigned char>(text[i]))) {
            value = value * 10 + static_cast<unsigned>(text[i] - '0');
            if (value > 65535) return false;
            ++i;
        }
        parts[count++] = value;
        if (i < text.size()) {
            if (text[i] != '.') return false;
            ++i;
            if (i == text.size()) return false;
        }
    }
    if (count == 0 || i != text.size()) return false;
    out.major = static_cast<uint16_t>(parts[0]);
    out.minor = static_cast<uint16_t>(parts[1]);
    out.patch = static_cast<uint16_t>(parts[2]);
    return true;
}

inline int CompareVersions(const PluginVersion& a, const PluginVersion& b) {
    if (a.major != b.major) return a.major < b.major ? -1 : 1;
    if (a.minor != b.minor) return a.minor < b.minor ? -1 : 1;
    if (a.patch != b.patch) return a.patch < b.patch ? -1 : 1;
    return 0;
}

} // namespace detail

// Whether `version` meets `constraint`: "" or "*" (any), "1.2.3" or "=1.2.3"
// (exactly), ">=", ">", "<=", "<", "^1.2.3" (at least, same major; same
// minor when the major is 0), "~1.2.3" (at least, same major.minor).
// Several separated by spaces must all hold (">=1.0 <2.0"). A constraint
// it cannot read is false with the reason in `error`.
inline bool VersionMeetsConstraint(const PluginVersion& version, const std::string& constraint,
                                   std::string* error = nullptr) {
    std::istringstream words(constraint);
    std::string word;
    while (words >> word) {
        if (word == "*") continue;
        std::string op;
        size_t start = 0;
        for (const char* candidate : {">=", "<=", ">", "<", "=", "^", "~"}) {
            const std::string c(candidate);
            if (word.compare(0, c.size(), c) == 0) {
                op = c;
                start = c.size();
                break;
            }
        }
        PluginVersion wanted;
        if (!detail::ParseConstraintVersion(word.substr(start), wanted)) {
            if (error) *error = "cannot read the version \"" + word + "\" (use forms like 1.2.0, >=1.0.0, ^1.2.0)";
            return false;
        }
        const int c = detail::CompareVersions(version, wanted);
        bool ok = false;
        if (op.empty() || op == "=") ok = c == 0;
        else if (op == ">=") ok = c >= 0;
        else if (op == ">") ok = c > 0;
        else if (op == "<=") ok = c <= 0;
        else if (op == "<") ok = c < 0;
        else if (op == "^") ok = c >= 0 && version.major == wanted.major &&
                                 (wanted.major != 0 || version.minor == wanted.minor);
        else if (op == "~") ok = c >= 0 && version.major == wanted.major && version.minor == wanted.minor;
        if (!ok) return false;
    }
    return true;
}

} // namespace cyxwiz::plugin
