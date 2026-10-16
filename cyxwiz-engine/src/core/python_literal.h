// Text as a Python string literal (TOFIX133 P0 item 16). Engine code that
// must pass a value into Python source (a file path, a breakpoint condition)
// quotes it with this, so backslashes, quotes and newlines stay data.
#pragma once

#include <cstdio>
#include <string>

namespace cyxwiz {

inline std::string PythonStringLiteral(const std::string& text) {
    std::string out = "'";
    for (const char c : text) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\'': out += "\\'"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\x%02x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;  // UTF-8 bytes pass through; Python source is UTF-8
                }
        }
    }
    out += "'";
    return out;
}

}  // namespace cyxwiz
