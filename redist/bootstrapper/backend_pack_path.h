#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

bool IsCanonicalBackendPackRelativePath(std::string_view value);
// A pack-relative path as compared (ASCII lowercase). Inline, so the Engine's
// plugin loader compares paths the same way without linking the pack service.
inline std::string FoldBackendPackPath(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}
std::filesystem::path BackendPackNativeRelativePath(std::string_view value);
// Filesystem access form only; never serialize this into signed metadata.
// Absolute Windows paths support long names without a registry dependency.
std::filesystem::path BackendPackIoPath(const std::filesystem::path& path);

}  // namespace cyxwiz::runtime
