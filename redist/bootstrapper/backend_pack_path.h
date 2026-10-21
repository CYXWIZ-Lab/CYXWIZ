#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

bool IsCanonicalBackendPackRelativePath(std::string_view value);
std::string FoldBackendPackPath(std::string value);
std::filesystem::path BackendPackNativeRelativePath(std::string_view value);
// Filesystem access form only; never serialize this into signed metadata.
// Absolute Windows paths support long names without a registry dependency.
std::filesystem::path BackendPackIoPath(const std::filesystem::path& path);

}  // namespace cyxwiz::runtime
