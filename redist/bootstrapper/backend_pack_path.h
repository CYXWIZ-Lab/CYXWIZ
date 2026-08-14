#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

bool IsCanonicalBackendPackRelativePath(std::string_view value);
std::string FoldBackendPackPath(std::string value);
std::filesystem::path BackendPackNativeRelativePath(std::string_view value);

}  // namespace cyxwiz::runtime
