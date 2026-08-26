#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

bool IsSafeProductVersion(std::string_view value);

bool LoadProductReleaseVersion(
    const std::filesystem::path& active_base_directory,
    std::string& product_version,
    std::string& error);

}  // namespace cyxwiz::runtime
