#include "product_release_version.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <system_error>

namespace cyxwiz::runtime {
namespace {

using Json = nlohmann::json;

constexpr std::uintmax_t kMaximumRuntimeVersionsBytes = 16 * 1024;

bool HasExactKeys(
    const Json& value,
    std::initializer_list<const char*> keys) {
    if (!value.is_object() || value.size() != keys.size()) return false;
    for (const char* key : keys) {
        if (!value.contains(key)) return false;
    }
    return true;
}

}  // namespace

bool IsSafeProductVersion(std::string_view value) {
    return !value.empty() && value.size() <= 64 &&
        std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '.' ||
                   character == '_' || character == '+' || character == '-';
        });
}

bool LoadProductReleaseVersion(
    const std::filesystem::path& active_base_directory,
    std::string& product_version,
    std::string& error) {
    product_version.clear();
    const auto path = active_base_directory / "RUNTIME_VERSIONS.json";
    std::error_code filesystem_error;
    const auto status = std::filesystem::symlink_status(path, filesystem_error);
    if (filesystem_error ||
        status.type() != std::filesystem::file_type::regular) {
        error = "The active base runtime version identity is missing or redirected";
        return false;
    }
    const auto size = std::filesystem::file_size(path, filesystem_error);
    if (filesystem_error || size == 0 || size > kMaximumRuntimeVersionsBytes) {
        error = "The active base runtime version identity violates its byte bound";
        return false;
    }

    std::ifstream stream(path, std::ios::binary);
    const Json document = Json::parse(stream, nullptr, false);
    if (stream.bad() || document.is_discarded() ||
        !HasExactKeys(document, {"arrayfire", "cyxwiz", "python"}) ||
        !document["arrayfire"].is_string() ||
        !document["cyxwiz"].is_string() ||
        !document["python"].is_string()) {
        error = "The active base runtime version identity schema is invalid";
        return false;
    }

    product_version = document["cyxwiz"].get<std::string>();
    if (!IsSafeProductVersion(product_version)) {
        product_version.clear();
        error = "The active base CyxWiz version is invalid";
        return false;
    }
    error.clear();
    return true;
}

}  // namespace cyxwiz::runtime
