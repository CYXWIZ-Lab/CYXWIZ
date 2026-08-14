#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace cyxwiz::runtime {

bool IsLowercaseSha256(std::string_view value);
bool Sha256Bytes(
    std::string_view bytes,
    std::string& digest,
    std::string& error);
bool Sha256File(
    const std::filesystem::path& path,
    std::string& digest,
    std::string& error);

}  // namespace cyxwiz::runtime
