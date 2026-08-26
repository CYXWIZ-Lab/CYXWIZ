#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace cyxwiz::runtime {

using AtomicFilePublishValidator = std::function<bool(
    const std::filesystem::path& temporary_file,
    std::string& error)>;

bool PublishRegularFileAtomic(
    const std::filesystem::path& source,
    const std::filesystem::path& destination,
    std::uintmax_t maximum_bytes,
    std::string& error,
    AtomicFilePublishValidator validator = {});

}  // namespace cyxwiz::runtime
