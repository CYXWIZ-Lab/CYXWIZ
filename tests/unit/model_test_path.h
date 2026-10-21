#pragma once

#include <chrono>
#include <filesystem>
#include <string>

namespace cyxwiz::test {

inline std::filesystem::path UniqueModelPath(const char* stem) {
    const auto suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
        (std::string(stem) + std::to_string(suffix) + ".cyxmodel");
}

} // namespace cyxwiz::test
