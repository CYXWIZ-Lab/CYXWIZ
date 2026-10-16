#pragma once

// Write a JSON document so readers never see a partial file: serialize to a
// sibling temporary, flush, then atomically replace the target. On failure
// the previous file stays as it was. Shared by the crash heartbeat and the
// training trace (tofix94).

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <optional>
#include <system_error>
#include <thread>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace cyxwiz {

inline bool ReplaceFileAtomically(const std::filesystem::path& temporary,
                                  const std::filesystem::path& target) {
#ifdef _WIN32
    return MoveFileExW(temporary.c_str(), target.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
#else
    std::error_code error;
    std::filesystem::rename(temporary, target, error);
    return !error;
#endif
}

inline bool WriteJsonFileAtomically(const std::filesystem::path& target,
                                    const nlohmann::json& document) {
    auto temporary = target;
    temporary += ".tmp";

    {
        std::ofstream file(temporary, std::ios::trunc);
        if (!file) {
            return false;
        }
        file << std::setw(2) << document << '\n';
        file.flush();
        if (!file) {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            return false;
        }
    }

    if (ReplaceFileAtomically(temporary, target)) {
        return true;
    }

    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    return false;
}

// Reads a file written by WriteJsonFileAtomically. On Windows the open can
// fail for a moment while the writer swaps the file in; that open is retried
// for a bounded ~20 ms. A file that opens but does not parse is not retried.
inline std::optional<nlohmann::json> ReadJsonFileWithRetry(const std::filesystem::path& path) {
    constexpr int kAttempts = 20;
    for (int attempt = 0; attempt < kAttempts; ++attempt) {
        std::ifstream file(path);
        if (!file.is_open()) {
            std::error_code error;
            if (!std::filesystem::exists(path, error)) {
                return std::nullopt;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        return nlohmann::json::parse(file);
    }
    return std::nullopt;
}

} // namespace cyxwiz
