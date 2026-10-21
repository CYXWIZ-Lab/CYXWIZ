#pragma once

// The last Python scan, kept so a start does not run every interpreter again
// (each one runs three times: version, venv, pip; about 5 s on the owner's
// PC). The cache is used while every interpreter it lists is still there
// unchanged (same size and modified time), one of them is usable, the saved
// path is among them, and the Engine needs the same Python as when it was
// written. Anything else, or "Scan again" in the Python dialog, scans fully.

#include "python_setup_presentation.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::pythonsetup {

struct FileStamp {
    std::uintmax_t size = 0;
    long long modified = 0;
    bool operator==(const FileStamp&) const = default;
};

struct ScanCache {
    static constexpr int kVersion = 1;
    std::string required;  // what this Engine needed ("3.12")
    struct Entry {
        Candidate candidate;
        FileStamp stamp;
    };
    std::vector<Entry> entries;
};

std::string ScanCacheToJson(const ScanCache& cache);
// nullopt when the text is not a cache this version reads.
std::optional<ScanCache> ScanCacheFromJson(const std::string& text);

// The file's stamp, or nullopt when it is not there.
using StampOf = std::function<std::optional<FileStamp>(const std::string& path)>;
std::optional<FileStamp> StampOfFile(const std::string& path);

// The scan result the cache gives, or nullopt when a full scan is needed.
std::optional<ScanResult> ResultFromCache(const ScanCache& cache, const std::string& configured,
                                          const std::string& required, const StampOf& stamp_of);

// A cache of a full scan's result.
ScanCache CacheOf(const ScanResult& scan, const std::string& required, const StampOf& stamp_of);

}  // namespace cyxwiz::pythonsetup
