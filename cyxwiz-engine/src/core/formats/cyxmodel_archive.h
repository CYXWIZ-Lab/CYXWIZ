#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz::formats {

using CyxModelAssets = std::map<std::string, std::vector<uint8_t>>;

struct CyxModelArchiveLimits {
    uint64_t payload_bytes = uint64_t{2} * 1024 * 1024 * 1024;
    uint32_t entries = 100000;
    uint32_t path_bytes = 4096;
};

// Storage for the existing package inventory, not a second model serializer.
// Throws on invalid input; Read publishes no partial assets. Binary output uses
// staged atomic replacement; directory output requires a new destination.
class CyxModelArchive {
public:
    static bool IsV3(const std::filesystem::path& path);
    static CyxModelAssets Read(const std::filesystem::path& path,
                              const CyxModelArchiveLimits& limits = {});
    static void WriteBinary(const std::filesystem::path& path,
                            const CyxModelAssets& assets,
                            const CyxModelArchiveLimits& limits = {});
    static void WriteDirectory(const std::filesystem::path& path,
                               const CyxModelAssets& assets,
                               const CyxModelArchiveLimits& limits = {});
};

} // namespace cyxwiz::formats
