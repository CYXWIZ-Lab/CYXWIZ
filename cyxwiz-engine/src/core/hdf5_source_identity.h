#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace cyxwiz {

struct Hdf5SourceStamp {
    std::string canonical_path;
    uint64_t size = 0;
    int64_t modified = 0;
    bool operator==(const Hdf5SourceStamp&) const = default;
};

// Filesystem identity only, available without HDF5. Same-size rewrites within
// timestamp granularity are indistinguishable; this is not a hash or file lock.
std::optional<Hdf5SourceStamp> ReadHdf5SourceStamp(
    const std::string& path, std::string& error);

} // namespace cyxwiz
