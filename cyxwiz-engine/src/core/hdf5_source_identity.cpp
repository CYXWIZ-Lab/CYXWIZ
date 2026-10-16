#include "hdf5_source_identity.h"

#include <filesystem>

namespace cyxwiz {

std::optional<Hdf5SourceStamp> ReadHdf5SourceStamp(
    const std::string& path, std::string& error) {
    namespace fs = std::filesystem;
    error.clear();
    if (path.empty() || path.find('\0') != std::string::npos) {
        error = "HDF5 source path is empty or contains a null byte";
        return std::nullopt;
    }
    std::error_code ec;
    const auto canonical = fs::canonical(path, ec);
    if (ec) {
        error = "Cannot resolve HDF5 source: " + ec.message();
        return std::nullopt;
    }
    if (!fs::is_regular_file(canonical, ec)) {
        error = ec ? "Cannot inspect HDF5 source type: " + ec.message()
                   : "HDF5 source must be a regular file";
        return std::nullopt;
    }
    const auto size = fs::file_size(canonical, ec);
    if (ec) {
        error = "Cannot inspect HDF5 source size: " + ec.message();
        return std::nullopt;
    }
    const auto modified = fs::last_write_time(canonical, ec);
    if (ec) {
        error = "Cannot inspect HDF5 source modification time: " + ec.message();
        return std::nullopt;
    }
    return Hdf5SourceStamp{canonical.string(), static_cast<uint64_t>(size),
                          static_cast<int64_t>(modified.time_since_epoch().count())};
}

} // namespace cyxwiz
