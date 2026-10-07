#pragma once

#include "hdf5_table_adapter.h"

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/H5File.hpp>
#include <stdexcept>

namespace cyxwiz::hdf5_detail {

struct ReadFailure : std::runtime_error {
    Hdf5TableStatus status;
    ReadFailure(Hdf5TableStatus code, const std::string& message)
        : std::runtime_error(message), status(code) {}
};

// Shared by numeric ingestion and bounded hierarchy inspection only.
HighFive::File OpenFile(const std::string& path);
void ValidatePathSyntax(const std::string& path, bool allow_root = false);
void ValidateLocalPath(const HighFive::File& file, const std::string& path,
                       bool allow_root = false);

} // namespace cyxwiz::hdf5_detail
#endif
