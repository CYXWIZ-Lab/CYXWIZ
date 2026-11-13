#pragma once

// One tensor in one file: u32 rank, u64 dims, u32 dtype, then the row-major
// data. Checkpoint parameters and the Studio Debugger's layer captures
// (TOFIX178 6.1c) are written in it.

#include <cyxwiz/tensor.h>

#include <filesystem>
#include <optional>
#include <string>

namespace cyxwiz {

enum class ArrayFireHostSyncCategory;  // algorithms/arrayfire_backend_utils.h

// Copies the tensor to the host once, counted under `category` and `site`.
// On failure returns false and says why in `error`.
bool WriteTensorFile(const std::filesystem::path& path, const Tensor& tensor, ArrayFireHostSyncCategory category,
                     const char* site, std::string* error = nullptr);

// Float32, Float64, Int32, Int64 and UInt8; anything else, a bad header or
// a short file is an error.
std::optional<Tensor> ReadTensorFile(const std::filesystem::path& path, std::string* error = nullptr);

}  // namespace cyxwiz
