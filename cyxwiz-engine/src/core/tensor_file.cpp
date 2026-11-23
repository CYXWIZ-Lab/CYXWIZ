#include "tensor_file.h"

#include "algorithms/arrayfire_backend_utils.h"

#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <vector>

namespace cyxwiz {

namespace {

bool SupportedDataType(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
        case DataType::Float64:
        case DataType::Int32:
        case DataType::Int64:
        case DataType::UInt8:
            return true;
    }
    return false;
}

template <typename T>
T Fail(std::string* error, std::string message, T result) {
    if (error) *error = std::move(message);
    return result;
}

}  // namespace

bool WriteTensorFile(const std::filesystem::path& path, const Tensor& tensor, ArrayFireHostSyncCategory category,
                     const char* site, std::string* error) {
    try {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file.is_open()) return Fail(error, "cannot open " + path.string() + " for writing", false);
        const auto& shape = tensor.Shape();
        const uint32_t rank = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&rank), sizeof(rank));
        for (size_t dim : shape) {
            const uint64_t d = static_cast<uint64_t>(dim);
            file.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }
        const uint32_t dtype = static_cast<uint32_t>(tensor.GetDataType());
        file.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
        const size_t num_bytes = tensor.NumBytes();
        if (num_bytes > static_cast<size_t>((std::numeric_limits<std::streamsize>::max)()))
            return Fail(error, "the tensor is too large to write to " + path.string(), false);
        const ScopedArrayFireHostSyncAttribution attribution(category, site);
        file.write(reinterpret_cast<const char*>(tensor.ReadData()), static_cast<std::streamsize>(num_bytes));
        if (!file.good()) return Fail(error, "writing " + path.string() + " failed", false);
        return true;
    } catch (const std::exception& e) {
        return Fail(error, "writing " + path.string() + " failed: " + e.what(), false);
    }
}

std::optional<Tensor> ReadTensorFile(const std::filesystem::path& path, std::string* error) {
    const std::optional<Tensor> none;
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return Fail(error, "the tensor file is unreadable: " + path.string(), none);
        uint32_t rank = 0;
        file.read(reinterpret_cast<char*>(&rank), sizeof(rank));
        if (!file.good() || rank > 16)
            return Fail(error, "the tensor file has an invalid rank header: " + path.string(), none);
        std::vector<size_t> shape(rank);
        for (auto& dim : shape) {
            uint64_t d = 0;
            file.read(reinterpret_cast<char*>(&d), sizeof(d));
            if (!file.good() || d == 0)
                return Fail(error, "the tensor file has an invalid shape header: " + path.string(), none);
            dim = static_cast<size_t>(d);
        }
        uint32_t dtype = 0;
        file.read(reinterpret_cast<char*>(&dtype), sizeof(dtype));
        if (!file.good())
            return Fail(error, "the tensor file has an invalid data type header: " + path.string(), none);
        const auto data_type = static_cast<DataType>(dtype);
        if (!SupportedDataType(data_type))
            return Fail(error, "the tensor file has an unsupported data type header: " + path.string(), none);
        Tensor tensor(shape, data_type);
        const size_t num_bytes = tensor.NumBytes();
        if (num_bytes > static_cast<size_t>((std::numeric_limits<std::streamsize>::max)()))
            return Fail(error, "the tensor in " + path.string() + " is too large to read", none);
        file.read(reinterpret_cast<char*>(tensor.MutableData()), static_cast<std::streamsize>(num_bytes));
        if (!file.good()) return Fail(error, "the tensor file is truncated: " + path.string(), none);
        return tensor;
    } catch (const std::exception& e) {
        return Fail(error, "reading " + path.string() + " failed: " + e.what(), none);
    }
}

}  // namespace cyxwiz
