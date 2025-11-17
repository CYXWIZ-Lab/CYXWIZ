#include "cyxwiz/tensor.h"
#include "cyxwiz/device.h"
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <spdlog/spdlog.h>

namespace cyxwiz {

Tensor::Tensor()
    : dtype_(DataType::Float32), device_(nullptr), data_(nullptr), owns_data_(false)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
}

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype)
    : shape_(shape), dtype_(dtype), device_(nullptr), owns_data_(true)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0) {
        data_ = malloc(num_bytes);
        memset(data_, 0, num_bytes);
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(const std::vector<size_t>& shape, const void* data, DataType dtype)
    : shape_(shape), dtype_(dtype), device_(nullptr), owns_data_(true)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0 && data) {
        data_ = malloc(num_bytes);
        memcpy(data_, data, num_bytes);
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), dtype_(other.dtype_), device_(other.device_), owns_data_(true)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(nullptr)
#endif
{
    size_t num_bytes = NumBytes();
    if (num_bytes > 0 && other.data_) {
        data_ = malloc(num_bytes);
        memcpy(data_, other.data_, num_bytes);
    } else {
        data_ = nullptr;
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), dtype_(other.dtype_), device_(other.device_),
      data_(other.data_), owns_data_(other.owns_data_)
#ifdef CYXWIZ_HAS_ARRAYFIRE
    , af_array_(other.af_array_)
#endif
{
    other.data_ = nullptr;
    other.owns_data_ = false;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    other.af_array_ = nullptr;
#endif
}

Tensor::~Tensor() {
    if (owns_data_ && data_) {
        free(data_);
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (af_array_) {
        delete af_array_;
    }
#endif
}

size_t Tensor::NumElements() const {
    size_t count = 1;
    for (size_t dim : shape_) {
        count *= dim;
    }
    return count;
}

size_t Tensor::NumBytes() const {
    size_t elem_size = 0;
    switch (dtype_) {
        case DataType::Float32: elem_size = 4; break;
        case DataType::Float64: elem_size = 8; break;
        case DataType::Int32: elem_size = 4; break;
        case DataType::Int64: elem_size = 8; break;
        case DataType::UInt8: elem_size = 1; break;
    }
    return NumElements() * elem_size;
}

void* Tensor::Data() {
    return data_;
}

const void* Tensor::Data() const {
    return data_;
}

Tensor Tensor::Zeros(const std::vector<size_t>& shape, DataType dtype) {
    return Tensor(shape, dtype);
}

Tensor Tensor::Ones(const std::vector<size_t>& shape, DataType dtype) {
    Tensor t(shape, dtype);

    // Fill with ones based on data type
    size_t num_elements = t.NumElements();
    switch (dtype) {
        case DataType::Float32: {
            float* data = static_cast<float*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1.0f;
            }
            break;
        }
        case DataType::Float64: {
            double* data = static_cast<double*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1.0;
            }
            break;
        }
        case DataType::Int32: {
            int32_t* data = static_cast<int32_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1;
            }
            break;
        }
        case DataType::Int64: {
            int64_t* data = static_cast<int64_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1;
            }
            break;
        }
        case DataType::UInt8: {
            uint8_t* data = static_cast<uint8_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = 1;
            }
            break;
        }
    }

    return t;
}

Tensor Tensor::Random(const std::vector<size_t>& shape, DataType dtype) {
    Tensor t(shape, dtype);

    // Fill with random values [0, 1) based on data type
    size_t num_elements = t.NumElements();
    switch (dtype) {
        case DataType::Float32: {
            float* data = static_cast<float*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            }
            break;
        }
        case DataType::Float64: {
            double* data = static_cast<double*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
            }
            break;
        }
        case DataType::Int32: {
            int32_t* data = static_cast<int32_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = rand() % 100;  // Random int [0, 99]
            }
            break;
        }
        case DataType::Int64: {
            int64_t* data = static_cast<int64_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = rand() % 100;  // Random int [0, 99]
            }
            break;
        }
        case DataType::UInt8: {
            uint8_t* data = static_cast<uint8_t*>(t.Data());
            for (size_t i = 0; i < num_elements; i++) {
                data[i] = rand() % 256;  // Random byte [0, 255]
            }
            break;
        }
    }

    return t;
}

Tensor Tensor::operator+(const Tensor& other) const {
    // Check shapes match
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise addition");
    }

    // Check data types match
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise addition");
    }

    // Create result tensor
    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    // Perform element-wise addition based on data type
    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] + b[i];
            }
            break;
        }
    }

    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise subtraction");
    }
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise subtraction");
    }

    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] - b[i];
            }
            break;
        }
    }

    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise multiplication");
    }
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise multiplication");
    }

    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] * b[i];
            }
            break;
        }
    }

    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::runtime_error("Tensor shapes must match for element-wise division");
    }
    if (dtype_ != other.dtype_) {
        throw std::runtime_error("Tensor data types must match for element-wise division");
    }

    Tensor result(shape_, dtype_);
    size_t num_elements = NumElements();

    switch (dtype_) {
        case DataType::Float32: {
            const float* a = static_cast<const float*>(Data());
            const float* b = static_cast<const float*>(other.Data());
            float* r = static_cast<float*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::Float64: {
            const double* a = static_cast<const double*>(Data());
            const double* b = static_cast<const double*>(other.Data());
            double* r = static_cast<double*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::Int32: {
            const int32_t* a = static_cast<const int32_t*>(Data());
            const int32_t* b = static_cast<const int32_t*>(other.Data());
            int32_t* r = static_cast<int32_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::Int64: {
            const int64_t* a = static_cast<const int64_t*>(Data());
            const int64_t* b = static_cast<const int64_t*>(other.Data());
            int64_t* r = static_cast<int64_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::UInt8: {
            const uint8_t* a = static_cast<const uint8_t*>(Data());
            const uint8_t* b = static_cast<const uint8_t*>(other.Data());
            uint8_t* r = static_cast<uint8_t*>(result.Data());
            for (size_t i = 0; i < num_elements; i++) {
                r[i] = a[i] / b[i];
            }
            break;
        }
    }

    return result;
}

} // namespace cyxwiz
