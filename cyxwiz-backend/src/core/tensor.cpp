#include "cyxwiz/tensor.h"
#include "cyxwiz/device.h"
#include <stdexcept>
#include <cstring>
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
    // TODO: Fill with ones
    return t;
}

Tensor Tensor::Random(const std::vector<size_t>& shape, DataType dtype) {
    Tensor t(shape, dtype);
    // TODO: Fill with random values
    return t;
}

Tensor Tensor::operator+(const Tensor& other) const {
    // TODO: Implement element-wise addition
    // For now, return a copy
    return Tensor(*this);
}

Tensor Tensor::operator-(const Tensor& other) const {
    // TODO: Implement element-wise subtraction
    // For now, return a copy
    return Tensor(*this);
}

Tensor Tensor::operator*(const Tensor& other) const {
    // TODO: Implement element-wise multiplication
    // For now, return a copy
    return Tensor(*this);
}

Tensor Tensor::operator/(const Tensor& other) const {
    // TODO: Implement element-wise division
    // For now, return a copy
    return Tensor(*this);
}

} // namespace cyxwiz
