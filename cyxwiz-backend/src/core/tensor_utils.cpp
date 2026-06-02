#include "tensor_utils.h"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace cyxwiz::tensor_utils {

bool SafeMultiply(size_t a, size_t b, size_t& result) {
    if (a != 0 && b > (std::numeric_limits<size_t>::max)() / a) {
        return false;
    }
    result = a * b;
    return true;
}

size_t ElementSize(DataType dtype) {
    switch (dtype) {
        case DataType::Float32: return 4;
        case DataType::Float64: return 8;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
    }
    throw std::runtime_error("Unsupported tensor data type");
}

int NormalizeDim(int dim, int rank, bool allow_end) {
    if (dim < 0) {
        dim += allow_end ? rank + 1 : rank;
    }
    const int upper = allow_end ? rank : rank - 1;
    if (dim < 0 || dim > upper) {
        throw std::runtime_error("Tensor dimension out of range");
    }
    return dim;
}

size_t CheckedProduct(const std::vector<size_t>& values,
                      size_t begin,
                      size_t end,
                      const char* overflow_message) {
    size_t product = 1;
    for (size_t i = begin; i < end; i++) {
        size_t next = 0;
        if (!SafeMultiply(product, values[i], next)) {
            throw std::overflow_error(overflow_message);
        }
        product = next;
    }
    return product;
}

std::vector<size_t> RowMajorStrides(const std::vector<size_t>& shape,
                                    const char* overflow_message,
                                    bool broadcast_zero_stride) {
    std::vector<size_t> strides(shape.size(), 1);
    size_t stride = 1;
    for (size_t i = shape.size(); i > 0; --i) {
        const size_t axis = i - 1;
        strides[axis] = broadcast_zero_stride && shape[axis] == 1 ? 0 : stride;

        size_t next = 0;
        if (!SafeMultiply(stride, shape[axis], next)) {
            throw std::overflow_error(overflow_message);
        }
        stride = next;
    }
    return strides;
}

void CopyElement(const Tensor& src_tensor, Tensor& dst_tensor, size_t src_linear, size_t dst_linear) {
    const size_t element_size = ElementSize(src_tensor.GetDataType());
    const auto* src = static_cast<const unsigned char*>(src_tensor.Data());
    auto* dst = static_cast<unsigned char*>(dst_tensor.Data());
    std::memcpy(dst + dst_linear * element_size, src + src_linear * element_size, element_size);
}

} // namespace cyxwiz::tensor_utils
