#include "cyxwiz/tensor.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"
#endif

namespace cyxwiz {

namespace {

bool SafeAdd(size_t a, size_t b, size_t& result) {
    if (b > (std::numeric_limits<size_t>::max)() - a) {
        return false;
    }
    result = a + b;
    return true;
}

void ValidatePositiveSize(int value, const char* name) {
    if (value <= 0) {
        throw std::runtime_error(name);
    }
}

void ValidateSplitSectionSize(int value) {
    if (value < 0) {
        throw std::runtime_error(
            "Tensor::Split: sizes must be non-negative");
    }
}

void ValidateIndexableDimension(size_t value, const char* operation_name) {
    if (value > static_cast<size_t>((std::numeric_limits<int>::max)())) {
        throw std::overflow_error(
            std::string(operation_name) +
            ": dimension exceeds the supported indexing range");
    }
}

bool IsPyTorchEmptyCatIdentity(const Tensor& tensor) {
    return tensor.Shape().size() == 1 && tensor.Shape()[0] == 0;
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
// Concatenate along axis on the device. Up to rank 4 the semantic views join on
// that axis; above, each row-major flat array is viewed as [after, extent,
// before] and joined on ArrayFire dim 1, which is the output's row-major order.
Tensor JoinOnDevice(const std::vector<const Tensor*>& parts, int axis, const std::vector<size_t>& out_shape) {
    const size_t rank = out_shape.size();
    dim_t before = 1;
    dim_t after = 1;
    for (size_t i = 0; i < rank; ++i) {
        if (i < static_cast<size_t>(axis)) before *= static_cast<dim_t>(out_shape[i]);
        if (i > static_cast<size_t>(axis)) after *= static_cast<dim_t>(out_shape[i]);
    }
    af::array joined;
    for (const Tensor* part : parts) {
        const af::array piece = rank <= 4
            ? part->GetSemanticArray()
            : af::moddims(part->GetRowMajorFlatArray(), after,
                          static_cast<dim_t>(part->Shape()[static_cast<size_t>(axis)]), before);
        joined = joined.isempty() ? piece : af::join(rank <= 4 ? axis : 1, joined, piece);
    }
    joined.eval();
    return rank <= 4 ? Tensor::FromSemanticArray(joined, out_shape)
                     : Tensor::FromRowMajorFlatArray(af::flat(joined), out_shape);
}
#endif

} // namespace

Tensor Tensor::Cat(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Tensor::Cat: tensor list must not be empty");
    }

    const Tensor* reference = &tensors.front();
    for (const Tensor& tensor : tensors) {
        if (!IsPyTorchEmptyCatIdentity(tensor)) {
            reference = &tensor;
            break;
        }
    }
    const auto& ref_shape = reference->Shape();
    const int rank = static_cast<int>(ref_shape.size());
    const int axis = tensor_utils::NormalizeDim(dim, rank);
    const DataType dtype = tensors.front().GetDataType();

    std::vector<size_t> out_shape = ref_shape;
    out_shape[static_cast<size_t>(axis)] = 0;

    for (const Tensor& tensor : tensors) {
        if (tensor.GetDataType() != dtype) {
            throw std::runtime_error("Tensor::Cat: all tensors must have the same data type");
        }
        if (IsPyTorchEmptyCatIdentity(tensor)) {
            continue;
        }
        if (tensor.Shape().size() != ref_shape.size()) {
            throw std::runtime_error("Tensor::Cat: all tensors must have the same rank");
        }

        for (size_t i = 0; i < ref_shape.size(); i++) {
            if (i != static_cast<size_t>(axis) && tensor.Shape()[i] != ref_shape[i]) {
                throw std::runtime_error("Tensor::Cat: shapes must match except along concat dimension");
            }
        }

        size_t total = 0;
        if (!SafeAdd(out_shape[static_cast<size_t>(axis)],
                     tensor.Shape()[static_cast<size_t>(axis)],
                     total)) {
            throw std::overflow_error("Tensor::Cat: concat dimension overflow");
        }
        out_shape[static_cast<size_t>(axis)] = total;
    }

    if (tensor_utils::CheckedProduct(
            out_shape, 0, out_shape.size(),
            "Tensor::Cat: output shape overflow") == 0) {
        return Tensor(out_shape, dtype);
    }

    std::vector<const Tensor*> parts;
    for (const Tensor& tensor : tensors) {
        if (!IsPyTorchEmptyCatIdentity(tensor) && tensor.Shape()[static_cast<size_t>(axis)] > 0) {
            parts.push_back(&tensor);
        }
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return JoinOnDevice(parts, axis, out_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Cat", e));
    }
#else
    throw std::runtime_error("Tensor::Cat runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor Tensor::Stack(const std::vector<Tensor>& tensors, int dim) {
    if (tensors.empty()) {
        throw std::runtime_error("Tensor::Stack: tensor list must not be empty");
    }

    const auto& input_shape = tensors.front().Shape();
    const int axis = tensor_utils::NormalizeDim(
        dim, static_cast<int>(input_shape.size()), true);
    const DataType dtype = tensors.front().GetDataType();
    for (const Tensor& tensor : tensors) {
        if (tensor.GetDataType() != dtype) {
            throw std::runtime_error(
                "Tensor::Stack: all tensors must have the same data type");
        }
        if (tensor.Shape() != input_shape) {
            throw std::runtime_error(
                "Tensor::Stack: all tensors must have the same shape");
        }
    }

    std::vector<size_t> out_shape = input_shape;
    out_shape.insert(
        out_shape.begin() + static_cast<ptrdiff_t>(axis), tensors.size());
    if (tensor_utils::CheckedProduct(
            out_shape, 0, out_shape.size(),
            "Tensor::Stack: output shape overflow") == 0) {
        return Tensor(out_shape, dtype);
    }

    std::vector<Tensor> expanded;
    expanded.reserve(tensors.size());
    std::vector<const Tensor*> parts;
    for (const Tensor& tensor : tensors) expanded.push_back(tensor.Unsqueeze(axis));
    for (const Tensor& tensor : expanded) parts.push_back(&tensor);
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        return JoinOnDevice(parts, axis, out_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Stack", e));
    }
#else
    throw std::runtime_error("Tensor::Stack runs on ArrayFire, and this build has no ArrayFire");
#endif
}

std::vector<Tensor> Tensor::Split(int split_size, int dim) const {
    ValidatePositiveSize(
        split_size, "Tensor::Split: split_size must be positive");

    const int axis = tensor_utils::NormalizeDim(dim, static_cast<int>(shape_.size()));
    const size_t dim_size = shape_[static_cast<size_t>(axis)];
    ValidateIndexableDimension(dim_size, "Tensor::Split");
    std::vector<Tensor> result;

    if (dim_size == 0) {
        result.emplace_back(shape_, dtype_);
        return result;
    }

    for (size_t start = 0; start < dim_size; start += static_cast<size_t>(split_size)) {
        const size_t end = (std::min)(start + static_cast<size_t>(split_size), dim_size);
        result.push_back(Slice(axis, static_cast<int>(start), static_cast<int>(end)));
    }
    return result;
}

std::vector<Tensor> Tensor::Split(const std::vector<int>& sizes, int dim) const {
    const int axis = tensor_utils::NormalizeDim(dim, static_cast<int>(shape_.size()));
    const size_t dim_size = shape_[static_cast<size_t>(axis)];
    ValidateIndexableDimension(dim_size, "Tensor::Split");

    std::vector<Tensor> result;
    result.reserve(sizes.size());

    size_t start = 0;
    for (int size : sizes) {
        ValidateSplitSectionSize(size);
        size_t end = 0;
        if (!SafeAdd(start, static_cast<size_t>(size), end) || end > dim_size) {
            throw std::runtime_error("Tensor::Split: split sizes exceed dimension size");
        }
        result.push_back(Slice(axis, static_cast<int>(start), static_cast<int>(end)));
        start = end;
    }

    if (start != dim_size) {
        throw std::runtime_error("Tensor::Split: split sizes must cover the full dimension");
    }
    return result;
}

std::vector<Tensor> Tensor::Chunk(int chunks, int dim) const {
    ValidatePositiveSize(chunks, "Tensor::Chunk: chunks must be positive");

    const int axis = tensor_utils::NormalizeDim(dim, static_cast<int>(shape_.size()));
    const size_t dim_size = shape_[static_cast<size_t>(axis)];
    ValidateIndexableDimension(dim_size, "Tensor::Chunk");
    if (dim_size == 0) {
        return std::vector<Tensor>(
            static_cast<size_t>(chunks), Tensor(shape_, dtype_));
    }

    const size_t chunk_count = static_cast<size_t>(chunks);
    const size_t chunk_size = ((dim_size - 1) / chunk_count) + 1;
    return Split(static_cast<int>(chunk_size), axis);
}

} // namespace cyxwiz
