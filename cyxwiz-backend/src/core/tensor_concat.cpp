#include "cyxwiz/tensor.h"
#include "tensor_backend_observation_utils.h"
#include "tensor_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
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
std::vector<std::vector<size_t>> BuildInputShapes(
    const std::vector<Tensor>& tensors) {
    std::vector<std::vector<size_t>> input_shapes;
    input_shapes.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        input_shapes.push_back(tensor.Shape());
    }
    return input_shapes;
}

void RecordConcatArrayFireFallback(
    const char* operation_name,
    const std::vector<Tensor>& tensors,
    DataType dtype,
    const std::vector<size_t>& output_shape,
    int axis,
    const char* error_message) {
    spdlog::warn(
        "{}",
        tensor_backend_observation::RecordArrayFireFallback(
            operation_name,
            tensor_backend_observation::DataTypeName(dtype),
            tensor_backend_observation::BuildTensorOpSignature(
                BuildInputShapes(tensors),
                output_shape,
                dtype,
                "dim=" + std::to_string(axis)),
            error_message));
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (out_shape.size() <= 4) {
        try {
            const Tensor* first = nullptr;
            for (const Tensor& tensor : tensors) {
                if (!IsPyTorchEmptyCatIdentity(tensor) &&
                    tensor.Shape()[static_cast<size_t>(axis)] > 0) {
                    first = &tensor;
                    break;
                }
            }
            if (!first) {
                return Tensor(out_shape, dtype);
            }

            af::array joined = first->GetSemanticArray();
            bool passed_first = false;
            for (const Tensor& tensor : tensors) {
                if (&tensor == first) {
                    passed_first = true;
                    continue;
                }
                if (!passed_first || IsPyTorchEmptyCatIdentity(tensor) ||
                    tensor.Shape()[static_cast<size_t>(axis)] == 0) {
                    continue;
                }
                joined = af::join(
                    static_cast<unsigned>(axis),
                    joined,
                    tensor.GetSemanticArray());
            }
            joined.eval();
            return Tensor::FromSemanticArray(joined, out_shape);
        } catch (const af::exception& e) {
            RecordConcatArrayFireFallback(
                "Tensor::Cat", tensors, dtype, out_shape, axis, e.what());
        }
    } else {
        RecordConcatArrayFireFallback(
            "Tensor::Cat",
            tensors,
            dtype,
            out_shape,
            axis,
            "ArrayFire concatenation supports ranks up to 4");
    }
#endif

    Tensor result(out_shape, dtype);
    const auto dst_strides = tensor_utils::RowMajorStrides(out_shape, "Tensor concat: stride overflow");
    const size_t element_size = tensor_utils::ElementSize(dtype);
    auto* dst = static_cast<unsigned char*>(result.MutableData());

    size_t axis_offset = 0;
    for (const Tensor& tensor : tensors) {
        if (IsPyTorchEmptyCatIdentity(tensor)) {
            continue;
        }
        const auto src_strides = tensor_utils::RowMajorStrides(tensor.Shape(), "Tensor concat: stride overflow");
        const auto* src = static_cast<const unsigned char*>(tensor.ReadData());
        std::vector<size_t> index(tensor.Shape().size(), 0);

        for (size_t src_linear = 0; src_linear < tensor.NumElements(); src_linear++) {
            size_t remaining = src_linear;
            for (size_t i = 0; i < tensor.Shape().size(); i++) {
                index[i] = remaining / src_strides[i];
                remaining %= src_strides[i];
            }

            size_t dst_linear = 0;
            for (size_t i = 0; i < index.size(); i++) {
                const size_t coord = i == static_cast<size_t>(axis) ? index[i] + axis_offset : index[i];
                dst_linear += coord * dst_strides[i];
            }

            std::memcpy(dst + dst_linear * element_size,
                        src + src_linear * element_size,
                        element_size);
        }

        axis_offset += tensor.Shape()[static_cast<size_t>(axis)];
    }

    return result;
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

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (out_shape.size() <= 4) {
        try {
            Tensor expanded = tensors.front().Unsqueeze(axis);
            af::array joined = expanded.GetSemanticArray();
            for (size_t index = 1; index < tensors.size(); ++index) {
                Tensor next = tensors[index].Unsqueeze(axis);
                joined = af::join(
                    static_cast<unsigned>(axis),
                    joined,
                    next.GetSemanticArray());
            }
            joined.eval();
            return Tensor::FromSemanticArray(joined, out_shape);
        } catch (const af::exception& e) {
            RecordConcatArrayFireFallback(
                "Tensor::Stack", tensors, dtype, out_shape, axis, e.what());
        }
    } else {
        RecordConcatArrayFireFallback(
            "Tensor::Stack",
            tensors,
            dtype,
            out_shape,
            axis,
            "ArrayFire stacking supports output ranks up to 4");
    }
#endif

    Tensor result(out_shape, dtype);
    const auto input_strides = tensor_utils::RowMajorStrides(
        input_shape, "Tensor stack: input stride overflow");
    const auto output_strides = tensor_utils::RowMajorStrides(
        out_shape, "Tensor stack: output stride overflow");
    const size_t element_size = tensor_utils::ElementSize(dtype);
    auto* dst = static_cast<unsigned char*>(result.MutableData());
    std::vector<size_t> coordinate(input_shape.size(), 0);

    for (size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
        const auto* src = static_cast<const unsigned char*>(
            tensors[tensor_index].ReadData());
        for (size_t src_linear = 0;
             src_linear < tensors[tensor_index].NumElements(); ++src_linear) {
            size_t remainder = src_linear;
            for (size_t input_axis = 0;
                 input_axis < input_shape.size(); ++input_axis) {
                coordinate[input_axis] = remainder / input_strides[input_axis];
                remainder %= input_strides[input_axis];
            }

            size_t dst_linear = tensor_index * output_strides[axis];
            for (size_t output_axis = 0, input_axis = 0;
                 output_axis < out_shape.size(); ++output_axis) {
                if (output_axis != static_cast<size_t>(axis)) {
                    dst_linear += coordinate[input_axis++] *
                                  output_strides[output_axis];
                }
            }
            std::memcpy(dst + dst_linear * element_size,
                        src + src_linear * element_size,
                        element_size);
        }
    }
    return result;
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
