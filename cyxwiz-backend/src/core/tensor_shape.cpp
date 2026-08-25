#include "cyxwiz/tensor.h"
#include "tensor_backend_observation_utils.h"
#include "tensor_utils.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#include <spdlog/spdlog.h>
#endif

namespace cyxwiz {

namespace {

#ifdef CYXWIZ_HAS_ARRAYFIRE
void RecordShapeArrayFireFallback(const char* operation_name,
                                  const Tensor& input,
                                  const std::vector<size_t>& output_shape,
                                  const std::string& attributes,
                                  const char* error_message) {
    tensor_backend_observation::RecordArrayFireFallback(
        operation_name,
        tensor_backend_observation::DataTypeName(input.GetDataType()),
        tensor_backend_observation::BuildTensorOpSignature(
            {input.Shape()},
            output_shape,
            input.GetDataType(),
            attributes),
        error_message);
}

af::array RowMajorLinear(const Tensor& input) {
    if (input.Shape().size() == 2) {
        return af::flat(af::transpose(input.GetArrayRowMajor2D()));
    }
    if (input.Shape().size() == 3) {
        return af::flat(af::reorder(input.GetArrayRowMajor3D(), 2, 1, 0));
    }
    return af::flat(input.GetArray());
}

af::array SemanticArrayFromRowMajorLinear(
    const af::array& linear,
    const std::vector<size_t>& shape) {
    af::dim4 reversed_dims(1, 1, 1, 1);
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        reversed_dims[static_cast<unsigned int>(axis)] =
            static_cast<dim_t>(shape[shape.size() - axis - 1]);
    }
    af::array reversed = af::moddims(linear, reversed_dims);

    switch (shape.size()) {
        case 1: return reversed;
        case 2: return af::reorder(reversed, 1, 0);
        case 3: return af::reorder(reversed, 2, 1, 0);
        case 4: return af::reorder(reversed, 3, 2, 1, 0);
        default:
            throw std::runtime_error(
                "ArrayFire semantic permutation supports ranks 1 through 4");
    }
}

Tensor TensorFromPermutedSemantic(
    const af::array& semantic,
    const std::vector<size_t>& shape) {
    if (shape.size() <= 4) {
        return Tensor::FromSemanticArray(semantic, shape);
    }
    throw std::runtime_error(
        "ArrayFire semantic permutation supports ranks 1 through 4");
}

Tensor PermuteArrayFire(const Tensor& input,
                        const std::vector<int>& normalized_dims,
                        const std::vector<size_t>& output_shape) {
    af::array semantic = SemanticArrayFromRowMajorLinear(
        RowMajorLinear(input), input.Shape());
    std::array<unsigned int, 4> order = {0, 1, 2, 3};
    for (size_t axis = 0; axis < normalized_dims.size(); ++axis) {
        order[axis] = static_cast<unsigned int>(normalized_dims[axis]);
    }
    af::array permuted =
        af::reorder(semantic, order[0], order[1], order[2], order[3]);
    permuted.eval();
    return TensorFromPermutedSemantic(permuted, output_shape);
}

std::string PermutationAttributes(const std::vector<int>& dims) {
    std::string attributes = "op=permute;dims=";
    for (size_t index = 0; index < dims.size(); ++index) {
        if (index > 0) {
            attributes += ',';
        }
        attributes += std::to_string(dims[index]);
    }
    return attributes;
}
#endif

} // namespace

Tensor Tensor::View(const std::vector<size_t>& new_shape) const {
    return Reshape(new_shape);
}

Tensor Tensor::Squeeze() const {
    std::vector<size_t> new_shape;
    for (size_t size : shape_) {
        if (size != 1) {
            new_shape.push_back(size);
        }
    }
    return Reshape(new_shape);
}

Tensor Tensor::Squeeze(int dim) const {
    const int rank = static_cast<int>(shape_.size());
    if (rank == 0) {
        if (dim < -1 || dim > 0) {
            throw std::runtime_error("Tensor dimension out of range");
        }
        return Reshape(shape_);
    }

    const int normalized = tensor_utils::NormalizeDim(dim, rank);
    if (shape_[static_cast<size_t>(normalized)] != 1) {
        return Reshape(shape_);
    }

    std::vector<size_t> new_shape;
    new_shape.reserve(shape_.size() - 1);
    for (size_t i = 0; i < shape_.size(); i++) {
        if (i != static_cast<size_t>(normalized)) {
            new_shape.push_back(shape_[i]);
        }
    }

    return Reshape(new_shape);
}

Tensor Tensor::Unsqueeze(int dim) const {
    const int rank = static_cast<int>(shape_.size());
    const int normalized = tensor_utils::NormalizeDim(dim, rank, true);

    std::vector<size_t> new_shape;
    new_shape.reserve(shape_.size() + 1);
    for (int i = 0; i < rank; i++) {
        if (i == normalized) {
            new_shape.push_back(1);
        }
        new_shape.push_back(shape_[static_cast<size_t>(i)]);
    }
    if (normalized == rank) {
        new_shape.push_back(1);
    }

    return Reshape(new_shape);
}

Tensor Tensor::Flatten() const {
    return Reshape({NumElements()});
}

Tensor Tensor::Flatten(int start_dim, int end_dim) const {
    const int rank = static_cast<int>(shape_.size());
    if (rank == 0) {
        return Reshape({1});
    }

    const int start = tensor_utils::NormalizeDim(start_dim, rank);
    const int end = tensor_utils::NormalizeDim(end_dim, rank);
    if (start > end) {
        throw std::runtime_error("Tensor::Flatten: start_dim must be <= end_dim");
    }

    std::vector<size_t> new_shape;
    new_shape.reserve(shape_.size() - static_cast<size_t>(end - start));

    for (int i = 0; i < start; i++) {
        new_shape.push_back(shape_[static_cast<size_t>(i)]);
    }
    new_shape.push_back(tensor_utils::CheckedProduct(
        shape_,
        static_cast<size_t>(start),
        static_cast<size_t>(end + 1),
        "Tensor shape operation: integer overflow in dimension product"));
    for (int i = end + 1; i < rank; i++) {
        new_shape.push_back(shape_[static_cast<size_t>(i)]);
    }

    return Reshape(new_shape);
}

Tensor Tensor::Transpose(int dim0, int dim1) const {
    const int rank = static_cast<int>(shape_.size());
    const int first = tensor_utils::NormalizeDim(dim0, rank);
    const int second = tensor_utils::NormalizeDim(dim1, rank);

    std::vector<int> dims;
    dims.reserve(shape_.size());
    for (int axis = 0; axis < rank; ++axis) {
        dims.push_back(axis);
    }
    std::swap(dims[static_cast<size_t>(first)],
              dims[static_cast<size_t>(second)]);
    return Permute(dims);
}

Tensor Tensor::Permute(const std::vector<int>& dims) const {
    const int rank = static_cast<int>(shape_.size());
    if (dims.size() != shape_.size()) {
        throw std::runtime_error("Tensor::Permute: dims must match tensor rank");
    }

    std::vector<int> normalized_dims;
    normalized_dims.reserve(dims.size());
    std::vector<bool> seen(dims.size(), false);
    std::vector<size_t> new_shape;
    new_shape.reserve(dims.size());

    for (int dim : dims) {
        const int normalized = tensor_utils::NormalizeDim(dim, rank);
        const size_t axis = static_cast<size_t>(normalized);
        if (seen[axis]) {
            throw std::runtime_error("Tensor::Permute: dims must not contain duplicates");
        }
        seen[axis] = true;
        normalized_dims.push_back(normalized);
        new_shape.push_back(shape_[axis]);
    }

    if (NumElements() == 0) {
        return Tensor(new_shape, dtype_);
    }

    bool identity = true;
    for (size_t axis = 0; axis < normalized_dims.size(); ++axis) {
        if (normalized_dims[axis] != static_cast<int>(axis)) {
            identity = false;
            break;
        }
    }
    if (identity) {
        return Reshape(shape_);
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    if (shape_.size() <= 4) {
        try {
            return PermuteArrayFire(*this, normalized_dims, new_shape);
        } catch (const af::exception& e) {
            RecordShapeArrayFireFallback(
                "Tensor::Permute",
                *this,
                new_shape,
                PermutationAttributes(normalized_dims),
                e.what());
        }
    } else {
        RecordShapeArrayFireFallback(
            "Tensor::Permute",
            *this,
            new_shape,
            PermutationAttributes(normalized_dims),
            "ArrayFire permutation supports ranks up to 4");
    }
#endif

    Tensor result(new_shape, dtype_);
    const size_t element_size = tensor_utils::ElementSize(dtype_);
    const auto src_strides = tensor_utils::RowMajorStrides(shape_, "Tensor stride calculation overflow");
    const auto dst_strides = tensor_utils::RowMajorStrides(new_shape, "Tensor stride calculation overflow");

    const auto* src = static_cast<const unsigned char*>(ReadData());
    auto* dst = static_cast<unsigned char*>(result.MutableData());

    const size_t total = NumElements();
    std::vector<size_t> src_index(shape_.size(), 0);
    std::vector<size_t> dst_index(shape_.size(), 0);

    for (size_t linear = 0; linear < total; linear++) {
        size_t rem = linear;
        for (size_t axis = 0; axis < shape_.size(); axis++) {
            src_index[axis] = rem / src_strides[axis];
            rem %= src_strides[axis];
        }

        for (size_t axis = 0; axis < normalized_dims.size(); axis++) {
            dst_index[axis] = src_index[static_cast<size_t>(normalized_dims[axis])];
        }

        size_t dst_linear = 0;
        for (size_t axis = 0; axis < dst_index.size(); axis++) {
            dst_linear += dst_index[axis] * dst_strides[axis];
        }

        std::memcpy(dst + dst_linear * element_size, src + linear * element_size, element_size);
    }

    return result;
}

} // namespace cyxwiz
