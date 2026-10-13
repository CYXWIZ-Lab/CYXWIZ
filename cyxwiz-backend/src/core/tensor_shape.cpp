#include "cyxwiz/tensor.h"
#include "tensor_math_utils.h"
#include "tensor_utils.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

#include <arrayfire.h>
#include "../algorithms/arrayfire_host_values.h"

namespace cyxwiz {

namespace {

// One af::reorder of a rank <= 4 semantic array.
af::array ReorderSemantic(const af::array& semantic, const std::vector<int>& order) {
    std::array<unsigned int, 4> axes = {0, 1, 2, 3};
    for (size_t axis = 0; axis < order.size(); ++axis) axes[axis] = static_cast<unsigned int>(order[axis]);
    return af::reorder(semantic, axes[0], axes[1], axes[2], axes[3]);
}

// General permutation of row-major flat storage: each output position's
// source index is computed on the device (div / mod per output axis) and
// gathered.
af::array PermuteGatherFlat(const af::array& flat, const std::vector<size_t>& shape,
                            const std::vector<int>& dims, const std::vector<size_t>& new_shape) {
    if (static_cast<unsigned long long>(flat.elements()) > (std::numeric_limits<unsigned>::max)()) {
        throw std::runtime_error("Tensor::Permute: input exceeds the ArrayFire index range");
    }
    std::vector<double> source_strides(shape.size(), 1.0);
    for (size_t axis = shape.size() - 1; axis-- > 0;) {
        source_strides[axis] = source_strides[axis + 1] * static_cast<double>(shape[axis + 1]);
    }
    af::array remaining = af::range(af::dim4(flat.elements()), 0, s64);
    af::array source = af::constant(0, af::dim4(flat.elements()), s64);
    // Extents and strides stay far below 2^53, so double scalars are exact.
    for (size_t axis = new_shape.size(); axis-- > 0;) {
        const double extent = static_cast<double>(new_shape[axis]);
        const af::array coordinate = af::mod(remaining, extent);
        remaining = (remaining - coordinate) / extent;
        source = source + coordinate * source_strides[static_cast<size_t>(dims[axis])];
    }
    return flat(source.as(u32));
}

// Permute on the device. Up to rank 4: reorder the semantic view. Above:
// consecutive output axes that read consecutive source axes merge into one
// run; with at most four runs it is one reorder of the collapsed view,
// otherwise a gather.
Tensor PermuteArrayFire(const Tensor& input,
                        const std::vector<int>& dims,
                        const std::vector<size_t>& new_shape) {
    const std::vector<size_t>& shape = input.Shape();
    if (shape.size() <= 4) {
        af::array permuted = ReorderSemantic(input.GetSemanticArray(), dims);
        permuted.eval();
        return Tensor::FromSemanticArray(permuted, new_shape);
    }
    struct Run { int source_start; size_t extent; };
    std::vector<Run> runs;  // in output order
    for (size_t axis = 0; axis < dims.size(); ++axis) {
        if (!runs.empty() && dims[axis] == dims[axis - 1] + 1) {
            runs.back().extent *= new_shape[axis];
        } else {
            runs.push_back({dims[axis], new_shape[axis]});
        }
    }
    const af::array flat = input.GetRowMajorFlatArray();
    af::array output;
    if (runs.size() <= 4) {
        std::vector<size_t> source_order(runs.size());
        for (size_t i = 0; i < runs.size(); ++i) source_order[i] = i;
        std::sort(source_order.begin(), source_order.end(),
                  [&runs](size_t a, size_t b) { return runs[a].source_start < runs[b].source_start; });
        std::vector<size_t> collapsed_source(runs.size());
        std::vector<int> collapsed_order(runs.size());
        for (size_t position = 0; position < source_order.size(); ++position) {
            collapsed_source[position] = runs[source_order[position]].extent;
            collapsed_order[source_order[position]] = static_cast<int>(position);
        }
        const af::array permuted = ReorderSemantic(
            tensor_math_utils::RowMajorFlatToSemantic(flat, collapsed_source), collapsed_order);
        output = tensor_math_utils::SemanticToRowMajorFlat(permuted, runs.size());
    } else {
        output = PermuteGatherFlat(flat, shape, dims, new_shape);
    }
    output.eval();
    return Tensor::FromRowMajorFlatArray(output, new_shape);
}

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

    try {
        return PermuteArrayFire(*this, normalized_dims, new_shape);
    } catch (const af::exception& e) {
        throw std::runtime_error(DeviceError("Tensor::Permute", e));
    }
}

} // namespace cyxwiz
