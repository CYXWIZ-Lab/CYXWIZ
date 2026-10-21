#include <cyxwiz/sequential.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

namespace {

std::vector<size_t> UnravelIndex(size_t index, const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size(), 0);
    for (size_t i = shape.size(); i-- > 0;) {
        indices[i] = index % shape[i];
        index /= shape[i];
    }
    return indices;
}

size_t RavelIndex(const std::vector<size_t>& indices,
                  const std::vector<size_t>& shape) {
    size_t linear = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        linear = linear * shape[i] + indices[i];
    }
    return linear;
}

} // namespace
// ============================================================================
// ReshapeModule Implementation
// ============================================================================

ReshapeModule::ReshapeModule(std::vector<size_t> target_sample_shape)
    : target_sample_shape_(std::move(target_sample_shape)) {
    if (target_sample_shape_.empty()) {
        throw std::runtime_error("ReshapeModule: target sample shape must not be empty");
    }
}

Tensor ReshapeModule::Forward(const Tensor& input) {
    original_shape_ = input.Shape();
    if (original_shape_.empty()) {
        throw std::runtime_error("ReshapeModule: input must include a batch dimension");
    }

    std::vector<size_t> target_shape;
    target_shape.reserve(target_sample_shape_.size() + 1);
    target_shape.push_back(original_shape_[0]);
    target_shape.insert(target_shape.end(),
                        target_sample_shape_.begin(),
                        target_sample_shape_.end());

    return input.Reshape(target_shape);
}

Tensor ReshapeModule::Backward(const Tensor& grad_output) {
    return grad_output.Reshape(original_shape_);
}

// ============================================================================
// PermuteModule Implementation
// ============================================================================

PermuteModule::PermuteModule(std::vector<int> sample_dims)
    : sample_dims_(std::move(sample_dims)) {
    if (sample_dims_.empty()) {
        throw std::runtime_error("PermuteModule: sample dims must not be empty");
    }

    inverse_sample_dims_.resize(sample_dims_.size());
    for (size_t i = 0; i < sample_dims_.size(); ++i) {
        const int dim = sample_dims_[i];
        if (dim < 0 || dim >= static_cast<int>(sample_dims_.size())) {
            throw std::runtime_error("PermuteModule: sample dims must be normalized");
        }
        inverse_sample_dims_[static_cast<size_t>(dim)] = static_cast<int>(i);
    }
}

Tensor PermuteModule::Forward(const Tensor& input) {
    if (input.Shape().size() != sample_dims_.size() + 1) {
        throw std::runtime_error("PermuteModule: input rank does not match sample dims");
    }

    std::vector<int> full_dims;
    full_dims.reserve(sample_dims_.size() + 1);
    full_dims.push_back(0);
    for (int dim : sample_dims_) {
        full_dims.push_back(dim + 1);
    }
    return input.Permute(full_dims);
}

Tensor PermuteModule::Backward(const Tensor& grad_output) {
    std::vector<int> full_inverse_dims;
    full_inverse_dims.reserve(inverse_sample_dims_.size() + 1);
    full_inverse_dims.push_back(0);
    for (int dim : inverse_sample_dims_) {
        full_inverse_dims.push_back(dim + 1);
    }
    return grad_output.Permute(full_inverse_dims);
}

// ============================================================================
// TensorUnaryModule Implementation
// ============================================================================

TensorUnaryModule::TensorUnaryModule(TensorUnaryOp op,
                                     float scalar,
                                     float scalar2)
    : op_(op),
      scalar_(scalar),
      scalar2_(scalar2) {}

Tensor TensorUnaryModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    Tensor output;
    switch (op_) {
        case TensorUnaryOp::Abs:
            output = input.Abs();
            break;
        case TensorUnaryOp::Exp:
            output = input.Exp();
            break;
        case TensorUnaryOp::Log:
            output = input.Log();
            break;
        case TensorUnaryOp::Sqrt:
            output = input.Sqrt();
            break;
        case TensorUnaryOp::Sign:
            output = input.Sign();
            break;
        case TensorUnaryOp::Pow:
            output = input.Pow(scalar_);
            break;
        case TensorUnaryOp::Clip:
            output = input.Clip(scalar_, scalar2_);
            break;
        default:
            throw std::runtime_error("TensorUnaryModule: unsupported unary op");
    }

    output_cache_ = output.Clone();
    return output;
}

Tensor TensorUnaryModule::Backward(const Tensor& grad_output) {
    switch (op_) {
        case TensorUnaryOp::Abs:
            return grad_output * input_cache_.Sign();
        case TensorUnaryOp::Exp:
            return grad_output * output_cache_;
        case TensorUnaryOp::Log:
            return grad_output / input_cache_;
        case TensorUnaryOp::Sqrt:
            return grad_output / (output_cache_ * 2.0f);
        case TensorUnaryOp::Sign:
            return Tensor::Zeros(grad_output.Shape(), grad_output.GetDataType());
        case TensorUnaryOp::Pow:
            if (scalar_ == 0.0f) {
                return Tensor::Zeros(grad_output.Shape(), grad_output.GetDataType());
            }
            return grad_output * (input_cache_.Pow(scalar_ - 1.0f) * scalar_);
        case TensorUnaryOp::Clip: {
            Tensor mask = Tensor::Zeros(grad_output.Shape(), grad_output.GetDataType());
            for (size_t i = 0; i < input_cache_.NumElements(); ++i) {
                const float value = input_cache_.At(i);
                if (value >= scalar_ && value <= scalar2_) {
                    mask.Set(i, 1.0f);
                }
            }
            return grad_output * mask;
        }
        default:
            throw std::runtime_error("TensorUnaryModule: unsupported unary op");
    }
}

std::string TensorUnaryModule::GetName() const {
    switch (op_) {
        case TensorUnaryOp::Abs:
            return "TensorAbs";
        case TensorUnaryOp::Exp:
            return "TensorExp";
        case TensorUnaryOp::Log:
            return "TensorLog";
        case TensorUnaryOp::Sqrt:
            return "TensorSqrt";
        case TensorUnaryOp::Sign:
            return "TensorSign";
        case TensorUnaryOp::Pow:
            return "TensorPow";
        case TensorUnaryOp::Clip:
            return "TensorClip";
        default:
            return "TensorUnary";
    }
}

// ============================================================================
// TensorReductionModule Implementation
// ============================================================================

TensorReductionModule::TensorReductionModule(TensorReductionOp op,
                                             int dim,
                                             bool keepdim)
    : op_(op),
      dim_(dim),
      keepdim_(keepdim) {}

Tensor TensorReductionModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    original_shape_ = input.Shape();
    if (original_shape_.empty()) {
        throw std::runtime_error("TensorReductionModule: input must include a batch dimension");
    }

    const size_t sample_rank = original_shape_.size() - 1;
    if (sample_rank == 0) {
        throw std::runtime_error("TensorReductionModule: input must include sample dimensions");
    }

    Tensor output = input.Clone();
    reduced_count_ = 1;

    if (dim_ == -1) {
        for (size_t i = 1; i < original_shape_.size(); ++i) {
            reduced_count_ *= original_shape_[i];
        }
        if (reduced_count_ == 0) {
            throw std::runtime_error("TensorReductionModule: cannot reduce empty input");
        }
        if (op_ == TensorReductionOp::Var || op_ == TensorReductionOp::Std) {
            std::vector<size_t> reduced_shape = keepdim_
                ? std::vector<size_t>(original_shape_.size(), 1)
                : std::vector<size_t>{original_shape_[0], 1};
            reduced_shape[0] = original_shape_[0];
            const DataType out_dtype = input.GetDataType() == DataType::Float64
                ? DataType::Float64
                : DataType::Float32;
            output = Tensor(reduced_shape, out_dtype);
            for (size_t batch = 0; batch < original_shape_[0]; ++batch) {
                const size_t offset = batch * reduced_count_;
                double total = 0.0;
                for (size_t i = 0; i < reduced_count_; ++i) {
                    total += static_cast<double>(input.At(offset + i));
                }
                const double mean = total / static_cast<double>(reduced_count_);
                double variance = 0.0;
                for (size_t i = 0; i < reduced_count_; ++i) {
                    const double diff = static_cast<double>(input.At(offset + i)) - mean;
                    variance += diff * diff;
                }
                variance /= static_cast<double>(reduced_count_);
                const double value = op_ == TensorReductionOp::Std
                    ? std::sqrt(variance)
                    : variance;
                output.Set(batch, static_cast<float>(value));
            }
            output_shape_ = output.Shape();
            output_cache_ = output.Clone();
            return output;
        }
        for (int axis = static_cast<int>(sample_rank); axis >= 1; --axis) {
            switch (op_) {
                case TensorReductionOp::Sum:
                case TensorReductionOp::Mean:
                    output = output.Sum(axis, true);
                    break;
                case TensorReductionOp::Max:
                    output = output.Max(axis, true);
                    break;
                case TensorReductionOp::Min:
                    output = output.Min(axis, true);
                    break;
                case TensorReductionOp::Prod:
                    output = output.Prod(axis, true);
                    break;
                default:
                    throw std::runtime_error("TensorReductionModule: unsupported reduction op");
            }
        }
        if (op_ == TensorReductionOp::Mean) {
            output = output / static_cast<float>(reduced_count_);
        }
        if (!keepdim_) {
            output = output.Reshape({original_shape_[0], 1});
        }
        output_shape_ = output.Shape();
        output_cache_ = output.Clone();
        return output;
    }

    if (dim_ < 0 || dim_ >= static_cast<int>(sample_rank)) {
        throw std::runtime_error("TensorReductionModule: dim is out of range");
    }

    const int full_dim = dim_ + 1;
    reduced_count_ = original_shape_[static_cast<size_t>(full_dim)];
    switch (op_) {
        case TensorReductionOp::Sum:
            output = input.Sum(full_dim, keepdim_);
            break;
        case TensorReductionOp::Mean:
            output = input.Mean(full_dim, keepdim_);
            break;
        case TensorReductionOp::Max:
            output = input.Max(full_dim, keepdim_);
            break;
        case TensorReductionOp::Min:
            output = input.Min(full_dim, keepdim_);
            break;
        case TensorReductionOp::Prod:
            output = input.Prod(full_dim, keepdim_);
            break;
        case TensorReductionOp::Var:
            output = input.Var(full_dim, keepdim_);
            break;
        case TensorReductionOp::Std:
            output = input.Std(full_dim, keepdim_);
            break;
        default:
            throw std::runtime_error("TensorReductionModule: unsupported reduction op");
    }
    if (!keepdim_ && output.Shape().size() == 1) {
        output = output.Reshape({original_shape_[0], 1});
    }

    output_shape_ = output.Shape();
    output_cache_ = output.Clone();
    return output;
}

Tensor TensorReductionModule::Backward(const Tensor& grad_output) {
    Tensor grad_input = Tensor::Zeros(original_shape_, grad_output.GetDataType());
    const int full_dim = dim_ >= 0 ? dim_ + 1 : -1;
    const float scale = op_ == TensorReductionOp::Mean
        ? 1.0f / static_cast<float>(reduced_count_)
        : 1.0f;

    auto grad_index_for_input = [&](const std::vector<size_t>& input_indices) {
        std::vector<size_t> grad_indices;

        if (dim_ == -1) {
            if (keepdim_) {
                grad_indices.assign(original_shape_.size(), 0);
                grad_indices[0] = input_indices[0];
            } else {
                grad_indices = {input_indices[0], 0};
            }
        } else {
            grad_indices.reserve(output_shape_.size());
            for (size_t axis = 0; axis < original_shape_.size(); ++axis) {
                if (static_cast<int>(axis) == full_dim) {
                    if (keepdim_) {
                        grad_indices.push_back(0);
                    }
                } else {
                    grad_indices.push_back(input_indices[axis]);
                }
            }
            if (!keepdim_ && grad_indices.size() == 1) {
                grad_indices.push_back(0);
            }
        }

        return RavelIndex(grad_indices, output_shape_);
    };

    std::vector<size_t> tie_counts(output_cache_.NumElements(), 0);
    std::vector<double> group_sums(output_cache_.NumElements(), 0.0);
    std::vector<double> group_nonzero_products(output_cache_.NumElements(), 1.0);
    std::vector<size_t> group_zero_counts(output_cache_.NumElements(), 0);
    if (op_ == TensorReductionOp::Max || op_ == TensorReductionOp::Min) {
        for (size_t i = 0; i < original_shape_.size(); ++i) {
            if (original_shape_[i] == 0) {
                throw std::runtime_error("TensorReductionModule: cannot reduce empty input");
            }
        }
        for (size_t i = 0; i < input_cache_.NumElements(); ++i) {
            const std::vector<size_t> input_indices = UnravelIndex(i, original_shape_);
            const size_t grad_index = grad_index_for_input(input_indices);
            if (input_cache_.At(i) == output_cache_.At(grad_index)) {
                tie_counts[grad_index] += 1;
            }
        }
    } else if (op_ == TensorReductionOp::Prod ||
               op_ == TensorReductionOp::Var ||
               op_ == TensorReductionOp::Std) {
        for (size_t i = 0; i < input_cache_.NumElements(); ++i) {
            const std::vector<size_t> input_indices = UnravelIndex(i, original_shape_);
            const size_t grad_index = grad_index_for_input(input_indices);
            const float input_value = input_cache_.At(i);
            if (op_ == TensorReductionOp::Prod) {
                if (input_value == 0.0f) {
                    group_zero_counts[grad_index] += 1;
                } else {
                    group_nonzero_products[grad_index] *= static_cast<double>(input_value);
                }
            } else {
                group_sums[grad_index] += static_cast<double>(input_value);
            }
        }
    }

    for (size_t i = 0; i < grad_input.NumElements(); ++i) {
        const std::vector<size_t> input_indices = UnravelIndex(i, original_shape_);
        const size_t grad_index = grad_index_for_input(input_indices);
        float value = grad_output.At(grad_index) * scale;
        if (op_ == TensorReductionOp::Max || op_ == TensorReductionOp::Min) {
            if (input_cache_.At(i) != output_cache_.At(grad_index)) {
                value = 0.0f;
            } else {
                value /= static_cast<float>(tie_counts[grad_index]);
            }
        } else if (op_ == TensorReductionOp::Prod) {
            const float input_value = input_cache_.At(i);
            double derivative = 0.0;
            if (group_zero_counts[grad_index] == 0) {
                derivative = group_nonzero_products[grad_index] /
                    static_cast<double>(input_value);
            } else if (group_zero_counts[grad_index] == 1 && input_value == 0.0f) {
                derivative = group_nonzero_products[grad_index];
            }
            value = grad_output.At(grad_index) * static_cast<float>(derivative);
        } else if (op_ == TensorReductionOp::Var ||
                   op_ == TensorReductionOp::Std) {
            const double mean = group_sums[grad_index] /
                static_cast<double>(reduced_count_);
            const double centered = static_cast<double>(input_cache_.At(i)) - mean;
            double derivative = 2.0 * centered / static_cast<double>(reduced_count_);
            if (op_ == TensorReductionOp::Std) {
                const double std_value = static_cast<double>(output_cache_.At(grad_index));
                derivative = std_value == 0.0
                    ? 0.0
                    : centered / (static_cast<double>(reduced_count_) * std_value);
            }
            value = grad_output.At(grad_index) * static_cast<float>(derivative);
        }
        grad_input.Set(i, value);
    }

    return grad_input;
}

std::string TensorReductionModule::GetName() const {
    switch (op_) {
        case TensorReductionOp::Sum:
            return "TensorSum";
        case TensorReductionOp::Mean:
            return "TensorMean";
        case TensorReductionOp::Max:
            return "TensorMax";
        case TensorReductionOp::Min:
            return "TensorMin";
        case TensorReductionOp::Prod:
            return "TensorProd";
        case TensorReductionOp::Var:
            return "TensorVar";
        case TensorReductionOp::Std:
            return "TensorStd";
        default:
            return "TensorReduction";
    }
}

// ============================================================================
// TensorShapeModule Implementation
// ============================================================================

TensorShapeModule::TensorShapeModule(TensorShapeOp op,
                                     std::vector<size_t> target_shape,
                                     int dim,
                                     std::vector<int> indices)
    : op_(op),
      target_shape_(std::move(target_shape)),
      dim_(dim),
      indices_(std::move(indices)) {}

Tensor TensorShapeModule::Forward(const Tensor& input) {
    original_shape_ = input.Shape();
    if (original_shape_.empty()) {
        throw std::runtime_error("TensorShapeModule: input must include a batch dimension");
    }

    const size_t sample_rank = original_shape_.size() - 1;
    if (op_ == TensorShapeOp::BroadcastTo || op_ == TensorShapeOp::Expand) {
        if (target_shape_.size() < sample_rank) {
            throw std::runtime_error("TensorShapeModule: target sample rank is too small");
        }

        sample_pad_ = target_shape_.size() - sample_rank;
        padded_input_shape_.clear();
        padded_input_shape_.reserve(target_shape_.size() + 1);
        padded_input_shape_.push_back(original_shape_[0]);
        for (size_t i = 0; i < sample_pad_; ++i) {
            padded_input_shape_.push_back(1);
        }
        for (size_t i = 1; i < original_shape_.size(); ++i) {
            padded_input_shape_.push_back(original_shape_[i]);
        }

        output_shape_.clear();
        output_shape_.reserve(target_shape_.size() + 1);
        output_shape_.push_back(original_shape_[0]);
        output_shape_.insert(output_shape_.end(), target_shape_.begin(), target_shape_.end());

        for (size_t axis = 0; axis < output_shape_.size(); ++axis) {
            const size_t in_dim = padded_input_shape_[axis];
            const size_t out_dim = output_shape_[axis];
            if (in_dim != 1 && in_dim != out_dim) {
                throw std::runtime_error("TensorShapeModule: incompatible target shape");
            }
        }

        Tensor reshaped = padded_input_shape_ == original_shape_
            ? input.Clone()
            : input.Reshape(padded_input_shape_);
        return op_ == TensorShapeOp::BroadcastTo
            ? reshaped.BroadcastTo(output_shape_)
            : reshaped.Expand(output_shape_);
    }

    if (op_ == TensorShapeOp::IndexSelect) {
        if (sample_rank == 0) {
            throw std::runtime_error("TensorShapeModule: input must include sample dimensions");
        }
        if (indices_.empty()) {
            throw std::runtime_error("TensorShapeModule: indices must not be empty");
        }
        int normalized_dim = dim_;
        if (normalized_dim < 0) {
            normalized_dim += static_cast<int>(sample_rank);
        }
        if (normalized_dim < 0 || normalized_dim >= static_cast<int>(sample_rank)) {
            throw std::runtime_error("TensorShapeModule: dim is out of range");
        }

        normalized_dim_ = normalized_dim;
        const int full_dim = normalized_dim_ + 1;
        const int dim_size = static_cast<int>(original_shape_[static_cast<size_t>(full_dim)]);
        normalized_indices_.clear();
        normalized_indices_.reserve(indices_.size());
        for (int index : indices_) {
            int normalized = index;
            if (normalized < 0) {
                normalized += dim_size;
            }
            if (normalized < 0 || normalized >= dim_size) {
                throw std::out_of_range("TensorShapeModule: selected index out of range");
            }
            normalized_indices_.push_back(normalized);
        }

        output_shape_ = original_shape_;
        output_shape_[static_cast<size_t>(full_dim)] = normalized_indices_.size();
        return input.IndexSelect(full_dim, indices_);
    }

    throw std::runtime_error("TensorShapeModule: unsupported shape op");
}

Tensor TensorShapeModule::Backward(const Tensor& grad_output) {
    Tensor grad_input = Tensor::Zeros(original_shape_, grad_output.GetDataType());

    if (op_ == TensorShapeOp::BroadcastTo || op_ == TensorShapeOp::Expand) {
        for (size_t i = 0; i < grad_output.NumElements(); ++i) {
            const std::vector<size_t> out_indices = UnravelIndex(i, output_shape_);
            std::vector<size_t> padded_indices(padded_input_shape_.size(), 0);
            for (size_t axis = 0; axis < output_shape_.size(); ++axis) {
                padded_indices[axis] = padded_input_shape_[axis] == 1 ? 0 : out_indices[axis];
            }

            std::vector<size_t> input_indices;
            input_indices.reserve(original_shape_.size());
            input_indices.push_back(padded_indices[0]);
            for (size_t axis = 1 + sample_pad_; axis < padded_indices.size(); ++axis) {
                input_indices.push_back(padded_indices[axis]);
            }

            const size_t input_index = RavelIndex(input_indices, original_shape_);
            grad_input.Set(input_index, grad_input.At(input_index) + grad_output.At(i));
        }
        return grad_input;
    }

    if (op_ == TensorShapeOp::IndexSelect) {
        const int full_dim = normalized_dim_ + 1;
        for (size_t i = 0; i < grad_output.NumElements(); ++i) {
            std::vector<size_t> out_indices = UnravelIndex(i, output_shape_);
            std::vector<size_t> input_indices = out_indices;
            input_indices[static_cast<size_t>(full_dim)] =
                static_cast<size_t>(normalized_indices_[out_indices[static_cast<size_t>(full_dim)]]);

            const size_t input_index = RavelIndex(input_indices, original_shape_);
            grad_input.Set(input_index, grad_input.At(input_index) + grad_output.At(i));
        }
        return grad_input;
    }

    throw std::runtime_error("TensorShapeModule: unsupported shape op");
}

std::string TensorShapeModule::GetName() const {
    switch (op_) {
        case TensorShapeOp::BroadcastTo:
            return "TensorBroadcastTo";
        case TensorShapeOp::Expand:
            return "TensorExpand";
        case TensorShapeOp::IndexSelect:
            return "TensorIndexSelect";
        default:
            return "TensorShape";
    }
}

// ============================================================================
// TensorMaskModule Implementation
// ============================================================================

TensorMaskModule::TensorMaskModule(TensorMaskOp op, float scalar)
    : op_(op),
      scalar_(scalar) {}

Tensor TensorMaskModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();
    Tensor output(input.Shape(), input.GetDataType());

    for (size_t i = 0; i < input.NumElements(); ++i) {
        const float value = input.At(i);
        bool keep = false;
        switch (op_) {
            case TensorMaskOp::CompareGreater:
                keep = value > scalar_;
                break;
            case TensorMaskOp::CompareGreaterEqual:
                keep = value >= scalar_;
                break;
            case TensorMaskOp::CompareLess:
                keep = value < scalar_;
                break;
            case TensorMaskOp::CompareLessEqual:
                keep = value <= scalar_;
                break;
            case TensorMaskOp::CompareEqual:
                keep = value == scalar_;
                break;
            case TensorMaskOp::CompareNotEqual:
                keep = value != scalar_;
                break;
            case TensorMaskOp::LogicalNot:
                keep = value == 0.0f;
                break;
            default:
                throw std::runtime_error("TensorMaskModule: unsupported mask op");
        }
        output.Set(i, keep ? 1.0f : 0.0f);
    }

    return output;
}

Tensor TensorMaskModule::Backward(const Tensor& grad_output) {
    return Tensor::Zeros(input_cache_.Shape(), grad_output.GetDataType());
}

std::string TensorMaskModule::GetName() const {
    switch (op_) {
        case TensorMaskOp::CompareGreater:
            return "TensorCompareGreater";
        case TensorMaskOp::CompareGreaterEqual:
            return "TensorCompareGreaterEqual";
        case TensorMaskOp::CompareLess:
            return "TensorCompareLess";
        case TensorMaskOp::CompareLessEqual:
            return "TensorCompareLessEqual";
        case TensorMaskOp::CompareEqual:
            return "TensorCompareEqual";
        case TensorMaskOp::CompareNotEqual:
            return "TensorCompareNotEqual";
        case TensorMaskOp::LogicalNot:
            return "TensorLogicalNot";
        default:
            return "TensorMask";
    }
}

} // namespace cyxwiz
