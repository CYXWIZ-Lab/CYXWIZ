// Conv3D on rows (TOFIX140 Group C): torch.nn.Conv3d on [N, C, D, H, W]
// samples carried as channel-major rows. See Conv3DModule in sequential.h.
#include <cyxwiz/sequential.h>
#include "../layers/layer_arrayfire_utils.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz {

// The sparse gather (column matrix rows <- input elements) and its transpose,
// built once on the device; the forward pass's column matrix for backward.
struct Conv3DModule::DeviceGather {
#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array gather;   // CSR [C*k^3 * P, C*D*H*W]
    af::array scatter;  // CSR [C*D*H*W, C*k^3 * P]
    af::array columns;  // [C*k^3, P*N] from the last Forward
#endif
};

namespace {

size_t OutputExtent(size_t extent, int kernel, int stride, int padding, const char* axis) {
    const long long padded = static_cast<long long>(extent) + 2LL * padding;
    if (padded < kernel) {
        throw std::invalid_argument("Conv3D kernel " + std::to_string(kernel) + " does not fit the " + axis +
                                    " " + std::to_string(extent) + " (padding " + std::to_string(padding) + ")");
    }
    return static_cast<size_t>((padded - kernel) / stride + 1);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
[[noreturn]] void ThrowConv3DDeviceError(const char* operation, const af::exception& error) {
    throw std::runtime_error(std::string(operation) + " failed on the ArrayFire device: " + error.what());
}
#endif

}  // namespace

Conv3DModule::Conv3DModule(size_t depth, size_t height, size_t width, size_t in_channels, size_t filters,
                           int kernel_size, int stride, int padding, bool use_bias)
    : depth_(depth), height_(height), width_(width), in_channels_(in_channels), filters_(filters),
      kernel_size_(kernel_size), stride_(stride), padding_(padding), use_bias_(use_bias),
      device_(std::make_shared<DeviceGather>()) {
    if (depth_ == 0 || height_ == 0 || width_ == 0 || in_channels_ == 0 || filters_ == 0) {
        throw std::invalid_argument("Conv3D needs a non-empty [D, H, W, C] sample and at least one filter");
    }
    if (kernel_size_ <= 0 || stride_ <= 0 || padding_ < 0) {
        throw std::invalid_argument("Conv3D needs a positive kernel and stride and a non-negative padding");
    }
    out_depth_ = OutputExtent(depth_, kernel_size_, stride_, padding_, "depth");
    out_height_ = OutputExtent(height_, kernel_size_, stride_, padding_, "height");
    out_width_ = OutputExtent(width_, kernel_size_, stride_, padding_, "width");
    const size_t k = static_cast<size_t>(kernel_size_);
    column_size_ = in_channels_ * k * k * k;
    patches_ = out_depth_ * out_height_ * out_width_;

#ifndef CYXWIZ_HAS_ARRAYFIRE
    throw std::runtime_error("Conv3D runs on ArrayFire, and this build has no ArrayFire");
#else
    // For patch p and column k, the input element at gather[p * column_size_ + k]
    // (channel-major index within a sample), or -1 in the zero padding.
    std::vector<long long> gather(patches_ * column_size_, -1);
    size_t p = 0;
    for (size_t od = 0; od < out_depth_; ++od) {
        for (size_t oh = 0; oh < out_height_; ++oh) {
            for (size_t ow = 0; ow < out_width_; ++ow, ++p) {
                long long* row = gather.data() + p * column_size_;
                size_t col = 0;
                for (size_t c = 0; c < in_channels_; ++c) {
                    for (size_t kd = 0; kd < k; ++kd) {
                        const long long id = static_cast<long long>(od) * stride_ + kd - padding_;
                        for (size_t kh = 0; kh < k; ++kh) {
                            const long long ih = static_cast<long long>(oh) * stride_ + kh - padding_;
                            for (size_t kw = 0; kw < k; ++kw, ++col) {
                                const long long iw = static_cast<long long>(ow) * stride_ + kw - padding_;
                                if (id < 0 || ih < 0 || iw < 0 || id >= static_cast<long long>(depth_) ||
                                    ih >= static_cast<long long>(height_) || iw >= static_cast<long long>(width_))
                                    continue;
                                row[col] = ((static_cast<long long>(c) * depth_ + id) * height_ + ih) * width_ + iw;
                            }
                        }
                    }
                }
            }
        }
    }

    // Each column-matrix row (column k of patch p, row p*K + k) reads at most
    // one input element; the transpose sums every patch that read an element.
    const size_t samples = in_channels_ * depth_ * height_ * width_;
    const size_t rows = gather.size();
    std::vector<int> gather_offsets(rows + 1, 0), gather_columns;
    std::vector<int> scatter_offsets(samples + 1, 0);
    gather_columns.reserve(rows);
    for (size_t r = 0; r < rows; ++r) {
        gather_offsets[r + 1] = gather_offsets[r];
        if (gather[r] < 0) continue;
        gather_columns.push_back(static_cast<int>(gather[r]));
        ++gather_offsets[r + 1];
        ++scatter_offsets[static_cast<size_t>(gather[r]) + 1];
    }
    for (size_t s = 0; s < samples; ++s) scatter_offsets[s + 1] += scatter_offsets[s];
    std::vector<int> scatter_columns(gather_columns.size());
    std::vector<int> fill(scatter_offsets.begin(), scatter_offsets.end() - 1);
    for (size_t r = 0; r < rows; ++r) {
        if (gather[r] >= 0) scatter_columns[static_cast<size_t>(fill[static_cast<size_t>(gather[r])]++)] = static_cast<int>(r);
    }
    const std::vector<float> ones(gather_columns.size(), 1.0f);
    const dim_t nnz = static_cast<dim_t>(ones.size());
    try {
        device_->gather = af::sparse(static_cast<dim_t>(rows), static_cast<dim_t>(samples), nnz, ones.data(),
                                     gather_offsets.data(), gather_columns.data(), f32, AF_STORAGE_CSR, afHost);
        device_->scatter = af::sparse(static_cast<dim_t>(samples), static_cast<dim_t>(rows), nnz, ones.data(),
                                      scatter_offsets.data(), scatter_columns.data(), f32, AF_STORAGE_CSR, afHost);
    } catch (const af::exception& e) {
        ThrowConv3DDeviceError("Conv3DModule (building the gather)", e);
    }

    // Kaiming uniform weights (as Conv2D), zero bias.
    const float limit = std::sqrt(6.0f / static_cast<float>(column_size_));
    weight_ = Tensor::Random({filters_, column_size_}) * (2.0f * limit) - limit;
    grad_weight_ = Tensor::Zeros({filters_, column_size_});
    if (use_bias_) {
        bias_ = Tensor::Zeros({filters_});
        grad_bias_ = Tensor::Zeros({filters_});
    }
#endif
}

Tensor Conv3DModule::Forward(const Tensor& input) {
    has_forward_ = false;
    const size_t samples = in_channels_ * depth_ * height_ * width_;
    if (input.GetDataType() != DataType::Float32 || input.Shape().size() != 2 || input.Shape()[1] != samples) {
        throw std::runtime_error("Conv3D expects Float32 rows [N, " + std::to_string(samples) + "] (C*D*H*W)");
    }
    input_ = input;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t n = static_cast<dim_t>(input.Shape()[0]);
        const dim_t k = static_cast<dim_t>(column_size_);
        const dim_t p = static_cast<dim_t>(patches_);
        const dim_t f = static_cast<dim_t>(filters_);
        af::array columns = af::moddims(af::matmul(device_->gather, af::transpose(TensorToAf(input))), k, p * n);
        af::array y = af::matmul(TensorToAf(weight_), columns);  // [F, P*N]
        if (use_bias_) y += af::tile(af::moddims(TensorToAf(bias_), f, 1), 1, static_cast<unsigned>(p * n));
        y = af::transpose(af::moddims(af::reorder(af::moddims(y, f, p, n), 1, 0, 2), p * f, n));
        y.eval();
        columns.eval();
        device_->columns = columns;
        has_forward_ = true;
        return Tensor::FromSemanticArray(y, {input.Shape()[0], filters_ * patches_});
    } catch (const af::exception& e) {
        device_->columns = af::array();
        ThrowConv3DDeviceError("Conv3DModule::Forward", e);
    }
#else
    throw std::runtime_error("Conv3D runs on ArrayFire, and this build has no ArrayFire");
#endif
}

Tensor Conv3DModule::Backward(const Tensor& grad_output) {
    if (!has_forward_) throw std::logic_error("Conv3DModule::Backward requires a successful Forward call");
    const size_t n_rows = input_.Shape()[0];
    if (grad_output.GetDataType() != DataType::Float32 ||
        grad_output.Shape() != std::vector<size_t>{n_rows, filters_ * patches_}) {
        throw std::runtime_error("Conv3D backward gradient shape does not match Forward output");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const dim_t n = static_cast<dim_t>(n_rows);
        const dim_t k = static_cast<dim_t>(column_size_);
        const dim_t p = static_cast<dim_t>(patches_);
        const dim_t f = static_cast<dim_t>(filters_);
        af::array columns = device_->columns;
        if (columns.isempty())
            columns = af::moddims(af::matmul(device_->gather, af::transpose(TensorToAf(input_))), k, p * n);
        // rows [N, F*P] -> [F, P*N]
        const af::array g = af::moddims(af::reorder(af::moddims(af::transpose(TensorToAf(grad_output)), p, f, n),
                                                    1, 0, 2),
                                        f, p * n);
        af::array grad_weight = af::matmulNT(g, columns);
        grad_weight.eval();
        if (use_bias_) {
            af::array grad_bias = af::moddims(af::sum(g, 1), f);
            grad_bias.eval();
            grad_bias_ = Tensor::FromSemanticArray(grad_bias, {filters_});
        }
        const af::array grad_columns = af::moddims(af::matmulTN(TensorToAf(weight_), g), k * p, n);
        af::array grad_input = af::transpose(af::matmul(device_->scatter, grad_columns));
        grad_input.eval();
        grad_weight_ = Tensor::FromSemanticArray(grad_weight, {filters_, column_size_});
        return Tensor::FromSemanticArray(grad_input, input_.Shape());
    } catch (const af::exception& e) {
        ThrowConv3DDeviceError("Conv3DModule::Backward", e);
    }
#else
    throw std::runtime_error("Conv3D runs on ArrayFire, and this build has no ArrayFire");
#endif
}

std::map<std::string, Tensor> Conv3DModule::GetParameters() {
    std::map<std::string, Tensor> parameters{{"weight", weight_}};
    if (use_bias_) parameters.emplace("bias", bias_);
    return parameters;
}

void Conv3DModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (const auto it = params.find("weight"); it != params.end()) {
        if (it->second.Shape() != std::vector<size_t>{filters_, column_size_}) {
            throw std::runtime_error("Conv3D weight must be [" + std::to_string(filters_) + ", " +
                                     std::to_string(column_size_) + "] (filters, C*k*k*k)");
        }
        weight_ = it->second;
        has_forward_ = false;
    }
    if (const auto it = params.find("bias"); it != params.end() && use_bias_) {
        if (it->second.Shape() != std::vector<size_t>{filters_}) {
            throw std::runtime_error("Conv3D bias must be [" + std::to_string(filters_) + "]");
        }
        bias_ = it->second;
        has_forward_ = false;
    }
}

std::map<std::string, Tensor> Conv3DModule::GetGradients() {
    std::map<std::string, Tensor> gradients{{"weight", grad_weight_}};
    if (use_bias_) gradients.emplace("bias", grad_bias_);
    return gradients;
}

std::string Conv3DModule::GetName() const {
    return "Conv3D(" + std::to_string(in_channels_) + " -> " + std::to_string(filters_) + ", kernel=" +
           std::to_string(kernel_size_) + ", stride=" + std::to_string(stride_) + ", padding=" +
           std::to_string(padding_) + ", [" + std::to_string(depth_) + "," + std::to_string(height_) + "," +
           std::to_string(width_) + "] -> [" + std::to_string(out_depth_) + "," + std::to_string(out_height_) +
           "," + std::to_string(out_width_) + "])";
}

}  // namespace cyxwiz
