#include <cyxwiz/sequential.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {
// ============================================================================
// BatchNormModule Implementation (BatchNorm1D for MLPs)
// ============================================================================

BatchNormModule::BatchNormModule(size_t num_features, float eps, float momentum)
    : num_features_(num_features)
    , eps_(eps)
    , momentum_(momentum)
{
    // Initialize gamma (scale) to 1, beta (shift) to 0
    gamma_ = Tensor({num_features}, DataType::Float32);
    beta_ = Tensor({num_features}, DataType::Float32);
    running_mean_ = Tensor({num_features}, DataType::Float32);
    running_var_ = Tensor({num_features}, DataType::Float32);
    grad_gamma_ = Tensor({num_features}, DataType::Float32);
    grad_beta_ = Tensor({num_features}, DataType::Float32);

#ifdef CYXWIZ_HAS_ARRAYFIRE
    gamma_ = Tensor(af::constant(1.0f, num_features));
    beta_ = Tensor(af::constant(0.0f, num_features));
    running_mean_ = Tensor(af::constant(0.0f, num_features));
    running_var_ = Tensor(af::constant(1.0f, num_features));
#else
    float* gamma_data = gamma_.Data<float>();
    float* beta_data = beta_.Data<float>();
    float* rm_data = running_mean_.Data<float>();
    float* rv_data = running_var_.Data<float>();
    for (size_t i = 0; i < num_features; ++i) {
        gamma_data[i] = 1.0f;
        beta_data[i] = 0.0f;
        rm_data[i] = 0.0f;
        rv_data[i] = 1.0f;
    }
#endif

    spdlog::debug("BatchNormModule({}) initialized", num_features);
}

Tensor BatchNormModule::Forward(const Tensor& input) {
    input_cache_ = input.Clone();

    const auto& shape = input.Shape();
    size_t batch_size = shape[0];
    size_t features = shape.size() > 1 ? shape[1] : shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array x = input.GetArray();
    // x is [batch, features] in our row-major view
    // ArrayFire sees it as [batch, features] with dims(0)=batch, dims(1)=features

    // gamma/beta are [features], we need them as [1, features] for broadcasting
    af::array gamma = af::moddims(gamma_.GetArray(), 1, features);
    af::array beta = af::moddims(beta_.GetArray(), 1, features);

    if (is_training_) {
        // Compute mean and variance per feature (across batch = dim 0)
        af::array mean = af::mean(x, 0);  // [1, features]
        af::array var = af::var(x, AF_VARIANCE_POPULATION, 0);  // [1, features]

        // Update running statistics
        af::array rm = af::moddims(running_mean_.GetArray(), 1, features);
        af::array rv = af::moddims(running_var_.GetArray(), 1, features);
        rm = (1.0f - momentum_) * rm + momentum_ * mean;
        rv = (1.0f - momentum_) * rv + momentum_ * var;
        running_mean_ = Tensor(af::flat(rm));
        running_var_ = Tensor(af::flat(rv));

        // Normalize: (x - mean) / sqrt(var + eps)
        af::array std_inv = 1.0f / af::sqrt(var + eps_);
        // Tile mean and std_inv to [batch, features]
        af::array mean_tiled = af::tile(mean, batch_size, 1);
        af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);
        af::array x_norm = (x - mean_tiled) * std_inv_tiled;

        // Scale and shift: gamma * x_norm + beta
        af::array gamma_tiled = af::tile(gamma, batch_size, 1);
        af::array beta_tiled = af::tile(beta, batch_size, 1);
        af::array out = gamma_tiled * x_norm + beta_tiled;

        // Cache for backward
        normalized_ = Tensor(x_norm);
        std_inv_ = Tensor(af::flat(std_inv));
        batch_mean_ = Tensor(af::flat(mean));

        return Tensor(out);
    } else {
        // Inference mode: use running statistics
        af::array rm = af::moddims(running_mean_.GetArray(), 1, features);
        af::array rv = af::moddims(running_var_.GetArray(), 1, features);
        af::array std_inv = 1.0f / af::sqrt(rv + eps_);

        af::array rm_tiled = af::tile(rm, batch_size, 1);
        af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);
        af::array x_norm = (x - rm_tiled) * std_inv_tiled;

        af::array gamma_tiled = af::tile(gamma, batch_size, 1);
        af::array beta_tiled = af::tile(beta, batch_size, 1);
        af::array out = gamma_tiled * x_norm + beta_tiled;

        return Tensor(out);
    }
#else
    // CPU fallback
    Tensor output({batch_size, features}, DataType::Float32);
    const float* x_data = input.Data<float>();
    float* out_data = output.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    const float* beta_data = beta_.Data<float>();

    if (is_training_) {
        // Compute mean per feature
        std::vector<float> mean(features, 0.0f);
        std::vector<float> var(features, 0.0f);

        for (size_t f = 0; f < features; ++f) {
            for (size_t b = 0; b < batch_size; ++b) {
                mean[f] += x_data[b * features + f];
            }
            mean[f] /= batch_size;
        }

        // Compute variance per feature
        for (size_t f = 0; f < features; ++f) {
            for (size_t b = 0; b < batch_size; ++b) {
                float diff = x_data[b * features + f] - mean[f];
                var[f] += diff * diff;
            }
            var[f] /= batch_size;
        }

        // Update running statistics
        float* rm_data = running_mean_.Data<float>();
        float* rv_data = running_var_.Data<float>();
        for (size_t f = 0; f < features; ++f) {
            rm_data[f] = (1.0f - momentum_) * rm_data[f] + momentum_ * mean[f];
            rv_data[f] = (1.0f - momentum_) * rv_data[f] + momentum_ * var[f];
        }

        // Normalize, scale, shift
        normalized_ = Tensor({batch_size, features}, DataType::Float32);
        std_inv_ = Tensor({features}, DataType::Float32);
        float* norm_data = normalized_.Data<float>();
        float* std_inv_data = std_inv_.Data<float>();

        for (size_t f = 0; f < features; ++f) {
            std_inv_data[f] = 1.0f / std::sqrt(var[f] + eps_);
        }

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                float x_norm = (x_data[b * features + f] - mean[f]) * std_inv_data[f];
                norm_data[b * features + f] = x_norm;
                out_data[b * features + f] = x_norm * gamma_data[f] + beta_data[f];
            }
        }
    } else {
        // Inference mode
        const float* rm_data = running_mean_.Data<float>();
        const float* rv_data = running_var_.Data<float>();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                float std_inv = 1.0f / std::sqrt(rv_data[f] + eps_);
                float x_norm = (x_data[b * features + f] - rm_data[f]) * std_inv;
                out_data[b * features + f] = x_norm * gamma_data[f] + beta_data[f];
            }
        }
    }

    return output;
#endif
}

Tensor BatchNormModule::Backward(const Tensor& grad_output) {
    const auto& shape = grad_output.Shape();
    size_t batch_size = shape[0];
    size_t features = shape.size() > 1 ? shape[1] : shape[0];

#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array grad = grad_output.GetArray();
    // grad is [batch, features]

    // Get cached values - x_norm is [batch, features]
    af::array x_norm = normalized_.GetArray();

    // gamma is [features], reshape to [1, features] for broadcasting
    af::array gamma = af::moddims(gamma_.GetArray(), 1, features);
    af::array std_inv = af::moddims(std_inv_.GetArray(), 1, features);

    // Gradient w.r.t gamma and beta (sum along batch dimension = dim 0)
    af::array d_gamma = af::sum(grad * x_norm, 0);  // [1, features]
    af::array d_beta = af::sum(grad, 0);  // [1, features]

    grad_gamma_ = Tensor(af::flat(d_gamma));
    grad_beta_ = Tensor(af::flat(d_beta));

    // Gradient w.r.t input
    // d_x = (1/N) * std_inv * (N * d_y * gamma - sum(d_y * gamma) - x_norm * sum(d_y * gamma * x_norm))
    float N = static_cast<float>(batch_size);

    af::array gamma_tiled = af::tile(gamma, batch_size, 1);  // [batch, features]
    af::array d_x_norm = grad * gamma_tiled;

    af::array sum_d_x_norm = af::tile(af::sum(d_x_norm, 0), batch_size, 1);  // [batch, features]
    af::array sum_d_x_norm_x_norm = af::tile(af::sum(d_x_norm * x_norm, 0), batch_size, 1);  // [batch, features]

    af::array std_inv_tiled = af::tile(std_inv, batch_size, 1);  // [batch, features]
    af::array d_x = (1.0f / N) * std_inv_tiled *
                    (N * d_x_norm - sum_d_x_norm - x_norm * sum_d_x_norm_x_norm);

    return Tensor(d_x);
#else
    // CPU fallback
    Tensor grad_input({batch_size, features}, DataType::Float32);
    const float* grad_data = grad_output.Data<float>();
    const float* x_norm_data = normalized_.Data<float>();
    const float* std_inv_data = std_inv_.Data<float>();
    const float* gamma_data = gamma_.Data<float>();
    float* grad_input_data = grad_input.Data<float>();
    float* d_gamma_data = grad_gamma_.Data<float>();
    float* d_beta_data = grad_beta_.Data<float>();

    // Initialize gradients
    for (size_t f = 0; f < features; ++f) {
        d_gamma_data[f] = 0.0f;
        d_beta_data[f] = 0.0f;
    }

    // Compute d_gamma, d_beta
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t f = 0; f < features; ++f) {
            d_gamma_data[f] += grad_data[b * features + f] * x_norm_data[b * features + f];
            d_beta_data[f] += grad_data[b * features + f];
        }
    }

    // Compute d_x
    float N = static_cast<float>(batch_size);
    for (size_t f = 0; f < features; ++f) {
        float sum_d_x_norm = 0.0f;
        float sum_d_x_norm_x_norm = 0.0f;

        for (size_t b = 0; b < batch_size; ++b) {
            float d_x_norm = grad_data[b * features + f] * gamma_data[f];
            sum_d_x_norm += d_x_norm;
            sum_d_x_norm_x_norm += d_x_norm * x_norm_data[b * features + f];
        }

        for (size_t b = 0; b < batch_size; ++b) {
            float d_x_norm = grad_data[b * features + f] * gamma_data[f];
            grad_input_data[b * features + f] = (1.0f / N) * std_inv_data[f] *
                (N * d_x_norm - sum_d_x_norm - x_norm_data[b * features + f] * sum_d_x_norm_x_norm);
        }
    }

    return grad_input;
#endif
}

std::map<std::string, Tensor> BatchNormModule::GetParameters() {
    return {
        {"gamma", gamma_},
        {"beta", beta_}
    };
}

void BatchNormModule::SetParameters(const std::map<std::string, Tensor>& params) {
    if (params.count("gamma")) gamma_ = params.at("gamma");
    if (params.count("beta")) beta_ = params.at("beta");
    if (params.count("running_mean")) running_mean_ = params.at("running_mean");
    if (params.count("running_var")) running_var_ = params.at("running_var");
}

std::map<std::string, Tensor> BatchNormModule::GetGradients() {
    return {
        {"gamma", grad_gamma_},
        {"beta", grad_beta_}
    };
}

std::string BatchNormModule::GetName() const {
    return "BatchNorm(" + std::to_string(num_features_) + ")";
}

} // namespace cyxwiz

