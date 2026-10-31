#include <cyxwiz/gradient_check.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <stdexcept>

namespace cyxwiz {

namespace {

std::vector<float> Host(const Tensor& tensor) {
    const float* data = tensor.ReadData<float>();
    return std::vector<float>(data, data + tensor.NumElements());
}

Tensor FromHost(const std::vector<size_t>& shape, const std::vector<float>& values) {
    return Tensor(shape, values.data());
}

std::vector<float> Random(size_t count, float range, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-range, range);
    std::vector<float> values(count);
    for (auto& v : values) {
        v = dist(rng);
        // Keep clear of exactly zero, where |x|-style functions have a kink.
        if (std::fabs(v) < 0.05f) v = v < 0.0f ? -0.05f : 0.05f;
    }
    return values;
}

double Dot(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) sum += static_cast<double>(a[i]) * b[i];
    return sum;
}

double RelativeError(double analytic, double numeric, double floor) {
    return std::fabs(analytic - numeric) / std::max(floor, std::fabs(analytic) + std::fabs(numeric));
}

// Elements probed: all when few, otherwise evenly spread.
std::vector<size_t> Probe(size_t count, size_t max_elements) {
    std::vector<size_t> indices;
    if (count <= max_elements) {
        for (size_t i = 0; i < count; ++i) indices.push_back(i);
        return indices;
    }
    for (size_t k = 0; k < max_elements; ++k) indices.push_back(k * count / max_elements);
    return indices;
}

void Record(GradientCheckResult& result, double error, const std::string& where) {
    if (error > result.max_relative_error) {
        result.max_relative_error = error;
        result.worst = where;
    }
}

}  // namespace

GradientCheckResult CheckModuleGradients(
    const std::function<std::unique_ptr<Module>(const std::vector<size_t>& input_shape)>& make,
    const GradientCheckOptions& options) {
    GradientCheckResult result;
    result.tolerance = options.tolerance;
    std::mt19937 rng(options.seed);

    try {
        for (const auto& shape : options.input_shapes) {
            auto module = make(shape);
            if (!module) {
                result.error = "the module could not be created for input " + std::to_string(shape.size()) + "-d";
                return result;
            }
            module->SetTraining(false);

            size_t count = 1;
            for (size_t d : shape) count *= d;
            const auto x = Random(count, options.input_range, rng);
            const Tensor input = FromHost(shape, x);

            // L = sum(Forward(x) * r); dL/dout = r.
            const Tensor out = module->Forward(input);
            const auto r = Random(out.NumElements(), 1.0f, rng);
            const Tensor upstream = FromHost(out.Shape(), r);
            const auto analytic_input = Host(module->Backward(upstream));
            const auto parameters = module->GetParameters();
            std::map<std::string, std::vector<float>> analytic_params;
            for (const auto& [key, gradient] : module->GetGradients()) analytic_params[key] = Host(gradient);

            const auto loss_at = [&](const Tensor& at) { return Dot(Host(module->Forward(at)), r); };

            // Input gradient.
            for (size_t i : Probe(count, options.max_elements)) {
                auto plus = x;
                auto minus = x;
                plus[i] += static_cast<float>(options.step);
                minus[i] -= static_cast<float>(options.step);
                const double numeric =
                    (loss_at(FromHost(shape, plus)) - loss_at(FromHost(shape, minus))) / (2.0 * options.step);
                Record(result, RelativeError(analytic_input.at(i), numeric, options.absolute_floor), "input");
            }

            // Parameter gradients.
            for (const auto& [key, value] : parameters) {
                const auto base = Host(value);
                const auto gradient = analytic_params.find(key);
                if (gradient == analytic_params.end()) {
                    result.error = "no gradient for parameter '" + key + "'";
                    return result;
                }
                for (size_t i : Probe(base.size(), options.max_elements)) {
                    auto plus = base;
                    auto minus = base;
                    plus[i] += static_cast<float>(options.step);
                    minus[i] -= static_cast<float>(options.step);
                    module->SetParameters({{key, FromHost(value.Shape(), plus)}});
                    const double l_plus = loss_at(input);
                    module->SetParameters({{key, FromHost(value.Shape(), minus)}});
                    const double l_minus = loss_at(input);
                    module->SetParameters({{key, FromHost(value.Shape(), base)}});
                    const double numeric = (l_plus - l_minus) / (2.0 * options.step);
                    Record(result, RelativeError(gradient->second.at(i), numeric, options.absolute_floor),
                           "parameter " + key);
                }
            }
        }
    } catch (const std::exception& e) {
        result.error = e.what();
        return result;
    }
    result.passed = result.max_relative_error <= options.tolerance;
    return result;
}

GradientCheckResult CheckLossGradients(
    const std::function<std::unique_ptr<Loss>()>& make,
    LossCheckTargets targets,
    const GradientCheckOptions& options) {
    GradientCheckResult result;
    result.tolerance = options.tolerance;
    std::mt19937 rng(options.seed);

    try {
        for (const auto& shape : options.input_shapes) {
            auto loss = make();
            if (!loss) {
                result.error = "the loss could not be created";
                return result;
            }
            if (shape.empty()) continue;
            size_t count = 1;
            for (size_t d : shape) count *= d;
            const auto p = Random(count, options.input_range, rng);
            std::vector<float> t;
            if (targets == LossCheckTargets::OneHot) {
                const size_t width = shape.back();
                t.assign(count, 0.0f);
                std::uniform_int_distribution<size_t> pick(0, width - 1);
                for (size_t row = 0; row < count / width; ++row) t[row * width + pick(rng)] = 1.0f;
            } else {
                t = Random(count, 1.0f, rng);
            }
            const Tensor target = FromHost(shape, t);

            const auto loss_at = [&](const std::vector<float>& at) {
                const Tensor value = loss->Forward(FromHost(shape, at), target);
                if (value.NumElements() != 1) {
                    throw std::runtime_error("forward must return one value, got " +
                                             std::to_string(value.NumElements()));
                }
                return static_cast<double>(Host(value)[0]);
            };
            loss_at(p);
            const Tensor analytic_tensor = loss->Backward(FromHost(shape, p), target);
            if (analytic_tensor.Shape() != shape) {
                throw std::runtime_error("backward must return the predictions' shape");
            }
            const auto analytic = Host(analytic_tensor);
            for (size_t i : Probe(count, options.max_elements)) {
                auto plus = p;
                auto minus = p;
                plus[i] += static_cast<float>(options.step);
                minus[i] -= static_cast<float>(options.step);
                const double numeric = (loss_at(plus) - loss_at(minus)) / (2.0 * options.step);
                Record(result, RelativeError(analytic.at(i), numeric, options.absolute_floor), "predictions");
            }
        }
    } catch (const std::exception& e) {
        result.error = e.what();
        return result;
    }
    result.passed = result.max_relative_error <= options.tolerance;
    return result;
}

std::string DescribeGradientCheck(const GradientCheckResult& result) {
    if (!result.error.empty()) return "Could not check: " + result.error;
    char text[160];
    std::snprintf(text, sizeof(text), "%s: largest relative error %.1e%s%s (tolerance %.1e)",
                  result.passed ? "Passed" : "Failed", result.max_relative_error,
                  result.worst.empty() ? "" : " in ", result.worst.c_str(), result.tolerance);
    return text;
}

}  // namespace cyxwiz
