#include "training_health.h"

#include <cyxwiz/sequential.h>

#include <arrayfire.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace cyxwiz {

namespace {

// Per-unit maximum over everything but the unit axis: channels of an
// [H, W, C, N] output, the last axis of [N, F] or [N, L, F].
af::array UnitMax(const Tensor& output) {
    const auto& shape = output.Shape();
    if (shape.size() == 4) {
        const af::array a = output.GetArrayRowMajor4D();  // (h, w, c, n)
        return af::flat(af::max(af::max(af::max(a, 0), 1), 3));
    }
    const size_t units = shape.empty() ? 0 : shape.back();
    if (units == 0) return af::array();
    const af::array flat = output.GetRowMajorFlatArray();
    // Row-major [..., F]: column-major (F, rest).
    return af::flat(af::max(af::moddims(flat, static_cast<dim_t>(units), flat.elements() / static_cast<dim_t>(units)), 1));
}

}  // namespace

struct TrainingHealthRecorder::DeviceState {
    std::map<std::string, af::array> gradient_norm_sums;  // device scalars
    long long steps = 0;
    std::map<size_t, af::array> relu_max;                  // per-unit maximum over validation
};

TrainingHealthRecorder::TrainingHealthRecorder() : state_(std::make_unique<DeviceState>()) {}
TrainingHealthRecorder::~TrainingHealthRecorder() = default;

size_t HealthLayerIndex(const std::string& parameter) {
    if (parameter.rfind("layer", 0) != 0) return std::string::npos;
    size_t end = 5;
    while (end < parameter.size() && parameter[end] >= '0' && parameter[end] <= '9') ++end;
    if (end == 5) return std::string::npos;
    return static_cast<size_t>(std::stoull(parameter.substr(5, end - 5)));
}

void TrainingHealthRecorder::OnStep(const std::map<std::string, Tensor>& gradients) {
    for (const auto& [name, gradient] : gradients) {
        if (gradient.Shape().size() < 2) continue;  // weights only, not biases or norms' scales
        const af::array g = gradient.GetRowMajorFlatArray();
        const af::array norm = af::sqrt(af::sum(g * g));
        auto found = state_->gradient_norm_sums.find(name);
        if (found == state_->gradient_norm_sums.end()) state_->gradient_norm_sums.emplace(name, norm);
        else found->second += norm;
    }
    ++state_->steps;
}

void TrainingHealthRecorder::OnValidationOutput(size_t module_index, const Module& module, const Tensor& output) {
    if (module.GetName() != "ReLU") return;
    const af::array unit_max = UnitMax(output);
    if (unit_max.isempty()) return;
    auto found = state_->relu_max.find(module_index);
    if (found == state_->relu_max.end() || found->second.elements() != unit_max.elements())
        state_->relu_max[module_index] = unit_max;
    else
        found->second = af::max(found->second, unit_max);
}

EpochHealth TrainingHealthRecorder::EndEpoch(int epoch, const std::map<std::string, Tensor>& parameters) {
    EpochHealth health;
    health.epoch = epoch;
    for (const auto& [name, tensor] : parameters) {
        if (tensor.Shape().size() < 2) continue;
        WeightHealth w;
        w.parameter = name;
        w.layer = HealthLayerIndex(name);
        const af::array values = tensor.GetRowMajorFlatArray().as(f32);
        w.count = static_cast<size_t>(values.elements());
        w.norm = std::sqrt(af::sum<double>(values * values));
        w.low = af::min<double>(values);
        w.high = af::max<double>(values);
        std::vector<unsigned> counts(kHealthHistogramBins, 0);
        if (w.high > w.low) {
            af::histogram(values, kHealthHistogramBins, w.low, w.high).as(u32).host(counts.data());
        } else {
            counts[0] = static_cast<unsigned>(w.count);
        }
        w.histogram.assign(counts.begin(), counts.end());
        const auto sum = state_->gradient_norm_sums.find(name);
        if (sum != state_->gradient_norm_sums.end() && state_->steps > 0)
            w.mean_gradient_norm = sum->second.scalar<float>() / static_cast<double>(state_->steps);
        health.weights.push_back(std::move(w));
    }
    for (const auto& [index, unit_max] : state_->relu_max) {
        ReluHealth relu;
        relu.layer = index;
        relu.units = static_cast<long long>(unit_max.elements());
        relu.dead = static_cast<long long>(af::count<unsigned>(unit_max <= 0.0f));
        health.relus.push_back(relu);
    }
    std::sort(health.weights.begin(), health.weights.end(),
              [](const WeightHealth& a, const WeightHealth& b) { return a.layer < b.layer; });
    state_->gradient_norm_sums.clear();
    state_->relu_max.clear();
    state_->steps = 0;
    return health;
}

}  // namespace cyxwiz
