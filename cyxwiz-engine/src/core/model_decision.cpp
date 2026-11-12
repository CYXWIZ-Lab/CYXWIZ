#include "model_decision.h"

#include "classification_decision.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz {

namespace {

double Sigmoid(double z) { return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z)); }

double Isotonic(const ModelDecision& d, double p) {
    const auto& x = d.isotonic_x;
    const auto& y = d.isotonic_y;
    if (x.empty() || x.size() != y.size()) return p;
    if (p <= x.front()) return y.front();
    if (p >= x.back()) return y.back();
    const size_t hi = static_cast<size_t>(std::upper_bound(x.begin(), x.end(), p) - x.begin());
    const size_t lo = hi - 1;
    return y[lo] + (p - x[lo]) / (x[hi] - x[lo]) * (y[hi] - y[lo]);
}

std::string Name(int index, const std::vector<std::string>& names) {
    return index >= 0 && static_cast<size_t>(index) < names.size() && !names[index].empty() ? names[index]
                                                                                           : "class " + std::to_string(index);
}

}  // namespace

std::vector<float> DecisionProbabilities(const ModelDecision& decision, const float* outputs, size_t width) {
    std::vector<float> p(width);
    if (width == 0) return p;
    if (decision.scores == "probabilities" && width == 1) {
        const float positive = std::clamp(outputs[0], 0.0f, 1.0f);
        return {1.0f - positive, positive};
    }
    if (decision.scores == "probabilities") {
        std::copy(outputs, outputs + width, p.begin());
    } else if (decision.scores == "sigmoid" && width == 1) {
        const float positive = static_cast<float>(Sigmoid(outputs[0]));
        return {1.0f - positive, positive};
    } else if (decision.scores == "sigmoid") {
        for (size_t c = 0; c < width; ++c) p[c] = static_cast<float>(Sigmoid(outputs[c]));
    } else {
        // Scores that already form a distribution (a Softmax output layer)
        // are used as they are, as Test does (ClassificationClassProbabilities).
        float sum = 0.0f;
        bool distribution = true;
        for (size_t c = 0; c < width; ++c) {
            distribution = distribution && outputs[c] >= 0.0f && outputs[c] <= 1.0f;
            sum += outputs[c];
        }
        if (distribution && std::abs(sum - 1.0f) <= 1e-3f) {
            std::copy(outputs, outputs + width, p.begin());
            return p;
        }
        const float top = *std::max_element(outputs, outputs + width);
        double total = 0.0;
        for (size_t c = 0; c < width; ++c) total += std::exp(static_cast<double>(outputs[c] - top));
        for (size_t c = 0; c < width; ++c) p[c] = static_cast<float>(std::exp(static_cast<double>(outputs[c] - top)) / total);
    }
    return p;
}

int ApplyModelDecision(const ModelDecision& decision, std::vector<float>& probabilities) {
    const size_t classes = probabilities.size();
    if (classes == 0) return -1;
    switch (decision.calibration) {
        case ModelDecision::Calibration::Temperature: {
            std::vector<double> z(classes);
            double top = -1e300, total = 0.0;
            for (size_t c = 0; c < classes; ++c) {
                z[c] = std::log(std::max(1e-12, static_cast<double>(probabilities[c]))) / decision.temperature;
                top = std::max(top, z[c]);
            }
            for (auto& v : z) total += (v = std::exp(v - top));
            for (size_t c = 0; c < classes; ++c) probabilities[c] = static_cast<float>(z[c] / total);
            break;
        }
        case ModelDecision::Calibration::Platt:
        case ModelDecision::Calibration::Isotonic: {
            if (classes != 2) break;
            const size_t pos = decision.positive == 0 ? 0 : 1;
            const double p = probabilities[pos];
            const double q = decision.calibration == ModelDecision::Calibration::Platt
                                 ? Sigmoid(-(decision.platt_a * p + decision.platt_b))
                                 : Isotonic(decision, p);
            probabilities[pos] = static_cast<float>(q);
            probabilities[1 - pos] = static_cast<float>(1.0 - q);
            break;
        }
        case ModelDecision::Calibration::None:
            break;
    }
    if (decision.has_threshold && classes == 2) {
        const int pos = decision.positive == 0 ? 0 : 1;
        return probabilities[pos] >= decision.threshold ? pos : 1 - pos;
    }
    return static_cast<int>(std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin());
}

std::string ModelDecisionToJson(const ModelDecision& d) {
    nlohmann::json j;
    j["scores"] = d.scores;
    if (d.has_threshold) j["threshold"] = {{"positive", d.positive}, {"value", d.threshold}};
    switch (d.calibration) {
        case ModelDecision::Calibration::Temperature:
            j["calibration"] = {{"method", "temperature"}, {"temperature", d.temperature}};
            break;
        case ModelDecision::Calibration::Platt:
            j["calibration"] = {{"method", "platt"}, {"positive", d.positive}, {"a", d.platt_a}, {"b", d.platt_b}};
            break;
        case ModelDecision::Calibration::Isotonic:
            j["calibration"] = {{"method", "isotonic"}, {"positive", d.positive}, {"x", d.isotonic_x}, {"y", d.isotonic_y}};
            break;
        case ModelDecision::Calibration::None:
            break;
    }
    return j.dump();
}

bool ModelDecisionFromJson(const std::string& text, ModelDecision& decision) {
    try {
        const auto j = nlohmann::json::parse(text);
        if (!j.is_object()) return false;
        ModelDecision d;
        d.scores = j.value("scores", std::string("softmax"));
        if (j.contains("threshold")) {
            d.has_threshold = true;
            d.positive = j["threshold"].value("positive", 1);
            d.threshold = j["threshold"].value("value", 0.5);
        }
        if (j.contains("calibration")) {
            const auto& c = j["calibration"];
            const std::string method = c.value("method", std::string());
            if (method == "temperature") {
                d.calibration = ModelDecision::Calibration::Temperature;
                d.temperature = c.value("temperature", 1.0);
                if (!(d.temperature > 0.0)) return false;
            } else if (method == "platt") {
                d.calibration = ModelDecision::Calibration::Platt;
                d.positive = c.value("positive", d.positive);
                d.platt_a = c.value("a", 0.0);
                d.platt_b = c.value("b", 0.0);
            } else if (method == "isotonic") {
                d.calibration = ModelDecision::Calibration::Isotonic;
                d.positive = c.value("positive", d.positive);
                d.isotonic_x = c.value("x", std::vector<double>{});
                d.isotonic_y = c.value("y", std::vector<double>{});
                if (d.isotonic_x.size() != d.isotonic_y.size()) return false;
            } else if (!method.empty()) {
                return false;
            }
        }
        decision = std::move(d);
        return true;
    } catch (...) {
        return false;
    }
}

std::string DescribeModelDecision(const ModelDecision& d, const std::vector<std::string>& class_names) {
    std::string text;
    char number[32];
    if (d.has_threshold) {
        std::snprintf(number, sizeof(number), "%.3g", d.threshold);
        text = "threshold " + std::string(number) + " for " + Name(d.positive, class_names);
    }
    if (d.calibration != ModelDecision::Calibration::None) {
        if (!text.empty()) text += "; ";
        if (d.calibration == ModelDecision::Calibration::Temperature) {
            std::snprintf(number, sizeof(number), "%.3f", d.temperature);
            text += "temperature " + std::string(number);
        } else {
            text += d.calibration == ModelDecision::Calibration::Platt ? "Platt calibration" : "isotonic calibration";
            text += " of " + Name(d.positive, class_names);
        }
    }
    return text;
}

std::string DecisionScoresForLoss(gui::NodeType loss_type) {
    switch (ClassificationDecisionModeForLoss(loss_type)) {
        case ClassificationDecisionMode::BinaryLogit: return "sigmoid";
        case ClassificationDecisionMode::BinaryProbability: return "probabilities";
        default: return "softmax";
    }
}

}  // namespace cyxwiz
