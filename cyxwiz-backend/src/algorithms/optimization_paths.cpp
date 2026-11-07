// An optimizer's path on a typed formula (TOFIX129 D10): the formula's
// variables are one parameter tensor, its exact gradient the gradient, and
// the update is the training optimizer's own Step, in float32 as training
// runs it.

#include "cyxwiz/optimization.h"

#include "cyxwiz/optimizers/adam.h"
#include "cyxwiz/optimizers/adaptive.h"
#include "cyxwiz/optimizers/sgd.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

namespace {

std::unique_ptr<Optimizer> Make(const OptimizerSetting& s, std::string& error) {
    switch (s.type) {
        case OptimizerType::SGD: return std::make_unique<SGDOptimizer>(s.learning_rate, s.momentum);
        case OptimizerType::Adam: return std::make_unique<AdamOptimizer>(s.learning_rate);
        case OptimizerType::AdamW: return std::make_unique<AdamWOptimizer>(s.learning_rate);
        case OptimizerType::NAdam: return std::make_unique<NAdamOptimizer>(s.learning_rate);
        case OptimizerType::RMSprop: return std::make_unique<RMSpropOptimizer>(s.learning_rate, 0.99, 1e-8, s.momentum);
        case OptimizerType::AdaGrad: return std::make_unique<AdaGradOptimizer>(s.learning_rate);
        default: break;
    }
    error = "This optimizer is not offered for paths (SGD, Adam, AdamW, NAdam, RMSprop, AdaGrad are)";
    return nullptr;
}

std::vector<double> ReadPoint(const Tensor& t, size_t n) {
    const float* p = t.ReadData<float>();
    return std::vector<double>(p, p + n);
}

Tensor FloatTensor(const std::vector<double>& values) {
    std::vector<float> narrow(values.size());
    std::transform(values.begin(), values.end(), narrow.begin(), [](double v) { return static_cast<float>(v); });
    return Tensor({values.size()}, narrow.data(), DataType::Float32);
}

bool Finite(const std::vector<double>& values) {
    for (const double v : values)
        if (!std::isfinite(v)) return false;
    return true;
}

}  // namespace

OptimizerPath Optimization::FollowOptimizer(const Expression& f, const std::vector<double>& start,
                                            const OptimizerSetting& setting, int steps, double tolerance,
                                            const std::function<bool()>& stop) {
    OptimizerPath path;
    if (!f.Valid()) {
        path.error_message = f.Error();
        return path;
    }
    const size_t n = f.Variables().size();
    if (n == 0 || start.size() != n) {
        path.error_message = "The start needs one value per variable";
        return path;
    }
    if (!Finite(start)) {
        path.error_message = "The start must be numbers";
        return path;
    }
    if (steps < 0) {
        path.error_message = "The number of steps cannot be negative";
        return path;
    }
    if (!(setting.learning_rate > 0.0)) {
        path.error_message = "The learning rate must be above 0";
        return path;
    }
    std::vector<Expression> gradient;
    for (size_t i = 0; i < n; ++i) {
        gradient.push_back(f.Derivative(i));
        if (!gradient.back().Valid()) {
            path.error_message = "The formula has no exact gradient: " + gradient.back().Error();
            return path;
        }
    }
    std::unique_ptr<Optimizer> optimizer = Make(setting, path.error_message);
    if (!optimizer) return path;

    try {
        std::map<std::string, Tensor> parameters{{"x", FloatTensor(start)}};
        std::vector<double> point = ReadPoint(parameters.at("x"), n);
        for (int step = 0;; ++step) {
            std::vector<double> g(n);
            double norm = 0.0;
            for (size_t i = 0; i < n; ++i) {
                g[i] = gradient[i].Evaluate(point);
                norm += g[i] * g[i];
            }
            norm = std::sqrt(norm);
            const double value = f.Evaluate(point);
            path.points.push_back(point);
            path.values.push_back(value);
            path.gradient_norms.push_back(norm);
            if (!std::isfinite(value) || !std::isfinite(norm) || !Finite(point)) {
                path.diverged = true;
                path.diverged_at = step;
                break;
            }
            if (norm < tolerance) {
                path.converged = true;
                break;
            }
            if (step == steps) break;
            if (stop && stop()) {
                path.stopped = true;
                break;
            }
            std::map<std::string, Tensor> gradients{{"x", FloatTensor(g)}};
            optimizer->Step(parameters, gradients);
            point = ReadPoint(parameters.at("x"), n);
        }
    } catch (const std::exception& e) {
        path.error_message = std::string("The optimizer failed: ") + e.what();
        return path;
    }
    path.success = true;
    return path;
}

} // namespace cyxwiz
