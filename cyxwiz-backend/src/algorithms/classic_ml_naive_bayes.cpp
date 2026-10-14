// Gaussian naive Bayes on ArrayFire (TOFIX145 Group C), following scikit-learn's
// GaussianNB: per-class means and population variances, every variance plus
// var_smoothing times the largest feature variance of the whole matrix, priors
// from class counts, joint log-likelihood argmax (first maximum on ties), and
// probabilities through log-sum-exp.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/classic_ml.h"
#include "analysis_device.h"
#include "arrayfire_indexed.h"
#include "classic_ml_detail.h"

#include <cmath>
#include <string>
#include <vector>

namespace cyxwiz {

using namespace analysis_detail;
using classic_detail::LabelRefusal;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// Joint log-likelihood [n, k] of every row under every class.
af::array JointLogLikelihood(const af::array& x, const af::array& theta, const af::array& var,
                             const af::array& log_prior) {
    const dim_t n = x.dims(0), k = theta.dims(0);
    af::array jll = af::constant(0.0, n, k, x.type());
    for (int c = 0; c < static_cast<int>(k); ++c) {
        const af::array mean = af::tile(theta(c, af::span), static_cast<unsigned>(n), 1);
        const af::array variance = af::tile(var(c, af::span), static_cast<unsigned>(n), 1);
        const af::array diff = x - mean;
        const double log_norm = -0.5 * af::sum<double>(af::log(kTwoPi * var(c, af::span)));
        jll(af::span, c) = log_norm - 0.5 * af::sum(diff * diff / variance, 1);
    }
    return jll + af::tile(log_prior, static_cast<unsigned>(n), 1);
}

}  // namespace

GaussianNBModel ClassicML::FitGaussianNB(const ClassicMatrix& data, const std::vector<int>& labels, int n_classes,
                                         double var_smoothing) {
    GaussianNBModel model;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    if (!(var_smoothing >= 0.0)) {
        model.error_message = "var_smoothing must be non-negative";
        return model;
    }
    std::vector<double> counts;
    if (const std::string refusal = LabelRefusal(labels, data.size(), n_classes, counts); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    constexpr const char* kOperation = "ClassicML::FitGaussianNB";
    try {
        const af::array x = MatrixToDevice(data);
        const dim_t d = x.dims(1);
        const af::array y = af::array(static_cast<dim_t>(labels.size()), labels.data());
        model.epsilon = var_smoothing * af::max<double>(af::var(x, AF_VARIANCE_POPULATION, 0));
        af::array theta = af::constant(0.0, n_classes, d, x.type());
        af::array var = af::constant(0.0, n_classes, d, x.type());
        for (int c = 0; c < n_classes; ++c) {
            const af::array rows = x(af::where(y == c), af::span);
            const af::array mean = af::mean(rows, 0);
            const af::array diff = rows - af::tile(mean, static_cast<unsigned>(rows.dims(0)), 1);
            theta(c, af::span) = mean;
            var(c, af::span) = af::mean(diff * diff, 0) + model.epsilon;
        }
        model.theta = MatrixFromDevice(theta, kOperation);
        model.var = MatrixFromDevice(var, kOperation);
        const double total = static_cast<double>(labels.size());
        model.class_count = counts;
        for (const double count : counts) model.class_prior.push_back(count / total);
        model.success = true;
    } catch (const Refusal& refusal) {
        model = GaussianNBModel{};
        model.error_message = refusal.what();
    } catch (const af::exception& e) {
        model = GaussianNBModel{};
        model.error_message = DeviceError(kOperation, e);
    }
    return model;
}

ClassicPrediction ClassicML::PredictGaussianNB(const GaussianNBModel& model, const ClassicMatrix& data) {
    ClassicPrediction result;
    if (!model.success || model.theta.empty() || model.theta.size() != model.var.size() ||
        model.class_prior.size() != model.theta.size()) {
        result.error_message = "The Gaussian naive Bayes model is not fitted";
        return result;
    }
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    if (data.front().size() != model.theta.front().size()) {
        result.error_message = "Rows have " + std::to_string(data.front().size()) + " features; the model was fitted on " +
                               std::to_string(model.theta.front().size());
        return result;
    }
    constexpr const char* kOperation = "ClassicML::PredictGaussianNB";
    try {
        const af::array x = MatrixToDevice(data);
        const af::array theta = MatrixToDevice(model.theta).as(x.type());
        const af::array var = MatrixToDevice(model.var).as(x.type());
        std::vector<double> log_prior;
        for (const double prior : model.class_prior) log_prior.push_back(std::log(prior));
        const af::array log_prior_row =
            af::moddims(HostValuesToDevice(log_prior), 1, static_cast<dim_t>(log_prior.size())).as(x.type());
        const af::array jll = JointLogLikelihood(x, theta, var, log_prior_row);
        const dim_t k = jll.dims(1);
        af::array peak, labels;
        MaxWithIndex(peak, labels, jll, 1);
        const af::array log_norm =
            peak + af::log(af::sum(af::exp(jll - af::tile(peak, 1, static_cast<unsigned>(k))), 1));
        const af::array proba = af::exp(jll - af::tile(log_norm, 1, static_cast<unsigned>(k)));
        result.labels = LabelsFromDevice(labels, kOperation);
        result.probabilities = MatrixFromDevice(proba, kOperation);
        result.success = true;
    } catch (const Refusal& refusal) {
        result = ClassicPrediction{};
        result.error_message = refusal.what();
    } catch (const af::exception& e) {
        result = ClassicPrediction{};
        result.error_message = DeviceError(kOperation, e);
    }
    return result;
}

}  // namespace cyxwiz
