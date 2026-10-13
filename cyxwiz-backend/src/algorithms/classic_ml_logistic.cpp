// Logistic regression on ArrayFire (TOFIX145 Group C), following scikit-learn's
// LogisticRegression with the lbfgs solver and an L2 penalty: two classes fit
// one coefficient row (sigmoid), more classes a multinomial softmax. The
// objective is the mean log loss plus ||coef||^2 / (2 C n), the intercept not
// penalised (sklearn's scaling, so tol means the same thing). It stops when the
// largest gradient entry is at most tol or a step no longer lowers the loss
// (sklearn's ftol, 64 eps). L-BFGS: the loss,
// the gradient and the two-loop recursion run on the device; only scalars
// (step sizes, curvature products) come back to the host.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/classic_ml.h"
#include "analysis_device.h"
#include "classic_ml_detail.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace cyxwiz {

using namespace analysis_detail;
using classic_detail::LabelRefusal;

namespace {

constexpr int kHistory = 10;  // scipy's L-BFGS-B default memory
// scikit-learn's lbfgs ftol: stop once a step lowers the loss by no more than
// this share, which is where float64 runs out (gradient ~1e-9 or below).
constexpr double kRelativeReduction = 64.0 * std::numeric_limits<double>::epsilon();

double Dot(const af::array& a, const af::array& b) { return af::sum<double>(a * b); }

// Rows with a trailing 1 for the intercept: [n, d + 1].
af::array WithBias(const af::array& x) {
    return af::join(1, x, af::constant(1.0, x.dims(0), 1, x.type()));
}

// Logits [n, k_out] from the flat parameters [(d + 1) * k_out] (column-major
// [d + 1, k_out]: coefficients then the intercept, per output).
af::array Logits(const af::array& xb, const af::array& w, dim_t k_out) {
    return af::matmul(xb, af::moddims(w, xb.dims(1), k_out));
}

// Row-wise log-sum-exp of [n, k].
af::array LogSumExp(const af::array& z) {
    const af::array peak = af::max(z, 1);
    return peak + af::log(af::sum(af::exp(z - af::tile(peak, 1, static_cast<unsigned>(z.dims(1)))), 1));
}

struct Objective {
    const af::array& xb;     // [n, d + 1]
    const af::array& y;      // binary: [n, 1] 0/1; multinomial: one-hot [n, k]
    const af::array& penal;  // [(d + 1) * k_out] 1 for coefficients, 0 for intercepts
    dim_t k_out;
    double l2;  // 1 / (C n)

    // Loss, and the gradient into `grad`.
    double operator()(const af::array& w, af::array& grad) const {
        const double n = static_cast<double>(xb.dims(0));
        const af::array z = Logits(xb, w, k_out);
        double loss = 0.0;
        af::array residual;
        if (k_out == 1) {
            // log(1 + e^z) - y z, stably.
            const af::array softplus = af::max(z, 0.0) + af::log1p(af::exp(-af::abs(z)));
            loss = af::sum<double>(softplus - y * z) / n;
            residual = af::sigmoid(z) - y;
        } else {
            const af::array lse = LogSumExp(z);
            loss = (af::sum<double>(lse) - af::sum<double>(y * z)) / n;
            residual = af::exp(z - af::tile(lse, 1, static_cast<unsigned>(k_out))) - y;
        }
        const af::array penalised = w * penal;
        loss += 0.5 * l2 * Dot(penalised, penalised);
        grad = af::flat(af::matmul(xb, residual, AF_MAT_TRANS)) / n + l2 * penalised;
        grad.eval();
        return loss;
    }
};

}  // namespace

LogisticModel ClassicML::FitLogistic(const ClassicMatrix& data, const std::vector<int>& labels, int n_classes, double C,
                                     int max_iter, double tol) {
    LogisticModel model;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    std::vector<double> counts;
    if (const std::string refusal = LabelRefusal(labels, data.size(), n_classes, counts); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    if (n_classes < 2) {
        model.error_message = "Logistic regression needs at least two classes";
        return model;
    }
    if (!(C > 0.0) || max_iter < 1 || !(tol > 0.0)) {
        model.error_message = "C and tol must be positive and max_iter at least 1";
        return model;
    }
    constexpr const char* kOperation = "ClassicML::FitLogistic";
    try {
        const af::array x = MatrixToDevice(data);
        const dim_t n = x.dims(0), d = x.dims(1);
        const dim_t k_out = n_classes == 2 ? 1 : n_classes;
        const af::array xb = WithBias(x);
        const af::array label_column = af::array(n, labels.data()).as(x.type());
        af::array y;
        if (k_out == 1) {
            y = label_column;
        } else {
            const af::array classes = af::tile(af::range(af::dim4(1, k_out), 1, x.type()), static_cast<unsigned>(n), 1);
            y = (af::tile(label_column, 1, static_cast<unsigned>(k_out)) == classes).as(x.type());
        }
        af::array penal = af::constant(1.0, d + 1, k_out, x.type());
        penal(static_cast<int>(d), af::span) = 0.0;
        penal = af::flat(penal);
        const Objective objective{xb, y, penal, k_out, 1.0 / (C * static_cast<double>(n))};

        af::array w = af::constant(0.0, (d + 1) * k_out, x.type());
        af::array grad;
        double loss = objective(w, grad);
        std::deque<af::array> s_hist, y_hist;
        std::deque<double> rho_hist;
        int iter = 0;
        model.converged = af::max<double>(af::abs(grad)) <= tol;
        while (!model.converged && iter < max_iter) {
            // Two-loop recursion: direction = -H grad.
            af::array q = grad;
            std::vector<double> alpha(s_hist.size());
            for (int i = static_cast<int>(s_hist.size()) - 1; i >= 0; --i) {
                alpha[static_cast<size_t>(i)] = rho_hist[static_cast<size_t>(i)] * Dot(s_hist[static_cast<size_t>(i)], q);
                q -= alpha[static_cast<size_t>(i)] * y_hist[static_cast<size_t>(i)];
                q.eval();
            }
            if (!s_hist.empty())
                q *= Dot(s_hist.back(), y_hist.back()) / Dot(y_hist.back(), y_hist.back());
            for (size_t i = 0; i < s_hist.size(); ++i) {
                const double beta = rho_hist[i] * Dot(y_hist[i], q);
                q += (alpha[i] - beta) * s_hist[i];
                q.eval();
            }
            af::array direction = -q;
            double slope = Dot(grad, direction);
            if (!(slope < 0.0)) {  // not a descent direction: restart from steepest descent
                s_hist.clear();
                y_hist.clear();
                rho_hist.clear();
                direction = -grad;
                slope = -Dot(grad, grad);
            }
            // Backtracking (Armijo); the first try of plain gradient descent is scaled.
            double step = s_hist.empty() ? 1.0 / std::max(1.0, std::sqrt(-slope)) : 1.0;
            af::array next_w, next_grad;
            double next_loss = 0.0;
            bool accepted = false;
            for (int tries = 0; tries < 60; ++tries) {
                next_w = w + step * direction;
                next_w.eval();  // keep each iterate a plain buffer, not a growing JIT tree
                next_loss = objective(next_w, next_grad);
                if (std::isfinite(next_loss) && next_loss <= loss + 1e-4 * step * slope) {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            ++iter;
            if (!accepted) break;  // no further decrease in float precision
            const af::array s = next_w - w;
            const af::array g_change = next_grad - grad;
            af::eval(s, g_change);
            const double sy = Dot(s, g_change);
            if (sy > 1e-12 * Dot(g_change, g_change)) {
                if (static_cast<int>(s_hist.size()) == kHistory) {
                    s_hist.pop_front();
                    y_hist.pop_front();
                    rho_hist.pop_front();
                }
                s_hist.push_back(s);
                y_hist.push_back(g_change);
                rho_hist.push_back(1.0 / sy);
            }
            const double reduction = loss - next_loss;
            const double scale = std::max({std::abs(loss), std::abs(next_loss), 1.0});
            w = next_w;
            grad = next_grad;
            loss = next_loss;
            model.converged = af::max<double>(af::abs(grad)) <= tol || reduction <= kRelativeReduction * scale;
        }

        const ClassicMatrix full = MatrixFromDevice(af::moddims(w, d + 1, k_out).T(), kOperation);  // [k_out, d + 1]
        for (const auto& row : full) {
            model.coef.emplace_back(row.begin(), row.end() - 1);
            model.intercept.push_back(row.back());
        }
        model.n_classes = n_classes;
        model.n_iter = iter;
        model.loss = loss;
        model.success = true;
    } catch (const Refusal& refusal) {
        model = LogisticModel{};
        model.error_message = refusal.what();
    } catch (const af::exception& e) {
        model = LogisticModel{};
        model.error_message = DeviceError(kOperation, e);
    }
    return model;
}

ClassicPrediction ClassicML::PredictLogistic(const LogisticModel& model, const ClassicMatrix& data) {
    ClassicPrediction result;
    if (!model.success || model.coef.empty() || model.coef.size() != model.intercept.size() || model.n_classes < 2) {
        result.error_message = "The logistic regression model is not fitted";
        return result;
    }
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    if (data.front().size() != model.coef.front().size()) {
        result.error_message = "Rows have " + std::to_string(data.front().size()) + " features; the model was fitted on " +
                               std::to_string(model.coef.front().size());
        return result;
    }
    constexpr const char* kOperation = "ClassicML::PredictLogistic";
    try {
        const af::array x = MatrixToDevice(data);
        ClassicMatrix params;  // [k_out, d + 1]
        for (size_t r = 0; r < model.coef.size(); ++r) {
            params.push_back(model.coef[r]);
            params.back().push_back(model.intercept[r]);
        }
        const af::array w = MatrixToDevice(params).as(x.type()).T();  // [d + 1, k_out]
        const af::array z = af::matmul(WithBias(x), w);
        af::array proba;
        if (model.coef.size() == 1) {
            const af::array p = af::sigmoid(z);
            proba = af::join(1, 1.0 - p, p);
        } else {
            const af::array lse = LogSumExp(z);
            proba = af::exp(z - af::tile(lse, 1, static_cast<unsigned>(z.dims(1))));
        }
        af::array peak, labels;
        if (model.coef.size() == 1) {
            labels = (z > 0.0).as(u32);  // sklearn: decision > 0 is class 1
        } else {
            af::max(peak, labels, z, 1);
        }
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
