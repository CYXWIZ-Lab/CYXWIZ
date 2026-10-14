// Support vector machines on ArrayFire (TOFIX145 Group C), following
// scikit-learn's SVC / SVR (libsvm): the dual is solved by SMO with libsvm's
// second-order working set selection and update rules. The gradient, the box
// masks and the kernel rows live on the device; each iteration brings back only
// the two chosen indices and a few scalars. Kernel rows are computed on demand
// (two per iteration), so memory stays O(n) and no n x n matrix is held.
//
// SVC: one-vs-one over the sorted classes; the first class of each pair is +1,
// and a pair's positive decision votes for it; the first class with the most
// votes wins (libsvm). SVR: epsilon-SVR as libsvm's 2l-variable problem.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/classic_ml.h"
#include "analysis_device.h"
#include "classic_ml_detail.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace cyxwiz {

using namespace analysis_detail;
using classic_detail::LabelRefusal;

namespace {

constexpr double kTau = 1e-12;  // libsvm TAU: a non-positive curvature is replaced by this

struct Kernel {
    std::string kind;  // linear, poly, rbf, sigmoid
    double gamma = 1.0;
    int degree = 3;
    double coef0 = 0.0;
};

// base^degree by repeated products, as libsvm's powi (af::pow gives NaN for a
// negative base).
af::array IntegerPower(const af::array& base, int degree) {
    af::array out = af::constant(1.0, base.dims(), base.type());
    for (int e = 0; e < degree; ++e) out *= base;
    return out;
}

// Kernel values [b, n] between `rows` [b, d] and `data` [n, d]. RBF uses
// |u|^2 + |v|^2 - 2 u.v, as libsvm does.
af::array KernelBlock(const af::array& rows, const af::array& data, const Kernel& k) {
    const af::array dot = af::matmul(rows, data, AF_MAT_NONE, AF_MAT_TRANS);
    if (k.kind == "linear") return dot;
    if (k.kind == "poly") return IntegerPower(k.gamma * dot + k.coef0, k.degree);
    if (k.kind == "sigmoid") return af::tanh(k.gamma * dot + k.coef0);
    const af::array row_sq = af::sum(rows * rows, 1);
    const af::array data_sq = af::sum(data * data, 1).T();
    return af::exp(-k.gamma * (af::tile(row_sq, 1, static_cast<unsigned>(data.dims(0))) +
                               af::tile(data_sq, static_cast<unsigned>(rows.dims(0)), 1) - 2.0 * dot));
}

// K(x, x) for every row [n].
af::array KernelDiagonal(const af::array& data, const Kernel& k) {
    const af::array sq = af::sum(data * data, 1);
    if (k.kind == "linear") return sq;
    if (k.kind == "poly") return IntegerPower(k.gamma * sq + k.coef0, k.degree);
    if (k.kind == "sigmoid") return af::tanh(k.gamma * sq + k.coef0);
    return af::constant(1.0, data.dims(0), data.type());
}

double At(const af::array& v, int i) { return af::sum<double>(v(i)); }

// Variable t of the dual is data row t % rows (SVR has two variables per row).
struct DualProblem {
    const af::array& data;   // [l, d]
    af::array y;             // [m] +1 / -1
    af::array p;             // [m] linear term
    std::vector<double> y_host;
    double C = 1.0;
    Kernel kernel;
};

struct DualSolution {
    std::vector<double> alpha;  // [m]
    double rho = 0.0;
    int iterations = 0;
    bool converged = false;
};

// Q row of variable t [m]: y_t * y * K(row t % l, every row), tiled for SVR.
af::array QRow(const DualProblem& prob, int t) {
    const dim_t l = prob.data.dims(0);
    const dim_t m = prob.y.dims(0);
    const int row = static_cast<int>(t % l);
    af::array k = KernelBlock(prob.data(row, af::span), prob.data, prob.kernel).T();  // [l]
    if (m != l) k = af::tile(k, static_cast<unsigned>(m / l));
    return prob.y_host[static_cast<size_t>(t)] * prob.y * k;
}

// libsvm Solver::Solve (no shrinking): min 0.5 a'Qa + p'a, 0 <= a <= C, y'a = 0.
DualSolution SolveDual(const DualProblem& prob, double eps, const char* operation) {
    const dim_t l = prob.data.dims(0);
    const int m = static_cast<int>(prob.y.dims(0));
    const auto type = prob.data.type();
    const double C = prob.C;
    af::array diag = KernelDiagonal(prob.data, prob.kernel);
    if (m != l) diag = af::tile(diag, static_cast<unsigned>(m / l));
    const std::vector<double> qd = DeviceToHostValues(diag, operation);

    af::array alpha = af::constant(0.0, m, type);
    std::vector<double> alpha_host(static_cast<size_t>(m), 0.0);
    af::array grad = prob.p;  // Q a + p at a = 0
    grad.eval();
    const af::array positive = prob.y > 0.0;
    const double inf = std::numeric_limits<double>::infinity();
    const int max_iter = std::max(10000000, m > INT_MAX / 100 ? INT_MAX : 100 * m);

    DualSolution out;
    int iter = 0;
    while (iter < max_iter) {
        // Working set: i maximises -y G over I_up; j minimises the second-order
        // objective over I_low (libsvm select_working_set).
        const af::array at_upper = alpha >= C;
        const af::array at_lower = alpha <= 0.0;
        const af::array in_up = (positive && !at_upper) || (!positive && !at_lower);
        const af::array in_low = (positive && !at_lower) || (!positive && !at_upper);
        const af::array minus_yg = -prob.y * grad;
        af::array g_max_value, i_index;
        af::max(g_max_value, i_index, af::select(in_up, minus_yg, -inf));
        const double g_max = af::sum<double>(g_max_value);
        const int i = static_cast<int>(af::sum<unsigned>(i_index));
        if (!std::isfinite(g_max)) {
            out.converged = true;
            break;
        }
        const af::array q_i = QRow(prob, i);
        const af::array yg = prob.y * grad;
        const double g_max2 = af::max<double>(af::select(in_low, yg, -inf));
        const af::array grad_diff = g_max + yg;
        // a = K_ii + K_tt - 2 K_it, from Q: Q_it = y_i y_t K_it.
        af::array curvature = qd[static_cast<size_t>(i)] + diag - 2.0 * prob.y_host[static_cast<size_t>(i)] * prob.y * q_i;
        curvature = af::select(curvature <= 0.0, kTau, curvature);
        const af::array objective =
            af::select(in_low && grad_diff > 0.0, -(grad_diff * grad_diff) / curvature, inf);
        af::array obj_min, j_index;
        af::min(obj_min, j_index, objective);
        if (g_max + g_max2 < eps || !std::isfinite(af::sum<double>(obj_min))) {
            out.converged = true;
            break;
        }
        const int j = static_cast<int>(af::sum<unsigned>(j_index));
        ++iter;

        // Two-variable update (libsvm Solver::Solve), box C for every variable.
        const af::array q_j = QRow(prob, j);
        const double y_i = prob.y_host[static_cast<size_t>(i)], y_j = prob.y_host[static_cast<size_t>(j)];
        const double g_i = At(grad, i), g_j = At(grad, j), q_ij = At(q_i, j);
        const double old_i = alpha_host[static_cast<size_t>(i)], old_j = alpha_host[static_cast<size_t>(j)];
        double a_i = old_i, a_j = old_j;
        if (y_i != y_j) {
            double quad = qd[static_cast<size_t>(i)] + qd[static_cast<size_t>(j)] + 2.0 * q_ij;
            if (quad <= 0.0) quad = kTau;
            const double delta = (-g_i - g_j) / quad;
            const double diff = a_i - a_j;
            a_i += delta;
            a_j += delta;
            if (diff > 0.0) {
                if (a_j < 0.0) { a_j = 0.0; a_i = diff; }
            } else if (a_i < 0.0) {
                a_i = 0.0;
                a_j = -diff;
            }
            if (diff > 0.0) {  // C_i - C_j == 0
                if (a_i > C) { a_i = C; a_j = C - diff; }
            } else if (a_j > C) {
                a_j = C;
                a_i = C + diff;
            }
        } else {
            double quad = qd[static_cast<size_t>(i)] + qd[static_cast<size_t>(j)] - 2.0 * q_ij;
            if (quad <= 0.0) quad = kTau;
            const double delta = (g_i - g_j) / quad;
            const double sum = a_i + a_j;
            a_i -= delta;
            a_j += delta;
            if (sum > C) {
                if (a_i > C) { a_i = C; a_j = sum - C; }
            } else if (a_j < 0.0) {
                a_j = 0.0;
                a_i = sum;
            }
            if (sum > C) {
                if (a_j > C) { a_j = C; a_i = sum - C; }
            } else if (a_i < 0.0) {
                a_i = 0.0;
                a_j = sum;
            }
        }
        alpha_host[static_cast<size_t>(i)] = a_i;
        alpha_host[static_cast<size_t>(j)] = a_j;
        alpha(i) = a_i;
        alpha(j) = a_j;
        grad += q_i * (a_i - old_i) + q_j * (a_j - old_j);
        af::eval(alpha, grad);
    }

    // rho (libsvm calculate_rho): the mean y G over free variables, else the
    // middle of the bounds.
    const af::array yg = prob.y * grad;
    const af::array at_upper = alpha >= C;
    const af::array at_lower = alpha <= 0.0;
    const af::array free = !at_upper && !at_lower;
    const double n_free = af::sum<double>(free.as(type));
    if (n_free > 0.0) {
        out.rho = af::sum<double>(af::select(free, yg, 0.0)) / n_free;
    } else {
        const af::array to_ub = (at_upper && !positive) || (at_lower && positive);
        const af::array to_lb = (at_upper && positive) || (at_lower && !positive);
        const double ub = af::min<double>(af::select(to_ub, yg, inf));
        const double lb = af::max<double>(af::select(to_lb, yg, -inf));
        out.rho = (ub + lb) / 2.0;
    }
    out.alpha = alpha_host;
    out.iterations = iter;
    return out;
}

std::string KernelRefusal(const std::string& kernel, double gamma, int degree) {
    if (kernel != "linear" && kernel != "poly" && kernel != "rbf" && kernel != "sigmoid")
        return "kernel must be linear, poly, rbf or sigmoid";
    if (!(gamma > 0.0) || !std::isfinite(gamma)) return "gamma must be positive";
    if (degree < 0) return "degree must be non-negative";
    return {};
}

// Row indices of `rows` as a device index array.
af::array Rows(const af::array& data, const std::vector<int>& rows) {
    return data(af::array(static_cast<dim_t>(rows.size()), rows.data()), af::span);
}

// One fitted dual: support rows (indices into the data), their coefficients
// (y alpha, summed over SVR's two variables) and rho.
void KeepSupport(const DualSolution& solution, const std::vector<int>& data_rows, size_t l,
                 std::vector<int>& support, std::vector<double>& coef, const std::vector<double>& y) {
    std::vector<double> per_row(l, 0.0);
    for (size_t t = 0; t < solution.alpha.size(); ++t) per_row[t % l] += y[t] * solution.alpha[t];
    std::vector<bool> used(l, false);
    for (size_t t = 0; t < solution.alpha.size(); ++t)
        if (solution.alpha[t] > 0.0) used[t % l] = true;
    for (size_t r = 0; r < l; ++r) {
        if (!used[r]) continue;
        support.push_back(data_rows[r]);
        coef.push_back(per_row[r]);
    }
}

}  // namespace

double ClassicML::ResolveSVMGamma(const ClassicMatrix& x, const std::string& gamma, std::string* error) {
    if (x.empty() || x.front().empty()) return 0.0;
    const double d = static_cast<double>(x.front().size());
    if (gamma == "auto") return 1.0 / d;
    if (gamma == "scale") {
        // 1 / (n_features * X.var()) over every entry; 1 when X is constant.
        try {
            const af::array all = af::flat(MatrixToDevice(x));
            const double mean = af::mean<double>(all);
            const af::array centred = all - mean;
            const double var = af::mean<double>(centred * centred);
            return var > 0.0 ? 1.0 / (d * var) : 1.0;
        } catch (const af::exception& e) {
            if (error) *error = DeviceError("ClassicML::ResolveSVMGamma", e);
            return 0.0;
        }
    }
    char* end = nullptr;
    const double value = std::strtod(gamma.c_str(), &end);
    return end != gamma.c_str() && *end == '\0' ? value : 0.0;
}

SVMModel ClassicML::FitSVC(const ClassicMatrix& data, const std::vector<int>& labels, int n_classes, double C,
                           const std::string& kernel, double gamma, int degree, double coef0, double tol) {
    SVMModel model;
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
        model.error_message = "An SVM classifier needs at least two classes";
        return model;
    }
    if (!(C > 0.0) || !(tol > 0.0)) {
        model.error_message = "C and tol must be positive";
        return model;
    }
    if (const std::string refusal = KernelRefusal(kernel, gamma, degree); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    constexpr const char* kOperation = "ClassicML::FitSVC";
    try {
        const af::array x = MatrixToDevice(data);
        const Kernel k{kernel, gamma, degree, coef0};
        std::vector<std::vector<int>> by_class(static_cast<size_t>(n_classes));
        for (size_t r = 0; r < labels.size(); ++r) by_class[static_cast<size_t>(labels[r])].push_back(static_cast<int>(r));
        std::vector<int> support_of_row(data.size(), -1);
        model.converged = true;
        for (int a = 0; a < n_classes; ++a) {
            for (int b = a + 1; b < n_classes; ++b) {
                std::vector<int> rows = by_class[static_cast<size_t>(a)];
                rows.insert(rows.end(), by_class[static_cast<size_t>(b)].begin(), by_class[static_cast<size_t>(b)].end());
                std::vector<double> y(rows.size(), -1.0);
                std::fill(y.begin(), y.begin() + static_cast<long long>(by_class[static_cast<size_t>(a)].size()), 1.0);
                const af::array sub = Rows(x, rows);
                const af::array y_dev = HostValuesToDevice(y).as(x.type());
                DualProblem prob{sub, y_dev, af::constant(-1.0, static_cast<dim_t>(y.size()), x.type()), y, C, k};
                const DualSolution solution = SolveDual(prob, tol, kOperation);
                model.converged = model.converged && solution.converged;
                model.iterations += solution.iterations;
                std::vector<int> support;
                std::vector<double> coef;
                KeepSupport(solution, rows, rows.size(), support, coef, y);
                for (int& row : support) {
                    if (support_of_row[static_cast<size_t>(row)] < 0) {
                        support_of_row[static_cast<size_t>(row)] = static_cast<int>(model.support_vectors.size());
                        model.support_vectors.push_back(data[static_cast<size_t>(row)]);
                    }
                    row = support_of_row[static_cast<size_t>(row)];
                }
                model.pair_first.push_back(a);
                model.pair_second.push_back(b);
                model.pair_support.push_back(std::move(support));
                model.pair_coef.push_back(std::move(coef));
                model.rho.push_back(solution.rho);
            }
        }
        model.kernel = kernel;
        model.gamma = gamma;
        model.degree = degree;
        model.coef0 = coef0;
        model.n_classes = n_classes;
        model.success = true;
    } catch (const Refusal& refusal) {
        model = SVMModel{};
        model.error_message = refusal.what();
    } catch (const af::exception& e) {
        model = SVMModel{};
        model.error_message = DeviceError(kOperation, e);
    }
    return model;
}

SVMModel ClassicML::FitSVR(const ClassicMatrix& data, const std::vector<double>& target, double C, double epsilon,
                           const std::string& kernel, double gamma, int degree, double coef0, double tol) {
    SVMModel model;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    if (target.size() != data.size()) {
        model.error_message = "Targets and rows differ in number";
        return model;
    }
    for (const double v : target) {
        if (!std::isfinite(v)) {
            model.error_message = "Targets must be finite numbers";
            return model;
        }
    }
    if (!(C > 0.0) || !(tol > 0.0) || !(epsilon >= 0.0)) {
        model.error_message = "C and tol must be positive and epsilon non-negative";
        return model;
    }
    if (const std::string refusal = KernelRefusal(kernel, gamma, degree); !refusal.empty()) {
        model.error_message = refusal;
        return model;
    }
    constexpr const char* kOperation = "ClassicML::FitSVR";
    try {
        const af::array x = MatrixToDevice(data);
        const size_t l = data.size();
        // libsvm solve_epsilon_svr: variables a (y = +1, p = eps - t) then a* (y = -1, p = eps + t).
        std::vector<double> y(2 * l), p(2 * l);
        for (size_t r = 0; r < l; ++r) {
            y[r] = 1.0;
            p[r] = epsilon - target[r];
            y[r + l] = -1.0;
            p[r + l] = epsilon + target[r];
        }
        const af::array y_dev = HostValuesToDevice(y).as(x.type());
        const af::array p_dev = HostValuesToDevice(p).as(x.type());
        DualProblem prob{x, y_dev, p_dev, y, C, Kernel{kernel, gamma, degree, coef0}};
        const DualSolution solution = SolveDual(prob, tol, kOperation);
        std::vector<int> rows(l);
        for (size_t r = 0; r < l; ++r) rows[r] = static_cast<int>(r);
        std::vector<int> support;
        std::vector<double> coef;
        KeepSupport(solution, rows, l, support, coef, y);
        for (int& row : support) {
            model.support_vectors.push_back(data[static_cast<size_t>(row)]);
            row = static_cast<int>(model.support_vectors.size()) - 1;
        }
        model.pair_support.push_back(std::move(support));
        model.pair_coef.push_back(std::move(coef));
        model.rho.push_back(solution.rho);
        model.iterations = solution.iterations;
        model.converged = solution.converged;
        model.kernel = kernel;
        model.gamma = gamma;
        model.degree = degree;
        model.coef0 = coef0;
        model.success = true;
    } catch (const Refusal& refusal) {
        model = SVMModel{};
        model.error_message = refusal.what();
    } catch (const af::exception& e) {
        model = SVMModel{};
        model.error_message = DeviceError(kOperation, e);
    }
    return model;
}

ClassicMatrix ClassicML::SVMDecision(const SVMModel& model, const ClassicMatrix& data, std::string* error) {
    if (!model.success || model.rho.empty() || model.pair_coef.size() != model.rho.size()) {
        if (error) *error = "The SVM model is not fitted";
        return {};
    }
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        if (error) *error = refusal;
        return {};
    }
    if (!model.support_vectors.empty() && data.front().size() != model.support_vectors.front().size()) {
        if (error)
            *error = "Rows have " + std::to_string(data.front().size()) + " features; the model was fitted on " +
                     std::to_string(model.support_vectors.front().size());
        return {};
    }
    constexpr const char* kOperation = "ClassicML::SVMDecision";
    try {
        const af::array x = MatrixToDevice(data);
        const dim_t n = x.dims(0);
        const size_t problems = model.rho.size();
        af::array decision = af::constant(0.0, n, static_cast<dim_t>(problems), x.type());
        if (!model.support_vectors.empty()) {
            const af::array sv = MatrixToDevice(model.support_vectors).as(x.type());
            const Kernel k{model.kernel, model.gamma, model.degree, model.coef0};
            // Coefficients [n_sv, problems]: column p holds problem p's y alpha.
            std::vector<double> coef(model.support_vectors.size() * problems, 0.0);
            for (size_t p = 0; p < problems; ++p)
                for (size_t s = 0; s < model.pair_support[p].size(); ++s)
                    coef[p * model.support_vectors.size() + static_cast<size_t>(model.pair_support[p][s])] =
                        model.pair_coef[p][s];
            const af::array coef_dev =
                af::moddims(HostValuesToDevice(coef), static_cast<dim_t>(model.support_vectors.size()),
                            static_cast<dim_t>(problems))
                    .as(x.type());
            const dim_t block = RowBlock(static_cast<dim_t>(model.support_vectors.size()));
            for (dim_t start = 0; start < n; start += block) {
                const dim_t end = std::min(n, start + block);
                const af::seq rows(static_cast<double>(start), static_cast<double>(end - 1));
                decision(rows, af::span) = af::matmul(KernelBlock(x(rows, af::span), sv, k), coef_dev);
            }
        }
        const af::array rho = HostValuesToDevice(model.rho).as(x.type()).T();
        decision -= af::tile(rho, static_cast<unsigned>(n), 1);
        return MatrixFromDevice(decision, kOperation);
    } catch (const Refusal& refusal) {
        if (error) *error = refusal.what();
    } catch (const af::exception& e) {
        if (error) *error = DeviceError(kOperation, e);
    }
    return {};
}

ClassicPrediction ClassicML::PredictSVC(const SVMModel& model, const ClassicMatrix& data) {
    ClassicPrediction result;
    if (model.n_classes < 2 || model.pair_first.size() != model.rho.size()) {
        result.error_message = "The SVM classifier is not fitted";
        return result;
    }
    const ClassicMatrix decision = SVMDecision(model, data, &result.error_message);
    if (decision.empty()) return result;
    constexpr const char* kOperation = "ClassicML::PredictSVC";
    try {
        // Votes [n, k] on the device: a pair's positive decision votes for its first class.
        const af::array dec = MatrixToDevice(decision);
        const dim_t n = dec.dims(0);
        af::array votes = af::constant(0.0, n, model.n_classes, dec.type());
        for (size_t p = 0; p < model.rho.size(); ++p) {
            const af::array first_wins = (dec(af::span, static_cast<int>(p)) > 0.0).as(dec.type());
            votes(af::span, model.pair_first[p]) += first_wins;
            votes(af::span, model.pair_second[p]) += 1.0 - first_wins;
        }
        af::array peak, labels;
        af::max(peak, labels, votes, 1);
        result.labels = LabelsFromDevice(labels, kOperation);
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

std::vector<double> ClassicML::PredictSVR(const SVMModel& model, const ClassicMatrix& data, std::string* error) {
    if (model.n_classes != 0 || model.rho.size() != 1) {
        if (error) *error = "The SVM regressor is not fitted";
        return {};
    }
    const ClassicMatrix decision = SVMDecision(model, data, error);
    std::vector<double> out;
    out.reserve(decision.size());
    for (const auto& row : decision) out.push_back(row.front());
    return out;
}

}  // namespace cyxwiz
