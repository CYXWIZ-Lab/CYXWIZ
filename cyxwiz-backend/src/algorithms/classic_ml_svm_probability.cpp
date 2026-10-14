// SVC probabilities (TOFIX145), as scikit-learn's SVC(probability=True), i.e.
// the libsvm vendored in scikit-learn 1.8 (svm.cpp: svm_binary_svc_probability,
// sigmoid_train, sigmoid_predict, multiclass_probability; newrand.h):
// - per class pair, the rows of the first class then the second, shuffled by
//   libsvm's generator and cut into 5 folds; a fold's decision values come from
//   an SVM fitted on the other four (on the device, ClassicML::FitSVC);
// - Platt's sigmoid fitted to those values by Newton's method with backtracking;
// - prediction couples the pairwise probabilities (Wu, Lin and Weng, method 2),
//   for all rows at once on the device.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "classic_ml_svm_probability.h"

#include "analysis_device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

using namespace analysis_detail;

namespace {

constexpr int kFolds = 5;

// libsvm's bounded_rand_int: a mt19937 draw mapped to [0, range) by Lemire's
// method with rejection (scikit-learn's newrand.h).
uint32_t BoundedRandInt(std::mt19937& generator, uint32_t range) {
    uint32_t x = static_cast<uint32_t>(generator());
    uint64_t m = static_cast<uint64_t>(x) * static_cast<uint64_t>(range);
    uint32_t low = static_cast<uint32_t>(m);
    if (low < range) {
        uint32_t threshold = 0u - range;
        if (threshold >= range) {
            threshold -= range;
            if (threshold >= range) threshold %= range;
        }
        while (low < threshold) {
            x = static_cast<uint32_t>(generator());
            m = static_cast<uint64_t>(x) * static_cast<uint64_t>(range);
            low = static_cast<uint32_t>(m);
        }
    }
    return static_cast<uint32_t>(m >> 32);
}

// Platt's negative log-likelihood of sigmoid (a, b) for targets t, written so no
// exp overflows (sigmoid_train's fval).
double SigmoidObjective(const af::array& dec, const af::array& t, double a, double b) {
    const af::array f = dec * a + b;
    const af::array value = af::select(f >= 0.0, t * f + af::log1p(af::exp(-f)), (t - 1.0) * f + af::log1p(af::exp(f)));
    return af::sum<double>(value);
}

// libsvm's sigmoid_train: prob_a, prob_b of P(+1 | d) = 1 / (1 + exp(a d + b)).
std::pair<double, double> SigmoidTrain(const af::array& dec, const std::vector<double>& y) {
    double prior1 = 0.0, prior0 = 0.0;
    for (const double v : y) (v > 0.0 ? prior1 : prior0) += 1.0;
    constexpr int kMaxIter = 100;
    constexpr double kMinStep = 1e-10, kSigma = 1e-12, kEps = 1e-5;
    const double hi = (prior1 + 1.0) / (prior1 + 2.0), lo = 1.0 / (prior0 + 2.0);
    std::vector<double> target(y.size());
    std::transform(y.begin(), y.end(), target.begin(), [&](double v) { return v > 0.0 ? hi : lo; });
    const af::array t = HostValuesToDevice(target).as(dec.type());

    double a = 0.0, b = std::log((prior0 + 1.0) / (prior1 + 1.0));
    double fval = SigmoidObjective(dec, t, a, b);
    for (int iter = 0; iter < kMaxIter; ++iter) {
        const af::array f = dec * a + b;
        const af::array e_neg = af::exp(-f), e_pos = af::exp(f);
        const af::array p = af::select(f >= 0.0, e_neg / (1.0 + e_neg), 1.0 / (1.0 + e_pos));
        const af::array q = af::select(f >= 0.0, 1.0 / (1.0 + e_neg), e_pos / (1.0 + e_pos));
        const af::array d2 = p * q, d1 = t - p;
        const double h11 = kSigma + af::sum<double>(dec * dec * d2);
        const double h22 = kSigma + af::sum<double>(d2);
        const double h21 = af::sum<double>(dec * d2);
        const double g1 = af::sum<double>(dec * d1);
        const double g2 = af::sum<double>(d1);
        if (std::abs(g1) < kEps && std::abs(g2) < kEps) break;

        const double det = h11 * h22 - h21 * h21;
        const double da = -(h22 * g1 - h21 * g2) / det;
        const double db = -(-h21 * g1 + h11 * g2) / det;
        const double gd = g1 * da + g2 * db;
        double step = 1.0;
        while (step >= kMinStep) {
            const double new_a = a + step * da, new_b = b + step * db;
            const double new_f = SigmoidObjective(dec, t, new_a, new_b);
            if (new_f < fval + 0.0001 * step * gd) {
                a = new_a;
                b = new_b;
                fval = new_f;
                break;
            }
            step /= 2.0;
        }
        if (step < kMinStep) break;  // libsvm: line search fails
    }
    return {a, b};
}

ClassicMatrix SelectRows(const ClassicMatrix& data, const std::vector<int>& rows) {
    ClassicMatrix out;
    out.reserve(rows.size());
    for (const int r : rows) out.push_back(data[static_cast<size_t>(r)]);
    return out;
}

}  // namespace

unsigned ClassicML::SVCProbabilitySeed(unsigned random_state) {
    // numpy's RandomState(seed) is mt19937 seeded like std::mt19937, and
    // randint(2**31 - 1) masks each 32-bit draw to 31 bits, rejecting 2**31 - 1.
    std::mt19937 generator(random_state);
    for (;;) {
        const unsigned value = static_cast<unsigned>(generator()) & 0x7fffffffu;
        if (value != 0x7fffffffu) return value;
    }
}

bool ClassicML::FitSVCProbability(SVMModel& model, const ClassicMatrix& data, const std::vector<int>& labels,
                                  double C, double tol, unsigned seed, std::string* error) {
    const auto fail = [&](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (!model.success || model.n_classes < 2 || model.rho.size() != model.pair_first.size())
        return fail("The SVM classifier is not fitted");
    if (labels.size() != data.size()) return fail("Labels and rows differ in number");
    constexpr const char* kOperation = "ClassicML::FitSVCProbability";
    try {
        std::vector<std::vector<int>> by_class(static_cast<size_t>(model.n_classes));
        for (size_t r = 0; r < labels.size(); ++r) {
            if (labels[r] < 0 || labels[r] >= model.n_classes) return fail("A label is outside 0..n_classes-1");
            by_class[static_cast<size_t>(labels[r])].push_back(static_cast<int>(r));
        }
        std::vector<double> prob_a, prob_b;
        for (size_t p = 0; p < model.rho.size(); ++p) {
            const auto& first = by_class[static_cast<size_t>(model.pair_first[p])];
            const auto& second = by_class[static_cast<size_t>(model.pair_second[p])];
            std::vector<int> rows = first;
            rows.insert(rows.end(), second.begin(), second.end());
            const int l = static_cast<int>(rows.size());
            std::vector<double> y(rows.size(), -1.0);
            std::fill(y.begin(), y.begin() + static_cast<long long>(first.size()), 1.0);

            // Every fold's SVM re-seeds libsvm's generator, so each pair shuffles
            // from the seed itself.
            std::mt19937 generator(seed);
            std::vector<int> perm(rows.size());
            std::iota(perm.begin(), perm.end(), 0);
            for (int i = 0; i < l; ++i) {
                const int j = i + static_cast<int>(BoundedRandInt(generator, static_cast<uint32_t>(l - i)));
                std::swap(perm[static_cast<size_t>(i)], perm[static_cast<size_t>(j)]);
            }

            std::vector<double> dec(rows.size(), 0.0);
            for (int fold = 0; fold < kFolds; ++fold) {
                const int begin = fold * l / kFolds, end = (fold + 1) * l / kFolds;
                std::vector<int> train_rows, train_labels, test_rows;
                for (int j = 0; j < l; ++j) {
                    const int local = perm[static_cast<size_t>(j)];
                    const int row = rows[static_cast<size_t>(local)];
                    if (j >= begin && j < end) {
                        test_rows.push_back(row);
                    } else {
                        train_rows.push_back(row);
                        train_labels.push_back(y[static_cast<size_t>(local)] > 0.0 ? 0 : 1);
                    }
                }
                const auto positives = std::count(train_labels.begin(), train_labels.end(), 0);
                const auto negatives = static_cast<long long>(train_labels.size()) - positives;
                std::vector<double> fold_dec;
                if (positives == 0 || negatives == 0) {
                    // libsvm: 0 with no rows, +1 with only the first class, -1 with only the second.
                    const double value = positives > 0 ? 1.0 : (negatives > 0 ? -1.0 : 0.0);
                    fold_dec.assign(test_rows.size(), value);
                } else {
                    const SVMModel sub = FitSVC(SelectRows(data, train_rows), train_labels, 2, C, model.kernel,
                                                model.gamma, model.degree, model.coef0, tol);
                    if (!sub.success) return fail("Probability fold: " + sub.error_message);
                    std::string decision_error;
                    const ClassicMatrix decision = SVMDecision(sub, SelectRows(data, test_rows), &decision_error);
                    if (decision.size() != test_rows.size()) return fail("Probability fold: " + decision_error);
                    for (const auto& row : decision) fold_dec.push_back(row.front());
                }
                for (int j = begin; j < end; ++j)
                    dec[static_cast<size_t>(perm[static_cast<size_t>(j)])] = fold_dec[static_cast<size_t>(j - begin)];
            }
            const auto [a, b] = SigmoidTrain(HostValuesToDevice(dec), y);
            prob_a.push_back(a);
            prob_b.push_back(b);
        }
        model.prob_a = std::move(prob_a);
        model.prob_b = std::move(prob_b);
        return true;
    } catch (const Refusal& refusal) {
        return fail(refusal.what());
    } catch (const af::exception& e) {
        return fail(DeviceError(kOperation, e));
    }
}

namespace classic_detail {

af::array SVCProbabilities(const af::array& decision, const SVMModel& model) {
    const dim_t n = decision.dims(0);
    const int k = model.n_classes;
    constexpr double kMinProb = 1e-7;
    // r[i][j] = P(i | i or j): Platt's sigmoid of the pair's decision, clipped.
    std::vector<std::vector<af::array>> r(static_cast<size_t>(k), std::vector<af::array>(static_cast<size_t>(k)));
    for (size_t p = 0; p < model.rho.size(); ++p) {
        const af::array f = decision(af::span, static_cast<int>(p)) * model.prob_a[p] + model.prob_b[p];
        const af::array sigmoid = af::select(f >= 0.0, af::exp(-f) / (1.0 + af::exp(-f)), 1.0 / (1.0 + af::exp(f)));
        const af::array clipped = af::min(af::max(sigmoid, kMinProb), 1.0 - kMinProb);
        const auto i = static_cast<size_t>(model.pair_first[p]), j = static_cast<size_t>(model.pair_second[p]);
        r[i][j] = clipped;
        r[j][i] = 1.0 - clipped;
    }
    // Q as libsvm builds it: Q[t][t] = sum_{j != t} r[j][t]^2, Q[t][j] = -r[j][t] r[t][j].
    std::vector<std::vector<af::array>> q(static_cast<size_t>(k), std::vector<af::array>(static_cast<size_t>(k)));
    for (size_t t = 0; t < static_cast<size_t>(k); ++t) {
        q[t][t] = af::constant(0.0, n, decision.type());
        for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
            if (j == t) continue;
            q[t][t] += r[j][t] * r[j][t];
            q[t][j] = -r[j][t] * r[t][j];
        }
    }
    std::vector<af::array> prob(static_cast<size_t>(k), af::constant(1.0 / k, n, decision.type()));
    // Rows that met the stopping rule stop changing, as libsvm's per-row break.
    af::array active = af::constant(1, n, b8);
    const int max_iter = std::max(100, k);
    const double eps = 0.005 / k;
    for (int iter = 0; iter < max_iter; ++iter) {
        std::vector<af::array> qp(static_cast<size_t>(k));
        af::array pqp = af::constant(0.0, n, decision.type());
        for (size_t t = 0; t < static_cast<size_t>(k); ++t) {
            qp[t] = af::constant(0.0, n, decision.type());
            for (size_t j = 0; j < static_cast<size_t>(k); ++j) qp[t] += q[t][j] * prob[j];
            pqp += prob[t] * qp[t];
        }
        af::array max_error = af::constant(0.0, n, decision.type());
        for (size_t t = 0; t < static_cast<size_t>(k); ++t) max_error = af::max(max_error, af::abs(qp[t] - pqp));
        active = active && (max_error >= eps);
        if (!af::anyTrue<bool>(active)) break;
        for (size_t t = 0; t < static_cast<size_t>(k); ++t) {
            const af::array diff = af::select(active, (-qp[t] + pqp) / q[t][t], 0.0);
            prob[t] += diff;
            pqp = (pqp + diff * (diff * q[t][t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
            for (size_t j = 0; j < static_cast<size_t>(k); ++j) {
                qp[j] = (qp[j] + diff * q[t][j]) / (1.0 + diff);
                prob[j] /= (1.0 + diff);
            }
            for (size_t j = 0; j < static_cast<size_t>(k); ++j) prob[j].eval();
            pqp.eval();
        }
    }
    af::array out = af::constant(0.0, n, k, decision.type());
    for (size_t t = 0; t < static_cast<size_t>(k); ++t) out(af::span, static_cast<int>(t)) = prob[t];
    return out;
}

}  // namespace classic_detail

}  // namespace cyxwiz
