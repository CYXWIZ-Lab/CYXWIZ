#include "cyxwiz/calibration.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace cyxwiz {

namespace {

double Sigmoid(double z) { return z >= 0 ? 1.0 / (1.0 + std::exp(-z)) : std::exp(z) / (1.0 + std::exp(z)); }

}  // namespace

Reliability ReliabilityOf(const std::vector<double>& confidence, const std::vector<int>& correct, int bins) {
    Reliability r;
    bins = std::max(1, bins);
    r.bins.resize(static_cast<size_t>(bins));
    std::vector<double> conf_sum(bins, 0.0), right(bins, 0.0);
    const size_t n = std::min(confidence.size(), correct.size());
    for (int b = 0; b < bins; ++b) {
        r.bins[b].low = static_cast<double>(b) / bins;
        r.bins[b].high = static_cast<double>(b + 1) / bins;
    }
    for (size_t i = 0; i < n; ++i) {
        const int b = std::min(bins - 1, std::max(0, static_cast<int>(confidence[i] * bins)));
        ++r.bins[b].count;
        conf_sum[b] += confidence[i];
        right[b] += correct[i] ? 1.0 : 0.0;
    }
    r.rows = static_cast<int>(n);
    for (int b = 0; b < bins; ++b) {
        auto& bin = r.bins[b];
        if (bin.count == 0) continue;
        bin.confidence = conf_sum[b] / bin.count;
        bin.accuracy = right[b] / bin.count;
        r.ece += static_cast<double>(bin.count) / static_cast<double>(n) * std::fabs(bin.accuracy - bin.confidence);
    }
    return r;
}

std::vector<double> WithTemperature(const float* probabilities, int classes, double temperature) {
    std::vector<double> z(classes);
    double top = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < classes; ++c) {
        z[c] = std::log(std::max(1e-12, static_cast<double>(probabilities[c]))) / temperature;
        top = std::max(top, z[c]);
    }
    double total = 0.0;
    for (auto& v : z) total += (v = std::exp(v - top));
    for (auto& v : z) v /= total;
    return z;
}

TemperatureFit FitTemperature(const std::vector<float>& probabilities, int classes, const std::vector<int>& truth,
                              const std::vector<size_t>& rows) {
    const auto nll = [&](double t) {
        double sum = 0.0;
        for (size_t r : rows) {
            const auto p = WithTemperature(&probabilities[r * classes], classes, t);
            sum -= std::log(std::max(1e-300, p[static_cast<size_t>(truth[r])]));
        }
        return rows.empty() ? 0.0 : sum / static_cast<double>(rows.size());
    };
    // Golden-section search on log T in [log 0.05, log 20].
    double lo = std::log(0.05), hi = std::log(20.0);
    const double g = (std::sqrt(5.0) - 1.0) / 2.0;
    double x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
    double f1 = nll(std::exp(x1)), f2 = nll(std::exp(x2));
    while (hi - lo > 1e-9) {
        if (f1 < f2) {
            hi = x2;
            x2 = x1;
            f2 = f1;
            x1 = hi - g * (hi - lo);
            f1 = nll(std::exp(x1));
        } else {
            lo = x1;
            x1 = x2;
            f1 = f2;
            x2 = lo + g * (hi - lo);
            f2 = nll(std::exp(x2));
        }
    }
    TemperatureFit fit;
    fit.temperature = std::exp((lo + hi) / 2.0);
    fit.nll_before = nll(1.0);
    fit.nll_after = nll(fit.temperature);
    return fit;
}

PlattFit FitPlatt(const std::vector<double>& scores, const std::vector<int>& labels) {
    const size_t n = std::min(scores.size(), labels.size());
    double negatives = 0.0;
    for (size_t i = 0; i < n; ++i) negatives += labels[i] <= 0;
    const double positives = static_cast<double>(n) - negatives;
    // Platt's targets and start (scikit-learn's _sigmoid_calibration).
    const double hi = (positives + 1.0) / (positives + 2.0), lo = 1.0 / (negatives + 2.0);
    double a = 0.0, b = std::log((negatives + 1.0) / (positives + 1.0));
    for (int iteration = 0; iteration < 100; ++iteration) {
        // p = 1 / (1 + exp(a s + b)); loss = -sum t log p + (1 - t) log(1 - p).
        double ga = 0.0, gb = 0.0, haa = 0.0, hab = 0.0, hbb = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double t = labels[i] > 0 ? hi : lo;
            const double p = Sigmoid(-(a * scores[i] + b));
            const double d = t - p;  // d loss / d (a s + b)
            ga += d * scores[i];
            gb += d;
            const double w = p * (1.0 - p);
            haa += w * scores[i] * scores[i];
            hab += w * scores[i];
            hbb += w;
        }
        haa += 1e-12;
        hbb += 1e-12;
        const double det = haa * hbb - hab * hab;
        if (std::fabs(det) < 1e-300) break;
        const double da = (hbb * ga - hab * gb) / det, db = (haa * gb - hab * ga) / det;
        a -= da;
        b -= db;
        if (std::fabs(da) < 1e-12 && std::fabs(db) < 1e-12) break;
    }
    return {a, b};
}

double ApplyPlatt(const PlattFit& fit, double score) { return Sigmoid(-(fit.a * score + fit.b)); }

IsotonicFit FitIsotonic(const std::vector<double>& scores, const std::vector<int>& labels) {
    const size_t n = std::min(scores.size(), labels.size());
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) { return scores[i] < scores[j]; });
    // Tied scores: one point, the mean label, weighted by the count.
    std::vector<double> x, y, w;
    for (size_t k = 0; k < n; ++k) {
        const double s = scores[order[k]], v = labels[order[k]] > 0 ? 1.0 : 0.0;
        if (!x.empty() && x.back() == s) {
            y.back() = (y.back() * w.back() + v) / (w.back() + 1.0);
            w.back() += 1.0;
        } else {
            x.push_back(s);
            y.push_back(v);
            w.push_back(1.0);
        }
    }
    // Pool adjacent violators.
    std::vector<double> block_y, block_w;
    std::vector<size_t> block_end;
    for (size_t i = 0; i < x.size(); ++i) {
        block_y.push_back(y[i]);
        block_w.push_back(w[i]);
        block_end.push_back(i);
        while (block_y.size() >= 2 && block_y[block_y.size() - 2] > block_y.back()) {
            const double wsum = block_w[block_w.size() - 2] + block_w.back();
            const double merged = (block_y[block_y.size() - 2] * block_w[block_w.size() - 2] + block_y.back() * block_w.back()) / wsum;
            block_y.pop_back();
            block_w.pop_back();
            const size_t end = block_end.back();
            block_end.pop_back();
            block_y.back() = merged;
            block_w.back() = wsum;
            block_end.back() = end;
        }
    }
    std::vector<double> fitted(x.size());
    size_t start = 0;
    for (size_t b = 0; b < block_y.size(); ++b) {
        for (size_t i = start; i <= block_end[b]; ++i) fitted[i] = block_y[b];
        start = block_end[b] + 1;
    }
    // Keep only the points where the fitted value changes (the ends of each run).
    IsotonicFit fit;
    for (size_t i = 0; i < x.size(); ++i) {
        const bool keep = i == 0 || i + 1 == x.size() || fitted[i - 1] != fitted[i] || fitted[i] != fitted[i + 1];
        if (!keep) continue;
        fit.x.push_back(x[i]);
        fit.y.push_back(fitted[i]);
    }
    return fit;
}

double ApplyIsotonic(const IsotonicFit& fit, double score) {
    if (fit.x.empty()) return score;
    if (score <= fit.x.front()) return fit.y.front();
    if (score >= fit.x.back()) return fit.y.back();
    const size_t hi = static_cast<size_t>(std::upper_bound(fit.x.begin(), fit.x.end(), score) - fit.x.begin());
    const size_t lo = hi - 1;
    const double t = (score - fit.x[lo]) / (fit.x[hi] - fit.x[lo]);
    return fit.y[lo] + t * (fit.y[hi] - fit.y[lo]);
}

}  // namespace cyxwiz
