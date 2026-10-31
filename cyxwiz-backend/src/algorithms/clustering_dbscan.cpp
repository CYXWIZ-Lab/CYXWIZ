// DBSCAN and the k-distance graph on ArrayFire (TOFIX140). Clusters are the
// connected components of the core points, found by min-label propagation with
// pointer jumping; numbering and border assignment reproduce scikit-learn's
// index-order expansion.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/clustering.h"
#include "analysis_device.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <string>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

using namespace analysis_detail;

namespace {

// For every point, the smallest `labels` value among its core neighbours
// (`none` when it has none). adjacency is [n, n] b8, labels [n, 1] s32.
af::array MinCoreNeighbourLabel(const af::array& adjacency, const af::array& core, const af::array& labels, int none) {
    const dim_t n = adjacency.dims(0);
    const dim_t block = RowBlock(n);
    const af::array candidate_row = af::select(core, labels, static_cast<double>(none)).T();  // [1, n]
    af::array result = af::constant(none, n, s32);
    for (dim_t start = 0; start < n; start += block) {
        const dim_t stop = std::min(n, start + block);
        const af::seq rows(static_cast<double>(start), static_cast<double>(stop - 1));
        const af::array candidates = af::tile(candidate_row, static_cast<unsigned>(stop - start), 1);
        const af::array masked = af::select(adjacency(rows, af::span), candidates, static_cast<double>(none));
        result(rows) = af::min(masked, 1);
    }
    return result;
}

}  // namespace

DBSCANResult Clustering::DBSCAN(
    const std::vector<std::vector<double>>& data,
    double eps,
    int min_samples,
    const std::string& metric
) {
    DBSCANResult result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    if (!(eps > 0.0) || min_samples < 1) {
        result.error_message = "eps must be positive and min_samples at least 1";
        return result;
    }
    if (!IsKnownMetric(metric)) {
        result.error_message = "Unknown metric '" + metric + "' (euclidean, manhattan or cosine)";
        return result;
    }
    try {
        const af::array x = MatrixToDevice(data);
        const dim_t n = x.dims(0);
        const int none = static_cast<int>(n);
        const double none_value = static_cast<double>(none);

        // Neighbourhoods (distance <= eps, the point itself included), in row blocks.
        af::array adjacency = af::constant(0, n, n, b8);
        const dim_t block = RowBlock(n);
        for (dim_t start = 0; start < n; start += block) {
            const dim_t stop = std::min(n, start + block);
            const af::seq rows(static_cast<double>(start), static_cast<double>(stop - 1));
            adjacency(rows, af::span) = PairwiseDistances(x(rows, af::span), x, metric) <= eps;
        }
        const af::array core = af::count(adjacency, 1) >= static_cast<unsigned>(min_samples);

        // Component of each core point = its lowest core index: propagate the minimum
        // label over core-core edges, jumping labels to their label's label each round.
        const af::array index = af::range(af::dim4(n), 0, s32);
        af::array labels = af::select(core, index, none_value);
        for (dim_t round = 0; round <= n; ++round) {
            af::array next = af::min(labels, MinCoreNeighbourLabel(adjacency, core, labels, none));
            next = af::select(core, af::lookup(next, af::min(next, af::constant(none - 1, next.dims(), s32))), none_value);
            next.eval();
            const bool changed = af::anyTrue<bool>(next != labels);
            labels = next;
            if (!changed) break;
        }
        // Border points take the earliest (lowest-root) cluster among their core neighbours.
        const af::array assigned = af::select(core, labels, MinCoreNeighbourLabel(adjacency, core, labels, none));
        // Number clusters 0, 1, ... in order of their root index.
        const af::array roots = (core && labels == index).as(s32);
        const af::array rank = af::accum(roots) - 1;
        const af::array safe = af::min(assigned, af::constant(none - 1, assigned.dims(), s32));
        const af::array final_labels = af::select(assigned < none, af::lookup(rank, safe), -1.0);

        result.labels = LabelsFromDevice(final_labels, "Clustering::DBSCAN");
        const std::vector<int> core_flags = LabelsFromDevice(core, "Clustering::DBSCAN");
        result.core_samples.assign(core_flags.begin(), core_flags.end());
        result.n_clusters = static_cast<int>(af::sum<double>(roots));
        result.n_noise_points = static_cast<int>(std::count(result.labels.begin(), result.labels.end(), -1));
        result.success = true;
    } catch (const af::exception& e) {
        result = DBSCANResult{};
        result.error_message = DeviceError("Clustering::DBSCAN", e);
    }
    return result;
}

std::vector<double> Clustering::ComputeKDistances(
    const std::vector<std::vector<double>>& data,
    int k
) {
    return ComputeKDistances(data, k, "euclidean");
}

std::vector<double> Clustering::ComputeKDistances(
    const std::vector<std::vector<double>>& data,
    int k,
    const std::string& metric
) {
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty() || k < 1 || data.size() < 2) {
        spdlog::error("Clustering::ComputeKDistances: {}", refusal.empty() ? "needs k >= 1 and 2 samples" : refusal);
        return {};
    }
    if (!IsKnownMetric(metric)) {
        spdlog::error("Clustering::ComputeKDistances: unknown metric '{}' (euclidean, manhattan or cosine)", metric);
        return {};
    }
    try {
        const af::array x = MatrixToDevice(data);
        const dim_t n = x.dims(0);
        const int column = static_cast<int>(std::min<dim_t>(k, n - 1));  // column 0 of a sorted row is the point itself
        af::array kth = af::constant(0.0, n, x.type());
        const dim_t block = RowBlock(n);
        for (dim_t start = 0; start < n; start += block) {
            const dim_t stop = std::min(n, start + block);
            const af::seq rows(static_cast<double>(start), static_cast<double>(stop - 1));
            const af::array sorted = af::sort(PairwiseDistances(x(rows, af::span), x, metric), 1);
            kth(rows) = sorted(af::span, column);
        }
        return DeviceToHostValues(af::sort(kth), "Clustering::ComputeKDistances");
    } catch (const af::exception& e) {
        spdlog::error("{}", DeviceError("Clustering::ComputeKDistances", e));
        return {};
    }
}

} // namespace cyxwiz
