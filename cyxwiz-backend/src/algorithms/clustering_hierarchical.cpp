// Agglomerative clustering on ArrayFire (TOFIX140). The cluster distance matrix
// stays on the device: each step finds the closest pair, applies the Lance-Williams
// update to the merged row and retires the other. A second matrix keeps the sum of
// original distances between clusters, which gives the cophenetic correlation
// without building the cophenetic matrix.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/clustering.h"
#include "analysis_device.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
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

// Labels after the first `merges` rows of a linkage matrix, numbered in order of
// first appearance.
std::vector<int> LabelsAfterMerges(const std::vector<std::vector<double>>& linkage, size_t merges, int n_samples) {
    std::vector<int> parent(2 * static_cast<size_t>(n_samples));
    std::iota(parent.begin(), parent.end(), 0);
    const std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
    for (size_t i = 0; i < merges && i < linkage.size(); ++i) {
        const int merged = n_samples + static_cast<int>(i);
        parent[find(static_cast<int>(linkage[i][0]))] = merged;
        parent[find(static_cast<int>(linkage[i][1]))] = merged;
    }
    std::vector<int> labels(static_cast<size_t>(n_samples));
    std::map<int, int> label_of_root;
    for (int i = 0; i < n_samples; ++i) {
        const auto [it, inserted] = label_of_root.try_emplace(find(i), static_cast<int>(label_of_root.size()));
        labels[static_cast<size_t>(i)] = it->second;
    }
    return labels;
}

}  // namespace

HierarchicalResult Clustering::Hierarchical(
    const std::vector<std::vector<double>>& data,
    int n_clusters,
    const std::string& linkage,
    const std::string& metric
) {
    HierarchicalResult result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    const int n = static_cast<int>(data.size());
    if (n < 2 || n_clusters < 1 || n_clusters > n) {
        result.error_message = "Needs at least 2 samples and 1 to n clusters";
        return result;
    }
    if (linkage != "ward" && linkage != "complete" && linkage != "average" && linkage != "single") {
        result.error_message = "Unknown linkage '" + linkage + "' (ward, complete, average or single)";
        return result;
    }
    if (!IsKnownMetric(metric)) {
        result.error_message = "Unknown metric '" + metric + "' (euclidean, manhattan or cosine)";
        return result;
    }
    if (linkage == "ward" && metric != "euclidean") {
        result.error_message = "Ward linkage needs the euclidean metric";
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array x = MatrixToDevice(data);
        const double inf = std::numeric_limits<double>::infinity();
        af::array sums = af::constant(0.0, n, n, x.type());  // original distances, summed per cluster pair
        const dim_t block = RowBlock(n);
        for (dim_t start = 0; start < n; start += block) {
            const dim_t stop = std::min<dim_t>(n, start + block);
            const af::seq rows(static_cast<double>(start), static_cast<double>(stop - 1));
            sums(rows, af::span) = PairwiseDistances(x(rows, af::span), x, metric);
        }
        const af::array diagonal = af::identity(n, n, b8);
        sums = af::select(diagonal, 0.0, sums);
        af::array distance = af::select(diagonal, inf, sums);  // cluster distances; retired rows are inf

        // Totals over the n(n-1)/2 sample pairs for the cophenetic correlation.
        const double pairs = 0.5 * n * (n - 1.0);
        const double sum_d = 0.5 * af::sum<double>(sums);
        const double sum_d2 = 0.5 * af::sum<double>(sums * sums);
        double sum_c = 0.0, sum_c2 = 0.0, sum_cd = 0.0;

        const af::array slot = af::range(af::dim4(n), 0, s32);
        af::array active = af::constant(1, n, b8);
        af::array sizes = af::constant(1.0, n, x.type());
        std::vector<double> host_sizes(static_cast<size_t>(n), 1.0);
        std::vector<int> cluster_id(static_cast<size_t>(n));
        std::iota(cluster_id.begin(), cluster_id.end(), 0);

        result.linkage_matrix.reserve(static_cast<size_t>(n - 1));
        for (int step = 0; step < n - 1; ++step) {
            af::array closest, flat_index;
            af::min(closest, flat_index, af::flat(distance));
            const std::vector<double> picked = DeviceToHostValues(
                af::join(0, closest, af::lookup(af::flat(sums), flat_index)), "Clustering::Hierarchical");
            const unsigned flat = flat_index.scalar<unsigned>();
            const int i = static_cast<int>(flat % static_cast<unsigned>(n)), j = static_cast<int>(flat / static_cast<unsigned>(n));
            const int a = std::min(i, j), b = std::max(i, j);
            const double height = picked[0], cross_sum = picked[1];
            const double na = host_sizes[static_cast<size_t>(a)], nb = host_sizes[static_cast<size_t>(b)];

            const af::array da = distance(af::span, a), db = distance(af::span, b);
            af::array merged;
            if (linkage == "single") {
                merged = af::min(da, db);
            } else if (linkage == "complete") {
                merged = af::max(da, db);
            } else if (linkage == "average") {
                merged = (na * da + nb * db) / (na + nb);
            } else {  // ward
                merged = af::sqrt(((na + sizes) * da * da + (nb + sizes) * db * db - sizes * height * height) /
                                  (na + nb + sizes));
            }
            merged = af::select(active && slot != a && slot != b, merged, inf);
            distance(af::span, a) = merged;
            distance(a, af::span) = merged.T();
            distance(af::span, b) = inf;
            distance(b, af::span) = inf;
            const af::array merged_sums = sums(af::span, a) + sums(af::span, b);
            sums(af::span, a) = merged_sums;
            sums(a, af::span) = merged_sums.T();
            sizes(a) = na + nb;
            active(b) = 0;
            distance.eval();
            sums.eval();

            const double pair_count = na * nb;
            sum_c += pair_count * height;
            sum_c2 += pair_count * height * height;
            sum_cd += height * cross_sum;

            const int id_a = cluster_id[static_cast<size_t>(a)], id_b = cluster_id[static_cast<size_t>(b)];
            result.linkage_matrix.push_back({static_cast<double>(std::min(id_a, id_b)),
                                             static_cast<double>(std::max(id_a, id_b)), height, na + nb});
            cluster_id[static_cast<size_t>(a)] = n + step;
            host_sizes[static_cast<size_t>(a)] = na + nb;
        }

        const double covariance = pairs * sum_cd - sum_c * sum_d;
        const double spread = (pairs * sum_c2 - sum_c * sum_c) * (pairs * sum_d2 - sum_d * sum_d);
        result.cophenetic_correlation = spread > 0.0 ? covariance / std::sqrt(spread) : 0.0;
        result.labels = LabelsAfterMerges(result.linkage_matrix, static_cast<size_t>(n - n_clusters), n);
        result.n_clusters = n_clusters;
        result.success = true;
    } catch (const af::exception& e) {
        result = HierarchicalResult{};
        result.error_message = DeviceError("Clustering::Hierarchical", e);
    }
#else
    result.error_message = NoArrayFire("Clustering::Hierarchical");
#endif
    return result;
}

std::vector<int> Clustering::CutDendrogram(
    const std::vector<std::vector<double>>& linkage_matrix,
    double height,
    int n_samples
) {
    size_t merges = 0;
    while (merges < linkage_matrix.size() && linkage_matrix[merges][2] <= height) ++merges;
    return LabelsAfterMerges(linkage_matrix, merges, n_samples);
}

} // namespace cyxwiz
