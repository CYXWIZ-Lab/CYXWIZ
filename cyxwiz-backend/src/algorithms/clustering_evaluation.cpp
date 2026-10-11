// Cluster quality metrics on ArrayFire (TOFIX140), as scikit-learn's
// silhouette_samples / silhouette_score, davies_bouldin_score and
// calinski_harabasz_score. Silhouette distances are taken in row blocks against a
// one-hot cluster matrix, so no n x n matrix is held. Noise points (-1) are left out.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/clustering.h"
#include "clustering_device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <vector>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace cyxwiz {

using namespace clustering_detail;

ClusterMetrics Clustering::EvaluateClustering(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    ClusterMetrics result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    if (labels.size() != data.size()) {
        result.error_message = "Need one label per sample";
        return result;
    }
    // Clusters in ascending label order; codes 0 .. k-1 for the kept samples.
    std::map<int, int> code_of;
    for (int label : labels) {
        if (label != -1) code_of.emplace(label, 0);
    }
    int next_code = 0;
    for (auto& entry : code_of) entry.second = next_code++;
    std::vector<int> kept;
    std::vector<int> codes;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] == -1) continue;
        kept.push_back(static_cast<int>(i));
        codes.push_back(code_of.at(labels[i]));
    }
    const int k = static_cast<int>(code_of.size()), m = static_cast<int>(kept.size());
    if (k < 2 || k > m - 1) {
        result.error_message = "Needs between 2 and n - 1 clusters (noise excluded); found " + std::to_string(k);
        return result;
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    try {
        const af::array all = MatrixToDevice(data);
        const af::array x = all(af::array(m, kept.data()), af::span);  // [m, d]
        const dim_t d = x.dims(1);
        const af::array code = af::array(m, codes.data());
        const af::array one_hot =
            (af::tile(code, 1, static_cast<unsigned>(k)) == af::tile(af::range(af::dim4(1, k), 1, s32), static_cast<unsigned>(m), 1))
                .as(x.type());                                              // [m, k]
        const af::array counts = af::sum(one_hot, 0);                       // [1, k]

        // Silhouette: a = mean distance to the own cluster (n_own - 1), b = smallest
        // mean distance to another cluster; 0 for singleton clusters.
        af::array silhouette = af::constant(0.0, m, x.type());
        const dim_t block = RowBlock(m);
        const double inf = std::numeric_limits<double>::infinity();
        for (dim_t start = 0; start < m; start += block) {
            const dim_t stop = std::min<dim_t>(m, start + block);
            const af::seq rows(static_cast<double>(start), static_cast<double>(stop - 1));
            const unsigned b = static_cast<unsigned>(stop - start);
            const af::array totals = af::matmul(PairwiseDistances(x(rows, af::span), x, "euclidean"), one_hot);  // [b, k]
            const af::array own = one_hot(rows, af::span);
            const af::array block_counts = af::tile(counts, b, 1);
            const af::array own_count = af::sum(own * block_counts, 1);
            const af::array a = af::sum(own * totals, 1) / af::max(own_count - 1.0, af::constant(1.0, own_count.dims(), x.type()));
            const af::array other = af::min(af::select(own > 0.0, inf, totals / block_counts), 1);
            const af::array denominator = af::max(a, other);
            af::array s = af::select(denominator > 0.0, (other - a) / denominator, 0.0);
            s = af::select(own_count > 1.0, s, 0.0);
            silhouette(rows) = s;
        }
        const std::vector<double> samples = DeviceToHostValues(silhouette, "Clustering::EvaluateClustering");
        result.per_sample_silhouette.assign(data.size(), 0.0);
        for (int i = 0; i < m; ++i) result.per_sample_silhouette[static_cast<size_t>(kept[static_cast<size_t>(i)])] = samples[static_cast<size_t>(i)];
        result.silhouette_score = af::mean<double>(silhouette);
        result.cluster_silhouettes =
            DeviceToHostValues(af::matmul(one_hot, silhouette, AF_MAT_TRANS, AF_MAT_NONE).T() / counts,
                               "Clustering::EvaluateClustering");

        // Centroids, within-cluster distances and dispersions.
        const af::array centroids = af::matmul(one_hot, x, AF_MAT_TRANS, AF_MAT_NONE) / af::tile(counts.T(), 1, static_cast<unsigned>(d));
        const af::array residual = x - af::matmul(one_hot, centroids);
        const af::array squared = af::sum(residual * residual, 1);  // [m, 1]
        const af::array intra = af::matmul(one_hot, af::sqrt(squared), AF_MAT_TRANS, AF_MAT_NONE).T() / counts;  // [1, k]

        // Davies-Bouldin: mean over clusters of max_j (s_i + s_j) / |c_i - c_j|.
        const af::array centroid_distance = PairwiseDistances(centroids, centroids, "euclidean");  // [k, k]
        const bool degenerate = af::allTrue<bool>(af::abs(intra) <= 1e-8) ||
                                af::allTrue<bool>(af::abs(centroid_distance) <= 1e-8);
        if (degenerate) {
            result.davies_bouldin_index = 0.0;
        } else {
            const af::array combined = af::tile(intra.T(), 1, static_cast<unsigned>(k)) + af::tile(intra, static_cast<unsigned>(k), 1);
            const af::array safe = af::select(centroid_distance == 0.0, inf, centroid_distance);
            result.davies_bouldin_index = af::mean<double>(af::max(combined / safe, 1));
        }

        // Calinski-Harabasz: between / within dispersion, scaled by (m - k) / (k - 1).
        const af::array spread = centroids - af::tile(af::mean(x, 0), static_cast<unsigned>(k), 1);
        const double between = af::sum<double>(counts.T() * af::sum(spread * spread, 1));
        const double within = af::sum<double>(squared);
        result.calinski_harabasz_score = within == 0.0 ? 1.0 : between * (m - k) / (within * (k - 1.0));

        result.n_clusters = k;
        result.n_samples = m;
        result.success = true;
    } catch (const af::exception& e) {
        result = ClusterMetrics{};
        result.error_message = DeviceError("Clustering::EvaluateClustering", e);
    }
#else
    result.error_message = NoArrayFire("Clustering::EvaluateClustering");
#endif
    return result;
}

double Clustering::ComputeSilhouetteScore(
    const std::vector<std::vector<double>>& data,
    const std::vector<int>& labels
) {
    const ClusterMetrics metrics = EvaluateClustering(data, labels);
    return metrics.success ? metrics.silhouette_score : std::numeric_limits<double>::quiet_NaN();
}

} // namespace cyxwiz
