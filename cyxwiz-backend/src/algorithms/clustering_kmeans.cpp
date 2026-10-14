// K-Means and the elbow analysis on ArrayFire (TOFIX140): Lloyd iterations with a
// one-hot matmul centroid update, scikit-learn's centre-shift tolerance.

#ifdef _WIN32
#define NOMINMAX
#endif

#include "cyxwiz/clustering.h"
#include "analysis_device.h"
#include "arrayfire_indexed.h"

#include <cmath>
#include <limits>
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

struct KMeansRun {
    af::array labels;
    af::array centers;
    double inertia = 0.0;
    int iterations = 0;
    bool converged = false;
    std::vector<double> history;
};

// Labels [n, 1] (u32) of the nearest center and the total squared distance.
std::pair<af::array, double> Assign(const af::array& data, const af::array& centers) {
    af::array nearest, labels;
    MinWithIndex(nearest, labels, SquaredDistances(data, centers), 1);
    return {labels, af::sum<double>(nearest)};
}

KMeansRun RunLloyd(const af::array& data, af::array centers, int max_iter, double tolerance,
                   const std::function<void(int, double)>& progress) {
    const dim_t n = data.dims(0), d = data.dims(1), k = centers.dims(0);
    const af::array cluster_ids = af::tile(af::range(af::dim4(1, k), 1, u32), static_cast<unsigned>(n), 1);
    KMeansRun run;
    for (int iter = 0; iter < max_iter; ++iter) {
        const auto [labels, inertia] = Assign(data, centers);
        run.history.push_back(inertia);
        if (progress) progress(iter + 1, inertia);

        const af::array one_hot = (af::tile(labels, 1, static_cast<unsigned>(k)) == cluster_ids).as(data.type());
        const af::array counts = af::sum(one_hot, 0).T();  // [k, 1]
        const af::array sums = af::matmul(one_hot, data, AF_MAT_TRANS, AF_MAT_NONE);  // [k, d]
        const af::array occupied = af::tile(counts > 0.0, 1, static_cast<unsigned>(d));
        const af::array safe_counts = af::max(counts, af::constant(1.0, counts.dims(), counts.type()));
        const af::array updated = af::select(occupied, sums / af::tile(safe_counts, 1, static_cast<unsigned>(d)), centers);
        const af::array step = updated - centers;
        const double shift = af::sum<double>(step * step);
        centers = updated;
        centers.eval();
        run.iterations = iter + 1;
        if (shift <= tolerance) {
            run.converged = true;
            break;
        }
    }
    // Labels and inertia for the final centers (scikit-learn's closing E-step).
    const auto [labels, inertia] = Assign(data, centers);
    run.labels = labels;
    run.centers = centers;
    run.inertia = inertia;
    return run;
}

af::array RandomRows(const af::array& data, int k, af::randomEngine& engine) {
    af::array keys, order;
    SortWithIndex(keys, order, af::randu(data.dims(0), f32, engine));
    const af::array chosen = order(af::seq(0, k - 1));
    return data(chosen, af::span);
}

}  // namespace

KMeansResult Clustering::KMeans(
    const std::vector<std::vector<double>>& data,
    int n_clusters,
    int max_iter,
    const std::string& init,
    int n_init,
    double tol,
    unsigned int seed,
    std::function<void(int, double)> progress_callback
) {
    KMeansResult result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    if (n_clusters <= 0 || n_clusters > static_cast<int>(data.size())) {
        result.error_message = "Number of clusters must be between 1 and the number of samples";
        return result;
    }
    if (max_iter < 1 || n_init < 1 || !(tol >= 0.0)) {
        result.error_message = "max_iter and n_init must be at least 1, and tol non-negative";
        return result;
    }
    if (init != "kmeans++" && init != "random") {
        result.error_message = "Unknown init '" + init + "' (kmeans++ or random)";
        return result;
    }
    try {
        const af::array x = MatrixToDevice(data);
        const double tolerance = tol * af::mean<double>(af::var(x, AF_VARIANCE_POPULATION, 0));
        af::randomEngine engine = SeededEngine(seed);
        KMeansRun best;
        best.inertia = std::numeric_limits<double>::infinity();
        for (int run = 0; run < n_init; ++run) {
            const af::array start = init == "random" ? RandomRows(x, n_clusters, engine)
                                                     : KMeansPlusPlus(x, n_clusters, engine);
            KMeansRun current = RunLloyd(x, start, max_iter, tolerance,
                                         run == 0 ? progress_callback : std::function<void(int, double)>{});
            if (current.inertia < best.inertia) best = std::move(current);
        }
        result.labels = LabelsFromDevice(best.labels, "Clustering::KMeans");
        result.centroids = MatrixFromDevice(best.centers, "Clustering::KMeans");
        result.inertia = best.inertia;
        result.inertia_history = std::move(best.history);
        result.n_iterations = best.iterations;
        result.converged = best.converged;
        result.n_clusters = n_clusters;
        result.success = true;
    } catch (const af::exception& e) {
        result = KMeansResult{};
        result.error_message = DeviceError("Clustering::KMeans", e);
    }
    return result;
}

ElbowAnalysis Clustering::ComputeElbowAnalysis(
    const std::vector<std::vector<double>>& data,
    int k_min,
    int k_max,
    std::function<void(int, int)> progress_callback
) {
    ElbowAnalysis result;
    if (const std::string refusal = MatrixRefusal(data); !refusal.empty()) {
        result.error_message = refusal;
        return result;
    }
    k_max = std::min(k_max, static_cast<int>(data.size()) - 1);
    if (k_min < 2 || k_min > k_max) {
        result.error_message = "Needs 2 <= k_min <= k_max <= samples - 1";
        return result;
    }
    for (int k = k_min; k <= k_max; ++k) {
        if (progress_callback) progress_callback(k - k_min + 1, k_max - k_min + 1);
        const KMeansResult kmeans = KMeans(data, k, 100, "kmeans++", 3, 1e-4, 0, nullptr);
        if (!kmeans.success) {
            result.error_message = "k = " + std::to_string(k) + ": " + kmeans.error_message;
            return result;
        }
        const ClusterMetrics metrics = EvaluateClustering(data, kmeans.labels);
        result.k_values.push_back(k);
        result.inertias.push_back(kmeans.inertia);
        result.silhouette_scores.push_back(metrics.success ? metrics.silhouette_score
                                                           : std::numeric_limits<double>::quiet_NaN());
    }
    // Elbow: the k with the largest second difference of the inertia curve.
    result.suggested_k = result.k_values.front();
    double sharpest = 0.0;
    for (size_t i = 1; i + 1 < result.inertias.size(); ++i) {
        const double bend = std::abs(result.inertias[i - 1] - 2.0 * result.inertias[i] + result.inertias[i + 1]);
        if (bend > sharpest) {
            sharpest = bend;
            result.suggested_k = result.k_values[i];
        }
    }
    result.success = true;
    return result;
}

} // namespace cyxwiz
