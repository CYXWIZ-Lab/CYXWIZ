// Clustering (the clustering panels and nodes) against PyTorch float64 references of
// scikit-learn's KMeans / DBSCAN / GaussianMixture / cluster metrics and scipy's
// linkage / fcluster / cophenet (TOFIX140, clustering on the device).
//
// fixtures/clustering_pytorch.json (generate_clustering_fixtures.py). Runs on the
// active ArrayFire backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu). Random
// starts (K-Means, GMM) are checked against the fixed point of well-separated blobs.
#include "test_device_selection.h"

#include <cyxwiz/clustering.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using Matrix = std::vector<std::vector<double>>;
using cyxwiz::Clustering;
using cyxwiz::test::DeviceTolerance;

// float64 rounding, or float32 rounding on a device without float64 (worst
// measured on the Intel Iris Xe, TOFIX145 item 7: 4.9e-7).
double Tolerance() { return DeviceTolerance(1e-8, 5e-5); }
// EM stopped at a 1e-12 likelihood change.
double EmTolerance() { return DeviceTolerance(1e-5, 1e-4); }

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void CheckValue(double actual, double expected, const std::string& what, double tolerance = Tolerance()) {
    Check(std::abs(actual - expected) <= tolerance * (1.0 + std::abs(expected)),
          what + ": " + std::to_string(actual) + ", reference " + std::to_string(expected));
}

void CheckMatrix(const Matrix& actual, const Matrix& expected, const std::string& what, double tolerance = Tolerance()) {
    Check(actual.size() == expected.size(), what + ": rows");
    for (size_t r = 0; r < expected.size(); ++r) {
        Check(actual[r].size() == expected[r].size(), what + ": cols");
        for (size_t c = 0; c < expected[r].size(); ++c) {
            CheckValue(actual[r][c], expected[r][c], what + "(" + std::to_string(r) + "," + std::to_string(c) + ")", tolerance);
        }
    }
}

// actual -> expected label map when the two labelings are the same partition.
std::map<int, int> SamePartition(const std::vector<int>& actual, const std::vector<int>& expected, const std::string& what) {
    Check(actual.size() == expected.size(), what + ": label count");
    std::map<int, int> forward, backward;
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto f = forward.try_emplace(actual[i], expected[i]).first;
        const auto b = backward.try_emplace(expected[i], actual[i]).first;
        Check(f->second == expected[i] && b->second == actual[i], what + ": different partition at sample " + std::to_string(i));
    }
    return forward;
}

void CheckRefused(const std::string& error, const json& c, const std::string& what) {
    const std::string message = c.at("message").get<std::string>();
    Check(error.find(message) != std::string::npos,
          what + ": expected a refusal containing '" + message + "', got '" + error + "'");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" / "clustering_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_CLUSTERING_FIXTURE);
}

void RunCase(const json& c, const std::string& name) {
    const std::string op = c.at("op").get<std::string>();
    const Matrix data = c.at("data").get<Matrix>();

    if (op == "kmeans") {
        const int k = c.at("k").get<int>();
        const auto result = Clustering::KMeans(data, k, 300, c.at("init").get<std::string>(), 10, 1e-4, 7);
        Check(result.success, name + ": " + result.error_message);
        const auto mapping = SamePartition(result.labels, c.at("truth").get<std::vector<int>>(), name);
        const Matrix expected = c.at("centroids").get<Matrix>();
        for (const auto& [ours, theirs] : mapping) {
            for (size_t f = 0; f < expected[0].size(); ++f) {
                CheckValue(result.centroids[static_cast<size_t>(ours)][f], expected[static_cast<size_t>(theirs)][f], name + " centroid");
            }
        }
        CheckValue(result.inertia, c.at("inertia").get<double>(), name + " inertia");
        Check(result.converged && !result.inertia_history.empty() &&
                  static_cast<int>(result.inertia_history.size()) == result.n_iterations,
              name + ": converged with one inertia per iteration");
        for (size_t i = 1; i < result.inertia_history.size(); ++i) {
            Check(result.inertia_history[i] <= result.inertia_history[i - 1] * (1.0 + DeviceTolerance(1e-12, 1e-6)),
                  name + ": inertia rose");
        }
        return;
    }
    if (op == "elbow") {
        const auto result = Clustering::ComputeElbowAnalysis(data, c.at("k_min").get<int>(), c.at("k_max").get<int>());
        Check(result.success, name + ": " + result.error_message);
        Check(result.k_values == std::vector<int>({2, 3, 4, 5, 6}) && result.silhouette_scores.size() == 5, name + ": k range");
        CheckValue(result.inertias[1], c.at("inertia_k3").get<double>(), name + " inertia at k = 3");
        Check(result.suggested_k == 3, name + ": suggested k " + std::to_string(result.suggested_k));
        return;
    }
    if (op == "dbscan") {
        const auto result = Clustering::DBSCAN(data, c.at("eps").get<double>(), c.at("min_samples").get<int>(),
                                               c.at("metric").get<std::string>());
        Check(result.success, name + ": " + result.error_message);
        const auto labels = c.at("labels").get<std::vector<int>>();
        Check(result.labels == labels, name + ": labels differ from scikit-learn's numbering");
        const auto core = c.at("core").get<std::vector<bool>>();
        Check(result.core_samples == core, name + ": core samples");
        Check(result.n_clusters == *std::max_element(labels.begin(), labels.end()) + 1, name + ": cluster count");
        Check(result.n_noise_points == static_cast<int>(std::count(labels.begin(), labels.end(), -1)), name + ": noise count");
        return;
    }
    if (op == "kdistances") {
        const auto result = Clustering::ComputeKDistances(data, c.at("k").get<int>());
        const auto expected = c.at("distances").get<std::vector<double>>();
        Check(result.size() == expected.size(), name + ": count");
        for (size_t i = 0; i < expected.size(); ++i) CheckValue(result[i], expected[i], name + "[" + std::to_string(i) + "]");
        return;
    }
    if (op == "hierarchical") {
        const int k = c.at("k").get<int>();
        const auto result = Clustering::Hierarchical(data, k, c.at("linkage").get<std::string>(), c.at("metric").get<std::string>());
        Check(result.success, name + ": " + result.error_message);
        CheckMatrix(result.linkage_matrix, c.at("linkage_matrix").get<Matrix>(), name + " linkage");
        CheckValue(result.cophenetic_correlation, c.at("cophenetic").get<double>(), name + " cophenetic correlation");
        SamePartition(result.labels, c.at("labels").get<std::vector<int>>(), name + " labels");
        const auto cut = Clustering::CutDendrogram(result.linkage_matrix, c.at("cut_height").get<double>(),
                                                   static_cast<int>(data.size()));
        SamePartition(cut, c.at("cut_labels").get<std::vector<int>>(), name + " cut");
        return;
    }
    if (op == "gmm") {
        const int k = c.at("k").get<int>();
        const auto result = Clustering::GMM(data, k, c.at("covariance_type").get<std::string>(), 2000, 1e-12, 3, 11);
        Check(result.success, name + ": " + result.error_message);
        const auto mapping = SamePartition(result.labels, c.at("labels").get<std::vector<int>>(), name + " labels");
        const Matrix means = c.at("means").get<Matrix>();
        const auto weights = c.at("weights").get<std::vector<double>>();
        const auto covariances = c.at("covariances").get<std::vector<Matrix>>();
        for (const auto& [ours, theirs] : mapping) {
            const auto o = static_cast<size_t>(ours), t = static_cast<size_t>(theirs);
            CheckMatrix({result.means[o]}, {means[t]}, name + " mean", EmTolerance());
            CheckValue(result.weights[o], weights[t], name + " weight", EmTolerance());
            CheckMatrix(result.covariances[o], covariances[t], name + " covariance", EmTolerance());
        }
        CheckValue(result.log_likelihood, c.at("log_likelihood").get<double>(), name + " log-likelihood", EmTolerance());
        CheckValue(result.bic, c.at("bic").get<double>(), name + " BIC", EmTolerance());
        CheckValue(result.aic, c.at("aic").get<double>(), name + " AIC", EmTolerance());
        Check(result.converged, name + ": converged");
        return;
    }
    if (op == "evaluate") {
        const auto labels = c.at("labels").get<std::vector<int>>();
        const auto metrics = Clustering::EvaluateClustering(data, labels);
        Check(metrics.success, name + ": " + metrics.error_message);
        CheckValue(metrics.silhouette_score, c.at("silhouette").get<double>(), name + " silhouette");
        CheckValue(Clustering::ComputeSilhouetteScore(data, labels), c.at("silhouette").get<double>(), name + " silhouette only");
        const auto per_sample = c.at("per_sample").get<std::vector<double>>();
        Check(metrics.per_sample_silhouette.size() == per_sample.size(), name + ": per-sample count");
        for (size_t i = 0; i < per_sample.size(); ++i) CheckValue(metrics.per_sample_silhouette[i], per_sample[i], name + " sample");
        const auto clusters = c.at("cluster_silhouettes").get<std::vector<double>>();
        Check(metrics.cluster_silhouettes.size() == clusters.size(), name + ": cluster count");
        for (size_t i = 0; i < clusters.size(); ++i) CheckValue(metrics.cluster_silhouettes[i], clusters[i], name + " cluster");
        CheckValue(metrics.davies_bouldin_index, c.at("davies_bouldin").get<double>(), name + " Davies-Bouldin");
        CheckValue(metrics.calinski_harabasz_score, c.at("calinski_harabasz").get<double>(), name + " Calinski-Harabasz");
        Check(metrics.n_clusters == c.at("n_clusters").get<int>() && metrics.n_samples == c.at("n_samples").get<int>(),
              name + ": counts");
        return;
    }
    if (op == "refuse_kmeans") return CheckRefused(Clustering::KMeans(data, 6).error_message, c, name);
    if (op == "refuse_hierarchical") return CheckRefused(Clustering::Hierarchical(data, 3, "ward", "manhattan").error_message, c, name);
    if (op == "refuse_dbscan") return CheckRefused(Clustering::DBSCAN(data, 1.0, 5, "chebyshev").error_message, c, name);
    if (op == "refuse_gmm") return CheckRefused(Clustering::GMM(data, 3, "banded").error_message, c, name);
    if (op == "refuse_evaluate") {
        CheckRefused(Clustering::EvaluateClustering(data, std::vector<int>(data.size(), 4)).error_message, c, name);
        return Check(std::isnan(Clustering::ComputeSilhouetteScore(data, std::vector<int>(data.size(), 4))), name + ": NaN");
    }
    Check(false, name + ": unknown op " + op);
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the clustering fixture");
    const json fixture = json::parse(in);
    size_t passed = 0;
    for (const auto& c : fixture.at("cases")) {
        const std::string name = c.at("op").get<std::string>() + " " + c.at("label").get<std::string>();
        try {
            RunCase(c, name);
        } catch (const std::exception& error) {
            Check(false, name + ": " + error.what());
        }
        ++passed;
    }
    std::cout << "clustering matches the references: " << passed << " cases" << std::endl;
    return 0;
}
