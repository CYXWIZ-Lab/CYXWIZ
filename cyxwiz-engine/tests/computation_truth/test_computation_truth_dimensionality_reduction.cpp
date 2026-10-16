// DimensionalityReduction (the Dimensionality Reduction panel and PCA node) against
// PyTorch float64 references of scikit-learn PCA / exact t-SNE and umap-learn's
// graph, a / b fit (scipy curve_fit) and expected layout updates (TOFIX140).
//
// fixtures/dimensionality_reduction_pytorch.json
// (generate_dimensionality_reduction_fixtures.py). Runs on the active ArrayFire
// backend (CYXWIZ_TEST_ARRAYFIRE_BACKEND=cuda|opencl|cpu).
#include "test_device_selection.h"

#include <cyxwiz/clustering.h>
#include <cyxwiz/dimensionality_reduction.h>

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using json = nlohmann::json;
using Matrix = std::vector<std::vector<double>>;
using cyxwiz::DimensionalityReduction;
using cyxwiz::test::DeviceTolerance;

// Closed-form results: float64 rounding, or float32 rounding on a device without
// float64 (worst measured on the Intel Iris Xe, TOFIX145 item 7: 8e-6, PCA
// transformed).
double Tolerance() { return DeviceTolerance(1e-8, 5e-5); }
// Iterative runs (kept short or slow: both are chaotic). float32 rounding grows
// along the run: 1.5e-2 on the fast t-SNE embedding, 4.4e-4 on its KL.
double TrajectoryTolerance() { return DeviceTolerance(1e-6, 5e-2); }

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

void CheckValues(const std::vector<double>& actual, const json& expected, const std::string& what, double tolerance = Tolerance()) {
    const auto values = expected.get<std::vector<double>>();
    Check(actual.size() == values.size(), what + ": count " + std::to_string(actual.size()));
    for (size_t i = 0; i < values.size(); ++i) CheckValue(actual[i], values[i], what + "[" + std::to_string(i) + "]", tolerance);
}

void CheckMatrix(const Matrix& actual, const json& expected_json, const std::string& what, double tolerance = Tolerance()) {
    const Matrix expected = expected_json.get<Matrix>();
    Check(actual.size() == expected.size(), what + ": rows");
    for (size_t r = 0; r < expected.size(); ++r) {
        Check(actual[r].size() == expected[r].size(), what + ": cols");
        for (size_t c = 0; c < expected[r].size(); ++c) {
            CheckValue(actual[r][c], expected[r][c], what + "(" + std::to_string(r) + "," + std::to_string(c) + ")", tolerance);
        }
    }
}

// The embedding keeps the blobs apart: silhouette against the true labels.
void CheckSeparates(const Matrix& embedding, const json& c, const std::string& what) {
    const auto metrics = cyxwiz::Clustering::EvaluateClustering(embedding, c.at("truth").get<std::vector<int>>());
    Check(metrics.success, what + ": " + metrics.error_message);
    Check(metrics.silhouette_score > 0.6, what + ": blobs not separated, silhouette " + std::to_string(metrics.silhouette_score));
}

void CheckRefused(const std::string& error, const json& c, const std::string& what) {
    const std::string message = c.at("message").get<std::string>();
    Check(error.find(message) != std::string::npos,
          what + ": expected a refusal containing '" + message + "', got '" + error + "'");
}

std::filesystem::path FixturePath(const char* argv0) {
    const auto beside = std::filesystem::path(argv0).parent_path() / "computation_truth_fixtures" /
                        "dimensionality_reduction_pytorch.json";
    if (std::filesystem::exists(beside)) return beside;
    return std::filesystem::path(CYXWIZ_DIMENSIONALITY_REDUCTION_FIXTURE);
}

void RunCase(const json& c, const std::string& name) {
    const std::string op = c.at("op").get<std::string>();
    const Matrix data = c.at("data").get<Matrix>();

    if (op == "pca") {
        const auto result = DimensionalityReduction::ComputePCA(data, c.at("n_components").get<int>(),
                                                                c.at("center").get<bool>(), c.at("scale").get<bool>());
        Check(result.success, name + ": " + result.error_message);
        CheckMatrix(result.components, c.at("components"), name + " components");
        CheckMatrix(result.transformed, c.at("transformed"), name + " transformed");
        CheckValues(result.explained_variance, c.at("explained_variance"), name + " variance");
        CheckValues(result.explained_variance_ratio, c.at("explained_variance_ratio"), name + " ratio");
        CheckValues(result.singular_values, c.at("singular_values"), name + " singular values");
        double total = 0.0;
        for (double ratio : result.explained_variance_ratio) total += ratio;
        CheckValue(result.total_variance_explained, total, name + " total explained");
        return Check(result.n_components == static_cast<int>(result.components.size()), name + ": n_components");
    }
    if (op == "tsne") {
        int callbacks = 0;
        const auto result = DimensionalityReduction::ComputetSNE(
            data, c.at("n_dims").get<int>(), c.at("perplexity").get<int>(), c.at("learning_rate").get<double>(),
            c.at("n_iterations").get<int>(), [&](int, double) { ++callbacks; });
        Check(result.success, name + ": " + result.error_message);
        CheckMatrix(result.embeddings, c.at("embeddings"), name + " embedding", TrajectoryTolerance());
        CheckValues(result.kl_divergence_history, c.at("kl_history"), name + " KL history", TrajectoryTolerance());
        CheckValue(result.final_kl_divergence, c.at("final_kl").get<double>(), name + " final KL", TrajectoryTolerance());
        Check(result.n_iterations == c.at("iterations_run").get<int>(), name + ": iterations run");
        return Check(callbacks == static_cast<int>(result.kl_divergence_history.size()), name + ": one callback per check");
    }
    if (op == "tsne_separates") {
        const auto result = DimensionalityReduction::ComputetSNE(data, 2, 10, 50.0, 1000);
        Check(result.success, name + ": " + result.error_message);
        return CheckSeparates(result.embeddings, c, name);
    }
    if (op == "umap") {
        const auto result = DimensionalityReduction::ComputeUMAP(data, c.at("n_dims").get<int>(), c.at("n_neighbors").get<int>(),
                                                                 c.at("min_dist").get<double>(), c.at("n_epochs").get<int>());
        Check(result.success, name + ": " + result.error_message);
        // a / b: scipy curve_fit polished to the exact least-squares optimum (the generator checks
        // it is scipy's fit to 1e-6).
        CheckValue(result.a, c.at("a").get<double>(), name + " a");
        CheckValue(result.b, c.at("b").get<double>(), name + " b");
        CheckMatrix(result.embeddings, c.at("embeddings"), name + " embedding", TrajectoryTolerance());
        return Check(result.n_epochs == c.at("n_epochs").get<int>(), name + ": epochs");
    }
    if (op == "umap_separates") {
        const auto result = DimensionalityReduction::ComputeUMAP(data, 2, 10, 0.1);
        Check(result.success && result.n_epochs == 500, name + ": " + result.error_message);
        return CheckSeparates(result.embeddings, c, name);
    }
    if (op == "refuse_tsne") return CheckRefused(DimensionalityReduction::ComputetSNE(data, 2, static_cast<int>(data.size())).error_message, c, name);
    if (op == "refuse_umap") return CheckRefused(DimensionalityReduction::ComputeUMAP(data, 2, 1).error_message, c, name);
    if (op == "refuse_pca") return CheckRefused(DimensionalityReduction::ComputePCA(data).error_message, c, name);
    Check(false, name + ": unknown op " + op);
}

}  // namespace

int main(int, char** argv) {
    Check(cyxwiz::test::SelectTestDeviceFromEnvironment(), "requested test device");
    std::ifstream in(FixturePath(argv[0]));
    Check(static_cast<bool>(in), "cannot open the dimensionality reduction fixture");
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
    std::cout << "dimensionality reduction matches the references: " << passed << " cases" << std::endl;
    return 0;
}
