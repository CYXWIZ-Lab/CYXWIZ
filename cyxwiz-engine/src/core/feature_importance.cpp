#include "feature_importance.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>

namespace cyxwiz {

// ============================================================================
// Utility Methods
// ============================================================================

std::vector<std::string> FeatureImportanceAnalyzer::GenerateFeatureNames(int n_features) {
    std::vector<std::string> names(n_features);
    for (int i = 0; i < n_features; i++) {
        names[i] = "Feature_" + std::to_string(i);
    }
    return names;
}

std::vector<int> FeatureImportanceAnalyzer::ComputeRanking(const std::vector<double>& importances) {
    std::vector<int> ranking(importances.size());
    std::iota(ranking.begin(), ranking.end(), 0);

    std::sort(ranking.begin(), ranking.end(), [&importances](int a, int b) {
        return importances[a] > importances[b];  // Descending order
    });

    return ranking;
}

void FeatureImportanceAnalyzer::NormalizeImportances(std::vector<double>& importances) {
    if (importances.empty()) return;

    double min_val = *std::min_element(importances.begin(), importances.end());
    double max_val = *std::max_element(importances.begin(), importances.end());

    // Handle negative values by shifting
    if (min_val < 0) {
        for (auto& v : importances) {
            v -= min_val;
        }
        max_val -= min_val;
        min_val = 0;
    }

    if (max_val > 1e-10) {
        for (auto& v : importances) {
            v /= max_val;
        }
    }
}

Tensor FeatureImportanceAnalyzer::CreateInputTensor(
    const std::vector<std::vector<double>>& X)
{
    if (X.empty()) {
        return Tensor({0}, DataType::Float32);
    }

    size_t n_samples = X.size();
    size_t n_features = X[0].size();

    Tensor tensor({n_samples, n_features}, DataType::Float32);
    float* data = tensor.Data<float>();

    if (data) {
        for (size_t i = 0; i < n_samples; i++) {
            for (size_t j = 0; j < n_features; j++) {
                data[i * n_features + j] = static_cast<float>(X[i][j]);
            }
        }
    }

    return tensor;
}

void FeatureImportanceAnalyzer::ShuffleColumn(
    std::vector<std::vector<double>>& X,
    int column_index,
    unsigned int seed)
{
    if (X.empty() || column_index < 0) return;

    std::vector<double> column_values;
    for (const auto& row : X) {
        if (column_index < static_cast<int>(row.size())) {
            column_values.push_back(row[column_index]);
        }
    }

    std::mt19937 rng(seed);
    std::shuffle(column_values.begin(), column_values.end(), rng);

    for (size_t i = 0; i < X.size() && i < column_values.size(); i++) {
        if (column_index < static_cast<int>(X[i].size())) {
            X[i][column_index] = column_values[i];
        }
    }
}

double FeatureImportanceAnalyzer::EvaluateModel(
    SequentialModel& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::string& scoring)
{
    if (X.empty() || y.empty()) return 0.0;

    Tensor input = CreateInputTensor(X);
    model.SetTraining(false);
    Tensor output = model.Forward(input);

    const float* output_data = output.Data<float>();
    if (!output_data) return 0.0;

    size_t n_samples = X.size();
    size_t output_size = output.NumElements() / n_samples;

    if (scoring == "accuracy") {
        // Classification accuracy
        int correct = 0;
        for (size_t i = 0; i < n_samples; i++) {
            // Find predicted class (argmax)
            int pred_class = 0;
            float max_val = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j < output_size; j++) {
                float val = output_data[i * output_size + j];
                if (val > max_val) {
                    max_val = val;
                    pred_class = static_cast<int>(j);
                }
            }

            if (pred_class == static_cast<int>(y[i])) {
                correct++;
            }
        }

        return static_cast<double>(correct) / n_samples;

    } else if (scoring == "mse") {
        // Mean squared error (return negative so higher is still better)
        double mse = 0.0;
        for (size_t i = 0; i < n_samples; i++) {
            double pred = output_data[i * output_size];  // Assume single output
            double diff = pred - y[i];
            mse += diff * diff;
        }
        mse /= n_samples;

        return -mse;  // Negative so higher is better

    } else {
        // Default to accuracy
        return EvaluateModel(model, X, y, "accuracy");
    }
}

// ============================================================================
// Permutation Importance
// ============================================================================

FeatureImportanceResult FeatureImportanceAnalyzer::ComputePermutationImportance(
    SequentialModel& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<std::string>& feature_names,
    int n_repeats,
    const std::string& scoring,
    std::function<void(int, int)> progress_callback)
{
    FeatureImportanceResult result;
    result.method = "permutation";

    if (X.empty() || y.empty()) {
        result.success = false;
        result.error_message = "Empty input data";
        return result;
    }

    int n_features = static_cast<int>(X[0].size());

    // Set feature names
    if (feature_names.empty()) {
        result.feature_names = GenerateFeatureNames(n_features);
    } else {
        result.feature_names = feature_names;
        while (result.feature_names.size() < static_cast<size_t>(n_features)) {
            result.feature_names.push_back("Feature_" + std::to_string(result.feature_names.size()));
        }
    }

    spdlog::info("Computing permutation importance for {} features, {} repeats",
                 n_features, n_repeats);

    // Compute baseline score
    result.baseline_score = EvaluateModel(model, X, y, scoring);
    spdlog::info("Baseline {} = {:.4f}", scoring, result.baseline_score);

    result.importances.resize(n_features, 0.0);
    result.importances_std.resize(n_features, 0.0);

    // For each feature
    for (int feat = 0; feat < n_features; feat++) {
        std::vector<double> repeat_scores;

        for (int rep = 0; rep < n_repeats; rep++) {
            // Create copy and shuffle column
            std::vector<std::vector<double>> X_shuffled = X;
            ShuffleColumn(X_shuffled, feat, static_cast<unsigned int>(feat * n_repeats + rep));

            // Evaluate on shuffled data
            double shuffled_score = EvaluateModel(model, X_shuffled, y, scoring);

            // Importance = baseline - shuffled (drop in performance)
            double importance = result.baseline_score - shuffled_score;
            repeat_scores.push_back(importance);
        }

        // Compute mean and std
        double mean = 0.0;
        for (double s : repeat_scores) {
            mean += s;
        }
        mean /= n_repeats;

        double variance = 0.0;
        for (double s : repeat_scores) {
            variance += (s - mean) * (s - mean);
        }
        variance /= n_repeats;

        result.importances[feat] = mean;
        result.importances_std[feat] = std::sqrt(variance);

        if (progress_callback) {
            progress_callback(feat + 1, n_features);
        }

        spdlog::debug("Feature {}: importance = {:.4f} +/- {:.4f}",
                     result.feature_names[feat], mean, std::sqrt(variance));
    }

    // Normalize importances
    NormalizeImportances(result.importances);

    // Compute ranking
    result.ranking = ComputeRanking(result.importances);

    result.success = true;

    spdlog::info("Permutation importance complete. Top feature: {}",
                 result.feature_names[result.ranking[0]]);

    return result;
}

// ============================================================================
// Drop-Column Importance
// ============================================================================

FeatureImportanceResult FeatureImportanceAnalyzer::ComputeDropColumnImportance(
    SequentialModel& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<double>& y,
    const std::vector<std::string>& feature_names,
    const std::string& scoring)
{
    FeatureImportanceResult result;
    result.method = "drop_column";

    if (X.empty() || y.empty()) {
        result.success = false;
        result.error_message = "Empty input data";
        return result;
    }

    int n_features = static_cast<int>(X[0].size());

    // Set feature names
    if (feature_names.empty()) {
        result.feature_names = GenerateFeatureNames(n_features);
    } else {
        result.feature_names = feature_names;
    }

    spdlog::info("Computing drop-column importance for {} features", n_features);

    // Compute baseline score
    result.baseline_score = EvaluateModel(model, X, y, scoring);

    result.importances.resize(n_features, 0.0);
    result.importances_std.resize(n_features, 0.0);

    // For each feature
    for (int feat = 0; feat < n_features; feat++) {
        // Create copy and zero out column
        std::vector<std::vector<double>> X_dropped = X;
        for (auto& row : X_dropped) {
            if (feat < static_cast<int>(row.size())) {
                row[feat] = 0.0;  // Set to zero (simplified drop)
            }
        }

        // Evaluate on dropped data
        double dropped_score = EvaluateModel(model, X_dropped, y, scoring);

        // Importance = baseline - dropped
        result.importances[feat] = result.baseline_score - dropped_score;
    }

    // Normalize importances
    NormalizeImportances(result.importances);

    // Compute ranking
    result.ranking = ComputeRanking(result.importances);

    result.success = true;

    return result;
}

// ============================================================================
// Weight-Based Importance
// ============================================================================

FeatureImportanceResult FeatureImportanceAnalyzer::ComputeWeightImportance(
    SequentialModel& model,
    const std::vector<std::string>& feature_names)
{
    FeatureImportanceResult result;
    result.method = "weight";

    // Find the first layer with parameters (should be Linear)
    Module* first_layer = nullptr;
    for (size_t i = 0; i < model.Size(); i++) {
        Module* module = model.GetModule(i);
        if (module && module->HasParameters()) {
            first_layer = module;
            break;
        }
    }

    if (!first_layer) {
        result.success = false;
        result.error_message = "No layer with parameters found";
        return result;
    }

    // Get weights
    auto params = first_layer->GetParameters();
    auto it = params.find("weight");
    if (it == params.end()) {
        result.success = false;
        result.error_message = "No weight parameter found in first layer";
        return result;
    }

    const Tensor& weights = it->second;
    const auto& shape = weights.Shape();

    if (shape.size() < 2) {
        result.success = false;
        result.error_message = "Unexpected weight shape";
        return result;
    }

    size_t out_features = shape[0];
    size_t in_features = shape[1];

    spdlog::info("Computing weight importance from layer '{}' with {} input features",
                 first_layer->GetName(), in_features);

    // Set feature names
    if (feature_names.empty()) {
        result.feature_names = GenerateFeatureNames(static_cast<int>(in_features));
    } else {
        result.feature_names = feature_names;
    }

    // Compute mean absolute weight for each input feature
    result.importances.resize(in_features, 0.0);

    const float* weight_data = weights.Data<float>();
    if (weight_data) {
        for (size_t j = 0; j < in_features; j++) {
            double sum_abs = 0.0;
            for (size_t i = 0; i < out_features; i++) {
                sum_abs += std::abs(weight_data[i * in_features + j]);
            }
            result.importances[j] = sum_abs / out_features;
        }
    }

    // Normalize
    NormalizeImportances(result.importances);

    // Compute ranking
    result.ranking = ComputeRanking(result.importances);

    result.success = true;

    return result;
}

} // namespace cyxwiz
