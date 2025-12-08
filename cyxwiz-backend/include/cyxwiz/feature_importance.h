#pragma once

#include "cyxwiz/api_export.h"
#include <cyxwiz/sequential.h>
#include <cyxwiz/tensor.h>
#include <vector>
#include <string>
#include <functional>

namespace cyxwiz {

/**
 * FeatureImportanceResult - Result of feature importance computation
 */
struct CYXWIZ_API FeatureImportanceResult {
    std::vector<std::string> feature_names;
    std::vector<double> importances;           // Importance scores (normalized 0-1)
    std::vector<double> importances_std;       // Standard deviation of importance
    std::vector<int> ranking;                  // Sorted indices (descending importance)
    std::string method;                        // "permutation", "drop_column"
    double baseline_score = 0.0;               // Original model score
    bool success = false;
    std::string error_message;
};

/**
 * FeatureImportanceAnalyzer - Compute feature importance for ML models
 *
 * Implements:
 * - Permutation Importance (model-agnostic)
 * - Drop-Column Importance
 */
class CYXWIZ_API FeatureImportanceAnalyzer {
public:
    /**
     * Compute permutation importance
     *
     * For each feature, shuffle its values and measure the decrease in model performance.
     * Features whose shuffling causes larger performance drops are more important.
     *
     * @param model The trained SequentialModel
     * @param X Input features [n_samples x n_features]
     * @param y True labels [n_samples]
     * @param feature_names Optional names for features
     * @param n_repeats Number of times to repeat shuffling (default 10)
     * @param scoring Scoring method: "accuracy" or "mse" (default "accuracy")
     * @param progress_callback Optional callback (current_feature, total_features)
     * @return FeatureImportanceResult with importance scores
     */
    static FeatureImportanceResult ComputePermutationImportance(
        SequentialModel& model,
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const std::vector<std::string>& feature_names = {},
        int n_repeats = 10,
        const std::string& scoring = "accuracy",
        std::function<void(int, int)> progress_callback = nullptr
    );

    /**
     * Compute drop-column importance
     *
     * For each feature, retrain the model without it and measure performance change.
     * More expensive than permutation importance but can capture feature interactions.
     *
     * Note: This simplified version doesn't actually retrain - it sets the column to zero.
     *
     * @param model The trained SequentialModel
     * @param X Input features [n_samples x n_features]
     * @param y True labels [n_samples]
     * @param feature_names Optional names for features
     * @param scoring Scoring method: "accuracy" or "mse" (default "accuracy")
     * @return FeatureImportanceResult with importance scores
     */
    static FeatureImportanceResult ComputeDropColumnImportance(
        SequentialModel& model,
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const std::vector<std::string>& feature_names = {},
        const std::string& scoring = "accuracy"
    );

    /**
     * Compute mean absolute weights for the first layer
     *
     * Simple heuristic: features connected to larger weights are more important.
     * Only applicable to models with a linear first layer.
     *
     * @param model The trained SequentialModel
     * @param feature_names Optional names for features
     * @return FeatureImportanceResult with weight-based importance
     */
    static FeatureImportanceResult ComputeWeightImportance(
        SequentialModel& model,
        const std::vector<std::string>& feature_names = {}
    );

private:
    /**
     * Evaluate model performance
     * @return Score (higher is better for accuracy, lower magnitude is better for MSE)
     */
    static double EvaluateModel(
        SequentialModel& model,
        const std::vector<std::vector<double>>& X,
        const std::vector<double>& y,
        const std::string& scoring
    );

    /**
     * Convert 2D vector to Tensor for model input
     */
    static Tensor CreateInputTensor(
        const std::vector<std::vector<double>>& X
    );

    /**
     * Shuffle a specific column in-place
     */
    static void ShuffleColumn(
        std::vector<std::vector<double>>& X,
        int column_index,
        unsigned int seed
    );

    /**
     * Compute ranking (indices sorted by descending importance)
     */
    static std::vector<int> ComputeRanking(const std::vector<double>& importances);

    /**
     * Normalize importances to [0, 1] range
     */
    static void NormalizeImportances(std::vector<double>& importances);

    /**
     * Generate default feature names
     */
    static std::vector<std::string> GenerateFeatureNames(int n_features);
};

} // namespace cyxwiz
