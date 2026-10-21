#pragma once

#include "gradient_boosting_model.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct GradientBoostingTrainingOptions {
    int n_estimators = 100;
    double learning_rate = 0.1;
    int max_depth = 3;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
};

class GradientBoostingTrainer {
public:
    explicit GradientBoostingTrainer(GradientBoostingTrainingOptions options);

    GradientBoostingModel Fit(
        const std::vector<std::vector<double>>& features,
        const std::vector<int>& labels,
        size_t class_count,
        const std::vector<std::string>& feature_names,
        const std::vector<std::string>& class_labels,
        bool numeric_labels) const;

private:
    struct SplitCandidate {
        bool valid = false;
        int feature_index = -1;
        double threshold = 0.0;
        double gain = 0.0;
    };

    GradientBoostingRegressionTree FitRegressionTree(
        const std::vector<std::vector<double>>& features,
        const std::vector<double>& residuals) const;
    int BuildRegressionNode(
        const std::vector<size_t>& indices,
        int depth,
        const std::vector<std::vector<double>>& features,
        const std::vector<double>& residuals,
        std::vector<GradientBoostingRegressionNode>& nodes) const;
    SplitCandidate FindBestSplit(
        const std::vector<size_t>& indices,
        const std::vector<std::vector<double>>& features,
        const std::vector<double>& residuals) const;
    double MeanValue(const std::vector<size_t>& indices,
                     const std::vector<double>& residuals) const;
    double SquaredError(const std::vector<size_t>& indices,
                        const std::vector<double>& residuals) const;

    GradientBoostingTrainingOptions options_;
};

} // namespace cyxwiz
