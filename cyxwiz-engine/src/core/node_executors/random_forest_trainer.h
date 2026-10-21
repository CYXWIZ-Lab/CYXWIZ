#pragma once

#include "decision_tree_trainer.h"
#include "random_forest_model.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cyxwiz {

struct RandomForestTrainingOptions {
    int n_estimators = 100;
    int max_depth = 10;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
    int seed = 42;
    std::string criterion = "gini";
    std::string max_features = "sqrt";
};

class RandomForestTrainer {
public:
    explicit RandomForestTrainer(RandomForestTrainingOptions options);

    RandomForestModel Fit(const std::vector<std::vector<double>>& features,
                          const std::vector<int>& labels,
                          size_t class_count,
                          const std::vector<std::string>& feature_names,
                          const std::vector<std::string>& class_labels,
                          bool numeric_labels) const;

private:
    size_t ResolveFeatureSubsetSize(size_t feature_count) const;
    std::vector<size_t> SampleFeatureIndices(size_t feature_count,
                                             uint32_t tree_seed) const;

    RandomForestTrainingOptions options_;
};

} // namespace cyxwiz
