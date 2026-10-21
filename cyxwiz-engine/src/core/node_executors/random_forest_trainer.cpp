#include "random_forest_trainer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

namespace {

std::vector<double> ProjectRow(const std::vector<double>& row,
                               const std::vector<size_t>& feature_indices) {
    std::vector<double> projected;
    projected.reserve(feature_indices.size());
    for (size_t feature_index : feature_indices) {
        projected.push_back(row[feature_index]);
    }
    return projected;
}

uint32_t MixSeed(int seed, int tree_index) {
    uint32_t value = static_cast<uint32_t>(seed);
    value ^= 0x9e3779b9u + static_cast<uint32_t>(tree_index) +
             (value << 6) + (value >> 2);
    return value;
}

} // namespace

RandomForestTrainer::RandomForestTrainer(RandomForestTrainingOptions options)
    : options_(std::move(options)) {}

RandomForestModel RandomForestTrainer::Fit(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    size_t class_count,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_labels,
    bool numeric_labels) const {
    if (features.empty()) {
        throw std::runtime_error("RandomForest: no training rows");
    }
    if (features.size() != labels.size()) {
        throw std::runtime_error("RandomForest: feature/label row mismatch");
    }
    if (features.front().empty()) {
        throw std::runtime_error("RandomForest: no feature columns");
    }
    if (options_.n_estimators < 1) {
        throw std::runtime_error("RandomForest: n_estimators must be >= 1");
    }

    DecisionTreeTrainingOptions tree_options;
    tree_options.max_depth = options_.max_depth;
    tree_options.min_samples_split = options_.min_samples_split;
    tree_options.min_samples_leaf = options_.min_samples_leaf;
    tree_options.criterion = options_.criterion;
    DecisionTreeTrainer tree_trainer(tree_options);

    std::vector<RandomForestTree> trees;
    trees.reserve(static_cast<size_t>(options_.n_estimators));
    std::uniform_int_distribution<size_t> row_dist(0, features.size() - 1);

    for (int tree_index = 0; tree_index < options_.n_estimators; ++tree_index) {
        const uint32_t tree_seed = MixSeed(options_.seed, tree_index);
        std::mt19937 rng(tree_seed);
        std::vector<size_t> feature_indices =
            SampleFeatureIndices(features.front().size(), tree_seed);

        std::vector<std::vector<double>> boot_features;
        std::vector<int> boot_labels;
        boot_features.reserve(features.size());
        boot_labels.reserve(labels.size());
        for (size_t sample = 0; sample < features.size(); ++sample) {
            const size_t row = row_dist(rng);
            boot_features.push_back(ProjectRow(features[row], feature_indices));
            boot_labels.push_back(labels[row]);
        }

        std::vector<std::string> selected_feature_names;
        selected_feature_names.reserve(feature_indices.size());
        for (size_t feature_index : feature_indices) {
            selected_feature_names.push_back(feature_names[feature_index]);
        }

        RandomForestTree tree;
        tree.feature_indices = std::move(feature_indices);
        tree.model = tree_trainer.Fit(
            boot_features,
            boot_labels,
            class_count,
            selected_feature_names,
            class_labels,
            numeric_labels);
        trees.push_back(std::move(tree));
    }

    RandomForestModel model;
    model.SetTrees(std::move(trees));
    model.SetFeatureNames(feature_names);
    model.SetClassLabels(class_labels, numeric_labels);
    return model;
}

size_t RandomForestTrainer::ResolveFeatureSubsetSize(size_t feature_count) const {
    if (feature_count == 0) {
        return 0;
    }
    if (options_.max_features == "all") {
        return feature_count;
    }
    if (options_.max_features == "log2") {
        return std::max<size_t>(
            1, static_cast<size_t>(std::floor(std::log2(feature_count))));
    }
    return std::max<size_t>(
        1, static_cast<size_t>(std::floor(std::sqrt(feature_count))));
}

std::vector<size_t> RandomForestTrainer::SampleFeatureIndices(
    size_t feature_count,
    uint32_t tree_seed) const {
    std::vector<size_t> indices(feature_count);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(tree_seed ^ 0x85ebca6bu);
    std::shuffle(indices.begin(), indices.end(), rng);
    indices.resize(ResolveFeatureSubsetSize(feature_count));
    std::sort(indices.begin(), indices.end());
    return indices;
}

} // namespace cyxwiz
