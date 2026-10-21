#include "gradient_boosting_trainer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

namespace {

double Sigmoid(double value) {
    if (value >= 35.0) {
        return 1.0;
    }
    if (value <= -35.0) {
        return 0.0;
    }
    return 1.0 / (1.0 + std::exp(-value));
}

} // namespace

GradientBoostingTrainer::GradientBoostingTrainer(
    GradientBoostingTrainingOptions options)
    : options_(std::move(options)) {}

GradientBoostingModel GradientBoostingTrainer::Fit(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    size_t class_count,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_labels,
    bool numeric_labels) const {
    if (features.empty()) {
        throw std::runtime_error("GradientBoosting: no training rows");
    }
    if (features.size() != labels.size()) {
        throw std::runtime_error("GradientBoosting: feature/label row mismatch");
    }
    if (features.front().empty()) {
        throw std::runtime_error("GradientBoosting: no feature columns");
    }
    if (class_count < 2) {
        throw std::runtime_error(
            "GradientBoosting: at least two classes are required");
    }
    if (options_.n_estimators < 1) {
        throw std::runtime_error(
            "GradientBoosting: n_estimators must be >= 1");
    }
    if (options_.learning_rate <= 0.0) {
        throw std::runtime_error(
            "GradientBoosting: learning_rate must be > 0");
    }

    std::vector<size_t> class_counts(class_count, 0);
    for (int label : labels) {
        if (label < 0 || label >= static_cast<int>(class_count)) {
            throw std::runtime_error("GradientBoosting: label id is out of range");
        }
        ++class_counts[static_cast<size_t>(label)];
    }

    const double eps = 1.0e-6;
    std::vector<double> initial_scores(class_count, 0.0);
    for (size_t cls = 0; cls < class_count; ++cls) {
        const double positives = static_cast<double>(class_counts[cls]) + eps;
        const double negatives =
            static_cast<double>(features.size() - class_counts[cls]) + eps;
        initial_scores[cls] = std::log(positives / negatives);
    }

    std::vector<std::vector<double>> scores(
        features.size(), std::vector<double>(class_count, 0.0));
    for (auto& row_scores : scores) {
        row_scores = initial_scores;
    }

    std::vector<std::vector<GradientBoostingRegressionTree>> trees;
    trees.reserve(static_cast<size_t>(options_.n_estimators));
    for (int estimator = 0; estimator < options_.n_estimators; ++estimator) {
        std::vector<GradientBoostingRegressionTree> round;
        round.reserve(class_count);
        for (size_t cls = 0; cls < class_count; ++cls) {
            std::vector<double> residuals(features.size(), 0.0);
            for (size_t row = 0; row < features.size(); ++row) {
                const double y =
                    labels[row] == static_cast<int>(cls) ? 1.0 : 0.0;
                residuals[row] = y - Sigmoid(scores[row][cls]);
            }
            auto tree = FitRegressionTree(features, residuals);
            for (size_t row = 0; row < features.size(); ++row) {
                scores[row][cls] +=
                    options_.learning_rate * tree.PredictValue(features[row]);
            }
            round.push_back(std::move(tree));
        }
        trees.push_back(std::move(round));
    }

    GradientBoostingModel model;
    model.SetInitialScores(std::move(initial_scores));
    model.SetTrees(std::move(trees));
    model.SetLearningRate(options_.learning_rate);
    model.SetFeatureNames(feature_names);
    model.SetClassLabels(class_labels, numeric_labels);
    return model;
}

GradientBoostingRegressionTree GradientBoostingTrainer::FitRegressionTree(
    const std::vector<std::vector<double>>& features,
    const std::vector<double>& residuals) const {
    std::vector<size_t> indices(features.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<GradientBoostingRegressionNode> nodes;
    BuildRegressionNode(indices, 0, features, residuals, nodes);

    GradientBoostingRegressionTree tree;
    tree.SetNodes(std::move(nodes));
    return tree;
}

int GradientBoostingTrainer::BuildRegressionNode(
    const std::vector<size_t>& indices,
    int depth,
    const std::vector<std::vector<double>>& features,
    const std::vector<double>& residuals,
    std::vector<GradientBoostingRegressionNode>& nodes) const {
    GradientBoostingRegressionNode node;
    node.value = MeanValue(indices, residuals);
    const int node_index = static_cast<int>(nodes.size());
    nodes.push_back(node);

    if (depth >= options_.max_depth ||
        indices.size() < static_cast<size_t>(options_.min_samples_split) ||
        indices.size() <= static_cast<size_t>(options_.min_samples_leaf * 2)) {
        return node_index;
    }

    const SplitCandidate split = FindBestSplit(indices, features, residuals);
    if (!split.valid || split.gain <= 0.0) {
        return node_index;
    }

    std::vector<size_t> left;
    std::vector<size_t> right;
    left.reserve(indices.size());
    right.reserve(indices.size());
    for (size_t row : indices) {
        if (features[row][static_cast<size_t>(split.feature_index)] <=
            split.threshold) {
            left.push_back(row);
        } else {
            right.push_back(row);
        }
    }
    if (left.size() < static_cast<size_t>(options_.min_samples_leaf) ||
        right.size() < static_cast<size_t>(options_.min_samples_leaf)) {
        return node_index;
    }

    const int left_child =
        BuildRegressionNode(left, depth + 1, features, residuals, nodes);
    const int right_child =
        BuildRegressionNode(right, depth + 1, features, residuals, nodes);

    auto& stored = nodes[static_cast<size_t>(node_index)];
    stored.is_leaf = false;
    stored.feature_index = split.feature_index;
    stored.threshold = split.threshold;
    stored.left_child = left_child;
    stored.right_child = right_child;
    return node_index;
}

GradientBoostingTrainer::SplitCandidate GradientBoostingTrainer::FindBestSplit(
    const std::vector<size_t>& indices,
    const std::vector<std::vector<double>>& features,
    const std::vector<double>& residuals) const {
    SplitCandidate best;
    if (indices.empty() || features.front().empty()) {
        return best;
    }

    const double base_error = SquaredError(indices, residuals);
    const size_t feature_count = features.front().size();
    for (size_t feature = 0; feature < feature_count; ++feature) {
        std::vector<double> values;
        values.reserve(indices.size());
        for (size_t row : indices) {
            values.push_back(features[row][feature]);
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        if (values.size() < 2) {
            continue;
        }

        for (size_t i = 1; i < values.size(); ++i) {
            const double threshold = (values[i - 1] + values[i]) * 0.5;
            std::vector<size_t> left;
            std::vector<size_t> right;
            left.reserve(indices.size());
            right.reserve(indices.size());
            for (size_t row : indices) {
                if (features[row][feature] <= threshold) {
                    left.push_back(row);
                } else {
                    right.push_back(row);
                }
            }
            if (left.size() < static_cast<size_t>(options_.min_samples_leaf) ||
                right.size() < static_cast<size_t>(options_.min_samples_leaf)) {
                continue;
            }

            const double gain = base_error - SquaredError(left, residuals) -
                                SquaredError(right, residuals);
            if (!best.valid || gain > best.gain) {
                best.valid = true;
                best.feature_index = static_cast<int>(feature);
                best.threshold = threshold;
                best.gain = gain;
            }
        }
    }
    return best;
}

double GradientBoostingTrainer::MeanValue(
    const std::vector<size_t>& indices,
    const std::vector<double>& residuals) const {
    if (indices.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t row : indices) {
        sum += residuals[row];
    }
    return sum / static_cast<double>(indices.size());
}

double GradientBoostingTrainer::SquaredError(
    const std::vector<size_t>& indices,
    const std::vector<double>& residuals) const {
    if (indices.empty()) {
        return 0.0;
    }
    const double mean = MeanValue(indices, residuals);
    double error = 0.0;
    for (size_t row : indices) {
        const double delta = residuals[row] - mean;
        error += delta * delta;
    }
    return error;
}

} // namespace cyxwiz
