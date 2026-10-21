#include "gradient_boosting_model.h"

#include <algorithm>
#include <utility>

namespace cyxwiz {

double GradientBoostingRegressionTree::PredictValue(
    const std::vector<double>& row) const {
    if (nodes_.empty()) {
        return 0.0;
    }

    int node_index = 0;
    while (node_index >= 0 && node_index < static_cast<int>(nodes_.size())) {
        const auto& node = nodes_[static_cast<size_t>(node_index)];
        if (node.is_leaf) {
            return node.value;
        }
        const double value =
            node.feature_index >= 0 &&
                    node.feature_index < static_cast<int>(row.size())
                ? row[static_cast<size_t>(node.feature_index)]
                : 0.0;
        node_index = value <= node.threshold ? node.left_child
                                             : node.right_child;
    }
    return 0.0;
}

void GradientBoostingRegressionTree::SetNodes(
    std::vector<GradientBoostingRegressionNode> nodes) {
    nodes_ = std::move(nodes);
}

int GradientBoostingRegressionTree::MaxDepth() const {
    if (nodes_.empty()) {
        return 0;
    }
    return MaxDepthFromNode(0);
}

int GradientBoostingRegressionTree::MaxDepthFromNode(int node_index) const {
    if (node_index < 0 || node_index >= static_cast<int>(nodes_.size())) {
        return 0;
    }
    const auto& node = nodes_[static_cast<size_t>(node_index)];
    if (node.is_leaf) {
        return 0;
    }
    return 1 + std::max(MaxDepthFromNode(node.left_child),
                        MaxDepthFromNode(node.right_child));
}

int GradientBoostingModel::PredictClass(const std::vector<double>& row) const {
    if (class_labels_.empty()) {
        return 0;
    }

    std::vector<double> scores = initial_scores_;
    if (scores.size() != class_labels_.size()) {
        scores.assign(class_labels_.size(), 0.0);
    }
    for (const auto& round : trees_) {
        for (size_t cls = 0; cls < round.size() && cls < scores.size(); ++cls) {
            scores[cls] += learning_rate_ * round[cls].PredictValue(row);
        }
    }

    int best_class = 0;
    double best_score = scores.empty() ? 0.0 : scores.front();
    for (size_t cls = 1; cls < scores.size(); ++cls) {
        if (scores[cls] > best_score) {
            best_score = scores[cls];
            best_class = static_cast<int>(cls);
        }
    }
    return best_class;
}

std::vector<int> GradientBoostingModel::PredictClasses(
    const std::vector<std::vector<double>>& rows) const {
    std::vector<int> out;
    out.reserve(rows.size());
    for (const auto& row : rows) {
        out.push_back(PredictClass(row));
    }
    return out;
}

void GradientBoostingModel::SetInitialScores(std::vector<double> scores) {
    initial_scores_ = std::move(scores);
}

void GradientBoostingModel::SetTrees(
    std::vector<std::vector<GradientBoostingRegressionTree>> trees) {
    trees_ = std::move(trees);
}

void GradientBoostingModel::SetLearningRate(double learning_rate) {
    learning_rate_ = learning_rate;
}

void GradientBoostingModel::SetFeatureNames(
    std::vector<std::string> feature_names) {
    feature_names_ = std::move(feature_names);
}

void GradientBoostingModel::SetClassLabels(std::vector<std::string> labels,
                                           bool numeric_labels) {
    class_labels_ = std::move(labels);
    numeric_labels_ = numeric_labels;
}

int GradientBoostingModel::MaxDepth() const {
    int depth = 0;
    for (const auto& round : trees_) {
        for (const auto& tree : round) {
            depth = std::max(depth, tree.MaxDepth());
        }
    }
    return depth;
}

} // namespace cyxwiz
