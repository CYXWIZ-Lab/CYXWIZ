#include "decision_tree_model.h"

#include <algorithm>

namespace cyxwiz {

int DecisionTreeModel::PredictClass(const std::vector<double>& row) const {
    if (nodes_.empty()) {
        return 0;
    }

    int node_index = 0;
    while (node_index >= 0 &&
           node_index < static_cast<int>(nodes_.size())) {
        const auto& node = nodes_[static_cast<size_t>(node_index)];
        if (node.is_leaf || node.feature_index < 0 ||
            node.feature_index >= static_cast<int>(row.size())) {
            return node.predicted_class;
        }
        node_index = row[static_cast<size_t>(node.feature_index)] <=
                         node.threshold
                         ? node.left_child
                         : node.right_child;
    }
    return nodes_.front().predicted_class;
}

std::vector<int> DecisionTreeModel::PredictClasses(
    const std::vector<std::vector<double>>& rows) const {
    std::vector<int> out;
    out.reserve(rows.size());
    for (const auto& row : rows) {
        out.push_back(PredictClass(row));
    }
    return out;
}

void DecisionTreeModel::SetNodes(std::vector<DecisionTreeNode> nodes) {
    nodes_ = std::move(nodes);
}

void DecisionTreeModel::SetFeatureNames(std::vector<std::string> feature_names) {
    feature_names_ = std::move(feature_names);
}

void DecisionTreeModel::SetClassLabels(std::vector<std::string> labels,
                                       bool numeric_labels) {
    class_labels_ = std::move(labels);
    numeric_labels_ = numeric_labels;
}

int DecisionTreeModel::MaxDepth() const {
    if (nodes_.empty()) {
        return 0;
    }
    return MaxDepthFromNode(0);
}

int DecisionTreeModel::MaxDepthFromNode(int node_index) const {
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

} // namespace cyxwiz
