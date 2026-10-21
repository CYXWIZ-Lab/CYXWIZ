#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct DecisionTreeNode {
    bool is_leaf = true;
    int feature_index = -1;
    double threshold = 0.0;
    int left_child = -1;
    int right_child = -1;
    int predicted_class = 0;
    double impurity = 0.0;
    size_t sample_count = 0;
};

class DecisionTreeModel {
public:
    int PredictClass(const std::vector<double>& row) const;
    std::vector<int> PredictClasses(
        const std::vector<std::vector<double>>& rows) const;

    void SetNodes(std::vector<DecisionTreeNode> nodes);
    void SetFeatureNames(std::vector<std::string> feature_names);
    void SetClassLabels(std::vector<std::string> labels, bool numeric_labels);

    const std::vector<DecisionTreeNode>& Nodes() const { return nodes_; }
    const std::vector<std::string>& FeatureNames() const { return feature_names_; }
    const std::vector<std::string>& ClassLabels() const { return class_labels_; }
    bool HasNumericLabels() const { return numeric_labels_; }
    int MaxDepth() const;

private:
    int MaxDepthFromNode(int node_index) const;

    std::vector<DecisionTreeNode> nodes_;
    std::vector<std::string> feature_names_;
    std::vector<std::string> class_labels_;
    bool numeric_labels_ = false;
};

} // namespace cyxwiz
