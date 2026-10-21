#pragma once

#include "decision_tree_model.h"

#include <cstddef>
#include <string>
#include <vector>

namespace cyxwiz {

struct RandomForestTree {
    DecisionTreeModel model;
    std::vector<size_t> feature_indices;
};

class RandomForestModel {
public:
    int PredictClass(const std::vector<double>& row) const;
    std::vector<int> PredictClasses(
        const std::vector<std::vector<double>>& rows) const;

    void SetTrees(std::vector<RandomForestTree> trees);
    void SetClassLabels(std::vector<std::string> labels, bool numeric_labels);
    void SetFeatureNames(std::vector<std::string> feature_names);

    const std::vector<RandomForestTree>& Trees() const { return trees_; }
    const std::vector<std::string>& ClassLabels() const { return class_labels_; }
    const std::vector<std::string>& FeatureNames() const { return feature_names_; }
    bool HasNumericLabels() const { return numeric_labels_; }
    int MaxDepth() const;

private:
    std::vector<RandomForestTree> trees_;
    std::vector<std::string> class_labels_;
    std::vector<std::string> feature_names_;
    bool numeric_labels_ = false;
};

} // namespace cyxwiz
