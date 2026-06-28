#pragma once

#include <string>
#include <vector>

namespace cyxwiz {

struct GradientBoostingRegressionNode {
    bool is_leaf = true;
    int feature_index = -1;
    double threshold = 0.0;
    int left_child = -1;
    int right_child = -1;
    double value = 0.0;
};

class GradientBoostingRegressionTree {
public:
    double PredictValue(const std::vector<double>& row) const;

    void SetNodes(std::vector<GradientBoostingRegressionNode> nodes);
    const std::vector<GradientBoostingRegressionNode>& Nodes() const {
        return nodes_;
    }
    int MaxDepth() const;

private:
    int MaxDepthFromNode(int node_index) const;

    std::vector<GradientBoostingRegressionNode> nodes_;
};

class GradientBoostingModel {
public:
    int PredictClass(const std::vector<double>& row) const;
    std::vector<int> PredictClasses(
        const std::vector<std::vector<double>>& rows) const;

    void SetInitialScores(std::vector<double> scores);
    void SetTrees(std::vector<std::vector<GradientBoostingRegressionTree>> trees);
    void SetLearningRate(double learning_rate);
    void SetFeatureNames(std::vector<std::string> feature_names);
    void SetClassLabels(std::vector<std::string> labels, bool numeric_labels);

    const std::vector<std::vector<GradientBoostingRegressionTree>>& Trees() const {
        return trees_;
    }
    const std::vector<std::string>& ClassLabels() const { return class_labels_; }
    const std::vector<std::string>& FeatureNames() const { return feature_names_; }
    bool HasNumericLabels() const { return numeric_labels_; }
    int MaxDepth() const;

private:
    std::vector<double> initial_scores_;
    std::vector<std::vector<GradientBoostingRegressionTree>> trees_;
    std::vector<std::string> class_labels_;
    std::vector<std::string> feature_names_;
    double learning_rate_ = 0.1;
    bool numeric_labels_ = false;
};

} // namespace cyxwiz
