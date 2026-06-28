#pragma once

#include "decision_tree_model.h"

#include <string>
#include <vector>

namespace cyxwiz {

struct DecisionTreeTrainingOptions {
    int max_depth = 10;
    int min_samples_split = 2;
    int min_samples_leaf = 1;
    std::string criterion = "gini";
};

class DecisionTreeTrainer {
public:
    explicit DecisionTreeTrainer(DecisionTreeTrainingOptions options);

    DecisionTreeModel Fit(const std::vector<std::vector<double>>& features,
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

    int BuildNode(const std::vector<size_t>& indices,
                  int depth,
                  const std::vector<std::vector<double>>& features,
                  const std::vector<int>& labels,
                  size_t class_count,
                  std::vector<DecisionTreeNode>& nodes) const;

    SplitCandidate FindBestSplit(
        const std::vector<size_t>& indices,
        const std::vector<std::vector<double>>& features,
        const std::vector<int>& labels,
        size_t class_count) const;

    double Impurity(const std::vector<size_t>& indices,
                    const std::vector<int>& labels,
                    size_t class_count) const;
    int MajorityClass(const std::vector<size_t>& indices,
                      const std::vector<int>& labels,
                      size_t class_count) const;

    DecisionTreeTrainingOptions options_;
};

} // namespace cyxwiz
