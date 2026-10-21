#include "random_forest_model.h"

#include <algorithm>
#include <utility>

namespace cyxwiz {

int RandomForestModel::PredictClass(const std::vector<double>& row) const {
    if (trees_.empty() || class_labels_.empty()) {
        return 0;
    }

    std::vector<size_t> votes(class_labels_.size(), 0);
    for (const auto& tree : trees_) {
        std::vector<double> projected;
        projected.reserve(tree.feature_indices.size());
        for (size_t feature_index : tree.feature_indices) {
            projected.push_back(feature_index < row.size() ? row[feature_index] : 0.0);
        }
        const int cls = tree.model.PredictClass(projected);
        if (cls >= 0 && cls < static_cast<int>(votes.size())) {
            ++votes[static_cast<size_t>(cls)];
        }
    }

    int best_class = 0;
    size_t best_votes = 0;
    for (size_t cls = 0; cls < votes.size(); ++cls) {
        if (votes[cls] > best_votes) {
            best_votes = votes[cls];
            best_class = static_cast<int>(cls);
        }
    }
    return best_class;
}

std::vector<int> RandomForestModel::PredictClasses(
    const std::vector<std::vector<double>>& rows) const {
    std::vector<int> out;
    out.reserve(rows.size());
    for (const auto& row : rows) {
        out.push_back(PredictClass(row));
    }
    return out;
}

void RandomForestModel::SetTrees(std::vector<RandomForestTree> trees) {
    trees_ = std::move(trees);
}

void RandomForestModel::SetClassLabels(std::vector<std::string> labels,
                                       bool numeric_labels) {
    class_labels_ = std::move(labels);
    numeric_labels_ = numeric_labels;
}

void RandomForestModel::SetFeatureNames(std::vector<std::string> feature_names) {
    feature_names_ = std::move(feature_names);
}

int RandomForestModel::MaxDepth() const {
    int depth = 0;
    for (const auto& tree : trees_) {
        depth = std::max(depth, tree.model.MaxDepth());
    }
    return depth;
}

} // namespace cyxwiz
