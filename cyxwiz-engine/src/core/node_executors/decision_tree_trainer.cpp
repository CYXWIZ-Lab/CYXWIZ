#include "decision_tree_trainer.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cyxwiz {

namespace {

std::vector<size_t> RangeIndices(size_t n) {
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

bool IsPure(const std::vector<size_t>& indices,
            const std::vector<int>& labels) {
    if (indices.empty()) {
        return true;
    }
    const int first = labels[indices.front()];
    for (size_t row : indices) {
        if (labels[row] != first) {
            return false;
        }
    }
    return true;
}

} // namespace

DecisionTreeTrainer::DecisionTreeTrainer(DecisionTreeTrainingOptions options)
    : options_(std::move(options)) {}

DecisionTreeModel DecisionTreeTrainer::Fit(
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    size_t class_count,
    const std::vector<std::string>& feature_names,
    const std::vector<std::string>& class_labels,
    bool numeric_labels) const {
    if (features.empty()) {
        throw std::runtime_error("DecisionTree: no training rows");
    }
    if (features.size() != labels.size()) {
        throw std::runtime_error("DecisionTree: feature/label row mismatch");
    }
    if (features.front().empty()) {
        throw std::runtime_error("DecisionTree: no feature columns");
    }
    if (class_count < 2) {
        throw std::runtime_error("DecisionTree: need at least two classes");
    }
    for (const auto& row : features) {
        if (row.size() != features.front().size()) {
            throw std::runtime_error("DecisionTree: ragged feature matrix");
        }
        for (double value : row) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "DecisionTree: feature matrix contains non-finite values");
            }
        }
    }

    std::vector<DecisionTreeNode> nodes;
    const auto indices = RangeIndices(features.size());
    BuildNode(indices, 0, features, labels, class_count, nodes);

    DecisionTreeModel model;
    model.SetNodes(std::move(nodes));
    model.SetFeatureNames(feature_names);
    model.SetClassLabels(class_labels, numeric_labels);
    return model;
}

int DecisionTreeTrainer::BuildNode(
    const std::vector<size_t>& indices,
    int depth,
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    size_t class_count,
    std::vector<DecisionTreeNode>& nodes) const {
    DecisionTreeNode node;
    node.sample_count = indices.size();
    node.impurity = Impurity(indices, labels, class_count);
    node.predicted_class = MajorityClass(indices, labels, class_count);

    const int node_index = static_cast<int>(nodes.size());
    nodes.push_back(node);

    if (indices.size() < static_cast<size_t>(options_.min_samples_split) ||
        depth >= options_.max_depth || IsPure(indices, labels)) {
        return node_index;
    }

    const SplitCandidate split =
        FindBestSplit(indices, features, labels, class_count);
    if (!split.valid || split.gain <= 1e-12) {
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
        BuildNode(left, depth + 1, features, labels, class_count, nodes);
    const int right_child =
        BuildNode(right, depth + 1, features, labels, class_count, nodes);

    auto& stored = nodes[static_cast<size_t>(node_index)];
    stored.is_leaf = false;
    stored.feature_index = split.feature_index;
    stored.threshold = split.threshold;
    stored.left_child = left_child;
    stored.right_child = right_child;
    return node_index;
}

DecisionTreeTrainer::SplitCandidate DecisionTreeTrainer::FindBestSplit(
    const std::vector<size_t>& indices,
    const std::vector<std::vector<double>>& features,
    const std::vector<int>& labels,
    size_t class_count) const {
    SplitCandidate best;
    const double parent_impurity = Impurity(indices, labels, class_count);
    if (parent_impurity <= 0.0 || indices.empty()) {
        return best;
    }

    const size_t feature_count = features.front().size();
    std::vector<std::pair<double, int>> sorted;
    sorted.reserve(indices.size());

    for (size_t feature = 0; feature < feature_count; ++feature) {
        sorted.clear();
        for (size_t row : indices) {
            sorted.emplace_back(features[row][feature], labels[row]);
        }
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) {
                      if (a.first == b.first) {
                          return a.second < b.second;
                      }
                      return a.first < b.first;
                  });

        std::vector<size_t> left_counts(class_count, 0);
        std::vector<size_t> right_counts(class_count, 0);
        for (const auto& item : sorted) {
            ++right_counts[static_cast<size_t>(item.second)];
        }

        size_t left_size = 0;
        size_t right_size = sorted.size();
        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const int label = sorted[i].second;
            ++left_counts[static_cast<size_t>(label)];
            --right_counts[static_cast<size_t>(label)];
            ++left_size;
            --right_size;

            if (sorted[i].first == sorted[i + 1].first) {
                continue;
            }
            if (left_size < static_cast<size_t>(options_.min_samples_leaf) ||
                right_size < static_cast<size_t>(options_.min_samples_leaf)) {
                continue;
            }

            const auto impurity_from_counts =
                [this](const std::vector<size_t>& counts, size_t total) {
                    if (total == 0) {
                        return 0.0;
                    }
                    double impurity = 0.0;
                    if (options_.criterion == "entropy") {
                        for (size_t count : counts) {
                            if (count == 0) continue;
                            const double p = static_cast<double>(count) /
                                             static_cast<double>(total);
                            impurity -= p * std::log2(p);
                        }
                    } else {
                        impurity = 1.0;
                        for (size_t count : counts) {
                            const double p = static_cast<double>(count) /
                                             static_cast<double>(total);
                            impurity -= p * p;
                        }
                    }
                    return impurity;
                };

            const double left_impurity =
                impurity_from_counts(left_counts, left_size);
            const double right_impurity =
                impurity_from_counts(right_counts, right_size);
            const double weighted_impurity =
                (static_cast<double>(left_size) * left_impurity +
                 static_cast<double>(right_size) * right_impurity) /
                static_cast<double>(sorted.size());
            const double gain = parent_impurity - weighted_impurity;
            if (gain > best.gain + 1e-12) {
                best.valid = true;
                best.feature_index = static_cast<int>(feature);
                best.threshold = (sorted[i].first + sorted[i + 1].first) / 2.0;
                best.gain = gain;
            }
        }
    }

    return best;
}

double DecisionTreeTrainer::Impurity(const std::vector<size_t>& indices,
                                     const std::vector<int>& labels,
                                     size_t class_count) const {
    if (indices.empty()) {
        return 0.0;
    }
    std::vector<size_t> counts(class_count, 0);
    for (size_t row : indices) {
        ++counts[static_cast<size_t>(labels[row])];
    }

    if (options_.criterion == "entropy") {
        double entropy = 0.0;
        for (size_t count : counts) {
            if (count == 0) continue;
            const double p = static_cast<double>(count) /
                             static_cast<double>(indices.size());
            entropy -= p * std::log2(p);
        }
        return entropy;
    }

    double gini = 1.0;
    for (size_t count : counts) {
        const double p =
            static_cast<double>(count) / static_cast<double>(indices.size());
        gini -= p * p;
    }
    return gini;
}

int DecisionTreeTrainer::MajorityClass(const std::vector<size_t>& indices,
                                       const std::vector<int>& labels,
                                       size_t class_count) const {
    std::vector<size_t> counts(class_count, 0);
    for (size_t row : indices) {
        ++counts[static_cast<size_t>(labels[row])];
    }

    int best_class = 0;
    size_t best_count = 0;
    for (size_t cls = 0; cls < counts.size(); ++cls) {
        if (counts[cls] > best_count) {
            best_count = counts[cls];
            best_class = static_cast<int>(cls);
        }
    }
    return best_class;
}

} // namespace cyxwiz
