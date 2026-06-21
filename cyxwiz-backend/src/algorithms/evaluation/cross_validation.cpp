#include "cyxwiz/model_evaluation.h"

#include <algorithm>
#include <numeric>
#include <random>

namespace cyxwiz {

std::vector<std::pair<std::vector<int>, std::vector<int>>> ModelEvaluation::KFoldSplit(
    int n_samples,
    int n_folds,
    bool shuffle,
    unsigned int seed) {

    std::vector<std::pair<std::vector<int>, std::vector<int>>> folds;

    if (n_samples < n_folds || n_folds < 2) {
        return folds;
    }

    // Create index array
    std::vector<int> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::mt19937 gen(seed);
        std::shuffle(indices.begin(), indices.end(), gen);
    }

    // Compute fold sizes
    int fold_size = n_samples / n_folds;
    int remainder = n_samples % n_folds;

    int start = 0;
    for (int f = 0; f < n_folds; ++f) {
        int current_fold_size = fold_size + (f < remainder ? 1 : 0);
        int end = start + current_fold_size;

        std::vector<int> test_idx(indices.begin() + start, indices.begin() + end);
        std::vector<int> train_idx;
        train_idx.reserve(n_samples - current_fold_size);

        for (int i = 0; i < start; ++i) train_idx.push_back(indices[i]);
        for (int i = end; i < n_samples; ++i) train_idx.push_back(indices[i]);

        folds.emplace_back(train_idx, test_idx);
        start = end;
    }

    return folds;
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> ModelEvaluation::StratifiedKFoldSplit(
    const std::vector<int>& labels,
    int n_folds,
    bool shuffle,
    unsigned int seed) {

    std::vector<std::pair<std::vector<int>, std::vector<int>>> folds;
    int n_samples = static_cast<int>(labels.size());

    if (n_samples < n_folds || n_folds < 2) {
        return folds;
    }

    // Group indices by class
    std::map<int, std::vector<int>> class_indices;
    for (int i = 0; i < n_samples; ++i) {
        class_indices[labels[i]].push_back(i);
    }

    // Shuffle within each class if requested
    if (shuffle) {
        std::mt19937 gen(seed);
        for (auto& [cls, indices] : class_indices) {
            std::shuffle(indices.begin(), indices.end(), gen);
        }
    }

    // Initialize fold indices
    std::vector<std::vector<int>> fold_indices(n_folds);

    // Distribute samples from each class across folds
    for (auto& [cls, indices] : class_indices) {
        int n_cls = static_cast<int>(indices.size());
        int fold_size = n_cls / n_folds;
        int remainder = n_cls % n_folds;

        int start = 0;
        for (int f = 0; f < n_folds; ++f) {
            int current_size = fold_size + (f < remainder ? 1 : 0);
            for (int i = start; i < start + current_size; ++i) {
                fold_indices[f].push_back(indices[i]);
            }
            start += current_size;
        }
    }

    // Create train/test splits
    for (int f = 0; f < n_folds; ++f) {
        std::vector<int> test_idx = fold_indices[f];
        std::vector<int> train_idx;
        train_idx.reserve(n_samples - test_idx.size());

        for (int g = 0; g < n_folds; ++g) {
            if (g != f) {
                train_idx.insert(train_idx.end(), fold_indices[g].begin(), fold_indices[g].end());
            }
        }

        folds.emplace_back(train_idx, test_idx);
    }

    return folds;
}

std::vector<int> ModelEvaluation::GenerateTrainSizes(
    int n_samples,
    int n_points,
    double min_ratio,
    double max_ratio) {

    std::vector<int> sizes;

    if (n_points < 2 || n_samples < 10) {
        sizes.push_back(n_samples);
        return sizes;
    }

    int min_size = static_cast<int>(n_samples * min_ratio);
    int max_size = static_cast<int>(n_samples * max_ratio);

    min_size = (min_size < 10) ? 10 : min_size;
    max_size = (max_size > n_samples) ? n_samples : max_size;

    // Generate linearly spaced sizes
    for (int i = 0; i < n_points; ++i) {
        double ratio = static_cast<double>(i) / (n_points - 1);
        int size = min_size + static_cast<int>(ratio * (max_size - min_size));
        sizes.push_back(size);
    }

    return sizes;
}

} // namespace cyxwiz
