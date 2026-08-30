#pragma once

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz::classification_metric_detail {

inline bool ValidateBinaryScores(
    const std::vector<int>& y_true,
    const std::vector<double>& y_scores,
    std::string& error_message,
    std::optional<double> threshold = std::nullopt) {
    if (y_true.empty()) {
        error_message = "Inputs must not be empty";
        return false;
    }
    if (y_true.size() != y_scores.size()) {
        error_message = "True labels and scores must have equal length";
        return false;
    }
    if (y_true.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error_message = "Sample count exceeds the public result range";
        return false;
    }
    if (threshold && !std::isfinite(*threshold)) {
        error_message = "Threshold must be finite";
        return false;
    }
    for (size_t index = 0; index < y_true.size(); ++index) {
        if (y_true[index] != 0 && y_true[index] != 1) {
            error_message = "Binary labels must be 0 or 1 at index " +
                            std::to_string(index);
            return false;
        }
        if (!std::isfinite(y_scores[index])) {
            error_message = "Scores must be finite at index " +
                            std::to_string(index);
            return false;
        }
    }
    return true;
}

inline bool ValidateMulticlassScores(
    const std::vector<int>& y_true,
    const std::vector<std::vector<double>>& y_scores,
    size_t& class_count,
    std::string& error_message) {
    class_count = 0;
    if (y_true.empty() || y_scores.empty()) {
        error_message = "Inputs must not be empty";
        return false;
    }
    if (y_true.size() != y_scores.size()) {
        error_message = "True labels and score rows must have equal length";
        return false;
    }
    if (y_true.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error_message = "Sample count exceeds the public result range";
        return false;
    }
    class_count = y_scores.front().size();
    if (class_count < 2 ||
        class_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        error_message = "Multiclass scores must contain at least two classes";
        return false;
    }

    std::vector<bool> seen(class_count, false);
    for (size_t row = 0; row < y_scores.size(); ++row) {
        if (y_scores[row].size() != class_count) {
            error_message = "Every score row must have the same class width";
            return false;
        }
        if (y_true[row] < 0 ||
            static_cast<size_t>(y_true[row]) >= class_count) {
            error_message = "Target label is outside the score-column range at row " +
                            std::to_string(row);
            return false;
        }
        seen[static_cast<size_t>(y_true[row])] = true;
        for (size_t column = 0; column < class_count; ++column) {
            if (!std::isfinite(y_scores[row][column])) {
                error_message = "Scores must be finite at row " +
                                std::to_string(row) + ", column " +
                                std::to_string(column);
                return false;
            }
        }
    }
    for (size_t class_index = 0; class_index < class_count; ++class_index) {
        if (!seen[class_index]) {
            error_message = "Every score column must have a target sample; missing class " +
                            std::to_string(class_index);
            return false;
        }
    }
    return true;
}

} // namespace cyxwiz::classification_metric_detail
