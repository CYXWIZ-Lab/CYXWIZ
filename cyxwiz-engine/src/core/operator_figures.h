#pragma once

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

// Named figures and facts that a step reports about its own run, for views
// that chart it (the Word Embeddings node's epochs on the Training Dashboard).
// A light copy of an operator's progress event: no Arrow, no executor.
struct OperatorFigures {
    int node_id = -1;
    std::string node_name;
    std::string stage;
    std::string message;
    float progress = 0.0f;
    std::vector<std::pair<std::string, double>> figures;
    std::vector<std::pair<std::string, std::string>> notes;

    bool Has(const std::string& name) const {
        for (const auto& figure : figures) {
            if (figure.first == name) return true;
        }
        return false;
    }
    double Figure(const std::string& name, double fallback = -1.0) const {
        for (const auto& figure : figures) {
            if (figure.first == name) return figure.second;
        }
        return fallback;
    }
    std::string Note(const std::string& name) const {
        for (const auto& note : notes) {
            if (note.first == name) return note.second;
        }
        return {};
    }
};

}  // namespace cyxwiz
