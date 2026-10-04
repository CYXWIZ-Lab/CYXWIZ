#include "plot_tree_model.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>

namespace cyxwiz::plot {

namespace {

using json = nlohmann::json;

// A double NaN: json::value(key, NAN) would read the number as a float.
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// A split value as people read it: 61.5, 0.25, and from 10,000 up a whole
// number with thousands separators (239,066 rather than 2.391e+05).
std::string Threshold(double v) {
    char buf[48];
    if (!std::isfinite(v) || std::abs(v) < 1e4 || std::abs(v) >= 1e15) {
        std::snprintf(buf, sizeof(buf), "%.4g", v);
        return buf;
    }
    std::snprintf(buf, sizeof(buf), "%.0f", std::abs(v));
    const std::string digits = buf;
    std::string out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i > 0 && (digits.size() - i) % 3 == 0) out += ',';
        out += digits[i];
    }
    return (v < 0 ? "-" : "") + out;
}

std::string SignedValue(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%+.3g", v);
    return buf;
}

std::string At(const std::vector<std::string>& names, long long i, const std::string& fallback) {
    return i >= 0 && static_cast<size_t>(i) < names.size() ? names[static_cast<size_t>(i)] : fallback;
}

// One tree's nodes, from the root (node 0) down; nodes no split reaches are
// left out. `boosting`: regression nodes (value) instead of classifier nodes.
void AddTree(TreeModelRows& out, const json& nodes, const std::vector<std::string>& features,
             const std::vector<std::string>& classes, int tree, const std::string& tree_name, const std::string& prefix,
             bool boosting, const std::string& tree_class) {
    const size_t n = nodes.size();
    if (n == 0) return;
    std::vector<std::string> text(n);
    std::vector<long long> parent(n, -1);
    std::vector<char> reached(n, 0);
    std::vector<size_t> order{0};
    reached[0] = 1;
    for (size_t k = 0; k < order.size(); ++k) {
        const json& nd = nodes[order[k]];
        if (nd.value("is_leaf", false)) continue;
        for (const char* side : {"left_child", "right_child"}) {
            const long long c = nd.value(side, -1LL);
            if (c < 0 || static_cast<size_t>(c) >= n || reached[static_cast<size_t>(c)]) continue;
            reached[static_cast<size_t>(c)] = 1;
            parent[static_cast<size_t>(c)] = static_cast<long long>(order[k]);
            order.push_back(static_cast<size_t>(c));
        }
    }
    for (size_t i : order) {
        const json& nd = nodes[i];
        const bool leaf = nd.value("is_leaf", false);
        const std::string id = prefix + std::to_string(i);
        std::string rule = "leaf", cls = tree_class;
        double samples = NAN, value = NAN;
        if (boosting) {
            if (leaf) value = nd.value("value", kNaN);
        } else {
            cls = At(classes, nd.value("predicted_class", -1LL), std::to_string(nd.value("predicted_class", -1LL)));
            samples = static_cast<double>(nd.value("sample_count", 0ULL));
            value = nd.value("impurity", kNaN);
        }
        if (!leaf) {
            const long long f = nd.value("feature_index", -1LL);
            rule = At(features, f, "feature " + std::to_string(f)) + " <= " + Threshold(nd.value("threshold", 0.0));
            text[i] = id + ": " + rule;
        } else {
            text[i] = "leaf " + id + ": " + (boosting ? SignedValue(value) : cls);
        }
        out.tree.push_back(tree);
        out.tree_name.push_back(tree_name);
        out.node.push_back(text[i]);
        out.parent.push_back(parent[i] >= 0 ? text[static_cast<size_t>(parent[i])] : std::string());
        out.rule.push_back(rule);
        out.samples.push_back(samples);
        out.cls.push_back(cls);
        out.value.push_back(value);
    }
}

std::vector<std::string> Names(const json& model, const char* key) {
    return model.contains(key) ? model.at(key).get<std::vector<std::string>>() : std::vector<std::string>{};
}

}  // namespace

const std::vector<std::string>& TreeModelColumns() {
    static const std::vector<std::string> kColumns = {"tree", "tree_name", "node", "parent", "rule", "samples", "class", "value", "trees"};
    return kColumns;
}

TreeModelRows ReadTreeModel(const std::string& json_text) {
    TreeModelRows out;
    try {
        const json doc = json::parse(json_text);
        if (!doc.is_object() || doc.value("format", "") != "cyxwiz_tree_model" || !doc.contains("model")) {
            out.error = "This is not a CyxWiz tree model.";
            return out;
        }
        out.model_type = doc.value("model_type", "");
        const json& model = doc.at("model");
        const auto classes = Names(model, "class_labels");
        if (out.model_type == "DecisionTreeClassifier") {
            out.trees = 1;
            AddTree(out, model.at("nodes"), Names(model, "feature_names"), classes, 1, "Tree", "", false, "");
        } else if (out.model_type == "RandomForestClassifier") {
            const json& trees = model.at("trees");
            out.trees = static_cast<int>(trees.size());
            for (size_t t = 0; t < trees.size(); ++t) {
                // Each tree saves the names of its own features.
                const json& inner = trees[t].at("model");
                const int number = static_cast<int>(t) + 1;
                AddTree(out, inner.at("nodes"), Names(inner, "feature_names"), classes, number, "Tree " + std::to_string(number),
                        std::to_string(number) + ".", false, "");
            }
        } else if (out.model_type == "GradientBoostingClassifier") {
            // trees[round][class]: one regression tree per class each round (one against the rest).
            const auto features = Names(model, "feature_names");
            const json& rounds = model.at("trees");
            int number = 0;
            for (size_t r = 0; r < rounds.size(); ++r)
                for (size_t c = 0; c < rounds[r].size(); ++c) {
                    ++number;
                    const std::string cls = rounds[r].size() == 1 && classes.size() == 2 ? classes[1] : At(classes, static_cast<long long>(c), std::to_string(c));
                    AddTree(out, rounds[r][c].at("nodes"), features, classes, number,
                            "Round " + std::to_string(r + 1) + " \xC2\xB7 " + cls, std::to_string(number) + ".", true, cls);
                }
            out.trees = number;
        } else {
            out.error = "Unknown tree model type \"" + out.model_type + "\".";
            return out;
        }
        if (out.node.empty()) out.error = "The model has no tree nodes.";
    } catch (const std::exception& ex) {
        out = TreeModelRows{};
        out.error = std::string("The tree model cannot be read: ") + ex.what();
    }
    return out;
}

}  // namespace cyxwiz::plot
