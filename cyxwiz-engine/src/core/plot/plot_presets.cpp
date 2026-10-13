#include "plot_presets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::plot {

namespace {

bool Has(const std::vector<std::string>& columns, std::initializer_list<const char*> names) {
    for (const char* n : names)
        if (std::find(columns.begin(), columns.end(), n) == columns.end()) return false;
    return true;
}

std::string WithScore(const char* title, const char* name, double v) {
    if (!std::isfinite(v)) return title;
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s \xC2\xB7 %s %.3f", title, name, v);
    return buf;
}

}  // namespace

std::optional<PlotSpec> EvaluationPreset(const std::vector<std::string>& columns,
                                         const std::function<double(const std::string&)>& first_value) {
    PlotSpec s;
    // Confusion Matrix: one row per (actual, predicted) cell.
    if (Has(columns, {"actual_label", "predicted_label", "count"})) {
        s.kind = Kind::Heatmap;
        s.x_column = "predicted_label";
        s.y_columns = {"actual_label"};
        // "value" is the count, or its share when the node normalizes.
        s.value_column = Has(columns, {"value"}) ? "value" : "count";
        s.title = "Confusion matrix";
        s.x_label = "Predicted";
        s.y_label = "Actual";
        return s;
    }
    // ROC Curve: false and true positive rate per threshold, AUC repeated.
    if (Has(columns, {"fpr", "tpr"})) {
        s.kind = Kind::Line;
        s.x_column = "fpr";
        s.y_columns = {"tpr"};
        s.show_diagonal = true;  // chance
        s.title = WithScore("ROC curve", "AUC", Has(columns, {"auc"}) ? first_value("auc") : NAN);
        s.x_label = "False positive rate";
        s.y_label = "True positive rate";
        return s;
    }
    // PR Curve: precision by recall, average precision repeated.
    if (Has(columns, {"precision", "recall"})) {
        s.kind = Kind::Line;
        s.x_column = "recall";
        s.y_columns = {"precision"};
        s.title = WithScore("Precision-recall curve", "AP",
                            Has(columns, {"average_precision"}) ? first_value("average_precision") : NAN);
        s.x_label = "Recall";
        s.y_label = "Precision";
        return s;
    }
    // A trained tree model (TOFIX134 P4.7, plot_tree_model): the Tree, sized
    // by training rows and coloured by class; a forest shows its first tree.
    if (Has(columns, {"tree", "tree_name", "node", "parent", "rule", "trees"})) {
        s.kind = Kind::Tree;
        s.x_column = "node";
        s.y_columns = {"parent"};
        const bool boosting = !std::isfinite(first_value("samples"));
        s.value_column = boosting ? "value" : "samples";
        if (!boosting) s.color_column = "class";
        const double trees = first_value("trees");
        if (std::isfinite(trees) && trees > 1) {
            s.rows = RowMode::Filter;
            s.conditions = {{"tree", "=", "1"}};
        }
        s.title = boosting ? "Boosted trees" : std::isfinite(trees) && trees > 1 ? "Random forest" : "Decision tree";
        return s;
    }
    return std::nullopt;
}

}  // namespace cyxwiz::plot
