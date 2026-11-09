#include "where_it_looks_presentation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace cyxwiz::where_it_looks {

namespace {

constexpr const char* kTimes = " \xC3\x97 ";  // " × "

std::string Fixed(double value, int digits) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", digits, value);
    return buffer;
}

}  // namespace

std::vector<std::string> OutputNames(layer_view::OutputRule rule, size_t outputs,
                                     const std::vector<std::string>& class_names) {
    using layer_view::OutputRule;
    const bool one_logit = outputs == 1 && (rule == OutputRule::Sigmoid || rule == OutputRule::Probability);
    const size_t count = one_logit ? 2 : outputs;
    std::vector<std::string> names;
    for (size_t i = 0; i < count; ++i) {
        if (rule == OutputRule::Value || rule == OutputRule::Unknown)
            names.push_back("output " + std::to_string(i));
        else
            names.push_back(layer_view::ClassName(static_cast<int>(i), class_names));
    }
    return names;
}

std::vector<Target> Targets(const std::vector<std::string>& names, int predicted, int truth) {
    std::vector<Target> targets;
    const auto valid = [&](int i) { return i >= 0 && static_cast<size_t>(i) < names.size(); };
    if (valid(predicted))
        targets.push_back({predicted, "Predicted: " + names[predicted], "The class the model picked for this sample"});
    if (valid(truth) && truth != predicted)
        targets.push_back({truth, "True: " + names[truth], "The sample's label"});
    for (size_t i = 0; i < names.size(); ++i) {
        const int index = static_cast<int>(i);
        if ((valid(predicted) && index == predicted) || (valid(truth) && index == truth)) continue;
        targets.push_back({index, names[i], "Another output of the model"});
    }
    return targets;
}

std::string CoverText(const occlusion::Plan& plan, const std::string& fill_source) {
    switch (plan.kind) {
        case occlusion::InputKind::Image:
            return "Patch " + std::to_string(plan.patch) + kTimes + std::to_string(plan.patch) + " px, every " +
                   std::to_string(plan.step) + " px, filled with " + fill_source;
        case occlusion::InputKind::Tokens:
            return "Each token in turn, replaced by " + fill_source + "; padding is left out";
        case occlusion::InputKind::Features:
            return "Each feature in turn, replaced by " + fill_source;
    }
    return {};
}

std::string PassesText(const occlusion::Plan& plan, size_t pieces, size_t chunk) {
    std::string text = std::to_string(pieces) + " forward pass" + (pieces == 1 ? "" : "es");
    if (plan.kind == occlusion::InputKind::Image) {
        const auto [rows, columns] = occlusion::PatchGrid(plan);
        text += " (" + std::to_string(rows) + kTimes + std::to_string(columns) + " patch positions)";
    } else {
        text += plan.kind == occlusion::InputKind::Tokens ? ", one per token" : ", one per feature";
    }
    return text + ", " + std::to_string(std::min(pieces, chunk)) + " at a time on the device, and one on the "
                  "unchanged input.";
}

std::vector<float> Scaled(const std::vector<double>& change) {
    double top = 0.0;
    for (double v : change)
        if (std::isfinite(v)) top = std::max(top, std::fabs(v));
    std::vector<float> scaled;
    scaled.reserve(change.size());
    for (double v : change) scaled.push_back(top > 0.0 && std::isfinite(v) ? static_cast<float>(v / top) : 0.0f);
    return scaled;
}

std::string PieceText(const occlusion::Plan& plan, const occlusion::Piece& piece,
                      const std::vector<std::string>& piece_names) {
    if (plan.kind == occlusion::InputKind::Image) {
        return "rows " + std::to_string(piece.row) + "-" + std::to_string(piece.row + piece.rows - 1) + ", columns " +
               std::to_string(piece.column) + "-" + std::to_string(piece.column + piece.columns - 1);
    }
    if (piece.position < piece_names.size() && !piece_names[piece.position].empty())
        return plan.kind == occlusion::InputKind::Tokens ? "\"" + piece_names[piece.position] + "\""
                                                         : piece_names[piece.position];
    return (plan.kind == occlusion::InputKind::Tokens ? "the token at " : "feature ") + std::to_string(piece.position);
}

std::string Headline(const occlusion::Plan& plan, const std::vector<occlusion::Piece>& pieces,
                     const occlusion::Result& result, const std::string& target_name,
                     const std::vector<std::string>& piece_names) {
    if (result.change.empty() || pieces.empty()) return "Nothing was covered.";
    size_t lowest = 0;
    for (size_t i = 1; i < result.change.size() && i < pieces.size(); ++i)
        if (result.change[i] < result.change[lowest]) lowest = i;
    const std::string verb = plan.kind == occlusion::InputKind::Image ? "Covering " : "Replacing ";
    const std::string score = "the " + target_name + " score";
    if (!(result.change[lowest] < 0.0))
        return "No covered piece lowers " + score + " (" + Fixed(result.baseline, 3) + "): the model does not rely on "
               "any one of them.";
    return verb + PieceText(plan, pieces[lowest], piece_names) + " lowers " + score + " most: " +
           Fixed(result.baseline, 3) + " to " + Fixed(result.baseline + result.change[lowest], 3) + ".";
}

const std::string& InspectorHint() {
    static const std::string hint =
        "Occlusion covers one piece of the input at a time and runs the model again: where covering a piece lowers "
        "the picked class's score most, the model relies on that piece. Hover the map for each piece's change. A "
        "Local Debug run's model has its starting weights; capture a training run to see where a trained model "
        "looks.";
    return hint;
}

std::string ChangeText(double change) {
    if (std::fabs(change) < 0.0005) return "0.000";
    return (change > 0 ? "+" : "") + Fixed(change, 3);
}

}  // namespace cyxwiz::where_it_looks
