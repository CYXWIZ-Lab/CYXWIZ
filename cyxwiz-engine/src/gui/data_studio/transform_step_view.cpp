#include "transform_step_view.h"

#include "../../core/data_registry.h"
#include "../../core/node_metadata_registry.h"
#include "../../core/preprocessing_state.h"
#include "../icons.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace cyxwiz {

namespace ts = transform_step;
using ::gui::NodeType;

namespace {

std::string P(const std::map<std::string, std::string>& params, const char* key, const char* fallback = "") {
    const auto it = params.find(key);
    return it == params.end() || it->second.empty() ? fallback : it->second;
}

double Number(const std::string& text, double fallback) {
    try {
        return std::stod(text);
    } catch (...) {
        return fallback;
    }
}

std::string Text(double v) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.6g", v);
    return text;
}

std::vector<std::string> Split(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        const auto b = item.find_first_not_of(" \t");
        const auto e = item.find_last_not_of(" \t");
        if (b != std::string::npos) out.push_back(item.substr(b, e - b + 1));
    }
    return out;
}

std::string Join(const std::vector<std::string>& items) {
    std::string out;
    for (const auto& item : items) out += (out.empty() ? "" : ",") + item;
    return out;
}

// A dim label before a row of controls.
void Label(const char* text) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
    ImGui::SameLine();
}

void Faint(const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(ui::CurrentTokens().text_faint, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

// Bars of a histogram, no outline; then its range under it.
void Histogram(const ts::Stats& s, float width, float height, const ImVec4& colour, bool axis = true) {
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    if (!s.ok || s.bins.empty()) return;
    const int64_t top = *std::max_element(s.bins.begin(), s.bins.end());
    if (top <= 0) return;
    ImDrawList* dl = ImGui::GetWindowDrawList();
    const float bw = width / static_cast<float>(s.bins.size());
    for (size_t i = 0; i < s.bins.size(); ++i) {
        if (s.bins[i] == 0) continue;
        const float h = std::max(1.0f, (height - 4.0f) * static_cast<float>(s.bins[i]) / static_cast<float>(top));
        const float x = at.x + static_cast<float>(i) * bw;
        dl->AddRectFilled(ImVec2(x + 1.0f, at.y + height - h), ImVec2(x + std::max(2.0f, bw - 1.0f), at.y + height),
                          ui::ToU32(colour), 2.0f, ImDrawFlags_RoundCornersTop);
    }
    if (!axis) return;
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string lo = ts::Value(s.min), hi = ts::Value(s.max);
    const float x0 = ImGui::GetCursorPosX();
    ImGui::TextColored(t.text_faint, "%s", lo.c_str());
    ImGui::SameLine();
    // SetCursorPosX, not SameLine(offset): inside a group SameLine adds the group's indent again.
    ImGui::SetCursorPosX(x0 + std::max(0.0f, width - ImGui::CalcTextSize(hi.c_str()).x));
    ImGui::TextColored(t.text_faint, "%s", hi.c_str());
}

// The log-likelihood by lambda, the best lambda marked (and a typed one).
void LikelihoodCurve(const ts::PowerFigures& p, float width, float height) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(width, height));
    if (p.lambdas.size() < 2) return;
    double lo = 0.0, hi = 0.0;
    bool any = false;
    for (double y : p.log_likelihoods) {
        if (!std::isfinite(y)) continue;
        lo = any ? std::min(lo, y) : y;
        hi = any ? std::max(hi, y) : y;
        any = true;
    }
    if (!any || hi <= lo) return;
    const double l0 = p.lambdas.front(), l1 = p.lambdas.back();
    const auto px = [&](double l) { return at.x + 10.0f + (width - 20.0f) * static_cast<float>((l - l0) / (l1 - l0)); };
    const auto py = [&](double y) { return at.y + height - 16.0f - (height - 28.0f) * static_cast<float>((y - lo) / (hi - lo)); };
    std::vector<ImVec2> points;
    for (size_t i = 0; i < p.lambdas.size() && i < p.log_likelihoods.size(); ++i)
        if (std::isfinite(p.log_likelihoods[i])) points.emplace_back(px(p.lambdas[i]), py(p.log_likelihoods[i]));
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->AddPolyline(points.data(), static_cast<int>(points.size()), ui::ToU32(t.accent_text), ImDrawFlags_None, 2.0f);
    const auto mark = [&](double l, const ImVec4& colour, const std::string& label, bool dashed) {
        if (!std::isfinite(l) || l < l0 || l > l1) return;
        const float x = px(l);
        for (float y = at.y + 6.0f; y < at.y + height - 16.0f; y += dashed ? 7.0f : height)
            dl->AddLine(ImVec2(x, y), ImVec2(x, std::min(y + (dashed ? 4.0f : height), at.y + height - 16.0f)), ui::ToU32(colour), 1.0f);
        const float w = ImGui::CalcTextSize(label.c_str()).x;
        const float lx = std::min(x + 4.0f, at.x + width - w);
        dl->AddText(ImVec2(lx, at.y), ui::ToU32(colour), label.c_str());
    };
    mark(p.best, t.caution, "best " + ts::Value(p.best), true);
    if (!p.auto_lambda && std::isfinite(p.lambda) && std::fabs(p.lambda - p.best) > 1e-9)
        mark(p.lambda, t.text_bright, "typed " + ts::Value(p.lambda), false);
    const std::string left = ts::Value(l0), right = ts::Value(l1);
    dl->AddText(ImVec2(at.x + 6.0f, at.y + height - ImGui::GetTextLineHeight()), ui::ToU32(t.text_faint), left.c_str());
    dl->AddText(ImVec2(at.x + width - 6.0f - ImGui::CalcTextSize(right.c_str()).x, at.y + height - ImGui::GetTextLineHeight()),
                ui::ToU32(t.text_faint), right.c_str());
}

// The table a node's result is kept under: a Data Input's loaded dataset, else its last run's output.
std::string ResultOf(const studio::GraphNode* node) {
    if (!node) return {};
    return node->output_dataset.empty() ? node->dataset_name : node->output_dataset;
}

bool BeginFigureTable(const char* id, int columns) {
    return ImGui::BeginTable(id, columns,
                             ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoSavedSettings);
}

void NumberCell(const std::string& text, const ImVec4* colour = nullptr) {
    ImGui::TableNextColumn();
    const float w = ImGui::CalcTextSize(text.c_str()).x;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
    if (colour) ImGui::TextColored(*colour, "%s", text.c_str());
    else ImGui::TextUnformatted(text.c_str());
}

void HeaderCells(std::initializer_list<const char*> names) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::TableNextRow();
    bool first = true;
    for (const char* name : names) {
        ImGui::TableNextColumn();
        if (first) {
            ImGui::TextColored(t.text_dim, "%s", name);
            first = false;
        } else {
            const float w = ImGui::CalcTextSize(name).x;
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - w));
            ImGui::TextColored(t.text_dim, "%s", name);
        }
    }
}

}  // namespace

void TransformStepView::Set(const std::string& key, const std::string& value) {
    if (!editor_ || !editor_->SetStepParameters(step_, {{key, value}})) return;
    message_ = "Changed; Run step shows its figures (Ctrl+Z on the canvas undoes it).";
    message_error_ = false;
}

void TransformStepView::Render(::gui::NodeEditor* editor, const studio::GraphNode& node, const studio::GraphNode* input) {
    const ui::Tokens& t = ui::CurrentTokens();
    editor_ = editor;
    if (step_ != node.id) {
        // A method switch replaced the step: its message stays with the new node.
        const bool replaced = editor && step_ >= 0 && editor->ReplacedBy(step_) == node.id;
        step_ = node.id;
        comparing_ = false;
        column_.clear();
        if (!replaced) message_.clear();
        figures_.Reset();
        compare_.Reset();
    }
    const ::gui::MLNode* step = editor ? editor->FindStep(node.id) : nullptr;
    if (!step) return;
    const std::map<std::string, std::string> params = step->parameters;  // Set changes the node
    const std::string input_dataset = ResultOf(input);
    if (schema_of_ != input_dataset) {
        schema_of_ = input_dataset;
        number_columns_.clear();
        if (!input_dataset.empty())
            if (const auto table = DataRegistry::Instance().ReadTablePreview(input_dataset, 0))
                number_columns_ = ts::NumberColumns(*table->schema());
    }
    ImGui::PushID("transform_step");
    RenderHeader(node, input, params);
    if (comparing_) {
        RenderCompare(node, input);
        ImGui::PopID();
        return;
    }
    RenderSettings(node, params);
    if (!message_.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(message_error_ ? t.error : t.text_dim, "%s", message_.c_str());
        ImGui::PopTextWrapPos();
    }
    ImGui::Spacing();
    const bool ran = node.ran && !node.output_dataset.empty() && node.error.empty() && !input_dataset.empty();
    if (editor->IsDataPipelineRunning()) {
        // Prepare's result line says it is running.
    } else if (!ran) {
        if (node.error.empty()) Faint("Run step shows the before / after of each column here.");
    } else {
        AskFigures(node, input, params);
        if (const auto* figures = figures_.Answer(figures_key_)) RenderFigures(node, *figures);
        else if (!figures_.Error().empty()) Faint("Before / after not worked out: " + figures_.Error());
        else Faint("Working out the before / after...");
    }
    RenderAllSettings(params);
    ImGui::PopID();
}

void TransformStepView::RenderHeader(const studio::GraphNode& node, const studio::GraphNode* input,
                                     const std::map<std::string, std::string>& params) {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::string line = (input ? "after " + input->name + " \xC2\xB7 " : std::string()) + ts::Formula(node.type, params);
    ImGui::TextColored(t.text_faint, "%s", line.c_str());
    if (node.ran && node.error.empty() && input) {
        const std::string run = ts::RunLine(input->rows, node.rows);
        ui::SameLineRight(ImGui::CalcTextSize(run.c_str()).x);
        ImGui::TextColored(t.success, "%s", run.c_str());
    }
}

void TransformStepView::RenderSettings(const studio::GraphNode& node, const std::map<std::string, std::string>& params) {
    const ui::Tokens& t = ui::CurrentTokens();
    switch (ts::KindOf(node.type)) {
        case ts::Kind::Scale: {
            const auto& methods = ts::ScaleMethods();
            std::vector<const char*> labels;
            int current = 0;
            for (size_t i = 0; i < methods.size(); ++i) {
                labels.push_back(methods[i].label);
                if (methods[i].type == node.type) current = static_cast<int>(i);
            }
            int chosen = current;
            Label("Method");
            if (ui::SegmentedControl("##method", labels.data(), static_cast<int>(labels.size()), &chosen) && chosen != current) {
                const auto& m = methods[static_cast<size_t>(chosen)];
                editor_->RequestReplaceStep(node.id, m.type, ts::CarriedParameters(params));
                message_ = std::string("Switched to ") + m.label + ": its node takes this step's place, links and columns "
                           "(Ctrl+Z on the canvas undoes it). Run step shows its figures.";
                message_error_ = false;
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Replaces the node with another scaling method, keeping its links and columns");
            ImGui::SameLine(0.0f, 12.0f);
            if (ui::LinkButton("Compare methods")) {
                comparing_ = true;
                if (compare_column_.empty()) compare_column_ = column_;
            }
            RenderColumns(params);
            if (node.type == NodeType::MinMaxScaler) {
                double lo = Number(P(params, "min", "0"), 0.0), hi = Number(P(params, "max", "1"), 1.0);
                Label("Range");
                ui::NumberFieldOptions range;
                range.width = 70.0f;
                range.hint = "Typed; the range must rise";
                const bool lo_changed = ui::NumberField("##range_min", &lo, range);
                ImGui::SameLine();
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(t.text_dim, "to");
                ImGui::SameLine();
                const bool hi_changed = ui::NumberField("##range_max", &hi, range);
                if (lo_changed || hi_changed) {
                    if (lo < hi) {
                        editor_->SetStepParameters(step_, {{"min", Text(lo)}, {"max", Text(hi)}});
                        message_ = "Changed; Run step shows its figures (Ctrl+Z on the canvas undoes it).";
                        message_error_ = false;
                    } else {
                        message_ = "The range must rise: the first number below the second.";
                        message_error_ = true;
                    }
                }
                ImGui::SameLine();
                ImGui::AlignTextToFramePadding();
                ImGui::TextColored(t.text_faint, "typed; the range must rise");
            } else if (node.type == NodeType::QuantileTransform) {
                static const char* const kShapes[] = {"Uniform", "Normal"};
                int shape = P(params, "output_distribution", "uniform") == "normal" ? 1 : 0;
                Label("Shape");
                if (ui::SegmentedControl("##shape", kShapes, 2, &shape)) Set("output_distribution", shape ? "normal" : "uniform");
                ImGui::SameLine(0.0f, 16.0f);
                Label("Quantiles");
                int quantiles = static_cast<int>(Number(P(params, "n_quantiles", "1000"), 1000.0));
                if (ui::NumberField("##quantiles", &quantiles, 2, 100000, 0, 70.0f, "How many quantiles to fit (at most the number of values)"))
                    Set("n_quantiles", std::to_string(quantiles));
            }
            RenderFit(params);
            break;
        }
        case ts::Kind::Power: {
            static const char* const kFamilies[] = {"Box-Cox", "Yeo-Johnson"};
            static const char* const kModes[] = {"Find the best", "Typed"};
            int family = P(params, "family", "yeo-johnson") == "box-cox" ? 0 : 1;
            int mode = P(params, "lambda_mode", "auto") == "fixed" ? 1 : 0;
            Label("Family");
            if (ui::SegmentedControl("##family", kFamilies, 2, &family)) Set("family", family == 0 ? "box-cox" : "yeo-johnson");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Box-Cox needs values above 0; Yeo-Johnson takes any value");
            ImGui::SameLine(0.0f, 16.0f);
            Label("Lambda");
            if (ui::SegmentedControl("##lambda_mode", kModes, 2, &mode)) Set("lambda_mode", mode ? "fixed" : "auto");
            if (mode == 1) {
                ImGui::SameLine();
                double lambda = Number(P(params, "lambda", "0"), 0.0);
                ui::NumberFieldOptions options;
                options.min = -10.0;
                options.max = 10.0;
                options.width = 60.0f;
                options.hint = "Lambda for every column (0: the log, for Box-Cox)";
                if (ui::NumberField("##lambda", &lambda, options)) Set("lambda", Text(lambda));
                for (const char* preset : {"-1", "-0.5", "0", "0.5", "1"}) {
                    ImGui::SameLine();
                    ImGui::PushID(preset);
                    if (ui::SecondaryButton(preset)) Set("lambda", preset);
                    ImGui::PopID();
                }
            }
            RenderColumns(params);
            RenderFit(params);
            break;
        }
        case ts::Kind::Log: {
            static const char* const kBases[] = {"e", "10", "2"};
            const std::string base = P(params, "base", "natural");
            int b = base == "log10" ? 1 : base == "log2" ? 2 : 0;
            Label("Base");
            if (ui::SegmentedControl("##base", kBases, 3, &b)) Set("base", b == 1 ? "log10" : b == 2 ? "log2" : "natural");
            ImGui::SameLine(0.0f, 16.0f);
            bool log1p = P(params, "use_log1p", "true") != "false";
            if (ImGui::Checkbox("log(1 + x)", &log1p)) Set("use_log1p", log1p ? "true" : "false");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Keeps zeros; values must be above -1. Off: log(x), values above 0.");
            RenderColumns(params);
            break;
        }
        case ts::Kind::None:
            break;
    }
}

void TransformStepView::RenderColumns(const std::map<std::string, std::string>& params) {
    const ui::Tokens& t = ui::CurrentTokens();
    const std::vector<std::string> chosen = Split(P(params, "columns"));
    const bool automatic = chosen.empty();
    const std::string label = P(params, "label_col");
    const std::vector<std::string> excluded = Split(P(params, "exclude_columns"));
    std::vector<std::string> shown = chosen;
    if (automatic)
        for (const auto& name : number_columns_)
            if (name != label && std::find(excluded.begin(), excluded.end(), name) == excluded.end()) shown.push_back(name);
    Label("Columns");
    const float start = ImGui::GetCursorPosX();
    for (size_t i = 0; i < shown.size(); ++i) {
        const std::string chip = shown[i] + "  " ICON_FA_XMARK;
        if (i > 0) {
            ImGui::SameLine();
            if (ImGui::GetContentRegionAvail().x < ui::ChipButtonWidth(chip.c_str())) {
                ImGui::NewLine();
                ImGui::SetCursorPosX(start);
            }
        }
        ImGui::PushID(static_cast<int>(i));
        if (ui::ChipButton(chip.c_str())) {
            if (shown.size() == 1) {
                message_ = "A step needs at least one column.";
                message_error_ = true;
            } else {
                std::vector<std::string> rest = shown;
                rest.erase(rest.begin() + static_cast<std::ptrdiff_t>(i));
                Set("columns", Join(rest));
            }
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Leave this column out");
        ImGui::PopID();
    }
    if (shown.empty()) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_dim, "every number column");
    }
    ImGui::SameLine();
    if (ImGui::GetContentRegionAvail().x < ui::ButtonWidth("Pick columns", ui::ButtonSize::Small)) {
        ImGui::NewLine();
        ImGui::SetCursorPosX(start);
    }
    if (ui::LinkButton("Pick columns")) ImGui::OpenPopup("##pick_columns");
    if (ImGui::BeginPopup("##pick_columns")) {
        if (ImGui::MenuItem("Every number column", nullptr, automatic)) Set("columns", "");
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Each number column of the input, except the label and the excluded columns, also ones added later");
        ImGui::Separator();
        if (number_columns_.empty()) ImGui::TextDisabled("Run the step's input to list its columns.");
        for (const auto& name : number_columns_) {
            const bool on = std::find(shown.begin(), shown.end(), name) != shown.end();
            if (ImGui::MenuItem(name.c_str(), nullptr, on, !(on && shown.size() == 1))) {
                std::vector<std::string> next;
                for (const auto& n : number_columns_) {
                    const bool keep = n == name ? !on : std::find(shown.begin(), shown.end(), n) != shown.end();
                    if (keep) next.push_back(n);
                }
                Set("columns", Join(next));
            }
        }
        ImGui::EndPopup();
    }
    if (automatic && !label.empty()) Faint("Every number column; the label " + label + " is left out.");
}

void TransformStepView::RenderFit(const std::map<std::string, std::string>& params) {
    const ui::Tokens& t = ui::CurrentTokens();
    static const char* const kFit[] = {"Fit here", "Use saved values"};
    const bool reuse = P(params, "operation_mode", "fit_transform") == "transform_only";
    int mode = reuse ? 1 : 0;
    Label("Fit");
    if (ui::SegmentedControl("##fit", kFit, 2, &mode)) Set("operation_mode", mode ? "transform_only" : "fit_transform");
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("Fit here: fitted on the rows that reach this step.\nUse saved values: values fitted on training data, "
                          "for validation, test and inference.");
    const std::string path = P(params, "state_path");
    if (!reuse) {
        ImGui::SameLine(0.0f, 12.0f);
        bool save = P(params, "save_state", "false") == "true";
        if (ImGui::Checkbox("Save the fitted values", &save)) Set("save_state", save ? "true" : "false");
        Faint(save ? "Saved to " + (path.empty() ? std::string("a state file named when it runs") : path) +
                         "; a validation or test Data Input reuses them."
                   : "Fitted on the rows that reach this step, each run.");
    } else if (path.empty()) {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.warning, "Pick the saved values file (.cyxstate.json) in the node's dialog.");
        ImGui::PopTextWrapPos();
    } else {
        Faint("Reads " + path + ".");
    }
}

void TransformStepView::AskFigures(const studio::GraphNode& node, const studio::GraphNode* input,
                                   const std::map<std::string, std::string>& params) {
    std::string key = node.output_dataset + "@" + node.finished_at + "|" + ResultOf(input) + "@" + input->finished_at;
    for (const char* k : {"columns", "label_col", "exclude_columns", "family", "lambda_mode", "lambda", "operation_mode", "state_path"})
        key += "|" + P(params, k);
    figures_key_ = key;
    const NodeType type = node.type;
    const std::string in = ResultOf(input), out = node.output_dataset;
    figures_.Ask("Prepare: before / after", key, [type, params, in, out](LambdaTask&) {
        ts::StepFigures f;
        const auto before = DataRegistry::Instance().ReadTablePreview(in, ts::kFigureRows);
        const auto after = DataRegistry::Instance().ReadTablePreview(out, ts::kFigureRows);
        if (!before || !after) {
            f.error = "the step's last run is no longer in memory; run it again.";
            return f;
        }
        std::map<std::string, double> saved;
        if (type == NodeType::PowerTransform && P(params, "operation_mode") == "transform_only") {
            FittedPreprocessingState state;
            std::string error;
            if (LoadFittedPreprocessingState(P(params, "state_path"), "PowerTransform", state, error))
                for (const auto& feature : state.features)
                    if (const auto it = feature.numeric_values.find("lambda"); it != feature.numeric_values.end())
                        saved[feature.name] = it->second;
        }
        f = ts::ComputeFigures(type, params, *before, *after, saved);
        f.sampled = before->num_rows() >= ts::kFigureRows || after->num_rows() >= ts::kFigureRows;
        return f;
    });
}

const ts::ColumnFigures* TransformStepView::Picked(const ts::StepFigures& figures) {
    for (const auto& c : figures.columns)
        if (c.name == column_) return &c;
    if (figures.columns.empty()) return nullptr;
    column_ = figures.columns.front().name;
    return &figures.columns.front();
}

void TransformStepView::RenderFigures(const studio::GraphNode& node, const ts::StepFigures& figures) {
    const ui::Tokens& t = ui::CurrentTokens();
    if (!figures.ok()) {
        Faint(figures.error);
        return;
    }
    ImGui::TextColored(t.text_dim, "BEFORE / AFTER (LAST RUN)");
    if (figures.sampled) {
        ImGui::SameLine();
        ImGui::TextColored(t.text_faint, "\xC2\xB7 the first 200,000 rows");
    }
    if (node.type == NodeType::PowerTransform) {
        RenderPowerFigures(figures);
        return;
    }
    const ts::ColumnFigures* picked = Picked(figures);
    if (BeginFigureTable("##before_after", 6)) {
        ImGui::TableSetupColumn("column", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("range before", ImGuiTableColumnFlags_WidthStretch, 1.5f);
        ImGui::TableSetupColumn("range after", ImGuiTableColumnFlags_WidthStretch, 1.5f);
        for (int i = 0; i < 3; ++i) ImGui::TableSetupColumn("figure", ImGuiTableColumnFlags_WidthStretch, 0.9f);
        HeaderCells({"column", "range before", "range after", "mean before", "mean after", "std after"});
        for (const auto& c : figures.columns) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(c.name.c_str());
            ui::PushListRowColours();
            if (ImGui::Selectable(c.name.c_str(), picked && picked->name == c.name, ImGuiSelectableFlags_SpanAllColumns))
                column_ = c.name;
            ui::PopListRowColours();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show its histograms below");
            ImGui::PopID();
            NumberCell(ts::Range(c.before));
            NumberCell(ts::Range(c.after));
            NumberCell(ts::Value(c.before.mean));
            NumberCell(ts::Value(c.after.mean));
            NumberCell(ts::Value(c.after.std));
        }
        ImGui::EndTable();
    }
    int64_t missing = 0;
    for (const auto& c : figures.columns) missing += c.after.missing;
    if (missing > 0) Faint("Missing values stay missing in their rows (" + std::to_string(missing) + " in these columns).");
    if (picked) {
        const float gap = 14.0f;
        const float half = std::max(120.0f, (ImGui::GetContentRegionAvail().x - gap) * 0.5f);
        ImGui::BeginGroup();
        ImGui::TextColored(t.text_dim, "%s before", picked->name.c_str());
        Histogram(picked->before, half, 90.0f, t.accent_text);
        ImGui::EndGroup();
        ImGui::SameLine(0.0f, gap);
        ImGui::BeginGroup();
        ImGui::TextColored(t.text_dim, "after");
        Histogram(picked->after, half, 90.0f, t.caution);
        ImGui::EndGroup();
    }
    if (node.type == NodeType::StandardScaler) Faint(ts::BeyondZ3Line(figures));
    if (ts::KindOf(node.type) == ts::Kind::Scale)
        Faint("The shape does not change for Min-Max, Z-score, Robust and Max-Abs, only the scale; Quantile changes the shape.");
}

void TransformStepView::RenderPowerFigures(const ts::StepFigures& figures) {
    const ui::Tokens& t = ui::CurrentTokens();
    const ts::ColumnFigures* picked = Picked(figures);
    if (!figures.columns.empty() && figures.columns.front().power.box_cox) Faint(ts::BoxCoxNote(figures));
    if (BeginFigureTable("##power", 6)) {
        ImGui::TableSetupColumn("column", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        for (int i = 0; i < 5; ++i) ImGui::TableSetupColumn("figure", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        HeaderCells({"column", "lambda", "skew before", "skew after", "normal p before", "normal p after"});
        for (const auto& c : figures.columns) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(c.name.c_str());
            ui::PushListRowColours();
            if (ImGui::Selectable(c.name.c_str(), picked && picked->name == c.name, ImGuiSelectableFlags_SpanAllColumns))
                column_ = c.name;
            ui::PopListRowColours();
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("Show its likelihood curve and histograms below");
            ImGui::PopID();
            const auto p_colour = [&](double p) { return std::isfinite(p) && p < 0.05 ? &t.caution : &t.success; };
            NumberCell(ts::Value(c.power.lambda));
            NumberCell(ts::Value(c.before.skew));
            NumberCell(ts::Value(c.after.skew));
            NumberCell(ts::PValue(c.power.p_before), std::isfinite(c.power.p_before) ? p_colour(c.power.p_before) : nullptr);
            NumberCell(ts::PValue(c.power.p_after), std::isfinite(c.power.p_after) ? p_colour(c.power.p_after) : nullptr);
        }
        ImGui::EndTable();
    }
    if (!picked) return;
    const float gap = 14.0f;
    const float avail = ImGui::GetContentRegionAvail().x;
    const float curve_w = std::max(200.0f, (avail - gap) * 0.55f);
    const float hist_w = std::max(140.0f, avail - gap - curve_w);
    ImGui::BeginGroup();
    ImGui::TextColored(t.text_dim, "Likelihood by lambda \xC2\xB7 %s", picked->name.c_str());
    LikelihoodCurve(picked->power, curve_w, 120.0f);
    ImGui::EndGroup();
    ImGui::SameLine(0.0f, gap);
    ImGui::BeginGroup();
    ImGui::TextColored(t.text_dim, "before / after");
    Histogram(picked->before, hist_w, 50.0f, t.accent_text, false);
    Histogram(picked->after, hist_w, 50.0f, t.caution, false);
    ImGui::EndGroup();
    const std::string sentence = ts::PowerSentence(*picked);
    if (!sentence.empty()) {
        const bool not_normal = std::isfinite(picked->power.p_after) && picked->power.p_after < 0.05;
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(not_normal || !picked->power.note.empty() ? t.caution : t.text_dim, "%s", sentence.c_str());
        ImGui::PopTextWrapPos();
    }
}

void TransformStepView::RenderCompare(const studio::GraphNode& node, const studio::GraphNode* input) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::TextColored(t.text_bright, "Compare methods");
    ImGui::SameLine();
    if (compare_column_.empty() && !number_columns_.empty()) compare_column_ = number_columns_.front();
    ImGui::TextColored(t.text_faint, "%s through every scaling method (preview; nothing is added)", compare_column_.c_str());
    ui::SameLineRight(ui::ButtonWidth(ICON_FA_ARROW_LEFT " Back to the step", ui::ButtonSize::Small));
    if (ui::LinkButton(ICON_FA_ARROW_LEFT " Back to the step")) comparing_ = false;
    if (ResultOf(input).empty()) {
        Faint("Run the step's input first: Compare reads its values.");
        return;
    }
    Label("Column");
    ImGui::SetNextItemWidth(220.0f);
    if (ImGui::BeginCombo("##compare_column", compare_column_.c_str())) {
        for (const auto& name : number_columns_)
            if (ImGui::Selectable(name.c_str(), name == compare_column_)) compare_column_ = name;
        ImGui::EndCombo();
    }
    const std::string key = ResultOf(input) + "@" + input->finished_at + "|" + compare_column_;
    const std::string dataset = ResultOf(input), column = compare_column_;
    compare_.Ask("Prepare: compare methods", key, [dataset, column](LambdaTask&) {
        const auto table = DataRegistry::Instance().ReadTablePreview(dataset, ts::kFigureRows);
        if (!table) {
            ts::CompareFigures f;
            f.error = "the step's input is no longer in memory; run it again.";
            return f;
        }
        return ts::CompareMethods(column, ts::ColumnValues(*table, column));
    });
    const ts::CompareFigures* f = compare_.Answer(key);
    if (!f) {
        Faint(compare_.Error().empty() ? "Working out every method..." : "Not compared: " + compare_.Error());
        return;
    }
    if (!f->ok()) {
        Faint(f->error);
        return;
    }
    const auto& methods = ts::ScaleMethods();
    const float avail = ImGui::GetContentRegionAvail().x;
    const int per_row = std::clamp(static_cast<int>(avail / 200.0f), 1, 3);
    if (ImGui::BeginTable("##compare_panes", per_row, ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_NoSavedSettings)) {
        for (size_t i = 0; i <= methods.size(); ++i) {
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(i));
            const bool original = i == 0;
            const ts::Stats& s = original ? f->original : f->methods[i - 1];
            ImGui::TextColored(t.text_dim, "%s", original ? "Original" : methods[i - 1].label);
            const float w = ImGui::GetContentRegionAvail().x;
            Histogram(s, w, 60.0f, original ? t.accent_text : t.caution);
            if (!original) {
                if (methods[i - 1].type == node.type) {
                    ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK " in use");
                } else if (ui::LinkButton("Use this method")) {
                    if (const ::gui::MLNode* step = editor_->FindStep(node.id)) {
                        editor_->RequestReplaceStep(node.id, methods[i - 1].type, ts::CarriedParameters(step->parameters));
                        message_ = std::string("Switched to ") + methods[i - 1].label +
                                   ": its node takes this step's place, links and columns (Ctrl+Z on the canvas undoes it).";
                        message_error_ = false;
                        comparing_ = false;
                    }
                }
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (BeginFigureTable("##compare_summary", 6)) {
        ImGui::TableSetupColumn("method", ImGuiTableColumnFlags_WidthStretch, 1.4f);
        for (int i = 0; i < 5; ++i) ImGui::TableSetupColumn("figure", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        HeaderCells({"method", "min", "max", "mean", "std", "skewness"});
        for (size_t i = 0; i <= methods.size(); ++i) {
            const ts::Stats& s = i == 0 ? f->original : f->methods[i - 1];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(i == 0 ? "Original" : methods[i - 1].label);
            NumberCell(ts::Value(s.min));
            NumberCell(ts::Value(s.max));
            NumberCell(ts::Value(s.mean));
            NumberCell(ts::Value(s.std));
            NumberCell(ts::Value(s.skew));
        }
        ImGui::EndTable();
    }
    ImGui::PushTextWrapPos(0.0f);
    for (const auto& m : methods) {
        ImGui::Bullet();
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s", m.advice);
    }
    ImGui::PopTextWrapPos();
}

void TransformStepView::RenderAllSettings(const std::map<std::string, std::string>& params) {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::Spacing();
    if (!ImGui::TreeNodeEx("All settings", ImGuiTreeNodeFlags_SpanAvailWidth)) return;
    if (BeginFigureTable("##all_settings", 2)) {
        ImGui::TableSetupColumn("setting", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        for (const auto& [key, value] : params) {
            if (value.empty()) continue;
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextColored(t.text_dim, "%s", key.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(value.size() > 160 ? (value.substr(0, 157) + "...").c_str() : value.c_str());
        }
        ImGui::EndTable();
    }
    Faint("Settings not shown above are changed in the node's dialog.");
    ImGui::TreePop();
}

}  // namespace cyxwiz
