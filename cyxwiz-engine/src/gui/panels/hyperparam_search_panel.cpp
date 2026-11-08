#include "hyperparam_search_panel.h"

#include "gui/icons.h"
#include "../graph_training_launcher.h"
#include "../node_editor.h"
#include "../ui_buttons.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"
#include "../../core/file_dialogs.h"
#include "../../core/hyperparam_search_presentation.h"
#include "../../core/live_graph_compile.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>

namespace cyxwiz {

namespace sp = search_presentation;

namespace {

const gui::MLNode* FindNode(const std::vector<gui::MLNode>& nodes, int id) {
    const auto it = std::find_if(nodes.begin(), nodes.end(), [id](const gui::MLNode& n) { return n.id == id; });
    return it == nodes.end() ? nullptr : &*it;
}

// The compiled graph when the compile is of the canvas as it is now and found
// no error; nullptr otherwise.
const TrainingConfiguration* CurrentCompile(const gui::NodeEditor* editor) {
    const LiveGraphCompile* live = editor ? editor->GetLiveCompile() : nullptr;
    const TrainingConfiguration* config = live ? live->Config() : nullptr;
    if (!config || !live->UpToDate()) return nullptr;
    const bool error = std::any_of(config->issues.begin(), config->issues.end(),
                                   [](const ValidationIssue& i) { return i.level == IssueLevel::Error; });
    return error ? nullptr : config;
}

void Heading(const char* text) {
    ImGui::PushStyleColor(ImGuiCol_Text, ui::CurrentTokens().text_bright);
    ImGui::TextUnformatted(text);
    ImGui::PopStyleColor();
}

void Wrapped(const std::string& text, const ImVec4& colour) {
    ImGui::PushStyleColor(ImGuiCol_Text, colour);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopTextWrapPos();
    ImGui::PopStyleColor();
}

}  // namespace

HyperparamSearchPanel::HyperparamSearchPanel() : Panel("Hyperparameter Search", false) {
    search_engine_ = std::make_unique<HyperparamSearch>();
}

bool HyperparamSearchPanel::IsSearching() const {
    return stage_ == Stage::Preparing || stage_ == Stage::Running;
}

void HyperparamSearchPanel::Render() {
    // The launch can stop after it started (the data preparation failed):
    // nothing is preparing or waiting any more and no trial began.
    if (stage_ == Stage::Preparing && !(launch_pending_ && launch_pending_()) &&
        !::gui::HasActiveGraphTrainingPreparation()) {
        NotStarted("Not started: the data preparation stopped before the trials (the Tasks panel says why).");
    }
    if (stage_ == Stage::Running && !search_engine_->IsRunning()) stage_ = Stage::Finished;

    if (!visible_) return;
    ImGui::SetNextWindowSize(ImVec2(760, 820), ImGuiCond_FirstUseEver);
    if (ImGui::Begin(ICON_FA_MAGNIFYING_GLASS_CHART " Hyperparameter Search###HyperparamSearch", &visible_)) {
        RenderHeader();
        ImGui::Spacing();
        ImGui::BeginChild("##search_body", ImVec2(0, 0), ImGuiChildFlags_None);
        if (show_trials_) RenderTrials();
        else RenderSetup();
        ImGui::EndChild();
    }
    ImGui::End();
}

void HyperparamSearchPanel::RenderHeader() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.accent_text, ICON_FA_MAGNIFYING_GLASS_CHART);
    ImGui::SameLine();
    Heading("Hyperparameter Search");
    if (node_editor_) {
        const std::string& path = node_editor_->GetCurrentFilePath();
        const std::string graph = path.empty() ? "Untitled graph" : std::filesystem::path(path).stem().string();
        ImGui::SameLine();
        ImGui::TextColored(t.text_dim, "%s \xC2\xB7 %zu nodes", graph.c_str(), node_editor_->GetNodes().size());
        ImGui::SameLine();
        if (CurrentCompile(node_editor_)) ui::Chip("Compiled", t.success);
        else ui::Chip("Not compiled", t.pending);
    }
    static const char* const kViews[] = {"Setup", "Trials"};
    ui::SameLineRight(ui::SegmentedControlWidth(kViews, 2));
    int view = show_trials_ ? 1 : 0;
    if (ui::SegmentedControl("##search_view", kViews, 2, &view)) show_trials_ = view == 1;
}

// ===== Setup =====

void HyperparamSearchPanel::RenderSetup() {
    Wrapped("Each trial trains this graph the way Train does: the same prepared data and split, a fresh model, its "
            "own scratch checkpoints. The model in use and the Training Dashboard's runs stay as they are.",
            ui::CurrentTokens().text_dim);
    ImGui::Spacing();
    ImGui::Spacing();
    RenderParameters();
    ImGui::Spacing();
    ImGui::Spacing();
    RenderHowToSearch();
    ImGui::Spacing();
    ImGui::Spacing();

    const std::string blocker = StartBlocker();
    if (ui::PrimaryButton("Start search", blocker.empty(), blocker.c_str())) StartSearch();
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(ui::CurrentTokens().text_dim, "The time per trial shows after the first one.");
    if (!not_started_.empty()) {
        ImGui::Spacing();
        ui::NoticeLine(not_started_, ui::CurrentTokens().caution);
    }
}

void HyperparamSearchPanel::RenderParameters() {
    const ui::Tokens& t = ui::CurrentTokens();
    ImGui::AlignTextToFramePadding();
    Heading("What to vary");
    const char* add_label = picker_open_ ? "Close the list" : ICON_FA_PLUS " Add from the graph";
    ui::SameLineRight(ui::ButtonWidth(add_label, ui::ButtonSize::Small));
    if (ui::SecondaryButton(add_label, !IsSearching(), "A search is running")) picker_open_ = !picker_open_;

    const std::vector<gui::MLNode> no_nodes;
    const auto& nodes = node_editor_ ? node_editor_->GetNodes() : no_nodes;
    if (config_.space.parameters.empty()) {
        ImGui::TextColored(t.text_dim, "Nothing yet: add the properties to vary from the graph.");
    } else if (ImGui::BeginTable("##search_params", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Node", ImGuiTableColumnFlags_WidthStretch, 1.1f);
        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableSetupColumn("Values", ImGuiTableColumnFlags_WidthStretch, 1.3f);
        ImGui::TableSetupColumn("Range", ImGuiTableColumnFlags_WidthStretch, 1.6f);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFrameHeight());
        ImGui::TableHeadersRow();
        int remove = -1;
        for (size_t i = 0; i < config_.space.parameters.size(); ++i) {
            SearchParameter& p = config_.space.parameters[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::AlignTextToFramePadding();
            const gui::MLNode* node = FindNode(nodes, p.node_id);
            ImGui::TextUnformatted(node ? node->name.c_str() : "(removed)");
            ImGui::TableSetColumnIndex(1);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted(sp::PropertyLabel(p.property).c_str());
            ImGui::TableSetColumnIndex(2);
            ImGui::AlignTextToFramePadding();
            ImGui::TextColored(t.text_dim, "%s", sp::KindText(p).c_str());
            ImGui::TableSetColumnIndex(3);
            ImGui::BeginDisabled(IsSearching());
            if (p.type == ParamType::Categorical) {
                std::string text = sp::RangeText(p);
                char buffer[256];
                std::snprintf(buffer, sizeof(buffer), "%s", text.c_str());
                ImGui::SetNextItemWidth(-1);
                if (ImGui::InputText("##choices", buffer, sizeof(buffer))) {
                    std::vector<std::string> options;
                    std::string item;
                    for (const char* c = buffer;; ++c) {
                        if (*c == ',' || *c == '\0') {
                            const auto a = item.find_first_not_of(' ');
                            if (a != std::string::npos) options.push_back(item.substr(a, item.find_last_not_of(' ') - a + 1));
                            item.clear();
                            if (*c == '\0') break;
                        } else {
                            item += *c;
                        }
                    }
                    p.options = std::move(options);
                }
                ui::Tooltip("The values to try, separated by commas");
            } else {
                ui::NumberFieldOptions range;
                range.width = 70.0f;
                range.format = p.type == ParamType::Discrete ? "%.0f" : "%.4g";
                if (p.type == ParamType::LogScale) range.min = 1e-12;
                ui::NumberField("##low", &p.min_value, range);
                ImGui::SameLine(0, 4);
                ImGui::TextUnformatted("-");
                ImGui::SameLine(0, 4);
                ui::NumberField("##high", &p.max_value, range);
            }
            ImGui::EndDisabled();
            ImGui::TableSetColumnIndex(4);
            if (ui::DangerButton(ICON_FA_XMARK, !IsSearching(), "A search is running")) remove = static_cast<int>(i);
            ImGui::PopID();
        }
        ImGui::EndTable();
        if (remove >= 0) config_.space.parameters.erase(config_.space.parameters.begin() + remove);
    }
    if (picker_open_) {
        ImGui::Spacing();
        RenderPicker();
    }
}

void HyperparamSearchPanel::RenderPicker() {
    const ui::Tokens& t = ui::CurrentTokens();
    ui::BeginInsetList("##search_picker");
    const TrainingConfiguration* compiled = CurrentCompile(node_editor_);
    if (!compiled) {
        ImGui::TextColored(t.text_dim, "Compile the graph first: the list comes from the compiled model.");
        ui::EndInsetList();
        return;
    }
    ImGui::TextColored(t.text_dim, "Properties a trial can change, from the compiled graph. Click one to add it.");
    const auto& nodes = node_editor_->GetNodes();
    const auto targets = sp::SearchTargets(nodes, node_editor_->GetLinks(), *compiled);
    const float name_width = ImGui::CalcTextSize("DataLoader 12").x + 16.0f;
    for (const auto& target : targets) {
        ImGui::PushID(target.node_id);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(target.node_name.c_str());
        ImGui::SameLine(name_width);
        if (!target.note.empty()) {
            ImGui::TextColored(t.text_dim, "%s", target.note.c_str());
        }
        const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;
        bool first = true;
        for (const auto& property : target.properties) {
            const bool added = std::any_of(config_.space.parameters.begin(), config_.space.parameters.end(),
                                           [&](const SearchParameter& p) {
                                               return p.node_id == target.node_id && p.property == property;
                                           });
            if (added) continue;
            const std::string label = sp::PropertyLabel(property);
            if (!first) {
                ImGui::SameLine(0, 6);
                if (ImGui::GetCursorScreenPos().x + ui::ChipButtonWidth(label.c_str()) > right) {
                    ImGui::NewLine();
                    ImGui::SetCursorPosX(name_width);
                }
            }
            first = false;
            if (ui::ChipButton(label.c_str())) {
                if (const gui::MLNode* node = FindNode(nodes, target.node_id))
                    sp::AddParameter(config_.space, sp::DefaultParameter(*node, property));
            }
        }
        if (first && target.note.empty()) ImGui::TextColored(t.text_faint, "all added");
        ImGui::PopID();
    }
    ui::EndInsetList();
}

void HyperparamSearchPanel::RenderHowToSearch() {
    const ui::Tokens& t = ui::CurrentTokens();
    Heading("How to search");
    ImGui::BeginDisabled(IsSearching());
    static const char* const kStrategies[] = {"Grid", "Random", "Bayesian"};
    int strategy = static_cast<int>(config_.strategy);
    if (ui::SegmentedControl("##strategy", kStrategies, 3, &strategy))
        config_.strategy = static_cast<SearchStrategy>(strategy);
    Wrapped(sp::StrategyText(config_.strategy), t.text_dim);
    ImGui::Spacing();

    const float label_width = 150.0f;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Trials");
    ImGui::SameLine(label_width);
    if (config_.strategy == SearchStrategy::Grid) {
        ImGui::AlignTextToFramePadding();
        const size_t combinations = config_.space.GetTotalCombinations();
        ImGui::TextColored(t.text_dim, "every combination: %zu", combinations);
    } else {
        ui::NumberField("##trials", &config_.max_trials, 1, 1000, 1, 70.0f);
    }

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Stop after");
    ImGui::SameLine(label_width);
    float hours = config_.max_time_seconds / 3600.0f;
    ui::NumberFieldOptions hours_options;
    hours_options.min = 0.05;
    hours_options.max = 168.0;
    hours_options.format = "%.2g";
    hours_options.width = 70.0f;
    if (ui::NumberField("##hours", &hours, hours_options)) config_.max_time_seconds = hours * 3600.0f;
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextColored(t.text_dim, "hours, then keep the best so far");

    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Score by");
    ImGui::SameLine(label_width);
    const auto& objectives = sp::Objectives();
    const auto current = std::find_if(objectives.begin(), objectives.end(),
                                      [this](const sp::Objective& o) { return config_.objective == o.key; });
    ImGui::SetNextItemWidth(240.0f);
    if (ImGui::BeginCombo("##objective", current != objectives.end() ? current->label : objectives.front().label)) {
        for (const auto& objective : objectives) {
            if (ImGui::Selectable(objective.label, config_.objective == objective.key)) {
                config_.objective = objective.key;
                config_.minimize = objective.minimize;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();

    ImGui::Spacing();
    ui::BeginCard("##search_defaults_note");
    Wrapped("Not listed above, the DataLoader's own settings apply to every trial: its epochs, batch size and early "
            "stopping. The trials split the data as the graph's Data Split does.",
            t.text);
    ui::EndCard();
}

std::string HyperparamSearchPanel::StartBlocker() const {
    if (IsSearching()) return "A search is running";
    if (!node_editor_ || !start_) return "No graph is open";
    if (!CurrentCompile(node_editor_)) return "Compile the graph first (it has errors or is compiling)";
    if (config_.space.parameters.empty()) return "Add at least one property to vary";
    for (const auto& p : config_.space.parameters) {
        if (!FindNode(node_editor_->GetNodes(), p.node_id)) return p.name + ": its node is no longer on the graph";
        if (p.type == ParamType::Categorical ? p.options.empty() : p.min_value >= p.max_value)
            return p.name + ": " + (p.type == ParamType::Categorical ? "give at least one value"
                                                                     : "the low end must be below the high end");
    }
    if (!config_.space.IsValid()) return "A range is not valid (a log scale starts above 0)";
    return {};
}

void HyperparamSearchPanel::StartSearch() {
    if (!StartBlocker().empty()) return;
    not_started_.clear();
    applied_note_.clear();
    stage_ = Stage::Preparing;
    show_trials_ = true;
    spdlog::info("Hyperparameter search: preparing the data as Train does");
    start_();
}

void HyperparamSearchPanel::BeginTrials(TrialSetup setup) {
    if (stage_ != Stage::Preparing) return;
    auto shared = std::make_shared<const TrialSetup>(std::move(setup));
    search_engine_->SetTrialFunction([shared](const std::vector<TrialChange>& changes, const TrialCallbacks& callbacks) {
        return RunTrial(*shared, changes, callbacks);
    });
    search_engine_->SetConfig(config_);
    if (!search_engine_->Start()) {
        NotStarted("Not started: the search could not start (the Console says why).");
        return;
    }
    stage_ = Stage::Running;
}

void HyperparamSearchPanel::NotStarted(const std::string& why) {
    if (stage_ != Stage::Preparing) return;
    stage_ = Stage::Setup;
    show_trials_ = false;
    not_started_ = why;
    spdlog::warn("Hyperparameter search: {}", why);
}

// ===== Trials =====

void HyperparamSearchPanel::RenderTrials() {
    const ui::Tokens& t = ui::CurrentTokens();
    if (stage_ == Stage::Setup) {
        ui::EmptyState(ICON_FA_MAGNIFYING_GLASS_CHART, "No search yet", "Set it up and press Start search.");
        return;
    }
    const std::vector<TrialResult> results = search_engine_->GetResults();
    const int best_id = search_engine_->GetBestTrialId();
    const bool running = stage_ == Stage::Running;

    if (stage_ == Stage::Preparing) {
        ImGui::TextColored(t.running, "Preparing");
        ImGui::SameLine();
        ImGui::TextUnformatted("the data, as Train does; the trials start when it is ready.");
    } else {
        const sp::RunStatus status =
            sp::Status(running, search_engine_->GetCurrentTrial(), search_engine_->GetTotalTrials(),
                       search_engine_->GetTrialEpoch(), search_engine_->GetTrialEpochs(), results);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(running ? t.running : t.success, running ? "Running" : "Finished");
        ImGui::SameLine();
        ImGui::TextUnformatted(status.headline.c_str());
        if (!status.timing.empty()) {
            ImGui::SameLine();
            ImGui::TextColored(t.text_dim, "%s", status.timing.c_str());
        }
        if (running) {
            ui::SameLineRight(ui::ButtonWidth("Stop", ui::ButtonSize::Small));
            if (ui::DangerButton("Stop")) search_engine_->Stop();
        }
        // A thin bar of the trials done.
        const ImVec2 at = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        ImDrawList* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(at, ImVec2(at.x + width, at.y + 6.0f), ui::ToU32(t.bg_input), 3.0f);
        if (status.fraction > 0.0f)
            draw->AddRectFilled(at, ImVec2(at.x + width * status.fraction, at.y + 6.0f), ui::ToU32(t.accent_text), 3.0f);
        ImGui::Dummy(ImVec2(width, 6.0f));
    }
    ImGui::Spacing();
    ImGui::Spacing();

    if (best_id >= 0) RenderBest(results, best_id);
    if (!results.empty()) {
        ImGui::Spacing();
        RenderScoreBars(results, best_id);
        ImGui::Spacing();
        RenderTrialTable(results, best_id);
    }
}

void HyperparamSearchPanel::RenderBest(const std::vector<TrialResult>& results, int best_id) {
    const ui::Tokens& t = ui::CurrentTokens();
    const auto best = std::find_if(results.begin(), results.end(), [best_id](const TrialResult& r) { return r.trial_id == best_id; });
    if (best == results.end()) return;
    const SearchConfig& ran = search_engine_->GetConfig();
    ui::BeginCard("##best_trial");
    ImGui::TextColored(t.accent_text, ICON_FA_STAR);
    ImGui::SameLine();
    Heading(("Best so far: trial " + std::to_string(best_id + 1)).c_str());
    ImGui::TextUnformatted(sp::ScoreText(*best).c_str());
    Wrapped(sp::ValuesText(ran, *best), t.text_dim);
    ImGui::Spacing();
    if (ui::PrimaryButton("Apply best to the graph", node_editor_ != nullptr, "No graph is open")) ApplyBest();
    ImGui::SameLine();
    if (ui::SecondaryButton("Export CSV", true, nullptr, ui::ButtonSize::Regular)) ExportCsv();
    Wrapped(applied_note_.empty() ? "Apply sets these properties on the canvas; Ctrl+Z takes them back. Train the "
                                    "graph afterwards to get the model."
                                  : applied_note_,
            t.text_dim);
    ui::EndCard();
}

void HyperparamSearchPanel::RenderScoreBars(const std::vector<TrialResult>& results, int best_id) {
    const ui::Tokens& t = ui::CurrentTokens();
    const SearchConfig& ran = search_engine_->GetConfig();
    ImGui::TextColored(t.text_dim, "%s per trial", sp::ObjectiveName(ran.objective).c_str());
    float top = 0.0f;
    for (const auto& r : results)
        if (r.completed) top = std::max(top, ObjectiveValue(ran, r));
    if (top <= 0.0f) top = 1.0f;

    const float height = 90.0f;
    const float label_height = ImGui::GetTextLineHeight();
    const ImVec2 at = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(at, ImVec2(at.x + width, at.y + height), ui::ToU32(t.bg_panel), 6.0f);
    const float gap = 6.0f;
    const size_t n = results.size();
    const float bar = std::clamp((width - 16.0f - gap * static_cast<float>(n - 1)) / static_cast<float>(n), 2.0f, 48.0f);
    const float plot_top = at.y + 6.0f;
    const float base = at.y + height - label_height - 6.0f;
    for (size_t i = 0; i < n; ++i) {
        const TrialResult& r = results[i];
        const float x = at.x + 8.0f + static_cast<float>(i) * (bar + gap);
        const float h = r.completed ? std::max(2.0f, (base - plot_top) * ObjectiveValue(ran, r) / top) : 4.0f;
        const ImVec4 colour = !r.completed ? ui::WithAlpha(t.error, 0.55f) : r.trial_id == best_id ? t.accent_text : t.accent;
        draw->AddRectFilled(ImVec2(x, base - h), ImVec2(x + bar, base), ui::ToU32(colour), 3.0f, ImDrawFlags_RoundCornersTop);
        const std::string label = std::to_string(r.trial_id + 1);
        const float label_width = ImGui::CalcTextSize(label.c_str()).x;
        if (label_width <= bar + gap)
            draw->AddText(ImVec2(x + (bar - label_width) * 0.5f, base + 2.0f), ui::ToU32(t.text_dim), label.c_str());
        if (ImGui::IsMouseHoveringRect(ImVec2(x, plot_top), ImVec2(x + bar, base + label_height)))
            ImGui::SetTooltip("Trial %d: %s", r.trial_id + 1, sp::ScoreText(r).c_str());
    }
    ImGui::Dummy(ImVec2(width, height));
}

void HyperparamSearchPanel::RenderTrialTable(const std::vector<TrialResult>& results, int best_id) {
    const ui::Tokens& t = ui::CurrentTokens();
    const SearchConfig& ran = search_engine_->GetConfig();
    const bool regression = std::any_of(results.begin(), results.end(), [](const TrialResult& r) { return r.regression; });
    const int columns = static_cast<int>(ran.space.parameters.size()) + 5;
    if (!ImGui::BeginTable("##trials", columns, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) return;
    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000").x);
    for (const auto& p : ran.space.parameters) ImGui::TableSetupColumn(p.name.c_str(), ImGuiTableColumnFlags_WidthStretch, 1.0f);
    ImGui::TableSetupColumn("val loss", ImGuiTableColumnFlags_WidthStretch, 0.8f);
    ImGui::TableSetupColumn(regression ? "MAE" : "accuracy", ImGuiTableColumnFlags_WidthStretch, 0.8f);
    ImGui::TableSetupColumn("time", ImGuiTableColumnFlags_WidthStretch, 0.6f);
    ImGui::TableSetupColumn("status", ImGuiTableColumnFlags_WidthStretch, 0.8f);
    ImGui::TableHeadersRow();

    const auto row = [&](const TrialResult* r, int trial) {
        ImGui::TableNextRow();
        if (r && r->trial_id == best_id)
            ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, ui::ToU32(ui::WithAlpha(t.accent, 0.18f)));
        ImGui::TableNextColumn();
        ImGui::Text("%d", trial + 1);
        for (const auto& p : ran.space.parameters) {
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r ? sp::ValueText(p, *r).c_str() : "...");
        }
        ImGui::TableNextColumn();
        if (r && r->completed) ImGui::Text("%.4g", r->val_loss);
        else ImGui::TextColored(t.text_faint, r ? "-" : "...");
        ImGui::TableNextColumn();
        if (r && r->completed) {
            if (regression) ImGui::Text("%.4g", r->val_mae);
            else ImGui::Text("%.1f %%", r->val_accuracy * 100.0f);
        } else {
            ImGui::TextColored(t.text_faint, r ? "-" : "...");
        }
        ImGui::TableNextColumn();
        if (r) ImGui::TextColored(t.text_dim, "%.0f s", r->duration_seconds);
        else ImGui::TextColored(t.text_faint, "...");
        ImGui::TableNextColumn();
        if (!r) {
            ImGui::TextColored(t.running, "Training");
        } else if (!r->completed) {
            ImGui::TextColored(t.caution, "Failed");
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", r->error_message.c_str());
        } else if (r->trial_id == best_id) {
            ImGui::TextColored(t.accent_text, "Best");
        } else {
            ImGui::TextColored(t.success, "Done");
        }
    };
    for (const auto& r : results) row(&r, r.trial_id);
    if (stage_ == Stage::Running && search_engine_->IsRunning()) row(nullptr, search_engine_->GetCurrentTrial());
    ImGui::EndTable();
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "Hover a failed trial for its reason. A failed trial does not stop the search; three in a row do.");
}

void HyperparamSearchPanel::ApplyBest() {
    if (!node_editor_) return;
    const TrialResult best = search_engine_->GetBestResult();
    if (best.trial_id < 0 || !best.completed) return;
    const auto changes = TrialChangesFor(search_engine_->GetConfig().space, best.parameters, best.categorical_params);
    node_editor_->ReplaceWithEditedGraph(sp::WithChanges(node_editor_->GetNodes(), changes), node_editor_->GetLinks());
    applied_note_ = "Trial " + std::to_string(best.trial_id + 1) +
                    "'s values are on the canvas (Ctrl+Z takes them back). Train the graph to get the model.";
    spdlog::info("Hyperparameter search: trial {}'s {} values applied to the graph", best.trial_id + 1, changes.size());
}

void HyperparamSearchPanel::ExportCsv() {
    const auto path = FileDialogs::SaveFile("Export Search Results", {{"CSV Files", "csv"}, {"All Files", "*"}},
                                            nullptr, "hyperparameter_search.csv");
    if (!path) return;
    std::ofstream out(*path, std::ios::binary);
    out << SearchResultsCsv(search_engine_->GetConfig(), search_engine_->GetResults(), search_engine_->GetBestTrialId());
    if (out) spdlog::info("Search results saved to {}", *path);
    else spdlog::error("Could not write {}", *path);
}

} // namespace cyxwiz
