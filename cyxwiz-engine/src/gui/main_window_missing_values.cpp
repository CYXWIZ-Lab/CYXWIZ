// The one-click fix for missing values in the "Cannot start training" report
// and on the Training Dashboard (TOFIX223 0.3, owner decision D4; mockup
// approved 2026-11-21). A thin layer over core/missing_values_presentation:
// the model words the cards and names the steps; this file adds and removes
// the steps on the canvas, keeps what the card says in step with the canvas,
// and draws the report's cards.

#include "main_window.h"

#include "missing_values_view.h"
#include "ui_buttons.h"
#include "ui_widgets.h"
#include "data_studio/data_studio_panel.h"
#include "node_editor.h"
#include "ui_tokens.h"
#include "panels/training_plot_panel.h"
#include "../core/column_role_store.h"
#include "../core/dataset_catalog.h"
#include "../core/training_input_preparation.h"

#include <imgui.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace gui {

namespace mv = cyxwiz::missing_values_fix;

namespace {

int LargestNodeId(const std::vector<MLNode>& nodes) {
    int largest = -1;
    for (const auto& n : nodes) largest = std::max(largest, n.id);
    return largest;
}

const MLNode* FindNode(const std::vector<MLNode>& nodes, int id) {
    for (const auto& n : nodes)
        if (n.id == id) return &n;
    return nullptr;
}

// The node is still the step the fix added: same id, type and name.
template <class Step>
bool IsOurs(const std::vector<MLNode>& nodes, const Step& step) {
    const MLNode* n = FindNode(nodes, step.id);
    return n && n->type == step.type && n->name == step.name;
}

// What a group's steps say was done: a Missing Value step fills, a Row
// Filter drops; nothing on the canvas (and nothing on its way) is open.
template <class Step>
mv::Done DoneFrom(const std::vector<Step>& on_canvas, const std::vector<Step>& awaiting) {
    const auto& steps = !awaiting.empty() ? awaiting : on_canvas;
    if (steps.empty()) return mv::Done::No;
    return steps.front().type == NodeType::FillMissingValues ? mv::Done::Filled : mv::Done::Dropped;
}

}  // namespace

void MainWindow::ShowMissingValuesReport(const std::vector<MissingValuesBlock>& blocks,
                                         const std::vector<MLNode>& nodes, const std::string& source_dataset) {
    missing_values_.clear();
    for (const auto& block : blocks) {
        MissingValuesEntry e;
        e.block = block;
        e.source_dataset = block.supplied ? block.dataset_name : source_dataset;
        if (const MLNode* n = FindNode(nodes, block.source_node_id)) e.input_name = n->name;
        if (e.input_name.empty()) e.input_name = block.dataset_name;
        if (node_editor_) e.graph_generation = node_editor_->GraphGeneration();
        missing_values_.push_back(std::move(e));
    }
    compile_result_success_ = false;
    compile_result_issues_.clear();
    ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked, "the data has missing values");
}

void MainWindow::RefreshMissingValuesReport() {
    compile_report_input_.missing_values.clear();
    for (const auto& e : missing_values_) {
        mv::Input in = mv::MakeInput(e.block.columns, e.block.total_rows, e.input_name,
                                     e.block.supplied ? e.block.role : std::string());
        in.number_fill = missing_values_fill_;
        in.features = e.features;
        in.label = e.label;
        in.left_out = e.left_out;
        compile_report_input_.missing_values.push_back(mv::BuildCard(in));
    }
    compile_report_ = cyxwiz::compile_report::BuildReport(compile_report_input_);
}

// The steps go after the step that feeds the Data Split's Train input, as one
// chain in the card's order: requested in reverse after the same node (each
// insert takes over that node's output), so a fill comes before the label's
// filter whether one button or Fix and train adds them.
void MainWindow::AddMissingValuesSteps(MissingValuesEntry& e, const std::vector<mv::Step>& steps) {
    if (!node_editor_ || steps.empty()) return;
    const auto& nodes = node_editor_->GetNodes();
    int after = cyxwiz::TrainingInputFeeder(nodes, node_editor_->GetLinks());
    if (after < 0) after = e.block.source_node_id;
    if (after < 0) {
        spdlog::warn("Missing values: no step feeds the Data Split; the fix was not added");
        return;
    }
    if (e.awaiting_feature.empty() && e.awaiting_label.empty()) e.awaiting_after_id = LargestNodeId(nodes);
    for (auto it = steps.rbegin(); it != steps.rend(); ++it)
        node_editor_->RequestStepAfter(after, it->type, it->name, it->params);
    for (const auto& step : steps) (step.label ? e.awaiting_label : e.awaiting_feature).push_back({-1, step.type, step.name});
    e.features = DoneFrom(e.feature_steps, e.awaiting_feature);
    e.label = DoneFrom(e.label_steps, e.awaiting_label);
    spdlog::info("Missing values: adding {} step(s) after node {} for {}", steps.size(), after, e.input_name);
    RefreshMissingValuesReport();
}

// Undo asks the canvas to remove the group's steps that are still ours; the
// card follows when the canvas has removed them (a refused removal, busy or
// inside a recipe, leaves the card as it is).
void MainWindow::RemoveMissingValuesSteps(MissingValuesEntry& e, bool label) {
    if (!node_editor_) return;
    if (SyncMissingValuesEntry(e)) RefreshMissingValuesReport();
    for (const auto& step : label ? e.label_steps : e.feature_steps) node_editor_->RequestRemoveStep(step.id);
}

// "Leave the column out" (M-7): its role becomes Ignore in the Data Input and
// in the project's column roles (Data Studio's Profile shows the same).
void MainWindow::LeaveColumnOut(size_t index, const std::string& column) {
    if (!node_editor_ || index >= missing_values_.size()) return;
    auto& e = missing_values_[index];
    std::string current;
    if (const MLNode* n = FindNode(node_editor_->GetNodes(), e.block.source_node_id)) {
        const auto it = n->parameters.find("column_roles");
        if (it != n->parameters.end()) current = it->second;
    }
    nlohmann::json roles = current.empty() ? nlohmann::json::object() : nlohmann::json::parse(current, nullptr, false);
    if (!roles.is_object()) roles = nlohmann::json::object();
    roles[column] = "ignore";
    if (!node_editor_->SetStepParameters(e.block.source_node_id, {{"column_roles", roles.dump()}})) {
        spdlog::warn("Missing values: the Data Input node {} did not take the role of '{}'", e.block.source_node_id,
                     column);
        return;
    }
    if (const auto entry = cyxwiz::DatasetCatalog::Instance().Resolve(e.source_dataset)) {
        auto& store = cyxwiz::ProjectColumnRoles();
        store.SetRole(cyxwiz::RoleSourceKey(entry->source_path, entry->Shown()), column, cyxwiz::ColumnRole::Ignore);
        std::string error;
        if (!store.Save(&error)) spdlog::warn("Column roles: {}", error);
    }
    e.left_out.push_back(column);
    spdlog::info("Missing values: '{}' is left out of the features of {}", column, e.input_name);
    RefreshMissingValuesReport();
}

bool MainWindow::SyncMissingValuesEntry(MissingValuesEntry& e) {
    if (!node_editor_) return false;
    const auto before = std::make_pair(e.features, e.label);
    // Another graph (open, new, clear): nothing the card said holds there.
    if (e.graph_generation != node_editor_->GraphGeneration()) {
        const bool had = !e.feature_steps.empty() || !e.label_steps.empty() || !e.awaiting_feature.empty() ||
                         !e.awaiting_label.empty() || !e.left_out.empty();
        e.feature_steps.clear();
        e.label_steps.clear();
        e.undone_feature_steps.clear();
        e.undone_label_steps.clear();
        e.awaiting_feature.clear();
        e.awaiting_label.clear();
        e.left_out.clear();
        e.graph_generation = node_editor_->GraphGeneration();
        e.features = e.label = mv::Done::No;
        return had;
    }
    if (node_editor_->HasPendingStepRequests()) return false;  // the canvas applies them next frame
    const auto& nodes = node_editor_->GetNodes();
    const auto sync = [&](std::vector<MissingValuesEntry::AddedStep>& on_canvas,
                          std::vector<MissingValuesEntry::AddedStep>& undone,
                          std::vector<MissingValuesEntry::AddedStep>& awaiting) {
        // Requested steps the canvas added: new nodes of that type and name.
        // One it refused (busy) is simply not there: the group stays open.
        for (auto& step : awaiting)
            for (const auto& n : nodes)
                if (n.id > e.awaiting_after_id && n.type == step.type && n.name == step.name &&
                    std::none_of(on_canvas.begin(), on_canvas.end(), [&](const auto& s) { return s.id == n.id; })) {
                    step.id = n.id;
                    on_canvas.push_back(step);
                    break;
                }
        awaiting.clear();
        // Removed (Undo, or Ctrl+Z on the canvas): kept to take back on a redo.
        for (auto it = on_canvas.begin(); it != on_canvas.end();) {
            if (IsOurs(nodes, *it)) {
                ++it;
            } else {
                undone.push_back(*it);
                it = on_canvas.erase(it);
            }
        }
        for (auto it = undone.begin(); it != undone.end();) {
            if (IsOurs(nodes, *it)) {
                on_canvas.push_back(*it);
                it = undone.erase(it);
            } else {
                ++it;
            }
        }
    };
    sync(e.feature_steps, e.undone_feature_steps, e.awaiting_feature);
    sync(e.label_steps, e.undone_label_steps, e.awaiting_label);
    e.awaiting_after_id = -1;
    e.features = DoneFrom(e.feature_steps, e.awaiting_feature);
    e.label = DoneFrom(e.label_steps, e.awaiting_label);
    return before != std::make_pair(e.features, e.label);
}

// Per frame: each card follows the canvas, and the training Fix and train or
// the Dashboard's Train again asked for starts once every step is there (and
// only when every open group got its step).
void MainWindow::TrackMissingValuesSteps() {
    if (!node_editor_) return;
    bool changed = false;
    for (auto& e : missing_values_) changed = SyncMissingValuesEntry(e) || changed;
    SyncMissingValuesEntry(dashboard_missing_);
    if (changed) RefreshMissingValuesReport();
    if (reopen_missing_values_report_) {
        reopen_missing_values_report_ = false;
        show_compile_result_popup_ = true;
    }
    if (!train_after_missing_values_fix_ || node_editor_->HasPendingStepRequests()) return;
    train_after_missing_values_fix_ = false;
    bool fixed = true;
    if (train_after_from_dashboard_) {
        const auto& e = dashboard_missing_;
        mv::Input in = mv::MakeInput(e.block.columns, e.block.total_rows, e.input_name, e.supplied_role);
        in.features = e.features;
        in.label = e.label;
        in.left_out = e.left_out;
        fixed = mv::BuildCard(in).resolved;
    } else {
        for (const auto& card : compile_report_.missing_values) fixed = fixed && card.resolved;
    }
    if (!fixed) {
        // A step was refused (the canvas was busy) or removed: the card says
        // what is still open instead of a run that would be refused again.
        spdlog::warn("Missing values: not every step was added; training did not start");
        if (!train_after_from_dashboard_) reopen_missing_values_report_ = true;
        train_after_from_dashboard_ = false;
        return;
    }
    train_after_from_dashboard_ = false;
    StartTrainingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
    if (training_plot_panel_) training_plot_panel_->SetVisible(true);
}

// The cards, in the Findings tab of the report (mockup board 1).
void MainWindow::RenderMissingValuesCards() {
    // A copy: an action rebuilds the report while the cards are drawn.
    const auto cards = compile_report_.missing_values;
    for (size_t i = 0; i < cards.size() && i < missing_values_.size(); ++i) {
        ImGui::PushID(missing_values_[i].block.role.c_str());
        for (const bool label : {false, true}) {
            const mv::Group& g = label ? cards[i].label : cards[i].features;
            const auto events = cyxwiz::ui::MissingValuesGroup(label ? "label" : "features", g, label, missing_values_fill_);
            if (events.fill_changed) RefreshMissingValuesReport();
            if (events.fill) AddMissingValuesSteps(missing_values_[i], g.fill_steps);
            if (events.drop) AddMissingValuesSteps(missing_values_[i], g.drop_steps);
            if (events.undo) RemoveMissingValuesSteps(missing_values_[i], label);
            if (!events.leave_out.empty()) LeaveColumnOut(i, events.leave_out);
        }
        // The input and where to look at it (mockup board 1).
        const auto& e = missing_values_[i];
        const auto& t = cyxwiz::ui::CurrentTokens();
        cyxwiz::ui::FlowRow links;
        links.Next(ImGui::CalcTextSize(e.input_name.c_str()).x);
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(t.text_faint, "%s", e.input_name.c_str());
        if (e.block.source_node_id >= 0 && node_editor_) {
            links.Next(ImGui::CalcTextSize("Show on canvas").x);
            if (cyxwiz::ui::LinkButton("Show on canvas")) {
                node_editor_->FocusNode(e.block.source_node_id);
                ImGui::CloseCurrentPopup();
            }
        }
        if (data_studio_panel_ && !e.source_dataset.empty()) {
            const char* studio = "Open Data Studio (Explore > Missing)";
            links.Next(ImGui::CalcTextSize(studio).x);
            if (cyxwiz::ui::LinkButton(studio)) {
                data_studio_panel_->SetVisible(true);
                data_studio_panel_->ShowExplore(e.source_dataset, cyxwiz::DataStudioPanel::ExploreTab::Missing);
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::Dummy(ImVec2(0.0f, t.space_xs));
        ImGui::PopID();
    }
}

// The Training Dashboard's fix after a run refused for missing values (mockup
// board 2): the same steps as the report, kept in one state (dashboard_missing_)
// that follows the canvas like the report's.
void MainWindow::WireMissingValuesDashboard() {
    if (!training_plot_panel_) return;
    cyxwiz::TrainingPlotPanel::MissingValuesActions actions;
    // The run's refusal is a new one when its columns differ: start over.
    const auto entry_for = [this](const mv::Input& in) -> MissingValuesEntry& {
        auto& e = dashboard_missing_;
        const auto same = [](const cyxwiz::MissingColumn& a, const cyxwiz::MissingColumn& b) {
            return a.name == b.name && a.rows == b.rows && a.label == b.label;
        };
        if (!std::equal(e.block.columns.begin(), e.block.columns.end(), in.columns.begin(), in.columns.end(), same) ||
            e.supplied_role != in.role_name) {
            e = MissingValuesEntry{};
            e.block.role = in.supplied_role ? in.role_name : "Train";
            e.block.columns = in.columns;
            e.block.total_rows = in.total_rows;
            e.input_name = in.input_name;
            e.supplied_role = in.supplied_role ? in.role_name : "";
            if (node_editor_) e.graph_generation = node_editor_->GraphGeneration();
        }
        return e;
    };
    actions.state = [entry_for](mv::Input& in) {
        const auto& e = entry_for(in);
        in.features = e.features;
        in.label = e.label;
        in.left_out = e.left_out;
    };
    actions.add = [this, entry_for](bool, const std::vector<mv::Step>& steps, const mv::Input& in) {
        if (node_editor_ && in.steps_reach_input) AddMissingValuesSteps(entry_for(in), steps);
    };
    actions.undo = [this](bool label) { RemoveMissingValuesSteps(dashboard_missing_, label); };
    // Not from inside the panel's drawing (it holds its data lock): the run
    // starts in TrackMissingValuesSteps, once the steps are on the canvas.
    actions.train_again = [this] {
        train_after_missing_values_fix_ = true;
        train_after_from_dashboard_ = true;
    };
    actions.show_input = [this] {
        if (!node_editor_) return;
        const int feeder = cyxwiz::TrainingInputFeeder(node_editor_->GetNodes(), node_editor_->GetLinks());
        if (feeder >= 0) node_editor_->FocusNode(feeder);
    };
    training_plot_panel_->SetMissingValuesActions(std::move(actions));
}

}  // namespace gui
