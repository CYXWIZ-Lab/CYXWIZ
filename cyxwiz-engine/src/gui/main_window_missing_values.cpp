// The one-click fix for missing values in the "Cannot start training" report
// (TOFIX223 0.3, owner decision D4; mockup approved 2026-11-21). A thin layer
// over core/missing_values_presentation: the model words the cards and names
// the steps; this file keeps what the user did, adds and removes the steps on
// the canvas, and draws the cards.

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

void Wrapped(const ImVec4& colour, const std::string& text) {
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(colour, "%s", text.c_str());
    ImGui::PopTextWrapPos();
}

int LargestNodeId(const std::vector<MLNode>& nodes) {
    int largest = -1;
    for (const auto& n : nodes) largest = std::max(largest, n.id);
    return largest;
}

}  // namespace

void MainWindow::ShowMissingValuesReport(const std::vector<MissingValuesBlock>& blocks,
                                         const std::vector<MLNode>& nodes, const std::string& source_dataset) {
    missing_values_.clear();
    for (const auto& block : blocks) {
        MissingValuesEntry e;
        e.block = block;
        e.source_dataset = block.supplied ? block.dataset_name : source_dataset;
        for (const auto& n : nodes)
            if (n.id == block.source_node_id) e.input_name = n.name;
        if (e.input_name.empty()) e.input_name = block.dataset_name;
        missing_values_.push_back(std::move(e));
    }
    compile_result_success_ = false;
    compile_result_issues_.clear();
    ShowCompileReport(cyxwiz::compile_report::Mode::TrainBlocked, "the data has missing values");
}

void MainWindow::RefreshMissingValuesReport() {
    compile_report_input_.missing_values.clear();
    for (const auto& e : missing_values_) {
        mv::Input in;
        in.columns = e.block.columns;
        in.total_rows = e.block.total_rows;
        in.input_name = e.input_name;
        in.supplied_role = e.block.supplied;
        in.role_name = e.block.role;
        // A supplied Validation / Test is read from its own Data Input: a step
        // after it would not reach training (a question for the owner, M-6).
        in.steps_reach_input = !e.block.supplied;
        in.number_fill = missing_values_fill_;
        in.features = e.features;
        in.label = e.label;
        in.left_out = e.left_out;
        compile_report_input_.missing_values.push_back(mv::BuildCard(in));
    }
    compile_report_ = cyxwiz::compile_report::BuildReport(compile_report_input_);
}

// Each step after the step that feeds the Data Split (or after the supplied
// input), in the card's order: requested in reverse after the same node.
void MainWindow::AddMissingValuesSteps(size_t index, bool label, const std::vector<mv::Step>& steps) {
    if (!node_editor_ || index >= missing_values_.size() || steps.empty()) return;
    auto& e = missing_values_[index];
    const auto& nodes = node_editor_->GetNodes();
    int after = e.block.supplied ? e.block.source_node_id
                                 : cyxwiz::TrainingInputFeeder(nodes, node_editor_->GetLinks());
    if (after < 0) after = e.block.source_node_id;
    if (after < 0) {
        spdlog::warn("Missing values: no step feeds the Data Split; the fix was not added");
        return;
    }
    if (e.awaiting_after_id < 0) e.awaiting_after_id = LargestNodeId(nodes);
    auto& awaiting = label ? e.awaiting_label_names : e.awaiting_feature_names;
    for (auto step = steps.rbegin(); step != steps.rend(); ++step) {
        node_editor_->RequestStepAfter(after, step->type, step->name, step->params);
        awaiting.push_back(step->name);
    }
    (label ? e.label : e.features) =
        label ? mv::Done::Dropped
              : (steps.front().type == NodeType::FilterRows ? mv::Done::Dropped : mv::Done::Filled);
    spdlog::info("Missing values: adding {} step(s) after node {} for {}", steps.size(), after, e.input_name);
    RefreshMissingValuesReport();
}

void MainWindow::RemoveMissingValuesSteps(size_t index, bool label) {
    if (!node_editor_ || index >= missing_values_.size()) return;
    auto& e = missing_values_[index];
    auto& ids = label ? e.label_nodes : e.feature_nodes;
    for (const int id : ids) node_editor_->RequestRemoveStep(id);
    ids.clear();
    (label ? e.awaiting_label_names : e.awaiting_feature_names).clear();
    (label ? e.label : e.features) = mv::Done::No;
    RefreshMissingValuesReport();
}

// "Leave the column out" (M-7): its role becomes Ignore in the Data Input and
// in the project's column roles (Data Studio's Profile shows the same).
void MainWindow::LeaveColumnOut(size_t index, const std::string& column) {
    if (!node_editor_ || index >= missing_values_.size()) return;
    auto& e = missing_values_[index];
    std::string current;
    for (const auto& n : node_editor_->GetNodes())
        if (n.id == e.block.source_node_id) {
            const auto it = n.parameters.find("column_roles");
            if (it != n.parameters.end()) current = it->second;
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

// Per frame: the ids of the steps once the canvas added them (for Undo), and
// the training "Fix and train" asked for once every step is there.
void MainWindow::TrackMissingValuesSteps() {
    if (!node_editor_ || node_editor_->HasPendingStepRequests()) return;
    bool changed = false;
    for (auto& e : missing_values_) {
        if (e.awaiting_feature_names.empty() && e.awaiting_label_names.empty()) continue;
        const auto take = [&](std::vector<std::string>& names, std::vector<int>& ids, mv::Done& done) {
            if (names.empty()) return;
            for (const auto& n : node_editor_->GetNodes())
                if (n.id > e.awaiting_after_id && std::find(names.begin(), names.end(), n.name) != names.end() &&
                    std::find(ids.begin(), ids.end(), n.id) == ids.end())
                    ids.push_back(n.id);
            // The canvas refused the insert (busy): the group is open again.
            if (ids.empty()) done = mv::Done::No;
            names.clear();
            changed = true;
        };
        take(e.awaiting_feature_names, e.feature_nodes, e.features);
        take(e.awaiting_label_names, e.label_nodes, e.label);
        e.awaiting_after_id = -1;
    }
    if (changed) RefreshMissingValuesReport();
    if (train_after_missing_values_fix_) {
        train_after_missing_values_fix_ = false;
        StartTrainingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
        if (training_plot_panel_) training_plot_panel_->SetVisible(true);
    }
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
            if (events.fill) AddMissingValuesSteps(i, false, g.fill_steps);
            if (events.drop) AddMissingValuesSteps(i, label, g.drop_steps);
            if (events.undo) RemoveMissingValuesSteps(i, label);
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
// board 2): the same steps as the report, for the Train input.
void MainWindow::WireMissingValuesDashboard() {
    if (!training_plot_panel_) return;
    cyxwiz::TrainingPlotPanel::MissingValuesActions actions;
    actions.add = [this](bool label, const std::vector<mv::Step>& steps,
                         const std::vector<cyxwiz::MissingColumn>& columns) {
        if (!node_editor_) return false;
        // The run's columns, as an entry of the fix (the report's when open).
        if (missing_values_.empty() || missing_values_.front().block.role != "Train") {
            MissingValuesEntry e;
            e.block.role = "Train";
            e.block.columns = columns;
            missing_values_.insert(missing_values_.begin(), std::move(e));
        }
        const size_t before = node_editor_->GetNodes().size();
        AddMissingValuesSteps(0, label, steps);
        return node_editor_->HasPendingStepRequests() || node_editor_->GetNodes().size() != before;
    };
    actions.undo = [this](bool label) {
        if (!missing_values_.empty() && missing_values_.front().block.role == "Train") RemoveMissingValuesSteps(0, label);
    };
    actions.train_again = [this] {
        if (!node_editor_) return;
        StartTrainingFromGraph(node_editor_->GetNodes(), node_editor_->GetLinks());
    };
    actions.show_input = [this] {
        if (!node_editor_) return;
        const int feeder = cyxwiz::TrainingInputFeeder(node_editor_->GetNodes(), node_editor_->GetLinks());
        if (feeder >= 0) node_editor_->FocusNode(feeder);
    };
    training_plot_panel_->SetMissingValuesActions(std::move(actions));
}

}  // namespace gui
