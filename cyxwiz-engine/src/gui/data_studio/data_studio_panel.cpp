// Data Studio: the graph read into the shell's model, and the routes in
// (ShowQuery, ShowProfile) and out (Show on canvas). Drawing is in
// data_studio_panel_render.cpp.

#include "data_studio_panel.h"

#include "../../core/dataset_catalog.h"
#include "../../core/sql_editing.h"
#include "../../core/dataset_picker_presentation.h"
#include "../node_editor.h"

#include <algorithm>
#include <filesystem>
#include <set>

namespace cyxwiz {

DataStudioPanel::DataStudioPanel() {
    query_editor_ = std::make_unique<QueryEditor>();
    profile_ = std::make_unique<ProfileView>();
    visualizer_ = std::make_unique<Visualizer>();
    readiness_ = std::make_unique<ReadinessView>();
    readiness_->on_profile = [this](const std::string& dataset) { ShowProfile(dataset); };
    readiness_->on_show_node = [this](int id) { ShowOnCanvas({id}); };
    query_editor_->step_blocker = [this](const std::vector<std::string>& inputs) { return StepBlocker(inputs); };
    query_editor_->on_turn_into_step = [this](const std::string& sql, const std::vector<std::string>& inputs) {
        return TurnIntoStep(sql, inputs);
    };
}

void DataStudioPanel::ShowQuery(const std::string& dataset_name, const std::string& sql) {
    SetActiveDataset(dataset_name);
    visible_ = true;
    focus_next_ = true;
    job_ = studio::Job::Explore;
    explore_tab_ = ExploreTab::Query;
    if (query_editor_) {
        query_editor_->SetQuery(sql);
        query_editor_->ExecuteQuery();
    }
}

void DataStudioPanel::ShowProfile(const std::string& dataset_name) {
    SetActiveDataset(dataset_name);
    visible_ = true;
    focus_next_ = true;
    job_ = studio::Job::Explore;
    explore_tab_ = ExploreTab::Profile;
}

void DataStudioPanel::SetActiveDataset(const std::string& dataset_name) {
    active_dataset_ = dataset_name;
    if (query_editor_) query_editor_->SetActiveDataset(dataset_name);
    if (profile_) profile_->SetActiveDataset(dataset_name);
    if (visualizer_) visualizer_->SetActiveDataset(dataset_name);
}

void DataStudioPanel::RefreshView() {
    studio::Input in;
    canvas_node_.clear();
    if (node_editor_) {
        const std::string& path = node_editor_->GetCurrentFilePath();
        if (!path.empty()) in.graph_name = std::filesystem::path(path).stem().string();
        const auto& facts = node_editor_->GetNodeRunFacts();
        std::set<int> seen;
        const auto add = [&](int id, ::gui::NodeType type, const std::string& name,
                             const std::map<std::string, std::string>* parameters) {
            if (!seen.insert(id).second) return false;
            studio::GraphNode n;
            n.id = id;
            n.type = type;
            n.name = name;
            if (parameters) {
                const auto it = parameters->find("file_path");
                if (it != parameters->end()) n.file_path = it->second;
                const auto registry = parameters->find("dataset_name");
                if (type == ::gui::NodeType::DataInput && registry != parameters->end()) n.dataset_name = registry->second;
                const auto mode = parameters->find("balance_mode");
                if (type == ::gui::NodeType::DataLoader && mode != parameters->end()) n.balance = mode->second;
            }
            if (const auto f = facts.find(id); f != facts.end()) {
                n.ran = true;
                n.rows = f->second.rows;
                n.columns = f->second.columns;
                n.seconds = f->second.seconds;
                n.outcome = f->second.outcome;
                n.error = f->second.error;
                n.output_dataset = f->second.output_dataset;
                n.finished_at = f->second.finished_at;
            }
            in.nodes.push_back(std::move(n));
            return true;
        };
        const auto& links = node_editor_->GetLinks();
        for (const auto& node : node_editor_->GetNodes()) {
            if (!add(node.id, node.type, node.name, &node.parameters)) continue;
            if (node.type == ::gui::NodeType::DataSplit) {
                // The node linked to each input pin (ML readiness: train, validation, test).
                auto& from = in.nodes.back().input_from;
                for (const auto& pin : node.inputs) {
                    int source = -1;
                    for (const auto& link : links)
                        if (link.to_node == node.id && link.to_pin == pin.id) source = link.from_node;
                    from.push_back(source);
                }
            }
            if (node.type != ::gui::NodeType::Subgraph) continue;
            // A recipe's steps count as the graph's nodes too (checks inside
            // it are Validate's); a collapsed recipe stands for them on the canvas.
            const auto summary = node_editor_->GetSubgraphSummary(node.id);
            const size_t at = in.nodes.size() - 1;
            in.nodes[at].recipe = summary.recipe;
            in.nodes[at].recipe_steps = static_cast<int>(summary.steps.size());
            for (const auto& step : summary.steps) {
                if (!summary.expanded) canvas_node_[step.id] = node.id;
                add(step.id, step.type, step.name, nullptr);
            }
        }
    }
    for (const auto& entry : DatasetCatalog::Instance().List()) {
        studio::Dataset d;
        d.name = entry.name;
        d.label = entry.label;
        d.rows = entry.rows;
        d.columns = entry.columns;
        d.tabular = IsTabularDataset(entry);
        in.datasets.push_back(std::move(d));
    }
    graph_nodes_ = in.nodes;
    view_ = studio::Build(in);
    if (readiness_) {
        readiness_->SetSplitSets(view_.split_sets, view_.balance);
        // The tab's caption says what the last check found ("5 · no leakage").
        const std::string caption = readiness_->Caption();
        if (!caption.empty()) view_.jobs[static_cast<size_t>(studio::Job::Ready)].state = caption;
    }
}

const studio::GraphNode* DataStudioPanel::NodeMaking(const std::string& table) const {
    const std::string registry = DatasetCatalog::Instance().NameFor(table);
    for (const auto& n : graph_nodes_)
        if (!n.output_dataset.empty() && (n.output_dataset == registry || n.output_dataset == table)) return &n;
    return nullptr;
}

std::string DataStudioPanel::StepBlocker(const std::vector<std::string>& inputs) const {
    if (!node_editor_) return "No graph is open";
    if (inputs.empty()) return "The query reads no table of the graph";
    if (inputs.size() > 1) return "It reads " + std::to_string(inputs.size()) + " tables; an SQL step reads one input";
    const studio::GraphNode* from = NodeMaking(inputs.front());
    if (!from) return "\"" + inputs.front() + "\" is not a result of this graph's nodes (run the graph first)";
    if (from->type != ::gui::NodeType::SQLQuery)
        return "\"" + inputs.front() + "\" comes from \"" + from->name + "\"; a query step can follow an SQL step";
    return "";
}

std::string DataStudioPanel::TurnIntoStep(const std::string& sql, const std::vector<std::string>& inputs) {
    if (!StepBlocker(inputs).empty()) return "";
    const studio::GraphNode* from = NodeMaking(inputs.front());
    // A collapsed recipe opens first so the step joins it on the canvas.
    if (const auto it = canvas_node_.find(from->id); it != canvas_node_.end())
        node_editor_->RequestSubgraphAction(it->second, ::gui::NodeEditor::SubgraphRequest::Expand);
    node_editor_->RequestSqlStepAfter(from->id, from->name + " (query)", sqledit::StepQueryFromScratch(sql, inputs.front()));
    return "Added the SQL step \"" + from->name + " (query)\" after \"" + from->name +
           "\" (it takes over what read that step). It shows on the canvas (Ctrl+Z there undoes it); run the graph to fill it.";
}

void DataStudioPanel::ShowOnCanvas(const std::vector<int>& node_ids) {
    if (!node_editor_ || node_ids.empty()) return;
    std::vector<int> ids;
    for (int id : node_ids) {
        const auto it = canvas_node_.find(id);
        const int shown = it == canvas_node_.end() ? id : it->second;
        if (std::find(ids.begin(), ids.end(), shown) == ids.end()) ids.push_back(shown);
    }
    if (open_node_editor_callback_) open_node_editor_callback_();
    node_editor_->ShowNodes(std::move(ids));
}

}  // namespace cyxwiz
