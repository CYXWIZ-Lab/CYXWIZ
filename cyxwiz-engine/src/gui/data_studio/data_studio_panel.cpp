// Data Studio: the graph read into the shell's model, and the routes in
// (ShowQuery, ShowProfile) and out (Show on canvas). Drawing is in
// data_studio_panel_render.cpp.

#include "data_studio_panel.h"

#include "../../core/dataset_catalog.h"
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
        for (const auto& node : node_editor_->GetNodes()) {
            if (!add(node.id, node.type, node.name, &node.parameters) || node.type != ::gui::NodeType::Subgraph) continue;
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
    view_ = studio::Build(in);
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
