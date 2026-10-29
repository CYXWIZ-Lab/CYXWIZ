// Data Studio: the graph read into the shell's model, and the routes in
// (ShowQuery, ShowProfile) and out (Show on canvas). Drawing is in
// data_studio_panel_render.cpp.

#include "data_studio_panel.h"
#include "../../core/node_executors/hypothesis_test_operator.h"
#include "../../core/stats_presentation.h"
#include "../separate_windows.h"

#include "../../core/data_studio_publish.h"
#include "../../core/explore_presentation.h"
#include "../../core/guided_sql.h"
#include "../../core/dataset_catalog.h"
#include "../../core/sql_editing.h"
#include "../../core/dataset_picker_presentation.h"
#include "../../core/image_index_service.h"
#include "../node_editor.h"

#include <algorithm>
#include <filesystem>
#include <set>

namespace cyxwiz {

DataStudioPanel::DataStudioPanel() {
    query_editor_ = std::make_unique<QueryEditor>();
    profile_ = std::make_unique<ProfileView>();
    profile_->on_missing = [this](const std::string& column) { ShowExplore(active_dataset_, ExploreTab::Missing, column); };
    profile_->on_outliers = [this](const std::string& column) { ShowExplore(active_dataset_, ExploreTab::Outliers, column); };
    profile_->on_distribution = [this](const std::string& column) { ShowExplore(active_dataset_, ExploreTab::Distribution, column); };
    profile_->on_correlation = [this] { ShowExplore(active_dataset_, ExploreTab::Correlation); };
    missing_ = std::make_unique<MissingView>();
    missing_->on_open_query = [this](const std::string& sql) { ShowQuery(active_dataset_, sql); };
    missing_->on_open_profile = [this](const std::string& column) { ShowExplore(active_dataset_, ExploreTab::Profile, column); };
    missing_->drop_blocker = [this]() -> std::string {
        if (!node_editor_) return "No graph is open";
        if (!NodeOfDataset(active_dataset_)) return "This data is not a table of this graph's nodes; pick a Data Input or a node result in the list";
        return "";
    };
    outliers_ = std::make_unique<OutliersView>();
    correlation_ = std::make_unique<CorrelationView>();
    distribution_ = std::make_unique<DistributionView>();
    tests_ = std::make_unique<TestsView>();
    regression_ = std::make_unique<RegressionView>();
    // Explore's Tests and Regression keep what they found as nodes reading the data in use (TOFIX129 D3).
    tests_->keep_as_node = [this](const stats::TestRequest& r) -> std::string {
        const studio::GraphNode* from = NodeOfDataset(active_dataset_);
        if (!from || !node_editor_) return "This data is not a table of this graph's nodes; pick a Data Input or a node result in the list.";
        const std::string name = stats_view::TestName(r) + " (" + (r.values.empty() ? r.groups : r.values) + ")";
        node_editor_->RequestBranchAfter(from->id, ::gui::NodeType::HypothesisTest, HypothesisTestOperator::ParametersFromRequest(r), name);
        return "Added the Hypothesis Test node \"" + name + "\" reading \"" + from->name + "\" (Ctrl+Z on the canvas undoes it).";
    };
    regression_->add_to_graph = [this](const stats::RegressionRequest& r) -> std::string {
        const studio::GraphNode* from = NodeOfDataset(active_dataset_);
        if (!from || !node_editor_) return "This data is not a table of this graph's nodes; pick a Data Input or a node result in the list.";
        const std::string name = "Predict " + r.response;
        if (r.polynomial) {
            node_editor_->RequestBranchAfter(from->id, ::gui::NodeType::PolynomialRegressionNode,
                                             {{"feature_col", r.predictors.front()}, {"target_col", r.response}, {"degree", std::to_string(r.degree)}},
                                             name);
        } else {
            std::string features;
            for (const auto& p : r.predictors) features += (features.empty() ? "" : ",") + p;
            node_editor_->RequestBranchAfter(from->id, ::gui::NodeType::LinearRegressionNode,
                                             {{"feature_cols", features}, {"target_col", r.response}, {"fit_intercept", "true"}}, name);
        }
        return std::string("Added the ") + (r.polynomial ? "Polynomial" : "Linear") + " Regression node \"" + name + "\" reading \"" +
               from->name + "\" (Ctrl+Z on the canvas undoes it).";
    };
    outliers_->on_open_query = [this](const std::string& sql) { ShowQuery(active_dataset_, sql); };
    missing_->on_drop_rows = [this](const std::string& column, const std::vector<std::string>& texts) {
        return AddKeepRowsStep(column, texts);
    };
    visualizer_ = std::make_unique<Visualizer>();
    prepare_ = std::make_unique<PrepareView>();
    prepare_->raise_canvas = [this] {
        if (open_node_editor_callback_) open_node_editor_callback_();
    };
    prepare_->show_on_canvas = [this](const std::vector<int>& ids) { ShowOnCanvas(ids); };
    validate_ = std::make_unique<ValidateView>();
    validate_->show_on_canvas = [this](const std::vector<int>& ids) { ShowOnCanvas(ids); };
    results_ = std::make_unique<ResultsView>();
    // Keep as node: the kept file's Data Input and Test Evaluation node,
    // added when CyxWiz Studio draws (brought forward first).
    results_->on_keep = [this](const std::string& path, const std::string& name) {
        if (!node_editor_) return;
        if (open_node_editor_callback_) open_node_editor_callback_();
        node_editor_->AddKeptTest(path, name);
    };
    publish_ = std::make_unique<PublishView>();
    publish_->show_on_canvas = [this](const std::vector<int>& ids) { ShowOnCanvas(ids); };
    readiness_ = std::make_unique<ReadinessView>();
    readiness_->on_profile = [this](const std::string& dataset) { ShowProfile(dataset); };
    readiness_->on_show_node = [this](int id) { ShowOnCanvas({id}); };
    visualizer_->keep_blocker = [this](const std::string& dataset) -> std::string {
        if (!node_editor_) return "No graph is open";
        if (!NodeOfDataset(dataset)) return "This data is not a table of this graph's nodes; pick a Data Input or a node result in the rail";
        return "";
    };
    visualizer_->on_keep = [this](const std::string& dataset, const std::string& title, const std::string& spec_json) -> std::string {
        const studio::GraphNode* from = NodeOfDataset(dataset);
        if (!from || !node_editor_) return "The data's node is no longer in the graph.";
        const std::string name = "Plot: " + title;
        node_editor_->RequestBranchAfter(from->id, ::gui::NodeType::Plot, {{"plot_spec", spec_json}}, name);
        return "Added the Plot node \"" + name + "\" reading \"" + from->name + "\" (Ctrl+Z on the canvas undoes it).";
    };
    query_editor_->step_blocker = [this](const std::vector<std::string>& inputs) { return StepBlocker(inputs); };
    query_editor_->on_turn_into_step = [this](const std::string& sql, const std::vector<std::string>& inputs) {
        return TurnIntoStep(sql, inputs);
    };
}

void DataStudioPanel::ShowQuery(const std::string& dataset_name, const std::string& sql) {
    SetActiveDataset(dataset_name);
    visible_ = true;
    ::gui::BringToFront("Data Studio");
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
    ::gui::BringToFront("Data Studio");
    job_ = studio::Job::Explore;
    explore_tab_ = ExploreTab::Profile;
}

void DataStudioPanel::ShowResults(ResultsView::View view) {
    visible_ = true;
    ::gui::BringToFront("Data Studio");
    job_ = studio::Job::Results;
    if (results_) {
        results_->Activate();
        results_->Show(view);
    }
}

void DataStudioPanel::ShowExplore(const std::string& dataset_name, ExploreTab tab, const std::string& column) {
    SetActiveDataset(dataset_name);
    visible_ = true;
    ::gui::BringToFront("Data Studio");
    job_ = studio::Job::Explore;
    explore_tab_ = tab;
    if (column.empty()) return;
    if (tab == ExploreTab::Profile && profile_) profile_->SelectColumn(column);
    if (tab == ExploreTab::Missing && missing_) missing_->Select(column);
    if (tab == ExploreTab::Outliers && outliers_) outliers_->Select(column);
    if (tab == ExploreTab::Distribution && distribution_) distribution_->Select(column);
    if (tab == ExploreTab::Tests && tests_) tests_->Select(column);
    if (tab == ExploreTab::Regression && regression_) regression_->Select(column);
}

ExploreContext DataStudioPanel::MakeExploreContext() const {
    ExploreContext c;
    c.dataset = active_dataset_;
    if (!profile_ || active_dataset_.empty()) return c;
    c.shown = profile_->Shown();
    c.profile = profile_->Current();
    c.missing_texts = profile_->MissingTexts();
    c.profiling = profile_->IsRunning();
    c.error = profile_->Error();
    return c;
}

std::string DataStudioPanel::AddKeepRowsStep(const std::string& column, const std::vector<std::string>& texts) {
    const studio::GraphNode* from = NodeOfDataset(active_dataset_);
    if (!from || !node_editor_) return "The data's node is no longer in the graph.";
    guided::GuidedStep step = explore_view::KeepRowsWithValue(column, texts);
    step.saved_context.inputs = {"input"};
    if (const auto profile = profile_ ? profile_->Current() : nullptr) {
        for (const auto& c : profile->columns) step.saved_context.columns["input"].push_back(c.facts.name);
    }
    const guided::SavedStep saved = guided::SaveStep(step);
    if (!saved.generated.ok) {
        std::string why;
        for (const auto& e : saved.generated.errors)
            if (!e.empty()) why += (why.empty() ? "" : "; ") + e;
        return "Could not write the step: " + why;
    }
    const std::string name = "Keep rows with " + column;
    node_editor_->RequestStepAfter(from->id, ::gui::NodeType::SQLQuery, name,
                                   {{guided::kGuidedOpsParameter, guided::ToJson(saved.step)}, {"query", saved.generated.sql}});
    return "Added the guided step \"" + name + "\" after \"" + from->name + "\"; Prepare lists it (Ctrl+Z on the canvas undoes it).";
}

void DataStudioPanel::AddTransformStep(::gui::NodeType type, std::map<std::string, std::string> parameters, bool compare) {
    visible_ = true;
    ::gui::BringToFront("Data Studio");
    job_ = studio::Job::Prepare;
    if (!prepare_ || !node_editor_) return;
    const studio::GraphNode* from = NodeOfDataset(active_dataset_);
    if (!from) {
        std::vector<const studio::GraphNode*> inputs;
        for (const auto& n : graph_nodes_)
            if (n.type == ::gui::NodeType::DataInput) inputs.push_back(&n);
        if (inputs.empty()) {
            prepare_->Notice("Nothing to transform yet: bring in data first (a Data Input node).");
            return;
        }
        if (inputs.size() > 1) {
            prepare_->Notice("Pick the data to transform in the rail (a Data Input or a node's result), then choose "
                             "Data > Transform again, or use Add a step.");
            return;
        }
        from = inputs.front();
    }
    prepare_->AddAfter(from->id, type, std::move(parameters), compare);
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
                // Validate shows a check's (or validator's) rule from its settings,
                // Publish an output's file.
                if (type == ::gui::NodeType::RowCountCheck || type == ::gui::NodeType::DataValidator ||
                    studio::IsOutputNode(type))
                    n.parameters = *parameters;
                const auto it = parameters->find("file_path");
                if (it != parameters->end()) n.file_path = it->second;
                if (type == ::gui::NodeType::DataInput) {
                    if (const auto f = parameters->find("folder_path"); f != parameters->end()) n.folder_path = f->second;
                    if (const auto c = parameters->find("labels_csv"); c != parameters->end()) n.labels_csv = c->second;
                }
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
                n.written_path = f->second.written_path;
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
                const ::gui::MLNode* stored = node_editor_->FindStep(step.id);
                add(step.id, step.type, step.name, stored ? &stored->parameters : nullptr);
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
        d.images = entry.modality == DatasetModality::Image && !d.tabular;
        if (d.images) {
            const ImageIndexStatus index = GetImageIndexStatus(d.name);
            if (index.state == ImageIndexStatus::State::Building)
                in.image_index_progress[d.name] = studio::Thousands(static_cast<int64_t>(index.done)) + " of " + studio::Thousands(static_cast<int64_t>(index.total));
        }
        in.datasets.push_back(std::move(d));
    }
    // What feeds each check, in the graph as it runs (recipes lowered): read
    // only while Validate shows, since it builds the pipeline document.
    std::string pipeline_error;
    if (node_editor_ && (job_ == studio::Job::Validate || job_ == studio::Job::Publish)) {
        const auto links = node_editor_->GetDataPipelineLinks(&pipeline_error);
        for (auto& n : in.nodes) {
            if (n.type != ::gui::NodeType::RowCountCheck && n.type != ::gui::NodeType::DataValidator && !studio::IsOutputNode(n.type))
                continue;
            n.input_from.clear();
            int pin = -1;
            for (const auto& link : links)
                if (link.to == n.id && (pin < 0 || link.pin < pin)) {
                    pin = link.pin;
                    n.input_from = {link.from};
                }
        }
    }
    if (results_) {
        in.preview_runs = results_->PreviewRunCount();
        in.test_rows = results_->TestRowCount();
        results_->graph_name = in.graph_name;
        results_->graph_open = node_editor_ != nullptr;
    }
    graph_nodes_ = in.nodes;
    view_ = studio::Build(in);
    if (prepare_) prepare_->SetGraph(node_editor_, graph_nodes_, view_.jobs[static_cast<size_t>(studio::Job::Prepare)].node_ids);
    if (validate_) {
        std::vector<int> recipes;
        for (const auto& n : graph_nodes_)
            if (n.recipe) recipes.push_back(n.id);
        validate_->SetGraph(node_editor_, graph_nodes_, recipes, active_dataset_, pipeline_error);
    }
    if (publish_) publish_->SetGraph(node_editor_, graph_nodes_, active_dataset_);
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

const studio::GraphNode* DataStudioPanel::NodeOfDataset(const std::string& dataset) const {
    if (const studio::GraphNode* n = NodeMaking(dataset)) return n;
    for (const auto& n : graph_nodes_)
        if (n.type == ::gui::NodeType::DataInput && !dataset.empty() && n.dataset_name == dataset) return &n;
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
