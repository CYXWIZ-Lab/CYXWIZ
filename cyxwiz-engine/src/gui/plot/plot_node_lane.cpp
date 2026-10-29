#include "plot_node_lane.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/parquet_backed_dataset.h"
#include "../../core/pipeline_execution_task.h"
#include "../../core/pipeline_executor.h"
#include "../../core/training_input_preparation.h"
#include "../graph_training_launcher.h"
#include "../../core/plot/plot_tree_model.h"
#include "../../core/project_manager.h"

#include <arrow/api.h>

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>

namespace cyxwiz::plot {

namespace {

// The table a plot reads (TOFIX154): a table in memory, or about this many
// rows from row groups spread over a table kept on disk; `rows` is the
// dataset's own count.
constexpr int64_t kPlotRowsFromDisk = 500000;
std::shared_ptr<arrow::Table> PlotTable(const std::string& name, int64_t& rows) {
    auto& registry = DataRegistry::Instance();
    if (const auto memory = registry.GetArrowDataset(name)) {
        auto table = memory->GetArrowTable();
        rows = table ? table->num_rows() : 0;
        return table;
    }
    if (const auto disk = registry.GetParquetBackedDataset(name)) {
        rows = disk->GetNumRows();
        return disk->ReadSpreadRowGroups(kPlotRowsFromDisk);
    }
    rows = 0;
    return nullptr;
}

std::string ClockNow() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char buf[8];
    std::strftime(buf, sizeof(buf), "%H:%M", &tm);
    return buf;
}

bool Loaded(const std::string& name) { return DataRegistry::Instance().IsArrowDataset(name); }

// A fitted tree model's file as a table of its nodes (TOFIX134 P4.7): the
// Plot opens it as a Tree. Empty with `error` set when it cannot be read.
std::shared_ptr<arrow::Table> TreeModelTable(const std::string& path, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *error = "The model file " + path + " cannot be opened.";
        return nullptr;
    }
    std::stringstream text;
    text << in.rdbuf();
    const TreeModelRows rows = ReadTreeModel(text.str());
    if (!rows.error.empty()) {
        *error = rows.error;
        return nullptr;
    }
    arrow::Int64Builder tree, trees;
    arrow::StringBuilder tree_name, node, parent, rule, cls;
    arrow::DoubleBuilder samples, value;
    bool ok = true;
    for (size_t i = 0; i < rows.node.size(); ++i) {
        ok = ok && tree.Append(rows.tree[i]).ok() && tree_name.Append(rows.tree_name[i]).ok() && node.Append(rows.node[i]).ok() &&
             parent.Append(rows.parent[i]).ok() && rule.Append(rows.rule[i]).ok() &&
             (std::isnan(rows.samples[i]) ? samples.AppendNull() : samples.Append(rows.samples[i])).ok() && cls.Append(rows.cls[i]).ok() &&
             (std::isnan(rows.value[i]) ? value.AppendNull() : value.Append(rows.value[i])).ok() && trees.Append(rows.trees).ok();
    }
    std::vector<std::shared_ptr<arrow::Array>> arrays(9);
    ok = ok && tree.Finish(&arrays[0]).ok() && tree_name.Finish(&arrays[1]).ok() && node.Finish(&arrays[2]).ok() &&
         parent.Finish(&arrays[3]).ok() && rule.Finish(&arrays[4]).ok() && samples.Finish(&arrays[5]).ok() && cls.Finish(&arrays[6]).ok() &&
         value.Finish(&arrays[7]).ok() && trees.Finish(&arrays[8]).ok();
    if (!ok) {
        *error = "The model's nodes could not be made into a table.";
        return nullptr;
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    const auto& names = TreeModelColumns();
    for (size_t c = 0; c < names.size(); ++c) fields.push_back(arrow::field(names[c], arrays[c]->type()));
    return arrow::Table::Make(arrow::schema(fields), arrays);
}

}  // namespace

const PlotNodeLane::Status& PlotNodeLane::StatusOf(int plot_id) { return entries_[plot_id].status; }

void PlotNodeLane::Forget(int plot_id) {
    Cancel(plot_id);
    if (const auto it = entries_.find(plot_id); it != entries_.end()) ReleaseModelTable(it->second.model_table, plot_id);
    entries_.erase(plot_id);
}

void PlotNodeLane::ReleaseModelTable(const std::string& name, int plot_id) {
    if (name.empty()) return;
    for (const auto& [id, entry] : entries_)
        if (id != plot_id && entry.model_table == name) return;
    DataRegistry::Instance().UnregisterTabularDataset(name);
}

PipelineExecutor& PlotNodeLane::Owned() {
    if (!owned_) owned_ = std::make_shared<PipelineExecutor>();
    return *owned_;
}

void PlotNodeLane::ReleaseNodes(const std::set<int>& node_ids) {
    if (owned_) owned_->ReleaseNodes(node_ids);
}

void PlotNodeLane::ReleaseAll() {
    if (owned_) owned_->ReleaseAll();
}

std::set<std::string> PlotNodeLane::OwnedNames() { return owned_ ? owned_->OwnedNames() : std::set<std::string>{}; }

void PlotNodeLane::Cancel(int plot_id) {
    auto it = entries_.find(plot_id);
    if (it == entries_.end() || it->second.task_id == 0) return;
    AsyncTaskManager::Instance().Cancel(it->second.task_id);
}

void PlotNodeLane::Plan(int plot_id, Entry& e, const std::vector<gui::MLNode>& nodes,
                        const std::vector<gui::NodeLink>& links) {
    e.plan = PlanNodeResult(plot_id, nodes, links, Loaded);
    Status& s = e.status;
    s.feeder_name = e.plan.feeder_name;
    s.needs_run = e.plan.state == NodeResultPlan::State::Run;
    s.run_node_count = e.plan.run_node_count;
    if (s.state == Status::State::Running) return;  // finishes first
    // Nothing to show: the rows it read go (a deleted or unwired input's
    // table stayed held here).
    const auto drop_table = [&]() {
        if (!s.table && !s.sparse && s.dataset_name.empty()) return;
        s.table.reset();
        s.sparse = false;
        s.dataset_name.clear();
        s.source_rows = 0;
        e.result_fingerprint.clear();
        ++s.data_version;
    };
    switch (e.plan.state) {
        case NodeResultPlan::State::NotConnected:
            s.state = Status::State::NotConnected;
            drop_table();
            break;
        case NodeResultPlan::State::Unavailable:
            s.state = Status::State::Unavailable;
            drop_table();
            s.reason = e.plan.reason;
            s.alternative_id = e.plan.alternative_id;
            s.alternative_name = e.plan.alternative_name;
            break;
        case NodeResultPlan::State::Loaded:
            // Cheap: read again whenever the Data Input changed.
            if (!s.table || e.result_fingerprint != e.plan.fingerprint) ReadLoaded(e);
            break;
        case NodeResultPlan::State::Run:
            if (!s.table && !s.sparse) {
                if (s.state != Status::State::Failed) s.state = Status::State::Idle;
            } else {
                s.state = e.result_fingerprint == e.plan.fingerprint ? Status::State::Ready : Status::State::OutOfDate;
            }
            break;
    }
}

void PlotNodeLane::ReadLoaded(Entry& e) {
    Status& s = e.status;
    int64_t rows = 0;
    auto table = PlotTable(e.plan.dataset_name, rows);
    if (!table) {
        s.state = Status::State::Idle;
        return;
    }
    s.table = table;
    s.source_rows = rows;
    s.dataset_name = e.plan.dataset_name;
    s.read_at = ClockNow();
    ++s.data_version;
    s.state = Status::State::Ready;
    s.error.clear();
    e.result_fingerprint = e.plan.fingerprint;
}

void PlotNodeLane::Start(int plot_id, Entry& e, const std::shared_ptr<const void>& owner) {
    auto executor = std::make_shared<PipelineExecutor>();
    auto& pm = ProjectManager::Instance();
    executor->SetArtifactRoot(pm.GetArtifactsPath());
    executor->SetExportRoot(pm.GetExportsPath());
    executor->SetIngestionCacheRoot(pm.GetIngestionCachePath());
    executor->SetProjectRoot(pm.GetProjectRoot());
    const std::string name = "Plot data: " + (e.plan.feeder_name.empty() ? std::string("node") : e.plan.feeder_name);
    auto submission = SubmitPipelineExecutionTask(name, e.plan.pipeline_json, executor, owner);
    e.task_id = submission.task_id;
    e.executor = submission.executor;
    e.run_fingerprint = e.plan.fingerprint;
    e.status.state = Status::State::Running;
    e.status.progress = 0.0f;
    e.status.progress_text = "Starting";
    e.status.error.clear();
    spdlog::info("Plot node {}: running {} node(s) above it (task {})", plot_id, e.plan.run_node_count, e.task_id);
}

void PlotNodeLane::StartSparse(int plot_id, Entry& e, const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                               const std::shared_ptr<const void>& owner) {
    (void)owner;
    // The steps above the vectorizer (the graph's own node and link types).
    std::vector<gui::MLNode> closure;
    for (const auto& n : nodes)
        if (std::find(e.plan.closure_ids.begin(), e.plan.closure_ids.end(), n.id) != e.plan.closure_ids.end()) closure.push_back(n);
    std::vector<gui::NodeLink> closure_links;
    for (const auto& l : links)
        if (std::find(e.plan.closure_ids.begin(), e.plan.closure_ids.end(), l.from_node) != e.plan.closure_ids.end() &&
            std::find(e.plan.closure_ids.begin(), e.plan.closure_ids.end(), l.to_node) != e.plan.closure_ids.end())
            closure_links.push_back(l);
    auto& pm = ProjectManager::Instance();
    auto result = std::make_shared<std::string>();
    Owned();
    const std::string name = "Sparse features: " + (e.plan.feeder_name.empty() ? std::string("node") : e.plan.feeder_name);
    auto task = std::make_shared<LambdaTask>(
        name, [plan = e.plan, closure = std::move(closure), closure_links = std::move(closure_links), result, owned = owned_,
               artifacts = pm.GetArtifactsPath(), exports = pm.GetExportsPath(), ingestion = pm.GetIngestionCachePath(),
               project = pm.GetProjectRoot(),
               // Training's prepared-data cache: the matrix is built once and read after
               // (by the Dashboard and by training) until the data or the vectorizer changes.
               cache = ::gui::GraphMaterializationCacheConfig(pm.GetProjectRoot())](LambdaTask& t) mutable {
            // The vectorizer's sparse features as training makes them: kept on
            // disk, else a run of the steps above it (TOFIX178 S0-f2); the run's
            // tables go with this lane's owner.
            for (auto& n : closure)
                if (n.id == plan.source_input_id && !plan.dataset_name.empty()) n.parameters["dataset_name"] = plan.dataset_name;
            const auto prepared = PrepareStepResult(
                closure, closure_links, plan.feeder_id, DataRegistry::Instance(), {project, artifacts, exports, ingestion},
                cache, [&t]() { return t.ShouldStop(); },
                [&t](float progress, const std::string& message) { t.ReportProgress(std::clamp(progress, 0.0f, 1.0f), message); },
                owned.get());
            if (t.ShouldStop() || prepared.cancelled) return;
            if (!prepared.ok) throw std::runtime_error(prepared.error.empty() ? std::string("The features could not be made.") : prepared.error);
            if (!DataRegistry::Instance().IsSparseFeatureDataset(prepared.dataset))
                throw std::runtime_error(plan.feeder_name + " gave no sparse features.");
            spdlog::info("Sparse features for the dashboard: {} ({})", prepared.dataset, prepared.from_disk ? "read from disk" : "built");
            *result = prepared.dataset;
            t.MarkCompleted("Sparse features ready");
        });
    e.task_id = AsyncTaskManager::Instance().Submit(task);
    e.executor.reset();
    e.sparse_result = result;
    e.run_fingerprint = e.plan.fingerprint;
    e.status.state = Status::State::Running;
    e.status.progress = 0.0f;
    e.status.progress_text = "Starting";
    e.status.error.clear();
    spdlog::info("Plot node {}: making sparse features with {} (task {})", plot_id, e.plan.feeder_name, e.task_id);
}

void PlotNodeLane::Finish(Entry& e) {
    auto task = AsyncTaskManager::Instance().GetTask(e.task_id);
    Status& s = e.status;
    if (task) {
        const TaskState state = task->GetState();
        if (state == TaskState::Pending || state == TaskState::Running) {
            s.progress = task->GetProgress();
            s.progress_text = task->GetStatusMessage();
            return;
        }
        if (state == TaskState::Failed) {
            s.state = Status::State::Failed;
            s.error = task->GetErrorMessage().empty() ? std::string("The run failed.") : task->GetErrorMessage();
        } else if (state == TaskState::Cancelled) {
            s.state = s.table || s.sparse ? Status::State::OutOfDate : Status::State::Idle;
        } else if (e.plan.sparse) {
            // Sparse features: no table; a Dashboard reads the matrix by name.
            if (e.sparse_result && !e.sparse_result->empty()) {
                s.table.reset();
                s.sparse = true;
                s.dataset_name = *e.sparse_result;
                s.read_at = ClockNow();
                ++s.data_version;
                s.state = Status::State::Ready;
                e.result_fingerprint = e.run_fingerprint;
            } else {
                s.state = Status::State::Failed;
                s.error = e.plan.feeder_name + " gave no sparse features.";
            }
        } else if (e.plan.model) {
            // A trainer's Model pin: its fitted model, as the nodes of its trees.
            const auto& models = e.executor->ModelArtifacts();
            auto it = models.find(e.plan.feeder_id);
            std::string error = e.plan.feeder_name + " gave no model to plot.";
            auto table = it != models.end() ? TreeModelTable(it->second, &error) : nullptr;
            const std::string name = "ds_plot_model_" + std::to_string(e.plan.feeder_id);
            if (table && DataRegistry::Instance().RegisterArrowTable(table, name)) {
                if (e.model_table != name) {
                    const std::string previous = e.model_table;
                    e.model_table = name;
                    ReleaseModelTable(previous, -1);  // another feeder's model, no longer shown here
                }
                s.table = table;
                s.source_rows = table->num_rows();
                s.dataset_name = name;
                s.read_at = ClockNow();
                ++s.data_version;
                s.state = Status::State::Ready;
                e.result_fingerprint = e.run_fingerprint;
            } else {
                s.state = Status::State::Failed;
                s.error = error;
            }
        } else {
            // A link from a second output (Word Embeddings' Rows) reads that
            // pin's table, not the node's main result.
            std::string result_name;
            if (e.plan.feeder_pin > 0) {
                const auto& pins = e.executor->NodePinResults();
                const auto pin = pins.find({e.plan.feeder_id, e.plan.feeder_pin});
                if (pin != pins.end()) result_name = pin->second;
            } else {
                const auto& results = e.executor->NodeResults();
                const auto it = results.find(e.plan.feeder_id);
                if (it != results.end()) result_name = it->second;
            }
            int64_t rows = 0;
            auto table = result_name.empty() ? nullptr : PlotTable(result_name, rows);
            if (table) {
                s.table = table;
                s.source_rows = rows;
                s.dataset_name = result_name;
                s.read_at = ClockNow();
                ++s.data_version;
                s.state = Status::State::Ready;
                e.result_fingerprint = e.run_fingerprint;
            } else {
                s.state = Status::State::Failed;
                s.error = e.plan.feeder_name + " gave no table to plot.";
            }
        }
    } else {
        s.state = s.table ? Status::State::OutOfDate : Status::State::Idle;
    }
    e.task_id = 0;
    // Its in-memory copies of disk inputs go now (the plot holds the rows it
    // read); its other names stay owned here until their nodes leave.
    if (e.executor) {
        e.executor->ReleaseAdmittedInputs();
        Owned().AdoptOwnership(*e.executor);
    }
    e.executor.reset();
    e.sparse_result.reset();
}

void PlotNodeLane::Poll(const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                        const std::shared_ptr<const void>& owner) {
    (void)owner;
    for (auto& [id, e] : entries_)
        if (e.task_id != 0) Finish(e);
    const double now = ImGui::GetTime();
    if (last_plan_time_ >= 0 && now - last_plan_time_ < 0.5) return;
    last_plan_time_ = now;
    std::vector<int> alive;
    for (const auto& n : nodes) {
        if (n.type != gui::NodeType::Plot && n.type != gui::NodeType::Dashboard) continue;
        alive.push_back(n.id);
        Plan(n.id, entries_[n.id], nodes, links);
    }
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (std::find(alive.begin(), alive.end(), it->first) == alive.end() && it->second.task_id == 0) {
            ReleaseModelTable(it->second.model_table, it->first);
            it = entries_.erase(it);
        }
        else
            ++it;
    }
}

void PlotNodeLane::Refresh(int plot_id, const std::vector<gui::MLNode>& nodes, const std::vector<gui::NodeLink>& links,
                           const std::shared_ptr<const void>& owner) {
    Entry& e = entries_[plot_id];
    if (e.task_id != 0) return;  // running already
    Plan(plot_id, e, nodes, links);
    if (e.plan.state == NodeResultPlan::State::Loaded) ReadLoaded(e);
    else if (e.plan.state == NodeResultPlan::State::Run) {
        if (e.plan.sparse) StartSparse(plot_id, e, nodes, links, owner);
        else Start(plot_id, e, owner);
    }
}

}  // namespace cyxwiz::plot
