#include "plot_node_lane.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/pipeline_execution_task.h"
#include "../../core/pipeline_executor.h"
#include "../../core/project_manager.h"

#include <imgui.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <ctime>

namespace cyxwiz::plot {

namespace {

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

}  // namespace

const PlotNodeLane::Status& PlotNodeLane::StatusOf(int plot_id) { return entries_[plot_id].status; }

void PlotNodeLane::Forget(int plot_id) {
    Cancel(plot_id);
    entries_.erase(plot_id);
}

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
    switch (e.plan.state) {
        case NodeResultPlan::State::NotConnected:
            s.state = Status::State::NotConnected;
            break;
        case NodeResultPlan::State::Unavailable:
            s.state = Status::State::Unavailable;
            s.reason = e.plan.reason;
            s.alternative_id = e.plan.alternative_id;
            s.alternative_name = e.plan.alternative_name;
            break;
        case NodeResultPlan::State::Loaded:
            // Cheap: read again whenever the Data Input changed.
            if (!s.table || e.result_fingerprint != e.plan.fingerprint) ReadLoaded(e);
            break;
        case NodeResultPlan::State::Run:
            if (!s.table) {
                if (s.state != Status::State::Failed) s.state = Status::State::Idle;
            } else {
                s.state = e.result_fingerprint == e.plan.fingerprint ? Status::State::Ready : Status::State::OutOfDate;
            }
            break;
    }
}

void PlotNodeLane::ReadLoaded(Entry& e) {
    auto dataset = DataRegistry::Instance().GetArrowDataset(e.plan.dataset_name);
    Status& s = e.status;
    if (!dataset || !dataset->GetArrowTable()) {
        s.state = Status::State::Idle;
        return;
    }
    s.table = dataset->GetArrowTable();
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
            s.state = s.table ? Status::State::OutOfDate : Status::State::Idle;
        } else {
            const auto& results = e.executor->NodeResults();
            auto it = results.find(e.plan.feeder_id);
            auto dataset = it != results.end() ? DataRegistry::Instance().GetArrowDataset(it->second) : nullptr;
            if (dataset && dataset->GetArrowTable()) {
                s.table = dataset->GetArrowTable();
                s.dataset_name = it->second;
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
    e.executor.reset();
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
        if (std::find(alive.begin(), alive.end(), it->first) == alive.end() && it->second.task_id == 0)
            it = entries_.erase(it);
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
    else if (e.plan.state == NodeResultPlan::State::Run) Start(plot_id, e, owner);
}

}  // namespace cyxwiz::plot
