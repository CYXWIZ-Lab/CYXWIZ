#pragma once
#include "graph_compiler.h"
#include "async_task_manager.h"

namespace cyxwiz {
// Worker consumes owned graph copies. Completion runs only when the UI pumps
// AsyncTaskManager. Callers must validate their window/model/graph lifetime.
using GraphCompileFunction = std::function<TrainingConfiguration(
    const std::vector<gui::MLNode>&, const std::vector<gui::NodeLink>&)>;
using GraphCompileCompletion = std::function<void(
    bool, const std::string&, TrainingConfiguration)>;
std::shared_ptr<AsyncTask> SubmitGraphCompileTask(
    std::vector<gui::MLNode> nodes, std::vector<gui::NodeLink> links,
    GraphCompileCompletion completion, GraphCompileFunction compile = {});
} // namespace cyxwiz
