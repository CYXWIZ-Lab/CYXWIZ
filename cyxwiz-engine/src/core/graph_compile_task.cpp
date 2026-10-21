#include "graph_compile_task.h"
#include <chrono>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace cyxwiz {
std::shared_ptr<AsyncTask> SubmitGraphCompileTask(
    std::vector<gui::MLNode> nodes, std::vector<gui::NodeLink> links,
    GraphCompileCompletion completion, GraphCompileFunction compile) {
    auto result = std::make_shared<TrainingConfiguration>();
    auto task = std::make_shared<LambdaTask>("Preparing Run Test",
        [nodes = std::move(nodes), links = std::move(links), result,
         compile = std::move(compile)](LambdaTask& task) {
            if (task.ShouldStop()) return;
            task.ReportProgress(0.05f, "Compiling graph on worker...");
            const auto start = std::chrono::steady_clock::now();
            *result = compile ? compile(nodes, links) : GraphCompiler{}.Compile(nodes, links);
            spdlog::info("Run Test preparation: compile_ms={:.3f} worker_thread={}",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count(),
                std::hash<std::thread::id>{}(std::this_thread::get_id()));
            if (task.ShouldStop()) return;
            if (!result->is_valid)
                throw std::runtime_error(result->error_message.empty()
                    ? "Run Test graph compilation failed" : result->error_message);
            task.ReportProgress(1.0f, "Graph prepared; waiting for UI handoff");
            task.MarkCompleted();
        });
    std::weak_ptr<AsyncTask> weak_task = task;
    task->SetCompletionCallback([result, weak_task, completion = std::move(completion)](
                                   bool success, const std::string& error) mutable {
        const auto task = weak_task.lock();
        const bool cancelled = !task || task->IsCancelRequested();
        if (completion) completion(success && !cancelled,
            cancelled ? "Run Test preparation cancelled" : error, std::move(*result));
    });
    AsyncTaskManager::Instance().Submit(task);
    return task;
}
} // namespace cyxwiz
