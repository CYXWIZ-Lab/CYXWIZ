#include "variables_service.h"

#include "scripting_engine.h"
#include "../core/async_task_manager.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace scripting {

namespace {
// The variable a path names, for the Task viewer: its last key (`df`, `x[3]`
// shows as `3`), or "a value".
std::string PathLabel(const std::string& path_json) {
    const auto path = nlohmann::json::parse(path_json.empty() ? "[]" : path_json, nullptr, false);
    if (path.is_array() && !path.empty() && path.back().is_array() && path.back().size() >= 2) {
        const auto& key = path.back()[1];
        return key.is_string() ? key.get<std::string>() : key.dump();
    }
    return "a value";
}
}  // namespace

const char* VariablesFunctionName(VariablesService::Kind kind) {
    switch (kind) {
        case VariablesService::Kind::List: return "list_variables";
        case VariablesService::Kind::Children: return "children";
        case VariablesService::Kind::Value: return "value_text";
        case VariablesService::Kind::Delete: return "delete";
        case VariablesService::Kind::SaveCsv: return "save_csv";
        case VariablesService::Kind::Table: return "table";
        case VariablesService::Kind::Evaluate: return "evaluate";
        case VariablesService::Kind::Console: return "console";
    }
    return "list_variables";
}

VariablesService::VariablesService(ScriptingEngine* engine) : engine_(engine) {
    worker_ = std::thread([this] { Run(); });
}

VariablesService::~VariablesService() { Stop(); }

void VariablesService::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        queue_.clear();
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::uint64_t VariablesService::Submit(Request request) {
    std::uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return 0;
        id = request.id = next_id_++;
        if (request.kind == Kind::List) {
            // Only the newest list of a scope matters.
            queue_.erase(std::remove_if(queue_.begin(), queue_.end(),
                                        [&](const Request& r) { return r.kind == Kind::List && r.scope == request.scope; }),
                         queue_.end());
        }
        queue_.push_back(std::move(request));
    }
    wake_.notify_one();
    return id;
}

std::vector<VariablesService::Result> VariablesService::Poll() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Result> out;
    out.swap(done_);
    return out;
}

void VariablesService::Run() {
    while (true) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            request = std::move(queue_.front());
            queue_.pop_front();
        }
        Result result;
        result.kind = request.kind;
        result.id = request.id;
        result.scope = request.scope;
        if (request.kind == Kind::Table) {
            ScriptingEngine::VariableTable table;
            // A table read can take a while: the Task viewer lists it (it runs
            // here, on the Python worker; a Python call cannot be stopped).
            cyxwiz::AsyncTaskManager::Instance().RunHere(
                "Read " + PathLabel(request.path_json) + " from Python",
                [&](cyxwiz::LambdaTask& task) {
                    engine_->ReadVariableTable(request.scope, request.path_json, request.max_rows, request.index,
                                               &table, &result.busy);
                    if (result.busy) task.MarkCancelled("A run was active; read again when it ends");
                    else if (!table.error.empty()) task.MarkFailed(table.error);
                    else task.MarkCompleted(std::to_string(table.shown) + " rows");
                },
                false);
            result.table = std::move(table.table);
            result.table_kind = std::move(table.kind);
            result.shape = std::move(table.shape);
            result.rows = table.rows;
            result.shown = table.shown;
            result.dtypes = std::move(table.dtypes);
            result.slice = std::move(table.slice);
            result.error = std::move(table.error);
        } else if (request.kind == Kind::Evaluate || request.kind == Kind::Console) {
            nlohmann::json args = {{request.kind == Kind::Evaluate ? "expr" : "source", request.expression},
                                   {"index", request.frame}};
            result.json = engine_->CallDebugTool(VariablesFunctionName(request.kind), args.dump(), &result.busy);
        } else {
            nlohmann::json args = nlohmann::json::object();
            if (request.kind == Kind::List) args["show_all"] = request.show_all;
            else {
                auto path = nlohmann::json::parse(request.path_json.empty() ? "[]" : request.path_json, nullptr, false);
                args["path"] = path.is_discarded() ? nlohmann::json::array() : path;
            }
            if (request.kind == Kind::SaveCsv) args["file_path"] = request.file;
            const auto call = [&] {
                result.json = engine_->CallVariablesTool(VariablesFunctionName(request.kind), args.dump(), request.scope,
                                                         &result.busy);
            };
            if (request.kind == Kind::SaveCsv) {
                cyxwiz::AsyncTaskManager::Instance().RunHere(
                    "Save " + PathLabel(request.path_json) + " as CSV",
                    [&](cyxwiz::LambdaTask& task) {
                        call();
                        if (result.busy) task.MarkCancelled("A run was active; save again when it ends");
                    },
                    false);
            } else {
                call();
            }
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return;
        done_.push_back(std::move(result));
    }
}

}  // namespace scripting
