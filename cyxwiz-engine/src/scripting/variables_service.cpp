#include "variables_service.h"

#include "scripting_engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace scripting {

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
            engine_->ReadVariableTable(request.scope, request.path_json, request.max_rows, request.index, &table, &result.busy);
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
            result.json = engine_->CallVariablesTool(VariablesFunctionName(request.kind), args.dump(), request.scope, &result.busy);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return;
        done_.push_back(std::move(result));
    }
}

}  // namespace scripting
