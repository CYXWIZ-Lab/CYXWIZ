#include "language_service.h"

#include "scripting_engine.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>

namespace scripting {

const char* LanguageFunctionName(LanguageService::Kind kind) {
    switch (kind) {
        case LanguageService::Kind::Complete: return "complete";
        case LanguageService::Kind::Describe: return "describe";
        case LanguageService::Kind::Hover: return "hover";
        case LanguageService::Kind::Signatures: return "signatures";
        case LanguageService::Kind::Definition: return "definition";
        case LanguageService::Kind::Diagnostics: return "diagnostics";
    }
    return "status";
}

LanguageService::LanguageService(ScriptingEngine* engine) : engine_(engine) {
    worker_ = std::thread([this] { Run(); });
}

LanguageService::~LanguageService() { Stop(); }

void LanguageService::Stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        queue_.clear();
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::uint64_t LanguageService::Submit(Request request) {
    std::uint64_t id;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return 0;
        id = request.id = next_id_++;
        // A newer request of the same kind replaces one that has not started.
        queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [&](const Request& r) { return r.kind == request.kind; }),
                     queue_.end());
        queue_.push_back(std::move(request));
    }
    wake_.notify_one();
    return id;
}

std::vector<LanguageService::Result> LanguageService::Poll() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<Result> out;
    out.swap(done_);
    return out;
}

void LanguageService::Run() {
    while (true) {
        Request request;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (stop_) return;
            request = std::move(queue_.front());
            queue_.pop_front();
        }
        nlohmann::json args = {{"source", request.source}, {"path", request.path}};
        switch (request.kind) {
            case Kind::Diagnostics:
                args["builtins_extra"] = request.known_names;
                break;
            case Kind::Describe:
                args["name"] = request.name;
                [[fallthrough]];
            default:
                args["line"] = request.line;
                args["column"] = request.column;
                args["project_root"] = request.project_root;
                break;
        }
        Result result;
        result.kind = request.kind;
        result.id = request.id;
        result.json = engine_->CallLanguageTool(LanguageFunctionName(request.kind), args.dump(), request.namespace_key);
        std::lock_guard<std::mutex> lock(mutex_);
        if (stop_) return;
        done_.push_back(std::move(result));
    }
}

}  // namespace scripting
