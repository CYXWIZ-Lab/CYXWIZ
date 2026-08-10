#include "debug_run_store.h"
#include "training_trace_collector.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>

namespace cyxwiz {

namespace {

std::filesystem::path StoreRoot() {
    return std::filesystem::current_path() / ".cyxwiz" / "debug_runs" / "studio";
}

std::filesystem::path RunPath(const std::string& run_id) {
    return StoreRoot() / run_id / "session.json";
}

const char* IssueLevelName(IssueLevel level) {
    switch (level) {
        case IssueLevel::Error: return "Error";
        case IssueLevel::Warning: return "Warning";
        case IssueLevel::Info: return "Info";
    }
    return "Unknown";
}

IssueLevel IssueLevelFromString(const std::string& value) {
    if (value == "Error") return IssueLevel::Error;
    if (value == "Warning") return IssueLevel::Warning;
    return IssueLevel::Info;
}

DebugTraceRole DebugTraceRoleFromString(const std::string& value) {
    if (value == "RawInput") return DebugTraceRole::RawInput;
    if (value == "PreprocessingOutput") return DebugTraceRole::PreprocessingOutput;
    if (value == "FeatureTensor") return DebugTraceRole::FeatureTensor;
    if (value == "ModelInput") return DebugTraceRole::ModelInput;
    if (value == "Activation") return DebugTraceRole::Activation;
    if (value == "Parameter") return DebugTraceRole::Parameter;
    if (value == "Gradient") return DebugTraceRole::Gradient;
    if (value == "Prediction") return DebugTraceRole::Prediction;
    if (value == "Target") return DebugTraceRole::Target;
    if (value == "Loss") return DebugTraceRole::Loss;
    if (value == "OptimizerStep") return DebugTraceRole::OptimizerStep;
    if (value == "CompileArtifact") return DebugTraceRole::CompileArtifact;
    if (value == "GeneratedCode") return DebugTraceRole::GeneratedCode;
    if (value == "StudioEvent") return DebugTraceRole::StudioEvent;
    if (value == "Warning") return DebugTraceRole::Warning;
    if (value == "Error") return DebugTraceRole::Error;
    return DebugTraceRole::Activation;
}

DebugRecommendationSeverity RecommendationSeverityFromString(const std::string& value) {
    if (value == "Critical") return DebugRecommendationSeverity::Critical;
    if (value == "Warning") return DebugRecommendationSeverity::Warning;
    return DebugRecommendationSeverity::Info;
}

nlohmann::json IssueToJson(const ValidationIssue& issue) {
    return {
        {"level", IssueLevelName(issue.level)},
        {"node_id", issue.node_id},
        {"node_name", issue.node_name},
        {"error_code", issue.error_code},
        {"message", issue.message}
    };
}

ValidationIssue IssueFromJson(const nlohmann::json& j) {
    ValidationIssue issue;
    issue.level = IssueLevelFromString(j.value("level", "Info"));
    issue.node_id = j.value("node_id", -1);
    issue.node_name = j.value("node_name", "");
    issue.error_code = j.value("error_code", "");
    issue.message = j.value("message", "");
    return issue;
}

nlohmann::json TraceToJson(const DebugTraceRecord& trace) {
    nlohmann::json issues = nlohmann::json::array();
    for (const auto& issue : trace.issues) {
        issues.push_back(IssueToJson(issue));
    }
    return {
        {"run_id", trace.run_id},
        {"node_id", trace.node_id},
        {"node_name", trace.node_name},
        {"node_type", trace.node_type},
        {"phase", trace.phase},
        {"role", DebugTraceRoleName(trace.role)},
        {"input_shape", trace.input_shape},
        {"output_shape", trace.output_shape},
        {"dtype", trace.dtype},
        {"duration_ms", trace.duration_ms},
        {"status", trace.status},
        {"issues", issues},
        {"payload", trace.payload}
    };
}

DebugTraceRecord TraceFromJson(const nlohmann::json& j) {
    DebugTraceRecord trace;
    trace.run_id = j.value("run_id", "");
    trace.node_id = j.value("node_id", -1);
    trace.node_name = j.value("node_name", "");
    trace.node_type = j.value("node_type", "");
    trace.phase = j.value("phase", "");
    trace.role = DebugTraceRoleFromString(j.value("role", "Activation"));
    trace.input_shape = j.value("input_shape", std::vector<size_t>{});
    trace.output_shape = j.value("output_shape", std::vector<size_t>{});
    trace.dtype = j.value("dtype", "");
    trace.duration_ms = j.value("duration_ms", 0.0f);
    trace.status = j.value("status", "");
    if (j.contains("issues") && j["issues"].is_array()) {
        for (const auto& item : j["issues"]) {
            trace.issues.push_back(IssueFromJson(item));
        }
    }
    if (j.contains("payload")) {
        trace.payload = j["payload"];
    }
    return trace;
}

nlohmann::json EventToJson(const StudioEventRecord& event) {
    return {
        {"run_id", event.run_id},
        {"timestamp", event.timestamp},
        {"graph_hash", event.graph_hash},
        {"selected_node_id", event.selected_node_id},
        {"action", event.action},
        {"status", event.status},
        {"message", event.message}
    };
}

StudioEventRecord EventFromJson(const nlohmann::json& j) {
    StudioEventRecord event;
    event.run_id = j.value("run_id", "");
    event.timestamp = j.value("timestamp", "");
    event.graph_hash = j.value("graph_hash", static_cast<uint64_t>(0));
    event.selected_node_id = j.value("selected_node_id", -1);
    event.action = j.value("action", "");
    event.status = j.value("status", "");
    event.message = j.value("message", "");
    return event;
}

nlohmann::json RecommendationToJson(const DebugRecommendation& rec) {
    return {
        {"severity", DebugRecommendationSeverityName(rec.severity)},
        {"node_id", rec.node_id},
        {"category", rec.category},
        {"title", rec.title},
        {"detail", rec.detail},
        {"action", rec.action}
    };
}

DebugRecommendation RecommendationFromJson(const nlohmann::json& j) {
    DebugRecommendation rec;
    rec.severity = RecommendationSeverityFromString(j.value("severity", "Info"));
    rec.node_id = j.value("node_id", -1);
    rec.category = j.value("category", "");
    rec.title = j.value("title", "");
    rec.detail = j.value("detail", "");
    rec.action = j.value("action", "");
    return rec;
}

nlohmann::json ExecutionSummaryToJson(
    const DebugRunExecutionSummary& execution) {
    return {
        {"available", execution.available},
        {"training_run_id", execution.training_run_id},
        {"status", execution.status},
        {"requested_backend", execution.requested_backend},
        {"requested_device_id", execution.requested_device_id},
        {"effective_backend", execution.effective_backend},
        {"effective_device_id", execution.effective_device_id},
        {"effective_device_name", execution.effective_device_name},
        {"execution_context_id", execution.execution_context_id},
        {"placement_fingerprint", execution.placement_fingerprint},
        {"residency_verdict", execution.residency_verdict},
        {"native_cpu_fallback_count", execution.native_cpu_fallback_count},
        {"transfer_event_count", execution.transfer_event_count},
        {"transfer_known_bytes", execution.transfer_known_bytes},
        {"synchronization_event_count", execution.synchronization_event_count},
        {"synchronization_known_bytes", execution.synchronization_known_bytes}
    };
}

DebugRunExecutionSummary ExecutionSummaryFromJson(const nlohmann::json& j) {
    DebugRunExecutionSummary execution;
    execution.available = j.value("available", false);
    execution.training_run_id = j.value("training_run_id", "");
    execution.status = j.value("status", "");
    execution.requested_backend = j.value("requested_backend", "");
    execution.requested_device_id = j.value("requested_device_id", 0);
    execution.effective_backend = j.value("effective_backend", "");
    execution.effective_device_id = j.value("effective_device_id", 0);
    execution.effective_device_name = j.value("effective_device_name", "");
    execution.execution_context_id = j.value("execution_context_id", "");
    execution.placement_fingerprint = j.value("placement_fingerprint", "");
    execution.residency_verdict = j.value("residency_verdict", "");
    execution.native_cpu_fallback_count =
        j.value("native_cpu_fallback_count", static_cast<size_t>(0));
    execution.transfer_event_count =
        j.value("transfer_event_count", static_cast<size_t>(0));
    execution.transfer_known_bytes =
        j.value("transfer_known_bytes", static_cast<uint64_t>(0));
    execution.synchronization_event_count =
        j.value("synchronization_event_count", static_cast<size_t>(0));
    execution.synchronization_known_bytes =
        j.value("synchronization_known_bytes", static_cast<uint64_t>(0));
    return execution;
}

DebugRunStoreSummary SummaryFromJson(const nlohmann::json& j,
                                     const std::filesystem::path& path) {
    DebugRunStoreSummary summary;
    summary.run_id = j.value("run_id", "");
    summary.timestamp = j.value("timestamp", "");
    summary.graph_hash = j.value("graph_hash", static_cast<uint64_t>(0));
    summary.success = j.value("success", false);
    summary.issue_count = j.value("issue_count", static_cast<size_t>(0));
    summary.trace_count = j.value("trace_count", static_cast<size_t>(0));
    summary.event_count = j.value("event_count", static_cast<size_t>(0));
    summary.recommendation_count = j.value("recommendation_count", static_cast<size_t>(0));
    summary.summary = j.value("summary", "");
    summary.file_path = path.string();
    if (j.contains("execution") && j["execution"].is_object()) {
        summary.execution = ExecutionSummaryFromJson(j["execution"]);
    }
    return summary;
}

DebugRunStoreRecord RecordFromJson(const nlohmann::json& j,
                                   const std::filesystem::path& path) {
    DebugRunStoreRecord record;
    record.summary = SummaryFromJson(j, path);

    if (j.contains("issues") && j["issues"].is_array()) {
        for (const auto& item : j["issues"]) {
            record.issues.push_back(IssueFromJson(item));
        }
    }
    if (j.contains("traces") && j["traces"].is_array()) {
        for (const auto& item : j["traces"]) {
            record.traces.push_back(TraceFromJson(item));
        }
    }
    if (j.contains("studio_events") && j["studio_events"].is_array()) {
        for (const auto& item : j["studio_events"]) {
            record.studio_events.push_back(EventFromJson(item));
        }
    }
    if (j.contains("recommendations") && j["recommendations"].is_array()) {
        for (const auto& item : j["recommendations"]) {
            record.recommendations.push_back(RecommendationFromJson(item));
        }
    }

    return record;
}

} // namespace

DebugRunExecutionSummary MakeDebugRunExecutionSummary(
    const TrainingTraceSummary& trace) {
    DebugRunExecutionSummary execution;
    execution.available = trace.available && !trace.run_id.empty();
    execution.training_run_id = trace.run_id;
    execution.status = trace.status;
    execution.requested_backend = trace.requested_backend;
    execution.requested_device_id = trace.requested_device_id;
    execution.effective_backend = trace.effective_backend;
    execution.effective_device_id = trace.effective_device_id;
    execution.effective_device_name = trace.effective_device_name;
    execution.execution_context_id = trace.execution_context_id;
    execution.placement_fingerprint = trace.placement_fingerprint;
    execution.residency_verdict = trace.residency_verdict;
    execution.native_cpu_fallback_count = trace.native_cpu_fallback_count;
    execution.transfer_event_count = trace.transfer_event_count;
    execution.transfer_known_bytes = trace.transfer_known_bytes;
    execution.synchronization_event_count = trace.synchronization_event_count;
    execution.synchronization_known_bytes = trace.synchronization_known_bytes;
    return execution;
}

bool DebugRunStore::Save(const DebugRunStoreRecord& record) {
    if (record.summary.run_id.empty()) {
        return false;
    }

    try {
        const auto path = RunPath(record.summary.run_id);
        std::filesystem::create_directories(path.parent_path());

        nlohmann::json issues = nlohmann::json::array();
        for (const auto& issue : record.issues) {
            issues.push_back(IssueToJson(issue));
        }

        nlohmann::json traces = nlohmann::json::array();
        for (const auto& trace : record.traces) {
            traces.push_back(TraceToJson(trace));
        }

        nlohmann::json events = nlohmann::json::array();
        for (const auto& event : record.studio_events) {
            events.push_back(EventToJson(event));
        }

        nlohmann::json recommendations = nlohmann::json::array();
        for (const auto& rec : record.recommendations) {
            recommendations.push_back(RecommendationToJson(rec));
        }

        nlohmann::json j = {
            {"run_id", record.summary.run_id},
            {"timestamp", record.summary.timestamp},
            {"graph_hash", record.summary.graph_hash},
            {"success", record.summary.success},
            {"issue_count", record.issues.size()},
            {"trace_count", record.traces.size()},
            {"event_count", record.studio_events.size()},
            {"recommendation_count", record.recommendations.size()},
            {"summary", record.summary.summary},
            {"execution", ExecutionSummaryToJson(record.summary.execution)},
            {"issues", issues},
            {"traces", traces},
            {"studio_events", events},
            {"recommendations", recommendations}
        };

        std::ofstream file(path, std::ios::trunc);
        file << std::setw(2) << j << '\n';
        return true;
    } catch (...) {
        return false;
    }
}

std::optional<DebugRunStoreRecord> DebugRunStore::Load(const std::string& run_id) {
    if (run_id.empty()) {
        return std::nullopt;
    }

    const auto path = RunPath(run_id);
    if (!std::filesystem::exists(path)) {
        return std::nullopt;
    }

    try {
        std::ifstream file(path);
        nlohmann::json j;
        file >> j;
        return RecordFromJson(j, path);
    } catch (...) {
        return std::nullopt;
    }
}

std::vector<DebugRunStoreSummary> DebugRunStore::ListRecent(size_t max_runs) {
    std::vector<DebugRunStoreSummary> out;
    const auto root = StoreRoot();
    if (!std::filesystem::exists(root)) {
        return out;
    }

    for (const auto& entry : std::filesystem::directory_iterator(root)) {
        if (!entry.is_directory()) {
            continue;
        }
        const auto path = entry.path() / "session.json";
        if (!std::filesystem::exists(path)) {
            continue;
        }
        try {
            std::ifstream file(path);
            nlohmann::json j;
            file >> j;
            out.push_back(SummaryFromJson(j, path));
        } catch (...) {
        }
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return a.timestamp > b.timestamp;
    });
    if (out.size() > max_runs) {
        out.resize(max_runs);
    }
    return out;
}

} // namespace cyxwiz
