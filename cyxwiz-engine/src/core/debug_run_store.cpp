#include "debug_run_store.h"
#include "debug_run_paths.h"
#include "debug_session.h"
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
    return GetDebugRunRoot() / "studio";
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

DebugReplayCompiledConfigSummary CompiledConfigSummaryFromJson(
    const nlohmann::json& j) {
    DebugReplayCompiledConfigSummary config;
    config.available = j.value("available", false);
    config.valid = j.value("valid", false);
    config.layer_count = j.value("layer_count", static_cast<size_t>(0));
    config.input_shape = j.value("input_shape", std::vector<size_t>{});
    config.input_size = j.value("input_size", static_cast<size_t>(0));
    config.output_size = j.value("output_size", static_cast<size_t>(0));
    config.batch_size = j.value("batch_size", 0);
    config.epochs = j.value("epochs", 0);
    config.shuffle = j.value("shuffle", false);
    config.drop_last = j.value("drop_last", false);
    config.num_workers = j.value("num_workers", 0);
    config.prefetch_factor = j.value("prefetch_factor", 0);
    config.log_interval = j.value("log_interval", 0);
    config.validation_freq = j.value("validation_freq", 0);
    config.grad_accum_steps = j.value("grad_accum_steps", 0);
    config.train_ratio = j.value("train_ratio", 0.0f);
    config.val_ratio = j.value("val_ratio", 0.0f);
    config.test_ratio = j.value("test_ratio", 0.0f);
    config.stratified = j.value("stratified", false);
    config.loss = j.value("loss", "");
    config.optimizer = j.value("optimizer", "");
    config.learning_rate = j.value("learning_rate", 0.0f);
    config.momentum = j.value("momentum", 0.0f);
    config.beta1 = j.value("beta1", 0.0f);
    config.beta2 = j.value("beta2", 0.0f);
    config.weight_decay = j.value("weight_decay", 0.0f);
    config.compiler_placement_fingerprint =
        j.value("compiler_placement_fingerprint", "");
    config.backend_placement_count =
        j.value("backend_placement_count", static_cast<size_t>(0));
    config.forbid_native_cpu_fallback =
        j.value("forbid_native_cpu_fallback", false);
    return config;
}

std::map<std::string, std::string> ReplayEnvironmentSummary() {
    std::map<std::string, std::string> environment;
#if defined(_WIN32)
    environment["platform"] = "windows";
#elif defined(__APPLE__)
    environment["platform"] = "macos";
#elif defined(__linux__)
    environment["platform"] = "linux";
#else
    environment["platform"] = "unknown";
#endif

#if defined(_M_X64) || defined(__x86_64__)
    environment["architecture"] = "x86_64";
#elif defined(_M_ARM64) || defined(__aarch64__)
    environment["architecture"] = "arm64";
#elif defined(_M_IX86) || defined(__i386__)
    environment["architecture"] = "x86";
#else
    environment["architecture"] = "unknown";
#endif

#if defined(NDEBUG)
    environment["build_configuration"] = "release";
#else
    environment["build_configuration"] = "debug";
#endif
    environment["compute_platform"] = "ArrayFire";
    return environment;
}

nlohmann::json ExecutionSummaryToJson(
    const DebugRunExecutionSummary& execution) {
    return {
        {"available", execution.available},
        {"correlated", execution.correlated},
        {"evidence_scope", execution.evidence_scope},
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
    execution.correlated = j.value("correlated", false);
    execution.evidence_scope = j.value(
        "evidence_scope",
        execution.available ? "latest_training_run_unlinked" : "unobserved");
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

    if (j.contains("replay_capsule") && j["replay_capsule"].is_object()) {
        record.replay_capsule =
            DebugRunReplayCapsuleFromJson(j["replay_capsule"]);
    }

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

DebugRunReplayCapsule MakeDebugRunReplayCapsule(
    const DebugSession& session,
    const TrainingConfiguration* config,
    const DebugRunExecutionSummary& execution,
    size_t smoke_sample_limit) {
    DebugRunReplayCapsule capsule;
    capsule.available = !session.run_id.empty();
    capsule.mode = session.mode_name;
    capsule.graph_hash = session.graph_hash;
    capsule.selected_sample_index = session.selected_sample_index;
    capsule.smoke_sample_limit = smoke_sample_limit;
    capsule.environment = ReplayEnvironmentSummary();

    for (size_t i = 0; i < session.traces.size(); ++i) {
        if (session.traces[i].phase == "GraphSnapshot") {
            capsule.graph_snapshot_trace_available = true;
            capsule.graph_snapshot_trace_index = i;
            break;
        }
    }

    if (config) {
        capsule.dataset_reference = config->dataset_name;
        capsule.smoke_batch_size_limit = smoke_sample_limit == 0
            ? 0
            : static_cast<size_t>(std::max(1, std::min(config->batch_size, 32)));
        capsule.split_seed = config->split_seed;
        capsule.dataloader_seed = config->dataloader_seed;
        capsule.balance_seed = config->balance_seed;

        auto& compiled = capsule.compiled_config;
        compiled.available = true;
        compiled.valid = config->is_valid;
        compiled.layer_count = config->layers.size();
        compiled.input_shape = config->input_shape;
        compiled.input_size = config->input_size;
        compiled.output_size = config->output_size;
        compiled.batch_size = config->batch_size;
        compiled.epochs = config->epochs;
        compiled.shuffle = config->shuffle;
        compiled.drop_last = config->drop_last;
        compiled.num_workers = config->num_workers;
        compiled.prefetch_factor = config->prefetch_factor;
        compiled.log_interval = config->log_interval;
        compiled.validation_freq = config->validation_freq;
        compiled.grad_accum_steps = config->grad_accum_steps;
        compiled.train_ratio = config->train_ratio;
        compiled.val_ratio = config->val_ratio;
        compiled.test_ratio = config->test_ratio;
        compiled.stratified = config->stratified;
        compiled.loss = config->GetLossName();
        compiled.optimizer = config->GetOptimizerName();
        compiled.learning_rate = config->learning_rate;
        compiled.momentum = config->momentum;
        compiled.beta1 = config->beta1;
        compiled.beta2 = config->beta2;
        compiled.weight_decay = config->weight_decay;
        compiled.compiler_placement_fingerprint =
            config->compiler_placement_fingerprint;
        compiled.backend_placement_count = config->backend_placements.size();
        compiled.forbid_native_cpu_fallback =
            config->forbid_native_cpu_fallback;
    }

    if (execution.available) {
        capsule.backend_evidence_scope = execution.evidence_scope;
        capsule.backend_source_run_id = execution.training_run_id;
        capsule.requested_backend = execution.requested_backend;
        capsule.requested_device_id = execution.requested_device_id;
        capsule.effective_backend = execution.effective_backend;
        capsule.effective_device_id = execution.effective_device_id;
        capsule.effective_device_name = execution.effective_device_name;
    }
    return capsule;
}

nlohmann::json DebugRunReplayCapsuleToJson(
    const DebugRunReplayCapsule& capsule) {
    const auto& config = capsule.compiled_config;
    return {
        {"schema", DebugRunReplayCapsule::kSchema},
        {"available", capsule.available},
        {"mode", capsule.mode},
        {"replay_scope", capsule.replay_scope},
        {"graph_hash", capsule.graph_hash},
        {"graph_snapshot_trace_available",
         capsule.graph_snapshot_trace_available},
        {"graph_snapshot_trace_index", capsule.graph_snapshot_trace_index},
        {"dataset_reference", capsule.dataset_reference},
        {"selected_sample_index", capsule.selected_sample_index},
        {"smoke_sample_limit", capsule.smoke_sample_limit},
        {"smoke_batch_size_limit", capsule.smoke_batch_size_limit},
        {"compiled_config", {
            {"available", config.available},
            {"valid", config.valid},
            {"scope", "reconstruction_critical_summary"},
            {"graph_snapshot_is_source_of_truth", true},
            {"layer_count", config.layer_count},
            {"input_shape", config.input_shape},
            {"input_size", config.input_size},
            {"output_size", config.output_size},
            {"batch_size", config.batch_size},
            {"epochs", config.epochs},
            {"shuffle", config.shuffle},
            {"drop_last", config.drop_last},
            {"num_workers", config.num_workers},
            {"prefetch_factor", config.prefetch_factor},
            {"log_interval", config.log_interval},
            {"validation_freq", config.validation_freq},
            {"grad_accum_steps", config.grad_accum_steps},
            {"train_ratio", config.train_ratio},
            {"val_ratio", config.val_ratio},
            {"test_ratio", config.test_ratio},
            {"stratified", config.stratified},
            {"loss", config.loss},
            {"optimizer", config.optimizer},
            {"learning_rate", config.learning_rate},
            {"momentum", config.momentum},
            {"beta1", config.beta1},
            {"beta2", config.beta2},
            {"weight_decay", config.weight_decay},
            {"compiler_placement_fingerprint",
             config.compiler_placement_fingerprint},
            {"backend_placement_count", config.backend_placement_count},
            {"forbid_native_cpu_fallback",
             config.forbid_native_cpu_fallback}
        }},
        {"seeds", {
            {"split", capsule.split_seed},
            {"dataloader", capsule.dataloader_seed},
            {"class_balance", capsule.balance_seed}
        }},
        {"backend_selection", {
            {"evidence_scope", capsule.backend_evidence_scope},
            {"source_run_id", capsule.backend_source_run_id},
            {"requested_backend", capsule.requested_backend},
            {"requested_device_id", capsule.requested_device_id},
            {"effective_backend", capsule.effective_backend},
            {"effective_device_id", capsule.effective_device_id},
            {"effective_device_name", capsule.effective_device_name}
        }},
        {"environment", capsule.environment},
        {"trace_records_embedded", capsule.trace_records_embedded},
        {"issues_embedded", capsule.issues_embedded},
        {"data_values_included", capsule.raw_dataset_values_included},
        {"exact_replay_claimed", capsule.exact_replay_claimed}
    };
}

DebugRunReplayCapsule DebugRunReplayCapsuleFromJson(
    const nlohmann::json& value) {
    DebugRunReplayCapsule capsule;
    capsule.available = value.value("available", false);
    capsule.mode = value.value("mode", "");
    capsule.replay_scope =
        value.value("replay_scope", "explain_and_recompile");
    capsule.graph_hash = value.value("graph_hash", static_cast<uint64_t>(0));
    capsule.graph_snapshot_trace_available =
        value.value("graph_snapshot_trace_available", false);
    capsule.graph_snapshot_trace_index =
        value.value("graph_snapshot_trace_index", static_cast<size_t>(0));
    capsule.dataset_reference = value.value("dataset_reference", "");
    capsule.selected_sample_index =
        value.value("selected_sample_index", static_cast<size_t>(0));
    capsule.smoke_sample_limit =
        value.value("smoke_sample_limit", static_cast<size_t>(0));
    capsule.smoke_batch_size_limit =
        value.value("smoke_batch_size_limit", static_cast<size_t>(0));
    if (value.contains("compiled_config") &&
        value["compiled_config"].is_object()) {
        capsule.compiled_config =
            CompiledConfigSummaryFromJson(value["compiled_config"]);
    }
    if (value.contains("seeds") && value["seeds"].is_object()) {
        const auto& seeds = value["seeds"];
        capsule.split_seed = seeds.value("split", 0);
        capsule.dataloader_seed = seeds.value("dataloader", 0);
        capsule.balance_seed = seeds.value("class_balance", 0);
    }
    if (value.contains("backend_selection") &&
        value["backend_selection"].is_object()) {
        const auto& backend = value["backend_selection"];
        capsule.backend_evidence_scope =
            backend.value("evidence_scope", "unobserved");
        capsule.backend_source_run_id = backend.value("source_run_id", "");
        capsule.requested_backend = backend.value("requested_backend", "");
        capsule.requested_device_id = backend.value("requested_device_id", 0);
        capsule.effective_backend = backend.value("effective_backend", "");
        capsule.effective_device_id = backend.value("effective_device_id", 0);
        capsule.effective_device_name =
            backend.value("effective_device_name", "");
    }
    capsule.environment = value.value(
        "environment", std::map<std::string, std::string>{});
    capsule.trace_records_embedded =
        value.value("trace_records_embedded", true);
    capsule.issues_embedded = value.value("issues_embedded", true);
    capsule.raw_dataset_values_included = value.value(
        "data_values_included",
        value.value("raw_dataset_values_included", false));
    capsule.exact_replay_claimed =
        value.value("exact_replay_claimed", false);
    return capsule;
}

DebugRunExecutionSummary MakeDebugRunExecutionSummary(
    const TrainingTraceSummary& trace,
    bool explicitly_selected) {
    DebugRunExecutionSummary execution;
    execution.available = trace.available && !trace.run_id.empty();
    execution.correlated = execution.available && explicitly_selected;
    execution.evidence_scope = !execution.available
        ? "unobserved"
        : (execution.correlated
            ? "selected_training_run"
            : "latest_training_run_unlinked");
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
            {"replay_capsule",
             DebugRunReplayCapsuleToJson(record.replay_capsule)},
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
