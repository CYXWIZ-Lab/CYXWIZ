#include "debug_recommendation_engine.h"

#include <cmath>

namespace cyxwiz {

namespace {

bool PayloadBool(const DebugTraceRecord& trace, const char* key, bool fallback = false) {
    auto it = trace.payload.find(key);
    return it != trace.payload.end() && it->is_boolean() ? it->get<bool>() : fallback;
}

double PayloadNumber(const DebugTraceRecord& trace, const char* key, double fallback = 0.0) {
    auto it = trace.payload.find(key);
    return it != trace.payload.end() && it->is_number() ? it->get<double>() : fallback;
}

std::string PayloadString(const DebugTraceRecord& trace, const char* key) {
    auto it = trace.payload.find(key);
    return it != trace.payload.end() && it->is_string() ? it->get<std::string>() : std::string{};
}

void Add(std::vector<DebugRecommendation>& out,
         DebugRecommendationSeverity severity,
         int node_id,
         const std::string& category,
         const std::string& title,
         const std::string& detail,
         const std::string& action) {
    out.push_back({severity, node_id, category, title, detail, action});
}

std::string IssueKey(const ValidationIssue& issue) {
    return std::to_string(static_cast<int>(issue.level)) + "|" +
           std::to_string(issue.node_id) + "|" + issue.node_name + "|" +
           issue.error_code + "|" + issue.message;
}

bool RememberIssue(std::vector<std::string>& seen_issue_keys,
                   const ValidationIssue& issue) {
    const std::string key = IssueKey(issue);
    for (const auto& seen : seen_issue_keys) {
        if (seen == key) {
            return false;
        }
    }
    seen_issue_keys.push_back(key);
    return true;
}

} // namespace

const char* DebugRecommendationSeverityName(DebugRecommendationSeverity severity) {
    switch (severity) {
        case DebugRecommendationSeverity::Info: return "Info";
        case DebugRecommendationSeverity::Warning: return "Warning";
        case DebugRecommendationSeverity::Critical: return "Critical";
    }
    return "Unknown";
}

std::vector<DebugRecommendation> DebugRecommendationEngine::Build(
    const std::vector<DebugTraceRecord>& traces,
    const std::vector<ValidationIssue>& issues,
    const SmokeRunResult& smoke_result,
    const CrashRunSummary& last_run,
    const TrainingTraceSummary& training_trace) const {
    std::vector<DebugRecommendation> out;

    std::vector<std::string> seen_issue_keys;
    for (const auto& issue : issues) {
        if (!RememberIssue(seen_issue_keys, issue)) {
            continue;
        }
        if (issue.level == IssueLevel::Error) {
            Add(out, DebugRecommendationSeverity::Critical, issue.node_id,
                "Issue", issue.node_name.empty() ? "Blocking issue" : issue.node_name,
                issue.message,
                "Fix this before running full training.");
        } else if (issue.level == IssueLevel::Warning) {
            Add(out, DebugRecommendationSeverity::Warning, issue.node_id,
                "Issue", issue.node_name.empty() ? "Warning" : issue.node_name,
                issue.message,
                "Review this before trusting training metrics.");
        }
    }

    for (const auto& trace : traces) {
        for (const auto& issue : trace.issues) {
            if (!RememberIssue(seen_issue_keys, issue)) {
                continue;
            }
            if (issue.level == IssueLevel::Error) {
                Add(out, DebugRecommendationSeverity::Critical, issue.node_id,
                    "Trace", "Trace error reported",
                    issue.message,
                    "Fix the trace-reported issue before running full training.");
            } else if (issue.level == IssueLevel::Warning) {
                Add(out, DebugRecommendationSeverity::Warning, issue.node_id,
                    "Trace", "Trace warning reported",
                    issue.message,
                    "Review the trace-reported warning before trusting debugger results.");
            }
        }

        if (trace.phase == "TextVocabulary") {
            const double unknown_ratio = PayloadNumber(trace, "unknown_token_ratio", 0.0);
            if (unknown_ratio > 0.20) {
                Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                    "Preprocessing", "High unknown-token ratio",
                    "The selected sample has many tokens missing from the vocabulary.",
                    "Increase vocab size, lower min_word_freq, rebuild the vocab from the training corpus, or inspect tokenizer settings.");
            } else if (unknown_ratio > 0.05) {
                Add(out, DebugRecommendationSeverity::Info, trace.node_id,
                    "Preprocessing", "Some unknown tokens detected",
                    "The selected sample has a modest unknown-token rate.",
                    "Keep an eye on aggregate smoke statistics before changing the vocabulary.");
            }
        }

        if (trace.phase == "TextPadding") {
            const bool truncated = PayloadBool(trace, "truncated", false);
            const double pad_ratio = PayloadNumber(trace, "pad_ratio", 0.0);
            if (truncated) {
                Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                    "Preprocessing", "Text sample was truncated",
                    "At least one inspected sample exceeded max_length.",
                    "Increase TextPadding max_length or inspect whether long text should be summarized/windowed.");
            } else if (pad_ratio > 0.80) {
                Add(out, DebugRecommendationSeverity::Info, trace.node_id,
                    "Preprocessing", "First sample is mostly padding",
                    "This may be normal for one short sample, but it can waste compute if common.",
                    "Use aggregate Smoke Run padding statistics before reducing max_length.");
            }
        }

        if (trace.status == "shape_mismatch") {
            Add(out, DebugRecommendationSeverity::Critical, trace.node_id,
                "Shapes", "Shape mismatch detected",
                "A runtime shape did not match the compiled graph expectation.",
                "Inspect the selected node wiring and its input/output shape settings before training.");
        }

        if (trace.phase == "Forward" &&
            (PayloadBool(trace, "has_nan", false) ||
             PayloadBool(trace, "has_inf", false))) {
            Add(out, DebugRecommendationSeverity::Critical, trace.node_id,
                "Numerics", "Local Debug produced non-finite activation",
                "A Local Debug forward trace reported NaN or Inf output values.",
                "Inspect input scaling, layer initialization, activation choice, and learning-rate-sensitive paths before training.");
        }

        if (trace.phase == "Backward" &&
            trace.role == DebugTraceRole::Gradient &&
            !PayloadBool(trace, "has_gradient", true)) {
            Add(out, DebugRecommendationSeverity::Critical, trace.node_id,
                "Gradients", "Local Debug gradient is missing",
                "A Local Debug gradient trace did not find a gradient tensor for a trainable parameter.",
                "Inspect parameter registration, loss wiring, and disconnected branches before training.");
        }

        if (trace.phase == "Backward" &&
            trace.role == DebugTraceRole::Gradient &&
            PayloadBool(trace, "is_nan", false)) {
            Add(out, DebugRecommendationSeverity::Critical, trace.node_id,
                "Gradients", "Local Debug gradient is NaN",
                "A Local Debug gradient trace reported a NaN parameter norm.",
                "Inspect the loss, labels, activation ranges, and backend numerical stability before training.");
        }

        if (trace.phase == "Backward" &&
            trace.role == DebugTraceRole::Gradient &&
            PayloadBool(trace, "has_gradient", true) &&
            PayloadBool(trace, "is_zero", false)) {
            Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                "Gradients", "Local Debug gradient is zero",
                "A Local Debug gradient trace reported a zero parameter norm.",
                "Inspect disconnected layers, saturated activations, frozen parameters, and loss wiring before training.");
        }

        if (trace.phase == "ExportCorrelation") {
            if (!PayloadBool(trace, "compile_success", true) ||
                trace.status == "failed") {
                Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                    "Export", "Export correlation failed",
                    "The generated-code/export trace reported a failed compile or correlation status.",
                    "Inspect the export compile status and generated artifact before using the exported model.");
            }
            if (PayloadString(trace, "artifact_path").empty()) {
                Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                    "Export", "Export artifact path missing",
                    "The export correlation trace did not include an artifact path.",
                    "Verify the exporter wrote an artifact and records its path in the debug trace.");
            }
        }

        if (trace.phase == "WindowsCrashImport") {
            const bool report_available = PayloadBool(trace, "report_available", true);
            const bool matched = PayloadBool(trace, "matched", true);
            if (!report_available) {
                Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                    "Crash", "Windows crash report unavailable",
                    "The crash importer did not find a Windows Error Reporting artifact for this run.",
                    "Inspect the crash heartbeat and Windows Event Viewer if the run stopped unexpectedly.");
            } else if (!matched) {
                Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                    "Crash", "Windows crash report not confidently matched",
                    "A Windows crash report was imported but could not be tied confidently to this debug run.",
                    "Compare run id, process name, report id, and crash timestamp before acting on the report.");
            }
        }

        if (trace.phase == "SmokeRun.Loss") {
            const bool pred_bad = PayloadBool(trace, "predictions_have_non_finite", false);
            const double loss = PayloadNumber(trace, "loss", 0.0);
            if (pred_bad || !std::isfinite(loss)) {
                Add(out, DebugRecommendationSeverity::Critical, trace.node_id,
                    "Optimization", "Smoke Run produced invalid values",
                    "Predictions or loss became NaN/Inf during the short real-data run.",
                    "Lower learning rate, inspect labels and input ranges, and verify backend recurrent path stability.");
            }
        }

        if (trace.phase == "SmokeRun.Backward") {
            const double grad_count = PayloadNumber(trace, "gradient_tensor_count", 0.0);
            const double zero_count = PayloadNumber(trace, "zero_gradient_tensor_count", 0.0);
            if (grad_count <= 0.0) {
                Add(out, DebugRecommendationSeverity::Critical, trace.node_id,
                    "Gradients", "No gradients observed",
                    "Smoke Run did not capture parameter gradients.",
                    "Inspect loss wiring and trainable parameter bookkeeping.");
            } else if (zero_count > 0.0 && zero_count >= grad_count) {
                Add(out, DebugRecommendationSeverity::Warning, trace.node_id,
                    "Gradients", "All gradients are zero",
                    "Every captured gradient tensor was zero in the Smoke Run batch.",
                    "Inspect dead activations, disconnected layers, and recurrent backward flow.");
            }
        }
    }

    if (smoke_result.supported) {
        if (!smoke_result.success) {
            Add(out, DebugRecommendationSeverity::Warning, -1,
                "Smoke Run", "Smoke Run needs attention",
                smoke_result.summary,
                "Do not start full training until the Smoke Run passes on real data.");
        } else if (smoke_result.last_accuracy < 0.15f) {
            Add(out, DebugRecommendationSeverity::Info, -1,
                "Smoke Run", "Low initial smoke accuracy",
                "Initial accuracy is low, which can be normal before training but is worth tracking.",
                "Compare this value after graph or preprocessing changes instead of waiting for full epochs.");
        }
    }

    if (last_run.available && last_run.suspected_crash) {
        Add(out, DebugRecommendationSeverity::Critical, -1,
            "Runtime", "Previous run may have crashed",
            last_run.warning.empty() ? "The last run did not write a clean completion marker." : last_run.warning,
            "Inspect Crash / Last Run before starting another long training run.");
    }

    if (training_trace.available) {
        for (const auto& warning : training_trace.warnings) {
            Add(out, DebugRecommendationSeverity::Warning, -1,
                "Training Trace", "Training-time warning",
                warning,
                "Inspect the Training Trace section for the stage and batch that produced this warning.");
        }
        if (training_trace.status == "running" &&
            training_trace.latest_stage == "ComputeLoss" &&
            !std::isfinite(training_trace.latest_loss)) {
            Add(out, DebugRecommendationSeverity::Critical, -1,
                "Training Trace", "Live training loss is not finite",
                "The latest training trace shows NaN or Inf loss.",
                "Stop training, lower the learning rate, and inspect labels/input ranges.");
        }
    }

    if (out.empty()) {
        Add(out, DebugRecommendationSeverity::Info, -1,
            "Summary", "No immediate debugger recommendations",
            "Preflight, local debug, and smoke traces did not trigger the current rules.",
            "Proceed to a short controlled training run and compare metrics.");
    }

    return out;
}

} // namespace cyxwiz
