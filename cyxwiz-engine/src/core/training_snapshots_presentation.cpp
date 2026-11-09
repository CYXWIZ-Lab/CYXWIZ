#include "training_snapshots_presentation.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>

namespace cyxwiz::training_snapshots {

namespace {

constexpr const char* kDot = " \xC2\xB7 ";  // " · "

std::string Fixed(double value, int digits) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", digits, value);
    return buffer;
}

std::string MetricText(float value, bool regression) {
    return regression ? Fixed(value, 3) : Fixed(value * 100.0, 2) + " %";
}

const TrainingEpochRecord* EpochOf(const std::optional<TrainingRunRecord>& record, int epoch) {
    if (!record) return nullptr;
    for (const auto& e : record->epochs)
        if (e.epoch == epoch) return &e;
    return nullptr;
}

std::vector<int> SnapshotEpochs(const std::vector<LayerCaptureInfo>& captures) {
    std::vector<int> epochs;
    for (const auto& c : captures)
        if (c.epoch > 0) epochs.push_back(c.epoch);
    std::sort(epochs.begin(), epochs.end());
    return epochs;
}

std::string JoinEpochs(const std::vector<int>& epochs) {
    std::string text;
    for (size_t i = 0; i < epochs.size(); ++i) text += (i ? ", " : "") + std::to_string(epochs[i]);
    return text;
}

}  // namespace

std::string WhenFromRunId(const std::string& run_id) {
    size_t end = run_id.size(), begin = end;
    while (begin > 0 && std::isdigit(static_cast<unsigned char>(run_id[begin - 1]))) --begin;
    if (end - begin < 12 || end - begin > 15) return {};
    const std::time_t seconds = static_cast<std::time_t>(std::stoll(run_id.substr(begin)) / 1000);
    std::tm local{};
#if defined(_WIN32)
    if (localtime_s(&local, &seconds) != 0) return {};
#else
    if (!localtime_r(&seconds, &local)) return {};
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M", &local);
    return buffer;
}

std::string SizeText(std::uintmax_t bytes) {
    if (bytes < 1024 * 1024) return Fixed(static_cast<double>(bytes) / 1024.0, 1) + " KB";
    return Fixed(static_cast<double>(bytes) / 1048576.0, 1) + " MB";
}

std::string SnapshotName(const std::string& label) {
    int epoch = 0;
    if (std::sscanf(label.c_str(), "epoch_%d", &epoch) == 1) return "epoch " + std::to_string(epoch);
    return label;
}

RunRow TrainingRunRow(const std::string& run_id, const std::optional<TrainingRunRecord>& record,
                      const std::vector<LayerCaptureInfo>& captures) {
    RunRow row;
    row.run_id = run_id;
    row.training = true;
    row.kind = "Training";
    row.title = run_id + (record && !record->graph_name.empty() ? kDot + record->graph_name : "");
    row.when = WhenFromRunId(run_id);
    const auto epochs = SnapshotEpochs(captures);
    const std::string snapshots = std::to_string(epochs.size()) + " layer snapshot" + (epochs.size() == 1 ? "" : "s") +
                                  (epochs.empty() ? "" : " (epoch" + std::string(epochs.size() == 1 ? " " : "s ") +
                                                             JoinEpochs(epochs) + ")");
    if (!record || record->epochs.empty()) {
        row.detail = snapshots + kDot + "epochs not recorded";
        return row;
    }
    std::string detail = std::to_string(record->total_epochs) + " epochs";
    if (static_cast<int>(record->epochs.size()) < record->total_epochs)
        detail = std::to_string(record->epochs.back().epoch) + " of " + detail;
    const TrainingEpochRecord* best = nullptr;
    for (const auto& e : record->epochs) {
        if (!e.val_metric) continue;
        if (!best || (record->regression ? *e.val_metric < *best->val_metric : *e.val_metric > *best->val_metric))
            best = &e;
    }
    if (best)
        detail += kDot + std::string("best validation ") + (record->regression ? "MAE " : "") +
                  MetricText(*best->val_metric, record->regression) + " (epoch " + std::to_string(best->epoch) + ")";
    else
        detail += kDot + std::string("final training loss ") + Fixed(record->epochs.back().train_loss, 3);
    row.detail = detail + kDot + snapshots;
    return row;
}

RunRow DebugRunRow(const DebugRunStoreSummary& run) {
    RunRow row;
    row.run_id = run.run_id;
    row.kind = "Local Debug";
    row.title = run.run_id;
    row.when = run.timestamp;
    row.detail = std::string(run.success ? "Passed" : "Needs attention") +
                 (run.issue_count ? kDot + std::to_string(run.issue_count) + (run.issue_count == 1 ? " issue" : " issues")
                                  : "") +
                 (run.summary.empty() ? "" : kDot + run.summary);
    const auto add = [&row](std::string key, std::string value) {
        if (!value.empty()) row.details.emplace_back(std::move(key), std::move(value));
    };
    add("Traces", std::to_string(run.trace_count));
    add("Events", std::to_string(run.event_count));
    add("Recommendations", std::to_string(run.recommendation_count));
    char hash[24];
    std::snprintf(hash, sizeof(hash), "0x%016llx", static_cast<unsigned long long>(run.graph_hash));
    add("Graph", hash);
    if (run.execution.available) {
        const auto& e = run.execution;
        add("Training run", e.training_run_id);
        add("Evidence", e.evidence_scope);
        add("Backend", e.effective_backend.empty() ? "unknown"
                                                   : e.effective_backend + ":" + std::to_string(e.effective_device_id));
        add("Device", e.effective_device_name.empty() ? "unknown" : e.effective_device_name);
        add("Residency", e.residency_verdict.empty() ? "unknown" : e.residency_verdict);
        add("Transfers", std::to_string(e.transfer_event_count) + ", " + SizeText(e.transfer_known_bytes));
        add("Synchronizations",
            std::to_string(e.synchronization_event_count) + ", " + SizeText(e.synchronization_known_bytes));
    }
    add("Saved at", run.file_path);
    return row;
}

SnapshotTable SnapshotTableFor(const std::string& run_id, const std::optional<TrainingRunRecord>& record,
                               const std::vector<LayerCaptureInfo>& captures) {
    SnapshotTable table;
    const bool regression = record && record->regression;
    table.heading = "Snapshots of " + run_id + (record && !record->graph_name.empty() ? kDot + record->graph_name : "") +
                    (record && record->total_epochs > 0 ? " (" + std::to_string(record->total_epochs) + " epochs)" : "");
    table.metric_header = regression ? "Validation MAE" : "Validation accuracy";
    for (const auto& c : captures) {
        if (c.epoch <= 0) continue;
        SnapshotRow row;
        row.epoch = c.epoch;
        row.label = c.label;
        row.size = SizeText(c.bytes);
        if (const TrainingEpochRecord* e = EpochOf(record, c.epoch)) {
            row.train_loss = Fixed(e->train_loss, 3);
            row.val_loss = e->val_loss ? Fixed(*e->val_loss, 3) : "no validation";
            row.val_metric = e->val_metric ? MetricText(*e->val_metric, regression) : "no validation";
        } else {
            row.train_loss = row.val_loss = row.val_metric = "not recorded";
        }
        table.rows.push_back(std::move(row));
    }
    std::sort(table.rows.begin(), table.rows.end(), [](const auto& a, const auto& b) { return a.epoch < b.epoch; });
    table.note = record ? "Each snapshot's metrics are its epoch's, saved with the run when the snapshot was taken."
                        : "This run's epochs were not recorded (snapshots saved before metrics were kept with them).";
    return table;
}

std::string StepperText(size_t index, size_t count, int epoch, const std::optional<TrainingRunRecord>& record) {
    std::string text = std::to_string(index + 1) + " of " + std::to_string(count) + " snapshot" + (count == 1 ? "" : "s");
    if (const TrainingEpochRecord* e = EpochOf(record, epoch); e && e->val_metric)
        text += kDot + std::string("validation ") + (record->regression ? "MAE " : "") +
                MetricText(*e->val_metric, record->regression);
    return text;
}

}  // namespace cyxwiz::training_snapshots
