#pragma once

// Remote training screens (TOFIX118 P4 GUI): the "Will this job fit on the
// node?" card in Server Connection, and the failure / resume / progress /
// checkpoint facts in the P2P Training panel. Pure data: no ImGui, no
// network or backend calls. The fit verdict uses the node's own admission
// rule (job_admission.h), so the Engine and the node always agree.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

enum class RemoteStatusKind { Ok, Failed, Pending, None };

// ---- Will this job fit? ------------------------------------------------------

enum class JobEstimateState { NotStarted, Running, Ready, Failed };

struct JobFitInputs {
    JobEstimateState estimate = JobEstimateState::NotStarted;
    std::uint64_t training_bytes = 0;    // ProbeGraphJobMemory result
    std::string estimate_backend;        // e.g. arrayfire_cuda
    std::string estimate_device;         // e.g. GTX 1050 Ti
    std::string estimate_error;          // why the estimate could not run
    std::string node_device;             // e.g. GTX 1050 Ti ("" when unknown)
    std::uint64_t node_device_bytes = 0; // total device memory; 0 = not reported
    std::string rejection_reason;        // the node's words when it refused ("" none)
};

struct JobFitCard {
    std::string status;
    RemoteStatusKind kind = RemoteStatusKind::None;
    std::string note;
    bool has_bars = false;
    std::string needs;      // "about 1.04 GB (training 0.54 GB + runtime)"
    std::string device_label;
    std::string has;        // "4.00 GB (3.80 GB usable)"
    double fill = 0.0;      // need / usable, clamped to [0, 1]
    bool has_reason = false;
    std::string reason;
    std::string next;
    bool start_enabled = true;
    std::string start_note;
    std::uint64_t estimated_memory = 0;  // what to send as JobConfig.estimated_memory
    std::vector<std::pair<std::string, std::string>> details;
};

JobFitCard BuildJobFitCard(const JobFitInputs& inputs);

// ---- P2P Training panel ------------------------------------------------------

// The node's category for a failure ("OUT_OF_MEMORY", ...) or a rejection
// text ("Out of memory: ..."): label, and what to do next.
struct RemoteFailureCard {
    std::string title;    // "Out of memory"
    std::string message;  // the node's words, without the label prefix
    std::string next;     // what to do next
    std::string when;     // "at batch 17,920" / "before starting"
};

RemoteFailureCard BuildFailureCardFromCode(const std::string& wire_code, const std::string& message,
                                           long long at_batch);
RemoteFailureCard BuildFailureCardFromRejection(const std::string& rejection_reason);

struct RemoteCheckpointRow {
    int epoch = 0;
    long long next_batch = -1;   // -1: epoch end
    long long total_batches = 0;
    std::string received;        // "06:29:11"
    double loss = -1.0;          // loss when it arrived; < 0 unknown
    std::string hash;
};

struct RemoteCheckpointView {
    std::string epoch;
    std::string continues_at;  // "17,500 of 19,426" / "epoch end"
    std::string received;
    std::string loss;          // "2.434" / "-"
    std::vector<std::pair<std::string, std::string>> details;
};

RemoteCheckpointView BuildCheckpointView(const RemoteCheckpointRow& row);

// "Last checkpoint: epoch 1, batch 17,500 of 19,426 (90%), saved on the node at 06:29:11."
std::string ResumeSummary(const RemoteCheckpointRow& last);

struct RemoteProgressInputs {
    int epoch = 0, total_epochs = 0;
    long long batch = 0, total_batches = 0;
    long long eta_seconds = -1, elapsed_seconds = -1;
    double samples_per_second = 0.0;
    int batch_size = 0;
};

struct RemoteProgressView {
    std::string progress;  // "Epoch 1/1 - batch 17,920 of 19,426"
    std::string eta;       // "about 31 min" / "-"
    std::string elapsed;   // "6 h 25 min" / "-"
    std::string speed;     // "0.8 batches/s (6.4 samples/s)" / "-"
};

RemoteProgressView BuildProgressView(const RemoteProgressInputs& inputs);

// Helpers shared by the views.
std::string FormatRemoteDuration(long long seconds);  // "31 min", "6 h 25 min", "45 s"
std::string FormatRemoteGigabytes(std::uint64_t bytes);  // "1.04 GB"
std::string FormatRemoteCount(long long value);          // "19,426"

}  // namespace cyxwiz
