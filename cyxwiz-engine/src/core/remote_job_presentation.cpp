#include "remote_job_presentation.h"

#include "job_admission.h"
#include "training_failure.h"

#include <algorithm>
#include <cstdio>

namespace cyxwiz {

namespace {

constexpr const char* kWireCodes[] = {"REFUSED", "DATA_ERROR", "DEVICE_ERROR", "OUT_OF_MEMORY",
                                      "CANCELLED", "RESERVATION_ENDED", "INTERNAL"};

std::string NextStepFor(const std::string& wire_code) {
    if (wire_code == "OUT_OF_MEMORY") {
        return "Resume on a node with more device memory, or lower the batch size and start over.";
    }
    if (wire_code == "DATA_ERROR") return "Fix the data or the graph's data inputs, then start over.";
    if (wire_code == "DEVICE_ERROR") {
        return "This node's device could not run the job: resume on another node.";
    }
    if (wire_code == "REFUSED") return "The node cannot train this graph as it is: check the reason, then change the graph.";
    if (wire_code == "CANCELLED") return "Resume from the last checkpoint, or start over.";
    if (wire_code == "RESERVATION_ENDED") {
        return "Reserve a node again, then resume from the last checkpoint.";
    }
    return "Try again; if it happens again, keep the node's log for a report.";
}

}  // namespace

std::string FormatRemoteCount(long long value) {
    const bool negative = value < 0;
    const std::string digits = std::to_string(negative ? -value : value);
    std::string out;
    const size_t lead = digits.size() % 3;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i != 0 && i >= lead && (i - lead) % 3 == 0) out += ',';
        out += digits[i];
    }
    return negative ? "-" + out : out;
}

std::string FormatRemoteGigabytes(std::uint64_t bytes) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.2f GB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buffer;
}

std::string FormatRemoteDuration(long long seconds) {
    if (seconds < 0) return "-";
    if (seconds < 60) return std::to_string(seconds) + " s";
    const long long minutes = (seconds + 30) / 60;
    if (minutes < 60) return std::to_string(minutes) + " min";
    const long long whole_minutes = seconds / 60;
    return std::to_string(whole_minutes / 60) + " h " + std::to_string(whole_minutes % 60) + " min";
}

JobFitCard BuildJobFitCard(const JobFitInputs& inputs) {
    JobFitCard card;
    card.device_label = inputs.node_device.empty() ? "Node device" : "Node device (" + inputs.node_device + ")";

    if (!inputs.rejection_reason.empty()) {
        const auto failure = BuildFailureCardFromRejection(inputs.rejection_reason);
        card.status = "Refused: " + failure.title;
        card.kind = RemoteStatusKind::Failed;
        card.note = "The node refused the job before starting. Nothing was trained.";
        card.has_reason = true;
        card.reason = inputs.rejection_reason;
        card.next = failure.next;
        card.start_enabled = false;
        card.start_note = "Start Training is off until the job or the node changes.";
    }

    switch (inputs.estimate) {
        case JobEstimateState::NotStarted:
            if (card.status.empty()) {
                card.status = "Not estimated";
                card.note = "Connect to the node to measure the job. The node still checks before it starts.";
            }
            break;
        case JobEstimateState::Running:
            if (card.status.empty()) {
                card.status = "Estimating...";
                card.kind = RemoteStatusKind::Pending;
                card.note = "Running one training step on this machine to measure the job. Takes a few seconds.";
                card.start_note = "You can start now; the node still checks.";
            }
            break;
        case JobEstimateState::Failed:
            if (card.status.empty()) {
                card.status = "Not estimated";
                card.note = "The job could not be measured here (" + inputs.estimate_error +
                            "). The node still checks before it starts.";
            }
            break;
        case JobEstimateState::Ready: {
            AdmissionFacts facts;
            facts.route = inputs.node_device.empty() ? "the node's device" : inputs.node_device;
            facts.route_verified = true;  // the node checks its own route; this is the memory check
            facts.device_memory = inputs.node_device_bytes;
            facts.job_memory = inputs.training_bytes;
            const auto decision = EvaluateJobAdmission(facts);
            const std::uint64_t need = inputs.training_bytes + kAdmissionRuntimeReserveBytes;
            card.estimated_memory = inputs.training_bytes;
            card.has_bars = true;
            card.needs = "about " + FormatRemoteGigabytes(need) + " (training " +
                         FormatRemoteGigabytes(inputs.training_bytes) + " + runtime)";
            const double usable = kAdmissionMemoryShare * static_cast<double>(inputs.node_device_bytes);
            if (inputs.node_device_bytes == 0) {
                card.has = "not reported";
                card.fill = 0.0;
            } else {
                card.has = FormatRemoteGigabytes(inputs.node_device_bytes) + " (" +
                           FormatRemoteGigabytes(static_cast<std::uint64_t>(usable)) + " usable)";
                card.fill = std::clamp(static_cast<double>(need) / usable, 0.0, 1.0);
            }
            if (!card.status.empty()) break;  // a refusal already decided the card
            if (inputs.node_device_bytes == 0) {
                card.status = "Unknown";
                card.note = "This node did not report its device memory. The estimate is still sent and the node "
                            "checks it before starting.";
            } else if (decision.accepted) {
                card.status = "Fits";
                card.kind = RemoteStatusKind::Ok;
                card.note = "The node checks again before it starts.";
            } else {
                card.status = "Won't fit";
                card.kind = RemoteStatusKind::Failed;
                card.note = "Starting would be refused by the node.";
                card.has_reason = true;
                card.reason = "The job needs about " + FormatRemoteGigabytes(need) + "; the node has " +
                              FormatRemoteGigabytes(static_cast<std::uint64_t>(usable)) + " usable.";
                card.next = NextStepFor("OUT_OF_MEMORY");
                card.start_enabled = false;
                card.start_note = "Start Training is off: the job will not fit.";
            }
            break;
        }
    }

    if (inputs.estimate == JobEstimateState::Ready || inputs.estimate == JobEstimateState::Failed) {
        card.details.emplace_back("How it was measured", "One training step on this machine");
    }
    if (!inputs.estimate_device.empty() || !inputs.estimate_backend.empty()) {
        card.details.emplace_back("Measured on", inputs.estimate_device +
                                                     (inputs.estimate_backend.empty() ? "" : " (" + inputs.estimate_backend + ")"));
    }
    if (inputs.estimate == JobEstimateState::Ready) {
        char mb[32];
        std::snprintf(mb, sizeof(mb), "%.1f MB", static_cast<double>(inputs.training_bytes) / (1024.0 * 1024.0));
        card.details.emplace_back("Training memory", mb);
        card.details.emplace_back("Runtime margin", "512 MB (node rule)");
        card.details.emplace_back("Node limit", "95% of device memory");
        card.details.emplace_back("Sent to the node as", "estimated_memory");
    }
    card.details.emplace_back("Final check", "The node compares again before it starts");
    return card;
}

RemoteFailureCard BuildFailureCardFromCode(const std::string& wire_code, const std::string& message,
                                           long long at_batch) {
    RemoteFailureCard card;
    card.title = TrainingFailureLabel(wire_code);
    card.message = message;
    card.next = NextStepFor(wire_code);
    if (at_batch > 0) card.when = "at batch " + FormatRemoteCount(at_batch);
    return card;
}

RemoteFailureCard BuildFailureCardFromRejection(const std::string& rejection_reason) {
    // The node writes "<label>: <reason>" (job_execution_service SendJob).
    for (const char* code : kWireCodes) {
        const std::string label = TrainingFailureLabel(code);
        if (rejection_reason.rfind(label + ": ", 0) == 0) {
            auto card = BuildFailureCardFromCode(code, rejection_reason.substr(label.size() + 2), 0);
            card.when = "before starting";
            return card;
        }
    }
    auto card = BuildFailureCardFromCode("", rejection_reason, 0);
    card.title = "Refused by the node";
    card.when = "before starting";
    return card;
}

RemoteCheckpointView BuildCheckpointView(const RemoteCheckpointRow& row) {
    RemoteCheckpointView view;
    view.epoch = std::to_string(row.epoch);
    view.continues_at = row.next_batch < 0
        ? "epoch end"
        : FormatRemoteCount(row.next_batch) + (row.total_batches > 0 ? " of " + FormatRemoteCount(row.total_batches) : "");
    view.received = row.received.empty() ? "-" : row.received;
    if (row.loss >= 0.0) {
        char loss[32];
        std::snprintf(loss, sizeof(loss), "%.3f", row.loss);
        view.loss = loss;
    } else {
        view.loss = "-";
    }
    if (!row.hash.empty()) view.details.emplace_back("SHA-256 of saved weights", row.hash);
    view.details.emplace_back("Kept", "on the node until the job succeeds");
    return view;
}

std::string ResumeSummary(const RemoteCheckpointRow& last) {
    std::string where = "epoch " + std::to_string(last.epoch);
    if (last.next_batch >= 0 && last.total_batches > 0) {
        const int percent = static_cast<int>(100.0 * static_cast<double>(last.next_batch) /
                                             static_cast<double>(last.total_batches));
        where += ", batch " + FormatRemoteCount(last.next_batch) + " of " + FormatRemoteCount(last.total_batches) +
                 " (" + std::to_string(percent) + "%)";
    } else if (last.next_batch < 0) {
        where += " end";
    }
    return "Last checkpoint: " + where + ", saved on the node" +
           (last.received.empty() ? "" : " at " + last.received) +
           ". Resuming sends the same job again; the node continues from here.";
}

RemoteProgressView BuildProgressView(const RemoteProgressInputs& inputs) {
    RemoteProgressView view;
    view.progress = "Epoch " + std::to_string(inputs.epoch) + "/" + std::to_string(inputs.total_epochs) +
                    " - batch " + FormatRemoteCount(inputs.batch) +
                    (inputs.total_batches > 0 ? " of " + FormatRemoteCount(inputs.total_batches) : "");
    view.eta = inputs.eta_seconds < 0 ? "-" : "about " + FormatRemoteDuration(inputs.eta_seconds);
    view.elapsed = FormatRemoteDuration(inputs.elapsed_seconds);
    if (inputs.samples_per_second > 0.0) {
        char speed[64];
        if (inputs.batch_size > 0) {
            std::snprintf(speed, sizeof(speed), "%.1f batches/s (%.1f samples/s)",
                          inputs.samples_per_second / inputs.batch_size, inputs.samples_per_second);
        } else {
            std::snprintf(speed, sizeof(speed), "%.1f samples/s", inputs.samples_per_second);
        }
        view.speed = speed;
    } else {
        view.speed = "-";
    }
    return view;
}

}  // namespace cyxwiz
