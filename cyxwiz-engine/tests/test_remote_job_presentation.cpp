// Remote training screens (TOFIX118 P4 GUI): fit card, failure card,
// checkpoints, resume and progress, on this machine's real numbers - the
// Berean T7 probe on the GTX 1050 Ti (557.6 MB) and the Dell full-data run.
#include "../src/core/job_admission.h"
#include "../src/core/remote_job_presentation.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

constexpr std::uint64_t kBereanProbeBytes = 584689664;  // 557.6 MB, arrayfire_cuda, 2026-09-29
constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;

cyxwiz::JobFitInputs Ready(std::uint64_t node_bytes, const std::string& node_device) {
    cyxwiz::JobFitInputs inputs;
    inputs.estimate = cyxwiz::JobEstimateState::Ready;
    inputs.training_bytes = kBereanProbeBytes;
    inputs.estimate_backend = "arrayfire_cuda";
    inputs.estimate_device = "GTX 1050 Ti";
    inputs.node_device = node_device;
    inputs.node_device_bytes = node_bytes;
    return inputs;
}

std::string Detail(const cyxwiz::JobFitCard& card, const std::string& key) {
    for (const auto& [k, v] : card.details) {
        if (k == key) return v;
    }
    return {};
}

void CheckFitCard() {
    const auto fits = cyxwiz::BuildJobFitCard(Ready(4 * kGiB, "GTX 1050 Ti"));
    Check(fits.status == "Fits" && fits.kind == cyxwiz::RemoteStatusKind::Ok && fits.start_enabled,
          "Berean T7 fits a 4 GB GTX 1050 Ti");
    Check(fits.needs == "about 1.04 GB (training 0.54 GB + runtime)", "need text: " + fits.needs);
    Check(fits.has == "4.00 GB (3.80 GB usable)" && fits.device_label == "Node device (GTX 1050 Ti)",
          "device text: " + fits.has);
    Check(std::abs(fits.fill - 0.2749) < 0.001, "bar fill is need / usable");
    Check(fits.estimated_memory == kBereanProbeBytes, "the measured bytes are what gets sent");
    Check(Detail(fits, "Training memory") == "557.6 MB" && Detail(fits, "Measured on") == "GTX 1050 Ti (arrayfire_cuda)",
          "details name the measurement");

    const auto small = cyxwiz::BuildJobFitCard(Ready(1 * kGiB, ""));
    Check(small.status == "Won't fit" && !small.start_enabled && small.has_reason, "a 1 GB device does not fit");
    Check(small.reason == "The job needs about 1.04 GB; the node has 0.95 GB usable.", "reason: " + small.reason);
    Check(small.fill == 1.0, "a job that does not fit fills the bar");

    // Same verdict as the node's own admission rule.
    cyxwiz::AdmissionFacts facts;
    facts.route = "cuda:0";
    facts.route_verified = true;
    facts.device_memory = 1 * kGiB;
    facts.job_memory = kBereanProbeBytes;
    Check(!cyxwiz::EvaluateJobAdmission(facts).accepted, "the node would refuse it too");
    facts.device_memory = 4 * kGiB;
    Check(cyxwiz::EvaluateJobAdmission(facts).accepted, "and accept it on 4 GB");

    const auto unknown = cyxwiz::BuildJobFitCard(Ready(0, ""));
    Check(unknown.status == "Unknown" && unknown.start_enabled && unknown.has == "not reported",
          "unknown node memory still lets the node decide");

    cyxwiz::JobFitInputs running;
    running.estimate = cyxwiz::JobEstimateState::Running;
    const auto estimating = cyxwiz::BuildJobFitCard(running);
    Check(estimating.status == "Estimating..." && estimating.start_enabled && !estimating.has_bars,
          "Start stays available while estimating");

    auto refused_inputs = Ready(1 * kGiB, "");
    refused_inputs.rejection_reason =
        "Out of memory: the job needs about 1.0 GB of device memory (training 0.5 GB + runtime); cuda:0 has 1.0 GB";
    const auto refused = cyxwiz::BuildJobFitCard(refused_inputs);
    Check(refused.status == "Refused: Out of memory" && !refused.start_enabled && refused.reason == refused_inputs.rejection_reason,
          "a node refusal keeps the node's words: " + refused.status);
    Check(refused.next.rfind("Resume on a node with more device memory", 0) == 0, "and says what to do next");
}

void CheckFailureAndProgress() {
    const auto oom = cyxwiz::BuildFailureCardFromCode("OUT_OF_MEMORY", "The node ran out of device memory.", 17920);
    Check(oom.title == "Out of memory" && oom.when == "at batch 17,920", "failure card from the node's code");
    const auto older = cyxwiz::BuildFailureCardFromCode("TRAINING_FAILED", "x", 0);
    Check(older.title == "Training failed" && older.when.empty(), "an older node's code reads as a plain failure");
    const auto device = cyxwiz::BuildFailureCardFromRejection(
        "Device problem: this node's compute route arrayfire_cuda:0 is not verified");
    Check(device.title == "Device problem" && device.when == "before starting" &&
              device.message == "this node's compute route arrayfire_cuda:0 is not verified",
          "rejection text splits into label and message");
    Check(cyxwiz::BuildFailureCardFromRejection("Node at capacity").title == "Refused by the node",
          "an unlabelled refusal still reads plainly");

    cyxwiz::RemoteCheckpointRow row;
    row.epoch = 1;
    row.next_batch = 17500;
    row.total_batches = 19426;
    row.received = "06:29:11";
    row.loss = 2.4343;
    row.hash = "abc123";
    const auto view = cyxwiz::BuildCheckpointView(row);
    Check(view.continues_at == "17,500 of 19,426" && view.loss == "2.434" && view.received == "06:29:11",
          "checkpoint row");
    Check(cyxwiz::ResumeSummary(row) ==
              "Last checkpoint: epoch 1, batch 17,500 of 19,426 (90%), saved on the node at 06:29:11. "
              "Resuming sends the same job again; the node continues from here.",
          "resume summary: " + cyxwiz::ResumeSummary(row));
    row.next_batch = -1;
    Check(cyxwiz::BuildCheckpointView(row).continues_at == "epoch end", "epoch-end checkpoint");

    cyxwiz::RemoteProgressInputs progress;
    progress.epoch = 1;
    progress.total_epochs = 1;
    progress.batch = 17920;
    progress.total_batches = 19426;
    progress.eta_seconds = 1882;
    progress.elapsed_seconds = 23146;
    progress.samples_per_second = 6.4;
    progress.batch_size = 8;
    const auto p = cyxwiz::BuildProgressView(progress);
    Check(p.progress == "Epoch 1/1 - batch 17,920 of 19,426", "progress: " + p.progress);
    Check(p.eta == "about 31 min" && p.elapsed == "6 h 25 min", "time: " + p.eta + " / " + p.elapsed);
    Check(p.speed == "0.8 batches/s (6.4 samples/s)", "speed: " + p.speed);
    progress.eta_seconds = -1;
    progress.samples_per_second = 0.0;
    const auto unknown = cyxwiz::BuildProgressView(progress);
    Check(unknown.eta == "-" && unknown.speed == "-", "missing node facts show a dash, not a guess");
}

}  // namespace

int main() {
    CheckFitCard();
    CheckFailureAndProgress();
    std::cout << "Remote job presentation passed\n";
    return 0;
}
