// Training throughput profile (TOFIX118 P8): trains a saved graph with the
// shared core for a fixed number of batches and reports where a training step
// spends its time - per-batch wall time after warm-up, tokens/s, the stage
// breakdown and the host syncs from the training trace.
//
// Usage: cyxwiz-training-profile <graph.cyxgraph> [--batches N] [--warmup N]
//                                [--out result.json]
// Set CYXWIZ_PROFILE_STAGE_SYNC=1 to charge device time to the stage that
// issued it (removes CPU/GPU overlap; compare both runs).
#include "../src/core/compute_runtime_paths.h"
#include "../src/core/graph_training_job.h"
#include "../src/core/route_qualification_snapshot.h"
#include "../src/core/training_trace_collector.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

double Percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1, static_cast<size_t>(p * (values.size() - 1) + 0.5));
    return values[index];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: cyxwiz-training-profile <graph.cyxgraph> [--batches N] [--warmup N] [--out file]\n";
        return 2;
    }
    const fs::path graph_path = argv[1];
    int batches = 100;
    int warmup = 10;
    fs::path out;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string flag = argv[i];
        if (flag == "--batches") batches = std::max(1, std::atoi(argv[i + 1]));
        else if (flag == "--warmup") warmup = std::max(0, std::atoi(argv[i + 1]));
        else if (flag == "--out") out = argv[i + 1];
    }

    // This machine's verified routes (Preferences > Devices > Verify), as the
    // Engine and the node load them; unverified devices are refused.
    const auto qualification =
        cyxwiz::LoadAndInstallRouteQualificationSnapshot(cyxwiz::GetRouteQualificationCachePath());
    if (!qualification.loaded) std::cerr << "route qualification: " << qualification.message << "\n";

    std::ifstream in(graph_path, std::ios::binary);
    if (!in) {
        std::cerr << "cannot open " << graph_path << "\n";
        return 2;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const json graph = json::parse(buffer.str());
    int tokens_per_sample = 0;
    for (const auto& node : graph["nodes"]) {
        const auto params = node.value("parameters", json::object());
        if (params.contains("max_sequence_length")) {
            tokens_per_sample = std::max(tokens_per_sample, std::atoi(params["max_sequence_length"].get<std::string>().c_str()));
        }
    }

    const fs::path work = fs::current_path() / "training_profile_checkpoints";
    cyxwiz::GraphTrainingJobRequest request;
    request.graph_json = buffer.str();
    request.epochs_override = 1;
    request.checkpoint_dir_override = work.string();

    int batch_size = 0;
    std::atomic<int> seen{0};
    std::vector<double> batch_ms;
    std::vector<float> losses;
    auto last = std::chrono::steady_clock::now();
    const auto start = last;
    cyxwiz::GraphTrainingJobCallbacks callbacks;
    callbacks.on_start = [&](int, int size) {
        batch_size = size;
        last = std::chrono::steady_clock::now();
    };
    callbacks.on_batch = [&](int, int, int, float loss, float) {
        const auto now = std::chrono::steady_clock::now();
        batch_ms.push_back(std::chrono::duration<double, std::milli>(now - last).count());
        last = now;
        losses.push_back(loss);
        // Stage/layer totals cover steady state only (warm-up compiles kernels).
        if (++seen == warmup) cyxwiz::TrainingTraceCollector::Instance().ResetTimings();
    };
    callbacks.should_cancel = [&] { return seen.load() >= batches; };

    std::cout << "profiling " << graph_path.filename().string() << " for " << batches << " batches\n";
    const auto result = cyxwiz::RunGraphTrainingJob(request, callbacks);
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!result.cancelled && !result.ok) {
        std::cerr << "training failed: " << result.error << "\n";
        return 1;
    }

    std::vector<double> steady(batch_ms.begin() + std::min<size_t>(warmup, batch_ms.size()), batch_ms.end());
    const double median = Percentile(steady, 0.5);
    const double tokens = static_cast<double>(batch_size) * tokens_per_sample;
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    const double steady_batches = static_cast<double>(steady.size());

    json report{{"graph", graph_path.string()},
                {"batches_measured", steady.size()},
                {"warmup_batches", warmup},
                {"batch_size", batch_size},
                {"tokens_per_sample", tokens_per_sample},
                {"stage_sync", std::getenv("CYXWIZ_PROFILE_STAGE_SYNC") != nullptr},
                {"device", trace.effective_backend + ":" + std::to_string(trace.effective_device_id) + " " +
                               trace.effective_device_name},
                {"batch_ms_median", median},
                {"batch_ms_p10", Percentile(steady, 0.1)},
                {"batch_ms_p90", Percentile(steady, 0.9)},
                {"tokens_per_second", median > 0 ? tokens * 1000.0 / median : 0.0},
                {"wall_seconds_including_setup", wall_s},
                {"native_cpu_fallbacks", trace.native_cpu_fallback_count},
                {"host_syncs_per_batch", seen ? double(trace.arrayfire_host_sync_count) / seen.load() : 0.0},
                {"host_sync_bytes_per_batch", seen ? double(trace.arrayfire_host_sync_bytes) / seen.load() : 0.0},
                {"first_loss", losses.empty() ? 0.0f : losses.front()},
                {"last_loss", losses.empty() ? 0.0f : losses.back()}};
    json stages = json::array();
    double stage_total = 0.0;
    json layer_report = json::array();
    for (const auto& t : trace.stage_timings) {
        if (t.stage.rfind("Model", 0) == 0 || t.stage.find('.') != std::string::npos) continue;  // layers and sub-spans
        stage_total += t.total_ms;
    }
    for (const auto& t : trace.stage_timings) {
        if (t.stage.rfind("Model", 0) == 0) {
            layer_report.push_back({{"layer", t.stage}, {"mean_ms", t.count ? t.total_ms / t.count : 0.0},
                                    {"max_ms", t.max_ms}});
            continue;
        }
        stages.push_back({{"stage", t.stage},
                          {"count", t.count},
                          {"mean_ms", t.count ? t.total_ms / t.count : 0.0},
                          {"max_ms", t.max_ms},
                          {"share", stage_total > 0 ? t.total_ms / stage_total : 0.0}});
    }
    report["stages"] = stages;
    report["layers"] = layer_report;
    json syncs = json::array();
    for (const auto& g : trace.arrayfire_host_sync_groups) {
        syncs.push_back({{"category", g.category}, {"operation", g.operation}, {"events", g.event_count},
                         {"bytes", g.bytes}});
    }
    report["host_sync_groups"] = syncs;

    (void)steady_batches;

    std::cout << report.dump(2) << "\n";
    if (!out.empty()) {
        std::ofstream file(out);
        file << report.dump(2) << "\n";
    }
    std::error_code ec;
    fs::remove_all(work, ec);
    return 0;
}
