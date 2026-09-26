#pragma once

// The standard training benchmark (TOFIX118 P3): a fixed decoder language
// model (vocabulary 8192, width 128, 4 blocks, 256-token windows, batch 8)
// trained on seeded synthetic token windows through the shared training core
// (RunGraphTrainingJob), on the currently selected route. It measures what a
// training job gets: steady-state step time and tokens per second, with the
// device's CPU fallbacks. Nodes report it as their measured capability.

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

inline constexpr const char* kTrainingBenchmarkId = "cyxwiz-causal-lm-train-v1";

struct TrainingBenchmarkOptions {
    int warmup_steps = 5;      // excluded: kernel compilation, first allocations
    int measured_steps = 20;
    // Scratch space for the synthetic dataset and checkpoints (removed after).
    std::filesystem::path work_dir;
    std::function<bool()> should_cancel;
};

struct TrainingBenchmarkResult {
    bool ok = false;
    std::string error;
    std::string benchmark_id = kTrainingBenchmarkId;
    std::string build;                 // CyxWiz version that measured it
    std::string measured_at;           // UTC, ISO 8601
    // The route that ran it.
    std::string backend;               // e.g. arrayfire_opencl
    int device_id = 0;
    std::string device_name;
    std::string physical_fingerprint;  // the verified route's identity
    // Measurement.
    int batch_size = 0;
    int tokens_per_step = 0;
    int steps_measured = 0;
    double step_ms_median = 0.0;
    double step_ms_p10 = 0.0;
    double step_ms_p90 = 0.0;
    double tokens_per_second = 0.0;    // at the median step
    double wall_seconds = 0.0;         // including setup
    long long native_cpu_fallbacks = 0;
    float first_loss = 0.0f;
    float last_loss = 0.0f;
};

// Runs the benchmark on the selected route (CommitExecutionDeviceSelectionState).
TrainingBenchmarkResult RunTrainingBenchmark(const TrainingBenchmarkOptions& options);

// training-benchmark.json next to route-qualification.json: one result per
// benchmarked route, failures included (ok = false with the reason: a
// verified route that cannot train is worth reporting). A result is stale
// when its build or the route's fingerprint no longer matches.
std::filesystem::path GetTrainingBenchmarkCachePath();
bool SaveTrainingBenchmarkResults(const std::filesystem::path& path,
                                  const std::vector<TrainingBenchmarkResult>& results, std::string& error);
bool LoadTrainingBenchmarkResults(const std::filesystem::path& path,
                                  std::vector<TrainingBenchmarkResult>& results, std::string& error);

}  // namespace cyxwiz
