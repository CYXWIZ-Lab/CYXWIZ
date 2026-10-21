// Standard training benchmark (TOFIX118 P3): the fixed causal-LM benchmark
// trains on its synthetic windows through the shared core on the selected
// route, reports a steady-state measurement, and its cache round-trips.
#include "../src/core/compute_runtime_paths.h"
#include "../src/core/training_benchmark.h"
#include "route_qualification_test_fixture.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

}  // namespace

int main() {
    cyxwiz::test::InstallQualifiedRouteSnapshot();
    const fs::path root = fs::temp_directory_path() / "cyxwiz_test_training_benchmark";
    std::error_code ec;
    fs::remove_all(root, ec);
    cyxwiz::ScopedComputeRuntimeRootOverrideForTesting runtime_root(root / "runtime");

    std::cout << "benchmark run\n";
    cyxwiz::TrainingBenchmarkOptions options;
    options.warmup_steps = 1;
    options.measured_steps = 3;
    options.work_dir = root / "work";
    const auto result = cyxwiz::RunTrainingBenchmark(options);
    if (!result.ok) std::cout << "  error: " << result.error << "\n";
    Check(result.ok, "the benchmark trains on its synthetic windows");
    Check(result.benchmark_id == cyxwiz::kTrainingBenchmarkId, "result names the benchmark");
    Check(result.batch_size == 8 && result.tokens_per_step == 8 * 256, "batch 8 x 256 tokens per step");
    Check(result.steps_measured == 3, "warm-up steps are excluded from the measurement");
    Check(result.step_ms_median > 0.0 && result.step_ms_p10 <= result.step_ms_median &&
              result.step_ms_median <= result.step_ms_p90,
          "step time percentiles are ordered");
    Check(std::abs(result.tokens_per_second - result.tokens_per_step * 1000.0 / result.step_ms_median) < 1e-6,
          "tokens per second follow the median step");
    Check(!result.backend.empty() && !result.build.empty() && !result.measured_at.empty(),
          "result records the route, build and time");
    // Uniform random tokens over 8192 ids: the loss starts near ln(8192).
    Check(std::abs(result.first_loss - std::log(8192.0f)) < 1.0f, "first loss is near ln(vocabulary)");
    Check(!fs::exists(options.work_dir), "scratch data is removed");

    std::cout << "cancellation\n";
    cyxwiz::TrainingBenchmarkOptions cancelled = options;
    cancelled.should_cancel = [] { return true; };
    const auto stopped = cyxwiz::RunTrainingBenchmark(cancelled);
    Check(!stopped.ok && stopped.error.find("cancelled") != std::string::npos, "a cancelled benchmark reports no result");

    std::cout << "cache\n";
    std::string error;
    const auto path = cyxwiz::GetTrainingBenchmarkCachePath();
    Check(path.parent_path() == cyxwiz::GetComputeRuntimeRoot(), "cache sits next to the route evidence");
    Check(cyxwiz::SaveTrainingBenchmarkResults(path, {result, stopped}, error), "results save");
    std::vector<cyxwiz::TrainingBenchmarkResult> loaded;
    Check(cyxwiz::LoadTrainingBenchmarkResults(path, loaded, error), "results load");
    Check(loaded.size() == 2 && loaded[0].ok && !loaded[1].ok && loaded[1].error == stopped.error,
          "failed routes are kept with their reason");
    if (!loaded.empty()) {
        Check(loaded[0].tokens_per_second == result.tokens_per_second &&
                  loaded[0].physical_fingerprint == result.physical_fingerprint &&
                  loaded[0].backend == result.backend && loaded[0].build == result.build,
              "a loaded result matches the saved one");
    }
    Check(!cyxwiz::LoadTrainingBenchmarkResults(root / "missing.json", loaded, error), "a missing cache is reported");
    {
        std::ofstream(root / "early.json")
            << R"({"schema":1,"benchmark_id":"cyxwiz-causal-lm-train-v1","routes":[{"backend":"arrayfire_cuda","tokens_per_second":20000.0}]})";
        Check(cyxwiz::LoadTrainingBenchmarkResults(root / "early.json", loaded, error) && loaded.size() == 1 &&
                  loaded[0].ok,
              "an early cache without ok flags holds successes");
    }

    fs::remove_all(root, ec);
    std::cout << (g_failures == 0 ? "PASS" : "FAILED") << " (" << g_failures << " failures)\n";
    return g_failures == 0 ? 0 : 1;
}
