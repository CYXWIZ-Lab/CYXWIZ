// tofix94: the crash heartbeat is replaced atomically, so a reader polling
// LoadLastRun() while heartbeats are written never sees an empty or partial
// document.

#include "core/crash_run_recorder.h"
#include "core/debug_run_paths.h"
#include "core/graph_compiler.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

using namespace cyxwiz;

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("cyxwiz_heartbeat_atomic_" + std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    ScopedDebugRunRootOverrideForTesting root_override(root);

    TrainingConfiguration config;
    config.dataset_name = "heartbeat_contract";
    auto& recorder = CrashRunRecorder::Instance();
    recorder.StartTrainingRun(config, 3, 32, 1000);

    if (!CrashRunRecorder::LoadLastRun()) {
        std::cerr << "FAIL: first heartbeat was not readable\n";
        return EXIT_FAILURE;
    }

    std::atomic<bool> writing{true};
    std::thread writer([&]() {
        // Panel events grow the document so a torn write would be visible.
        for (int i = 0; i < 1500; ++i) {
            recorder.MarkPanelEvent("TrainingPlotPanel.WriteBatchProgress",
                                    "batch " + std::to_string(i) +
                                        " of a long detail string for the heartbeat");
            recorder.MarkStage(TrainingTraceStage::Forward, 1, i, 1500, 0.5f, 0.75f);
            // Per-batch marks write the file every 200 ms: about 1.5 s of
            // marks so the file is replaced several times while it is read.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        writing = false;
    });

    int reads = 0;
    int failures = 0;
    while (writing.load()) {
        auto summary = CrashRunRecorder::LoadLastRun();
        ++reads;
        if (!summary || summary->dataset_name != "heartbeat_contract") {
            ++failures;
        }
    }
    writer.join();

    // Per-batch marks do not each replace the file (TOFIX178 0.6): 4000 of
    // them take far less than 4000 atomic writes would, and the run's end
    // writes the latest state (the last batch).
    const auto burst_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 4000; ++i) recorder.MarkStage(TrainingTraceStage::Backward, 2, i, 4000, 0.25f, 0.5f);
    const auto burst_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - burst_start).count();
    recorder.MarkCompleted();
    const auto finished = CrashRunRecorder::LoadLastRun();
    if (burst_ms > 1500) {
        std::cerr << "FAIL: 4000 stage marks took " << burst_ms << " ms (each one writes the file?)\n";
        return EXIT_FAILURE;
    }
    if (!finished || finished->batch != 3999 || finished->status != "completed") {
        std::cerr << "FAIL: the run's end did not write the latest batch\n";
        return EXIT_FAILURE;
    }

    std::error_code ignored;
    std::filesystem::remove_all(root, ignored);

    if (failures != 0) {
        std::cerr << "FAIL: " << failures << " of " << reads
                  << " heartbeat reads were empty or malformed\n";
        return EXIT_FAILURE;
    }
    std::cout << "crash heartbeat atomic contract passed (" << reads << " reads)\n";
    return EXIT_SUCCESS;
}
