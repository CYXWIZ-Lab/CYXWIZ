#pragma once

#include "test_executor.h"
#include "graph_compiler.h"
#include "data_registry.h"
#include "async_task_manager.h"
#include <atomic>
#include <mutex>
#include <memory>
#include <functional>
#include <string>
#include <thread>

namespace cyxwiz {

/**
 * TestManager - Centralized manager for model testing sessions
 *
 * This singleton ensures only one testing session runs at a time.
 * Tests the trained model on test data and computes metrics.
 * Integrates with the async task system to show testing progress.
 */
class TestManager {
public:
    // Singleton access
    static TestManager& Instance();

    // Check if testing is currently active
    bool IsTestingActive() const { return is_testing_.load(); }

    // Get the current testing task ID (0 if none)
    uint64_t GetCurrentTaskId() const { return current_task_id_.load(); }

    // Get current testing metrics (thread-safe)
    TestingMetrics GetCurrentMetrics() const;

    // Get results from last completed test
    const TestingMetrics& GetLastResults() const { return last_results_; }

    // Check if we have results from a completed test
    bool HasResults() const { return has_results_; }

    /**
     * Start testing with compiled graph configuration
     * @param config Compiled training configuration from GraphCompiler
     * @param dataset Dataset handle
     * @param batch_size Batch size for inference
     * @param model Optional pre-trained model (if nullptr, builds from config)
     * @param on_complete Callback when testing completes
     * @return true if testing started, false if already testing
     */
    bool StartTesting(
        TrainingConfiguration config,
        DatasetHandle dataset,
        int batch_size,
        std::shared_ptr<SequentialModel> model = nullptr,
        TestCompleteCallback on_complete = nullptr
    );

    /**
     * Stop the current testing session
     */
    void StopTesting();

    /**
     * Export results to CSV file
     * @param filepath Path to output file
     * @return true on success
     */
    bool ExportResultsToCSV(const std::string& filepath);

    /**
     * Export results to JSON file
     * @param filepath Path to output file
     * @return true on success
     */
    bool ExportResultsToJSON(const std::string& filepath);

    // Callbacks for testing events
    using TestingStartCallback = std::function<void(const std::string& description)>;
    using TestingEndCallback = std::function<void(bool success, const TestingMetrics& metrics)>;
    using ProgressCallback = std::function<void(int batch, int total, float accuracy)>;

    void SetOnTestingStart(TestingStartCallback callback) { on_testing_start_ = callback; }
    void SetOnTestingEnd(TestingEndCallback callback) { on_testing_end_ = callback; }
    void SetOnProgress(ProgressCallback callback) { on_progress_ = callback; }

private:
    TestManager() = default;
    ~TestManager();

    TestManager(const TestManager&) = delete;
    TestManager& operator=(const TestManager&) = delete;

    void TestingThreadFunc(
        std::unique_ptr<TestExecutor> executor,
        int batch_size,
        TestCompleteCallback on_complete
    );

    std::atomic<bool> is_testing_{false};
    std::atomic<bool> stop_requested_{false};
    std::atomic<uint64_t> current_task_id_{0};

    mutable std::mutex mutex_;
    std::unique_ptr<TestExecutor> current_executor_;
    std::unique_ptr<std::thread> testing_thread_;

    // Cached metrics for thread-safe access
    mutable std::mutex metrics_mutex_;
    TestingMetrics cached_metrics_;

    // Last completed test results
    TestingMetrics last_results_;
    bool has_results_ = false;

    // Callbacks
    TestingStartCallback on_testing_start_;
    TestingEndCallback on_testing_end_;
    ProgressCallback on_progress_;
};

} // namespace cyxwiz
