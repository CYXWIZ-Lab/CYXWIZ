#include "test_manager.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace cyxwiz {

TestManager& TestManager::Instance() {
    static TestManager instance;
    return instance;
}

TestManager::~TestManager() {
    StopTesting();
    if (testing_thread_ && testing_thread_->joinable()) {
        testing_thread_->join();
    }
}

TestingMetrics TestManager::GetCurrentMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return cached_metrics_;
}

bool TestManager::StartTesting(
    TrainingConfiguration config,
    DatasetHandle dataset,
    int batch_size,
    std::shared_ptr<SequentialModel> model,
    TestCompleteCallback on_complete)
{
    // Check if already testing
    if (is_testing_.load()) {
        spdlog::warn("TestManager: Cannot start testing - already testing");
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Double-check after acquiring lock
    if (is_testing_.load()) {
        return false;
    }

    // Create executor
    auto executor = std::make_unique<TestExecutor>(std::move(config), dataset);

    // Set model if provided
    if (model) {
        executor->SetModel(model);
    }

    // Set state
    is_testing_.store(true);
    stop_requested_.store(false);

    // Create async task for visibility in Tasks panel
    std::string task_name = "Testing Model";
    current_task_id_.store(AsyncTaskManager::Instance().RunAsync(
        task_name,
        [](LambdaTask& task) {
            // This task just tracks the testing - actual work is in testing thread
            while (!task.ShouldStop()) {
                auto& mgr = TestManager::Instance();
                if (!mgr.IsTestingActive()) {
                    break;
                }
                auto metrics = mgr.GetCurrentMetrics();
                float progress = metrics.total_batches > 0 ?
                    static_cast<float>(metrics.current_batch) / metrics.total_batches : 0.0f;
                task.ReportProgress(progress,
                    "Batch " + std::to_string(metrics.current_batch) + "/" +
                    std::to_string(metrics.total_batches) +
                    " - Acc: " + std::to_string(static_cast<int>(metrics.test_accuracy * 100)) + "%");
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            task.MarkCompleted();
        },
        nullptr,  // progress callback
        nullptr   // completion callback
    ));

    // Notify start callback
    if (on_testing_start_) {
        on_testing_start_("Testing Model");
    }

    // Wait for previous thread if any
    if (testing_thread_ && testing_thread_->joinable()) {
        testing_thread_->join();
    }

    // Start testing thread
    testing_thread_ = std::make_unique<std::thread>(
        &TestManager::TestingThreadFunc, this,
        std::move(executor), batch_size, on_complete
    );

    spdlog::info("TestManager: Started testing (batch_size={})", batch_size);
    return true;
}

void TestManager::StopTesting() {
    if (!is_testing_.load()) {
        return;
    }

    spdlog::info("TestManager: Stopping testing...");
    stop_requested_.store(true);

    // Stop the executor
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_executor_) {
            current_executor_->Stop();
        }
    }

    // Cancel the async task
    uint64_t task_id = current_task_id_.load();
    if (task_id != 0) {
        AsyncTaskManager::Instance().Cancel(task_id);
    }
}

void TestManager::TestingThreadFunc(
    std::unique_ptr<TestExecutor> executor,
    int batch_size,
    TestCompleteCallback on_complete)
{
    spdlog::info("TestManager: Testing thread started");

    // Store executor reference
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_executor_ = std::move(executor);
    }

    // Initialize cached metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        cached_metrics_ = TestingMetrics();
        cached_metrics_.is_testing = true;
    }

    TestingMetrics final_metrics;

    // Check if stop was requested before testing started
    if (stop_requested_.load()) {
        spdlog::info("TestManager: Testing cancelled before start");
    } else {
        // Run testing
        TestExecutor* exec = nullptr;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            exec = current_executor_.get();
        }

        if (exec) {
            // Set up batch callback for progress updates
            auto batch_callback = [this](int batch, int total, float accuracy) {
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex_);
                    cached_metrics_.current_batch = batch;
                    cached_metrics_.total_batches = total;
                    cached_metrics_.test_accuracy = accuracy;
                }

                if (on_progress_) {
                    on_progress_(batch, total, accuracy);
                }
            };

            exec->Test(
                batch_size,
                batch_callback,
                [this, &final_metrics](const TestingMetrics& metrics) {
                    final_metrics = metrics;
                }
            );
        }
    }

    // Get final metrics
    {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        if (final_metrics.is_complete) {
            cached_metrics_ = final_metrics;
        }
        cached_metrics_.is_testing = false;
        cached_metrics_.is_complete = !stop_requested_.load();
    }

    // Store last results
    bool success = !stop_requested_.load() && final_metrics.is_complete;
    if (success) {
        last_results_ = final_metrics;
        has_results_ = true;
    }

    // Cleanup
    is_testing_.store(false);
    current_task_id_.store(0);

    // Notify end callback
    if (on_testing_end_) {
        on_testing_end_(success, final_metrics);
    }

    // Notify completion callback
    if (on_complete && success) {
        on_complete(final_metrics);
    }

    // Clear current executor
    {
        std::lock_guard<std::mutex> lock(mutex_);
        current_executor_.reset();
    }

    if (success) {
        spdlog::info("TestManager: Testing completed! Accuracy: {:.2f}%",
                     final_metrics.test_accuracy * 100);
    } else {
        spdlog::info("TestManager: Testing stopped");
    }
}

bool TestManager::ExportResultsToCSV(const std::string& filepath) {
    if (!has_results_) {
        spdlog::warn("TestManager: No results to export");
        return false;
    }

    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("TestManager: Failed to open file for export: {}", filepath);
        return false;
    }

    // Write overview
    file << "Test Results Summary\n";
    file << "Metric,Value\n";
    file << "Accuracy," << std::fixed << std::setprecision(4) << last_results_.test_accuracy << "\n";
    file << "Loss," << last_results_.test_loss << "\n";
    file << "Total Samples," << last_results_.total_samples << "\n";
    file << "Macro Precision," << last_results_.macro_precision << "\n";
    file << "Macro Recall," << last_results_.macro_recall << "\n";
    file << "Macro F1," << last_results_.macro_f1 << "\n";
    file << "Weighted F1," << last_results_.weighted_f1 << "\n";
    file << "Time (seconds)," << last_results_.total_time_seconds << "\n";
    file << "Samples/sec," << last_results_.samples_per_second << "\n";

    file << "\nPer-Class Metrics\n";
    file << "Class,Precision,Recall,F1,Support\n";
    for (const auto& cm : last_results_.per_class_metrics) {
        file << cm.class_name << ","
             << cm.precision << ","
             << cm.recall << ","
             << cm.f1_score << ","
             << cm.support << "\n";
    }

    file << "\nConfusion Matrix\n";
    const auto& conf = last_results_.confusion_matrix;
    file << ",";
    for (int j = 0; j < conf.num_classes; ++j) {
        file << "Pred_" << j;
        if (j < conf.num_classes - 1) file << ",";
    }
    file << "\n";
    for (int i = 0; i < conf.num_classes; ++i) {
        file << "Actual_" << i << ",";
        for (int j = 0; j < conf.num_classes; ++j) {
            file << conf.matrix[i][j];
            if (j < conf.num_classes - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
    spdlog::info("TestManager: Exported results to CSV: {}", filepath);
    return true;
}

bool TestManager::ExportResultsToJSON(const std::string& filepath) {
    if (!has_results_) {
        spdlog::warn("TestManager: No results to export");
        return false;
    }

    nlohmann::json j;

    // Overview
    j["accuracy"] = last_results_.test_accuracy;
    j["loss"] = last_results_.test_loss;
    j["total_samples"] = last_results_.total_samples;
    j["macro_precision"] = last_results_.macro_precision;
    j["macro_recall"] = last_results_.macro_recall;
    j["macro_f1"] = last_results_.macro_f1;
    j["weighted_f1"] = last_results_.weighted_f1;
    j["time_seconds"] = last_results_.total_time_seconds;
    j["samples_per_second"] = last_results_.samples_per_second;

    // Per-class metrics
    nlohmann::json per_class = nlohmann::json::array();
    for (const auto& cm : last_results_.per_class_metrics) {
        nlohmann::json c;
        c["class_id"] = cm.class_id;
        c["class_name"] = cm.class_name;
        c["precision"] = cm.precision;
        c["recall"] = cm.recall;
        c["f1_score"] = cm.f1_score;
        c["support"] = cm.support;
        c["true_positives"] = cm.true_positives;
        c["false_positives"] = cm.false_positives;
        c["false_negatives"] = cm.false_negatives;
        per_class.push_back(c);
    }
    j["per_class_metrics"] = per_class;

    // Confusion matrix
    nlohmann::json conf = nlohmann::json::array();
    for (const auto& row : last_results_.confusion_matrix.matrix) {
        conf.push_back(row);
    }
    j["confusion_matrix"] = conf;

    std::ofstream file(filepath);
    if (!file.is_open()) {
        spdlog::error("TestManager: Failed to open file for export: {}", filepath);
        return false;
    }

    file << j.dump(2);
    file.close();
    spdlog::info("TestManager: Exported results to JSON: {}", filepath);
    return true;
}

} // namespace cyxwiz
