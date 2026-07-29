#pragma once

#include "graph_compiler.h"
#include "dataset_batcher.h"
#include "data_registry.h"
#include "test_dataset_selection.h"
#include "text_dataset_batcher.h"
#include <cyxwiz/tensor.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/loss.h>
#include <functional>
#include <atomic>
#include <cmath>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include <chrono>

namespace cyxwiz {

class ArrowDataset;
class ParquetBackedDataset;

struct RegressionMetricAccumulator {
    double absolute_error_sum = 0.0;
    double squared_error_sum = 0.0;
    size_t value_count = 0;

    void Reset() {
        absolute_error_sum = 0.0;
        squared_error_sum = 0.0;
        value_count = 0;
    }

    void Add(const float* predictions,
             const float* targets,
             size_t count) {
        for (size_t i = 0; i < count; ++i) {
            const double error =
                static_cast<double>(predictions[i]) -
                static_cast<double>(targets[i]);
            absolute_error_sum += std::abs(error);
            squared_error_sum += error * error;
        }
        value_count += count;
    }

    float Mae() const {
        return value_count == 0
            ? 0.0f
            : static_cast<float>(
                  absolute_error_sum / static_cast<double>(value_count));
    }

    float Rmse() const {
        return value_count == 0
            ? 0.0f
            : static_cast<float>(std::sqrt(
                  squared_error_sum / static_cast<double>(value_count)));
    }
};

/**
 * Per-class metrics for detailed analysis
 */
struct ClassMetrics {
    int class_id = 0;
    std::string class_name;
    int true_positives = 0;
    int false_positives = 0;
    int false_negatives = 0;
    float precision = 0.0f;
    float recall = 0.0f;
    float f1_score = 0.0f;
    int support = 0;  // Number of actual samples in this class
};

/**
 * Confusion matrix for classification results
 */
struct ConfusionMatrix {
    int num_classes = 0;
    std::vector<std::vector<int>> matrix;  // [actual][predicted]
    std::vector<std::string> class_names;

    void Resize(int n);
    void Add(int actual, int predicted);
    float GetAccuracy() const;
    int GetTotal() const;
};

/**
 * Testing metrics computed during evaluation
 */
struct TestingMetrics {
    // Progress
    int current_batch = 0;
    int total_batches = 0;
    int total_samples = 0;
    int correct_predictions = 0;

    // Overall metrics
    float test_loss = 0.0f;
    float test_accuracy = 0.0f;
    float macro_precision = 0.0f;
    float macro_recall = 0.0f;
    float macro_f1 = 0.0f;
    float weighted_f1 = 0.0f;
    bool regression_mode = false;
    float test_mae = 0.0f;
    float test_rmse = 0.0f;
    size_t total_target_values = 0;

    // Detailed results
    ConfusionMatrix confusion_matrix;
    std::vector<ClassMetrics> per_class_metrics;

    // Raw predictions (for detailed analysis)
    std::vector<int> predictions;
    std::vector<int> ground_truth;
    std::vector<float> confidences;

    // Timing
    float total_time_seconds = 0.0f;
    float samples_per_second = 0.0f;

    // State
    bool is_testing = false;
    bool is_complete = false;
    std::string status_message;
};

/**
 * Callback types for testing progress
 */
using TestBatchCallback = std::function<void(int batch, int total, float running_accuracy)>;
using TestCompleteCallback = std::function<void(const TestingMetrics&)>;

/**
 * TestExecutor - Runs inference on test data and computes metrics
 *
 * This class handles model evaluation:
 * - Forward pass only (no gradients)
 * - Computes loss, accuracy, confusion matrix
 * - Calculates per-class precision, recall, F1
 * - Reports progress via callbacks
 */
class TestExecutor {
public:
    /**
     * Create a test executor
     * @param config Compiled training configuration from GraphCompiler
     * @param dataset Dataset handle from DataRegistry
     */
    TestExecutor(TrainingConfiguration config, DatasetHandle dataset);
    TestExecutor(TrainingConfiguration config,
                 std::shared_ptr<ArrowDataset> arrow_dataset,
                 std::string label_column,
                 TestDatasetScope dataset_scope);
    TestExecutor(TrainingConfiguration config,
                 std::shared_ptr<ParquetBackedDataset> parquet_dataset,
                 std::string label_column,
                 TestDatasetScope dataset_scope);
    TestExecutor(TrainingConfiguration config, const DataRegistry::TextDatasetEntry& text_entry);

    ~TestExecutor();

    /**
     * Run testing (blocking - should be called from a background thread)
     * @param batch_size Batch size for inference
     * @param batch_cb Callback for each batch (optional)
     * @param complete_cb Callback when testing completes (optional)
     */
    void Test(
        int batch_size,
        TestBatchCallback batch_cb = nullptr,
        TestCompleteCallback complete_cb = nullptr
    );

    /**
     * Stop testing (thread-safe, cooperative cancellation)
     */
    void Stop();

    /**
     * Check if testing is currently running
     */
    bool IsTesting() const { return is_testing_.load(); }

    /**
     * Get current testing metrics (thread-safe)
     */
    TestingMetrics GetMetrics() const;

    /**
     * Set a pre-trained model for testing
     * If not set, will build model from config
     */
    void SetModel(std::shared_ptr<SequentialModel> model);

    /**
     * Get the testing configuration
     */
    const TrainingConfiguration& GetConfig() const { return config_; }

private:
    TrainingConfiguration config_;
    DatasetHandle dataset_;
    std::shared_ptr<ArrowDataset> arrow_dataset_;
    std::shared_ptr<ParquetBackedDataset> parquet_dataset_;
    DataRegistry::TextDatasetEntry text_entry_;
    bool use_text_dataset_ = false;
    bool use_arrow_dataset_ = false;
    bool use_parquet_dataset_ = false;
    TestDatasetScope dataset_scope_ =
        TestDatasetScope::ConfiguredTestSplit;
    std::string arrow_label_column_;
    std::string parquet_label_column_;

    // Thread safety
    std::atomic<bool> is_testing_{false};
    std::atomic<bool> stop_requested_{false};

    mutable std::mutex metrics_mutex_;
    TestingMetrics metrics_;

    // Model and loss (no optimizer needed for testing)
    std::shared_ptr<SequentialModel> model_;
    std::unique_ptr<Loss> loss_;
    RegressionMetricAccumulator regression_metrics_;

    // Internal methods

    /**
     * Initialize testing components
     */
    bool Initialize(int batch_size);

    /**
     * Build model from configuration if not already set
     */
    bool BuildModelFromConfig();

    /**
     * Process a single batch
     */
    void ProcessBatch(const Batch& batch);

    /**
     * Compute per-class metrics from confusion matrix
     */
    void ComputePerClassMetrics();

    /**
     * Compute macro and weighted averages
     */
    void ComputeAggregateMetrics();

    /**
     * Update metrics (thread-safe)
     */
    void UpdateMetrics(const std::function<void(TestingMetrics&)>& updater);

    /**
     * Check if we should stop (for cooperative cancellation)
     */
    bool ShouldStop() const { return stop_requested_.load(); }

    /**
     * Forward pass through the model (no gradient tracking)
     */
    Tensor Forward(const Tensor& input);

    /**
     * Compute loss between predictions and targets
     */
    float ComputeLoss(const Tensor& predictions, const Tensor& targets);

    /**
     * Apply preprocessing to batch data
     */
    void PreprocessBatch(Batch& batch);
};

} // namespace cyxwiz
