#pragma once

#include "graph_compiler.h"
#include "metric_learning_metrics.h"
#include "dataset_batcher.h"
#include "data_registry.h"
#include "test_dataset_selection.h"
#include "executable_model.h"
#include <cyxwiz/tensor.h>
#include <cyxwiz/device.h>
#include <cyxwiz/sequential.h>
#include <cyxwiz/loss.h>
#include <cyxwiz/model_evaluation.h>
#include <functional>
#include <atomic>
#include <cmath>
#include <mutex>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <optional>

namespace cyxwiz {

class ArrowDataset;
class ParquetBackedDataset;

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
    int batch_limit = 0;      // > 0: a quick test that stops after this many batches
    bool partial = false;     // true when the quick test stopped before the end
    int correct_predictions = 0;

    // Overall metrics
    float test_loss = 0.0f;
    float test_accuracy = 0.0f;
    float macro_precision = 0.0f;
    float macro_recall = 0.0f;
    float macro_f1 = 0.0f;
    float weighted_f1 = 0.0f;
    bool regression_mode = false;
    bool causal_lm_mode = false;
    float test_mae = 0.0f;
    float test_rmse = 0.0f;
    size_t total_target_values = 0;
    // Metric-learning model (TOFIX140 A5): accuracy, confusion matrix and
    // per-class metrics are those of a 1-NN classifier over the test rows
    // (each row takes the class of its nearest other row); the Pair /
    // Retrieval Metrics nodes add their results.
    bool metric_learning_mode = false;
    std::optional<PairMetricResult> pair_metrics;
    std::optional<RetrievalMetricResult> retrieval_metrics;

    // Detailed results
    ConfusionMatrix confusion_matrix;
    std::vector<ClassMetrics> per_class_metrics;

    // Raw predictions (for detailed analysis)
    std::vector<int> predictions;
    std::vector<int> ground_truth;
    std::vector<float> confidences;
    // Regression: the true and predicted targets per sample, row-major
    // [samples][regression_width]; empty when the test set has too many
    // values to keep (regression_note says so).
    std::vector<float> regression_truth;
    std::vector<float> regression_predictions;
    int regression_width = 0;
    std::string regression_note;
    // The dataset row of each sample (as Batch::row_ids), in the table named
    // source_dataset; empty when the test batcher does not report rows.
    std::vector<int64_t> row_ids;
    std::string source_dataset;

    // Per-sample class probabilities, row-major [total_samples][probability_classes]
    // (binary: P(class 0), P(class 1)), kept for the ROC / PR curves. Empty when
    // the test set is too large to keep them; curves_note then says so.
    std::vector<float> class_probabilities;
    int probability_classes = 0;
    // ROC and precision-recall over the test set, on the device at the end:
    // binary - the positive class (1); multi-class - one-vs-rest per class,
    // skipping a class the test set has no sample of. curves_note says why
    // there are no curves, or which classes were skipped.
    std::optional<ROCCurveData> roc;
    std::optional<PRCurveData> pr;
    std::string curves_note;

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
// The end of a classifier test: per-class precision / recall / F1 from the
// confusion matrix (a class without a name gets "Class N"), the macro and
// support-weighted averages, accuracy, and the ROC / precision-recall curves
// from the kept class probabilities. A test run and a test read back from its
// per-row table (test_predictions_metrics.h) both finish with it, so every
// view shows the same numbers.
void FinishClassifierMetrics(TestingMetrics& m);

// One class against the rest (TOFIX129 D4: the positive class of Results >
// Test's curves and threshold): its ROC and precision-recall curves and how
// many test samples are of it. Two classes: class 1's are the run's own
// curves, class 0's are worked out on the device from P(class 0). More
// classes: the run's one-vs-rest curves. False with `error` when the run
// kept no class scores or the class has no curve (no sample of it).
struct ClassCurves {
    ROCCurveData roc;
    PRCurveData pr;
    int positives = 0;
    int negatives = 0;
};
bool CurvesForClass(const TestingMetrics& m, int positive, ClassCurves& out, std::string& error);

// A regression test's R-squared: 1 - SSE / SST per target over the kept rows,
// averaged over the targets (scikit-learn's r2_score); NaN when no rows were
// kept or a target is constant.
double RegressionRSquared(const TestingMetrics& m);

using TestBatchCallback = std::function<void(int batch, int total, float running_accuracy)>;
using TestCompleteCallback = std::function<void(const TestingMetrics&)>;

// Test batcher built by the host when the run starts (batch size known), and
// optional class names to label metrics with.
struct ExternalTestSource {
    std::function<std::unique_ptr<IBatcher>(const TrainingConfiguration& config, int batch_size)> make_batcher;
    std::vector<std::string> class_names;
};

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
     * Create a test executor for a host-built test batcher (e.g. DataRegistry
     * datasets, see legacy_dataset_batchers.h).
     */
    TestExecutor(TrainingConfiguration config, ExternalTestSource source);
    TestExecutor(TrainingConfiguration config,
                 std::shared_ptr<ArrowDataset> arrow_dataset,
                 std::string label_column,
                 TestDatasetScope dataset_scope);
    TestExecutor(TrainingConfiguration config,
                 std::shared_ptr<ParquetBackedDataset> parquet_dataset,
                 std::string label_column,
                 TestDatasetScope dataset_scope);

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
    // Quick test (TOFIX129 G3): stop after this many batches, 0 = all.
    void SetBatchLimit(int batches) { batch_limit_ = batches; }

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
    // Pass the recorded owner when handing off from a different thread.
    // Omitting it means the calling thread owns the supplied model's tensors.
    // Any trained model: sequential or a graph with branches (TOFIX157).
    void SetModel(std::shared_ptr<IExecutableModel> model,
                  std::optional<ProcessDeviceSelection> model_device = std::nullopt);
    // A layer stack someone else holds (shared, not copied).
    void SetModel(std::shared_ptr<SequentialModel> model,
                  std::optional<ProcessDeviceSelection> model_device = std::nullopt);

    /**
     * Get the testing configuration
     */
    const TrainingConfiguration& GetConfig() const { return config_; }

private:
    TrainingConfiguration config_;
    std::shared_ptr<ArrowDataset> arrow_dataset_;
    std::shared_ptr<ParquetBackedDataset> parquet_dataset_;
    ExternalTestSource external_source_;
    bool use_arrow_dataset_ = false;
    bool use_parquet_dataset_ = false;
    TestDatasetScope dataset_scope_ =
        TestDatasetScope::ConfiguredTestSplit;
    std::string arrow_label_column_;
    std::string parquet_label_column_;

    // Thread safety
    std::atomic<bool> is_testing_{false};
    std::atomic<bool> stop_requested_{false};
    int batch_limit_ = 0;

    mutable std::mutex metrics_mutex_;
    TestingMetrics metrics_;

    // Model and loss (no optimizer needed for testing)
    std::shared_ptr<IExecutableModel> model_;
    std::optional<ProcessDeviceSelection> model_device_;
    std::unique_ptr<Loss> loss_;
    RegressionMetricAccumulator regression_metrics_;

    // Internal methods

    /**
     * Initialize testing components
     */
    bool Initialize(int batch_size);
    void TestImpl(int batch_size, TestBatchCallback batch_cb, TestCompleteCallback complete_cb);
    void TestCausalSequence(int batch_size, TestBatchCallback batch_cb, TestCompleteCallback complete_cb);

    /**
     * Build model from configuration if not already set
     */
    bool BuildModelFromConfig();

    /**
     * Process a single batch
     */
    void ProcessBatch(const Batch& batch);
    // Metric-learning models: the loss over the builder's pairs / triplets and
    // the 1-NN / Pair / Retrieval evaluation of the test rows.
    void TestMetricLearning(IBatcher& rows, int batch_size);

    void ApplyLabelClassNames(const std::vector<std::string>& names);

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
