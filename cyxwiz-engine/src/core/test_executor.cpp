#include "test_executor.h"
#include "classification_decision.h"
#include "model_builder.h"
#include "training_batcher_setup.h"

#include <cstdint>
#include <spdlog/spdlog.h>
#include <cmath>
#include <stdexcept>

namespace cyxwiz {

// ============================================================================
// ConfusionMatrix Implementation
// ============================================================================

void ConfusionMatrix::Resize(int n) {
    num_classes = n;
    matrix.clear();
    matrix.resize(n, std::vector<int>(n, 0));
}

void ConfusionMatrix::Add(int actual, int predicted) {
    if (actual >= 0 && actual < num_classes && predicted >= 0 && predicted < num_classes) {
        matrix[actual][predicted]++;
    }
}

float ConfusionMatrix::GetAccuracy() const {
    if (num_classes == 0) return 0.0f;

    int correct = 0;
    int total = 0;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            total += matrix[i][j];
            if (i == j) correct += matrix[i][j];
        }
    }
    return total > 0 ? static_cast<float>(correct) / total : 0.0f;
}

int ConfusionMatrix::GetTotal() const {
    int total = 0;
    for (int i = 0; i < num_classes; ++i) {
        for (int j = 0; j < num_classes; ++j) {
            total += matrix[i][j];
        }
    }
    return total;
}

// ============================================================================
// TestExecutor Implementation
// ============================================================================

TestExecutor::TestExecutor(TrainingConfiguration config, DatasetHandle dataset)
    : config_(std::move(config))
    , dataset_(dataset)
{
    spdlog::info("TestExecutor: Created with {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TestExecutor::TestExecutor(
    TrainingConfiguration config,
    std::shared_ptr<ArrowDataset> arrow_dataset,
    std::string label_column,
    TestDatasetScope dataset_scope)
    : config_(std::move(config))
    , arrow_dataset_(std::move(arrow_dataset))
    , use_arrow_dataset_(true)
    , dataset_scope_(dataset_scope)
    , arrow_label_column_(std::move(label_column))
{
    spdlog::info("TestExecutor: Created for Arrow dataset with {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}
TestExecutor::TestExecutor(
    TrainingConfiguration config,
    std::shared_ptr<ParquetBackedDataset> parquet_dataset,
    std::string label_column,
    TestDatasetScope dataset_scope)
    : config_(std::move(config))
    , parquet_dataset_(std::move(parquet_dataset))
    , use_parquet_dataset_(true)
    , dataset_scope_(dataset_scope)
    , parquet_label_column_(std::move(label_column))
{
    spdlog::info("TestExecutor: Created for Parquet-backed dataset with {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TestExecutor::TestExecutor(
    TrainingConfiguration config,
    const DataRegistry::TextDatasetEntry& text_entry)
    : config_(std::move(config))
    , text_entry_(text_entry)
    , use_text_dataset_(true)
{
    spdlog::info("TestExecutor: Created for text dataset with {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TestExecutor::~TestExecutor() {
    Stop();
}

void TestExecutor::SetModel(std::shared_ptr<SequentialModel> model) {
    model_ = model;
    spdlog::info("TestExecutor: External model set");
}

bool TestExecutor::BuildModelFromConfig() {
    if (model_) {
        spdlog::info("TestExecutor: Using pre-set model");
        return true;
    }

    spdlog::info(
        "TestExecutor: Building model from {} layer configs through ModelBuilder",
        config_.layers.size());
    BuiltModel built = BuildSequentialFromConfig(config_);
    if (!built.ok()) {
        spdlog::error(
            "TestExecutor: ModelBuilder rejected test model: {}",
            built.error_message.empty() ? "unknown model construction failure"
                                        : built.error_message);
        return false;
    }

    model_ = std::shared_ptr<SequentialModel>(std::move(built.model));
    return true;
}

bool TestExecutor::Initialize(int /*batch_size*/) {
    std::string target_transform_error;
    if (!ResolveRegressionTargetTransform(
            config_.regression_target_transform,
            target_transform_error) ||
        !config_.regression_target_transform.IsResolvedForWidth(
            config_.output_size)) {
        if (target_transform_error.empty()) {
            target_transform_error =
                "resolved target transform width does not match model output";
        }
        spdlog::error(
            "TestExecutor: regression target transform is invalid: {}",
            target_transform_error);
        return false;
    }
    regression_metrics_.SetTargetTransform(
        &config_.regression_target_transform);
    if (config_.regression_target_transform.enabled) {
        spdlog::info(
            "TestExecutor: resolved StandardScaler regression target state "
            "'{}' for {} outputs; loss is transformed-space and MAE/RMSE "
            "are original-unit metrics",
            config_.regression_target_transform.state_path,
            config_.regression_target_transform.scales.size());
    }

    // Build model from configuration if not already set
    if (!BuildModelFromConfig()) {
        spdlog::error("TestExecutor: Failed to build model from config");
        return false;
    }

    try {
        loss_ = BuildLossFromConfig(config_);
    } catch (const std::exception& e) {
        spdlog::error(
            "TestExecutor: failed to build configured loss: {}", e.what());
        return false;
    }

    const bool regression_mode = UsesContinuousTargetMetrics(config_);
    UpdateMetrics([regression_mode](TestingMetrics& m) {
        m.regression_mode = regression_mode;
    });

    // Classification-only detail structures. Continuous targets use
    // MAE/RMSE across every output value instead of class decisions.
    if (regression_mode) {
        return true;
    }

    // Initialize confusion matrix
    int num_classes = UsesScalarBinaryTargets(config_.loss_type)
        ? 2
        : static_cast<int>(config_.output_size);
    UpdateMetrics([num_classes](TestingMetrics& m) {
        m.confusion_matrix.Resize(num_classes);
        m.per_class_metrics.resize(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            m.per_class_metrics[i].class_id = i;
            m.per_class_metrics[i].class_name = "Class " + std::to_string(i);
        }
    });

    if (use_text_dataset_ && !text_entry_.class_names.empty()) {
        UpdateMetrics([this](TestingMetrics& m) {
            m.confusion_matrix.class_names = text_entry_.class_names;
            for (size_t i = 0; i < m.per_class_metrics.size() &&
                               i < text_entry_.class_names.size(); ++i) {
                m.per_class_metrics[i].class_name = text_entry_.class_names[i];
            }
        });
    }

    return true;
}

void TestExecutor::Test(
    int batch_size,
    TestBatchCallback batch_cb,
    TestCompleteCallback complete_cb)
{
    if (is_testing_.load()) {
        spdlog::warn("TestExecutor: Already testing");
        return;
    }

    is_testing_.store(true);
    stop_requested_.store(false);

    // Initialize
    if (!Initialize(batch_size)) {
        const std::string detail =
            "TestExecutor could not initialize the model or "
            "configured loss";
        UpdateMetrics([&detail](TestingMetrics& m) {
            m.is_testing = false;
            m.is_complete = false;
            m.status_message = "Testing failed: " + detail;
        });
        spdlog::error("TestExecutor: {}", detail);
        is_testing_.store(false);
        throw std::runtime_error(detail);
    }

    // Setup metrics
    regression_metrics_.Reset();
    const bool regression_mode = UsesContinuousTargetMetrics(config_);
    UpdateMetrics([regression_mode](TestingMetrics& m) {
        m.current_batch = 0;
        m.total_batches = 0;
        m.total_samples = 0;
        m.correct_predictions = 0;
        m.test_loss = 0.0f;
        m.test_accuracy = 0.0f;
        m.regression_mode = regression_mode;
        m.test_mae = 0.0f;
        m.test_rmse = 0.0f;
        m.total_target_values = 0;
        m.is_testing = true;
        m.is_complete = false;
        m.status_message = "Starting testing...";
        m.predictions.clear();
        m.ground_truth.clear();
        m.confidences.clear();
    });

    // Create the test batcher using the dataset type that was trained.
    std::unique_ptr<DatasetBatcher> legacy_test_batcher;
    std::unique_ptr<TextDatasetBatcher> text_test_batcher;
    std::unique_ptr<ArrowDatasetBatcher> arrow_test_batcher;
    std::unique_ptr<ParquetArrowBatcher> parquet_test_batcher;

    if (use_arrow_dataset_) {
        TrainingConfiguration test_config =
            ConfigureTestDatasetScope(config_, dataset_scope_);
        auto batchers = BuildArrowTrainingBatchers(
            test_config, arrow_dataset_, arrow_label_column_, batch_size);
        arrow_test_batcher =
            dataset_scope_ == TestDatasetScope::EntireProvidedDataset
                ? std::move(batchers.arrow_train)
                : std::move(batchers.arrow_test);
    } else if (use_parquet_dataset_) {
        TrainingConfiguration test_config =
            ConfigureTestDatasetScope(config_, dataset_scope_);
        auto batchers = BuildParquetTrainingBatchers(
            test_config, parquet_dataset_, parquet_label_column_, batch_size);
        parquet_test_batcher =
            dataset_scope_ == TestDatasetScope::EntireProvidedDataset
                ? std::move(batchers.parquet_train)
                : std::move(batchers.parquet_test);
    } else if (use_text_dataset_) {
        text_test_batcher = std::make_unique<TextDatasetBatcher>(
            text_entry_,
            config_.text_preprocessing,
            batch_size,
            config_.train_ratio,
            config_.val_ratio,
            config_.test_ratio,
            false,
            config_.num_workers,
            static_cast<uint32_t>(config_.dataloader_seed),
            config_.stratified,
            static_cast<uint32_t>(std::max(0, config_.split_seed)),
            false,
            "none",
            "max",
            static_cast<uint32_t>(std::max(0, config_.balance_seed)));
        text_test_batcher->SetPhase(BatcherPhase::Test);
        text_test_batcher->Reset();

        if (UsesScalarBinaryTargets(config_.loss_type)) {
            text_test_batcher->SetScalarLabelMode(true);
        } else if (config_.preprocessing.has_onehot &&
                   config_.preprocessing.num_classes > 0) {
            text_test_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
        } else if (config_.output_size > 0) {
            text_test_batcher->SetOneHotEncoding(config_.output_size);
        }
    } else {
        legacy_test_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Test, false, false, config_.num_workers,
            static_cast<uint32_t>(config_.dataloader_seed));

        if (config_.preprocessing.has_normalization) {
            legacy_test_batcher->SetLegacyNormalization(config_.preprocessing.norm_mean,
                                                        config_.preprocessing.norm_std);
        }

        if (UsesScalarBinaryTargets(config_.loss_type)) {
            legacy_test_batcher->SetLegacyScalarLabelMode(true);
        } else if (config_.preprocessing.has_onehot) {
            legacy_test_batcher->SetLegacyOneHotEncoding(config_.preprocessing.num_classes);
        }

        legacy_test_batcher->SetFlatten(true);
    }

    size_t total_batches = 0;
    if (use_arrow_dataset_) {
        total_batches = arrow_test_batcher ? arrow_test_batcher->GetNumBatches() : 0;
    } else if (use_parquet_dataset_) {
        total_batches = parquet_test_batcher ? parquet_test_batcher->GetNumBatches() : 0;
    } else if (use_text_dataset_) {
        total_batches = text_test_batcher ? text_test_batcher->GetNumBatches() : 0;
    } else {
        total_batches = legacy_test_batcher ? legacy_test_batcher->GetNumBatches() : 0;
    }
    UpdateMetrics([total_batches](TestingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
    });

    spdlog::info(
        "TestExecutor: Starting testing with batch_size={}, {} batches, "
        "dataset_scope={}",
        batch_size,
        total_batches,
        dataset_scope_ == TestDatasetScope::EntireProvidedDataset
            ? "entire_provided_dataset"
            : "configured_test_split");

    if (total_batches == 0) {
        const std::string message =
            "Testing has no test batches. Check that the trained dataset has a "
            "non-empty test split and that Tools > Test uses the same effective "
            "dataset/materialized dataset that training used.";
        UpdateMetrics([&message](TestingMetrics& m) {
            m.is_testing = false;
            m.is_complete = false;
            m.status_message = message;
        });
        is_testing_.store(false);
        spdlog::error("TestExecutor: {}", message);
        throw std::runtime_error(message);
    }

    // Set model to evaluation mode (disables dropout, etc.)
    model_->SetTraining(false);

    auto start_time = std::chrono::steady_clock::now();

    // Testing loop
    float total_loss = 0.0f;
    int batch_num = 0;

    while (true) {
        if (ShouldStop()) break;

        Batch batch;
        if (use_arrow_dataset_) {
            batch = arrow_test_batcher->GetNextBatch();
        } else if (use_parquet_dataset_) {
            batch = parquet_test_batcher->GetNextBatch();
        } else if (use_text_dataset_) {
            batch = text_test_batcher->GetNextBatch();
        } else {
            batch = legacy_test_batcher->GetNextBatch();
        }

        if (!batch.IsValid()) break;

        batch_num++;

        // Process batch
        ProcessBatch(batch);

        // Compute current loss for this batch
        Tensor predictions = Forward(batch.data);
        float batch_loss = ComputeLoss(predictions, batch.labels);
        total_loss += batch_loss;

        // Update running metrics
        TestingMetrics current = GetMetrics();
        float running_accuracy = current.total_samples > 0 ?
            static_cast<float>(current.correct_predictions) / current.total_samples : 0.0f;

        UpdateMetrics([batch_num, total_loss, running_accuracy](TestingMetrics& m) {
            m.current_batch = batch_num;
            m.test_loss = total_loss / batch_num;
            m.test_accuracy = running_accuracy;
            m.status_message = "Testing batch " + std::to_string(batch_num) + "...";
        });

        // Batch callback
        if (batch_cb) {
            batch_cb(batch_num, static_cast<int>(total_batches), running_accuracy);
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    float total_time = std::chrono::duration<float>(end_time - start_time).count();

    // Compute final metrics
    if (!regression_mode) {
        ComputePerClassMetrics();
        ComputeAggregateMetrics();
    }

    TestingMetrics final_metrics = GetMetrics();
    float samples_per_sec = final_metrics.total_samples > 0 ?
        final_metrics.total_samples / total_time : 0.0f;

    // Update final state
    UpdateMetrics([total_time, samples_per_sec](TestingMetrics& m) {
        m.total_time_seconds = total_time;
        m.samples_per_second = samples_per_sec;
        m.is_testing = false;
        m.is_complete = true;
        m.status_message = "Testing complete";
    });

    is_testing_.store(false);

    // Log results
    final_metrics = GetMetrics();
    spdlog::info("TestExecutor: Testing complete");
    spdlog::info("  Loss: {:.4f}", final_metrics.test_loss);
    if (final_metrics.regression_mode) {
        spdlog::info("  MAE: {:.4f}", final_metrics.test_mae);
        spdlog::info("  RMSE: {:.4f}", final_metrics.test_rmse);
    } else {
        spdlog::info("  Accuracy: {:.2f}%",
                     final_metrics.test_accuracy * 100);
        spdlog::info("  Macro F1: {:.4f}", final_metrics.macro_f1);
    }
    spdlog::info("  Samples: {}", final_metrics.total_samples);
    spdlog::info("  Time: {:.2f}s ({:.0f} samples/sec)", total_time, samples_per_sec);

    // Complete callback
    if (complete_cb) {
        complete_cb(final_metrics);
    }
}

void TestExecutor::ProcessBatch(const Batch& batch) {
    // Forward pass
    Tensor predictions = Forward(batch.data);

    const float* pred_data = predictions.Data<float>();
    const float* target_data = batch.labels.Data<float>();

    const size_t output_width = config_.output_size;
    if (UsesContinuousTargetMetrics(config_)) {
        const size_t value_count = batch.size * output_width;
        regression_metrics_.Add(
            pred_data, target_data, value_count, output_width);
        UpdateMetrics([this, &batch, value_count](TestingMetrics& m) {
            m.total_samples += static_cast<int>(batch.size);
            m.total_target_values += value_count;
            m.test_mae = regression_metrics_.Mae();
            m.test_rmse = regression_metrics_.Rmse();
        });
        return;
    }

    const auto decision_mode =
        ClassificationDecisionModeForLoss(config_.loss_type);

    for (size_t b = 0; b < batch.size; ++b) {
        const float* sample_pred = pred_data + b * output_width;
        int pred_class = ClassificationPredictedClass(
            sample_pred, output_width, decision_mode);

        // Get ground truth (argmax of one-hot or direct class index)
        const float* sample_target = target_data + b * output_width;
        int true_class = ClassificationTargetClass(
            sample_target, output_width, decision_mode);

        // Get confidence
        float confidence = ClassificationConfidence(
            sample_pred, output_width, pred_class, decision_mode);

        // Update metrics
        UpdateMetrics([pred_class, true_class, confidence](TestingMetrics& m) {
            m.total_samples++;
            if (pred_class == true_class) {
                m.correct_predictions++;
            }
            m.predictions.push_back(pred_class);
            m.ground_truth.push_back(true_class);
            m.confidences.push_back(confidence);
            m.confusion_matrix.Add(true_class, pred_class);
        });
    }
}

void TestExecutor::ComputePerClassMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    int num_classes = metrics_.confusion_matrix.num_classes;

    for (int c = 0; c < num_classes; ++c) {
        ClassMetrics& cm = metrics_.per_class_metrics[c];
        cm.class_id = c;

        // True positives = diagonal element
        cm.true_positives = metrics_.confusion_matrix.matrix[c][c];

        // False positives = sum of column c minus diagonal
        cm.false_positives = 0;
        for (int i = 0; i < num_classes; ++i) {
            if (i != c) {
                cm.false_positives += metrics_.confusion_matrix.matrix[i][c];
            }
        }

        // False negatives = sum of row c minus diagonal
        cm.false_negatives = 0;
        for (int j = 0; j < num_classes; ++j) {
            if (j != c) {
                cm.false_negatives += metrics_.confusion_matrix.matrix[c][j];
            }
        }

        // Support = sum of row (actual samples in this class)
        cm.support = 0;
        for (int j = 0; j < num_classes; ++j) {
            cm.support += metrics_.confusion_matrix.matrix[c][j];
        }

        // Precision = TP / (TP + FP)
        int pred_positives = cm.true_positives + cm.false_positives;
        cm.precision = pred_positives > 0 ?
            static_cast<float>(cm.true_positives) / pred_positives : 0.0f;

        // Recall = TP / (TP + FN)
        int actual_positives = cm.true_positives + cm.false_negatives;
        cm.recall = actual_positives > 0 ?
            static_cast<float>(cm.true_positives) / actual_positives : 0.0f;

        // F1 = 2 * (precision * recall) / (precision + recall)
        float pr_sum = cm.precision + cm.recall;
        cm.f1_score = pr_sum > 0 ? 2.0f * cm.precision * cm.recall / pr_sum : 0.0f;
    }
}

void TestExecutor::ComputeAggregateMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);

    int num_classes = static_cast<int>(metrics_.per_class_metrics.size());
    if (num_classes == 0) return;

    // Macro averages (unweighted mean of per-class metrics)
    float sum_precision = 0.0f;
    float sum_recall = 0.0f;
    float sum_f1 = 0.0f;

    // Weighted averages (weighted by support)
    float weighted_f1 = 0.0f;
    int total_support = 0;

    for (const auto& cm : metrics_.per_class_metrics) {
        sum_precision += cm.precision;
        sum_recall += cm.recall;
        sum_f1 += cm.f1_score;
        weighted_f1 += cm.f1_score * cm.support;
        total_support += cm.support;
    }

    metrics_.macro_precision = sum_precision / num_classes;
    metrics_.macro_recall = sum_recall / num_classes;
    metrics_.macro_f1 = sum_f1 / num_classes;
    metrics_.weighted_f1 = total_support > 0 ? weighted_f1 / total_support : 0.0f;

    // Final accuracy from confusion matrix
    metrics_.test_accuracy = metrics_.confusion_matrix.GetAccuracy();
}

Tensor TestExecutor::Forward(const Tensor& input) {
    if (!model_) {
        spdlog::error("TestExecutor::Forward: Model not initialized");
        return Tensor();
    }
    return model_->Forward(input);
}

float TestExecutor::ComputeLoss(const Tensor& predictions, const Tensor& targets) {
    if (!loss_) return 0.0f;

    Tensor loss_tensor = loss_->Forward(predictions, targets);
    const float* loss_data = loss_tensor.Data<float>();
    return loss_data[0];
}

void TestExecutor::Stop() {
    stop_requested_.store(true);
}

TestingMetrics TestExecutor::GetMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void TestExecutor::UpdateMetrics(const std::function<void(TestingMetrics&)>& updater) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    updater(metrics_);
}

void TestExecutor::PreprocessBatch(Batch& /*batch*/) {
    // Preprocessing is handled by DatasetBatcher
}

} // namespace cyxwiz
