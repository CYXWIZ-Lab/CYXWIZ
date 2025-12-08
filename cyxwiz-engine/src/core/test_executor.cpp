#include "test_executor.h"
#include <spdlog/spdlog.h>
#include <cmath>
#include <algorithm>
#include <numeric>

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

    model_ = std::make_shared<SequentialModel>();

    spdlog::info("TestExecutor: Building model from {} layer configs", config_.layers.size());

    // Track input size for each layer
    size_t current_input_size = config_.input_size;

    for (size_t i = 0; i < config_.layers.size(); ++i) {
        const auto& layer_cfg = config_.layers[i];

        switch (layer_cfg.type) {
            case gui::NodeType::Dense: {
                size_t out_features = layer_cfg.units > 0 ? layer_cfg.units : 64;
                model_->Add<LinearModule>(current_input_size, out_features, true);
                spdlog::info("  [{}] Linear({} -> {})", i, current_input_size, out_features);
                current_input_size = out_features;
                break;
            }

            case gui::NodeType::ReLU:
                model_->Add<ReLUModule>();
                break;

            case gui::NodeType::Sigmoid:
                model_->Add<SigmoidModule>();
                break;

            case gui::NodeType::Tanh:
                model_->Add<TanhModule>();
                break;

            case gui::NodeType::LeakyReLU: {
                float slope = layer_cfg.negative_slope > 0 ? layer_cfg.negative_slope : 0.01f;
                model_->Add<LeakyReLUModule>(slope);
                break;
            }

            case gui::NodeType::ELU: {
                float alpha = layer_cfg.alpha > 0 ? layer_cfg.alpha : 1.0f;
                model_->Add<ELUModule>(alpha);
                break;
            }

            case gui::NodeType::GELU:
                model_->Add<GELUModule>();
                break;

            case gui::NodeType::Swish:
                model_->Add<SwishModule>();
                break;

            case gui::NodeType::Mish:
                model_->Add<MishModule>();
                break;

            case gui::NodeType::Softmax:
                model_->Add<SoftmaxModule>();
                break;

            case gui::NodeType::Dropout:
                // Skip dropout in testing (no-op in eval mode)
                break;

            case gui::NodeType::Flatten:
                model_->Add<FlattenModule>(1);
                break;

            case gui::NodeType::Output: {
                size_t out_features = config_.output_size;
                model_->Add<LinearModule>(current_input_size, out_features, true);
                current_input_size = out_features;
                break;
            }

            // Skip non-layer nodes
            case gui::NodeType::DatasetInput:
            case gui::NodeType::DataLoader:
            case gui::NodeType::Augmentation:
            case gui::NodeType::DataSplit:
            case gui::NodeType::TensorReshape:
            case gui::NodeType::Normalize:
            case gui::NodeType::OneHotEncode:
            case gui::NodeType::MSELoss:
            case gui::NodeType::CrossEntropyLoss:
            case gui::NodeType::BCELoss:
            case gui::NodeType::BCEWithLogits:
            case gui::NodeType::L1Loss:
            case gui::NodeType::SmoothL1Loss:
            case gui::NodeType::HuberLoss:
            case gui::NodeType::NLLLoss:
            case gui::NodeType::SGD:
            case gui::NodeType::Adam:
            case gui::NodeType::AdamW:
                break;

            default:
                break;
        }
    }

    if (model_->Size() == 0) {
        spdlog::error("TestExecutor: No layers were added to the model!");
        return false;
    }

    return true;
}

bool TestExecutor::Initialize(int batch_size) {
    // Build model from configuration if not already set
    if (!BuildModelFromConfig()) {
        spdlog::error("TestExecutor: Failed to build model from config");
        return false;
    }

    // Create loss function based on config (for loss computation, not training)
    switch (config_.loss_type) {
        case gui::NodeType::CrossEntropyLoss:
            loss_ = CreateLoss(LossType::CrossEntropy);
            break;
        case gui::NodeType::MSELoss:
            loss_ = CreateLoss(LossType::MSE);
            break;
        case gui::NodeType::BCELoss:
            loss_ = CreateLoss(LossType::BinaryCrossEntropy);
            break;
        case gui::NodeType::BCEWithLogits:
            loss_ = CreateLoss(LossType::BCEWithLogits);
            break;
        default:
            loss_ = CreateLoss(LossType::CrossEntropy);
            break;
    }

    // Initialize confusion matrix
    int num_classes = static_cast<int>(config_.output_size);
    UpdateMetrics([num_classes](TestingMetrics& m) {
        m.confusion_matrix.Resize(num_classes);
        m.per_class_metrics.resize(num_classes);
        for (int i = 0; i < num_classes; ++i) {
            m.per_class_metrics[i].class_id = i;
            m.per_class_metrics[i].class_name = "Class " + std::to_string(i);
        }
    });

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
        spdlog::error("TestExecutor: Failed to initialize");
        is_testing_.store(false);
        return;
    }

    // Setup metrics
    UpdateMetrics([](TestingMetrics& m) {
        m.current_batch = 0;
        m.total_batches = 0;
        m.total_samples = 0;
        m.correct_predictions = 0;
        m.test_loss = 0.0f;
        m.test_accuracy = 0.0f;
        m.is_testing = true;
        m.is_complete = false;
        m.status_message = "Starting testing...";
        m.predictions.clear();
        m.ground_truth.clear();
        m.confidences.clear();
    });

    // Create test batcher (use Test split if available, otherwise validation)
    DatasetBatcher test_batcher(dataset_, batch_size, DatasetSplit::Test, false, false);

    // Apply preprocessing settings
    if (config_.preprocessing.has_normalization) {
        test_batcher.SetNormalization(config_.preprocessing.norm_mean,
                                       config_.preprocessing.norm_std);
    }

    if (config_.preprocessing.has_onehot) {
        test_batcher.SetOneHotEncoding(config_.preprocessing.num_classes);
    }

    // Flatten input for MLP
    test_batcher.SetFlatten(true);

    size_t total_batches = test_batcher.GetNumBatches();
    UpdateMetrics([total_batches](TestingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
    });

    spdlog::info("TestExecutor: Starting testing with batch_size={}, {} batches",
                 batch_size, total_batches);

    // Set model to evaluation mode (disables dropout, etc.)
    model_->SetTraining(false);

    auto start_time = std::chrono::steady_clock::now();

    // Testing loop
    float total_loss = 0.0f;
    int batch_num = 0;

    while (!test_batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        Batch batch = test_batcher.GetNextBatch();
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
    ComputePerClassMetrics();
    ComputeAggregateMetrics();

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
    spdlog::info("  Accuracy: {:.2f}%", final_metrics.test_accuracy * 100);
    spdlog::info("  Loss: {:.4f}", final_metrics.test_loss);
    spdlog::info("  Samples: {}", final_metrics.total_samples);
    spdlog::info("  Time: {:.2f}s ({:.0f} samples/sec)", total_time, samples_per_sec);
    spdlog::info("  Macro F1: {:.4f}", final_metrics.macro_f1);

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

    size_t num_classes = config_.output_size;

    for (size_t b = 0; b < batch.size; ++b) {
        // Get prediction (argmax)
        const float* sample_pred = pred_data + b * num_classes;
        int pred_class = ArgMax(sample_pred, num_classes);

        // Get ground truth (argmax of one-hot or direct class index)
        const float* sample_target = target_data + b * num_classes;
        int true_class = ArgMax(sample_target, num_classes);

        // Get confidence
        float confidence = GetConfidence(sample_pred, num_classes, pred_class);

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

int TestExecutor::ArgMax(const float* data, size_t size) {
    if (size == 0) return 0;

    int max_idx = 0;
    float max_val = data[0];
    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = static_cast<int>(i);
        }
    }
    return max_idx;
}

float TestExecutor::GetConfidence(const float* data, size_t size, int predicted_class) {
    if (size == 0 || predicted_class < 0 || predicted_class >= static_cast<int>(size)) {
        return 0.0f;
    }

    // If output is already softmax (sums to ~1), return directly
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }

    if (std::abs(sum - 1.0f) < 0.1f) {
        // Already normalized
        return data[predicted_class];
    }

    // Apply softmax
    float max_val = *std::max_element(data, data + size);
    float exp_sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        exp_sum += std::exp(data[i] - max_val);
    }
    return std::exp(data[predicted_class] - max_val) / exp_sum;
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

void TestExecutor::PreprocessBatch(Batch& batch) {
    // Preprocessing is handled by DatasetBatcher
}

} // namespace cyxwiz
