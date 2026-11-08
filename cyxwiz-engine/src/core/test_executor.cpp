#include "test_executor.h"
#include "execution_device_context.h"
#include "metric_learning_evaluation.h"
#include "metric_learning_sampling.h"
#include "spatial_batch_layout.h"
#include "spatial_sequential_head.h"
#include "classification_decision.h"
#include "model_builder.h"
#include "training_batcher_setup.h"

#include <arrayfire.h>

#include <algorithm>
#include <iterator>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <cmath>
#include <stdexcept>

namespace cyxwiz {

// ============================================================================
// TestExecutor Implementation
// ============================================================================

TestExecutor::TestExecutor(TrainingConfiguration config, ExternalTestSource source)
    : config_(std::move(config))
    , external_source_(std::move(source))
{
    spdlog::info("TestExecutor: Created with a host-built test batcher, {} layers, "
                 "input_size={}, output_size={}",
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

TestExecutor::~TestExecutor() {
    Stop();
}

void TestExecutor::SetModel(std::shared_ptr<SequentialModel> model,
                            std::optional<ProcessDeviceSelection> model_device) {
    SetModel(model ? std::make_shared<SequentialExecutableModel>(std::move(model)) : nullptr, model_device);
}

void TestExecutor::SetModel(std::shared_ptr<IExecutableModel> model,
                            std::optional<ProcessDeviceSelection> model_device) {
    model_ = std::move(model);
    model_device_ = model_device;
    if (model_ && !model_device_) {
        if (const auto* current = Device::GetCurrentDevice()) {
            model_device_ = ProcessDeviceSelection{current->GetType(), current->GetDeviceId()};
        }
    }
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

    model_ = std::make_shared<SequentialExecutableModel>(std::move(built.model));
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
    if (regression_mode || config_.sequence_batch.create_causal_lm_targets) {
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

    if (!external_source_.class_names.empty()) {
        UpdateMetrics([this](TestingMetrics& m) {
            m.confusion_matrix.class_names = external_source_.class_names;
            for (size_t i = 0; i < m.per_class_metrics.size() &&
                               i < external_source_.class_names.size(); ++i) {
                m.per_class_metrics[i].class_name = external_source_.class_names[i];
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
    if (is_testing_.exchange(true)) {
        spdlog::warn("TestExecutor: Already testing");
        return;
    }
    stop_requested_.store(false);
    UpdateMetrics([](TestingMetrics& m) {
        m = TestingMetrics{};
        m.is_testing = true;
        m.status_message = "Activating model device for testing";
    });
    try {
        // All test modalities share this boundary. Activate the recorded owner
        // BEFORE batch tensors, loss construction or the first model forward.
        // Binding execution metadata alone does not switch ArrayFire's backend.
        ScopedActiveExecutionDeviceContext active;
        auto owner = model_device_;
        if (!model_) owner = Device::GetProcessDevice();
        if (model_ && !owner) {
            throw std::runtime_error(
                "Run Test has no model device ownership. Reload the checkpoint "
                "through Train > Test > Load Checkpoint for Testing.");
        }
        if (owner) {
            const auto activation = Device(owner->type, owner->device_id).ActivateExact(false);
            if (!activation.success) {
                throw std::runtime_error(fmt::format(
                    "Run Test could not activate the model device (type={}, id={}): {}. "
                    "No fallback was attempted. Make that device available or reload "
                    "the checkpoint on an available device before testing.",
                    static_cast<int>(owner->type), owner->device_id, activation.message));
            }
        }
        const auto policy = config_.strict_execution
            ? ExecutionPolicy::Strict
            : ExecutionPolicy::Compatibility;
        const auto context = CaptureCurrentExecutionDeviceContext(policy);
        if (!context.valid) {
            throw std::runtime_error("Run Test execution context is invalid: " + context.error);
        }
        if (!model_) {
            if (const auto* current = Device::GetCurrentDevice()) {
                model_device_ = ProcessDeviceSelection{current->GetType(), current->GetDeviceId()};
            }
        }
        ScopedExecutionDeviceContext binding(context);
        spdlog::info(
            "TestExecutor: model device activated; owner={}:{} effective={}:{} "
            "device='{}' execution_policy={} selection_fallback=false",
            context.requested_backend, context.requested_device_id,
            context.effective_backend, context.effective_device_id,
            context.device_name, context.ExecutionPolicyName());
        TestImpl(batch_size, batch_cb, complete_cb);
    } catch (const std::exception& e) {
        is_testing_.store(false);
        UpdateMetrics([&](TestingMetrics& m) {
            m.is_testing = false;
            m.is_complete = false;
            m.status_message = std::string("Testing failed: ") + e.what();
        });
        throw;
    }
}

void TestExecutor::TestImpl(int batch_size, TestBatchCallback batch_cb,
                            TestCompleteCallback complete_cb) {
    if (config_.sequence_batch.enabled) {
        TestCausalSequence(batch_size, batch_cb, complete_cb);
        return;
    }

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
    const std::string source_dataset = use_arrow_dataset_ && arrow_dataset_       ? arrow_dataset_->GetName()
                                       : use_parquet_dataset_ && parquet_dataset_ ? parquet_dataset_->GetName()
                                                                                  : external_source_.source_dataset;
    UpdateMetrics([regression_mode, &source_dataset](TestingMetrics& m) {
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
        m.regression_truth.clear();
        m.regression_predictions.clear();
        m.regression_width = 0;
        m.regression_note.clear();
        m.row_ids.clear();
        m.row_files.clear();
        m.source_dataset = source_dataset;
        m.class_probabilities.clear();
        m.probability_classes = 0;
        m.roc.reset();
        m.pr.reset();
        m.curves_note.clear();
        m.kept_inputs.clear();
        m.kept_input_bytes = 0;
        m.layer_features.clear();
        m.layer_features_note.clear();
    });
    dropped_feature_layers_.clear();
    feature_layer_of_module_.clear();

    // Create the test batcher using the dataset type that was trained.
    std::unique_ptr<IBatcher> external_test_batcher;
    std::unique_ptr<ArrowDatasetBatcher> arrow_test_batcher;
    std::unique_ptr<ParquetArrowBatcher> parquet_test_batcher;

    if (use_arrow_dataset_) {
        TrainingConfiguration test_config =
            ConfigureTestDatasetScope(config_, dataset_scope_);
        auto batchers = BuildArrowTrainingBatchers(
            test_config, arrow_dataset_, arrow_label_column_, batch_size);
        ApplyLabelClassNames(batchers.label_classes);
        arrow_test_batcher =
            dataset_scope_ == TestDatasetScope::EntireProvidedDataset
                ? std::move(batchers.arrow_train)
                : std::move(batchers.arrow_test);
    } else if (use_parquet_dataset_) {
        TrainingConfiguration test_config =
            ConfigureTestDatasetScope(config_, dataset_scope_);
        auto batchers = BuildParquetTrainingBatchers(
            test_config, parquet_dataset_, parquet_label_column_, batch_size);
        ApplyLabelClassNames(batchers.label_classes);
        parquet_test_batcher =
            dataset_scope_ == TestDatasetScope::EntireProvidedDataset
                ? std::move(batchers.parquet_train)
                : std::move(batchers.parquet_test);
    } else if (external_source_.make_batcher) {
        external_test_batcher = external_source_.make_batcher(config_, batch_size);
        if (!external_test_batcher) {
            throw std::runtime_error("The test dataset could not be prepared");
        }
    } else {
        throw std::runtime_error("TestExecutor has no test dataset");
    }

    size_t total_batches = 0;
    if (use_arrow_dataset_) {
        total_batches = arrow_test_batcher ? arrow_test_batcher->GetNumBatches() : 0;
    } else if (use_parquet_dataset_) {
        total_batches = parquet_test_batcher ? parquet_test_batcher->GetNumBatches() : 0;
    } else {
        total_batches = external_test_batcher ? external_test_batcher->GetNumBatches() : 0;
    }
    UpdateMetrics([total_batches, this](TestingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.batch_limit = batch_limit_;
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
            "non-empty test split and that Train > Test uses the same effective "
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

    if (config_.metric_sampling != MetricSampling::None) {
        IBatcher* rows = use_arrow_dataset_ ? static_cast<IBatcher*>(arrow_test_batcher.get())
                         : use_parquet_dataset_ ? static_cast<IBatcher*>(parquet_test_batcher.get())
                                                : external_test_batcher.get();
        TestMetricLearning(*rows, batch_size);
    }
    while (config_.metric_sampling == MetricSampling::None) {
        if (ShouldStop()) break;

        Batch batch;
        if (use_arrow_dataset_) {
            batch = arrow_test_batcher->GetNextBatch();
        } else if (use_parquet_dataset_) {
            batch = parquet_test_batcher->GetNextBatch();
        } else {
            batch = external_test_batcher->GetNextBatch();
        }

        if (!batch.IsValid()) break;

        if (batch_limit_ > 0 && batch_num >= batch_limit_) {
            UpdateMetrics([](TestingMetrics& m) { m.partial = true; });
            spdlog::info("TestExecutor: quick test stopped after {} batches", batch_num);
            break;
        }

        batch_num++;

        // Process batch
        WatchLayerFeatures(true);
        const Tensor predictions = ProcessBatch(batch);
        WatchLayerFeatures(false);

        // Compute current loss for this batch
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
        UpdateMetrics(FinishClassifierMetrics);
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

void TestExecutor::WatchLayerFeatures(bool on) {
    SequentialModel* stack = model_ ? model_->AsSequentialModel() : nullptr;
    if (!stack || UsesContinuousTargetMetrics(config_)) return;
    if (!on) {
        stack->SetForwardObserver({});
        return;
    }
    if (feature_layer_of_module_.empty()) {
        // Each weight layer, and the activation right after it when there is one.
        static const char* const kActivations[] = {"ReLU", "LeakyReLU", "GELU", "Tanh", "Sigmoid", "ELU",
                                                   "SELU", "Swish", "Mish", "Softmax"};
        for (size_t i = 0; i < stack->Size(); ++i) {
            bool weights = false;
            for (const auto& [name, tensor] : stack->GetModule(i)->GetParameters())
                weights = weights || tensor.Shape().size() >= 2;
            if (!weights) continue;
            size_t recorded = i;
            if (i + 1 < stack->Size()) {
                const std::string next = stack->GetModule(i + 1)->GetName();
                for (const char* activation : kActivations)
                    if (next.rfind(activation, 0) == 0) recorded = i + 1;
            }
            feature_layer_of_module_[recorded] = i;
        }
    }
    stack->SetForwardObserver([this](size_t index, const Module&, const Tensor& output) {
        const auto found = feature_layer_of_module_.find(index);
        if (found == feature_layer_of_module_.end()) return;
        const auto& shape = output.Shape();
        af::array rows;  // [samples, width]
        if (shape.size() == 4) {
            rows = af::moddims(af::mean(af::mean(output.GetArrayRowMajor4D(), 0), 1),
                               static_cast<dim_t>(shape[2]), static_cast<dim_t>(shape[3])).T();
        } else if (shape.size() == 3) {
            rows = af::mean(output.GetArrayRowMajor3D(), 1);
            rows = af::moddims(rows, static_cast<dim_t>(shape[0]), static_cast<dim_t>(shape[2]));
        } else if (shape.size() == 2) {
            rows = output.GetSemanticArray();
        } else {
            return;
        }
        const size_t samples = static_cast<size_t>(rows.dims(0)), width = static_cast<size_t>(rows.dims(1));
        std::vector<float> host(samples * width);
        af::flat(rows.T()).as(f32).host(host.data());  // row-major [samples, width]
        const size_t layer = found->second;
        UpdateMetrics([&](TestingMetrics& m) {
            auto it = std::find_if(m.layer_features.begin(), m.layer_features.end(),
                                   [layer](const TestingMetrics::LayerFeatures& f) { return f.layer == layer; });
            if (it == m.layer_features.end()) {
                if (dropped_feature_layers_.count(layer)) return;
                m.layer_features.push_back({layer, width, {}});
                it = std::prev(m.layer_features.end());
            }
            if (it->width != width) return;
            size_t bytes = 0;
            for (const auto& f : m.layer_features) bytes += f.values.size() * sizeof(float);
            if (bytes + host.size() * sizeof(float) > TestingMetrics::kLayerFeatureBytes) {
                m.layer_features_note += (m.layer_features_note.empty() ? "Too large to keep for the Projector: layer " : ", layer ") +
                                         std::to_string(layer);
                dropped_feature_layers_.insert(layer);
                m.layer_features.erase(it);
                return;
            }
            it->values.insert(it->values.end(), host.begin(), host.end());
        });
    });
}

Tensor TestExecutor::ProcessBatch(const Batch& batch) {
    // Forward pass
    Tensor predictions = Forward(batch.data);

    const float* pred_data = predictions.Data<float>();
    const float* target_data = batch.labels.Data<float>();

    const size_t output_width = config_.output_size;
    // The samples' dataset rows; a batch without them leaves row_ids shorter
    // than the samples, which readers take as "rows not known".
    if (batch.row_ids.size() == batch.size) {
        std::vector<std::string> files;
        if (external_source_.row_file) {
            files.reserve(batch.size);
            for (const int64_t row : batch.row_ids) files.push_back(external_source_.row_file(row));
        }
        UpdateMetrics([&batch, &files](TestingMetrics& m) {
            m.row_ids.insert(m.row_ids.end(), batch.row_ids.begin(), batch.row_ids.end());
            m.row_files.insert(m.row_files.end(), std::make_move_iterator(files.begin()),
                               std::make_move_iterator(files.end()));
        });
    }
    if (UsesContinuousTargetMetrics(config_)) {
        const size_t value_count = batch.size * output_width;
        regression_metrics_.Add(
            pred_data, target_data, value_count, output_width);
        UpdateMetrics([this, &batch, value_count, pred_data, target_data, output_width](TestingMetrics& m) {
            m.total_samples += static_cast<int>(batch.size);
            m.total_target_values += value_count;
            m.test_mae = regression_metrics_.Mae();
            m.test_rmse = regression_metrics_.Rmse();
            // Kept up to kMaxRegressionValues per side (about 80 MB each);
            // past that the per-row table is skipped.
            constexpr size_t kMaxRegressionValues = 20'000'000;
            if (!m.regression_note.empty()) return;
            if (m.regression_truth.size() + value_count > kMaxRegressionValues) {
                m.regression_truth.clear();
                m.regression_predictions.clear();
                m.regression_note = "Per-row results skipped: the test set has too many values to keep";
                return;
            }
            // In the target's own units, like MAE / RMSE: a scaled target
            // (StandardScaler) is scaled back.
            const RegressionTargetTransform& transform = config_.regression_target_transform;
            m.regression_width = static_cast<int>(output_width);
            for (size_t i = 0; i < value_count; ++i) {
                m.regression_truth.push_back(static_cast<float>(transform.InverseValue(target_data[i], i % output_width)));
                m.regression_predictions.push_back(
                    static_cast<float>(transform.InverseValue(pred_data[i], i % output_width)));
            }
        });
        return predictions;
    }

    const auto decision_mode =
        ClassificationDecisionModeForLoss(config_.loss_type);

    // Inputs Explain can use: not token ids (an Embedding's input).
    const bool token_ids = !config_.layers.empty() && (config_.layers.front().type == gui::NodeType::Embedding ||
                                                       config_.layers.front().type == gui::NodeType::Concatenate);
    const bool keep_inputs = !token_ids && batch.size > 0 && batch.data.NumElements() % batch.size == 0;
    const size_t sample_width = batch.size > 0 ? batch.data.NumElements() / batch.size : 0;
    const float* input_data = nullptr;  // read from the device once, when a sample is kept

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

        // Class probabilities for the ROC / PR curves
        std::vector<float> probabilities;
        const size_t classes = ClassificationClassProbabilities(
            sample_pred, output_width, decision_mode, probabilities);

        // Update metrics
        UpdateMetrics([&, pred_class, true_class, confidence, classes](TestingMetrics& m) {
            // Kept up to kMaxCurveValues (about 80 MB); past that the curves
            // are skipped rather than the test running out of memory.
            constexpr size_t kMaxCurveValues = 20'000'000;
            if (m.curves_note.empty()) {
                if (m.class_probabilities.size() + classes > kMaxCurveValues) {
                    m.class_probabilities.clear();
                    m.curves_note = "ROC / PR skipped: the test set has too many scores to keep";
                } else {
                    m.probability_classes = static_cast<int>(classes);
                    m.class_probabilities.insert(m.class_probabilities.end(), probabilities.begin(),
                                                 probabilities.end());
                }
            }
            const int sample = m.total_samples;
            if (keep_inputs && (pred_class != true_class || sample < TestingMetrics::kKeptFirstSamples) &&
                m.kept_input_bytes + sample_width * sizeof(float) <= TestingMetrics::kKeptInputBytes) {
                if (!input_data) input_data = batch.data.Data<float>();
                if (input_data) {
                    const float* row = input_data + b * sample_width;
                    m.kept_inputs.push_back({sample, std::vector<float>(row, row + sample_width)});
                    m.kept_input_bytes += sample_width * sizeof(float);
                }
            }
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
    return predictions;
}

void TestExecutor::TestMetricLearning(IBatcher& rows, int batch_size) {
    // The loss over the builder's pairs / triplets, as in training.
    MetricBatchSampler sampler(rows, config_.metric_sampling, config_.metric_mining,
                               static_cast<uint64_t>(std::max(config_.dataloader_seed, 0)));
    double loss_sum = 0.0;
    int loss_batches = 0;
    sampler.Reset();
    while (!ShouldStop() && !sampler.IsEpochComplete()) {
        if (batch_limit_ > 0 && loss_batches >= batch_limit_) break;
        Batch batch = sampler.GetNextBatch();
        if (!batch.IsValid()) break;
        const Tensor embeddings = Forward(batch.data);
        loss_sum += ComputeLoss(embeddings, batch.labels);
        ++loss_batches;
    }

    // 1-NN classification + the Pair / Retrieval Metrics over the plain rows.
    MetricEvaluationSpec spec = MetricEvaluationSpecFor(config_);
    spec.nearest_neighbours = true;
    if (batch_limit_ > 0) {
        spec.max_rows = std::min(spec.max_rows, static_cast<size_t>(batch_limit_) * static_cast<size_t>(batch_size));
    }
    const MetricEvaluation evaluation =
        EvaluateMetricLearning(rows, [this](const Batch& batch) { return Forward(batch.data); }, spec);
    int64_t max_class = 0;
    for (int64_t id : evaluation.class_ids) max_class = std::max(max_class, id);
    const int num_classes = static_cast<int>(max_class) + 1;

    UpdateMetrics([&](TestingMetrics& m) {
        m.metric_learning_mode = true;
        m.pair_metrics = evaluation.pair;
        m.retrieval_metrics = evaluation.retrieval;
        m.partial = evaluation.truncated;
        m.test_loss = loss_batches > 0 ? static_cast<float>(loss_sum / loss_batches) : 0.0f;
        m.current_batch = loss_batches;
        m.confusion_matrix.Resize(num_classes);
        m.per_class_metrics.assign(static_cast<size_t>(num_classes), ClassMetrics{});
        for (int i = 0; i < num_classes; ++i) {
            m.per_class_metrics[static_cast<size_t>(i)].class_id = i;
            m.per_class_metrics[static_cast<size_t>(i)].class_name =
                static_cast<size_t>(i) < external_source_.class_names.size()
                    ? external_source_.class_names[static_cast<size_t>(i)]
                    : "Class " + std::to_string(i);
        }
        for (size_t i = 0; i < evaluation.nearest_classes.size(); ++i) {
            const int truth = static_cast<int>(evaluation.class_ids[i]);
            const int predicted = static_cast<int>(evaluation.nearest_classes[i]);
            m.total_samples++;
            if (predicted == truth) m.correct_predictions++;
            m.predictions.push_back(predicted);
            m.ground_truth.push_back(truth);
            m.confidences.push_back(1.0f);
            m.confusion_matrix.Add(truth, predicted);
        }
    });
    spdlog::info("TestExecutor: metric-learning test: 1-NN accuracy over {} rows; {}",
                 evaluation.nearest_classes.size(), DescribeMetricEvaluation(evaluation));
}

void TestExecutor::ApplyLabelClassNames(const std::vector<std::string>& names) {
    // A text label column's classes name the results (TOFIX145); an external
    // test source's own names win.
    if (names.empty() || !external_source_.class_names.empty()) return;
    UpdateMetrics([&names](TestingMetrics& m) {
        m.confusion_matrix.class_names = names;
        for (size_t i = 0; i < m.per_class_metrics.size() && i < names.size(); ++i)
            m.per_class_metrics[i].class_name = names[i];
    });
}

Tensor TestExecutor::Forward(const Tensor& input) {
    if (!model_) {
        spdlog::error("TestExecutor::Forward: Model not initialized");
        return Tensor();
    }
    return UsesSpatialSequentialInput(config_)
        ? model_->Forward(SpatialBatchFromRows(input, config_.input_shape))
        : model_->Forward(input);
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
