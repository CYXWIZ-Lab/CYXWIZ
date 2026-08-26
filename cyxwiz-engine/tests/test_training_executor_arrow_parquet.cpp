#include "../src/core/arrow_dataset.h"
#include "../src/core/classification_decision.h"
#include "../src/core/debug_run_paths.h"
#include "../src/core/model_builder.h"
#include "../src/core/ner_sequence_builder.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/preprocessing_state.h"
#include "../src/core/runtime_log_store.h"
#include "../src/core/sequence_batcher.h"
#include "../src/core/sequence_tag_metrics.h"
#include "../src/core/sequence_training_step.h"
#include "../src/core/sequence_vocabulary.h"
#include "../src/core/test_executor.h"
#include "../src/core/training_batcher_setup.h"
#include "../src/core/training_executor.h"
#include "../src/core/training_trace_collector.h"
#include "../src/core/execution_device_context.h"
#include "../src/core/execution_device_preferences.h"
#include "../src/plugin/interfaces/i_training_hook.h"
#include "../src/plugin/registries/plugin_training_hook_manager.h"
#include "route_qualification_test_fixture.h"
#include "algorithms/arrayfire_backend_utils.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using json = nlohmann::json;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

void CheckNear(double actual,
               double expected,
               double tolerance,
               const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << message << " actual=" << actual
                  << " expected=" << expected << '\n';
        std::exit(1);
    }
}

#ifndef NDEBUG
constexpr const char* kForceFallbackEnv =
    "CYXWIZ_TEST_FORCE_ARRAYFIRE_FALLBACK";

void SetEnvVar(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void ClearEnvVar(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value)
        : name_(name) {
        const char* previous = std::getenv(name);
        if (previous != nullptr) {
            had_previous_ = true;
            previous_ = previous;
        }
        if (value == nullptr) {
            ClearEnvVar(name_);
        } else {
            SetEnvVar(name_, value);
        }
    }

    ~ScopedEnvVar() {
        if (had_previous_) {
            SetEnvVar(name_, previous_.c_str());
        } else {
            ClearEnvVar(name_);
        }
    }

private:
    const char* name_;
    bool had_previous_ = false;
    std::string previous_;
};
#endif

std::shared_ptr<arrow::Array> FinishFloatArray(const std::vector<float>& values) {
    arrow::FloatBuilder builder;
    for (float value : values) {
        auto status = builder.Append(value);
        Check(status.ok(), status.ToString());
    }

    std::shared_ptr<arrow::Array> array;
    auto status = builder.Finish(&array);
    Check(status.ok(), status.ToString());
    return array;
}

std::shared_ptr<arrow::Table> MakeTrainingTable() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::float32()),
    });

    return arrow::Table::Make(
        schema,
        {
            FinishFloatArray({0.0f, 0.1f, 0.9f, 1.0f, 0.2f, 0.8f}),
            FinishFloatArray({0.0f, 0.2f, 0.8f, 1.0f, 0.1f, 0.9f}),
            FinishFloatArray({0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f}),
        },
        6);
}

json LoadTrainingCoreFixture(const std::filesystem::path& executable_path,
                             const char* explicit_path = nullptr) {
    const auto fixture_path = explicit_path != nullptr
        ? std::filesystem::path(explicit_path)
        : std::filesystem::absolute(executable_path).parent_path() /
              "computation_truth_fixtures" / "training_core_pytorch.json";
    std::ifstream stream(fixture_path);
    Check(stream.is_open(),
          "unable to open PyTorch fixture: " + fixture_path.string());
    json fixture;
    stream >> fixture;
    Check(fixture.value("schema_version", 0) == 1 &&
              fixture.at("oracle").value("name", "") == "PyTorch" &&
              fixture.at("oracle").value("device", "") == "cpu",
          "gradient accumulation fixture must declare a PyTorch CPU oracle");
    return fixture;
}

std::vector<size_t> FixtureShape(const json& value) {
    return value.at("shape").get<std::vector<size_t>>();
}

std::vector<float> FixtureFloats(const json& value) {
    return value.at("values").get<std::vector<float>>();
}

void CheckParameterFixture(const cyxwiz::Tensor& actual,
                           const json& expected,
                           double tolerance,
                           const std::string& context) {
    const auto expected_shape = FixtureShape(expected);
    const auto expected_values = FixtureFloats(expected);
    Check(actual.Shape() == expected_shape,
          context + " shape should match PyTorch");
    Check(actual.GetDataType() == cyxwiz::DataType::Float32,
          context + " dtype should remain Float32");
    const float* actual_values = actual.ReadData<float>();
    Check(actual.NumElements() == expected_values.size(),
          context + " element count should match PyTorch");
    for (size_t i = 0; i < expected_values.size(); ++i) {
        CheckNear(actual_values[i], expected_values[i], tolerance,
                  context + " value " + std::to_string(i));
    }
}

std::shared_ptr<arrow::Table> MakeUnevenValidationTable() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::float32()),
    });

    return arrow::Table::Make(
        schema,
        {
            FinishFloatArray(
                {0.0f, 0.1f, 0.2f, 0.3f, 0.7f, 0.8f, 0.9f, 1.0f}),
            FinishFloatArray(
                {0.0f, 0.2f, 0.1f, 0.3f, 0.7f, 0.9f, 0.8f, 1.0f}),
            FinishFloatArray(
                {0.0f, 0.0f, -100.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f}),
        },
        8);
}

std::shared_ptr<arrow::Table> MakeImbalancedTrainingTable();
std::vector<size_t> CollectBatchSizes(cyxwiz::IBatcher& batcher);
std::vector<float> CollectBatchFeatures(cyxwiz::IBatcher& batcher);
std::vector<float> CollectBatchLabels(cyxwiz::IBatcher& batcher);

std::shared_ptr<arrow::Table> MakeRegressionTable() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("target", arrow::float32()),
        arrow::field("target_1", arrow::float32()),
    });

    return arrow::Table::Make(
        schema,
        {
            FinishFloatArray({0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f}),
            FinishFloatArray({1.0f, 0.8f, 0.6f, 0.4f, 0.2f, 0.0f}),
            FinishFloatArray({1.5f, 1.7f, 1.9f, 2.1f, 2.3f, 2.5f}),
            FinishFloatArray({-0.5f, -0.1f, 0.3f, 0.7f, 1.1f, 1.5f}),
        },
        6);
}

void WriteParquetWithRowGroupSize(const cyxwiz::ArrowDataset& dataset,
                                  const std::string& path,
                                  int64_t row_group_size) {
    auto table = dataset.GetArrowTable();
    Check(table != nullptr, "source table should exist for row-group parquet write");
    auto output = arrow::io::FileOutputStream::Open(path);
    Check(output.ok(), output.status().ToString());
    auto status = parquet::arrow::WriteTable(*table,
                                             arrow::default_memory_pool(),
                                             *output,
                                             row_group_size);
    Check(status.ok(), status.ToString());
}

cyxwiz::TrainingConfiguration MakeConfig(const std::filesystem::path& checkpoint_dir) {
    cyxwiz::TrainingConfiguration config;
    config.dataset_name = "training_executor_parity";
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.train_ratio = 0.67f;
    config.shuffle = false;
    config.num_workers = 0;
    config.batch_size = 2;
    config.epochs = 1;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

std::shared_ptr<cyxwiz::ArrowDataset> MakeGradientAccumulationDataset(
    const json& test_case) {
    const auto input_shape = FixtureShape(test_case.at("input"));
    const auto input_values = FixtureFloats(test_case.at("input"));
    const auto targets =
        test_case.at("targets").at("values").get<std::vector<int64_t>>();
    Check(input_shape.size() == 2 && input_shape[1] == 2 &&
              input_shape[0] == targets.size(),
          "gradient accumulation fixture must contain [N,2] inputs and N targets");

    std::vector<float> x0(targets.size());
    std::vector<float> x1(targets.size());
    std::vector<float> labels(targets.size());
    for (size_t row = 0; row < targets.size(); ++row) {
        x0[row] = input_values[row * 2];
        x1[row] = input_values[row * 2 + 1];
        labels[row] = static_cast<float>(targets[row]);
    }
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::float32()),
    });
    return std::make_shared<cyxwiz::ArrowDataset>(
        arrow::Table::Make(
            schema,
            {FinishFloatArray(x0), FinishFloatArray(x1),
             FinishFloatArray(labels)},
            static_cast<int64_t>(targets.size())),
        "gradient_accumulation_" + test_case.at("name").get<std::string>());
}

void TestGradientAccumulationParityCase(
    const json& test_case,
    const std::filesystem::path& work_dir) {
    Check(test_case.value("operation", "").find("torch.nn.Linear") == 0,
          "gradient accumulation case must use the PyTorch end-to-end oracle");
    const std::string loss_reduction =
        test_case.value("loss_reduction", "");
    Check(loss_reduction == "mean" || loss_reduction == "sum",
          "gradient accumulation case must use scalar CrossEntropy");
    const auto dataset = MakeGradientAccumulationDataset(test_case);
    auto config = MakeConfig(
        work_dir / test_case.at("name").get<std::string>());
    config.dataset_name = test_case.at("name").get<std::string>();
    config.learning_rate = test_case.at("learning_rate").get<float>();
    config.batch_size = test_case.at("microbatch_size").get<int>();
    config.grad_accum_steps = test_case.at("grad_accum_steps").get<int>();
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.has_data_split = true;
    config.shuffle = false;
    config.log_interval = 0;
    config.forbid_native_cpu_fallback = true;
    config.loss_params["reduction"] = loss_reduction;
    const auto class_weights =
        test_case.value("class_weights", std::vector<float>{});
    if (!class_weights.empty()) {
        std::string serialized_weights = "[";
        for (size_t i = 0; i < class_weights.size(); ++i) {
            if (i > 0) serialized_weights += ",";
            serialized_weights += std::to_string(class_weights[i]);
        }
        serialized_weights += "]";
        config.loss_params["class_weights"] = serialized_weights;
    }
    config.loss_params["ignore_index"] =
        std::to_string(test_case.value("ignore_index", -100));
    config.loss_params["label_smoothing"] =
        std::to_string(test_case.value("label_smoothing", 0.0f));

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    const auto& expected_steps = test_case.at("expected_steps");
    const double tolerance =
        test_case.at("tolerance").at("atol").get<double>();
    size_t observed_step = 0;
    int batch_callback_count = 0;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        1,
        config.batch_size,
        [&](int, int batch, int, float, float) {
            ++batch_callback_count;
            if (observed_step >= expected_steps.size() ||
                batch != expected_steps[observed_step]
                             .at("ending_microbatch").get<int>()) {
                return;
            }
            const auto current = executor.GetMetrics();
            Check(current.optimizer_step_count ==
                      static_cast<int>(observed_step + 1),
                  "optimizer step count must advance exactly at the PyTorch window boundary");
            auto* model = executor.GetModel();
            Check(model != nullptr,
                  "accumulation parity callback should expose the active model");
            const auto actual = model->GetParameters();
            const auto& expected = expected_steps[observed_step];
            CheckParameterFixture(
                actual.at("layer0.bias"), expected.at("bias"), tolerance,
                test_case.at("name").get<std::string>() +
                    " step bias " + std::to_string(observed_step + 1));
            ++observed_step;
        },
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
        });

    Check(observed_step == expected_steps.size(),
          "executor must observe every PyTorch optimizer boundary");
    Check(batch_callback_count ==
              static_cast<int>(test_case.at("targets").at("shape")[0]
                                   .get<size_t>() + config.batch_size - 1) /
                  config.batch_size,
          "executor must consume every configured microbatch exactly once");
    Check(final_metrics.optimizer_step_count ==
              test_case.at("expected_optimizer_step_count").get<int>(),
          "final optimizer step count must match PyTorch");
    CheckNear(final_metrics.train_accuracy,
              test_case.at("expected_train_accuracy").get<double>(),
              1e-6,
              test_case.at("name").get<std::string>() +
                  " train accuracy must exclude ignored targets");
    CheckNear(final_metrics.train_loss,
              test_case.at("expected_train_loss").get<double>(),
              tolerance,
              test_case.at("name").get<std::string>() +
                  " train loss must preserve reduction semantics");
    Check(final_metrics.current_epoch == 1 &&
              final_metrics.last_executed_epoch == 1 &&
              final_metrics.terminal_status == "completed" &&
              final_metrics.terminal_reason == "completed_all_epochs",
          "gradient accumulation parity run must preserve terminal lifecycle truth");
    auto* model = executor.GetModel();
    Check(model != nullptr,
          "completed accumulation parity run should retain its active model");
    const auto final_parameters = model->GetParameters();
    CheckParameterFixture(
        final_parameters.at("layer0.bias"),
        test_case.at("expected").at("bias"), tolerance,
        test_case.at("name").get<std::string>() + " final bias");
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.native_cpu_fallback_count == 0,
          "gradient accumulation parity must remain ArrayFire-resident");
}

void TestGradientAccumulationPyTorchParity(
    const json& cases,
    const std::filesystem::path& work_dir) {
    const auto& matrix =
        cases.at("gradient_accumulation_linear_ce_sgd_f32");
    Check(matrix.is_array() && matrix.size() == 4,
          "gradient accumulation fixture matrix must cover mean and sum lifecycle cases");
    Check(std::count_if(
              matrix.begin(), matrix.end(), [](const json& test_case) {
                  return test_case.value("loss_reduction", "") == "sum";
              }) == 1,
          "gradient accumulation fixture matrix must contain one sum-reduction oracle");
    for (const auto& test_case : matrix) {
        TestGradientAccumulationParityCase(test_case, work_dir);
    }
}

cyxwiz::TrainingConfiguration MakeRegressionConfig(
    const std::filesystem::path& checkpoint_dir) {
    auto config = MakeConfig(checkpoint_dir);
    config.dataset_name = "training_executor_regression";
    config.output_size = 2;
    config.loss_type = gui::NodeType::MSELoss;
    config.target.required_by_objective = true;
    config.target.origin = cyxwiz::TargetOrigin::DatasetColumn;
    config.target.value_kind = cyxwiz::TargetValueKind::Continuous;
    config.target.primary_column = "target";
    config.target.width = 2;
    config.layers.front().units = 2;
    return config;
}

cyxwiz::TrainingMetrics RunOneEpochForDataContract(
    cyxwiz::TrainingConfiguration config,
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset) {
    cyxwiz::TrainingExecutor executor(std::move(config), dataset, "label");
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
        });
    return final_metrics;
}

struct FullBatchMetricOracle {
    float loss = 0.0f;
    float accuracy = 0.0f;
    float mae = 0.0f;
    float rmse = 0.0f;
};

FullBatchMetricOracle EvaluateCurrentModelAsOneBatch(
    cyxwiz::TrainingExecutor& executor,
    const cyxwiz::TrainingConfiguration& config,
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::string& label_column,
    cyxwiz::BatcherPhase phase = cyxwiz::BatcherPhase::Train) {
    auto full_batch_config = config;
    full_batch_config.batch_size =
        static_cast<int>(dataset->GetNumRows());
    auto batchers = cyxwiz::BuildArrowTrainingBatchers(
        full_batch_config,
        dataset,
        label_column,
        full_batch_config.batch_size);
    cyxwiz::IBatcher* batcher = nullptr;
    if (phase == cyxwiz::BatcherPhase::Val) {
        batcher = batchers.val;
    } else if (phase == cyxwiz::BatcherPhase::Test) {
        batcher = batchers.test;
    } else {
        batcher = batchers.train;
    }
    Check(batcher != nullptr,
          "full-batch metric oracle should resolve the requested role");
    const size_t expected_observations = batcher->GetNumSamples();
    auto batch = batcher->GetNextBatch();
    Check(batch.IsValid() &&
              batch.size == expected_observations,
          "full-batch metric oracle should contain every requested-role row");

    auto* model = executor.GetModel();
    Check(model != nullptr,
          "full-batch metric oracle requires the retained trained model");
    const auto predictions = model->Forward(batch.data);
    auto loss = cyxwiz::BuildLossFromConfig(config);
    Check(loss != nullptr, "full-batch metric oracle should build the loss");
    const auto loss_tensor = loss->Forward(predictions, batch.labels);

    FullBatchMetricOracle oracle;
    oracle.loss = loss_tensor.ReadData<float>()[0];
    if (cyxwiz::UsesRegressionMetrics(config)) {
        const float* prediction_values = predictions.ReadData<float>();
        const float* target_values = batch.labels.ReadData<float>();
        double absolute_error_sum = 0.0;
        double squared_error_sum = 0.0;
        for (size_t index = 0; index < predictions.NumElements(); ++index) {
            const double error = static_cast<double>(prediction_values[index]) -
                static_cast<double>(target_values[index]);
            absolute_error_sum += std::abs(error);
            squared_error_sum += error * error;
        }
        const double count = static_cast<double>(predictions.NumElements());
        oracle.mae = static_cast<float>(absolute_error_sum / count);
        oracle.rmse = static_cast<float>(std::sqrt(squared_error_sum / count));
    } else {
        const float* prediction_values = predictions.ReadData<float>();
        if (batch.labels.GetDataType() == cyxwiz::DataType::Int32 ||
            batch.labels.GetDataType() == cyxwiz::DataType::Int64) {
            const auto loss_config = cyxwiz::ResolveLossConfiguration(config);
            const int32_t* labels32 =
                batch.labels.GetDataType() == cyxwiz::DataType::Int32
                    ? batch.labels.ReadData<int32_t>()
                    : nullptr;
            const int64_t* labels64 =
                batch.labels.GetDataType() == cyxwiz::DataType::Int64
                    ? batch.labels.ReadData<int64_t>()
                    : nullptr;
            size_t correct = 0;
            size_t valid = 0;
            for (size_t row = 0; row < batch.size; ++row) {
                const int64_t target = labels32
                    ? static_cast<int64_t>(labels32[row])
                    : labels64[row];
                if (loss_config.ignore_index_applicable &&
                    target == loss_config.ignore_index) {
                    continue;
                }
                if (cyxwiz::ClassificationPredictedClass(
                        prediction_values + row * config.output_size,
                        config.output_size,
                        cyxwiz::ClassificationDecisionModeForLoss(
                            config.loss_type)) == target) {
                    ++correct;
                }
                ++valid;
            }
            oracle.accuracy = valid > 0
                ? static_cast<float>(correct) / static_cast<float>(valid)
                : 0.0f;
        } else {
            const float* target_values = batch.labels.ReadData<float>();
            oracle.accuracy = cyxwiz::ClassificationAccuracy(
                prediction_values,
                target_values,
                batch.size,
                config.output_size,
                cyxwiz::ClassificationDecisionModeForLoss(config.loss_type));
        }
    }
    return oracle;
}

void TestUnevenFinalBatchMetricAggregation(
    const std::filesystem::path& work_dir) {
    const auto classification_dataset =
        std::make_shared<cyxwiz::ArrowDataset>(
            MakeTrainingTable(), "uneven_classification_metrics");
    auto classification_config = MakeConfig(
        work_dir / "uneven-classification-metrics");
    classification_config.train_ratio = 1.0f;
    classification_config.val_ratio = 0.0f;
    classification_config.test_ratio = 0.0f;
    classification_config.has_data_split = true;
    classification_config.batch_size = 4;
    classification_config.learning_rate = 0.0f;
    classification_config.log_interval = 0;
    classification_config.forbid_native_cpu_fallback = true;

    cyxwiz::TrainingExecutor classification_executor(
        classification_config, classification_dataset, "label");
    cyxwiz::TrainingMetrics classification_metrics;
    classification_executor.Train(
        1,
        classification_config.batch_size,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            classification_metrics = metrics;
        });
    const auto classification_trace =
        cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(classification_trace.native_cpu_fallback_count == 0,
          "uneven classification aggregation should remain ArrayFire-resident");
    const auto classification_oracle = EvaluateCurrentModelAsOneBatch(
        classification_executor,
        classification_config,
        classification_dataset,
        "label");
    Check(classification_metrics.total_batches == 2 &&
              classification_metrics.train_sample_count == 6,
          "classification fixture should execute a 4+2 uneven epoch");
    CheckNear(classification_metrics.train_loss,
              classification_oracle.loss,
              1e-5,
              "classification epoch loss should equal the full-batch mean");
    CheckNear(classification_metrics.train_accuracy,
              classification_oracle.accuracy,
              1e-6,
              "classification epoch accuracy should use all six samples");
    Check(classification_metrics.loss_history.size() == 1 &&
              classification_metrics.accuracy_history.size() == 1,
          "classification history should publish one aggregated epoch point");
    CheckNear(classification_metrics.loss_history.front(),
              classification_oracle.loss,
              1e-5,
              "classification history should retain the full-epoch mean");

    auto weighted_config = classification_config;
    weighted_config.checkpoint_dir =
        (work_dir / "uneven-weighted-smoothed-classification").string();
    weighted_config.loss_params["class_weights"] = "[1.0, 4.0]";
    weighted_config.loss_params["label_smoothing"] = "0.2";
    cyxwiz::TrainingExecutor weighted_executor(
        weighted_config, classification_dataset, "label");
    cyxwiz::TrainingMetrics weighted_metrics;
    weighted_executor.Train(
        1,
        weighted_config.batch_size,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            weighted_metrics = metrics;
        });
    const auto weighted_trace =
        cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(weighted_trace.native_cpu_fallback_count == 0,
          "weighted smoothed aggregation should remain ArrayFire-resident");
    const auto weighted_oracle = EvaluateCurrentModelAsOneBatch(
        weighted_executor,
        weighted_config,
        classification_dataset,
        "label");
    Check(weighted_metrics.total_batches == 2 &&
              weighted_metrics.train_sample_count == 6,
          "weighted smoothed fixture should execute a 4+2 uneven epoch");
    CheckNear(weighted_metrics.train_loss,
              weighted_oracle.loss,
              1e-5,
              "weighted smoothed epoch loss should equal the full-batch mean");
    Check(weighted_metrics.loss_history.size() == 1,
          "weighted smoothed history should publish one epoch point");
    CheckNear(weighted_metrics.loss_history.front(),
              weighted_oracle.loss,
              1e-5,
              "weighted smoothed history should retain the full-epoch mean");

    const auto validation_dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeUnevenValidationTable(), "uneven_validation_metrics");
    auto validation_config = MakeConfig(
        work_dir / "uneven-validation-metrics");
    validation_config.train_ratio = 0.25f;
    validation_config.val_ratio = 0.75f;
    validation_config.test_ratio = 0.0f;
    validation_config.has_data_split = true;
    validation_config.batch_size = 4;
    validation_config.learning_rate = 0.0f;
    validation_config.log_interval = 0;
    validation_config.forbid_native_cpu_fallback = true;
    validation_config.loss_params["ignore_index"] = "-100";

    cyxwiz::TrainingExecutor validation_executor(
        validation_config, validation_dataset, "label");
    cyxwiz::TrainingMetrics validation_metrics;
    validation_executor.Train(
        1,
        validation_config.batch_size,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            validation_metrics = metrics;
        });
    const auto validation_trace =
        cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(validation_trace.native_cpu_fallback_count == 0,
          "uneven validation aggregation should remain ArrayFire-resident");
    const auto validation_oracle = EvaluateCurrentModelAsOneBatch(
        validation_executor,
        validation_config,
        validation_dataset,
        "label",
        cyxwiz::BatcherPhase::Val);
    Check(validation_metrics.val_sample_count == 6 &&
              validation_metrics.has_validation_metrics,
          "validation fixture should evaluate a 4+2 uneven role");
    CheckNear(validation_metrics.val_loss,
              validation_oracle.loss,
              1e-5,
              "validation loss should equal the full-role batch mean");
    CheckNear(validation_metrics.val_accuracy,
              validation_oracle.accuracy,
              1e-6,
              "validation accuracy should use five valid targets and exclude "
              "the ignored row");
    Check(validation_metrics.val_loss_history.size() == 1 &&
              validation_metrics.val_accuracy_history.size() == 1,
          "validation history should publish one aggregated role point");

    const auto regression_dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeRegressionTable(), "uneven_regression_metrics");
    auto regression_config = MakeRegressionConfig(
        work_dir / "uneven-regression-metrics");
    regression_config.train_ratio = 1.0f;
    regression_config.val_ratio = 0.0f;
    regression_config.test_ratio = 0.0f;
    regression_config.has_data_split = true;
    regression_config.batch_size = 4;
    regression_config.learning_rate = 0.0f;
    regression_config.log_interval = 0;
    regression_config.forbid_native_cpu_fallback = true;

    cyxwiz::TrainingExecutor regression_executor(
        regression_config, regression_dataset, "target");
    cyxwiz::TrainingMetrics regression_metrics;
    regression_executor.Train(
        1,
        regression_config.batch_size,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            regression_metrics = metrics;
        });
    const auto regression_trace =
        cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(regression_trace.native_cpu_fallback_count == 0,
          "uneven regression aggregation should remain ArrayFire-resident");
    const auto regression_oracle = EvaluateCurrentModelAsOneBatch(
        regression_executor,
        regression_config,
        regression_dataset,
        "target");
    Check(regression_metrics.total_batches == 2 &&
              regression_metrics.train_sample_count == 6,
          "regression fixture should execute a 4+2 uneven epoch");
    CheckNear(regression_metrics.train_loss,
              regression_oracle.loss,
              1e-5,
              "regression epoch loss should equal the full-batch mean");
    CheckNear(regression_metrics.train_mae,
              regression_oracle.mae,
              1e-5,
              "regression epoch MAE should use all output values");
    CheckNear(regression_metrics.train_rmse,
              regression_oracle.rmse,
              1e-5,
              "regression epoch RMSE should use all output values");
}

void TestWeightedSamplerEpochAndUpdateCount(
    const std::filesystem::path& work_dir) {
    auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeImbalancedTrainingTable(), "weighted_sampler_contract");
    auto config = MakeConfig(work_dir / "weighted-sampler");
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.has_data_split = true;
    config.batch_size = 2;
    config.shuffle = false;
    config.balance_classes = true;
    config.balance_mode = "weighted_sampler";
    config.balance_target = "max";
    config.balance_seed = 23;

    auto first = cyxwiz::BuildArrowTrainingBatchers(
        config, dataset, "label", config.batch_size);
    auto second = cyxwiz::BuildArrowTrainingBatchers(
        config, dataset, "label", config.batch_size);
    Check(first.num_train_samples == 5 && second.num_train_samples == 5,
          "weighted sampling should preserve the configured five-sample "
          "epoch instead of expanding it to oversampling size");
    Check(first.train->GetNumBatches() == 3 &&
              second.train->GetNumBatches() == 3,
          "weighted sampling should preserve the baseline partial-batch count");
    const auto first_labels = CollectBatchLabels(*first.train);
    const auto second_labels = CollectBatchLabels(*second.train);
    Check(first_labels == second_labels && first_labels.size() == 10,
          "weighted sampling should be deterministic by seed and emit five "
          "two-class targets");

    auto baseline_config = config;
    baseline_config.balance_classes = false;
    baseline_config.balance_mode = "none";
    baseline_config.checkpoint_dir =
        (work_dir / "weighted-sampler-baseline").string();
    const auto baseline = RunOneEpochForDataContract(
        baseline_config, dataset);
    const auto weighted = RunOneEpochForDataContract(config, dataset);
    Check(baseline.optimizer_step_count == 3 &&
              weighted.optimizer_step_count == 3 &&
              baseline.total_batches == weighted.total_batches,
          "weighted sampling should preserve one optimizer update per "
          "baseline full/partial batch");
}

void TestArrowParquetBatchBoundaryParity(
    const std::shared_ptr<cyxwiz::ArrowDataset>& arrow_dataset,
    const std::shared_ptr<cyxwiz::ParquetBackedDataset>& parquet_dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "batch-boundary-parity");
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.has_data_split = true;
    config.batch_size = 4;
    config.shuffle = false;

    auto arrow_keep = cyxwiz::BuildArrowTrainingBatchers(
        config, arrow_dataset, "label", config.batch_size);
    auto parquet_keep = cyxwiz::BuildParquetTrainingBatchers(
        config, parquet_dataset, "label", config.batch_size);
    Check(arrow_keep.num_train_samples == 6 &&
              parquet_keep.num_train_samples == 6,
          "Arrow and Parquet should expose the same six-sample Train role");
    Check(CollectBatchSizes(*arrow_keep.train) ==
              std::vector<size_t>({4, 2}) &&
              CollectBatchSizes(*parquet_keep.train) ==
              std::vector<size_t>({4, 2}),
          "Arrow and Parquet should both keep the final partial batch");

    config.drop_last = true;
    auto arrow_drop = cyxwiz::BuildArrowTrainingBatchers(
        config, arrow_dataset, "label", config.batch_size);
    auto parquet_drop = cyxwiz::BuildParquetTrainingBatchers(
        config, parquet_dataset, "label", config.batch_size);
    Check(arrow_drop.train->GetNumBatches() == 1 &&
              parquet_drop.train->GetNumBatches() == 1,
          "Arrow and Parquet drop_last should floor the Train batch count");
    Check(CollectBatchSizes(*arrow_drop.train) ==
              std::vector<size_t>({4}) &&
              CollectBatchSizes(*parquet_drop.train) ==
              std::vector<size_t>({4}),
          "Arrow and Parquet drop_last should suppress the same partial batch");
}

void TestArrowParquetSeedDeterminism(
    const std::shared_ptr<cyxwiz::ArrowDataset>& arrow_dataset,
    const std::shared_ptr<cyxwiz::ParquetBackedDataset>& parquet_dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "seed-determinism");
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.has_data_split = true;
    config.batch_size = 2;
    config.shuffle = true;
    config.dataloader_seed = 2468;

    auto first_arrow = cyxwiz::BuildArrowTrainingBatchers(
        config, arrow_dataset, "label", config.batch_size);
    auto second_arrow = cyxwiz::BuildArrowTrainingBatchers(
        config, arrow_dataset, "label", config.batch_size);
    Check(CollectBatchFeatures(*first_arrow.train) ==
              CollectBatchFeatures(*second_arrow.train),
          "matching Arrow seeds should reproduce the complete sample order");
    Check(CollectBatchLabels(*first_arrow.train) ==
              CollectBatchLabels(*second_arrow.train),
          "matching Arrow seeds should reproduce the complete label order");

    auto first_parquet = cyxwiz::BuildParquetTrainingBatchers(
        config, parquet_dataset, "label", config.batch_size);
    auto second_parquet = cyxwiz::BuildParquetTrainingBatchers(
        config, parquet_dataset, "label", config.batch_size);
    Check(CollectBatchFeatures(*first_parquet.train) ==
              CollectBatchFeatures(*second_parquet.train),
          "matching Parquet seeds should reproduce the complete sample order");
    Check(CollectBatchLabels(*first_parquet.train) ==
              CollectBatchLabels(*second_parquet.train),
          "matching Parquet seeds should reproduce the complete label order");
}

void TestSequenceBatchContract() {
    cyxwiz::SequenceBatch empty;
    Check(!empty.IsValid(), "empty SequenceBatch should be invalid");
    Check(!empty.IsSupervised(),
          "empty SequenceBatch should not be supervised");

    const std::vector<int64_t> word_ids = {1, 2, 0, 3, 4, 0};
    const std::vector<int64_t> mask = {1, 1, 0, 1, 1, 0};
    cyxwiz::SequenceBatch inference;
    inference.word_ids =
        cyxwiz::Tensor({2, 3}, word_ids.data(), cyxwiz::DataType::Int64);
    inference.attention_mask =
        cyxwiz::Tensor({2, 3}, mask.data(), cyxwiz::DataType::Int64);
    inference.size = 2;
    inference.sequence_length = 3;
    Check(inference.IsValid(),
          "SequenceBatch with word ids should be valid");
    Check(inference.HasAttentionMask(),
          "SequenceBatch should report attention mask");
    Check(!inference.IsSupervised(),
          "SequenceBatch without tag ids should not be supervised");

    const std::vector<int64_t> tag_ids = {5, 6, -100, 7, 8, -100};
    inference.tag_ids =
        cyxwiz::Tensor({2, 3}, tag_ids.data(), cyxwiz::DataType::Int64);
    Check(inference.IsSupervised(),
          "SequenceBatch with tag ids should be supervised");
}

void TestSequenceBatcherPadsNamedPayloads() {
    std::vector<cyxwiz::SequenceSample> samples = {
        {{11, 12, 13}, {1, 2, 3}, {5, 6, 7}},
        {{21}, {4}, {8}},
        {{31, 32, 33, 34}, {9, 10, 11, 12}, {1, 2, 3, 4}},
    };

    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 2;
    config.max_sequence_length = 3;
    config.shuffle = false;
    config.create_attention_mask = true;
    config.tag_ignore_index = -100;

    cyxwiz::SequenceBatcher batcher(samples, config);
    Check(batcher.GetNumSamples() == 3,
          "SequenceBatcher should report sample count");
    Check(batcher.GetNumBatches() == 2,
          "SequenceBatcher should ceil partial final batch");

    auto batch = batcher.GetNextSequenceBatch();
    Check(batch.IsSupervised(),
          "first sequence batch should be supervised");
    Check(batch.HasPosIds(), "first sequence batch should include POS ids");
    Check(batch.HasAttentionMask(),
          "first sequence batch should include attention mask");
    Check(batch.word_ids.Shape() == std::vector<size_t>({2, 3}),
          "word_ids should be [batch, seq]");
    Check(batch.tag_ids.Shape() == std::vector<size_t>({2, 3}),
          "tag_ids should be [batch, seq]");

    const auto* words = batch.word_ids.Data<int64_t>();
    const auto* mask = batch.attention_mask.Data<int64_t>();
    const auto* tags = batch.tag_ids.Data<int64_t>();
    Check(words[0] == 11 && words[1] == 12 && words[2] == 13,
          "first row words should copy exactly");
    Check(words[3] == 21 && words[4] == 0 && words[5] == 0,
          "short row words should pad with word_pad_id");
    Check(mask[0] == 1 && mask[1] == 1 && mask[2] == 1,
          "full row mask should be all ones");
    Check(mask[3] == 1 && mask[4] == 0 && mask[5] == 0,
          "short row mask should mark padding");
    Check(tags[3] == 8 && tags[4] == -100 && tags[5] == -100,
          "short row tags should pad with ignore_index");

    auto final_batch = batcher.GetNextSequenceBatch();
    Check(final_batch.size == 1,
          "final sequence batch should keep partial batch by default");
    const auto* final_words = final_batch.word_ids.Data<int64_t>();
    const auto* final_tags = final_batch.tag_ids.Data<int64_t>();
    Check(final_words[0] == 31 && final_words[1] == 32 &&
              final_words[2] == 33,
          "long row words should truncate to max_sequence_length");
    Check(final_tags[0] == 1 && final_tags[1] == 2 && final_tags[2] == 3,
          "long row tags should truncate with words");
    Check(batcher.IsEpochComplete(),
          "sequence batcher should complete after final batch");
}

void TestSequenceBatcherDropLast() {
    std::vector<cyxwiz::SequenceSample> samples = {
        {{1}, {}, {2}},
        {{3}, {}, {4}},
        {{5}, {}, {6}},
    };

    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 2;
    config.drop_last = true;
    cyxwiz::SequenceBatcher batcher(samples, config);
    Check(batcher.GetNumBatches() == 1,
          "drop_last should floor sequence batch count");
    Check(batcher.GetNextSequenceBatch().size == 2,
          "drop_last first batch should be full");
    Check(!batcher.GetNextSequenceBatch().IsValid(),
          "drop_last should suppress partial final batch");
}

void TestSequenceBatcherSeedDeterminism() {
    const std::vector<cyxwiz::SequenceSample> samples = {
        {{11}, {}, {1}},
        {{21}, {}, {2}},
        {{31}, {}, {3}},
        {{41}, {}, {4}},
        {{51}, {}, {5}},
        {{61}, {}, {6}},
    };

    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 2;
    config.max_sequence_length = 1;
    config.shuffle = true;
    config.seed = 31415;
    cyxwiz::SequenceBatcher first(samples, config);
    cyxwiz::SequenceBatcher second(samples, config);

    auto collect_words = [](cyxwiz::SequenceBatcher& batcher) {
        std::vector<int64_t> words;
        batcher.Reset();
        while (!batcher.IsEpochComplete()) {
            const auto batch = batcher.GetNextSequenceBatch();
            Check(batch.IsValid(), "seeded sequence batch should be valid");
            const int64_t* values = batch.word_ids.ReadData<int64_t>();
            words.insert(words.end(), values,
                         values + batch.word_ids.NumElements());
        }
        return words;
    };

    Check(collect_words(first) == collect_words(second),
          "matching SequenceBatcher seeds should reproduce the complete "
          "sample order");
}

std::shared_ptr<arrow::Table> MakeImbalancedTrainingTable() {
    auto schema = arrow::schema({
        arrow::field("x0", arrow::float32()),
        arrow::field("x1", arrow::float32()),
        arrow::field("label", arrow::float32()),
    });

    return arrow::Table::Make(
        schema,
        {
            FinishFloatArray({0.0f, 0.1f, 0.2f, 0.3f, 1.0f}),
            FinishFloatArray({0.0f, 0.2f, 0.4f, 0.6f, 1.0f}),
            FinishFloatArray({0.0f, 0.0f, 0.0f, 0.0f, 1.0f}),
        },
        5);
}

std::vector<size_t> CollectBatchSizes(cyxwiz::IBatcher& batcher) {
    std::vector<size_t> sizes;
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        if (!batch.IsValid()) {
            break;
        }
        sizes.push_back(batch.size);
    }
    return sizes;
}

std::vector<float> CollectBatchFeatures(cyxwiz::IBatcher& batcher) {
    std::vector<float> features;
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        if (!batch.IsValid()) {
            break;
        }
        const float* values = batch.data.ReadData<float>();
        features.insert(features.end(), values,
                        values + batch.data.NumElements());
    }
    return features;
}

std::vector<float> CollectBatchLabels(cyxwiz::IBatcher& batcher) {
    std::vector<float> labels;
    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        auto batch = batcher.GetNextBatch();
        if (!batch.IsValid()) {
            break;
        }
        const float* values = batch.labels.ReadData<float>();
        labels.insert(labels.end(), values,
                      values + batch.labels.NumElements());
    }
    return labels;
}

void TestSequencePhaseSwitchRequiresExplicitReset() {
    const std::vector<cyxwiz::SequenceSample> samples = {
        {{1, 2}, {}, {0, 1}},
        {{3, 4}, {}, {1, 0}},
        {{5, 6}, {}, {0, 1}},
    };

    cyxwiz::SequenceBatcherConfig config;
    config.batch_size = 1;
    config.max_sequence_length = 2;
    config.shuffle = true;
    config.seed = 17;
    config.train_indices = {0, 1};
    config.val_indices = {2};

    cyxwiz::SequenceBatcher batcher(samples, config);
    while (!batcher.IsEpochComplete()) {
        (void)batcher.GetNextSequenceBatch();
    }
    Check(batcher.IsEpochComplete(),
          "sequence phase contract fixture should consume the Train phase");

    batcher.SetPhase(cyxwiz::BatcherPhase::Val);
    Check(batcher.IsEpochComplete(),
          "SequenceBatcher::SetPhase must not implicitly reset iteration");
    batcher.Reset();
    Check(!batcher.IsEpochComplete() && batcher.GetNumSamples() == 1,
          "an explicit Reset should start the selected sequence phase");
}

void TestSequenceTagMetrics() {
    const std::vector<std::string> labels = {
        "O",
        "B-PER",
        "I-PER",
        "B-LOC",
        "I-LOC",
    };

    const std::vector<float> logits = {
        0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 5.0f, 0.0f, 0.0f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 5.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 5.0f, 0.0f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        5.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 5.0f, 0.0f,
    };
    const std::vector<int64_t> gold = {
        1, 2, 0, -100,
        3, 4, 0, -100,
    };

    cyxwiz::Tensor logits_tensor(
        {2, 4, labels.size()}, logits.data(), cyxwiz::DataType::Float32);
    cyxwiz::Tensor gold_tensor({2, 4}, gold.data(), cyxwiz::DataType::Int64);
    const auto metrics = cyxwiz::ComputeSequenceTagMetricsFromLogits(
        logits_tensor, gold_tensor, labels, -100);

    Check(metrics.correct_tokens == 5,
          "token metrics should count correct non-ignored tokens");
    Check(metrics.total_tokens == 6,
          "token metrics should skip ignored padding labels");
    CheckNear(metrics.token_accuracy, 5.0 / 6.0, 1e-9,
              "token accuracy should use non-ignored denominator");
    Check(metrics.predicted_entities == 2,
          "BIO metrics should ignore entity predictions on padding labels");
    Check(metrics.gold_entities == 2,
          "BIO metrics should count gold PER and LOC entities");
    Check(metrics.matched_entities == 1,
          "BIO metrics should require exact span/type match");
    CheckNear(metrics.entity_precision, 0.5, 1e-9,
              "BIO precision should match exact entities over predictions");
    CheckNear(metrics.entity_recall, 0.5, 1e-9,
              "BIO recall should match exact entities over gold spans");
    CheckNear(metrics.entity_f1, 0.5, 1e-9,
              "BIO F1 should combine exact precision and recall");
}

void TestSequenceVocabulary() {
    std::vector<std::vector<std::string>> token_sequences = {
        {"John", "lives", "in", "Berlin"},
        {"john", "works", "in", "Berlin"},
        {"Mary", "lives", "there"},
    };

    cyxwiz::SequenceVocabularyConfig token_config;
    token_config.kind = cyxwiz::SequenceVocabularyKind::Token;
    token_config.lowercase = true;
    token_config.min_frequency = 2;
    token_config.max_size = 5;

    const auto token_vocab =
        cyxwiz::BuildSequenceVocabulary(token_sequences, token_config);
    Check(token_vocab.Size() == 5,
          "token vocabulary should honor max_size including PAD/UNK");
    Check(token_vocab.PadId() == 0,
          "token vocabulary should reserve PAD id first");
    Check(token_vocab.UnkId() == 1,
          "token vocabulary should reserve UNK id second");
    Check(token_vocab.IdFor("berlin") == 2,
          "token vocabulary should sort by frequency then lexical order");
    Check(token_vocab.IdFor("in") == 3,
          "token vocabulary should keep frequent tokens");
    Check(token_vocab.IdFor("john") == 4,
          "token vocabulary should lowercase before counting");
    Check(token_vocab.IdFor("mary") == token_vocab.UnkId(),
          "token vocabulary should map filtered tokens to UNK");

    std::vector<std::vector<std::string>> tag_sequences = {
        {"B-PER", "I-PER", "O"},
        {"B-LOC", "O"},
    };
    cyxwiz::SequenceVocabularyConfig tag_config;
    tag_config.kind = cyxwiz::SequenceVocabularyKind::Tag;
    const auto tag_vocab =
        cyxwiz::BuildSequenceVocabulary(tag_sequences, tag_config);
    Check(!tag_vocab.HasPad() && !tag_vocab.HasUnk(),
          "tag vocabulary should not reserve PAD/UNK ids");
    Check(tag_vocab.ValueFor(0) == "O",
          "tag vocabulary should keep O at id zero when present");
    Check(tag_vocab.Contains("B-PER") && tag_vocab.Contains("I-PER"),
          "tag vocabulary should contain BIO labels");
    bool unknown_tag_failed = false;
    try {
        (void)tag_vocab.IdFor("B-ORG");
    } catch (const std::runtime_error&) {
        unknown_tag_failed = true;
    }
    Check(unknown_tag_failed,
          "tag vocabulary should reject unknown labels instead of using UNK");
}

void TestNERSequenceBuilder() {
    std::vector<cyxwiz::NERSequenceRow> rows = {
        {{"John", "lives", "in", "Berlin"},
         {"NNP", "VBZ", "IN", "NNP"},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"john", "works"},
         {"NNP", "VBZ"},
         {"B-PER", "O"}},
    };

    cyxwiz::NERSequenceBuilderConfig config;
    config.token_vocabulary.lowercase = true;
    config.batcher.batch_size = 2;
    config.batcher.max_sequence_length = 5;
    config.batcher.shuffle = false;

    const auto built = cyxwiz::BuildNERSequenceData(rows, config);
    Check(built.has_pos_tags,
          "NERSequenceBuilder should detect POS payloads");
    Check(built.has_tags,
          "NERSequenceBuilder should detect supervised NER tags");
    Check(built.samples.size() == 2,
          "NERSequenceBuilder should produce one sample per row");
    Check(built.token_vocabulary.PadId() == 0,
          "NER token vocabulary should reserve PAD id");
    Check(built.token_vocabulary.UnkId() == 1,
          "NER token vocabulary should reserve UNK id");
    Check(built.pos_vocabulary.PadId() == 0,
          "NER POS vocabulary should reserve PAD id");
    Check(built.tag_vocabulary.ValueFor(0) == "O",
          "NER tag vocabulary should keep O at id zero");
    Check(built.samples[0].word_ids[0] ==
              built.token_vocabulary.IdFor("john"),
          "NERSequenceBuilder should lowercase tokens during encoding");
    Check(built.samples[0].pos_ids[0] ==
              built.pos_vocabulary.IdFor("NNP"),
          "NERSequenceBuilder should encode POS tags");
    Check(built.samples[0].tag_ids[0] ==
              built.tag_vocabulary.IdFor("B-PER"),
          "NERSequenceBuilder should encode BIO tags");

    auto batcher = built.CreateBatcher();
    const auto batch = batcher.GetNextSequenceBatch();
    Check(batch.IsSupervised(),
          "NERSequenceBuilder batch should be supervised");
    Check(batch.HasPosIds(),
          "NERSequenceBuilder batch should include POS ids");
    Check(batch.HasAttentionMask(),
          "NERSequenceBuilder batch should include attention mask");
    Check(batch.word_ids.Shape() == std::vector<size_t>({2, 5}),
          "NERSequenceBuilder batch word ids should use configured length");

    const auto* words = batch.word_ids.Data<int64_t>();
    const auto* pos = batch.pos_ids.Data<int64_t>();
    const auto* tags = batch.tag_ids.Data<int64_t>();
    const auto* mask = batch.attention_mask.Data<int64_t>();
    Check(words[4] == built.token_vocabulary.PadId(),
          "NERSequenceBuilder should pad word ids with token PAD id");
    Check(pos[4] == built.pos_vocabulary.PadId(),
          "NERSequenceBuilder should pad POS ids with POS PAD id");
    Check(tags[4] == -100,
          "NERSequenceBuilder should pad tags with ignore_index");
    Check(mask[0] == 1 && mask[3] == 1 && mask[4] == 0,
          "NERSequenceBuilder should build attention masks from token length");

    bool mismatch_failed = false;
    try {
        (void)cyxwiz::BuildNERSequenceData({
            {{"bad", "row"}, {}, {"O"}},
        });
    } catch (const std::runtime_error&) {
        mismatch_failed = true;
    }
    Check(mismatch_failed,
          "NERSequenceBuilder should reject mismatched tag lengths");
}

void TestSequenceTrainingStep() {
    std::vector<cyxwiz::NERSequenceRow> rows = {
        {{"John", "lives", "in", "Berlin"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Mary", "works", "in", "Paris"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
    };

    cyxwiz::NERSequenceBuilderConfig builder_config;
    builder_config.use_pos_tags = false;
    builder_config.token_vocabulary.lowercase = true;
    builder_config.batcher.batch_size = 2;
    builder_config.batcher.max_sequence_length = 4;
    builder_config.batcher.shuffle = false;
    builder_config.batcher.tag_ignore_index = -100;

    const auto built = cyxwiz::BuildNERSequenceData(rows, builder_config);
    auto batcher = built.CreateBatcher();

    cyxwiz::TrainingConfiguration config;
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = built.tag_vocabulary.Size();
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.loss_params["ignore_index"] = "-100";
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.sequence_batch.enabled = true;
    config.sequence_batch.ignore_index = -100;

    cyxwiz::CompiledLayer embedding;
    embedding.type = gui::NodeType::Embedding;
    embedding.parameters["num_embeddings"] =
        std::to_string(built.token_vocabulary.Size());
    embedding.parameters["embedding_dim"] = "6";
    config.layers.push_back(embedding);

    cyxwiz::CompiledLayer token_head;
    token_head.type = gui::NodeType::TimeDistributed;
    token_head.units = static_cast<int>(built.tag_vocabulary.Size());
    config.layers.push_back(token_head);

    const auto result = cyxwiz::TrainSequenceTaggerEpoch(
        config, batcher, built.tag_vocabulary.Values());

    Check(result.success,
          "sequence training step should succeed: " + result.error);
    Check(result.batches == 1,
          "sequence training step should consume one batch");
    Check(result.samples == 2,
          "sequence training step should report trained samples");
    Check(std::isfinite(result.mean_loss),
          "sequence training step should produce finite loss");
    Check(result.metrics.total_tokens == 8,
          "sequence training step should score non-padding tokens");
    Check(result.metrics.token_accuracy >= 0.0 &&
              result.metrics.token_accuracy <= 1.0,
          "sequence training step token accuracy should be a probability");
}

void TestSequenceTrainingExecutor() {
    const std::vector<cyxwiz::NERSequenceRow> rows = {
        {{"John", "lives", "in", "Berlin"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Mary", "works", "in", "Paris"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
    };

    cyxwiz::NERSequenceBuilderConfig builder_config;
    builder_config.use_pos_tags = false;
    builder_config.token_vocabulary.lowercase = true;
    builder_config.batcher.batch_size = 2;
    builder_config.batcher.max_sequence_length = 4;
    builder_config.batcher.shuffle = false;
    builder_config.batcher.tag_ignore_index = -100;

    const auto built = cyxwiz::BuildNERSequenceData(rows, builder_config);

    cyxwiz::TrainingConfiguration config;
    config.input_size = 4;
    config.input_shape = {4};
    config.output_size = built.tag_vocabulary.Size();
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.loss_params["ignore_index"] = "-100";
    config.optimizer_type = gui::NodeType::SGD;
    config.learning_rate = 0.01f;
    config.sequence_batch.enabled = true;
    config.sequence_batch.ignore_index = -100;
    config.save_best_checkpoint = false;
    const auto checkpoint_dir =
        std::filesystem::temp_directory_path() /
        "cyxwiz_sequence_training_executor";
    std::filesystem::remove_all(checkpoint_dir);
    config.checkpoint_dir = checkpoint_dir.string();

    cyxwiz::CompiledLayer embedding;
    embedding.type = gui::NodeType::Embedding;
    embedding.parameters["num_embeddings"] =
        std::to_string(built.token_vocabulary.Size());
    embedding.parameters["embedding_dim"] = "6";
    config.layers.push_back(embedding);

    cyxwiz::CompiledLayer token_head;
    token_head.type = gui::NodeType::TimeDistributed;
    token_head.units = static_cast<int>(built.tag_vocabulary.Size());
    config.layers.push_back(token_head);

    auto batcher = std::make_unique<cyxwiz::SequenceBatcher>(
        built.samples, built.batcher_config);
    cyxwiz::TrainingExecutor executor(
        config, std::move(batcher), built.tag_vocabulary.Values());

    bool saw_batch = false;
    bool saw_epoch = false;
    bool completed = false;
    cyxwiz::TrainingMetrics final_metrics;

    executor.Train(
        1,
        2,
        [&](int epoch, int batch, int total_batches, float loss, float acc) {
            Check(epoch == 1,
                  "sequence executor batch callback should report epoch 1");
            Check(batch == 1,
                  "sequence executor batch callback should report batch 1");
            Check(total_batches == 1,
                  "sequence executor should report one batch");
            Check(std::isfinite(loss),
                  "sequence executor batch loss should be finite");
            Check(acc >= 0.0f && acc <= 1.0f,
                  "sequence executor batch accuracy should be a probability");
            saw_batch = true;
        },
        [&](int epoch,
            float train_loss,
            float train_acc,
            float val_loss,
            float val_acc,
            float) {
            Check(epoch == 1,
                  "sequence executor epoch callback should report epoch 1");
            Check(std::isfinite(train_loss),
                  "sequence executor train loss should be finite");
            Check(std::isfinite(val_loss),
                  "sequence executor val loss should be finite");
            Check(train_acc >= 0.0f && train_acc <= 1.0f,
                  "sequence executor train token accuracy should be a probability");
            Check(val_acc >= 0.0f && val_acc <= 1.0f,
                  "sequence executor val token accuracy should be a probability");
            saw_epoch = true;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(saw_batch, "sequence executor should run a batch callback");
    Check(saw_epoch, "sequence executor should run an epoch callback");
    Check(completed, "sequence executor should run completion callback");
    Check(final_metrics.is_complete,
          "sequence executor should mark training complete");
    Check(final_metrics.total_batches == 1,
          "sequence executor should report one training batch");
    Check(final_metrics.train_token_count == 8,
          "sequence executor should score train tokens");
    Check(final_metrics.val_token_count == 8,
          "sequence executor should score validation tokens");
    Check(final_metrics.test_sample_count == 0 &&
              !final_metrics.has_test_metrics &&
              final_metrics.test_token_count == 0,
          "sequence executor without an explicit Test phase must not "
          "evaluate Train data as held-out Test");
    Check(final_metrics.train_token_accuracy == final_metrics.train_accuracy,
          "sequence executor should mirror token accuracy to train_accuracy");
    Check(final_metrics.val_token_accuracy == final_metrics.val_accuracy,
          "sequence executor should mirror val token accuracy to val_accuracy");
    Check(final_metrics.train_entity_f1 >= 0.0f &&
              final_metrics.train_entity_f1 <= 1.0f,
          "sequence executor train entity F1 should be a probability");
    Check(final_metrics.val_entity_f1 >= 0.0f &&
              final_metrics.val_entity_f1 <= 1.0f,
          "sequence executor val entity F1 should be a probability");

    std::filesystem::remove_all(checkpoint_dir);
}

void TestArrowDataLoaderSeedDeterminism(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset) {
    cyxwiz::ArrowDatasetBatcher first(
        dataset,
        "label",
        2,
        true,
        1.0f,
        true,
        "",
        0,
        0,
        cyxwiz::BatcherPhase::Train,
        0.0f,
        1234);
    cyxwiz::ArrowDatasetBatcher second(
        dataset,
        "label",
        2,
        true,
        1.0f,
        true,
        "",
        0,
        0,
        cyxwiz::BatcherPhase::Train,
        0.0f,
        1234);
    first.SetOneHotEncoding(2);
    second.SetOneHotEncoding(2);

    const cyxwiz::Batch first_batch = first.GetNextBatch();
    const cyxwiz::Batch second_batch = second.GetNextBatch();
    Check(first_batch.IsValid(),
          "seeded Arrow batcher should produce a first batch");
    Check(second_batch.IsValid(),
          "matching seeded Arrow batcher should produce a first batch");
    Check(first_batch.data.NumElements() == second_batch.data.NumElements(),
          "matching seeds should produce same-sized data batches");
    Check(first_batch.labels.NumElements() == second_batch.labels.NumElements(),
          "matching seeds should produce same-sized label batches");

    const float* first_data = first_batch.data.Data<float>();
    const float* second_data = second_batch.data.Data<float>();
    for (size_t i = 0; i < first_batch.data.NumElements(); ++i) {
        CheckNear(first_data[i],
                  second_data[i],
                  0.0,
                  "matching seeds should produce identical data order");
    }

    const float* first_labels = first_batch.labels.Data<float>();
    const float* second_labels = second_batch.labels.Data<float>();
    for (size_t i = 0; i < first_batch.labels.NumElements(); ++i) {
        CheckNear(first_labels[i],
                  second_labels[i],
                  0.0,
                  "matching seeds should produce identical label order");
    }
}

class CountingTrainingHook final
    : public cyxwiz::plugin::ITrainingHook {
public:
    void OnTrainingStart(cyxwiz::plugin::TrainingContext&) override {
        ++training_start_count;
    }

    void OnTrainingEnd(cyxwiz::plugin::TrainingContext&) override {
        ++training_end_count;
    }

    void OnEpochStart(cyxwiz::plugin::TrainingContext& context) override {
        epoch_start_epochs.push_back(context.current_epoch);
    }

    void OnEpochEnd(cyxwiz::plugin::TrainingContext& context) override {
        epoch_end_epochs.push_back(context.current_epoch);
        epoch_end_val_losses.push_back(context.val_loss);
    }

    int training_start_count = 0;
    int training_end_count = 0;
    std::vector<int> epoch_start_epochs;
    std::vector<int> epoch_end_epochs;
    std::vector<float> epoch_end_val_losses;
};

void TestValidationEarlyStoppingLifecycle(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir,
    bool save_best_checkpoint) {
    const std::string label = save_best_checkpoint
        ? "early stopping with best checkpoint"
        : "early stopping without best checkpoint";
    auto config = MakeConfig(
        work_dir /
        (save_best_checkpoint ? "early-stop-with-checkpoint"
                              : "early-stop-without-checkpoint"));
    config.epochs = 5;
    config.learning_rate = 0.0f;
    config.validation_freq = 1;
    config.early_stopping_patience = 1;
    config.save_best_checkpoint = save_best_checkpoint;
    config.log_interval = 0;

    CountingTrainingHook hook;
    const std::string plugin_id = save_best_checkpoint
        ? "plan39.lifecycle.plateau-with-checkpoint"
        : "plan39.lifecycle.plateau-without-checkpoint";
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(plugin_id, &hook);

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    int epoch_callback_count = 0;
    int completion_callback_count = 0;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        nullptr,
        [&](int epoch, float, float, float val_loss, float, float) {
            ++epoch_callback_count;
            Check(epoch == epoch_callback_count,
                  label + " should report ordered epoch callbacks");
            Check(std::isfinite(val_loss),
                  label + " should report finite validation loss");
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(plugin_id);

    Check(epoch_callback_count == 2,
          label + " should stop after the first non-improving validation");
    Check(completion_callback_count == 1,
          label + " should invoke completion exactly once");
    Check(final_metrics.is_complete && !final_metrics.is_training,
          label + " should finish the executor lifecycle");
    Check(final_metrics.current_epoch == 2 &&
              final_metrics.last_executed_epoch == 2,
          label + " should preserve the actual stopping epoch");
    Check(final_metrics.total_epochs == 5,
          label + " should preserve the configured epoch count");
    Check(final_metrics.loss_history.size() == 2 &&
              final_metrics.val_loss_history.size() == 2,
          label + " should retain complete executed-run history");
    Check(final_metrics.terminal_status == "early_stopped",
          label + " should report early_stopped");
    Check(final_metrics.terminal_reason ==
              "validation_loss_plateau_patience_1",
          label + " should report the exact plateau reason");
    Check(hook.training_start_count == 1 &&
              hook.training_end_count == 1 &&
              hook.epoch_start_epochs == std::vector<int>({1, 2}) &&
              hook.epoch_end_epochs == std::vector<int>({1, 2}),
          label + " should close every fully executed plugin epoch hook");
    Check(hook.epoch_end_val_losses.size() == 2 &&
              std::isfinite(hook.epoch_end_val_losses[0]) &&
              std::isfinite(hook.epoch_end_val_losses[1]),
          label + " should publish validation values to both epoch-end hooks");

    if (save_best_checkpoint) {
        Check(!final_metrics.checkpoint_used.empty(),
              label + " should restore a concrete best checkpoint");
        Check(final_metrics.restored_checkpoint_epoch == 1,
              label + " should report the independently restored epoch");
        Check(final_metrics.restored_checkpoint_step > 0,
              label + " should report the restored optimizer/global step");
        Check(final_metrics.active_model_provenance ==
                  "restored_best_checkpoint",
              label + " should report restored active-model provenance");
        Check(final_metrics.current_epoch !=
                  final_metrics.restored_checkpoint_epoch,
              label + " must not rewrite run history with checkpoint epoch");
    } else {
        Check(final_metrics.checkpoint_used.empty(),
              label + " should not invent a checkpoint path");
        Check(final_metrics.restored_checkpoint_epoch == 0 &&
                  final_metrics.restored_checkpoint_step == 0,
              label + " should not report restored checkpoint state");
        Check(final_metrics.active_model_provenance == "run_final_state",
              label + " should retain final-run model provenance");
    }

    const auto crash_run = cyxwiz::CrashRunRecorder::LoadLastRun();
    Check(crash_run.has_value(), label + " should persist debug-run truth");
    Check(crash_run->status == "early_stopped" &&
              crash_run->terminal_reason ==
                  "validation_loss_plateau_patience_1",
          label + " debug-run status/reason mismatch");
    Check(crash_run->epoch == 2 && crash_run->epochs == 5,
          label + " debug-run epoch truth mismatch");
    Check(crash_run->last_executed_epoch == 2,
          label + " debug-run last-executed epoch truth mismatch");
    if (save_best_checkpoint) {
        Check(crash_run->checkpoint_used == final_metrics.checkpoint_used &&
                  crash_run->restored_checkpoint_epoch == 1 &&
                  crash_run->restored_checkpoint_step > 0 &&
                  crash_run->active_model_provenance ==
                      "restored_best_checkpoint",
              label + " debug-run checkpoint provenance mismatch");
    } else {
        Check(crash_run->checkpoint_used.empty() &&
                  crash_run->restored_checkpoint_epoch == 0 &&
                  crash_run->restored_checkpoint_step == 0 &&
                  crash_run->active_model_provenance == "run_final_state",
              label + " debug-run should preserve final-run provenance");
    }

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.available && trace.status == "early_stopped",
          label + " trace should finish as early_stopped");
    bool found_terminal_event = false;
    bool found_restore_event = false;
    for (const auto& event : trace.recent_events) {
        if (event.stage == "EarlyStopped" && event.epoch == 2 &&
            event.status == "early_stopped" &&
            event.terminal_reason ==
                "validation_loss_plateau_patience_1") {
            found_terminal_event = true;
        }
        if (event.stage == "CheckpointRestored" && event.epoch == 1 &&
            event.checkpoint_path == final_metrics.checkpoint_used &&
            event.metric_scope == "active_model") {
            found_restore_event = true;
        }
    }
    Check(found_terminal_event,
          label + " trace should contain the exact terminal event");
    Check(found_restore_event == save_best_checkpoint,
          label + " trace checkpoint-restored provenance mismatch");
}

void CheckTerminalEvidence(const std::string& expected_status,
                           const std::string& expected_reason,
                           int expected_epoch,
                           int expected_last_executed_epoch,
                           const std::string& label) {
    const auto crash_run = cyxwiz::CrashRunRecorder::LoadLastRun();
    Check(crash_run.has_value(), label + " should persist debug-run truth");
    Check(crash_run->status == expected_status,
          label + " debug-run status mismatch");
    Check(crash_run->terminal_reason == expected_reason,
          label + " debug-run terminal reason mismatch");
    Check(crash_run->epoch == expected_epoch,
          label + " debug-run terminal epoch mismatch");
    Check(crash_run->last_executed_epoch ==
              expected_last_executed_epoch,
          label + " debug-run last-executed epoch mismatch");
    if (expected_status == "failed") {
        Check(crash_run->failure_reason == expected_reason,
              label + " debug-run failure reason mismatch");
    }

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.available && trace.status == expected_status,
          label + " trace status mismatch");
    int terminal_event_count = 0;
    for (const auto& event : trace.recent_events) {
        if ((event.stage == "EarlyStopped" ||
             event.stage == "TrainingTerminal") &&
            event.status == expected_status &&
            event.terminal_reason == expected_reason) {
            ++terminal_event_count;
            Check(event.epoch == expected_epoch,
                  label + " trace terminal epoch mismatch");
        }
    }
    Check(terminal_event_count == 1,
          label + " should record exactly one canonical terminal event");
}

void CheckHeldOutTestEvidence(
    const cyxwiz::TrainingMetrics& metrics,
    int expected_checkpoint_epoch,
    const std::string& label) {
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    int restored_index = -1;
    int test_index = -1;
    int index = 0;
    for (const auto& event : trace.recent_events) {
        if (event.stage == "CheckpointRestored") {
            restored_index = index;
        }
        if (event.stage == "HeldOutTestCompleted") {
            Check(test_index < 0,
                  label + " should record one held-out Test event");
            test_index = index;
            CheckNear(event.loss, metrics.test_loss, 1e-6,
                      label + " trace Test loss");
            CheckNear(event.accuracy, metrics.test_accuracy, 1e-6,
                      label + " trace Test accuracy");
            Check(event.metric_scope == "test",
                  label + " trace should classify held-out metrics as Test");
            Check(event.checkpoint_path == metrics.checkpoint_used,
                  label + " trace should identify the evaluated model checkpoint");
            Check(event.message.find(metrics.active_model_provenance) !=
                      std::string::npos,
                  label + " trace should identify active-model provenance");
        }
        ++index;
    }

    Check(test_index >= 0,
          label + " should trace held-out Test completion");
    if (expected_checkpoint_epoch > 0) {
        Check(metrics.restored_checkpoint_epoch == expected_checkpoint_epoch &&
                  metrics.active_model_provenance ==
                      "restored_best_checkpoint" &&
                  !metrics.checkpoint_used.empty(),
              label + " should preserve restored model provenance");
        Check(restored_index >= 0 && restored_index < test_index,
              label + " must restore the best checkpoint before held-out Test");
    } else {
        Check(restored_index < 0 && metrics.checkpoint_used.empty() &&
                  metrics.restored_checkpoint_epoch == 0 &&
                  metrics.active_model_provenance == "run_final_state",
              label + " should evaluate the final run state without restore");
    }
}

void CheckScheduledValidationValues(
    const std::vector<float>& values,
    const std::vector<int>& validation_epochs,
    int total_epochs,
    const std::string& label) {
    Check(values.size() == static_cast<size_t>(total_epochs),
          label + " should publish one value per executed epoch");
    for (int epoch = 1; epoch <= total_epochs; ++epoch) {
        const bool should_have_validation =
            std::find(validation_epochs.begin(), validation_epochs.end(), epoch) !=
            validation_epochs.end();
        if (should_have_validation) {
            Check(std::isfinite(values[static_cast<size_t>(epoch - 1)]) &&
                      values[static_cast<size_t>(epoch - 1)] >= 0.0f,
                  label + " should publish finite validation at epoch " +
                      std::to_string(epoch));
        } else {
            Check(values[static_cast<size_t>(epoch - 1)] == -1.0f,
                  label + " should publish the skipped-validation sentinel at epoch " +
                      std::to_string(epoch));
        }
    }
}

void TestScheduledValidationLifecycle(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "scheduled-final-validation");
    config.epochs = 5;
    config.learning_rate = 0.0f;
    config.validation_freq = 2;
    config.early_stopping_patience = 0;
    config.save_best_checkpoint = false;
    config.log_interval = 0;

    CountingTrainingHook hook;
    constexpr const char* kPluginId =
        "plan39.lifecycle.scheduled-final-validation";
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(kPluginId, &hook);

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    int batch_callback_count = 0;
    int completion_callback_count = 0;
    std::vector<int> callback_epochs;
    std::vector<float> callback_val_losses;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) { ++batch_callback_count; },
        [&](int epoch, float, float, float val_loss, float, float) {
            callback_epochs.push_back(epoch);
            callback_val_losses.push_back(val_loss);
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(kPluginId);

    const std::vector<int> expected_epochs = {1, 2, 3, 4, 5};
    const std::vector<int> validation_epochs = {2, 4, 5};
    Check(batch_callback_count == 10,
          "scheduled validation should keep every batch callback responsive");
    Check(completion_callback_count == 1,
          "scheduled validation should invoke completion exactly once");
    Check(callback_epochs == expected_epochs,
          "scheduled validation should preserve ordered epoch callbacks");
    CheckScheduledValidationValues(
        callback_val_losses, validation_epochs, 5,
        "scheduled public epoch callback");
    Check(hook.training_start_count == 1 && hook.training_end_count == 1 &&
              hook.epoch_start_epochs == expected_epochs &&
              hook.epoch_end_epochs == expected_epochs,
          "scheduled validation should preserve symmetric plugin lifecycle hooks");
    CheckScheduledValidationValues(
        hook.epoch_end_val_losses, validation_epochs, 5,
        "scheduled plugin epoch-end callback");
    Check(final_metrics.terminal_status == "completed" &&
              final_metrics.terminal_reason == "completed_all_epochs" &&
              final_metrics.last_executed_epoch == 5 &&
              final_metrics.loss_history.size() == 5 &&
              final_metrics.val_loss_history.size() == 3,
          "scheduled validation should retain exact final/history truth");

    std::vector<int> traced_validation_epochs;
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    for (const auto& event : trace.recent_events) {
        if (event.stage == "ValidationCompleted") {
            traced_validation_epochs.push_back(event.epoch);
        }
    }
    Check(traced_validation_epochs == validation_epochs,
          "scheduled validation trace should contain epochs 2, 4, and final epoch 5");
    CheckTerminalEvidence(
        "completed", "completed_all_epochs", 5, 5,
        "scheduled final validation");
}

void TestScheduledValidationPatienceLifecycle(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "scheduled-validation-patience");
    config.epochs = 7;
    config.learning_rate = 0.0f;
    config.validation_freq = 2;
    config.early_stopping_patience = 2;
    config.save_best_checkpoint = false;
    config.log_interval = 0;

    CountingTrainingHook hook;
    constexpr const char* kPluginId =
        "plan39.lifecycle.scheduled-validation-patience";
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(kPluginId, &hook);

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    std::vector<int> callback_epochs;
    std::vector<float> callback_val_losses;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        nullptr,
        [&](int epoch, float, float, float val_loss, float, float) {
            callback_epochs.push_back(epoch);
            callback_val_losses.push_back(val_loss);
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(kPluginId);

    const std::vector<int> expected_epochs = {1, 2, 3, 4, 5, 6};
    const std::vector<int> validation_epochs = {2, 4, 6};
    Check(callback_epochs == expected_epochs &&
              hook.epoch_start_epochs == expected_epochs &&
              hook.epoch_end_epochs == expected_epochs,
          "scheduled patience should close all six fully executed epochs");
    CheckScheduledValidationValues(
        callback_val_losses, validation_epochs, 6,
        "scheduled patience public epoch callback");
    CheckScheduledValidationValues(
        hook.epoch_end_val_losses, validation_epochs, 6,
        "scheduled patience plugin epoch-end callback");
    Check(final_metrics.terminal_status == "early_stopped" &&
              final_metrics.terminal_reason ==
                  "validation_loss_plateau_patience_2" &&
              final_metrics.last_executed_epoch == 6 &&
              final_metrics.loss_history.size() == 6 &&
              final_metrics.val_loss_history.size() == 3,
          "scheduled patience should count only validation epochs and stop at epoch 6");
    CheckTerminalEvidence(
        "early_stopped", "validation_loss_plateau_patience_2", 6, 6,
        "scheduled validation patience");
}

void CheckExternalResolvedRoleLifecycleCase(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    cyxwiz::TrainingConfiguration config,
    const std::string& case_name,
    const std::vector<int>& expected_epochs,
    const std::vector<int>& validation_epochs,
    const std::string& expected_terminal_status,
    const std::string& expected_terminal_reason) {
    auto batchers = cyxwiz::BuildArrowTrainingBatchers(
        config, dataset, "label", config.batch_size);
    auto resolved = cyxwiz::TakeResolvedExternalBatchers(
        std::move(batchers));
    Check(resolved.train != nullptr && resolved.dev != nullptr &&
              resolved.test != nullptr,
          case_name + " should resolve explicit Train/Dev/Test batchers");

    CountingTrainingHook hook;
    const std::string plugin_id =
        "plan39.lifecycle.external-resolved." + case_name;
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(plugin_id, &hook);

    cyxwiz::TrainingExecutor executor(config, std::move(resolved));
    int batch_callback_count = 0;
    int completion_callback_count = 0;
    std::vector<int> callback_epochs;
    std::vector<float> callback_val_losses;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) { ++batch_callback_count; },
        [&](int epoch, float, float, float val_loss, float, float) {
            callback_epochs.push_back(epoch);
            callback_val_losses.push_back(val_loss);
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(plugin_id);

    Check(final_metrics.total_batches == 2 &&
              batch_callback_count ==
                  final_metrics.total_batches *
                      static_cast<int>(expected_epochs.size()),
          case_name + " should invoke one callback for each external Train batch");
    Check(completion_callback_count == 1,
          case_name + " should invoke completion exactly once");
    Check(callback_epochs == expected_epochs,
          case_name + " should preserve ordered public epoch callbacks");
    CheckScheduledValidationValues(
        callback_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " public epoch callback");
    Check(hook.training_start_count == 1 && hook.training_end_count == 1 &&
              hook.epoch_start_epochs == expected_epochs &&
              hook.epoch_end_epochs == expected_epochs,
          case_name + " should preserve symmetric plugin lifecycle hooks");
    CheckScheduledValidationValues(
        hook.epoch_end_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " plugin epoch-end callback");
    Check(final_metrics.train_sample_count == 3 &&
              final_metrics.val_sample_count == 1 &&
              final_metrics.test_sample_count == 2,
          case_name + " should retain resolved external role sample counts");
    Check(final_metrics.last_executed_epoch == expected_epochs.back() &&
              final_metrics.loss_history.size() == expected_epochs.size() &&
              final_metrics.val_loss_history.size() == validation_epochs.size(),
          case_name + " should preserve exact external run history");
    Check(final_metrics.terminal_status == expected_terminal_status &&
              final_metrics.terminal_reason == expected_terminal_reason,
          case_name + " should preserve exact terminal truth");

    std::vector<int> traced_validation_epochs;
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    for (const auto& event : trace.recent_events) {
        if (event.stage == "ValidationCompleted") {
            traced_validation_epochs.push_back(event.epoch);
        }
    }
    Check(traced_validation_epochs == validation_epochs,
          case_name + " trace should record only scheduled validation epochs");
    CheckTerminalEvidence(
        expected_terminal_status,
        expected_terminal_reason,
        expected_epochs.back(),
        expected_epochs.back(),
        case_name);
}

void TestExternalResolvedRoleLifecycle(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto completed_config = MakeConfig(
        work_dir / "external-resolved-scheduled-final-validation");
    completed_config.epochs = 3;
    completed_config.learning_rate = 0.0f;
    completed_config.validation_freq = 2;
    completed_config.early_stopping_patience = 0;
    completed_config.save_best_checkpoint = false;
    completed_config.log_interval = 0;
    completed_config.train_ratio = 0.5f;
    completed_config.val_ratio = 0.25f;
    completed_config.test_ratio = 0.25f;
    completed_config.has_data_split = true;
    CheckExternalResolvedRoleLifecycleCase(
        dataset,
        completed_config,
        "external-resolved-scheduled-final-validation",
        {1, 2, 3},
        {2, 3},
        "completed",
        "completed_all_epochs");

    auto plateau_config = MakeConfig(
        work_dir / "external-resolved-scheduled-patience");
    plateau_config.epochs = 5;
    plateau_config.learning_rate = 0.0f;
    plateau_config.validation_freq = 2;
    plateau_config.early_stopping_patience = 1;
    plateau_config.save_best_checkpoint = false;
    plateau_config.log_interval = 0;
    plateau_config.train_ratio = 0.5f;
    plateau_config.val_ratio = 0.25f;
    plateau_config.test_ratio = 0.25f;
    plateau_config.has_data_split = true;
    CheckExternalResolvedRoleLifecycleCase(
        dataset,
        plateau_config,
        "external-resolved-scheduled-patience",
        {1, 2, 3, 4},
        {2, 4},
        "early_stopped",
        "validation_loss_plateau_patience_1");
}

class PhaseTrackingBatcher final : public cyxwiz::IBatcher {
public:
    cyxwiz::Batch GetNextBatch() override {
        if (IsEpochComplete()) {
            return {};
        }

        const float phase_offset = current_phase_ == cyxwiz::BatcherPhase::Val
            ? 0.25f
            : 0.0f;
        const std::vector<float> features = {
            phase_offset + 0.0f, phase_offset + 0.1f,
            phase_offset + 0.9f, phase_offset + 1.0f,
        };
        const std::vector<float> labels = {
            1.0f, 0.0f,
            0.0f, 1.0f,
        };

        cyxwiz::Batch batch;
        batch.data = cyxwiz::Tensor(
            {2, 2}, features.data(), cyxwiz::DataType::Float32);
        batch.labels = cyxwiz::Tensor(
            {2, 2}, labels.data(), cyxwiz::DataType::Float32);
        batch.size = 2;
        ++current_batch_;
        if (current_phase_ == cyxwiz::BatcherPhase::Val) {
            ++val_batch_count_;
        } else {
            ++train_batch_count_;
        }
        return batch;
    }

    void Reset() override {
        current_batch_ = 0;
        if (current_phase_ == cyxwiz::BatcherPhase::Val) {
            ++val_reset_count_;
        } else {
            ++train_reset_count_;
        }
    }

    bool IsEpochComplete() const override {
        return current_batch_ >= GetNumBatches();
    }

    size_t GetNumBatches() const override {
        if (current_phase_ == cyxwiz::BatcherPhase::Val) {
            return 1;
        }
        if (current_phase_ == cyxwiz::BatcherPhase::Test) {
            return 0;
        }
        return 2;
    }

    size_t GetNumSamples() const override {
        return GetNumBatches() * 2;
    }

    void SetNormalization(float, float) override {}
    void SetOneHotEncoding(size_t) override {}
    void SetFlatten(bool) override {}

    void SetPhase(cyxwiz::BatcherPhase phase) override {
        current_phase_ = phase;
    }

    int TrainResetCount() const { return train_reset_count_; }
    int ValResetCount() const { return val_reset_count_; }
    int TrainBatchCount() const { return train_batch_count_; }
    int ValBatchCount() const { return val_batch_count_; }

private:
    cyxwiz::BatcherPhase current_phase_ = cyxwiz::BatcherPhase::Train;
    size_t current_batch_ = 0;
    int train_reset_count_ = 0;
    int val_reset_count_ = 0;
    int train_batch_count_ = 0;
    int val_batch_count_ = 0;
};

void CheckSharedPhaseExternalLifecycleCase(
    cyxwiz::TrainingConfiguration config,
    const std::string& case_name,
    const std::vector<int>& expected_epochs,
    const std::vector<int>& validation_epochs,
    const std::string& expected_terminal_status,
    const std::string& expected_terminal_reason) {
    auto batcher = std::make_unique<PhaseTrackingBatcher>();
    auto* phase_tracking_batcher = batcher.get();
    CountingTrainingHook hook;
    const std::string plugin_id =
        "plan39.lifecycle.external-shared-phase." + case_name;
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(plugin_id, &hook);

    cyxwiz::TrainingExecutor executor(config, std::move(batcher));
    int batch_callback_count = 0;
    int completion_callback_count = 0;
    std::vector<int> callback_epochs;
    std::vector<float> callback_val_losses;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) { ++batch_callback_count; },
        [&](int epoch, float, float, float val_loss, float, float) {
            callback_epochs.push_back(epoch);
            callback_val_losses.push_back(val_loss);
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(plugin_id);

    const int expected_train_batches =
        static_cast<int>(expected_epochs.size()) * 2;
    const int expected_val_batches =
        static_cast<int>(validation_epochs.size());
    Check(batch_callback_count == expected_train_batches &&
              phase_tracking_batcher->TrainBatchCount() ==
                  expected_train_batches &&
              phase_tracking_batcher->ValBatchCount() ==
                  expected_val_batches,
          case_name + " should consume exact shared-phase Train and Dev batches");
    Check(completion_callback_count == 1,
          case_name + " should invoke completion exactly once");
    Check(callback_epochs == expected_epochs &&
              hook.training_start_count == 1 &&
              hook.training_end_count == 1 &&
              hook.epoch_start_epochs == expected_epochs &&
              hook.epoch_end_epochs == expected_epochs,
          case_name + " should preserve exact shared-phase lifecycle hooks");
    CheckScheduledValidationValues(
        callback_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " public epoch callback");
    CheckScheduledValidationValues(
        hook.epoch_end_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " plugin epoch-end callback");
    Check(phase_tracking_batcher->TrainResetCount() ==
              static_cast<int>(expected_epochs.size()) -
                  (expected_terminal_status == "early_stopped" ? 1 : 0) &&
              phase_tracking_batcher->ValResetCount() ==
                  expected_val_batches,
          case_name + " must reset once per executed semantic boundary");
    Check(final_metrics.train_sample_count == 4 &&
              final_metrics.val_sample_count == 2 &&
              final_metrics.test_sample_count == 0 &&
              final_metrics.last_executed_epoch == expected_epochs.back() &&
              final_metrics.loss_history.size() == expected_epochs.size() &&
              final_metrics.val_loss_history.size() ==
                  validation_epochs.size() &&
              final_metrics.terminal_status == expected_terminal_status &&
              final_metrics.terminal_reason == expected_terminal_reason,
          case_name + " should preserve shared-phase role and terminal truth");

    std::vector<int> traced_validation_epochs;
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    for (const auto& event : trace.recent_events) {
        if (event.stage == "ValidationCompleted") {
            traced_validation_epochs.push_back(event.epoch);
        }
    }
    Check(traced_validation_epochs == validation_epochs,
          case_name + " trace should record only scheduled validation epochs");
    CheckTerminalEvidence(
        expected_terminal_status,
        expected_terminal_reason,
        expected_epochs.back(),
        expected_epochs.back(),
        case_name);
}

void TestSharedPhaseExternalLifecycle(
    const std::filesystem::path& work_dir) {
    auto completed_config = MakeConfig(
        work_dir / "external-shared-phase-scheduled-final-validation");
    completed_config.epochs = 3;
    completed_config.learning_rate = 0.0f;
    completed_config.validation_freq = 2;
    completed_config.early_stopping_patience = 0;
    completed_config.save_best_checkpoint = false;
    completed_config.log_interval = 0;
    CheckSharedPhaseExternalLifecycleCase(
        completed_config,
        "external-shared-phase-scheduled-final-validation",
        {1, 2, 3},
        {2, 3},
        "completed",
        "completed_all_epochs");

    auto plateau_config = MakeConfig(
        work_dir / "external-shared-phase-scheduled-patience");
    plateau_config.epochs = 5;
    plateau_config.learning_rate = 0.0f;
    plateau_config.validation_freq = 2;
    plateau_config.early_stopping_patience = 1;
    plateau_config.save_best_checkpoint = false;
    plateau_config.log_interval = 0;
    CheckSharedPhaseExternalLifecycleCase(
        plateau_config,
        "external-shared-phase-scheduled-patience",
        {1, 2, 3, 4},
        {2, 4},
        "early_stopped",
        "validation_loss_plateau_patience_1");
}

void CheckSequenceLifecycleCase(
    const cyxwiz::NERSequenceBuildResult& built,
    cyxwiz::TrainingConfiguration config,
    const std::string& case_name,
    const std::vector<int>& expected_epochs,
    const std::vector<int>& validation_epochs,
    const std::string& expected_terminal_status,
    const std::string& expected_terminal_reason,
    int expected_checkpoint_epoch) {
    auto batcher = std::make_unique<cyxwiz::SequenceBatcher>(
        built.samples, built.batcher_config);
    CountingTrainingHook hook;
    const std::string plugin_id =
        "plan39.lifecycle.sequence." + case_name;
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(plugin_id, &hook);

    cyxwiz::TrainingExecutor executor(
        config, std::move(batcher), built.tag_vocabulary.Values());
    int batch_callback_count = 0;
    int completion_callback_count = 0;
    std::vector<int> callback_epochs;
    std::vector<float> callback_val_losses;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) { ++batch_callback_count; },
        [&](int epoch, float, float, float val_loss, float, float) {
            callback_epochs.push_back(epoch);
            callback_val_losses.push_back(val_loss);
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(plugin_id);

    Check(batch_callback_count ==
              static_cast<int>(expected_epochs.size()) * 2 &&
              final_metrics.optimizer_step_count ==
                  static_cast<int>(expected_epochs.size()) * 2,
          case_name + " should execute two Sequence batches per epoch");
    Check(completion_callback_count == 1,
          case_name + " should invoke completion exactly once");
    Check(callback_epochs == expected_epochs &&
              hook.training_start_count == 1 &&
              hook.training_end_count == 1 &&
              hook.epoch_start_epochs == expected_epochs &&
              hook.epoch_end_epochs == expected_epochs,
          case_name + " should preserve symmetric Sequence lifecycle hooks");
    CheckScheduledValidationValues(
        callback_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " public epoch callback");
    CheckScheduledValidationValues(
        hook.epoch_end_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " plugin epoch-end callback");
    Check(final_metrics.train_sample_count == 2 &&
              final_metrics.val_sample_count == 1 &&
              final_metrics.test_sample_count == 1 &&
              final_metrics.train_token_count == 8 &&
              final_metrics.val_token_count == 4 &&
              final_metrics.test_token_count == 4 &&
              final_metrics.has_test_metrics &&
              std::isfinite(final_metrics.test_loss) &&
              final_metrics.test_token_accuracy ==
                  final_metrics.test_accuracy &&
              final_metrics.test_entity_f1 >= 0.0f &&
              final_metrics.test_entity_f1 <= 1.0f &&
              final_metrics.test_accuracy >= 0.0f &&
              final_metrics.test_accuracy <= 1.0f,
          case_name + " should preserve Sequence role and token counts");
    Check(final_metrics.current_epoch == expected_epochs.back() &&
              final_metrics.last_executed_epoch == expected_epochs.back() &&
              final_metrics.loss_history.size() == expected_epochs.size() &&
              final_metrics.val_loss_history.size() ==
                  validation_epochs.size() &&
              final_metrics.terminal_status == expected_terminal_status &&
              final_metrics.terminal_reason == expected_terminal_reason,
          case_name + " should preserve exact Sequence terminal truth");

    std::vector<int> traced_validation_epochs;
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    for (const auto& event : trace.recent_events) {
        if (event.stage == "ValidationCompleted") {
            traced_validation_epochs.push_back(event.epoch);
        }
    }
    Check(traced_validation_epochs == validation_epochs,
          case_name + " trace should record only Sequence validation epochs");
    CheckHeldOutTestEvidence(
        final_metrics, expected_checkpoint_epoch, case_name);
    CheckTerminalEvidence(
        expected_terminal_status,
        expected_terminal_reason,
        expected_epochs.back(),
        expected_epochs.back(),
        case_name);
}

void TestSequenceLifecycleCadence(const std::filesystem::path& work_dir) {
    const std::vector<cyxwiz::NERSequenceRow> rows = {
        {{"John", "lives", "in", "Berlin"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Mary", "works", "in", "Paris"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Alice", "visits", "New", "York"},
         {},
         {"B-PER", "O", "B-LOC", "I-LOC"}},
        {{"Bob", "works", "in", "London"},
         {},
         {"B-PER", "O", "O", "B-LOC"}},
    };

    cyxwiz::NERSequenceBuilderConfig builder_config;
    builder_config.use_pos_tags = false;
    builder_config.token_vocabulary.lowercase = true;
    builder_config.batcher.batch_size = 1;
    builder_config.batcher.max_sequence_length = 4;
    builder_config.batcher.shuffle = true;
    builder_config.batcher.seed = 17;
    builder_config.batcher.tag_ignore_index = -100;
    builder_config.batcher.train_indices = {0, 1};
    builder_config.batcher.val_indices = {2};
    builder_config.batcher.test_indices = {3};
    const auto built = cyxwiz::BuildNERSequenceData(rows, builder_config);

    auto make_config = [&](const std::string& checkpoint_name) {
        cyxwiz::TrainingConfiguration config;
        config.dataset_name = "sequence-lifecycle";
        config.input_size = 4;
        config.input_shape = {4};
        config.output_size = built.tag_vocabulary.Size();
        config.loss_type = gui::NodeType::CrossEntropyLoss;
        config.loss_params["ignore_index"] = "-100";
        config.optimizer_type = gui::NodeType::SGD;
        config.learning_rate = 0.0f;
        config.batch_size = 1;
        config.sequence_batch.enabled = true;
        config.sequence_batch.ignore_index = -100;
        config.save_best_checkpoint = false;
        config.log_interval = 0;
        config.checkpoint_dir = (work_dir / checkpoint_name).string();

        cyxwiz::CompiledLayer embedding;
        embedding.type = gui::NodeType::Embedding;
        embedding.parameters["num_embeddings"] =
            std::to_string(built.token_vocabulary.Size());
        embedding.parameters["embedding_dim"] = "6";
        config.layers.push_back(embedding);

        cyxwiz::CompiledLayer token_head;
        token_head.type = gui::NodeType::TimeDistributed;
        token_head.units = static_cast<int>(built.tag_vocabulary.Size());
        config.layers.push_back(token_head);
        return config;
    };

    auto completed_config = make_config("sequence-scheduled-final-validation");
    completed_config.epochs = 3;
    completed_config.validation_freq = 2;
    completed_config.early_stopping_patience = 0;
    CheckSequenceLifecycleCase(
        built,
        completed_config,
        "sequence-scheduled-final-validation",
        {1, 2, 3},
        {2, 3},
        "completed",
        "completed_all_epochs",
        0);

    auto plateau_config = make_config("sequence-scheduled-patience");
    plateau_config.epochs = 5;
    plateau_config.validation_freq = 2;
    plateau_config.early_stopping_patience = 1;
    plateau_config.save_best_checkpoint = true;
    CheckSequenceLifecycleCase(
        built,
        plateau_config,
        "sequence-scheduled-patience",
        {1, 2, 3, 4},
        {2, 4},
        "early_stopped",
        "validation_loss_plateau_patience_1",
        2);
}

void CheckParquetLifecycleCase(
    const std::shared_ptr<cyxwiz::ParquetBackedDataset>& dataset,
    cyxwiz::TrainingConfiguration config,
    const std::string& case_name,
    const std::vector<int>& expected_epochs,
    const std::vector<int>& validation_epochs,
    const std::string& expected_terminal_status,
    const std::string& expected_terminal_reason,
    int expected_checkpoint_epoch) {
    CountingTrainingHook hook;
    const std::string plugin_id =
        "plan39.lifecycle.parquet." + case_name;
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(plugin_id, &hook);

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    int batch_callback_count = 0;
    int completion_callback_count = 0;
    std::vector<int> callback_epochs;
    std::vector<float> callback_val_losses;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) { ++batch_callback_count; },
        [&](int epoch, float, float, float val_loss, float, float) {
            callback_epochs.push_back(epoch);
            callback_val_losses.push_back(val_loss);
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(plugin_id);

    Check(batch_callback_count ==
              static_cast<int>(expected_epochs.size()) * 2 &&
              final_metrics.optimizer_step_count ==
                  static_cast<int>(expected_epochs.size()) * 2,
          case_name + " should execute two Parquet Train batches per epoch");
    Check(completion_callback_count == 1,
          case_name + " should invoke completion exactly once");
    Check(callback_epochs == expected_epochs &&
              hook.training_start_count == 1 &&
              hook.training_end_count == 1 &&
              hook.epoch_start_epochs == expected_epochs &&
              hook.epoch_end_epochs == expected_epochs,
          case_name + " should preserve symmetric Parquet lifecycle hooks");
    CheckScheduledValidationValues(
        callback_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " public epoch callback");
    CheckScheduledValidationValues(
        hook.epoch_end_val_losses,
        validation_epochs,
        static_cast<int>(expected_epochs.size()),
        case_name + " plugin epoch-end callback");
    Check(final_metrics.train_sample_count == 3 &&
              final_metrics.val_sample_count == 1 &&
              final_metrics.test_sample_count == 2 &&
              final_metrics.has_test_metrics &&
              std::isfinite(final_metrics.test_loss) &&
              final_metrics.test_accuracy >= 0.0f &&
              final_metrics.test_accuracy <= 1.0f,
          case_name + " should preserve Parquet role counts");
    Check(final_metrics.current_epoch == expected_epochs.back() &&
              final_metrics.last_executed_epoch == expected_epochs.back() &&
              final_metrics.loss_history.size() == expected_epochs.size() &&
              final_metrics.val_loss_history.size() ==
                  validation_epochs.size() &&
              final_metrics.terminal_status == expected_terminal_status &&
              final_metrics.terminal_reason == expected_terminal_reason,
          case_name + " should preserve exact Parquet terminal truth");

    std::vector<int> traced_validation_epochs;
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    for (const auto& event : trace.recent_events) {
        if (event.stage == "ValidationCompleted") {
            traced_validation_epochs.push_back(event.epoch);
        }
    }
    Check(traced_validation_epochs == validation_epochs,
          case_name + " trace should record only Parquet validation epochs");
    CheckHeldOutTestEvidence(
        final_metrics, expected_checkpoint_epoch, case_name);
    CheckTerminalEvidence(
        expected_terminal_status,
        expected_terminal_reason,
        expected_epochs.back(),
        expected_epochs.back(),
        case_name);
}

void TestParquetLifecycleCadence(
    const std::shared_ptr<cyxwiz::ParquetBackedDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto completed_config = MakeConfig(
        work_dir / "parquet-scheduled-final-validation");
    completed_config.epochs = 3;
    completed_config.learning_rate = 0.0f;
    completed_config.validation_freq = 2;
    completed_config.early_stopping_patience = 0;
    completed_config.save_best_checkpoint = false;
    completed_config.log_interval = 0;
    completed_config.train_ratio = 0.5f;
    completed_config.val_ratio = 0.25f;
    completed_config.test_ratio = 0.25f;
    completed_config.has_data_split = true;
    CheckParquetLifecycleCase(
        dataset,
        completed_config,
        "parquet-scheduled-final-validation",
        {1, 2, 3},
        {2, 3},
        "completed",
        "completed_all_epochs",
        0);

    auto plateau_config = MakeConfig(
        work_dir / "parquet-scheduled-patience");
    plateau_config.epochs = 5;
    plateau_config.learning_rate = 0.0f;
    plateau_config.validation_freq = 2;
    plateau_config.early_stopping_patience = 1;
    plateau_config.save_best_checkpoint = true;
    plateau_config.log_interval = 0;
    plateau_config.train_ratio = 0.5f;
    plateau_config.val_ratio = 0.25f;
    plateau_config.test_ratio = 0.25f;
    plateau_config.has_data_split = true;
    CheckParquetLifecycleCase(
        dataset,
        plateau_config,
        "parquet-scheduled-patience",
        {1, 2, 3, 4},
        {2, 4},
        "early_stopped",
        "validation_loss_plateau_patience_1",
        2);
}

struct MetricCadenceRunResult {
    cyxwiz::TrainingMetrics metrics;
    std::map<std::string, std::vector<float>> parameters;
    int batch_callback_count = 0;
    uint64_t loss_scalar_readbacks = 0;
    uint64_t accuracy_scalar_readbacks = 0;
    std::string reporting_cadence;
};

MetricCadenceRunResult RunMetricCadenceCase(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& checkpoint_dir,
    int log_interval) {
    auto config = MakeConfig(checkpoint_dir);
    config.epochs = 2;
    config.batch_size = 1;
    config.validation_freq = 2;
    config.log_interval = log_interval;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;

    // Model construction happens inside Train. Reset the ArrayFire RNG before
    // each case so reporting cadence is the only changed input.
    af::setSeed(390039);
    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    MetricCadenceRunResult result;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) {
            ++result.batch_callback_count;
        },
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            result.metrics = metrics;
        });

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    for (const auto& group : trace.arrayfire_host_sync_groups) {
        if (group.operation == "TrainingExecutor::ReadAccumulatedLoss") {
            result.loss_scalar_readbacks += group.event_count;
        } else if (group.operation ==
                   "TrainingExecutor::ReadAccumulatedAccuracy") {
            result.accuracy_scalar_readbacks += group.event_count;
        }
    }
    for (const auto& event : trace.recent_events) {
        if (event.stage == "TrainingExecutor.ReportingCadence") {
            result.reporting_cadence = event.message;
        }
    }

    auto* model = executor.GetModel();
    Check(model != nullptr,
          "metric cadence case should retain its trained sequential model");
    for (const auto& [name, tensor] : model->GetParameters()) {
        Check(tensor.GetDataType() == cyxwiz::DataType::Float32,
              "metric cadence fixture expects Float32 model parameters");
        const float* values = tensor.ReadData<float>();
        result.parameters[name] = std::vector<float>(
            values, values + tensor.NumElements());
    }
    return result;
}

void TestMetricReportingCadenceInvariance(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    const auto first_final = RunMetricCadenceCase(
        dataset, work_dir / "metric-cadence-first-final", 0);
    const auto every_second = RunMetricCadenceCase(
        dataset, work_dir / "metric-cadence-every-two", 2);

    Check(first_final.batch_callback_count == 8 &&
              every_second.batch_callback_count == 8,
          "metric sampling cadence must not throttle batch callbacks");
    Check(first_final.metrics.optimizer_step_count == 8 &&
              every_second.metrics.optimizer_step_count == 8,
          "metric sampling cadence must not change optimizer step count");
    Check(first_final.metrics.terminal_status == "completed" &&
              every_second.metrics.terminal_status == "completed" &&
              first_final.metrics.last_executed_epoch == 2 &&
              every_second.metrics.last_executed_epoch == 2,
          "metric sampling cadence must not change terminal epoch truth");
    Check(first_final.metrics.loss_history.size() == 2 &&
              every_second.metrics.loss_history.size() == 2 &&
              first_final.metrics.val_loss_history.size() == 1 &&
              every_second.metrics.val_loss_history.size() == 1,
          "metric sampling cadence must not change epoch/validation history counts");
    for (size_t i = 0; i < first_final.metrics.loss_history.size(); ++i) {
        CheckNear(first_final.metrics.loss_history[i],
                  every_second.metrics.loss_history[i],
                  1e-6,
                  "metric sampling cadence must preserve train loss history");
        CheckNear(first_final.metrics.accuracy_history[i],
                  every_second.metrics.accuracy_history[i],
                  1e-6,
                  "metric sampling cadence must preserve train accuracy history");
    }
    CheckNear(first_final.metrics.val_loss_history[0],
              every_second.metrics.val_loss_history[0],
              1e-6,
              "metric sampling cadence must preserve final validation loss");
    Check(first_final.parameters.size() == every_second.parameters.size(),
          "metric sampling cadence runs should expose the same parameter set");
    for (const auto& [name, first_values] : first_final.parameters) {
        const auto found = every_second.parameters.find(name);
        Check(found != every_second.parameters.end() &&
                  found->second.size() == first_values.size(),
              "metric sampling cadence runs should preserve parameter shape for " +
                  name);
        for (size_t i = 0; i < first_values.size(); ++i) {
            CheckNear(first_values[i], found->second[i], 1e-6,
                      "metric sampling cadence must preserve final parameter " +
                          name);
        }
    }

    Check(first_final.loss_scalar_readbacks == 4 &&
              first_final.accuracy_scalar_readbacks == 4,
          "first/final cadence should read each training scalar twice per epoch");
    Check(every_second.loss_scalar_readbacks == 6 &&
              every_second.accuracy_scalar_readbacks == 6,
          "interval-2 cadence should read each training scalar three times per epoch");
    Check(first_final.reporting_cadence.find("first and final batch") !=
              std::string::npos &&
              every_second.reporting_cadence.find("every 2 batches") !=
              std::string::npos,
          "trace should report each effective metric sampling cadence");
}

class StopBeforeSecondEpochHook final
    : public cyxwiz::plugin::ITrainingHook {
public:
    void OnTrainingStart(cyxwiz::plugin::TrainingContext&) override {
        ++training_start_count;
    }

    void OnTrainingEnd(cyxwiz::plugin::TrainingContext&) override {
        ++training_end_count;
    }

    void OnEpochStart(cyxwiz::plugin::TrainingContext&) override {
        ++epoch_start_count;
    }

    void OnEpochEnd(cyxwiz::plugin::TrainingContext&) override {
        ++epoch_end_count;
    }

    bool ShouldStopEarly(
        const cyxwiz::plugin::TrainingContext& context) override {
        ++stop_poll_count;
        return context.current_epoch >= 2;
    }

    int training_start_count = 0;
    int training_end_count = 0;
    int epoch_start_count = 0;
    int epoch_end_count = 0;
    int stop_poll_count = 0;
};

void TestPluginStopLifecycle(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "plugin-stop");
    config.epochs = 4;
    config.log_interval = 0;

    StopBeforeSecondEpochHook hook;
    constexpr const char* kPluginId = "plan39.lifecycle.plugin-stop";
    auto& hooks = cyxwiz::plugin::PluginTrainingHookManager::Instance();
    hooks.RegisterHook(kPluginId, &hook);

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    int batch_callback_count = 0;
    int epoch_callback_count = 0;
    int completion_callback_count = 0;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) { ++batch_callback_count; },
        [&](int, float, float, float, float, float) {
            ++epoch_callback_count;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });
    hooks.RemoveByPlugin(kPluginId);

    Check(batch_callback_count == 2 && epoch_callback_count == 1,
          "plugin stop should execute exactly one complete epoch");
    Check(completion_callback_count == 1,
          "plugin stop should invoke completion exactly once");
    Check(final_metrics.current_epoch == 1 &&
              final_metrics.last_executed_epoch == 1 &&
              final_metrics.loss_history.size() == 1,
          "plugin stop should preserve one executed epoch");
    Check(final_metrics.terminal_status == "early_stopped" &&
              final_metrics.terminal_reason ==
                  "plugin_requested_early_stop",
          "plugin stop should report its exact terminal truth");
    Check(hook.training_start_count == 1 &&
              hook.training_end_count == 1 &&
              hook.epoch_start_count == 1 &&
              hook.epoch_end_count == 1 &&
              hook.stop_poll_count == 2,
          "plugin stop should preserve exact plugin callback cadence");
    CheckTerminalEvidence(
        "early_stopped", "plugin_requested_early_stop", 1, 1,
        "plugin stop");
}

void TestUserCancellationLifecycle(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "user-cancel");
    config.epochs = 3;
    config.log_interval = 0;

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    int batch_callback_count = 0;
    int epoch_callback_count = 0;
    int completion_callback_count = 0;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        config.epochs,
        config.batch_size,
        [&](int, int, int, float, float) {
            ++batch_callback_count;
            executor.Stop();
        },
        [&](int, float, float, float, float, float) {
            ++epoch_callback_count;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            ++completion_callback_count;
            final_metrics = metrics;
        });

    Check(batch_callback_count == 1 && epoch_callback_count == 0,
          "user cancellation should stop after one partial-epoch batch");
    Check(completion_callback_count == 1,
          "user cancellation should invoke completion exactly once");
    Check(final_metrics.current_epoch == 1 &&
              final_metrics.last_executed_epoch == 0 &&
              final_metrics.current_batch == 1 &&
              final_metrics.loss_history.empty(),
          "user cancellation should distinguish active from completed epoch");
    Check(final_metrics.terminal_status == "cancelled" &&
              final_metrics.terminal_reason == "user_cancelled",
          "user cancellation should report its exact terminal truth");
    CheckTerminalEvidence(
        "cancelled", "user_cancelled", 1, 0, "user cancellation");
}

void TestInjectedRuntimeFailureLifecycle(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const std::filesystem::path& work_dir) {
    auto config = MakeConfig(work_dir / "runtime-failure");
    config.epochs = 3;
    config.log_interval = 0;

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    int completion_callback_count = 0;
    bool threw = false;
    try {
        executor.Train(
            config.epochs,
            config.batch_size,
            [](int, int, int, float, float) {
                throw std::runtime_error("injected_lifecycle_failure");
            },
            nullptr,
            [&](const cyxwiz::TrainingMetrics&) {
                ++completion_callback_count;
            });
    } catch (const std::runtime_error& error) {
        threw = std::string(error.what()).find(
                    "injected_lifecycle_failure") != std::string::npos;
    }

    const auto final_metrics = executor.GetMetrics();
    Check(threw, "injected runtime failure should propagate to the caller");
    Check(completion_callback_count == 0,
          "failed training should not invoke the success completion callback");
    Check(final_metrics.is_complete && !final_metrics.is_training &&
              !final_metrics.is_paused,
          "failed training should close its lifecycle state");
    Check(final_metrics.current_epoch == 1 &&
              final_metrics.last_executed_epoch == 0,
          "failed training should preserve its partial-epoch truth");
    Check(final_metrics.terminal_status == "failed" &&
              final_metrics.terminal_reason.find(
                  "injected_lifecycle_failure") != std::string::npos,
          "failed training should report a non-empty coded reason");
    CheckTerminalEvidence(
        "failed", final_metrics.terminal_reason, 1, 0,
        "injected runtime failure");
}

enum class PausedControlAction {
    Resume,
    Cancel,
};

using PausedExecutorFactory = std::function<
    std::unique_ptr<cyxwiz::TrainingExecutor>(const std::filesystem::path&)>;

void CheckPausedControlCase(
    const PausedExecutorFactory& make_executor,
    const std::filesystem::path& work_dir,
    const std::string& mode,
    int expected_batches,
    bool strict_residency,
    PausedControlAction action) {
    const bool cancel = action == PausedControlAction::Cancel;
    const std::string action_name = cancel ? "cancel" : "resume";
    const std::string label = mode + " pause/" + action_name;
    auto executor = make_executor(work_dir / (mode + "-" + action_name));
    std::atomic<int> batch_callback_count{0};
    std::atomic<bool> pause_requested{false};
    cyxwiz::TrainingMetrics final_metrics;
    int completion_callback_count = 0;
    std::exception_ptr training_error;
    std::thread training_thread([&]() {
        try {
            executor->Train(
                1,
                2,
                [&](int, int batch, int, float, float) {
                    ++batch_callback_count;
                    if (batch == 1) {
                        executor->Pause();
                        pause_requested.store(true);
                    }
                },
                nullptr,
                [&](const cyxwiz::TrainingMetrics& metrics) {
                    ++completion_callback_count;
                    final_metrics = metrics;
                });
        } catch (...) {
            training_error = std::current_exception();
        }
    });

    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(10);
    while (!pause_requested.load() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!pause_requested.load() || !executor->IsPaused()) {
        executor->Stop();
        training_thread.join();
        Check(false, label + " should reach the paused state after batch one");
    }
    const auto paused_metrics = executor->GetMetrics();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto still_paused_metrics = executor->GetMetrics();
    Check(paused_metrics.current_batch == 1 &&
              paused_metrics.optimizer_step_count == 1 &&
              still_paused_metrics.current_batch == 1 &&
              still_paused_metrics.optimizer_step_count == 1 &&
              batch_callback_count.load() == 1,
          label + " should not advance or duplicate work while paused");

    if (cancel) {
        executor->Stop();
    } else {
        executor->Resume();
    }
    training_thread.join();
    Check(training_error == nullptr,
          label + " training should not throw");
    Check(completion_callback_count == 1,
          label + " should invoke completion exactly once");

    if (cancel) {
        Check(batch_callback_count.load() == 1 &&
                  final_metrics.optimizer_step_count == 1 &&
                  final_metrics.current_batch == 1,
              label + " must not consume work after cancellation");
        Check(final_metrics.current_epoch == 1 &&
                  final_metrics.last_executed_epoch == 0 &&
                  final_metrics.loss_history.empty() &&
                  final_metrics.terminal_status == "cancelled" &&
                  final_metrics.terminal_reason == "user_cancelled" &&
                  !final_metrics.is_paused,
              label + " should preserve partial-epoch cancellation truth");
        CheckTerminalEvidence(
            "cancelled", "user_cancelled", 1, 0, label);
    } else {
        Check(batch_callback_count.load() == expected_batches &&
                  final_metrics.optimizer_step_count == expected_batches &&
                  final_metrics.current_batch == expected_batches,
              label + " should execute each batch and optimizer step once");
        Check(final_metrics.current_epoch == 1 &&
                  final_metrics.last_executed_epoch == 1 &&
                  final_metrics.loss_history.size() == 1 &&
                  final_metrics.terminal_status == "completed" &&
                  final_metrics.terminal_reason == "completed_all_epochs" &&
                  !final_metrics.is_paused,
              label + " should preserve normal terminal truth");
        CheckTerminalEvidence(
            "completed", "completed_all_epochs", 1, 1, label);
    }

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.effective_backend.rfind("arrayfire_", 0) == 0,
          label + " should retain effective ArrayFire execution truth");
    if (strict_residency) {
        Check(trace.native_cpu_fallback_count == 0 &&
                  trace.fallback_policy == "forbid_native_cpu_fallback",
              label + " should retain strict zero-fallback execution truth");
    } else {
        Check(trace.fallback_policy == "allow_native_cpu_fallback",
              label + " should expose its declared compatibility policy");
    }
}

void CheckPausedControlMode(
    const PausedExecutorFactory& make_executor,
    const std::filesystem::path& work_dir,
    const std::string& mode,
    int expected_batches,
    bool strict_residency) {
    CheckPausedControlCase(
        make_executor, work_dir, mode, expected_batches, strict_residency,
        PausedControlAction::Resume);
    CheckPausedControlCase(
        make_executor, work_dir, mode, expected_batches, strict_residency,
        PausedControlAction::Cancel);
}

void TestPausedControlAcrossModernDatasetModes(
    const std::filesystem::path& work_dir) {
    const auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeTrainingTable(), "paused_control_arrow");

    const auto make_tabular_config = [&](const std::filesystem::path& path) {
        auto config = MakeConfig(path);
        config.train_ratio = 1.0f;
        config.val_ratio = 0.0f;
        config.test_ratio = 0.0f;
        config.has_data_split = true;
        config.log_interval = 0;
        config.forbid_native_cpu_fallback = true;
        return config;
    };

    CheckPausedControlMode(
        [&](const std::filesystem::path& path) {
            return std::make_unique<cyxwiz::TrainingExecutor>(
                make_tabular_config(path), dataset, "label");
        },
        work_dir,
        "arrow",
        3,
        true);

    const auto parquet_path = work_dir / "paused-control.parquet";
    WriteParquetWithRowGroupSize(*dataset, parquet_path.string(), 1);
    const auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), "paused_control_parquet");
    Check(parquet_dataset != nullptr,
          "paused-control Parquet fixture should open");
    CheckPausedControlMode(
        [&](const std::filesystem::path& path) {
            return std::make_unique<cyxwiz::TrainingExecutor>(
                make_tabular_config(path), parquet_dataset, "label");
        },
        work_dir,
        "parquet",
        3,
        true);

    CheckPausedControlMode(
        [&](const std::filesystem::path& path) {
            auto config = make_tabular_config(path);
            return std::make_unique<cyxwiz::TrainingExecutor>(
                std::move(config), std::make_unique<PhaseTrackingBatcher>());
        },
        work_dir,
        "external",
        2,
        true);

    const std::vector<cyxwiz::NERSequenceRow> sequence_rows = {
        {{"John", "lives", "in", "Berlin"}, {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Mary", "works", "in", "Paris"}, {},
         {"B-PER", "O", "O", "B-LOC"}},
        {{"Alice", "visits", "New", "York"}, {},
         {"B-PER", "O", "B-LOC", "I-LOC"}},
        {{"Bob", "works", "in", "London"}, {},
         {"B-PER", "O", "O", "B-LOC"}},
    };
    cyxwiz::NERSequenceBuilderConfig builder_config;
    builder_config.use_pos_tags = false;
    builder_config.token_vocabulary.lowercase = true;
    builder_config.batcher.batch_size = 2;
    builder_config.batcher.max_sequence_length = 4;
    builder_config.batcher.shuffle = false;
    builder_config.batcher.tag_ignore_index = -100;
    builder_config.batcher.train_indices = {0, 1, 2, 3};
    const auto sequence = cyxwiz::BuildNERSequenceData(
        sequence_rows, builder_config);

    CheckPausedControlMode(
        [&](const std::filesystem::path& path) {
            cyxwiz::TrainingConfiguration config;
            config.dataset_name = "paused_control_sequence";
            config.input_size = 4;
            config.input_shape = {4};
            config.output_size = sequence.tag_vocabulary.Size();
            config.loss_type = gui::NodeType::CrossEntropyLoss;
            config.loss_params["ignore_index"] = "-100";
            config.optimizer_type = gui::NodeType::SGD;
            config.learning_rate = 0.01f;
            config.batch_size = 2;
            config.sequence_batch.enabled = true;
            config.sequence_batch.ignore_index = -100;
            config.save_best_checkpoint = false;
            config.log_interval = 0;
            config.forbid_native_cpu_fallback = false;
            config.checkpoint_dir = path.string();

            cyxwiz::CompiledLayer embedding;
            embedding.type = gui::NodeType::Embedding;
            embedding.parameters["num_embeddings"] =
                std::to_string(sequence.token_vocabulary.Size());
            embedding.parameters["embedding_dim"] = "6";
            config.layers.push_back(embedding);

            cyxwiz::CompiledLayer token_head;
            token_head.type = gui::NodeType::TimeDistributed;
            token_head.units = static_cast<int>(
                sequence.tag_vocabulary.Size());
            config.layers.push_back(token_head);

            auto batcher = std::make_unique<cyxwiz::SequenceBatcher>(
                sequence.samples, sequence.batcher_config);
            return std::make_unique<cyxwiz::TrainingExecutor>(
                std::move(config), std::move(batcher),
                sequence.tag_vocabulary.Values());
        },
        work_dir,
        "sequence",
        2,
        false);
}

void RunExecutor(cyxwiz::TrainingExecutor& executor,
                 const std::string& label,
                 int expected_epochs = 1,
                 int expected_validation_points = 1,
                 int expected_optimizer_steps = -1) {
    int saw_epochs = 0;
    bool completed = false;
    bool saw_active_execution_context = false;
    cyxwiz::TrainingMetrics final_metrics;

    Check(!cyxwiz::HasActiveExecutionDeviceContext(),
          label + " should start without an active execution context");
    executor.Train(
        expected_epochs,
        2,
        [&](int, int, int, float, float) {
            Check(cyxwiz::HasActiveExecutionDeviceContext(),
                  label + " should hold an active execution context during batches");
            Check(cyxwiz::CurrentExecutionDeviceContext() != nullptr,
                  label + " should expose the current execution context during batches");
            saw_active_execution_context = true;
        },
        [&](int epoch,
            float train_loss,
            float,
            float val_loss,
            float,
            float) {
            Check(epoch >= 1 && epoch <= expected_epochs,
                  label + " epoch callback should report a valid epoch");
            Check(std::isfinite(train_loss), label + " train loss should be finite");
            if (val_loss >= 0.0f) {
                Check(std::isfinite(val_loss), label + " val loss should be finite");
            }
            ++saw_epochs;
        },
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(saw_epochs == expected_epochs, label + " should run each epoch callback");
    Check(saw_active_execution_context,
          label + " should run a batch under an active execution context");
    Check(!cyxwiz::HasActiveExecutionDeviceContext(),
          label + " should clear the active execution context after training");
    Check(completed, label + " should run completion callback");
    Check(final_metrics.is_complete, label + " should mark training complete");
    Check(!final_metrics.is_training, label + " should clear training state");
    Check(final_metrics.current_epoch == expected_epochs, label + " should finish expected epoch");
    Check(final_metrics.last_executed_epoch == expected_epochs,
          label + " should preserve the final executed epoch");
    Check(final_metrics.total_epochs == expected_epochs, label + " should keep total epochs");
    Check(final_metrics.terminal_status == "completed" &&
              final_metrics.terminal_reason == "completed_all_epochs",
          label + " should report exact normal-completion truth");
    Check(final_metrics.total_batches == 2, label + " should train two batches");
    const int expected_steps = expected_optimizer_steps >= 0
        ? expected_optimizer_steps
        : expected_epochs * 2;
    Check(final_metrics.optimizer_step_count == expected_steps,
          label + " should report expected optimizer step count");
    Check(final_metrics.loss_history.size() == static_cast<size_t>(expected_epochs),
          label + " should store one train loss history entry per epoch");
    Check(final_metrics.val_loss_history.size() == static_cast<size_t>(expected_validation_points),
          label + " should store validation history only for validation epochs");
    Check(std::isfinite(final_metrics.train_loss),
          label + " final train loss should be finite");
    Check(std::isfinite(final_metrics.val_loss),
          label + " final validation loss should be finite");
}

#ifndef NDEBUG
void TestAllowedTrainingRecordsForcedLinearFallback(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        return;
    }

    ScopedEnvVar env(kForceFallbackEnv, "LinearLayer::Forward");
    auto allowed_config = config;
    allowed_config.forbid_native_cpu_fallback = false;
    cyxwiz::TrainingExecutor executor(allowed_config, dataset, "label");

    bool completed = false;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            completed = metrics.is_complete;
        });

    Check(completed,
          "allowed fallback training should finish with native CPU fallback");
    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.available, "allowed fallback training should leave a trace");
    Check(trace.native_cpu_fallback_count > 0,
          "allowed fallback training should record native CPU fallback count");
    Check(trace.residency_verdict == "native_cpu_fallback_observed",
          "allowed fallback training should record fallback residency verdict");

    bool saw_linear_forward = false;
    int execution_context_bind_count = 0;
    for (const auto& event : trace.recent_events) {
        if (event.stage == "ExecutionDeviceContext.Bind") {
            ++execution_context_bind_count;
            Check(event.status == "ok",
                  "execution context bind should be valid");
            Check(event.execution_platform == "arrayfire",
                  "execution context should record ArrayFire platform");
            Check(!event.requested_backend.empty(),
                  "execution context should record requested backend");
            Check(!event.effective_backend.empty(),
                  "execution context should record effective backend");
            Check(event.requested_backend == event.effective_backend,
                  "current run context should bind requested/effective backend");
            Check(!event.execution_context_id.empty(),
                  "execution context should record stable identity");
            Check(event.capability_generation > 0,
                  "execution context should record capability generation");
            Check(event.activation_succeeded,
                  "execution context should record exact activation success");
            Check(event.execution_validated,
                  "execution context should record bounded execution validation");
            Check(event.preflight_stage == "complete",
                  "execution context should record completed preflight stage");
            Check(event.fallback_policy == "allow_native_cpu_fallback",
                  "execution context should record fallback policy");
        }
        if (!event.native_cpu_fallback) {
            continue;
        }
        if (event.fallback_operation == "LinearLayer::Forward") {
            saw_linear_forward = true;
            Check(event.status == "warning",
                  "allowed fallback event should be a warning");
            Check(event.fallback_target == "native_cpu",
                  "fallback target should distinguish native CPU");
            Check(event.fallback_policy == "allow_native_cpu_fallback",
                  "allowed fallback event should record allow policy");
            Check(!event.compute_backend.empty(),
                  "fallback event should record selected ArrayFire backend");
        }
    }
    Check(execution_context_bind_count == 1,
          "training trace should record one execution device context bind");
    Check(saw_linear_forward,
          "allowed fallback trace should name Linear forward");

    const auto runtime_events =
        cyxwiz::RuntimeLogStore::Instance().Snapshot().events;
    bool saw_device_lifecycle = false;
    bool saw_fallback_lifecycle = false;
    for (const auto& event : runtime_events) {
        if (event.run_id != trace.run_id) continue;
        if (event.event_name == "ExecutionDeviceContext.Bind") {
            saw_device_lifecycle = event.category == "device" &&
                event.backend == trace.effective_backend &&
                event.device_id == trace.effective_device_id;
        }
        if (event.event_name == "ArrayFire.NativeCpuFallback") {
            saw_fallback_lifecycle = event.category == "training" &&
                event.level == cyxwiz::RuntimeLogLevel::Warning &&
                event.primary_error_code == "CW-G-0501" &&
                event.message.find("LinearLayer::Forward") !=
                    std::string::npos;
        }
    }
    Check(saw_device_lifecycle,
          "runtime log should retain one structured run-bound device bind");
    Check(saw_fallback_lifecycle,
          "runtime log should retain structured native CPU fallback evidence");
}

void TestStrictTrainingRejectsForcedLinearFallback(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        return;
    }

    ScopedEnvVar env(kForceFallbackEnv, "LinearLayer::Forward");
    auto strict_config = config;
    strict_config.forbid_native_cpu_fallback = true;
    cyxwiz::TrainingExecutor executor(strict_config, dataset, "label");

    bool threw = false;
    try {
        executor.Train(1, 2);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string message = e.what();
        Check(message.find("LinearLayer::Forward") != std::string::npos,
              "strict training fallback error should name Linear forward");
        Check(message.find("native CPU fallback is forbidden") !=
                  std::string::npos,
              "strict training fallback error should forbid native CPU fallback");
    }

    Check(threw, "strict training should reject forced Linear fallback");
    Check(!executor.IsTraining(),
          "strict training fallback failure should clear training state");
}

void TestStrictTrainingSkipsFirstBatchDebugHostDump(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    if (!cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        return;
    }

    auto strict_config = config;
    strict_config.forbid_native_cpu_fallback = true;
    cyxwiz::TrainingExecutor executor(strict_config, dataset, "label");

    bool completed = false;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            completed = metrics.is_complete;
        });

    Check(completed,
          "strict training should complete when supported operations stay ArrayFire-backed");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    bool saw_skip_event = false;
    bool saw_loss_boundary = false;
    for (const auto& event : trace.recent_events) {
        if (event.stage == "TrainingExecutor.DebugSampleDump") {
            saw_skip_event = true;
            Check(event.status == "ok",
                  "debug sample dump skip should be recorded as an ok runtime event");
            Check(event.message.find("strict ArrayFire residency") !=
                      std::string::npos,
                  "debug sample dump skip should explain strict residency");
        }
        if (event.stage == "TrainingExecutor.OutputBoundary" &&
            event.message.find("loss_scalar_readback") != std::string::npos) {
            saw_loss_boundary = true;
            Check(event.status == "ok",
                  "loss scalar output boundary should be an ok runtime event");
            Check(event.message.find("not native CPU compute fallback") !=
                      std::string::npos,
                  "loss scalar output boundary should distinguish reporting from fallback");
        }
    }
    Check(saw_skip_event,
          "strict training should record skipped first-batch debug host dump");
    Check(saw_loss_boundary,
          "strict training should declare scalar loss readback boundary");
}
#endif

void TestPendingExecutionDeviceSelectionAppliesAndClears(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    auto* current_device = cyxwiz::Device::GetCurrentDevice();
    if (current_device == nullptr) {
        return;
    }

    cyxwiz::SetPendingExecutionDeviceSelection(
        current_device->GetType(),
        current_device->GetDeviceId());
    Check(cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
          "pending execution device selection should be queued before training");

    cyxwiz::TrainingExecutor executor(config, dataset, "label");
    bool saw_batch = false;
    executor.Train(
        1,
        2,
        [&](int, int, int, float, float) {
            Check(cyxwiz::HasActiveExecutionDeviceContext(),
                  "pending device training should bind active context");
            saw_batch = true;
        },
        nullptr,
        nullptr);

    Check(saw_batch,
          "pending execution device selection should still allow training");
    Check(!cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
          "pending execution device selection should clear after apply");
}

void TestPendingExecutionDeviceSelectionRejectsNonArrayFireBackends() {
    bool rejected_metal = false;
    try {
        cyxwiz::SetPendingExecutionDeviceSelection(
            cyxwiz::DeviceType::METAL,
            0);
    } catch (const std::invalid_argument&) {
        rejected_metal = true;
    }
    Check(rejected_metal,
          "pending execution device selection should reject Metal as non-ArrayFire");

    bool rejected_vulkan = false;
    try {
        cyxwiz::SetPendingExecutionDeviceSelection(
            cyxwiz::DeviceType::VULKAN,
            0);
    } catch (const std::invalid_argument&) {
        rejected_vulkan = true;
    }
    Check(rejected_vulkan,
          "pending execution device selection should reject Vulkan as non-ArrayFire");
}

std::optional<cyxwiz::DeviceInfo> FirstDeviceOfType(
    const std::vector<cyxwiz::DeviceInfo>& devices,
    cyxwiz::DeviceType type) {
    for (const auto& device : devices) {
        if (device.type == type) {
            return device;
        }
    }
    return std::nullopt;
}

void TestRunPreflightEnforcesRouteQualification() {
    const auto inventory = cyxwiz::Device::GetAvailableDevices();
    const auto rejected = std::find_if(
        inventory.begin(), inventory.end(),
        [](const cyxwiz::DeviceInfo& device) {
            return device.type == cyxwiz::DeviceType::CUDA ||
                   device.type == cyxwiz::DeviceType::OPENCL;
        });
    if (rejected == inventory.end()) {
        std::cout
            << "SKIP: route qualification recovery requires an accelerator route\n";
        return;
    }

    auto snapshot = cyxwiz::test::MakeQualifiedRouteSnapshot(
        inventory, "test-run-preflight-rejection");
    for (auto& route : snapshot.routes) {
        if (route.type == rejected->type &&
            route.device_id == rejected->device_id) {
            route.pass_count = 0;
            route.failure_count = 1;
            route.certified = false;
        }
    }
    cyxwiz::InstallRouteQualificationSnapshot(std::move(snapshot));

    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(
        rejected->type, rejected->device_id);
    bool strict_rejected = false;
    try {
        (void)cyxwiz::PrepareExecutionDeviceForRun(
            cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        strict_rejected =
            !message.empty() &&
            message.find("test-run-preflight-rejection") == std::string::npos;
    }
    Check(strict_rejected,
          "strict run preflight should reject an uncertified requested route "
          "without exposing the internal evidence identifier");

    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(
        rejected->type, rejected->device_id);
    const auto recovered = cyxwiz::PrepareExecutionDeviceForRun(
        cyxwiz::ArrayFireFallbackPolicy::AllowNativeCpuFallback);
    Check(recovered.selection_fallback_applied,
          "compatibility preflight should record ArrayFire CPU route recovery");
    Check(!recovered.requested_qualification.qualified,
          "recovered context should retain rejected requested qualification");
    Check(recovered.effective_backend == "arrayfire_cpu",
          "compatibility preflight should recover to ArrayFire CPU");
    Check(recovered.effective_qualification.qualified,
          "compatibility preflight should require certified CPU recovery");
    Check(recovered.effective_qualification.matrix_id ==
              "test-run-preflight-rejection",
          "recovered context should retain effective qualification matrix");

    cyxwiz::test::InstallQualifiedRouteSnapshot(inventory);
}

#ifndef NDEBUG
void TestStrictArrayFireCpuDenseTrainingDoesNotFallback(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = FirstDeviceOfType(devices, cyxwiz::DeviceType::CPU);
    Check(cpu.has_value(),
          "strict ArrayFire CPU regression requires ArrayFire CPU discovery");

    cyxwiz::ClearPendingExecutionDeviceSelection();
    cyxwiz::SetPendingExecutionDeviceSelection(
        cyxwiz::DeviceType::CPU,
        cpu->device_id);
    cyxwiz::ClearNextRunExecutionPolicy();
    cyxwiz::SetNextRunExecutionPolicy(
        cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback);

    auto strict_config = config;
    strict_config.forbid_native_cpu_fallback = false;
    strict_config.log_interval = 0;
    strict_config.save_best_checkpoint = true;
    strict_config.loss_params["class_weight"] = "manual";
    strict_config.loss_params["class_weights"] = "[1.0, 2.0]";
    strict_config.loss_params["label_smoothing"] = "0.1";
    cyxwiz::TrainingExecutor executor(strict_config, dataset, "label");

    bool completed = false;
    int batch_callback_count = 0;
    int callback_total_batches = 0;
    executor.Train(
        1,
        2,
        [&](int, int batch, int total_batches, float, float) {
            ++batch_callback_count;
            callback_total_batches = total_batches;
            Check(batch == batch_callback_count,
                  "strict training batch callback should remain responsive every batch");
        },
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            completed = metrics.is_complete;
        });

    Check(completed,
          "strict ArrayFire CPU dense training should complete");
    Check(callback_total_batches >= 2,
          "cadence regression requires at least two training batches");
    Check(batch_callback_count == callback_total_batches,
          "metric throttling must not throttle batch progress callbacks");
    Check(!cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
          "strict ArrayFire CPU run should consume pending selection");
    Check(cyxwiz::GetNextRunExecutionPolicy() ==
              cyxwiz::ArrayFireFallbackPolicy::ForbidNativeCpuFallback,
          "GUI execution policy preference should persist for later runs");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    const auto latest_trace =
        cyxwiz::TrainingTraceCollector::LatestTrace();
    Check(latest_trace.run_id == trace.run_id,
          "latest trace authority should prefer the active in-memory run");
    Check(latest_trace.requested_backend == trace.requested_backend,
          "latest trace authority should preserve requested backend truth");
    Check(latest_trace.effective_backend == trace.effective_backend,
          "latest trace authority should preserve effective backend truth");
    Check(latest_trace.placement_fingerprint == trace.placement_fingerprint,
          "latest trace authority should preserve placement fingerprint truth");
    Check(trace.native_cpu_fallback_count == 0,
          "strict ArrayFire CPU dense training should record zero native CPU fallback events");
    Check(trace.residency_verdict ==
              "strict_arrayfire_declared_boundaries",
          "strict ArrayFire CPU run should record strict residency verdict");
    Check(trace.execution_platform == "arrayfire",
          "strict ArrayFire CPU summary should record ArrayFire platform");
    Check(trace.requested_backend == "arrayfire_cpu",
          "strict ArrayFire CPU summary should record requested CPU backend");
    Check(trace.effective_backend == "arrayfire_cpu",
          "strict ArrayFire CPU summary should record effective CPU backend");
    Check(trace.requested_route_qualified,
          "strict ArrayFire CPU summary should record requested certification");
    Check(trace.effective_route_qualified,
          "strict ArrayFire CPU summary should record effective certification");
    Check(trace.requested_qualification_matrix_id ==
              "test-qualified-routes",
          "strict ArrayFire CPU summary should retain qualification matrix");
    Check(!trace.identity_confidence.empty(),
          "strict ArrayFire CPU summary should retain identity confidence");
    Check(trace.fallback_policy == "forbid_native_cpu_fallback",
          "strict ArrayFire CPU summary should record strict fallback policy");
    Check(trace.declared_output_boundary_count > 0,
          "strict ArrayFire CPU summary should count declared output boundaries");
    Check(trace.arrayfire_host_sync_count > 0,
          "strict ArrayFire CPU summary should count ArrayFire host synchronizations");
    Check(trace.arrayfire_host_sync_bytes > 0,
          "strict ArrayFire CPU summary should count ArrayFire host synchronization bytes");
    Check(trace.transfer_event_count >= trace.arrayfire_host_sync_count,
          "strict ArrayFire CPU summary should count transfer events");
    Check(trace.transfer_known_bytes >= trace.arrayfire_host_sync_bytes,
          "strict ArrayFire CPU summary should count known transfer bytes");
    Check(trace.transfer_summary.find("arrayfire_to_host") !=
              std::string::npos,
          "strict ArrayFire CPU summary should explain transfer modes and reasons");
    Check(trace.synchronization_event_count == trace.arrayfire_host_sync_count,
          "strict ArrayFire CPU summary should count synchronization events");
    Check(trace.synchronization_known_bytes == trace.arrayfire_host_sync_bytes,
          "strict ArrayFire CPU summary should count synchronization bytes");
    Check(trace.synchronization_summary.find("tensor_host_materialization") !=
              std::string::npos,
          "strict ArrayFire CPU summary should explain synchronization reasons");
    uint64_t grouped_host_sync_count = 0;
    uint64_t grouped_host_sync_bytes = 0;
    bool saw_loss_scalar_group = false;
    bool saw_metric_scalar_group = false;
    bool saw_layout_conversion_group = false;
    bool saw_checkpoint_output_group = false;
    bool saw_unknown_group = false;
    uint64_t cadence_loss_scalar_readbacks = 0;
    uint64_t cadence_metric_scalar_readbacks = 0;
    for (const auto& group : trace.arrayfire_host_sync_groups) {
        grouped_host_sync_count += group.event_count;
        grouped_host_sync_bytes += group.bytes;
        Check(!group.reason.empty(),
              "host sync groups should retain the synchronization reason");
        saw_loss_scalar_group = saw_loss_scalar_group ||
            group.category == "loss_scalar_readback";
        saw_metric_scalar_group = saw_metric_scalar_group ||
            group.category == "metric_scalar_readback";
        if (group.operation == "TrainingExecutor::ReadAccumulatedLoss") {
            cadence_loss_scalar_readbacks += group.event_count;
        }
        if (group.operation ==
              "TrainingExecutor::ReadAccumulatedAccuracy") {
            cadence_metric_scalar_readbacks += group.event_count;
        }
        saw_layout_conversion_group = saw_layout_conversion_group ||
            group.category == "layout_conversion";
        saw_checkpoint_output_group = saw_checkpoint_output_group ||
            group.category == "checkpoint_output";
        saw_unknown_group = saw_unknown_group ||
            group.category == "unknown";
    }
    Check(grouped_host_sync_count == trace.arrayfire_host_sync_count,
          "host sync groups should account for every synchronization event");
    Check(grouped_host_sync_bytes == trace.arrayfire_host_sync_bytes,
          "host sync groups should account for every synchronized byte");
    Check(saw_loss_scalar_group,
          "host sync groups should attribute loss scalar readbacks");
    Check(saw_metric_scalar_group,
          "host sync groups should attribute metric scalar readbacks");
    Check(cadence_loss_scalar_readbacks == 2,
          "first/final cadence should read one loss scalar at two boundaries");
    Check(cadence_metric_scalar_readbacks == 2,
          "first/final cadence should read one metric scalar at two boundaries");
    Check(!saw_layout_conversion_group,
          "strict dense training should not synchronize for 2D layout conversion");
    Check(saw_checkpoint_output_group,
          "checkpoint parameter reads should be a named output boundary");
    Check(!saw_unknown_group,
          "strict dense training should not record unattributed host synchronization");
    Check(!trace.arrayfire_host_sync_summary.empty(),
          "strict ArrayFire CPU summary should format host sync groups");
    Check(!trace.placement_fingerprint.empty(),
          "strict ArrayFire CPU summary should record placement fingerprint");
    Check(trace.placement_entry_count >
              static_cast<uint64_t>(strict_config.backend_placements.size()),
          "strict ArrayFire CPU summary should add dense runtime placement entries");
    Check(!trace.placement_summary.empty(),
          "strict ArrayFire CPU summary should record placement summary");
    Check(trace.placement_summary.find("=gpu(") == std::string::npos,
          "strict ArrayFire CPU placement must not retain stale GPU entries");
    Check(trace.placement_summary.find("=arrayfire_cpu(") !=
              std::string::npos,
          "strict ArrayFire CPU placement should resolve compiler entries to the bound backend");
    Check(trace.placement_summary.find("dataset_ingress") !=
              std::string::npos,
          "placement summary should include dataset ingress");
    Check(trace.placement_summary.find("ModelForward.Dense") !=
              std::string::npos,
          "placement summary should include dense forward");
    Check(trace.placement_summary.find("Loss.") != std::string::npos,
          "placement summary should include loss stage");
    Check(trace.placement_summary.find("metrics") != std::string::npos,
          "placement summary should include metrics stage");
    Check(trace.placement_summary.find("optimizer") != std::string::npos,
          "placement summary should include optimizer stage");
    Check(trace.placement_summary.find("loss_scalar_readback") !=
              std::string::npos,
          "placement summary should include declared scalar output boundary");

    bool saw_cpu_bind = false;
    bool saw_placement_plan = false;
    bool saw_cpu_stage = false;
    bool saw_host_sync = false;
    bool saw_reporting_cadence = false;
    bool saw_timed_batch_fetch = false;
    bool saw_timed_optimizer_step = false;
    for (const auto& event : trace.recent_events) {
        if (!event.stage_backend.empty()) {
            Check(event.stage_backend == "arrayfire_cpu",
                  "active run events should record ArrayFire CPU stage backend");
            Check(event.stage_device_id == cpu->device_id,
                  "active run events should record ArrayFire CPU stage device id");
            if (event.stage != "ExecutionDeviceContext.Bind") {
                saw_cpu_stage = true;
            }
        }
        if (event.stage == "TrainingExecutor.PlacementPlan") {
            saw_placement_plan = true;
            Check(event.placement_fingerprint == trace.placement_fingerprint,
                  "placement plan event should match summary fingerprint");
            Check(event.placement_entry_count ==
                      trace.placement_entry_count,
                  "placement plan event should match summary entry count");
            Check(event.placement_summary == trace.placement_summary,
                  "placement plan event should match summary placement text");
        }
        if (event.stage == "TrainingExecutor.ReportingCadence") {
            saw_reporting_cadence = true;
            Check(event.message.find("first and final batch") !=
                      std::string::npos,
                  "trace should record the effective first/final metric cadence");
        }
        if (event.stage == "GetNextBatch") {
            saw_timed_batch_fetch = true;
            Check(event.duration_ms >= 0.0f,
                  "batch fetch trace should carry host wall-clock duration");
        }
        if (event.stage == "UpdateParameters") {
            saw_timed_optimizer_step = true;
            Check(event.duration_ms >= 0.0f,
                  "optimizer trace should carry host wall-clock duration");
        }
        if (event.stage == "ArrayFire.HostSync") {
            saw_host_sync = true;
            Check(event.transfer_mode == "arrayfire_to_host",
                  "host sync event should identify ArrayFire-to-host transfer mode");
            Check(event.arrayfire_host_sync_bytes > 0,
                  "host sync event should record byte count");
            Check(!event.arrayfire_host_sync_category.empty(),
                  "host sync event should record an attribution category");
            Check(!event.arrayfire_host_sync_shape.empty(),
                  "host sync event should record tensor shape");
            Check(!event.arrayfire_host_sync_dtype.empty(),
                  "host sync event should record tensor dtype");
            Check(!event.arrayfire_host_sync_layout.empty(),
                  "host sync event should record tensor layout");
        }
        if (event.stage != "ExecutionDeviceContext.Bind") {
            continue;
        }
        saw_cpu_bind = true;
        Check(event.execution_platform == "arrayfire",
              "strict CPU run should bind the ArrayFire platform");
        Check(event.requested_backend == "arrayfire_cpu",
              "strict CPU run should record requested ArrayFire CPU backend");
        Check(event.effective_backend == "arrayfire_cpu",
              "strict CPU run should activate ArrayFire CPU backend");
        Check(event.requested_device_id == cpu->device_id,
              "strict CPU run should record requested CPU device id");
        Check(event.effective_device_id == cpu->device_id,
              "strict CPU run should record effective CPU device id");
        Check(event.fallback_policy == "forbid_native_cpu_fallback",
              "strict CPU run should record forbidden native CPU fallback policy");
    }
    Check(saw_cpu_bind,
          "strict ArrayFire CPU run should record execution context bind");
    Check(saw_placement_plan,
          "strict ArrayFire CPU run should record placement plan fingerprint");
    Check(saw_cpu_stage,
          "strict ArrayFire CPU run should record stage backend/device fields");
    Check(saw_host_sync,
          "strict ArrayFire CPU run should record at least one host sync event");
    Check(saw_reporting_cadence,
          "strict ArrayFire CPU run should record metric reporting cadence");
    Check(saw_timed_batch_fetch,
          "strict training should record batch fetch timing");
    Check(saw_timed_optimizer_step,
          "strict training should record optimizer timing");

    const auto persisted_trace =
        cyxwiz::TrainingTraceCollector::LoadLastTrace();
    Check(persisted_trace.has_value(),
          "strict ArrayFire CPU run should persist the training trace");
    if (persisted_trace.has_value()) {
        Check(persisted_trace->requested_route_qualified,
              "persisted trace should retain requested route certification");
        Check(persisted_trace->effective_route_qualified,
              "persisted trace should retain effective route certification");
        Check(persisted_trace->effective_qualification_matrix_id ==
                  "test-qualified-routes",
              "persisted trace should retain qualification matrix identity");
        Check(persisted_trace->transfer_event_count ==
                  trace.transfer_event_count,
              "persisted trace should preserve transfer event count");
        Check(persisted_trace->transfer_known_bytes ==
                  trace.transfer_known_bytes,
              "persisted trace should preserve known transfer bytes");
        Check(persisted_trace->transfer_summary == trace.transfer_summary,
              "persisted trace should preserve transfer summary");
        Check(persisted_trace->synchronization_event_count ==
                  trace.synchronization_event_count,
              "persisted trace should preserve synchronization event count");
        Check(persisted_trace->synchronization_known_bytes ==
                  trace.synchronization_known_bytes,
              "persisted trace should preserve synchronization bytes");
        Check(persisted_trace->synchronization_summary ==
                  trace.synchronization_summary,
              "persisted trace should preserve synchronization summary");
        Check(persisted_trace->arrayfire_host_sync_groups.size() ==
                  trace.arrayfire_host_sync_groups.size(),
              "persisted trace should preserve host sync groups");
        Check(persisted_trace->arrayfire_host_sync_summary ==
                  trace.arrayfire_host_sync_summary,
              "persisted trace should preserve formatted host sync summary");
    }

    auto* active_device = cyxwiz::Device::GetCurrentDevice();
    Check(active_device != nullptr,
          "strict ArrayFire CPU run should leave runtime device queryable");
    Check(active_device->GetType() == cyxwiz::DeviceType::CPU,
          "strict ArrayFire CPU run should leave ArrayFire CPU active");
    Check(active_device->GetDeviceId() == cpu->device_id,
          "strict ArrayFire CPU run should leave selected CPU device active");
    cyxwiz::ClearNextRunExecutionPolicy();
}

void TestStrictPlacementPreflightRejectsKnownNativeCpuStage(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    auto strict_config = config;
    strict_config.forbid_native_cpu_fallback = true;
    strict_config.backend_placements.clear();

    cyxwiz::BackendPlacementEntry cpu_stage;
    cpu_stage.node_id = strict_config.layers.front().node_id;
    cpu_stage.node_name = strict_config.layers.front().name;
    cpu_stage.node_type = "Dense";
    cpu_stage.expected_backend = "CPU";
    cpu_stage.fallback_backend = "CPU";
    cpu_stage.status = cyxwiz::BackendPlacementStatus::Cpu;
    cpu_stage.reason_code =
        cyxwiz::BackendPlacementReason::GraphRuntimeCpuBacked;
    strict_config.backend_placements.push_back(cpu_stage);

    cyxwiz::TrainingExecutor executor(strict_config, dataset, "label");
    bool saw_batch = false;
    bool completed = false;
    executor.Train(
        1,
        2,
        [&](int, int, int, float, float) { saw_batch = true; },
        nullptr,
        [&](const cyxwiz::TrainingMetrics&) { completed = true; });

    Check(!saw_batch,
          "strict placement preflight should reject before the first batch");
    Check(!completed,
          "strict placement preflight rejection should not report completion");
    Check(!executor.IsTraining(),
          "strict placement preflight rejection should clear training state");
    const auto failed_metrics = executor.GetMetrics();
    Check(failed_metrics.is_complete &&
              failed_metrics.terminal_status == "failed" &&
              failed_metrics.terminal_reason.find(
                  "placement_preflight_failed") != std::string::npos,
          "strict placement preflight should close metrics with its reason");
    Check(failed_metrics.current_epoch == 0 &&
              failed_metrics.last_executed_epoch == 0 &&
              failed_metrics.current_batch == 0,
          "strict placement preflight should report zero executed work");
    CheckTerminalEvidence(
        "failed", failed_metrics.terminal_reason, 0, 0,
        "strict placement preflight");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.status == "failed",
          "strict placement preflight rejection should terminate the trace");
    Check(trace.native_cpu_fallback_count == 0,
          "known placement rejection should not attempt native CPU fallback");
    Check(trace.residency_verdict == "terminal_without_residency_pass",
          "strict placement preflight rejection should not claim residency");
    bool saw_preflight_warning = false;
    bool saw_preflight_terminal = false;
    for (const auto& warning : trace.warnings) {
        if (warning.find("placement_preflight_failed") != std::string::npos &&
            warning.find("Dense") != std::string::npos) {
            saw_preflight_warning = true;
        }
    }
    for (const auto& event : trace.recent_events) {
        if (event.terminal_reason.find("placement_preflight_failed") !=
                std::string::npos) {
            saw_preflight_terminal = true;
        }
    }
    Check(saw_preflight_warning,
          "trace should identify the compiler-known blocking stage");
    Check(saw_preflight_terminal,
          "terminal trace should preserve the placement preflight reason");
}

void TestExecutablePreflightRejectsUnsupportedOptimizer(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    auto unsupported_config = config;
    unsupported_config.optimizer_type = gui::NodeType::Output;

    cyxwiz::TrainingExecutor executor(
        unsupported_config, dataset, "label");
    bool saw_batch = false;
    bool completed = false;
    executor.Train(
        1,
        2,
        [&](int, int, int, float, float) { saw_batch = true; },
        nullptr,
        [&](const cyxwiz::TrainingMetrics&) { completed = true; });

    Check(!saw_batch,
          "unsupported optimizer preflight should reject before the first batch");
    Check(!completed,
          "unsupported optimizer preflight should not report completion");
    Check(!executor.IsTraining(),
          "unsupported optimizer preflight should clear training state");
    const auto failed_metrics = executor.GetMetrics();
    Check(failed_metrics.is_complete &&
              failed_metrics.terminal_status == "failed" &&
              failed_metrics.terminal_reason.find(
                  "execution_preflight_failed") != std::string::npos,
          "unsupported optimizer preflight should close metrics with its reason");
    Check(failed_metrics.current_epoch == 0 &&
              failed_metrics.last_executed_epoch == 0 &&
              failed_metrics.current_batch == 0,
          "unsupported optimizer preflight should report zero executed work");
    CheckTerminalEvidence(
        "failed", failed_metrics.terminal_reason, 0, 0,
        "unsupported optimizer preflight");

    const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
    Check(trace.status == "failed",
          "unsupported optimizer preflight should terminate the trace");
    Check(trace.native_cpu_fallback_count == 0,
          "unsupported optimizer preflight should not attempt CPU fallback");
    bool saw_execution_preflight = false;
    for (const auto& warning : trace.warnings) {
        if (warning.find("execution_preflight_failed") != std::string::npos &&
            warning.find("optimizer_unsupported") != std::string::npos) {
            saw_execution_preflight = true;
        }
    }
    Check(saw_execution_preflight,
          "trace should preserve the unsupported optimizer preflight reason");
}
#endif

void TestTrainingDeviceSelectionSwitchesBetweenRuns(
    const std::shared_ptr<cyxwiz::ArrowDataset>& dataset,
    const cyxwiz::TrainingConfiguration& config) {
    const auto devices = cyxwiz::Device::GetAvailableDevices();
    const auto cpu = FirstDeviceOfType(devices, cyxwiz::DeviceType::CPU);
    Check(cpu.has_value(),
          "device switching regression requires ArrayFire CPU discovery");

    std::vector<cyxwiz::DeviceInfo> run_order;
    run_order.push_back(*cpu);

    const auto cuda = FirstDeviceOfType(devices, cyxwiz::DeviceType::CUDA);
    if (cuda.has_value()) {
        run_order.push_back(*cuda);
        run_order.push_back(*cpu);
    }

    const auto oneapi = FirstDeviceOfType(devices, cyxwiz::DeviceType::ONEAPI);
    const char* oneapi_training =
        std::getenv("CYXWIZ_TEST_ONEAPI_TRAINING");
    if (oneapi.has_value() && oneapi_training != nullptr &&
        std::string(oneapi_training) == "1") {
        run_order.push_back(*oneapi);
        run_order.push_back(*cpu);
    } else if (oneapi.has_value()) {
        std::cout
            << "SKIP: oneAPI full training matrix is opt-in; bounded exact "
               "activation is covered by test_device\n";
    }

    const auto opencl = FirstDeviceOfType(devices, cyxwiz::DeviceType::OPENCL);
    if (opencl.has_value()) {
        run_order.push_back(*opencl);
        run_order.push_back(*cpu);
    }

    if (run_order.size() <= 1) {
        return;
    }

    for (const auto& selection : run_order) {
        cyxwiz::ClearPendingExecutionDeviceSelection();
        cyxwiz::SetPendingExecutionDeviceSelection(
            selection.type,
            selection.device_id);

        cyxwiz::TrainingExecutor executor(config, dataset, "label");
        bool saw_batch = false;
        executor.Train(
            1,
            2,
            [&](int, int, int, float, float) {
                Check(cyxwiz::HasActiveExecutionDeviceContext(),
                      "device switch run should bind active context");
                saw_batch = true;
            },
            nullptr,
            nullptr);
        Check(saw_batch,
              "device switch run should execute at least one training batch");

        const std::string expected_backend =
            cyxwiz::ExecutionDeviceSelectionBackendName(selection.type);
        const auto trace = cyxwiz::TrainingTraceCollector::Instance().Snapshot();
        bool saw_bind = false;
        for (const auto& event : trace.recent_events) {
            if (event.stage != "ExecutionDeviceContext.Bind") {
                continue;
            }
            saw_bind = true;
            Check(event.requested_backend == expected_backend,
                  "device switch bind should record requested backend");
            Check(event.requested_device_id == selection.device_id,
                  "device switch bind should record requested device id");
            Check(event.effective_backend == expected_backend,
                  "device switch bind should activate requested backend");
            Check(event.effective_device_id == selection.device_id,
                  "device switch bind should activate requested device id");
            Check(event.activation_succeeded,
                  "device switch bind should record exact activation success");
            Check(event.execution_validated,
                  "device switch bind should record execution validation");
            Check(event.requested_route_qualified,
                  "device switch bind should record requested certification");
            Check(event.effective_route_qualified,
                  "device switch bind should record effective certification");
            Check(event.requested_qualification_matrix_id ==
                      "test-qualified-routes",
                  "device switch bind should retain qualification matrix");
            Check(event.preflight_stage == "complete",
                  "device switch bind should record completed preflight");
        }
        Check(saw_bind,
              "device switch run should record execution context bind");

        auto* active_device = cyxwiz::Device::GetCurrentDevice();
        Check(active_device != nullptr,
              "device switch run should leave ArrayFire runtime queryable");
        Check(active_device->GetType() == selection.type,
              "device switch run should leave selected backend active");
        Check(active_device->GetDeviceId() == selection.device_id,
              "device switch run should leave selected device active");
        Check(!cyxwiz::GetPendingExecutionDeviceSelection().has_value(),
              "device switch run should clear pending selection");
    }
}

void TestObjectiveAwareRegressionMetrics(
    const std::filesystem::path& checkpoint_dir) {
    const auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeRegressionTable(), "training_executor_regression");
    const auto config = MakeRegressionConfig(checkpoint_dir);
    Check(cyxwiz::UsesContinuousTargetMetrics(config),
          "continuous target contract should select regression metrics");

    cyxwiz::TrainingExecutor executor(config, dataset, "target");
    bool completed = false;
    cyxwiz::TrainingMetrics final_metrics;
    executor.Train(
        1,
        2,
        nullptr,
        nullptr,
        [&](const cyxwiz::TrainingMetrics& metrics) {
            final_metrics = metrics;
            completed = true;
        });

    Check(completed, "regression executor should complete");
    Check(std::isfinite(final_metrics.train_mae),
          "regression train MAE should be finite");
    Check(std::isfinite(final_metrics.train_rmse),
          "regression train RMSE should be finite");
    Check(std::isfinite(final_metrics.val_mae),
          "regression validation MAE should be finite");
    Check(std::isfinite(final_metrics.val_rmse),
          "regression validation RMSE should be finite");
    Check(final_metrics.mae_history.size() == 1,
          "regression should store one MAE point per epoch");
    Check(final_metrics.rmse_history.size() == 1,
          "regression should store one RMSE point per epoch");
    Check(final_metrics.val_mae_history.size() == 1,
          "regression should store validation MAE history");
    Check(final_metrics.val_rmse_history.size() == 1,
          "regression should store validation RMSE history");
    Check(final_metrics.accuracy_history.empty(),
          "regression must not manufacture classification accuracy history");
    Check(final_metrics.val_accuracy_history.empty(),
          "regression must not manufacture validation accuracy history");
    CheckNear(final_metrics.train_accuracy, 0.0, 0.0,
              "regression train accuracy should remain unset");
    CheckNear(final_metrics.val_accuracy, 0.0, 0.0,
              "regression validation accuracy should remain unset");

#ifndef NDEBUG
    if (cyxwiz::IsCurrentArrayFireBackendAvailable()) {
        auto strict_config = MakeRegressionConfig(
            checkpoint_dir / "strict_residency");
        strict_config.forbid_native_cpu_fallback = true;

        cyxwiz::TrainingExecutor strict_executor(
            strict_config, dataset, "target");
        bool strict_completed = false;
        cyxwiz::TrainingMetrics strict_metrics;
        strict_executor.Train(
            1,
            2,
            nullptr,
            nullptr,
            [&](const cyxwiz::TrainingMetrics& metrics) {
                strict_metrics = metrics;
                strict_completed = true;
            });

        Check(strict_completed,
              "strict regression executor should complete with ArrayFire metrics");
        Check(std::isfinite(strict_metrics.train_mae),
              "strict regression train MAE should be finite");
        Check(std::isfinite(strict_metrics.val_rmse),
              "strict regression validation RMSE should be finite");

        const auto trace =
            cyxwiz::TrainingTraceCollector::Instance().Snapshot();
        Check(trace.native_cpu_fallback_count == 0,
              "strict regression metrics should not use native CPU fallback");
    }
#endif
}

void TestRegressionMetricAccumulator(
    const std::filesystem::path&) {
    cyxwiz::RegressionMetricAccumulator metrics;
    const float predictions[] = {2.0f, -1.0f, 4.0f, 6.0f};
    const float targets[] = {1.0f, 1.0f, 5.0f, 3.0f};
    metrics.Add(predictions, targets, 4);

    Check(metrics.value_count == 4,
          "regression metrics should count every target horizon");
    CheckNear(metrics.Mae(), 1.75, 1e-6,
              "regression metrics should compute elementwise MAE");
    CheckNear(metrics.Rmse(), std::sqrt(3.75), 1e-6,
              "regression metrics should compute elementwise RMSE");

    metrics.Reset();
    Check(metrics.value_count == 0,
          "regression metrics reset should clear the target count");
    CheckNear(metrics.Mae(), 0.0, 0.0,
              "empty regression metrics should have zero MAE");
    CheckNear(metrics.Rmse(), 0.0, 0.0,
              "empty regression metrics should have zero RMSE");
}

void TestRegressionTargetTransform(
    const std::filesystem::path& work_dir) {
    cyxwiz::FittedPreprocessingState state;
    state.operator_name = "StandardScaler";
    state.fit_rows = 4;
    state.input_schema_fingerprint = "fixture";
    state.configuration["with_mean"] = "true";
    state.configuration["with_std"] = "true";

    cyxwiz::PreprocessingFeatureState first;
    first.name = "target";
    first.data_type = "float";
    first.numeric_values["mean"] = 100.0;
    first.numeric_values["scale"] = 10.0;
    state.features.push_back(first);

    cyxwiz::PreprocessingFeatureState second;
    second.name = "target_1";
    second.data_type = "float";
    second.numeric_values["mean"] = 200.0;
    second.numeric_values["scale"] = 2.0;
    state.features.push_back(second);

    const auto path = work_dir / "target_scaler.cyxstate.json";
    std::string error;
    Check(cyxwiz::SaveFittedPreprocessingState(
              path.string(), state, false, error),
          "target scaler fixture should save: " + error);

    cyxwiz::RegressionTargetTransform transform;
    transform.enabled = true;
    transform.operator_name = "StandardScaler";
    transform.state_path = path.string();
    transform.target_columns = {"target", "target_1"};
    Check(cyxwiz::ResolveRegressionTargetTransform(transform, error),
          "target scaler fixture should resolve: " + error);
    Check(transform.IsResolvedForWidth(2),
          "resolved target scaler should match output width");
    CheckNear(transform.InverseValue(1.5, 0), 115.0, 1e-9,
              "first horizon should inverse-transform with its state");
    CheckNear(transform.InverseValue(-2.0, 1), 196.0, 1e-9,
              "second horizon should inverse-transform with its state");

    cyxwiz::RegressionMetricAccumulator metrics(&transform);
    const float predictions[] = {1.0f, 1.0f, 2.0f, 2.0f};
    const float targets[] = {0.0f, 0.0f, 1.0f, 1.0f};
    metrics.Add(predictions, targets, 4, 2);
    CheckNear(metrics.Mae(), 6.0, 1e-6,
              "target-scaled MAE should be reported in original units");
    CheckNear(metrics.Rmse(), std::sqrt(52.0), 1e-6,
              "target-scaled RMSE should be reported in original units");

    auto wrong_order = transform;
    wrong_order.target_columns = {"target_1", "target"};
    Check(!cyxwiz::ResolveRegressionTargetTransform(wrong_order, error),
          "target scaler should reject reordered target columns");
    Check(error.find("expected") != std::string::npos,
          "target scaler order error should explain the mismatch");
}

nlohmann::json ReadPersistedTrainingTrace(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "persisted training trace should be readable: " + path.string());
    }
    return nlohmann::json::parse(input);
}

void TestTrainingTracePersistenceCoalescing() {
    auto& collector = cyxwiz::TrainingTraceCollector::Instance();
    const auto original_settings = collector.GetSettings();
    Check(original_settings.persist_every_n_events >= 1000,
          "default training trace persistence cadence should stay off the hot path");

    cyxwiz::TrainingTraceSettings settings;
    settings.persist_enabled = true;
    settings.persist_every_n_events = 3;
    settings.max_recent_events = 20;
    collector.Configure(settings);
    collector.StartRun("trace-coalescing-contract");

    const auto trace_path =
        cyxwiz::GetDebugRunRoot() / "current_training_trace.json";
    Check(ReadPersistedTrainingTrace(trace_path).at("events").empty(),
          "starting a run should persist an empty trace");

    collector.RecordRuntimeEvent("Routine.One", "first routine event");
    collector.RecordRuntimeEvent("Routine.Two", "second routine event");
    Check(ReadPersistedTrainingTrace(trace_path).at("events").empty(),
          "routine runtime events should be coalesced until the configured cadence");

    collector.RecordRuntimeEvent("Routine.Three", "third routine event");
    Check(ReadPersistedTrainingTrace(trace_path).at("events").size() == 3,
          "the configured routine-event cadence should flush the latest snapshot");

    collector.RecordStage(
        cyxwiz::TrainingTraceStage::ComputeLoss, 1, 1, 2,
        0.5f, 0.0f, 0.0f, "device_resident");
    Check(ReadPersistedTrainingTrace(trace_path).at("events").size() == 3,
          "successful device-resident stages should not force persistence");
    collector.RecordRuntimeWarning("TrainingExecutor", "forced warning flush");
    const auto warning_trace = ReadPersistedTrainingTrace(trace_path);
    Check(warning_trace.at("events").size() == 5,
          "a warning should flush pending routine events immediately");
    Check(!warning_trace.at("warnings").empty(),
          "an immediate warning flush should persist warning evidence");

    collector.RecordTaskProgress(
        42,
        "Prepare graph training",
        "MemoryPreflight",
        0.04f,
        "Materialization estimate requires confirmation.",
        "warning",
        17,
        "TF-IDF",
        4096,
        0,
        10,
        "warning",
        8192,
        6144,
        true,
        2048,
        3072,
        256,
        "private_commit",
        "test");
    const auto materialization_trace =
        ReadPersistedTrainingTrace(trace_path);
    Check(materialization_trace.at("materialization_events").size() == 1,
          "node-scoped task progress should persist as materialization evidence");
    const auto& materialization_event =
        materialization_trace.at("materialization_events").front();
    Check(materialization_event.at("node_id") == 17 &&
              materialization_event.at("estimated_memory_bytes") == 4096 &&
              materialization_event.at("process_resident_growth_bytes") == 256,
          "persisted materialization evidence should retain node, estimate, and actual growth");

    Check(collector.ContinueRun("trace-coalescing-contract-runtime"),
          "an active preparation trace should accept the runtime run ID");
    const auto continued_trace = collector.Snapshot();
    Check(continued_trace.run_id == "trace-coalescing-contract-runtime" &&
              continued_trace.materialization_events.size() == 1 &&
              continued_trace.materialization_events.front().run_id ==
                  continued_trace.run_id,
          "run-ID handoff should preserve and rebind materialization evidence");
    const auto continued_persisted = ReadPersistedTrainingTrace(trace_path);
    Check(continued_persisted.at("run_id") ==
              "trace-coalescing-contract-runtime" &&
              continued_persisted.at("materialization_events").size() == 1,
          "run-ID handoff should persist the preserved materialization evidence");

    std::atomic<bool> stop_reader = false;
    std::atomic<bool> reader_saw_invalid_trace = false;
    std::thread reader([&] {
        while (!stop_reader.load()) {
            try {
                const auto persisted = ReadPersistedTrainingTrace(trace_path);
                const auto snapshot = collector.Snapshot();
                if (!persisted.is_object() ||
                    snapshot.recent_events.size() > settings.max_recent_events) {
                    reader_saw_invalid_trace.store(true);
                }
            } catch (...) {
                reader_saw_invalid_trace.store(true);
            }
        }
    });
    for (int index = 0; index < 30; ++index) {
        collector.RecordRuntimeWarning(
            "AtomicPersistence", "forced reader-safety flush");
    }
    stop_reader.store(true);
    reader.join();
    Check(!reader_saw_invalid_trace.load(),
          "concurrent readers should observe complete persisted traces and bounded snapshots");

    for (int index = 0; index < 25; ++index) {
        collector.RecordStage(
            cyxwiz::TrainingTraceStage::Forward, 1, index + 1, 25);
    }
    Check(collector.Snapshot().recent_events.size() == 20,
          "training trace event retention should remain bounded");

    collector.RecordTerminalEvent(
        "completed", "trace persistence contract complete", 1, 0.5f, 0.75f);
    collector.FinishRun("completed");
    const auto terminal_trace = ReadPersistedTrainingTrace(trace_path);
    Check(terminal_trace.at("status") == "completed",
          "finish should persist the terminal run status");
    Check(terminal_trace.at("events").back().at("stage") ==
              "TrainingTerminal",
          "terminal persistence should include the latest terminal event");
    Check(!std::filesystem::exists(trace_path.string() + ".tmp"),
          "atomic trace persistence should not leave a temporary file");

    collector.Configure(original_settings);
}

} // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    const fs::path work_dir =
        fs::temp_directory_path() / "cyxwiz_training_executor_arrow_parquet";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);
    const cyxwiz::ScopedDebugRunRootOverrideForTesting debug_root(
        work_dir / "debug_runs");
    cyxwiz::test::InstallQualifiedRouteSnapshot();

    if (argc >= 2 &&
        std::string(argv[1]) == "--gradient-accumulation-parity-only") {
        const auto fixture = LoadTrainingCoreFixture(
            argc > 0 ? fs::path(argv[0]) : fs::path{},
            argc >= 3 ? argv[2] : nullptr);
        TestGradientAccumulationPyTorchParity(
            fixture.at("cases"), work_dir / "gradient-accumulation");
        fs::remove_all(work_dir);
        std::cout << "Gradient accumulation PyTorch parity passed\n";
        return 0;
    }

    if (argc == 2 &&
        std::string(argv[1]) == "--uneven-epoch-metrics-only") {
        TestUnevenFinalBatchMetricAggregation(work_dir);
        fs::remove_all(work_dir);
        std::cout << "Uneven final-batch metric aggregation passed\n";
        return 0;
    }

    if (argc == 2 &&
        std::string(argv[1]) == "--paused-control-matrix-only") {
        TestPausedControlAcrossModernDatasetModes(
            work_dir / "paused-control-matrix");
        fs::remove_all(work_dir);
        std::cout << "Paused-control dataset-mode matrix passed\n";
        return 0;
    }

#ifndef NDEBUG
    if (argc == 2 && std::string(argv[1]) == "--strict-dense-only") {
        const auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
            MakeTrainingTable(), "strict_dense_training");
        const auto config = MakeConfig(work_dir / "checkpoints");
        TestStrictArrayFireCpuDenseTrainingDoesNotFallback(dataset, config);
        fs::remove_all(work_dir);
        std::cout << "Strict dense ArrayFire residency passed\n";
        return 0;
    }
#endif

    TestTrainingTracePersistenceCoalescing();

    TestRunPreflightEnforcesRouteQualification();
    TestWeightedSamplerEpochAndUpdateCount(work_dir);

    TestSequenceBatchContract();
    TestSequenceBatcherPadsNamedPayloads();
    TestSequenceBatcherDropLast();
    TestSequenceBatcherSeedDeterminism();
    TestSequencePhaseSwitchRequiresExplicitReset();
    TestSequenceTagMetrics();
    TestSequenceVocabulary();
    TestNERSequenceBuilder();
    TestSequenceTrainingStep();
    TestSequenceTrainingExecutor();

    const auto dataset = std::make_shared<cyxwiz::ArrowDataset>(
        MakeTrainingTable(), "training_executor_arrow");
    const auto config = MakeConfig(work_dir / "checkpoints");
    TestArrowDataLoaderSeedDeterminism(dataset);
    TestValidationEarlyStoppingLifecycle(dataset, work_dir, false);
    TestValidationEarlyStoppingLifecycle(dataset, work_dir, true);
    TestScheduledValidationLifecycle(dataset, work_dir);
    TestScheduledValidationPatienceLifecycle(dataset, work_dir);
    TestExternalResolvedRoleLifecycle(dataset, work_dir);
    TestSharedPhaseExternalLifecycle(work_dir);
    TestSequenceLifecycleCadence(work_dir);
    TestMetricReportingCadenceInvariance(dataset, work_dir);
    TestUnevenFinalBatchMetricAggregation(work_dir);
    const auto training_core_fixture = LoadTrainingCoreFixture(
        argc > 0 ? fs::path(argv[0]) : fs::path{});
    TestGradientAccumulationPyTorchParity(
        training_core_fixture.at("cases"),
        work_dir / "gradient-accumulation");
    TestPluginStopLifecycle(dataset, work_dir);
    TestUserCancellationLifecycle(dataset, work_dir);
    TestInjectedRuntimeFailureLifecycle(dataset, work_dir);
    TestPausedControlAcrossModernDatasetModes(
        work_dir / "paused-control-matrix");

#ifndef NDEBUG
    TestAllowedTrainingRecordsForcedLinearFallback(dataset, config);
    TestStrictTrainingRejectsForcedLinearFallback(dataset, config);
    TestStrictTrainingSkipsFirstBatchDebugHostDump(dataset, config);
    TestStrictArrayFireCpuDenseTrainingDoesNotFallback(dataset, config);
    TestStrictPlacementPreflightRejectsKnownNativeCpuStage(dataset, config);
    TestExecutablePreflightRejectsUnsupportedOptimizer(dataset, config);
#endif
    TestPendingExecutionDeviceSelectionAppliesAndClears(dataset, config);
    TestPendingExecutionDeviceSelectionRejectsNonArrayFireBackends();
    TestTrainingDeviceSelectionSwitchesBetweenRuns(dataset, config);

    {
        auto sequence_config = config;
        sequence_config.sequence_batch.enabled = true;
        sequence_config.sequence_batch.token_column = "tokens";
        sequence_config.sequence_batch.tag_column = "ner_tags";
        cyxwiz::TrainingExecutor sequence_executor(
            sequence_config, dataset, "label");
        bool saw_batch = false;
        bool saw_epoch = false;
        bool completed = false;
        sequence_executor.Train(
            1,
            sequence_config.batch_size,
            [&](int, int, int, float, float) { saw_batch = true; },
            [&](int, float, float, float, float, float) { saw_epoch = true; },
            [&](const cyxwiz::TrainingMetrics&) { completed = true; });
        Check(!saw_batch,
              "sequence batch guard should reject before any training batch");
        Check(!saw_epoch,
              "sequence batch guard should reject before epoch callback");
        Check(!completed,
              "sequence batch guard should reject before completion callback");
        Check(!sequence_executor.IsTraining(),
              "sequence batch guard should clear executor training state");
    }

    {
        cyxwiz::TrainingExecutor arrow_executor(config, dataset, "label");
        RunExecutor(arrow_executor, "Arrow TrainingExecutor");
    }

    TestObjectiveAwareRegressionMetrics(work_dir / "regression_checkpoints");
    TestRegressionMetricAccumulator(
        work_dir / "regression_test_checkpoints");
    TestRegressionTargetTransform(work_dir);

    {
        auto grad_accum_config = config;
        grad_accum_config.epochs = 3;
        grad_accum_config.grad_accum_steps = 2;
        cyxwiz::TrainingExecutor grad_accum_executor(
            grad_accum_config, dataset, "label");
        RunExecutor(grad_accum_executor,
                    "Arrow gradient accumulation TrainingExecutor",
                    3,
                    3,
                    3);
    }

    const fs::path parquet_path = work_dir / "training_executor.parquet";
    WriteParquetWithRowGroupSize(*dataset, parquet_path.string(), 1);
    auto parquet_dataset = cyxwiz::ParquetBackedDataset::Open(
        parquet_path.string(), "training_executor_parquet");
    Check(parquet_dataset != nullptr, "Parquet fixture should open");

    TestArrowParquetBatchBoundaryParity(dataset, parquet_dataset, work_dir);
    TestArrowParquetSeedDeterminism(dataset, parquet_dataset, work_dir);
    TestParquetLifecycleCadence(parquet_dataset, work_dir);

    {
        cyxwiz::TrainingExecutor parquet_executor(config, parquet_dataset, "label");
        RunExecutor(parquet_executor, "Parquet TrainingExecutor");
    }

    Check(fs::exists(
              cyxwiz::GetDebugRunRoot() / "current_training_trace.json"),
          "training traces should use the injected debug-run root");
    Check(fs::exists(cyxwiz::GetDebugRunRoot() / "current_run.json"),
          "crash run records should use the injected debug-run root");

    parquet_dataset.reset();
    fs::remove_all(work_dir);

    std::cout << "TrainingExecutor Arrow/Parquet parity passed\n";
    return 0;
}
