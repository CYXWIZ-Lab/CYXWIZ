#include "training_executor.h"
#include "classification_decision.h"
#include "checkpoint_manager.h"
#include "crash_run_recorder.h"
#include "error_codes.h"
#include "algorithms/arrayfire_backend_utils.h"
#include "algorithms/arrayfire_host_materialization.h"
#include <cyxwiz/debug_hooks.h>
#include "execution_device_context.h"
#include "execution_device_preferences.h"
#include "execution_placement_plan.h"
#include "training_trace_collector.h"
#include "backend_placement_capabilities.h"
#include "model_builder.h"
#include "runtime_log_store.h"
#include "sequence_model_input.h"
#include "sequence_training_step.h"
#include "training_batcher_setup.h"
#include "training_parameter_contract.h"
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
#include "data_registry.h"
#include "../preprocessing/preprocessing_config.h"
#include "../preprocessing/statistics_calculator.h"
#endif
#include "../plugin/registries/plugin_training_hook_manager.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef CYXWIZ_HAS_ARRAYFIRE
#include <arrayfire.h>
#endif

namespace cyxwiz {

namespace {

void AppendExecutionDeviceLifecycleEvent(
    const ExecutionDeviceContext& context, const std::string& run_id) {
    RuntimeLogEvent event;
    event.timestamp_utc = std::chrono::system_clock::now();
    event.level = context.valid ? RuntimeLogLevel::Info
                                : RuntimeLogLevel::Warning;
    event.category = "device";
    event.source = "TrainingExecutor";
    event.event_name = "ExecutionDeviceContext.Bind";
    event.run_id = run_id;
    event.backend = context.effective_backend;
    event.device_id = context.effective_device_id;
    event.device_name = context.device_name;
    event.message = context.Describe();
    if (!context.valid) {
        event.primary_error_code = errors::Runtime::ExecutionFailed;
    }
    event.details.emplace_back("requested_backend", context.requested_backend);
    event.details.emplace_back(
        "requested_device_id", std::to_string(context.requested_device_id));
    event.details.emplace_back(
        "fallback_policy", context.FallbackPolicyName());
    RuntimeLogStore::Instance().Append(std::move(event));
}

void AppendNativeCpuFallbackLifecycleEvent(
    const ArrayFireNativeCpuFallbackEvent& fallback,
    const std::string& run_id, const std::string& detail) {
    RuntimeLogEvent event;
    event.timestamp_utc = std::chrono::system_clock::now();
    event.level = fallback.fallback_forbidden ? RuntimeLogLevel::Error
                                              : RuntimeLogLevel::Warning;
    event.category = "training";
    event.source = "TrainingExecutor";
    event.event_name = "ArrayFire.NativeCpuFallback";
    event.run_id = run_id;
    event.backend = fallback.selected_backend;
    event.primary_error_code = errors::Gpu::KernelExecutionFailed;
    event.message = "Native CPU fallback: " + detail;
    event.details.emplace_back("operation", fallback.operation_name);
    event.details.emplace_back("reason", fallback.reason_code);
    event.details.emplace_back(
        "policy", fallback.fallback_forbidden
            ? "forbid_native_cpu_fallback"
            : "allow_native_cpu_fallback");
    RuntimeLogStore::Instance().Append(std::move(event));
}

void LogTrainingBackendPlacementPlan(const TrainingConfiguration& config) {
    if (config.backend_placements.empty()) {
        return;
    }

    const auto summary = config.SummarizeBackendPlacements();
    spdlog::info(
        "TrainingExecutor: Backend placement plan: total={}, gpu={}, cpu={}, mixed={}, risk={}, unsupported={}, unknown={}",
        summary.total,
        summary.gpu,
        summary.cpu,
        summary.mixed,
        summary.risk,
        summary.unsupported,
        summary.unknown);
    for (const auto& placement : config.backend_placements) {
        const std::string layer =
            placement.node_type +
            (placement.node_name.empty()
                 ? std::string()
                 : " '" + placement.node_name + "'");
        const std::string detail =
            layer + " -> expected=" + placement.expected_backend +
            (placement.fallback_backend.empty()
                 ? std::string()
                 : ", fallback=" + placement.fallback_backend) +
            (placement.reason_code.empty()
                 ? std::string()
                 : ", reason=" + placement.reason_code);

        if (placement.NeedsUserAttention()) {
            spdlog::warn("TrainingExecutor: {}", detail);
            if (!placement.explanation.empty()) {
                spdlog::warn("TrainingExecutor: {}", placement.explanation);
            }
        } else {
            spdlog::info("TrainingExecutor: {}", detail);
        }
    }
}

std::string DescribePinMemoryTransferStatus(
    const PinMemoryTransferStatus& status) {
    return fmt::format(
        "pin_memory requested={}, effective_mode={}, reason={}, backend={}, "
        "batch_size={}, node_id={}, node_name='{}': {}",
        status.requested ? "true" : "false",
        status.effective_mode,
        status.reason_code,
        status.backend,
        status.batch_size,
        status.node_id,
        status.node_name,
        status.message);
}

void ReportPinMemoryTransferStatus(const TrainingConfiguration& config) {
    const auto& status = config.pin_memory_transfer;
    if (!status.requested) {
        return;
    }

    const std::string detail = DescribePinMemoryTransferStatus(status);
    const std::string severity = status.NeedsUserWarning() ? "warning" : "ok";
    if (status.NeedsUserWarning()) {
        spdlog::warn("TrainingExecutor: {}", detail);
    } else {
        spdlog::info("TrainingExecutor: {}", detail);
    }
    TrainingTraceCollector::Instance().RecordPinMemoryTransferStatus(
        status,
        detail,
        severity);
}

void RecordArrayFireNativeCpuFallbackForActiveTrace(
    const ArrayFireNativeCpuFallbackEvent& fallback) {
    std::string detail = fallback.operation_name;
    if (!fallback.reason_code.empty()) {
        detail += " reason=" + fallback.reason_code;
    }
    if (!fallback.selected_backend.empty()) {
        detail += " selected_backend=" + fallback.selected_backend;
    }
    if (!fallback.context.empty()) {
        detail += " " + fallback.context;
    }
    CrashRunRecorder::Instance().MarkBackendEvent(
        "ArrayFire.NativeCpuFallback",
        detail);
    TrainingTraceCollector::Instance().RecordNativeCpuFallback(fallback);
    const auto trace = TrainingTraceCollector::Instance().Snapshot();
    AppendNativeCpuFallbackLifecycleEvent(fallback, trace.run_id, detail);
}

void RecordArrayFireHostSyncForActiveTrace(
    const ArrayFireHostSyncEvent& sync) {
    TrainingTraceCollector::Instance().RecordArrayFireHostSync(sync);
}

bool ShouldLogTrainingBatch(const TrainingConfiguration& config, int batch_num) {
    if (batch_num <= 1) {
        return true;
    }
    return config.log_interval > 0 && batch_num % config.log_interval == 0;
}

bool ShouldReportTrainingBatch(const TrainingConfiguration& config,
                               int batch_num,
                               int total_batches,
                               bool epoch_complete) {
    if (batch_num <= 1 || epoch_complete ||
        (total_batches > 0 && batch_num >= total_batches)) {
        return true;
    }
    return config.log_interval > 0 &&
           batch_num % config.log_interval == 0;
}

void AccumulateDeviceValue(Tensor& accumulator,
                           bool& initialized,
                           const Tensor& value,
                           size_t expected_elements,
                           const char* value_name) {
    if (value.GetDataType() != DataType::Float32 ||
        value.NumElements() != expected_elements) {
        throw std::runtime_error(
            std::string("Training ") + value_name + " accumulator requires " +
            std::to_string(expected_elements) + " Float32 value(s)");
    }

    Tensor next = initialized ? accumulator + value : value;
#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array materialized = next.GetSemanticArray();
    materialized.eval();
    next.SetFromSemanticArray(materialized, {expected_elements});
#endif
    accumulator = std::move(next);
    initialized = true;
}

void AccumulateDeviceScalar(Tensor& accumulator,
                            bool& initialized,
                            const Tensor& value) {
    AccumulateDeviceValue(
        accumulator, initialized, value, 1, "scalar");
}

void AccumulateDeviceClassificationCounts(Tensor& accumulator,
                                          bool& initialized,
                                          const Tensor& value) {
    AccumulateDeviceValue(
        accumulator, initialized, value, 2, "classification count");
}

std::optional<int> ClassificationMetricIgnoreIndex(
    const TrainingConfiguration& config) {
    const auto loss_config = ResolveLossConfiguration(config);
    if (!loss_config.ignore_index_applicable) {
        return std::nullopt;
    }
    return loss_config.ignore_index;
}

Tensor ApplyDeviceScalar(const Tensor& value,
                         const Tensor& scalar,
                         bool divide) {
    if (value.GetDataType() != DataType::Float32 ||
        scalar.GetDataType() != DataType::Float32 ||
        scalar.NumElements() != 1) {
        throw std::runtime_error(
            "Training device-scalar operations require Float32 tensors and one scalar value");
    }
#ifdef CYXWIZ_HAS_ARRAYFIRE
    af::array result = divide
        ? value.GetSemanticArray() / scalar.GetSemanticArray()
        : value.GetSemanticArray() * scalar.GetSemanticArray();
    result.eval();
    return Tensor::FromSemanticArray(result, value.Shape());
#else
    const float host_scalar = scalar.ReadData<float>()[0];
    return divide ? value / host_scalar : value * host_scalar;
#endif
}

struct LossAggregationWeightValue {
    float host = 1.0f;
    const Tensor* device = nullptr;
};

LossAggregationWeightValue ResolveLossAggregationWeight(
    const Loss& loss,
    size_t observation_count) {
    switch (loss.GetReduction()) {
        case Reduction::Mean:
            return {
                static_cast<float>(observation_count),
                loss.GetLastMeanReductionDenominator(),
            };
        case Reduction::Sum:
            return {1.0f, nullptr};
        case Reduction::None:
            throw std::runtime_error(
                "TrainingExecutor requires a scalar loss reduction");
    }
    throw std::runtime_error("TrainingExecutor received an unknown loss reduction");
}

float LossAggregationWeight(const Loss& loss, size_t observation_count) {
    return ResolveLossAggregationWeight(loss, observation_count).host;
}

float FinalizeAggregatedLoss(const Loss& loss,
                             float weighted_loss_sum,
                             float weight_sum) {
    if (loss.GetReduction() == Reduction::Sum) {
        return weighted_loss_sum;
    }
    return weight_sum > 0.0f ? weighted_loss_sum / weight_sum : 0.0f;
}

float ReadAccumulatedLoss(const Tensor& loss_sum,
                          const Loss& loss,
                          float weight_sum,
                          const Tensor* device_weight_sum = nullptr,
                          std::string_view operation =
                              "TrainingExecutor::ReadAccumulatedLoss") {
    if (loss.GetReduction() == Reduction::Mean &&
        device_weight_sum != nullptr) {
        Tensor safe_weight = device_weight_sum->Clip(
            std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max());
        Tensor mean_loss = ApplyDeviceScalar(
            loss_sum, safe_weight, true);
        const ScopedArrayFireHostSyncAttribution loss_sync_attribution(
            ArrayFireHostSyncCategory::LossScalarReadback,
            std::string(operation));
        return mean_loss.ReadData<float>()[0];
    }
    if (weight_sum <= 0.0f) {
        return 0.0f;
    }
    const ScopedArrayFireHostSyncAttribution loss_sync_attribution(
        ArrayFireHostSyncCategory::LossScalarReadback,
        std::string(operation));
    return FinalizeAggregatedLoss(
        loss, loss_sum.ReadData<float>()[0], weight_sum);
}

bool ShouldRunValidationEpoch(const TrainingConfiguration& config,
                              int epoch,
                              int total_epochs) {
    const int validation_freq = std::max(1, config.validation_freq);
    return epoch == total_epochs || validation_freq <= 1 ||
           epoch % validation_freq == 0;
}

bool ShouldMaterializeFirstBatchDebugSamples(
    const TrainingConfiguration& config) {
    return !config.forbid_native_cpu_fallback;
}

void RecordSkippedFirstBatchDebugSampleDump() {
    TrainingTraceCollector::Instance().RecordRuntimeEvent(
        "TrainingExecutor.DebugSampleDump",
        "Skipped first-batch debug host tensor dump because strict ArrayFire "
        "residency forbids nonessential host materialization.");
}

void RecordDeclaredScalarLossOutputBoundary() {
    const std::string message =
        "Declared bounded host output boundary: loss_scalar_readback reads one "
        "scalar loss value for reporting and callbacks; this is not native CPU "
        "compute fallback.";
    CrashRunRecorder::Instance().MarkBackendEvent(
        "TrainingExecutor.OutputBoundary",
        message);
    TrainingTraceCollector::Instance().RecordRuntimeEvent(
        "TrainingExecutor.OutputBoundary",
        message);
}

std::string BuildRegressionMetricFallbackContext(
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width) {
    return BuildArrayFireBackendFallbackContext(
        BuildTensorShapeContext("predictions", predictions.Shape()) +
        "; " +
        BuildTensorShapeContext("targets", targets.Shape()) +
        fmt::format("; batch_size={}; output_width={}",
                    batch_size,
                    output_width));
}

void AddRegressionMetricsNativeCpuFallback(
    RegressionMetricAccumulator& regression,
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width) {
    const float* pred_data = predictions.ReadData<float>();
    const float* target_data = targets.ReadData<float>();
    regression.Add(
        pred_data,
        target_data,
        batch_size * output_width,
        output_width);
}

#ifdef CYXWIZ_HAS_ARRAYFIRE
bool CanUseArrayFireRegressionMetrics(const Tensor& predictions,
                                      const Tensor& targets,
                                      size_t batch_size,
                                      size_t output_width,
                                      BackendFallbackReason& reason) {
    if (predictions.GetDataType() != DataType::Float32 ||
        targets.GetDataType() != DataType::Float32) {
        reason = BackendFallbackReason::UnsupportedDtype;
        return false;
    }

    const auto& pred_shape = predictions.Shape();
    const auto& target_shape = targets.Shape();
    if (batch_size == 0 || output_width == 0 ||
        batch_size > std::numeric_limits<unsigned>::max() ||
        output_width > std::numeric_limits<unsigned>::max() ||
        pred_shape.size() != 2 || target_shape.size() != 2 ||
        pred_shape[0] != batch_size || pred_shape[1] != output_width ||
        target_shape[0] != batch_size || target_shape[1] != output_width) {
        reason = BackendFallbackReason::UnsupportedShape;
        return false;
    }

    return true;
}

void AddRegressionMetricsArrayFire(
    RegressionMetricAccumulator& regression,
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    const RegressionTargetTransform& transform) {
    af::array error =
        predictions.GetArrayRowMajor2D() - targets.GetArrayRowMajor2D();

    const bool restore_original_units =
        transform.enabled && transform.resolved &&
        transform.scales.size() == output_width;
    if (restore_original_units) {
        std::vector<float> scales;
        scales.reserve(output_width);
        for (double scale : transform.scales) {
            scales.push_back(static_cast<float>(scale));
        }

        const af::array scale_row(
            1,
            static_cast<dim_t>(output_width),
            scales.data());
        error = error * af::tile(scale_row,
                                 static_cast<unsigned>(batch_size),
                                 1U);
    }

    af::array absolute_error =
        af::sum(af::flat(af::abs(error.as(f32))));
    af::array squared_error =
        af::sum(af::flat((error * error).as(f32)));
    absolute_error.eval();
    squared_error.eval();

    float absolute_error_sum = 0.0f;
    float squared_error_sum = 0.0f;
    MaterializeArrayFireToHost(
        absolute_error,
        &absolute_error_sum,
        ArrayFireHostSyncCategory::MetricScalarReadback,
        "TrainingExecutor::RegressionAbsoluteError",
        "arrayfire_native",
        "bounded_scalar_readback",
        "scalar=absolute_error_sum");
    MaterializeArrayFireToHost(
        squared_error,
        &squared_error_sum,
        ArrayFireHostSyncCategory::MetricScalarReadback,
        "TrainingExecutor::RegressionSquaredError",
        "arrayfire_native",
        "bounded_scalar_readback",
        "scalar=squared_error_sum");

    regression.absolute_error_sum +=
        static_cast<double>(absolute_error_sum);
    regression.squared_error_sum +=
        static_cast<double>(squared_error_sum);
    regression.value_count += batch_size * output_width;
}
#endif

void AddRegressionMetricScalars(
    RegressionMetricAccumulator& regression,
    const Tensor& predictions,
    const Tensor& targets,
    size_t batch_size,
    size_t output_width,
    const TrainingConfiguration& config) {
    if (batch_size == 0 || output_width == 0) {
        return;
    }

#ifdef CYXWIZ_HAS_ARRAYFIRE
    BackendFallbackReason unsupported_reason =
        BackendFallbackReason::UnsupportedShape;
    const std::string context = BuildRegressionMetricFallbackContext(
        predictions, targets, batch_size, output_width);
    if (!CanUseArrayFireRegressionMetrics(
            predictions, targets, batch_size, output_width,
            unsupported_reason)) {
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "RegressionMetricAccumulator",
            unsupported_reason,
            "unsupported regression metric tensor layout",
            context);
        if (ShouldLogArrayFireBackendFallbackOnce(
                "RegressionMetricAccumulator",
                unsupported_reason,
                context)) {
            spdlog::warn("{}",
                         BuildArrayFireBackendFallbackMessage(
                             "RegressionMetricAccumulator",
                             unsupported_reason,
                             true,
                             "unsupported regression metric tensor layout",
                             context));
        }
        AddRegressionMetricsNativeCpuFallback(
            regression, predictions, targets, batch_size, output_width);
        return;
    }

    try {
        AddRegressionMetricsArrayFire(
            regression,
            predictions,
            targets,
            batch_size,
            output_width,
            config.regression_target_transform);
        return;
    } catch (const af::exception& e) {
        const BackendFallbackReason reason =
            ClassifyArrayFireBackendFallbackReason(e.what());
        ThrowIfArrayFireNativeCpuFallbackForbidden(
            "RegressionMetricAccumulator",
            reason,
            e.what(),
            context);
        if (ShouldLogArrayFireBackendFallbackOnce(
                "RegressionMetricAccumulator",
                reason,
                context)) {
            spdlog::warn("{}",
                         BuildArrayFireBackendFallbackMessage(
                             "RegressionMetricAccumulator",
                             reason,
                             true,
                             e.what(),
                             context));
        }
    }
#else
    const std::string context = BuildRegressionMetricFallbackContext(
        predictions, targets, batch_size, output_width);
    ThrowIfArrayFireNativeCpuFallbackForbidden(
        "RegressionMetricAccumulator",
        BackendFallbackReason::BackendInternalError,
        "ArrayFire support is unavailable",
        context);
#endif

    AddRegressionMetricsNativeCpuFallback(
        regression, predictions, targets, batch_size, output_width);
}

} // namespace

// ============================================================================
// TrainingExecutor Implementation
// ============================================================================

TrainingExecutor::TrainingExecutor(TrainingConfiguration config, DatasetHandle dataset)
    : config_(std::move(config))
    , dataset_(dataset)
    , mode_(DatasetMode::Legacy)
{
    spdlog::info("TrainingExecutor: Created with {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   std::shared_ptr<ArrowDataset> arrow_dataset,
                                   const std::string& label_column)
    : config_(std::move(config))
    , arrow_dataset_(arrow_dataset)
    , label_column_(label_column)
    , mode_(DatasetMode::Arrow)
{
    spdlog::info("TrainingExecutor: Created with Arrow dataset ({} rows, {} cols), label='{}'",
                 arrow_dataset_ ? arrow_dataset_->GetNumRows() : 0,
                 arrow_dataset_ ? arrow_dataset_->GetNumColumns() : 0,
                 label_column_);
    spdlog::info("TrainingExecutor: Model has {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   std::shared_ptr<ParquetBackedDataset> parquet_dataset,
                                   const std::string& label_column)
    : config_(std::move(config))
    , parquet_dataset_(parquet_dataset)
    , label_column_(label_column)
    , mode_(DatasetMode::Parquet)
{
    spdlog::info("TrainingExecutor: Created with Parquet-backed dataset "
                 "({} rows, {} cols, {} row groups, {:.1f} MB on disk), label='{}'",
                 parquet_dataset_ ? parquet_dataset_->GetNumRows() : 0,
                 parquet_dataset_ ? parquet_dataset_->GetNumColumns() : 0,
                 parquet_dataset_ ? parquet_dataset_->GetNumRowGroups() : 0,
                 parquet_dataset_ ? parquet_dataset_->GetFileSizeBytes() / (1024.0 * 1024.0) : 0.0,
                 label_column_);
    spdlog::info("TrainingExecutor: Model has {} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   std::unique_ptr<IBatcher> external_batcher)
    : config_(std::move(config))
    , mode_(DatasetMode::External)
{
    external_batchers_.train = std::shared_ptr<IBatcher>(std::move(external_batcher));
    external_batchers_.dev = external_batchers_.train;
    spdlog::info("TrainingExecutor: Created with external IBatcher, "
                 "{} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(TrainingConfiguration config,
                                   ResolvedExternalBatchers external_batchers)
    : config_(std::move(config))
    , mode_(DatasetMode::External)
    , external_batchers_(std::move(external_batchers))
{
    spdlog::info("TrainingExecutor: Created with resolved external role batchers, "
                 "{} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::TrainingExecutor(
    TrainingConfiguration config,
    std::unique_ptr<ISequenceBatcher> sequence_batcher,
    std::vector<std::string> id_to_label)
    : config_(std::move(config))
    , mode_(DatasetMode::SequenceExternal)
    , sequence_batcher_(std::move(sequence_batcher))
    , sequence_id_to_label_(std::move(id_to_label))
{
    spdlog::info("TrainingExecutor: Created with external ISequenceBatcher, "
                 "{} layers, input_size={}, output_size={}, labels={}",
                 config_.layers.size(), config_.input_size,
                 config_.output_size, sequence_id_to_label_.size());
}

TrainingExecutor::~TrainingExecutor() {
    Stop();
}

bool TrainingExecutor::ConfigureScheduler(
    TrainingSchedulerSpec specification,
    std::string& error) {
    error.clear();
    if (is_training_.load()) {
        error = "cannot configure a scheduler while training is active";
        return false;
    }
    if (!ValidateTrainingSchedulerSpec(specification, error)) {
        return false;
    }

    scheduler_specification_ = std::move(specification);
    scheduler_resume_state_.reset();
    scheduler_controller_.reset();
    return true;
}

bool TrainingExecutor::ConfigureScheduler(
    TrainingSchedulerSpec specification,
    TrainingSchedulerResumeState resume_state,
    std::string& error) {
    error.clear();
    if (is_training_.load()) {
        error = "cannot configure a scheduler while training is active";
        return false;
    }
    if (!ValidateTrainingSchedulerSpec(specification, error)) {
        return false;
    }
    if (resume_state.completed_epochs < 0 ||
        resume_state.completed_optimizer_steps < 0) {
        error = "scheduler resume cursors cannot be negative";
        return false;
    }
    if (resume_state.scheduler_state.scheduler_type.empty()) {
        error = "scheduler resume requires a populated backend scheduler state";
        return false;
    }

    scheduler_specification_ = std::move(specification);
    scheduler_resume_state_ = std::move(resume_state);
    scheduler_controller_.reset();
    return true;
}

std::optional<TrainingSchedulerResumeState>
TrainingExecutor::ExportSchedulerResumeState(std::string& error) const {
    error.clear();
    if (!scheduler_controller_) {
        error = "training executor has no initialized scheduler";
        return std::nullopt;
    }
    TrainingSchedulerResumeState state;
    if (!scheduler_controller_->ExportResumeState(state, error)) {
        return std::nullopt;
    }
    return state;
}

bool TrainingExecutor::Initialize(int /*batch_size*/) {
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
            "TrainingExecutor: regression target transform is invalid: {}",
            target_transform_error);
        return false;
    }
    if (config_.regression_target_transform.enabled) {
        spdlog::info(
            "TrainingExecutor: resolved StandardScaler regression target "
            "state '{}' for {} outputs; loss is transformed-space and "
            "MAE/RMSE are original-unit metrics",
            config_.regression_target_transform.state_path,
            config_.regression_target_transform.scales.size());
    }

    if (config_.sequence_batch.enabled &&
        mode_ != DatasetMode::SequenceExternal) {
        spdlog::error("TrainingExecutor: {}",
                      errors::FormatError(
                          errors::Runtime::UnsupportedNode,
                          SequenceBatchRuntimeUnsupportedMessage()));
        return false;
    }
    if (mode_ == DatasetMode::SequenceExternal) {
        if (!config_.sequence_batch.enabled) {
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Training::InvalidTrainingSetup,
                              "sequence batch config is not enabled"));
            return false;
        }
        if (!sequence_batcher_) {
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Training::InvalidTrainingSetup,
                              "sequence batcher is null"));
            return false;
        }
        if (sequence_id_to_label_.empty()) {
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Data::RequiredLabelColumnMissing,
                              "sequence label vocabulary is empty"));
            return false;
        }
    }

    auto built = BuildExecutableFromConfig(config_);
    if (!built.ok()) {
        spdlog::error("TrainingExecutor: {}",
                      built.error_message.empty()
                          ? errors::FormatError(
                                errors::Training::ModelBuildFailed,
                                "Failed to build model from config")
                          : built.error_message);
        return false;
    }
    model_ = std::move(built.model);
    loss_ = std::move(built.loss);
    optimizer_ = std::move(built.optimizer);
    scheduler_controller_.reset();
    if (scheduler_specification_.has_value()) {
        scheduler_controller_ =
            std::make_unique<TrainingSchedulerController>(
                *scheduler_specification_);
        std::string scheduler_error;
        if (!scheduler_controller_->Attach(
                *optimizer_, scheduler_resume_state_, scheduler_error)) {
            spdlog::error(
                "TrainingExecutor: scheduler initialization failed: {}",
                scheduler_error);
            scheduler_controller_.reset();
            return false;
        }
        spdlog::info(
            "TrainingExecutor: initialized {} scheduler with {} cadence at "
            "epoch_cursor={} optimizer_step_cursor={} lr={:.9g}",
            scheduler_controller_->GetScheduler()->GetName(),
            TrainingSchedulerCadenceName(
                scheduler_controller_->GetCadence()),
            scheduler_controller_->GetCompletedEpochs(),
            scheduler_controller_->GetCompletedOptimizerSteps(),
            scheduler_controller_->GetScheduler()->GetLR());
    }
    return true;
}

void TrainingExecutor::Train(
    int epochs,
    int batch_size,
    BatchCallback batch_cb,
    EpochCallback epoch_cb,
    TrainingCompleteCallback complete_cb)
{
    if (is_training_.load()) {
        spdlog::warn("TrainingExecutor: Already training");
        return;
    }

    is_training_.store(true);
    stop_requested_.store(false);
    is_paused_.store(false);

    const auto next_run_execution_policy = GetNextRunExecutionPolicy();
    if (next_run_execution_policy.has_value()) {
        config_.forbid_native_cpu_fallback =
            *next_run_execution_policy ==
            ArrayFireFallbackPolicy::ForbidNativeCpuFallback;
        spdlog::info(
            "TrainingExecutor: applied next-run execution policy: {}",
            ExecutionPolicyDisplayName(*next_run_execution_policy));
    }

    const ArrayFireFallbackPolicy fallback_policy_value =
        config_.forbid_native_cpu_fallback
            ? ArrayFireFallbackPolicy::ForbidNativeCpuFallback
            : ArrayFireFallbackPolicy::AllowNativeCpuFallback;
    const ScopedArrayFireFallbackPolicy fallback_policy(fallback_policy_value);
    ExecutionDeviceContext execution_context;
    try {
        execution_context =
            PrepareExecutionDeviceForRun(fallback_policy_value);
        spdlog::info(
            "TrainingExecutor: ArrayFire device preflight completed: {}",
            execution_context.Describe());
    } catch (const std::exception& e) {
        is_training_.store(false);
        spdlog::error(
            "TrainingExecutor: ArrayFire device preflight failed: {}",
            e.what());
        throw;
    }
    const ScopedExecutionDeviceContext execution_context_scope(
        execution_context);
    const ScopedActiveExecutionDeviceContext active_execution_context_scope;

    if (config_.forbid_native_cpu_fallback) {
        spdlog::info(
            "TrainingExecutor: strict ArrayFire residency enabled; native CPU fallback is forbidden");
    } else {
        spdlog::info(
            "TrainingExecutor: ArrayFire-first execution; native CPU fallback remains available for declared gaps");
    }

    CrashRunRecorder::Instance().StartTrainingRun(
        config_,
        epochs,
        batch_size,
        0);
    const auto started_run = CrashRunRecorder::LoadLastRun();
    const std::string training_run_id =
        started_run ? started_run->run_id : "training-run";
    auto& training_trace = TrainingTraceCollector::Instance();
    if (!training_trace.ContinueRun(training_run_id)) {
        training_trace.StartRun(training_run_id);
    } else {
        training_trace.RecordRuntimeEvent(
            "TrainingSetup",
            "Training executor attached to existing preparation trace");
    }
    CrashRunRecorder::Instance().MarkBackendEvent(
        "ExecutionDeviceContext.Bind",
        execution_context.Describe());
    training_trace.RecordExecutionDeviceContext(execution_context);
    AppendExecutionDeviceLifecycleEvent(execution_context, training_run_id);
    const auto execution_placement_plan =
        BuildExecutionPlacementPlan(config_, execution_context);
    TrainingTraceCollector::Instance().RecordPlacementPlan(
        execution_placement_plan.fingerprint,
        static_cast<uint64_t>(execution_placement_plan.entries.size()),
        execution_placement_plan.summary,
        fmt::format("placement_entries={} fingerprint={} summary={}",
                    execution_placement_plan.entries.size(),
                    execution_placement_plan.fingerprint,
                    execution_placement_plan.summary));
    const ScopedArrayFireNativeCpuFallbackObserver fallback_observer(
        &RecordArrayFireNativeCpuFallbackForActiveTrace);
    const ScopedArrayFireHostSyncObserver host_sync_observer(
        &RecordArrayFireHostSyncForActiveTrace);
    TrainingTraceCollector::Instance().RecordRuntimeEvent(
        "TrainingDevicePolicy",
        config_.forbid_native_cpu_fallback
            ? "ArrayFire-first execution; native CPU fallback forbidden"
            : "ArrayFire-first execution; native CPU fallback allowed and recorded");
    RecordDeclaredScalarLossOutputBoundary();
    auto fail_run = [&](const std::string& reason) {
        UpdateMetrics([&](TrainingMetrics& m) {
            m.total_epochs = epochs;
            m.is_training = false;
            m.is_paused = false;
            m.is_complete = true;
            m.terminal_status = "failed";
            m.terminal_reason = reason;
            m.status_message = "Training failed: " + reason;
        });
        const TrainingMetrics failed_metrics = GetMetrics();
        CrashRunRecorder::Instance().MarkFailed(reason);
        TrainingTraceCollector::Instance().RecordTerminalEvent(
            "failed",
            reason,
            failed_metrics.current_epoch,
            failed_metrics.train_loss,
            failed_metrics.train_accuracy);
        TrainingTraceCollector::Instance().FinishRun("failed");
        BackendDebugHooks::SetDebugEventCallback({});
        is_paused_.store(false);
        is_training_.store(false);
    };

    if (!execution_placement_plan.IsExecutable()) {
        const std::string reason =
            "execution_preflight_failed: " +
            execution_placement_plan.FatalBlockerSummary();
        TrainingTraceCollector::Instance().RecordRuntimeWarning(
            "TrainingExecutor.ExecutionPreflight", reason);
        spdlog::error("TrainingExecutor: {}", reason);
        fail_run(reason);
        return;
    }

    if (config_.forbid_native_cpu_fallback &&
        !execution_placement_plan.IsStrictlyExecutable()) {
        const std::string reason =
            "placement_preflight_failed: " +
            execution_placement_plan.StrictBlockerSummary();
        TrainingTraceCollector::Instance().RecordRuntimeWarning(
            "TrainingExecutor.PlacementPreflight", reason);
        spdlog::error("TrainingExecutor: {}", reason);
        fail_run(reason);
        return;
    }

    try {
        // Initialize
        if (!Initialize(batch_size)) {
            spdlog::error("TrainingExecutor: {}",
                          errors::FormatError(
                              errors::Training::InvalidTrainingSetup,
                              "Failed to initialize"));
            fail_run("initialization_failed");
            return;
        }
        LogTrainingBackendPlacementPlan(config_);

    // Setup metrics
    const double initial_learning_rate = optimizer_
        ? optimizer_->GetLearningRate()
        : static_cast<double>(config_.learning_rate);
    UpdateMetrics([epochs, initial_learning_rate](TrainingMetrics& m) {
        m.total_epochs = epochs;
        m.current_epoch = 0;
        m.last_executed_epoch = 0;
        m.restored_checkpoint_epoch = 0;
        m.restored_checkpoint_step = 0;
        m.checkpoint_used.clear();
        m.active_model_provenance = "run_final_state";
        m.is_training = true;
        m.is_complete = false;
        m.status_message = "Starting training...";
        m.terminal_status.clear();
        m.terminal_reason.clear();
        m.loss_history.clear();
        m.accuracy_history.clear();
        m.val_loss_history.clear();
        m.val_accuracy_history.clear();
        m.has_validation_metrics = false;
        m.optimizer_step_count = 0;
        m.scheduler_step_count = 0;
        m.learning_rate = initial_learning_rate;
        m.learning_rate_history.clear();
        m.has_test_metrics = false;
        m.train_token_accuracy = 0.0f;
        m.val_token_accuracy = 0.0f;
        m.train_entity_f1 = 0.0f;
        m.val_entity_f1 = 0.0f;
        m.test_token_accuracy = 0.0f;
        m.test_entity_f1 = 0.0f;
        m.train_token_count = 0;
        m.val_token_count = 0;
        m.test_token_count = 0;
    });
    CrashRunRecorder::Instance().MarkActiveModelCheckpoint(
        "", 0, 0, "run_final_state");

    // Create batchers - Arrow in-memory, Parquet disk-backed, external, or
    // legacy. Modern paths flow through IBatcher pointers; legacy keeps the
    // existing DatasetBatcher loop for now.
    TrainingBatcherSet modern_batchers;
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
    std::unique_ptr<DatasetBatcher> legacy_train_batcher;
    std::unique_ptr<DatasetBatcher> legacy_val_batcher;
#endif

    // Non-owning IBatcher pointers point at whichever concrete batcher the
    // selected mode owns. Arrow, Parquet, image, audio, and text all share the
    // IBatcher-aware loop; legacy DatasetHandle keeps its older loop.
    IBatcher* active_train_ibatcher = nullptr;
    IBatcher* active_val_ibatcher = nullptr;
    IBatcher* active_test_ibatcher = nullptr;
    BatcherPhase train_batcher_phase = BatcherPhase::Train;
    BatcherPhase val_batcher_phase = BatcherPhase::Val;
    BatcherPhase test_batcher_phase = BatcherPhase::Test;
    ISequenceBatcher* active_sequence_batcher = nullptr;

    size_t num_train_samples = 0;
    size_t num_val_samples = 0;
    size_t num_test_samples = 0;

    if (mode_ == DatasetMode::Arrow) {
        modern_batchers = BuildArrowTrainingBatchers(
            config_, arrow_dataset_, label_column_, batch_size);
        num_train_samples = modern_batchers.num_train_samples;
        active_train_ibatcher = modern_batchers.train;
        active_val_ibatcher = modern_batchers.val;
        active_test_ibatcher = modern_batchers.test;
    } else if (mode_ == DatasetMode::Parquet) {
        modern_batchers = BuildParquetTrainingBatchers(
            config_, parquet_dataset_, label_column_, batch_size);
        num_train_samples = modern_batchers.num_train_samples;
        active_train_ibatcher = modern_batchers.train;
        active_val_ibatcher = modern_batchers.val;
        active_test_ibatcher = modern_batchers.test;
    } else if (mode_ == DatasetMode::External) {
        // External batchers are constructed by TrainingManager for
        // image/audio/text datasets with the compiled graph config already
        // applied. The executor only owns the common training loop.
        spdlog::info("TrainingExecutor: Using external batcher for training "
                     "(batch_size={}, num_workers={}, {} samples)",
                     batch_size, config_.num_workers,
                     external_batchers_.train ? external_batchers_.train->GetNumSamples() : 0);

        if (!external_batchers_.train) {
            spdlog::error("TrainingExecutor: external batcher mode but no Train batcher");
            fail_run("missing_external_train_batcher");
            return;
        }

        active_train_ibatcher = external_batchers_.train.get();
        active_val_ibatcher = external_batchers_.dev
            ? external_batchers_.dev.get() : active_train_ibatcher;
        active_test_ibatcher = external_batchers_.test.get();
        train_batcher_phase = external_batchers_.train_phase;
        val_batcher_phase = external_batchers_.dev
            ? external_batchers_.dev_phase : BatcherPhase::Val;
        test_batcher_phase = external_batchers_.test_phase;
    } else if (mode_ == DatasetMode::SequenceExternal) {
        spdlog::info("TrainingExecutor: Using external sequence batcher for "
                     "token tagging ({} samples, {} batches)",
                     sequence_batcher_ ? sequence_batcher_->GetNumSamples() : 0,
                     sequence_batcher_ ? sequence_batcher_->GetNumBatches() : 0);

        if (!sequence_batcher_) {
            spdlog::error("TrainingExecutor: sequence mode but no sequence batcher");
            fail_run("missing_sequence_batcher");
            return;
        }

        num_train_samples = sequence_batcher_->GetNumSamples();
        active_sequence_batcher = sequence_batcher_.get();
    } else {
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
        // Legacy DatasetHandle batching
        spdlog::info("TrainingExecutor: Using legacy dataset for training "
                     "(batch_size={}, shuffle={}, drop_last={}, num_workers={})",
                     batch_size, config_.shuffle, config_.drop_last, config_.num_workers);

        // Honor DataLoader node config (or defaults if no such node).
        // Validation batcher never shuffles and never drops the last batch.
        legacy_train_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Train,
            config_.shuffle, config_.drop_last, config_.num_workers,
            static_cast<uint32_t>(config_.dataloader_seed));
        legacy_val_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Validation, false, false, config_.num_workers,
            static_cast<uint32_t>(config_.dataloader_seed));

        // Apply NEW preprocessing pipeline (if configured)
        const std::string dataset_name = !config_.dataset_name.empty()
            ? config_.dataset_name
            : dataset_.GetName();
        DataRegistry& registry = DataRegistry::Instance();

        if (registry.HasPreprocessingConfig(dataset_name)) {
            spdlog::info("TrainingExecutor: Found preprocessing config for dataset '{}'", dataset_name);

            PreprocessingConfig preprocessing_config = registry.GetPreprocessingConfig(dataset_name);

            if (preprocessing_config.enabled) {
                legacy_train_batcher->SetPreprocessingConfig(preprocessing_config);
                legacy_val_batcher->SetPreprocessingConfig(preprocessing_config);

                spdlog::info("TrainingExecutor: Computing dataset statistics...");
                DatasetStatistics stats = StatisticsCalculator::Compute(
                    dataset_name, &registry,
                    [](float progress) {
                        spdlog::debug("Statistics computation: {:.1f}%", progress * 100.0f);
                    }
                );

                if (stats.is_valid) {
                    legacy_train_batcher->InitializePreprocessing(stats);
                    legacy_val_batcher->InitializePreprocessing(stats);
                    spdlog::info("TrainingExecutor: Preprocessing pipeline initialized");
                }
            }
        }

        // Load augmentation pipeline
        if (registry.HasAugmentationPipeline(dataset_name)) {
            auto aug_pipeline = registry.GetAugmentationPipeline(dataset_name);
            if (aug_pipeline) {
                legacy_train_batcher->SetAugmentationPipeline(aug_pipeline);
                legacy_train_batcher->SetApplyAugmentationOnTrain(true);
            }
        }

        // Apply OLD preprocessing settings
        if (config_.preprocessing.has_normalization) {
            legacy_train_batcher->SetLegacyNormalization(config_.preprocessing.norm_mean,
                                                         config_.preprocessing.norm_std);
            legacy_val_batcher->SetLegacyNormalization(config_.preprocessing.norm_mean,
                                                       config_.preprocessing.norm_std);
        }

        if (UsesScalarBinaryTargets(config_.loss_type)) {
            legacy_train_batcher->SetLegacyScalarLabelMode(true);
            legacy_val_batcher->SetLegacyScalarLabelMode(true);
        } else if (config_.preprocessing.has_onehot) {
            legacy_train_batcher->SetLegacyOneHotEncoding(config_.preprocessing.num_classes);
            legacy_val_batcher->SetLegacyOneHotEncoding(config_.preprocessing.num_classes);
        }

        legacy_train_batcher->SetFlatten(true);
        legacy_val_batcher->SetFlatten(true);

        num_train_samples = legacy_train_batcher->GetNumSamples();
        num_val_samples = legacy_val_batcher->GetNumSamples();
#else
        spdlog::error("TrainingExecutor: legacy DatasetHandle mode is disabled "
                      "for this modern-only test build");
        fail_run("legacy_dataset_mode_disabled");
        return;
#endif
    }

    if (mode_ == DatasetMode::SequenceExternal && active_sequence_batcher) {
        active_sequence_batcher->SetPhase(BatcherPhase::Train);
        num_train_samples = active_sequence_batcher->GetNumSamples();
        active_sequence_batcher->SetPhase(BatcherPhase::Val);
        num_val_samples = active_sequence_batcher->GetNumSamples();
        if (active_sequence_batcher->HasPhase(BatcherPhase::Test)) {
            active_sequence_batcher->SetPhase(BatcherPhase::Test);
            num_test_samples = active_sequence_batcher->GetNumSamples();
        }
        active_sequence_batcher->SetPhase(BatcherPhase::Train);
    } else if (active_train_ibatcher) {
        active_train_ibatcher->SetPhase(train_batcher_phase);
        num_train_samples = active_train_ibatcher->GetNumSamples();

        if (active_val_ibatcher) {
            active_val_ibatcher->SetPhase(val_batcher_phase);
            num_val_samples = active_val_ibatcher->GetNumSamples();
            active_val_ibatcher->SetPhase(train_batcher_phase);
        }

        if (active_test_ibatcher) {
            active_test_ibatcher->SetPhase(test_batcher_phase);
            num_test_samples = active_test_ibatcher->GetNumSamples();
            active_test_ibatcher->SetPhase(train_batcher_phase);
        }

        active_train_ibatcher->SetPhase(train_batcher_phase);
    }

    UpdateMetrics([num_train_samples, num_val_samples, num_test_samples](
                      TrainingMetrics& m) {
        m.train_sample_count = num_train_samples;
        m.val_sample_count = num_val_samples;
        m.test_sample_count = num_test_samples;
    });
    FinalizePartitionManifest(
        config_.dataset_roles,
        static_cast<int64_t>(num_train_samples),
        static_cast<int64_t>(num_val_samples),
        static_cast<int64_t>(num_test_samples));

    spdlog::info("TrainingExecutor: Starting training for {} epochs, batch_size={}, samples={}",
                 epochs, batch_size, num_train_samples);

    spdlog::debug("TrainingExecutor: Step 1 - Notifying plugin hooks");
    // Notify plugin hooks: training start
    {
        cyxwiz::plugin::TrainingContext ctx;
        ctx.total_epochs = epochs;
        ctx.learning_rate = optimizer_
            ? static_cast<float>(optimizer_->GetLearningRate())
            : config_.learning_rate;
        cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyTrainingStart(ctx);
    }

    spdlog::debug("TrainingExecutor: Step 2 - Setting model to training mode");
    // Set model to training mode
    model_->SetTraining(true);

    spdlog::debug("TrainingExecutor: Step 3 - Entering training loop");
    CrashRunRecorder::Instance().UpdateSampleCount(num_train_samples);
    BackendDebugHooks::SetDebugEventCallback([](const std::string& source,
                                                const std::string& message) {
        if (source.rfind("Model", 0) == 0) {
            TrainingTraceCollector::Instance().RecordRuntimeEvent(source, message);
        } else {
            CrashRunRecorder::Instance().MarkBackendEvent(source, message);
            TrainingTraceCollector::Instance().RecordRuntimeWarning(source, message);
        }
    });
    const auto last_run = CrashRunRecorder::LoadLastRun();
    const auto trace_snapshot = TrainingTraceCollector::Instance().Snapshot();
    if (!trace_snapshot.available || trace_snapshot.status != "running") {
        TrainingTraceCollector::Instance().StartRun(
            last_run ? last_run->run_id : "training-run");
    } else {
        TrainingTraceCollector::Instance().RecordRuntimeEvent(
            "TrainingLoop",
            "Training loop attached to existing setup trace");
    }
    ReportPinMemoryTransferStatus(config_);
    gradient_accumulator_.clear();
    gradient_accumulated_batches_ = 0;
    gradient_accumulation_weight_ = 0.0f;
    gradient_accumulation_device_weight_ = Tensor();
    gradient_accumulation_device_weight_initialized_ = false;

    std::unique_ptr<CheckpointManager> checkpoint_manager;
    std::optional<TrainingSchedulerResumeState> best_scheduler_state;
    float best_val_loss = std::numeric_limits<float>::infinity();
    int epochs_without_improvement = 0;
    const int early_stopping_patience = std::max(0, config_.early_stopping_patience);
    const bool save_best_checkpoint = config_.save_best_checkpoint;
    const int validation_freq = std::max(1, config_.validation_freq);
    const int log_interval = std::max(0, config_.log_interval);
    spdlog::info("TrainingExecutor: DataLoader runtime policy validation_freq={} epoch(s), metric_report_interval={} batch(es), seed={}, grad_accum_steps={}",
                 validation_freq, log_interval, config_.dataloader_seed,
                 training_contract::ClampGradientAccumulationSteps(
                     config_.grad_accum_steps));
    const std::string reporting_cadence = log_interval > 0
        ? fmt::format(
              "Metrics are sampled on batch 1, every {} batches, and the final batch.",
              log_interval)
        : "Metrics are sampled on the first and final batch.";
    TrainingTraceCollector::Instance().RecordRuntimeEvent(
        "TrainingExecutor.ReportingCadence",
        reporting_cadence);
    std::string terminal_status;
    std::string terminal_reason;

    std::filesystem::path checkpoint_root = config_.checkpoint_dir.empty()
        ? (std::filesystem::current_path() / ".cyxwiz" / "checkpoints")
        : std::filesystem::path(config_.checkpoint_dir);
    checkpoint_root /= last_run ? last_run->run_id : "training-run";
    checkpoint_manager = std::make_unique<CheckpointManager>(checkpoint_root.string());

    // Training loop
    const bool regression_metrics = UsesRegressionMetrics(config_);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        spdlog::debug("TrainingExecutor: Epoch {} starting", epoch);
        if (ShouldStop()) break;
        // Check plugin early stopping
        {
            cyxwiz::plugin::TrainingContext stop_ctx;
            stop_ctx.current_epoch = epoch;
            stop_ctx.total_epochs = epochs;
            stop_ctx.learning_rate = optimizer_
                ? static_cast<float>(optimizer_->GetLearningRate())
                : config_.learning_rate;
            if (cyxwiz::plugin::PluginTrainingHookManager::Instance().ShouldStopEarly(stop_ctx)) {
                spdlog::info("TrainingExecutor: Plugin requested early stop");
                terminal_status = "early_stopped";
                terminal_reason = "plugin_requested_early_stop";
                UpdateMetrics([&](TrainingMetrics& m) {
                    m.terminal_status = terminal_status;
                    m.terminal_reason = terminal_reason;
                    m.status_message = "Early stopping: plugin requested stop";
                });
                break;
            }
        }
        if (!WaitWhilePaused()) break;

        auto epoch_start = std::chrono::steady_clock::now();

        UpdateMetrics([epoch](TrainingMetrics& m) {
            m.current_epoch = epoch;
            m.status_message = "Training epoch " + std::to_string(epoch) + "...";
        });

        // Notify plugin hooks: epoch start
        {
            cyxwiz::plugin::TrainingContext ctx;
            ctx.current_epoch = epoch;
            ctx.total_epochs = epochs;
            ctx.learning_rate = optimizer_
                ? static_cast<float>(optimizer_->GetLearningRate())
                : config_.learning_rate;
            cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyEpochStart(ctx);
        }

        spdlog::debug("TrainingExecutor: About to call RunTrainingEpoch");
        // Run training epoch - dispatch by dataset mode. Arrow and Parquet
        // batchers both flow through RunTrainingEpochArrow via their shared
        // IBatcher base; legacy DatasetBatcher stays on its own path.
        if (mode_ == DatasetMode::Legacy) {
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
            RunTrainingEpoch(*legacy_train_batcher, epoch, batch_cb);
#else
            break;
#endif
        } else if (mode_ == DatasetMode::SequenceExternal &&
                   active_sequence_batcher) {
            RunTrainingEpochSequence(*active_sequence_batcher, epoch, batch_cb);
        } else if (active_train_ibatcher) {
            RunTrainingEpochArrow(*active_train_ibatcher, epoch, batch_cb);
        }

        if (ShouldStop()) break;

        // Run validation (eval mode).
        //
        // For image/audio batchers, active_train_ibatcher and
        // active_val_ibatcher point to the *same* instance - the batcher
        // holds both train_indices_ and val_indices_ internally and switches
        // between them via SetPhase. Without SetPhase(Val) the val pass
        // would iterate the training indices, producing bogus "perfect val"
        // metrics (this was the source of the suspicious 100% val acc).
        // For Arrow/Parquet/legacy paths these are separate instances so
        // SetPhase is a no-op on the default IBatcher impl.
        const bool should_validate_this_epoch =
            ShouldRunValidationEpoch(config_, epoch, epochs);
        bool validation_ran_this_epoch = false;
        if (should_validate_this_epoch) {
            model_->SetTraining(false);
        }
        if (should_validate_this_epoch && mode_ == DatasetMode::Legacy) {
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
            RunValidation(*legacy_val_batcher);
            validation_ran_this_epoch = true;
#endif
        } else if (should_validate_this_epoch &&
                   mode_ == DatasetMode::SequenceExternal &&
                   active_sequence_batcher) {
            active_sequence_batcher->SetPhase(BatcherPhase::Val);
            RunValidationSequence(*active_sequence_batcher);
            active_sequence_batcher->SetPhase(BatcherPhase::Train);
            validation_ran_this_epoch = true;
        } else if (should_validate_this_epoch && active_val_ibatcher) {
            active_val_ibatcher->SetPhase(val_batcher_phase);
            RunValidationArrow(*active_val_ibatcher);
            active_val_ibatcher->SetPhase(train_batcher_phase);
            validation_ran_this_epoch = true;
        } else if (!should_validate_this_epoch) {
            spdlog::debug("TrainingExecutor: Skipping validation at epoch {} (validation_freq={})",
                          epoch, validation_freq);
        }
        if (should_validate_this_epoch) {
            model_->SetTraining(true);
        }

        auto epoch_end = std::chrono::steady_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();

        // Get current metrics for callback
        TrainingMetrics current = GetMetrics();

        // Compute samples per second
        float samples_per_sec = static_cast<float>(num_train_samples) / epoch_time;

        // Update history
        UpdateMetrics([&](TrainingMetrics& m) {
            m.last_executed_epoch = epoch;
            m.epoch_time_seconds = epoch_time;
            m.samples_per_second = samples_per_sec;
            m.loss_history.push_back(m.train_loss);
            if (regression_metrics) {
                m.mae_history.push_back(m.train_mae);
                m.rmse_history.push_back(m.train_rmse);
            } else {
                m.accuracy_history.push_back(m.train_accuracy);
            }
            if (validation_ran_this_epoch) {
                m.val_loss_history.push_back(m.val_loss);
                if (regression_metrics) {
                    m.val_mae_history.push_back(m.val_mae);
                    m.val_rmse_history.push_back(m.val_rmse);
                } else {
                    m.val_accuracy_history.push_back(m.val_accuracy);
                }
            }
        });
        CrashRunRecorder::Instance().UpdateLastExecutedEpoch(epoch);

        if (scheduler_controller_) {
            const std::optional<float> scheduler_validation_loss =
                validation_ran_this_epoch &&
                        current.val_sample_count > 0 &&
                        std::isfinite(current.val_loss)
                    ? std::optional<float>(current.val_loss)
                    : std::nullopt;
            ApplySchedulerAdvance(
                scheduler_controller_->OnEpochCompleted(
                    scheduler_validation_loss));
            current = GetMetrics();
        }

        // Epoch callback
        if (epoch_cb) {
            epoch_cb(epoch, current.train_loss, current.train_accuracy,
                     validation_ran_this_epoch ? current.val_loss : -1.0f,
                     validation_ran_this_epoch ? current.val_accuracy : -1.0f,
                     epoch_time);
        }

        if (validation_ran_this_epoch && regression_metrics) {
            spdlog::info(
                "Epoch {}/{}: loss={:.4f}, mae={:.4f}, rmse={:.4f}, "
                "val_loss={:.4f}, val_mae={:.4f}, val_rmse={:.4f} "
                "({:.1f}s, {:.0f} samples/sec)",
                epoch, epochs, current.train_loss, current.train_mae,
                current.train_rmse, current.val_loss, current.val_mae,
                current.val_rmse, epoch_time, samples_per_sec);
            TrainingTraceCollector::Instance().RecordValidationMetrics(
                epoch, current.train_loss, 0.0f, current.val_loss, 0.0f,
                epoch_time * 1000.0f);
        } else if (validation_ran_this_epoch) {
            spdlog::info("Epoch {}/{}: loss={:.4f}, acc={:.2f}%, val_loss={:.4f}, val_acc={:.2f}% ({:.1f}s, {:.0f} samples/sec)",
                         epoch, epochs, current.train_loss, current.train_accuracy * 100,
                         current.val_loss, current.val_accuracy * 100, epoch_time, samples_per_sec);
            TrainingTraceCollector::Instance().RecordValidationMetrics(
                epoch,
                current.train_loss,
                current.train_accuracy,
                current.val_loss,
                current.val_accuracy,
                epoch_time * 1000.0f);
        } else if (regression_metrics) {
            spdlog::info(
                "Epoch {}/{}: loss={:.4f}, mae={:.4f}, rmse={:.4f}, "
                "validation skipped (validation_freq={}) "
                "({:.1f}s, {:.0f} samples/sec)",
                epoch, epochs, current.train_loss, current.train_mae,
                current.train_rmse, validation_freq, epoch_time,
                samples_per_sec);
        } else {
            spdlog::info("Epoch {}/{}: loss={:.4f}, acc={:.2f}%, validation skipped (validation_freq={}) ({:.1f}s, {:.0f} samples/sec)",
                         epoch, epochs, current.train_loss, current.train_accuracy * 100,
                         validation_freq, epoch_time, samples_per_sec);
        }

        bool stop_after_epoch = false;
        if (validation_ran_this_epoch && std::isfinite(current.val_loss)) {
            if (current.val_loss < best_val_loss) {
                best_val_loss = current.val_loss;
                epochs_without_improvement = 0;
                if (checkpoint_manager && save_best_checkpoint) {
                    if (SequentialModel* sequential_model =
                            model_->AsSequentialModel()) {
                        if (checkpoint_manager->SaveBestModel(
                                *sequential_model,
                                optimizer_.get(),
                                current,
                                current.val_loss)) {
                            if (scheduler_controller_) {
                                TrainingSchedulerResumeState scheduler_state;
                                std::string scheduler_error;
                                if (!scheduler_controller_->ExportResumeState(
                                        scheduler_state,
                                        scheduler_error)) {
                                    throw std::runtime_error(
                                        "TrainingExecutor: best checkpoint "
                                        "scheduler state export failed: " +
                                        scheduler_error);
                                }
                                best_scheduler_state =
                                    std::move(scheduler_state);
                            }
                            TrainingTraceCollector::Instance()
                                .RecordCheckpointSaved(
                                    epoch,
                                    checkpoint_manager->GetBestCheckpoint(),
                                    current.val_loss,
                                    current.val_accuracy,
                                    true);
                            spdlog::info(
                                "TrainingExecutor: Best validation checkpoint saved at epoch {} (val_loss={:.4f})",
                                epoch,
                                current.val_loss);
                        }
                    } else {
                        spdlog::warn(
                            "TrainingExecutor: save_best_checkpoint is not available for graph executable models yet");
                        checkpoint_manager.reset();
                    }
                }
            } else {
                ++epochs_without_improvement;
                spdlog::info("TrainingExecutor: No validation improvement for {} epoch(s) (best val_loss={:.4f})",
                             epochs_without_improvement, best_val_loss);
                if (early_stopping_patience > 0 &&
                    epochs_without_improvement >= early_stopping_patience) {
                    spdlog::info("TrainingExecutor: Early stopping after {} non-improving epoch(s) (patience={})",
                                 epochs_without_improvement, early_stopping_patience);
                    terminal_status = "early_stopped";
                    terminal_reason =
                        "validation_loss_plateau_patience_" +
                        std::to_string(early_stopping_patience);
                    UpdateMetrics([&](TrainingMetrics& m) {
                        m.terminal_status = terminal_status;
                        m.terminal_reason = terminal_reason;
                        m.status_message = "Early stopping: validation loss plateaued";
                    });
                    stop_after_epoch = true;
                }
            }
        }

        // Notify plugin hooks: epoch end
        {
            cyxwiz::plugin::TrainingContext ctx;
            ctx.current_epoch = epoch;
            ctx.total_epochs = epochs;
            ctx.train_loss = current.train_loss;
            ctx.train_accuracy = current.train_accuracy;
            ctx.val_loss = validation_ran_this_epoch
                ? current.val_loss
                : -1.0f;
            ctx.val_accuracy = validation_ran_this_epoch
                ? current.val_accuracy
                : -1.0f;
            ctx.learning_rate = optimizer_
                ? static_cast<float>(optimizer_->GetLearningRate())
                : config_.learning_rate;
            cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyEpochEnd(ctx);
        }

        if (stop_after_epoch) {
            break;
        }

        // Reset batchers for next epoch
        if (mode_ == DatasetMode::Legacy) {
#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
            legacy_train_batcher->Reset();
            legacy_val_batcher->Reset();
#endif
        } else {
            if (mode_ == DatasetMode::SequenceExternal &&
                active_sequence_batcher) {
                active_sequence_batcher->Reset();
            } else {
                active_train_ibatcher->Reset();
                if (active_val_ibatcher != active_train_ibatcher) {
                    active_val_ibatcher->Reset();
                }
                if (active_test_ibatcher &&
                    active_test_ibatcher != active_train_ibatcher &&
                    active_test_ibatcher != active_val_ibatcher) {
                    active_test_ibatcher->Reset();
                }
            }
        }
    }

    TrainingMetrics final_metrics = GetMetrics();

    // Mark complete
    if (terminal_status.empty()) {
        terminal_status = stop_requested_.load() ? "cancelled" : "completed";
        terminal_reason = stop_requested_.load()
            ? "user_cancelled"
            : "completed_all_epochs";
    }
    UpdateMetrics([&](TrainingMetrics& m) {
        m.is_training = false;
        m.is_paused = false;
        m.is_complete = true;
        m.terminal_status = terminal_status;
        m.terminal_reason = terminal_reason;
        if (terminal_status == "early_stopped") {
            m.status_message = "Training early-stopped: " + terminal_reason;
        } else if (terminal_status == "cancelled") {
            m.status_message = "Training cancelled";
        } else {
            m.status_message = "Training complete";
        }
    });
    final_metrics = GetMetrics();
    TrainingTraceCollector::Instance().RecordTerminalEvent(
        terminal_status,
        terminal_reason,
        final_metrics.current_epoch,
        final_metrics.train_loss,
        final_metrics.train_accuracy);

    if (!stop_requested_.load() && checkpoint_manager && save_best_checkpoint) {
        const std::string best_checkpoint = checkpoint_manager->GetBestCheckpoint();
        if (!best_checkpoint.empty()) {
            std::optional<CheckpointMetadata> restored;
            if (SequentialModel* sequential_model = model_->AsSequentialModel()) {
                restored = checkpoint_manager->LoadCheckpoint(*sequential_model,
                                                              optimizer_.get(),
                                                              "best");
            }
            if (restored) {
                if (scheduler_controller_ && best_scheduler_state) {
                    std::string scheduler_error;
                    if (!scheduler_controller_->Restore(
                            *best_scheduler_state,
                            scheduler_error)) {
                        throw std::runtime_error(
                            "TrainingExecutor: best checkpoint scheduler "
                            "restore failed: " + scheduler_error);
                    }
                }
                const double active_learning_rate = optimizer_
                    ? optimizer_->GetLearningRate()
                    : static_cast<double>(config_.learning_rate);
                final_metrics.checkpoint_used = best_checkpoint;
                final_metrics.restored_checkpoint_epoch = restored->epoch;
                final_metrics.restored_checkpoint_step = restored->global_step;
                final_metrics.active_model_provenance =
                    "restored_best_checkpoint";
                UpdateMetrics([&](TrainingMetrics& m) {
                    m.checkpoint_used = best_checkpoint;
                    m.restored_checkpoint_epoch = restored->epoch;
                    m.restored_checkpoint_step = restored->global_step;
                    m.active_model_provenance =
                        "restored_best_checkpoint";
                    m.learning_rate = active_learning_rate;
                    m.terminal_status = terminal_status;
                    m.terminal_reason = terminal_reason;
                    m.status_message = std::string(
                        "Restored best validation checkpoint after ") +
                        terminal_status + ": " + terminal_reason;
                });
                CrashRunRecorder::Instance().MarkActiveModelCheckpoint(
                    best_checkpoint,
                    restored->epoch,
                    restored->global_step,
                    "restored_best_checkpoint");
                TrainingTraceCollector::Instance().RecordCheckpointRestored(
                    restored->epoch,
                    best_checkpoint,
                    restored->val_loss,
                    restored->val_accuracy);
                spdlog::info("TrainingExecutor: Restored best checkpoint from epoch {} (val_loss={:.4f})",
                             restored->epoch, restored->val_loss);
            }
        }
    }

    if (!stop_requested_.load() && mode_ == DatasetMode::SequenceExternal &&
        active_sequence_batcher) {
        if (!active_sequence_batcher->HasPhase(BatcherPhase::Test)) {
            spdlog::debug(
                "TrainingExecutor: no explicit Sequence Test phase; "
                "held-out metrics were skipped");
        } else {
            active_sequence_batcher->SetPhase(BatcherPhase::Test);
        }
        if (active_sequence_batcher->HasPhase(BatcherPhase::Test) &&
            active_sequence_batcher->GetNumSamples() > 0) {
            model_->SetTraining(false);
            const auto test_evaluation =
                EvaluateSequenceBatcher(*active_sequence_batcher);
            UpdateMetrics([test_evaluation](TrainingMetrics& m) {
                m.test_loss = test_evaluation.loss;
                m.test_accuracy = test_evaluation.accuracy;
                m.test_token_accuracy = test_evaluation.accuracy;
                m.test_entity_f1 = test_evaluation.entity_f1;
                m.test_token_count = test_evaluation.token_count;
                m.has_test_metrics = true;
            });
            final_metrics.test_loss = test_evaluation.loss;
            final_metrics.test_accuracy = test_evaluation.accuracy;
            final_metrics.test_token_accuracy = test_evaluation.accuracy;
            final_metrics.test_entity_f1 = test_evaluation.entity_f1;
            final_metrics.test_token_count = test_evaluation.token_count;
            final_metrics.has_test_metrics = true;
            TrainingTraceCollector::Instance().RecordHeldOutTestMetrics(
                final_metrics.last_executed_epoch,
                test_evaluation.loss,
                test_evaluation.accuracy,
                final_metrics.active_model_provenance,
                final_metrics.checkpoint_used);
            spdlog::info(
                "TrainingExecutor: Held-out Sequence test metrics "
                "test_loss={:.4f}, token_acc={:.2f}%, entity_f1={:.2f}% "
                "({} samples, {} tokens)",
                test_evaluation.loss,
                test_evaluation.accuracy * 100.0f,
                test_evaluation.entity_f1 * 100.0f,
                active_sequence_batcher->GetNumSamples(),
                test_evaluation.token_count);
        } else if (active_sequence_batcher->HasPhase(BatcherPhase::Test)) {
            spdlog::warn(
                "TrainingExecutor: configured Sequence test split produced "
                "0 held-out samples; test metrics were skipped");
        }
    } else if (!stop_requested_.load() && active_test_ibatcher &&
        active_test_ibatcher->GetNumSamples() > 0) {
        model_->SetTraining(false);
        active_test_ibatcher->SetPhase(test_batcher_phase);
        const auto test_evaluation = EvaluateArrowBatcher(*active_test_ibatcher);
        active_test_ibatcher->SetPhase(train_batcher_phase);
        UpdateMetrics([test_evaluation](TrainingMetrics& m) {
            m.test_loss = test_evaluation.loss;
            m.test_accuracy = test_evaluation.accuracy;
            m.test_mae = test_evaluation.mae;
            m.test_rmse = test_evaluation.rmse;
            m.has_test_metrics = true;
        });
        final_metrics.test_loss = test_evaluation.loss;
        final_metrics.test_accuracy = test_evaluation.accuracy;
        final_metrics.test_mae = test_evaluation.mae;
        final_metrics.test_rmse = test_evaluation.rmse;
        final_metrics.has_test_metrics = true;
        TrainingTraceCollector::Instance().RecordHeldOutTestMetrics(
            final_metrics.last_executed_epoch,
            test_evaluation.loss,
            test_evaluation.accuracy,
            final_metrics.active_model_provenance,
            final_metrics.checkpoint_used);
        if (regression_metrics) {
            spdlog::info(
                "TrainingExecutor: Held-out test metrics test_loss={:.4f}, "
                "test_mae={:.4f}, test_rmse={:.4f} ({} samples)",
                test_evaluation.loss, test_evaluation.mae,
                test_evaluation.rmse,
                active_test_ibatcher->GetNumSamples());
        } else {
            spdlog::info("TrainingExecutor: Held-out test metrics test_loss={:.4f}, test_acc={:.2f}% ({} samples)",
                         test_evaluation.loss, test_evaluation.accuracy * 100.0f,
                         active_test_ibatcher->GetNumSamples());
        }
    } else if (!stop_requested_.load() && active_test_ibatcher) {
        spdlog::warn("TrainingExecutor: configured test split produced 0 held-out samples; "
                     "test metrics were skipped");
    }

    // Notify plugin hooks: training end
    final_metrics = GetMetrics();
    {
        cyxwiz::plugin::TrainingContext ctx;
        ctx.current_epoch = final_metrics.current_epoch;
        ctx.total_epochs = final_metrics.total_epochs;
        ctx.train_loss = final_metrics.train_loss;
        ctx.train_accuracy = final_metrics.train_accuracy;
        ctx.val_loss = final_metrics.val_loss;
        ctx.val_accuracy = final_metrics.val_accuracy;
        ctx.learning_rate = optimizer_
            ? static_cast<float>(optimizer_->GetLearningRate())
            : config_.learning_rate;
        cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyTrainingEnd(ctx);
    }

    if (stop_requested_.load()) {
        CrashRunRecorder::Instance().MarkCancelled();
        TrainingTraceCollector::Instance().FinishRun("cancelled");
    } else if (terminal_status == "early_stopped") {
        CrashRunRecorder::Instance().MarkEarlyStopped(terminal_reason);
        TrainingTraceCollector::Instance().FinishRun("early_stopped");
    } else {
        CrashRunRecorder::Instance().MarkCompleted();
        TrainingTraceCollector::Instance().FinishRun("completed");
    }
    BackendDebugHooks::SetDebugEventCallback({});

    is_training_.store(false);

    // Complete callback
    if (complete_cb) {
        complete_cb(GetMetrics());
    }

    spdlog::info("TrainingExecutor: Training complete");
    } catch (const std::exception& e) {
        const std::string coded_error = errors::FormatError(
            errors::Training::TrainingExecutionFailed,
            "Training failed",
            e.what());
        fail_run(coded_error);
        spdlog::error("TrainingExecutor: {}", coded_error);
        throw;
    } catch (...) {
        const std::string coded_error = errors::FormatError(
            errors::Training::TrainingExecutionFailed,
            "Training failed with unknown native exception");
        fail_run(coded_error);
        spdlog::error("TrainingExecutor: {}", coded_error);
        throw;
    }
}

#ifndef CYXWIZ_TRAINING_EXECUTOR_MODERN_ONLY
void TrainingExecutor::RunTrainingEpoch(
    DatasetBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    float epoch_loss = 0.0f;
    float loss_weight_sum = 0.0f;
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
    Tensor device_loss_sum;
    bool device_loss_sum_initialized = false;
    Tensor device_loss_weight_sum;
    bool device_loss_weight_sum_initialized = false;
    Tensor device_accuracy_counts;
    bool device_accuracy_counts_initialized = false;
    const auto metric_ignore_index = ClassificationMetricIgnoreIndex(config_);
    int batch_num = 0;
    float current_loss = 0.0f;
    float current_acc = 0.0f;

    size_t total_batches = batcher.GetNumBatches();

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    // Epoch wall-clock start for the periodic progress log below -
    // mirrors RunTrainingEpochArrow so both training paths have the
    // same "training is alive" feedback loop.
    const auto epoch_start_time = std::chrono::steady_clock::now();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        if (!WaitWhilePaused()) break;

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        const auto fetch_start = std::chrono::steady_clock::now();
        Batch batch = batcher.GetNextBatch();
        const auto fetch_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - fetch_start).count();
        if (!batch.IsValid()) break;

        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches), 0.0f, 0.0f, fetch_ms);

        batch_num++;

        // Forward pass through model
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches));
        const auto forward_start = std::chrono::steady_clock::now();
        Tensor predictions = Forward(batch.data);
        const auto forward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - forward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches), 0.0f, 0.0f, forward_ms);

        // DEBUG: Log sample values for first batch of first epoch.
        if (epoch == 1 && batch_num == 1 &&
            !ShouldMaterializeFirstBatchDebugSamples(config_)) {
            RecordSkippedFirstBatchDebugSampleDump();
        } else if (epoch == 1 && batch_num == 1) {
            const ScopedArrayFireHostSyncAttribution debug_sync_attribution(
                ArrayFireHostSyncCategory::DebugSampleDump,
                "TrainingExecutor::FirstBatchDebugSampleDump");
            const float* input_data = batch.data.ReadData<float>();
            const float* pred_data_debug = predictions.ReadData<float>();
            const float* target_data_debug = batch.labels.ReadData<float>();

            // Log input data range
            float min_input = input_data[0], max_input = input_data[0];
            const auto& input_shape = batch.data.Shape();
            if (input_shape.size() < 2) {
                spdlog::error("TrainingExecutor: Expected 2D input, got {}D", input_shape.size());
                break;
            }
            size_t input_size = input_shape[0] * input_shape[1];
            for (size_t i = 1; i < std::min(input_size, size_t(1000)); ++i) {
                min_input = std::min(min_input, input_data[i]);
                max_input = std::max(max_input, input_data[i]);
            }
            spdlog::info("DEBUG: Input data range: [{:.4f}, {:.4f}]", min_input, max_input);

            // Log first sample prediction
            spdlog::info("DEBUG: First sample predictions:");
            std::string pred_str = "  [";
            for (size_t c = 0; c < config_.output_size; ++c) {
                pred_str += fmt::format("{:.4f}", pred_data_debug[c]);
                if (c < config_.output_size - 1) pred_str += ", ";
            }
            pred_str += "]";
            spdlog::info("{}", pred_str);

            // Log first sample target
            spdlog::info("DEBUG: First sample target:");
            std::string target_str = "  [";
            for (size_t c = 0; c < config_.output_size; ++c) {
                target_str += fmt::format("{:.1f}", target_data_debug[c]);
                if (c < config_.output_size - 1) target_str += ", ";
            }
            target_str += "]";
            spdlog::info("{}", target_str);
        }

        // Compute loss
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches));
        float batch_loss = current_loss;
        std::string loss_status = "device_resident";
        LossAggregationWeightValue loss_weight;
        if (regression_metrics) {
            batch_loss = ComputeLoss(predictions, batch.labels);
            loss_weight = ResolveLossAggregationWeight(*loss_, batch.size);
            loss_weight_sum += loss_weight.host;
            epoch_loss += batch_loss * loss_weight.host;
            loss_status = std::isfinite(batch_loss) ? "ok" : "failed";
        } else {
            Tensor loss_tensor = ComputeLossTensor(predictions, batch.labels);
            loss_weight = ResolveLossAggregationWeight(*loss_, batch.size);
            Tensor weighted_loss = loss_weight.device != nullptr
                ? ApplyDeviceScalar(loss_tensor, *loss_weight.device, false)
                : loss_tensor * loss_weight.host;
            AccumulateDeviceScalar(
                device_loss_sum,
                device_loss_sum_initialized,
                weighted_loss);
            if (loss_weight.device != nullptr) {
                AccumulateDeviceScalar(
                    device_loss_weight_sum,
                    device_loss_weight_sum_initialized,
                    *loss_weight.device);
            } else {
                loss_weight_sum += loss_weight.host;
            }
        }
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, 0.0f,
            loss_status,
            loss_status == "failed" ? "Training loss became NaN or Inf." : "");

        // Compute objective-appropriate metrics.
        if (regression_metrics) {
            AddRegressionMetricScalars(
                regression,
                predictions,
                batch.labels,
                batch.size,
                config_.output_size,
                config_);
        } else {
            const auto accuracy_scalar = BuildClassificationDecisionScalar(
                predictions, batch.labels, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type),
                metric_ignore_index);
            AccumulateDeviceClassificationCounts(
                device_accuracy_counts,
                device_accuracy_counts_initialized,
                accuracy_scalar.counts);
        }

        // Backward pass
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        const auto backward_start = std::chrono::steady_clock::now();
        Backward(predictions, batch.labels);
        const auto backward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - backward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, backward_ms);

        const bool should_report = ShouldReportTrainingBatch(
            config_, batch_num, static_cast<int>(total_batches),
            batcher.IsEpochComplete());
        if (!regression_metrics && should_report) {
            current_loss = ReadAccumulatedLoss(
                device_loss_sum, *loss_, loss_weight_sum,
                device_loss_weight_sum_initialized
                    ? &device_loss_weight_sum
                    : nullptr);
            const auto accuracy_count = ReadClassificationDecisionScalar(
                ClassificationDecisionScalar{device_accuracy_counts},
                "TrainingExecutor::ReadAccumulatedAccuracy");
            current_acc = accuracy_count.total > 0
                ? static_cast<float>(accuracy_count.correct) /
                      static_cast<float>(accuracy_count.total)
                : 0.0f;
            batch_loss = current_loss;
        } else if (regression_metrics) {
            current_loss = FinalizeAggregatedLoss(
                *loss_, epoch_loss, loss_weight_sum);
        }

        // Update metrics
        const float current_mae = regression.Mae();
        const float current_rmse = regression.Rmse();

        AccumulateGradientsAndMaybeStep(
            epoch, batch_num, static_cast<int>(total_batches),
            batch_loss, current_acc, loss_weight.host, loss_weight.device,
            batcher.IsEpochComplete());

        UpdateMetrics([batch_num, current_loss, current_acc,
                       current_mae, current_rmse,
                       regression_metrics](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
            if (regression_metrics) {
                m.train_mae = current_mae;
                m.train_rmse = current_rmse;
            }
        });

        // Periodic progress log - mirror of RunTrainingEpochArrow's
        // version so non-Arrow training paths (legacy DatasetBatcher)
        // also get per-50-batch liveness signals.
        if (ShouldLogTrainingBatch(config_, batch_num)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            if (regression_metrics) {
                spdlog::info(
                    "Epoch {} [{}/{}] loss={:.4f} mae={:.4f} rmse={:.4f} "
                    "({:.1f}s, {:.1f} batches/s)",
                    epoch, batch_num, total_batches, current_loss,
                    current_mae, current_rmse, elapsed_s, rate);
            } else {
                spdlog::info("Epoch {} [{}/{}] loss={:.4f} acc={:.2f}% "
                             "({:.1f}s, {:.1f} batches/s)",
                             epoch, batch_num, total_batches,
                             current_loss, current_acc * 100.0f,
                             elapsed_s, rate);
            }
        }

        // Batch callback
        if (batch_cb) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            TrainingTraceCollector::Instance().RecordStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            batch_cb(epoch, batch_num, static_cast<int>(total_batches), current_loss, current_acc);
        }
    }

    // Final epoch metrics
    float final_loss = current_loss;
    float final_acc = current_acc;
    const float final_mae = regression.Mae();
    const float final_rmse = regression.Rmse();

    UpdateMetrics([final_loss, final_acc, final_mae, final_rmse,
                   regression_metrics](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
        if (regression_metrics) {
            m.train_mae = final_mae;
            m.train_rmse = final_rmse;
        }
    });
    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
}

void TrainingExecutor::RunValidation(DatasetBatcher& batcher) {
    float val_loss = 0.0f;
    float loss_weight_sum = 0.0f;
    Tensor device_loss_sum;
    bool device_loss_sum_initialized = false;
    Tensor device_loss_weight_sum;
    bool device_loss_weight_sum_initialized = false;
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
    int correct = 0;
    int total = 0;
    const auto metric_ignore_index = ClassificationMetricIgnoreIndex(config_);
    int batch_num = 0;

    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass only (no backprop)
        Tensor predictions = Forward(batch.data);

        // Compute loss
        if (regression_metrics) {
            const float batch_loss = ComputeLoss(predictions, batch.labels);
            const float loss_weight = LossAggregationWeight(*loss_, batch.size);
            val_loss += batch_loss * loss_weight;
            loss_weight_sum += loss_weight;
        } else {
            Tensor loss_tensor = ComputeLossTensor(predictions, batch.labels);
            const auto loss_weight =
                ResolveLossAggregationWeight(*loss_, batch.size);
            Tensor weighted_loss = loss_weight.device != nullptr
                ? ApplyDeviceScalar(loss_tensor, *loss_weight.device, false)
                : loss_tensor * loss_weight.host;
            AccumulateDeviceScalar(
                device_loss_sum, device_loss_sum_initialized, weighted_loss);
            if (loss_weight.device != nullptr) {
                AccumulateDeviceScalar(
                    device_loss_weight_sum,
                    device_loss_weight_sum_initialized,
                    *loss_weight.device);
            } else {
                loss_weight_sum += loss_weight.host;
            }
        }

        // Compute objective-appropriate metrics.
        if (regression_metrics) {
            AddRegressionMetricScalars(
                regression,
                predictions,
                batch.labels,
                batch.size,
                config_.output_size,
                config_);
        } else {
            const auto accuracy_count = CountClassificationDecisionScalars(
                predictions, batch.labels, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type),
                metric_ignore_index);
            correct += static_cast<int>(accuracy_count.correct);
            total += static_cast<int>(accuracy_count.total);
        }
    }

    float final_loss = 0.0f;
    if (regression_metrics) {
        final_loss = FinalizeAggregatedLoss(
            *loss_, val_loss, loss_weight_sum);
    } else if (device_loss_sum_initialized) {
        final_loss = ReadAccumulatedLoss(
            device_loss_sum,
            *loss_,
            loss_weight_sum,
            device_loss_weight_sum_initialized
                ? &device_loss_weight_sum
                : nullptr,
            "TrainingExecutor::ReadValidationLoss");
    }
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;
    const float final_mae = regression.Mae();
    const float final_rmse = regression.Rmse();

    UpdateMetrics([final_loss, final_acc, final_mae, final_rmse,
                   regression_metrics](TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
        if (regression_metrics) {
            m.val_mae = final_mae;
            m.val_rmse = final_rmse;
        }
        m.has_validation_metrics = true;
    });
}
#endif

Tensor TrainingExecutor::Forward(const Tensor& input) {
    if (!model_) {
        spdlog::error("TrainingExecutor::Forward: Model not initialized");
        return Tensor();
    }

    last_predictions_ = model_->Forward(input);
    return last_predictions_;
}

Tensor TrainingExecutor::ComputeLossTensor(
    const Tensor& predictions,
    const Tensor& targets) {
    if (!loss_) {
        spdlog::error("TrainingExecutor::ComputeLoss: No loss function");
        return Tensor::Zeros({1}, DataType::Float32);
    }

    return loss_->Forward(predictions, targets);
}

float TrainingExecutor::ComputeLoss(
    const Tensor& predictions,
    const Tensor& targets) {
    Tensor loss_tensor = ComputeLossTensor(predictions, targets);
    const ScopedArrayFireHostSyncAttribution loss_sync_attribution(
        ArrayFireHostSyncCategory::LossScalarReadback,
        "TrainingExecutor::ComputeLoss");
    const float* loss_data = loss_tensor.ReadData<float>();
    return loss_data[0];
}

float TrainingExecutor::ComputeAccuracy(const Tensor& predictions, const Tensor& targets) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2) return 0.0f;

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    const auto accuracy_count = CountClassificationDecisionScalars(
        predictions, targets, batch_size, num_classes,
        ClassificationDecisionModeForLoss(config_.loss_type),
        ClassificationMetricIgnoreIndex(config_));
    return accuracy_count.total > 0
        ? static_cast<float>(accuracy_count.correct) /
              static_cast<float>(accuracy_count.total)
        : 0.0f;
}

void TrainingExecutor::Backward(const Tensor& predictions, const Tensor& targets) {
    if (!model_) {
        spdlog::error("TrainingExecutor::Backward: Model not initialized");
        return;
    }

    if (!loss_) {
        spdlog::error("TrainingExecutor::Backward: No loss function");
        return;
    }

    // Compute loss gradient
    Tensor grad = loss_->Backward(predictions, targets);

    // ArrayFire can flatten trailing-1 dimensions (e.g. [16,1] -> [16]).
    // Restore the semantic model shape without materializing on the host.
    if (grad.Shape() != predictions.Shape() &&
        grad.NumElements() == predictions.NumElements()) {
        grad = grad.Reshape(predictions.Shape());
    }

    // Backward through model
    model_->Backward(grad);
}

bool TrainingExecutor::AccumulateGradientsAndMaybeStep(
    int epoch,
    int batch_num,
    int total_batches,
    float batch_loss,
    float current_acc,
    float gradient_weight,
    const Tensor* device_gradient_weight,
    bool force_step) {
    if (!model_ || !optimizer_) {
        return false;
    }

    const auto grads = model_->GetGradients();
    if (grads.empty()) {
        return false;
    }

    if (device_gradient_weight == nullptr &&
        (!std::isfinite(gradient_weight) || gradient_weight < 0.0f)) {
        throw std::runtime_error(
            "TrainingExecutor: gradient accumulation weight must be finite and non-negative");
    }
    const bool mean_reduction = loss_->GetReduction() == Reduction::Mean;
    if (mean_reduction && gradient_accumulated_batches_ > 0 &&
        gradient_accumulation_device_weight_initialized_ !=
            (device_gradient_weight != nullptr)) {
        throw std::runtime_error(
            "TrainingExecutor: gradient accumulation cannot mix host and device mean denominators");
    }
    for (const auto& [name, grad] : grads) {
        if (grad.GetDataType() != DataType::Float32) {
            throw std::runtime_error(
                "TrainingExecutor: gradient accumulation requires Float32 gradients for '" +
                name + "'");
        }

        Tensor weighted_grad = grad.Clone();
        if (mean_reduction) {
            weighted_grad = device_gradient_weight != nullptr
                ? ApplyDeviceScalar(grad, *device_gradient_weight, false)
                : grad * gradient_weight;
        }
        auto found = gradient_accumulator_.find(name);
        if (found == gradient_accumulator_.end()) {
            gradient_accumulator_[name] = std::move(weighted_grad);
            continue;
        }

        Tensor& accumulated = found->second;
        if (accumulated.Shape() != grad.Shape()) {
            throw std::runtime_error(
                "TrainingExecutor: gradient accumulation shape mismatch for '" +
                name + "'");
        }

        accumulated = accumulated + weighted_grad;
    }

    ++gradient_accumulated_batches_;
    if (mean_reduction) {
        if (device_gradient_weight != nullptr) {
            AccumulateDeviceScalar(
                gradient_accumulation_device_weight_,
                gradient_accumulation_device_weight_initialized_,
                *device_gradient_weight);
        } else {
            gradient_accumulation_weight_ += gradient_weight;
        }
    }
    const int grad_accum_steps =
        training_contract::ClampGradientAccumulationSteps(
            config_.grad_accum_steps);
    if (!force_step && gradient_accumulated_batches_ < grad_accum_steps) {
        return false;
    }

    std::map<std::string, Tensor> averaged_grads;
    const float host_scale = mean_reduction &&
                             !gradient_accumulation_device_weight_initialized_
        ? (gradient_accumulation_weight_ > 0.0f
               ? 1.0f / gradient_accumulation_weight_
               : 0.0f)
        : 1.0f;
    for (const auto& [name, accumulated] : gradient_accumulator_) {
        if (mean_reduction &&
            gradient_accumulation_device_weight_initialized_) {
            Tensor safe_weight = gradient_accumulation_device_weight_.Clip(
                std::numeric_limits<float>::min(),
                std::numeric_limits<float>::max());
            averaged_grads[name] = ApplyDeviceScalar(
                accumulated, safe_weight, true);
        } else {
            averaged_grads[name] = accumulated * host_scale;
        }
    }

    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::UpdateParameters, epoch, batch_num,
        total_batches, batch_loss, current_acc);

    const auto optimizer_start = std::chrono::steady_clock::now();
    auto params = model_->GetParameters();
    optimizer_->Step(params, averaged_grads);
    model_->SetParameters(params);
    const auto optimizer_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - optimizer_start).count();

    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::UpdateParameters, epoch, batch_num,
        total_batches, batch_loss, current_acc, optimizer_ms);

    UpdateMetrics([](TrainingMetrics& m) {
        ++m.optimizer_step_count;
    });
    if (scheduler_controller_) {
        ApplySchedulerAdvance(
            scheduler_controller_->OnOptimizerStep());
    }

    gradient_accumulator_.clear();
    gradient_accumulated_batches_ = 0;
    gradient_accumulation_weight_ = 0.0f;
    gradient_accumulation_device_weight_ = Tensor();
    gradient_accumulation_device_weight_initialized_ = false;
    return true;
}

void TrainingExecutor::ApplySchedulerAdvance(
    const TrainingSchedulerAdvanceResult& result) {
    if (!result.ok) {
        throw std::runtime_error(
            "TrainingExecutor: scheduler lifecycle failed: " +
            result.error);
    }
    if (!result.stepped) {
        return;
    }

    UpdateMetrics([&](TrainingMetrics& metrics) {
        ++metrics.scheduler_step_count;
        metrics.learning_rate = result.learning_rate;
        metrics.learning_rate_history.push_back(result.learning_rate);
    });

    const auto* scheduler = scheduler_controller_
        ? scheduler_controller_->GetScheduler()
        : nullptr;
    const std::string scheduler_name = scheduler
        ? scheduler->GetName()
        : "unknown";
    const auto cadence = scheduler_controller_
        ? scheduler_controller_->GetCadence()
        : TrainingSchedulerCadence::CompletedEpoch;
    TrainingTraceCollector::Instance().RecordRuntimeEvent(
        "TrainingScheduler.Advance",
        fmt::format(
            "scheduler={} cadence={} completed_epochs={} "
            "completed_optimizer_steps={} learning_rate={:.9g}",
            scheduler_name,
            TrainingSchedulerCadenceName(cadence),
            result.completed_epochs,
            result.completed_optimizer_steps,
            result.learning_rate));
}

void TrainingExecutor::Stop() {
    stop_requested_.store(true);
    is_paused_.store(false);  // Unpause so thread can exit
}

void TrainingExecutor::Pause() {
    is_paused_.store(true);
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_paused = true;
        m.status_message = "Training paused";
    });
}

void TrainingExecutor::Resume() {
    is_paused_.store(false);
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_paused = false;
        m.status_message = "Training resumed";
    });
}

TrainingMetrics TrainingExecutor::GetMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void TrainingExecutor::UpdateMetrics(const std::function<void(TrainingMetrics&)>& updater) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    updater(metrics_);
}

bool TrainingExecutor::WaitWhilePaused() {
    while (is_paused_.load() && !stop_requested_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return !stop_requested_.load();
}

void TrainingExecutor::PreprocessBatch(Batch& /*batch*/) {
    // Preprocessing is handled by DatasetBatcher
}

namespace {

int64_t SequenceTargetIdAt(const Tensor& targets, size_t offset) {
    if (targets.GetDataType() == DataType::Int64) {
        return targets.ReadData<int64_t>()[offset];
    }
    if (targets.GetDataType() == DataType::Int32) {
        return static_cast<int64_t>(targets.ReadData<int32_t>()[offset]);
    }
    throw std::runtime_error("language-model target_ids must be Int64 or Int32");
}

struct NextTokenAccuracyCount {
    size_t correct = 0;
    size_t valid = 0;
};

NextTokenAccuracyCount CountNextTokenAccuracyFromLogits(
    const Tensor& logits,
    const Tensor& target_ids,
    int64_t ignore_index) {

    const auto& logit_shape = logits.Shape();
    const auto& target_shape = target_ids.Shape();
    if (logits.GetDataType() != DataType::Float32 ||
        logit_shape.size() != 3 ||
        target_shape.size() != 2 ||
        target_shape[0] != logit_shape[0] ||
        target_shape[1] != logit_shape[1]) {
        throw std::runtime_error(
            "language-model accuracy expects logits [batch, seq, vocab] "
            "and target_ids [batch, seq]");
    }

    const size_t batch_size = logit_shape[0];
    const size_t sequence_length = logit_shape[1];
    const size_t vocab_size = logit_shape[2];
    const float* data = logits.ReadData<float>();

    NextTokenAccuracyCount result;
    for (size_t row = 0; row < batch_size; ++row) {
        for (size_t col = 0; col < sequence_length; ++col) {
            const size_t target_offset = row * sequence_length + col;
            const int64_t target = SequenceTargetIdAt(target_ids, target_offset);
            if (target == ignore_index) {
                continue;
            }
            if (target < 0 || static_cast<size_t>(target) >= vocab_size) {
                throw std::runtime_error(
                    "language-model target id is outside the vocabulary range");
            }

            const size_t logit_offset = target_offset * vocab_size;
            size_t predicted = 0;
            float best = data[logit_offset];
            for (size_t vocab = 1; vocab < vocab_size; ++vocab) {
                const float value = data[logit_offset + vocab];
                if (value > best) {
                    best = value;
                    predicted = vocab;
                }
            }
            if (predicted == static_cast<size_t>(target)) {
                ++result.correct;
            }
            ++result.valid;
        }
    }

    return result;
}

} // namespace

void TrainingExecutor::RunTrainingEpochSequence(
    ISequenceBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    float epoch_loss = 0.0f;
    float loss_weight_sum = 0.0f;
    int batch_num = 0;
    size_t sample_count = 0;
    SequenceTagMetrics aggregate_metrics;

    const size_t total_batches = batcher.GetNumBatches();
    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    const auto epoch_start_time = std::chrono::steady_clock::now();
    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        if (!WaitWhilePaused()) break;

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        const auto fetch_start = std::chrono::steady_clock::now();
        SequenceBatch batch = batcher.GetNextSequenceBatch();
        const auto fetch_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - fetch_start).count();
        if (!batch.IsValid()) break;
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches), 0.0f, 0.0f, fetch_ms);
        if (!batch.IsSupervised()) {
            throw std::runtime_error(
                "TrainingExecutor: sequence batch is missing tag_ids or target_ids");
        }
        const bool is_language_modeling = batch.HasTargetIds();
        const Tensor& targets =
            is_language_modeling ? batch.target_ids : batch.tag_ids;
        const int64_t ignore_index = is_language_modeling
            ? config_.sequence_batch.target_ignore_index
            : config_.sequence_batch.ignore_index;

        ++batch_num;
        sample_count += batch.size;

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches));
        const auto forward_start = std::chrono::steady_clock::now();
        Tensor model_input = BuildSequenceModelInput(batch, config_);
        Tensor predictions = Forward(model_input);
        const auto forward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - forward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches), 0.0f, 0.0f, forward_ms);

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches));
        const float batch_loss = ComputeLoss(predictions, targets);
        const std::string loss_status =
            std::isfinite(batch_loss) ? "ok" : "failed";
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, 0.0f,
            loss_status,
            std::isfinite(batch_loss) ? "" :
                "Sequence training loss became NaN or Inf.");
        if (!std::isfinite(batch_loss)) {
            throw std::runtime_error(
                "TrainingExecutor: sequence training loss is not finite");
        }
        size_t batch_loss_observations = 0;
        if (is_language_modeling) {
            const auto accuracy_count = CountNextTokenAccuracyFromLogits(
                predictions, targets, ignore_index);
            batch_loss_observations = accuracy_count.valid;
            aggregate_metrics.total_tokens += accuracy_count.valid;
            aggregate_metrics.correct_tokens += accuracy_count.correct;
            aggregate_metrics.token_accuracy =
                aggregate_metrics.total_tokens == 0 ? 0.0 :
                    static_cast<double>(aggregate_metrics.correct_tokens) /
                    static_cast<double>(aggregate_metrics.total_tokens);
        } else {
            const auto batch_metrics = ComputeSequenceTagMetricsFromLogits(
                predictions, targets, sequence_id_to_label_, ignore_index);
            batch_loss_observations = batch_metrics.total_tokens;
            AccumulateSequenceTagMetrics(aggregate_metrics, batch_metrics);
            FinalizeSequenceTagMetricRates(aggregate_metrics);
        }
        const float loss_weight = LossAggregationWeight(
            *loss_, batch_loss_observations);
        epoch_loss += batch_loss * loss_weight;
        loss_weight_sum += loss_weight;

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        const auto backward_start = std::chrono::steady_clock::now();
        Backward(predictions, targets);
        const auto backward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - backward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, backward_ms);

        const float current_loss = FinalizeAggregatedLoss(
            *loss_, epoch_loss, loss_weight_sum);
        const float current_acc =
            static_cast<float>(aggregate_metrics.token_accuracy);
        const float current_f1 =
            static_cast<float>(aggregate_metrics.entity_f1);

        AccumulateGradientsAndMaybeStep(
            epoch, batch_num, static_cast<int>(total_batches),
            batch_loss, current_acc, loss_weight, nullptr,
            batcher.IsEpochComplete());

        UpdateMetrics([batch_num,
                       current_loss,
                       current_acc,
                       current_f1,
                       token_count = aggregate_metrics.total_tokens]
                      (TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
            m.train_token_accuracy = current_acc;
            m.train_entity_f1 = current_f1;
            m.train_token_count = token_count;
        });

        if (ShouldLogTrainingBatch(config_, batch_num)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            spdlog::info("Epoch {} [{}/{}] seq_loss={:.4f} "
                         "token_acc={:.2f}% entity_f1={:.2f}% "
                         "({:.1f}s, {:.1f} batches/s)",
                         epoch, batch_num, total_batches, current_loss,
                         current_acc * 100.0f, current_f1 * 100.0f,
                         elapsed_s, rate);
        }

        if (batch_cb) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            TrainingTraceCollector::Instance().RecordStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            batch_cb(epoch, batch_num, static_cast<int>(total_batches),
                     batch_loss, current_acc);
        }
    }

    FinalizeSequenceTagMetricRates(aggregate_metrics);
    const float final_loss = FinalizeAggregatedLoss(
        *loss_, epoch_loss, loss_weight_sum);
    const float final_acc =
        static_cast<float>(aggregate_metrics.token_accuracy);
    const float final_f1 =
        static_cast<float>(aggregate_metrics.entity_f1);

    UpdateMetrics([final_loss,
                   final_acc,
                   final_f1,
                   token_count = aggregate_metrics.total_tokens]
                  (TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
        m.train_token_accuracy = final_acc;
        m.train_entity_f1 = final_f1;
        m.train_token_count = token_count;
    });
    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);

    spdlog::debug("TrainingExecutor: sequence epoch {} consumed {} samples",
                  epoch, sample_count);
}

void TrainingExecutor::RunValidationSequence(ISequenceBatcher& batcher) {
    const auto evaluation = EvaluateSequenceBatcher(batcher);
    UpdateMetrics([evaluation](TrainingMetrics& m) {
        m.val_loss = evaluation.loss;
        m.val_accuracy = evaluation.accuracy;
        m.has_validation_metrics = true;
        m.val_token_accuracy = evaluation.accuracy;
        m.val_entity_f1 = evaluation.entity_f1;
        m.val_token_count = evaluation.token_count;
    });
}

TrainingExecutor::SequenceEvaluationMetrics
TrainingExecutor::EvaluateSequenceBatcher(ISequenceBatcher& batcher) {
    float evaluation_loss = 0.0f;
    float loss_weight_sum = 0.0f;
    int batch_num = 0;
    SequenceTagMetrics aggregate_metrics;

    batcher.Reset();
    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        SequenceBatch batch = batcher.GetNextSequenceBatch();
        if (!batch.IsValid()) break;
        if (!batch.IsSupervised()) {
            throw std::runtime_error(
                "TrainingExecutor: sequence validation batch is missing tag_ids or target_ids");
        }
        const bool is_language_modeling = batch.HasTargetIds();
        const Tensor& targets =
            is_language_modeling ? batch.target_ids : batch.tag_ids;
        const int64_t ignore_index = is_language_modeling
            ? config_.sequence_batch.target_ignore_index
            : config_.sequence_batch.ignore_index;

        ++batch_num;
        Tensor model_input = BuildSequenceModelInput(batch, config_);
        Tensor predictions = Forward(model_input);
        const float batch_loss = ComputeLoss(predictions, targets);
        if (!std::isfinite(batch_loss)) {
            throw std::runtime_error(
                "TrainingExecutor: sequence evaluation loss is not finite");
        }
        size_t batch_loss_observations = 0;
        if (is_language_modeling) {
            const auto accuracy_count = CountNextTokenAccuracyFromLogits(
                predictions, targets, ignore_index);
            batch_loss_observations = accuracy_count.valid;
            aggregate_metrics.total_tokens += accuracy_count.valid;
            aggregate_metrics.correct_tokens += accuracy_count.correct;
            aggregate_metrics.token_accuracy =
                aggregate_metrics.total_tokens == 0 ? 0.0 :
                    static_cast<double>(aggregate_metrics.correct_tokens) /
                    static_cast<double>(aggregate_metrics.total_tokens);
        } else {
            const auto batch_metrics = ComputeSequenceTagMetricsFromLogits(
                predictions, targets, sequence_id_to_label_, ignore_index);
            batch_loss_observations = batch_metrics.total_tokens;
            AccumulateSequenceTagMetrics(aggregate_metrics, batch_metrics);
        }
        const float loss_weight = LossAggregationWeight(
            *loss_, batch_loss_observations);
        evaluation_loss += batch_loss * loss_weight;
        loss_weight_sum += loss_weight;
    }

    FinalizeSequenceTagMetricRates(aggregate_metrics);
    SequenceEvaluationMetrics evaluation;
    evaluation.loss = FinalizeAggregatedLoss(
        *loss_, evaluation_loss, loss_weight_sum);
    evaluation.accuracy =
        static_cast<float>(aggregate_metrics.token_accuracy);
    evaluation.entity_f1 =
        static_cast<float>(aggregate_metrics.entity_f1);
    evaluation.token_count = aggregate_metrics.total_tokens;
    return evaluation;
}

// =============================================================================
// Arrow-specific training methods
// =============================================================================

void TrainingExecutor::RunTrainingEpochArrow(
    IBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    spdlog::debug("RunTrainingEpochArrow: Entered, epoch={}", epoch);

    float epoch_loss = 0.0f;
    float loss_weight_sum = 0.0f;
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
    Tensor device_loss_sum;
    bool device_loss_sum_initialized = false;
    Tensor device_loss_weight_sum;
    bool device_loss_weight_sum_initialized = false;
    Tensor device_accuracy_counts;
    bool device_accuracy_counts_initialized = false;
    const auto metric_ignore_index = ClassificationMetricIgnoreIndex(config_);
    int batch_num = 0;
    float current_loss = 0.0f;
    float current_acc = 0.0f;
    double fetch_total_ms = 0.0;
    double fetch_max_ms = 0.0;

    spdlog::debug("RunTrainingEpochArrow: Getting num batches");
    size_t total_batches = batcher.GetNumBatches();
    spdlog::debug("RunTrainingEpochArrow: total_batches={}", total_batches);

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    // Epoch wall-clock start for the periodic progress log below.
    // Without per-batch logging the training run goes completely
    // silent between the first-batch debug dump and the epoch-end
    // summary, which can be 100+ seconds on a real dataset - making
    // it impossible to tell "alive but slow" from "hung". This is
    // the fix for the tofix.md entry "Training logs silent mid-epoch".
    const auto epoch_start_time = std::chrono::steady_clock::now();

    spdlog::debug("RunTrainingEpochArrow: Entering batch loop");
    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        if (!WaitWhilePaused()) break;

        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches));
        const auto fetch_start = std::chrono::steady_clock::now();
        Batch batch = batcher.GetNextBatch();
        const double fetch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fetch_start).count();
        if (!batch.IsValid()) break;

        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::GetNextBatch, epoch, batch_num + 1,
            static_cast<int>(total_batches), 0.0f, 0.0f,
            static_cast<float>(fetch_ms));

        batch_num++;
        fetch_total_ms += fetch_ms;
        fetch_max_ms = std::max(fetch_max_ms, fetch_ms);

        // Forward pass through model
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches));
        const auto forward_start = std::chrono::steady_clock::now();
        Tensor predictions = Forward(batch.data);
        const auto forward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - forward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Forward, epoch, batch_num,
            static_cast<int>(total_batches), 0.0f, 0.0f, forward_ms);

        // DEBUG: Log sample values for first batch of first epoch.
        if (epoch == 1 && batch_num == 1 &&
            !ShouldMaterializeFirstBatchDebugSamples(config_)) {
            RecordSkippedFirstBatchDebugSampleDump();
        } else if (epoch == 1 && batch_num == 1) {
            const ScopedArrayFireHostSyncAttribution debug_sync_attribution(
                ArrayFireHostSyncCategory::DebugSampleDump,
                "TrainingExecutor::FirstArrowBatchDebugSampleDump");
            const float* input_data = batch.data.ReadData<float>();
            float min_input = input_data[0], max_input = input_data[0];
            const auto& input_shape = batch.data.Shape();
            if (input_shape.size() >= 2) {
                size_t input_size = input_shape[0] * input_shape[1];
                for (size_t i = 1; i < std::min(input_size, size_t(1000)); ++i) {
                    min_input = std::min(min_input, input_data[i]);
                    max_input = std::max(max_input, input_data[i]);
                }
                spdlog::info("DEBUG Arrow: Input data range: [{:.4f}, {:.4f}]", min_input, max_input);
            }

            // Debug labels
            const auto& label_shape = batch.labels.Shape();
            std::string shape_str;
            for (auto d : label_shape) shape_str += std::to_string(d) + " ";
            spdlog::info("DEBUG Arrow: Label shape: [{}], output_size={}", shape_str, config_.output_size);

            size_t label_count = 0;
            if (!label_shape.empty()) {
                label_count = 1;
                for (size_t dim : label_shape) {
                    label_count *= dim;
                }
            }
            if (label_count > 0) {
                const size_t sample_count = std::min<size_t>(label_count, 3);
                std::string label_str = "  [";
                if (batch.labels.GetDataType() == DataType::Int32) {
                    const int32_t* label_data =
                        batch.labels.ReadData<int32_t>();
                    for (size_t i = 0; i < sample_count; ++i) {
                        label_str += fmt::format("{}", label_data[i]);
                        if (i + 1 < sample_count) label_str += ", ";
                    }
                } else {
                    const float* label_data =
                        batch.labels.ReadData<float>();
                    for (size_t i = 0; i < sample_count; ++i) {
                        label_str += fmt::format("{:.1f}", label_data[i]);
                        if (i + 1 < sample_count) label_str += ", ";
                    }
                }
                label_str += "]";
                spdlog::info("DEBUG Arrow: First label values: {}", label_str);
            } else {
                spdlog::error("DEBUG Arrow: Labels tensor is empty or invalid!");
            }

            // Debug predictions
            const auto& pred_shape = predictions.Shape();
            std::string pred_shape_str;
            for (auto d : pred_shape) pred_shape_str += std::to_string(d) + " ";
            spdlog::info("DEBUG Arrow: Predictions shape: [{}]", pred_shape_str);
        }

        // Compute loss
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches));
        float batch_loss = current_loss;
        std::string loss_status = "device_resident";
        LossAggregationWeightValue loss_weight;
        if (regression_metrics) {
            batch_loss = ComputeLoss(predictions, batch.labels);
            loss_weight = ResolveLossAggregationWeight(*loss_, batch.size);
            loss_weight_sum += loss_weight.host;
            epoch_loss += batch_loss * loss_weight.host;
            loss_status = std::isfinite(batch_loss) ? "ok" : "failed";
        } else {
            Tensor loss_tensor = ComputeLossTensor(predictions, batch.labels);
            loss_weight = ResolveLossAggregationWeight(*loss_, batch.size);
            Tensor weighted_loss = loss_weight.device != nullptr
                ? ApplyDeviceScalar(loss_tensor, *loss_weight.device, false)
                : loss_tensor * loss_weight.host;
            AccumulateDeviceScalar(
                device_loss_sum,
                device_loss_sum_initialized,
                weighted_loss);
            if (loss_weight.device != nullptr) {
                AccumulateDeviceScalar(
                    device_loss_weight_sum,
                    device_loss_weight_sum_initialized,
                    *loss_weight.device);
            } else {
                loss_weight_sum += loss_weight.host;
            }
        }
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::ComputeLoss, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, 0.0f,
            loss_status,
            loss_status == "failed" ? "Training loss became NaN or Inf." : "");

        // Compute accuracy
        if (regression_metrics) {
            AddRegressionMetricScalars(
                regression,
                predictions,
                batch.labels,
                batch.size,
                config_.output_size,
                config_);
        } else {
            const auto accuracy_scalar = BuildClassificationDecisionScalar(
                predictions, batch.labels, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type),
                metric_ignore_index);
            AccumulateDeviceClassificationCounts(
                device_accuracy_counts,
                device_accuracy_counts_initialized,
                accuracy_scalar.counts);
        }

        // Backward pass
        CrashRunRecorder::Instance().MarkStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss);
        const auto backward_start = std::chrono::steady_clock::now();
        Backward(predictions, batch.labels);
        const auto backward_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - backward_start).count();
        TrainingTraceCollector::Instance().RecordStage(
            TrainingTraceStage::Backward, epoch, batch_num,
            static_cast<int>(total_batches), batch_loss, 0.0f, backward_ms);

        const bool should_report = ShouldReportTrainingBatch(
            config_, batch_num, static_cast<int>(total_batches),
            batcher.IsEpochComplete());
        if (!regression_metrics && should_report) {
            current_loss = ReadAccumulatedLoss(
                device_loss_sum, *loss_, loss_weight_sum,
                device_loss_weight_sum_initialized
                    ? &device_loss_weight_sum
                    : nullptr);
            const auto accuracy_count = ReadClassificationDecisionScalar(
                ClassificationDecisionScalar{device_accuracy_counts},
                "TrainingExecutor::ReadAccumulatedAccuracy");
            current_acc = accuracy_count.total > 0
                ? static_cast<float>(accuracy_count.correct) /
                      static_cast<float>(accuracy_count.total)
                : 0.0f;
            batch_loss = current_loss;
        } else if (regression_metrics) {
            current_loss = FinalizeAggregatedLoss(
                *loss_, epoch_loss, loss_weight_sum);
        }

        // Update metrics
        const float current_mae = regression.Mae();
        const float current_rmse = regression.Rmse();

        AccumulateGradientsAndMaybeStep(
            epoch, batch_num, static_cast<int>(total_batches),
            batch_loss, current_acc, loss_weight.host, loss_weight.device,
            batcher.IsEpochComplete());

        UpdateMetrics([batch_num, current_loss, current_acc,
                       current_mae, current_rmse](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
            m.train_mae = current_mae;
            m.train_rmse = current_rmse;
        });

        // Periodic progress log so the user knows training is alive.
        // Fires on batch 1 (so they see immediate feedback that the
        // loop entered) and every 50 batches after. Throughput is
        // computed against epoch_start_time so the "batches/s" reading
        // naturally warms up as the batcher + GPU pools stabilize.
        if (ShouldLogTrainingBatch(config_, batch_num)) {
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - epoch_start_time).count();
            const float elapsed_s = elapsed_ms / 1000.0f;
            const float rate = elapsed_ms > 0
                ? (batch_num * 1000.0f / static_cast<float>(elapsed_ms))
                : 0.0f;
            if (regression_metrics) {
                spdlog::info(
                    "Epoch {} [{}/{}] loss={:.4f} mae={:.4f} rmse={:.4f} "
                    "({:.1f}s, {:.1f} batches/s)",
                    epoch, batch_num, total_batches,
                    current_loss, current_mae, current_rmse,
                    elapsed_s, rate);
            } else {
                spdlog::info("Epoch {} [{}/{}] loss={:.4f} acc={:.2f}% "
                             "({:.1f}s, {:.1f} batches/s)",
                             epoch, batch_num, total_batches,
                             current_loss, current_acc * 100.0f,
                             elapsed_s, rate);
            }
        }

        // Batch callback
        if (batch_cb) {
            CrashRunRecorder::Instance().MarkStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            TrainingTraceCollector::Instance().RecordStage(
                TrainingTraceStage::BatchCallback, epoch, batch_num,
                static_cast<int>(total_batches), batch_loss, current_acc);
            batch_cb(epoch, batch_num, static_cast<int>(total_batches), current_loss, current_acc);
        }
    }

    // Final epoch metrics
    float final_loss = current_loss;
    float final_acc = current_acc;
    const float final_mae = regression.Mae();
    const float final_rmse = regression.Rmse();

    UpdateMetrics([final_loss, final_acc, final_mae, final_rmse](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
        m.train_mae = final_mae;
        m.train_rmse = final_rmse;
    });
    if (batch_num > 0) {
        spdlog::info(
            "Arrow loader timing epoch {}: batches={}, avg_fetch={:.2f} ms, max_fetch={:.2f} ms",
            epoch,
            batch_num,
            fetch_total_ms / static_cast<double>(batch_num),
            fetch_max_ms);
    }
    CrashRunRecorder::Instance().MarkStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
    TrainingTraceCollector::Instance().RecordStage(
        TrainingTraceStage::EpochComplete, epoch, batch_num,
        static_cast<int>(total_batches), final_loss, final_acc);
}

ObjectiveEvaluationMetrics TrainingExecutor::EvaluateArrowBatcher(
    IBatcher& batcher) {
    float val_loss = 0.0f;
    float loss_weight_sum = 0.0f;
    Tensor device_loss_sum;
    bool device_loss_sum_initialized = false;
    Tensor device_loss_weight_sum;
    bool device_loss_weight_sum_initialized = false;
    const bool regression_metrics = UsesRegressionMetrics(config_);
    RegressionMetricAccumulator regression(
        &config_.regression_target_transform);
    int correct = 0;
    int total = 0;
    const auto metric_ignore_index = ClassificationMetricIgnoreIndex(config_);
    int batch_num = 0;
    double fetch_total_ms = 0.0;
    double fetch_max_ms = 0.0;

    batcher.Reset();

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;

        const auto fetch_start = std::chrono::steady_clock::now();
        Batch batch = batcher.GetNextBatch();
        const double fetch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - fetch_start).count();
        if (!batch.IsValid()) break;

        batch_num++;
        fetch_total_ms += fetch_ms;
        fetch_max_ms = std::max(fetch_max_ms, fetch_ms);

        // Forward pass only (no backprop)
        Tensor predictions = Forward(batch.data);

        // Compute loss
        if (regression_metrics) {
            const float batch_loss = ComputeLoss(predictions, batch.labels);
            const float loss_weight = LossAggregationWeight(*loss_, batch.size);
            val_loss += batch_loss * loss_weight;
            loss_weight_sum += loss_weight;
        } else {
            Tensor loss_tensor = ComputeLossTensor(predictions, batch.labels);
            const auto loss_weight =
                ResolveLossAggregationWeight(*loss_, batch.size);
            Tensor weighted_loss = loss_weight.device != nullptr
                ? ApplyDeviceScalar(loss_tensor, *loss_weight.device, false)
                : loss_tensor * loss_weight.host;
            AccumulateDeviceScalar(
                device_loss_sum, device_loss_sum_initialized, weighted_loss);
            if (loss_weight.device != nullptr) {
                AccumulateDeviceScalar(
                    device_loss_weight_sum,
                    device_loss_weight_sum_initialized,
                    *loss_weight.device);
            } else {
                loss_weight_sum += loss_weight.host;
            }
        }

        // Compute accuracy
        if (regression_metrics) {
            AddRegressionMetricScalars(
                regression,
                predictions,
                batch.labels,
                batch.size,
                config_.output_size,
                config_);
        } else {
            const auto accuracy_count = CountClassificationDecisionScalars(
                predictions, batch.labels, batch.size, config_.output_size,
                ClassificationDecisionModeForLoss(config_.loss_type),
                metric_ignore_index);
            correct += static_cast<int>(accuracy_count.correct);
            total += static_cast<int>(accuracy_count.total);
        }
    }

    float final_loss = 0.0f;
    if (regression_metrics) {
        final_loss = FinalizeAggregatedLoss(
            *loss_, val_loss, loss_weight_sum);
    } else if (device_loss_sum_initialized) {
        final_loss = ReadAccumulatedLoss(
            device_loss_sum,
            *loss_,
            loss_weight_sum,
            device_loss_weight_sum_initialized
                ? &device_loss_weight_sum
                : nullptr,
            "TrainingExecutor::ReadValidationLoss");
    }
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    if (batch_num > 0) {
        spdlog::info(
            "Arrow loader timing validation: batches={}, avg_fetch={:.2f} ms, max_fetch={:.2f} ms",
            batch_num,
            fetch_total_ms / static_cast<double>(batch_num),
            fetch_max_ms);
    }

    return {final_loss, final_acc, regression.Mae(), regression.Rmse()};
}

void TrainingExecutor::RunValidationArrow(IBatcher& batcher) {
    const auto evaluation = EvaluateArrowBatcher(batcher);

    UpdateMetrics([evaluation](TrainingMetrics& m) {
        m.val_loss = evaluation.loss;
        m.val_accuracy = evaluation.accuracy;
        m.val_mae = evaluation.mae;
        m.val_rmse = evaluation.rmse;
        m.has_validation_metrics = true;
    });
}

} // namespace cyxwiz
