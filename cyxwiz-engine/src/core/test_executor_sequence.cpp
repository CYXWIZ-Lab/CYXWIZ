#include "test_executor.h"
#include "sequence_arrow_batcher.h"
#include "sequence_model_input.h"
#include "sequence_tag_metrics.h"
#include "execution_device_context.h"
#include <spdlog/spdlog.h>

namespace cyxwiz {

void TestExecutor::TestCausalSequence(int batch_size,
                                    TestBatchCallback batch_cb,
                                    TestCompleteCallback complete_cb) {
    // Public Test owns exception cleanup and the testing flag.
    // Two scopes: the configured test split of the model's prepared dataset, or
    // an entire supplied Test dataset (e.g. frozen token windows). Either way the
    // batcher's vocabulary must equal the model's frozen vocabulary (checked
    // below), so a corpus encoded with a different tokenizer fails closed.
    const bool entire = dataset_scope_ == TestDatasetScope::EntireProvidedDataset;
    if (batch_size <= 0 || !model_ || !use_arrow_dataset_ ||
        !config_.sequence_batch.create_causal_lm_targets) {
        throw std::runtime_error(
            "Sequence Run Test requires an active causal-LM model and an Arrow "
            "sequence dataset (the configured test split, or a supplied Test dataset "
            "encoded with the model's vocabulary).");
    }
    if (config_.sequence_batch.expected_token_vocabulary.empty()) {
        throw std::runtime_error(
            "Sequence Run Test has no frozen vocabulary contract. Train or "
            "load the model with its validated sequence preparation first.");
    }
    // A supplied Test dataset is scored whole: every row through the train
    // phase (whole-dataset semantics), without the training run's roles.
    TrainingConfiguration build_config =
        entire ? ConfigureTestDatasetScope(config_, dataset_scope_) : config_;
    if (entire) {
        build_config.dataset_roles.dev = {};
        build_config.dataset_roles.test = {};
    }
    auto built = BuildSequenceBatcherFromArrowDataset(arrow_dataset_, build_config, batch_size);
    if (!built.success()) throw std::runtime_error(built.error_message);
    if (built.id_to_label != config_.sequence_batch.expected_token_vocabulary ||
        built.token_vocabulary_size != config_.output_size) {
        throw std::runtime_error("Sequence Run Test vocabulary differs from the active model.");
    }
    built.batcher->SetPhase(entire ? BatcherPhase::Train : BatcherPhase::Test);
    built.batcher->Reset();
    if (built.batcher->GetNumBatches() == 0)
        throw std::runtime_error("Sequence Run Test has no test windows.");
    const auto policy = config_.forbid_native_cpu_fallback
        ? ArrayFireFallbackPolicy::ForbidNativeCpuFallback
        : ArrayFireFallbackPolicy::AllowNativeCpuFallback;
    const auto context = CaptureCurrentExecutionDeviceContext(policy);
    if (!context.valid) throw std::runtime_error("Sequence test execution context is invalid.");
    ScopedActiveExecutionDeviceContext active;
    ScopedExecutionDeviceContext binding(context);
    ScopedArrayFireFallbackPolicy fallback(policy);
    if (!Initialize(batch_size)) throw std::runtime_error("Sequence test initialization failed.");
    if (loss_->GetReduction() == Reduction::None)
        throw std::runtime_error("Sequence Run Test requires a scalar loss reduction.");
    model_->SetTraining(false);
    UpdateMetrics([&](TestingMetrics& m) {
        m = TestingMetrics{};
        m.causal_lm_mode = true;
        m.is_testing = true;
        m.total_batches = static_cast<int>(built.batcher->GetNumBatches());
    });
    const auto start = std::chrono::steady_clock::now();
    double loss_sum = 0.0, denominator_sum = 0.0;
    size_t correct = 0, valid = 0;
    int windows = 0, batches = 0;
    while (!built.batcher->IsEpochComplete() && !ShouldStop()) {
        auto batch = built.batcher->GetNextSequenceBatch();
        if (!batch.IsValid() || !batch.HasTargetIds())
            throw std::runtime_error("Sequence test batch is missing generated integer targets.");
        const auto predictions = Forward(BuildSequenceModelInput(batch, config_));
        const auto counts = CountNextTokenAccuracyFromLogits(
            predictions, batch.target_ids, config_.sequence_batch.target_ignore_index);
        if (counts.valid != 0) {
            const ScopedArrayFireHostSyncAttribution attribution(
                ArrayFireHostSyncCategory::LossScalarReadback, "TestExecutor::SequenceLoss");
            const float loss = ComputeLoss(predictions, batch.target_ids);
            if (!std::isfinite(loss)) throw std::runtime_error("Sequence test loss is not finite.");
            double weight = 1.0;
            if (loss_->GetReduction() == Reduction::Mean) {
                weight = static_cast<double>(counts.valid);
                if (const Tensor* denominator = loss_->GetLastMeanReductionDenominator())
                    weight = denominator->ReadData<float>()[0];
            }
            loss_sum += loss * weight;
            denominator_sum += weight;
        }
        correct += counts.correct;
        valid += counts.valid;
        windows += static_cast<int>(batch.word_ids.Shape()[0]);
        ++batches;
        UpdateMetrics([&](TestingMetrics& m) {
            m.current_batch = batches;
            m.total_samples = windows;
            m.total_target_values = valid;
            m.correct_predictions = static_cast<int>(correct);
            m.test_accuracy = valid ? static_cast<float>(double(correct) / valid) : 0.0f;
            m.test_loss = static_cast<float>(loss_->GetReduction() == Reduction::Sum
                ? loss_sum : (denominator_sum > 0 ? loss_sum / denominator_sum : 0.0));
            m.status_message = "Testing next-token predictions";
        });
        if (batch_cb) batch_cb(batches, GetMetrics().total_batches, GetMetrics().test_accuracy);
    }
    if (!ShouldStop() && valid == 0)
        throw std::runtime_error("Sequence Run Test has no valid target tokens.");
    const float seconds = std::chrono::duration<float>(std::chrono::steady_clock::now()-start).count();
    UpdateMetrics([&](TestingMetrics& m) {
        m.is_testing = false;
        m.is_complete = !ShouldStop();
        m.status_message = ShouldStop() ? "Testing cancelled" : "Causal LM testing complete";
        m.total_time_seconds = seconds;
        m.samples_per_second = seconds > 0 ? windows / seconds : 0;
    });
    is_testing_.store(false);
    const auto result = GetMetrics();
    spdlog::info("TestExecutor: causal_lm status='{}' windows={} tokens={} loss={:.6f} token_accuracy={:.6f}",
        result.status_message, windows, valid, result.test_loss, result.test_accuracy);
    if (complete_cb) complete_cb(result);
}

} // namespace cyxwiz
