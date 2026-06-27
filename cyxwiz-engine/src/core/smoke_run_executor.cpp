#include "smoke_run_executor.h"

#include <cstdint>

#include "data_registry.h"
#include "label_column_resolver.h"
#include "model_builder.h"
#include "pipeline_materializer.h"
#include "text_dataset_batcher.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>

namespace cyxwiz {

namespace {

float ExtractLossScalar(const Tensor& t) {
    if (t.NumElements() == 0 || t.GetDataType() != DataType::Float32) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    return t.Data<float>()[0];
}

bool HasNonFinite(const Tensor& t) {
    if (t.GetDataType() != DataType::Float32) {
        return false;
    }
    const float* p = t.Data<float>();
    for (size_t i = 0; i < t.NumElements(); ++i) {
        if (std::isnan(p[i]) || std::isinf(p[i])) {
            return true;
        }
    }
    return false;
}

float Accuracy(const Tensor& predictions, const Tensor& targets) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2 || shape[0] == 0 || shape[1] == 0) {
        return 0.0f;
    }

    const size_t batch_size = shape[0];
    const size_t num_classes = shape[1];
    const float* pred = predictions.Data<float>();
    const float* target = targets.Data<float>();
    int correct = 0;

    for (size_t b = 0; b < batch_size; ++b) {
        int pred_class = 0;
        int true_class = 0;
        float max_pred = pred[b * num_classes];
        float max_target = target[b * num_classes];
        for (size_t c = 0; c < num_classes; ++c) {
            if (pred[b * num_classes + c] > max_pred) {
                max_pred = pred[b * num_classes + c];
                pred_class = static_cast<int>(c);
            }
            if (target[b * num_classes + c] > max_target) {
                max_target = target[b * num_classes + c];
                true_class = static_cast<int>(c);
            }
        }
        if (pred_class == true_class) {
            correct++;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(batch_size);
}

float L2Norm(const Tensor& t) {
    if (t.GetDataType() != DataType::Float32) {
        return 0.0f;
    }
    const float* p = t.Data<float>();
    double acc = 0.0;
    for (size_t i = 0; i < t.NumElements(); ++i) {
        acc += static_cast<double>(p[i]) * static_cast<double>(p[i]);
    }
    return static_cast<float>(std::sqrt(acc));
}

std::vector<size_t> ShapeOf(const Tensor& t) {
    return t.Shape();
}

int FindFirstModelNode(const std::vector<gui::MLNode>& nodes) {
    for (const auto& node : nodes) {
        switch (node.type) {
            case gui::NodeType::Dense:
            case gui::NodeType::Embedding:
            case gui::NodeType::GRU:
            case gui::NodeType::LSTM:
            case gui::NodeType::RNN:
            case gui::NodeType::Flatten:
            case gui::NodeType::Reshape:
            case gui::NodeType::View:
            case gui::NodeType::Permute:
            case gui::NodeType::Squeeze:
            case gui::NodeType::Unsqueeze:
            case gui::NodeType::TensorAbs:
            case gui::NodeType::TensorExp:
            case gui::NodeType::TensorLog:
            case gui::NodeType::TensorSqrt:
            case gui::NodeType::TensorSign:
            case gui::NodeType::TensorPow:
            case gui::NodeType::TensorClip:
            case gui::NodeType::TensorSum:
            case gui::NodeType::TensorMean:
            case gui::NodeType::TensorMax:
            case gui::NodeType::TensorMin:
            case gui::NodeType::TensorProd:
            case gui::NodeType::TensorVar:
            case gui::NodeType::TensorStd:
            case gui::NodeType::TensorBroadcastTo:
            case gui::NodeType::TensorExpand:
            case gui::NodeType::TensorIndexSelect:
            case gui::NodeType::TensorCompare:
            case gui::NodeType::TensorLogicalMask:
            case gui::NodeType::Dropout:
                return node.id;
            default:
                break;
        }
    }
    return -1;
}

DebugTraceRecord MakeSmokeRecord(const std::string& run_id,
                                 const std::string& phase,
                                 DebugTraceRole role,
                                 int node_id,
                                 const std::string& status) {
    DebugTraceRecord record;
    record.run_id = run_id;
    record.node_id = node_id;
    record.node_name = "SmokeRun";
    record.node_type = "SmokeRun";
    record.phase = phase;
    record.role = role;
    record.status = status;
    return record;
}

} // namespace

SmokeRunResult SmokeRunExecutor::RunTextSmoke(
    TrainingConfiguration config,
    const std::vector<gui::MLNode>& nodes,
    const std::vector<gui::NodeLink>& links,
    const std::string& run_id,
    int max_samples) const {
    SmokeRunResult result;
    result.requested_samples = max_samples;

    if (config.preprocessing_domain != PreprocessingDomain::Text) {
        result.summary = "Smoke Run currently supports text graphs in this slice.";
        return result;
    }
    result.supported = true;

    if (config.sequence_batch.enabled) {
        result.summary = SequenceBatchRuntimeUnsupportedMessage();
        result.issues.push_back({
            IssueLevel::Error, -1, "SequenceBatch", result.summary
        });
        return result;
    }

    auto& registry = DataRegistry::Instance();
    const int batch_size = std::max(1, std::min(config.batch_size, 32));
    std::unique_ptr<IBatcher> batcher;
    std::string batcher_source = "legacy text";

    if (registry.IsArrowDataset(config.dataset_name) && !links.empty()) {
        auto materialized = PipelineMaterializer::Materialize(
            nodes, links, registry, config.dataset_name);
        if (!materialized.success) {
            result.summary = materialized.error_message;
            result.issues.push_back({
                IssueLevel::Error, -1, "PipelineMaterializer", result.summary
            });
            return result;
        }

        if (materialized.operators_applied > 0) {
            auto arrow_dataset =
                registry.GetArrowDataset(materialized.effective_dataset_name);
            if (!arrow_dataset || !arrow_dataset->GetSchema()) {
                result.summary = "Materialized Arrow text table is unavailable: " +
                                 materialized.effective_dataset_name;
                result.issues.push_back({
                    IssueLevel::Error, -1, "PipelineMaterializer", result.summary
                });
                return result;
            }

            const int label_idx =
                FindCommonLabelColumnIndex(arrow_dataset->GetSchema());
            if (label_idx < 0) {
                result.summary = "Materialized Arrow text table has no label column.";
                result.issues.push_back({
                    IssueLevel::Error, -1, "ArrowDataset", result.summary
                });
                return result;
            }

            const std::string label_column =
                arrow_dataset->GetSchema()->field(label_idx)->name();
            batcher = std::make_unique<ArrowDatasetBatcher>(
                arrow_dataset,
                label_column,
                static_cast<size_t>(batch_size),
                /*shuffle=*/false,
                config.train_ratio,
                /*is_training=*/true,
                "",
                0,
                config.num_workers,
                BatcherPhase::Train,
                0.0f,
                static_cast<uint32_t>(config.dataloader_seed));
            batcher_source = "materialized Arrow text";

            if (arrow_dataset->GetNumColumns() > 1) {
                config.input_size =
                    static_cast<size_t>(arrow_dataset->GetNumColumns() - 1);
                config.input_shape = {config.input_size};
            }
        }
    }

    if (!batcher) {
        const auto* entry = registry.GetTextDatasetEntry(config.dataset_name);
        if (!entry) {
            result.summary = "Text dataset is not registered: " + config.dataset_name;
            result.issues.push_back({IssueLevel::Error, -1, "TextDataset", result.summary});
            return result;
        }

        auto text_batcher = std::make_unique<TextDatasetBatcher>(
            *entry,
            config.text_preprocessing,
            batch_size,
            config.train_ratio,
            0.0f,
            0.0f,
            config.shuffle,
            config.num_workers,
            static_cast<uint32_t>(config.dataloader_seed),
            config.stratified,
            static_cast<uint32_t>(std::max(0, config.split_seed)));

        config.input_size = static_cast<size_t>(text_batcher->GetMaxLength());
        config.input_shape = {static_cast<size_t>(text_batcher->GetMaxLength())};
        batcher = std::move(text_batcher);
    }

    if (batcher->GetNumSamples() == 0) {
        result.summary = "Text dataset has no training samples.";
        result.issues.push_back({IssueLevel::Error, -1, "TextDataset", result.summary});
        return result;
    }

    if (config.preprocessing.has_onehot && config.preprocessing.num_classes > 0) {
        batcher->SetOneHotEncoding(config.preprocessing.num_classes);
    } else if (config.output_size > 0) {
        batcher->SetOneHotEncoding(config.output_size);
    }

    BuiltModel built = BuildSequentialFromConfig(config);
    if (!built.ok() || !built.loss || !built.optimizer) {
        result.summary = "Smoke Run failed to build model/loss/optimizer.";
        result.issues.push_back({IssueLevel::Error, -1, "ModelBuilder", result.summary});
        return result;
    }

    const int max_batches = std::max(
        1,
        static_cast<int>((std::min<size_t>(batcher->GetNumSamples(), static_cast<size_t>(max_samples)) +
                          static_cast<size_t>(batch_size) - 1) /
                         static_cast<size_t>(batch_size)));
    const int model_node_id = FindFirstModelNode(nodes);
    float loss_sum = 0.0f;

    for (int batch_index = 1; batch_index <= max_batches && !batcher->IsEpochComplete(); ++batch_index) {
        Batch batch = batcher->GetNextBatch();
        if (!batch.IsValid()) {
            break;
        }

        result.samples_seen += static_cast<int>(batch.size);
        result.batches_seen++;

        auto input_record = MakeSmokeRecord(
            run_id, "SmokeRun.BatchInput", DebugTraceRole::ModelInput,
            model_node_id, "ok");
        input_record.input_shape = ShapeOf(batch.data);
        input_record.output_shape = ShapeOf(batch.labels);
        input_record.payload["batch"] = batch_index;
        input_record.payload["samples"] = batch.size;
        result.traces.push_back(std::move(input_record));

        Tensor predictions = built.model->Forward(batch.data);
        const bool pred_bad = HasNonFinite(predictions);
        if (pred_bad) {
            result.issues.push_back({
                IssueLevel::Error, model_node_id, "SmokeRun",
                "Smoke Run predictions contain NaN or Inf."
            });
        }

        Tensor loss_tensor = built.loss->Forward(predictions, batch.labels);
        const float loss = ExtractLossScalar(loss_tensor);
        const bool loss_ok = std::isfinite(loss);
        if (!loss_ok) {
            result.issues.push_back({
                IssueLevel::Error, model_node_id, "SmokeRun",
                "Smoke Run loss is not finite."
            });
        } else {
            loss_sum += loss;
        }

        result.last_accuracy = Accuracy(predictions, batch.labels);

        auto loss_record = MakeSmokeRecord(
            run_id, "SmokeRun.Loss", DebugTraceRole::Loss,
            model_node_id, loss_ok && !pred_bad ? "ok" : "failed");
        loss_record.input_shape = ShapeOf(predictions);
        loss_record.output_shape = ShapeOf(loss_tensor);
        loss_record.payload["batch"] = batch_index;
        loss_record.payload["loss"] = loss;
        loss_record.payload["accuracy"] = result.last_accuracy;
        loss_record.payload["predictions_have_non_finite"] = pred_bad;
        result.traces.push_back(std::move(loss_record));

        if (!loss_ok || pred_bad) {
            break;
        }

        Tensor grad = built.loss->Backward(predictions, batch.labels);
        built.model->Backward(grad);
        built.model->UpdateParameters(built.optimizer.get());

        size_t grad_count = 0;
        size_t zero_grad_count = 0;
        float max_grad_norm = 0.0f;
        for (const auto& [name, tensor] : built.model->GetGradients()) {
            (void)name;
            const float norm = L2Norm(tensor);
            max_grad_norm = std::max(max_grad_norm, norm);
            grad_count++;
            if (norm == 0.0f) {
                zero_grad_count++;
            }
        }

        auto grad_record = MakeSmokeRecord(
            run_id, "SmokeRun.Backward", DebugTraceRole::Gradient,
            model_node_id, grad_count > 0 ? "ok" : "warning");
        grad_record.payload["batch"] = batch_index;
        grad_record.payload["gradient_tensor_count"] = grad_count;
        grad_record.payload["zero_gradient_tensor_count"] = zero_grad_count;
        grad_record.payload["max_gradient_l2_norm"] = max_grad_norm;
        result.traces.push_back(std::move(grad_record));

        if (grad_count == 0) {
            result.issues.push_back({
                IssueLevel::Warning, model_node_id, "SmokeRun",
                "Smoke Run did not observe parameter gradients."
            });
        }
    }

    result.average_loss = result.batches_seen > 0
        ? loss_sum / static_cast<float>(result.batches_seen)
        : 0.0f;

    const bool has_error = std::any_of(
        result.issues.begin(), result.issues.end(),
        [](const ValidationIssue& issue) { return issue.level == IssueLevel::Error; });
    result.success = result.batches_seen > 0 && !has_error;

    std::ostringstream out;
    out << "Smoke Run: " << (result.success ? "passed" : "failed")
        << ", source=" << batcher_source
        << ", samples=" << result.samples_seen
        << ", batches=" << result.batches_seen
        << ", avg_loss=" << result.average_loss
        << ", last_acc=" << (result.last_accuracy * 100.0f) << "%";
    result.summary = out.str();

    return result;
}

} // namespace cyxwiz
