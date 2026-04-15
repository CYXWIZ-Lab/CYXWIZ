#include "training_executor.h"
#include "data_registry.h"
#include "../preprocessing/preprocessing_config.h"
#include "../preprocessing/statistics_calculator.h"
#include "../plugin/registries/plugin_training_hook_manager.h"
#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>
#include <cmath>
#include <algorithm>

namespace cyxwiz {

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
    , external_batcher_(std::move(external_batcher))
    , mode_(DatasetMode::Image)
{
    spdlog::info("TrainingExecutor: Created with external IBatcher (Image mode), "
                 "{} layers, input_size={}, output_size={}",
                 config_.layers.size(), config_.input_size, config_.output_size);
}

TrainingExecutor::~TrainingExecutor() {
    Stop();
}

bool TrainingExecutor::BuildModelFromConfig() {
    model_ = std::make_unique<SequentialModel>();

    spdlog::info("TrainingExecutor: Building model from {} layer configs", config_.layers.size());

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

            case gui::NodeType::Embedding: {
                // Read num_embeddings (vocab size) and embedding_dim from
                // the generic parameters map. Defaults cover the case
                // where the dialog-created node still has its factory
                // defaults (10000 / 256). Both params come from the
                // node editor's Properties panel.
                size_t num_embeddings = 10000;
                size_t embedding_dim = 256;
                auto ne_it = layer_cfg.parameters.find("num_embeddings");
                if (ne_it != layer_cfg.parameters.end()) {
                    try { num_embeddings = static_cast<size_t>(std::stoi(ne_it->second)); }
                    catch (...) {}
                }
                auto ed_it = layer_cfg.parameters.find("embedding_dim");
                if (ed_it != layer_cfg.parameters.end()) {
                    try { embedding_dim = static_cast<size_t>(std::stoi(ed_it->second)); }
                    catch (...) {}
                }

                // Promote num_embeddings if the dataset vocab is larger.
                // This is the "num_embeddings too small" recovery path.
                // config_.input_size holds seq_len for text; separate
                // vocab info lives on TrainingConfiguration only if the
                // text dispatch stores it there — for now, we trust the
                // node param and warn if it's suspiciously small vs the
                // input token ID range.
                if (num_embeddings < 2) num_embeddings = 2;
                if (embedding_dim < 1) embedding_dim = 1;

                model_->Add<EmbeddingModule>(num_embeddings, embedding_dim);

                // Shape tracking: input is [batch, seq_len] with
                // current_input_size = seq_len. Embedding output is
                // [batch, seq_len, embedding_dim].
                //
                // Lookahead: if the next layer is a recurrent layer
                // (LSTM / GRU / RNN), keep current_input_size =
                // embedding_dim because the recurrent layer's
                // `input_size` is the per-timestep feature count, not
                // the flattened sequence length. Otherwise collapse
                // to seq_len * embedding_dim so the downstream
                // Flatten/Dense head gets the right feature count
                // even if the user didn't drop a Flatten node.
                const size_t seq_len = current_input_size;
                bool next_is_recurrent = false;
                if (i + 1 < config_.layers.size()) {
                    const auto nt = config_.layers[i + 1].type;
                    if (nt == gui::NodeType::LSTM ||
                        nt == gui::NodeType::GRU  ||
                        nt == gui::NodeType::RNN) {
                        next_is_recurrent = true;
                    }
                }
                if (next_is_recurrent) {
                    spdlog::info("  [{}] Embedding({} x {}) — shape "
                                 "[seq_len={}] -> [seq_len={}, embed={}], "
                                 "next layer is recurrent: input_size={} "
                                 "(per-timestep features)",
                                 i, num_embeddings, embedding_dim,
                                 seq_len, seq_len, embedding_dim,
                                 embedding_dim);
                    current_input_size = embedding_dim;
                } else {
                    const size_t new_size = seq_len * embedding_dim;
                    spdlog::info("  [{}] Embedding({} x {}) — shape "
                                 "[seq_len={}] -> [seq_len={}, embed={}], "
                                 "next Flatten/Dense sees {} features",
                                 i, num_embeddings, embedding_dim,
                                 seq_len, seq_len, embedding_dim, new_size);
                    current_input_size = new_size;
                }
                break;
            }

            case gui::NodeType::LSTM: {
                // Read hidden_size / num_layers / bidirectional /
                // return_sequences from the node parameters. Defaults
                // mirror Keras's `LSTM(hidden_size)` shorthand for the
                // common "one recurrent layer feeding a classifier" case.
                size_t hidden_size = 128;
                size_t num_layers  = 1;
                bool bidirectional = false;
                bool return_sequences = false;

                auto hs_it = layer_cfg.parameters.find("hidden_size");
                if (hs_it != layer_cfg.parameters.end()) {
                    try { hidden_size = static_cast<size_t>(std::stoi(hs_it->second)); }
                    catch (...) {}
                }
                auto nl_it = layer_cfg.parameters.find("num_layers");
                if (nl_it != layer_cfg.parameters.end()) {
                    try { num_layers = static_cast<size_t>(std::stoi(nl_it->second)); }
                    catch (...) {}
                }
                auto bi_it = layer_cfg.parameters.find("bidirectional");
                if (bi_it != layer_cfg.parameters.end()) {
                    bidirectional = (bi_it->second == "true" ||
                                     bi_it->second == "1");
                }
                auto rs_it = layer_cfg.parameters.find("return_sequences");
                if (rs_it != layer_cfg.parameters.end()) {
                    return_sequences = (rs_it->second == "true" ||
                                        rs_it->second == "1");
                }
                if (hidden_size < 1) hidden_size = 1;
                if (num_layers < 1) num_layers = 1;

                // LSTM input_size = current_input_size, which should be
                // the per-timestep feature count (e.g., embedding_dim
                // if the previous layer is Embedding with our lookahead
                // fix above). If the previous layer was a non-recurrent
                // one that already flattened, this will be the flattened
                // size and the user will get a shape mismatch at runtime
                // — that's a graph-design error, not a compiler bug.
                model_->Add<LSTMModule>(current_input_size, hidden_size,
                                        num_layers, bidirectional,
                                        return_sequences);

                const size_t output_features = hidden_size *
                                               (bidirectional ? 2 : 1);
                spdlog::info("  [{}] LSTM(in={}, hidden={}, layers={}, "
                             "bidir={}, return_seq={}) — output "
                             "[batch, {}] ({} features)",
                             i, current_input_size, hidden_size,
                             num_layers, bidirectional, return_sequences,
                             output_features, output_features);
                current_input_size = output_features;
                break;
            }

            case gui::NodeType::ReLU: {
                model_->Add<ReLUModule>();
                spdlog::info("  [{}] ReLU", i);
                break;
            }

            case gui::NodeType::Sigmoid: {
                model_->Add<SigmoidModule>();
                spdlog::info("  [{}] Sigmoid", i);
                break;
            }

            case gui::NodeType::Tanh: {
                model_->Add<TanhModule>();
                spdlog::info("  [{}] Tanh", i);
                break;
            }

            case gui::NodeType::LeakyReLU: {
                float slope = layer_cfg.negative_slope > 0 ? layer_cfg.negative_slope : 0.01f;
                model_->Add<LeakyReLUModule>(slope);
                spdlog::info("  [{}] LeakyReLU(slope={})", i, slope);
                break;
            }

            case gui::NodeType::ELU: {
                float alpha = layer_cfg.alpha > 0 ? layer_cfg.alpha : 1.0f;
                model_->Add<ELUModule>(alpha);
                spdlog::info("  [{}] ELU(alpha={})", i, alpha);
                break;
            }

            case gui::NodeType::GELU: {
                model_->Add<GELUModule>();
                spdlog::info("  [{}] GELU", i);
                break;
            }

            case gui::NodeType::Swish: {
                model_->Add<SwishModule>();
                spdlog::info("  [{}] Swish", i);
                break;
            }

            case gui::NodeType::Mish: {
                model_->Add<MishModule>();
                spdlog::info("  [{}] Mish", i);
                break;
            }

            case gui::NodeType::Softmax: {
                model_->Add<SoftmaxModule>();
                spdlog::info("  [{}] Softmax", i);
                break;
            }

            case gui::NodeType::Dropout: {
                float p = layer_cfg.dropout_rate > 0 ? layer_cfg.dropout_rate : 0.5f;
                model_->Add<DropoutModule>(p);
                spdlog::info("  [{}] Dropout(p={})", i, p);
                break;
            }

            case gui::NodeType::Flatten: {
                model_->Add<FlattenModule>(1);
                spdlog::info("  [{}] Flatten", i);
                break;
            }

            case gui::NodeType::BatchNorm: {
                // BatchNorm uses current feature size (output of previous Dense layer)
                float eps = layer_cfg.eps > 0 ? layer_cfg.eps : 1e-5f;
                float momentum = layer_cfg.momentum > 0 ? layer_cfg.momentum : 0.1f;
                model_->Add<BatchNormModule>(current_input_size, eps, momentum);
                spdlog::info("  [{}] BatchNorm({})", i, current_input_size);
                break;
            }

            case gui::NodeType::Output: {
                // Output node is just a marker, not an actual layer
                // The actual output transformation is done by the preceding Dense layer
                spdlog::info("  [{}] Output (marker, no layer added)", i);
                break;
            }

            // Skip non-layer nodes (preprocessing, loss functions, optimizers)
            case gui::NodeType::DatasetInput:
            case gui::NodeType::DataLoader:
            case gui::NodeType::Augmentation:
            case gui::NodeType::DataSplit:
            case gui::NodeType::TensorReshape:
            case gui::NodeType::Normalize:
            case gui::NodeType::OneHotEncode:
            // Loss functions
            case gui::NodeType::MSELoss:
            case gui::NodeType::CrossEntropyLoss:
            case gui::NodeType::BCELoss:
            case gui::NodeType::BCEWithLogits:
            case gui::NodeType::L1Loss:
            case gui::NodeType::SmoothL1Loss:
            case gui::NodeType::HuberLoss:
            case gui::NodeType::NLLLoss:
            // Optimizers
            case gui::NodeType::SGD:
            case gui::NodeType::Adam:
            case gui::NodeType::AdamW:
                // These are not layers in the sequential model
                break;

            // CNN layers (not yet supported in SequentialModel, need CNN module wrappers)
            case gui::NodeType::Conv2D:
            case gui::NodeType::MaxPool2D:
            case gui::NodeType::AvgPool2D:
            case gui::NodeType::GlobalMaxPool:
            case gui::NodeType::GlobalAvgPool:
                spdlog::warn("  [{}] CNN layer {} not yet supported in SequentialModel",
                             i, static_cast<int>(layer_cfg.type));
                break;

            default:
                spdlog::warn("  [{}] Unknown layer type: {}", i, static_cast<int>(layer_cfg.type));
                break;
        }
    }

    if (model_->Size() == 0) {
        spdlog::error("TrainingExecutor: No layers were added to the model!");
        return false;
    }

    // Print model summary
    model_->Summary();

    return true;
}

bool TrainingExecutor::Initialize(int /*batch_size*/) {
    // Build model from configuration
    if (!BuildModelFromConfig()) {
        spdlog::error("TrainingExecutor: Failed to build model from config");
        return false;
    }

    // Create loss function based on config
    switch (config_.loss_type) {
        case gui::NodeType::CrossEntropyLoss:
            loss_ = CreateLoss(LossType::CrossEntropy);
            spdlog::info("TrainingExecutor: Using CrossEntropy loss");
            break;
        case gui::NodeType::MSELoss:
            loss_ = CreateLoss(LossType::MSE);
            spdlog::info("TrainingExecutor: Using MSE loss");
            break;
        case gui::NodeType::BCELoss:
            loss_ = CreateLoss(LossType::BinaryCrossEntropy);
            spdlog::info("TrainingExecutor: Using BCE loss");
            break;
        case gui::NodeType::BCEWithLogits:
            loss_ = CreateLoss(LossType::BCEWithLogits);
            spdlog::info("TrainingExecutor: Using BCEWithLogits loss");
            break;
        case gui::NodeType::L1Loss:
            loss_ = CreateLoss(LossType::L1);
            spdlog::info("TrainingExecutor: Using L1 loss");
            break;
        case gui::NodeType::SmoothL1Loss:
        case gui::NodeType::HuberLoss:
            loss_ = CreateLoss(LossType::SmoothL1);
            spdlog::info("TrainingExecutor: Using SmoothL1/Huber loss");
            break;
        case gui::NodeType::NLLLoss:
            loss_ = CreateLoss(LossType::NLLLoss);
            spdlog::info("TrainingExecutor: Using NLL loss");
            break;
        default:
            loss_ = CreateLoss(LossType::CrossEntropy);
            spdlog::info("TrainingExecutor: Defaulting to CrossEntropy loss");
            break;
    }

    // Create optimizer from backend
    optimizer_ = CreateOptimizer(config_.GetOptimizerType(), config_.learning_rate);

    spdlog::info("TrainingExecutor: Using {} optimizer with lr={}",
                 config_.GetOptimizerName(), config_.learning_rate);

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

    // Initialize
    if (!Initialize(batch_size)) {
        spdlog::error("TrainingExecutor: Failed to initialize");
        is_training_.store(false);
        return;
    }

    // Setup metrics
    UpdateMetrics([epochs](TrainingMetrics& m) {
        m.total_epochs = epochs;
        m.current_epoch = 0;
        m.is_training = true;
        m.is_complete = false;
        m.status_message = "Starting training...";
        m.loss_history.clear();
        m.accuracy_history.clear();
        m.val_loss_history.clear();
        m.val_accuracy_history.clear();
    });

    // Create batchers - Arrow in-memory, Parquet disk-backed, or legacy.
    // All three end up driving the training loop through IBatcher pointers.
    std::unique_ptr<ArrowDatasetBatcher> arrow_train_batcher;
    std::unique_ptr<ArrowDatasetBatcher> arrow_val_batcher;
    std::unique_ptr<ParquetArrowBatcher> parquet_train_batcher;
    std::unique_ptr<ParquetArrowBatcher> parquet_val_batcher;
    std::unique_ptr<DatasetBatcher> legacy_train_batcher;
    std::unique_ptr<DatasetBatcher> legacy_val_batcher;

    // Non-owning IBatcher pointers — point at whichever concrete batcher
    // the mode selected. The Arrow and Parquet paths both flow through the
    // same IBatcher-aware training loops; the legacy path stays on the
    // legacy-specific functions.
    IBatcher* active_train_ibatcher = nullptr;
    IBatcher* active_val_ibatcher = nullptr;

    size_t num_train_samples = 0;

    if (mode_ == DatasetMode::Arrow) {
        // Arrow-based batching (Data Studio)
        spdlog::info("TrainingExecutor: Using Arrow dataset for training "
                     "(batch_size={}, shuffle={}, train_ratio={:.2f})",
                     batch_size, config_.shuffle, config_.train_ratio);

        // Honor DataLoader/DataSplit node config (or defaults if no such nodes)
        // Validation batcher never shuffles regardless of config.
        arrow_train_batcher = std::make_unique<ArrowDatasetBatcher>(
            arrow_dataset_, label_column_, batch_size,
            config_.shuffle, config_.train_ratio, true);
        arrow_val_batcher = std::make_unique<ArrowDatasetBatcher>(
            arrow_dataset_, label_column_, batch_size,
            false, config_.train_ratio, false);

        if (config_.drop_last) {
            spdlog::warn("TrainingExecutor: drop_last=true requested but ArrowDatasetBatcher "
                         "does not yet support it - last partial batch will be kept");
        }
        if (config_.has_data_split && config_.test_ratio > 0.01f) {
            spdlog::warn("TrainingExecutor: test_ratio={:.2f} configured on DataSplit but "
                         "ArrowDatasetBatcher has no held-out test split - the test portion "
                         "will be merged into validation. train={:.2f}, val+test={:.2f}",
                         config_.test_ratio, config_.train_ratio, 1.0f - config_.train_ratio);
        }

        // Apply preprocessing from config
        if (config_.preprocessing.has_normalization) {
            arrow_train_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                   config_.preprocessing.norm_std);
            arrow_val_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                 config_.preprocessing.norm_std);
        }

        // One-hot encoding for classification
        if (config_.preprocessing.has_onehot) {
            arrow_train_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
            arrow_val_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
        } else {
            // Default to output_size for classification
            arrow_train_batcher->SetOneHotEncoding(config_.output_size);
            arrow_val_batcher->SetOneHotEncoding(config_.output_size);
        }

        num_train_samples = arrow_train_batcher->GetNumSamples();
        active_train_ibatcher = arrow_train_batcher.get();
        active_val_ibatcher = arrow_val_batcher.get();
    } else if (mode_ == DatasetMode::Parquet) {
        // Disk-backed Parquet batching — rows are fetched lazily from the
        // memory-mapped Parquet cache one row group at a time. Same output
        // shape as the Arrow path, so training loops don't need to know.
        spdlog::info("TrainingExecutor: Using Parquet-backed dataset for training "
                     "(batch_size={}, shuffle={}, train_ratio={:.2f})",
                     batch_size, config_.shuffle, config_.train_ratio);

        parquet_train_batcher = std::make_unique<ParquetArrowBatcher>(
            parquet_dataset_, label_column_, batch_size,
            config_.shuffle, config_.train_ratio, true);
        parquet_val_batcher = std::make_unique<ParquetArrowBatcher>(
            parquet_dataset_, label_column_, batch_size,
            false, config_.train_ratio, false);

        if (config_.drop_last) {
            spdlog::warn("TrainingExecutor: drop_last=true requested but ParquetArrowBatcher "
                         "does not yet support it - last partial batch will be kept");
        }
        if (config_.has_data_split && config_.test_ratio > 0.01f) {
            spdlog::warn("TrainingExecutor: test_ratio={:.2f} configured on DataSplit but "
                         "ParquetArrowBatcher has no held-out test split - the test portion "
                         "will be merged into validation. train={:.2f}, val+test={:.2f}",
                         config_.test_ratio, config_.train_ratio, 1.0f - config_.train_ratio);
        }

        // Apply preprocessing from config (same as Arrow path).
        if (config_.preprocessing.has_normalization) {
            parquet_train_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                     config_.preprocessing.norm_std);
            parquet_val_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                   config_.preprocessing.norm_std);
        }

        if (config_.preprocessing.has_onehot) {
            parquet_train_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
            parquet_val_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
        } else {
            parquet_train_batcher->SetOneHotEncoding(config_.output_size);
            parquet_val_batcher->SetOneHotEncoding(config_.output_size);
        }

        num_train_samples = parquet_train_batcher->GetNumSamples();
        active_train_ibatcher = parquet_train_batcher.get();
        active_val_ibatcher = parquet_val_batcher.get();
    } else if (mode_ == DatasetMode::Image) {
        // Image mode: the batcher was constructed externally by
        // StartTrainingImage with the correct target size and
        // augmentation pipeline. We just point the training loop at it.
        // Validation uses the same batcher reset for now — Phase 1.4
        // can add a separate validation batcher without augmentation.
        spdlog::info("TrainingExecutor: Using Image dataset for training "
                     "(batch_size={}, {} samples)",
                     batch_size, external_batcher_ ? external_batcher_->GetNumSamples() : 0);

        if (!external_batcher_) {
            spdlog::error("TrainingExecutor: Image mode but no external batcher");
            return;
        }

        num_train_samples = external_batcher_->GetNumSamples();
        active_train_ibatcher = external_batcher_.get();
        active_val_ibatcher = external_batcher_.get();
    } else {
        // Legacy DatasetHandle batching
        spdlog::info("TrainingExecutor: Using legacy dataset for training "
                     "(batch_size={}, shuffle={}, drop_last={})",
                     batch_size, config_.shuffle, config_.drop_last);

        // Honor DataLoader node config (or defaults if no such node).
        // Validation batcher never shuffles and never drops the last batch.
        legacy_train_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Train,
            config_.shuffle, config_.drop_last);
        legacy_val_batcher = std::make_unique<DatasetBatcher>(
            dataset_, batch_size, DatasetSplit::Validation, false, false);

        // Apply NEW preprocessing pipeline (if configured)
        std::string dataset_name = dataset_.GetName();
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
        std::string dataset_name_aug = dataset_.GetName();
        if (registry.HasAugmentationPipeline(dataset_name_aug)) {
            auto aug_pipeline = registry.GetAugmentationPipeline(dataset_name_aug);
            if (aug_pipeline) {
                legacy_train_batcher->SetAugmentationPipeline(aug_pipeline);
                legacy_train_batcher->SetApplyAugmentationOnTrain(true);
            }
        }

        // Apply OLD preprocessing settings
        if (config_.preprocessing.has_normalization) {
            legacy_train_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                    config_.preprocessing.norm_std);
            legacy_val_batcher->SetNormalization(config_.preprocessing.norm_mean,
                                                  config_.preprocessing.norm_std);
        }

        if (config_.preprocessing.has_onehot) {
            legacy_train_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
            legacy_val_batcher->SetOneHotEncoding(config_.preprocessing.num_classes);
        }

        legacy_train_batcher->SetFlatten(true);
        legacy_val_batcher->SetFlatten(true);

        num_train_samples = legacy_train_batcher->GetNumSamples();
    }

    spdlog::info("TrainingExecutor: Starting training for {} epochs, batch_size={}, samples={}",
                 epochs, batch_size, num_train_samples);

    spdlog::debug("TrainingExecutor: Step 1 - Notifying plugin hooks");
    // Notify plugin hooks: training start
    {
        cyxwiz::plugin::TrainingContext ctx;
        ctx.total_epochs = epochs;
        ctx.learning_rate = config_.learning_rate;
        cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyTrainingStart(ctx);
    }

    spdlog::debug("TrainingExecutor: Step 2 - Setting model to training mode");
    // Set model to training mode
    model_->SetTraining(true);

    spdlog::debug("TrainingExecutor: Step 3 - Entering training loop");
    // Training loop
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        spdlog::debug("TrainingExecutor: Epoch {} starting", epoch);
        if (ShouldStop()) break;
        // Check plugin early stopping
        {
            cyxwiz::plugin::TrainingContext stop_ctx;
            stop_ctx.current_epoch = epoch;
            stop_ctx.total_epochs = epochs;
            stop_ctx.learning_rate = config_.learning_rate;
            if (cyxwiz::plugin::PluginTrainingHookManager::Instance().ShouldStopEarly(stop_ctx)) {
                spdlog::info("TrainingExecutor: Plugin requested early stop");
                break;
            }
        }
        WaitWhilePaused();

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
            ctx.learning_rate = config_.learning_rate;
            cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyEpochStart(ctx);
        }

        spdlog::debug("TrainingExecutor: About to call RunTrainingEpoch");
        // Run training epoch - dispatch by dataset mode. Arrow and Parquet
        // batchers both flow through RunTrainingEpochArrow via their shared
        // IBatcher base; legacy DatasetBatcher stays on its own path.
        if (mode_ == DatasetMode::Legacy) {
            RunTrainingEpoch(*legacy_train_batcher, epoch, batch_cb);
        } else if (active_train_ibatcher) {
            RunTrainingEpochArrow(*active_train_ibatcher, epoch, batch_cb);
        }

        if (ShouldStop()) break;

        // Run validation (eval mode).
        //
        // For image/audio batchers, active_train_ibatcher and
        // active_val_ibatcher point to the *same* instance — the batcher
        // holds both train_indices_ and val_indices_ internally and switches
        // between them via SetPhase. Without SetPhase(Val) the val pass
        // would iterate the training indices, producing bogus "perfect val"
        // metrics (this was the source of the suspicious 100% val acc).
        // For Arrow/Parquet/legacy paths these are separate instances so
        // SetPhase is a no-op on the default IBatcher impl.
        model_->SetTraining(false);
        if (mode_ == DatasetMode::Legacy) {
            RunValidation(*legacy_val_batcher);
        } else if (active_val_ibatcher) {
            active_val_ibatcher->SetPhase(BatcherPhase::Val);
            RunValidationArrow(*active_val_ibatcher);
            active_val_ibatcher->SetPhase(BatcherPhase::Train);
        }
        model_->SetTraining(true);

        auto epoch_end = std::chrono::steady_clock::now();
        float epoch_time = std::chrono::duration<float>(epoch_end - epoch_start).count();

        // Get current metrics for callback
        TrainingMetrics current = GetMetrics();

        // Compute samples per second
        float samples_per_sec = static_cast<float>(num_train_samples) / epoch_time;

        // Update history
        UpdateMetrics([&](TrainingMetrics& m) {
            m.epoch_time_seconds = epoch_time;
            m.samples_per_second = samples_per_sec;
            m.loss_history.push_back(m.train_loss);
            m.accuracy_history.push_back(m.train_accuracy);
            m.val_loss_history.push_back(m.val_loss);
            m.val_accuracy_history.push_back(m.val_accuracy);
        });

        // Epoch callback
        if (epoch_cb) {
            epoch_cb(epoch, current.train_loss, current.train_accuracy,
                     current.val_loss, current.val_accuracy, epoch_time);
        }

        spdlog::info("Epoch {}/{}: loss={:.4f}, acc={:.2f}%, val_loss={:.4f}, val_acc={:.2f}% ({:.1f}s, {:.0f} samples/sec)",
                     epoch, epochs, current.train_loss, current.train_accuracy * 100,
                     current.val_loss, current.val_accuracy * 100, epoch_time, samples_per_sec);

        // Notify plugin hooks: epoch end
        {
            cyxwiz::plugin::TrainingContext ctx;
            ctx.current_epoch = epoch;
            ctx.total_epochs = epochs;
            ctx.train_loss = current.train_loss;
            ctx.train_accuracy = current.train_accuracy;
            ctx.val_loss = current.val_loss;
            ctx.val_accuracy = current.val_accuracy;
            ctx.learning_rate = config_.learning_rate;
            cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyEpochEnd(ctx);
        }

        // Reset batchers for next epoch
        if (mode_ == DatasetMode::Legacy) {
            legacy_train_batcher->Reset();
            legacy_val_batcher->Reset();
        } else {
            active_train_ibatcher->Reset();
            active_val_ibatcher->Reset();
        }
    }

    // Notify plugin hooks: training end
    {
        TrainingMetrics final_metrics = GetMetrics();
        cyxwiz::plugin::TrainingContext ctx;
        ctx.current_epoch = final_metrics.current_epoch;
        ctx.total_epochs = final_metrics.total_epochs;
        ctx.train_loss = final_metrics.train_loss;
        ctx.train_accuracy = final_metrics.train_accuracy;
        ctx.val_loss = final_metrics.val_loss;
        ctx.val_accuracy = final_metrics.val_accuracy;
        ctx.learning_rate = config_.learning_rate;
        cyxwiz::plugin::PluginTrainingHookManager::Instance().NotifyTrainingEnd(ctx);
    }

    // Mark complete
    UpdateMetrics([](TrainingMetrics& m) {
        m.is_training = false;
        m.is_complete = true;
        m.status_message = "Training complete";
    });

    is_training_.store(false);

    // Complete callback
    if (complete_cb) {
        complete_cb(GetMetrics());
    }

    spdlog::info("TrainingExecutor: Training complete");
}

void TrainingExecutor::RunTrainingEpoch(
    DatasetBatcher& batcher,
    int epoch,
    BatchCallback batch_cb)
{
    float epoch_loss = 0.0f;
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    size_t total_batches = batcher.GetNumBatches();

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        WaitWhilePaused();

        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass through model
        Tensor predictions = Forward(batch.data);

        // DEBUG: Log sample values for first batch of first epoch
        if (epoch == 1 && batch_num == 1) {
            const float* input_data = batch.data.Data<float>();
            const float* pred_data_debug = predictions.Data<float>();
            const float* target_data_debug = batch.labels.Data<float>();

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
        float batch_loss = ComputeLoss(predictions, batch.labels);
        epoch_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            // Start from c=0 to properly compare all classes including class 0
            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }

        // Backward pass
        Backward(predictions, batch.labels);

        // Update weights using optimizer
        model_->UpdateParameters(optimizer_.get());

        // Update metrics
        float current_loss = epoch_loss / batch_num;
        float current_acc = static_cast<float>(correct) / total;

        UpdateMetrics([batch_num, current_loss, current_acc](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
        });

        // Batch callback
        if (batch_cb) {
            batch_cb(epoch, batch_num, static_cast<int>(total_batches), batch_loss, current_acc);
        }
    }

    // Final epoch metrics
    float final_loss = batch_num > 0 ? epoch_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
    });
}

void TrainingExecutor::RunValidation(DatasetBatcher& batcher) {
    float val_loss = 0.0f;
    int correct = 0;
    int total = 0;
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
        float batch_loss = ComputeLoss(predictions, batch.labels);
        val_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            // Start from c=0 to properly compare all classes including class 0
            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }
    }

    float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
    });
}

Tensor TrainingExecutor::Forward(const Tensor& input) {
    if (!model_) {
        spdlog::error("TrainingExecutor::Forward: Model not initialized");
        return Tensor();
    }

    last_predictions_ = model_->Forward(input);
    return last_predictions_;
}

float TrainingExecutor::ComputeLoss(const Tensor& predictions, const Tensor& targets) {
    if (!loss_) {
        spdlog::error("TrainingExecutor::ComputeLoss: No loss function");
        return 0.0f;
    }

    Tensor loss_tensor = loss_->Forward(predictions, targets);
    const float* loss_data = loss_tensor.Data<float>();
    return loss_data[0];
}

float TrainingExecutor::ComputeAccuracy(const Tensor& predictions, const Tensor& targets) {
    const auto& shape = predictions.Shape();
    if (shape.size() != 2) return 0.0f;

    size_t batch_size = shape[0];
    size_t num_classes = shape[1];

    const float* pred_data = predictions.Data<float>();
    const float* target_data = targets.Data<float>();

    int correct = 0;
    for (size_t b = 0; b < batch_size; ++b) {
        int pred_class = 0, true_class = 0;
        float max_pred = pred_data[b * num_classes];
        float max_target = target_data[b * num_classes];

        // Start from c=0 to properly compare all classes including class 0
        for (size_t c = 0; c < num_classes; ++c) {
            if (pred_data[b * num_classes + c] > max_pred) {
                max_pred = pred_data[b * num_classes + c];
                pred_class = static_cast<int>(c);
            }
            if (target_data[b * num_classes + c] > max_target) {
                max_target = target_data[b * num_classes + c];
                true_class = static_cast<int>(c);
            }
        }
        if (pred_class == true_class) correct++;
    }

    return static_cast<float>(correct) / batch_size;
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

    // Backward through model
    model_->Backward(grad);
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

void TrainingExecutor::WaitWhilePaused() {
    while (is_paused_.load() && !stop_requested_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void TrainingExecutor::PreprocessBatch(Batch& /*batch*/) {
    // Preprocessing is handled by DatasetBatcher
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
    int correct = 0;
    int total = 0;
    int batch_num = 0;

    spdlog::debug("RunTrainingEpochArrow: Getting num batches");
    size_t total_batches = batcher.GetNumBatches();
    spdlog::debug("RunTrainingEpochArrow: total_batches={}", total_batches);

    UpdateMetrics([total_batches](TrainingMetrics& m) {
        m.total_batches = static_cast<int>(total_batches);
        m.current_batch = 0;
    });

    spdlog::debug("RunTrainingEpochArrow: Entering batch loop");
    while (!batcher.IsEpochComplete()) {
        if (ShouldStop()) break;
        WaitWhilePaused();

        Batch batch = batcher.GetNextBatch();
        if (!batch.IsValid()) break;

        batch_num++;

        // Forward pass through model
        Tensor predictions = Forward(batch.data);

        // DEBUG: Log sample values for first batch of first epoch
        if (epoch == 1 && batch_num == 1) {
            const float* input_data = batch.data.Data<float>();
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

            const float* label_data = batch.labels.Data<float>();
            if (label_data && !label_shape.empty()) {
                spdlog::info("DEBUG Arrow: First label values: {:.1f}, {:.1f}, {:.1f}",
                             label_data[0], label_data[1], label_data[2]);
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
        float batch_loss = ComputeLoss(predictions, batch.labels);
        epoch_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }

        // Backward pass
        Backward(predictions, batch.labels);

        // Update weights using optimizer
        model_->UpdateParameters(optimizer_.get());

        // Update metrics
        float current_loss = epoch_loss / batch_num;
        float current_acc = static_cast<float>(correct) / total;

        UpdateMetrics([batch_num, current_loss, current_acc](TrainingMetrics& m) {
            m.current_batch = batch_num;
            m.train_loss = current_loss;
            m.train_accuracy = current_acc;
        });

        // Batch callback
        if (batch_cb) {
            batch_cb(epoch, batch_num, static_cast<int>(total_batches), batch_loss, current_acc);
        }
    }

    // Final epoch metrics
    float final_loss = batch_num > 0 ? epoch_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.train_loss = final_loss;
        m.train_accuracy = final_acc;
    });
}

void TrainingExecutor::RunValidationArrow(IBatcher& batcher) {
    float val_loss = 0.0f;
    int correct = 0;
    int total = 0;
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
        float batch_loss = ComputeLoss(predictions, batch.labels);
        val_loss += batch_loss;

        // Compute accuracy
        const float* pred_data = predictions.Data<float>();
        const float* target_data = batch.labels.Data<float>();

        for (size_t b = 0; b < batch.size; ++b) {
            int pred_class = 0, true_class = 0;
            float max_pred = pred_data[b * config_.output_size];
            float max_target = target_data[b * config_.output_size];

            for (size_t c = 0; c < config_.output_size; ++c) {
                if (pred_data[b * config_.output_size + c] > max_pred) {
                    max_pred = pred_data[b * config_.output_size + c];
                    pred_class = static_cast<int>(c);
                }
                if (target_data[b * config_.output_size + c] > max_target) {
                    max_target = target_data[b * config_.output_size + c];
                    true_class = static_cast<int>(c);
                }
            }
            if (pred_class == true_class) correct++;
            total++;
        }
    }

    float final_loss = batch_num > 0 ? val_loss / batch_num : 0.0f;
    float final_acc = total > 0 ? static_cast<float>(correct) / total : 0.0f;

    UpdateMetrics([final_loss, final_acc](TrainingMetrics& m) {
        m.val_loss = final_loss;
        m.val_accuracy = final_acc;
    });
}

} // namespace cyxwiz
