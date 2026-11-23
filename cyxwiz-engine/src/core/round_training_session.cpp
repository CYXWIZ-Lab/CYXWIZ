#include "round_training_session.h"

#include "arrow_dataset.h"
#include "executable_model.h"
#include "label_vocabulary.h"
#include "sequence_arrow_batcher.h"
#include "training_batcher_setup.h"
#include "training_executor.h"
#include "training_resume_checkpoint.h"

#include <arrow/api.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace cyxwiz {

namespace {

// The last round's files stay loaded; a worker's leases move a chunk at a time.
constexpr size_t kLoadedFiles = 4;

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

RoundFailure Failure(TrainingFailureKind kind, std::string text) {
    spdlog::error("Round training session: {}", text);
    return RoundFailure{kind, std::move(text)};
}

// What a round cannot honour: these belong to one node training alone.
std::string RefusalFor(const TrainingConfiguration& config) {
    if (config.sequence_batch.enabled && !config.sequence_batch.create_causal_lm_targets) {
        return "of the sequence graphs only language models train in a group; token tagging builds its label "
               "list from the data, and every part of the data would build its own";
    }
    if (config.scheduler.has_value() || config.lr_schedule != "none") {
        return "a learning-rate schedule cannot follow a group yet: remove the scheduler node (or set the optimizer's "
               "schedule to none) to train in a group";
    }
    if (config.learning_rate_range.enabled) return "a learning-rate range test runs on one node, not in a group";
    if (!config.initialize_from_checkpoint.empty()) {
        return "a group starts from the state its coordinator sends, not from a checkpoint named in the graph";
    }
    return {};
}

}  // namespace

RoundTrainingSession::RoundTrainingSession() = default;
RoundTrainingSession::~RoundTrainingSession() = default;

std::unique_ptr<RoundTrainingSession> RoundTrainingSession::Open(const GraphTrainingJobRequest& request,
                                                                 uint32_t worker, RoundFailure& failure) {
    failure = {};
    PreparedGraphJob prepared;
    if (auto refused = PrepareGraphJob(request, prepared); !refused.error.empty()) {
        failure = RoundFailure{refused.failure, refused.error};
        return nullptr;
    }
    TrainingConfiguration& config = prepared.config;
    if (const std::string refusal = RefusalFor(config); !refusal.empty()) {
        failure = Failure(TrainingFailureKind::Refused, refusal);
        return nullptr;
    }
    const auto sample = prepared.datasets.find(config.dataset_name);
    if (sample == prepared.datasets.end()) {
        failure = Failure(TrainingFailureKind::DataError,
                          "the file given for '" + config.dataset_name + "' is too large to hold in memory; a part "
                          "of a group's data must fit (split the dataset into smaller parts)");
        return nullptr;
    }
    const std::string label_column = config.dataset_roles.train.label_column;
    const bool sequence = config.sequence_batch.enabled;
    if (sequence) {
        // Token windows carry their tokenizer's vocabulary with them, so every
        // part of the data numbers the tokens alike. Text does not.
        const std::string column = config.sequence_batch.token_column.empty() ? "tokens" : config.sequence_batch.token_column;
        const auto table = sample->second->GetArrowTable();
        const int index = table && table->schema() ? table->schema()->GetFieldIndex(column) : -1;
        if (index < 0 || table->field(index)->type()->id() != arrow::Type::LIST) {
            failure = Failure(TrainingFailureKind::Refused,
                              "a language model trains in a group on token windows (the Tokenize step's output); "
                              "column '" + column + "' holds text, and every part of the data would build its own vocabulary");
            return nullptr;
        }
    } else if (TextLabelColumn(sample->second->GetArrowTable(), label_column)) {
        failure = Failure(TrainingFailureKind::Refused,
                          "the label column '" + label_column + "' holds text; in a group every part of the data "
                          "would number the classes on its own. Store the labels as class numbers");
        return nullptr;
    }

    // A round trains every leased row, in order, and nothing else: the
    // group's coordinator owns validation, checkpoints and when to stop.
    config.train_ratio = 1.0f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.0f;
    config.stratified = false;
    config.has_data_split = false;
    config.shuffle = false;
    config.save_best_checkpoint = false;
    config.early_stopping_patience = 0;
    config.generation_preview.enabled = false;
    if (request.batch_size_override > 0) config.batch_size = request.batch_size_override;
    if (config.model_seed >= 0) {
        // A seed of this worker's own, kept a non-negative int like the graph's.
        config.model_seed = static_cast<int>(
            TrainingEpochSeed(static_cast<std::uint64_t>(config.model_seed), static_cast<int>(worker & 0x7fffffffu)) & 0x7fffffffu);
    } else {
        spdlog::warn("Round training session: the graph's Data Loader has no model seed, so this worker's rounds "
                     "cannot be reproduced");
    }

    std::unique_ptr<RoundTrainingSession> session(new RoundTrainingSession());
    session->sequence_ = sequence;
    session->label_column_ = label_column;
    session->dataset_name_ = config.dataset_name;
    session->batch_size_ = std::max(1, config.batch_size);
    if (sequence) {
        // The windows' vocabulary and length complete the model's shape; later
        // parts must carry the same vocabulary (the batcher builder checks).
        const auto first = BuildSequenceBatcherFromArrowDataset(sample->second, config, session->batch_size_);
        if (!first.success()) {
            failure = Failure(TrainingFailureKind::DataError, "sequence batcher: " + first.error_message);
            return nullptr;
        }
        ApplySequenceBatcherBuildResultToTrainingConfig(first, config);
        session->round_config_ = config;
        session->executor_ = std::make_unique<TrainingExecutor>(config, std::unique_ptr<ISequenceBatcher>{}, first.id_to_label);
    } else {
        session->round_config_ = config;
        session->executor_ = std::make_unique<TrainingExecutor>(config, ResolvedExternalBatchers{});
    }

    // Build the model now, on the sample file, without training: the layout
    // and the starting state are needed before the first round.
    std::string state_error;
    TrainingExecutor::RoundRun setup;
    setup.round = 1;
    setup.setup_only = true;
    setup.on_ready = [&](IExecutableModel& model, std::string& error) {
        return DescribeModelState(model, session->layout_, error) &&
               ReadModelState(model, session->layout_, true, session->initial_state_, error);
    };
    TrainingExecutor::RoundRunResult built;
    std::string build_error;
    if (!session->RunRound(setup, sample->second, built, build_error) || !built.ok) {
        failure = Failure(TrainingFailureKind::Internal,
                          "the model could not be built: " + (build_error.empty() ? built.error : build_error));
        return nullptr;
    }
    spdlog::info("Round training session: model built, {} values exchanged of {} ({} tensors), layout {}",
                 session->layout_.exchanged_count, session->layout_.total_count, session->layout_.segments.size(),
                 session->layout_.hash);
    return session;
}

bool RoundTrainingSession::RunRound(const TrainingExecutor::RoundRun& run, const std::shared_ptr<ArrowDataset>& dataset,
                                    TrainingExecutor::RoundRunResult& result, std::string& error) {
    if (sequence_) {
        auto built = BuildSequenceBatcherFromArrowDataset(dataset, round_config_, batch_size_);
        if (!built.success()) {
            error = "sequence batcher: " + built.error_message;
            return false;
        }
        result = executor_->TrainRound(run, batch_size_, std::move(built.batcher));
    } else {
        result = executor_->TrainRound(
            run, batch_size_,
            TakeResolvedExternalBatchers(BuildArrowTrainingBatchers(round_config_, dataset, label_column_, batch_size_)));
    }
    return true;
}

bool RoundTrainingSession::LoadInitialState(std::span<const float> full_state, std::string& error) {
    if (full_state.size() != layout_.total_count) {
        error = "the starting state has " + std::to_string(full_state.size()) + " values, the model has " +
                std::to_string(layout_.total_count);
        return false;
    }
    if (!std::all_of(full_state.begin(), full_state.end(), [](float value) { return std::isfinite(value); })) {
        error = "the starting state holds a value that is not a finite number";
        return false;
    }
    pending_full_state_.assign(full_state.begin(), full_state.end());
    return true;
}

std::shared_ptr<ArrowDataset> RoundTrainingSession::LoadLeases(const RoundData& data, std::vector<uint64_t>& lease_rows,
                                                               RoundFailure& failure) {
    if (data.leases.empty()) {
        failure = Failure(TrainingFailureKind::DataError, "the round leases no data");
        return nullptr;
    }
    std::vector<std::pair<std::string, std::shared_ptr<ArrowDataset>>> used;
    std::vector<std::shared_ptr<arrow::Table>> parts;
    lease_rows.clear();
    for (const RowRange& lease : data.leases) {
        const std::string path = lease.file.lexically_normal().string();
        std::shared_ptr<ArrowDataset> file;
        for (const auto& candidates : {used, loaded_files_}) {
            for (const auto& [loaded_path, loaded] : candidates) {
                if (!file && loaded_path == path) file = loaded;
            }
        }
        if (!file) {
            const std::string extension = Lower(lease.file.extension().string());
            if (extension == ".parquet") {
                file = ArrowDataset::FromParquet(path, dataset_name_);
            } else if (extension == ".arrow" || extension == ".feather" || extension == ".ipc") {
                file = ArrowDataset::FromFile(path, dataset_name_);
            }
            if (!file) {
                failure = Failure(TrainingFailureKind::DataError, "the leased file could not be read: " + path);
                return nullptr;
            }
        }
        if (std::none_of(used.begin(), used.end(), [&](const auto& entry) { return entry.first == path; })) {
            used.emplace_back(path, file);
        }
        const uint64_t file_rows = static_cast<uint64_t>(file->GetNumRows());
        const uint64_t rows = lease.rows == 0 && lease.row_offset < file_rows ? file_rows - lease.row_offset : lease.rows;
        if (rows == 0 || lease.row_offset > file_rows || rows > file_rows - lease.row_offset) {
            failure = Failure(TrainingFailureKind::DataError,
                              "a lease asks for rows " + std::to_string(lease.row_offset) + " to " +
                              std::to_string(lease.row_offset + lease.rows) + " of " + path + ", which has " +
                              std::to_string(file_rows));
            return nullptr;
        }
        parts.push_back(file->GetArrowTable()->Slice(static_cast<int64_t>(lease.row_offset), static_cast<int64_t>(rows)));
        lease_rows.push_back(rows);
    }
    if (used.size() > kLoadedFiles) used.erase(used.begin(), used.end() - static_cast<std::ptrdiff_t>(kLoadedFiles));
    loaded_files_ = std::move(used);

    if (parts.size() == 1) return std::make_shared<ArrowDataset>(parts.front(), dataset_name_);
    auto joined = arrow::ConcatenateTables(parts);
    if (!joined.ok()) {
        failure = Failure(TrainingFailureKind::DataError,
                          "the leased files do not have the same columns: " + joined.status().ToString());
        return nullptr;
    }
    // The first part's file notes (a token-window file's tokenizer) stay with the rows.
    return std::make_shared<ArrowDataset>((*joined)->ReplaceSchemaMetadata(parts.front()->schema()->metadata()),
                                          dataset_name_);
}

bool RoundTrainingSession::TrainRound(uint64_t round, std::span<const float> theta, const RoundData& data,
                                      const RoundBudget& budget, const RoundCallbacks& callbacks,
                                      RoundOutcome& outcome, RoundFailure& failure) {
    outcome = {};
    failure = {};
    if (theta.size() != layout_.exchanged_count) {
        failure = Failure(TrainingFailureKind::Internal,
                          "the round's state has " + std::to_string(theta.size()) + " values, the model exchanges " +
                          std::to_string(layout_.exchanged_count));
        return false;
    }
    if (budget.train_seconds <= 0.0 && budget.max_steps == 0) {
        failure = Failure(TrainingFailureKind::Internal, "a round needs a time or a step budget");
        return false;
    }
    std::vector<uint64_t> lease_rows;
    const auto dataset = LoadLeases(data, lease_rows, failure);
    if (!dataset) return false;

    std::vector<float> local;
    std::string read_error;
    TrainingExecutor::RoundRun run;
    run.round = static_cast<int>(std::min<uint64_t>(round, 1'000'000'000));
    run.train_seconds = budget.train_seconds;
    run.max_steps = budget.max_steps;
    run.should_stop = callbacks.should_stop;
    run.on_progress = callbacks.on_progress;
    run.on_ready = [&](IExecutableModel& model, std::string& error) {
        if (!pending_full_state_.empty()) {
            if (!WriteModelState(model, layout_, pending_full_state_, true, error)) return false;
            pending_full_state_.clear();
            pending_full_state_.shrink_to_fit();
        }
        return WriteModelState(model, layout_, theta, false, error);
    };
    run.on_done = [&](IExecutableModel& model) { ReadModelState(model, layout_, false, local, read_error); };

    TrainingExecutor::RoundRunResult result;
    std::string data_error;
    if (!RunRound(run, dataset, result, data_error)) {
        failure = Failure(TrainingFailureKind::DataError, data_error);
        return false;
    }
    if (result.cancelled) {
        failure = Failure(TrainingFailureKind::Cancelled, "the round was stopped");
        return false;
    }
    if (!result.ok) {
        failure = Failure(ClassifyTrainingFailure(result.error), "the round failed: " + result.error);
        return false;
    }
    if (local.size() != theta.size()) {
        failure = Failure(TrainingFailureKind::Internal, "the trained state could not be read: " + read_error);
        return false;
    }
    // The update, in place: the state the round started from minus the state it reached.
    for (size_t i = 0; i < local.size(); ++i) local[i] = theta[i] - local[i];
    if (!std::all_of(local.begin(), local.end(), [](float value) { return std::isfinite(value); })) {
        failure = Failure(TrainingFailureKind::Internal,
                          "training produced a value that is not a finite number; the update is not sent");
        return false;
    }
    outcome.delta = std::move(local);
    outcome.steps = result.steps;
    outcome.samples = result.samples;
    outcome.last_loss = result.last_loss;
    outcome.train_seconds = result.train_seconds;
    uint64_t left = result.samples;
    for (const uint64_t rows : lease_rows) {
        const uint64_t done = std::min(left, rows);
        outcome.rows_done.push_back(done);
        left -= done;
    }
    return true;
}

}  // namespace cyxwiz
