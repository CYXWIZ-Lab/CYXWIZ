#include "tabular_loader.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/classification_decision.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_audit.h"
#include "../../core/graph_compiler.h"  // PreprocessingDomain
#include "../../core/ner_sequence_builder.h"
#include "../../core/node_executors/text_column_utils.h"
#include "../../core/parquet_backed_dataset.h"
#include "../../core/training_manager.h"
#include "../../core/training_batcher_setup.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <cctype>

namespace fs = std::filesystem;

namespace cyxwiz::loaders {

namespace {
// Local copy of DataInputDialog::FormatBytes so the loader module
// doesn't depend on the dialog header. The format is identical — KB /
// MB / GB / TB with one decimal above 1 KB, bare byte count below.
std::string FormatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_idx = 0;
    double size = static_cast<double>(bytes);
    while (size >= 1024.0 && unit_idx < 4) {
        size /= 1024.0;
        ++unit_idx;
    }
    char buf[32];
    if (unit_idx == 0) std::snprintf(buf, sizeof(buf), "%zu %s", bytes, units[unit_idx]);
    else               std::snprintf(buf, sizeof(buf), "%.1f %s", size, units[unit_idx]);
    return std::string(buf);
}

std::vector<std::string> SplitSequenceTokens(const std::string& value) {
    std::vector<std::string> tokens;
    std::string current;
    for (char ch : value) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }
    return tokens;
}

bool LoadTabularSequenceRows(
    const std::shared_ptr<arrow::Table>& table,
    const std::string& token_column,
    const std::string& pos_column,
    const std::string& tag_column,
    std::vector<NERSequenceRow>& rows,
    std::string& error) {

    if (!table) {
        error = "sequence dataset table is null";
        return false;
    }
    if (token_column.empty()) {
        error = "sequence token column is empty";
        return false;
    }
    if (tag_column.empty()) {
        error = "sequence tag column is empty";
        return false;
    }

    auto read_string_column = [&](const std::string& column_name,
                                  std::vector<std::string>& out) -> bool {
        auto column = table->GetColumnByName(column_name);
        if (!column) {
            error = "sequence column '" + column_name + "' was not found";
            return false;
        }
        std::string type_name;
        if (!cyxwiz::ReadColumnAsStrings(column, out, type_name)) {
            error = "sequence column '" + column_name + "' must be text, got " +
                    type_name;
            return false;
        }
        return true;
    };

    std::vector<std::string> token_cells;
    std::vector<std::string> pos_cells;
    std::vector<std::string> tag_cells;
    if (!read_string_column(token_column, token_cells)) {
        return false;
    }
    if (!pos_column.empty() && !read_string_column(pos_column, pos_cells)) {
        return false;
    }
    if (!read_string_column(tag_column, tag_cells)) {
        return false;
    }
    if (token_cells.size() != tag_cells.size() ||
        (!pos_column.empty() && token_cells.size() != pos_cells.size())) {
        error = "sequence column row counts do not match";
        return false;
    }

    rows.clear();
    rows.reserve(token_cells.size());
    for (size_t i = 0; i < token_cells.size(); ++i) {
        NERSequenceRow row;
        row.tokens = SplitSequenceTokens(token_cells[i]);
        if (row.tokens.empty()) {
            continue;
        }

        if (!pos_column.empty()) {
            row.pos_tags = SplitSequenceTokens(pos_cells[i]);
            if (!row.pos_tags.empty() && row.pos_tags.size() != row.tokens.size()) {
                error = "sequence POS row " + std::to_string(i) +
                        " does not match token count";
                return false;
            }
        }

        row.ner_tags = SplitSequenceTokens(tag_cells[i]);
        if (row.ner_tags.size() != row.tokens.size()) {
            error = "sequence tag row " + std::to_string(i) +
                    " does not match token count";
            return false;
        }

        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        error = "sequence dataset contained no usable rows";
        return false;
    }
    return true;
}

std::shared_ptr<arrow::Table> LoadSequenceSourceTable(
    const std::string& dataset_name) {
    auto& registry = cyxwiz::DataRegistry::Instance();
    if (auto arrow_ds = registry.GetArrowDataset(dataset_name)) {
        return arrow_ds->GetArrowTable();
    }
    if (auto parquet_ds = registry.GetParquetBackedDataset(dataset_name)) {
        std::vector<int> all_row_groups;
        all_row_groups.reserve(static_cast<size_t>(parquet_ds->GetNumRowGroups()));
        for (int i = 0; i < parquet_ds->GetNumRowGroups(); ++i) {
            all_row_groups.push_back(i);
        }
        return parquet_ds->ReadRowGroups(all_row_groups);
    }
    return nullptr;
}

}  // namespace

bool TabularLoader::ValidateApplyContext(const ApplyContext& ctx,
                                         std::string& err) const {
    if (ctx.source_path.empty()) {
        err = "Tabular load needs a file path";
        return false;
    }
    if (ctx.dataset_name.empty()) {
        err = "Dataset name is empty";
        return false;
    }
    const std::string file_type =
        NormalizeTabularFileType(ctx.detected_file_type);
    if (IsUnsupportedTabularFileType(file_type)) {
        err = UnsupportedTabularFileTypeMessage(file_type);
        return false;
    }
    return true;
}

uint64_t TabularLoader::LaunchAsyncLoad(const ApplyContext& ctx,
                                        std::shared_ptr<AsyncLoadState> state) {
    if (!state) return 0;

    // Cross-Apply cleanup: if the user re-Applies with a different file,
    // the new auto-generated dataset_name differs from the old one
    // (both are derived from the file stem). LoadTabularCSV will
    // UnregisterTabularDataset(new_name) on entry, but it doesn't know
    // about the old one — so we clear it here before launching the
    // worker. Same guarantee the pre-refactor inline code provided.
    auto& registry = cyxwiz::DataRegistry::Instance();
    if (!ctx.previous_dataset_name.empty() &&
        ctx.previous_dataset_name != ctx.dataset_name) {
        registry.UnregisterTabularDataset(ctx.previous_dataset_name);
    }

    // Snapshot everything by value so the worker never races dialog state.
    const std::string path       = ctx.source_path;
    const std::string name       = ctx.dataset_name;
    const std::string file_type =
        ResolveTabularFileType(ctx.detected_file_type, path);
    const bool has_header        = ctx.has_header;
    const char delim             = (file_type == "tsv") ? '\t' : ctx.delimiter;
    const auto missing_tokens    =
        cyxwiz::ParseMissingValueTokens(ctx.missing_value_tokens);
    const int skip_rows          = ctx.skip_rows;
    const int64_t max_rows       = ctx.max_rows;
    const bool force_disk        = ctx.force_disk_backed;
    const std::string label_col  = ctx.label_column;
    const auto selected_columns  = ctx.selected_columns;

    state->dataset_name = name;
    state->source_path  = path;

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading " + name,
        [path, name, file_type, has_header, delim, missing_tokens, skip_rows, max_rows,
         force_disk, label_col, selected_columns, state]
        (cyxwiz::LambdaTask& task) {
            try {
                task.ReportProgress(0.1f, "Reading " + file_type);
                auto& reg = cyxwiz::DataRegistry::Instance();

                if (file_type == "csv" || file_type == "tsv") {
                    // CSV family — LoadTabularCSV auto-picks Arrow
                    // in-memory vs Parquet disk-backed.
                    auto backend = reg.LoadTabularCSV(
                        path, name, has_header, delim, skip_rows, max_rows,
                        force_disk, missing_tokens, selected_columns);
                    task.ReportProgress(0.9f, "Finalizing");

                    if (backend == cyxwiz::DataRegistry::TabularLoadBackend::InMemory) {
                        auto ds = reg.GetArrowDataset(name);
                        if (ds) {
                            state->success = true;
                            state->backend = 1;
                            state->rows    = ds->GetNumRows();
                            state->cols    = ds->GetNumColumns();
                            state->bytes   = ds->GetMemoryUsage();
                            state->message = "Loaded in memory";
                            auto audit = cyxwiz::DatasetAudit::AuditTabular(name, ds, label_col);
                            state->audit_errors = audit.ErrorCount();
                            state->audit_warnings = audit.WarningCount();
                            state->audit_message = cyxwiz::FormatAuditSummary(audit);
                            state->audit_issue_lines = cyxwiz::FormatAuditIssueLines(audit);
                        } else {
                            state->success = false;
                            state->message = "Load completed but dataset missing from registry";
                        }
                    } else if (backend == cyxwiz::DataRegistry::TabularLoadBackend::DiskBacked) {
                        auto pq = reg.GetParquetBackedDataset(name);
                        if (pq) {
                            state->success = true;
                            state->backend = 2;
                            state->rows    = pq->GetNumRows();
                            state->cols    = pq->GetNumColumns();
                            state->bytes   = pq->GetMemoryUsage();
                            state->message = "Loaded via Parquet cache";
                            auto audit = cyxwiz::DatasetAudit::AuditParquet(name, pq, label_col);
                            state->audit_errors = audit.ErrorCount();
                            state->audit_warnings = audit.WarningCount();
                            state->audit_message = cyxwiz::FormatAuditSummary(audit);
                            state->audit_issue_lines = cyxwiz::FormatAuditIssueLines(audit);
                        } else {
                            state->success = false;
                            state->message = "Disk-backed load completed but dataset missing";
                        }
                    } else {
                        state->success = false;
                        state->message =
                            "Failed to load CSV. Check delimiter/header; if "
                            "metadata precedes the header, set Skip first N "
                            "source rows.";
                    }
                } else {
                    // Non-CSV supported formats go straight to Arrow
                    // in-memory on the AsyncTaskManager worker.
                    std::shared_ptr<cyxwiz::ArrowDataset> dataset;
                    if (file_type == "parquet") {
                        dataset = reg.LoadParquetToArrow(path, name);
                    } else {
                        // auto/feather/arrow/ipc use Arrow's table loader.
                        dataset = reg.LoadArrowTable(path, name);
                    }
                    if (dataset && !selected_columns.empty()) {
                        auto projected = dataset->SelectColumns(selected_columns);
                        if (!projected ||
                            projected->GetNumColumns() !=
                                static_cast<int64_t>(selected_columns.size())) {
                            reg.UnregisterTabularDataset(name);
                            dataset.reset();
                            state->message =
                                "Selected columns do not match the loaded schema";
                        } else {
                            dataset = reg.RegisterArrowTable(
                                projected->GetArrowTable(), name);
                        }
                    }
                    task.ReportProgress(0.9f, "Finalizing");

                    if (dataset) {
                        state->success = true;
                        state->backend = 1;
                        state->rows    = dataset->GetNumRows();
                        state->cols    = dataset->GetNumColumns();
                        state->bytes   = dataset->GetMemoryUsage();
                        state->message = "Loaded " + file_type;
                        auto audit = cyxwiz::DatasetAudit::AuditTabular(name, dataset, label_col);
                        state->audit_errors = audit.ErrorCount();
                        state->audit_warnings = audit.WarningCount();
                        state->audit_message = cyxwiz::FormatAuditSummary(audit);
                        state->audit_issue_lines = cyxwiz::FormatAuditIssueLines(audit);
                    } else {
                        state->success = false;
                        if (state->message.empty()) {
                            state->message = "Failed to load " + file_type +
                                             " - check file format";
                        }
                    }
                }
            } catch (const std::exception& e) {
                state->success = false;
                state->message = std::string("Error: ") + e.what();
                spdlog::error("TabularLoader async: {}", state->message);
            }
            // Publish barrier — must be set LAST so PollAsyncLoadResult
            // only reads a fully initialized state.
            state->done.store(true);
        });
}

bool TabularLoader::IsRegistered(const std::string& name) const {
    auto& reg = cyxwiz::DataRegistry::Instance();
    return reg.IsArrowDataset(name) || reg.IsParquetBackedDataset(name);
}

void TabularLoader::Unregister(const std::string& name) {
    cyxwiz::DataRegistry::Instance().UnregisterTabularDataset(name);
}

bool TabularLoader::RestoreFromRegistry(const std::string& name,
                                        const gui::MLNode& /*node*/,
                                        RestoreState& out) const {
    auto& reg = cyxwiz::DataRegistry::Instance();

    if (auto ds = reg.GetArrowDataset(name)) {
        out.found   = true;
        out.rows    = ds->GetNumRows();
        out.cols    = ds->GetNumColumns();
        out.bytes   = ds->GetMemoryUsage();
        out.backend = 1;
        out.memory_is_estimate = false;
        out.status_message = "Loaded " + name + " (" +
            std::to_string(out.rows) + " rows, " +
            std::to_string(out.cols) + " cols, " +
            FormatBytes(out.bytes) + ")";
        return true;
    }
    if (auto pq = reg.GetParquetBackedDataset(name)) {
        out.found   = true;
        out.rows    = pq->GetNumRows();
        out.cols    = pq->GetNumColumns();
        out.bytes   = pq->GetMemoryUsage();
        out.backend = 2;
        out.memory_is_estimate = false;
        out.status_message = "Loaded " + name + " via Parquet cache (" +
            std::to_string(out.rows) + " rows, " +
            std::to_string(out.cols) + " cols, " +
            FormatBytes(out.bytes) + " on disk)";
        return true;
    }
    return false;
}

CompletedLoadDescription TabularLoader::DescribeCompletedLoad(
    const AsyncLoadState& state) const {
    CompletedLoadDescription d;
    d.memory_is_estimate = false;

    const bool disk_backed = (state.backend == 2);
    const std::string backing_suffix = disk_backed ? " (disk-backed)" : "";
    d.node_description_suffix =
        std::to_string(state.rows) + " rows, " +
        std::to_string(state.cols) + " cols" + backing_suffix;

    fs::path p(state.source_path);
    d.default_status_message = "Loaded " + p.filename().string() +
        (disk_backed ? " via Parquet cache (" : " (") +
        std::to_string(state.rows) + " rows, " +
        std::to_string(state.cols) + " cols, " +
        FormatBytes(state.bytes) + (disk_backed ? " on disk" : "") + ")";
    return d;
}

cyxwiz::PreprocessingDomain TabularLoader::Domain(
    const std::string& file_category) const {
    // TimeSeries nodes (file_category == "timeseries") share
    // TabularLoader's load path but get their own PreprocessingDomain
    // because the GraphCompiler's downstream checks (TimeSeriesWindow
    // required, train/val split by chronology, etc.) differ.
    if (file_category == "timeseries") {
        return cyxwiz::PreprocessingDomain::TimeSeries;
    }
    return cyxwiz::PreprocessingDomain::Tabular;
}

bool TabularLoader::LaunchTraining(
    cyxwiz::TrainingConfiguration config,
    const std::string& dataset_name,
    const std::string& label_column,
    int epochs,
    int batch_size,
    std::weak_ptr<cyxwiz::TrainingPlotPanel> plot_panel,
    std::function<void(bool)> node_editor_callback) {
    auto& registry = cyxwiz::DataRegistry::Instance();
    auto& tm       = cyxwiz::TrainingManager::Instance();
    const bool has_external_role = config.dataset_roles.dev.IsSupplied() ||
                                   config.dataset_roles.test.IsSupplied();
    if (has_external_role && !config.sequence_batch.enabled) {
        const std::string train_role_name =
            config.dataset_roles.train.dataset_name.empty()
                ? dataset_name
                : config.dataset_roles.train.dataset_name;
        const std::string train_role_label =
            config.dataset_roles.train.label_column.empty()
                ? label_column
                : config.dataset_roles.train.label_column;

        auto train_arrow = registry.GetArrowDataset(train_role_name);
        auto train_parquet = registry.GetParquetBackedDataset(train_role_name);
        auto dev_arrow = config.dataset_roles.dev.IsSupplied()
            ? registry.GetArrowDataset(config.dataset_roles.dev.dataset_name)
            : nullptr;
        auto dev_parquet = config.dataset_roles.dev.IsSupplied()
            ? registry.GetParquetBackedDataset(config.dataset_roles.dev.dataset_name)
            : nullptr;
        auto test_arrow = config.dataset_roles.test.IsSupplied()
            ? registry.GetArrowDataset(config.dataset_roles.test.dataset_name)
            : nullptr;
        auto test_parquet = config.dataset_roles.test.IsSupplied()
            ? registry.GetParquetBackedDataset(config.dataset_roles.test.dataset_name)
            : nullptr;

        const bool missing_train = !train_arrow && !train_parquet;
        const bool missing_dev = config.dataset_roles.dev.IsSupplied() &&
            !dev_arrow && !dev_parquet;
        const bool missing_test = config.dataset_roles.test.IsSupplied() &&
            !test_arrow && !test_parquet;
        if (missing_train || missing_dev || missing_test) {
            spdlog::error("TabularLoader: explicit tabular roles require "
                          "registered Arrow or Parquet-backed sources");
            return false;
        }

        // Explicit Dev/Test roles replace splits derived from Train. Delay
        // prefetch wrapping until those replacements are final so each wrapper
        // and its owning concrete source cannot diverge.
        auto assembly_config = config;
        assembly_config.prefetch_factor = 0;
        assembly_config.has_data_split = true;
        if (config.dataset_roles.dev.IsSupplied()) {
            assembly_config.val_ratio = 0.0f;
        }
        if (config.dataset_roles.test.IsSupplied()) {
            assembly_config.test_ratio = 0.0f;
        }
        assembly_config.train_ratio = std::max(
            0.0f,
            1.0f - assembly_config.val_ratio - assembly_config.test_ratio);
        config.train_ratio = assembly_config.train_ratio;
        config.val_ratio = assembly_config.val_ratio;
        config.test_ratio = assembly_config.test_ratio;
        spdlog::info(
            "TabularLoader: explicit role mapping uses Train split "
            "train={:.2f}, dev={:.2f}, test={:.2f}; "
            "external dev={}, external test={}",
            config.train_ratio, config.val_ratio, config.test_ratio,
            config.dataset_roles.dev.IsSupplied(),
            config.dataset_roles.test.IsSupplied());
        auto batchers = train_arrow
            ? BuildArrowTrainingBatchers(
                  assembly_config, train_arrow, train_role_label, batch_size)
            : BuildParquetTrainingBatchers(
                  assembly_config, train_parquet, train_role_label, batch_size);

        auto make_arrow_role = [&](std::shared_ptr<cyxwiz::ArrowDataset> ds,
                                   const std::string& role_label) {
            return std::make_unique<cyxwiz::ArrowDatasetBatcher>(
                ds, role_label, batch_size, false, 1.0f, true, "", 0,
                config.num_workers, cyxwiz::BatcherPhase::Train, 0.0f,
                static_cast<uint32_t>(config.dataloader_seed));
        };
        auto make_parquet_role = [&, batch_size](
            std::shared_ptr<cyxwiz::ParquetBackedDataset> ds,
            const std::string& role_label) {
            return std::make_unique<cyxwiz::ParquetArrowBatcher>(
                ds, role_label, batch_size, false, 1.0f, true, "", 0,
                config.num_workers, cyxwiz::BatcherPhase::Train, 0.0f,
                static_cast<uint32_t>(config.dataloader_seed));
        };

        if (dev_arrow) {
            batchers.parquet_val.reset();
            batchers.arrow_val = make_arrow_role(
                dev_arrow, config.dataset_roles.dev.label_column);
            batchers.val = batchers.arrow_val.get();
        } else if (dev_parquet) {
            batchers.arrow_val.reset();
            batchers.parquet_val = make_parquet_role(
                dev_parquet, config.dataset_roles.dev.label_column);
            batchers.val = batchers.parquet_val.get();
        }
        if (test_arrow) {
            batchers.parquet_test.reset();
            batchers.arrow_test = make_arrow_role(
                test_arrow, config.dataset_roles.test.label_column);
            batchers.test = batchers.arrow_test.get();
        } else if (test_parquet) {
            batchers.arrow_test.reset();
            batchers.parquet_test = make_parquet_role(
                test_parquet, config.dataset_roles.test.label_column);
            batchers.test = batchers.parquet_test.get();
        }

        auto apply_role_transforms = [&](cyxwiz::IBatcher* b) {
            if (!b) return;
            if (config.preprocessing.has_normalization) {
                b->SetNormalization(config.preprocessing.norm_mean,
                                    config.preprocessing.norm_std);
            }
            if (config.is_time_series ||
                UsesScalarBinaryTargets(config.loss_type)) {
                if (auto* arrow_b =
                        dynamic_cast<cyxwiz::ArrowDatasetBatcher*>(b)) {
                    arrow_b->SetScalarLabelMode(true);
                } else if (auto* parquet_b =
                               dynamic_cast<cyxwiz::ParquetArrowBatcher*>(b)) {
                    parquet_b->SetScalarLabelMode(true);
                }
            } else if (config.preprocessing.has_onehot) {
                b->SetOneHotEncoding(config.preprocessing.num_classes);
            } else {
                b->SetOneHotEncoding(config.output_size);
            }
        };
        apply_role_transforms(batchers.val);
        apply_role_transforms(batchers.test);
        batchers.num_val_samples =
            batchers.val ? batchers.val->GetNumSamples() : 0;
        batchers.num_test_samples =
            batchers.test ? batchers.test->GetNumSamples() : 0;
        AttachTrainingBatcherPrefetchWrappers(
            batchers, config, "explicit tabular roles");

        return tm.StartTrainingExternal(
            std::move(config), TakeResolvedExternalBatchers(std::move(batchers)),
            epochs, batch_size, plot_panel, std::move(node_editor_callback));
    }
    if (config.sequence_batch.enabled) {
        auto table = LoadSequenceSourceTable(dataset_name);
        if (!table) {
            spdlog::error("TabularLoader: '{}' is registered but no Arrow/Parquet "
                          "table could be retrieved for sequence training",
                          dataset_name);
            return false;
        }

        std::vector<NERSequenceRow> rows;
        std::string error;
        if (!LoadTabularSequenceRows(
                table,
                config.sequence_batch.token_column,
                config.sequence_batch.pos_column,
                config.sequence_batch.tag_column,
                rows,
                error)) {
            spdlog::error("TabularLoader: sequence materialization failed for '{}': {}",
                          dataset_name, error);
            return false;
        }

        NERSequenceBuilderConfig builder_config;
        builder_config.use_pos_tags = !config.sequence_batch.pos_column.empty();
        builder_config.require_tags = true;
        builder_config.batcher.batch_size = static_cast<size_t>(std::max(1, batch_size));
        builder_config.batcher.shuffle = config.shuffle;
        builder_config.batcher.drop_last = config.drop_last;
        builder_config.batcher.create_attention_mask =
            config.sequence_batch.create_attention_mask;
        builder_config.batcher.tag_ignore_index = config.sequence_batch.ignore_index;
        builder_config.token_vocabulary.lowercase = true;
        builder_config.pos_vocabulary.lowercase = false;

        auto built = BuildNERSequenceData(rows, builder_config);
        if (!built.has_tags || built.samples.empty()) {
            spdlog::error("TabularLoader: sequence build produced no supervised samples for '{}'",
                          dataset_name);
            return false;
        }

        spdlog::info("TabularLoader: Starting sequence training: dataset={}, epochs={}, "
                     "batch_size={}, {} samples, {} labels",
                     dataset_name, epochs, batch_size,
                     built.samples.size(), built.tag_vocabulary.Size());
        return tm.StartTrainingSequence(
            std::move(config),
            std::make_unique<SequenceBatcher>(built.samples, built.batcher_config),
            built.tag_vocabulary.Values(),
            epochs,
            batch_size,
            plot_panel,
            std::move(node_editor_callback));
    }

    // Tabular covers both backends: in-memory Arrow (the default fast
    // path) and disk-backed Parquet (picked by LoadTabularCSV when the
    // CSV was too big to fit in RAM). Check Arrow first since the
    // Apply dispatch prefers it.
    if (auto arrow_ds = registry.GetArrowDataset(dataset_name)) {
        spdlog::info("TabularLoader: Starting Arrow training: dataset={}, epochs={}, "
                     "batch_size={}, label={}",
                     dataset_name, epochs, batch_size, label_column);
        return tm.StartTrainingArrow(std::move(config), arrow_ds, label_column,
                                     epochs, batch_size, plot_panel,
                                     std::move(node_editor_callback));
    }
    if (auto pq_ds = registry.GetParquetBackedDataset(dataset_name)) {
        spdlog::info("TabularLoader: Starting Parquet-backed training: dataset={}, "
                     "epochs={}, batch_size={}, label={}, {:.1f} MB on disk",
                     dataset_name, epochs, batch_size, label_column,
                     pq_ds->GetFileSizeBytes() / (1024.0 * 1024.0));
        return tm.StartTrainingParquet(std::move(config), pq_ds, label_column,
                                       epochs, batch_size, plot_panel,
                                       std::move(node_editor_callback));
    }
    spdlog::error("TabularLoader: '{}' is registered but neither Arrow nor Parquet "
                  "dataset can be retrieved", dataset_name);
    return false;
}

std::vector<ParamSchema> TabularLoader::NodeParams() const {
    // Per-category keys the Tabular branch of DataInputDialog::Apply
    // writes to the node. Keys not in any loader's schema are
    // considered "common" and never pruned.
    return {
        {"type",              "auto",   "Detected file type (csv/tsv/...)"},
        {"has_header",        "true",   "First row contains column names"},
        {"delimiter",         ",",      "Column separator"},
        {"skip_rows",         "0",      "Rows to skip at file start"},
        {"max_rows",          "0",      "Max rows to load (0 = all)"},
        {"selected_columns",  "[]",     "JSON array of included column names"},
        {"encoding",          "0",      "Text encoding index"},
        {"label_column",      "",       "Column name used as training label"},
        {"force_disk_backed", "false",  "Force Parquet disk-backed cache"},
        {"sheet_idx",         "0",      "Excel sheet index"},
        {"sheet_range",       "",       "Excel cell range"},
        {"json_lines",        "false",  "JSON lines (NDJSON) mode"},
        {"json_path",         "",       "JSONPath selector"},
        {"hdf5_dataset",      "data",   "HDF5 dataset name"},
    };
}

SyntheticBatch TabularLoader::MakeSynthetic(
    const cyxwiz::TrainingConfiguration& config, uint32_t seed) const {
    const bool time_series =
        config.preprocessing_domain == cyxwiz::PreprocessingDomain::TimeSeries;
    return MakeSyntheticForDomain(
        config,
        time_series ? cyxwiz::PreprocessingDomain::TimeSeries
                    : cyxwiz::PreprocessingDomain::Tabular,
        seed,
        time_series ? "TimeSeries" : "Tabular");
}

}  // namespace cyxwiz::loaders
