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
#include "../../core/project_manager.h"
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
    if ((file_type == "csv" || file_type == "tsv") &&
        (ctx.delimiter == '\0' || ctx.decimal_point == '\0')) {
        err = "CSV delimiter and decimal separator must each be one character";
        return false;
    }
    if ((file_type == "csv" || file_type == "tsv") &&
        ctx.delimiter == ctx.decimal_point) {
        err = "CSV delimiter and decimal separator must be different";
        return false;
    }
    return true;
}

uint64_t TabularLoader::LaunchAsyncLoad(const ApplyContext& ctx,
                                        std::shared_ptr<AsyncLoadState> state) {
    if (!state) return 0;

    // Keep the prior registration alive while the replacement loads. The UI
    // removes it only after this worker succeeds and the dataset passes audit.
    // Snapshot everything by value so the worker never races dialog state.
    const std::string path       = ctx.source_path;
    const std::string name       = ctx.dataset_name;
    const std::string file_type =
        ResolveTabularFileType(ctx.detected_file_type, path);
    const bool has_header        = ctx.has_header;
    const char delim             = (file_type == "tsv") ? '\t' : ctx.delimiter;
    const char decimal_point     = ctx.decimal_point;
    const bool use_threads       = ctx.use_threads;
    const auto missing_tokens    =
        cyxwiz::ParseMissingValueTokens(ctx.missing_value_tokens);
    const int skip_rows          = ctx.skip_rows;
    const int64_t max_rows       = ctx.max_rows;
    const bool force_disk        = ctx.force_disk_backed;
    const std::string label_col  = ctx.label_column;
    const auto selected_columns  = ctx.selected_columns;
    const std::string ingestion_cache_directory =
        cyxwiz::ProjectManager::Instance().GetIngestionCachePath();

    state->dataset_name = name;
    state->previous_dataset_name = ctx.previous_dataset_name;
    state->source_path  = path;
    if (!ctx.previous_dataset_name.empty()) {
        auto& registry = cyxwiz::DataRegistry::Instance();
        state->previous_arrow_dataset =
            registry.GetArrowDataset(ctx.previous_dataset_name);
        state->previous_parquet_dataset =
            registry.GetParquetBackedDataset(ctx.previous_dataset_name);
        if (auto previous_source =
                registry.GetTabularSourcePath(ctx.previous_dataset_name)) {
            state->previous_source_path = *previous_source;
        }
    }

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading " + name,
        [path, name, file_type, has_header, delim, decimal_point, use_threads,
         missing_tokens, skip_rows, max_rows, force_disk, label_col,
         selected_columns, ingestion_cache_directory, state]
        (cyxwiz::LambdaTask& task) {
            try {
                task.ReportProgress(0.05f, "Preparing " + file_type + " source");
                auto& reg = cyxwiz::DataRegistry::Instance();
                const cyxwiz::CsvProgressCallback csv_progress = [&task](
                    float source_progress, const std::string& message) {
                    task.ReportProgress(
                        0.05f + 0.83f *
                            std::clamp(source_progress, 0.0f, 1.0f),
                        message);
                    return !task.ShouldStop();
                };
                cyxwiz::DatasetAuditOptions audit_options;
                audit_options.should_cancel = [&task]() {
                    return task.ShouldStop();
                };
                audit_options.report_progress = [&task](
                    float audit_progress, const std::string& message) {
                    task.ReportProgress(
                        0.9f + 0.09f * std::clamp(audit_progress, 0.0f, 1.0f),
                        message);
                };

                if (file_type == "csv" || file_type == "tsv") {
                    // CSV family — LoadTabularCSV auto-picks Arrow
                    // in-memory vs Parquet disk-backed.
                    std::string load_status;
                    auto backend = reg.LoadTabularCSV(
                        path, name, has_header, delim, skip_rows, max_rows,
                        force_disk, missing_tokens, selected_columns,
                        decimal_point, use_threads, csv_progress, &load_status,
                        ingestion_cache_directory);
                    task.ReportProgress(0.9f, "Finalizing");

                    if (backend == cyxwiz::DataRegistry::TabularLoadBackend::InMemory) {
                        auto ds = reg.GetArrowDataset(name);
                        if (ds) {
                            state->success = true;
                            state->backend = 1;
                            state->rows    = ds->GetNumRows();
                            state->cols    = ds->GetNumColumns();
                            state->bytes   = ds->GetMemoryUsage();
                            state->message = load_status.empty()
                                ? "Loaded in memory"
                                : load_status;
                            auto audit = cyxwiz::DatasetAudit::AuditTabular(
                                name, ds, label_col, audit_options);
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
                            state->message = load_status.empty()
                                ? "Loaded via Parquet cache"
                                : load_status;
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
                        auto audit = cyxwiz::DatasetAudit::AuditTabular(
                            name, dataset, label_col, audit_options);
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
        if (!config.dataset_roles.train.IsSupplied()) {
            config.dataset_roles.train.dataset_name = dataset_name;
            config.dataset_roles.train.label_column = label_column;
        }
        ResolvedTabularDatasets datasets;
        datasets.train_arrow = registry.GetArrowDataset(
            config.dataset_roles.train.dataset_name);
        datasets.train_parquet = registry.GetParquetBackedDataset(
            config.dataset_roles.train.dataset_name);
        if (config.dataset_roles.dev.IsSupplied()) {
            datasets.dev_arrow = registry.GetArrowDataset(
                config.dataset_roles.dev.dataset_name);
            datasets.dev_parquet = registry.GetParquetBackedDataset(
                config.dataset_roles.dev.dataset_name);
        }
        if (config.dataset_roles.test.IsSupplied()) {
            datasets.test_arrow = registry.GetArrowDataset(
                config.dataset_roles.test.dataset_name);
            datasets.test_parquet = registry.GetParquetBackedDataset(
                config.dataset_roles.test.dataset_name);
        }
        auto built = BuildResolvedTabularTrainingBatchers(
            config, datasets, batch_size);
        if (!built.ok()) {
            spdlog::error(
                "TabularLoader: could not build resolved partitions: {}",
                built.error_message);
            return false;
        }

        return tm.StartTrainingExternal(
            std::move(config),
            TakeResolvedExternalBatchers(std::move(built.batchers)),
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
