#include <stdexcept>
#include "tabular_loader.h"
#include "../../core/archive_text_source.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/classification_decision.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_audit.h"
#include "../../core/graph_compiler.h"  // PreprocessingDomain
#include "../../core/parquet_backed_dataset.h"
#include "../../core/project_manager.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdio>
#include <filesystem>

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

}  // namespace

bool TabularLoader::ValidateApplyContext(const ApplyContext& ctx,
                                         std::string& err) const {
    return ValidateTabularApplyContext(ctx, err);
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
    const auto archive_member = ctx.archive_member;
    const bool json_lines        = ctx.json_lines;
    const bool has_header        = ctx.has_header;
    const char delim             = (file_type == "tsv") ? '\t' : ctx.delimiter;
    const char decimal_point     = ctx.decimal_point;
    const char quote_char        = ctx.quote_char;
    const bool use_threads       = ctx.use_threads;
    const auto missing_tokens    =
        cyxwiz::ParseMissingValueTokens(ctx.missing_value_tokens);
    const int skip_rows          = ctx.skip_rows;
    const int64_t max_rows       = ctx.max_rows;
    const bool force_disk        = ctx.force_disk_backed;
    const std::string label_col  = ctx.label_column;
    const std::string time_col   = ctx.time_column;
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
        [path, name, file_type, archive_member, json_lines, has_header, delim, decimal_point, quote_char,
         use_threads,
         missing_tokens, skip_rows, max_rows, force_disk, label_col, time_col,
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

                const bool parquet = file_type == "parquet";
                const bool external = cyxwiz::data_input::IsExternalFormatName(file_type);
                if (file_type == "csv" || file_type == "tsv" || parquet || external) {
                    // CSV family, Parquet and plugin formats (streamed into a
                    // Parquet cache first): the registry picks Arrow
                    // in-memory or a Parquet file on disk by the data budget.
                    std::string load_status;
                    auto backend = external
                        ? reg.LoadTabularExternal(file_type, path, name, force_disk, selected_columns,
                                                  &load_status, ingestion_cache_directory,
                                                  [&task] { return task.ShouldStop(); })
                        : parquet
                        ? reg.LoadTabularParquet(path, name, force_disk, selected_columns,
                                                 &load_status, ingestion_cache_directory)
                        : reg.LoadTabularCSV(
                              path, name, has_header, delim, skip_rows, max_rows,
                              force_disk, missing_tokens, selected_columns,
                              decimal_point, use_threads, csv_progress, &load_status,
                              ingestion_cache_directory, quote_char);
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
                                ? "Kept on disk"
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
                        state->message = external
                            ? (load_status.empty() ? "Failed to load the file" : load_status)
                            : parquet
                            ? "Failed to load Parquet. Check that the file is a Parquet file "
                              "and the selected columns exist."
                            : "Failed to load CSV. Check delimiter/header; if "
                              "metadata precedes the header, set Skip first N "
                              "source rows.";
                    }
                } else {
                    // Non-CSV supported formats go straight to Arrow
                    // in-memory on the AsyncTaskManager worker.
                    std::shared_ptr<cyxwiz::ArrowDataset> dataset;
                    if (file_type == "zip_text") {
                        dataset = reg.RegisterArrowTable(
                            LoadZipTextSelection(path, archive_member, {},
                                [&task] { return task.ShouldStop(); }), name);
                    } else if (file_type == "json") {
                        dataset = reg.LoadJSONToArrow(path, name,
                            json_lines || std::filesystem::path(path).extension() == ".jsonl");
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
                // A time series keeps its rows in time order (windows and the
                // chronological split follow the row order).
                if (state->success && !time_col.empty()) {
                    std::string sort_error;
                    if (state->backend == 2) {
                        state->success = false;
                        state->message = "The table is kept on disk, so its rows cannot be put in time order "
                                         "here. Clear Time column, or sort the file first.";
                    } else if (!reg.SortArrowTableByColumn(name, time_col, sort_error)) {
                        state->success = false;
                        state->message = sort_error;
                    } else {
                        state->message += ", sorted by " + time_col;
                    }
                    if (!state->success) reg.UnregisterTabularDataset(name);
                }
            } catch (const std::exception& e) {
                state->success = false;
                state->message = std::string("Error: ") + e.what();
                spdlog::error("TabularLoader async: {}", state->message);
            }
            // Publish barrier — must be set LAST so PollAsyncLoadResult
            // only reads a fully initialized state.
            state->done.store(true);
            // The dialog reads the published state; the task itself must also
            // end Failed (not Completed) so the Tasks panel tells the truth.
            if (!state->success && !task.ShouldStop()) {
                throw std::runtime_error(state->message.empty() ? std::string("Load failed") : state->message);
            }
        });
}

bool TabularLoader::IsRegistered(const std::string& name) const {
    auto& reg = cyxwiz::DataRegistry::Instance();
    // Sparse CSR outputs (TF-IDF / Count vectorizer with output_format=sparse)
    // train through the run resolver's sparse path, so this loader owns them too.
    return reg.IsArrowDataset(name) || reg.IsParquetBackedDataset(name) ||
           reg.IsSparseFeatureDataset(name);
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
        out.status_message = "Loaded " + name + " on disk (" +
            std::to_string(out.rows) + " rows, " +
            std::to_string(out.cols) + " cols, " +
            FormatBytes(out.bytes) + " file)";
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
        (disk_backed ? " on disk (" : " (") +
        std::to_string(state.rows) + " rows, " +
        std::to_string(state.cols) + " cols, " +
        FormatBytes(state.bytes) + (disk_backed ? " file" : "") + ")";
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

std::vector<ParamSchema> TabularLoader::NodeParams() const {
    // Per-category keys the Tabular branch of DataInputDialog::Apply
    // writes to the node. Keys not in any loader's schema are
    // considered "common" and never pruned.
    return {
        {"type",              "auto",   "Detected file type (csv/tsv/...)"},
        {"has_header",        "true",   "First row contains column names"},
        {"delimiter",         ",",      "Column separator"},
        {"quote_char",        "\"",     "Quote character (empty = no quoting)"},
        {"skip_rows",         "0",      "Rows to skip at file start"},
        {"max_rows",          "0",      "Max rows to load (0 = all)"},
        {"selected_columns",  "[]",     "JSON array of included column names"},
        {"label_column",      "",       "Column name used as training label"},
        {"time_column",       "",       "Time series: rows sorted by this column"},
        {"force_disk_backed", "false",  "Force Parquet disk-backed cache"},
        {"sheet_idx",         "0",      "Excel sheet index"},
        {"sheet_range",       "",       "Excel cell range"},
        {"json_lines",        "false",  "JSON lines (NDJSON) mode"},
        {"json_path",         "",       "JSONPath selector"},
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
