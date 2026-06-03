#include "tabular_loader.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/dataset_audit.h"
#include "../../core/graph_compiler.h"  // PreprocessingDomain
#include "../../core/parquet_backed_dataset.h"
#include "../../core/training_manager.h"

#include <spdlog/spdlog.h>

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
    if (ctx.source_path.empty()) {
        err = "Tabular load needs a file path";
        return false;
    }
    if (ctx.dataset_name.empty()) {
        err = "Dataset name is empty";
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
    const std::string file_type  = ctx.detected_file_type;
    const bool has_header        = ctx.has_header;
    const char delim             = (file_type == "tsv") ? '\t' : ctx.delimiter;
    const int skip_rows          = ctx.skip_rows;
    const int64_t max_rows       = ctx.max_rows;
    const bool force_disk        = ctx.force_disk_backed;
    const bool json_lines        = ctx.json_lines;
    const int excel_sheet        = ctx.excel_sheet_idx;
    const std::string label_col  = ctx.label_column;

    state->dataset_name = name;
    state->source_path  = path;

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading " + name,
        [path, name, file_type, has_header, delim, skip_rows, max_rows,
         force_disk, json_lines, excel_sheet, label_col, state]
        (cyxwiz::LambdaTask& task) {
            try {
                task.ReportProgress(0.1f, "Reading " + file_type);
                auto& reg = cyxwiz::DataRegistry::Instance();

                if (file_type == "csv" || file_type == "tsv" ||
                    file_type == "txt" || file_type == "arff") {
                    // CSV family — LoadTabularCSV auto-picks Arrow
                    // in-memory vs Parquet disk-backed.
                    auto backend = reg.LoadTabularCSV(
                        path, name, has_header, delim, skip_rows, max_rows,
                        force_disk);
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
                        state->message = "Failed to load CSV - check file format";
                    }
                } else {
                    // Non-CSV tabular formats go straight to Arrow in-memory.
                    // Previously these ran synchronously on the UI thread; the
                    // refactor puts them on the AsyncTaskManager worker along
                    // with CSV so a large Parquet or Excel file no longer
                    // freezes the dialog.
                    std::shared_ptr<cyxwiz::ArrowDataset> dataset;
                    if (file_type == "parquet") {
                        dataset = reg.LoadParquetToArrow(path, name);
                    } else if (file_type == "json") {
                        dataset = reg.LoadJSONToArrow(path, name, json_lines);
                    } else if (file_type == "excel") {
                        dataset = reg.LoadExcelToArrow(path, name, excel_sheet);
                    } else {
                        // "auto" + anything else — LoadArrowTable auto-detects.
                        dataset = reg.LoadArrowTable(path, name);
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
                        state->message = "Failed to load " + file_type +
                                         " - check file format";
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
