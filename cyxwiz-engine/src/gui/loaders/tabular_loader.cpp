#include "tabular_loader.h"

#include "../../core/arrow_dataset.h"
#include "../../core/async_task_manager.h"
#include "../../core/data_registry.h"
#include "../../core/parquet_backed_dataset.h"

#include <spdlog/spdlog.h>

#include <filesystem>

namespace fs = std::filesystem;

namespace cyxwiz::loaders {

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

    state->dataset_name = name;
    state->source_path  = path;

    auto& mgr = cyxwiz::AsyncTaskManager::Instance();
    return mgr.RunAsync(
        "Loading " + name,
        [path, name, file_type, has_header, delim, skip_rows, max_rows,
         force_disk, json_lines, excel_sheet, state]
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

}  // namespace cyxwiz::loaders
