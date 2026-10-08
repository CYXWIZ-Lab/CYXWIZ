#include "hdf5_source_load_task.h"
#include "async_task_manager.h"

#ifdef CYXWIZ_HAS_HDF5
#include "arrow_dataset.h"
#include "dataset_audit.h"
#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <algorithm>
#endif

#include <new>
#include <stdexcept>
#include <utility>

namespace cyxwiz {
namespace {

struct LoadCancelled {};

Hdf5SourceLoadResult Failure(Hdf5TableStatus status, std::string error, bool changed = false) {
    Hdf5SourceLoadResult result;
    result.read.status = status;
    result.read.error = std::move(error);
    result.source_changed = changed;
    return result;
}

#ifdef CYXWIZ_HAS_HDF5
void CheckCancellation(const std::function<bool()>& cancelled) {
    if (cancelled()) throw LoadCancelled{};
}

void CheckArrow(const arrow::Status& status) {
    if (status.IsOutOfMemory()) throw std::bad_alloc{};
    if (!status.ok()) throw std::runtime_error(status.ToString());
}

Hdf5SourceLoadResult ChangedSource() {
    return Failure(Hdf5TableStatus::ReadFailed,
                   "HDF5 source changed or became unavailable. Refresh the source and retry loading.", true);
}

Hdf5SourceLoadResult Prepare(const Hdf5SourceLoadRequest& request, LambdaTask& task) {
    const std::function<bool()> cancelled = [&task, external = request.cancel_requested] {
        return task.ShouldStop() || (external && external());
    };
    CheckCancellation(cancelled);
    std::string error;
    if (!ValidateHdf5InputSettings(request.settings, error))
        return Failure(Hdf5TableStatus::InvalidSelection, std::move(error));

    task.ReportProgress(0.05f, "Checking HDF5 source");
    CheckCancellation(cancelled);
    const auto before = ReadHdf5SourceStamp(request.path, error);
    CheckCancellation(cancelled);
    if (!before)
        return request.expected_source ? ChangedSource() : Failure(Hdf5TableStatus::InvalidFile, std::move(error));
    if (request.expected_source && *request.expected_source != *before) return ChangedSource();

    Hdf5TableReadOptions options;
    options.selection = request.settings.selection;
    options.numeric_policy = request.settings.numeric_policy;
    options.max_materialized_bytes = request.settings.max_materialized_bytes;
    options.cancel_requested = cancelled;
    task.ReportProgress(0.15f, "Reading HDF5 table");
    CheckCancellation(cancelled);
    auto read = ReadHdf5Table(request.path, options);
    CheckCancellation(cancelled);
    if (read.status == Hdf5TableStatus::Cancelled) throw LoadCancelled{};

    std::shared_ptr<DatasetAuditResult> audit;
    if (read.status == Hdf5TableStatus::Ok) {
        task.ReportProgress(0.75f, "Validating HDF5 table");
        CheckCancellation(cancelled);
        if (!read.table) throw std::runtime_error("HDF5 read returned no table");
        CheckArrow(read.table->ValidateFull());
        CheckCancellation(cancelled);
        const auto existing = read.table->schema()->metadata();
        auto metadata = existing ? existing->Copy() : std::make_shared<arrow::KeyValueMetadata>();
        CheckArrow(metadata->Set("hdf5.source_path", before->canonical_path));
        CheckArrow(metadata->Set("hdf5.source_size", std::to_string(before->size)));
        CheckArrow(metadata->Set("hdf5.source_modified", std::to_string(before->modified)));
        CheckArrow(metadata->Set("hdf5.import_mode", "numeric_table"));
        read.table = read.table->ReplaceSchemaMetadata(std::move(metadata));

        CheckCancellation(cancelled);
        DatasetAuditOptions audit_options;
        audit_options.should_cancel = cancelled;
        audit_options.report_progress = [&task](float progress, const std::string& message) {
            task.ReportProgress(0.78f + 0.14f * std::clamp(progress, 0.0f, 1.0f), message);
        };
        auto dataset = std::make_shared<ArrowDataset>(read.table, "HDF5 source");
        audit = std::make_shared<DatasetAuditResult>(DatasetAudit::AuditTabular(
            "HDF5 source", dataset, read.labels ? "label" : "", audit_options));
        CheckCancellation(cancelled);
        if (audit->cancelled) throw LoadCancelled{};
    }

    // No progress callbacks after the final stamp: callbacks can change the file.
    task.ReportProgress(0.95f, "Verifying HDF5 source");
    CheckCancellation(cancelled);
    const auto after = ReadHdf5SourceStamp(request.path, error);
    CheckCancellation(cancelled);
    if (!after || *after != *before) return ChangedSource();
    if (read.status != Hdf5TableStatus::Ok)
        return Failure(read.status, std::move(read.error));
    if (audit->HasErrors()) {
        auto result = Failure(Hdf5TableStatus::ReadFailed,
                              "HDF5 source refused by dataset audit. " + FormatAuditSummary(*audit));
        result.audit = std::move(audit);
        return result;
    }
    Hdf5SourceLoadResult result;
    result.read = std::move(read);
    result.source = after;
    result.audit = std::move(audit);
    return result;
}
#endif

} // namespace

std::shared_ptr<AsyncTask> MakeHdf5SourceLoadTask(
    Hdf5SourceLoadRequest request, std::shared_ptr<Hdf5SourceLoadTaskResult> state) {
    if (!state) throw std::invalid_argument("HDF5 source load requires result storage");
    return std::make_shared<LambdaTask>("Prepare HDF5 source",
        [captured_request = std::move(request), captured_state = std::move(state)](LambdaTask& task) mutable {
            auto request = std::move(captured_request);
            auto state = std::move(captured_state);
            if (!state) return;
            try {
#ifdef CYXWIZ_HAS_HDF5
                state->result = Prepare(request, task);
                if (task.IsCancelRequested()) throw LoadCancelled{};
#else
                state->result = Failure(Hdf5TableStatus::DependencyUnavailable,
                                        "HDF5 support is not compiled into this build");
#endif
            } catch (const LoadCancelled&) {
                state->result = Failure(Hdf5TableStatus::Cancelled, "HDF5 source load cancelled");
            } catch (const std::bad_alloc&) {
                state->result = Failure(Hdf5TableStatus::ResourceLimit, "HDF5 source load could not allocate memory");
            } catch (const std::exception& error) {
                state->result = Failure(Hdf5TableStatus::ReadFailed,
                                        "HDF5 source load failed: " + std::string(error.what()));
            } catch (...) {
                state->result = Failure(Hdf5TableStatus::ReadFailed, "HDF5 source load failed with an unknown error");
            }
            if (state->result.read.status == Hdf5TableStatus::Cancelled) task.MarkCancelled("HDF5 source load cancelled");
            else if (state->result.read.status == Hdf5TableStatus::Ok) task.MarkCompleted("HDF5 table prepared");
            else task.MarkFailed(state->result.read.error.empty() ? "HDF5 source load failed" : state->result.read.error);
            state->done.store(true, std::memory_order_release);
        }, true);
}

} // namespace cyxwiz
