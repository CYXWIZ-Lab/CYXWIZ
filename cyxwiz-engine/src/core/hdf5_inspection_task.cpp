#include "hdf5_inspection_task.h"
#include "async_task_manager.h"

#ifdef CYXWIZ_HAS_HDF5
#include <arrow/api.h>
#endif

#include <new>
#include <stdexcept>
#include <utility>

namespace cyxwiz {
namespace {

struct InspectionCancelled {};

Hdf5InspectionResult Failure(Hdf5TableStatus status, std::string error, bool changed = false) {
    Hdf5InspectionResult result;
    result.status = status;
    result.error = std::move(error);
    result.source_changed = changed;
    return result;
}

#ifdef CYXWIZ_HAS_HDF5
void CheckCancellation(const std::function<bool()>& cancelled) {
    if (cancelled()) throw InspectionCancelled{};
}

Hdf5InspectionResult ChangedSource() {
    return Failure(Hdf5TableStatus::ReadFailed,
                   "HDF5 source changed or became unavailable. Refresh the source and retry inspection.", true);
}

DataPreviewPage FormatPreview(const Hdf5TableReadResult& sampled, LambdaTask& task,
                              const std::function<bool()>& cancelled) {
    const auto& table = sampled.table;
    if (!table || sampled.data.shape.empty())
        throw std::runtime_error("HDF5 preview returned no table or source shape");
    DataPreviewPage page;
    page.backend = "HDF5 source";
    page.total_rows = static_cast<int64_t>(sampled.data.shape[0]);
    page.total_columns = static_cast<int64_t>(sampled.data.shape.size() == 1 ? 1 : sampled.data.shape[1]) +
                         (sampled.labels ? 1 : 0);
    page.offset = static_cast<int64_t>(sampled.row_offset);
    page.schema.reserve(static_cast<size_t>(table->num_columns()));
    for (const auto& field : table->schema()->fields()) {
        CheckCancellation(cancelled);
        page.schema.push_back({field->name(), field->type()->ToString(), field->nullable()});
    }
    page.rows.reserve(static_cast<size_t>(table->num_rows()));
    for (int64_t row = 0; row < table->num_rows(); ++row) {
        CheckCancellation(cancelled);
        std::vector<std::string> values;
        values.reserve(static_cast<size_t>(table->num_columns()));
        for (int column = 0; column < table->num_columns(); ++column) {
            CheckCancellation(cancelled);
            const auto scalar = table->column(column)->GetScalar(row);
            if (!scalar.ok()) throw std::runtime_error(scalar.status().ToString());
            const auto& value = scalar.ValueOrDie();
            auto& schema = page.schema[static_cast<size_t>(column)];
            ++schema.sampled_values;
            if (!value || !value->is_valid) {
                ++schema.sampled_nulls;
                values.emplace_back("<null>");
            } else {
                values.push_back(value->ToString());
            }
        }
        page.rows.push_back(std::move(values));
        task.ReportProgress(0.60f + 0.25f * static_cast<float>(row + 1) /
                                             static_cast<float>(table->num_rows()),
                            "Formatting HDF5 preview");
    }
    CheckCancellation(cancelled);
    page.rows_returned = static_cast<int64_t>(page.rows.size());
    page.next_offset = page.offset + page.rows_returned;
    page.has_next = page.next_offset < page.total_rows;
    page.ok = true;
    page.status = DataPreviewStatus::Ready;
    return page;
}

Hdf5InspectionResult Inspect(const Hdf5InspectionRequest& request, LambdaTask& task) {
    const std::function<bool()> cancelled = [&task, external = request.browse.cancel_requested] {
        return task.ShouldStop() || (external && external());
    };
    CheckCancellation(cancelled);
    if (request.kind != Hdf5InspectionKind::Hierarchy && request.kind != Hdf5InspectionKind::Preview)
        return Failure(Hdf5TableStatus::InvalidSelection, "Invalid HDF5 inspection kind");

    task.ReportProgress(0.05f, "Checking HDF5 source");
    CheckCancellation(cancelled);
    std::string stamp_error;
    const auto before = ReadHdf5SourceStamp(request.path, stamp_error);
    CheckCancellation(cancelled);
    if (!before)
        return request.expected_source ? ChangedSource() : Failure(Hdf5TableStatus::InvalidFile, stamp_error);
    if (request.expected_source && *request.expected_source != *before) return ChangedSource();

    Hdf5InspectionResult result;
    task.ReportProgress(0.15f, request.kind == Hdf5InspectionKind::Hierarchy
        ? "Browsing HDF5 group" : "Reading HDF5 preview");
    CheckCancellation(cancelled);
    if (request.kind == Hdf5InspectionKind::Hierarchy) {
        auto browse = request.browse;
        browse.cancel_requested = cancelled;
        result.hierarchy = HDF5Browser::BrowsePage(request.path, browse);
        result.status = result.hierarchy.status;
        result.error = result.hierarchy.error;
    } else {
        Hdf5TableReadOptions options;
        options.selection = request.selection;
        options.numeric_policy = Hdf5NumericPolicy::Preserve;
        options.max_materialized_bytes = 16ULL * 1024 * 1024;
        options.cancel_requested = cancelled;
        auto sampled = PreviewHdf5Table(request.path, options, request.window);
        result.status = sampled.status;
        result.error = sampled.error;
        if (sampled.status == Hdf5TableStatus::Ok) {
            task.ReportProgress(0.60f, "Formatting HDF5 preview");
            CheckCancellation(cancelled);
            result.preview = FormatPreview(sampled, task, cancelled);
        }
        result.data = std::move(sampled.data);
        result.labels = std::move(sampled.labels);
    }
    if (result.status == Hdf5TableStatus::Cancelled) throw InspectionCancelled{};
    CheckCancellation(cancelled);
    const auto after = ReadHdf5SourceStamp(request.path, stamp_error);
    CheckCancellation(cancelled);
    if (!after || *after != *before) return ChangedSource();
    result.source = after;
    task.ReportProgress(0.95f, "Verified HDF5 source");
    CheckCancellation(cancelled);
    return result;
}
#endif

} // namespace

std::shared_ptr<AsyncTask> MakeHdf5InspectionTask(
    Hdf5InspectionRequest request, std::shared_ptr<Hdf5InspectionTaskResult> state) {
    if (!state) throw std::invalid_argument("HDF5 inspection requires result storage");
    return std::make_shared<LambdaTask>("Inspect HDF5 source",
        [captured_request = std::move(request), captured_state = std::move(state)](LambdaTask& task) mutable {
            auto request = std::move(captured_request);
            auto state = std::move(captured_state);
            if (!state) return;
            try {
#ifdef CYXWIZ_HAS_HDF5
                state->result = Inspect(request, task);
                if (task.IsCancelRequested()) throw InspectionCancelled{};
#else
                state->result = Failure(Hdf5TableStatus::DependencyUnavailable,
                                        "HDF5 support is not compiled into this build");
#endif
            } catch (const InspectionCancelled&) {
                state->result = Failure(Hdf5TableStatus::Cancelled, "HDF5 inspection cancelled");
            } catch (const std::bad_alloc&) {
                state->result = Failure(Hdf5TableStatus::ResourceLimit, "HDF5 inspection could not allocate memory");
            } catch (const std::exception& error) {
                state->result = Failure(Hdf5TableStatus::ReadFailed,
                                        "HDF5 inspection failed: " + std::string(error.what()));
            } catch (...) {
                state->result = Failure(Hdf5TableStatus::ReadFailed, "HDF5 inspection failed with an unknown error");
            }
            if (state->result.status == Hdf5TableStatus::Cancelled) task.MarkCancelled("HDF5 inspection cancelled");
            else if (state->result.status == Hdf5TableStatus::Ok) task.MarkCompleted("HDF5 inspection ready");
            else task.MarkFailed(state->result.error.empty() ? "HDF5 inspection failed" : state->result.error);
            state->done.store(true, std::memory_order_release);
        }, true);
}

} // namespace cyxwiz
