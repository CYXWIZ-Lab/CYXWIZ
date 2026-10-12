#include "hdf5_apply_preparation.h"
#include "arrow_dataset.h"
#include "async_task_manager.h"

#include <new>
#include <stdexcept>
#include <utility>

namespace cyxwiz {
namespace {
struct Cancelled {};

Hdf5ApplyPreparationResult Failure(Hdf5TableStatus status, std::string error) {
    Hdf5ApplyPreparationResult result;
    result.status = status;
    result.error = std::move(error);
    return result;
}

Hdf5ApplyPreparationResult Prepare(Hdf5ApplyPreparationRequest request, LambdaTask& task) {
    auto external = std::move(request.source.cancel_requested);
    request.source.cancel_requested = [&task, external = std::move(external)] {
        return task.ShouldStop() || (external && external());
    };
    const auto check_cancelled = [&] {
        if (request.source.cancel_requested()) throw Cancelled{};
    };
    check_cancelled();
    if (request.dataset_name.empty())
        return Failure(Hdf5TableStatus::InvalidSelection, "HDF5 Apply requires a dataset name");

    auto loaded = PrepareHdf5Source(request.source, [&task](float value, const std::string& message) {
        task.ReportProgress(value * 0.9f, message);
    });
    check_cancelled();
    if (loaded.read.status != Hdf5TableStatus::Ok) {
        auto result = Failure(loaded.read.status, std::move(loaded.read.error));
        result.audit = std::move(loaded.audit);
        result.source_changed = loaded.source_changed;
        return result;
    }
    if (!loaded.source || !loaded.read.table)
        throw std::runtime_error("HDF5 preparation returned no source receipt or table");

    task.ReportProgress(0.92f, "Staging HDF5 publication");
    check_cancelled();
    Hdf5ApplyPreparationResult result;
    auto dataset = std::make_shared<ArrowDataset>(loaded.read.table, request.dataset_name);
    result.publication = DataRegistry::PrepareArrowTablePublication(
        request.publication, dataset, loaded.source->canonical_path, result.error);
    check_cancelled();
    if (!result.publication) return Failure(Hdf5TableStatus::ReadFailed, std::move(result.error));

    result.parameters = std::move(request.parameters);
    if (!WriteHdf5InputSettings(request.source.settings, result.parameters, result.error))
        return Failure(Hdf5TableStatus::InvalidSelection, std::move(result.error));
    result.parameters["dataset_name"] = request.dataset_name;
    result.parameters["configured"] = "true";
    result.parameters["data_loaded"] = "true";
    result.parameters["loaded_rows"] = std::to_string(loaded.read.table->num_rows());
    result.parameters["loaded_cols"] = std::to_string(loaded.read.table->num_columns());
    result.parameters["memory_bytes"] = std::to_string(dataset->GetMemoryUsage());
    result.parameters["label_column"] = request.source.settings.selection.label_path.empty() ? "" : "label";
    result.parameters.erase("dataset_role");

    // Staging and parameter allocation may take time. Verify after both, with
    // no user progress callback between this stamp and publishing the result.
    task.ReportProgress(0.98f, "Verifying staged HDF5 source");
    check_cancelled();
    std::string error;
    const auto final_source = ReadHdf5SourceStamp(request.source.path, error);
    check_cancelled();
    if (!final_source || *final_source != *loaded.source) {
        auto changed = Failure(Hdf5TableStatus::ReadFailed,
                              "HDF5 source changed during Apply preparation. Refresh and retry.");
        changed.source_changed = true;
        return changed;
    }
    result.source = std::move(loaded.source);
    result.audit = std::move(loaded.audit);
    result.status = Hdf5TableStatus::Ok;
    return result;
}
} // namespace

std::shared_ptr<AsyncTask> MakeHdf5ApplyPreparationTask(
    Hdf5ApplyPreparationRequest request, const std::shared_ptr<Hdf5ApplyPreparationState>& state) {
    if (!state) throw std::invalid_argument("HDF5 Apply preparation requires result storage");
    return std::make_shared<LambdaTask>("Prepare HDF5 Apply",
        [request = std::move(request), storage = std::weak_ptr<Hdf5ApplyPreparationState>(state)]
        (LambdaTask& task) mutable {
            auto state = storage.lock();
            if (!state) {
                task.MarkCancelled("HDF5 Apply preparation owner was released");
                return;
            }
            if (state->done.load(std::memory_order_acquire)) return;
            try {
                state->result = Prepare(std::move(request), task);
                if (task.ShouldStop()) throw Cancelled{};
            } catch (const Cancelled&) {
                state->result = Failure(Hdf5TableStatus::Cancelled, "HDF5 Apply preparation cancelled");
            } catch (const std::bad_alloc&) {
                state->result = Failure(Hdf5TableStatus::ResourceLimit, "HDF5 Apply preparation could not allocate memory");
            } catch (const std::exception& error) {
                state->result = Failure(Hdf5TableStatus::ReadFailed, error.what());
            } catch (...) {
                state->result = Failure(Hdf5TableStatus::ReadFailed, "HDF5 Apply preparation failed");
            }
            if (state->result.status == Hdf5TableStatus::Ok) task.MarkCompleted("HDF5 Apply prepared");
            else if (state->result.status == Hdf5TableStatus::Cancelled) task.MarkCancelled(state->result.error);
            else task.MarkFailed(state->result.error);
            state->done.store(true, std::memory_order_release);
        }, true);
}
} // namespace cyxwiz
