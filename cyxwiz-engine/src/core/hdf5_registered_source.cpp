#include "hdf5_registered_source.h"
#include "arrow_dataset.h"
#include "hdf5_source_identity.h"

#include <arrow/util/key_value_metadata.h>
#include <new>
#include <stdexcept>
#include <utility>

namespace cyxwiz {
namespace {
struct Cancelled {};

Hdf5RegisteredSourceResult Failure(Hdf5TableStatus status, std::string error) {
    Hdf5RegisteredSourceResult result;
    result.status = status;
    result.error = std::move(error);
    return result;
}
} // namespace

Hdf5RegisteredSourceResult VerifyHdf5RegisteredSource(
    const Hdf5RegisteredSourceRequest& request, const std::function<bool()>& cancelled) {
    const auto check_cancelled = [&] { if (cancelled && cancelled()) throw Cancelled{}; };
    try {
        check_cancelled();
        std::string error;
        if (!ValidateHdf5InputSettings(request.settings, error))
            return Failure(Hdf5TableStatus::InvalidSelection, std::move(error));
        if (request.dataset_name.empty() || !request.dataset ||
            request.dataset->GetName() != request.dataset_name)
            return Failure(Hdf5TableStatus::ReadFailed, "No matching HDF5 Arrow registration");
        const auto table = request.dataset->GetArrowTable();
        if (!table || table->num_rows() <= 0 || table->num_columns() <= 0 || !table->schema()->metadata())
            return Failure(Hdf5TableStatus::ReadFailed, "Registered table has no HDF5 source identity");
        const auto before = ReadHdf5SourceStamp(request.resolved_path, error);
        check_cancelled();
        if (!before) return Failure(Hdf5TableStatus::InvalidFile, std::move(error));
        if (request.registered_source_path != before->canonical_path)
            return Failure(Hdf5TableStatus::ReadFailed, "Registered HDF5 source path does not match the saved source");

        const auto metadata = table->schema()->metadata();
        const auto expect = [&](const char* key, const std::string& expected) {
            check_cancelled();
            int64_t count = 0;
            bool matches = false;
            for (int64_t i = 0; i < metadata->size(); ++i) {
                if (metadata->key(i) == key) {
                    ++count;
                    matches = metadata->value(i) == expected;
                }
            }
            if (count != 1 || !matches)
                throw std::runtime_error(std::string("Registered HDF5 identity mismatch: ") + key);
        };
        expect("hdf5.source_path", before->canonical_path);
        expect("hdf5.source_size", std::to_string(before->size));
        expect("hdf5.source_modified", std::to_string(before->modified));
        expect("hdf5.import_mode", "numeric_table");
        expect("hdf5.data_path", request.settings.selection.data_path);
        expect("hdf5.label_path", request.settings.selection.label_path);
        expect("hdf5.numeric_policy", request.settings.numeric_policy == Hdf5NumericPolicy::Preserve
            ? "preserve" : "float64");
        expect("hdf5.max_materialized_bytes", std::to_string(request.settings.max_materialized_bytes));
        expect("label_column", request.settings.selection.label_path.empty() ? "" : "label");
        const bool has_label = table->schema()->GetFieldIndex("label") >= 0;
        if (has_label != !request.settings.selection.label_path.empty())
            return Failure(Hdf5TableStatus::ReadFailed, "Registered HDF5 label column does not match the selection");

        Hdf5RegisteredSourceResult result;
        result.rows = table->num_rows();
        result.columns = table->num_columns();
        result.bytes = request.dataset->GetMemoryUsage();
        check_cancelled();
        const auto after = ReadHdf5SourceStamp(request.resolved_path, error);
        check_cancelled();
        if (!after || *before != *after)
            return Failure(Hdf5TableStatus::ReadFailed, "HDF5 source changed during loaded-state verification");
        result.status = Hdf5TableStatus::Ok;
        return result;
    } catch (const Cancelled&) {
        return Failure(Hdf5TableStatus::Cancelled, "HDF5 loaded-state verification cancelled");
    } catch (const std::bad_alloc&) {
        return Failure(Hdf5TableStatus::ResourceLimit, "HDF5 loaded-state verification could not allocate memory");
    } catch (const std::exception& error) {
        return Failure(Hdf5TableStatus::ReadFailed, error.what());
    } catch (...) {
        return Failure(Hdf5TableStatus::ReadFailed, "HDF5 loaded-state verification failed");
    }
}
} // namespace cyxwiz
