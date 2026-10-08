#include "hdf5_table_adapter.h"
#include "hdf5_object_path.h"
#include "hdf5_table_internal.h"

#ifdef CYXWIZ_HAS_HDF5
#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <highfive/highfive.hpp>

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#endif

namespace cyxwiz {

bool Hdf5TableSupportAvailable() {
#ifdef CYXWIZ_HAS_HDF5
    return true;
#else
    return false;
#endif
}

#ifdef CYXWIZ_HAS_HDF5
namespace {

using hdf5_detail::ReadFailure;

void CancelIfRequested(const Hdf5TableReadOptions& options) {
    if (options.cancel_requested && options.cancel_requested())
        throw ReadFailure(Hdf5TableStatus::Cancelled, "HDF5 table read was cancelled");
}

uint64_t CheckedProduct(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b)
        throw ReadFailure(Hdf5TableStatus::ResourceLimit, "HDF5 materialization size overflows the resource estimate");
    return a * b;
}

uint64_t CheckedSum(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b)
        throw ReadFailure(Hdf5TableStatus::ResourceLimit, "HDF5 materialization size overflows the resource estimate");
    return a + b;
}

void CheckArrow(const arrow::Status& status) {
    if (!status.ok())
        throw ReadFailure(status.IsOutOfMemory() ? Hdf5TableStatus::ResourceLimit : Hdf5TableStatus::ReadFailed,
            "HDF5 Arrow materialization failed: " + status.ToString());
}

} // namespace

HighFive::File hdf5_detail::OpenFile(const std::string& path) {
    if (path.find('\0') != std::string::npos)
        throw ReadFailure(Hdf5TableStatus::InvalidFile, "HDF5 file path contains a null byte");
    htri_t signature = -1;
    H5E_BEGIN_TRY {
        signature = H5Fis_hdf5(path.c_str());
    } H5E_END_TRY;
    if (signature <= 0)
        throw ReadFailure(Hdf5TableStatus::InvalidFile, "The selected file does not have a valid HDF5 signature: " + path);
    return HighFive::File(path, HighFive::File::ReadOnly);
}

void hdf5_detail::ValidatePathSyntax(const std::string& path, bool allow_root) {
    std::string error;
    if (!ValidateHdf5ObjectPath(path, error, allow_root))
        throw ReadFailure(Hdf5TableStatus::InvalidSelection, error);
}

void hdf5_detail::ValidateLocalPath(const HighFive::File& file, const std::string& path,
                                   bool allow_root) {
    ValidatePathSyntax(path, allow_root);
    // Inspect each component before following it, so a nested external or soft
    // link cannot bypass the local-file-only contract.
    for (size_t begin = 1; begin < path.size();) {
        auto end = path.find('/', begin);
        if (end == std::string::npos) end = path.size();
        const auto prefix = path.substr(0, end);
        if (!file.exist(prefix))
            throw ReadFailure(Hdf5TableStatus::MissingDataset, "Dataset '" + path + "' does not exist");
        H5L_info2_t link{};
        if (H5Lget_info2(file.getId(), prefix.c_str(), &link, H5P_DEFAULT) < 0)
            throw ReadFailure(Hdf5TableStatus::ReadFailed, "Cannot inspect HDF5 link '" + prefix + "'");
        if (link.type != H5L_TYPE_HARD)
            throw ReadFailure(Hdf5TableStatus::UnsupportedLayout, "Dataset '" + path + "' uses an unsupported indirect link");
        begin = end + 1;
    }
}

namespace {

HighFive::DataSet OpenDataset(const HighFive::File& file, const std::string& path) {
    hdf5_detail::ValidateLocalPath(file, path);
    if (file.getObjectType(path) != HighFive::ObjectType::Dataset)
        throw ReadFailure(Hdf5TableStatus::InvalidSelection, "Selected HDF5 object '" + path + "' is not a dataset");
    return file.getDataSet(path);
}

std::shared_ptr<arrow::DataType> NumericType(const HighFive::DataSet& dataset, const std::string& path) {
    const auto type = dataset.getDataType();
    const auto id = type.getId();
    const auto size = type.getSize();
    if (H5Tget_class(id) == H5T_INTEGER && H5Tget_precision(id) == size * 8 && H5Tget_offset(id) == 0) {
        const auto sign = H5Tget_sign(id);
        if (sign == H5T_SGN_NONE) {
            switch (size) {
                case 1: return arrow::uint8();
                case 2: return arrow::uint16();
                case 4: return arrow::uint32();
                case 8: return arrow::uint64();
            }
        } else if (sign == H5T_SGN_2) {
            switch (size) {
                case 1: return arrow::int8();
                case 2: return arrow::int16();
                case 4: return arrow::int32();
                case 8: return arrow::int64();
            }
        }
    }
    if (H5Tget_class(id) == H5T_FLOAT) {
        if (H5Tequal(id, H5T_IEEE_F32LE) > 0 || H5Tequal(id, H5T_IEEE_F32BE) > 0) return arrow::float32();
        if (H5Tequal(id, H5T_IEEE_F64LE) > 0 || H5Tequal(id, H5T_IEEE_F64BE) > 0) return arrow::float64();
    }
    throw ReadFailure(Hdf5TableStatus::UnsupportedType,
        "Dataset '" + path + "' has an unsupported dtype; expected a primitive integer or IEEE float32/float64");
}

void Describe(const HighFive::DataSet& dataset, const std::string& path, bool label,
              Hdf5TableDatasetInfo& info) {
    const auto dimensions = dataset.getDimensions();
    info.path = path;
    info.shape.assign(dimensions.begin(), dimensions.end());
    const auto type = dataset.getDataType();
    switch (H5Tget_class(type.getId())) {
        case H5T_STRING: info.source_type = "string"; break;
        case H5T_COMPOUND: info.source_type = "compound"; break;
        case H5T_VLEN: info.source_type = "vlen"; break;
        default: info.source_type = "unsupported"; break;
    }
    // Preserve inspectable metadata even when the selected layout cannot load.
    try { info.source_type = NumericType(dataset, path)->ToString(); }
    catch (const ReadFailure& error) {
        if (error.status != Hdf5TableStatus::UnsupportedType) throw;
    }
    const auto properties = dataset.getCreatePropertyList();
    if (H5Pget_layout(properties.getId()) == H5D_VIRTUAL ||
        H5Pget_external_count(properties.getId()) != 0)
        throw ReadFailure(Hdf5TableStatus::UnsupportedLayout, "Dataset '" + path + "' uses virtual or external storage");
    if (dimensions.empty() || dimensions.size() > (label ? 1U : 2U))
        throw ReadFailure(Hdf5TableStatus::UnsupportedRank, "Dataset '" + path + "' has rank " +
            std::to_string(dimensions.size()) + (label ? "; labels require rank 1" : "; numeric table mode supports rank 1 or 2"));
    if (std::find(dimensions.begin(), dimensions.end(), 0) != dimensions.end())
        throw ReadFailure(Hdf5TableStatus::EmptyDataset, "Dataset '" + path + "' is empty");
    (void)NumericType(dataset, path);
}

struct ReadWindow {
    uint64_t row_offset = 0;
    uint64_t rows = 0;
    uint64_t column_offset = 0;
    uint64_t columns = 0;
};

uint64_t DecodedChunkBytes(const HighFive::DataSet& dataset) {
    const auto properties = dataset.getCreatePropertyList();
    const auto layout = H5Pget_layout(properties.getId());
    if (layout == H5D_LAYOUT_ERROR)
        throw ReadFailure(Hdf5TableStatus::ReadFailed, "Cannot inspect HDF5 storage layout");
    if (layout != H5D_CHUNKED) return 0;
    std::array<hsize_t, 2> dimensions{};
    const int rank = H5Pget_chunk(properties.getId(), static_cast<int>(dimensions.size()), dimensions.data());
    if (rank < 1 || rank > static_cast<int>(dimensions.size()))
        throw ReadFailure(Hdf5TableStatus::ReadFailed, "Cannot inspect HDF5 chunk dimensions");
    uint64_t bytes = dataset.getDataType().getSize();
    for (int i = 0; i < rank; ++i) bytes = CheckedProduct(bytes, dimensions[i]);
    return bytes;
}

template <typename ArrowType>
void ReadColumns(const HighFive::DataSet& dataset, const Hdf5TableDatasetInfo& info,
                 bool labels, const Hdf5TableReadOptions& options, const ReadWindow& window,
                 std::vector<std::shared_ptr<arrow::Field>>& fields,
                 std::vector<std::shared_ptr<arrow::Array>>& arrays) {
    using Value = typename ArrowType::c_type;
    const auto rows = window.rows;
    const auto columns = window.columns;
    std::vector<Value> scratch(static_cast<size_t>(std::min<uint64_t>(rows, 65536)));
    for (uint64_t index = 0; index < columns; ++index) {
        const auto column = window.column_offset + index;
        CancelIfRequested(options);
        arrow::NumericBuilder<ArrowType> builder;
        CheckArrow(builder.Reserve(static_cast<int64_t>(rows)));
        for (uint64_t row = 0; row < rows;) {
            CancelIfRequested(options);
            const size_t count = static_cast<size_t>(std::min<uint64_t>(rows - row, scratch.size()));
            if (info.shape.size() == 1)
                dataset.select({static_cast<size_t>(window.row_offset + row)}, {count}).read_raw(scratch.data());
            else
                dataset.select({static_cast<size_t>(window.row_offset + row), static_cast<size_t>(column)}, {count, 1}).read_raw(scratch.data());
            CheckArrow(builder.AppendValues(scratch.data(), static_cast<int64_t>(count)));
            row += count;
        }
        std::shared_ptr<arrow::Array> array;
        CheckArrow(builder.Finish(&array));
        const auto name = labels ? "label" : (info.shape.size() == 1 ? "value" : "col_" + std::to_string(column));
        fields.push_back(arrow::field(name, array->type(), false));
        arrays.push_back(std::move(array));
    }
}

void Materialize(const HighFive::DataSet& dataset, const Hdf5TableDatasetInfo& info,
                 bool labels, const Hdf5TableReadOptions& options, const ReadWindow& window,
                 std::vector<std::shared_ptr<arrow::Field>>& fields,
                 std::vector<std::shared_ptr<arrow::Array>>& arrays) {
    const auto type = options.numeric_policy == Hdf5NumericPolicy::Float64
        ? arrow::float64() : NumericType(dataset, info.path);
    switch (type->id()) {
        case arrow::Type::INT8: return ReadColumns<arrow::Int8Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::INT16: return ReadColumns<arrow::Int16Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::INT32: return ReadColumns<arrow::Int32Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::INT64: return ReadColumns<arrow::Int64Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::UINT8: return ReadColumns<arrow::UInt8Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::UINT16: return ReadColumns<arrow::UInt16Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::UINT32: return ReadColumns<arrow::UInt32Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::UINT64: return ReadColumns<arrow::UInt64Type>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::FLOAT: return ReadColumns<arrow::FloatType>(dataset, info, labels, options, window, fields, arrays);
        case arrow::Type::DOUBLE: return ReadColumns<arrow::DoubleType>(dataset, info, labels, options, window, fields, arrays);
        default: throw ReadFailure(Hdf5TableStatus::UnsupportedType, "Unsupported HDF5 numeric type");
    }
}

} // namespace
#endif

namespace {

Hdf5TableReadResult ReadTable(const std::string& path, const Hdf5TableReadOptions& options,
                            const Hdf5TablePreviewRequest* preview, bool probe) {
    Hdf5TableReadResult result;
#ifdef CYXWIZ_HAS_HDF5
    try {
        CancelIfRequested(options);
        if (options.numeric_policy != Hdf5NumericPolicy::Preserve && options.numeric_policy != Hdf5NumericPolicy::Float64)
            throw ReadFailure(Hdf5TableStatus::InvalidSelection, "Unsupported HDF5 numeric coercion policy");
        if (preview && (preview->row_limit == 0 || preview->row_limit > 200 ||
                        preview->column_limit == 0 || preview->column_limit > 64))
            throw ReadFailure(Hdf5TableStatus::InvalidSelection, "HDF5 preview requires 1-200 rows and 1-64 data columns");
        const auto file = hdf5_detail::OpenFile(path);
        auto data = OpenDataset(file, options.selection.data_path);
        Describe(data, options.selection.data_path, false, result.data);
        std::optional<HighFive::DataSet> labels;
        if (!options.selection.label_path.empty()) {
            labels = OpenDataset(file, options.selection.label_path);
            result.labels.emplace();
            Describe(*labels, options.selection.label_path, true, *result.labels);
            if (result.labels->shape[0] != result.data.shape[0])
                throw ReadFailure(Hdf5TableStatus::LabelRowMismatch, "Label dataset '" + result.labels->path +
                    "' has " + std::to_string(result.labels->shape[0]) + " rows; data dataset has " + std::to_string(result.data.shape[0]));
        }
        ReadWindow window{0, result.data.shape[0], 0, result.data.shape.size() == 1 ? 1 : result.data.shape[1]};
        if (window.rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            CheckedSum(window.columns, labels ? 1 : 0) > 4096)
            throw ReadFailure(Hdf5TableStatus::ResourceLimit, "HDF5 table exceeds the supported row range or 4096-column limit");
        if (preview) {
            if (preview->row_offset > window.rows || preview->column_offset >= window.columns)
                throw ReadFailure(Hdf5TableStatus::InvalidSelection, "HDF5 preview offset is outside the selected dataset");
            window.row_offset = preview->row_offset;
            window.rows = std::min(window.rows - window.row_offset, preview->row_limit);
            window.column_offset = preview->column_offset;
            window.columns = std::min(window.columns - window.column_offset, preview->column_limit);
        }
        result.row_offset = window.row_offset;
        result.column_offset = window.column_offset;
        const auto rows = window.rows;
        const auto columns = CheckedSum(window.columns, labels ? 1 : 0);
        result.estimated_materialized_bytes = CheckedSum(
            CheckedSum(CheckedProduct(CheckedProduct(rows, columns), 16), CheckedProduct(columns, 256)),
            CheckedProduct(std::min<uint64_t>(rows, 65536), 8));
        if (preview && rows != 0) {
            // A one-cell hyperslab may still require decoding a whole chunk.
            result.estimated_materialized_bytes = CheckedSum(result.estimated_materialized_bytes,
                                                              DecodedChunkBytes(data));
            if (labels) result.estimated_materialized_bytes = CheckedSum(
                result.estimated_materialized_bytes, DecodedChunkBytes(*labels));
        }
        if (result.estimated_materialized_bytes > options.max_materialized_bytes)
            throw ReadFailure(Hdf5TableStatus::ResourceLimit, "The estimated HDF5 materialization exceeds the configured memory policy");
        CancelIfRequested(options);
        if (probe) {
            result.status = Hdf5TableStatus::Ok;
            return result;
        }
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        fields.reserve(static_cast<size_t>(columns));
        arrays.reserve(static_cast<size_t>(columns));
        Materialize(data, result.data, false, options, window, fields, arrays);
        if (labels) Materialize(*labels, *result.labels, true, options,
                               {window.row_offset, window.rows, 0, 1}, fields, arrays);
        CancelIfRequested(options);
        const auto metadata = arrow::key_value_metadata(
            {"hdf5.data_path", "hdf5.label_path", "hdf5.numeric_policy", "label_column"},
            {result.data.path, options.selection.label_path,
             options.numeric_policy == Hdf5NumericPolicy::Preserve ? "preserve" : "float64", labels ? "label" : ""});
        result.table = arrow::Table::Make(arrow::schema(fields, metadata), arrays);
        result.status = Hdf5TableStatus::Ok;
    } catch (const ReadFailure& error) {
        result.status = error.status;
        result.error = error.what();
    } catch (const std::bad_alloc&) {
        result.status = Hdf5TableStatus::ResourceLimit;
        result.error = "HDF5 materialization could not allocate the required memory";
    } catch (const std::exception& error) {
        result.status = Hdf5TableStatus::ReadFailed;
        result.error = "HDF5 table read failed: " + std::string(error.what());
    }
#else
    (void)path;
    (void)options;
    (void)preview;
    (void)probe;
    result.status = Hdf5TableStatus::DependencyUnavailable;
    result.error = "HDF5 support is not compiled into this build";
#endif
    return result;
}

} // namespace

Hdf5TableReadResult ReadHdf5Table(const std::string& path, const Hdf5TableReadOptions& options) {
    return ReadTable(path, options, nullptr, false);
}

Hdf5TableReadResult ProbeHdf5Table(const std::string& path, const Hdf5TableReadOptions& options) {
    return ReadTable(path, options, nullptr, true);
}

Hdf5TableReadResult PreviewHdf5Table(const std::string& path, const Hdf5TableReadOptions& options,
                                  const Hdf5TablePreviewRequest& request) {
    return ReadTable(path, options, &request, false);
}

} // namespace cyxwiz
