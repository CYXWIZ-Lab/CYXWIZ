#include "../src/core/hdf5_table_adapter.h"

#ifdef CYXWIZ_HAS_HDF5
#include <arrow/api.h>
#include <arrow/util/key_value_metadata.h>
#include <highfive/highfive.hpp>
#endif

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

int checks = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void CheckFailure(const cyxwiz::Hdf5TableReadResult& result,
                  cyxwiz::Hdf5TableStatus expected, const std::string& context) {
    Check(result.status == expected, context + ": unexpected status: " + result.error);
    Check(!result.table, context + ": failed reads must not return a table");
    Check(!result.error.empty(), context + ": failure must explain the cause");
}

#ifdef CYXWIZ_HAS_HDF5
struct Workspace {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cyxwiz_hdf5_adapter_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Workspace() { Check(std::filesystem::create_directory(path), "unique workspace"); }
    ~Workspace() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

template <typename T>
void WriteVector(HighFive::File& file, const std::string& path,
                 const std::vector<T>& values) {
    file.createDataSet<T>(path, HighFive::DataSpace::From(values)).write(values);
}

template <typename Scalar>
typename Scalar::ValueType Cell(const std::shared_ptr<arrow::Table>& table,
                               int column, int64_t row) {
    auto scalar = table->column(column)->GetScalar(row);
    Check(scalar.ok(), "cell lookup should succeed");
    auto typed = std::dynamic_pointer_cast<Scalar>(scalar.ValueOrDie());
    Check(typed && typed->is_valid, "cell must have the expected numeric type and value");
    return typed->value;
}

std::shared_ptr<arrow::Table> CheckTable(const cyxwiz::Hdf5TableReadResult& result,
                                       int64_t rows, int columns) {
    Check(result.status == cyxwiz::Hdf5TableStatus::Ok && result.table,
          "read should succeed: " + result.error);
    Check(result.error.empty(), "successful read should not report an error");
    Check(result.table->num_rows() == rows && result.table->num_columns() == columns,
          "selected table shape");
    Check(result.table->ValidateFull().ok(), "materialized Arrow table must be valid");
    return result.table;
}

template <typename T, typename Scalar>
void CheckPrimitive(HighFive::File& file, const std::string& path,
                    arrow::Type::type type, T first, T last) {
    WriteVector<T>(file, path, {first, last});
    file.flush();
    cyxwiz::Hdf5TableReadOptions options;
    options.selection.data_path = path;
    auto result = cyxwiz::ReadHdf5Table(file.getName(), options);
    auto table = CheckTable(result, 2, 1);
    Check(table->field(0)->type()->id() == type, path + ": preserve primitive width/sign");
    Check(Cell<Scalar>(table, 0, 0) == first && Cell<Scalar>(table, 0, 1) == last,
          path + ": preserve numeric values");
}
#endif

} // namespace

int main() try {
    using cyxwiz::Hdf5TableStatus;
    cyxwiz::Hdf5TableReadOptions options;
    Check(options.numeric_policy == cyxwiz::Hdf5NumericPolicy::Preserve,
          "native numeric preservation is the default");
    Check(options.max_materialized_bytes == 256ULL * 1024 * 1024,
          "default materialization budget is 256 MiB");
#ifdef CYXWIZ_HAS_HDF5
    Check(cyxwiz::Hdf5TableSupportAvailable(), "compiled HDF5 support is available");
    Workspace workspace;
    const auto path = (workspace.path / "input.h5").string();
    const uint64_t large = 9007199254740993ULL;
    {
        HighFive::File file(path, HighFive::File::Overwrite);
        file.createGroup("/nested");
        const std::vector<std::vector<int32_t>> matrix{{1, -2}, {3, -4}, {5, -6}};
        file.createDataSet<int32_t>("/nested/data", HighFive::DataSpace::From(matrix)).write(matrix);
        WriteVector<uint8_t>(file, "/nested/labels", {2, 1, 0});
        WriteVector<uint64_t>(file, "/nested/large", {large, large + 2, std::numeric_limits<uint64_t>::max()});
        WriteVector<int32_t>(file, "/short_labels", {0, 1});
        WriteVector<std::string>(file, "/strings", {"one", "two", "three"});
        file.createDataSet<int32_t>("/scalar", HighFive::DataSpace::From(int32_t{7})).write(int32_t{7});
        file.createDataSet<int32_t>("/rank3", HighFive::DataSpace(std::vector<size_t>{1, 2, 3}));
        file.createDataSet<int32_t>("/empty", HighFive::DataSpace(std::vector<size_t>{0}));
        file.createDataSet<int32_t>("/empty_columns", HighFive::DataSpace(std::vector<size_t>{3, 0}));
        file.createDataSet<int32_t>("/too_wide", HighFive::DataSpace(std::vector<size_t>{1, 4097}));
        WriteVector<int32_t>(file, "/slabs", std::vector<int32_t>(131073, 42));
        Check(H5Lcreate_hard(file.getId(), "/nested/data", file.getId(), "/hard_data",
                            H5P_DEFAULT, H5P_DEFAULT) >= 0, "create hard link");
        Check(H5Lcreate_soft("/nested/data", file.getId(), "/soft_data",
                            H5P_DEFAULT, H5P_DEFAULT) >= 0, "create soft dataset link");
        Check(H5Lcreate_soft("/nested", file.getId(), "/soft_group",
                            H5P_DEFAULT, H5P_DEFAULT) >= 0, "create soft group link");
        Check(H5Lcreate_external(path.c_str(), "/nested/data", file.getId(), "/external_data",
                                H5P_DEFAULT, H5P_DEFAULT) >= 0, "create external dataset link");
        CheckPrimitive<int8_t, arrow::Int8Scalar>(file, "/i8", arrow::Type::INT8, -128, 127);
        CheckPrimitive<int16_t, arrow::Int16Scalar>(file, "/i16", arrow::Type::INT16, -30000, 30000);
        CheckPrimitive<int32_t, arrow::Int32Scalar>(file, "/i32", arrow::Type::INT32, -100000, 100000);
        CheckPrimitive<int64_t, arrow::Int64Scalar>(file, "/i64", arrow::Type::INT64, -9007199254740993LL, 9007199254740993LL);
        CheckPrimitive<uint8_t, arrow::UInt8Scalar>(file, "/u8", arrow::Type::UINT8, 0, 255);
        CheckPrimitive<uint16_t, arrow::UInt16Scalar>(file, "/u16", arrow::Type::UINT16, 0, 65535);
        CheckPrimitive<uint32_t, arrow::UInt32Scalar>(file, "/u32", arrow::Type::UINT32, 0, 4000000000U);
        CheckPrimitive<float, arrow::FloatScalar>(file, "/f32", arrow::Type::FLOAT, -1.25f, 2.5f);
        CheckPrimitive<double, arrow::DoubleScalar>(file, "/f64", arrow::Type::DOUBLE, -1.25, 2.5);
    }

    options.selection = {"/nested/data", ""};
    auto matrix_result = cyxwiz::ReadHdf5Table(path, options);
    auto matrix = CheckTable(matrix_result, 3, 2);
    Check(matrix_result.data.path == "/nested/data" &&
              matrix_result.data.shape == std::vector<uint64_t>({3, 2}) && !matrix_result.labels,
          "rank-2 selection metadata");
    for (int64_t row = 0; row < 3; ++row) {
        Check(Cell<arrow::Int32Scalar>(matrix, 0, row) == 2 * row + 1 &&
                  Cell<arrow::Int32Scalar>(matrix, 1, row) == -(2 * row + 2),
              "rank-2 values must retain row/column orientation");
    }
    options.selection.data_path = "/hard_data";
    auto hard = CheckTable(cyxwiz::ReadHdf5Table(path, options), 3, 2);
    Check(hard->Equals(*matrix, false), "hard-linked datasets are supported");

    options.selection = {"/nested/large", "/nested/labels"};
    auto labeled_result = cyxwiz::ReadHdf5Table(path, options);
    auto labeled = CheckTable(labeled_result, 3, 2);
    Check(labeled_result.data.shape == std::vector<uint64_t>({3}) &&
              labeled_result.labels && labeled_result.labels->path == "/nested/labels" &&
              labeled_result.labels->shape == std::vector<uint64_t>({3}), "rank-1 and label metadata");
    Check(labeled->field(0)->type()->id() == arrow::Type::UINT64 &&
              labeled->field(1)->name() == "label" && labeled->field(1)->type()->id() == arrow::Type::UINT8,
          "labels append last and preserve their native type");
    Check(Cell<arrow::UInt64Scalar>(labeled, 0, 0) == large &&
              Cell<arrow::UInt64Scalar>(labeled, 0, 1) == large + 2 &&
              Cell<arrow::UInt64Scalar>(labeled, 0, 2) == std::numeric_limits<uint64_t>::max(),
          "uint64 values above float64 exact range must remain exact");
    Check(Cell<arrow::UInt8Scalar>(labeled, 1, 0) == 2 &&
              Cell<arrow::UInt8Scalar>(labeled, 1, 2) == 0, "labels retain row alignment");
    const auto metadata = labeled->schema()->metadata();
    Check(metadata != nullptr, "HDF5 selection metadata is attached to the schema");
    for (const auto& entry : {std::pair<const char*, const char*>{"hdf5.data_path", "/nested/large"},
                              {"hdf5.label_path", "/nested/labels"},
                              {"hdf5.numeric_policy", "preserve"}, {"label_column", "label"}}) {
        auto value = metadata->Get(entry.first);
        Check(value.ok() && value.ValueOrDie() == entry.second, "HDF5 schema metadata value");
    }
    options.numeric_policy = cyxwiz::Hdf5NumericPolicy::Float64;
    auto compatible = CheckTable(cyxwiz::ReadHdf5Table(path, options), 3, 2);
    Check(compatible->field(0)->type()->id() == arrow::Type::DOUBLE &&
              compatible->field(1)->type()->id() == arrow::Type::DOUBLE, "explicit float64 converts data and labels");
    Check(Cell<arrow::DoubleScalar>(compatible, 0, 0) == static_cast<double>(large) &&
              Cell<arrow::DoubleScalar>(compatible, 1, 0) == 2.0, "float64 compatibility permits rounding");
    auto policy = compatible->schema()->metadata()->Get("hdf5.numeric_policy");
    Check(policy.ok() && policy.ValueOrDie() == "float64", "metadata records explicit float64 coercion");

    options = {};
    options.selection = {"/nested/data", "/nested/labels"};
    CheckTable(cyxwiz::ReadHdf5Table(path, options), 3, 3);
    for (const auto* invalid : {"", "nested/data", "/", "/nested/data/", "/nested//data", "/nested/../data"}) {
        options.selection = {invalid, ""};
        CheckFailure(cyxwiz::ReadHdf5Table(path, options), Hdf5TableStatus::InvalidSelection, invalid);
    }
    struct Rejection { const char* path; Hdf5TableStatus status; };
    for (const auto& invalid : {
             Rejection{"/missing", Hdf5TableStatus::MissingDataset},
             Rejection{"/nested", Hdf5TableStatus::InvalidSelection},
             Rejection{"/scalar", Hdf5TableStatus::UnsupportedRank},
             Rejection{"/rank3", Hdf5TableStatus::UnsupportedRank},
             Rejection{"/strings", Hdf5TableStatus::UnsupportedType},
             Rejection{"/empty", Hdf5TableStatus::EmptyDataset},
             Rejection{"/empty_columns", Hdf5TableStatus::EmptyDataset},
             Rejection{"/too_wide", Hdf5TableStatus::ResourceLimit},
             Rejection{"/soft_data", Hdf5TableStatus::UnsupportedLayout},
             Rejection{"/soft_group/data", Hdf5TableStatus::UnsupportedLayout},
             Rejection{"/external_data", Hdf5TableStatus::UnsupportedLayout}}) {
        options.selection = {invalid.path, ""};
        CheckFailure(cyxwiz::ReadHdf5Table(path, options), invalid.status, invalid.path);
    }
    for (const auto& invalid : {
             Rejection{"nested/labels", Hdf5TableStatus::InvalidSelection},
             Rejection{"/missing", Hdf5TableStatus::MissingDataset},
             Rejection{"/short_labels", Hdf5TableStatus::LabelRowMismatch},
             Rejection{"/nested/data", Hdf5TableStatus::UnsupportedRank},
             Rejection{"/scalar", Hdf5TableStatus::UnsupportedRank},
             Rejection{"/strings", Hdf5TableStatus::UnsupportedType}}) {
        options.selection = {"/nested/data", invalid.path};
        CheckFailure(cyxwiz::ReadHdf5Table(path, options), invalid.status, std::string("label ") + invalid.path);
    }

    options.selection = {"/nested/data", ""};
    const auto bad = (workspace.path / "invalid.h5").string();
    { std::ofstream out(bad, std::ios::binary); out << "not an HDF5 file"; }
    CheckFailure(cyxwiz::ReadHdf5Table(bad, options), Hdf5TableStatus::InvalidFile, "invalid signature");
    CheckFailure(cyxwiz::ReadHdf5Table((workspace.path / "absent.h5").string(), options),
                 Hdf5TableStatus::InvalidFile, "missing file");
    const uint64_t payload = 3 * 2 * sizeof(int32_t);
    Check(matrix_result.estimated_materialized_bytes > payload, "estimate includes scratch beyond Arrow payload");
    for (uint64_t budget : {uint64_t{0}, uint64_t{1}, payload, matrix_result.estimated_materialized_bytes - 1}) {
        options.max_materialized_bytes = budget;
        CheckFailure(cyxwiz::ReadHdf5Table(path, options), Hdf5TableStatus::ResourceLimit, "materialization budget");
    }
    options.max_materialized_bytes = matrix_result.estimated_materialized_bytes;
    CheckTable(cyxwiz::ReadHdf5Table(path, options), 3, 2);
    options.selection.label_path = "/nested/labels";
    CheckFailure(cyxwiz::ReadHdf5Table(path, options), Hdf5TableStatus::ResourceLimit, "labels count toward budget");

    options = {};
    options.selection.data_path = "/slabs";
    options.cancel_requested = [] { return true; };
    CheckFailure(cyxwiz::ReadHdf5Table(path, options), Hdf5TableStatus::Cancelled, "cancel before load");
    int polls = 0;
    // Allow validation and the first slab, then cancel a subsequent slab.
    options.cancel_requested = [&] { return ++polls >= 5; };
    CheckFailure(cyxwiz::ReadHdf5Table(path, options), Hdf5TableStatus::Cancelled, "cancel during load");
    Check(polls >= 5, "reader polls cancellation during materialization");
    options.cancel_requested = [] { return false; };
    auto slabs = CheckTable(cyxwiz::ReadHdf5Table(path, options), 131073, 1);
    Check(Cell<arrow::Int32Scalar>(slabs, 0, 131072) == 42, "final partial slab is retained");
#else
    Check(!cyxwiz::Hdf5TableSupportAvailable(), "disabled HDF5 support is reported");
    CheckFailure(cyxwiz::ReadHdf5Table("unavailable.h5", options),
                 Hdf5TableStatus::DependencyUnavailable, "disabled adapter");
#endif
    std::cout << "HDF5 adapter: " << checks << " checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
