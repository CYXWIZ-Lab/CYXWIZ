// A pipeline run loads an HDF5 Data Input (TOFIX189): one rank-2 numeric
// dataset as columns col_0.., its rank-1 labels as "label", and a clear refusal
// for a path the file does not have. Linked into test_data_convert_pipeline_input.
#include "core/arrow_dataset.h"
#include "core/data_registry.h"
#include "core/hdf5_input_settings.h"
#include "core/pipeline_executor.h"

#include <arrow/api.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef CYXWIZ_HAS_HDF5
#include "core/gcc_diagnostics.h"
// HighFive's own path constructor: GCC -Wuninitialized false positive (TOFIX187).
CYXWIZ_GCC_IGNORE_BEGIN("-Wuninitialized")
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
CYXWIZ_GCC_IGNORE_END
#endif

namespace fs = std::filesystem;

namespace {

void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

#ifdef CYXWIZ_HAS_HDF5
std::string Graph(const fs::path& file, const std::string& data_path, const std::string& label_path) {
    cyxwiz::Hdf5InputSettings settings;
    settings.selection.data_path = data_path;
    settings.selection.label_path = label_path;
    std::map<std::string, std::string> parameters{
        {"source_type", "file"}, {"file_path", file.string()}, {"type", "hdf5"}};
    std::string error;
    Require(cyxwiz::WriteHdf5InputSettings(settings, parameters, error), "HDF5 settings: " + error);
    nlohmann::json graph;
    graph["nodes"] = nlohmann::json::array(
        {{{"id", 1}, {"type", "DataInput"}, {"name", "Input"}, {"parameters", parameters}}});
    graph["links"] = nlohmann::json::array();
    return graph.dump();
}

double Value(const std::shared_ptr<arrow::Table>& table, const std::string& column, int64_t row) {
    const auto chunked = table->GetColumnByName(column);
    Require(chunked != nullptr, "column " + column + " is missing");
    const auto scalar = chunked->GetScalar(row);
    Require(scalar.ok(), "column " + column + " has no row " + std::to_string(row));
    const auto as_double = (*scalar)->CastTo(arrow::float64());
    Require(as_double.ok(), "column " + column + " is not numeric");
    return std::static_pointer_cast<arrow::DoubleScalar>(*as_double)->value;
}
#endif

}  // namespace

void RunDataInputHdf5PipelineTests(const fs::path& work_dir) {
#ifdef CYXWIZ_HAS_HDF5
    const fs::path file = work_dir / "measurements.h5";
    const std::vector<std::vector<float>> features = {
        {0.5f, 1.0f, -2.0f}, {1.5f, 2.0f, -1.0f}, {2.5f, 3.0f, 0.0f}, {3.5f, 4.0f, 1.0f}};
    const std::vector<int32_t> labels = {0, 1, 1, 0};
    {
        HighFive::File h5(file.string(), HighFive::File::Overwrite);
        h5.createDataSet<float>("/train/features", HighFive::DataSpace::From(features)).write(features);
        h5.createDataSet<int32_t>("/train/labels", HighFive::DataSpace::From(labels)).write(labels);
    }

    cyxwiz::PipelineExecutor executor;
    executor.SetIngestionCacheRoot((work_dir / "ingestion_hdf5").string());
    Require(executor.ExecutePipeline(Graph(file, "/train/features", "/train/labels")),
            "an HDF5 Data Input should run: " + executor.GetLastError());
    const auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset("ds_datainput_1");
    Require(dataset != nullptr, "the HDF5 Data Input should register its table");
    const auto table = dataset->GetArrowTable();
    Require(table->num_rows() == 4 && table->num_columns() == 4,
            "4 rows of 3 features plus the label, got " + std::to_string(table->num_rows()) + " x " +
                std::to_string(table->num_columns()));
    Require(table->schema()->field(3)->name() == "label", "the labels come last, as \"label\"");
    for (size_t row = 0; row < features.size(); ++row) {
        for (size_t column = 0; column < 3; ++column) {
            Require(Value(table, "col_" + std::to_string(column), static_cast<int64_t>(row)) ==
                        features[row][column],
                    "value at row " + std::to_string(row) + ", column " + std::to_string(column));
        }
        Require(Value(table, "label", static_cast<int64_t>(row)) == labels[row],
                "label at row " + std::to_string(row));
    }

    cyxwiz::PipelineExecutor missing;
    Require(!missing.ExecutePipeline(Graph(file, "/train/no_such", "")),
            "a dataset path the file does not have must fail the run");
    Require(missing.GetLastError().find("HDF5") != std::string::npos,
            "the refusal names HDF5: " + missing.GetLastError());

    // A file another writer made: h5py (examples/datasets/make_metric_blobs_h5.py)
    // from metric_blobs.csv, so every value must match the CSV it came from.
    const fs::path example = fs::path(CYXWIZ_SOURCE_ROOT) / "examples" / "datasets";
    cyxwiz::PipelineExecutor blobs;
    blobs.SetIngestionCacheRoot((work_dir / "ingestion_blobs").string());
    Require(blobs.ExecutePipeline(Graph(example / "metric_blobs.h5", "/blobs/features", "/blobs/labels")),
            "the h5py example should load: " + blobs.GetLastError());
    const auto blob_table = cyxwiz::DataRegistry::Instance().GetArrowDataset("ds_datainput_1")->GetArrowTable();
    std::ifstream csv(example / "metric_blobs.csv");
    std::string line;
    std::getline(csv, line);  // f0..f7,class
    int64_t row = 0;
    while (std::getline(csv, line)) {
        std::stringstream fields(line);
        std::string field;
        for (int column = 0; column < 9 && std::getline(fields, field, ','); ++column) {
            const std::string name = column < 8 ? "col_" + std::to_string(column) : "label";
            // float32 in the file: the CSV's 4 decimals survive within float precision.
            const double want = std::stod(field);
            const double got = Value(blob_table, name, row);
            Require(got > want - 1e-4 && got < want + 1e-4,
                    "metric_blobs row " + std::to_string(row) + " " + name + ": got " + std::to_string(got) +
                        ", CSV " + field);
        }
        ++row;
    }
    Require(row == 480 && blob_table->num_rows() == 480, "metric_blobs has 480 rows");
    std::cout << "Data Input HDF5 pipeline tests passed\n";
#else
    (void)work_dir;
    std::cout << "Data Input HDF5 pipeline tests skipped: no HDF5 in this build\n";
#endif
}
