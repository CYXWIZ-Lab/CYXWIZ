#include "../src/gui/loaders/tabular_loader.h"
#include "../src/core/csv_ingestion_options.h"

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    const auto missing_tokens =
        cyxwiz::ParseMissingValueTokens(" na, ?,na , missing ");
    Check(missing_tokens.size() == 3,
          "missing-token parsing should trim and deduplicate values");
    Check(missing_tokens[0] == "na" && missing_tokens[1] == "?" &&
              missing_tokens[2] == "missing",
          "missing-token parsing should preserve configured order");
    const auto convert_options =
        cyxwiz::MakeTabularCsvConvertOptions(missing_tokens);
    Check(convert_options.strings_can_be_null,
          "tabular CSV string columns should honor configured null values");
    Check(std::find(convert_options.null_values.begin(),
                    convert_options.null_values.end(), "na") !=
              convert_options.null_values.end(),
          "tabular CSV conversion should include lowercase na");

    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("csv"),
          "CSV should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("tsv"),
          "TSV should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("parquet"),
          "Parquet should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("feather"),
          "Feather should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("arrow"),
          "Arrow should remain an accepted tabular loader type");
    Check(!cyxwiz::loaders::IsUnsupportedTabularFileType("ipc"),
          "IPC should remain an accepted tabular loader type");
    Check(cyxwiz::loaders::ResolveTabularFileType(
              "auto", "D:\\datasets\\aps_failure_training_set.csv") == "csv",
          "Auto should resolve a CSV source before async loading");
    Check(cyxwiz::loaders::ResolveTabularFileType(
              "AUTO", "/datasets/table.TSV") == "tsv",
          "Auto extension resolution should be case-insensitive");
    Check(cyxwiz::loaders::ResolveTabularFileType(
              "parquet", "/datasets/table.csv") == "parquet",
          "An explicit format should override the source extension");

    for (const auto* extension : {"h5", "hdf5", "hdf", "H5", "HDF5", "HDF"}) {
        const auto resolved = cyxwiz::loaders::ResolveTabularFileType(
            " auto ", std::string("/datasets/table.") + extension);
        Check(resolved == "hdf5" && cyxwiz::loaders::IsUnsupportedTabularFileType(resolved),
              "Auto HDF5 must resolve and fail closed before generic loading");
    }
    Check(cyxwiz::loaders::ResolveTabularFileType("auto", "/datasets/table.zip") == "zip_text",
          "Auto ZIP should require the document selection contract");
    Check(cyxwiz::loaders::ResolveTabularFileType("auto", "/datasets/table.xlsx") == "excel",
          "Auto Excel should resolve to the unsupported format");
    Check(cyxwiz::loaders::ResolveTabularFileType("unknown", "/datasets/table.csv") == "unknown",
          "An unknown explicit format must not fall back to extension detection");
    const auto* hdf5 = cyxwiz::data_input::FindFormat("hdf5");
    Check(hdf5 && !hdf5->executable &&
              !cyxwiz::data_input::BuildAvailable(*hdf5, false) &&
              cyxwiz::data_input::BuildAvailable(*hdf5, true),
          "HighFive availability must not enable the incomplete Data Input adapter");
    Check(std::string(cyxwiz::data_input::UnsupportedReason("h5", false)).find("not compiled") != std::string::npos,
          "A build without HDF5 must report its missing dependency");
    Check(cyxwiz::data_input::ExtensionFilter().find("h5") == std::string::npos &&
              cyxwiz::data_input::ExtensionFilter().find("hdf") == std::string::npos,
          "The production file filter must not advertise HDF5");

    cyxwiz::loaders::ApplyContext context;
    context.dataset_name = "validation_test";
    context.source_path = "not-opened.H5";
    context.detected_file_type = "auto";
    std::string validation_error;
    Check(!cyxwiz::loaders::ValidateTabularApplyContext(context, validation_error) &&
              validation_error.find("HDF5 loading is not supported") != std::string::npos,
          "Auto HDF5 must be rejected by Apply preflight before worker launch");
    context.source_path = "not-opened.tsv";
    context.delimiter = ',';
    context.decimal_point = ',';
    Check(cyxwiz::loaders::ValidateTabularApplyContext(context, validation_error),
          "Auto TSV must validate its actual tab delimiter and allow decimal commas");
    context.source_path = "not-opened.csv";
    Check(!cyxwiz::loaders::ValidateTabularApplyContext(context, validation_error),
          "CSV must reject equal field and decimal separators");
    context.source_path = "not-opened.zip";
    Check(!cyxwiz::loaders::ValidateTabularApplyContext(context, validation_error) &&
              validation_error.find("exact member path") != std::string::npos,
          "Auto ZIP must require a document selection before worker launch");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("json"),
          "JSON should fail closed before async tabular load");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("json").find(
              "JSON loading is not supported") != std::string::npos,
          "JSON failure should explain unsupported loader");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("excel"),
          "Excel should fail closed before async tabular load");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("excel").find(
              "Excel loading is not supported") != std::string::npos,
          "Excel failure should explain unsupported loader");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("hdf5"),
          "HDF5 should fail closed before async tabular load");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("hdf5").find(
              "HDF5 loading is not supported") != std::string::npos,
          "HDF5 failure should explain unsupported loader");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("txt"),
          "TXT should fail closed on tabular path");
    Check(cyxwiz::loaders::UnsupportedTabularFileTypeMessage("txt").find(
              "use Text source") != std::string::npos,
          "TXT failure should point to Text source");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("arff"),
          "ARFF should fail closed before async tabular load");

    Check(cyxwiz::loaders::IsUnsupportedTabularFileType("JSON"),
          "JSON validation should be case-insensitive");

    return 0;
}
