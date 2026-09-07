#include "core/data_convert_service.h"
#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
int checks = 0;
void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
struct Workspace {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("cyxwiz_convert_reload_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Workspace() { Check(std::filesystem::create_directory(path), "unique workspace"); }
    ~Workspace() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
}

int main() try {
    namespace fs = std::filesystem;
    using cyxwiz::DataConvertOptions;
    using cyxwiz::DataConvertService;
    Workspace workspace;
    const auto input = workspace.path / "input.csv";
    { std::ofstream out(input); out << "left,right\n1.5,2.25\n-3.5,4.75\n"; }
    std::vector<std::string> formats = {
        "csv", "tsv", "jsonl", "arff", "npy", "parquet", "feather", "arrow", "ipc"};
#ifdef CYXWIZ_HAS_HDF5
    formats.push_back("hdf5");
#endif
    std::string error;
#ifndef CYXWIZ_HAS_HDF5
    DataConvertOptions unavailable;
    unavailable.input_path = input.string();
    unavailable.output_path = (workspace.path / "unavailable.hdf5").string();
    unavailable.output_format = "hdf5";
    auto unavailable_result = DataConvertService::Convert(unavailable);
    Check(!unavailable_result.ok && !fs::exists(unavailable.output_path), "disabled HDF5 cannot publish output");
#endif
    for (const auto& format : formats) {
        DataConvertOptions options;
        options.input_path = input.string();
        options.output_path = (workspace.path / ("output." + format)).string();
        options.output_format = format;
        auto result = DataConvertService::Convert(options);
        Check(result.ok && !result.output_table, format + " writes without retained table: " + result.error);
        DataConvertOptions reload;
        reload.input_path = result.output_path;
        reload.input_format = format;
        reload.auto_detect_delimiter = true;
        auto table = DataConvertService::LoadTable(reload, error);
        Check(table && table->num_rows() == 2 && table->num_columns() == 2, format + " reload shape: " + error);
        const bool matrix = format == "npy" || format == "hdf5";
        Check(table->field(0)->name() == (matrix ? "col_0" : "left"), format + " column name contract");
        Check(table->field(0)->type()->id() == arrow::Type::DOUBLE &&
              table->field(1)->type()->id() == arrow::Type::DOUBLE, format + " numeric schema");
        const double expected[2][2] = {{1.5, 2.25}, {-3.5, 4.75}};
        for (int row = 0; row < 2; ++row) {
            for (int column = 0; column < 2; ++column) {
                const auto scalar = table->column(column)->GetScalar(row).ValueOrDie();
                Check(scalar->is_valid && static_cast<const arrow::DoubleScalar&>(*scalar).value == expected[row][column],
                      format + " disk values");
            }
        }
        options.retain_output_table = true;
        auto cached = DataConvertService::Convert(options);
        Check(cached.ok && cached.skipped_fresh_output && !cached.output_table,
              format + " cache forces disk reload despite retention request");
        auto cached_table = DataConvertService::LoadTable(reload, error);
        Check(cached_table && cached_table->Equals(*table, false), format + " cached output disk values: " + error);
    }
    const auto text_input = workspace.path / "lines.txt";
    // Parquet can reconstruct a different Arrow schema (for example timezone
    // metadata) unless explicitly stored. Retention must describe disk truth.
    const auto timestamp_type = arrow::timestamp(arrow::TimeUnit::NANO, "Europe/Paris");
    arrow::TimestampBuilder timestamp_builder(timestamp_type, arrow::default_memory_pool());
    Check(timestamp_builder.Append(123456789).ok(), "timestamp fixture value");
    std::shared_ptr<arrow::Array> timestamp_array;
    Check(timestamp_builder.Finish(&timestamp_array).ok(), "timestamp fixture array");
    auto timestamp_table = arrow::Table::Make(
        arrow::schema({arrow::field("when", timestamp_type)}), {timestamp_array});
    DataConvertOptions retained;
    retained.input_table = timestamp_table;
    retained.output_path = (workspace.path / "timestamp.parquet").string();
    retained.retain_output_table = true;
    auto retained_result = DataConvertService::Convert(retained);
    Check(retained_result.ok && retained_result.output_table, retained_result.error);
    DataConvertOptions timestamp_reload;
    timestamp_reload.input_path = retained.output_path;
    auto timestamp_disk = DataConvertService::LoadTable(timestamp_reload, error);
    Check(timestamp_disk && retained_result.output_table->Equals(*timestamp_disk, false),
          "retained Parquet schema must match its disk reload");
    { std::ofstream out(text_input); out << "first line\nsecond line\n"; }
    DataConvertOptions text_options;
    text_options.input_path = text_input.string();
    text_options.output_path = (workspace.path / "output.txt").string();
    text_options.output_format = "txt";
    Check(DataConvertService::Convert(text_options).ok, "text conversion");
    DataConvertOptions text_reload;
    text_reload.input_path = text_options.output_path;
    auto lines = DataConvertService::LoadTable(text_reload, error);
    Check(lines && lines->num_rows() == 2 && lines->num_columns() == 1, "text shape: " + error);
    Check(lines->field(0)->name() == "text" && lines->field(0)->type()->id() == arrow::Type::STRING, "text schema");
    Check(lines->column(0)->GetScalar(1).ValueOrDie()->ToString() == "second line", "text disk values");
    auto cached_text = DataConvertService::Convert(text_options);
    Check(cached_text.ok && cached_text.skipped_fresh_output && !cached_text.output_table, "text cached disk branch");

    DataConvertOptions options;
    options.input_path = input.string();
    options.output_path = (workspace.path / "cache.parquet").string();
    Check(DataConvertService::Convert(options).ok, "file cache baseline");
    auto source = DataConvertService::LoadTable(options, error);
    Check(source != nullptr, error);
    options.input_table = source->Slice(1, 1);
    options.overwrite = true;
    Check(DataConvertService::Preview(options).rows == 1, "preview prioritizes Arrow table over file");
    auto memory = DataConvertService::Convert(options);
    Check(memory.ok && !memory.skipped_fresh_output && memory.rows_written == 1, "Arrow input cannot reuse file cache");
    options.input_table.reset();
    options.overwrite = false;
    Check(!DataConvertService::Convert(options).ok, "Arrow manifest cannot impersonate file cache");
    options.overwrite = true;
    Check(DataConvertService::Convert(options).rows_written == 2, "file input restored");
    options.overwrite = false;
    const auto output = fs::path(options.output_path);
    const auto output_time = fs::last_write_time(output);
    { std::fstream out(output, std::ios::binary | std::ios::in | std::ios::out); out.write("FAIL", 4); }
    fs::last_write_time(output, output_time + std::chrono::seconds(2));
    Check(!DataConvertService::Convert(options).ok, "same-size modified output is not fresh");
    options.overwrite = true;
    Check(DataConvertService::Convert(options).ok, "replace corrupted output");
    options.overwrite = false;
    const auto manifest_path = options.output_path + ".manifest.json";
    nlohmann::json manifest;
    { std::ifstream in(manifest_path); in >> manifest; }
    manifest["rows_written"] = "invalid type";
    { std::ofstream out(manifest_path); out << manifest; }
    Check(!DataConvertService::Convert(options).ok, "wrongly typed manifest is a miss, not an exception or success");
    options.overwrite = true;
    Check(DataConvertService::Convert(options).ok, "replace malformed manifest");
    options.overwrite = false;
    fs::last_write_time(input, fs::last_write_time(input) + std::chrono::seconds(2));
    Check(!DataConvertService::Convert(options).ok, "changed input invalidates cache");
    options.input_table = source;
    options.input_path.clear();
    Check(DataConvertService::Preview(options).ok, "pathless Arrow preview succeeds");
    std::cout << "DataConvert disk reload/cache: " << checks << " checks passed; HDF5 "
#ifdef CYXWIZ_HAS_HDF5
              << "enabled\n";
#else
              << "unavailable (not tested)\n";
#endif
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
