#include "../src/core/arrow_dataset.h"
#include "../src/core/data_convert_service.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const fs::path work_dir = fs::temp_directory_path() / "cyxwiz_data_convert_test";
    fs::remove_all(work_dir);
    fs::create_directories(work_dir);

    const fs::path csv_path = work_dir / "input.csv";
    const fs::path parquet_path = work_dir / "output.parquet";

    {
        std::ofstream csv(csv_path, std::ios::binary);
        csv << "id,value,label\n";
        csv << "1,1.5,a\n";
        csv << "2,2.5,b\n";
        csv << "3,3.5,a\n";
    }

    cyxwiz::DataConvertOptions options;
    options.input_path = csv_path.string();
    options.output_path = parquet_path.string();
    options.parquet_compression = "snappy";
    options.overwrite = false;
    options.write_manifest = true;

    auto preview = cyxwiz::DataConvertService::PreviewCsv(options);
    Check(preview.ok, "CSV preview should succeed: " + preview.error);
    Check(preview.rows == 3, "preview row count should match CSV rows");
    Check(preview.columns == 3, "preview column count should match CSV columns");
    Check(preview.schema.size() == 3, "preview schema should have three columns");
    Check(preview.schema[0].name == "id", "preview should preserve header names");
    Check(preview.sample_rows.size() == 3, "preview should include sample rows");
    Check(preview.sample_rows[0][0] == "1", "preview should expose cell values");

    auto result = cyxwiz::DataConvertService::ConvertCsvToParquet(options);
    Check(result.ok, "CSV to Parquet conversion should succeed: " + result.error);
    Check(result.rows_read == 3, "rows_read should match input");
    Check(result.rows_written == 3, "rows_written should match input");
    Check(result.columns == 3, "column count should match input");
    Check(result.output_path == parquet_path.string(),
          "result should report the Parquet output path");
    Check(fs::exists(parquet_path), "Parquet output should exist");
    Check(fs::exists(parquet_path.string() + ".manifest.json"),
          "manifest should exist");

    auto parquet = cyxwiz::ArrowDataset::FromParquet(
        parquet_path.string(), "converted");
    Check(parquet != nullptr, "converted Parquet should load");
    Check(parquet->GetNumRows() == 3, "converted Parquet row count should match");
    Check(parquet->GetNumColumns() == 3,
          "converted Parquet column count should match");

    auto blocked = cyxwiz::DataConvertService::ConvertCsvToParquet(options);
    Check(!blocked.ok, "overwrite=false should block existing output");

    const fs::path semicolon_csv_path = work_dir / "semicolon.csv";
    const fs::path semicolon_parquet_path = work_dir / "semicolon.parquet";
    {
        std::ofstream csv(semicolon_csv_path, std::ios::binary);
        csv << "id;value;label\n";
        csv << "1;4.5;x\n";
        csv << "2;5.5;y\n";
    }

    cyxwiz::DataConvertOptions auto_options;
    auto_options.input_path = semicolon_csv_path.string();
    auto_options.output_path = semicolon_parquet_path.string();
    auto_options.auto_detect_delimiter = true;

    auto auto_preview = cyxwiz::DataConvertService::PreviewCsv(auto_options);
    Check(auto_preview.ok,
          "auto delimiter preview should succeed: " + auto_preview.error);
    Check(auto_preview.detected_delimiter == ';',
          "auto delimiter should detect semicolon");
    Check(auto_preview.columns == 3,
          "auto delimiter preview should split semicolon columns");

    auto auto_result =
        cyxwiz::DataConvertService::ConvertCsvToParquet(auto_options);
    Check(auto_result.ok,
          "auto delimiter conversion should succeed: " + auto_result.error);
    Check(auto_result.detected_delimiter == ';',
          "auto delimiter conversion should report semicolon");

    const fs::path multiline_csv_path = work_dir / "multiline.csv";
    const fs::path multiline_parquet_path = work_dir / "multiline.parquet";
    {
        std::ofstream csv(multiline_csv_path, std::ios::binary);
        csv << "id,statement,status\n";
        csv << "1,\"first line\nsecond line\",Anxiety\n";
        csv << "2,\"plain text\",Normal\n";
    }

    cyxwiz::DataConvertOptions multiline_options;
    multiline_options.input_path = multiline_csv_path.string();
    multiline_options.output_path = multiline_parquet_path.string();
    multiline_options.auto_detect_delimiter = true;
    multiline_options.allow_newlines_in_values = true;

    auto multiline_preview =
        cyxwiz::DataConvertService::PreviewCsv(multiline_options);
    Check(multiline_preview.ok,
          "quoted multiline CSV preview should succeed: " +
              multiline_preview.error);
    Check(multiline_preview.rows == 2,
          "quoted multiline CSV should keep logical row count");

    auto multiline_result =
        cyxwiz::DataConvertService::ConvertCsvToParquet(multiline_options);
    Check(multiline_result.ok,
          "quoted multiline CSV conversion should succeed: " +
              multiline_result.error);

    fs::remove_all(work_dir);
    return 0;
}
