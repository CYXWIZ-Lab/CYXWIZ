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

    auto result = cyxwiz::DataConvertService::ConvertCsvToParquet(options);
    Check(result.ok, "CSV to Parquet conversion should succeed: " + result.error);
    Check(result.rows_read == 3, "rows_read should match input");
    Check(result.rows_written == 3, "rows_written should match input");
    Check(result.columns == 3, "column count should match input");
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

    fs::remove_all(work_dir);
    return 0;
}
