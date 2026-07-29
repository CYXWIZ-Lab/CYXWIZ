#include "../src/core/arrow_dataset.h"
#include "../src/core/data_convert_service.h"

#ifdef CYXWIZ_HAS_HDF5
#include <highfive/highfive.hpp>
#endif

#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

void WriteNpyFloat64Matrix(const std::filesystem::path& path,
                           const std::vector<double>& values,
                           int64_t rows,
                           int64_t columns) {
    std::ofstream out(path, std::ios::binary);
    Check(static_cast<bool>(out), "test should open npy fixture for writing");

    std::ostringstream header;
    header << "{'descr': '<f8', 'fortran_order': False, 'shape': ("
           << rows << ", " << columns << "), }";
    std::string header_text = header.str();
    const size_t preamble_size = 10;
    size_t padded_size = header_text.size() + 1;
    while ((preamble_size + padded_size) % 16 != 0) {
        ++padded_size;
    }
    header_text.append(padded_size - header_text.size() - 1, ' ');
    header_text.push_back('\n');

    const char magic[] = "\x93NUMPY";
    out.write(magic, 6);
    const unsigned char version[2] = {1, 0};
    out.write(reinterpret_cast<const char*>(version), 2);
    const uint16_t header_length =
        static_cast<uint16_t>(header_text.size());
    const unsigned char length_bytes[2] = {
        static_cast<unsigned char>(header_length & 0xff),
        static_cast<unsigned char>((header_length >> 8) & 0xff)};
    out.write(reinterpret_cast<const char*>(length_bytes), 2);
    out.write(header_text.data(),
              static_cast<std::streamsize>(header_text.size()));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
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

    const fs::path late_decimal_csv_path = work_dir / "late_decimal.csv";
    const fs::path late_decimal_parquet_path =
        work_dir / "late_decimal.parquet";
    {
        std::ofstream csv(late_decimal_csv_path, std::ios::binary);
        csv << "feature;label\n";
        for (int row = 0; row < 300000; ++row) {
            csv << "0;0\n";
        }
        csv << "3,5;1\n";
    }

    cyxwiz::DataConvertOptions late_decimal_options;
    late_decimal_options.input_path = late_decimal_csv_path.string();
    late_decimal_options.input_format = "csv";
    late_decimal_options.output_path = late_decimal_parquet_path.string();
    late_decimal_options.output_format = "parquet";
    late_decimal_options.delimiter = ';';
    late_decimal_options.decimal_point = ',';
    late_decimal_options.overwrite = true;
    late_decimal_options.retain_output_table = true;
    auto late_decimal_result =
        cyxwiz::DataConvertService::Convert(late_decimal_options);
    Check(late_decimal_result.ok,
          "late decimal conversion should succeed with a stable numeric schema: " +
              late_decimal_result.error);
    Check(late_decimal_result.output_table &&
              late_decimal_result.output_table->num_rows() == 300001,
          "late decimal conversion should retain every source row");
    Check(late_decimal_result.output_table &&
              late_decimal_result.output_table->schema()->field(0)->type()->id() ==
                  arrow::Type::DOUBLE,
          "integer-looking first blocks should promote to float64 before full parsing");
    if (late_decimal_result.output_table) {
        auto scalar_result =
            late_decimal_result.output_table->column(0)->GetScalar(300000);
        Check(scalar_result.ok() &&
                  std::abs(std::static_pointer_cast<arrow::DoubleScalar>(
                               scalar_result.ValueOrDie())->value - 3.5) < 1e-12,
              "late comma-decimal value should be parsed numerically");
    }

    auto parquet = cyxwiz::ArrowDataset::FromParquet(
        parquet_path.string(), "converted");
    Check(parquet != nullptr, "converted Parquet should load");
    Check(parquet->GetNumRows() == 3, "converted Parquet row count should match");
    Check(parquet->GetNumColumns() == 3,
          "converted Parquet column count should match");
    Check(parquet->GetArrowTable()->schema()->field(0)->type()->id() ==
              arrow::Type::INT64,
          "integer-only columns should retain their inferred integer type");

    const fs::path memory_csv_path = work_dir / "memory_input.csv";
    cyxwiz::DataConvertOptions memory_to_csv;
    memory_to_csv.input_table = parquet->GetArrowTable();
    memory_to_csv.output_path = memory_csv_path.string();
    memory_to_csv.output_format = "csv";
    memory_to_csv.overwrite = true;
    auto memory_result = cyxwiz::DataConvertService::Convert(memory_to_csv);
    Check(memory_result.ok,
          "in-memory Arrow table conversion should succeed without input_path: " +
              memory_result.error);
    Check(fs::exists(memory_csv_path),
          "in-memory Arrow table conversion should create CSV output");

    cyxwiz::DataConvertOptions mismatched_output;
    mismatched_output.input_path = parquet_path.string();
    mismatched_output.input_format = "parquet";
    mismatched_output.output_path = (work_dir / "wrong.csv").string();
    mismatched_output.output_format = "parquet";
    mismatched_output.overwrite = true;
    auto mismatch_result = cyxwiz::DataConvertService::Convert(mismatched_output);
    Check(!mismatch_result.ok,
          "explicit output format should reject mismatched output extension");

    const fs::path roundtrip_csv_path = work_dir / "roundtrip.csv";
    cyxwiz::DataConvertOptions parquet_to_csv;
    parquet_to_csv.input_path = parquet_path.string();
    parquet_to_csv.input_format = "parquet";
    parquet_to_csv.output_path = roundtrip_csv_path.string();
    parquet_to_csv.output_format = "csv";
    parquet_to_csv.overwrite = true;
    auto csv_result = cyxwiz::DataConvertService::Convert(parquet_to_csv);
    Check(csv_result.ok, "Parquet to CSV conversion should succeed: " + csv_result.error);
    Check(fs::exists(roundtrip_csv_path), "CSV output should exist");
    {
        std::ifstream csv(roundtrip_csv_path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(csv)),
                         std::istreambuf_iterator<char>());
        Check(text.find("id") != std::string::npos &&
                  text.find("value") != std::string::npos &&
                  text.find("label") != std::string::npos,
              "CSV output should include the header");
        Check(text.find("1,1.5") != std::string::npos,
              "CSV output should include data rows");
    }

    const fs::path feather_path = work_dir / "roundtrip.feather";
    cyxwiz::DataConvertOptions parquet_to_feather;
    parquet_to_feather.input_path = parquet_path.string();
    parquet_to_feather.input_format = "parquet";
    parquet_to_feather.output_path = feather_path.string();
    parquet_to_feather.output_format = "feather";
    parquet_to_feather.overwrite = true;
    auto feather_result = cyxwiz::DataConvertService::Convert(parquet_to_feather);
    Check(feather_result.ok,
          "Parquet to Feather conversion should succeed: " + feather_result.error);
    auto feather = cyxwiz::ArrowDataset::FromFeather(
        feather_path.string(), "converted_feather");
    Check(feather != nullptr, "converted Feather should load");
    Check(feather->GetNumRows() == 3,
          "converted Feather row count should match");

    const fs::path jsonl_path = work_dir / "input.jsonl";
    const fs::path jsonl_parquet_path = work_dir / "jsonl_output.parquet";
    {
        std::ofstream jsonl(jsonl_path, std::ios::binary);
        jsonl << "{\"id\":1,\"value\":1.5,\"label\":\"a\"}\n";
        jsonl << "{\"id\":2,\"value\":2.5,\"label\":\"b\"}\n";
        jsonl << "{\"id\":3,\"value\":3.5,\"label\":\"a\"}\n";
    }

    cyxwiz::DataConvertOptions jsonl_to_parquet;
    jsonl_to_parquet.input_path = jsonl_path.string();
    jsonl_to_parquet.input_format = "jsonl";
    jsonl_to_parquet.output_path = jsonl_parquet_path.string();
    jsonl_to_parquet.output_format = "parquet";
    jsonl_to_parquet.overwrite = true;
    auto jsonl_result = cyxwiz::DataConvertService::Convert(jsonl_to_parquet);
    Check(jsonl_result.ok,
          "JSONL to Parquet conversion should succeed: " + jsonl_result.error);
    auto jsonl_parquet = cyxwiz::ArrowDataset::FromParquet(
        jsonl_parquet_path.string(), "converted_jsonl");
    Check(jsonl_parquet != nullptr, "converted JSONL Parquet should load");
    Check(jsonl_parquet->GetNumRows() == 3,
          "converted JSONL row count should match");
    Check(jsonl_parquet->GetNumColumns() == 3,
          "converted JSONL column count should match");

    const fs::path parquet_to_jsonl_path = work_dir / "roundtrip.jsonl";
    cyxwiz::DataConvertOptions parquet_to_jsonl;
    parquet_to_jsonl.input_path = parquet_path.string();
    parquet_to_jsonl.input_format = "parquet";
    parquet_to_jsonl.output_path = parquet_to_jsonl_path.string();
    parquet_to_jsonl.output_format = "jsonl";
    parquet_to_jsonl.overwrite = true;
    auto parquet_to_jsonl_result =
        cyxwiz::DataConvertService::Convert(parquet_to_jsonl);
    Check(parquet_to_jsonl_result.ok,
          "Parquet to JSONL conversion should succeed: " +
              parquet_to_jsonl_result.error);
    {
        std::ifstream jsonl(parquet_to_jsonl_path, std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(jsonl)),
                         std::istreambuf_iterator<char>());
        Check(text.find("\"id\"") != std::string::npos,
              "JSONL output should include column names");
        Check(text.find("\"label\"") != std::string::npos,
              "JSONL output should include string columns");
    }

    const fs::path text_path = work_dir / "input.txt";
    const fs::path text_parquet_path = work_dir / "text_output.parquet";
    {
        std::ofstream text(text_path, std::ios::binary);
        text << "first line\n";
        text << "second line\n";
        text << "third line\n";
    }

    cyxwiz::DataConvertOptions text_to_parquet;
    text_to_parquet.input_path = text_path.string();
    text_to_parquet.input_format = "txt";
    text_to_parquet.output_path = text_parquet_path.string();
    text_to_parquet.output_format = "parquet";
    text_to_parquet.overwrite = true;
    auto text_result = cyxwiz::DataConvertService::Convert(text_to_parquet);
    Check(text_result.ok,
          "TXT to Parquet conversion should succeed: " + text_result.error);
    auto text_parquet = cyxwiz::ArrowDataset::FromParquet(
        text_parquet_path.string(), "converted_text");
    Check(text_parquet != nullptr, "converted TXT Parquet should load");
    Check(text_parquet->GetNumRows() == 3,
          "converted TXT row count should match");
    Check(text_parquet->GetNumColumns() == 1,
          "converted TXT should create one text column");

    const fs::path parquet_to_text_path = work_dir / "roundtrip.txt";
    cyxwiz::DataConvertOptions parquet_to_text;
    parquet_to_text.input_path = text_parquet_path.string();
    parquet_to_text.input_format = "parquet";
    parquet_to_text.output_path = parquet_to_text_path.string();
    parquet_to_text.output_format = "txt";
    parquet_to_text.overwrite = true;
    auto parquet_to_text_result =
        cyxwiz::DataConvertService::Convert(parquet_to_text);
    Check(parquet_to_text_result.ok,
          "single-column Parquet to TXT conversion should succeed: " +
              parquet_to_text_result.error);
    {
        std::ifstream text(parquet_to_text_path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(text)),
                            std::istreambuf_iterator<char>());
        Check(content.find("first line") != std::string::npos,
              "TXT output should include source lines");
    }

    const fs::path arff_path = work_dir / "input.arff";
    const fs::path arff_parquet_path = work_dir / "arff_output.parquet";
    {
        std::ofstream arff(arff_path, std::ios::binary);
        arff << "@RELATION sentiment\n\n";
        arff << "@ATTRIBUTE id NUMERIC\n";
        arff << "@ATTRIBUTE score REAL\n";
        arff << "@ATTRIBUTE label STRING\n\n";
        arff << "@DATA\n";
        arff << "1,0.5,positive\n";
        arff << "2,0.2,negative\n";
        arff << "3,?,neutral\n";
    }

    cyxwiz::DataConvertOptions arff_to_parquet;
    arff_to_parquet.input_path = arff_path.string();
    arff_to_parquet.input_format = "arff";
    arff_to_parquet.output_path = arff_parquet_path.string();
    arff_to_parquet.output_format = "parquet";
    arff_to_parquet.overwrite = true;
    auto arff_result = cyxwiz::DataConvertService::Convert(arff_to_parquet);
    Check(arff_result.ok,
          "ARFF to Parquet conversion should succeed: " + arff_result.error);
    auto arff_parquet = cyxwiz::ArrowDataset::FromParquet(
        arff_parquet_path.string(), "converted_arff");
    Check(arff_parquet != nullptr, "converted ARFF Parquet should load");
    Check(arff_parquet->GetNumRows() == 3,
          "converted ARFF row count should match");
    Check(arff_parquet->GetNumColumns() == 3,
          "converted ARFF column count should match");

    const fs::path parquet_to_arff_path = work_dir / "roundtrip.arff";
    cyxwiz::DataConvertOptions parquet_to_arff;
    parquet_to_arff.input_path = parquet_path.string();
    parquet_to_arff.input_format = "parquet";
    parquet_to_arff.output_path = parquet_to_arff_path.string();
    parquet_to_arff.output_format = "arff";
    parquet_to_arff.overwrite = true;
    auto parquet_to_arff_result =
        cyxwiz::DataConvertService::Convert(parquet_to_arff);
    Check(parquet_to_arff_result.ok,
          "Parquet to ARFF conversion should succeed: " +
              parquet_to_arff_result.error);
    {
        std::ifstream arff(parquet_to_arff_path, std::ios::binary);
        std::string content((std::istreambuf_iterator<char>(arff)),
                            std::istreambuf_iterator<char>());
        Check(content.find("@RELATION") != std::string::npos,
              "ARFF output should include relation header");
        Check(content.find("@ATTRIBUTE") != std::string::npos,
              "ARFF output should include attribute declarations");
        Check(content.find("@DATA") != std::string::npos,
              "ARFF output should include data section");
    }

    const fs::path npy_path = work_dir / "input.npy";
    const fs::path npy_parquet_path = work_dir / "npy_output.parquet";
    WriteNpyFloat64Matrix(npy_path, {1.0, 2.0, 3.0, 4.0}, 2, 2);

    cyxwiz::DataConvertOptions npy_to_parquet;
    npy_to_parquet.input_path = npy_path.string();
    npy_to_parquet.input_format = "npy";
    npy_to_parquet.output_path = npy_parquet_path.string();
    npy_to_parquet.output_format = "parquet";
    npy_to_parquet.overwrite = true;
    auto npy_result = cyxwiz::DataConvertService::Convert(npy_to_parquet);
    Check(npy_result.ok,
          "NPY to Parquet conversion should succeed: " + npy_result.error);
    auto npy_parquet = cyxwiz::ArrowDataset::FromParquet(
        npy_parquet_path.string(), "converted_npy");
    Check(npy_parquet != nullptr, "converted NPY Parquet should load");
    Check(npy_parquet->GetNumRows() == 2,
          "converted NPY row count should match");
    Check(npy_parquet->GetNumColumns() == 2,
          "converted NPY column count should match");

    const fs::path parquet_to_npy_path = work_dir / "roundtrip.npy";
    cyxwiz::DataConvertOptions parquet_to_npy;
    parquet_to_npy.input_path = npy_parquet_path.string();
    parquet_to_npy.input_format = "parquet";
    parquet_to_npy.output_path = parquet_to_npy_path.string();
    parquet_to_npy.output_format = "npy";
    parquet_to_npy.overwrite = true;
    auto parquet_to_npy_result =
        cyxwiz::DataConvertService::Convert(parquet_to_npy);
    Check(parquet_to_npy_result.ok,
          "numeric Parquet to NPY conversion should succeed: " +
              parquet_to_npy_result.error);
    cyxwiz::DataConvertOptions npy_reload;
    npy_reload.input_path = parquet_to_npy_path.string();
    npy_reload.input_format = "npy";
    std::string npy_reload_error;
    auto npy_reload_table =
        cyxwiz::DataConvertService::LoadTable(npy_reload, npy_reload_error);
    Check(npy_reload_table != nullptr,
          "written NPY output should reload: " + npy_reload_error);
    Check(npy_reload_table->num_rows() == 2,
          "written NPY reload should preserve row count");
    Check(npy_reload_table->num_columns() == 2,
          "written NPY reload should preserve column count");

#ifdef CYXWIZ_HAS_HDF5
    const fs::path hdf5_path = work_dir / "input.h5";
    const fs::path hdf5_parquet_path = work_dir / "hdf5_output.parquet";
    {
        HighFive::File file(hdf5_path.string(), HighFive::File::Overwrite);
        std::vector<std::vector<double>> values = {
            {1.0, 2.0},
            {3.0, 4.0},
            {5.0, 6.0},
        };
        auto dataset = file.createDataSet<double>(
            "data", HighFive::DataSpace::From(values));
        dataset.write(values);
    }

    cyxwiz::DataConvertOptions hdf5_to_parquet;
    hdf5_to_parquet.input_path = hdf5_path.string();
    hdf5_to_parquet.input_format = "hdf5";
    hdf5_to_parquet.output_path = hdf5_parquet_path.string();
    hdf5_to_parquet.output_format = "parquet";
    hdf5_to_parquet.overwrite = true;
    auto hdf5_result =
        cyxwiz::DataConvertService::Convert(hdf5_to_parquet);
    Check(hdf5_result.ok,
          "HDF5 to Parquet conversion should succeed: " + hdf5_result.error);
    auto hdf5_parquet = cyxwiz::ArrowDataset::FromParquet(
        hdf5_parquet_path.string(), "converted_hdf5");
    Check(hdf5_parquet != nullptr, "converted HDF5 Parquet should load");
    Check(hdf5_parquet->GetNumRows() == 3,
          "converted HDF5 row count should match");
    Check(hdf5_parquet->GetNumColumns() == 2,
          "converted HDF5 column count should match");

    const fs::path parquet_to_hdf5_path = work_dir / "roundtrip.h5";
    cyxwiz::DataConvertOptions parquet_to_hdf5;
    parquet_to_hdf5.input_path = npy_parquet_path.string();
    parquet_to_hdf5.input_format = "parquet";
    parquet_to_hdf5.output_path = parquet_to_hdf5_path.string();
    parquet_to_hdf5.output_format = "hdf5";
    parquet_to_hdf5.overwrite = true;
    auto parquet_to_hdf5_result =
        cyxwiz::DataConvertService::Convert(parquet_to_hdf5);
    Check(parquet_to_hdf5_result.ok,
          "numeric Parquet to HDF5 conversion should succeed: " +
              parquet_to_hdf5_result.error);
    cyxwiz::DataConvertOptions hdf5_reload;
    hdf5_reload.input_path = parquet_to_hdf5_path.string();
    hdf5_reload.input_format = "hdf5";
    std::string hdf5_reload_error;
    auto hdf5_reload_table =
        cyxwiz::DataConvertService::LoadTable(hdf5_reload, hdf5_reload_error);
    Check(hdf5_reload_table != nullptr,
          "written HDF5 output should reload: " + hdf5_reload_error);
    Check(hdf5_reload_table->num_rows() == 2,
          "written HDF5 reload should preserve row count");
    Check(hdf5_reload_table->num_columns() == 2,
          "written HDF5 reload should preserve column count");
#endif

    auto skipped = cyxwiz::DataConvertService::ConvertCsvToParquet(options);
    Check(skipped.ok, "fresh output should be reusable without overwrite: " + skipped.error);
    Check(skipped.skipped_fresh_output,
          "fresh output should report skipped_fresh_output");
    Check(skipped.rows_written == 3, "fresh manifest should preserve row count");

    fs::remove(parquet_path.string() + ".manifest.json");
    auto blocked = cyxwiz::DataConvertService::ConvertCsvToParquet(options);
    Check(!blocked.ok,
          "overwrite=false should block existing output when no fresh manifest exists");

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
