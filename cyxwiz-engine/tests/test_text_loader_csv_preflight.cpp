#include "../src/core/arrow_dataset.h"
#include "../src/gui/loaders/text_csv_preflight.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::filesystem::path WriteTempFile(const std::string& name,
                                    const std::string& content) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary);
    Check(out.is_open(), "should create temp CSV");
    out << content;
    out.close();
    return path;
}

} // namespace

int main() {
    {
        const auto path = WriteTempFile(
            "cyxwiz_text_preflight_valid.csv",
            "text,label\n"
            "\"hello, world\",greeting\n"
            "\"multi\nline\",note\n");
        const auto result =
            cyxwiz::loaders::ValidateTextCsvRowWidths(path.string(), ',');
        Check(result.ok, "quoted commas and embedded newlines should pass");

        auto read_options = arrow::csv::ReadOptions::Defaults();
        auto parse_options = arrow::csv::ParseOptions::Defaults();
        auto convert_options = arrow::csv::ConvertOptions::Defaults();
        parse_options.delimiter = ',';
        parse_options.newlines_in_values = true;
        auto raw_arrow = cyxwiz::ArrowDataset::FromCSV(
            path.string(), "cyxwiz_text_preflight_valid",
            read_options, parse_options, convert_options);
        Check(raw_arrow != nullptr && raw_arrow->GetArrowTable() != nullptr,
              "TextLoader Arrow options should accept embedded newlines after preflight");
        Check(raw_arrow->GetArrowTable()->num_rows() == 2,
              "embedded-newline CSV should register two data rows");
        std::filesystem::remove(path);
    }

    {
        const auto path = WriteTempFile(
            "cyxwiz_text_preflight_short.csv",
            "text,label\n"
            "only_text\n");
        const auto result =
            cyxwiz::loaders::ValidateTextCsvRowWidths(path.string(), ',');
        Check(!result.ok, "short row should fail preflight");
        Check(result.message.find("row 2 has 1 fields but header has 2 fields") !=
                  std::string::npos,
              "short row failure should include row and width details: " +
                  result.message);
        Check(result.message.find("delimiter ','") != std::string::npos,
              "short row failure should include delimiter hint: " +
                  result.message);
        std::filesystem::remove(path);
    }

    {
        const auto path = WriteTempFile(
            "cyxwiz_text_preflight_tsv.tsv",
            "text\tlabel\n"
            "hello\tpositive\textra\n");
        const auto result =
            cyxwiz::loaders::ValidateTextCsvRowWidths(path.string(), '\t');
        Check(!result.ok, "wide TSV row should fail preflight");
        Check(result.message.find("delimiter '\\t'") != std::string::npos,
              "TSV failure should show tab delimiter: " + result.message);
        std::filesystem::remove(path);
    }

    {
        const auto path = WriteTempFile(
            "cyxwiz_text_preflight_unclosed.csv",
            "text,label\n"
            "\"unterminated,label\n");
        const auto result =
            cyxwiz::loaders::ValidateTextCsvRowWidths(path.string(), ',');
        Check(!result.ok, "unterminated quote should fail preflight");
        Check(result.message.find("unterminated quoted field") !=
                  std::string::npos,
              "unterminated quote failure should explain parse issue: " +
                  result.message);
        std::filesystem::remove(path);
    }

    std::cout << "Text loader CSV preflight validation passed\n";
    {
        // E3 2026-10-08: a missing file or a load that read nothing must not
        // register anything under the dataset name.
        const auto missing = cyxwiz::loaders::ValidateTextSource(
            (std::filesystem::temp_directory_path() / "cyxwiz_no_such_text_source.csv").string());
        Check(!missing.ok && missing.message.rfind("file not found: ", 0) == 0,
              "a missing text source is refused before loading");
        const auto path = WriteTempFile("cyxwiz_text_preflight_exists.csv", "text,label\nhi,a\n");
        Check(cyxwiz::loaders::ValidateTextSource(path.string()).ok, "an existing file passes");
        Check(cyxwiz::loaders::ValidateTextSource(std::filesystem::temp_directory_path().string()).ok,
              "a corpus directory passes");
        std::filesystem::remove(path);
        const auto none = cyxwiz::loaders::ValidateTextSamples(0, "x.csv");
        Check(!none.ok && none.message.find("no text samples could be read from 'x.csv'") != std::string::npos,
              "zero samples are refused");
        Check(cyxwiz::loaders::ValidateTextSamples(1, "x.csv").ok, "one sample passes");
    }

    return 0;
}
