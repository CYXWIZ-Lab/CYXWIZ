#include "../src/core/large_text_file.h"

#include <chrono>
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

} // namespace

int main() {
    namespace fs = std::filesystem;
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path test_dir = fs::temp_directory_path() /
        ("cyxwiz_large_text_test_" + std::to_string(suffix));
    const fs::path test_file = test_dir / "sample.csv";
    const fs::path empty_file = test_dir / "empty.txt";
    fs::create_directories(test_dir);

    {
        std::ofstream output(test_file, std::ios::binary);
        Check(output.is_open(), "test file should open");
        for (int line = 0; line < 2050; ++line) {
            output << "line_" << line;
            if (line == 100) {
                output << std::string(80, 'x');
            }
            if (line + 1 < 2050) {
                output << "\r\n";
            }
        }
    }
    {
        std::ofstream output(empty_file, std::ios::binary);
    }

    cyxwiz::LargeTextFileIndex index;
    std::string error;
    Check(
        cyxwiz::LargeTextFile::BuildIndex(
            test_file.string(), 128, index, error),
        "indexing should succeed: " + error);
    Check(index.line_count == 2050, "line count should include final unterminated line");
    Check(index.checkpoint_stride == 128, "checkpoint stride should be preserved");
    Check(index.checkpoint_offsets.size() == 17, "sparse checkpoints should cover the file");
    Check(index.file_size == fs::file_size(test_file), "indexed file size should match disk");

    cyxwiz::LargeTextFilePage page;
    Check(
        cyxwiz::LargeTextFile::ReadPage(
            test_file.string(), index, 126, 5, 1024, page, error),
        "cross-checkpoint page read should succeed: " + error);
    Check(page.first_line == 126, "page should retain requested first line");
    Check(page.lines.size() == 5, "page should contain requested number of lines");
    Check(page.lines.front() == "line_126", "page should begin at requested line");
    Check(page.lines.back() == "line_130", "page should cross checkpoint boundary correctly");

    Check(
        cyxwiz::LargeTextFile::ReadPage(
            test_file.string(), index, 100, 1, 16, page, error),
        "bounded long-line page read should succeed: " + error);
    Check(page.lines.size() == 1, "bounded page should return one line");
    Check(page.lines.front().size() == 16, "displayed line should obey byte limit");
    Check(page.truncated_lines.front(), "long line should be marked truncated");

    Check(
        cyxwiz::LargeTextFile::ReadPage(
            test_file.string(), index, 2048, 10, 1024, page, error),
        "final partial page should succeed: " + error);
    Check(page.lines.size() == 2, "final page should stop at end of file");
    Check(page.lines.back() == "line_2049", "final unterminated line should be readable");

    cyxwiz::LargeTextFileIndex empty_index;
    Check(
        cyxwiz::LargeTextFile::BuildIndex(
            empty_file.string(), 128, empty_index, error),
        "empty file indexing should succeed: " + error);
    Check(empty_index.line_count == 0, "empty file should have zero lines");

    cyxwiz::LargeTextFileIndex cancelled_index;
    Check(
        !cyxwiz::LargeTextFile::BuildIndex(
            test_file.string(),
            128,
            cancelled_index,
            error,
            []() { return true; }),
        "cancelled indexing should stop");
    Check(error == "Cancelled", "cancelled indexing should report cancellation");

    fs::remove(test_file);
    fs::remove(empty_file);
    fs::remove(test_dir);
    std::cout << "Large text file paging contract passed\n";
    return 0;
}
