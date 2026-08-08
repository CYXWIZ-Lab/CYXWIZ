#include "large_text_file.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <vector>

namespace cyxwiz {
namespace {

constexpr std::size_t kScanChunkBytes = 1024 * 1024;

bool ReadBoundedLine(
    std::istream& input,
    std::size_t max_line_bytes,
    std::string& line,
    bool& truncated,
    const LargeTextFile::CancelCheck& should_cancel) {
    line.clear();
    truncated = false;

    std::size_t consumed = 0;
    char value = '\0';
    while (input.get(value)) {
        if (value == '\n') {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return true;
        }

        if (line.size() < max_line_bytes) {
            line.push_back(value);
        } else {
            truncated = true;
        }

        ++consumed;
        if ((consumed % (64 * 1024)) == 0 && should_cancel && should_cancel()) {
            return false;
        }
    }

    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return consumed > 0;
}

} // namespace

bool LargeTextFile::BuildIndex(
    const std::string& path,
    std::uint64_t checkpoint_stride,
    LargeTextFileIndex& index,
    std::string& error,
    const CancelCheck& should_cancel,
    const ProgressCallback& report_progress) {
    error.clear();
    index = {};

    if (checkpoint_stride == 0) {
        error = "Checkpoint stride must be greater than zero";
        return false;
    }

    std::error_code size_error;
    const auto raw_file_size = std::filesystem::file_size(path, size_error);
    if (size_error) {
        error = "Could not read file size: " + size_error.message();
        return false;
    }

    index.file_size = static_cast<std::uint64_t>(raw_file_size);
    index.checkpoint_stride = checkpoint_stride;
    index.checkpoint_offsets.push_back(0);

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        error = "Could not open file";
        return false;
    }

    std::vector<char> buffer(kScanChunkBytes);
    std::uint64_t bytes_read = 0;
    std::uint64_t newline_count = 0;
    char last_byte = '\0';

    while (input) {
        if (should_cancel && should_cancel()) {
            error = "Cancelled";
            return false;
        }

        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = static_cast<std::size_t>(input.gcount());
        if (count == 0) {
            break;
        }

        for (std::size_t offset = 0; offset < count; ++offset) {
            if (buffer[offset] != '\n') {
                continue;
            }

            ++newline_count;
            if ((newline_count % checkpoint_stride) == 0) {
                index.checkpoint_offsets.push_back(
                    bytes_read + static_cast<std::uint64_t>(offset) + 1);
            }
        }

        last_byte = buffer[count - 1];
        bytes_read += static_cast<std::uint64_t>(count);

        if (report_progress && index.file_size > 0) {
            const float fraction = static_cast<float>(
                static_cast<double>(bytes_read) / static_cast<double>(index.file_size));
            report_progress(std::clamp(fraction, 0.0f, 1.0f), "Indexing lines...");
        }
    }

    if (input.bad()) {
        error = "Failed while reading file";
        return false;
    }

    if (index.file_size == 0) {
        index.line_count = 0;
    } else {
        index.line_count = newline_count + (last_byte == '\n' ? 0 : 1);
    }

    if (report_progress) {
        report_progress(1.0f, "Index complete");
    }
    return true;
}

bool LargeTextFile::ReadPage(
    const std::string& path,
    const LargeTextFileIndex& index,
    std::uint64_t first_line,
    std::size_t max_lines,
    std::size_t max_line_bytes,
    LargeTextFilePage& page,
    std::string& error,
    const CancelCheck& should_cancel) {
    error.clear();
    page = {};

    if (index.checkpoint_stride == 0 || index.checkpoint_offsets.empty()) {
        error = "Large text index is not initialized";
        return false;
    }
    if (max_lines == 0 || max_line_bytes == 0) {
        error = "Page and line limits must be greater than zero";
        return false;
    }

    page.first_line = std::min(first_line, index.line_count);
    if (page.first_line >= index.line_count) {
        return true;
    }

    const std::uint64_t checkpoint = page.first_line / index.checkpoint_stride;
    if (checkpoint >= index.checkpoint_offsets.size()) {
        error = "Large text index does not cover the requested line";
        return false;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        error = "Could not open file";
        return false;
    }

    input.seekg(static_cast<std::streamoff>(index.checkpoint_offsets[checkpoint]), std::ios::beg);
    if (!input) {
        error = "Could not seek to requested page";
        return false;
    }

    const std::uint64_t checkpoint_line = checkpoint * index.checkpoint_stride;
    const std::uint64_t lines_to_skip = page.first_line - checkpoint_line;
    for (std::uint64_t skipped = 0; skipped < lines_to_skip; ++skipped) {
        if (should_cancel && should_cancel()) {
            error = "Cancelled";
            return false;
        }
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        if (!input) {
            error = "File ended before the requested line";
            return false;
        }
    }

    page.lines.reserve(max_lines);
    page.truncated_lines.reserve(max_lines);

    while (page.lines.size() < max_lines &&
           page.first_line + page.lines.size() < index.line_count) {
        if (should_cancel && should_cancel()) {
            error = "Cancelled";
            return false;
        }

        std::string line;
        bool truncated = false;
        if (!ReadBoundedLine(input, max_line_bytes, line, truncated, should_cancel)) {
            if (should_cancel && should_cancel()) {
                error = "Cancelled";
                return false;
            }
            break;
        }

        page.lines.push_back(std::move(line));
        page.truncated_lines.push_back(truncated);
    }

    return true;
}

} // namespace cyxwiz
