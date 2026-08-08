#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

struct LargeTextFileIndex {
    std::uint64_t file_size = 0;
    std::uint64_t line_count = 0;
    std::uint64_t checkpoint_stride = 1024;
    std::vector<std::uint64_t> checkpoint_offsets;
};

struct LargeTextFilePage {
    std::uint64_t first_line = 0;
    std::vector<std::string> lines;
    std::vector<bool> truncated_lines;
};

class LargeTextFile {
public:
    using CancelCheck = std::function<bool()>;
    using ProgressCallback = std::function<void(float, const std::string&)>;

    static bool BuildIndex(
        const std::string& path,
        std::uint64_t checkpoint_stride,
        LargeTextFileIndex& index,
        std::string& error,
        const CancelCheck& should_cancel = {},
        const ProgressCallback& report_progress = {});

    static bool ReadPage(
        const std::string& path,
        const LargeTextFileIndex& index,
        std::uint64_t first_line,
        std::size_t max_lines,
        std::size_t max_line_bytes,
        LargeTextFilePage& page,
        std::string& error,
        const CancelCheck& should_cancel = {});
};

} // namespace cyxwiz
