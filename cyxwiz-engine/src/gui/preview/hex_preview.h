#pragma once

#include <string>
#include <vector>

namespace gui {

/**
 * Hex dump preview for binary/unknown files
 * Classic layout: offset | hex bytes | ASCII
 */
class HexPreviewRenderer {
public:
    HexPreviewRenderer() = default;

    // Load file (reads first max_bytes)
    void LoadFile(const std::string& filepath, size_t max_bytes = 16384);

    // Set raw data
    void SetData(const std::vector<uint8_t>& data);

    // Render hex dump
    void Render();

    bool HasData() const { return !data_.empty(); }
    size_t GetDataSize() const { return data_.size(); }
    size_t GetFileSize() const { return file_size_; }

private:
    std::vector<uint8_t> data_;
    size_t file_size_ = 0;  // Actual file size (may be larger than data_)
    int bytes_per_row_ = 16;
};

} // namespace gui
