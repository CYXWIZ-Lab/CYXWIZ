#pragma once

#include <string>
#include <vector>

namespace gui {

/**
 * Table preview for CSV/TSV data with column headers and pagination
 */
class TablePreviewRenderer {
public:
    TablePreviewRenderer() = default;

    // Load from raw file (parses first N rows)
    void LoadFromFile(const std::string& filepath, char delimiter = ',', int max_rows = 1000);

    // Set data directly
    void SetData(const std::vector<std::string>& headers,
                  const std::vector<std::vector<std::string>>& rows);

    // Render the table
    void Render();

    // Clear cached data (call when dataset changes)
    void Clear();

    bool HasData() const { return !rows_.empty(); }

private:
    std::vector<std::string> headers_;
    std::vector<std::vector<std::string>> rows_;
    int current_page_ = 0;
    int rows_per_page_ = 50;
    char search_buffer_[256] = "";
    bool has_header_ = true;

    // Cache tracking to detect source changes
    std::string cached_source_;  // File path or dataset name

    void RenderPagination();
};

} // namespace gui
