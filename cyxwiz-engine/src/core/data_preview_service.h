#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

class DataRegistry;

enum class DataPreviewStatus {
    Ready,
    InvalidRequest,
    Unsupported,
    Cancelled,
    Failed,
};

struct DataPreviewColumn {
    std::string name;
    std::string type;
    bool nullable = false;
    int64_t sampled_values = 0;
    int64_t sampled_nulls = 0;
};

struct DataPreviewRequest {
    std::string dataset_name;
    int64_t offset = 0;
    int64_t row_limit = 20;
    std::vector<std::string> selected_columns;
    std::function<bool()> cancel_requested;
};

struct DataPreviewPage {
    bool ok = false;
    DataPreviewStatus status = DataPreviewStatus::Failed;
    std::string reason;
    std::string backend;
    std::string dataset_name;
    int64_t total_rows = 0;
    int64_t total_columns = 0;
    int64_t offset = 0;
    int64_t rows_returned = 0;
    bool has_next = false;
    int64_t next_offset = 0;
    std::vector<DataPreviewColumn> schema;
    std::vector<std::vector<std::string>> rows;
};

class DataPreviewService {
public:
    static DataPreviewPage PreviewRegisteredTabular(
        DataRegistry& registry,
        const DataPreviewRequest& request);
};

} // namespace cyxwiz
