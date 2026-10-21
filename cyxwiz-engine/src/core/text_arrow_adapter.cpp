#include "text_arrow_adapter.h"

#include <arrow/builder.h>

#include <string>
#include <vector>

namespace cyxwiz {

arrow::Result<std::shared_ptr<arrow::Table>> BuildRawTextArrowTable(
    const TextDataset& dataset,
    const std::string& text_column,
    const std::string& label_column) {

    const DatasetInfo info = dataset.GetInfo();
    const size_t rows = dataset.Size();
    const bool include_labels =
        info.num_classes > 0 && !label_column.empty();

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    arrow::StringBuilder text_builder(pool);
    ARROW_RETURN_NOT_OK(text_builder.Reserve(static_cast<int64_t>(rows)));

    arrow::Int32Builder label_builder(pool);
    if (include_labels) {
        ARROW_RETURN_NOT_OK(label_builder.Reserve(static_cast<int64_t>(rows)));
    }

    for (size_t i = 0; i < rows; ++i) {
        ARROW_RETURN_NOT_OK(text_builder.Append(dataset.GetText(i)));
        if (include_labels) {
            const int label = dataset.GetLabel(i);
            if (label < 0) {
                return arrow::Status::Invalid(
                    "BuildRawTextArrowTable: missing label at row " +
                    std::to_string(i));
            }
            ARROW_RETURN_NOT_OK(label_builder.Append(label));
        }
    }

    std::shared_ptr<arrow::Array> text_array;
    ARROW_RETURN_NOT_OK(text_builder.Finish(&text_array));

    std::vector<std::shared_ptr<arrow::Field>> fields = {
        arrow::field(text_column.empty() ? "text" : text_column, arrow::utf8()),
    };
    std::vector<std::shared_ptr<arrow::Array>> arrays = {text_array};

    if (include_labels) {
        std::shared_ptr<arrow::Array> label_array;
        ARROW_RETURN_NOT_OK(label_builder.Finish(&label_array));
        fields.push_back(arrow::field(label_column, arrow::int32()));
        arrays.push_back(label_array);
    }

    return arrow::Table::Make(
        arrow::schema(fields), arrays, static_cast<int64_t>(rows));
}

} // namespace cyxwiz
