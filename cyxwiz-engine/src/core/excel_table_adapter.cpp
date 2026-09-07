#include "excel_table_adapter.h"

#ifdef CYXWIZ_HAS_XLSX
#include "xlsx_archive_guard.h"
#include <OpenXLSX.hpp>
#include <arrow/api.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <vector>

namespace {
using OpenXLSX::XLValueType;

void CheckArrow(const arrow::Status& status) {
    if (!status.ok()) throw std::runtime_error(status.ToString());
}

XLValueType MergeType(XLValueType previous, XLValueType current,
                      const std::string& location) {
    if (current == XLValueType::Error)
        throw std::runtime_error(location + ": Excel error cell is unsupported");
    if (current == XLValueType::Empty) return previous;
    if (previous == XLValueType::Empty || previous == current) return current;
    const auto numeric = [](XLValueType type) {
        return type == XLValueType::Integer || type == XLValueType::Float;
    };
    if (numeric(previous) && numeric(current)) return XLValueType::Float;
    throw std::runtime_error(location + ": mixed column types are unsupported");
}

std::shared_ptr<arrow::DataType> ArrowType(XLValueType type) {
    switch (type) {
    case XLValueType::Boolean: return arrow::boolean();
    case XLValueType::Integer: return arrow::int64();
    case XLValueType::Float: return arrow::float64();
    default: return arrow::utf8(); // Includes columns consisting entirely of blanks.
    }
}

void AppendValue(arrow::ArrayBuilder& builder, const OpenXLSX::XLCellValue& value,
                 size_t& string_bytes) {
    if (value.type() == XLValueType::Empty) {
        CheckArrow(builder.AppendNull());
    } else if (builder.type()->id() == arrow::Type::DOUBLE) {
        if (value.type() == XLValueType::Integer) {
            const auto integer = value.get<int64_t>();
            if (integer < -9007199254740992LL || integer > 9007199254740992LL)
                throw std::runtime_error("integer cannot be promoted to float64 without precision loss");
        }
        const double number = value.get<double>();
        if (!std::isfinite(number)) throw std::runtime_error("non-finite number is unsupported");
        CheckArrow(static_cast<arrow::DoubleBuilder&>(builder).Append(number));
    } else if (value.type() == XLValueType::Integer) {
        CheckArrow(static_cast<arrow::Int64Builder&>(builder).Append(value.get<int64_t>()));
    } else if (value.type() == XLValueType::Boolean) {
        CheckArrow(static_cast<arrow::BooleanBuilder&>(builder).Append(value.get<bool>()));
    } else {
        const auto text = value.get<std::string>();
        constexpr size_t limit = 64 * 1024 * 1024;
        if (text.size() > limit - string_bytes)
            throw std::runtime_error("XLSX Arrow string-byte limit exceeded");
        string_bytes += text.size();
        CheckArrow(static_cast<arrow::StringBuilder&>(builder).Append(text));
    }
}
} // namespace
#endif

namespace cyxwiz {

std::shared_ptr<arrow::Table> ReadExcelTable(
    const std::string& path, const ExcelTableReadOptions& options, std::string& error) {
    error.clear();
#ifdef CYXWIZ_HAS_XLSX
    std::string location = path;
    try {
        if (options.skip_rows < 0 || options.max_cells == 0 || options.max_cells > 1000000)
            throw std::runtime_error("skip_rows must be non-negative and max_cells must be between 1 and 1000000");
        const std::filesystem::path file_path(std::u8string(path.begin(), path.end()));
        std::string extension = file_path.extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (extension != ".xlsx")
            throw std::runtime_error("only .xlsx workbooks are supported; .xls and macro files are unsupported");
        PreparedXlsxArchive snapshot(file_path);
        OpenXLSX::XLDocument document;
        const auto staged_utf8 = snapshot.Path().u8string();
        document.open(std::string(staged_utf8.begin(), staged_utf8.end()));
        auto workbook = document.workbook();
        const auto sheets = workbook.worksheetNames();
        if (sheets.empty()) throw std::runtime_error("workbook has no worksheets");
        const auto name = options.sheet_name.empty() ? sheets.front() : options.sheet_name;
        if (std::find(sheets.begin(), sheets.end(), name) == sheets.end())
            throw std::runtime_error("worksheet '" + name + "' does not exist");
        auto sheet = workbook.worksheet(name);
        location = name;
        if (sheet.merges().count() != 0)
            throw std::runtime_error("merged cells are unsupported; unmerge the worksheet first");
        const uint32_t rows = sheet.rowCount();
        const uint16_t columns = sheet.columnCount();
        if (rows > 1048576 || columns > 16384)
            throw std::runtime_error("worksheet dimensions exceed XLSX limits");
        if (rows == 0 || columns == 0 || static_cast<uint64_t>(options.skip_rows) >= rows)
            throw std::runtime_error("worksheet has no selected rows");
        if (static_cast<uint64_t>(rows) > options.max_cells / columns)
            throw std::runtime_error("worksheet exceeds the configured cell limit");

        const uint32_t first = static_cast<uint32_t>(options.skip_rows) + 1;
        const uint32_t data_first = first + (options.has_header ? 1 : 0);
        std::vector<std::string> names;
        std::set<std::string> unique_names;
        size_t string_bytes = 0;
        for (uint16_t col = 1; col <= columns; ++col) {
            std::string field_name = "col_" + std::to_string(col - 1);
            if (options.has_header) {
                const OpenXLSX::XLCellReference ref(first, col);
                location = name + "!" + ref.address();
                auto cell = sheet.findCell(ref);
                if (cell.empty() || cell.hasFormula() || cell.value().type() != XLValueType::String)
                    throw std::runtime_error("header must be a non-empty string without a formula");
                field_name = cell.value().get<std::string>();
                if (field_name.empty()) throw std::runtime_error("header cannot be empty");
            }
            if (!unique_names.insert(field_name).second)
                throw std::runtime_error("duplicate header '" + field_name + "'");
            if (field_name.size() > 1024 * 1024 - string_bytes)
                throw std::runtime_error("XLSX header-byte limit exceeded");
            string_bytes += field_name.size();
            names.push_back(std::move(field_name));
        }

        std::vector<XLValueType> types(columns, XLValueType::Empty);
        bool any_value = options.has_header;
        for (uint32_t row = data_first; row <= rows; ++row) {
            for (uint16_t col = 1; col <= columns; ++col) {
                const OpenXLSX::XLCellReference ref(row, col);
                location = name + "!" + ref.address();
                auto cell = sheet.findCell(ref);
                if (cell.empty()) continue;
                if (cell.hasFormula()) throw std::runtime_error("formula cells are unsupported; export values first");
                const auto type = cell.value().type();
                types[col - 1] = MergeType(types[col - 1], type, location);
                any_value = any_value || type != XLValueType::Empty;
            }
        }
        if (!any_value) throw std::runtime_error("worksheet contains no values");

        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        for (uint16_t col = 1; col <= columns; ++col) {
            const auto type = ArrowType(types[col - 1]);
            std::unique_ptr<arrow::ArrayBuilder> builder;
            CheckArrow(arrow::MakeBuilder(arrow::default_memory_pool(), type, &builder));
            CheckArrow(builder->Reserve(rows - data_first + 1));
            for (uint32_t row = data_first; row <= rows; ++row) {
                const OpenXLSX::XLCellReference ref(row, col);
                location = name + "!" + ref.address();
                auto cell = sheet.findCell(ref);
                if (cell.empty()) CheckArrow(builder->AppendNull());
                else AppendValue(*builder, cell.value(), string_bytes);
            }
            std::shared_ptr<arrow::Array> array;
            CheckArrow(builder->Finish(&array));
            fields.push_back(arrow::field(names[col - 1], type));
            arrays.push_back(std::move(array));
        }
        auto table = arrow::Table::Make(arrow::schema(fields), arrays);
        CheckArrow(table->ValidateFull());
        return table;
    } catch (const std::exception& exception) {
        error = "XLSX read failed at " + location + ": " + exception.what();
        return nullptr;
    }
#else
    (void)path;
    (void)options;
    error = "XLSX support is not compiled into this build (OpenXLSX is required).";
    return nullptr;
#endif
}

} // namespace cyxwiz
