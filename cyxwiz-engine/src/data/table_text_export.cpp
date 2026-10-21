#include "table_text_export.h"

#include <nlohmann/json.hpp>

#include <charconv>

namespace cyxwiz {

namespace {
std::string Exact(const DataTable::CellValue& v) {
    if (const auto* s = std::get_if<std::string>(&v)) return *s;
    if (const auto* n = std::get_if<int64_t>(&v)) return std::to_string(*n);
    if (const auto* d = std::get_if<double>(&v)) {
        char buf[32];
        const auto r = std::to_chars(buf, buf + sizeof(buf), *d);
        return std::string(buf, r.ptr);
    }
    return "";
}

std::string Delimited(const std::string& value, char separator) {
    if (value.find_first_of(std::string(1, separator) + "\"\r\n") == std::string::npos) return value;
    std::string quoted = "\"";
    for (char ch : value) {
        if (ch == '"') quoted += '"';
        quoted += ch;
    }
    return quoted + "\"";
}
}  // namespace

std::string TableText(const DataTable& table, const std::vector<size_t>& rows, TableTextFormat format, bool headers) {
    const auto& names = table.GetHeaders();
    const size_t cols = table.GetColumnCount();
    if (format == TableTextFormat::Json) {
        nlohmann::json out = nlohmann::json::array();
        for (size_t r : rows) {
            nlohmann::json row = nlohmann::json::object();
            for (size_t c = 0; c < cols; ++c) {
                const std::string key = c < names.size() && !names[c].empty() ? names[c] : "column " + std::to_string(c + 1);
                const auto v = table.GetCell(r, c);
                if (const auto* s = std::get_if<std::string>(&v)) row[key] = *s;
                else if (const auto* n = std::get_if<int64_t>(&v)) row[key] = *n;
                else if (const auto* d = std::get_if<double>(&v)) row[key] = *d;
                else row[key] = nullptr;
            }
            out.push_back(std::move(row));
        }
        return out.dump(2) + "\n";
    }
    const char separator = format == TableTextFormat::Csv ? ',' : '\t';
    std::string text;
    if (headers) {
        for (size_t c = 0; c < cols; ++c) {
            if (c) text += separator;
            text += Delimited(c < names.size() ? names[c] : std::string(), separator);
        }
        text += "\n";
    }
    for (size_t r : rows) {
        for (size_t c = 0; c < cols; ++c) {
            if (c) text += separator;
            text += Delimited(Exact(table.GetCell(r, c)), separator);
        }
        text += "\n";
    }
    return text;
}

std::shared_ptr<DataTable> TableRows(const DataTable& table, const std::vector<size_t>& rows) {
    auto out = std::make_shared<DataTable>();
    out->SetName(table.GetName());
    out->SetHeaders(table.GetHeaders());
    const size_t cols = table.GetColumnCount();
    for (size_t r : rows) {
        std::vector<DataTable::CellValue> row;
        row.reserve(cols);
        for (size_t c = 0; c < cols; ++c) row.push_back(table.GetCell(r, c));
        out->AddRow(std::move(row));
    }
    return out;
}

}  // namespace cyxwiz
