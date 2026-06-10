#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::loaders {

struct TextCsvPreflightResult {
    bool ok = true;
    std::string message;
};

namespace detail {

inline std::string DisplayTextCsvDelimiter(char delimiter) {
    if (delimiter == '\t') return "\\t";
    if (delimiter == '\n') return "\\n";
    if (delimiter == '\r') return "\\r";
    return std::string(1, delimiter);
}

inline bool ReadTextCsvRow(std::istream& in,
                           char delimiter,
                           std::vector<std::string>& out_fields,
                           std::string& error) {
    out_fields.clear();
    std::string field;
    bool in_quotes = false;
    bool any_char_read = false;
    int c;

    while ((c = in.get()) != EOF) {
        any_char_read = true;
        if (in_quotes) {
            if (c == '"') {
                if (in.peek() == '"') {
                    field.push_back('"');
                    in.get();
                } else {
                    in_quotes = false;
                }
            } else {
                field.push_back(static_cast<char>(c));
            }
        } else {
            if (c == '"') {
                in_quotes = true;
            } else if (c == delimiter) {
                out_fields.push_back(std::move(field));
                field.clear();
            } else if (c == '\r') {
                if (in.peek() == '\n') in.get();
                out_fields.push_back(std::move(field));
                return true;
            } else if (c == '\n') {
                out_fields.push_back(std::move(field));
                return true;
            } else {
                field.push_back(static_cast<char>(c));
            }
        }
    }

    if (!any_char_read) {
        return false;
    }

    if (in_quotes) {
        error = "unterminated quoted field";
        return false;
    }

    out_fields.push_back(std::move(field));
    return true;
}

inline bool IsBlankTextCsvRow(const std::vector<std::string>& fields) {
    return fields.size() == 1 && fields[0].empty();
}

} // namespace detail

inline TextCsvPreflightResult ValidateTextCsvRowWidths(
    const std::string& path,
    char delimiter) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return {false, "Text CSV preflight failed: cannot open file '" + path + "'"};
    }

    std::vector<std::string> fields;
    std::string parse_error;
    if (!detail::ReadTextCsvRow(file, delimiter, fields, parse_error)) {
        if (!parse_error.empty()) {
            return {false, "Text CSV preflight failed: " + parse_error +
                               " in header row"};
        }
        return {false, "Text CSV preflight failed: file is empty"};
    }

    const size_t expected_width = fields.size();
    if (expected_width == 0) {
        return {false, "Text CSV preflight failed: header row has no fields"};
    }

    size_t row_number = 1;
    while (true) {
        parse_error.clear();
        const bool has_row =
            detail::ReadTextCsvRow(file, delimiter, fields, parse_error);
        if (!has_row) {
            if (!parse_error.empty()) {
                std::ostringstream msg;
                msg << "Text CSV preflight failed: " << parse_error
                    << " near row " << (row_number + 1);
                return {false, msg.str()};
            }
            break;
        }

        ++row_number;
        if (detail::IsBlankTextCsvRow(fields)) {
            continue;
        }

        if (fields.size() != expected_width) {
            std::ostringstream msg;
            msg << "Text CSV preflight failed: row " << row_number << " has "
                << fields.size() << " fields but header has "
                << expected_width << " fields using delimiter '"
                << detail::DisplayTextCsvDelimiter(delimiter)
                << "'. Check the delimiter, quoting, or malformed row before applying.";
            return {false, msg.str()};
        }
    }

    return {};
}

} // namespace cyxwiz::loaders
