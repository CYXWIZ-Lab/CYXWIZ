#include "csv_records.h"

#include <cerrno>
#include <cstdlib>

namespace cyxwiz::csv {

namespace {
std::string Trim(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t");
    if (a == std::string::npos) return {};
    const size_t b = s.find_last_not_of(" \t");
    return s.substr(a, b - a + 1);
}
}  // namespace

std::vector<std::vector<std::string>> ReadRecords(const std::string& text) {
    std::vector<std::vector<std::string>> records;
    std::vector<std::string> record;
    std::string field;
    bool quoted = false;      // inside quotes
    bool was_quoted = false;  // this field started with a quote
    size_t i = 0;
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF && static_cast<unsigned char>(text[1]) == 0xBB &&
        static_cast<unsigned char>(text[2]) == 0xBF)
        i = 3;
    auto end_field = [&]() {
        record.push_back(was_quoted ? field : Trim(field));
        field.clear();
        was_quoted = false;
    };
    auto end_record = [&]() {
        end_field();
        if (!(record.size() == 1 && record[0].empty())) records.push_back(std::move(record));  // skip blank lines
        record.clear();
    };
    for (; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < text.size() && text[i + 1] == '"') {
                    field += '"';
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                field += c;
            }
            continue;
        }
        if (c == '"' && Trim(field).empty()) {
            field.clear();
            quoted = true;
            was_quoted = true;
        } else if (c == ',') {
            end_field();
        } else if (c == '\r') {
            if (i + 1 < text.size() && text[i + 1] == '\n') ++i;
            end_record();
        } else if (c == '\n') {
            end_record();
        } else {
            field += c;
        }
    }
    if (!field.empty() || !record.empty() || was_quoted) end_record();
    return records;
}

std::string Field(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

bool ParseInt(const std::string& text, long long& out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const long long v = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || end != text.c_str() + text.size()) return false;
    out = v;
    return true;
}

bool ParseDouble(const std::string& text, double& out) {
    if (text.empty()) return false;
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (errno != 0 || end != text.c_str() + text.size()) return false;
    out = v;
    return true;
}

}  // namespace cyxwiz::csv
