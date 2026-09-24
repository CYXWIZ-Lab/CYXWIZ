#pragma once

#include <cctype>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

// SQL step contract (TOFIX101 packages C and C2), shared by the executor, the
// node editor (input pins) and Preparation Recipe validation. Header-only and
// dependency-free so the editor and recipe code need no DuckDB.
inline constexpr const char* kSqlContractParameter = "sql_contract_version";
inline constexpr const char* kSqlContractVersion = "1";
inline constexpr const char* kSqlQueryParameter = "query";
inline constexpr const char* kSqlInputAliasParameter = "input_alias";
inline constexpr const char* kSqlInputAliasesParameter = "input_aliases";
inline constexpr const char* kSqlDefaultInputAlias = "input";
inline constexpr size_t kSqlMaxInputs = 8;

// The SQL step's input tables, in input-pin order. `input_aliases` (a
// comma-separated list such as "chapters, manifest") wins when set; otherwise
// the single `input_alias` (default "input"). Surrounding blanks are trimmed
// and empty entries dropped; validity of each name is checked where the query
// runs (SqlInputAliasRejection), so a typo is reported rather than hidden.
inline std::vector<std::string> SqlStepInputAliases(
    const std::map<std::string, std::string>& params) {
    const auto trim = [](std::string text) {
        size_t start = 0;
        while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start]))) ++start;
        size_t end = text.size();
        while (end > start && std::isspace(static_cast<unsigned char>(text[end - 1]))) --end;
        return text.substr(start, end - start);
    };
    std::vector<std::string> aliases;
    const auto list = params.find(kSqlInputAliasesParameter);
    if (list != params.end() && !trim(list->second).empty()) {
        std::string item;
        for (char c : list->second + ",") {
            if (c == ',') {
                item = trim(item);
                if (!item.empty()) aliases.push_back(item);
                item.clear();
            } else {
                item += c;
            }
        }
        return aliases;
    }
    const auto single = params.find(kSqlInputAliasParameter);
    const std::string alias = single == params.end() ? std::string() : trim(single->second);
    aliases.push_back(alias.empty() ? kSqlDefaultInputAlias : alias);
    return aliases;
}

}  // namespace cyxwiz
