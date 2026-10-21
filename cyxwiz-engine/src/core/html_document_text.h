#pragma once
#include <functional>
#include <memory>
#include <string>
namespace arrow {
class Table;
}
namespace cyxwiz {
bool HtmlDocumentParserAvailable();
// Preserves all input fields and row order, appends <column>_cleaned,
// <column>_html_title and <column>_html_policy. Throws; never publishes partial
// output.
std::shared_ptr<arrow::Table> CleanHtmlDocumentTable(
    const std::shared_ptr<arrow::Table> &input, const std::string &column,
    const std::string &begin_comment = {}, const std::string &end_comment = {},
    const std::function<bool()> &cancelled = {});
} // namespace cyxwiz
