#include "core/html_document_text.h"
#include <arrow/api.h>
#include <iostream>
#include <stdexcept>

void Check(bool value, const char *why) {
  if (!value)
    throw std::runtime_error(why);
}
template <class F> void Reject(F f, const std::string &message) {
  try {
    f();
  } catch (const std::exception &e) {
    Check(std::string(e.what()).find(message) != std::string::npos, e.what());
    return;
  }
  throw std::runtime_error("Expected rejection: " + message);
}
std::shared_ptr<arrow::Table> Table(const std::vector<std::string> &rows) {
  arrow::StringBuilder builder;
  for (const auto &row : rows)
    Check(builder.Append(row).ok(), "append");
  auto array = builder.Finish();
  Check(array.ok(), "finish");
  return arrow::Table::Make(
      arrow::schema({arrow::field("text", arrow::utf8())}), {*array});
}
std::string Cell(const std::shared_ptr<arrow::Table> &table, const char *column,
                 int64_t row) {
  auto scalar = table->GetColumnByName(column)->GetScalar(row);
  Check(scalar.ok(), "scalar");
  return (*scalar)->ToString();
}
int main() {
  try {
    if (!cyxwiz::HtmlDocumentParserAvailable()) {
      Reject([] { cyxwiz::CleanHtmlDocumentTable(Table({"<p>x"}), "text"); },
             "Parsed HTML unavailable");
      std::cout << "Unavailable-provider contract passed\n";
      return 0;
    }
    auto input = Table({"<title>T &amp; U</title><p>A<b>B</b>C<p>D&nbsp;E &lt; "
                        "F<script>bad()</script><style>bad</style>",
                        ""});
    arrow::StringBuilder builder;
    Check(builder.AppendNull().ok(), "null append");
    auto nulls = builder.Finish();
    Check(nulls.ok(), "null finish");
    auto chunks = std::make_shared<arrow::ChunkedArray>(
        arrow::ArrayVector{input->column(0)->chunk(0), *nulls});
    input = arrow::Table::Make(input->schema(), {chunks});
    auto output = cyxwiz::CleanHtmlDocumentTable(input, "text");
    Check(output->num_rows() == 3 && output->num_columns() == 4, "shape");
    Check(output->column(0)->Equals(input->column(0)), "raw input preserved");
    Check(Cell(output, "text_cleaned", 0) == "ABC\n\nD E < F",
          "entities/inline/paragraphs");
    Check(Cell(output, "text_html_title", 0) == "T & U", "title");
    Check(Cell(output, "text_cleaned", 1).empty(), "empty remains empty");
    Check(!output->GetColumnByName("text_cleaned")
               ->GetScalar(2)
               .ValueOrDie()
               ->is_valid,
          "null preserved");
    Check(Cell(output, "text_html_policy", 0).find("parser_version") !=
              std::string::npos,
          "policy persisted");
    const std::string html =
        "<p>nav</p><!-- START "
        "--><table><tr><td>Heading</td><td>Date</td></tr></table><p>Content</"
        "p><!-- END --><p>footer</p>";
    Check(Cell(cyxwiz::CleanHtmlDocumentTable(Table({html}), "text", "START",
                                              "END"),
               "text_cleaned", 0) == "Heading Date\n\nContent",
          "generic markers preserve tables");
    Reject([&] { cyxwiz::CleanHtmlDocumentTable(output, "text"); },
           "already exists");
    Reject([&] { cyxwiz::CleanHtmlDocumentTable(input, "missing"); },
           "exist exactly once");
    Reject([&] { cyxwiz::CleanHtmlDocumentTable(input, "text", "START", ""); },
           "two different");
    Reject([&] { cyxwiz::CleanHtmlDocumentTable(input, "text", "A", "A"); },
           "two different");
    Reject(
        [&] { cyxwiz::CleanHtmlDocumentTable(input, "text", "START", "END"); },
        "Missing exact");
    Reject(
        [] {
          cyxwiz::CleanHtmlDocumentTable(Table({"<body><!-- END --><!-- START --></body>"}),
                                         "text", "START", "END");
        },
        "Ambiguous end");
    Reject(
        [] {
          cyxwiz::CleanHtmlDocumentTable(
              Table({"<body><!-- START --><!-- START --><!-- END --></body>"}), "text",
              "START", "END");
        },
        "Ambiguous begin");
    Reject(
        [] {
          cyxwiz::CleanHtmlDocumentTable(Table({std::string("a\0b", 3)}),
                                         "text");
        },
        "without NUL");
    Reject(
        [] {
          cyxwiz::CleanHtmlDocumentTable(Table({std::string(1048577, 'x')}),
                                         "text");
        },
        "1 MiB");
    Reject(
        [&] {
          cyxwiz::CleanHtmlDocumentTable(input, "text", {}, {},
                                         [] { return true; });
        },
        "cancelled");
    int polls = 0;
    Reject(
        [&] {
          cyxwiz::CleanHtmlDocumentTable(Table({std::string(100000, 'x')}),
                                         "text", {}, {},
                                         [&] { return ++polls == 4; });
        },
        "cancelled");
    std::cout
        << "HTML Arrow contract passed: text/title/policy, null/chunks/order, "
           "selection, collision, limits, cancellation\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
