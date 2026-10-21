#include "../src/core/label_column_resolver.h"

#include <arrow/api.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    Check(cyxwiz::IsCommonLabelColumnName("label"), "label should match");
    Check(cyxwiz::IsCommonLabelColumnName("Label"), "Label should match");
    Check(cyxwiz::IsCommonLabelColumnName("class"), "class should match");
    Check(cyxwiz::IsCommonLabelColumnName("target"), "target should match");
    Check(cyxwiz::IsCommonLabelColumnName("y"), "y should match");
    Check(cyxwiz::IsCommonLabelColumnName("category"), "category should match");

    Check(!cyxwiz::IsCommonLabelColumnName("tok_0"), "tok_0 should not match");
    Check(!cyxwiz::IsCommonLabelColumnName("text"), "text should not match");
    Check(!cyxwiz::IsCommonLabelColumnName("not_label"),
          "not_label should not match");

    Check(cyxwiz::FindCommonLabelColumnIndex(nullptr) == -1,
          "null schema should not match");

    auto token_schema = arrow::schema({
        arrow::field("tok_0", arrow::float32()),
        arrow::field("tok_1", arrow::float32()),
        arrow::field("y", arrow::int32()),
    });
    Check(cyxwiz::FindCommonLabelColumnIndex(token_schema) == 2,
          "y should be detected after token columns");

    auto first_match_schema = arrow::schema({
        arrow::field("feature", arrow::float32()),
        arrow::field("target", arrow::int32()),
        arrow::field("y", arrow::int32()),
    });
    Check(cyxwiz::FindCommonLabelColumnIndex(first_match_schema) == 1,
          "first common label column should win");
    Check(cyxwiz::ResolveLabelColumnIndex(first_match_schema, "y") == 2,
          "an explicit label should take precedence over common-name order");
    Check(cyxwiz::ResolveLabelColumnName(first_match_schema) == "target",
          "an empty selection should resolve the first common label name");
    Check(cyxwiz::ResolveLabelColumnIndex(first_match_schema, "missing") == -1,
          "an invalid explicit label must not silently select another column");

    auto tabular_schema = arrow::schema({
        arrow::field("class", arrow::utf8()),
        arrow::field("sensor_a", arrow::float64()),
        arrow::field("sensor_b", arrow::int64()),
        arrow::field("comment", arrow::utf8()),
        arrow::field("__partition__", arrow::int8()),
    });
    const int tabular_label =
        cyxwiz::ResolveLabelColumnIndex(tabular_schema, "class");
    Check(cyxwiz::CountNumericBatchFeatureColumns(
              tabular_schema, tabular_label) == 2,
          "feature width should match numeric batch columns while excluding label and internal columns");

    auto no_label_schema = arrow::schema({
        arrow::field("text", arrow::utf8()),
        arrow::field("tok_0", arrow::float32()),
    });
    Check(cyxwiz::FindCommonLabelColumnIndex(no_label_schema) == -1,
          "schema without common label should not match");

    std::cout << "Label column resolver tests passed\n";
    return 0;
}
