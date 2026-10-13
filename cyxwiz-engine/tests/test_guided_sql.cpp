// Guided SQL steps (TOFIX146 step 5): the operation library, JSON round trip,
// column tracking, errors, summaries and the shape of the generated SQL.
// DuckDB runs every operation in test_pipeline_executor_operator_routing
// --guided.
#include "../src/core/guided_sql.h"
#include "../src/core/sql_step_presentation.h"

#include <iostream>
#include <string>

namespace {
int checks = 0;
int failures = 0;

void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

using namespace cyxwiz;
using json = nlohmann::json;

guided::Context SermonContext() {
    guided::Context context;
    context.inputs = {"sermons", "manifest"};
    context.columns["sermons"] = {"sermon_id", "title", "book_code", "chapter", "text", "word_count", "split"};
    context.columns["manifest"] = {"partition_document_id", "split"};
    return context;
}

void TestLibrary() {
    Check(guided::Library().size() == 20, "20 operations, got " + std::to_string(guided::Library().size()));
    std::string categories;
    for (const auto& op : guided::Library()) {
        if (categories.find(op.category) == std::string::npos) categories += std::string(op.category) + ",";
        Check(!guided::DefaultParams(op.id, "text").empty(), std::string("defaults for ") + op.id);
    }
    Check(categories == "Rows,Columns,Text,Combine,Summarise,Splits,Checks,", "7 categories in order: " + categories);
    Check(guided::FindOp("look_up") && !guided::FindOp("nope"), "find an operation by id");
}

void TestJson() {
    guided::GuidedStep step;
    step.ops.push_back({"keep_rows", true, guided::DefaultParams("keep_rows", "text")});
    step.ops.push_back({"sort_rows", false, guided::DefaultParams("sort_rows", "title")});
    const auto back = guided::FromJson(guided::ToJson(step));
    Check(back && back->ops.size() == 2 && back->ops[1].type == "sort_rows" && !back->ops[1].on &&
              back->ops[0].params == step.ops[0].params,
          "round trip");
    // The inputs' columns travel with the operations.
    step.saved_context = SermonContext();
    step.saved_context.whole_number_columns = {"word_count"};
    const auto with_context = guided::FromJson(guided::ToJson(step));
    Check(with_context && with_context->saved_context.inputs.size() == 2 &&
              with_context->saved_context.columns.at("manifest").size() == 2 &&
              with_context->saved_context.whole_number_columns.count("word_count") == 1,
          "the saved context round-trips");
    std::string error;
    Check(!guided::FromJson("{\"version\":1,\"ops\":[{\"type\":\"drop_table\"}]}", &error) && error.find("drop_table") != std::string::npos,
          "unknown operations are refused: " + error);
    Check(!guided::FromJson("{\"version\":2,\"ops\":[]}", &error), "a newer version is refused");
    Check(!guided::FromJson("not json", &error), "bad JSON is refused");
}

void TestColumnsAndSql() {
    const auto context = SermonContext();
    guided::GuidedStep step;
    // The board 3 example: look up the chapter's split, keep matched long
    // sermons, choose and rename columns, check.
    step.ops.push_back({"look_up", true,
                        {{"table", "manifest"},
                         {"key", json::array({{{"kind", "column"}, {"value", "book_code"}}, {{"kind", "text"}, {"value", " "}},
                                              {{"kind", "column"}, {"value", "chapter"}}, {{"kind", "text"}, {"value", ":1"}}})},
                         {"other_key", "partition_document_id"},
                         {"bring", json::array({{{"column", "split"}, {"as", "chapter_split"}}})},
                         {"no_match", "keep"}}});
    step.ops.push_back({"keep_rows", true,
                        {{"match", "all"},
                         {"conditions", json::array({{{"column", "chapter_split"}, {"op", "is not empty"}, {"value", ""}},
                                                     {{"column", "word_count"}, {"op", "at least"}, {"value", "5000"}}})}}});
    step.ops.push_back({"choose_columns", true,
                        {{"columns", json::array({{{"name", "sermon_id"}, {"keep", true}},
                                                  {{"name", "title"}, {"keep", true}},
                                                  {{"name", "text"}, {"keep", true}},
                                                  {{"name", "word_count"}, {"keep", true}},
                                                  {{"name", "split"}, {"keep", false}},
                                                  {{"name", "chapter_split"}, {"keep", true}, {"rename", "split"}}})}}});
    step.ops.push_back({"add_checks", true,
                        {{"checks", json::array({{{"kind", "allowed_values"}, {"column", "split"}, {"values", "train, validation, test"}},
                                                 {{"kind", "unique"}, {"column", "sermon_id"}}})}}});
    const auto columns = guided::TrackColumns(step, context);
    Check(columns.size() == 5 && columns[1].names.back() == "chapter_split", "look up adds chapter_split");
    std::string result;
    for (const auto& name : columns[4].names) result += name + ",";
    Check(result == "sermon_id,title,text,word_count,split,", "columns after choose: " + result);

    const auto generated = guided::GenerateSql(step, context);
    Check(generated.ok, "the example generates: " + generated.errors[0] + generated.errors[1] + generated.errors[2] + generated.errors[3]);
    Check(generated.sql.find("LEFT JOIN \"manifest\" r ON r.\"partition_document_id\" = CAST(l.\"book_code\" AS VARCHAR) || ' ' || CAST(l.\"chapter\" AS VARCHAR) || ':1'") !=
              std::string::npos,
          "the composite key:\n" + generated.sql);
    Check(generated.sql.find("\"word_count\" >= '5000'") != std::string::npos, "at least 5000");
    Check(generated.sql.find("\"chapter_split\" AS \"split\"") != std::string::npos, "rename");
    Check(generated.sql.find("error('Check failed: sermon_id repeats')") != std::string::npos, "the unique check");
    // The outline reads one part per operation, with its summary as the description.
    const auto outline = sqlstep::OutlineSqlQuery(generated.sql);
    Check(outline.structured && outline.parts.size() == 5, "4 parts and the result, got " + std::to_string(outline.parts.size()));
    if (outline.parts.size() == 5) {
        Check(outline.parts[0].name == "look_up_1" && outline.parts[3].name == "add_checks_4", "part names follow the operations");
        Check(outline.parts[1].description == "2. Keep rows where chapter_split is not empty and word_count at least 5000.",
              "descriptions are the summaries: " + outline.parts[1].description);
    }

    // An operation that is off is left out; the next reads the one before.
    step.ops[1].on = false;
    const auto skipped = guided::GenerateSql(step, context);
    Check(skipped.part_names[1].empty() && skipped.sql.find("FROM look_up_1") != std::string::npos, "an operation that is off is skipped");
    step.ops[1].on = true;

    // Errors name the problem and leave the operation out.
    step.ops[1].params["conditions"][1]["value"] = "";
    const auto broken = guided::GenerateSql(step, context);
    Check(!broken.ok && broken.errors[1].find("give a value") != std::string::npos && broken.part_names[1].empty(),
          "a missing value is reported: " + broken.errors[1]);
    guided::GuidedStep empty;
    Check(guided::GenerateSql(empty, context).sql.find("SELECT * FROM \"sermons\"") != std::string::npos, "no operations passes the input on");
}

void TestEveryOperationGenerates() {
    const auto context = SermonContext();
    for (const auto& info : guided::Library()) {
        guided::GuidedStep step;
        json params = guided::DefaultParams(info.id, "text");
        // Fill what the defaults leave blank.
        const std::string id = info.id;
        if (id == "find_replace") params["find"] = "x";
        if (id == "keep_between") params["start"] = "START";
        if (id == "extract_pattern") params["pattern"] = "[0-9]+";
        if (id == "look_up") {
            params["table"] = "manifest";
            params["key"] = json::array({{{"kind", "column"}, {"value", "split"}}});
            params["other_key"] = "split";
            params["bring"] = json::array({{{"column", "partition_document_id"}, {"as", "doc"}}});
        }
        if (id == "append_rows") params["table"] = "manifest";
        if (id == "pivot_unpivot") {
            params["names_from"] = "split";
            params["values"] = "train, test";
            params["values_from"] = "word_count";
        }
        if (id == "choose_columns") params["columns"] = json::array({{{"name", "title"}, {"keep", true}}});
        step.ops.push_back({id, true, params});
        const auto generated = guided::GenerateSql(step, context);
        Check(generated.ok, id + " generates: " + generated.errors[0]);
        Check(!guided::Summarise(step.ops[0]).empty(), id + " has a summary");
    }
}
}  // namespace

int main() {
    TestLibrary();
    TestJson();
    TestColumnsAndSql();
    TestEveryOperationGenerates();
    if (failures) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "Guided SQL steps passed: " << checks << " checks\n";
    return 0;
}
