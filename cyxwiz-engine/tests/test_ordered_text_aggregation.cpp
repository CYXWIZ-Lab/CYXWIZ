#include "core/arrow_dataset.h"
#include "core/data_registry.h"
#include "core/pipeline_executor.h"
#include <arrow/api.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <string>

namespace {
void Require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: ordered text aggregation: " << message << '\n';
        std::exit(1);
    }
}
void Ok(const arrow::Status& status) { Require(status.ok(), status.ToString()); }
}

void CheckOrderedTextAggregation() {
    namespace fs = std::filesystem;
    using json = nlohmann::json;
    auto& registry = cyxwiz::DataRegistry::Instance();
    const auto dir = fs::temp_directory_path() / ("cyxwiz_ordered_text_" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const auto input = dir / "input.parquet";
    const auto output = dir / "output.parquet";
    arrow::StringBuilder groups, words;
    arrow::Int64Builder sequence;
    arrow::BooleanBuilder flags;
    const std::string group_values[] = {"A", "A", "A", "A", "A", "B", "C", "D", "D"};
    const char* text_values[] = {"third", "first 'quoted'", "second", "alpha", nullptr, "other", nullptr, "tail", "head"};
    const int64_t positions[] = {3, 1, 2, 2, 4, 1, 1, 0, 1};
    for (int i = 0; i < 9; ++i) {
        Ok(groups.Append(group_values[i]));
        if (text_values[i]) Ok(words.Append(text_values[i])); else Ok(words.AppendNull());
        if (i == 7) Ok(sequence.AppendNull()); else Ok(sequence.Append(positions[i]));
        Ok(flags.Append(true));
    }
    std::shared_ptr<arrow::Array> g, w, s, f;
    Ok(groups.Finish(&g)); Ok(words.Finish(&w)); Ok(sequence.Finish(&s)); Ok(flags.Finish(&f));
    auto table = arrow::Table::Make(arrow::schema({
        arrow::field("book\"name", arrow::utf8()), arrow::field("verse text", arrow::utf8()),
        arrow::field("verse", arrow::int64()), arrow::field("flag", arrow::boolean())}), {g, w, s, f});
    Require(registry.RegisterArrowTable(table, "ordered_text_fixture") != nullptr, "register input");
    Require(registry.ExportArrowToParquet("ordered_text_fixture", input.string()), "write fixture");
    const json base_params = {{"group_columns", "book\"name"},
        {"aggregations", "STRING_AGG(verse text) AS chapter_text, COUNT(*) AS n, SUM(verse) AS total"},
        {"text_order_by", "verse"}};
    int next_id = 94600;
    auto run = [&](json parameters, const std::string& separator,
                   const std::string& expected_error = "") {
        const int in_id = next_id++, group_id = next_id++, export_id = next_id++;
        const std::string group_name = "ds_groupby_" + std::to_string(group_id);
        const json graph = {{"nodes", json::array({
            {{"id", in_id}, {"type", "DataInput"}, {"name", "Input"},
             {"parameters", {{"source_type", "file"}, {"file_type", "parquet"}, {"file_path", input.generic_string()}}}},
            {{"id", group_id}, {"type", "GroupBy"}, {"name", "Group"}, {"parameters", parameters}},
            {{"id", export_id}, {"type", "ExportParquet"}, {"name", "Export"},
             {"parameters", {{"file_path", output.generic_string()}}}}
        })}, {"links", json::array({{{"start_node", in_id}, {"end_node", group_id}},
                                    {{"start_node", group_id}, {"end_node", export_id}}})}};
        cyxwiz::PipelineExecutor executor;
        const bool succeeded = executor.ExecutePipeline(graph.dump());
        if (!expected_error.empty()) {
            Require(!succeeded, "invalid configuration must fail");
            Require(executor.GetLastError().find(expected_error) != std::string::npos,
                    "expected diagnostic '" + expected_error + "', found: " + executor.GetLastError());
            Require(!registry.GetArrowDataset(group_name), "invalid group must not publish output");
        } else {
            Require(succeeded, executor.GetLastError());
            auto dataset = registry.GetArrowDataset(group_name);
            Require(dataset != nullptr, "group output registered");
            auto actual = dataset->GetArrowTable();
            Ok(actual->ValidateFull());
            Require(actual->num_rows() == 4, "four isolated groups");
            std::map<std::string, std::string> expected = {
                {"A", "first 'quoted'" + separator + "alpha" + separator + "second" + separator + "third"},
                {"B", "other"}, {"D", "head" + separator + "tail"}};
            for (int64_t row = 0; row < actual->num_rows(); ++row) {
                auto key = actual->GetColumnByName("book\"name")->GetScalar(row);
                auto text = actual->GetColumnByName("chapter_text")->GetScalar(row);
                Require(key.ok() && text.ok(), "read grouped values");
                const auto group = (*key)->ToString();
                if (group == "C") Require(!(*text)->is_valid, "all-null text remains null");
                else Require((*text)->is_valid && (*text)->ToString() == expected.at(group),
                             "shuffled order, stable ties, nulls and separator for " + group);
                if (group == "A") {
                    auto count = actual->GetColumnByName("n")->GetScalar(row);
                    auto sum = actual->GetColumnByName("total")->GetScalar(row);
                    Require(count.ok() && sum.ok(), "numeric aggregates readable");
                    Require((*count)->ToString() == "5" && (*sum)->ToString() == "12", "numeric aggregates preserved");
                }
            }
            auto reloaded = registry.LoadParquetToArrow(output.string(), "ordered_text_reload");
            Require(reloaded && reloaded->GetArrowTable()->Equals(*actual), "Parquet round trip");
            registry.UnloadDataset("ordered_text_reload");
        }
        registry.UnloadDataset("ds_datainput_" + std::to_string(in_id));
        registry.UnloadDataset(group_name);
    };
    run(base_params, " ");
    auto params = base_params;
    params["text_separator"] = "';\n";
    run(params, "';\n");
    params["text_separator"] = "";
    run(params, "");
    params = base_params; params["text_order_by"] = "verse, verse text";
    run(params, " ");
    params = base_params; params.erase("text_order_by");
    run(params, "", "requires 'text_order_by'");
    params = base_params; params["text_order_by"] = "verse; DROP TABLE anything";
    run(params, "", "not found");
    params = base_params; params["aggregations"] = "STRING_AGG(verse) AS chapter_text";
    run(params, "", "string/large_string");
    params = base_params; params["text_order_by"] = "flag";
    run(params, "", "must be numeric or text");
    params = base_params; params["aggregations"] = "STRING_AGG(verse text) AS x; DROP TABLE anything";
    run(params, "", "not a valid identifier");
    params = base_params; params["text_separator"] = std::string(1, '\0');
    run(params, "", "cannot contain NUL");
    registry.UnloadDataset("ordered_text_fixture");
    fs::remove(input); fs::remove(output); fs::remove(dir);
    std::cout << "PASS: ordered text aggregation (4 exports, 6 rejected configurations)\n";
}
