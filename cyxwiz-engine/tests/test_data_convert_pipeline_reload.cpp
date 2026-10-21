#include "core/arrow_dataset.h"
#include "core/data_convert_service.h"
#include "core/data_registry.h"
#include "core/pipeline_executor.h"
#include <nlohmann/json.hpp>
#ifdef CYXWIZ_HAS_XLSX
#include <OpenXLSX.hpp>
#endif
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
namespace fs = std::filesystem;
int checks = 0;
void Check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
std::shared_ptr<arrow::Table> Registered(int node_id) {
    auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(
        "ds_dataconvert_" + std::to_string(node_id));
    return dataset ? dataset->GetArrowTable() : nullptr;
}
void Forget(int node_id) {
    cyxwiz::DataRegistry::Instance().UnregisterTabularDataset(
        "ds_dataconvert_" + std::to_string(node_id));
}
Json Graph(const cyxwiz::DataConvertOptions& options, int id, const fs::path& downstream) {
    Json convert = {{"id", id}, {"type", "DataConvert"}, {"name", "Reload"}};
    convert["parameters"] = {
        {"input_path", options.input_path}, {"output_path", options.output_path},
        {"input_format", options.input_format}, {"output_format", options.output_format},
        {"excel_sheet", options.excel_sheet}, {"skip_rows", std::to_string(options.skip_rows)},
        {"excel_start_column", options.excel_start_column},
        {"has_header", options.has_header ? "true" : "false"},
        {"delimiter", "auto"}, {"overwrite", "false"}, {"write_manifest", "true"}};
    Json next = {{"id", id + 1}, {"type", "DataConvert"}, {"name", "Downstream"}};
    next["parameters"] = {{"output_path", downstream.string()}, {"output_format", "parquet"},
        {"overwrite", "true"}, {"write_manifest", "false"}};
    return {{"nodes", Json::array({convert, next})},
            {"links", Json::array({{{"start_node", id}, {"end_node", id + 1}}})}};
}
std::shared_ptr<arrow::Table> Run(const Json& graph, int id, const std::string& label) {
    Forget(id);
    Forget(id + 1);
    cyxwiz::PipelineExecutor executor; // No lazy-execution cache carried between runs.
    std::set<int> completed;
    executor.SetNodeExecutionCallback([&](int node, cyxwiz::PipelineNodeExecutionEvent event, const std::string&) {
        if (event == cyxwiz::PipelineNodeExecutionEvent::Completed) completed.insert(node);
    });
    const bool ok = executor.ExecutePipeline(graph.dump());
    Check(ok, label + ": " + executor.GetLastError());
    Check(completed.count(id) == 1 && completed.count(id + 1) == 1, label + " both nodes executed");
    auto table = Registered(id);
    auto downstream = Registered(id + 1);
    Check(table && downstream && table->Equals(*downstream, false), label + " downstream values/schema");
    return table;
}
}

void RunDataConvertPipelineReloadTests(const fs::path& work_dir) {
    using cyxwiz::DataConvertOptions;
    using cyxwiz::DataConvertService;
    const auto numeric = work_dir / "numbers.csv";
    { std::ofstream out(numeric); out << "left,right\n1,2.25\n-3,4.75\n"; }
    const auto text = work_dir / "lines.txt";
    { std::ofstream out(text); out << "first line\nsecond line\n"; }
    std::vector<std::string> formats = {
        "csv", "tsv", "jsonl", "txt", "arff", "npy", "parquet", "feather", "arrow", "ipc"};
#ifdef CYXWIZ_HAS_HDF5
    formats.push_back("hdf5");
#endif
    int id = 100;
    for (const auto& format : formats) {
        DataConvertOptions options;
        options.input_path = (format == "txt" ? text : numeric).string();
        options.output_path = (work_dir / ("graph." + format)).string();
        options.output_format = format;
        options.auto_detect_delimiter = true;
        const auto graph = Graph(options, id, work_dir / (format + "_downstream.parquet"));
        auto first = Run(graph, id, format + " first run");
        DataConvertOptions reload;
        reload.input_path = options.output_path;
        reload.auto_detect_delimiter = true;
        std::string error;
        auto persisted = DataConvertService::LoadTable(reload, error);
        Check(persisted && first->Equals(*persisted, false), format + " first run matches persisted output: " + error);
        Check(first->num_rows() == 2 && first->num_columns() == (format == "txt" ? 1 : 2), format + " shape");
        const bool matrix = format == "npy" || format == "hdf5";
        Check(first->field(0)->name() == (format == "txt" ? "text" : matrix ? "col_0" : "left"), format + " field contract");
        const auto expected_type = format == "txt" ? arrow::Type::STRING :
            (matrix || format == "arff") ? arrow::Type::DOUBLE : arrow::Type::INT64;
        Check(first->field(0)->type()->id() == expected_type, format + " dtype contract");
        Check(first->column(0)->GetScalar(1).ValueOrDie()->ToString() == (format == "txt" ? "second line" : "-3"), format + " values");
        const auto stamp = fs::last_write_time(options.output_path);
        options.retain_output_table = true;
        auto cached = DataConvertService::Convert(options);
        Check(cached.ok && cached.skipped_fresh_output && !cached.output_table, format + " service requires disk reload");
        auto second = Run(graph, id, format + " fresh-cache run");
        Check(second->Equals(*first, false), format + " first/cache schema and value parity");
        Check(fs::last_write_time(options.output_path) == stamp, format + " cached output was not rewritten");
        Forget(id);
        Forget(id + 1);
        id += 2;
    }

    // A matching size/mtime manifest is only a cache hint. If the bytes are
    // unreadable, the actual graph disk branch must fail without publishing.
    DataConvertOptions corrupt;
    corrupt.input_path = numeric.string();
    corrupt.output_path = (work_dir / "graph.parquet").string();
    corrupt.auto_detect_delimiter = true;
    const auto stamp = fs::last_write_time(corrupt.output_path);
    {
        std::fstream out(corrupt.output_path, std::ios::binary | std::ios::in | std::ios::out);
        out.write("FAIL", 4);
        out.seekp(-4, std::ios::end);
        out.write("FAIL", 4);
        Check(static_cast<bool>(out), "corrupt fixture write");
    }
    fs::last_write_time(corrupt.output_path, stamp);
    Check(DataConvertService::Convert(corrupt).skipped_fresh_output, "corrupt fixture reaches cache reload, not writer");
    Forget(id);
    Forget(id + 1);
    cyxwiz::PipelineExecutor invalid_cache;
    auto corrupt_graph = Graph(corrupt, id, work_dir / "must_not_exist.parquet");
    Check(!invalid_cache.ExecutePipeline(corrupt_graph.dump()), "unreadable cached artifact must fail graph");
    Check(!Registered(id) && !Registered(id + 1) && !fs::exists(work_dir / "must_not_exist.parquet"),
          "failed disk reload cannot register or feed downstream data");
    id += 2;

#ifdef CYXWIZ_HAS_XLSX
    const auto workbook = work_dir / "sheets.xlsx";
    {
        OpenXLSX::XLDocument doc;
        doc.create(workbook.string(), OpenXLSX::XLDoNotOverwrite);
        auto first = doc.workbook().worksheet("Sheet1");
        first.cell("A1").value() = "wrong_sheet";
        first.cell("A2").value() = 99;
        doc.workbook().addWorksheet("Chosen sheet");
        auto chosen = doc.workbook().worksheet("Chosen sheet");
        chosen.cell("B1").value() = "preamble";
        chosen.cell("B2").value() = "score";
        chosen.cell("B3").value() = 7.25;
        doc.save();
    }
    DataConvertOptions excel;
    excel.input_path = workbook.string();
    excel.input_format = "xlsx";
    excel.output_path = (work_dir / "excel.parquet").string();
    excel.excel_sheet = "Chosen sheet";
    excel.excel_start_column = "B";
    excel.skip_rows = 1;
    excel.auto_detect_delimiter = true;
    auto graph = Graph(excel, id, work_dir / "excel_downstream.parquet");
    auto first = Run(graph, id, "named XLSX first run");
    Check(first->num_rows() == 1 && first->field(0)->name() == "score" &&
          first->column(0)->GetScalar(0).ValueOrDie()->ToString() == "7.25", "named sheet and skipped header reach graph");
    auto cached = DataConvertService::Convert(excel);
    Check(cached.ok && cached.skipped_fresh_output && !cached.output_table, "XLSX cached graph must reload disk");
    auto second = Run(graph, id, "named XLSX cached run");
    Check(second->Equals(*first, false), "XLSX first/cache parity");
    graph["nodes"][0]["parameters"]["excel_start_column"] = "A";
    Forget(id);
    Forget(id + 1);
    cyxwiz::PipelineExecutor changed_column;
    Check(!changed_column.ExecutePipeline(graph.dump()) && !Registered(id),
          "changed start column cannot reuse previous graph output");
    graph["nodes"][0]["parameters"]["excel_start_column"] = "B";
    graph["nodes"][0]["parameters"]["excel_sheet"] = "Missing";
    Forget(id);
    Forget(id + 1);
    cyxwiz::PipelineExecutor invalid;
    Check(!invalid.ExecutePipeline(graph.dump()) && !Registered(id) && !Registered(id + 1), "changed worksheet cannot reuse old output or publish stale registry data");
    graph["nodes"][0]["parameters"]["overwrite"] = "true";
    Check(!invalid.ExecutePipeline(graph.dump()) && invalid.GetLastError().find("does not exist") != std::string::npos, "missing sheet reports adapter error");
    graph["nodes"][0]["parameters"]["skip_rows"] = "4294967296";
    Check(!invalid.ExecutePipeline(graph.dump()) && invalid.GetLastError().find("skip_rows") != std::string::npos, "skip_rows cannot wrap to zero");
    graph["nodes"][0]["parameters"]["skip_rows"] = "2";
    graph["nodes"][0]["parameters"]["excel_sheet"] = "Chosen sheet";
    graph["nodes"][0]["parameters"]["has_header"] = "false";
    auto no_header = Run(graph, id, "XLSX no-header run");
    Check(no_header->num_rows() == 1 && no_header->field(0)->name() == "col_0", "XLSX header option reaches adapter");
#endif
    std::cout << "DataConvert graph reload: " << checks << " checks passed; HDF5 "
#ifdef CYXWIZ_HAS_HDF5
              << "enabled; XLSX "
#else
              << "unavailable; XLSX "
#endif
#ifdef CYXWIZ_HAS_XLSX
              << "enabled\n";
#else
              << "unavailable\n";
#endif
}
