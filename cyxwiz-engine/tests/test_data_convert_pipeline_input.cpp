#include "core/arrow_dataset.h"
#include "core/data_registry.h"
#include "core/pipeline_executor.h"

#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>

void RunDataConvertPipelineReloadTests(const std::filesystem::path& work_dir);

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::string JsonEscapePath(const std::string& path) {
    std::string escaped;
    escaped.reserve(path.size());
    for (char c : path) {
        escaped += (c == '\\') ? "\\\\" : std::string(1, c);
    }
    return escaped;
}

} // namespace

int main() try {
    namespace fs = std::filesystem;
    const fs::path work_dir = fs::temp_directory_path() /
        ("cyxwiz_convert_graph_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    Check(fs::create_directory(work_dir), "fixture workspace must be unique");
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); }
    } cleanup{work_dir};
    const fs::path csv_path = work_dir / "input.csv";
    const fs::path parquet_path = work_dir / "upstream.parquet";

    {
        std::ofstream csv(csv_path, std::ios::binary);
        csv << "id,value,label\n";
        csv << "1,1.5,a\n";
        csv << "2,2.5,b\n";
        csv << "3,3.5,a\n";
    }

    const std::string graph_json =
        R"({"nodes":[)"
        R"({"id":1,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":2,"type":"DataConvert","name":"Convert","parameters":{)"
        R"("output_path":")" + JsonEscapePath(parquet_path.string()) +
        R"(","output_format":"parquet","overwrite":"true","write_manifest":"true"}})"
        R"(],"links":[{"start_node":1,"end_node":2}]})";

    cyxwiz::PipelineExecutor executor;
    executor.SetIngestionCacheRoot((work_dir / "ingestion").string());
    Check(executor.ExecutePipeline(graph_json),
          "DataConvert should accept an upstream dataset without input_path: " +
              executor.GetLastError());

    Check(fs::exists(parquet_path),
          "DataConvert should write the Parquet output");
    Check(fs::exists(parquet_path.string() + ".manifest.json"),
          "DataConvert should write the sidecar manifest");

    auto dataset = cyxwiz::DataRegistry::Instance().GetArrowDataset(
        "ds_dataconvert_2");
    Check(dataset != nullptr,
          "DataConvert should register its converted output dataset");
    Check(dataset->GetNumRows() == 3,
          "DataConvert output dataset should preserve row count");
    Check(dataset->GetNumColumns() == 3,
          "DataConvert output dataset should preserve column count");

    RunDataConvertPipelineReloadTests(work_dir);
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
