// What a pipeline run registers is released when its nodes leave the graph
// (TOFIX160 P2): a node's own results go, a name another kept node uses
// stays, a Save output the user named stays, and ReleaseAll leaves only that.
#include "core/data_registry.h"
#include "core/data_budget.h"
#include "core/pipeline_executor.h"

#include <arrow/memory_pool.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string JsonEscapePath(const std::string& path) {
    std::string escaped;
    for (char c : path) escaped += (c == '\\') ? "\\\\" : std::string(1, c);
    return escaped;
}

bool Held(const std::string& name) {
    return cyxwiz::DataRegistry::Instance().GetArrowDataset(name) != nullptr ||
           cyxwiz::DataRegistry::Instance().GetParquetBackedDataset(name) != nullptr;
}

}  // namespace

int main() try {
    namespace fs = std::filesystem;
    const fs::path work_dir = fs::temp_directory_path() /
        ("cyxwiz_run_release_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Check(fs::create_directory(work_dir), "fixture workspace must be unique");
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); }
    } cleanup{work_dir};
    const fs::path csv_path = work_dir / "input.csv";
    const fs::path parquet_path = work_dir / "converted.parquet";
    {
        std::ofstream csv(csv_path, std::ios::binary);
        csv << "id,value,label\n1,1.5,a\n2,2.5,b\n3,3.5,a\n";
    }

    // Data Input -> Data Convert -> Save (named by the user).
    const std::string graph_json =
        R"({"nodes":[)"
        R"({"id":1,"type":"DataInput","name":"Input","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
        R"(","type":"csv","has_header":"true"}},)"
        R"({"id":2,"type":"DataConvert","name":"Convert","parameters":{)"
        R"("output_path":")" + JsonEscapePath(parquet_path.string()) +
        R"(","output_format":"parquet","overwrite":"true"}},)"
        R"({"id":3,"type":"SaveDataset","name":"Save","parameters":{"name":"kept_output"}})"
        R"(],"links":[{"start_node":1,"end_node":2},{"start_node":2,"end_node":3}]})";

    cyxwiz::PipelineExecutor executor;
    executor.SetIngestionCacheRoot((work_dir / "ingestion").string());
    Check(executor.ExecutePipeline(graph_json), "the pipeline runs: " + executor.GetLastError());
    Check(Held("ds_datainput_1"), "the run registered its copy of the input");
    Check(Held("ds_dataconvert_2"), "the run registered the convert step's output");
    Check(Held("kept_output"), "the run registered the Save output");

    // The convert step leaves the graph: its output goes, the rest stays.
    executor.ReleaseNodes({2});
    Check(!Held("ds_dataconvert_2"), "a released node's output is unregistered");
    Check(Held("ds_datainput_1"), "a kept node's result stays");
    Check(Held("kept_output"), "the Save output stays");
    Check(executor.NodeResults().count(2) == 0, "the released node's result is forgotten");

    // Releasing a node twice, or one never run, does nothing.
    executor.ReleaseNodes({2, 99});
    Check(Held("ds_datainput_1"), "releasing an unknown node touches nothing");

    // The graph is cleared: everything it registered goes, except what the user named.
    executor.ReleaseAll();
    Check(!Held("ds_datainput_1"), "ReleaseAll unregisters the input copy");
    Check(Held("kept_output"), "ReleaseAll keeps the user-named Save output");
    Check(executor.NodeResults().empty(), "ReleaseAll forgets every result");

    // A second run registers again (nothing stale left behind).
    Check(executor.ExecutePipeline(graph_json), "the pipeline runs again: " + executor.GetLastError());
    Check(Held("ds_datainput_1") && Held("ds_dataconvert_2"), "the second run registers its names again");

    cyxwiz::DataRegistry::Instance().ClearAllTabularDatasets();

    // Lifecycle (TOFIX160 P8): graph A loads and runs, graph B replaces it,
    // the project closes - the registry and Arrow's memory pool come back to
    // where they started.
    auto& registry = cyxwiz::DataRegistry::Instance();
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    const int64_t baseline = pool->bytes_allocated();
    const fs::path big_csv = work_dir / "big.csv";
    {
        std::ofstream csv(big_csv, std::ios::binary);
        csv << "id,value,label\n";
        for (int i = 0; i < 100000; ++i) csv << i << ',' << (i * 0.5) << ",row" << (i % 97) << '\n';
    }
    const std::string graph_b =
        R"({"nodes":[{"id":1,"type":"DataInput","name":"Big","parameters":{)"
        R"("source_type":"file","file_path":")" + JsonEscapePath(big_csv.string()) +
        R"(","type":"csv","has_header":"true"}}],"links":[]})";
    {
        cyxwiz::PipelineExecutor graph_a_run;
        graph_a_run.SetIngestionCacheRoot((work_dir / "ingestion").string());
        Check(graph_a_run.ExecutePipeline(graph_json), "graph A runs: " + graph_a_run.GetLastError());
        // Graph B replaces A on the canvas: A's run data goes (the editor's ReleaseAll).
        graph_a_run.ReleaseAll();
        Check(!Held("ds_datainput_1") && !Held("ds_dataconvert_2"), "graph A's run data is released");
    }
    {
        cyxwiz::PipelineExecutor graph_b_run;
        graph_b_run.SetIngestionCacheRoot((work_dir / "ingestion").string());
        Check(graph_b_run.ExecutePipeline(graph_b), "graph B runs: " + graph_b_run.GetLastError());
        Check(Held("ds_datainput_1"), "graph B's input is held");
        const auto loaded = registry.LoadedData();
        Check(loaded.memory_bytes > 1000000, "graph B's 100,000 rows are held in memory");
        Check(pool->bytes_allocated() > baseline + 1000000, "Arrow's pool holds them");
        // The project closes.
        registry.ClearAllTabularDatasets();
        graph_b_run.ReleaseAll();
    }
    Check(registry.LoadedData().entries.empty(), "nothing is held after the project closes");
    Check(pool->bytes_allocated() <= baseline + 256 * 1024,
          "Arrow's pool is back near its baseline (" + std::to_string(pool->bytes_allocated() - baseline) +
              " bytes over)");

    // Two executors ran the same graph (a run started while one ran): the
    // older one's release keeps the names the current one owns.
    {
        cyxwiz::PipelineExecutor older, current;
        older.SetIngestionCacheRoot((work_dir / "ingestion").string());
        current.SetIngestionCacheRoot((work_dir / "ingestion").string());
        Check(older.ExecutePipeline(graph_json), "older run: " + older.GetLastError());
        Check(current.ExecutePipeline(graph_json), "current run: " + current.GetLastError());
        const auto owned = current.OwnedNames();
        Check(owned.count("ds_datainput_1") && owned.count("ds_dataconvert_2"), "the current executor owns its names");
        Check(!owned.count("kept_output"), "a Save output is not owned for release");
        older.ReleaseAll(owned);
        Check(Held("ds_datainput_1") && Held("ds_dataconvert_2"), "the older release keeps the current run's names");
        current.ReleaseAll();
        Check(!Held("ds_datainput_1") && !Held("ds_dataconvert_2"), "the current release then frees them");
        registry.ClearAllTabularDatasets();
    }

    // A one-shot run (a plot's data) hands its names to an executor that
    // outlives it; deleting the node then frees them (its run copy stayed).
    {
        cyxwiz::PipelineExecutor owner;
        {
            cyxwiz::PipelineExecutor one_shot;
            one_shot.SetIngestionCacheRoot((work_dir / "ingestion").string());
            Check(one_shot.ExecutePipeline(graph_json), "one-shot run: " + one_shot.GetLastError());
            owner.AdoptOwnership(one_shot);
            Check(one_shot.OwnedNames().empty(), "the one-shot run owns nothing after handing over");
        }
        Check(Held("ds_datainput_1") && Held("ds_dataconvert_2"), "its names outlive the one-shot executor");
        const auto adopted = owner.OwnedNames();
        Check(adopted.count("ds_datainput_1") && adopted.count("ds_dataconvert_2") && !adopted.count("kept_output"),
              "the owner holds the run's names, not the Save output");
        owner.ReleaseNodes({1});
        Check(!Held("ds_datainput_1"), "releasing the deleted input frees its run copy");
        Check(Held("ds_dataconvert_2"), "another node's result stays");
        owner.ReleaseAll();
        Check(!Held("ds_dataconvert_2") && Held("kept_output"), "ReleaseAll frees the rest, the Save output stays");
        registry.ClearAllTabularDatasets();
    }

    // Training reads the canvas run's result while it is current (TOFIX178
    // S0-d): Data Input -> Standard Scaler -> Min-Max Scaler.
    {
        const auto graph = [&](const std::string& with_mean, const std::string& max) {
            return std::string(R"({"nodes":[)"
                               R"({"id":11,"type":"DataInput","name":"Input","parameters":{)"
                               R"("source_type":"file","file_path":")") + JsonEscapePath(csv_path.string()) +
                   R"(","type":"csv","has_header":"true"}},)"
                   R"({"id":12,"type":"StandardScaler","name":"Scale","parameters":{"columns":"value","with_mean":")" +
                   with_mean + R"(","with_std":"true"}},)"
                   R"({"id":13,"type":"MinMaxScaler","name":"Range","parameters":{"columns":"value","min":"0","max":")" +
                   max + R"("}}],"links":[{"start_node":11,"end_node":12},{"start_node":12,"end_node":13}]})";
        };
        cyxwiz::PipelineExecutor canvas;
        canvas.SetIngestionCacheRoot((work_dir / "ingestion").string());
        Check(!canvas.CurrentNodeResult(graph("true", "1"), 12), "nothing is current before a run");
        Check(canvas.ExecutePipeline(graph("true", "1")), "the canvas run: " + canvas.GetLastError());
        const std::string scaled = canvas.NodeResults().at(12);
        const auto current = canvas.CurrentNodeResult(graph("true", "1"), 12);
        Check(current && *current == scaled, "an unchanged step's result is current: " + scaled);
        const auto downstream_edit = canvas.CurrentNodeResult(graph("true", "2"), 12);
        Check(downstream_edit && *downstream_edit == scaled, "an edit after the step keeps it current");
        Check(!canvas.CurrentNodeResult(graph("false", "1"), 12), "an edit of the step makes it stale");
        Check(!canvas.CurrentNodeResult(graph("false", "1"), 13), "an edit before a step makes it stale");
        Check(Held(scaled), "asking ran nothing and released nothing");
        cyxwiz::DataRegistry::Instance().UnloadDataset(scaled);
        Check(!canvas.CurrentNodeResult(graph("true", "1"), 12), "an unloaded result is not current");
        canvas.ReleaseAll();
        registry.ClearAllTabularDatasets();
    }

    // The runtime memory guard (TOFIX178 S0-e2): a step during which the
    // process grows by the safe budget stops with the numbers. The probe
    // reports 1 TB of growth after the step starts.
    {
        const std::string scale =
            R"({"nodes":[{"id":21,"type":"DataInput","name":"Input","parameters":{)"
            R"("source_type":"file","file_path":")" + JsonEscapePath(csv_path.string()) +
            R"(","type":"csv","has_header":"true"}},)"
            R"({"id":22,"type":"StandardScaler","name":"Scale","parameters":{"columns":"value"}}],)"
            R"("links":[{"start_node":21,"end_node":22}]})";
        cyxwiz::PipelineExecutor guarded;
        guarded.SetIngestionCacheRoot((work_dir / "ingestion_guard").string());
        int reads = 0;
        guarded.SetProcessMemoryProbe([&reads] {
            cyxwiz::ProcessMemorySnapshot snapshot;
            snapshot.detected = true;
            snapshot.resident_bytes = (1ULL << 30) + (reads++ == 0 ? 0 : (1ULL << 40));
            return snapshot;
        });
        const bool ran = guarded.ExecutePipeline(scale);
        const std::string error = guarded.GetLastError();
        Check(!ran && error.find("the memory guard stopped 'Scale'") != std::string::npos,
              "the guard stops the step with its reason: " + error);
        Check(error.find("Sample Rows") != std::string::npos, "the reason says what to do");
        guarded.ReleaseAll();
        cyxwiz::PipelineExecutor unguarded;
        unguarded.SetIngestionCacheRoot((work_dir / "ingestion_guard").string());
        Check(unguarded.ExecutePipeline(scale), "the same step runs when memory stays: " + unguarded.GetLastError());
        unguarded.ReleaseAll();
        registry.ClearAllTabularDatasets();
    }

    // Unload from memory frees a dataset from every map (TOFIX160 P6).
    const auto table = cyxwiz::DataRegistry::Instance().LoadCSVToArrow(big_csv.string(), "unload_me");
    Check(table != nullptr && Held("unload_me"), "a dataset is loaded to unload");
    registry.UnloadDataset("unload_me");
    Check(!Held("unload_me"), "UnloadDataset removes an Arrow dataset");

    std::cout << "test_pipeline_run_release: all checks passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
