// Out-of-core baseline benchmark (TOFIX154 step 1).
//
// Measures how today's Data Studio pipeline behaves when the data is large
// next to the memory it may use. The principle is the ratio data/budget, so
// 1/5/20 GB files under a 512 MB or 2 GB budget stand in for terabytes.
//
//   cyxwiz-bench-out-of-core generate --dir D --gb N
//       Writes bench.csv and bench.parquet (tier1-like rows: id, split,
//       category, value, score, text), streamed in 128K-row groups so the
//       generator itself stays small.
//   cyxwiz-bench-out-of-core run --pipeline P --file F [--disk] [--budget-mb B]
//       Runs one pipeline (load, filter_select, groupby, sort, scaler) through
//       PipelineExecutor and prints one JSON line. --disk sets Data Input's
//       force_disk_backed (the Memory tab's "Force disk-backed cache").
//   cyxwiz-bench-out-of-core probe --file F
//       Executor bookkeeping (inventory item 10): two independent SQL steps
//       after one Data Input, run twice on the same executor. Serial branches
//       show as total = load + both steps; lazy reuse as a faster second run.
//   cyxwiz-bench-out-of-core suite --dir D --budget-mb M [--out results.jsonl] [--only P] [--parquet-only]
//       [--device cpu]
//       Runs every pipeline on both files (the CSV also with --disk), each in
//       a child process under a
//       Job Object memory limit (the budget), and records time, peak memory
//       and the outcome.

#include "core/data_budget.h"
#include <cyxwiz/device.h>
#include "core/pipeline_executor.h"
#include "core/pipeline_node_run_facts.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr int64_t kRowGroup = 128 * 1024;
const char* const kPipelines[] = {"load", "filter_select", "groupby", "sort", "scaler"};

std::string Arg(int argc, char** argv, const std::string& name, const std::string& fallback = "") {
    for (int i = 2; i + 1 < argc; ++i) {
        if (name == argv[i]) return argv[i + 1];
    }
    return fallback;
}

int Usage() {
    std::cerr << "usage: cyxwiz-bench-out-of-core generate --dir D --gb N\n"
                 "       cyxwiz-bench-out-of-core run --pipeline P --file F [--disk] [--budget-mb B]\n"
                 "       cyxwiz-bench-out-of-core probe --file F\n"
                 "       cyxwiz-bench-out-of-core suite --dir D --budget-mb M [--out results.jsonl] [--only P] [--parquet-only] [--device cpu]\n";
    return 2;
}

// ---------------------------------------------------------------- generate

struct Batch {
    std::vector<int64_t> id;
    std::vector<std::string> split, category, text;
    std::vector<double> value, score;
};

const char* const kWords[] = {
    "the", "word", "light", "grace", "faith", "spirit", "truth", "life", "love", "peace",
    "hope", "law", "king", "people", "heart", "lord", "glory", "mercy", "power", "earth",
    "heaven", "water", "bread", "way", "kingdom", "son", "father", "covenant", "promise", "joy"};

void FillBatch(Batch& b, int64_t first, int64_t rows, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> normal(50.0, 15.0);
    std::uniform_int_distribution<int> word(0, 29), length(20, 40), group(0, 999);
    b = Batch{};
    for (int64_t r = 0; r < rows; ++r) {
        const double s = unit(rng);
        b.id.push_back(first + r);
        b.split.push_back(s < 0.8 ? "train" : s < 0.9 ? "validation" : "test");
        b.category.push_back("c" + std::to_string(group(rng)));
        b.value.push_back(normal(rng));
        b.score.push_back(unit(rng));
        std::string t;
        for (int w = length(rng); w > 0; --w) {
            if (!t.empty()) t += ' ';
            t += kWords[word(rng)];
        }
        b.text.push_back(std::move(t));
    }
}

std::shared_ptr<arrow::Table> ToTable(const Batch& b) {
    arrow::Int64Builder id;
    arrow::StringBuilder split, category, text;
    arrow::DoubleBuilder value, score;
    (void)id.AppendValues(b.id);
    (void)split.AppendValues(b.split);
    (void)category.AppendValues(b.category);
    (void)value.AppendValues(b.value);
    (void)score.AppendValues(b.score);
    (void)text.AppendValues(b.text);
    auto schema = arrow::schema({arrow::field("id", arrow::int64()), arrow::field("split", arrow::utf8()),
                                 arrow::field("category", arrow::utf8()), arrow::field("value", arrow::float64()),
                                 arrow::field("score", arrow::float64()), arrow::field("text", arrow::utf8())});
    return arrow::Table::Make(schema, {id.Finish().ValueOrDie(), split.Finish().ValueOrDie(),
                                       category.Finish().ValueOrDie(), value.Finish().ValueOrDie(),
                                       score.Finish().ValueOrDie(), text.Finish().ValueOrDie()});
}

int Generate(const fs::path& dir, double gb) {
    fs::create_directories(dir);
    const auto target = static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
    std::ofstream csv(dir / "bench.csv", std::ios::binary);
    csv << "id,split,category,value,score,text\n";
    auto out = arrow::io::FileOutputStream::Open((dir / "bench.parquet").string()).ValueOrDie();
    std::unique_ptr<parquet::arrow::FileWriter> writer;
    std::mt19937_64 rng(154);
    Batch b;
    int64_t rows = 0;
    char number[64];
    // The CSV size sets the row count: "N GB" means an N GB CSV file.
    while (static_cast<uint64_t>(csv.tellp()) < target) {
        FillBatch(b, rows, kRowGroup, rng);
        for (size_t r = 0; r < b.id.size(); ++r) {
            csv << b.id[r] << ',' << b.split[r] << ',' << b.category[r] << ',';
            std::snprintf(number, sizeof(number), "%.6f,%.6f,", b.value[r], b.score[r]);
            csv << number << b.text[r] << '\n';
        }
        auto table = ToTable(b);
        if (!writer) {
            auto props = parquet::WriterProperties::Builder().compression(parquet::Compression::SNAPPY)->build();
            writer = parquet::arrow::FileWriter::Open(*table->schema(), arrow::default_memory_pool(), out, props)
                         .ValueOrDie();
        }
        if (!writer->WriteTable(*table, kRowGroup).ok()) {
            std::cerr << "parquet write failed\n";
            return 1;
        }
        rows += static_cast<int64_t>(b.id.size());
    }
    if (!writer->Close().ok() || !out->Close().ok()) return 1;
    csv.close();
    json report = {{"rows", rows}, {"csv_bytes", fs::file_size(dir / "bench.csv")},
                   {"parquet_bytes", fs::file_size(dir / "bench.parquet")}};
    std::ofstream(dir / "bench.json") << report.dump(2);
    std::cout << report.dump() << '\n';
    return 0;
}

// --------------------------------------------------------------------- run

json PipelineGraph(const std::string& pipeline, const fs::path& file, bool disk) {
    const bool parquet = file.extension() == ".parquet";
    json input = {{"id", 1}, {"type", "DataInput"}, {"name", "Bench"},
                  {"parameters", {{"source_type", "file"}, {"file_path", file.generic_string()},
                                  {"type", parquet ? "parquet" : "csv"}, {"has_header", "true"},
                                  {"force_disk_backed", disk ? "true" : "false"}}}};
    json step;
    if (pipeline == "filter_select") {
        step = {{"type", "SQLQuery"},
                {"parameters", {{"sql_contract_version", "1"}, {"input_alias", "input"},
                                {"query", "SELECT id, category, value FROM input WHERE score > 0.5"}}}};
    } else if (pipeline == "groupby") {
        step = {{"type", "GroupBy"},
                {"parameters", {{"group_columns", "category"}, {"aggregations", "COUNT(*) AS n, AVG(value) AS mean_value"}}}};
    } else if (pipeline == "sort") {
        step = {{"type", "SortRows"}, {"parameters", {{"columns", "value"}, {"order", "ASC"}}}};
    } else if (pipeline == "scaler") {
        step = {{"type", "StandardScaler"},
                {"parameters", {{"columns", "value,score"}, {"with_mean", "true"}, {"with_std", "true"}}}};
    }
    json graph = {{"nodes", json::array({input})}, {"links", json::array()}};
    if (!step.is_null()) {
        step["id"] = 2;
        step["name"] = pipeline;
        graph["nodes"].push_back(step);
        graph["links"].push_back({{"start_node", 1}, {"end_node", 2}});
    }
    return graph;
}

uint64_t PeakWorkingSet() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) return counters.PeakWorkingSetSize;
#endif
    return 0;
}

#ifdef _WIN32
// A crash under the budget gets a report naming the module it happened in,
// so the suite can say where (a device library, the CSV reader, DuckDB).
LONG WINAPI ReportCrash(EXCEPTION_POINTERS* info) {
    const auto address = info->ExceptionRecord->ExceptionAddress;
    HMODULE module = nullptr;
    char path[MAX_PATH] = "unknown";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(address), &module)) {
        GetModuleFileNameA(module, path, MAX_PATH);
    }
    const auto offset = static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(address) -
                                                        reinterpret_cast<uintptr_t>(module));
    json report = {{"crash", {{"code", static_cast<int64_t>(info->ExceptionRecord->ExceptionCode)},
                              {"module", fs::path(path).filename().string()}, {"offset", offset},
                              {"peak_working_set_mb", PeakWorkingSet() / (1024.0 * 1024.0)}}}};
    std::cout << report.dump() << std::endl;
    return EXCEPTION_CONTINUE_SEARCH;
}
#endif

int Run(const std::string& pipeline, const fs::path& file, bool disk) {
#ifdef _WIN32
    SetUnhandledExceptionFilter(ReportCrash);
#endif
    json report = {{"pipeline", pipeline}, {"file", file.filename().string()}, {"disk", disk},
                   {"data_budget_mb", cyxwiz::DataBudgetBytes() >> 20}};
    const auto start = std::chrono::steady_clock::now();
    bool ok = false;
    try {
        cyxwiz::PipelineExecutor executor;
        // The CSV ingestion cache sits next to the data; the suite empties it
        // before every run, so each run starts cold.
        executor.SetIngestionCacheRoot((file.parent_path() / "cache").string());
        ok = executor.ExecutePipeline(PipelineGraph(pipeline, file, disk).dump());
        if (!ok) report["error"] = executor.GetLastError();
        for (const auto& [id, facts] : executor.GetNodeRunFacts()) {
            report["nodes"][std::to_string(id)] = {{"rows", facts.rows}, {"columns", facts.columns},
                                                   {"seconds", facts.seconds}, {"error", facts.error}};
        }
    } catch (const std::bad_alloc&) {
        report["error"] = "out of memory (std::bad_alloc)";
    } catch (const std::exception& e) {
        report["error"] = std::string("exception: ") + e.what();
    }
    report["ok"] = ok;
    report["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    report["peak_working_set_mb"] = PeakWorkingSet() / (1024.0 * 1024.0);
    std::cout << report.dump() << '\n';
    return ok ? 0 : 1;
}

// ------------------------------------------------------------------- probe

int Probe(const fs::path& file) {
    json graph = PipelineGraph("load", file, false);
    const char* const queries[] = {"SELECT category, COUNT(*) AS n FROM input GROUP BY category",
                                   "SELECT split, AVG(value) AS mean_value FROM input GROUP BY split"};
    for (int i = 0; i < 2; ++i) {
        graph["nodes"].push_back({{"id", 2 + i}, {"type", "SQLQuery"}, {"name", "branch " + std::to_string(i + 1)},
                                  {"parameters", {{"sql_contract_version", "1"}, {"input_alias", "input"},
                                                  {"query", queries[i]}}}});
        graph["links"].push_back({{"start_node", 1}, {"end_node", 2 + i}});
    }
    cyxwiz::PipelineExecutor executor;
    executor.SetIngestionCacheRoot((file.parent_path() / "cache").string());
    json report = {{"file", file.filename().string()}, {"runs", json::array()}};
    for (int run = 0; run < 2; ++run) {
        const auto start = std::chrono::steady_clock::now();
        const bool ok = executor.ExecutePipeline(graph.dump());
        json entry = {{"ok", ok}, {"seconds", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()}};
        double steps = 0.0;
        for (const auto& [id, facts] : executor.GetNodeRunFacts()) {
            entry["nodes"][std::to_string(id)] = facts.seconds;
            steps += facts.seconds;
        }
        entry["sum_of_node_seconds"] = steps;
        report["runs"].push_back(entry);
    }
    std::cout << report.dump() << '\n';
    return 0;
}

// ------------------------------------------------------------------- suite

#ifdef _WIN32
json RunChild(const std::string& self, const std::string& pipeline, const fs::path& file, bool disk,
              uint64_t budget, const std::string& device) {
    json result = {{"pipeline", pipeline}, {"file", file.filename().string()}, {"disk", disk},
                   {"budget_mb", budget >> 20}};
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limit{};
    limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    limit.JobMemoryLimit = budget;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limit, sizeof(limit));

    const fs::path log = fs::temp_directory_path() / ("cyxwiz_bench_" + std::to_string(GetCurrentProcessId()) + ".jsonl");
    SECURITY_ATTRIBUTES inherit{sizeof(inherit), nullptr, TRUE};
    HANDLE out = CreateFileW(log.wstring().c_str(), GENERIC_WRITE, FILE_SHARE_READ, &inherit, CREATE_ALWAYS,
                             FILE_ATTRIBUTE_NORMAL, nullptr);
    std::wstring command = L"\"" + fs::path(self).wstring() + L"\" run --pipeline " + fs::path(pipeline).wstring() +
                           L" --file \"" + file.wstring() + L"\"" + (disk ? L" --disk" : L"") +
                           L" --budget-mb " + std::to_wstring((budget >> 20) / 2) +
                           (device.empty() ? L"" : L" --device " + fs::path(device).wstring());
    STARTUPINFOW startup{sizeof(startup)};
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = out;
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    const auto start = std::chrono::steady_clock::now();
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr,
                        &startup, &process)) {
        result["error"] = "could not start the child process";
        CloseHandle(out);
        CloseHandle(job);
        return result;
    }
    AssignProcessToJobObject(job, process.hProcess);
    ResumeThread(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(process.hProcess, &code);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION used{};
    QueryInformationJobObject(job, JobObjectExtendedLimitInformation, &used, sizeof(used), nullptr);
    result["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    result["peak_commit_mb"] = used.PeakJobMemoryUsed / (1024.0 * 1024.0);
    result["exit_code"] = static_cast<int64_t>(static_cast<int32_t>(code));
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(out);
    CloseHandle(job);

    std::ifstream child(log);
    std::string line, last;
    while (std::getline(child, line)) {
        if (!line.empty() && line.front() == '{') last = line;
    }
    child.close();
    fs::remove(log);
    if (!last.empty()) {
        const json run = json::parse(last, nullptr, false);
        if (!run.is_discarded()) result["run"] = run;
    }
    // ok: finished and right; failed: reported an error; crashed: killed or died
    // without a report (the budget was hit inside a library that cannot fail cleanly).
    const bool crashed = !result.contains("run") || result["run"].contains("crash");
    result["outcome"] = code == 0 ? "ok" : crashed ? "crashed" : "failed";
    return result;
}
#endif

int Suite(const std::string& self, const fs::path& dir, uint64_t budget_mb, const fs::path& out,
          const std::string& only, bool parquet_only, const std::string& device) {
#ifdef _WIN32
    std::ofstream results(out, std::ios::app);
    int failures = 0;
    const std::pair<const char*, bool> inputs[] = {{"bench.parquet", false}, {"bench.csv", false}, {"bench.csv", true}};
    for (const auto& [name, disk] : inputs) {
        if (parquet_only && std::string(name) != "bench.parquet") continue;
        for (const char* pipeline : kPipelines) {
            if (!only.empty() && only != pipeline) continue;
            fs::remove_all(dir / "cache");
            json result = RunChild(self, pipeline, dir / name, disk, budget_mb << 20, device);
            if (!device.empty()) result["device"] = device;
            result["data"] = dir.filename().string();
            std::cout << result.dump() << '\n';
            results << result.dump() << '\n';
            results.flush();
            if (result["outcome"] != "ok") ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
#else
    (void)self; (void)dir; (void)budget_mb; (void)out; (void)only; (void)parquet_only; (void)device;
    std::cerr << "suite needs Windows Job Objects\n";
    return 2;
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return Usage();
    const std::string command = argv[1];
    if (command == "generate") {
        const std::string dir = Arg(argc, argv, "--dir");
        if (dir.empty()) return Usage();
        return Generate(dir, std::stod(Arg(argc, argv, "--gb", "1")));
    }
    if (command == "run") {
        const std::string pipeline = Arg(argc, argv, "--pipeline"), file = Arg(argc, argv, "--file");
        if (pipeline.empty() || file.empty()) return Usage();
        bool disk = false;
        for (int i = 2; i < argc; ++i) disk = disk || std::string(argv[i]) == "--disk";
        // The suite passes half its memory cap: the D2 rule (half of RAM) with
        // the cap standing in for the machine.
        const std::string budget = Arg(argc, argv, "--budget-mb");
        if (!budget.empty()) cyxwiz::SetDataBudgetBytes(std::stoull(budget) << 20);
        // --device cpu: ArrayFire's CPU backend, as chosen in Preferences; a
        // GPU driver's start-up does not fit a few hundred MB cap.
        if (Arg(argc, argv, "--device") == "cpu") cyxwiz::Device::RecordProcessDevice(cyxwiz::DeviceType::CPU, 0);
        return Run(pipeline, file, disk);
    }
    if (command == "probe") {
        const std::string file = Arg(argc, argv, "--file");
        if (file.empty()) return Usage();
        return Probe(file);
    }
    if (command == "suite") {
        const std::string dir = Arg(argc, argv, "--dir");
        if (dir.empty()) return Usage();
        const fs::path out = Arg(argc, argv, "--out", (fs::path(dir) / "results.jsonl").string());
        std::string self = argv[0];
#ifdef _WIN32
        wchar_t module[MAX_PATH];
        if (GetModuleFileNameW(nullptr, module, MAX_PATH)) self = fs::path(module).string();
#endif
        return Suite(self, dir, std::stoull(Arg(argc, argv, "--budget-mb", "2048")), out,
                     Arg(argc, argv, "--only"),
                     std::find(argv + 2, argv + argc, std::string("--parquet-only")) != argv + argc,
                     Arg(argc, argv, "--device"));
    }
    return Usage();
}
