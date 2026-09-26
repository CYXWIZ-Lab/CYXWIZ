#include "training_benchmark.h"

#include "compute_runtime_paths.h"
#include "execution_device_preferences.h"
#include "graph_training_job.h"
#include "training_trace_collector.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <cyxwiz/cyxwiz.h>
#include <cyxwiz/tokenizer.h>
#include <nlohmann/json.hpp>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

namespace cyxwiz {
namespace {

#include "training_benchmark_graph.inc"

constexpr int kVocabularySize = 8192;
constexpr int kContext = 256;
constexpr int kBatchSize = 8;
constexpr unsigned kDataSeed = 52;

namespace fs = std::filesystem;
using json = nlohmann::json;

double Percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(p * static_cast<double>(values.size() - 1) + 0.5);
    return values[std::min(index, values.size() - 1)];
}

std::string UtcNow() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

// Seeded token windows in the format the Engine's Token Windows step writes
// (list<int64> of context + 1 ids, window bounds, frozen vocabulary in the
// schema metadata). Ids avoid the special tokens.
bool WriteSyntheticTokenWindows(const fs::path& path, int rows, std::string& error) {
    Vocabulary vocabulary;
    vocabulary.SetVocabulary({});  // the special tokens take the first ids
    const auto specials = static_cast<int64_t>(vocabulary.Size());
    for (int word = 0; vocabulary.Size() < static_cast<size_t>(kVocabularySize); ++word) {
        vocabulary.AddWord("w" + std::to_string(word));
    }
    std::ostringstream artifact;
    if (!vocabulary.SaveToStream(artifact)) {
        error = "cannot serialize the benchmark vocabulary";
        return false;
    }

    std::mt19937 engine(kDataSeed);
    std::uniform_int_distribution<int64_t> token(specials, kVocabularySize - 1);
    auto values = std::make_shared<arrow::Int64Builder>();
    arrow::ListBuilder ids(arrow::default_memory_pool(), values, arrow::list(arrow::field("element", arrow::int64())));
    arrow::StringBuilder documents;
    arrow::Int64Builder indices, starts, ends, valid;
    for (int row = 0; row < rows; ++row) {
        if (!ids.Append().ok()) return false;
        for (int i = 0; i <= kContext; ++i) {
            if (!values->Append(token(engine)).ok()) return false;
        }
        if (!documents.Append("benchmark-" + std::to_string(row)).ok() || !indices.Append(0).ok() || !starts.Append(0).ok() ||
            !ends.Append(kContext + 1).ok() || !valid.Append(kContext).ok()) {
            error = "cannot build the benchmark windows";
            return false;
        }
    }
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    auto finish = [&](const char* name, arrow::ArrayBuilder& builder) {
        auto array = builder.Finish();
        if (!array.ok()) return false;
        fields.push_back(arrow::field(name, (*array)->type(), false));
        arrays.push_back(*array);
        return true;
    };
    if (!finish("document_id", documents) || !finish("token_ids", ids) || !finish("__window_index", indices) ||
        !finish("__token_start", starts) || !finish("__token_end", ends) || !finish("__valid_targets", valid)) {
        error = "cannot build the benchmark windows";
        return false;
    }
    auto metadata = std::make_shared<arrow::KeyValueMetadata>();
    metadata->Append("cyxwiz.token_windows.version", "1");
    metadata->Append("cyxwiz.token_windows.context", std::to_string(kContext));
    metadata->Append("cyxwiz.token_windows.vocabulary", artifact.str());
    metadata->Append("cyxwiz.token_windows.tokenizer_type", "0");
    metadata->Append("cyxwiz.token_windows.lowercase", "false");
    const auto table = arrow::Table::Make(arrow::schema(fields, metadata), arrays, rows);

    auto out = arrow::io::FileOutputStream::Open(path.string());
    if (!out.ok()) {
        error = "cannot write " + path.string() + ": " + out.status().ToString();
        return false;
    }
    const auto properties = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    const auto status = parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), *out, 1024,
                                                   parquet::default_writer_properties(), properties);
    if (!status.ok()) {
        error = "cannot write " + path.string() + ": " + status.ToString();
        return false;
    }
    return (*out)->Close().ok();
}

json ToJson(const TrainingBenchmarkResult& r) {
    return {{"ok", r.ok},
            {"error", r.error},
            {"benchmark_id", r.benchmark_id},
            {"build", r.build},
            {"measured_at", r.measured_at},
            {"backend", r.backend},
            {"device_id", r.device_id},
            {"device_name", r.device_name},
            {"physical_fingerprint", r.physical_fingerprint},
            {"batch_size", r.batch_size},
            {"tokens_per_step", r.tokens_per_step},
            {"steps_measured", r.steps_measured},
            {"step_ms_median", r.step_ms_median},
            {"step_ms_p10", r.step_ms_p10},
            {"step_ms_p90", r.step_ms_p90},
            {"tokens_per_second", r.tokens_per_second},
            {"wall_seconds", r.wall_seconds},
            {"native_cpu_fallbacks", r.native_cpu_fallbacks},
            {"first_loss", r.first_loss},
            {"last_loss", r.last_loss}};
}

TrainingBenchmarkResult FromJson(const json& j) {
    TrainingBenchmarkResult r;
    // Files written before failures were kept hold successes only.
    r.ok = j.value("ok", true);
    r.error = j.value("error", std::string{});
    r.benchmark_id = j.value("benchmark_id", std::string{});
    r.build = j.value("build", std::string{});
    r.measured_at = j.value("measured_at", std::string{});
    r.backend = j.value("backend", std::string{});
    r.device_id = j.value("device_id", 0);
    r.device_name = j.value("device_name", std::string{});
    r.physical_fingerprint = j.value("physical_fingerprint", std::string{});
    r.batch_size = j.value("batch_size", 0);
    r.tokens_per_step = j.value("tokens_per_step", 0);
    r.steps_measured = j.value("steps_measured", 0);
    r.step_ms_median = j.value("step_ms_median", 0.0);
    r.step_ms_p10 = j.value("step_ms_p10", 0.0);
    r.step_ms_p90 = j.value("step_ms_p90", 0.0);
    r.tokens_per_second = j.value("tokens_per_second", 0.0);
    r.wall_seconds = j.value("wall_seconds", 0.0);
    r.native_cpu_fallbacks = j.value("native_cpu_fallbacks", 0LL);
    r.first_loss = j.value("first_loss", 0.0f);
    r.last_loss = j.value("last_loss", 0.0f);
    return r;
}

}  // namespace

TrainingBenchmarkResult RunTrainingBenchmark(const TrainingBenchmarkOptions& options) {
    TrainingBenchmarkResult result;
    result.build = GetVersionString();
    result.measured_at = UtcNow();
    if (const auto selection = GetSavedExecutionDeviceSelection()) {
        result.physical_fingerprint = selection->physical_fingerprint;
    }

    const int warmup = std::max(0, options.warmup_steps);
    const int measured = std::max(1, options.measured_steps);
    std::error_code ec;
    const fs::path work = options.work_dir.empty() ? fs::temp_directory_path(ec) / "cyxwiz-training-benchmark"
                                                   : options.work_dir;
    fs::create_directories(work, ec);
    const fs::path data = work / "benchmark_windows.parquet";
    // One spare batch so the run never reaches its end before the last measured step.
    if (!WriteSyntheticTokenWindows(data, (warmup + measured + 1) * kBatchSize, result.error)) return result;

    GraphTrainingJobRequest request;
    request.graph_json = kTrainingBenchmarkGraphJson;
    request.dataset_files["cyxwiz_training_benchmark"] = data.string();
    request.epochs_override = 1;
    request.checkpoint_dir_override = (work / "checkpoints").string();

    std::vector<double> step_ms;
    std::vector<float> losses;
    int seen = 0;
    auto last = std::chrono::steady_clock::now();
    const auto start = last;
    GraphTrainingJobCallbacks callbacks;
    callbacks.on_start = [&](int, int batch_size) {
        result.batch_size = batch_size;
        last = std::chrono::steady_clock::now();
    };
    callbacks.on_batch = [&](int, int, int, float loss, float) {
        // Cancellation is polled between batches, so a batch may finish past the count.
        if (seen >= warmup + measured) return;
        const auto now = std::chrono::steady_clock::now();
        step_ms.push_back(std::chrono::duration<double, std::milli>(now - last).count());
        last = now;
        losses.push_back(loss);
        ++seen;
    };
    callbacks.should_cancel = [&] {
        return seen >= warmup + measured || (options.should_cancel && options.should_cancel());
    };

    const auto run = RunGraphTrainingJob(request, callbacks);
    result.wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fs::remove_all(work, ec);
    if (!run.ok && !run.cancelled) {
        result.error = "benchmark training failed: " + run.error;
        return result;
    }
    if (seen < warmup + measured) {
        result.error = options.should_cancel && options.should_cancel()
                           ? "benchmark cancelled"
                           : "benchmark ended after " + std::to_string(seen) + " steps";
        return result;
    }

    const std::vector<double> steady(step_ms.begin() + warmup, step_ms.end());
    const auto trace = TrainingTraceCollector::Instance().Snapshot();
    result.backend = trace.effective_backend;
    result.device_id = trace.effective_device_id;
    result.device_name = trace.effective_device_name;
    result.tokens_per_step = result.batch_size * kContext;
    result.steps_measured = static_cast<int>(steady.size());
    result.step_ms_median = Percentile(steady, 0.5);
    result.step_ms_p10 = Percentile(steady, 0.1);
    result.step_ms_p90 = Percentile(steady, 0.9);
    result.tokens_per_second =
        result.step_ms_median > 0.0 ? result.tokens_per_step * 1000.0 / result.step_ms_median : 0.0;
    result.native_cpu_fallbacks = static_cast<long long>(trace.native_cpu_fallback_count);
    result.first_loss = losses.front();
    result.last_loss = losses.back();
    result.ok = true;
    return result;
}

fs::path GetTrainingBenchmarkCachePath() {
    return GetComputeRuntimeRoot() / "training-benchmark.json";
}

bool SaveTrainingBenchmarkResults(const fs::path& path, const std::vector<TrainingBenchmarkResult>& results,
                                  std::string& error) {
    json routes = json::array();
    for (const auto& r : results) routes.push_back(ToJson(r));
    const json document{{"schema", 1}, {"benchmark_id", kTrainingBenchmarkId}, {"routes", routes}};
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    const fs::path temp = path.string() + ".tmp";
    {
        std::ofstream out(temp, std::ios::binary | std::ios::trunc);
        if (!out || !(out << document.dump(2))) {
            error = "cannot write " + temp.string();
            return false;
        }
    }
    fs::rename(temp, path, ec);
    if (ec) {
        error = "cannot replace " + path.string() + ": " + ec.message();
        return false;
    }
    return true;
}

bool LoadTrainingBenchmarkResults(const fs::path& path, std::vector<TrainingBenchmarkResult>& results,
                                  std::string& error) {
    results.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "no training benchmark at " + path.string();
        return false;
    }
    const json document = json::parse(in, nullptr, false);
    if (document.is_discarded() || document.value("schema", 0) != 1 || !document.contains("routes") ||
        !document["routes"].is_array()) {
        error = "unreadable training benchmark " + path.string();
        return false;
    }
    for (const auto& route : document["routes"]) {
        if (route.is_object()) results.push_back(FromJson(route));
    }
    return true;
}

}  // namespace cyxwiz
