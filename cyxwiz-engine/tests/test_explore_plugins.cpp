// Explore > Plugins (TOFIX181 step 4): the presentation model, the column
// stream from memory and from disk, and the example plugin's analytic run on
// Weka's weather data the way the tab runs it.
//   - the model: the analytic's label, which columns it reads and why not,
//     the default columns, when Run can start, how an answer reads, Copy / CSV;
//   - OpenDatasetColumns streams only the chosen columns, every row, from a
//     table in memory and row group by row group from disk; it cancels, and
//     names a missing column;
//   - Coefficient of variation from plugins/examples/data_formats gives
//     temperature 0.0893 and humidity 0.1260 (sample std / |mean|), on either.
#include "../src/core/data_registry.h"
#include "../src/core/dataset_column_stream.h"
#include "../src/core/explore_plugins_presentation.h"
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_analytics_registry.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace ep = cyxwiz::explore_plugins;
using cyxwiz::DataRegistry;

constexpr const char* kPluginId = "com.cyxwiz.examples.data-formats";
constexpr const char* kAnalytic = "com.cyxwiz.examples.data-formats.variation";

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

fs::path PluginDir() {
    auto dir = fs::current_path();
    while (!dir.empty()) {
        const auto candidate = dir / "plugins" / "examples" / "data_formats";
        if (fs::exists(candidate / "plugin.json")) return candidate;
        const auto parent = dir.parent_path();
        if (parent == dir) break;
        dir = parent;
    }
    return {};
}

ep::Analytic Variation() {
    ep::Analytic a;
    a.id = kAnalytic;
    a.name = "Coefficient of variation";
    a.category = "Distribution";
    a.plugin_name = "Example Data Formats";
    a.plugin_version = "1.0.0";
    a.kind = ep::ColumnKind::Numeric;
    return a;
}

std::vector<ep::Column> WeatherColumns() {
    return {{"outlook", false, true}, {"temperature", true, false}, {"humidity", true, false},
            {"windy", false, true}, {"play", false, true}};
}

void CheckModel() {
    const auto a = Variation();
    const auto columns = WeatherColumns();
    Check(ep::Label(a) == "Coefficient of variation (Distribution)", "label: " + ep::Label(a));
    Check(ep::FromLine(a) == "from Example Data Formats 1.0.0", "from line: " + ep::FromLine(a));

    const auto refused = ep::RefusedReasons(a, columns);
    Check(refused.size() == 5 && refused[0] == "text · this analytic reads numbers" && refused[1].empty() &&
              refused[2].empty(),
          "text columns are refused with the reason, numbers are not: " + refused[0]);
    Check(ep::DefaultColumns(a, columns) == std::vector<std::string>{"temperature", "humidity"},
          "the default is every number column in table order");
    Check(ep::ColumnsHint(a, columns) == "numbers only: 2 of 5 columns", "hint: " + ep::ColumnsHint(a, columns));
    auto limited = a;
    limited.max_columns = 1;
    Check(ep::DefaultColumns(limited, columns) == std::vector<std::string>{"temperature"}, "the default stops at the maximum");

    std::vector<std::string> picked = {"humidity", "outlook", "gone"};
    ep::KeepReadable(a, columns, picked);
    Check(picked == std::vector<std::string>{"humidity"}, "picks the analytic cannot read, or the table lacks, are dropped");

    auto run = ep::CanRun(a, {}, 14, true);
    Check(!run.enabled && run.reason == "Pick at least 1 column (this analytic reads 1 or more).", "no columns: " + run.reason);
    run = ep::CanRun(a, {"temperature", "humidity"}, 14, true);
    Check(run.enabled && run.summary == "reads 14 rows of 2 columns", "ready: " + run.summary);
    run = ep::CanRun(a, {"temperature"}, 1234567, true);
    Check(run.summary == "reads 1,234,567 rows of 1 column", "grouped rows: " + run.summary);
    run = ep::CanRun(a, {"temperature"}, 14, false);
    Check(!run.enabled && run.reason == "Example Data Formats is not loaded, so this analytic cannot run.",
          "plugin gone: " + run.reason);
    run = ep::CanRun(limited, {"temperature", "humidity"}, 14, true);
    Check(!run.enabled && run.reason == "Pick at most 1 column.", "too many: " + run.reason);

    Check(ep::Key("weather", kAnalytic, {"temperature"}) != ep::Key("weather", kAnalytic, {"humidity"}),
          "the key follows the columns");

    Check(ep::NumberText(0.0893) == "0.0893" && ep::NumberText(0.126) == "0.1260" &&
              ep::NumberText(1234.5) == "1234.5" && ep::NumberText(12345678.0) == "1.235e+07" &&
              ep::NumberText(0.0) == "0.0000",
          "numbers: " + ep::NumberText(12345678.0));

    ep::Result r;
    r.ok = true;
    r.name = "Coefficient of variation";
    r.labels = {"temperature", "humidity"};
    r.values = {0.0893, 0.1260};
    r.text = "0 means every value is the same.";
    r.plugin_name = "Example Data Formats";
    r.plugin_version = "1.0.0";
    r.rows = 14;
    r.milliseconds = 3.2;
    r.finished_at = "11:24:05";
    const auto v = ep::Present(r);
    Check(v.headline.empty(), "no single number, no headline");
    Check(v.bars.size() == 2 && v.bars[1].fraction == 1.0f && std::fabs(v.bars[0].fraction - 0.0893f / 0.1260f) < 1e-5f &&
              v.bars[0].value == "0.0893" && v.bars[1].value == "0.1260",
          "bars scale to the largest value and print it");
    Check(v.by_line == "Computed by Example Data Formats 1.0.0 on 14 rows in 3 ms, 11:24:05", "by line: " + v.by_line);
    Check(v.value_header == "coefficient of variation", "value column header");
    Check(ep::CopyText(r) == "column\tcoefficient of variation\ntemperature\t0.0893\nhumidity\t0.1260\n",
          "copy text: " + ep::CopyText(r));
    Check(ep::Csv(r).rfind("column,coefficient of variation\ntemperature,0.089", 0) == 0, "csv: " + ep::Csv(r));
    r.has_scalar = true;
    r.scalar = 2.5;
    Check(ep::Present(r).headline == "2.5000", "a single number is the headline");
}

std::shared_ptr<arrow::Table> Weather() {
    arrow::StringBuilder outlook, windy, play;
    arrow::DoubleBuilder temperature, humidity;
    const char* o[] = {"sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast",
                       "sunny", "sunny", "rainy", "sunny", "overcast", "overcast", "rainy"};
    const double t[] = {85, 80, 83, 70, 68, 65, 64, 72, 69, 75, 75, 72, 81, 71};
    const double h[] = {85, 90, 86, 96, 80, 70, 65, 95, 70, 80, 70, 90, 75, 91};
    const bool w[] = {false, true, false, false, false, true, true, false, false, false, true, true, false, true};
    const bool p[] = {false, false, true, true, true, false, true, false, true, true, true, true, true, false};
    for (int i = 0; i < 14; ++i) {
        (void)outlook.Append(o[i]);
        (void)temperature.Append(t[i]);
        (void)humidity.Append(h[i]);
        (void)windy.Append(w[i] ? "TRUE" : "FALSE");
        (void)play.Append(p[i] ? "yes" : "no");
    }
    std::shared_ptr<arrow::Array> a, b, c, d, e;
    (void)outlook.Finish(&a);
    (void)temperature.Finish(&b);
    (void)humidity.Finish(&c);
    (void)windy.Finish(&d);
    (void)play.Finish(&e);
    auto schema = arrow::schema({arrow::field("outlook", arrow::utf8()), arrow::field("temperature", arrow::float64()),
                                 arrow::field("humidity", arrow::float64()), arrow::field("windy", arrow::utf8()),
                                 arrow::field("play", arrow::utf8())});
    return arrow::Table::Make(schema, {a, b, c, d, e});
}

struct Read {
    int batches = 0;
    int64_t rows = 0;
    int columns = 0;
    std::string error;
};

Read Drain(const std::string& dataset, const std::vector<std::string>& columns, cyxwiz::ColumnStreamOptions options = {}) {
    Read out;
    auto reader = cyxwiz::OpenDatasetColumns(dataset, columns, options, out.error);
    if (!reader) return out;
    out.columns = reader->schema()->num_fields();
    while (true) {
        std::shared_ptr<arrow::RecordBatch> batch;
        const auto status = reader->ReadNext(&batch);
        if (!status.ok()) {
            out.error = status.message();
            break;
        }
        if (!batch) break;
        ++out.batches;
        out.rows += batch->num_rows();
    }
    return out;
}

void CheckVariation(const std::string& dataset, const char* where) {
    std::string error;
    auto reader = cyxwiz::OpenDatasetColumns(dataset, {"temperature", "humidity"}, {}, error);
    Check(reader != nullptr, std::string(where) + ": the stream opens: " + error);
    const auto r = cyxwiz::plugin::PluginAnalyticsRegistry::Instance().Compute(kAnalytic, std::move(reader));
    Check(r.success && r.vector_labels == std::vector<std::string>{"temperature", "humidity"} && r.vector_value.size() == 2,
          std::string(where) + ": the analytic answers per column: " + r.error_message);
    Check(std::fabs(r.vector_value[0] - 0.0893) < 1e-4 && std::fabs(r.vector_value[1] - 0.1260) < 1e-4,
          std::string(where) + ": temperature 0.0893 and humidity 0.1260, got " + std::to_string(r.vector_value[0]) +
              " and " + std::to_string(r.vector_value[1]));
}

}  // namespace

int main() {
    CheckModel();

    const auto dir = PluginDir();
    Check(!dir.empty(), "plugins/examples/data_formats not found; run from the repository root");
    const auto work = fs::temp_directory_path() / "cyxwiz_test_explore_plugins";
    fs::remove_all(work);
    fs::create_directories(work / "cache");
    const auto parquet_path = (work / "weather.parquet").string();
    {
        auto out = arrow::io::FileOutputStream::Open(parquet_path);
        Check(out.ok(), "the weather fixture opens for writing");
        Check(parquet::arrow::WriteTable(*Weather(), arrow::default_memory_pool(), *out, 5).ok(),
              "the weather fixture is written in row groups of 5");
        (void)(*out)->Close();
    }
    auto& registry = DataRegistry::Instance();
    const std::string cache = (work / "cache").string();
    Check(registry.LoadTabularParquet(parquet_path, "weather_memory", false, {}, nullptr, cache) ==
              DataRegistry::TabularLoadBackend::InMemory,
          "the table loads in memory");
    Check(registry.LoadTabularParquet(parquet_path, "weather_disk", true, {}, nullptr, cache) ==
              DataRegistry::TabularLoadBackend::DiskBacked,
          "the table stays on disk");

    // --- The stream ---
    auto read = Drain("weather_memory", {"temperature", "humidity"});
    Check(read.error.empty() && read.rows == 14 && read.columns == 2, "memory: 14 rows of the 2 chosen columns: " + read.error);
    read = Drain("weather_disk", {"humidity"});
    Check(read.error.empty() && read.rows == 14 && read.columns == 1 && read.batches == 3,
          "disk: 14 rows of 1 column, one batch per row group (3), got " + std::to_string(read.batches));
    int64_t seen = 0;
    cyxwiz::ColumnStreamOptions counting;
    counting.on_rows = [&](int64_t n) { seen = n; };
    (void)Drain("weather_disk", {"temperature"}, counting);
    Check(seen == 14, "the row count reaches every row");
    int calls = 0;
    cyxwiz::ColumnStreamOptions cancel;
    cancel.should_cancel = [&] { return ++calls > 1; };
    read = Drain("weather_disk", {"temperature"}, cancel);
    Check(read.error == "Cancelled" && read.batches == 1, "cancel stops after the batch in hand: " + read.error);
    read = Drain("weather_memory", {"pressure"});
    Check(read.error == "The table has no column 'pressure'", "a missing column is named: " + read.error);
    read = Drain("nothing", {"temperature"});
    Check(read.error.find("not a table held in memory or on disk") != std::string::npos, "an unknown table: " + read.error);

    // --- The example plugin's analytic, as the tab runs it ---
    auto& manager = cyxwiz::plugin::PluginManager::Instance();
    Check(manager.LoadPlugin(dir) && manager.InitializePlugin(kPluginId), "the data formats example loads and starts");
    const auto analytics = cyxwiz::plugin::PluginAnalyticsRegistry::Instance().Analytics();
    Check(analytics.size() == 1 && analytics[0].plugin_version == "1.0.0", "the registry names the plugin's version");
    CheckVariation("weather_memory", "memory");
    CheckVariation("weather_disk", "disk");
    manager.UnloadPlugin(kPluginId);

    registry.UnregisterTabularDataset("weather_memory");
    registry.UnregisterTabularDataset("weather_disk");
    std::error_code ignored;
    fs::remove_all(work, ignored);
    std::cout << "test_explore_plugins: all checks passed\n";
    return 0;
}
