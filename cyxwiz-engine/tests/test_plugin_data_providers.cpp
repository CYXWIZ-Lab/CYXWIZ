// Plugin dataset formats and analytics through the real plugin manager and
// registries (TOFIX181). Uses the example plugin plugins/examples/data_formats
// (an ARFF loader and a Coefficient of variation analytic, written on the
// Arrow C stream without an Arrow library):
//   - the loader registers, matches .arff files and streams them in batches;
//   - '?' becomes null, quoted text keeps its commas, a bad row names its line;
//   - the analytic computes on the Engine's data and refuses a text column;
//   - a plugin.json declaring plugin API 1.2 gets neither registered;
//   - unloading removes both; data already read stays readable (the Engine
//     copies each batch, so it never points into a freed library);
//   - a plugin that crashes in CanLoad, OpenDataset, get_next or Compute
//     fails that call with its reason, and the Engine keeps running.
#include "../src/plugin/plugin_manager.h"
#include "../src/plugin/registries/plugin_analytics_registry.h"
#include "../src/plugin/registries/plugin_data_loader_registry.h"

#include <arrow/api.h>
#include <arrow/c/bridge.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
using cyxwiz::plugin::PluginAnalyticsRegistry;
using cyxwiz::plugin::PluginDataLoaderRegistry;

constexpr const char* kPluginId = "com.cyxwiz.examples.data-formats";
constexpr const char* kLoader = "com.cyxwiz.examples.data-formats.arff";
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

void Write(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

std::shared_ptr<arrow::Table> ReadAll(const std::string& path, std::string* error = nullptr) {
    auto reader = PluginDataLoaderRegistry::Instance().Open(kLoader, path);
    if (!reader.ok()) {
        if (error) *error = reader.status().message();
        return nullptr;
    }
    auto table = (*reader)->ToTable();
    if (!table.ok()) {
        if (error) *error = table.status().message();
        return nullptr;
    }
    return *table;
}

std::shared_ptr<arrow::RecordBatchReader> ReaderOf(const std::shared_ptr<arrow::Table>& table) {
    return std::make_shared<arrow::TableBatchReader>(table);
}

// A hardware fault, as a plugin with a bug would raise.
void Crash() {
    volatile int* nowhere = nullptr;
    *nowhere = 1;
}

int CrashingGetNext(ArrowArrayStream*, ArrowArray*) {
    Crash();
    return 0;
}

// A data plugin that crashes where it is told to (registered straight into
// the registries, as PluginManager does after QueryInterface).
class CrashingProvider final : public cyxwiz::plugin::IDataProvider, public cyxwiz::plugin::IAnalyticsProvider {
public:
    enum class Where { CanLoad, OpenDataset, GetNext };
    Where where = Where::CanLoad;

    std::vector<cyxwiz::plugin::PluginDataLoaderInfo> GetLoaders() override {
        return {{"test.crashing.loader", "Crashing", {"crashy"}, ""}};
    }
    bool CanLoad(const std::string&, const std::string&) override {
        if (where == Where::CanLoad) Crash();
        return true;
    }
    bool OpenDataset(const std::string&, const std::string&, ArrowArrayStream* out, std::string&) override {
        if (where == Where::OpenDataset) Crash();
        arrow::DoubleBuilder builder;
        (void)builder.Append(1.0);
        auto table = arrow::Table::Make(arrow::schema({arrow::field("x", arrow::float64())}),
                                        {builder.Finish().ValueOrDie()});
        if (!arrow::ExportRecordBatchReader(ReaderOf(table), out).ok()) return false;
        out->get_next = CrashingGetNext;  // a valid schema, then a crash on the first batch
        return true;
    }
    std::vector<cyxwiz::plugin::PluginAnalyticsInfo> GetAnalytics() override {
        cyxwiz::plugin::PluginAnalyticsInfo info;
        info.analytics_id = "test.crashing.analytic";
        info.name = "Crashing";
        return {info};
    }
    cyxwiz::plugin::AnalyticsResult Compute(const std::string&, ArrowArrayStream*) override {
        Crash();
        return {};
    }
};

}  // namespace

int main() {
    const auto dir = PluginDir();
    Check(!dir.empty(), "plugins/examples/data_formats not found; run from the repository root");
    const auto work = fs::temp_directory_path() / "cyxwiz_test_plugin_data_providers";
    fs::remove_all(work);
    fs::create_directories(work);

    auto& loaders = PluginDataLoaderRegistry::Instance();
    auto& analytics = PluginAnalyticsRegistry::Instance();
    auto& manager = cyxwiz::plugin::PluginManager::Instance();

    // --- Registration ---
    const auto generation = loaders.Generation();
    Check(manager.LoadPlugin(dir), "the data formats example loads from " + dir.string());
    Check(manager.InitializePlugin(kPluginId), "the data formats example initialises");
    Check(loaders.Generation() != generation, "registering a loader changes the generation");
    const auto all = loaders.Loaders();
    Check(all.size() == 1 && all[0].info.loader_id == kLoader && all[0].plugin_id == kPluginId,
          "the ARFF loader is registered once");
    Check(all[0].info.extensions.size() == 1 && all[0].info.extensions[0] == "arff",
          "its extension is stored lower case without the dot");
    Check(analytics.Analytics().size() == 1 && analytics.Analytics()[0].info.analytics_id == kAnalytic,
          "the analytic is registered once");

    // --- Matching ---
    const auto small = work / "Iris Small.ARFF";
    Write(small,
          "% a comment\n"
          "@RELATION iris\n"
          "@ATTRIBUTE 'sepal length' NUMERIC\n"
          "@ATTRIBUTE note STRING\n"
          "@ATTRIBUTE class {setosa,versicolor}\n"
          "@DATA\n"
          "5.1,'hello, world',setosa\n"
          "?,plain,versicolor\n"
          "% comment between rows\n"
          "4.9,?,setosa\n");
    Check(loaders.LoaderForFile(small.string()) == kLoader, "an upper-case .ARFF file matches the loader");
    const auto not_arff = work / "fake.arff";
    Write(not_arff, "a,b,c\n1,2,3\n");
    Check(loaders.LoaderForFile(not_arff.string()).empty(), "CanLoad refuses a .arff file without an ARFF header");
    Check(loaders.LoaderForFile((work / "x.csv").string()).empty(), "a .csv file matches no plugin loader");

    // --- Reading ---
    std::string error;
    auto table = ReadAll(small.string(), &error);
    Check(table != nullptr, "the small file reads: " + error);
    Check(table->num_rows() == 3 && table->num_columns() == 3, "3 rows and 3 columns");
    Check(table->schema()->field(0)->name() == "sepal length" &&
              table->schema()->field(0)->type()->id() == arrow::Type::DOUBLE,
          "a numeric attribute is a float64 column with its quoted name");
    Check(table->schema()->field(1)->type()->id() == arrow::Type::STRING &&
              table->schema()->field(2)->type()->id() == arrow::Type::STRING,
          "string and nominal attributes are text columns");
    Check(table->column(0)->null_count() == 1 && table->column(1)->null_count() == 1,
          "'?' is a null value");
    auto note = std::static_pointer_cast<arrow::StringArray>(table->column(1)->chunk(0));
    Check(note->GetString(0) == "hello, world", "a quoted value keeps its comma, got " + note->GetString(0));

    const auto big = work / "big.arff";
    {
        std::ofstream out(big, std::ios::binary);
        out << "@relation big\n@attribute x numeric\n@attribute y numeric\n@data\n";
        for (int i = 0; i < 120001; ++i) out << i << ',' << (i % 7) << '\n';
    }
    table = ReadAll(big.string(), &error);
    Check(table != nullptr, "the big file reads: " + error);
    Check(table->num_rows() == 120001, "every row of the big file arrives, got " + std::to_string(table->num_rows()));
    Check(table->column(0)->num_chunks() == 3, "the big file streams in 3 batches of at most 50,000 rows");

    const auto bad = work / "bad.arff";
    Write(bad, "@relation bad\n@attribute a numeric\n@attribute b numeric\n@data\n1,2\n3\n");
    table = ReadAll(bad.string(), &error);
    Check(table == nullptr && error.find("line 6") != std::string::npos,
          "a row with too few values fails and names its line, got: " + error);
    const auto no_data = work / "nodata.arff";
    Write(no_data, "@relation r\n@attribute a numeric\n");
    table = ReadAll(no_data.string(), &error);
    Check(table == nullptr && error.find("ARFF (Weka) could not open") != std::string::npos &&
              error.find("no @data") != std::string::npos,
          "a file without @data is refused with the plugin's reason, got: " + error);
    auto missing = loaders.Open("com.example.not-installed", small.string());
    Check(!missing.ok() && missing.status().message().find("not installed") != std::string::npos,
          "an unknown loader id says the format is not installed");

    // --- Analytic ---
    arrow::DoubleBuilder a_builder;
    arrow::Int64Builder b_builder;
    (void)a_builder.AppendValues({2.0, 4.0, 6.0});
    (void)b_builder.AppendValues({5, 5, 5});
    (void)b_builder.AppendNull();
    (void)a_builder.Append(8.0);
    auto numbers = arrow::Table::Make(
        arrow::schema({arrow::field("a", arrow::float64()), arrow::field("b", arrow::int64())}),
        {a_builder.Finish().ValueOrDie(), b_builder.Finish().ValueOrDie()});
    auto result = analytics.Compute(kAnalytic, ReaderOf(numbers));
    Check(result.success, "the analytic runs on numeric columns: " + result.error_message);
    const double expected_a = std::sqrt(20.0 / 3.0) / 5.0;  // values 2,4,6,8
    Check(result.vector_labels.size() == 2 && result.vector_labels[0] == "a" &&
              std::abs(result.vector_value[0] - expected_a) < 1e-5 && result.vector_value[1] == 0.0f,
          "coefficient of variation per column, nulls skipped");
    Check(result.name == "Coefficient of variation" && result.category == "Distribution",
          "the result carries the analytic's name and category");

    arrow::StringBuilder text_builder;
    (void)text_builder.Append("x");
    auto text = arrow::Table::Make(arrow::schema({arrow::field("t", arrow::utf8())}),
                                   {text_builder.Finish().ValueOrDie()});
    result = analytics.Compute(kAnalytic, ReaderOf(text));
    Check(!result.success && result.error_message.find("not numeric") != std::string::npos,
          "a text column is refused, got: " + result.error_message);

    // --- Unload ---
    manager.UnloadPlugin(kPluginId);
    Check(loaders.Loaders().empty() && analytics.Analytics().empty(), "unloading removes the loader and the analytic");
    // Data read from the plugin is the Engine's copy: it outlives the plugin's library.
    Check(note->GetString(0) == "hello, world", "a column read from the plugin is still readable after unloading it");
    note.reset();

    // --- A plugin built for API 1.2 ---
    const auto old_dir = work / "old_api";
    fs::create_directories(old_dir / "bin");
    for (const auto& entry : fs::directory_iterator(dir / "bin"))
        fs::copy_file(entry.path(), old_dir / "bin" / entry.path().filename());
    {
        std::ifstream in(dir / "plugin.json");
        std::string manifest((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        manifest.replace(manifest.find("\"1.3.0\""), 7, "\"1.2.0\"");
        Write(old_dir / "plugin.json", manifest);
    }
    Check(manager.LoadPlugin(old_dir), "the API 1.2 copy loads");
    Check(manager.InitializePlugin(kPluginId), "the API 1.2 copy initialises");
    Check(loaders.Loaders().empty() && analytics.Analytics().empty(),
          "a plugin built for API 1.2 gets no loader or analytic registered");
    const auto* loaded = manager.GetLoadedPlugin(kPluginId);
    bool reason = false;
    for (const auto& skipped : loaded->skipped_nodes)
        if (skipped.find("IDataProvider: built for plugin API 1.2.0; it needs 1.3.0") != std::string::npos) reason = true;
    Check(reason, "the skipped reason names the plugin API");
    manager.UnloadPlugin(kPluginId);

    // --- A plugin that crashes ---
    CrashingProvider crashing;
    loaders.Register("test.crashing", "1.0.0", &crashing);
    analytics.Register("test.crashing", "1.0.0", &crashing);
    const auto crashy = (work / "data.crashy").string();
    Write(crashy, "x\n1\n");

    Check(loaders.LoaderForFile(crashy).empty(), "a loader that crashes in CanLoad does not take the file");

    crashing.where = CrashingProvider::Where::OpenDataset;
    const auto failed_open = loaders.Open("test.crashing.loader", crashy);
    Check(!failed_open.ok() && failed_open.status().message().find("crashed during OpenDataset") != std::string::npos &&
              failed_open.status().message().find("the Engine kept running") != std::string::npos,
          "a crash in OpenDataset fails the open, got: " + failed_open.status().ToString());

    crashing.where = CrashingProvider::Where::GetNext;
    {
        auto opened = loaders.Open("test.crashing.loader", crashy);
        Check(opened.ok(), "the crashing stream opens (its schema is valid): " + opened.status().ToString());
        std::shared_ptr<arrow::RecordBatch> batch;
        const auto read = (*opened)->ReadNext(&batch);
        Check(!read.ok() && !batch && read.message().find("crashed during get_next") != std::string::npos,
              "a crash in get_next fails the read, got: " + read.ToString());
        Check(!(*opened)->ReadNext(&batch).ok(), "a stream that crashed stays failed");
    }  // the reader releases the stream here, in the guard

    const auto crashed = analytics.Compute("test.crashing.analytic", ReaderOf(numbers));
    Check(!crashed.success && crashed.error_message.find("crashed during Compute") != std::string::npos,
          "a crash in Compute fails the analytic, got: " + crashed.error_message);

    loaders.RemoveByPlugin("test.crashing");
    analytics.RemoveByPlugin("test.crashing");
    Check(loaders.Loaders().empty() && analytics.Analytics().empty(), "the crashing provider is removed");

    fs::remove_all(work);
    std::cout << "test_plugin_data_providers: all checks passed\n";
    return 0;
}
