// One run resolver for every host (Engine 2.0 core, step 2.3a): ResolveRun
// picks how a prepared configuration trains - sequence, a host's media
// dataset, supplied roles, a table in memory, a file on disk - and the Engine
// and the headless job get the same answer for the same configuration,
// whatever their dataset stores look like. Linked against cyxwiz-training-core
// only (what a Server Node links). The runs are built, not trained.
#include "oracle/oracle_test.h"
#include "../src/core/arrow_dataset.h"
#include "../src/core/parquet_backed_dataset.h"
#include "../src/core/run_resolver.h"
#include "../src/core/sparse_feature_dataset.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace fs = std::filesystem;

namespace {

int g_failures = 0;

void Check(bool condition, const std::string& what) {
    std::cout << (condition ? "  ok   " : "  FAIL ") << what << "\n";
    if (!condition) ++g_failures;
}

bool Contains(const std::string& text, const std::string& part) {
    return text.find(part) != std::string::npos;
}

// Two features and a two-class label, eight rows.
std::shared_ptr<arrow::Table> MakeTable() {
    arrow::FloatBuilder x0, x1;
    arrow::Int32Builder label;
    for (int row = 0; row < 8; ++row) {
        const bool ok = x0.Append(static_cast<float>(row)).ok() &&
                        x1.Append(static_cast<float>(10 * (row % 2))).ok() && label.Append(row % 2).ok();
        if (!ok) return nullptr;
    }
    std::shared_ptr<arrow::Array> x0_values, x1_values, label_values;
    if (!x0.Finish(&x0_values).ok() || !x1.Finish(&x1_values).ok() || !label.Finish(&label_values).ok()) {
        return nullptr;
    }
    return arrow::Table::Make(arrow::schema({arrow::field("x0", arrow::float32()),
                                             arrow::field("x1", arrow::float32()),
                                             arrow::field("label", arrow::int32())}),
                              {x0_values, x1_values, label_values}, 8);
}

bool WriteParquet(const arrow::Table& table, const fs::path& path) {
    auto out = arrow::io::FileOutputStream::Open(path.string());
    return out.ok() && parquet::arrow::WriteTable(table, arrow::default_memory_pool(), *out, 4).ok() &&
           (*out)->Close().ok();
}

// What launch preparation hands the resolver: the training dataset's name
// and label in the configuration.
cyxwiz::TrainingConfiguration MakeConfig(const std::string& dataset) {
    cyxwiz::TrainingConfiguration config;
    config.is_valid = true;
    config.dataset_name = dataset;
    config.input_size = 2;
    config.input_shape = {2};
    config.output_size = 2;
    config.loss_type = gui::NodeType::CrossEntropyLoss;
    config.optimizer_type = gui::NodeType::SGD;
    config.train_ratio = 0.75f;
    config.val_ratio = 0.0f;
    config.test_ratio = 0.25f;
    config.has_data_split = true;
    config.shuffle = false;
    config.num_workers = 0;
    config.prefetch_factor = 0;
    config.batch_size = 2;
    config.epochs = 1;
    config.target.primary_column = "label";
    config.dataset_roles.train = {dataset, "label", 1, true};
    config.dataset_roles.policy.train_ratio = 0.75f;
    config.dataset_roles.policy.dev_ratio = 0.0f;
    config.dataset_roles.policy.test_ratio = 0.25f;
    cyxwiz::CompiledLayer dense;
    dense.type = gui::NodeType::Dense;
    dense.units = 2;
    config.layers.push_back(dense);
    return config;
}

// The Engine's kind of store: one object that answers every lookup (as its
// data registry does), with a sparse lookup and a media lookup that know
// nothing here.
struct RegistryLike {
    std::map<std::string, std::shared_ptr<cyxwiz::ArrowDataset>> memory;
    std::map<std::string, std::shared_ptr<cyxwiz::ParquetBackedDataset>> disk;
    std::map<std::string, std::shared_ptr<const cyxwiz::SparseFeatureDataset>> sparse;

    cyxwiz::RunDatasets Datasets() const {
        cyxwiz::RunDatasets data;
        data.arrow = [this](const std::string& name) -> std::shared_ptr<cyxwiz::ArrowDataset> {
            const auto it = memory.find(name);
            return it == memory.end() ? nullptr : it->second;
        };
        data.parquet = [this](const std::string& name) -> std::shared_ptr<cyxwiz::ParquetBackedDataset> {
            const auto it = disk.find(name);
            return it == disk.end() ? nullptr : it->second;
        };
        data.sparse = [this](const std::string& name) -> std::shared_ptr<const cyxwiz::SparseFeatureDataset> {
            const auto it = sparse.find(name);
            return it == sparse.end() ? nullptr : it->second;
        };
        data.media = [](const cyxwiz::TrainingConfiguration&,
                        const std::string&) -> std::optional<cyxwiz::TrainingRecipe> { return std::nullopt; };
        return data;
    }
};

// The headless job's kind of store: two maps of its own, no sparse sets and
// no media (those lookups stay empty).
cyxwiz::RunDatasets JobLike(const std::map<std::string, std::shared_ptr<cyxwiz::ArrowDataset>>& datasets,
                            const std::map<std::string, std::shared_ptr<cyxwiz::ParquetBackedDataset>>& disk) {
    cyxwiz::RunDatasets data;
    data.arrow = [&datasets](const std::string& name) -> std::shared_ptr<cyxwiz::ArrowDataset> {
        const auto it = datasets.find(name);
        return it == datasets.end() ? nullptr : it->second;
    };
    data.parquet = [&disk](const std::string& name) -> std::shared_ptr<cyxwiz::ParquetBackedDataset> {
        const auto it = disk.find(name);
        return it == disk.end() ? nullptr : it->second;
    };
    return data;
}

// The recipe builds a run for a copy of `config`; the reason when it cannot.
// `input_size`, when given, gets the run's model input width as the recipe
// set it from the data.
bool Makes(const cyxwiz::TrainingRecipe& recipe, const cyxwiz::TrainingConfiguration& config, std::string& error,
           size_t* input_size = nullptr) {
    if (!recipe) {
        error = recipe.error;
        return false;
    }
    error.clear();
    const auto executor = recipe.make(config, config.batch_size, error);
    if (executor && input_size) *input_size = executor->GetConfig().input_size;
    return executor != nullptr;
}

}  // namespace

int main() {
    if (!cyxwiz::test::BeginOracleTest("test_run_resolver")) return 1;
    const fs::path work = fs::temp_directory_path() / "cyxwiz_run_resolver_test";
    std::error_code ec;
    fs::remove_all(work, ec);
    fs::create_directories(work);

    const auto table = MakeTable();
    if (!table || !WriteParquet(*table, work / "on_disk.parquet")) {
        std::cerr << "could not make the fixture tables\n";
        return EXIT_FAILURE;
    }
    {
        RegistryLike registry;
        registry.memory["train"] = std::make_shared<cyxwiz::ArrowDataset>(table, "train");
        registry.memory["validation"] = std::make_shared<cyxwiz::ArrowDataset>(table, "validation");
        registry.disk["on_disk"] = cyxwiz::ParquetBackedDataset::Open((work / "on_disk.parquet").string(), "on_disk");
        Check(registry.disk["on_disk"] != nullptr, "the Parquet fixture opens");
        // The same tables as a job holds them.
        const std::map<std::string, std::shared_ptr<cyxwiz::ArrowDataset>> job_memory = registry.memory;
        const std::map<std::string, std::shared_ptr<cyxwiz::ParquetBackedDataset>> job_disk = registry.disk;
        const cyxwiz::RunDatasets engine = registry.Datasets();
        const cyxwiz::RunDatasets job = JobLike(job_memory, job_disk);
        std::string error;

        std::cout << "a table in memory\n";
        {
            const auto config = MakeConfig("train");
            const auto recipe = cyxwiz::ResolveRun(config, engine);
            Check(static_cast<bool>(recipe), "resolved " + recipe.error);
            Check(recipe.task_name == "Training Model (Arrow)", "it is the Arrow recipe: " + recipe.task_name);
            // The run reads the label the configuration names: two inputs
            // (x0, x1), not three with the label counted as a feature.
            size_t inputs = 0;
            Check(Makes(recipe, config, error, &inputs), "the recipe builds a run " + error);
            Check(inputs == 2, "the label column is kept out of the inputs: " + std::to_string(inputs));
            Check(cyxwiz::ResolveRun(config, job).task_name == recipe.task_name, "the job resolves it the same");
        }

        std::cout << "a Parquet file kept on disk\n";
        {
            const auto config = MakeConfig("on_disk");
            const auto recipe = cyxwiz::ResolveRun(config, engine);
            Check(static_cast<bool>(recipe), "resolved " + recipe.error);
            Check(recipe.task_name == "Training Model (Parquet)", "it is the Parquet recipe: " + recipe.task_name);
            size_t inputs = 0;
            Check(Makes(recipe, config, error, &inputs), "the recipe builds a run " + error);
            Check(inputs == 2, "the label column is kept out of the inputs: " + std::to_string(inputs));
            Check(cyxwiz::ResolveRun(config, job).task_name == recipe.task_name, "the job resolves it the same");
        }

        std::cout << "a supplied validation dataset\n";
        {
            auto config = MakeConfig("train");
            config.dataset_roles.dev = {"validation", "label", 2, true};
            const auto recipe = cyxwiz::ResolveRun(config, engine);
            Check(static_cast<bool>(recipe), "resolved " + recipe.error);
            Check(recipe.task_name == "Training Model (Resolved Roles)",
                  "it is the resolved-roles recipe: " + recipe.task_name);
            Check(Makes(recipe, config, error), "the recipe builds a run " + error);
            const auto from_job = cyxwiz::ResolveRun(config, job);
            Check(from_job.task_name == recipe.task_name, "the job resolves it the same: " + from_job.task_name);
            Check(Makes(from_job, config, error), "the job's recipe builds a run " + error);

            // A role that is not loaded stops the run with its name, not a
            // run that trains without it.
            config.dataset_roles.dev.dataset_name = "never_loaded";
            const auto missing = cyxwiz::ResolveRun(config, engine);
            Check(!Makes(missing, config, error) && Contains(error, "never_loaded"),
                  "a supplied role that is not loaded is refused by name: " + error);
        }

        std::cout << "a sparse feature set\n";
        {
            // Four rows, four features in CSR, a two-class label.
            cyxwiz::SparseFeatureDataset::Contents contents;
            contents.name = "tfidf";
            contents.num_rows = 4;
            contents.num_features = 4;
            contents.row_offsets = {0, 1, 2, 3, 4};
            contents.column_indices = {0, 1, 2, 3};
            contents.values = {1.0f, 1.0f, 1.0f, 1.0f};
            contents.feature_names = {"a", "b", "c", "d"};
            arrow::Int32Builder codes;
            std::shared_ptr<arrow::Array> code_values;
            const bool built = codes.AppendValues({0, 1, 0, 1}).ok() && codes.Finish(&code_values).ok();
            Check(built, "the sparse labels build");
            contents.labels = std::make_shared<arrow::ChunkedArray>(code_values);
            contents.label_name = "y";
            auto sparse = cyxwiz::SparseFeatureDataset::Create(std::move(contents));
            Check(sparse.ok(), "the sparse fixture is valid");
            if (sparse.ok()) registry.sparse["tfidf"] = *sparse;
            const cyxwiz::RunDatasets with_sparse = registry.Datasets();
            const auto config = MakeConfig("tfidf");
            const auto recipe = cyxwiz::ResolveRun(config, with_sparse);
            Check(static_cast<bool>(recipe) && recipe.task_name == "Training Model (Resolved Roles)",
                  "a sparse set resolves to its own batchers: " + recipe.task_name + recipe.error);
            Check(Makes(recipe, config, error), "the recipe builds a run " + error);
            // The job holds no sparse sets: it refuses the name.
            const auto headless = cyxwiz::ResolveRun(config, job);
            Check(!headless && Contains(headless.error, "tfidf"),
                  "a host without sparse sets refuses it: " + headless.error);
        }

        std::cout << "a host's media dataset\n";
        {
            int asked = 0;
            cyxwiz::RunDatasets with_media = registry.Datasets();
            with_media.media = [&asked](const cyxwiz::TrainingConfiguration&,
                                        const std::string& name) -> std::optional<cyxwiz::TrainingRecipe> {
                ++asked;
                if (name != "pictures" && name != "validation") return std::nullopt;
                return cyxwiz::MakeTrainingRecipe(
                    "Training Model (Media)", "Training from a media dataset",
                    [](cyxwiz::TrainingConfiguration, int,
                       std::string& why) -> std::unique_ptr<cyxwiz::TrainingExecutor> {
                        why = "the test builds no media run";
                        return nullptr;
                    });
            };
            const auto pictures = cyxwiz::ResolveRun(MakeConfig("pictures"), with_media);
            Check(static_cast<bool>(pictures) && pictures.task_name == "Training Model (Media)",
                  "the host's recipe is used for a name that is no table: " + pictures.task_name + pictures.error);
            // A media dataset may keep a table under its own name (raw text
            // does): the host's answer comes first.
            const auto both = cyxwiz::ResolveRun(MakeConfig("validation"), with_media);
            Check(both.task_name == "Training Model (Media)", "media comes before the tables: " + both.task_name);
            // A name the host does not own as media falls through to the tables.
            asked = 0;
            const auto plain = cyxwiz::ResolveRun(MakeConfig("train"), with_media);
            Check(asked == 1 && plain.task_name == "Training Model (Arrow)",
                  "a media lookup that knows nothing falls through: " + plain.task_name);
            // A host without media refuses a name only another host would know.
            const auto headless = cyxwiz::ResolveRun(MakeConfig("pictures"), job);
            Check(!headless && Contains(headless.error, "pictures"),
                  "a host without media refuses the dataset by name: " + headless.error);
        }

        std::cout << "a dataset no lookup knows\n";
        {
            const auto config = MakeConfig("nowhere");
            const auto recipe = cyxwiz::ResolveRun(config, engine);
            Check(!recipe && Contains(recipe.error, "nowhere"), "refused by name: " + recipe.error);
            Check(cyxwiz::ResolveRun(config, job).error == recipe.error, "the job gives the same reason");
            // With a supplied role too: still the training dataset's refusal.
            auto with_role = config;
            with_role.dataset_roles.dev = {"validation", "label", 2, true};
            const auto refused = cyxwiz::ResolveRun(with_role, engine);
            Check(!refused && Contains(refused.error, "nowhere"),
                  "refused by name with a supplied role: " + refused.error);
            const auto unnamed = cyxwiz::ResolveRun(MakeConfig(""), engine);
            Check(!unnamed && !unnamed.error.empty(), "a run that names no dataset is refused: " + unnamed.error);
            // Lookups left empty know no dataset.
            const auto no_store = cyxwiz::ResolveRun(MakeConfig("train"), cyxwiz::RunDatasets{});
            Check(!no_store && Contains(no_store.error, "train"), "empty lookups know nothing: " + no_store.error);
        }

        std::cout << "a sequence graph\n";
        {
            auto config = MakeConfig("train");
            config.sequence_batch.enabled = true;
            const auto in_memory = cyxwiz::ResolveRun(config, engine);
            Check(static_cast<bool>(in_memory) && in_memory.task_name == "Training Model (Sequence)",
                  "a table in memory resolves to the sequence recipe: " + in_memory.task_name + in_memory.error);
            Check(cyxwiz::ResolveRun(config, job).task_name == in_memory.task_name, "the job resolves it the same");

            // A role on disk beside a training set in memory cannot be read
            // by the in-memory builder: refused, by both hosts alike.
            config.dataset_roles.dev = {"on_disk", "label", 2, true};
            const auto mixed = cyxwiz::ResolveRun(config, engine);
            Check(!mixed && Contains(mixed.error, "kept on disk"),
                  "a role on disk needs the training set there: " + mixed.error);
            Check(cyxwiz::ResolveRun(config, job).error == mixed.error, "the job gives the same reason");

            // The training set on disk takes each role from where it is.
            auto on_disk = MakeConfig("on_disk");
            on_disk.sequence_batch.enabled = true;
            on_disk.dataset_roles.dev = {"validation", "label", 2, true};
            const auto streamed = cyxwiz::ResolveRun(on_disk, engine);
            Check(static_cast<bool>(streamed) && streamed.task_name == "Training Model (Sequence)",
                  "a training file on disk resolves with a role in memory: " + streamed.error);

            on_disk.dataset_roles.dev.dataset_name = "never_loaded";
            const auto missing = cyxwiz::ResolveRun(on_disk, engine);
            Check(!missing && Contains(missing.error, "never_loaded"),
                  "a sequence role that is not loaded is refused by name: " + missing.error);

            const auto unknown = cyxwiz::ResolveRun([] {
                auto nowhere = MakeConfig("nowhere");
                nowhere.sequence_batch.enabled = true;
                return nowhere;
            }(), engine);
            Check(!unknown && Contains(unknown.error, "nowhere"), "an unknown sequence dataset is refused by name");
        }
    }

    fs::remove_all(work, ec);
    if (g_failures > 0) {
        std::cout << g_failures << " check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "run resolver test passed\n";
    return EXIT_SUCCESS;
}
