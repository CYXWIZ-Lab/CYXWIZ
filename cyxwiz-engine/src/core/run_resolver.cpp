#include "run_resolver.h"

#include "arrow_dataset.h"
#include "parquet_backed_dataset.h"
#include "sequence_arrow_batcher.h"
#include "sparse_feature_dataset.h"
#include "training_batcher_setup.h"

#include <spdlog/spdlog.h>

#include <utility>
#include <vector>

namespace cyxwiz {

namespace {

// A lookup's answer. An unset lookup knows no dataset, and no dataset has an
// empty name (a role that is not supplied).
template <typename T>
std::shared_ptr<T> Find(const std::function<std::shared_ptr<T>(const std::string&)>& lookup,
                        const std::string& name) {
    if (!lookup || name.empty()) return nullptr;
    return lookup(name);
}

TrainingRecipe NotLoaded(const std::string& what, const std::string& name) {
    return FailedTrainingRecipe(what + " '" + name + "' is not loaded; Apply its Data Input first");
}

// Token / tag windows. The training set is a table in memory, or a file kept
// on disk that streams a row group at a time; a supplied Validation or Test
// role is a dataset of its own, in memory or on disk.
TrainingRecipe SequenceRecipe(const TrainingConfiguration& config, const RunDatasets& data, const std::string& name) {
    const auto train_memory = Find(data.arrow, name);
    const auto train_file = train_memory ? nullptr : Find(data.parquet, name);
    if (!train_memory && !train_file) return NotLoaded("dataset", name);

    // A supplied role's windows: its table in memory, else its file on disk.
    const auto source_of = [&data](const DatasetSourceRef& role) {
        SequenceWindowSource source;
        if (const auto memory = Find(data.arrow, role.dataset_name)) source.table = memory->GetArrowTable();
        else source.file = Find(data.parquet, role.dataset_name);
        return source;
    };
    const auto& roles = config.dataset_roles;
    const SequenceWindowSource dev_source = source_of(roles.dev);
    const SequenceWindowSource test_source = source_of(roles.test);
    if (roles.dev.IsSupplied() && dev_source.Empty()) {
        return NotLoaded("the supplied validation dataset", roles.dev.dataset_name);
    }
    if (roles.test.IsSupplied() && test_source.Empty()) {
        return NotLoaded("the supplied test dataset", roles.test.dataset_name);
    }

    // The builder for a table in memory reads every role from memory; only
    // the one that streams the training file takes each role from where it is.
    const bool on_disk = train_file != nullptr;
    const auto dev = Find(data.arrow, roles.dev.dataset_name);
    const auto test = Find(data.arrow, roles.test.dataset_name);
    if (!on_disk && ((roles.dev.IsSupplied() && !dev) || (roles.test.IsSupplied() && !test))) {
        return FailedTrainingRecipe("a supplied role is kept on disk, so the training set must be too: raise the "
                                    "data budget or keep all of them in memory");
    }
    SequenceWindowSource train_source;
    train_source.file = train_file;

    // The batches are built for each run's batch size.
    return SequenceTrainingRecipe(
        [on_disk, train_source, train_memory, dev, test, dev_source, test_source, name](
            TrainingConfiguration& run, int batch_size, std::unique_ptr<ISequenceBatcher>& batcher,
            std::vector<std::string>& id_to_label, std::string& error) {
            auto sequence = on_disk
                ? BuildSequenceBatcherFromWindowSources(train_source, run, batch_size, dev_source, test_source)
                : BuildSequenceBatcherFromArrowDataset(train_memory, run, batch_size, dev, test);
            if (!sequence.success()) {
                error = "sequence batcher materialization failed: " + sequence.error_message;
                return false;
            }
            ApplySequenceBatcherBuildResultToTrainingConfig(sequence, run);
            spdlog::info("ResolveRun: sequence training from '{}' ({} samples, {} labels, seq_len={}, "
                         "token_vocab={})", name, sequence.sample_count, sequence.id_to_label.size(),
                         sequence.sequence_length, sequence.token_vocabulary_size);
            batcher = std::move(sequence.batcher);
            id_to_label = std::move(sequence.id_to_label);
            return true;
        });
}

// A supplied Validation or Test dataset: one batcher per role, each from
// memory, from its file or from its sparse set. The roles' datasets are
// looked up when the run is made, so a search's later trials read what is
// loaded then.
TrainingRecipe SuppliedRolesRecipe(const RunDatasets& data, const std::string& name, const std::string& label) {
    return ResolvedBatchersTrainingRecipe(
        [data, name, label](TrainingConfiguration& run, int batch_size, ResolvedExternalBatchers& batchers,
                            std::string& error) {
            auto& roles = run.dataset_roles;
            if (!roles.train.IsSupplied()) {
                roles.train.dataset_name = name;
                roles.train.label_column = label;
            }
            ResolvedTabularDatasets datasets;
            const auto fill = [&data](const std::string& role, std::shared_ptr<ArrowDataset>& memory,
                                      std::shared_ptr<ParquetBackedDataset>& file,
                                      std::shared_ptr<const SparseFeatureDataset>& sparse) {
                memory = Find(data.arrow, role);
                file = Find(data.parquet, role);
                sparse = Find(data.sparse, role);
            };
            fill(roles.train.dataset_name, datasets.train_arrow, datasets.train_parquet, datasets.train_sparse);
            fill(roles.dev.dataset_name, datasets.dev_arrow, datasets.dev_parquet, datasets.dev_sparse);
            fill(roles.test.dataset_name, datasets.test_arrow, datasets.test_parquet, datasets.test_sparse);
            auto built = BuildResolvedTabularTrainingBatchers(run, datasets, batch_size);
            if (!built.ok()) {
                error = "could not build resolved partitions: " + built.error_message;
                return false;
            }
            batchers = TakeResolvedExternalBatchers(std::move(built.batchers));
            return true;
        },
        "Training Model (Resolved Roles)", "Training from resolved dataset roles");
}

// A sparse feature set (TF-IDF / Count Vectorizer output kept as CSR): its
// own batchers. Looked up again when the run is made, as the roles are.
TrainingRecipe SparseRecipe(const RunDatasets& data, const std::string& name) {
    return ResolvedBatchersTrainingRecipe(
        [sparse = data.sparse, name](TrainingConfiguration& run, int batch_size, ResolvedExternalBatchers& batchers,
                                     std::string& error) {
            auto sparse_dataset = Find(sparse, name);
            if (!sparse_dataset) {
                error = "the sparse dataset '" + name + "' is no longer loaded";
                return false;
            }
            if (!run.dataset_roles.train.IsSupplied()) {
                run.dataset_roles.train.dataset_name = name;
                run.dataset_roles.train.label_column = sparse_dataset->GetLabelName();
            }
            spdlog::info("ResolveRun: sparse CSR training: dataset={}, batch_size={}, features={}, label={}", name,
                         batch_size, sparse_dataset->GetNumFeatures(), sparse_dataset->GetLabelName());
            ResolvedTabularDatasets datasets;
            datasets.train_sparse = std::move(sparse_dataset);
            auto built = BuildResolvedTabularTrainingBatchers(run, datasets, batch_size);
            if (!built.ok()) {
                error = "could not build sparse partitions: " + built.error_message;
                return false;
            }
            batchers = TakeResolvedExternalBatchers(std::move(built.batchers));
            return true;
        },
        "Training Model (Resolved Roles)", "Training from resolved dataset roles");
}

}  // namespace

TrainingRecipe ResolveRun(const TrainingConfiguration& config, const RunDatasets& data) {
    const std::string& name = config.dataset_name;
    const std::string& label = config.dataset_roles.train.label_column;
    if (name.empty()) return FailedTrainingRecipe("the run names no training dataset");

    if (config.sequence_batch.enabled) return SequenceRecipe(config, data, name);

    // Images, audio and text come before the tables: a text dataset also keeps
    // its raw table under the same name, and an image or audio recipe reads
    // its supplied Validation / Test datasets itself.
    if (data.media) {
        if (auto media = data.media(config, name)) return std::move(*media);
    }

    const auto sparse = Find(data.sparse, name);
    const auto memory = Find(data.arrow, name);
    const auto file = Find(data.parquet, name);
    if (!sparse && !memory && !file) return NotLoaded("dataset", name);

    if (config.dataset_roles.dev.IsSupplied() || config.dataset_roles.test.IsSupplied()) {
        return SuppliedRolesRecipe(data, name, label);
    }
    if (sparse) return SparseRecipe(data, name);
    // In memory first: a table that fits is loaded there.
    if (memory) {
        spdlog::info("ResolveRun: Arrow training: dataset={}, label={}", name, label);
        return ArrowTrainingRecipe(memory, label);
    }
    spdlog::info("ResolveRun: Parquet-backed training: dataset={}, label={}, {:.1f} MB on disk", name, label,
                 static_cast<double>(file->GetFileSizeBytes()) / (1024.0 * 1024.0));
    return ParquetTrainingRecipe(file, label);
}

}  // namespace cyxwiz
