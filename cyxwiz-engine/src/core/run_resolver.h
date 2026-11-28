#pragma once

// Which data a training run reads and through which batchers, decided in one
// place for every host. The Engine's Train (and its searches) and a headless
// graph training job each say where their datasets are (RunDatasets) and get
// the same recipe (training_recipe.h) for the same configuration, so a job on
// a server node trains the way Train does on the same graph and data.
//
// The order of the decision:
//   1. a sequence graph (token / tag windows): the sequence batcher, from the
//      table in memory or a row group at a time from the file on disk;
//   2. a dataset the host owns as images, audio or text (`media`): the host's
//      recipe for it, which reads its supplied roles itself;
//   3. a supplied Validation or Test dataset: one batcher per role;
//   4. a sparse feature set: its sparse batchers;
//   5. a table in memory; 6. a Parquet file kept on disk;
//   7. otherwise a failed recipe naming the dataset.

#include "training_recipe.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace cyxwiz {

class SparseFeatureDataset;

// Where a host keeps its datasets, by name. A lookup returns null for a name
// it does not know; a lookup left empty knows none. The recipes that find
// their datasets when a run is made keep a copy of the lookups, so whatever a
// lookup captures must outlive the recipe.
struct RunDatasets {
    std::function<std::shared_ptr<ArrowDataset>(const std::string& name)> arrow;
    std::function<std::shared_ptr<ParquetBackedDataset>(const std::string& name)> parquet;
    std::function<std::shared_ptr<const SparseFeatureDataset>(const std::string& name)> sparse;
    // A dataset the host owns as images, audio or text: the host's recipe for
    // it (a failed one when it cannot train as it is), or nullopt when the
    // name is not such a dataset. Empty in hosts without media (the headless
    // job).
    std::function<std::optional<TrainingRecipe>(const TrainingConfiguration& config, const std::string& name)> media;
};

// The recipe for `config`: the training dataset is `config.dataset_name` and
// its label `config.dataset_roles.train.label_column` (both set by launch
// preparation, in the Engine and in the job). A recipe without `make` carries
// the reason in `error`.
TrainingRecipe ResolveRun(const TrainingConfiguration& config, const RunDatasets& data);

}  // namespace cyxwiz
