#pragma once

// The Engine's graph-training dispatch: what Train runs once the launcher has
// prepared the graph. GraphTrainingRecipe says how the prepared data trains:
// sequence graphs build the sequence batcher; other graphs ask the loader
// that owns the runtime dataset, else the legacy DatasetHandle path. Train
// starts that recipe through TrainingManager; a search builds its trials from
// the same recipe. Shared by MainWindow and the Engine/node parity test
// (TOFIX118 P2 slice 4).

#include "graph_training_launcher.h"

#include <string>

namespace cyxwiz {
class DataRegistry;
class TrainingManager;
struct TrainingConfiguration;
struct TrainingRecipe;
}  // namespace cyxwiz

namespace gui {

cyxwiz::TrainingRecipe GraphTrainingRecipe(cyxwiz::DataRegistry& registry, const cyxwiz::TrainingConfiguration& config,
                                           const std::string& dataset_name, const std::string& label_column);

GraphTrainingDispatch MakeEngineGraphTrainingDispatch(cyxwiz::DataRegistry& registry, cyxwiz::TrainingManager& tm);

}  // namespace gui
