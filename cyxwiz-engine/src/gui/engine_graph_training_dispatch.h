#pragma once

// The Engine's graph-training dispatch: what Train runs once the launcher has
// prepared the graph. GraphTrainingRecipe says how the prepared data trains:
// the run resolver every host shares (core/run_resolver.h) over the Engine's
// data registry, which also holds the image, audio and text datasets a
// headless job has none of. The training dataset and its label are the
// prepared configuration's (dataset_name, dataset_roles.train). Train starts
// that recipe through TrainingManager; a search builds its trials from the
// same recipe. Shared by MainWindow and the Engine/node parity test
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

cyxwiz::TrainingRecipe GraphTrainingRecipe(cyxwiz::DataRegistry& registry, const cyxwiz::TrainingConfiguration& config);

GraphTrainingDispatch MakeEngineGraphTrainingDispatch(cyxwiz::DataRegistry& registry, cyxwiz::TrainingManager& tm);

}  // namespace gui
