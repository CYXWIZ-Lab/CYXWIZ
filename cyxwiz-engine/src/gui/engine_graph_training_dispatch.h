#pragma once

// The Engine's graph-training dispatch: what Train runs once the launcher has
// prepared the graph. Sequence graphs build the sequence batcher and train
// through TrainingManager::StartTrainingSequence; other graphs go to the
// loader that owns the runtime dataset, else the legacy DatasetHandle path.
// Shared by MainWindow and the Engine/node parity test (TOFIX118 P2 slice 4).

#include "graph_training_launcher.h"

namespace cyxwiz {
class DataRegistry;
class TrainingManager;
}  // namespace cyxwiz

namespace gui {

GraphTrainingDispatch MakeEngineGraphTrainingDispatch(cyxwiz::DataRegistry& registry, cyxwiz::TrainingManager& tm);

}  // namespace gui
