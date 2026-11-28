#pragma once

// Training recipes (training_recipe.h) for the datasets only the Engine
// hosts: image folders and audio clips, read by their own batchers. They are
// built into the Engine, not the training core, because their batchers decode
// files with libraries a headless job does not carry. The Engine hands them
// to the run resolver through its media lookup (run_resolver.h).

#include "data_registry.h"
#include "training_recipe.h"

namespace cyxwiz {

// An image folder: decoded on the training thread at the graph's Resize size.
TrainingRecipe ImageTrainingRecipe(DataRegistry::ImageDatasetEntry entry);
// Audio clips: features from the graph's Spectrogram / Mel / MFCC nodes.
TrainingRecipe AudioTrainingRecipe(DataRegistry::AudioDatasetEntry entry);

}  // namespace cyxwiz
