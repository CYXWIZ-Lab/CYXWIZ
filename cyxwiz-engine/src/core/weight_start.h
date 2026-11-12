#pragma once

// Starting a built model from a published weight file: the model's layers as
// the matcher wants them, and the load itself. Used by the training run and
// by the Data Loader dialog's report.

#include "executable_model.h"
#include "model_builder.h"
#include "weight_match.h"

#include <filesystem>
#include <string>
#include <vector>

namespace cyxwiz {

// Every module of the model that holds parameters, with its node's name.
std::vector<weight_match::LayerSlot> WeightSlots(IExecutableModel& model,
                                                 const std::vector<BuiltModuleProvenance>& provenance);

// Reads the file, matches it to the model and sets the weights of the layers
// that matched. False with the reason when the file cannot be read or no
// layer matched; the model is then untouched. `report` says what happened to
// every layer either way.
bool StartFromWeightsFile(const std::filesystem::path& path, IExecutableModel& model,
                          const std::vector<BuiltModuleProvenance>& provenance, weight_match::Report& report,
                          std::string& error);

}  // namespace cyxwiz
