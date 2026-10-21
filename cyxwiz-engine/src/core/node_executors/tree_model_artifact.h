#pragma once

#include "decision_tree_model.h"
#include "gradient_boosting_model.h"
#include "random_forest_model.h"

#include <string>

namespace cyxwiz {

std::string ReadTreeModelArtifactType(const std::string& path,
                                      std::string* error = nullptr);

bool SaveDecisionTreeModelArtifact(const DecisionTreeModel& model,
                                   const std::string& path,
                                   std::string* error = nullptr);
bool LoadDecisionTreeModelArtifact(const std::string& path,
                                   DecisionTreeModel& model,
                                   std::string* error = nullptr);

bool SaveRandomForestModelArtifact(const RandomForestModel& model,
                                   const std::string& path,
                                   std::string* error = nullptr);
bool LoadRandomForestModelArtifact(const std::string& path,
                                   RandomForestModel& model,
                                   std::string* error = nullptr);

bool SaveGradientBoostingModelArtifact(const GradientBoostingModel& model,
                                       const std::string& path,
                                       std::string* error = nullptr);
bool LoadGradientBoostingModelArtifact(const std::string& path,
                                       GradientBoostingModel& model,
                                       std::string* error = nullptr);

} // namespace cyxwiz
