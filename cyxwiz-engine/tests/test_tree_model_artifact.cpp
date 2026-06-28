#include "../src/core/node_executors/decision_tree_trainer.h"
#include "../src/core/node_executors/gradient_boosting_trainer.h"
#include "../src/core/node_executors/random_forest_trainer.h"
#include "../src/core/node_executors/tree_model_artifact.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::vector<std::vector<double>> Features() {
    return {
        {0.00, 8.00},
        {0.10, 7.00},
        {0.20, 6.00},
        {0.30, 5.00},
        {0.70, 4.00},
        {0.80, 3.00},
        {0.90, 2.00},
        {1.00, 1.00},
    };
}

std::vector<int> Labels() {
    return {0, 0, 0, 0, 1, 1, 1, 1};
}

std::vector<std::string> FeatureNames() {
    return {"x", "z"};
}

std::vector<std::string> ClassLabels() {
    return {"0", "1"};
}

void TestDecisionTreeRoundTrip(const std::filesystem::path& root) {
    cyxwiz::DecisionTreeTrainingOptions options;
    options.max_depth = 2;
    cyxwiz::DecisionTreeTrainer trainer(options);
    const auto features = Features();
    auto model = trainer.Fit(
        features, Labels(), 2, FeatureNames(), ClassLabels(), true);

    const auto path = root / "decision_tree.cyx-tree.json";
    std::string error;
    Check(cyxwiz::SaveDecisionTreeModelArtifact(
              model, path.string(), &error),
          error);
    Check(cyxwiz::ReadTreeModelArtifactType(path.string(), &error) ==
              "DecisionTreeClassifier",
          "DecisionTree artifact type should be readable");

    cyxwiz::DecisionTreeModel loaded;
    Check(cyxwiz::LoadDecisionTreeModelArtifact(
              path.string(), loaded, &error),
          error);
    Check(loaded.PredictClasses(features) == model.PredictClasses(features),
          "DecisionTree predictions should survive artifact round-trip");
    Check(loaded.FeatureNames() == FeatureNames(),
          "DecisionTree feature names should survive artifact round-trip");
    Check(loaded.HasNumericLabels(), "DecisionTree numeric-label flag persists");
}

void TestRandomForestRoundTrip(const std::filesystem::path& root) {
    cyxwiz::RandomForestTrainingOptions options;
    options.n_estimators = 9;
    options.max_depth = 2;
    options.max_features = "all";
    options.seed = 7;
    cyxwiz::RandomForestTrainer trainer(options);
    const auto features = Features();
    auto model = trainer.Fit(
        features, Labels(), 2, FeatureNames(), ClassLabels(), true);

    const auto path = root / "random_forest.cyx-tree.json";
    std::string error;
    Check(cyxwiz::SaveRandomForestModelArtifact(
              model, path.string(), &error),
          error);

    cyxwiz::RandomForestModel loaded;
    Check(cyxwiz::LoadRandomForestModelArtifact(
              path.string(), loaded, &error),
          error);
    Check(loaded.PredictClasses(features) == model.PredictClasses(features),
          "RandomForest predictions should survive artifact round-trip");
    Check(loaded.Trees().size() == model.Trees().size(),
          "RandomForest tree count should survive artifact round-trip");
}

void TestGradientBoostingRoundTrip(const std::filesystem::path& root) {
    cyxwiz::GradientBoostingTrainingOptions options;
    options.n_estimators = 30;
    options.learning_rate = 0.4;
    options.max_depth = 2;
    cyxwiz::GradientBoostingTrainer trainer(options);
    const auto features = Features();
    auto model = trainer.Fit(
        features, Labels(), 2, FeatureNames(), ClassLabels(), true);

    const auto path = root / "gradient_boosting.cyx-tree.json";
    std::string error;
    Check(cyxwiz::SaveGradientBoostingModelArtifact(
              model, path.string(), &error),
          error);

    cyxwiz::GradientBoostingModel loaded;
    Check(cyxwiz::LoadGradientBoostingModelArtifact(
              path.string(), loaded, &error),
          error);
    Check(loaded.PredictClasses(features) == model.PredictClasses(features),
          "GradientBoosting predictions should survive artifact round-trip");
    Check(loaded.Trees().size() == model.Trees().size(),
          "GradientBoosting round count should survive artifact round-trip");
    Check(loaded.LearningRate() == model.LearningRate(),
          "GradientBoosting learning rate should survive artifact round-trip");
}

} // namespace

int main() {
    const auto root =
        std::filesystem::temp_directory_path() / "cyxwiz_tree_model_artifacts";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    TestDecisionTreeRoundTrip(root);
    TestRandomForestRoundTrip(root);
    TestGradientBoostingRoundTrip(root);

    std::filesystem::remove_all(root);
    std::cout << "Tree model artifact round-trip passed\n";
    return 0;
}
