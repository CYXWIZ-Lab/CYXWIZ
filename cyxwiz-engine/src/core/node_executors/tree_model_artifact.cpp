#include "tree_model_artifact.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace cyxwiz {

namespace {

using json = nlohmann::json;

void SetError(std::string* error, const std::string& message) {
    if (error) {
        *error = message;
    }
}

bool WriteJsonFile(const std::string& path,
                   const json& document,
                   std::string* error) {
    try {
        const std::filesystem::path out_path(path);
        if (out_path.empty()) {
            SetError(error, "model artifact path is empty");
            return false;
        }
        if (out_path.has_parent_path()) {
            std::filesystem::create_directories(out_path.parent_path());
        }
        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            SetError(error, "failed to open model artifact for writing: " + path);
            return false;
        }
        out << document.dump(2);
        return true;
    } catch (const std::exception& ex) {
        SetError(error, ex.what());
        return false;
    }
}

bool ReadJsonFile(const std::string& path, json& document, std::string* error) {
    try {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            SetError(error, "failed to open model artifact for reading: " + path);
            return false;
        }
        document = json::parse(in);
        return true;
    } catch (const std::exception& ex) {
        SetError(error, ex.what());
        return false;
    }
}

json DecisionTreeNodeToJson(const DecisionTreeNode& node) {
    return {
        {"is_leaf", node.is_leaf},
        {"feature_index", node.feature_index},
        {"threshold", node.threshold},
        {"left_child", node.left_child},
        {"right_child", node.right_child},
        {"predicted_class", node.predicted_class},
        {"impurity", node.impurity},
        {"sample_count", node.sample_count},
    };
}

DecisionTreeNode DecisionTreeNodeFromJson(const json& value) {
    DecisionTreeNode node;
    node.is_leaf = value.at("is_leaf").get<bool>();
    node.feature_index = value.at("feature_index").get<int>();
    node.threshold = value.at("threshold").get<double>();
    node.left_child = value.at("left_child").get<int>();
    node.right_child = value.at("right_child").get<int>();
    node.predicted_class = value.at("predicted_class").get<int>();
    node.impurity = value.at("impurity").get<double>();
    node.sample_count = value.at("sample_count").get<size_t>();
    return node;
}

json DecisionTreeToJson(const DecisionTreeModel& model) {
    json nodes = json::array();
    for (const auto& node : model.Nodes()) {
        nodes.push_back(DecisionTreeNodeToJson(node));
    }
    return {
        {"feature_names", model.FeatureNames()},
        {"class_labels", model.ClassLabels()},
        {"numeric_labels", model.HasNumericLabels()},
        {"nodes", nodes},
    };
}

DecisionTreeModel DecisionTreeFromJson(const json& value) {
    std::vector<DecisionTreeNode> nodes;
    for (const auto& node_json : value.at("nodes")) {
        nodes.push_back(DecisionTreeNodeFromJson(node_json));
    }

    DecisionTreeModel model;
    model.SetNodes(std::move(nodes));
    model.SetFeatureNames(value.at("feature_names").get<std::vector<std::string>>());
    model.SetClassLabels(
        value.at("class_labels").get<std::vector<std::string>>(),
        value.at("numeric_labels").get<bool>());
    return model;
}

json RegressionNodeToJson(const GradientBoostingRegressionNode& node) {
    return {
        {"is_leaf", node.is_leaf},
        {"feature_index", node.feature_index},
        {"threshold", node.threshold},
        {"left_child", node.left_child},
        {"right_child", node.right_child},
        {"value", node.value},
    };
}

GradientBoostingRegressionNode RegressionNodeFromJson(const json& value) {
    GradientBoostingRegressionNode node;
    node.is_leaf = value.at("is_leaf").get<bool>();
    node.feature_index = value.at("feature_index").get<int>();
    node.threshold = value.at("threshold").get<double>();
    node.left_child = value.at("left_child").get<int>();
    node.right_child = value.at("right_child").get<int>();
    node.value = value.at("value").get<double>();
    return node;
}

json RegressionTreeToJson(const GradientBoostingRegressionTree& tree) {
    json nodes = json::array();
    for (const auto& node : tree.Nodes()) {
        nodes.push_back(RegressionNodeToJson(node));
    }
    return {{"nodes", nodes}};
}

GradientBoostingRegressionTree RegressionTreeFromJson(const json& value) {
    std::vector<GradientBoostingRegressionNode> nodes;
    for (const auto& node_json : value.at("nodes")) {
        nodes.push_back(RegressionNodeFromJson(node_json));
    }
    GradientBoostingRegressionTree tree;
    tree.SetNodes(std::move(nodes));
    return tree;
}

json ArtifactEnvelope(const std::string& model_type, const json& model_json) {
    return {
        {"format", "cyxwiz_tree_model"},
        {"version", 1},
        {"model_type", model_type},
        {"model", model_json},
    };
}

bool ValidateEnvelope(const json& document,
                      const std::string& model_type,
                      std::string* error) {
    if (document.value("format", "") != "cyxwiz_tree_model") {
        SetError(error, "model artifact format is not cyxwiz_tree_model");
        return false;
    }
    if (document.value("model_type", "") != model_type) {
        SetError(error, "model artifact type mismatch; expected " + model_type);
        return false;
    }
    if (!document.contains("model")) {
        SetError(error, "model artifact is missing model payload");
        return false;
    }
    return true;
}

} // namespace

bool SaveDecisionTreeModelArtifact(const DecisionTreeModel& model,
                                   const std::string& path,
                                   std::string* error) {
    return WriteJsonFile(
        path, ArtifactEnvelope("DecisionTreeClassifier", DecisionTreeToJson(model)),
        error);
}

bool LoadDecisionTreeModelArtifact(const std::string& path,
                                   DecisionTreeModel& model,
                                   std::string* error) {
    try {
        json document;
        if (!ReadJsonFile(path, document, error) ||
            !ValidateEnvelope(document, "DecisionTreeClassifier", error)) {
            return false;
        }
        model = DecisionTreeFromJson(document.at("model"));
        return true;
    } catch (const std::exception& ex) {
        SetError(error, ex.what());
        return false;
    }
}

bool SaveRandomForestModelArtifact(const RandomForestModel& model,
                                   const std::string& path,
                                   std::string* error) {
    json trees = json::array();
    for (const auto& tree : model.Trees()) {
        trees.push_back({
            {"feature_indices", tree.feature_indices},
            {"model", DecisionTreeToJson(tree.model)},
        });
    }
    const json payload = {
        {"feature_names", model.FeatureNames()},
        {"class_labels", model.ClassLabels()},
        {"numeric_labels", model.HasNumericLabels()},
        {"trees", trees},
    };
    return WriteJsonFile(
        path, ArtifactEnvelope("RandomForestClassifier", payload), error);
}

bool LoadRandomForestModelArtifact(const std::string& path,
                                   RandomForestModel& model,
                                   std::string* error) {
    try {
        json document;
        if (!ReadJsonFile(path, document, error) ||
            !ValidateEnvelope(document, "RandomForestClassifier", error)) {
            return false;
        }
        const auto& payload = document.at("model");
        std::vector<RandomForestTree> trees;
        for (const auto& tree_json : payload.at("trees")) {
            RandomForestTree tree;
            tree.feature_indices =
                tree_json.at("feature_indices").get<std::vector<size_t>>();
            tree.model = DecisionTreeFromJson(tree_json.at("model"));
            trees.push_back(std::move(tree));
        }
        model.SetTrees(std::move(trees));
        model.SetFeatureNames(
            payload.at("feature_names").get<std::vector<std::string>>());
        model.SetClassLabels(
            payload.at("class_labels").get<std::vector<std::string>>(),
            payload.at("numeric_labels").get<bool>());
        return true;
    } catch (const std::exception& ex) {
        SetError(error, ex.what());
        return false;
    }
}

bool SaveGradientBoostingModelArtifact(const GradientBoostingModel& model,
                                       const std::string& path,
                                       std::string* error) {
    json rounds = json::array();
    for (const auto& round : model.Trees()) {
        json round_json = json::array();
        for (const auto& tree : round) {
            round_json.push_back(RegressionTreeToJson(tree));
        }
        rounds.push_back(std::move(round_json));
    }
    const json payload = {
        {"feature_names", model.FeatureNames()},
        {"class_labels", model.ClassLabels()},
        {"numeric_labels", model.HasNumericLabels()},
        {"initial_scores", model.InitialScores()},
        {"learning_rate", model.LearningRate()},
        {"trees", rounds},
    };
    return WriteJsonFile(
        path, ArtifactEnvelope("GradientBoostingClassifier", payload), error);
}

bool LoadGradientBoostingModelArtifact(const std::string& path,
                                       GradientBoostingModel& model,
                                       std::string* error) {
    try {
        json document;
        if (!ReadJsonFile(path, document, error) ||
            !ValidateEnvelope(document, "GradientBoostingClassifier", error)) {
            return false;
        }
        const auto& payload = document.at("model");
        std::vector<std::vector<GradientBoostingRegressionTree>> rounds;
        for (const auto& round_json : payload.at("trees")) {
            std::vector<GradientBoostingRegressionTree> round;
            for (const auto& tree_json : round_json) {
                round.push_back(RegressionTreeFromJson(tree_json));
            }
            rounds.push_back(std::move(round));
        }
        model.SetInitialScores(
            payload.at("initial_scores").get<std::vector<double>>());
        model.SetTrees(std::move(rounds));
        model.SetLearningRate(payload.at("learning_rate").get<double>());
        model.SetFeatureNames(
            payload.at("feature_names").get<std::vector<std::string>>());
        model.SetClassLabels(
            payload.at("class_labels").get<std::vector<std::string>>(),
            payload.at("numeric_labels").get<bool>());
        return true;
    } catch (const std::exception& ex) {
        SetError(error, ex.what());
        return false;
    }
}

} // namespace cyxwiz
