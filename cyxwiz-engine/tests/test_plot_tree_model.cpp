// A trained tree model as Tree rows (TOFIX134 P4.7): decision tree, forest
// (each tree's own feature names) and boosting (value leaves), broken input.

#include "../src/core/plot/plot_tree_model.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace cyxwiz::plot;

namespace {
void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

// Root splits on feature 1 at 61.5; left is a leaf (class 0), right splits on
// feature 0 into two leaves. Node 5 is not reached by any split.
const char* kTreeNodes = R"([
  {"is_leaf":false,"feature_index":1,"threshold":61.5,"left_child":1,"right_child":2,"predicted_class":0,"impurity":0.48,"sample_count":100},
  {"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"predicted_class":0,"impurity":0.1,"sample_count":60},
  {"is_leaf":false,"feature_index":0,"threshold":0.25,"left_child":3,"right_child":4,"predicted_class":1,"impurity":0.3,"sample_count":40},
  {"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"predicted_class":0,"impurity":0.0,"sample_count":15},
  {"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"predicted_class":1,"impurity":0.0,"sample_count":25},
  {"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"predicted_class":1,"impurity":0.0,"sample_count":1}
])";

std::string Tree(const char* features) {
    return std::string(R"({"feature_names":)") + features +
           R"(,"class_labels":["not popular","popular"],"numeric_labels":false,"nodes":)" + kTreeNodes + "}";
}
}  // namespace

int main() {
    // Decision tree.
    const std::string dt = std::string(R"({"format":"cyxwiz_tree_model","version":1,"model_type":"DecisionTreeClassifier","model":)") +
                           Tree(R"(["energy","artist_popularity"])") + "}";
    TreeModelRows r = ReadTreeModel(dt);
    Check(r.error.empty(), "decision tree reads: " + r.error);
    Check(r.trees == 1 && r.node.size() == 5, "5 reached nodes, node 5 left out");
    Check(r.node[0] == "0: artist_popularity <= 61.5" && r.parent[0].empty(), "root text, no parent: " + r.node[0]);
    Check(r.node[1] == "leaf 1: not popular" && r.parent[1] == r.node[0], "leaf under the root: " + r.node[1]);
    Check(r.node[2] == "2: energy <= 0.25" && r.cls[2] == "popular", "right split and its class");
    Check(r.samples[0] == 100 && r.samples[4] == 25 && std::abs(r.value[0] - 0.48) < 1e-9, "samples and impurity");
    Check(r.rule[1] == "leaf" && r.rule[2] == "energy <= 0.25", "rule column");
    Check(TreeModelColumns().size() == 9 && TreeModelColumns()[2] == "node", "column names");

    // Forest: node texts carry the tree number; features are each tree's own.
    const std::string rf = std::string(R"({"format":"cyxwiz_tree_model","version":1,"model_type":"RandomForestClassifier","model":{)") +
                           R"("feature_names":["energy","artist_popularity","tempo"],"class_labels":["not popular","popular"],"numeric_labels":false,"trees":[)" +
                           R"({"feature_indices":[0,1],"model":)" + Tree(R"(["energy","artist_popularity"])") + "}," +
                           R"({"feature_indices":[2,1],"model":)" + Tree(R"(["tempo","artist_popularity"])") + "}]}}";
    r = ReadTreeModel(rf);
    Check(r.error.empty(), "forest reads: " + r.error);
    Check(r.trees == 2 && r.node.size() == 10, "two trees of 5 nodes");
    Check(r.node[5] == "2.0: artist_popularity <= 61.5" && r.tree[5] == 2 && r.tree_name[5] == "Tree 2", "second tree's root");
    Check(r.node[7] == "2.2: tempo <= 0.25", "second tree uses its own feature names: " + r.node[7]);
    Check(r.parent[6] == "2.0: artist_popularity <= 61.5", "parents stay in their tree");

    // Boosting: trees[round][class], value leaves, no samples.
    const char* reg = R"({"nodes":[
      {"is_leaf":false,"feature_index":0,"threshold":3,"left_child":1,"right_child":2,"value":0},
      {"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"value":-0.2},
      {"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"value":0.35}]})";
    const std::string gb = std::string(R"({"format":"cyxwiz_tree_model","version":1,"model_type":"GradientBoostingClassifier","model":{)") +
                           R"("feature_names":["x"],"class_labels":["a","b"],"numeric_labels":false,"initial_scores":[0,0],"learning_rate":0.1,"trees":[[)" +
                           reg + "," + reg + "]]}}";
    r = ReadTreeModel(gb);
    Check(r.error.empty(), "boosting reads: " + r.error);
    Check(r.trees == 2 && r.node.size() == 6, "one round, two classes");
    Check(r.node[0] == "1.0: x <= 3" && std::isnan(r.samples[0]) && std::isnan(r.value[0]), "split: no samples, no value");
    Check(r.node[2] == "leaf 1.2: +0.35" && std::abs(r.value[2] - 0.35) < 1e-9, "leaf value: " + r.node[2]);
    Check(r.tree_name[3] == "Round 1 \xC2\xB7 b" && r.cls[3] == "b", "second class tree: " + r.tree_name[3]);

    // Large split values read with thousands separators, not 2.391e+05.
    const std::string big = std::string(R"({"format":"cyxwiz_tree_model","version":1,"model_type":"DecisionTreeClassifier","model":{)") +
                            R"("feature_names":["followers"],"class_labels":["a","b"],"numeric_labels":false,"nodes":[)" +
                            R"({"is_leaf":false,"feature_index":0,"threshold":239066.4,"left_child":1,"right_child":2,"predicted_class":0,"impurity":0.5,"sample_count":4},)" +
                            R"({"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"predicted_class":0,"impurity":0,"sample_count":2},)" +
                            R"({"is_leaf":true,"feature_index":-1,"threshold":0,"left_child":-1,"right_child":-1,"predicted_class":1,"impurity":0,"sample_count":2}]}})";
    r = ReadTreeModel(big);
    Check(r.error.empty() && r.node[0] == "0: followers <= 239,066", "thousands separators: " + r.node[0]);

    // Not a tree model, broken JSON, unknown type.
    Check(!ReadTreeModel(R"({"format":"cyxwiz_regression"})").error.empty(), "other artifact refused");
    Check(!ReadTreeModel("{not json").error.empty(), "broken JSON refused");
    Check(!ReadTreeModel(R"({"format":"cyxwiz_tree_model","model_type":"Svm","model":{}})").error.empty(), "unknown type refused");
    Check(ReadTreeModel(R"({"format":"cyxwiz_tree_model","model_type":"DecisionTreeClassifier","model":{"nodes":"x"}})").node.empty(),
          "broken nodes give no rows");

    std::cout << "test_plot_tree_model: all checks passed\n";
    return 0;
}
