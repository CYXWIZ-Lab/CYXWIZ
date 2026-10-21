#pragma once

// A trained tree model as Tree rows (TOFIX134 P4.7, board 17; owner: the
// model reaches the Plot through the trainer's Model pin). Reads the
// cyxwiz_tree_model JSON that Decision Tree, Random Forest and Gradient
// Boosting save, and gives one row per tree node: the Tree plot draws it
// with node / parent, sized by samples and coloured by class. Pure: the
// caller reads the file.

#include <string>
#include <vector>

namespace cyxwiz::plot {

struct TreeModelRows {
    std::string model_type;   // "DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"
    int trees = 0;            // trees in the model (boosting: rounds x classes)
    // Columns, one entry per node.
    std::vector<int> tree;                // 1-based
    std::vector<std::string> tree_name;   // "Tree 3", "Round 2 · popular"
    std::vector<std::string> node;        // "0: artist_popularity <= 61.5", "leaf 3: popular" (forests: "2.0: ...")
    std::vector<std::string> parent;      // the parent's node text; empty for a root
    std::vector<std::string> rule;        // "artist_popularity <= 61.5" or "leaf"
    std::vector<double> samples;          // training rows at the node (NaN for boosting)
    std::vector<std::string> cls;         // the predicted class (boosting: the tree's class)
    std::vector<double> value;            // impurity (classifier trees) or the leaf value (boosting; NaN on splits)
    std::string error;                    // not a tree model, or a broken one
};

// The column names of the table made from TreeModelRows, in order (the
// last, "trees", repeats the number of trees on every row).
const std::vector<std::string>& TreeModelColumns();

TreeModelRows ReadTreeModel(const std::string& json_text);

}  // namespace cyxwiz::plot
