// Node presets (TOFIX129 A7-2): built-ins, the saved store's round trip,
// replacement by name, and applying a preset to a node.
#include "../src/core/node_presets.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    using namespace cyxwiz::node_presets;

    const auto dense = BuiltinPresets(gui::NodeType::Dense);
    Check(dense.size() == 3 && dense[1].name == "Medium (256)" && dense[1].parameters.at("units") == "256",
          "Dense built-ins");
    Check(BuiltinPresets(gui::NodeType::ReLU).empty(), "no built-ins for ReLU");

    const auto file = std::filesystem::temp_directory_path() / "cyxwiz_test_node_presets" / "node_presets.json";
    std::error_code ec;
    std::filesystem::remove_all(file.parent_path(), ec);

    Store store;
    std::string error;
    Check(store.Load(file, &error) && store.Size() == 0, "a missing file is an empty store: " + error);
    store.Put("Dense", "Wide", {{"units", "2048"}, {"activation", "relu"}});
    store.Put("Dense", "Medium (256)", {{"units", "300"}});  // shadows the built-in
    store.Put("Adam", "Slow", {{"learning_rate", "0.00001"}});
    Check(store.Size() == 3, "three saved presets");
    Check(store.Save(file, &error), "save: " + error);

    Store again;
    Check(again.Load(file, &error) && again.Size() == 3, "round trip: " + error);
    const auto saved = again.PresetsFor("Dense");
    Check(saved.size() == 2 && saved[0].name == "Wide" && saved[0].parameters.at("activation") == "relu" && !saved[0].builtin,
          "saved Dense presets in order, not built-in");

    const auto all = AllPresets(gui::NodeType::Dense, "Dense", again);
    Check(all.size() == 4, "3 built-in + Wide, the shadowed one replaced: " + std::to_string(all.size()));
    Check(all[1].name == "Medium (256)" && all[1].parameters.at("units") == "300" && !all[1].builtin,
          "a saved preset with a built-in's name replaces it");
    Check(all[3].name == "Wide", "saved presets after the built-ins");

    gui::MLNode node;
    node.type = gui::NodeType::Dense;
    node.parameters["units"] = "64";
    node.parameters["_meta_x"] = "1";
    node.parameters["plot_spec"] = "{}";
    Check(Apply(all[3], node) == 2 && node.parameters["units"] == "2048" && node.parameters["activation"] == "relu",
          "apply writes the preset's keys");
    const auto to_save = ParametersToSave(node);
    Check(to_save.size() == 2 && to_save.count("units") && to_save.count("activation"),
          "saving skips _meta_ keys and view specs");

    Check(again.Remove("Dense", "Wide") && !again.Remove("Dense", "Wide"), "remove once");
    Check(again.PresetsFor("Dense").size() == 1, "one Dense preset left");

    // A broken file is reported and treated as empty.
    {
        std::ofstream broken(file);
        broken << "{ not json";
    }
    Store bad;
    Check(!bad.Load(file, &error) && bad.Size() == 0 && !error.empty(), "broken file reported");

    std::filesystem::remove_all(file.parent_path(), ec);
    std::cout << "node presets checks passed\n";
    return 0;
}
