// The Properties panel's view model (TOFIX129 A7) on the nodes of the A7 test
// graph: Dense (128), the Data Loader with dialog-owned truths, the Data Input
// (dialog-only) and the MSE loss, plus the empty state and the cache key.
#include "../src/core/properties_presentation.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

gui::MLNode MakeNode(int id, gui::NodeType type, std::string name) {
    gui::MLNode node;
    node.id = id;
    node.type = type;
    node.name = std::move(name);
    return node;
}

cyxwiz::ParameterDefinition Param(const char* name, const char* type, const char* def, const char* display,
                                  const char* group, bool required = false, bool advanced = false,
                                  const char* validation = "") {
    return cyxwiz::ParameterDefinition(name, type, def, "", {}, validation, display, group, required, advanced);
}

const cyxwiz::properties_view::Setting* FindRow(const cyxwiz::properties_view::View& v, const std::string& key) {
    for (const auto& g : v.groups) {
        for (const auto& r : g.rows) {
            if (r.key == key) return &r;
        }
    }
    return nullptr;
}

}  // namespace

int main() {
    using namespace cyxwiz;
    using namespace cyxwiz::properties_view;

    // ---- Dense (128): one row, its truth attached, nothing written ---------
    {
        NodeMetadata dense_meta(gui::NodeType::Dense, gui::NodeCategory::Layers, "Dense", "L");
        dense_meta.parameters = {Param("units", "int", "64", "Output Units", "Layer", true, false, "1-1048576")};
        auto dense = MakeNode(4, gui::NodeType::Dense, "Dense (128)");
        dense.parameters["units"] = "128";
        dense.inputs.push_back({});
        dense.outputs.push_back({});
        std::vector<gui::MLNode> nodes{dense};
        std::vector<gui::NodeLink> links{{1, 3, 0, 4, 0}, {2, 4, 0, 5, 0}};

        Inputs in;
        in.node = &nodes[0];
        in.metadata = &dense_meta;
        in.nodes = &nodes;
        in.links = &links;
        gui::properties_truth::PropertyTruth units;
        units.label = "Output units";
        units.canonical_key = "units";
        units.source_key = "units";
        units.effective_value = "128";
        units.owner = gui::properties_truth::TruthOwner::Compiler;
        units.quick_editable = true;
        units.statuses = {gui::properties_truth::TruthStatus::OK};
        units.message = "GraphCompiler and ModelBuilder use units as the linear output width.";
        in.truth.properties.push_back(units);
        gui::properties_truth::RawParameterTruth raw;
        raw.key = "units";
        raw.value = "128";
        raw.maps_to = "Output units";
        in.truth.raw_parameters.push_back(raw);

        const View v = Build(in);
        Check(v.header.name == "Dense (128)" && v.header.id == 4, "header: name and id");
        Check(v.header.type_name == "Dense" && v.header.category == "ML Layers", "header: type and category");
        Check(v.header.implementation == "Implemented" && v.header.implementation_kind == ChipKind::Ok,
              "header: implementation");
        Check(v.header.primary_label.empty(), "Dense has no dialog: no primary button");
        Check(v.route == SettingsRoute::Metadata && v.groups.size() == 1 && v.groups[0].name == "Layer",
              "one Layer group");
        const auto* row = FindRow(v, "units");
        Check(row && row->value == "128" && row->stored && row->required && row->differs_from_default,
              "units row: value 128, stored, required, differs from the default 64");
        Check(row->label == "Output Units", "label from display_name");
        Check(row->truth && row->truth->chips.size() == 1 && row->truth->chips[0].kind == ChipKind::Ok,
              "the truth is attached to the row with an OK chip");
        Check(row->truth->provenance == "Source: units \xC2\xB7 Owner: Compiler", "provenance line");
        Check(v.loose_truths.empty(), "no loose truth: the row consumed it");
        Check(v.links_in == 1 && v.links_out == 1, "connections counted from the links");
        Check(v.raw.size() == 1 && v.raw[0].maps_to == "Output units", "raw rows: the truth's mapping");
        Check(!v.has_compiled && !v.fusion.applies && !v.data, "no compile facts, no fusion, no data");
        Check(nodes[0].parameters.size() == 1, "building the view did not write into the node");
    }

    // ---- Data Loader: groups, advanced, short labels, a default row --------
    {
        NodeMetadata meta(gui::NodeType::DataLoader, gui::NodeCategory::DataPipeline, "Data Loader", "D");
        meta.parameters = {
            Param("epochs", "int", "10", "Epochs", "Training", false, false, "1-10000"),
            Param("batch_size", "int", "32", "Batch Size", "Training"),
            Param("generation_preview_every_epochs", "int", "20", "Generation Preview Every Epochs",
                  "Generation previews"),
            Param("model_seed", "int", "-1", "Model Seed", "Training", false, true),
            Param("balance_classes", "bool", "false", "Balance Classes", "Balancing", false, true),
        };
        auto loader = MakeNode(3, gui::NodeType::DataLoader, "Loader");
        loader.parameters["batch_size"] = "64";
        std::vector<gui::MLNode> nodes{loader};
        std::vector<gui::NodeLink> links;

        Inputs in;
        in.node = &nodes[0];
        in.metadata = &meta;
        in.nodes = &nodes;
        in.links = &links;
        in.has_dialog = true;
        gui::properties_truth::PropertyTruth seed;
        seed.label = "Model RNG seed";
        seed.canonical_key = "model_seed";
        seed.source_key = "model_seed";
        seed.effective_value = "-1";
        seed.owner = gui::properties_truth::TruthOwner::Runtime;
        seed.statuses = {gui::properties_truth::TruthStatus::OK, gui::properties_truth::TruthStatus::RequiresDialog};
        in.truth.properties.push_back(seed);
        gui::properties_truth::PropertyTruth label;
        label.label = "Label column";
        label.canonical_key = "label_column";
        label.effective_value = "track_popularity";
        label.owner = gui::properties_truth::TruthOwner::Loader;
        label.statuses = {gui::properties_truth::TruthStatus::OK};
        in.truth.properties.push_back(label);

        const View v = Build(in);
        Check(v.header.primary_label == "Open Dialog...", "a node with a dialog gets the primary button");
        Check(v.groups.size() == 3, "Training, Generation previews, Advanced settings");
        Check(v.groups[0].name == "Training" && !v.groups[0].advanced, "named groups first");
        Check(v.groups[2].name == "Advanced settings" && v.groups[2].advanced && v.groups[2].rows.size() == 2,
              "advanced parameters in one group at the end");
        const auto* epochs = FindRow(v, "epochs");
        Check(epochs && epochs->value == "10" && !epochs->stored && !epochs->differs_from_default,
              "a missing parameter shows its default and is not stored");
        Check(nodes[0].parameters.size() == 1, "displaying defaults wrote nothing into the node");
        const auto* batch = FindRow(v, "batch_size");
        Check(batch && batch->differs_from_default, "64 differs from the default 32");
        const auto* every = FindRow(v, "generation_preview_every_epochs");
        Check(every && every->label == "Every Epochs", "the group's words are dropped from the label: " + every->label);
        const auto* model_seed = FindRow(v, "model_seed");
        Check(model_seed && model_seed->truth && model_seed->truth->chips.size() == 2 &&
                  model_seed->truth->chips[1].kind == ChipKind::Dialog && model_seed->truth->chips[1].text == "dialog",
              "the dialog-owned truth sits on its row as a dialog chip");
        Check(v.loose_truths.size() == 1 && v.loose_truths[0]->label == "Label column",
              "a truth without a row is listed loose");
    }

    // ---- Data Input: dialog-only, data fact ---------------------------------
    {
        NodeMetadata meta(gui::NodeType::DataInput, gui::NodeCategory::DataSources, "Data Input", "I");
        meta.properties_editor = NodePropertiesEditor::Dialog;
        auto input = MakeNode(1, gui::NodeType::DataInput, "Spotify");
        input.parameters["file_path"] = "D:/demo/mrcj/datasets/spotify_data clean.csv";
        input.parameters["dataset_name"] = "spotify";
        std::vector<gui::MLNode> nodes{input};
        std::vector<gui::NodeLink> links;
        gui::properties_truth::DatasetTruthFact fact;
        fact.dataset_name = "spotify";

        Inputs in;
        in.node = &nodes[0];
        in.metadata = &meta;
        in.nodes = &nodes;
        in.links = &links;
        in.has_dialog = true;
        in.dialog_only = true;
        in.dataset = &fact;

        View v = Build(in);
        Check(v.route == SettingsRoute::DialogOnly && v.groups.empty(), "dialog-only: no rows");
        Check(v.data && v.data->label == "spotify_data clean.csv" && !v.data->loaded, "data row: file, not loaded");
        Check(v.data->note.rfind("Not loaded", 0) == 0, "not-loaded note: " + v.data->note);

        fact.found = true;
        fact.backing_store = "Arrow";
        fact.rows = 32833;
        fact.columns = std::vector<std::string>(23, "c");
        v = Build(in);
        Check(v.data && v.data->loaded && v.data->detail == "32,833 rows \xC3\x97 23 columns",
              "data row when loaded: " + v.data->detail);
        Check(v.settings_note == "Everything else about this node is set in its dialog.", "dialog-only note");
    }

    // ---- MSE: custom editor route keeps truths available by key ------------
    {
        NodeMetadata meta(gui::NodeType::MSELoss, gui::NodeCategory::Training, "MSE Loss", "M");
        meta.properties_editor = NodePropertiesEditor::Custom;
        auto mse = MakeNode(8, gui::NodeType::MSELoss, "MSE");
        mse.parameters["reduction"] = "mean";
        std::vector<gui::MLNode> nodes{mse};
        std::vector<gui::NodeLink> links;
        Inputs in;
        in.node = &nodes[0];
        in.metadata = &meta;
        in.nodes = &nodes;
        in.links = &links;
        in.custom_editor = true;
        gui::properties_truth::PropertyTruth red;
        red.label = "Reduction";
        red.canonical_key = "reduction";
        red.source_key = "reduction";
        red.effective_value = "mean";
        red.statuses = {gui::properties_truth::TruthStatus::OK};
        in.truth.properties.push_back(red);
        const View v = Build(in);
        Check(v.route == SettingsRoute::Custom && v.groups.empty(), "custom route: no metadata rows");
        Check(FindTruth(v.truths, "reduction") != nullptr, "the custom editor finds the truth by key");
        Check(v.loose_truths.size() == 1, "until a row claims it, the truth counts as loose");
        Check(v.raw.size() == 1 && v.raw[0].key == "reduction" && v.raw[0].maps_to.empty(),
              "a stored parameter without a truth mapping is still listed raw");
    }

    // ---- Short labels ---------------------------------------------------------
    Check(ShortLabel("Generation Preview Max New Tokens", "Generation previews") == "Max New Tokens",
          "short label drops the group words");
    Check(ShortLabel("Epochs", "Training") == "Epochs", "a label with nothing in common stays");
    Check(ShortLabel("Training", "Training") == "Training", "never strip the whole label");
    Check(ShortLabel("Units", "") == "Units", "no group, no change");

    // ---- Empty state ----------------------------------------------------------
    {
        TrainingConfiguration config;
        ValidationIssue issue;
        issue.level = IssueLevel::Error;
        issue.node_id = 8;
        issue.message = "Required input pin 'Targets' on node 'MSE' has no incoming connection";
        config.issues.push_back(issue);
        std::vector<gui::MLNode> nodes{MakeNode(8, gui::NodeType::MSELoss, "MSE")};
        const EmptyView e = BuildEmpty("", 10, 9, LiveCompileState::Failed, &config, &nodes);
        Check(e.graph_name == "Untitled graph" && e.nodes == 10 && e.links == 9, "empty: name and counts");
        Check(e.status == "Compile failed" && e.kind == CompiledStatusKind::Failed, "empty: status");
        Check(e.first_error == "1 error on MSE", "empty: first error: " + e.first_error);
        const EmptyView named = BuildEmpty("D:/x/cyxgraph/a7_props.cyxgraph", 0, 0, LiveCompileState::NotCompiled, nullptr, nullptr);
        Check(named.graph_name == "a7_props" && named.status.empty(), "empty: file stem, no status for an empty graph");
    }

    // ---- Cache key --------------------------------------------------------------
    {
        Key a{4, 10, 2, 0, 1};
        Key b = a;
        Check(a == b, "equal keys");
        b.edit_serial = 1;
        Check(!(a == b), "an edit changes the key");
    }

    std::cout << "properties presentation checks passed\n";
    return 0;
}
