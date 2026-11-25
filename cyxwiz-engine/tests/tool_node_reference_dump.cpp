// Writes what the generated Node reference of the docs is made from, as JSON:
// the node contract inventory (every node's pins, settings, status and
// support, the same document test_node_contract_gates writes) with, on each
// node, the facts the inventory leaves out: the palette placement, the words
// shown for each choice, the keywords, how a node that is not in the palette
// is made, and the Node Documentation tooltip text (usage, setting notes and
// tips).
//
//   tool_node_reference_dump --out <file.json>
//
// scripts/docs/generate_node_reference.py reads the file (or runs this tool)
// and writes docs/reference/nodes.

#include "node_contract_inventory.h"
#include "node_workflow_evidence.h"

#include "../src/core/node_metadata_registry.h"
#include "../src/core/node_palette.h"
#include "../src/gui/node_documentation.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

#ifndef CYXWIZ_NODE_REFERENCE_REPOSITORY_ROOT
#error "CYXWIZ_NODE_REFERENCE_REPOSITORY_ROOT must identify the source checkout"
#endif

namespace {

using Json = nlohmann::ordered_json;

const char* PlaceName(cyxwiz::palette::Place place) {
    switch (place) {
        case cyxwiz::palette::Place::Category: return "category";
        case cyxwiz::palette::Place::Advanced: return "advanced";
        case cyxwiz::palette::Place::CantRun: return "cant_run";
        case cyxwiz::palette::Place::Hidden: return "hidden";
    }
    return "unknown";
}

const char* EditorName(cyxwiz::NodePropertiesEditor editor) {
    switch (editor) {
        case cyxwiz::NodePropertiesEditor::Automatic: return "automatic";
        case cyxwiz::NodePropertiesEditor::Dialog: return "dialog";
        case cyxwiz::NodePropertiesEditor::Custom: return "custom";
    }
    return "unknown";
}

Json ReferenceJson(const cyxwiz::NodeMetadata& metadata) {
    const auto placement = cyxwiz::palette::PlaceNode(metadata);
    Json choice_labels = Json::object();
    for (const auto& parameter : metadata.parameters) {
        if (!parameter.enum_labels.empty()) {
            choice_labels[parameter.name] = parameter.enum_labels;
        }
    }
    Json legacy = nullptr;
    if (const auto* doc =
            gui::NodeDocumentationManager::Instance().GetDocumentation(metadata.type)) {
        Json parameters = Json::array();
        for (const auto& [name, description] : doc->parameters) {
            parameters.push_back({{"name", name}, {"description", description}});
        }
        legacy = {{"title", doc->title},
                  {"description", doc->description},
                  {"usage", doc->usage},
                  {"parameters", parameters},
                  {"tips", doc->tips},
                  {"category", doc->category}};
    }
    return {{"category_order", static_cast<int>(metadata.category)},
            {"keywords", metadata.keywords},
            {"made_by", metadata.made_by},
            {"properties_editor", EditorName(metadata.properties_editor)},
            {"palette", {{"place", PlaceName(placement.place)},
                         {"group", placement.advanced_group},
                         {"why", placement.why}}},
            {"choice_labels", choice_labels},
            {"node_documentation", legacy}};
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path output;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--out" && index + 1 < argc) {
            output = std::filesystem::absolute(argv[++index]);
        } else {
            std::cerr << "usage: tool_node_reference_dump --out <file.json>\n";
            return 2;
        }
    }
    if (output.empty()) {
        std::cerr << "usage: tool_node_reference_dump --out <file.json>\n";
        return 2;
    }

    // The registry reads the resource templates relative to the Engine
    // folder, as the contract gate does.
    const std::filesystem::path repository_root =
        CYXWIZ_NODE_REFERENCE_REPOSITORY_ROOT;
    std::error_code error_code;
    std::filesystem::current_path(repository_root / "cyxwiz-engine", error_code);
    if (error_code) {
        std::cerr << "cannot enter the Engine folder: " << error_code.message() << '\n';
        return 1;
    }

    auto& registry = cyxwiz::NodeMetadataRegistry::Instance();
    registry.Initialize();
    const auto all_metadata = registry.GetAllMetadata();
    const auto workflow_evidence = cyxwiz::test::LoadNodeWorkflowEvidenceCatalog(
        repository_root,
        repository_root / "cyxwiz-engine" / "tests" / "workflow_evidence_manifest.json",
        all_metadata);
    for (const auto& error : workflow_evidence.errors) {
        std::cerr << "workflow evidence: " << error << '\n';
    }
    if (!workflow_evidence.errors.empty()) return 1;

    cyxwiz::test::NodeContractInventorySummary summary;
    Json document = Json::parse(cyxwiz::test::BuildNodeContractInventoryJson(
        all_metadata, workflow_evidence, summary));
    auto& rows = document["nodes"];
    std::size_t row = 0;
    for (const auto* metadata : all_metadata) {
        if (metadata == nullptr) continue;
        if (row >= rows.size() || rows[row]["name"] != metadata->name) {
            std::cerr << "inventory rows do not follow the registry order\n";
            return 1;
        }
        rows[row++]["reference"] = ReferenceJson(*metadata);
    }

    std::string error;
    if (!cyxwiz::test::WriteNodeContractInventoryJson(output, document.dump(2) + '\n', error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::cout << "Node reference dump: " << output.string() << " (" << row << " nodes)\n";
    return 0;
}
