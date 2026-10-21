#pragma once

// A saved graph (.cyxgraph JSON, or the model_definition a job carries) read
// into the graph data model without the editor (TOFIX118 P2). The editor
// loads through the same code, so every host sees the same nodes, pins and
// links: nodes come from the node factory, parameters are migrated, data
// boundary pins follow the document's contract version and links resolve by
// pin index.

#include "graph_model.h"

#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

struct GraphDocument {
    std::vector<gui::MLNode> nodes;
    std::vector<gui::NodeLink> links;
    int next_node_id = 1;  // first unused ids
    int next_link_id = 1;
    int next_pin_id = 1;
};

struct GraphDocumentLoadOptions {
    // Runs after the nodes are built and before links resolve (the editor
    // restores subgraph wrapper pins here).
    std::function<void(std::vector<gui::MLNode>&, int& next_pin_id)> restore_extra_pins;
    // True: a link that does not resolve fails the load (otherwise it is
    // skipped with a warning, as the editor does for old graphs).
    bool strict_links = false;
};

// Builds nodes and links from `content` (the document's nodes/links, already
// flattened by the caller when it has subgraphs); contract versions are read
// from `document`. False with a reason when the graph cannot be loaded.
bool BuildGraphDocument(const nlohmann::json& document, const nlohmann::json& content,
                        const GraphDocumentLoadOptions& options, GraphDocument& out, std::string& error);

// Parses graph JSON text and builds it. Refuses documents with subgraph
// wrappers (they are a visual composition the runtime does not lower).
bool ParseGraphDocument(const std::string& json_text, GraphDocument& out, std::string& error);

}  // namespace cyxwiz

namespace gui {

// Saved-graph parameter migrations (also used by the editor's pattern import).
void MigrateLegacyNodeParameters(NodeType type, std::map<std::string, std::string>& params,
                                 bool prefer_legacy = false);

// True (and logged) when a node is a Dense standing in for a sequence/NER
// placeholder; such graphs must be re-authored with the real node type.
bool RejectDenseEncodedSequencePlaceholder(const MLNode& node, const std::string& source);

}  // namespace gui
