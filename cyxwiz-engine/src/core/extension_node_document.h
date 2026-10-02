#pragma once

// The "extension" block of a saved extension node (TOFIX125). It holds the
// node's identity and its pins, so a graph opens without loss when the
// extension is not installed: links are saved by pin index, and the saved
// pin list lets a placeholder have the same pins in the same order.
//
//   "extension": {
//     "contract": 1,
//     "type_id": "com.me.acts:Snake",
//     "version": "1.0.0",
//     "content_hash": "sha256:...",
//     "inputs":  [ { "name": "Input",  "type": "Tensor" } ],
//     "outputs": [ { "name": "Output", "type": "Tensor" } ]
//   }

#include "graph_model.h"

#include <nlohmann/json_fwd.hpp>

#include <string>

namespace cyxwiz {

constexpr int kExtensionBlockContract = 1;

const char* PinTypeToText(gui::PinType type);
bool PinTypeFromText(const std::string& text, gui::PinType& type);

// The block for an extension node; null for any other node.
nlohmann::json WriteExtensionBlock(const gui::MLNode& node);

// Builds the node a saved NodeType::PluginCustom entry describes. Installed
// extension: pins from the registry, with a note on the node when the
// version, the code or the pins differ from what was saved. Extension not
// installed: a node marked extension_missing with the saved pins. False with
// a reason when the entry has no valid extension block.
bool ReadExtensionNode(const nlohmann::json& node_json, int& next_node_id, int& next_pin_id,
                       gui::MLNode& node, std::string& error);

}  // namespace cyxwiz
