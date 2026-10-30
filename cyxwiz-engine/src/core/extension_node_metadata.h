#pragma once

// One way to get the metadata of a node on the canvas (TOFIX125). The
// catalog (NodeMetadataRegistry) is keyed by NodeType, so every extension
// node would resolve to the generic PluginCustom entry there.

#include "graph_model.h"
#include "node_metadata.h"
#include "pipeline_runtime_capabilities.h"

#include <optional>
#include <string>

namespace cyxwiz {

// Built-in node: its catalog entry. Extension node: its descriptor's
// metadata. Extension node whose extension is not registered: a placeholder
// built from the node's own pins, badged "Not installed". Empty only for a
// built-in type the catalog does not know.
std::optional<NodeMetadata> ResolveNodeMetadata(const gui::MLNode& node);

// Why generated code (PyTorch, TensorFlow, Keras, pycyxwiz) cannot contain
// this node, or nothing. A trainable extension node's computation is the
// extension's own code, and a node whose extension is not installed has no
// code at all: a script without them would leave the layer out and still run
// (TOFIX125 P6). Built-in nodes and signal nodes: nothing.
std::optional<std::string> ExtensionCodeExportRefusal(const gui::MLNode& node);

// What an extension node is to the toolbar's Train check, as the compiler
// counts it: a trainable layer (role Layer) is a model layer, a loss node the
// loss; nothing for activations, simulation and unknown nodes.
std::optional<PipelineTrainingSupportRole> ExtensionTrainingRole(const gui::MLNode& node);

// A simulation node (a plugin's signal node) in generated code: never a
// layer, since it is not part of a trained model (declaring one shifted every
// later self.layerN in the forward pass). A comment line, with the plugin's
// code text for `framework` ("pytorch", "tensorflow", "keras", "pycyxwiz")
// as further comment lines, each starting with `indent` (TOFIX125 P6).
std::string SimulationNodeCodeComment(const gui::MLNode& node, const char* framework, const std::string& indent);

}  // namespace cyxwiz
