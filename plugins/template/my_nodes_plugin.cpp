// SPDX-License-Identifier: Apache-2.0
// From the CyxWiz plugin SDK template: replace this licence with your own when you make it yours.
// My Nodes: a CyxWiz plugin with one trainable node.
//
// Bent Identity  y = (sqrt(x^2 + 1) - 1) / 2 + x
//
// A plugin is a shared library with one IPlugin class. The Engine asks it for
// ITensorNodeProvider, lists its nodes in the node editor and, when a graph
// uses one, calls CreateModule for a cyxwiz::Module that it trains like a
// built-in layer. Rename the node, change Forward and Backward, and rebuild.

#include "plugin/plugin_types.h"
#include "plugin/plugin_context.h"
#include "plugin/interfaces/i_tensor_node_provider.h"

#include <cyxwiz/sequential.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace my_nodes {

using namespace cyxwiz;
using namespace cyxwiz::plugin;

class BentIdentityModule : public Module {
public:
    Tensor Forward(const Tensor& x) override {
        input_cache_ = x;  // Backward needs the input
        return ((x * x + 1.0f).Sqrt() - 1.0f) / 2.0f + x;
    }

    // Receives dLoss/dOutput and returns dLoss/dInput, the input's shape.
    Tensor Backward(const Tensor& grad) override {
        const Tensor& x = input_cache_;
        return grad * (x / ((x * x + 1.0f).Sqrt() * 2.0f) + 1.0f);
    }

    // Elementwise: each position depends only on itself.
    IncrementalDecoding GetIncrementalDecoding() const override { return IncrementalDecoding::PositionWise; }
    std::string GetName() const override { return "BentIdentity"; }
};

class MyNodesPlugin : public IPlugin, public ITensorNodeProvider {
public:
    bool OnLoad(PluginContext& ctx) override {
        ctx.LogInfo("My Nodes loaded");
        state_ = PluginState::Loaded;
        return true;
    }
    bool OnInitialize(PluginContext&) override {
        state_ = PluginState::Active;
        return true;
    }
    void OnShutdown(PluginContext&) override { state_ = PluginState::Loaded; }
    void OnUnload(PluginContext&) override { state_ = PluginState::Unloaded; }

    const PluginManifest& GetManifest() const override { return manifest_; }
    PluginPermissionFlags GetRequiredPermissions() const override {
        return static_cast<uint32_t>(PluginPermission::Training);
    }
    PluginState GetState() const override { return state_; }
    void* QueryInterface(const char* name) override {
        if (std::string(name) == "ITensorNodeProvider") return static_cast<ITensorNodeProvider*>(this);
        return nullptr;
    }

    // What the node editor shows.
    std::vector<TensorNodeInfo> GetTensorNodes() override {
        TensorNodeInfo node;
        node.type_name = "BentIdentity";  // saved in graphs: do not rename once used
        node.display_name = "Bent Identity";
        node.category = "Activations";
        node.description = "(sqrt(x^2 + 1) - 1) / 2 + x: smooth, unbounded, never flat";
        node.version = "1.0.0";
        node.is_activation = true;
        node.pins = {{"Input", true}, {"Output", false}};
        return {node};
    }

    // Shapes are one sample's, without the batch dimension.
    ExtensionShapeResult InferOutputShape(const std::string&, const std::vector<size_t>& input_shape,
                                          const std::map<std::string, std::string>&) override {
        return {true, input_shape, {}};
    }

    std::unique_ptr<Module> CreateModule(const std::string& type_name, const std::vector<size_t>&,
                                         const std::map<std::string, std::string>&,
                                         std::string& error) override {
        if (type_name == "BentIdentity") return std::make_unique<BentIdentityModule>();
        error = "unknown node type " + type_name;
        return nullptr;
    }

private:
    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;
};

}  // namespace my_nodes

CYXWIZ_PLUGIN_ENTRY(my_nodes::MyNodesPlugin)
