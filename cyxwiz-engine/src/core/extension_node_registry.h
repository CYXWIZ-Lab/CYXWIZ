#pragma once

// The one registry of extension nodes (TOFIX125). Providers (the plugin
// manager, later the Python node provider) register here; the node factory,
// graph loader, compiler, model builder and editor read from here.
//
// Queries return copies or shared pointers, never pointers into the
// registry, and no registry lock is held while a provider is called.

#include "extension_node_contract.h"
#include "../plugin/interfaces/i_node_provider.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace cyxwiz {

class ExtensionNodeRegistry {
public:
    static ExtensionNodeRegistry& Instance();

    ExtensionNodeRegistry(const ExtensionNodeRegistry&) = delete;
    ExtensionNodeRegistry& operator=(const ExtensionNodeRegistry&) = delete;

    struct Registration {
        ExtensionNodeDescriptor descriptor;
        // TensorLayer nodes: required.
        std::shared_ptr<ITensorNodeFactory> tensor_factory;
        // Signal nodes: required, non-owning (the plugin manager owns the plugin).
        plugin::INodeProvider* signal_provider = nullptr;
        // Keeps the provider's code loaded while anything holds a copy.
        std::shared_ptr<void> provider_lease;
    };

    // False with a reason: invalid id, id already registered, or the kind's
    // factory/provider is missing. descriptor.type_id is filled from
    // provider_id and type_name.
    bool Register(Registration registration, std::string& error);

    // Removes every node of a provider; returns how many.
    size_t RemoveByProvider(const std::string& provider_id);

    std::optional<ExtensionNodeDescriptor> Find(const std::string& type_id) const;
    bool Has(const std::string& type_id) const;
    // Ordered by type id.
    std::vector<ExtensionNodeDescriptor> All() const;
    size_t Count() const;

    std::shared_ptr<ITensorNodeFactory> TensorFactory(const std::string& type_id) const;
    plugin::INodeProvider* SignalProvider(const std::string& type_id) const;
    std::shared_ptr<void> ProviderLease(const std::string& type_id) const;

    // Simulation nodes: calls into the provider. Empty result when the node is
    // unknown, is not a simulation node, or the provider throws (logged).
    plugin::DynamicPinResult ResolveDynamicPins(
        const std::string& type_id,
        const std::map<std::string, std::string>& parameters) const;
    std::string GenerateCode(const std::string& type_id,
                             const std::map<std::string, std::string>& parameters,
                             const std::string& framework) const;

    // Increases on every successful Register and every RemoveByProvider that
    // removed something. Caches of compile results key on it.
    uint64_t Generation() const;

    // Called after each change, on the thread that made it, with no registry
    // lock held. Returns a token for Unsubscribe.
    int Subscribe(std::function<void()> on_change);
    void Unsubscribe(int token);

private:
    ExtensionNodeRegistry() = default;

    void NotifyChanged();

    mutable std::mutex mutex_;
    std::map<std::string, Registration> nodes_;  // type id -> registration
    uint64_t generation_ = 0;
    std::map<int, std::function<void()>> subscribers_;
    int next_subscriber_token_ = 1;
};

// The descriptor of a plugin's simulation node. Pins become Tensor ports and
// default parameters become string parameters, as the node factory has
// always treated them.
ExtensionNodeDescriptor DescribeSignalNode(const std::string& provider_id,
                                           const plugin::PluginNodeTypeInfo& info);

}  // namespace cyxwiz
