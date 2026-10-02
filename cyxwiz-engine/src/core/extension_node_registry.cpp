#include "extension_node_registry.h"

#include <spdlog/spdlog.h>

#include <utility>

namespace cyxwiz {

std::string MakeExtensionTypeId(const std::string& provider_id, const std::string& type_name) {
    return provider_id + ":" + type_name;
}

bool ValidateExtensionTypeId(const std::string& provider_id, const std::string& type_name,
                             std::string& error) {
    if (provider_id.empty()) {
        error = "the provider id is empty";
        return false;
    }
    if (type_name.empty()) {
        error = "the node type name is empty";
        return false;
    }
    for (const char c : provider_id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '-' || c == '_';
        if (!ok) {
            error = "provider id '" + provider_id +
                    "' may only use a-z, 0-9, '.', '-' and '_'";
            return false;
        }
    }
    for (const char c : type_name) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_';
        if (!ok) {
            error = "node type name '" + type_name +
                    "' may only use letters, digits and '_'";
            return false;
        }
    }
    return true;
}

bool SplitExtensionTypeId(const std::string& type_id, std::string& provider_id,
                          std::string& type_name) {
    const auto colon = type_id.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= type_id.size()) {
        return false;
    }
    provider_id = type_id.substr(0, colon);
    type_name = type_id.substr(colon + 1);
    return true;
}

ExtensionNodeRegistry& ExtensionNodeRegistry::Instance() {
    static ExtensionNodeRegistry instance;
    return instance;
}

bool ExtensionNodeRegistry::Register(Registration registration, std::string& error) {
    auto& descriptor = registration.descriptor;
    if (!ValidateExtensionTypeId(descriptor.provider_id, descriptor.type_name, error)) {
        return false;
    }
    descriptor.type_id = MakeExtensionTypeId(descriptor.provider_id, descriptor.type_name);
    descriptor.metadata.type = gui::NodeType::PluginCustom;
    descriptor.metadata.category = gui::NodeCategory::Plugin;

    if (descriptor.kind == ExtensionNodeKind::TensorLayer && !registration.tensor_factory) {
        error = "trainable node '" + descriptor.type_id + "' has no tensor factory";
        return false;
    }
    if (descriptor.kind == ExtensionNodeKind::Signal && registration.signal_provider == nullptr) {
        error = "simulation node '" + descriptor.type_id + "' has no provider";
        return false;
    }

    {
        std::lock_guard lock(mutex_);
        const auto existing = nodes_.find(descriptor.type_id);
        if (existing != nodes_.end()) {
            error = "node type '" + descriptor.type_id + "' is already registered by '" +
                    existing->second.descriptor.provider_id + "' (" +
                    existing->second.descriptor.source_path + ")";
            return false;
        }
        spdlog::info("ExtensionNodeRegistry: registered {} ({})",
                     descriptor.metadata.name, descriptor.type_id);
        const std::string type_id = descriptor.type_id;
        nodes_.emplace(type_id, std::move(registration));
        ++generation_;
    }
    NotifyChanged();
    return true;
}

size_t ExtensionNodeRegistry::RemoveByProvider(const std::string& provider_id) {
    size_t removed = 0;
    {
        std::lock_guard lock(mutex_);
        for (auto it = nodes_.begin(); it != nodes_.end();) {
            if (it->second.descriptor.provider_id == provider_id) {
                it = nodes_.erase(it);
                ++removed;
            } else {
                ++it;
            }
        }
        if (removed > 0) ++generation_;
    }
    if (removed > 0) NotifyChanged();
    return removed;
}

std::optional<ExtensionNodeDescriptor> ExtensionNodeRegistry::Find(const std::string& type_id) const {
    std::lock_guard lock(mutex_);
    const auto it = nodes_.find(type_id);
    if (it == nodes_.end()) return std::nullopt;
    return it->second.descriptor;
}

bool ExtensionNodeRegistry::Has(const std::string& type_id) const {
    std::lock_guard lock(mutex_);
    return nodes_.count(type_id) > 0;
}

std::vector<ExtensionNodeDescriptor> ExtensionNodeRegistry::All() const {
    std::lock_guard lock(mutex_);
    std::vector<ExtensionNodeDescriptor> result;
    result.reserve(nodes_.size());
    for (const auto& [type_id, registration] : nodes_) {
        result.push_back(registration.descriptor);
    }
    return result;
}

size_t ExtensionNodeRegistry::Count() const {
    std::lock_guard lock(mutex_);
    return nodes_.size();
}

std::shared_ptr<ITensorNodeFactory> ExtensionNodeRegistry::TensorFactory(const std::string& type_id) const {
    std::lock_guard lock(mutex_);
    const auto it = nodes_.find(type_id);
    return it == nodes_.end() ? nullptr : it->second.tensor_factory;
}

plugin::INodeProvider* ExtensionNodeRegistry::SignalProvider(const std::string& type_id) const {
    std::lock_guard lock(mutex_);
    const auto it = nodes_.find(type_id);
    return it == nodes_.end() ? nullptr : it->second.signal_provider;
}

std::shared_ptr<void> ExtensionNodeRegistry::ProviderLease(const std::string& type_id) const {
    std::lock_guard lock(mutex_);
    const auto it = nodes_.find(type_id);
    return it == nodes_.end() ? nullptr : it->second.provider_lease;
}

plugin::DynamicPinResult ExtensionNodeRegistry::ResolveDynamicPins(
    const std::string& type_id,
    const std::map<std::string, std::string>& parameters) const {
    plugin::INodeProvider* provider = nullptr;
    std::string type_name;
    {
        std::lock_guard lock(mutex_);
        const auto it = nodes_.find(type_id);
        if (it == nodes_.end() || it->second.signal_provider == nullptr) return {};
        provider = it->second.signal_provider;
        type_name = it->second.descriptor.type_name;
    }
    try {
        return provider->ResolveDynamicPins(type_name, parameters);
    } catch (const std::exception& e) {
        spdlog::error("ExtensionNodeRegistry: ResolveDynamicPins failed for {}: {}", type_id, e.what());
        return {};
    }
}

std::string ExtensionNodeRegistry::GenerateCode(
    const std::string& type_id,
    const std::map<std::string, std::string>& parameters,
    const std::string& framework) const {
    plugin::INodeProvider* provider = nullptr;
    std::string type_name;
    {
        std::lock_guard lock(mutex_);
        const auto it = nodes_.find(type_id);
        if (it == nodes_.end() || it->second.signal_provider == nullptr) return {};
        provider = it->second.signal_provider;
        type_name = it->second.descriptor.type_name;
    }
    try {
        return provider->GenerateCode(type_name, parameters, framework);
    } catch (const std::exception& e) {
        spdlog::error("ExtensionNodeRegistry: code generation failed for {}: {}", type_id, e.what());
        return {};
    }
}

uint64_t ExtensionNodeRegistry::Generation() const {
    std::lock_guard lock(mutex_);
    return generation_;
}

int ExtensionNodeRegistry::Subscribe(std::function<void()> on_change) {
    std::lock_guard lock(mutex_);
    const int token = next_subscriber_token_++;
    subscribers_.emplace(token, std::move(on_change));
    return token;
}

void ExtensionNodeRegistry::Unsubscribe(int token) {
    std::lock_guard lock(mutex_);
    subscribers_.erase(token);
}

void ExtensionNodeRegistry::NotifyChanged() {
    std::vector<std::function<void()>> callbacks;
    {
        std::lock_guard lock(mutex_);
        callbacks.reserve(subscribers_.size());
        for (const auto& [token, callback] : subscribers_) {
            callbacks.push_back(callback);
        }
    }
    for (const auto& callback : callbacks) {
        if (callback) callback();
    }
}

ExtensionNodeDescriptor DescribeSignalNode(const std::string& provider_id,
                                           const plugin::PluginNodeTypeInfo& info) {
    ExtensionNodeDescriptor descriptor;
    descriptor.provider_id = provider_id;
    descriptor.type_name = info.type_name;
    descriptor.type_id = MakeExtensionTypeId(provider_id, info.type_name);
    descriptor.source = ExtensionSource::NativePlugin;
    descriptor.kind = ExtensionNodeKind::Signal;
    descriptor.menu_category = info.category;
    descriptor.color = info.color;
    descriptor.supports_dynamic_pins = info.supports_dynamic_pins;
    descriptor.dynamic_pin_trigger = info.dynamic_pin_trigger;

    auto& metadata = descriptor.metadata;
    metadata.type = gui::NodeType::PluginCustom;
    metadata.category = gui::NodeCategory::Plugin;
    metadata.name = info.display_name.empty() ? info.type_name : info.display_name;
    metadata.icon = info.icon;
    metadata.brief_description = info.description;
    for (const auto& pin : info.pins) {
        PortDefinition port(pin.name, PinType::Tensor);
        if (pin.is_input) {
            metadata.inputs.push_back(std::move(port));
        } else {
            metadata.outputs.push_back(std::move(port));
        }
    }
    for (const auto& [key, value] : info.default_parameters) {
        metadata.parameters.emplace_back(key, "string", value);
    }
    return descriptor;
}

}  // namespace cyxwiz
