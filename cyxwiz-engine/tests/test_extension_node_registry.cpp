// The extension node registry (TOFIX125 P1 step 1.1): identity rules,
// registration, removal, the generation counter and change notification.
// No GUI, no plugin library, no Python: providers are stand-ins in this file.
#include "../src/core/extension_node_registry.h"
#include "../src/plugin/interfaces/i_node_provider.h"

#include <cyxwiz/sequential.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class StubSignalProvider : public cyxwiz::plugin::INodeProvider {
public:
    bool throw_on_call = false;
    std::string last_type_name;

    std::vector<cyxwiz::plugin::PluginNodeTypeInfo> GetNodeTypes() override { return {}; }
    std::string GenerateCode(const std::string& type_name,
                             const std::map<std::string, std::string>& parameters,
                             const std::string& framework) override {
        if (throw_on_call) throw std::runtime_error("provider failed");
        last_type_name = type_name;
        const auto it = parameters.find("gain");
        return framework + ":" + type_name + ":" + (it == parameters.end() ? "" : it->second);
    }
    cyxwiz::plugin::DynamicPinResult ResolveDynamicPins(
        const std::string& type_name,
        const std::map<std::string, std::string>&) override {
        if (throw_on_call) throw std::runtime_error("provider failed");
        last_type_name = type_name;
        cyxwiz::plugin::DynamicPinResult result;
        result.pins.push_back({"joint_0", "Signal", true});
        result.pins.push_back({"state", "Signal", false});
        return result;
    }
};

// Never asked to build a module here; the registry only stores it.
class StubTensorFactory : public cyxwiz::ITensorNodeFactory {
public:
    cyxwiz::ExtensionShapeResult InferOutputShape(
        const std::vector<size_t>& input_shape,
        const std::map<std::string, std::string>&) const override {
        return {true, input_shape, {}};
    }
    std::unique_ptr<cyxwiz::Module> CreateModule(
        const std::vector<size_t>&, const std::map<std::string, std::string>&,
        std::string& error) const override {
        error = "not used in this test";
        return nullptr;
    }
};

cyxwiz::ExtensionNodeRegistry::Registration SignalNode(const std::string& provider_id,
                                                       const std::string& type_name,
                                                       cyxwiz::plugin::INodeProvider* provider) {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    registration.descriptor.provider_id = provider_id;
    registration.descriptor.type_name = type_name;
    registration.descriptor.kind = cyxwiz::ExtensionNodeKind::Signal;
    registration.descriptor.metadata.name = type_name;
    registration.descriptor.source_path = "test://" + provider_id;
    registration.signal_provider = provider;
    return registration;
}

cyxwiz::ExtensionNodeRegistry::Registration TensorNode(
    const std::string& provider_id, const std::string& type_name,
    std::shared_ptr<cyxwiz::ITensorNodeFactory> factory) {
    cyxwiz::ExtensionNodeRegistry::Registration registration;
    registration.descriptor.provider_id = provider_id;
    registration.descriptor.type_name = type_name;
    registration.descriptor.kind = cyxwiz::ExtensionNodeKind::TensorLayer;
    registration.descriptor.role = cyxwiz::ExtensionLayerRole::Activation;
    registration.descriptor.metadata.name = type_name;
    registration.descriptor.metadata.inputs = {{"Input"}};
    registration.descriptor.metadata.outputs = {{"Output"}};
    registration.descriptor.metadata.parameters = {
        {"initial_a", "float", "1.0", "Starting value of a"}};
    registration.descriptor.source_path = "test://" + provider_id;
    registration.tensor_factory = std::move(factory);
    return registration;
}

void TestIdentityRules() {
    std::string error;
    Check(cyxwiz::ValidateExtensionTypeId("com.me.acts", "Snake", error), "a plain id is valid");
    Check(cyxwiz::ValidateExtensionTypeId("com.cyxwiz.examples.image-nodes", "GaussianBlur", error),
          "the image example plugin's id is valid");
    Check(cyxwiz::ValidateExtensionTypeId("mujoco_sim", "MuJoCoPlant_2", error),
          "underscores and digits are valid");

    Check(!cyxwiz::ValidateExtensionTypeId("", "Snake", error), "an empty provider id is refused");
    Check(!error.empty(), "a refusal gives a reason");
    Check(!cyxwiz::ValidateExtensionTypeId("com.me", "", error), "an empty type name is refused");
    Check(!cyxwiz::ValidateExtensionTypeId("Com.Me", "Snake", error),
          "upper case in the provider id is refused");
    Check(!cyxwiz::ValidateExtensionTypeId("com.me:x", "Snake", error),
          "a colon in the provider id is refused");
    Check(!cyxwiz::ValidateExtensionTypeId("com.me", "Sna ke", error),
          "a space in the type name is refused");
    Check(!cyxwiz::ValidateExtensionTypeId("com.me", "Snake:2", error),
          "a colon in the type name is refused");

    Check(cyxwiz::MakeExtensionTypeId("com.me.acts", "Snake") == "com.me.acts:Snake",
          "the type id joins provider and name with a colon");

    std::string provider;
    std::string name;
    Check(cyxwiz::SplitExtensionTypeId("com.me.acts:Snake", provider, name) &&
              provider == "com.me.acts" && name == "Snake",
          "a type id splits into provider and name");
    Check(!cyxwiz::SplitExtensionTypeId("Snake", provider, name), "no colon does not split");
    Check(!cyxwiz::SplitExtensionTypeId(":Snake", provider, name), "an empty provider does not split");
    Check(!cyxwiz::SplitExtensionTypeId("com.me:", provider, name), "an empty name does not split");
}

void TestRegisterFindRemove() {
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    Check(registry.Count() == 0, "the registry starts empty");

    StubSignalProvider signal_provider;
    auto factory = std::make_shared<StubTensorFactory>();
    std::string error;

    Check(registry.Register(SignalNode("test.sim", "Plant", &signal_provider), error),
          "a simulation node registers: " + error);
    Check(registry.Register(TensorNode("test.acts", "Snake", factory), error),
          "a trainable node registers: " + error);
    Check(registry.Register(TensorNode("test.acts", "Softsign", factory), error),
          "a second node of the same provider registers: " + error);
    Check(registry.Count() == 3, "three nodes are registered");

    const auto snake = registry.Find("test.acts:Snake");
    Check(snake.has_value(), "a registered node is found by type id");
    Check(snake->type_id == "test.acts:Snake", "the registry fills the type id");
    Check(snake->metadata.type == gui::NodeType::PluginCustom,
          "an extension node's metadata type is PluginCustom");
    Check(snake->metadata.category == gui::NodeCategory::Plugin,
          "an extension node's metadata category is Plugin");
    Check(snake->metadata.parameters.size() == 1 && snake->metadata.parameters[0].type == "float",
          "typed parameters are kept");
    Check(snake->metadata.inputs.size() == 1 && snake->metadata.outputs.size() == 1,
          "ports are kept");

    Check(registry.Has("test.sim:Plant"), "Has finds a registered node");
    Check(!registry.Has("test.sim:Missing"), "Has does not find an unknown node");
    Check(!registry.Find("Snake").has_value(), "the display name is not an identity");

    Check(registry.TensorFactory("test.acts:Snake") == factory,
          "the tensor factory is returned for a trainable node");
    Check(registry.TensorFactory("test.sim:Plant") == nullptr,
          "a simulation node has no tensor factory");
    Check(registry.SignalProvider("test.sim:Plant") == &signal_provider,
          "the provider is returned for a simulation node");
    Check(registry.SignalProvider("test.acts:Snake") == nullptr,
          "a trainable node has no simulation provider");
    Check(registry.TensorFactory("test.acts:Missing") == nullptr,
          "an unknown node has no factory");

    const auto all = registry.All();
    Check(all.size() == 3 && all[0].type_id == "test.acts:Snake" &&
              all[1].type_id == "test.acts:Softsign" && all[2].type_id == "test.sim:Plant",
          "All is ordered by type id");

    Check(registry.RemoveByProvider("test.acts") == 2, "removing a provider removes its two nodes");
    Check(registry.Count() == 1, "the other provider's node stays");
    Check(!registry.Has("test.acts:Snake"), "a removed node is gone");
    Check(registry.RemoveByProvider("test.acts") == 0, "removing again removes nothing");
    Check(registry.RemoveByProvider("test.sim") == 1, "the last node is removed");
    Check(registry.Count() == 0, "the registry is empty again");
}

void TestRefusals() {
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    StubSignalProvider signal_provider;
    auto factory = std::make_shared<StubTensorFactory>();
    std::string error;

    Check(!registry.Register(TensorNode("Bad.Id", "Snake", factory), error),
          "an invalid provider id is refused");
    Check(!registry.Register(TensorNode("test.acts", "Sna ke", factory), error),
          "an invalid type name is refused");
    Check(!registry.Register(TensorNode("test.acts", "Snake", nullptr), error),
          "a trainable node without a factory is refused");
    Check(error.find("test.acts:Snake") != std::string::npos,
          "the refusal names the node: " + error);
    Check(!registry.Register(SignalNode("test.sim", "Plant", nullptr), error),
          "a simulation node without a provider is refused");
    Check(registry.Count() == 0, "nothing was registered by the refusals");

    Check(registry.Register(TensorNode("test.acts", "Snake", factory), error), "first registration");
    auto duplicate = TensorNode("test.acts", "Snake", factory);
    duplicate.descriptor.source_path = "test://second";
    Check(!registry.Register(std::move(duplicate), error), "a duplicate type id is refused");
    Check(error.find("test.acts:Snake") != std::string::npos &&
              error.find("test://test.acts") != std::string::npos,
          "the refusal names the node and where the first one came from: " + error);
    Check(registry.Find("test.acts:Snake")->source_path == "test://test.acts",
          "the first registration is kept");

    // The same type name under another provider is a different node.
    Check(registry.Register(TensorNode("other.acts", "Snake", factory), error),
          "the same name under another provider registers: " + error);
    registry.RemoveByProvider("test.acts");
    registry.RemoveByProvider("other.acts");
    Check(registry.Count() == 0, "cleanup");
}

void TestGenerationAndNotification() {
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    auto factory = std::make_shared<StubTensorFactory>();
    std::string error;

    int notified = 0;
    size_t count_seen_in_callback = 0;
    const int token = registry.Subscribe([&] {
        ++notified;
        // A subscriber may query the registry: no lock is held during the call.
        count_seen_in_callback = registry.Count();
    });

    const uint64_t start = registry.Generation();
    Check(registry.Register(TensorNode("test.acts", "Snake", factory), error), "register");
    Check(registry.Generation() == start + 1, "registering increases the generation");
    Check(notified == 1 && count_seen_in_callback == 1, "registering notifies once, after the change");

    Check(!registry.Register(TensorNode("test.acts", "Snake", factory), error), "duplicate");
    Check(registry.Generation() == start + 1, "a refused registration leaves the generation");
    Check(notified == 1, "a refused registration does not notify");

    Check(registry.RemoveByProvider("nobody") == 0, "removing an unknown provider");
    Check(registry.Generation() == start + 1 && notified == 1,
          "removing nothing changes nothing");

    Check(registry.RemoveByProvider("test.acts") == 1, "remove");
    Check(registry.Generation() == start + 2, "removing increases the generation");
    Check(notified == 2 && count_seen_in_callback == 0, "removing notifies once, after the change");

    registry.Unsubscribe(token);
    Check(registry.Register(TensorNode("test.acts", "Snake", factory), error), "register again");
    Check(notified == 2, "an unsubscribed callback is not called");
    registry.RemoveByProvider("test.acts");
}

void TestLeaseIsKept() {
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    auto factory = std::make_shared<StubTensorFactory>();
    auto lease = std::make_shared<int>(125);
    std::string error;

    auto registration = TensorNode("test.acts", "Snake", factory);
    registration.provider_lease = lease;
    Check(registry.Register(std::move(registration), error), "register with a lease");
    Check(lease.use_count() == 2, "the registry holds the lease");

    {
        const auto held = registry.ProviderLease("test.acts:Snake");
        Check(held == lease && lease.use_count() == 3, "a caller can hold the lease");
        registry.RemoveByProvider("test.acts");
        Check(lease.use_count() == 2, "after removal the caller's copy keeps the provider alive");
    }
    Check(lease.use_count() == 1, "the lease is released when the last holder lets go");
    Check(registry.ProviderLease("test.acts:Snake") == nullptr, "an unknown node has no lease");
}

void TestSignalNodeDescriptor() {
    cyxwiz::plugin::PluginNodeTypeInfo info;
    info.type_name = "MuJoCoPlant";
    info.display_name = "MuJoCo Plant";
    info.category = "Simulation";
    info.description = "Physics plant";
    info.color = 0xFF112233UL;
    info.icon = "icon";
    info.pins = {{"action", "Signal", true}, {"observation", "Signal", false},
                 {"reward", "Signal", false}, {"reset", "Signal", true}};
    info.default_parameters = {{"mjcf_path", ""}, {"frame_skip", "5"}};
    info.supports_dynamic_pins = true;
    info.dynamic_pin_trigger = "mjcf_path";

    const auto d = cyxwiz::DescribeSignalNode("com.cyxwiz.simulation.mujoco", info);
    Check(d.type_id == "com.cyxwiz.simulation.mujoco:MuJoCoPlant", "the type id is provider:name");
    Check(d.kind == cyxwiz::ExtensionNodeKind::Signal, "a plugin simulation node is a Signal node");
    Check(d.metadata.name == "MuJoCo Plant" && d.metadata.brief_description == "Physics plant",
          "name and description are kept");
    Check(d.menu_category == "Simulation" && d.color == 0xFF112233UL, "category and colour are kept");
    Check(d.supports_dynamic_pins && d.dynamic_pin_trigger == "mjcf_path", "dynamic pins are kept");
    Check(d.metadata.inputs.size() == 2 && d.metadata.inputs[0].name == "action" &&
              d.metadata.inputs[1].name == "reset",
          "inputs keep their declared order");
    Check(d.metadata.outputs.size() == 2 && d.metadata.outputs[0].name == "observation" &&
              d.metadata.outputs[1].name == "reward",
          "outputs keep their declared order");
    Check(d.metadata.inputs[0].type == gui::PinType::Tensor, "plugin pins are Tensor ports");
    Check(d.metadata.parameters.size() == 2, "default parameters become parameter definitions");
    bool found_frame_skip = false;
    for (const auto& parameter : d.metadata.parameters) {
        if (parameter.name == "frame_skip") {
            found_frame_skip = parameter.default_value == "5" && parameter.type == "string";
        }
    }
    Check(found_frame_skip, "a default parameter keeps its value");

    cyxwiz::plugin::PluginNodeTypeInfo unnamed;
    unnamed.type_name = "Bare";
    Check(cyxwiz::DescribeSignalNode("test.sim", unnamed).metadata.name == "Bare",
          "a node without a display name shows its type name");
}

void TestProviderCalls() {
    auto& registry = cyxwiz::ExtensionNodeRegistry::Instance();
    StubSignalProvider provider;
    auto factory = std::make_shared<StubTensorFactory>();
    std::string error;
    Check(registry.Register(SignalNode("test.sim", "Plant", &provider), error), "register signal");
    Check(registry.Register(TensorNode("test.acts", "Snake", factory), error), "register tensor");

    Check(registry.GenerateCode("test.sim:Plant", {{"gain", "2"}}, "pytorch") == "pytorch:Plant:2",
          "code generation reaches the provider with the unqualified type name");
    const auto pins = registry.ResolveDynamicPins("test.sim:Plant", {});
    Check(pins.pins.size() == 2 && pins.pins[0].name == "joint_0" && provider.last_type_name == "Plant",
          "dynamic pins reach the provider with the unqualified type name");

    Check(registry.GenerateCode("test.sim:Missing", {}, "pytorch").empty(),
          "an unknown node generates no code");
    Check(registry.ResolveDynamicPins("test.sim:Missing", {}).pins.empty(),
          "an unknown node resolves no pins");
    Check(registry.GenerateCode("test.acts:Snake", {}, "pytorch").empty(),
          "a trainable node has no simulation provider to ask");

    provider.throw_on_call = true;
    Check(registry.GenerateCode("test.sim:Plant", {}, "pytorch").empty(),
          "a provider exception during code generation is contained");
    Check(registry.ResolveDynamicPins("test.sim:Plant", {}).pins.empty(),
          "a provider exception during pin resolution is contained");

    registry.RemoveByProvider("test.sim");
    registry.RemoveByProvider("test.acts");
    Check(registry.Count() == 0, "cleanup");
}

}  // namespace

int main() {
    TestIdentityRules();
    TestRegisterFindRemove();
    TestRefusals();
    TestGenerationAndNotification();
    TestLeaseIsKept();
    TestSignalNodeDescriptor();
    TestProviderCalls();
    std::cout << "test_extension_node_registry: all checks passed\n";
    return 0;
}
