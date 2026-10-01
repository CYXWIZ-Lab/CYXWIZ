// Example plugin with trainable nodes (TOFIX125, plugin API 1.1).
//
// Two activations CyxWiz does not ship, each a real cyxwiz::Module:
//   Softsign  y = x / (1 + |x|)                      no parameters
//   Snake     y = x + sin^2(a x) / a                 one learnable a
//
// The engine finds them through ITensorNodeProvider, lists them in the node
// editor, compiles them into the model and trains them. Its guard checks
// the shapes these modules return.

#include "../../../cyxwiz-engine/src/plugin/plugin_types.h"
#include "../../../cyxwiz-engine/src/plugin/plugin_context.h"
#include "../../../cyxwiz-engine/src/plugin/interfaces/i_tensor_node_provider.h"

#include <cyxwiz/sequential.h>

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyxwiz::plugin::examples {

namespace {

float HostScalar(const Tensor& tensor) {
    return tensor.ReadData<float>()[0];
}

Tensor Scalar(float value) {
    return Tensor(std::vector<size_t>{1}, &value);
}

class SoftsignModule : public Module {
public:
    Tensor Forward(const Tensor& x) override {
        input_cache_ = x;
        return x / (x.Abs() + 1.0f);
    }
    Tensor Backward(const Tensor& grad) override {
        const Tensor d = input_cache_.Abs() + 1.0f;
        return grad / (d * d);
    }
    IncrementalDecoding GetIncrementalDecoding() const override { return IncrementalDecoding::PositionWise; }
    std::string GetName() const override { return "Softsign"; }
};

class SnakeModule : public Module {
public:
    explicit SnakeModule(float initial_a) : a_(Scalar(initial_a)), grad_a_(Scalar(0.0f)) {}

    Tensor Forward(const Tensor& x) override {
        input_cache_ = x;
        const float a = HostScalar(a_);
        const Tensor s = (x * a).Sin();
        return x + (s * s) / a;
    }

    Tensor Backward(const Tensor& grad) override {
        const float a = HostScalar(a_);
        const Tensor& x = input_cache_;
        const Tensor s = (x * a).Sin();
        const Tensor sin2 = (x * (2.0f * a)).Sin();
        // dy/da = x sin(2ax) / a - sin^2(ax) / a^2, summed against grad.
        const Tensor da = grad * ((x * sin2) / a - (s * s) / (a * a));
        const float* values = da.ReadData<float>();
        double sum = 0.0;
        for (size_t i = 0; i < da.NumElements(); ++i) sum += values[i];
        grad_a_ = Scalar(static_cast<float>(sum));
        // dy/dx = 1 + sin(2ax).
        return grad * (sin2 + 1.0f);
    }

    std::map<std::string, Tensor> GetParameters() override { return {{"a", a_}}; }
    void SetParameters(const std::map<std::string, Tensor>& params) override {
        const auto it = params.find("a");
        if (it != params.end()) a_ = it->second;
    }
    std::map<std::string, Tensor> GetGradients() override { return {{"a", grad_a_}}; }
    bool HasParameters() const override { return true; }
    IncrementalDecoding GetIncrementalDecoding() const override { return IncrementalDecoding::PositionWise; }
    std::string GetName() const override { return "Snake"; }

private:
    Tensor a_;
    Tensor grad_a_;
};

float ParseFloat(const std::map<std::string, std::string>& parameters, const std::string& key, float fallback) {
    const auto it = parameters.find(key);
    if (it == parameters.end() || it->second.empty()) return fallback;
    return std::stof(it->second);
}

}  // namespace

class ActivationNodesPlugin : public IPlugin, public ITensorNodeProvider {
public:
    bool OnLoad(PluginContext& ctx) override {
        ctx.LogInfo("Example Activation Nodes: OnLoad");
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

    std::vector<TensorNodeInfo> GetTensorNodes() override {
        TensorNodeInfo softsign;
        softsign.type_name = "Softsign";
        softsign.display_name = "Softsign";
        softsign.category = "Activations";
        softsign.description = "x / (1 + |x|): a smooth activation between -1 and 1";
        softsign.version = "1.0.0";
        softsign.color = 0xFF9C6A2E;  // IM_COL32(46, 106, 156)
        softsign.is_activation = true;
        softsign.pins = {{"Input", true}, {"Output", false}};

        TensorNodeInfo snake;
        snake.type_name = "Snake";
        snake.display_name = "Snake";
        snake.category = "Activations";
        snake.description = "x + sin^2(a x) / a, with a learnable frequency a; suits periodic signals";
        snake.version = "1.0.0";
        snake.color = 0xFF3C8F4A;  // IM_COL32(74, 143, 60)
        snake.is_activation = true;
        snake.pins = {{"Input", true}, {"Output", false}};
        snake.parameters = {{"initial_a", "float", "1.0", "Starting value of the frequency a (greater than 0)", {}}};
        return {softsign, snake};
    }

    ExtensionShapeResult InferOutputShape(const std::string& type_name,
                                          const std::vector<size_t>& input_shape,
                                          const std::map<std::string, std::string>& parameters) override {
        if (type_name == "Snake") {
            float a = 0.0f;
            try {
                a = ParseFloat(parameters, "initial_a", 1.0f);
            } catch (const std::exception&) {
                return {false, {}, "initial_a must be a number"};
            }
            if (!(a > 0.0f)) return {false, {}, "initial_a must be greater than 0"};
        }
        // Elementwise: the output has the input's shape.
        return {true, input_shape, {}};
    }

    std::unique_ptr<Module> CreateModule(const std::string& type_name,
                                         const std::vector<size_t>&,
                                         const std::map<std::string, std::string>& parameters,
                                         std::string& error) override {
        if (type_name == "Softsign") return std::make_unique<SoftsignModule>();
        if (type_name == "Snake") return std::make_unique<SnakeModule>(ParseFloat(parameters, "initial_a", 1.0f));
        error = "unknown node type " + type_name;
        return nullptr;
    }

private:
    PluginManifest manifest_;
    PluginState state_ = PluginState::Unloaded;
};

}  // namespace cyxwiz::plugin::examples

CYXWIZ_PLUGIN_ENTRY(cyxwiz::plugin::examples::ActivationNodesPlugin)
