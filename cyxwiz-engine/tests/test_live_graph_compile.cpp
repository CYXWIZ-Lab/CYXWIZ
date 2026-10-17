// LiveGraphCompile scheduling (TOFIX123, fix found in TOFIX125 2026-10-17):
// a compile made while training ran skips the parameter count. Once training
// ends the same graph is compiled again with the count, so Properties does
// not keep showing "Learnable parameters: not counted" until the next edit.
// A compile that already asked for the count is not repeated (no loop when
// the count cannot be made).
#include "../src/core/extension_node_registry.h"
#include "../src/core/live_graph_compile.h"

#include <cyxwiz/sequential.h>

#include <chrono>
#include <map>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = cyxwiz::LiveGraphCompile::Clock;

void Check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

class StubFactory : public cyxwiz::ITensorNodeFactory {
public:
    cyxwiz::ExtensionShapeResult InferOutputShape(const std::vector<size_t>& shape,
                                                  const std::map<std::string, std::string>&) const override {
        return {true, shape, {}};
    }
    std::unique_ptr<cyxwiz::Module> CreateModule(const std::vector<size_t>&,
                                                 const std::map<std::string, std::string>&,
                                                 std::string& error) const override {
        error = "not used";
        return nullptr;
    }
};

// Calls Update until the running compile has been collected.
void Finish(cyxwiz::LiveGraphCompile& live, const std::vector<gui::MLNode>& nodes,
            const std::vector<gui::NodeLink>& links, bool count, Clock::time_point& now) {
    for (int i = 0; i < 600 && live.ParametersCounting(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        now += std::chrono::milliseconds(10);
        live.Update(nodes, links, count, now);
    }
    Check(!live.ParametersCounting(), "the compile finishes");
}

}  // namespace

int main() {
    gui::MLNode dense;
    dense.id = 1;
    dense.type = gui::NodeType::Dense;
    dense.name = "Dense";
    dense.parameters["units"] = "4";
    const std::vector<gui::MLNode> nodes{dense};
    const std::vector<gui::NodeLink> links;

    cyxwiz::LiveGraphCompile live;
    Clock::time_point now = Clock::now();

    // While training: compiled without the count.
    live.Update(nodes, links, false, now);
    now += live.debounce + std::chrono::milliseconds(1);
    live.Update(nodes, links, false, now);
    Check(live.ParametersCounting(), "a compile starts after the debounce");
    Finish(live, nodes, links, false, now);
    Check(live.Config() != nullptr, "the compile result is kept");
    Check(!live.ParametersCounted(), "no count while training");
    const auto serial = live.ResultSerial();

    // Still training, graph unchanged: nothing new.
    live.Update(nodes, links, false, now);
    Check(!live.ParametersCounting(), "an unchanged graph is not compiled again while training");

    // Training ended, graph unchanged: compiled once more, with the count.
    live.Update(nodes, links, true, now);
    Check(live.ParametersCounting(), "the same graph is compiled again with the count after training");
    Finish(live, nodes, links, true, now);
    Check(live.ResultSerial() == serial + 1, "one more compile");

    // That compile asked for the count: no repeat, even if the count failed.
    for (int i = 0; i < 5; ++i) {
        now += live.debounce + std::chrono::milliseconds(1);
        live.Update(nodes, links, true, now);
        Check(!live.ParametersCounting(), "a compile that asked for the count is not repeated");
    }
    Check(live.ResultSerial() == serial + 1, "no further compiles");

    // An extension registered (a plugin loaded, a bundle approved) changes
    // what the graph compiles to: compiled again without an edit.
    {
        cyxwiz::ExtensionNodeRegistry::Registration registration;
        registration.descriptor.provider_id = "com.test.live";
        registration.descriptor.type_name = "Stub";
        registration.descriptor.kind = cyxwiz::ExtensionNodeKind::TensorLayer;
        registration.descriptor.metadata.inputs.emplace_back("Input");
        registration.descriptor.metadata.outputs.emplace_back("Output");
        registration.tensor_factory = std::make_shared<StubFactory>();
        std::string error;
        Check(cyxwiz::ExtensionNodeRegistry::Instance().Register(std::move(registration), error),
              "an extension registers: " + error);
    }
    now += live.debounce + std::chrono::milliseconds(1);
    live.Update(nodes, links, true, now);
    now += live.debounce + std::chrono::milliseconds(1);
    live.Update(nodes, links, true, now);
    Check(live.ParametersCounting(), "a registry change compiles the graph again");
    Finish(live, nodes, links, true, now);
    cyxwiz::ExtensionNodeRegistry::Instance().RemoveByProvider("com.test.live");
    Finish(live, nodes, links, true, now);
    for (int i = 0; i < 3; ++i) {
        now += live.debounce + std::chrono::milliseconds(1);
        live.Update(nodes, links, true, now);
        Finish(live, nodes, links, true, now);
    }
    const auto settled = live.ResultSerial();

    // Invalidate (Python started): compiled again without an edit.
    live.Invalidate();
    now += live.debounce + std::chrono::milliseconds(1);
    live.Update(nodes, links, true, now);
    now += live.debounce + std::chrono::milliseconds(1);
    live.Update(nodes, links, true, now);
    Finish(live, nodes, links, true, now);
    Check(live.ResultSerial() == settled + 1, "Invalidate compiles the graph once more");

    std::cout << "test_live_graph_compile: all checks passed\n";
    return 0;
}
