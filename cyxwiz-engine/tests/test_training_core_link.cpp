// Links only cyxwiz-training-core + cyxwiz-backend (no Engine sources, no
// stubs): proves the core is self-contained, as a Server Node needs it
// (TOFIX118 P2). Compiles a minimal graph and constructs the executor.
#include "../src/core/graph_compiler.h"
#include "../src/core/training_executor.h"

#include <cstdlib>
#include <iostream>

int main() {
    gui::MLNode input;
    input.id = 1;
    input.type = gui::NodeType::DataInput;
    input.name = "input";
    gui::MLNode dense;
    dense.id = 2;
    dense.type = gui::NodeType::Dense;
    dense.name = "dense";
    dense.parameters["units"] = "4";

    cyxwiz::GraphCompiler compiler;
    const cyxwiz::TrainingConfiguration config = compiler.Compile({input, dense}, {}, true);
    std::cout << "compiled: valid=" << config.is_valid << " issues=" << config.issues.size() << "\n";

    cyxwiz::TrainingExecutor executor(
        config, [](const cyxwiz::TrainingConfiguration&, int) { return cyxwiz::ResolvedExternalBatchers{}; });
    std::cout << "training core link check passed\n";
    return EXIT_SUCCESS;
}
