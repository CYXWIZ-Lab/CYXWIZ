#pragma once

// Recompiles the graph on the canvas in the background shortly after each
// edit (TOFIX123), so the Properties panel and canvas colours always show
// the compile of the current graph. One worker at a time; a newer edit
// waits for it and compiles again. The worker also builds the model once to
// count learnable parameters per layer (skipped while training runs).
//
// Not an AsyncTaskManager task on purpose: a compile per edit would flood
// Task View. Update() is called from the UI thread every frame.

#include "compiled_node_presentation.h"
#include "graph_compiler.h"

#include <chrono>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <vector>

namespace cyxwiz {

class LiveGraphCompile {
public:
    using Clock = std::chrono::steady_clock;

    ~LiveGraphCompile();

    // Call every frame with the canvas graph. Starts a compile once the graph
    // has been unchanged for the debounce delay; collects a finished one.
    void Update(const std::vector<gui::MLNode>& nodes,
                const std::vector<gui::NodeLink>& links,
                bool count_parameters,
                Clock::time_point now = Clock::now());

    LiveCompileState State() const;
    // The latest finished compile (possibly of an older revision while a new
    // one runs); nullptr before the first.
    const TrainingConfiguration* Config() const { return result_ ? &result_->config : nullptr; }
    const std::map<size_t, long long>& LayerParameters() const;
    bool ParametersCounted() const { return result_ && result_->parameters_counted; }
    bool ParametersCounting() const { return pending_.valid(); }
    // Increments each time a compile finishes (for callers that mirror it).
    uint64_t ResultSerial() const { return result_serial_; }

    // Error-level issues of the latest compile on this node.
    bool NodeHasError(int node_id) const;
    // First Error-level message on this node ("" when none).
    std::string NodeErrorMessage(int node_id) const;

    static uint64_t HashGraph(const std::vector<gui::MLNode>& nodes,
                              const std::vector<gui::NodeLink>& links);

    std::chrono::milliseconds debounce{400};

private:
    struct Result {
        uint64_t revision = 0;
        TrainingConfiguration config;
        std::map<size_t, long long> layer_parameters;
        bool parameters_counted = false;
    };

    void Collect();

    uint64_t seen_revision_ = 0;
    bool seen_any_ = false;
    Clock::time_point last_change_{};
    uint64_t requested_revision_ = 0;  // revision the running worker compiles
    std::future<std::shared_ptr<Result>> pending_;
    std::shared_ptr<Result> result_;
    uint64_t result_serial_ = 0;
};

}  // namespace cyxwiz
