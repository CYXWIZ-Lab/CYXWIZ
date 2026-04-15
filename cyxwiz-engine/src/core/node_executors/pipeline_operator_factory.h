#pragma once

#include "pipeline_operator.h"
#include "../../gui/node_editor.h"  // gui::NodeType
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cyxwiz {

/**
 * PipelineOperatorFactory — Category 1 factory.
 *
 * Maps gui::NodeType to a creator that builds a fresh IPipelineOperator
 * instance on demand. Unlike NodeExecutorFactory (Cat-2, which caches one
 * persistent UI executor per type) this factory hands out NEW instances per
 * call: pipeline operators carry per-evaluation state (configured params,
 * computed schemas) and are short-lived during graph execution.
 *
 * Register new operators by calling RegisterCreator from a Phase 4+ node's
 * own translation unit, or extend RegisterDefaults() in the .cpp.
 */
class PipelineOperatorFactory {
public:
    using Creator = std::function<std::unique_ptr<IPipelineOperator>()>;

    static PipelineOperatorFactory& Instance();

    /**
     * Build a new operator for `type`. Returns nullptr if no creator is
     * registered. Caller owns the result.
     */
    std::unique_ptr<IPipelineOperator> Create(gui::NodeType type) const;

    /** Register a creator for a NodeType. Overwrites any previous binding. */
    void RegisterCreator(gui::NodeType type, Creator creator);

    /** True iff a creator is registered for `type`. */
    bool HasOperator(gui::NodeType type) const;

    /** All NodeTypes with a registered creator. */
    std::vector<gui::NodeType> GetSupportedTypes() const;

private:
    PipelineOperatorFactory();

    void RegisterDefaults();

    std::unordered_map<gui::NodeType, Creator> creators_;
};

} // namespace cyxwiz
