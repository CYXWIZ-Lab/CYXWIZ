#pragma once

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

/**
 * Pipeline band classification — see CLAUDE.md "Pipeline Architecture: Four Bands".
 *
 * - Band1 (DataPrep):     stateless, cacheable, deterministic. Identical
 *                         behavior train/val/test. No phase flag.
 * - Band2 (Partition):    deterministic given seed, cacheable. Assigns rows
 *                         to train/val/test partitions.
 * - Band3 (PhaseAware):   reads training phase, NOT cacheable. e.g. Augmentation.
 *                         Cat-1 operators in Band 3 are rare — most Band 3 logic
 *                         lives in batchers, not pipeline operators.
 */
enum class PipelineBand {
    DataPrep,
    Partition,
    PhaseAware,
};

struct PipelineOperatorProgress {
    int node_id = -1;
    std::string node_name;
    std::string stage;
    std::string message;
    float progress = 0.0f;
    uint64_t estimated_memory_bytes = 0;
    std::string memory_risk_level;
    uint64_t processed_items = 0;
    uint64_t total_items = 0;
};

using PipelineOperatorProgressCallback =
    std::function<void(const PipelineOperatorProgress&)>;

/**
 * IPipelineOperator — Category 1 base interface (pipeline operations).
 *
 * Real operations that transform Arrow table data flowing through a graph,
 * executed automatically during training. The opposite of Category 2
 * introspection executors (INodeExecutor) which are manual click-Run UI tools.
 *
 * Contract:
 *   - Input  → arrow::Table from upstream node
 *   - Output → arrow::Table (or new schema) for downstream consumer
 *   - No UI rendering, no code generation, no user interaction
 *   - Configure() reads parameters from the node's params map; same map the
 *     node editor and graph compiler use, so JSON round-trip is automatic.
 *
 * Subclasses implement a single operation per CLAUDE.md's
 * single-responsibility node rule. Multi-step transforms compose at the
 * graph level, not inside a single operator.
 *
 * NOT for Category 2 (introspection / KMeans-style click-Run tools) — use
 * INodeExecutor for those.
 */
class IPipelineOperator {
public:
    virtual ~IPipelineOperator() = default;

    // === Identity ===
    virtual std::string GetName() const = 0;
    virtual PipelineBand GetBand() const = 0;

    // === Configuration ===
    /**
     * Read parameters from the node's param map (the same std::map used by
     * gui::MLNode::parameters and JSON serialization). Populate `error` on
     * failure and return false. Called once before Apply().
     */
    virtual bool Configure(
        const std::map<std::string, std::string>& params,
        std::string& error) = 0;

    // === Execution ===
    /**
     * Transform an input Arrow table into an output Arrow table.
     *
     * Must be deterministic for Band 1 / Band 2. Must not mutate `input`.
     * Errors propagate via arrow::Result; callers handle them with
     * ARROW_ASSIGN_OR_RAISE / ARROW_RETURN_NOT_OK / explicit checks.
     */
    virtual arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) = 0;

    virtual void SetProgressCallback(PipelineOperatorProgressCallback callback) {
        (void)callback;
    }

    /**
     * Optional schema-only inference for compile-gate validation. Default
     * returns the input schema unchanged (correct for any operator that
     * preserves schema). Override when the op adds, drops, or retypes columns.
     */
    virtual arrow::Result<std::shared_ptr<arrow::Schema>> InferOutputSchema(
        const std::shared_ptr<arrow::Schema>& input_schema) {
        return input_schema;
    }

    /**
     * Whether the operator's output is safely cacheable on disk between
     * training runs. True for stateless Band 1 / deterministic Band 2 ops.
     * False for Band 3 (phase-aware) ops.
     */
    virtual bool IsCacheable() const {
        return GetBand() != PipelineBand::PhaseAware;
    }
};

} // namespace cyxwiz
