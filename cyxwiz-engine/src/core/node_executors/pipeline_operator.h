#pragma once

#include "../materialization_memory_types.h"
#include "../process_memory_snapshot.h"

#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/type_fwd.h>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

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
    std::string status = "running";
    float progress = 0.0f;
    uint64_t estimated_memory_bytes = 0;
    uint64_t available_memory_bytes = 0;
    uint64_t safe_memory_budget_bytes = 0;
    std::string memory_risk_level;
    bool process_memory_detected = false;
    uint64_t process_resident_memory_bytes = 0;
    uint64_t process_private_memory_bytes = 0;
    uint64_t process_resident_growth_bytes = 0;
    std::string process_private_memory_name;
    std::string process_memory_source;
    uint64_t processed_items = 0;
    uint64_t total_items = 0;
};

using PipelineOperatorProgressCallback =
    std::function<void(const PipelineOperatorProgress&)>;

using PipelineOperatorCancellationQuery = std::function<bool()>;
using ProcessMemorySnapshotQuery = std::function<ProcessMemorySnapshot()>;

struct PipelineOperatorCacheDependency {
    std::string role;
    std::string path;
};

struct PipelineOperatorExecutionContext {
    MaterializationMemoryContext memory;
    PipelineOperatorCancellationQuery cancellation_requested;
    ProcessMemorySnapshotQuery process_memory_snapshot;
    // Pre-start planning uses the normal operator estimator and stops as soon
    // as that estimator publishes its first memory decision. The materializer
    // owns the control flow; operators never need a UI-specific branch.
    bool stop_after_memory_preflight = false;

    bool IsCancellationRequested() const {
        return cancellation_requested && cancellation_requested();
    }

    ProcessMemorySnapshot CaptureProcessMemory() const {
        return process_memory_snapshot
            ? process_memory_snapshot()
            : DetectProcessMemorySnapshot();
    }
};

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

    void SetMaterializationMemoryContext(MaterializationMemoryContext context) {
        execution_context_.memory = std::move(context);
    }

    void SetExecutionContext(PipelineOperatorExecutionContext context) {
        execution_context_ = std::move(context);
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

    /**
     * Validate and report external inputs that affect a persistent cache hit.
     * Implementations must fail closed when a required artifact is missing,
     * malformed, or incompatible with the configured operator. The
     * materializer owns path normalization and content fingerprinting.
     */
    virtual bool CollectCacheDependencies(
        std::vector<PipelineOperatorCacheDependency>& dependencies,
        std::string& error) const {
        (void)dependencies;
        (void)error;
        return true;
    }

protected:
    const MaterializationMemoryContext& GetMaterializationMemoryContext() const {
        return execution_context_.memory;
    }

    const PipelineOperatorCancellationQuery& GetCancellationQuery() const {
        return execution_context_.cancellation_requested;
    }

    bool IsCancellationRequested() const {
        return execution_context_.IsCancellationRequested();
    }

    arrow::Status CheckCancellation(const std::string& operation) const {
        if (IsCancellationRequested()) {
            return arrow::Status::Cancelled(
                operation + ": materialization cancelled");
        }
        return arrow::Status::OK();
    }

private:
    PipelineOperatorExecutionContext execution_context_;
};

} // namespace cyxwiz
