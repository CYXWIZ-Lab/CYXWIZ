#pragma once

#include "../profiler_trace.h"
#include "pipeline_operator.h"
#include <utility>

namespace cyxwiz {

/**
 * IdentityOperator — Cat-1 worked example.
 *
 * Passes its input Arrow table through unchanged. Exists to prove the
 * IPipelineOperator interface compiles, round-trips Arrow tables, and
 * registers cleanly via PipelineOperatorFactory. Phase 4 nodes (LogTransform,
 * Differencing, TimeSeriesWindow, ...) follow the same shape with real
 * transform logic in Apply().
 *
 * Bound to gui::NodeType::Identity, which already exists in the enum as a
 * generic passthrough utility node, so this example needs no enum or loader
 * changes.
 */
class IdentityOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "Identity"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override {
        progress_callback_ = std::move(callback);
    }

    bool Configure(
        const std::map<std::string, std::string>& /*params*/,
        std::string& /*error*/) override {
        return true;
    }

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override {
        CYXWIZ_PROFILE_ZONE("CyxWiz Identity Materializer");
        if (!input) {
            return arrow::Status::Invalid("IdentityOperator: input table is null");
        }
        if (progress_callback_) {
            PipelineOperatorProgress event;
            event.stage = "complete";
            event.message = "Identity passthrough complete";
            event.progress = 1.0f;
            event.processed_items = static_cast<uint64_t>(input->num_rows());
            event.total_items = static_cast<uint64_t>(input->num_rows());
            progress_callback_(event);
        }
        return input;
    }

private:
    PipelineOperatorProgressCallback progress_callback_;
};

} // namespace cyxwiz
