#pragma once

#include "pipeline_operator.h"

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

    bool Configure(
        const std::map<std::string, std::string>& /*params*/,
        std::string& /*error*/) override {
        return true;
    }

    arrow::Result<std::shared_ptr<arrow::Table>> Apply(
        const std::shared_ptr<arrow::Table>& input) override {
        if (!input) {
            return arrow::Status::Invalid("IdentityOperator: input table is null");
        }
        return input;
    }
};

} // namespace cyxwiz
