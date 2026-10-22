#pragma once

#include "pipeline_operator.h"
#include "../stats_explore.h"

#include <utility>

namespace cyxwiz {

/**
 * HypothesisTestOperator: the Hypothesis Test node (TOFIX129 D3; until then a
 * template with no executor). The same tests as Data Studio > Explore > Tests
 * (core/stats_explore, cyxwiz::StatisticalTests on the device), run on the
 * node's input: its output is one row with the test, statistic, df, p, the
 * estimate and its CI, the effect size, the rows used and left out, whether
 * it is significant, and the answer in words.
 *
 * Params: question (two_groups / paired / several_groups / one_value /
 * two_categories), values, groups, second, two_columns (bool), variances
 * (unequal / equal), alpha (0.001 .. 0.5), value (one value).
 */
class HypothesisTestOperator : public IPipelineOperator {
public:
    std::string GetName() const override { return "HypothesisTest"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }
    void SetProgressCallback(PipelineOperatorProgressCallback callback) override { progress_callback_ = std::move(callback); }
    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    std::string OutcomeSummary() const override { return outcome_; }

    // The request the parameters describe (Configure), and back (Keep as a node).
    static bool RequestFromParameters(const std::map<std::string, std::string>& params, stats::TestRequest& request,
                                      std::string& error);
    static std::map<std::string, std::string> ParametersFromRequest(const stats::TestRequest& request);

private:
    stats::TestRequest request_;
    std::string outcome_;
    PipelineOperatorProgressCallback progress_callback_;
};

}  // namespace cyxwiz
