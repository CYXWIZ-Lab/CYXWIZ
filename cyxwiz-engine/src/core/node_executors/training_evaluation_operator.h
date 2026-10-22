#pragma once

// Cross-Validation and Learning Curves nodes (TOFIX129 D4, decision D4-7):
// the settings of Data Studio > Results > Cross-validation / Learning curve,
// kept in the graph on the data path the way Data Split is. Train trains
// through them unchanged (pin 0 is the input); Train > Cross-validate /
// Learning curve and Results' Run read their settings. Pin 1 is the last
// run's table (one row per training: fold, share, rows, scores), published
// under `results_table` when a run finishes, so a Plot or Export node can use
// it; it has no result before the first run.

#include "pipeline_operator.h"
#include "../cross_validation_runner.h"

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace cyxwiz {

class TrainingEvaluationOperator : public IPipelineOperator {
public:
    explicit TrainingEvaluationOperator(bool learning_curve) : learning_curve_(learning_curve) {}

    std::string GetName() const override { return learning_curve_ ? "LearningCurves" : "CrossValidation"; }
    PipelineBand GetBand() const override { return PipelineBand::DataPrep; }

    bool Configure(const std::map<std::string, std::string>& params, std::string& error) override;
    arrow::Result<std::shared_ptr<arrow::Table>> Apply(const std::shared_ptr<arrow::Table>& input) override;
    std::vector<std::shared_ptr<arrow::Table>> ExtraOutputs() const override { return {results_}; }
    std::string OutcomeSummary() const override { return outcome_; }

    // Big tables stream straight through.
    bool SupportsStreaming() const override { return true; }
    bool PassesInputThrough() const override { return true; }
    arrow::Status BeginStream(const std::shared_ptr<arrow::Schema>&) override { return arrow::Status::OK(); }

    // The node's parameters as the runner's settings, and back. Cross-
    // validation: folds (2 .. 20), stratified (true / false), seed; learning
    // curve: folds, seed, train_sizes ("10%, 25%, 50%, 100%"). Both:
    // results_table, the name the last run's table is published under.
    static bool SettingsFromParameters(bool learning_curve, const std::map<std::string, std::string>& params,
                                       CrossValidationSettings& settings, std::string& results_table, std::string& error);
    static std::map<std::string, std::string> ParametersFromSettings(bool learning_curve, const CrossValidationSettings& settings,
                                                                     const std::string& results_table);
    static const char* DefaultResultsTable(bool learning_curve) {
        return learning_curve ? "learning_curve_results" : "cross_validation_results";
    }

    // How pin 1 finds the published table: the Engine installs a lookup into
    // its data registry (the pipeline core does not depend on it, as with the
    // compiler's dataset hooks); without one, pin 1 stays empty.
    using ResultsLookup = std::function<std::shared_ptr<arrow::Table>(const std::string& name)>;
    static void SetResultsLookup(ResultsLookup lookup);

private:
    bool learning_curve_;
    CrossValidationSettings settings_;
    std::string results_table_;
    std::shared_ptr<arrow::Table> results_;
    std::string outcome_;
};

}  // namespace cyxwiz
