#include "training_evaluation_operator.h"

#include "../cross_validation_presentation.h"

#include <cstdlib>
#include <mutex>

namespace cyxwiz {

namespace {

std::mutex g_lookup_mutex;
TrainingEvaluationOperator::ResultsLookup g_lookup;

std::string Param(const std::map<std::string, std::string>& params, const char* key, const char* fallback) {
    const auto it = params.find(key);
    return it == params.end() || it->second.empty() ? fallback : it->second;
}

bool WholeNumber(const std::string& text, long long& out) {
    char* end = nullptr;
    out = std::strtoll(text.c_str(), &end, 10);
    return !text.empty() && end && *end == '\0';
}

}  // namespace

bool TrainingEvaluationOperator::SettingsFromParameters(bool learning_curve, const std::map<std::string, std::string>& params,
                                                        CrossValidationSettings& settings, std::string& results_table,
                                                        std::string& error) {
    const std::string name = learning_curve ? "LearningCurves" : "CrossValidation";
    settings = CrossValidationSettings{};
    long long folds = 0, seed = 0;
    if (!WholeNumber(Param(params, "folds", "5"), folds) || folds < 2 || folds > 20) {
        error = name + ": 'folds' must be a whole number from 2 to 20 (got '" + Param(params, "folds", "") + "')";
        return false;
    }
    if (!WholeNumber(Param(params, "seed", "42"), seed) || seed < 0 || seed > 4294967295LL) {
        error = name + ": 'seed' must be a whole number of 0 or more (got '" + Param(params, "seed", "") + "')";
        return false;
    }
    settings.folds = static_cast<int>(folds);
    settings.seed = static_cast<uint32_t>(seed);
    if (learning_curve) {
        std::string why;
        if (!folds_view::ParseShares(Param(params, "train_sizes", "10%, 25%, 50%, 75%, 100%"), settings.train_shares, why)) {
            error = name + ": 'train_sizes': " + why;
            return false;
        }
    } else {
        const std::string stratified = Param(params, "stratified", "true");
        if (stratified != "true" && stratified != "false") {
            error = name + ": 'stratified' must be true or false (got '" + stratified + "')";
            return false;
        }
        settings.stratified = stratified == "true";
    }
    results_table = Param(params, "results_table", DefaultResultsTable(learning_curve));
    return true;
}

std::map<std::string, std::string> TrainingEvaluationOperator::ParametersFromSettings(bool learning_curve,
                                                                                      const CrossValidationSettings& settings,
                                                                                      const std::string& results_table) {
    std::map<std::string, std::string> params = {{"folds", std::to_string(settings.folds)},
                                                 {"seed", std::to_string(settings.seed)},
                                                 {"results_table", results_table.empty() ? DefaultResultsTable(learning_curve) : results_table}};
    if (learning_curve) params["train_sizes"] = folds_view::SharesText(settings.train_shares);
    else params["stratified"] = settings.stratified ? "true" : "false";
    return params;
}

void TrainingEvaluationOperator::SetResultsLookup(ResultsLookup lookup) {
    std::lock_guard<std::mutex> lock(g_lookup_mutex);
    g_lookup = std::move(lookup);
}

bool TrainingEvaluationOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    return SettingsFromParameters(learning_curve_, params, settings_, results_table_, error);
}

arrow::Result<std::shared_ptr<arrow::Table>> TrainingEvaluationOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    if (!input) return arrow::Status::Invalid(GetName(), ": input is null");
    // Pin 1: the last run's table, when one has been published.
    ResultsLookup lookup;
    {
        std::lock_guard<std::mutex> lock(g_lookup_mutex);
        lookup = g_lookup;
    }
    results_ = lookup ? lookup(results_table_) : nullptr;
    outcome_ = std::to_string(settings_.folds) + " folds" +
               (learning_curve_ ? ", training rows " + folds_view::SharesText(settings_.train_shares)
                                : std::string(settings_.stratified ? ", class shares kept" : "")) +
               (results_ ? "; last run: " + std::to_string(results_->num_rows()) + " trainings in '" + results_table_ + "'"
                         : "; not run yet (Train > " + std::string(learning_curve_ ? "Learning curve" : "Cross-validate") + ")");
    return input;
}

}  // namespace cyxwiz
