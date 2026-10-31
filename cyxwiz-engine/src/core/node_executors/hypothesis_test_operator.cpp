#include "hypothesis_test_operator.h"

#include "../profiler_trace.h"
#include "../session_query_engine.h"
#include "../stats_presentation.h"

#include <arrow/builder.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <filesystem>

namespace cyxwiz {

namespace {

using stats::Question;

struct QuestionName {
    Question question;
    const char* name;
};
constexpr QuestionName kQuestions[] = {{Question::TwoGroups, "two_groups"},
                                       {Question::Paired, "paired"},
                                       {Question::SeveralGroups, "several_groups"},
                                       {Question::OneValue, "one_value"},
                                       {Question::TwoCategories, "two_categories"}};

std::string Param(const std::map<std::string, std::string>& params, const char* key, const char* fallback = "") {
    const auto it = params.find(key);
    return it == params.end() || it->second.empty() ? fallback : it->second;
}

bool Number(const std::string& text, double& out) {
    try {
        size_t used = 0;
        out = std::stod(text, &used);
        return used == text.size() && std::isfinite(out);
    } catch (...) {
        return false;
    }
}

std::string Text(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

}  // namespace

bool HypothesisTestOperator::RequestFromParameters(const std::map<std::string, std::string>& params, stats::TestRequest& r,
                                                   std::string& error) {
    r = stats::TestRequest{};
    const std::string question = Param(params, "question", "two_groups");
    bool known = false;
    for (const auto& q : kQuestions)
        if (question == q.name) {
            r.question = q.question;
            known = true;
        }
    if (!known) {
        error = "HypothesisTest: 'question' must be two_groups, paired, several_groups, one_value or two_categories (got '" + question + "')";
        return false;
    }
    r.values = Param(params, "values");
    r.groups = Param(params, "groups");
    r.second = Param(params, "second");
    r.two_columns = Param(params, "two_columns", "false") == "true";
    const std::string variances = Param(params, "variances", "unequal");
    if (variances != "unequal" && variances != "equal") {
        error = "HypothesisTest: 'variances' must be unequal or equal (got '" + variances + "')";
        return false;
    }
    r.welch = variances == "unequal";
    double alpha = 0.05;
    if (!Number(Param(params, "alpha", "0.05"), alpha) || alpha < 0.001 || alpha > 0.5) {
        error = "HypothesisTest: 'alpha' must be a number from 0.001 to 0.5 (got '" + Param(params, "alpha") + "')";
        return false;
    }
    r.confidence = 1.0 - alpha;
    if (!Number(Param(params, "value", "0"), r.value)) {
        error = "HypothesisTest: 'value' must be a number (got '" + Param(params, "value") + "')";
        return false;
    }
    return true;
}

std::map<std::string, std::string> HypothesisTestOperator::ParametersFromRequest(const stats::TestRequest& r) {
    std::string question = "two_groups";
    for (const auto& q : kQuestions)
        if (q.question == r.question) question = q.name;
    // Only the columns the question uses (the tab keeps every pick for its other questions).
    const bool categories = r.question == Question::TwoCategories;
    const bool grouped = (r.question == Question::TwoGroups && !r.two_columns) || r.question == Question::SeveralGroups;
    const bool second = categories || r.question == Question::Paired || (r.question == Question::TwoGroups && r.two_columns);
    return {{"question", question},
            {"values", categories ? std::string() : r.values},
            {"groups", grouped || categories ? r.groups : std::string()},
            {"second", second ? r.second : std::string()},
            {"two_columns", r.question == Question::TwoGroups && r.two_columns ? "true" : "false"},
            {"variances", r.welch ? "unequal" : "equal"},
            {"alpha", Text(1.0 - r.confidence)},
            {"value", Text(r.value)}};
}

bool HypothesisTestOperator::Configure(const std::map<std::string, std::string>& params, std::string& error) {
    return RequestFromParameters(params, request_, error);
}

arrow::Result<std::shared_ptr<arrow::Table>> HypothesisTestOperator::Apply(const std::shared_ptr<arrow::Table>& input) {
    CYXWIZ_PROFILE_ZONE("CyxWiz HypothesisTest");
    if (!input) return arrow::Status::Invalid("HypothesisTest: input is null");
    // The node's input on a private query engine, so the test reads its
    // columns exactly as Explore > Tests does.
    static std::atomic<uint64_t> next{0};
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "cyxwiz_hypothesis_test" /
                                      std::to_string(++next);
    stats::TestAnswer a;
    {
        SessionQueryEngine engine(dir.string());
        std::string error;
        if (!engine.AttachArrow("input", 1, input, &error)) return arrow::Status::Invalid("HypothesisTest: " + error);
        const QueryRunner run = [&](const QueryRequest& q) { return engine.Run(q); };
        a = stats::RunTest("input", request_, run, {});
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    if (!a.ok()) return arrow::Status::Invalid("HypothesisTest: " + a.error);

    int64_t used = 0;
    for (const auto& g : a.groups) used += g.n;
    if (request_.question == Question::Paired) used = a.groups.empty() ? 0 : a.groups.front().n;
    if (request_.question == Question::TwoCategories)
        for (const auto& row : a.counts)
            for (const double c : row) used += static_cast<int64_t>(c);
    const stats_view::Sentence sentence = stats_view::Answer(a);

    arrow::StringBuilder question, test, effect_name, answer;
    arrow::DoubleBuilder statistic, df, df2, p, estimate, ci_low, ci_high, confidence, effect;
    arrow::Int64Builder n, left_out;
    arrow::BooleanBuilder significant;
    ARROW_RETURN_NOT_OK(question.Append(stats_view::QuestionLabel(request_.question)));
    ARROW_RETURN_NOT_OK(test.Append(stats_view::TestName(request_)));
    ARROW_RETURN_NOT_OK(statistic.Append(a.main.statistic));
    ARROW_RETURN_NOT_OK(df.Append(a.main.df));
    if (std::isfinite(a.main.df2)) ARROW_RETURN_NOT_OK(df2.Append(a.main.df2));
    else ARROW_RETURN_NOT_OK(df2.AppendNull());
    ARROW_RETURN_NOT_OK(p.Append(a.main.p));
    const bool has_estimate = std::isfinite(a.main.estimate);
    for (auto* b : {&estimate, &ci_low, &ci_high}) {
        if (!has_estimate) ARROW_RETURN_NOT_OK(b->AppendNull());
    }
    if (has_estimate) {
        ARROW_RETURN_NOT_OK(estimate.Append(a.main.estimate));
        ARROW_RETURN_NOT_OK(ci_low.Append(a.main.ci_low));
        ARROW_RETURN_NOT_OK(ci_high.Append(a.main.ci_high));
    }
    ARROW_RETURN_NOT_OK(confidence.Append(request_.confidence));
    ARROW_RETURN_NOT_OK(effect.Append(a.main.effect));
    ARROW_RETURN_NOT_OK(effect_name.Append(stats_view::EffectName(request_.question)));
    ARROW_RETURN_NOT_OK(n.Append(used));
    ARROW_RETURN_NOT_OK(left_out.Append(a.left_out));
    ARROW_RETURN_NOT_OK(significant.Append(sentence.significant));
    ARROW_RETURN_NOT_OK(answer.Append(sentence.text));

    std::vector<std::shared_ptr<arrow::Array>> columns(16);
    ARROW_RETURN_NOT_OK(question.Finish(&columns[0]));
    ARROW_RETURN_NOT_OK(test.Finish(&columns[1]));
    ARROW_RETURN_NOT_OK(statistic.Finish(&columns[2]));
    ARROW_RETURN_NOT_OK(df.Finish(&columns[3]));
    ARROW_RETURN_NOT_OK(df2.Finish(&columns[4]));
    ARROW_RETURN_NOT_OK(p.Finish(&columns[5]));
    ARROW_RETURN_NOT_OK(estimate.Finish(&columns[6]));
    ARROW_RETURN_NOT_OK(ci_low.Finish(&columns[7]));
    ARROW_RETURN_NOT_OK(ci_high.Finish(&columns[8]));
    ARROW_RETURN_NOT_OK(confidence.Finish(&columns[9]));
    ARROW_RETURN_NOT_OK(effect.Finish(&columns[10]));
    ARROW_RETURN_NOT_OK(effect_name.Finish(&columns[11]));
    ARROW_RETURN_NOT_OK(n.Finish(&columns[12]));
    ARROW_RETURN_NOT_OK(left_out.Finish(&columns[13]));
    ARROW_RETURN_NOT_OK(significant.Finish(&columns[14]));
    ARROW_RETURN_NOT_OK(answer.Finish(&columns[15]));
    const auto schema = arrow::schema({arrow::field("question", arrow::utf8()), arrow::field("test", arrow::utf8()),
                                       arrow::field("statistic", arrow::float64()), arrow::field("df", arrow::float64()),
                                       arrow::field("df2", arrow::float64()), arrow::field("p", arrow::float64()),
                                       arrow::field("estimate", arrow::float64()), arrow::field("ci_low", arrow::float64()),
                                       arrow::field("ci_high", arrow::float64()), arrow::field("confidence", arrow::float64()),
                                       arrow::field("effect", arrow::float64()), arrow::field("effect_name", arrow::utf8()),
                                       arrow::field("n", arrow::int64()), arrow::field("left_out", arrow::int64()),
                                       arrow::field("significant", arrow::boolean()), arrow::field("answer", arrow::utf8())});
    outcome_ = stats_view::TestName(request_) + ": p " + stats_view::P(a.main.p) + (sentence.significant ? " (significant)" : "");
    spdlog::info("HypothesisTest: {}", outcome_);
    return arrow::Table::Make(schema, columns);
}

}  // namespace cyxwiz
