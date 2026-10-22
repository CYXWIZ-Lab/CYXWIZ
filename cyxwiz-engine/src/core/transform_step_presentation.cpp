#include "transform_step_presentation.h"

#include "explore_presentation.h"

#include <cyxwiz/data_transform.h>

#include <arrow/compute/api.h>

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace cyxwiz::transform_step {

using gui::NodeType;

namespace {

std::string Param(const std::map<std::string, std::string>& params, const char* key, const char* fallback = "") {
    const auto it = params.find(key);
    return it == params.end() || it->second.empty() ? fallback : it->second;
}

std::vector<std::string> SplitList(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream in(text);
    std::string item;
    while (std::getline(in, item, ',')) {
        const auto b = item.find_first_not_of(" \t");
        const auto e = item.find_last_not_of(" \t");
        if (b != std::string::npos) out.push_back(item.substr(b, e - b + 1));
    }
    return out;
}

bool IsNumber(const arrow::DataType& type) {
    const auto id = type.id();
    return arrow::is_integer(id) || id == arrow::Type::FLOAT || id == arrow::Type::DOUBLE;
}

std::vector<double> Present(const std::vector<double>& values) {
    std::vector<double> out;
    out.reserve(values.size());
    std::copy_if(values.begin(), values.end(), std::back_inserter(out), [](double v) { return std::isfinite(v); });
    return out;
}

double NormalityP(const std::vector<double>& present) {
    const NormalityTestResult test = DataTransform::DAgostinoPearsonTest(present);
    return test.computed ? test.p_value : NAN;
}

}  // namespace

Kind KindOf(NodeType type) {
    switch (type) {
        case NodeType::StandardScaler:
        case NodeType::MinMaxScaler:
        case NodeType::RobustScaler:
        case NodeType::MaxAbsScaler:
        case NodeType::QuantileTransform:
            return Kind::Scale;
        case NodeType::PowerTransform:
            return Kind::Power;
        case NodeType::LogTransform:
            return Kind::Log;
        default:
            return Kind::None;
    }
}

const std::vector<Method>& ScaleMethods() {
    static const std::vector<Method> methods = {
        {NodeType::StandardScaler, "Z-score", "(x - mean) / std", "Z-score: most models; keeps outliers' distance."},
        {NodeType::MinMaxScaler, "Min-Max", "(x - min) / (max - min)",
         "Min-Max: inputs that must stay in a fixed range (images, some networks)."},
        {NodeType::RobustScaler, "Robust", "(x - median) / IQR", "Robust: columns with outliers."},
        {NodeType::MaxAbsScaler, "Max-Abs", "x / max |x|, to -1 .. 1", "Max-Abs: sparse data (keeps zeros)."},
        {NodeType::QuantileTransform, "Quantile", "through its quantiles", "Quantile: very skewed columns; changes the shape."},
    };
    return methods;
}

const Method* MethodOf(NodeType type) {
    for (const auto& m : ScaleMethods())
        if (m.type == type) return &m;
    return nullptr;
}

std::string Formula(NodeType type, const std::map<std::string, std::string>& params) {
    switch (type) {
        case NodeType::MinMaxScaler:
            return "(x - min) / (max - min), to " + Param(params, "min", "0") + " .. " + Param(params, "max", "1");
        case NodeType::QuantileTransform:
            return std::string("through its quantiles to a ") +
                   (Param(params, "output_distribution", "uniform") == "normal" ? "normal shape" : "uniform 0 .. 1");
        case NodeType::PowerTransform: {
            const std::string family = Param(params, "family", "yeo-johnson") == "box-cox" ? "Box-Cox" : "Yeo-Johnson";
            if (Param(params, "lambda_mode", "auto") == "fixed") return family + ", lambda " + Param(params, "lambda", "0");
            return family + ", lambda found per column";
        }
        case NodeType::LogTransform: {
            const std::string base = Param(params, "base", "natural");
            return std::string(Param(params, "use_log1p", "true") == "false" ? "log(x)" : "log(1 + x)") + ", base " +
                   (base == "log10" ? "10" : base == "log2" ? "2" : "e");
        }
        default:
            if (const Method* m = MethodOf(type)) return m->formula;
            return {};
    }
}

std::map<std::string, std::string> CarriedParameters(const std::map<std::string, std::string>& from) {
    std::map<std::string, std::string> out;
    for (const char* key : {"columns", "label_col", "exclude_columns", "save_state", "state_overwrite"})
        if (const auto it = from.find(key); it != from.end()) out[key] = it->second;
    out["operation_mode"] = "fit_transform";  // the saved values were another method's
    return out;
}

std::vector<std::string> NumberColumns(const arrow::Schema& schema) {
    std::vector<std::string> out;
    for (const auto& field : schema.fields())
        if (IsNumber(*field->type())) out.push_back(field->name());
    return out;
}

std::vector<std::string> StepColumns(const std::map<std::string, std::string>& params, const arrow::Schema& schema) {
    std::vector<std::string> chosen = SplitList(Param(params, "columns"));
    if (!chosen.empty()) return chosen;
    const std::string label = Param(params, "label_col");
    const std::vector<std::string> excluded = SplitList(Param(params, "exclude_columns"));
    for (const auto& name : NumberColumns(schema))
        if (name != label && std::find(excluded.begin(), excluded.end(), name) == excluded.end()) chosen.push_back(name);
    return chosen;
}

std::vector<double> ColumnValues(const arrow::Table& table, const std::string& name) {
    const auto column = table.GetColumnByName(name);
    if (!column || !IsNumber(*column->type())) return {};
    std::vector<double> out;
    out.reserve(static_cast<size_t>(column->length()));
    for (const auto& chunk : column->chunks()) {
        auto cast = arrow::compute::Cast(*chunk, arrow::float64());
        if (!cast.ok()) return {};
        const auto& doubles = static_cast<const arrow::DoubleArray&>(**cast);
        for (int64_t i = 0; i < doubles.length(); ++i) out.push_back(doubles.IsNull(i) ? NAN : doubles.Value(i));
    }
    return out;
}

Stats Describe(const std::vector<double>& values, int bins) {
    Stats s;
    const std::vector<double> present = Present(values);
    s.count = static_cast<int64_t>(present.size());
    s.missing = static_cast<int64_t>(values.size()) - s.count;
    if (present.empty()) return s;
    const ColumnStats stats = DataTransform::ComputeColumnStats(present);
    if (!stats.error_message.empty()) return s;
    s.min = stats.min;
    s.max = stats.max;
    s.mean = stats.mean;
    s.std = stats.std_dev;
    s.skew = stats.skewness;
    s.bins = DataTransform::Histogram(present, bins, s.min, s.max);
    s.beyond_z3 = std::count_if(present.begin(), present.end(), [](double v) { return std::fabs(v) > 3.0; });
    s.ok = true;
    return s;
}

StepFigures ComputeFigures(NodeType type, const std::map<std::string, std::string>& params, const arrow::Table& input,
                           const arrow::Table& output, const std::map<std::string, double>& saved_lambdas) {
    StepFigures f;
    f.rows_in = input.num_rows();
    f.rows_out = output.num_rows();
    const std::vector<std::string> columns = StepColumns(params, *input.schema());
    if (columns.empty()) {
        f.error = "The step's input has no number columns to transform.";
        return f;
    }
    const bool power = KindOf(type) == Kind::Power;
    const bool box_cox = Param(params, "family", "yeo-johnson") == "box-cox";
    const bool auto_lambda = Param(params, "lambda_mode", "auto") != "fixed";
    const bool saved = Param(params, "operation_mode", "fit_transform") == "transform_only";
    for (const auto& name : columns) {
        ColumnFigures c;
        c.name = name;
        const std::vector<double> before = ColumnValues(input, name);
        if (before.empty() && input.num_rows() > 0) continue;  // not a number column of the input
        c.before = Describe(before);
        c.after = Describe(ColumnValues(output, name));
        if (power) {
            c.has_power = true;
            PowerFigures& p = c.power;
            p.box_cox = box_cox;
            p.auto_lambda = auto_lambda;
            const std::vector<double> present = Present(before);
            if (box_cox && !DataTransform::CanApplyBoxCox(present)) {
                p.note = "'" + name + "' has values of 0 or less; Box-Cox needs values above 0. Use Yeo-Johnson.";
            } else if (!present.empty()) {
                const BoxCoxLambdaResult curve =
                    box_cox ? DataTransform::FindOptimalLambda(present) : DataTransform::FindOptimalYeoJohnsonLambda(present);
                if (curve.success) {
                    p.lambdas = curve.lambdas_tested;
                    p.log_likelihoods = curve.log_likelihoods;
                    p.best = curve.optimal_lambda;
                }
                if (!auto_lambda) {
                    try {
                        p.lambda = std::stod(Param(params, "lambda", "0"));
                    } catch (...) {
                        p.lambda = NAN;
                    }
                } else if (saved) {
                    const auto it = saved_lambdas.find(name);
                    p.lambda = it == saved_lambdas.end() ? NAN : it->second;
                    if (it == saved_lambdas.end()) p.note = "Uses saved values; their lambda for '" + name + "' could not be read.";
                } else {
                    p.lambda = p.best;
                }
            }
            p.p_before = NormalityP(present);
            p.p_after = NormalityP(Present(ColumnValues(output, name)));
        }
        f.columns.push_back(std::move(c));
    }
    if (f.columns.empty()) f.error = "None of the step's columns is a number column of its input.";
    return f;
}

CompareFigures CompareMethods(const std::string& column, const std::vector<double>& values) {
    CompareFigures f;
    f.column = column;
    f.original = Describe(values);
    if (!f.original.ok) {
        f.error = "'" + column + "' has no values to compare on.";
        return f;
    }
    const std::vector<std::vector<double>> data{Present(values)};
    for (const auto& m : ScaleMethods()) {
        TransformResult r;
        switch (m.type) {
            case NodeType::StandardScaler: r = DataTransform::Standardize(data); break;
            case NodeType::MinMaxScaler: r = DataTransform::Normalize(data); break;
            case NodeType::RobustScaler: r = DataTransform::RobustScale(data); break;
            case NodeType::MaxAbsScaler: r = DataTransform::MaxAbsScale(data); break;
            default: r = DataTransform::QuantileTransform(data, "normal", 1000); break;
        }
        f.methods.push_back(r.success && !r.transformed_data.empty() ? Describe(r.transformed_data[0]) : Stats{});
    }
    return f;
}

std::string Value(double v) {
    // A mean of 0 computed as -3e-10 reads "0", not "-0".
    std::string text = explore_view::Figure(v);
    if (!text.empty() && text[0] == '-' && text.find_first_not_of("-0.") == std::string::npos) text.erase(0, 1);
    return text;
}

std::string Range(const Stats& s) { return s.ok ? Value(s.min) + " .. " + Value(s.max) : "-"; }

std::string PValue(double p) {
    if (!std::isfinite(p)) return "-";
    char text[32];
    std::snprintf(text, sizeof(text), "%.2g", p);
    return text;
}

std::string RunLine(int64_t rows_in, int64_t rows_out) {
    return "Last run passed \xC2\xB7 " + explore_view::Number(static_cast<size_t>(std::max<int64_t>(0, rows_in))) +
           " rows in, " + explore_view::Number(static_cast<size_t>(std::max<int64_t>(0, rows_out))) + " out";
}

std::string PowerSentence(const ColumnFigures& c) {
    if (!c.has_power) return {};
    if (!c.power.note.empty()) return c.power.note;
    if (!c.before.ok || !c.after.ok) return {};
    char skew[96];
    std::snprintf(skew, sizeof(skew), "Skewness %.2f -> %.2f", c.before.skew, c.after.skew);
    if (!std::isfinite(c.power.p_after)) return std::string(skew) + " (the normality test needs at least 20 values).";
    if (c.power.p_after < 0.05)
        return std::string(skew) + ", but the values still do not look normal (p " + PValue(c.power.p_after) +
               "): no single power transform makes them normal; the column may mix groups (more than one hump).";
    return std::string(skew) + "; the values now look normal (p " + PValue(c.power.p_after) + ").";
}

std::string BoxCoxNote(const StepFigures& f) {
    std::vector<std::string> below;
    for (const auto& c : f.columns)
        if (c.before.ok && !(c.before.min > 0.0)) below.push_back(c.name);
    if (below.empty())
        return f.columns.size() == 1 ? "Box-Cox needs values above 0; the column's are."
                                     : "Box-Cox needs values above 0; all " + std::to_string(f.columns.size()) + " are.";
    std::string names;
    for (size_t i = 0; i < below.size(); ++i) names += (i ? ", " : "") + below[i];
    return "Box-Cox needs values above 0; " + names + (below.size() == 1 ? " has" : " have") +
           " 0 or less: use Yeo-Johnson.";
}

std::string BeyondZ3Line(const StepFigures& f) {
    int64_t total = 0;
    std::string parts;
    for (const auto& c : f.columns) {
        if (!c.after.ok || c.after.beyond_z3 == 0) continue;
        total += c.after.beyond_z3;
        parts += (parts.empty() ? "" : ", ") + c.name + " " + explore_view::Number(static_cast<size_t>(c.after.beyond_z3));
    }
    if (total == 0) return "Values beyond |z| 3: none in these columns.";
    return "Values beyond |z| 3: " + explore_view::Number(static_cast<size_t>(total)) + " (" + parts + ").";
}

}  // namespace cyxwiz::transform_step
