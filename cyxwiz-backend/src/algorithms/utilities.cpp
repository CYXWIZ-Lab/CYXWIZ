#include <cyxwiz/utilities.h>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <regex>
#include <cctype>
#include <nlohmann/json.hpp>

// OpenSSL headers if available
#ifdef CYXWIZ_HAS_OPENSSL
#include <openssl/md5.h>
#include <openssl/sha.h>
#endif

namespace cyxwiz {

// ============================================================================
// Constants
// ============================================================================

static const double PI = 3.14159265358979323846;
static const double E = 2.71828182845904523536;
static const double PHI = 1.61803398874989484820;  // Golden ratio

// ============================================================================
// Unit Conversion Tables
// ============================================================================

// Structure for unit info
struct UnitInfo {
    std::string category;
    double to_base;      // Multiply by this to convert to base unit
    std::string base_unit;
};

// Unit conversion table (unit name -> conversion info)
static std::map<std::string, UnitInfo> g_unit_table = {
    // Length (base: meter)
    {"meter", {"length", 1.0, "meter"}},
    {"kilometer", {"length", 1000.0, "meter"}},
    {"centimeter", {"length", 0.01, "meter"}},
    {"millimeter", {"length", 0.001, "meter"}},
    {"mile", {"length", 1609.344, "meter"}},
    {"yard", {"length", 0.9144, "meter"}},
    {"foot", {"length", 0.3048, "meter"}},
    {"inch", {"length", 0.0254, "meter"}},
    {"nautical_mile", {"length", 1852.0, "meter"}},

    // Mass (base: kilogram)
    {"kilogram", {"mass", 1.0, "kilogram"}},
    {"gram", {"mass", 0.001, "kilogram"}},
    {"milligram", {"mass", 0.000001, "kilogram"}},
    {"pound", {"mass", 0.45359237, "kilogram"}},
    {"ounce", {"mass", 0.028349523125, "kilogram"}},
    {"ton", {"mass", 1000.0, "kilogram"}},
    {"stone", {"mass", 6.35029318, "kilogram"}},

    // Time (base: second)
    {"second", {"time", 1.0, "second"}},
    {"minute", {"time", 60.0, "second"}},
    {"hour", {"time", 3600.0, "second"}},
    {"day", {"time", 86400.0, "second"}},
    {"week", {"time", 604800.0, "second"}},
    {"year", {"time", 31536000.0, "second"}},
    {"millisecond", {"time", 0.001, "second"}},
    {"microsecond", {"time", 0.000001, "second"}},

    // Data (base: byte)
    {"byte", {"data", 1.0, "byte"}},
    {"kilobyte", {"data", 1024.0, "byte"}},
    {"megabyte", {"data", 1048576.0, "byte"}},
    {"gigabyte", {"data", 1073741824.0, "byte"}},
    {"terabyte", {"data", 1099511627776.0, "byte"}},
    {"petabyte", {"data", 1125899906842624.0, "byte"}},
    {"bit", {"data", 0.125, "byte"}},
    {"kilobit", {"data", 128.0, "byte"}},
    {"megabit", {"data", 131072.0, "byte"}},
    {"gigabit", {"data", 134217728.0, "byte"}},

    // Area (base: square meter)
    {"square_meter", {"area", 1.0, "square_meter"}},
    {"square_kilometer", {"area", 1000000.0, "square_meter"}},
    {"hectare", {"area", 10000.0, "square_meter"}},
    {"acre", {"area", 4046.8564224, "square_meter"}},
    {"square_foot", {"area", 0.09290304, "square_meter"}},
    {"square_inch", {"area", 0.00064516, "square_meter"}},

    // Volume (base: liter)
    {"liter", {"volume", 1.0, "liter"}},
    {"milliliter", {"volume", 0.001, "liter"}},
    {"gallon", {"volume", 3.785411784, "liter"}},
    {"quart", {"volume", 0.946352946, "liter"}},
    {"pint", {"volume", 0.473176473, "liter"}},
    {"cup", {"volume", 0.2365882365, "liter"}},
    {"fluid_ounce", {"volume", 0.0295735295625, "liter"}},
    {"cubic_meter", {"volume", 1000.0, "liter"}},
    {"cubic_centimeter", {"volume", 0.001, "liter"}},

    // Speed (base: meter per second)
    {"meter_per_second", {"speed", 1.0, "meter_per_second"}},
    {"kilometer_per_hour", {"speed", 0.277777778, "meter_per_second"}},
    {"mile_per_hour", {"speed", 0.44704, "meter_per_second"}},
    {"knot", {"speed", 0.514444444, "meter_per_second"}},
    {"foot_per_second", {"speed", 0.3048, "meter_per_second"}},
};

// Temperature requires special handling (not linear conversion)
static double ConvertTemperature(double value, const std::string& from, const std::string& to) {
    // Convert to Celsius first
    double celsius;
    if (from == "celsius") celsius = value;
    else if (from == "fahrenheit") celsius = (value - 32.0) * 5.0 / 9.0;
    else if (from == "kelvin") celsius = value - 273.15;
    else return value;

    // Convert from Celsius to target
    if (to == "celsius") return celsius;
    else if (to == "fahrenheit") return celsius * 9.0 / 5.0 + 32.0;
    else if (to == "kelvin") return celsius + 273.15;
    return celsius;
}

// ============================================================================
// Calculator Implementation
// ============================================================================

void Utilities::SkipWhitespace(ParserState& state) {
    while (state.pos < state.expression.size() && std::isspace(state.expression[state.pos])) {
        state.pos++;
    }
}

char Utilities::Peek(ParserState& state) {
    SkipWhitespace(state);
    if (state.pos >= state.expression.size()) return '\0';
    return state.expression[state.pos];
}

char Utilities::Get(ParserState& state) {
    SkipWhitespace(state);
    if (state.pos >= state.expression.size()) return '\0';
    return state.expression[state.pos++];
}

bool Utilities::Match(ParserState& state, char c) {
    if (Peek(state) == c) {
        Get(state);
        return true;
    }
    return false;
}

double Utilities::ParseNumber(ParserState& state) {
    SkipWhitespace(state);
    std::string num;
    bool has_dot = false;
    bool has_exp = false;

    while (state.pos < state.expression.size()) {
        char c = state.expression[state.pos];
        if (std::isdigit(c)) {
            num += c;
            state.pos++;
        } else if (c == '.' && !has_dot) {
            num += c;
            has_dot = true;
            state.pos++;
        } else if ((c == 'e' || c == 'E') && !has_exp) {
            num += c;
            has_exp = true;
            state.pos++;
            if (state.pos < state.expression.size() &&
                (state.expression[state.pos] == '+' || state.expression[state.pos] == '-')) {
                num += state.expression[state.pos++];
            }
        } else {
            break;
        }
    }

    if (num.empty()) {
        state.error = "Expected number";
        return 0.0;
    }

    return std::stod(num);
}

double Utilities::ParseFunction(ParserState& state, const std::string& name) {
    if (!Match(state, '(')) {
        state.error = "Expected '(' after function name";
        return 0.0;
    }

    double arg1 = ParseExpression(state);
    double arg2 = 0.0;
    bool has_arg2 = false;

    if (Match(state, ',')) {
        arg2 = ParseExpression(state);
        has_arg2 = true;
    }

    if (!Match(state, ')')) {
        state.error = "Expected ')'";
        return 0.0;
    }

    // Angle conversion for trig functions
    double angle_factor = (state.angle_mode == "degrees") ? (PI / 180.0) : 1.0;

    // Single argument functions
    if (name == "sin") return std::sin(arg1 * angle_factor);
    if (name == "cos") return std::cos(arg1 * angle_factor);
    if (name == "tan") return std::tan(arg1 * angle_factor);
    if (name == "asin") return std::asin(arg1) / angle_factor;
    if (name == "acos") return std::acos(arg1) / angle_factor;
    if (name == "atan") return std::atan(arg1) / angle_factor;
    if (name == "sinh") return std::sinh(arg1);
    if (name == "cosh") return std::cosh(arg1);
    if (name == "tanh") return std::tanh(arg1);
    if (name == "sqrt") return std::sqrt(arg1);
    if (name == "cbrt") return std::cbrt(arg1);
    if (name == "exp") return std::exp(arg1);
    if (name == "log" || name == "ln") return std::log(arg1);
    if (name == "log10") return std::log10(arg1);
    if (name == "log2") return std::log2(arg1);
    if (name == "abs") return std::abs(arg1);
    if (name == "floor") return std::floor(arg1);
    if (name == "ceil") return std::ceil(arg1);
    if (name == "round") return std::round(arg1);
    if (name == "sign") return (arg1 > 0) ? 1.0 : ((arg1 < 0) ? -1.0 : 0.0);

    // Factorial
    if (name == "factorial" || name == "fact") {
        int n = static_cast<int>(arg1);
        if (n < 0) {
            state.error = "Factorial undefined for negative numbers";
            return 0.0;
        }
        double result = 1.0;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }

    // Two argument functions
    if (has_arg2) {
        if (name == "pow") return std::pow(arg1, arg2);
        if (name == "mod" || name == "fmod") return std::fmod(arg1, arg2);
        if (name == "min") return std::min(arg1, arg2);
        if (name == "max") return std::max(arg1, arg2);
        if (name == "atan2") return std::atan2(arg1, arg2) / angle_factor;
    }

    state.error = "Unknown function: " + name;
    return 0.0;
}

double Utilities::ParsePrimary(ParserState& state) {
    SkipWhitespace(state);

    // Parentheses
    if (Match(state, '(')) {
        double result = ParseExpression(state);
        if (!Match(state, ')')) {
            state.error = "Expected ')'";
        }
        return result;
    }

    // Negative sign
    if (Match(state, '-')) {
        return -ParsePrimary(state);
    }

    // Positive sign (optional)
    if (Match(state, '+')) {
        return ParsePrimary(state);
    }

    // Number
    if (std::isdigit(Peek(state)) || Peek(state) == '.') {
        return ParseNumber(state);
    }

    // Identifier (function or variable or constant)
    if (std::isalpha(Peek(state)) || Peek(state) == '_') {
        std::string name;
        while (state.pos < state.expression.size() &&
               (std::isalnum(state.expression[state.pos]) || state.expression[state.pos] == '_')) {
            name += state.expression[state.pos++];
        }

        // Check for constants
        std::string lower_name = name;
        std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        if (lower_name == "pi") return PI;
        if (lower_name == "e") return E;
        if (lower_name == "phi") return PHI;

        // Check for variables
        auto it = state.variables.find(name);
        if (it != state.variables.end()) {
            return it->second;
        }

        // Must be a function
        if (Peek(state) == '(') {
            return ParseFunction(state, lower_name);
        }

        state.error = "Unknown identifier: " + name;
        return 0.0;
    }

    state.error = "Unexpected character";
    return 0.0;
}

double Utilities::ParseFactor(ParserState& state) {
    double left = ParsePrimary(state);

    // Handle exponentiation (right-associative)
    if (Match(state, '^')) {
        double right = ParseFactor(state);  // Right-associative
        return std::pow(left, right);
    }

    return left;
}

double Utilities::ParseTerm(ParserState& state) {
    double left = ParseFactor(state);

    while (true) {
        if (Match(state, '*')) {
            left *= ParseFactor(state);
        } else if (Match(state, '/')) {
            double right = ParseFactor(state);
            if (right == 0.0) {
                state.error = "Division by zero";
                return 0.0;
            }
            left /= right;
        } else if (Match(state, '%')) {
            double right = ParseFactor(state);
            if (right == 0.0) {
                state.error = "Modulo by zero";
                return 0.0;
            }
            left = std::fmod(left, right);
        } else {
            break;
        }
    }

    return left;
}

double Utilities::ParseExpression(ParserState& state) {
    double left = ParseTerm(state);

    while (true) {
        if (Match(state, '+')) {
            left += ParseTerm(state);
        } else if (Match(state, '-')) {
            left -= ParseTerm(state);
        } else {
            break;
        }
    }

    return left;
}

CalculatorResult Utilities::Evaluate(
    const std::string& expression,
    const std::map<std::string, double>& variables,
    const std::string& angle_mode
) {
    CalculatorResult result;
    result.expression = expression;

    if (expression.empty()) {
        result.error_message = "Empty expression";
        return result;
    }

    ParserState state;
    state.expression = expression;
    state.pos = 0;
    state.variables = variables;
    state.angle_mode = angle_mode;

    try {
        result.result = ParseExpression(state);

        // Check for trailing characters
        SkipWhitespace(state);
        if (state.pos < state.expression.size()) {
            result.error_message = "Unexpected characters after expression";
            return result;
        }

        if (!state.error.empty()) {
            result.error_message = state.error;
            return result;
        }

        // Format result
        std::ostringstream oss;
        oss << std::setprecision(15) << result.result;
        result.formatted_result = oss.str();

        // Store variables used
        for (const auto& var : variables) {
            result.variables.push_back(var);
        }

        result.success = true;
    } catch (const std::exception& e) {
        result.error_message = e.what();
    }

    return result;
}

std::map<std::string, std::string> Utilities::GetSupportedFunctions() {
    return {
        {"sin", "Sine (sin(x))"},
        {"cos", "Cosine (cos(x))"},
        {"tan", "Tangent (tan(x))"},
        {"asin", "Arc sine (asin(x))"},
        {"acos", "Arc cosine (acos(x))"},
        {"atan", "Arc tangent (atan(x))"},
        {"atan2", "Arc tangent of y/x (atan2(y, x))"},
        {"sinh", "Hyperbolic sine (sinh(x))"},
        {"cosh", "Hyperbolic cosine (cosh(x))"},
        {"tanh", "Hyperbolic tangent (tanh(x))"},
        {"sqrt", "Square root (sqrt(x))"},
        {"cbrt", "Cube root (cbrt(x))"},
        {"exp", "Exponential (exp(x))"},
        {"log", "Natural logarithm (log(x))"},
        {"ln", "Natural logarithm (ln(x))"},
        {"log10", "Base-10 logarithm (log10(x))"},
        {"log2", "Base-2 logarithm (log2(x))"},
        {"abs", "Absolute value (abs(x))"},
        {"floor", "Floor (floor(x))"},
        {"ceil", "Ceiling (ceil(x))"},
        {"round", "Round (round(x))"},
        {"sign", "Sign function (sign(x))"},
        {"pow", "Power (pow(x, y))"},
        {"mod", "Modulo (mod(x, y))"},
        {"min", "Minimum (min(x, y))"},
        {"max", "Maximum (max(x, y))"},
        {"factorial", "Factorial (factorial(n))"},
    };
}

std::map<std::string, double> Utilities::GetSupportedConstants() {
    return {
        {"pi", PI},
        {"e", E},
        {"phi", PHI},
    };
}

// ============================================================================
// Unit Converter Implementation
// ============================================================================

std::string Utilities::GetCategoryForUnit(const std::string& unit) {
    auto it = g_unit_table.find(unit);
    if (it != g_unit_table.end()) {
        return it->second.category;
    }
    // Check temperature separately
    if (unit == "celsius" || unit == "fahrenheit" || unit == "kelvin") {
        return "temperature";
    }
    return "";
}

double Utilities::ConvertToBase(double value, const std::string& unit) {
    auto it = g_unit_table.find(unit);
    if (it != g_unit_table.end()) {
        return value * it->second.to_base;
    }
    return value;
}

double Utilities::ConvertFromBase(double value, const std::string& unit) {
    auto it = g_unit_table.find(unit);
    if (it != g_unit_table.end()) {
        return value / it->second.to_base;
    }
    return value;
}

UnitConversionResult Utilities::ConvertUnit(
    double value,
    const std::string& from_unit,
    const std::string& to_unit
) {
    UnitConversionResult result;
    result.input_value = value;
    result.input_unit = from_unit;
    result.output_unit = to_unit;

    std::string from_category = GetCategoryForUnit(from_unit);
    std::string to_category = GetCategoryForUnit(to_unit);

    if (from_category.empty()) {
        result.error_message = "Unknown source unit: " + from_unit;
        return result;
    }

    if (to_category.empty()) {
        result.error_message = "Unknown target unit: " + to_unit;
        return result;
    }

    if (from_category != to_category) {
        result.error_message = "Cannot convert between different categories: " +
                               from_category + " and " + to_category;
        return result;
    }

    result.category = from_category;

    // Handle temperature specially
    if (from_category == "temperature") {
        result.output_value = ConvertTemperature(value, from_unit, to_unit);

        // Generate formula
        if (from_unit == "celsius" && to_unit == "fahrenheit") {
            result.formula = "F = C * 9/5 + 32";
        } else if (from_unit == "fahrenheit" && to_unit == "celsius") {
            result.formula = "C = (F - 32) * 5/9";
        } else if (from_unit == "celsius" && to_unit == "kelvin") {
            result.formula = "K = C + 273.15";
        } else if (from_unit == "kelvin" && to_unit == "celsius") {
            result.formula = "C = K - 273.15";
        } else if (from_unit == "fahrenheit" && to_unit == "kelvin") {
            result.formula = "K = (F - 32) * 5/9 + 273.15";
        } else if (from_unit == "kelvin" && to_unit == "fahrenheit") {
            result.formula = "F = (K - 273.15) * 9/5 + 32";
        }
    } else {
        // Linear conversion through base unit
        double base_value = ConvertToBase(value, from_unit);
        result.output_value = ConvertFromBase(base_value, to_unit);

        // Generate formula
        double factor = g_unit_table[from_unit].to_base / g_unit_table[to_unit].to_base;
        std::ostringstream oss;
        oss << from_unit << " * " << std::setprecision(10) << factor << " = " << to_unit;
        result.formula = oss.str();
    }

    result.success = true;
    return result;
}

std::vector<std::string> Utilities::GetUnitCategories() {
    return {"length", "mass", "temperature", "time", "data", "area", "volume", "speed"};
}

std::vector<std::string> Utilities::GetUnitsForCategory(const std::string& category) {
    std::vector<std::string> units;

    if (category == "temperature") {
        return {"celsius", "fahrenheit", "kelvin"};
    }

    for (const auto& [unit, info] : g_unit_table) {
        if (info.category == category) {
            units.push_back(unit);
        }
    }

    return units;
}

UnitConversionResult Utilities::ConvertToAllUnits(double value, const std::string& from_unit) {
    UnitConversionResult result;
    result.input_value = value;
    result.input_unit = from_unit;

    std::string category = GetCategoryForUnit(from_unit);
    if (category.empty()) {
        result.error_message = "Unknown unit: " + from_unit;
        return result;
    }

    result.category = category;
    auto units = GetUnitsForCategory(category);

    for (const auto& unit : units) {
        auto conv = ConvertUnit(value, from_unit, unit);
        if (conv.success) {
            result.all_conversions.push_back({unit, conv.output_value});
        }
    }

    result.success = true;
    return result;
}

// ============================================================================
// Random Number Generator Implementation
// ============================================================================

void Utilities::ComputeStatistics(RandomNumberResult& result) {
    if (result.values.empty()) return;

    result.min_value = *std::min_element(result.values.begin(), result.values.end());
    result.max_value = *std::max_element(result.values.begin(), result.values.end());
    result.mean = std::accumulate(result.values.begin(), result.values.end(), 0.0) / result.values.size();

    double sq_sum = 0.0;
    for (double v : result.values) {
        sq_sum += (v - result.mean) * (v - result.mean);
    }
    result.std_dev = std::sqrt(sq_sum / result.values.size());
}

void Utilities::BuildHistogram(RandomNumberResult& result, int num_bins) {
    if (result.values.empty()) return;

    double range = result.max_value - result.min_value;
    if (range == 0) {
        result.histogram[0] = static_cast<int>(result.values.size());
        return;
    }

    double bin_width = range / num_bins;

    for (double v : result.values) {
        int bin = static_cast<int>((v - result.min_value) / bin_width);
        if (bin >= num_bins) bin = num_bins - 1;
        result.histogram[bin]++;
    }
}

RandomNumberResult Utilities::GenerateUniform(int count, double min, double max, int64_t seed) {
    RandomNumberResult result;
    result.distribution = "uniform";
    result.count = count;

    if (count <= 0) {
        result.error_message = "Count must be positive";
        return result;
    }

    std::mt19937_64 gen;
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }
    gen.seed(static_cast<uint64_t>(seed));
    result.seed_used = static_cast<uint64_t>(seed);

    std::uniform_real_distribution<double> dist(min, max);

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(dist(gen));
    }

    ComputeStatistics(result);
    BuildHistogram(result);
    result.success = true;
    return result;
}

RandomNumberResult Utilities::GenerateNormal(int count, double mean, double std_dev, int64_t seed) {
    RandomNumberResult result;
    result.distribution = "normal";
    result.count = count;

    if (count <= 0) {
        result.error_message = "Count must be positive";
        return result;
    }

    if (std_dev <= 0) {
        result.error_message = "Standard deviation must be positive";
        return result;
    }

    std::mt19937_64 gen;
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }
    gen.seed(static_cast<uint64_t>(seed));
    result.seed_used = static_cast<uint64_t>(seed);

    std::normal_distribution<double> dist(mean, std_dev);

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(dist(gen));
    }

    ComputeStatistics(result);
    BuildHistogram(result);
    result.success = true;
    return result;
}

RandomNumberResult Utilities::GenerateExponential(int count, double lambda, int64_t seed) {
    RandomNumberResult result;
    result.distribution = "exponential";
    result.count = count;

    if (count <= 0) {
        result.error_message = "Count must be positive";
        return result;
    }

    if (lambda <= 0) {
        result.error_message = "Lambda must be positive";
        return result;
    }

    std::mt19937_64 gen;
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }
    gen.seed(static_cast<uint64_t>(seed));
    result.seed_used = static_cast<uint64_t>(seed);

    std::exponential_distribution<double> dist(lambda);

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(dist(gen));
    }

    ComputeStatistics(result);
    BuildHistogram(result);
    result.success = true;
    return result;
}

RandomNumberResult Utilities::GeneratePoisson(int count, double lambda, int64_t seed) {
    RandomNumberResult result;
    result.distribution = "poisson";
    result.count = count;

    if (count <= 0) {
        result.error_message = "Count must be positive";
        return result;
    }

    if (lambda <= 0) {
        result.error_message = "Lambda must be positive";
        return result;
    }

    std::mt19937_64 gen;
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }
    gen.seed(static_cast<uint64_t>(seed));
    result.seed_used = static_cast<uint64_t>(seed);

    std::poisson_distribution<int> dist(lambda);

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(static_cast<double>(dist(gen)));
    }

    ComputeStatistics(result);
    BuildHistogram(result);
    result.success = true;
    return result;
}

RandomNumberResult Utilities::GenerateIntegers(int count, int min, int max, int64_t seed) {
    RandomNumberResult result;
    result.distribution = "integer";
    result.count = count;

    if (count <= 0) {
        result.error_message = "Count must be positive";
        return result;
    }

    if (min > max) {
        result.error_message = "Min must be less than or equal to max";
        return result;
    }

    std::mt19937_64 gen;
    if (seed < 0) {
        std::random_device rd;
        seed = rd();
    }
    gen.seed(static_cast<uint64_t>(seed));
    result.seed_used = static_cast<uint64_t>(seed);

    std::uniform_int_distribution<int> dist(min, max);

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(static_cast<double>(dist(gen)));
    }

    ComputeStatistics(result);
    BuildHistogram(result, std::min(max - min + 1, 20));
    result.success = true;
    return result;
}

std::vector<std::string> Utilities::GenerateUUIDs(int count) {
    std::vector<std::string> uuids;

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;

    for (int i = 0; i < count; i++) {
        uint64_t part1 = dist(gen);
        uint64_t part2 = dist(gen);

        // Format as UUID v4: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
        // Set version (4) and variant (8, 9, A, or B)
        part1 = (part1 & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;  // Version 4
        part2 = (part2 & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;  // Variant

        std::ostringstream oss;
        oss << std::hex << std::setfill('0');
        oss << std::setw(8) << ((part1 >> 32) & 0xFFFFFFFF) << "-";
        oss << std::setw(4) << ((part1 >> 16) & 0xFFFF) << "-";
        oss << std::setw(4) << (part1 & 0xFFFF) << "-";
        oss << std::setw(4) << ((part2 >> 48) & 0xFFFF) << "-";
        oss << std::setw(12) << (part2 & 0xFFFFFFFFFFFFULL);

        uuids.push_back(oss.str());
    }

    return uuids;
}

// ============================================================================
// Hash Generator Implementation
// ============================================================================

std::string Utilities::BytesToHex(const std::vector<uint8_t>& bytes) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint8_t b : bytes) {
        oss << std::setw(2) << static_cast<int>(b);
    }
    return oss.str();
}

// Simple portable MD5 implementation (RFC 1321)
std::string Utilities::ComputeMD5(const std::vector<uint8_t>& data) {
#ifdef CYXWIZ_HAS_OPENSSL
    unsigned char hash[MD5_DIGEST_LENGTH];
    MD5(data.data(), data.size(), hash);
    return BytesToHex(std::vector<uint8_t>(hash, hash + MD5_DIGEST_LENGTH));
#else
    // Simplified portable implementation
    // For production, consider using a proper library
    return "MD5 requires OpenSSL";
#endif
}

std::string Utilities::ComputeSHA1(const std::vector<uint8_t>& data) {
#ifdef CYXWIZ_HAS_OPENSSL
    unsigned char hash[SHA_DIGEST_LENGTH];
    SHA1(data.data(), data.size(), hash);
    return BytesToHex(std::vector<uint8_t>(hash, hash + SHA_DIGEST_LENGTH));
#else
    return "SHA1 requires OpenSSL";
#endif
}

std::string Utilities::ComputeSHA256(const std::vector<uint8_t>& data) {
#ifdef CYXWIZ_HAS_OPENSSL
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(data.data(), data.size(), hash);
    return BytesToHex(std::vector<uint8_t>(hash, hash + SHA256_DIGEST_LENGTH));
#else
    return "SHA256 requires OpenSSL";
#endif
}

std::string Utilities::ComputeSHA512(const std::vector<uint8_t>& data) {
#ifdef CYXWIZ_HAS_OPENSSL
    unsigned char hash[SHA512_DIGEST_LENGTH];
    SHA512(data.data(), data.size(), hash);
    return BytesToHex(std::vector<uint8_t>(hash, hash + SHA512_DIGEST_LENGTH));
#else
    return "SHA512 requires OpenSSL";
#endif
}

HashResult Utilities::HashText(const std::string& text, const std::string& algorithm) {
    HashResult result;
    result.input_text = text;
    result.input_size = text.size();
    result.algorithm = algorithm;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> data(text.begin(), text.end());

    if (algorithm == "md5" || algorithm == "all") {
        result.md5_hash = ComputeMD5(data);
    }
    if (algorithm == "sha1" || algorithm == "all") {
        result.sha1_hash = ComputeSHA1(data);
    }
    if (algorithm == "sha256" || algorithm == "all") {
        result.sha256_hash = ComputeSHA256(data);
    }
    if (algorithm == "sha512" || algorithm == "all") {
        result.sha512_hash = ComputeSHA512(data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.compute_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    result.success = true;
    return result;
}

HashResult Utilities::HashFile(const std::string& file_path, const std::string& algorithm) {
    HashResult result;
    result.input_file = file_path;
    result.algorithm = algorithm;

    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        result.error_message = "Could not open file: " + file_path;
        return result;
    }

    // Read file into memory
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
    file.close();

    result.input_size = data.size();

    auto start = std::chrono::high_resolution_clock::now();

    if (algorithm == "md5" || algorithm == "all") {
        result.md5_hash = ComputeMD5(data);
    }
    if (algorithm == "sha1" || algorithm == "all") {
        result.sha1_hash = ComputeSHA1(data);
    }
    if (algorithm == "sha256" || algorithm == "all") {
        result.sha256_hash = ComputeSHA256(data);
    }
    if (algorithm == "sha512" || algorithm == "all") {
        result.sha512_hash = ComputeSHA512(data);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.compute_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    result.success = true;
    return result;
}

bool Utilities::VerifyHash(
    const std::string& text,
    const std::string& expected_hash,
    const std::string& algorithm
) {
    auto result = HashText(text, algorithm);

    std::string computed;
    if (algorithm == "md5") computed = result.md5_hash;
    else if (algorithm == "sha1") computed = result.sha1_hash;
    else if (algorithm == "sha256") computed = result.sha256_hash;
    else if (algorithm == "sha512") computed = result.sha512_hash;

    // Case-insensitive comparison
    std::string lower_computed = computed;
    std::string lower_expected = expected_hash;
    std::transform(lower_computed.begin(), lower_computed.end(), lower_computed.begin(), ::tolower);
    std::transform(lower_expected.begin(), lower_expected.end(), lower_expected.begin(), ::tolower);

    return lower_computed == lower_expected;
}

// ============================================================================
// JSON Viewer Implementation
// ============================================================================

void Utilities::AnalyzeJSON(const std::string& json_str, JSONResult& result) {
    try {
        auto json = nlohmann::json::parse(json_str);

        std::function<void(const nlohmann::json&, int)> analyze = [&](const nlohmann::json& j, int depth) {
            result.depth = std::max(result.depth, depth);

            if (j.is_object()) {
                result.object_count++;
                for (auto& [key, value] : j.items()) {
                    if (depth == 0) {
                        result.keys.push_back(key);
                    }
                    analyze(value, depth + 1);
                }
            } else if (j.is_array()) {
                result.array_count++;
                for (const auto& item : j) {
                    analyze(item, depth + 1);
                }
            } else if (j.is_string()) {
                result.string_count++;
            } else if (j.is_number()) {
                result.number_count++;
            } else if (j.is_boolean()) {
                result.bool_count++;
            } else if (j.is_null()) {
                result.null_count++;
            }
        };

        analyze(json, 0);
    } catch (...) {
        // Ignore analysis errors
    }
}

JSONResult Utilities::ValidateJSON(const std::string& json_text) {
    JSONResult result;
    result.input_json = json_text;

    try {
        auto json = nlohmann::json::parse(json_text);
        result.is_valid = true;
        result.formatted_json = json.dump(2);
        result.minified_json = json.dump();

        AnalyzeJSON(json_text, result);
        result.success = true;
    } catch (const nlohmann::json::parse_error& e) {
        result.is_valid = false;
        result.error_detail = e.what();

        // Try to extract line/column from error message
        std::string msg = e.what();
        std::regex pos_regex("at position (\\d+)");
        std::smatch match;
        if (std::regex_search(msg, match, pos_regex)) {
            int pos = std::stoi(match[1]);
            // Count lines and columns
            int line = 1, col = 1;
            for (int i = 0; i < pos && i < static_cast<int>(json_text.size()); i++) {
                if (json_text[i] == '\n') {
                    line++;
                    col = 1;
                } else {
                    col++;
                }
            }
            result.error_line = line;
            result.error_column = col;
        }

        result.success = false;
        result.error_message = "Invalid JSON";
    }

    return result;
}

JSONResult Utilities::FormatJSON(const std::string& json_text, int indent_size) {
    JSONResult result;
    result.input_json = json_text;

    try {
        auto json = nlohmann::json::parse(json_text);
        result.is_valid = true;
        result.formatted_json = json.dump(indent_size);
        result.minified_json = json.dump();

        AnalyzeJSON(json_text, result);
        result.success = true;
    } catch (const nlohmann::json::parse_error& e) {
        result.is_valid = false;
        result.error_detail = e.what();
        result.error_message = "Invalid JSON";
    }

    return result;
}

JSONResult Utilities::MinifyJSON(const std::string& json_text) {
    JSONResult result;
    result.input_json = json_text;

    try {
        auto json = nlohmann::json::parse(json_text);
        result.is_valid = true;
        result.formatted_json = json.dump(2);
        result.minified_json = json.dump();
        result.success = true;
    } catch (const nlohmann::json::parse_error& e) {
        result.is_valid = false;
        result.error_detail = e.what();
        result.error_message = "Invalid JSON";
    }

    return result;
}

std::string Utilities::GetJSONValue(const std::string& json_text, const std::string& path) {
    try {
        auto json = nlohmann::json::parse(json_text);

        // Simple path parser: supports dot notation and array indices
        std::vector<std::string> parts;
        std::string current;

        for (size_t i = 0; i < path.size(); i++) {
            char c = path[i];
            if (c == '.') {
                if (!current.empty()) {
                    parts.push_back(current);
                    current.clear();
                }
            } else if (c == '[') {
                if (!current.empty()) {
                    parts.push_back(current);
                    current.clear();
                }
                // Find closing bracket
                size_t end = path.find(']', i);
                if (end != std::string::npos) {
                    parts.push_back(path.substr(i + 1, end - i - 1));
                    i = end;
                }
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            parts.push_back(current);
        }

        // Navigate the JSON
        nlohmann::json current_json = json;
        for (const auto& part : parts) {
            // Check if it's an array index
            bool is_index = true;
            for (char c : part) {
                if (!std::isdigit(c)) {
                    is_index = false;
                    break;
                }
            }

            if (is_index && current_json.is_array()) {
                int idx = std::stoi(part);
                if (idx >= 0 && idx < static_cast<int>(current_json.size())) {
                    current_json = current_json[idx];
                } else {
                    return "Index out of bounds";
                }
            } else if (current_json.is_object() && current_json.contains(part)) {
                current_json = current_json[part];
            } else {
                return "Path not found: " + part;
            }
        }

        if (current_json.is_string()) {
            return current_json.get<std::string>();
        } else {
            return current_json.dump();
        }
    } catch (const std::exception& e) {
        return std::string("Error: ") + e.what();
    }
}

// ============================================================================
// Regex Tester Implementation
// ============================================================================

RegexResult Utilities::TestRegex(
    const std::string& pattern,
    const std::string& text,
    const std::string& flags
) {
    RegexResult result;
    result.pattern = pattern;
    result.input_text = text;
    result.flags = flags;

    if (pattern.empty()) {
        result.error_message = "Empty pattern";
        return result;
    }

    try {
        // Build regex flags
        std::regex_constants::syntax_option_type regex_flags = std::regex_constants::ECMAScript;
        if (flags.find('i') != std::string::npos) {
            regex_flags |= std::regex_constants::icase;
        }
        // Note: multiline flag is not standard C++ (not in std::regex_constants)
        // The 'm' flag is silently ignored for portability

        std::regex re(pattern, regex_flags);
        result.is_valid_pattern = true;

        // Find all matches
        std::sregex_iterator iter(text.begin(), text.end(), re);
        std::sregex_iterator end;

        while (iter != end) {
            RegexMatch match;
            match.match_text = iter->str();
            match.start_pos = static_cast<int>(iter->position());
            match.end_pos = match.start_pos + static_cast<int>(match.match_text.length());

            // Calculate line number
            match.line_number = 1;
            for (int i = 0; i < match.start_pos; i++) {
                if (text[i] == '\n') match.line_number++;
            }

            // Capture groups
            for (size_t i = 1; i < iter->size(); i++) {
                match.groups.push_back((*iter)[i].str());
            }

            result.matches.push_back(match);
            ++iter;
        }

        result.match_count = static_cast<int>(result.matches.size());
        result.success = true;
    } catch (const std::regex_error& e) {
        result.is_valid_pattern = false;
        result.pattern_error = e.what();
        result.error_message = "Invalid regex pattern";
    }

    return result;
}

RegexResult Utilities::ReplaceRegex(
    const std::string& pattern,
    const std::string& text,
    const std::string& replacement,
    const std::string& flags
) {
    RegexResult result = TestRegex(pattern, text, flags);
    result.replacement_pattern = replacement;

    if (!result.is_valid_pattern) {
        return result;
    }

    try {
        std::regex_constants::syntax_option_type regex_flags = std::regex_constants::ECMAScript;
        if (flags.find('i') != std::string::npos) {
            regex_flags |= std::regex_constants::icase;
        }
        // Note: multiline flag is not standard C++ (not in std::regex_constants)
        // The 'm' flag is silently ignored for portability

        std::regex re(pattern, regex_flags);
        result.replaced_text = std::regex_replace(text, re, replacement);
        result.success = true;
    } catch (const std::regex_error& e) {
        result.error_message = e.what();
    }

    return result;
}

bool Utilities::IsValidRegex(const std::string& pattern) {
    try {
        std::regex re(pattern);
        return true;
    } catch (...) {
        return false;
    }
}

std::map<std::string, std::string> Utilities::GetCommonPatterns() {
    return {
        {"Email", R"([\w.+-]+@[\w-]+\.[\w.-]+)"},
        {"Phone (US)", R"(\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})"},
        {"URL", R"(https?://[\w\-._~:/?#\[\]@!$&'()*+,;=]+)"},
        {"IPv4", R"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"},
        {"Date (ISO)", R"(\d{4}-\d{2}-\d{2})"},
        {"Date (US)", R"(\d{1,2}/\d{1,2}/\d{4})"},
        {"Time (24h)", R"(([01]?[0-9]|2[0-3]):[0-5][0-9])"},
        {"Time (12h)", R"(([0]?[1-9]|1[0-2]):[0-5][0-9]\s?(AM|PM|am|pm))"},
        {"Hex Color", R"(#[0-9A-Fa-f]{6})"},
        {"ZIP Code (US)", R"(\d{5}(-\d{4})?)"},
        {"Credit Card", R"(\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4})"},
        {"SSN", R"(\d{3}-\d{2}-\d{4})"},
        {"Integer", R"(-?\d+)"},
        {"Decimal", R"(-?\d+\.?\d*)"},
        {"Word", R"(\w+)"},
        {"Whitespace", R"(\s+)"},
        {"HTML Tag", R"(<[^>]+>)"},
        {"Quoted String", R"("([^"\\]|\\.)*")"},
    };
}

} // namespace cyxwiz
