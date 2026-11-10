#include <cyxwiz/utilities.h>
#include <cyxwiz/expression.h>
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
#include <openssl/evp.h>
#endif
#include "cyxwiz/portable_random.h"

namespace cyxwiz {

namespace {

// The requested digests over one stream of bytes (text or a file read in
// blocks), so a large file is never held in memory.
class Digests {
public:
    // Leaves `error` empty when every name is known and OpenSSL is in the build.
    Digests(const std::string& algorithm, std::string& error) {
        const bool all = algorithm == "all";
        if (!all && algorithm != "md5" && algorithm != "sha1" && algorithm != "sha256" && algorithm != "sha512") {
            error = "Unknown hash algorithm '" + algorithm + "' (md5, sha1, sha256, sha512 or all)";
            return;
        }
#ifdef CYXWIZ_HAS_OPENSSL
        const std::pair<const char*, const EVP_MD*> known[] = {
            {"md5", EVP_md5()}, {"sha1", EVP_sha1()}, {"sha256", EVP_sha256()}, {"sha512", EVP_sha512()}};
        for (const auto& [name, md] : known) {
            if (!all && algorithm != name) continue;
            EVP_MD_CTX* ctx = EVP_MD_CTX_new();
            if (ctx == nullptr || EVP_DigestInit_ex(ctx, md, nullptr) != 1) {
                EVP_MD_CTX_free(ctx);
                error = std::string("Could not start ") + name;
                return;
            }
            contexts_.push_back({name, ctx});
        }
#else
        error = "Hashing needs OpenSSL, which this build does not include";
#endif
    }
    ~Digests() {
#ifdef CYXWIZ_HAS_OPENSSL
        for (auto& entry : contexts_) EVP_MD_CTX_free(entry.second);
#endif
    }
    Digests(const Digests&) = delete;
    Digests& operator=(const Digests&) = delete;

    bool Update(const void* data, size_t size) {
#ifdef CYXWIZ_HAS_OPENSSL
        for (auto& entry : contexts_)
            if (EVP_DigestUpdate(entry.second, data, size) != 1) return false;
#else
        (void)data;
        (void)size;
#endif
        return true;
    }

    // Lower-case hex digests into the result.
    bool Finish(HashResult& result) {
#ifdef CYXWIZ_HAS_OPENSSL
        for (auto& [name, ctx] : contexts_) {
            unsigned char digest[EVP_MAX_MD_SIZE];
            unsigned int size = 0;
            if (EVP_DigestFinal_ex(ctx, digest, &size) != 1) return false;
            std::ostringstream hex;
            hex << std::hex << std::setfill('0');
            for (unsigned int i = 0; i < size; ++i) hex << std::setw(2) << static_cast<int>(digest[i]);
            const std::string id = name;
            if (id == "md5") result.md5_hash = hex.str();
            if (id == "sha1") result.sha1_hash = hex.str();
            if (id == "sha256") result.sha256_hash = hex.str();
            if (id == "sha512") result.sha512_hash = hex.str();
        }
        return true;
#else
        (void)result;
        return false;
#endif
    }

private:
#ifdef CYXWIZ_HAS_OPENSSL
    std::vector<std::pair<const char*, EVP_MD_CTX*>> contexts_;
#endif
};

}  // namespace

// ============================================================================
// Constants
// ============================================================================

static const double PI = 3.14159265358979323846;
static const double E = 2.71828182845904523536;
static const double PHI = 1.61803398874989484820;  // Golden ratio

// ============================================================================
// Units (one table for the Utilities window, the UnitConverter node and
// Python). base = value * scale + offset, in the category's base unit:
// metre, kilogram, second, byte, square metre, litre, metre per second,
// kelvin. Factors are the exact definitions (NIST SP 811; international
// yard and pound, 1959; US customary and imperial volumes).
// ============================================================================

namespace {

constexpr double kInch = 0.0254;
constexpr double kFoot = 0.3048;
constexpr double kYard = 0.9144;
constexpr double kMile = 1609.344;
constexpr double kPound = 0.45359237;
constexpr double kUsGallon = 3.785411784;  // 231 cubic inches, in litres
constexpr double kImperialGallon = 4.54609;

const std::vector<UnitDefinition>& UnitTable() {
    static const std::vector<UnitDefinition> table = {
        // Length (base: metre)
        {"meter", "metre", "m", "length", 1.0, 0.0, {"metre", "meters", "metres"}},
        {"kilometer", "kilometre", "km", "length", 1000.0, 0.0, {"kilometre", "kilometers", "kilometres"}},
        {"centimeter", "centimetre", "cm", "length", 0.01, 0.0, {"centimetre", "centimeters", "centimetres"}},
        {"millimeter", "millimetre", "mm", "length", 0.001, 0.0, {"millimetre", "millimeters", "millimetres"}},
        {"micrometer", "micrometre", "um", "length", 1e-6, 0.0, {"micrometre", "micron", "microns"}},
        {"nanometer", "nanometre", "nm", "length", 1e-9, 0.0, {"nanometre"}},
        {"inch", "inch", "in", "length", kInch, 0.0, {"inches"}},
        {"foot", "foot", "ft", "length", kFoot, 0.0, {"feet"}},
        {"yard", "yard", "yd", "length", kYard, 0.0, {"yards"}},
        {"mile", "mile", "mi", "length", kMile, 0.0, {"miles"}},
        {"nautical_mile", "nautical mile", "nmi", "length", 1852.0, 0.0, {"nautical_miles"}},

        // Mass (base: kilogram)
        {"kilogram", "kilogram", "kg", "mass", 1.0, 0.0, {"kilograms"}},
        {"gram", "gram", "g", "mass", 0.001, 0.0, {"grams"}},
        {"milligram", "milligram", "mg", "mass", 1e-6, 0.0, {"milligrams"}},
        {"tonne", "tonne (1000 kg)", "t", "mass", 1000.0, 0.0, {"tonnes", "metric_ton"}},
        {"pound", "pound", "lb", "mass", kPound, 0.0, {"pounds", "lbs"}},
        {"ounce", "ounce", "oz", "mass", kPound / 16.0, 0.0, {"ounces"}},
        {"stone", "stone (14 lb)", "st", "mass", 14.0 * kPound, 0.0, {"stones"}},
        {"short_ton", "short ton (US, 2000 lb)", "ton (US)", "mass", 2000.0 * kPound, 0.0, {}},
        {"long_ton", "long ton (UK, 2240 lb)", "ton (UK)", "mass", 2240.0 * kPound, 0.0, {}},

        // Time (base: second)
        {"nanosecond", "nanosecond", "ns", "time", 1e-9, 0.0, {"nanoseconds"}},
        {"microsecond", "microsecond", "us", "time", 1e-6, 0.0, {"microseconds"}},
        {"millisecond", "millisecond", "ms", "time", 0.001, 0.0, {"milliseconds"}},
        {"second", "second", "s", "time", 1.0, 0.0, {"seconds", "sec"}},
        {"minute", "minute", "min", "time", 60.0, 0.0, {"minutes"}},
        {"hour", "hour", "h", "time", 3600.0, 0.0, {"hours", "hr"}},
        {"day", "day", "d", "time", 86400.0, 0.0, {"days"}},
        {"week", "week", "wk", "time", 604800.0, 0.0, {"weeks"}},
        {"year", "year (365 days)", "yr", "time", 365.0 * 86400.0, 0.0, {"years"}},

        // Data (base: byte). SI prefixes are powers of 1000, binary ones of 1024.
        {"bit", "bit", "bit", "data", 0.125, 0.0, {"bits"}},
        {"byte", "byte", "B", "data", 1.0, 0.0, {"bytes"}},
        {"kilobit", "kilobit", "kbit", "data", 125.0, 0.0, {"kilobits"}},
        {"megabit", "megabit", "Mbit", "data", 125e3, 0.0, {"megabits"}},
        {"gigabit", "gigabit", "Gbit", "data", 125e6, 0.0, {"gigabits"}},
        {"kilobyte", "kilobyte (1000 B)", "kB", "data", 1e3, 0.0, {"kilobytes"}},
        {"megabyte", "megabyte (10^6 B)", "MB", "data", 1e6, 0.0, {"megabytes"}},
        {"gigabyte", "gigabyte (10^9 B)", "GB", "data", 1e9, 0.0, {"gigabytes"}},
        {"terabyte", "terabyte (10^12 B)", "TB", "data", 1e12, 0.0, {"terabytes"}},
        {"petabyte", "petabyte (10^15 B)", "PB", "data", 1e15, 0.0, {"petabytes"}},
        {"kibibyte", "kibibyte (1024 B)", "KiB", "data", 1024.0, 0.0, {"kibibytes"}},
        {"mebibyte", "mebibyte (2^20 B)", "MiB", "data", 1048576.0, 0.0, {"mebibytes"}},
        {"gibibyte", "gibibyte (2^30 B)", "GiB", "data", 1073741824.0, 0.0, {"gibibytes"}},
        {"tebibyte", "tebibyte (2^40 B)", "TiB", "data", 1099511627776.0, 0.0, {"tebibytes"}},
        {"pebibyte", "pebibyte (2^50 B)", "PiB", "data", 1125899906842624.0, 0.0, {"pebibytes"}},

        // Area (base: square metre)
        {"square_meter", "square metre", "m2", "area", 1.0, 0.0, {"square_metre", "m^2", "sqm"}},
        {"square_centimeter", "square centimetre", "cm2", "area", 1e-4, 0.0, {"square_centimetre", "cm^2"}},
        {"square_kilometer", "square kilometre", "km2", "area", 1e6, 0.0, {"square_kilometre", "km^2"}},
        {"hectare", "hectare", "ha", "area", 1e4, 0.0, {"hectares"}},
        {"square_inch", "square inch", "in2", "area", kInch * kInch, 0.0, {"in^2", "sqin"}},
        {"square_foot", "square foot", "ft2", "area", kFoot * kFoot, 0.0, {"ft^2", "sqft"}},
        {"square_yard", "square yard", "yd2", "area", kYard * kYard, 0.0, {"yd^2"}},
        {"acre", "acre", "ac", "area", 4840.0 * kYard * kYard, 0.0, {"acres"}},
        {"square_mile", "square mile", "mi2", "area", kMile * kMile, 0.0, {"mi^2"}},

        // Volume (base: litre)
        {"liter", "litre", "L", "volume", 1.0, 0.0, {"litre", "liters", "litres", "l"}},
        {"milliliter", "millilitre", "mL", "volume", 0.001, 0.0, {"millilitre", "milliliters", "millilitres", "ml"}},
        {"cubic_meter", "cubic metre", "m3", "volume", 1000.0, 0.0, {"cubic_metre", "m^3"}},
        {"cubic_centimeter", "cubic centimetre", "cm3", "volume", 0.001, 0.0, {"cubic_centimetre", "cm^3", "cc"}},
        {"cubic_inch", "cubic inch", "in3", "volume", kInch * kInch * kInch * 1000.0, 0.0, {"in^3"}},
        {"cubic_foot", "cubic foot", "ft3", "volume", kFoot * kFoot * kFoot * 1000.0, 0.0, {"ft^3"}},
        {"gallon", "US gallon", "gal (US)", "volume", kUsGallon, 0.0, {"gallons", "us_gallon", "gal"}},
        {"quart", "US quart", "qt (US)", "volume", kUsGallon / 4.0, 0.0, {"quarts", "us_quart"}},
        {"pint", "US pint", "pt (US)", "volume", kUsGallon / 8.0, 0.0, {"pints", "us_pint"}},
        {"cup", "US cup", "cup (US)", "volume", kUsGallon / 16.0, 0.0, {"cups", "us_cup"}},
        {"fluid_ounce", "US fluid ounce", "fl oz (US)", "volume", kUsGallon / 128.0, 0.0, {"fluid_ounces", "us_fluid_ounce"}},
        {"imperial_gallon", "imperial gallon", "gal (UK)", "volume", kImperialGallon, 0.0, {"uk_gallon"}},
        {"imperial_pint", "imperial pint", "pt (UK)", "volume", kImperialGallon / 8.0, 0.0, {"uk_pint"}},

        // Speed (base: metre per second)
        {"meter_per_second", "metre per second", "m/s", "speed", 1.0, 0.0, {"metre_per_second", "mps"}},
        {"kilometer_per_hour", "kilometre per hour", "km/h", "speed", 1000.0 / 3600.0, 0.0, {"kilometre_per_hour", "kph", "kmh"}},
        {"mile_per_hour", "mile per hour", "mph", "speed", kMile / 3600.0, 0.0, {"miles_per_hour"}},
        {"knot", "knot", "kn", "speed", 1852.0 / 3600.0, 0.0, {"knots", "kt"}},
        {"foot_per_second", "foot per second", "ft/s", "speed", kFoot, 0.0, {"feet_per_second", "fps"}},

        // Temperature (base: kelvin)
        {"kelvin", "kelvin", "K", "temperature", 1.0, 0.0, {"k"}},
        {"celsius", "degree Celsius", "°C", "temperature", 1.0, 273.15, {"c", "degc"}},
        {"fahrenheit", "degree Fahrenheit", "°F", "temperature", 5.0 / 9.0, 459.67 * 5.0 / 9.0, {"f", "degf"}},
        {"rankine", "degree Rankine", "°R", "temperature", 5.0 / 9.0, 0.0, {"r", "degr"}},
    };
    return table;
}

// Lower case, spaces and hyphens as underscores.
std::string UnitKey(const std::string& name) {
    std::string key;
    for (const unsigned char c : name) {
        if (std::isspace(c) || c == '-') {
            if (!key.empty() && key.back() != '_') key += '_';
        } else {
            key += static_cast<char>(std::tolower(c));
        }
    }
    while (!key.empty() && key.back() == '_') key.pop_back();
    return key;
}

std::string FullNumber(double v) {
    std::ostringstream oss;
    oss << std::setprecision(17) << v;
    // Shortest text that reads back as the same double.
    for (int digits = 1; digits <= 17; ++digits) {
        std::ostringstream shorter;
        shorter << std::setprecision(digits) << v;
        if (std::stod(shorter.str()) == v) return shorter.str();
    }
    return oss.str();
}

}  // namespace

// ============================================================================
// Calculator Implementation
// ============================================================================

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

    std::vector<std::string> names;
    std::vector<double> values;
    for (const auto& [name, value] : variables) {
        names.push_back(name);
        values.push_back(value);
    }
    const Expression parsed = Expression::Parse(
        expression, names, angle_mode == "degrees" ? Expression::AngleUnit::Degrees : Expression::AngleUnit::Radians);
    if (!parsed.Valid()) {
        result.error_message = parsed.Error();
        return result;
    }

    try {
        std::string problem;
        result.result = parsed.Evaluate(values, &problem);
        if (!problem.empty()) {
            result.error_message = problem;
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

const std::vector<UnitDefinition>& Utilities::Units() {
    return UnitTable();
}

const UnitDefinition* Utilities::FindUnit(const std::string& name, const std::string& category) {
    const std::string key = UnitKey(name);
    if (key.empty()) return nullptr;
    const std::string wanted = UnitKey(category);
    const auto in_category = [&](const UnitDefinition& u) { return wanted.empty() || u.category == wanted; };
    // Exact symbols first (mm is not Mm), then ids, labels, symbols and aliases
    // in any case.
    for (const auto& u : UnitTable())
        if (in_category(u) && u.symbol == name) return &u;
    for (const auto& u : UnitTable()) {
        if (!in_category(u)) continue;
        if (u.id == key || UnitKey(u.label) == key || UnitKey(u.symbol) == key) return &u;
        for (const auto& alias : u.aliases)
            if (UnitKey(alias) == key) return &u;
    }
    return nullptr;
}

std::string Utilities::GetCategoryForUnit(const std::string& unit) {
    const UnitDefinition* u = FindUnit(unit);
    return u ? u->category : std::string();
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

    const UnitDefinition* from = FindUnit(from_unit);
    const UnitDefinition* to = from ? FindUnit(to_unit, from->category) : nullptr;
    if (!from) {
        result.error_message = "Unknown source unit: " + from_unit;
        return result;
    }
    if (!to) {
        const UnitDefinition* other = FindUnit(to_unit);
        result.error_message = other ? "Cannot convert " + from->category + " to " + other->category
                                     : "Unknown target unit: " + to_unit;
        return result;
    }
    if (!std::isfinite(value)) {
        result.error_message = "The value must be a finite number";
        return result;
    }
    result.category = from->category;
    const double base = value * from->scale + from->offset;
    if (from->category == "temperature" && base < 0.0) {
        result.error_message = "Below absolute zero (" + FullNumber(base) + " K)";
        return result;
    }
    // to = from * a + b
    const double a = from->scale / to->scale;
    const double b = (from->offset - to->offset) / to->scale;
    result.output_value = (base - to->offset) / to->scale;
    result.input_unit = from->id;
    result.output_unit = to->id;
    result.formula = to->symbol + " = " + from->symbol + " x " + FullNumber(a);
    if (b != 0.0) result.formula += (b > 0.0 ? " + " : " - ") + FullNumber(std::fabs(b));
    result.success = true;
    return result;
}

std::vector<std::string> Utilities::GetUnitCategories() {
    return {"length", "mass", "temperature", "time", "data", "area", "volume", "speed"};
}

std::vector<std::string> Utilities::GetUnitsForCategory(const std::string& category) {
    std::vector<std::string> units;
    for (const auto& u : UnitTable())
        if (u.category == category) units.push_back(u.id);
    return units;
}

UnitConversionResult Utilities::ConvertToAllUnits(double value, const std::string& from_unit) {
    UnitConversionResult result;
    result.input_value = value;
    result.input_unit = from_unit;
    const UnitDefinition* from = FindUnit(from_unit);
    if (!from) {
        result.error_message = "Unknown unit: " + from_unit;
        return result;
    }
    result.category = from->category;
    result.input_unit = from->id;
    for (const auto& unit : GetUnitsForCategory(from->category)) {
        auto conv = ConvertUnit(value, from->id, unit);
        if (!conv.success) {
            result.error_message = conv.error_message;
            return result;
        }
        result.all_conversions.push_back({unit, conv.output_value});
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
    const double n = static_cast<double>(result.values.size());
    result.std_dev_population = std::sqrt(sq_sum / n);
    result.std_dev_sample = result.values.size() > 1 ? std::sqrt(sq_sum / (n - 1.0)) : 0.0;

    std::vector<double> sorted = result.values;
    std::sort(sorted.begin(), sorted.end());
    // numpy.percentile's default: linear between the two nearest ranks.
    const auto quantile = [&sorted](double p) {
        const double at = p * static_cast<double>(sorted.size() - 1);
        const auto low = static_cast<size_t>(std::floor(at));
        const size_t high = std::min(low + 1, sorted.size() - 1);
        return sorted[low] + (sorted[high] - sorted[low]) * (at - static_cast<double>(low));
    };
    result.q1 = quantile(0.25);
    result.median = quantile(0.5);
    result.q3 = quantile(0.75);
}

void Utilities::BuildHistogram(RandomNumberResult& result, bool integers) {
    result.histogram_edges.clear();
    result.histogram_counts.clear();
    if (result.values.empty()) return;
    const double low = result.min_value, high = result.max_value;
    int bins = 20;
    double first = low, width = (high - low) / bins;
    if (integers) {
        const double values = high - low + 1.0;
        width = values <= 50.0 ? 1.0 : std::ceil(values / 20.0);
        bins = static_cast<int>(std::ceil(values / width));
        first = low - 0.5;  // a bin is centred on its integer
    } else if (!(high > low)) {
        bins = 1;
        first = low - 0.5;
        width = 1.0;
    }
    for (int i = 0; i <= bins; ++i) result.histogram_edges.push_back(first + width * i);
    result.histogram_counts.assign(static_cast<size_t>(bins), 0);
    for (double v : result.values) {
        auto bin = static_cast<int>(std::floor((v - first) / width));
        bin = std::clamp(bin, 0, bins - 1);  // the top edge belongs to the last bin
        ++result.histogram_counts[static_cast<size_t>(bin)];
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

    const auto dist = [low = min, high = max](auto& r) { return cyxwiz::portable::Uniform(r, low, high); };

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(dist(gen));
    }

    ComputeStatistics(result);
    BuildHistogram(result, false);
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

    const auto dist = [mean = mean, sd = std_dev](auto& r) { return cyxwiz::portable::Normal(r, mean, sd); };

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(dist(gen));
    }

    ComputeStatistics(result);
    BuildHistogram(result, false);
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

    const auto dist = [rate = lambda](auto& r) { return cyxwiz::portable::Exponential(r, rate); };

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(dist(gen));
    }

    ComputeStatistics(result);
    BuildHistogram(result, false);
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

    const auto dist = [mean = lambda](auto& r) { return static_cast<int>(cyxwiz::portable::Poisson(r, mean)); };

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(static_cast<double>(dist(gen)));
    }

    ComputeStatistics(result);
    BuildHistogram(result, true);
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

    const auto dist = [low = min, high = max](auto& r) { return cyxwiz::portable::UniformInt<int>(r, low, high); };

    result.values.reserve(count);
    for (int i = 0; i < count; i++) {
        result.values.push_back(static_cast<double>(dist(gen)));
    }

    ComputeStatistics(result);
    BuildHistogram(result, true);
    result.success = true;
    return result;
}

std::vector<std::string> Utilities::GenerateUUIDs(int count) {
    std::vector<std::string> uuids;

    // Every bit from the operating system's random source (random_device is
    // rand_s on Windows, getrandom / /dev/urandom on Linux and macOS). A
    // generator seeded once from 32 bits gave each call one of 2^32 streams,
    // so ids (.cyxmodel archives among them) could repeat.
    std::random_device rd;
    const auto word = [&rd] { return (static_cast<uint64_t>(rd()) << 32) | static_cast<uint64_t>(rd()); };

    for (int i = 0; i < count; i++) {
        uint64_t part1 = word();
        uint64_t part2 = word();

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

HashResult Utilities::HashText(const std::string& text, const std::string& algorithm) {
    HashResult result;
    result.input_text = text;
    result.input_size = text.size();
    result.algorithm = algorithm;
    const auto start = std::chrono::high_resolution_clock::now();
    Digests digests(algorithm, result.error_message);
    if (!result.error_message.empty()) return result;
    if (!digests.Update(text.data(), text.size()) || !digests.Finish(result)) {
        result.error_message = "The hash could not be computed";
        return result;
    }
    result.compute_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
    result.success = true;
    return result;
}

HashResult Utilities::HashFile(const std::string& file_path, const std::string& algorithm,
                               const std::function<bool(uint64_t, uint64_t)>& progress) {
    HashResult result;
    result.input_file = file_path;
    result.algorithm = algorithm;
    Digests digests(algorithm, result.error_message);
    if (!result.error_message.empty()) return result;

    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        result.error_message = "Could not open file: " + file_path;
        return result;
    }
    file.seekg(0, std::ios::end);
    const auto total = static_cast<uint64_t>(std::max<std::streamoff>(0, file.tellg()));
    file.seekg(0, std::ios::beg);

    const auto start = std::chrono::high_resolution_clock::now();
    std::vector<char> block(1 << 20);
    uint64_t done = 0;
    while (file) {
        file.read(block.data(), static_cast<std::streamsize>(block.size()));
        const auto got = static_cast<size_t>(file.gcount());
        if (got == 0) break;
        if (!digests.Update(block.data(), got)) {
            result.error_message = "The hash could not be computed";
            return result;
        }
        done += got;
        if (progress && !progress(done, total)) {
            result.error_message = "Stopped";
            return result;
        }
    }
    if (file.bad()) {
        result.error_message = "Could not read file: " + file_path;
        return result;
    }
    result.input_size = static_cast<size_t>(done);
    if (!digests.Finish(result)) {
        result.error_message = "The hash could not be computed";
        return result;
    }
    result.compute_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start).count();
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

namespace {

// Counts of each kind of value, the top-level keys and the deepest level.
void AnalyzeParsedJSON(const nlohmann::ordered_json& root, JSONResult& result) {
    std::function<void(const nlohmann::ordered_json&, int)> analyze = [&](const nlohmann::ordered_json& j, int depth) {
        result.depth = std::max(result.depth, depth);
        if (j.is_object()) {
            result.object_count++;
            for (auto& [key, value] : j.items()) {
                if (depth == 0) result.keys.push_back(key);
                analyze(value, depth + 1);
            }
        } else if (j.is_array()) {
            result.array_count++;
            for (const auto& item : j) analyze(item, depth + 1);
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
    analyze(root, 0);
}

// Parses once: formatted and minified text and the statistics, or the error
// with its line and column (from the parser's byte offset; its message no
// longer says "at position"). Keys keep the document's order (ordered_json):
// formatting must not sort them.
JSONResult ParseJSONText(const std::string& json_text, int indent_size) {
    JSONResult result;
    result.input_json = json_text;
    try {
        const auto json = nlohmann::ordered_json::parse(json_text);
        result.is_valid = true;
        result.formatted_json = json.dump(std::max(0, indent_size));
        result.minified_json = json.dump();
        AnalyzeParsedJSON(json, result);
        result.success = true;
    } catch (const nlohmann::json::parse_error& e) {
        result.is_valid = false;
        result.error_detail = e.what();
        result.error_message = "Invalid JSON";
        // byte: 1-based position of the last character read, the end of the
        // token the parser could not take. The place shown is that token's
        // start (as Python's json), and a trailing comma's own place.
        size_t stop = e.byte > 0 ? std::min(static_cast<size_t>(e.byte - 1), json_text.size()) : 0;
        if (stop < json_text.size()) {
            const char c = json_text[stop];
            const auto literal = [](char ch) {
                return std::isalnum(static_cast<unsigned char>(ch)) || ch == '.' || ch == '-' || ch == '+';
            };
            if (c == '"') {
                // The opening quote of the string that ends here.
                size_t k = stop;
                while (k > 0) {
                    --k;
                    size_t slashes = 0;
                    while (k >= slashes + 1 && json_text[k - slashes - 1] == '\\') ++slashes;
                    if (json_text[k] == '"' && slashes % 2 == 0) break;
                }
                stop = k;
            } else if (literal(c)) {
                while (stop > 0 && literal(json_text[stop - 1])) --stop;
            } else if (c == ']' || c == '}') {
                size_t k = stop;
                while (k > 0 && std::isspace(static_cast<unsigned char>(json_text[k - 1]))) --k;
                if (k > 0 && json_text[k - 1] == ',') stop = k - 1;
            }
        }
        int line = 1, column = 1;
        for (size_t i = 0; i < stop; ++i) {
            if (json_text[i] == '\n') {
                ++line;
                column = 1;
            } else if ((static_cast<unsigned char>(json_text[i]) & 0xC0) != 0x80) {
                ++column;  // characters, not UTF-8 continuation bytes
            }
        }
        result.error_line = line;
        result.error_column = column;
    }
    return result;
}

}  // namespace

JSONResult Utilities::ValidateJSON(const std::string& json_text) {
    return ParseJSONText(json_text, 2);
}

JSONResult Utilities::FormatJSON(const std::string& json_text, int indent_size) {
    return ParseJSONText(json_text, indent_size);
}

JSONResult Utilities::MinifyJSON(const std::string& json_text) {
    return ParseJSONText(json_text, 2);
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
        // ^ and $ at every line (C++17, LWG 2503; ECMAScript only).
        if (flags.find('m') != std::string::npos) {
            regex_flags |= std::regex_constants::multiline;
        }

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
        // ^ and $ at every line (C++17, LWG 2503; ECMAScript only).
        if (flags.find('m') != std::string::npos) {
            regex_flags |= std::regex_constants::multiline;
        }

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
