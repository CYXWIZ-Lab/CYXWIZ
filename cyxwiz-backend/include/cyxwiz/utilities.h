#pragma once

#include "api_export.h"
#include <vector>
#include <string>
#include <map>
#include <cstdint>

namespace cyxwiz {

// ============================================================================
// Result Structures
// ============================================================================

// Calculator Result
struct CYXWIZ_API CalculatorResult {
    double result = 0.0;
    std::string formatted_result;       // Formatted with precision
    std::string expression;             // Original expression
    std::string parsed_expression;      // Expanded/parsed form
    std::vector<std::pair<std::string, double>> variables;  // Variable values used
    bool success = false;
    std::string error_message;
};

// Unit Conversion Result
struct CYXWIZ_API UnitConversionResult {
    double input_value = 0.0;
    double output_value = 0.0;
    std::string input_unit;
    std::string output_unit;
    std::string category;               // "length", "mass", "temperature", etc.
    std::string formula;                // Conversion formula used
    std::vector<std::pair<std::string, double>> all_conversions;  // All units in category
    bool success = false;
    std::string error_message;
};

// Random Number Result
struct CYXWIZ_API RandomNumberResult {
    std::vector<double> values;         // Generated values
    double min_value = 0.0;
    double max_value = 0.0;
    double mean = 0.0;
    double std_dev = 0.0;
    std::string distribution;           // "uniform", "normal", "exponential", etc.
    uint64_t seed_used = 0;
    int count = 0;
    std::map<int, int> histogram;       // For visualization (bin -> count)
    bool success = false;
    std::string error_message;
};

// Hash Result
struct CYXWIZ_API HashResult {
    std::string input_text;
    std::string input_file;             // File path if file hashing
    size_t input_size = 0;              // Size in bytes
    std::string md5_hash;
    std::string sha1_hash;
    std::string sha256_hash;
    std::string sha512_hash;
    std::string algorithm;              // Algorithm used
    double compute_time_ms = 0.0;       // Time to compute
    bool success = false;
    std::string error_message;
};

// JSON Result
struct CYXWIZ_API JSONResult {
    std::string input_json;
    std::string formatted_json;         // Pretty-printed
    std::string minified_json;          // Compact form
    bool is_valid = false;
    int error_line = -1;
    int error_column = -1;
    std::string error_detail;
    std::vector<std::string> keys;      // Top-level keys
    int depth = 0;                      // Maximum nesting depth
    int object_count = 0;
    int array_count = 0;
    int string_count = 0;
    int number_count = 0;
    int bool_count = 0;
    int null_count = 0;
    bool success = false;
    std::string error_message;
};

// Regex Match
struct CYXWIZ_API RegexMatch {
    std::string match_text;
    int start_pos = 0;
    int end_pos = 0;
    int line_number = 0;
    std::vector<std::string> groups;    // Capture groups
};

// Regex Test Result
struct CYXWIZ_API RegexResult {
    std::string pattern;
    std::string input_text;
    std::string flags;                  // "i", "m", "g", etc.
    std::vector<RegexMatch> matches;
    int match_count = 0;
    std::string replaced_text;          // If replacement provided
    std::string replacement_pattern;
    bool is_valid_pattern = false;
    std::string pattern_error;
    bool success = false;
    std::string error_message;
};

// ============================================================================
// Utilities Class
// ============================================================================

class CYXWIZ_API Utilities {
public:
    // ==================== Calculator ====================

    /**
     * Evaluate a mathematical expression
     * @param expression Math expression (e.g., "2 + 3 * sin(pi/4)")
     * @param variables Named variables (e.g., {"x": 5.0})
     * @param angle_mode "radians" or "degrees"
     * @return CalculatorResult
     */
    static CalculatorResult Evaluate(
        const std::string& expression,
        const std::map<std::string, double>& variables = {},
        const std::string& angle_mode = "radians"
    );

    /**
     * Get list of supported functions
     * @return Map of function names to descriptions
     */
    static std::map<std::string, std::string> GetSupportedFunctions();

    /**
     * Get list of supported constants
     * @return Map of constant names to values
     */
    static std::map<std::string, double> GetSupportedConstants();

    // ==================== Unit Converter ====================

    /**
     * Convert between units
     * @param value Input value
     * @param from_unit Source unit
     * @param to_unit Target unit
     * @return UnitConversionResult
     */
    static UnitConversionResult ConvertUnit(
        double value,
        const std::string& from_unit,
        const std::string& to_unit
    );

    /**
     * Get available unit categories
     * @return List of categories (length, mass, temperature, etc.)
     */
    static std::vector<std::string> GetUnitCategories();

    /**
     * Get units for a category
     * @param category Category name
     * @return List of unit names
     */
    static std::vector<std::string> GetUnitsForCategory(const std::string& category);

    /**
     * Convert to all units in a category
     * @param value Input value
     * @param from_unit Source unit
     * @return UnitConversionResult with all_conversions populated
     */
    static UnitConversionResult ConvertToAllUnits(
        double value,
        const std::string& from_unit
    );

    // ==================== Random Number Generator ====================

    /**
     * Generate random numbers with uniform distribution
     * @param count Number of values to generate
     * @param min Minimum value
     * @param max Maximum value
     * @param seed Random seed (-1 for random)
     * @return RandomNumberResult
     */
    static RandomNumberResult GenerateUniform(
        int count,
        double min = 0.0,
        double max = 1.0,
        int64_t seed = -1
    );

    /**
     * Generate random numbers with normal distribution
     * @param count Number of values to generate
     * @param mean Mean value
     * @param std_dev Standard deviation
     * @param seed Random seed (-1 for random)
     * @return RandomNumberResult
     */
    static RandomNumberResult GenerateNormal(
        int count,
        double mean = 0.0,
        double std_dev = 1.0,
        int64_t seed = -1
    );

    /**
     * Generate random numbers with exponential distribution
     * @param count Number of values to generate
     * @param lambda Rate parameter
     * @param seed Random seed (-1 for random)
     * @return RandomNumberResult
     */
    static RandomNumberResult GenerateExponential(
        int count,
        double lambda = 1.0,
        int64_t seed = -1
    );

    /**
     * Generate random numbers with Poisson distribution
     * @param count Number of values to generate
     * @param lambda Expected value
     * @param seed Random seed (-1 for random)
     * @return RandomNumberResult
     */
    static RandomNumberResult GeneratePoisson(
        int count,
        double lambda = 1.0,
        int64_t seed = -1
    );

    /**
     * Generate random integers
     * @param count Number of values to generate
     * @param min Minimum value (inclusive)
     * @param max Maximum value (inclusive)
     * @param seed Random seed (-1 for random)
     * @return RandomNumberResult
     */
    static RandomNumberResult GenerateIntegers(
        int count,
        int min = 0,
        int max = 100,
        int64_t seed = -1
    );

    /**
     * Generate UUID v4
     * @param count Number of UUIDs to generate
     * @return Vector of UUID strings
     */
    static std::vector<std::string> GenerateUUIDs(int count = 1);

    // ==================== Hash Generator ====================

    /**
     * Compute hash of text
     * @param text Input text
     * @param algorithm "md5", "sha1", "sha256", "sha512", "all"
     * @return HashResult
     */
    static HashResult HashText(
        const std::string& text,
        const std::string& algorithm = "sha256"
    );

    /**
     * Compute hash of file
     * @param file_path Path to file
     * @param algorithm "md5", "sha1", "sha256", "sha512", "all"
     * @return HashResult
     */
    static HashResult HashFile(
        const std::string& file_path,
        const std::string& algorithm = "sha256"
    );

    /**
     * Verify hash matches expected value
     * @param text Input text
     * @param expected_hash Expected hash value
     * @param algorithm Algorithm to use
     * @return true if hashes match
     */
    static bool VerifyHash(
        const std::string& text,
        const std::string& expected_hash,
        const std::string& algorithm = "sha256"
    );

    // ==================== JSON Viewer ====================

    /**
     * Validate and analyze JSON
     * @param json_text Input JSON string
     * @return JSONResult
     */
    static JSONResult ValidateJSON(const std::string& json_text);

    /**
     * Format JSON with indentation
     * @param json_text Input JSON string
     * @param indent_size Number of spaces for indentation
     * @return JSONResult with formatted_json
     */
    static JSONResult FormatJSON(
        const std::string& json_text,
        int indent_size = 2
    );

    /**
     * Minify JSON (remove whitespace)
     * @param json_text Input JSON string
     * @return JSONResult with minified_json
     */
    static JSONResult MinifyJSON(const std::string& json_text);

    /**
     * Extract value at JSON path
     * @param json_text Input JSON string
     * @param path JSON path (e.g., "data.items[0].name")
     * @return String representation of value
     */
    static std::string GetJSONValue(
        const std::string& json_text,
        const std::string& path
    );

    // ==================== Regex Tester ====================

    /**
     * Test regex pattern against text
     * @param pattern Regex pattern
     * @param text Input text to search
     * @param flags Regex flags ("i"=ignore case, "m"=multiline)
     * @return RegexResult
     */
    static RegexResult TestRegex(
        const std::string& pattern,
        const std::string& text,
        const std::string& flags = ""
    );

    /**
     * Replace matches with replacement string
     * @param pattern Regex pattern
     * @param text Input text
     * @param replacement Replacement string (supports $1, $2, etc.)
     * @param flags Regex flags
     * @return RegexResult with replaced_text
     */
    static RegexResult ReplaceRegex(
        const std::string& pattern,
        const std::string& text,
        const std::string& replacement,
        const std::string& flags = ""
    );

    /**
     * Validate regex pattern
     * @param pattern Regex pattern to validate
     * @return true if pattern is valid
     */
    static bool IsValidRegex(const std::string& pattern);

    /**
     * Get common regex patterns
     * @return Map of pattern name to pattern string
     */
    static std::map<std::string, std::string> GetCommonPatterns();

private:
    // Expression parser state
    struct ParserState {
        std::string expression;
        size_t pos = 0;
        std::map<std::string, double> variables;
        std::string angle_mode;
        std::string error;
    };

    // Expression parser helpers (recursive descent)
    static double ParseExpression(ParserState& state);
    static double ParseTerm(ParserState& state);
    static double ParseFactor(ParserState& state);
    static double ParsePrimary(ParserState& state);
    static double ParseNumber(ParserState& state);
    static double ParseFunction(ParserState& state, const std::string& name);
    static void SkipWhitespace(ParserState& state);
    static char Peek(ParserState& state);
    static char Get(ParserState& state);
    static bool Match(ParserState& state, char c);

    // Unit conversion helpers
    static std::string GetCategoryForUnit(const std::string& unit);
    static double ConvertToBase(double value, const std::string& unit);
    static double ConvertFromBase(double value, const std::string& unit);

    // Hash implementation helpers
    static std::string ComputeMD5(const std::vector<uint8_t>& data);
    static std::string ComputeSHA1(const std::vector<uint8_t>& data);
    static std::string ComputeSHA256(const std::vector<uint8_t>& data);
    static std::string ComputeSHA512(const std::vector<uint8_t>& data);
    static std::string BytesToHex(const std::vector<uint8_t>& bytes);

    // JSON helpers
    static void AnalyzeJSON(const std::string& json, JSONResult& result);

    // Random number helpers
    static void ComputeStatistics(RandomNumberResult& result);
    static void BuildHistogram(RandomNumberResult& result, int num_bins = 20);
};

} // namespace cyxwiz
