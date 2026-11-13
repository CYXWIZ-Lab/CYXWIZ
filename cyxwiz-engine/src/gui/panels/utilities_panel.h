#pragma once

// Tools > Utilities (TOFIX129 D12): one window with six tabs over
// cyxwiz::Utilities, shown through core/utilities_presentation. Calculator
// and Units answer at once; Random, Hash, JSON and Regex run in the
// background with Stop. The menu items open the window on their tab; the
// user's defaults (tab, digits, angles, unit category, hash algorithm, JSON
// indent, regex flags) come from Preferences and "Save as my default".

#include "../../core/utilities_presentation.h"
#include "../data_studio/explore_tab.h"
#include "../preview/json_tree.h"

#include <cyxwiz/utilities.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class UtilitiesPanel {
public:
    UtilitiesPanel();

    void Render();
    bool IsVisible() const { return visible_; }
    void SetVisible(bool v) { visible_ = v; }
    // Opens the window on a tab ("calculator", "units", "random", "hash",
    // "json", "regex").
    void Open(const std::string& tab_id);

private:
    struct RandomJob {
        RandomNumberResult result;
        std::vector<double> parameters;
        std::string error;
        bool ok() const { return error.empty(); }
    };
    struct HashJob {
        HashResult result;
        std::string algorithm;
        std::string error;
        bool ok() const { return error.empty(); }
    };
    struct JsonJob {
        JSONResult result;
        std::string action;  // "check", "format", "minify"
        std::string path_value;
        std::string error;
        bool ok() const { return error.empty(); }
    };
    struct RegexJob {
        RegexResult result;
        RegexResult replaced;
        std::string text;
        std::string error;
        bool ok() const { return error.empty(); }
    };

    void RenderCalculator();
    void RenderUnits();
    void RenderRandom();
    void RenderHash();
    void RenderJson();
    void RenderRegex();
    void RenderDefaultsLink();
    void Evaluate();
    void ConvertUnits();
    void SaveDefaults();
    void SaveCsv(const std::string& name, const std::string& text);
    void SendToDataStudio(const RandomJob& job);

    bool visible_ = false;
    utilities_view::Tab tab_ = utilities_view::Tab::Calculator;
    bool select_tab_ = false;
    int digits_ = 10;
    std::string message_;
    bool message_error_ = false;

    // Calculator.
    std::string expression_ = "2 * pi + sqrt(16)";
    bool degrees_ = false;
    CalculatorResult calc_;
    struct HistoryEntry {
        std::string expression;
        double value = 0.0;
    };
    std::vector<HistoryEntry> history_;  // newest first, at most 50
    std::map<std::string, double> variables_;
    std::string new_name_, new_value_;
    std::string variable_error_;

    // Units.
    std::string unit_category_ = "length";
    std::string unit_value_ = "1";
    std::string from_unit_ = "centimeter", to_unit_ = "foot";
    UnitConversionResult conversion_, all_;

    // Random.
    int distribution_ = 1;  // Uniform, Normal, Exponential, Poisson, Integers
    double a_[5] = {0.0, 10.0, 1.0, 3.5, 0.0};
    double b_[5] = {1.0, 2.0, 0.0, 0.0, 100.0};
    int count_ = 1000;
    int seed_ = 42;
    int uuid_count_ = 5;
    std::vector<std::string> uuids_;
    ExploreJob<RandomJob> random_job_;
    std::string random_key_;
    int random_runs_ = 0;

    // Hash.
    int hash_source_ = 0;  // 0 text, 1 file
    std::string hash_text_ = "Hello, World!";
    std::string hash_file_;
    std::string hash_algorithm_ = "sha256";
    std::string expected_hash_;
    ExploreJob<HashJob> hash_job_;
    std::string hash_key_;
    int hash_runs_ = 0;

    // JSON.
    std::string json_text_;
    int json_indent_ = 2;
    std::string json_path_;
    ExploreJob<JsonJob> json_job_;
    std::string json_key_;
    int json_runs_ = 0;
    gui::JSONTreeRenderer json_tree_;
    std::string json_tree_for_;  // the text the tree shows

    // Regex.
    std::string pattern_ = "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}\\b";
    std::string regex_text_;
    std::string replacement_ = "[$&]";
    bool ignore_case_ = false;
    bool multiline_ = false;
    ExploreJob<RegexJob> regex_job_;
    std::string regex_key_;
    int regex_runs_ = 0;
    std::string regex_rows_key_;                 // the answer regex_rows_ were built from
    std::vector<utilities_view::RegexRow> regex_rows_;
};

}  // namespace cyxwiz
