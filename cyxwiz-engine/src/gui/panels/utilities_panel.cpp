#include "utilities_panel.h"

#include "../../core/data_registry.h"
#include "../../core/engine_config.h"
#include "../../core/file_dialogs.h"
#include "../icons.h"
#include "../ui_buttons.h"
#include "../ui_fonts.h"
#include "../ui_tokens.h"
#include "../ui_widgets.h"

#include <arrow/api.h>
#include <imgui.h>
#include <imgui_stdlib.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>

namespace cyxwiz {

namespace view = utilities_view;
using view::Tab;

namespace {

constexpr const char* kSampleJson = R"({
  "name": "CyxWiz",
  "version": "1.0.0",
  "features": ["Visual Node Editor", "GPU Acceleration"],
  "config": {
    "backend": "ArrayFire",
    "gpu_enabled": true,
    "max_workers": 10
  }
})";

constexpr const char* kSampleText = R"(Contact Information:
John Doe - john.doe@example.com - (555) 123-4567
Jane Smith - jane_smith123@company.org - 555.987.6543
Support Team - support@help-desk.io)";

constexpr const char* kDistributions[] = {"Uniform", "Normal", "Exponential", "Poisson", "Integers"};
constexpr const char* kAlgorithms[] = {"md5", "sha1", "sha256", "sha512"};

// A number typed in full: digits, a sign, a decimal point, an exponent.
bool ParseNumber(const std::string& text, double& value) {
    const char* begin = text.c_str();
    char* end = nullptr;
    value = std::strtod(begin, &end);
    if (end == begin) return false;
    while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end))) ++end;
    return *end == '\0' && std::isfinite(value);
}

bool IsIdentifier(const std::string& name) {
    if (name.empty() || !(std::isalpha(static_cast<unsigned char>(name[0])) || name[0] == '_')) return false;
    return std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isalnum(c) || c == '_'; });
}

void Copy(const std::string& text) { ImGui::SetClipboardText(text.c_str()); }

// A heading in the dim tone, as the other D windows label their areas.
void Label(const char* text) {
    ImGui::TextColored(ui::CurrentTokens().text_dim, "%s", text);
}

void BigNumber(const std::string& text) {
    const ui::FontScope heading(ui::Font::Heading);
    ImGui::TextUnformatted(text.c_str());
}

// A UTF-8 path (from the file dialogs) as a filesystem path.
std::filesystem::path Utf8Path(const std::string& s) { return std::filesystem::path(std::u8string(s.begin(), s.end())); }

// What to do next, in a table column (ui::EmptyState centres on the window).
void ColumnHint(const char* title, const char* hint) {
    const auto& t = ui::CurrentTokens();
    ImGui::Spacing();
    ImGui::TextColored(t.text_dim, "%s", title);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "%s", hint);
    ImGui::PopTextWrapPos();
}

// Widths for ui::FlowRow, so a row of controls wraps in a narrow (docked)
// window instead of running past its edge.
float TextWidth(const char* text) { return ImGui::CalcTextSize(text).x; }
float NumberWidth(float field) {
    return field + 2.0f * (ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.x);
}
float CheckboxWidth(const char* label) {
    return ImGui::GetFrameHeight() + ImGui::GetStyle().ItemInnerSpacing.x + TextWidth(label);
}
float LabelledWidth(const char* label, float control) {
    return TextWidth(label) + ImGui::GetStyle().ItemSpacing.x + control;
}

// Narrower than this, a tab's two areas stack (a docked window can be ~350 px).
constexpr float kSideBySideWidth = 640.0f;

// Two areas side by side, or the second under the first when `narrow`.
template <typename Left, typename Right>
void SideBySide(const char* id, ImGuiTableFlags flags, bool narrow, Left left, Right right) {
    if (narrow) {
        left();
        ImGui::Spacing();
        right();
        return;
    }
    if (!ImGui::BeginTable(id, 2, flags)) return;
    ImGui::TableNextColumn();
    left();
    ImGui::TableNextColumn();
    right();
    ImGui::EndTable();
}

// The running job's progress bar and Stop; true when it is running.
template <typename T>
bool Running(ExploreJob<T>& job) {
    if (!job.Running()) return false;
    std::string message;
    const float progress = job.Progress(&message);
    ImGui::ProgressBar(progress, ImVec2(-ui::ButtonWidth("Stop", ui::ButtonSize::Small) - 8.0f, 0.0f),
                       message.empty() ? nullptr : message.c_str());
    ImGui::SameLine();
    if (ui::SecondaryButton("Stop")) job.Cancel();
    return true;
}

}  // namespace

UtilitiesPanel::UtilitiesPanel() {
    const core::UtilitiesConfig defaults = core::EngineConfig::Instance().GetUtilities();
    tab_ = view::TabFromId(defaults.tab);
    digits_ = std::clamp(defaults.digits, 1, 17);
    degrees_ = defaults.degrees;
    // A saved category other than the shipped one starts on its first two units.
    if (defaults.unit_category != unit_category_) {
        const auto units = Utilities::GetUnitsForCategory(defaults.unit_category);
        if (units.size() >= 2) {
            unit_category_ = defaults.unit_category;
            from_unit_ = units[0];
            to_unit_ = units[1];
        }
    }
    if (std::find(std::begin(kAlgorithms), std::end(kAlgorithms), defaults.hash_algorithm) != std::end(kAlgorithms))
        hash_algorithm_ = defaults.hash_algorithm;
    json_indent_ = std::clamp(defaults.json_indent, 0, 8);
    ignore_case_ = defaults.regex_ignore_case;
    multiline_ = defaults.regex_multiline;
    json_text_ = kSampleJson;
    regex_text_ = kSampleText;
    Evaluate();
    ConvertUnits();
    uuids_ = Utilities::GenerateUUIDs(uuid_count_);
}

void UtilitiesPanel::Open(const std::string& tab_id) {
    visible_ = true;
    tab_ = view::TabFromId(tab_id);
    select_tab_ = true;
    ImGui::SetWindowFocus("###Utilities");
}

void UtilitiesPanel::Render() {
    if (!visible_) return;
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowSize(ImVec2(std::min(1040.0f, viewport->WorkSize.x * 0.9f), std::min(700.0f, viewport->WorkSize.y * 0.9f)),
                             ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowPos(viewport->GetWorkCenter(), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    if (!ImGui::Begin(ICON_FA_TOOLBOX " Utilities###Utilities", &visible_)) {
        ImGui::End();
        return;
    }
    RenderDefaultsLink();
    if (ImGui::BeginTabBar("##utilities_tabs")) {
        for (const Tab tab : view::kTabs) {
            const ImGuiTabItemFlags flags = select_tab_ && tab == tab_ ? ImGuiTabItemFlags_SetSelected : 0;
            if (!ImGui::BeginTabItem(view::TabLabel(tab), nullptr, flags)) continue;
            if (!select_tab_) tab_ = tab;
            switch (tab) {
                case Tab::Calculator: RenderCalculator(); break;
                case Tab::Units: RenderUnits(); break;
                case Tab::Random: RenderRandom(); break;
                case Tab::Hash: RenderHash(); break;
                case Tab::Json: RenderJson(); break;
                case Tab::Regex: RenderRegex(); break;
            }
            ImGui::EndTabItem();
        }
        select_tab_ = false;
        ImGui::EndTabBar();
    }
    ImGui::End();
}

// One row above the tabs: the last message, and "Save as my default" at the
// right (under a tab it fell below the results).
void UtilitiesPanel::RenderDefaultsLink() {
    const auto& t = ui::CurrentTokens();
    if (!message_.empty()) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextColored(message_error_ ? t.error : t.success, "%s", message_.c_str());
        ImGui::SameLine();
        if (ui::InlineLink("OK")) message_.clear();
    } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("");
    }
    ui::SameLineRight(ui::ButtonWidth("Save as my default", ui::ButtonSize::Small));
    if (ui::LinkButton("Save as my default")) SaveDefaults();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip("This tab, the digits, angle unit, unit category, hash algorithm, JSON indent and regex\n"
                          "flags open this way next time (also in Preferences > General).");
}

void UtilitiesPanel::SaveDefaults() {
    core::UtilitiesConfig c;
    c.tab = view::TabId(tab_);
    c.digits = digits_;
    c.degrees = degrees_;
    c.unit_category = unit_category_;
    c.hash_algorithm = hash_algorithm_;
    c.json_indent = json_indent_;
    c.regex_ignore_case = ignore_case_;
    c.regex_multiline = multiline_;
    auto& config = core::EngineConfig::Instance();
    config.SetUtilities(c);
    message_error_ = !config.Save();
    message_ = message_error_ ? "Could not save the Engine settings." : "Saved as your default.";
}

void UtilitiesPanel::SaveCsv(const std::string& name, const std::string& text) {
    if (const auto path = FileDialogs::SaveFile("Save as CSV", {{"CSV", "csv"}}, nullptr, name.c_str())) {
        std::ofstream out(Utf8Path(*path), std::ios::binary);
        out << text;
        message_error_ = !out;
        message_ = out ? "Saved " + *path : "Could not write " + *path;
    }
}

// ---------------------------------------------------------------- Calculator

void UtilitiesPanel::Evaluate() {
    calc_ = Utilities::Evaluate(expression_, variables_, degrees_ ? "degrees" : "radians");
    if (!calc_.success) return;
    if (history_.empty() || history_.front().expression != expression_) {
        history_.insert(history_.begin(), {expression_, calc_.result});
        if (history_.size() > 50) history_.pop_back();
    }
}

void UtilitiesPanel::RenderCalculator() {
    const auto& t = ui::CurrentTokens();
    const bool narrow = ImGui::GetContentRegionAvail().x < kSideBySideWidth;
    const float button = ui::ButtonWidth("Evaluate", ui::ButtonSize::Regular);
    ImGui::SetNextItemWidth(-button - ImGui::GetStyle().ItemSpacing.x);
    const bool enter = ImGui::InputTextWithHint("##expression", "An expression, e.g. 2 * pi + sqrt(16)", &expression_,
                                                ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ui::PrimaryButton("Evaluate") || enter) Evaluate();

    const char* angles[] = {"Radians", "Degrees"};
    int angle = degrees_ ? 1 : 0;
    ui::FlowRow row;
    row.Next(LabelledWidth("Angles", ui::SegmentedControlWidth(angles, 2)));
    ImGui::AlignTextToFramePadding();
    Label("Angles");
    ImGui::SameLine();
    if (ui::SegmentedControl("##angles", angles, 2, &angle)) {
        degrees_ = angle == 1;
        Evaluate();
    }
    row.Next(LabelledWidth("Digits", NumberWidth(40.0f)));
    ImGui::AlignTextToFramePadding();
    Label("Digits");
    ImGui::SameLine();
    ui::NumberField("##digits", &digits_, 1, 17, 1, 40.0f, "Significant digits shown (1 to 17)");

    SideBySide("##calc", ImGuiTableFlags_SizingStretchProp, narrow,
        [&] {
        Label("Result");
        if (calc_.success) {
            BigNumber(view::FormatNumber(calc_.result, digits_));
            ImGui::TextColored(t.text_dim, "%s in full", view::FullNumber(calc_.result).c_str());
            ImGui::SameLine();
            if (ui::InlineLink("Copy")) Copy(view::FullNumber(calc_.result));
        } else if (!calc_.error_message.empty()) {
            ui::NoticeLine(calc_.error_message, t.error);
        }
        ImGui::Spacing();
        Label("Variables");
        std::string remove;
        for (const auto& [name, value] : variables_) {
            ImGui::PushID(name.c_str());
            ImGui::Text("%s = %s", name.c_str(), view::FullNumber(value).c_str());
            ImGui::SameLine();
            if (ui::InlineLink("Remove")) remove = name;
            ImGui::PopID();
        }
        if (!remove.empty()) {
            variables_.erase(remove);
            Evaluate();
        }
        ui::FlowRow variable_row;
        variable_row.Next(90.0f);
        ImGui::SetNextItemWidth(90.0f);
        ImGui::InputTextWithHint("##var_name", "name", &new_name_);
        variable_row.Next(120.0f);
        ImGui::SetNextItemWidth(120.0f);
        ImGui::InputTextWithHint("##var_value", "value", &new_value_);
        variable_row.Next(ui::ButtonWidth("Add", ui::ButtonSize::Small));
        if (ui::SecondaryButton("Add", !new_name_.empty() && !new_value_.empty())) {
            const auto value = Utilities::Evaluate(new_value_, variables_, degrees_ ? "degrees" : "radians");
            if (!IsIdentifier(new_name_)) {
                variable_error_ = "A name starts with a letter and has only letters, digits and _.";
            } else if (!value.success) {
                variable_error_ = "The value: " + value.error_message;
            } else {
                variables_[new_name_] = value.result;
                new_name_.clear();
                new_value_.clear();
                variable_error_.clear();
                Evaluate();
            }
        }
        if (!variable_error_.empty()) ui::NoticeLine(variable_error_, t.error);

        },
        [&] {
        Label("History");
        if (history_.empty()) ImGui::TextColored(t.text_faint, "%s", "Nothing evaluated yet.");
        for (size_t i = 0; i < history_.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            const std::string line = history_[i].expression + "  =  " + view::FormatNumber(history_[i].value, digits_);
            if (ImGui::Selectable(line.c_str())) {
                expression_ = history_[i].expression;
                Evaluate();
            }
            if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", "Use this expression again");
            ImGui::PopID();
        }
        ImGui::Spacing();
        if (ImGui::CollapsingHeader("Functions and constants")) {
            std::string functions;
            for (const auto& [name, help] : Utilities::GetSupportedFunctions()) functions += name + "  ";
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(t.text_dim, "%s", functions.c_str());
            std::string constants;
            for (const auto& [name, value] : Utilities::GetSupportedConstants())
                constants += name + " = " + view::FullNumber(value) + "   ";
            ImGui::TextColored(t.text_dim, "%s", constants.c_str());
            ImGui::TextColored(t.text_faint, "%s", "The same parser as Optimization and Calculus and pycyxwiz.calculus.Expression.");
            ImGui::PopTextWrapPos();
        }
        });
}

// ---------------------------------------------------------------- Units

void UtilitiesPanel::ConvertUnits() {
    double value = 0.0;
    if (!ParseNumber(unit_value_, value)) {
        conversion_ = UnitConversionResult();
        conversion_.error_message = "Type a number, e.g. 1.5 or 2e-6.";
        all_ = UnitConversionResult();
        return;
    }
    conversion_ = Utilities::ConvertUnit(value, from_unit_, to_unit_);
    all_ = Utilities::ConvertToAllUnits(value, from_unit_);
}

void UtilitiesPanel::RenderUnits() {
    const auto& t = ui::CurrentTokens();
    const bool narrow = ImGui::GetContentRegionAvail().x < kSideBySideWidth;
    ui::FlowRow categories;
    for (const auto& category : Utilities::GetUnitCategories()) {
        const std::string label = view::CategoryLabel(category);
        categories.Next(ui::ButtonWidth(label.c_str(), ui::ButtonSize::Small));
        ImGui::PushID(category.c_str());
        if (ui::GhostButton(label.c_str(), true, nullptr, category == unit_category_)) {
            unit_category_ = category;
            const auto units = Utilities::GetUnitsForCategory(category);
            from_unit_ = units.at(0);
            to_unit_ = units.at(units.size() > 1 ? 1 : 0);
            ConvertUnits();
        }
        ImGui::PopID();
    }

    const auto units = view::UnitsOf(unit_category_);
    const auto unit_combo = [&](const char* id, std::string& chosen) {
        const UnitDefinition* current = Utilities::FindUnit(chosen, unit_category_);
        ImGui::SetNextItemWidth(250.0f);
        bool changed = false;
        if (ImGui::BeginCombo(id, current ? view::UnitLabel(*current).c_str() : chosen.c_str())) {
            for (const UnitDefinition* u : units) {
                if (ImGui::Selectable(view::UnitLabel(*u).c_str(), u == current)) {
                    chosen = u->id;
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        return changed;
    };

    SideBySide("##units", ImGuiTableFlags_SizingStretchProp, narrow,
        [&] {
        Label("Convert");
        ui::FlowRow convert_row;
        convert_row.Next(140.0f);
        ImGui::SetNextItemWidth(140.0f);
        bool changed = ImGui::InputText("##unit_value", &unit_value_);
        convert_row.Next(250.0f);
        changed |= unit_combo("##from", from_unit_);
        ui::FlowRow to_row;
        to_row.Next(ui::ButtonWidth("Swap", ui::ButtonSize::Small));
        if (ui::SecondaryButton("Swap")) {
            std::swap(from_unit_, to_unit_);
            changed = true;
        }
        to_row.Next(TextWidth("to"));
        ImGui::AlignTextToFramePadding();
        Label("to");
        to_row.Next(250.0f);
        changed |= unit_combo("##to", to_unit_);
        if (changed) ConvertUnits();
        ImGui::Spacing();
        if (conversion_.success) {
            BigNumber(view::FormatNumber(conversion_.output_value, digits_));
            if (const auto* u = Utilities::FindUnit(conversion_.output_unit)) {
                ImGui::SameLine();
                ImGui::TextColored(t.text_dim, "%s", u->symbol.c_str());
            }
            ImGui::TextColored(t.text_dim, "%s", conversion_.formula.c_str());
            if (ui::InlineLink("Copy")) Copy(view::FullNumber(conversion_.output_value));
        } else if (!conversion_.error_message.empty()) {
            ui::NoticeLine(conversion_.error_message, t.error);
        }
        ImGui::Spacing();
        Label("Digits");
        ImGui::SameLine();
        ui::NumberField("##unit_digits", &digits_, 1, 17, 1, 40.0f, "Significant digits shown (1 to 17)");

        },
        [&] {
        const UnitDefinition* from = Utilities::FindUnit(from_unit_, unit_category_);
        Label((unit_value_ + " " + (from ? from->symbol : from_unit_) + " in every " + unit_category_ + " unit").c_str());
        if (all_.success && ImGui::BeginTable("##all", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            for (const auto& [id, value] : all_.all_conversions) {
                const UnitDefinition* u = Utilities::FindUnit(id, unit_category_);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(u ? view::UnitLabel(*u).c_str() : id.c_str());
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(view::FormatNumber(value, digits_).c_str());
            }
            ImGui::EndTable();
        }
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(t.text_faint, "%s", view::CategoryNote(unit_category_).c_str());
        ImGui::PopTextWrapPos();
        });
}

// ---------------------------------------------------------------- Random

void UtilitiesPanel::RenderRandom() {
    const auto& t = ui::CurrentTokens();
    // Narrow: the settings above the results instead of a column beside them.
    const bool narrow = ImGui::GetContentRegionAvail().x < kSideBySideWidth;
    const auto begin_results = [narrow] {
        if (narrow) ImGui::Spacing();
        else ui::BeginResultsArea();
    };
    const auto end_results = [narrow] {
        if (!narrow) ui::EndResultsArea();
    };
    if (!narrow) ui::BeginSettingsColumn("utilities_random", 280.0f);
    ui::SettingsSection("Draw");
    ImGui::SetNextItemWidth(-1);
    ImGui::Combo("##distribution", &distribution_, kDistributions, IM_ARRAYSIZE(kDistributions));
    const int d = distribution_;
    const auto field = [](const char* label, const char* id, double* v, double min, double step, double max = 1e300) {
        ui::FieldLabel(label);
        ui::NumberFieldOptions o;
        o.min = min;
        o.max = max;
        o.step = step;
        o.format = "%.6g";
        o.width = 120.0f;
        ui::NumberField(id, v, o);
    };
    switch (d) {
        case 0: field("Low", "##a", &a_[0], -1e300, 1.0); field("High", "##b", &b_[0], -1e300, 1.0); break;
        case 1: field("Mean", "##a", &a_[1], -1e300, 1.0); field("SD", "##b", &b_[1], 1e-300, 0.5); break;
        case 2: field("Rate (1 / mean)", "##a", &a_[2], 1e-300, 0.1); break;
        case 3: field("Mean", "##a", &a_[3], 1e-300, 1.0, 1e9); break;
        default:
            // Whole numbers that fit an int.
            field("Low", "##a", &a_[4], -1e9, 1.0, 1e9);
            field("High", "##b", &b_[4], -1e9, 1.0, 1e9);
            a_[4] = std::round(a_[4]);
            b_[4] = std::round(b_[4]);
            break;
    }
    ui::FieldLabel("Count");
    ui::NumberField("##count", &count_, 1, 1000000, 100, 120.0f);
    ui::FieldLabel("Seed");
    ui::NumberField("##seed", &seed_, 0, 2147483647, 1, 120.0f, "The same seed draws the same values on every platform");
    ImGui::SameLine();
    if (ui::LinkButton("New seed")) seed_ = static_cast<int>(std::random_device{}() & 0x7FFFFFFF);
    std::string why;
    if ((d == 0 && !(b_[0] > a_[0])) || (d == 4 && b_[4] < a_[4])) why = "High must be above Low.";
    if (ui::PrimaryButton("Generate", why.empty(), why.c_str(), ui::ButtonSize::Regular, -1.0f)) {
        random_key_ = "random#" + std::to_string(++random_runs_);
        const int dist = d, count = count_, seed = seed_;
        const double a = a_[d], b = b_[d];
        random_job_.Ask("Utilities: random values", random_key_, [dist, count, seed, a, b](LambdaTask&) {
            RandomJob job;
            switch (dist) {
                case 0: job.result = Utilities::GenerateUniform(count, a, b, seed); job.parameters = {a, b}; break;
                case 1: job.result = Utilities::GenerateNormal(count, a, b, seed); job.parameters = {a, b}; break;
                case 2: job.result = Utilities::GenerateExponential(count, a, seed); job.parameters = {a}; break;
                case 3: job.result = Utilities::GeneratePoisson(count, a, seed); job.parameters = {a}; break;
                default:
                    job.result = Utilities::GenerateIntegers(count, static_cast<int>(a), static_cast<int>(b), seed);
                    job.parameters = {a, b};
                    break;
            }
            if (!job.result.success) job.error = job.result.error_message;
            return job;
        });
    }
    ImGui::Spacing();
    ui::SettingsSection("UUIDs (version 4)");
    ui::NumberField("##uuid_count", &uuid_count_, 1, 100, 1, 60.0f);
    ImGui::SameLine();
    if (ui::SecondaryButton("New UUIDs")) uuids_ = Utilities::GenerateUUIDs(uuid_count_);
    for (const auto& id : uuids_) ImGui::TextUnformatted(id.c_str());
    if (ui::InlineLink("Copy UUIDs")) {
        std::string all;
        for (const auto& id : uuids_) all += id + "\n";
        Copy(all);
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "%s", "From the operating system's random source (128 bits each), not from the seed.");
    ImGui::PopTextWrapPos();

    begin_results();
    if (Running(random_job_)) {
        end_results();
        return;
    }
    if (!random_job_.Error().empty()) ui::NoticeLine(random_job_.Error(), t.error);
    const RandomJob* job = random_job_.Answer(random_key_);
    if (!job) {
        if (narrow) ColumnHint("No values yet", "Pick a distribution, then Generate.");
        else ui::EmptyState(ICON_FA_DICE, "No values yet", "Pick a distribution, then Generate.");
        end_results();
        return;
    }
    const auto& r = job->result;
    Label(view::RandomTitle(r, job->parameters).c_str());
    // Histogram: bars to scale, the bin edges under them.
    const float height = 170.0f;
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    ImGui::Dummy(ImVec2(width, height + 18.0f));
    const bool over_histogram = ImGui::IsItemHovered();  // not under a popup or another window
    const int max_count = r.histogram_counts.empty() ? 1 : std::max(1, *std::max_element(r.histogram_counts.begin(), r.histogram_counts.end()));
    ImDrawList* draw = ImGui::GetWindowDrawList();
    draw->AddRectFilled(origin, ImVec2(origin.x + width, origin.y + height), ui::ToU32(t.plot_bg));
    const size_t bins = r.histogram_counts.size();
    const float bar = bins ? width / static_cast<float>(bins) : width;
    for (size_t i = 0; i < bins; ++i) {
        const float h = height * static_cast<float>(r.histogram_counts[i]) / static_cast<float>(max_count);
        const ImVec2 a(origin.x + bar * static_cast<float>(i) + 1.0f, origin.y + height - h);
        const ImVec2 b(origin.x + bar * static_cast<float>(i + 1) - 1.0f, origin.y + height);
        draw->AddRectFilled(a, b, ui::ToU32(t.series[0]));
        if (over_histogram && ImGui::IsMouseHoveringRect(ImVec2(a.x, origin.y), b))
            ImGui::SetTooltip("%s to %s: %d", view::FormatNumber(r.histogram_edges[i], 6).c_str(),
                              view::FormatNumber(r.histogram_edges[i + 1], 6).c_str(), r.histogram_counts[i]);
    }
    const size_t label_every = std::max<size_t>(1, bins / 6);
    for (size_t i = 0; i <= bins; i += label_every) {
        const std::string edge = view::FormatNumber(r.histogram_edges[i], 4);
        const float x = origin.x + bar * static_cast<float>(i) - ImGui::CalcTextSize(edge.c_str()).x * 0.5f;
        draw->AddText(ImVec2(std::max(origin.x, x), origin.y + height + 2.0f), ui::ToU32(t.text_faint), edge.c_str());
    }
    std::vector<ui::KeyValue> rows;
    for (const auto& f : view::RandomFigures(r, digits_)) rows.push_back({f.label, f.value});
    ui::KeyValueTable("##random_figures", rows, 2, true);
    std::string first;
    for (size_t i = 0; i < std::min<size_t>(5, r.values.size()); ++i) first += view::FormatNumber(r.values[i], digits_) + "   ";
    ImGui::TextColored(t.text_dim, "First values  %s", first.c_str());
    if (ui::SecondaryButton("Copy values")) Copy(view::ValuesText(r.values, ""));
    ImGui::SameLine();
    if (ui::SecondaryButton("Save as CSV...")) {
        SaveCsv("random_" + r.distribution + "_seed" + std::to_string(r.seed_used) + ".csv", view::ValuesText(r.values, "value"));
    }
    ImGui::SameLine();
    if (ui::SecondaryButton("Send to Data Studio")) SendToDataStudio(*job);
    end_results();
}

void UtilitiesPanel::SendToDataStudio(const RandomJob& job) {
    arrow::DoubleBuilder builder;
    std::shared_ptr<arrow::Array> column;
    if (!builder.AppendValues(job.result.values).ok() || !builder.Finish(&column).ok()) {
        message_ = "Could not build the table.";
        message_error_ = true;
        return;
    }
    const auto table = arrow::Table::Make(arrow::schema({arrow::field("value", arrow::float64())}), {column});
    const std::string base = "random_" + job.result.distribution + "_seed" + std::to_string(job.result.seed_used);
    std::string name = base;
    for (int n = 2; DataRegistry::Instance().GetArrowDataset(name); ++n) name = base + "_" + std::to_string(n);
    const auto dataset = DataRegistry::Instance().RegisterArrowTable(table, name);
    message_error_ = dataset == nullptr;
    message_ = dataset ? "Added the table " + name + " (" + view::FormatCount(job.result.values.size()) +
                             " rows, column value) to this session's data."
                       : "Could not add the table.";
}

// ---------------------------------------------------------------- Hash

void UtilitiesPanel::RenderHash() {
    const auto& t = ui::CurrentTokens();
    const bool narrow = ImGui::GetContentRegionAvail().x < kSideBySideWidth;
    const char* sources[] = {"Text", "File"};
    ui::FlowRow row;
    row.Next(ui::SegmentedControlWidth(sources, 2));
    ui::SegmentedControl("##hash_source", sources, 2, &hash_source_);
    row.Next(130.0f);
    ImGui::SetNextItemWidth(130.0f);
    if (ImGui::BeginCombo("##algorithm", view::AlgorithmLabel(hash_algorithm_).c_str())) {
        for (const char* a : kAlgorithms)
            if (ImGui::Selectable(view::AlgorithmLabel(a).c_str(), hash_algorithm_ == a)) hash_algorithm_ = a;
        ImGui::EndCombo();
    }
    row.Next(ui::ButtonWidth("Hash", ui::ButtonSize::Regular));
    const bool can = hash_source_ == 0 || !hash_file_.empty();
    if (ui::PrimaryButton("Hash", can, "Choose a file first") ) {
        hash_key_ = "hash#" + std::to_string(++hash_runs_);
        const bool file = hash_source_ == 1;
        const std::string input = file ? hash_file_ : hash_text_, algorithm = hash_algorithm_;
        hash_job_.Ask("Utilities: hash", hash_key_, [file, input, algorithm](LambdaTask& task) {
            HashJob job;
            job.algorithm = algorithm;
            if (file) {
                job.result = Utilities::HashFile(input, algorithm, [&task](uint64_t done, uint64_t total) {
                    task.ReportProgress(total ? static_cast<float>(static_cast<double>(done) / static_cast<double>(total)) : 1.0f,
                                        view::SizeText(done) + " of " + view::SizeText(total));
                    return !task.ShouldStop();
                });
            } else {
                job.result = Utilities::HashText(input, algorithm);
            }
            if (!job.result.success) job.error = job.result.error_message == "Stopped" ? "Cancelled." : job.result.error_message;
            return job;
        });
    }

    SideBySide("##hash", ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_Resizable, narrow,
        [&] {
        if (hash_source_ == 0) {
            Label("Text");
            ImGui::InputTextMultiline("##hash_text", &hash_text_, ImVec2(-1, 180));
            ImGui::TextColored(t.text_faint, "%s (UTF-8)", view::SizeText(hash_text_.size()).c_str());
        } else {
            Label("File");
            ImGui::TextWrapped("%s", hash_file_.empty() ? "No file chosen." : hash_file_.c_str());
            if (ui::SecondaryButton("Choose a file...")) {
                if (const auto path = FileDialogs::OpenFile("Choose a file to hash", {{"All files", "*"}})) hash_file_ = *path;
            }
            std::error_code ec;
            if (!hash_file_.empty()) {
                const auto size = std::filesystem::file_size(Utf8Path(hash_file_), ec);
                if (!ec) ImGui::TextColored(t.text_faint, "%s, read in 1 MB blocks", view::SizeText(size).c_str());
            }
        }

        },
        [&] {
        if (!Running(hash_job_)) {
            if (!hash_job_.Error().empty()) ui::NoticeLine(hash_job_.Error(), t.error);
            if (const HashJob* job = hash_job_.Answer(hash_key_)) {
                const std::string hash = view::HashOf(job->result, job->algorithm);
                Label(view::AlgorithmLabel(job->algorithm).c_str());
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextUnformatted(hash.c_str());
                ImGui::PopTextWrapPos();
                ImGui::TextColored(t.text_faint, "%s in %.1f ms", view::SizeText(job->result.input_size).c_str(),
                                   job->result.compute_time_ms);
                if (ui::SecondaryButton("Copy")) Copy(hash);
                ImGui::Spacing();
                Label("Check against");
                ImGui::SetNextItemWidth(-1);
                ImGui::InputTextWithHint("##expected", "Paste the hash you expect", &expected_hash_);
                const auto check = view::CheckHash(expected_hash_, hash);
                if (check.kind == view::HashCheck::Kind::Match) ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK " %s", check.text.c_str());
                if (check.kind == view::HashCheck::Kind::Differs) ImGui::TextColored(t.error, ICON_FA_TRIANGLE_EXCLAMATION " %s", check.text.c_str());
            } else if (hash_job_.Error().empty()) {
                ColumnHint("No hash yet", "Type text or choose a file, then Hash.");
            }
        }
        });
}

// ---------------------------------------------------------------- JSON

void UtilitiesPanel::RenderJson() {
    const auto& t = ui::CurrentTokens();
    const bool narrow = ImGui::GetContentRegionAvail().x < kSideBySideWidth;
    const auto run = [this](const std::string& action) {
        json_key_ = "json#" + std::to_string(++json_runs_);
        const std::string text = json_text_, path = json_path_;
        const int indent = json_indent_;
        json_job_.Ask("Utilities: JSON", json_key_, [text, path, indent, action](LambdaTask&) {
            JsonJob job;
            job.action = action;
            job.result = action == "minify" ? Utilities::MinifyJSON(text) : Utilities::FormatJSON(text, indent);
            if (job.result.is_valid && !path.empty()) {
                const JSONLookup found = Utilities::LookUpJSON(text, path);
                job.path_value = found.found ? "= " + found.value_json : found.error;
            }
            return job;  // an invalid text is an answer, shown with its place
        });
    };
    ui::FlowRow row;
    row.Next(ui::ButtonWidth("Check", ui::ButtonSize::Regular));
    std::string rerun;  // a path typed under the answer runs once it is drawn
    if (ui::PrimaryButton("Check")) run("check");
    row.Next(ui::ButtonWidth("Format", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Format")) run("format");
    row.Next(ui::ButtonWidth("Minify", ui::ButtonSize::Small));
    if (ui::SecondaryButton("Minify")) run("minify");
    row.Next(LabelledWidth("Indent", NumberWidth(36.0f)));
    ImGui::AlignTextToFramePadding();
    Label("Indent");
    ImGui::SameLine();
    ui::NumberField("##indent", &json_indent_, 0, 8, 1, 36.0f);
    row.Next(ui::ButtonWidth("Example", ui::ButtonSize::Small));
    if (ui::LinkButton("Example")) json_text_ = kSampleJson;
    row.Next(ui::ButtonWidth("Clear", ui::ButtonSize::Small));
    if (ui::LinkButton("Clear")) json_text_.clear();

    SideBySide("##json", ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_Resizable, narrow,
        [&] {
        Label("Input");
        ImGui::InputTextMultiline("##json_text", &json_text_, ImVec2(-1, narrow ? 200.0f : -ImGui::GetFrameHeightWithSpacing() * 2.0f),
                                  ImGuiInputTextFlags_AllowTabInput);
        ImGui::TextColored(t.text_faint, "%s", view::SizeText(json_text_.size()).c_str());

        },
        [&] {
        if (!Running(json_job_)) {
            const JsonJob* job = json_job_.Answer(json_key_);
            if (!job) {
                ColumnHint("Nothing checked yet", "Type or paste JSON, then Check, Format or Minify.");
            } else if (!job->result.is_valid) {
                const auto err = view::JsonError(job->result);
                ImGui::TextColored(t.error, ICON_FA_TRIANGLE_EXCLAMATION " %s", "Not valid JSON");
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextUnformatted(err.sentence.c_str());
                ImGui::PopTextWrapPos();
                std::string error_line, prefix;
                for (const auto& [n, line] : err.lines) {
                    char number[32];
                    std::snprintf(number, sizeof(number), "%4d | ", n);
                    ImGui::TextColored(t.text_dim, "%s%s", number, line.c_str());
                    error_line = line;
                    prefix = number;
                }
                if (err.caret_column > 0) {
                    // The bytes of the error line before its column (columns count characters).
                    size_t bytes = 0;
                    for (int c = 1; c < err.caret_column && bytes < error_line.size(); ++c) {
                        ++bytes;
                        while (bytes < error_line.size() && (static_cast<unsigned char>(error_line[bytes]) & 0xC0) == 0x80) ++bytes;
                    }
                    const float x = ImGui::GetCursorPosX() + ImGui::CalcTextSize(prefix.c_str()).x +
                                    ImGui::CalcTextSize(error_line.substr(0, bytes).c_str()).x;
                    ImGui::SetCursorPosX(x);
                    ImGui::TextColored(t.error, "^");
                }
            } else {
                ImGui::TextColored(t.success, ICON_FA_CIRCLE_CHECK " %s", "Valid JSON");
                ImGui::PushTextWrapPos(0.0f);
                ImGui::TextColored(t.text_dim, "%s", view::JsonSummary(job->result).c_str());
                ImGui::PopTextWrapPos();
                ImGui::SetNextItemWidth(220.0f);
                if (ImGui::InputTextWithHint("##json_path", "Path, e.g. config.max_workers", &json_path_,
                                             ImGuiInputTextFlags_EnterReturnsTrue))
                    rerun = job->action;
                if (!job->path_value.empty()) {
                    ImGui::SameLine();
                    ImGui::TextUnformatted(job->path_value.c_str());
                }
                const std::string& shown = job->action == "minify" ? job->result.minified_json : job->result.formatted_json;
                if (json_tree_for_ != json_key_) {
                    json_tree_.SetContent(job->result.minified_json);
                    json_tree_for_ = json_key_;
                }
                if (ImGui::BeginTabBar("##json_views")) {
                    if (ImGui::BeginTabItem("Tree")) {
                        ImGui::BeginChild("##tree", ImVec2(0, 0));
                        json_tree_.Render();
                        ImGui::EndChild();
                        ImGui::EndTabItem();
                    }
                    if (ImGui::BeginTabItem(job->action == "minify" ? "Minified" : "Formatted")) {
                        if (ui::SecondaryButton("Copy")) Copy(shown);
                        ImGui::SameLine();
                        if (ui::SecondaryButton("Use as input")) json_text_ = shown;
                        ImGui::InputTextMultiline("##json_out", const_cast<std::string*>(&shown), ImVec2(-1, -1),
                                                  ImGuiInputTextFlags_ReadOnly);
                        ImGui::EndTabItem();
                    }
                    ImGui::EndTabBar();
                }
            }
        }
        });
    if (!rerun.empty()) run(rerun);
}

// ---------------------------------------------------------------- Regex

void UtilitiesPanel::RenderRegex() {
    const auto& t = ui::CurrentTokens();
    const bool narrow = ImGui::GetContentRegionAvail().x < kSideBySideWidth;
    const auto run = [this] {
        regex_key_ = "regex#" + std::to_string(++regex_runs_);
        const std::string pattern = pattern_, text = regex_text_, replacement = replacement_;
        const std::string flags = std::string(ignore_case_ ? "i" : "") + (multiline_ ? "m" : "");
        regex_job_.Ask("Utilities: regex", regex_key_, [pattern, text, replacement, flags](LambdaTask&) {
            RegexJob job;
            job.text = text;
            job.result = Utilities::TestRegex(pattern, text, flags);
            if (job.result.is_valid_pattern) job.replaced = Utilities::ReplaceRegex(pattern, text, replacement, flags);
            return job;  // a bad pattern is an answer, shown with its reason
        });
    };
    bool rerun = false;  // Replace under the answer runs once it is drawn
    const float button = ui::ButtonWidth("Test", ui::ButtonSize::Regular);
    ImGui::SetNextItemWidth(-button - ImGui::GetStyle().ItemSpacing.x);
    const bool enter = ImGui::InputTextWithHint("##pattern", "A regular expression", &pattern_, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if (ui::PrimaryButton("Test") || enter) run();
    ui::FlowRow flags;
    flags.Next(CheckboxWidth("Ignore case"));
    if (ImGui::Checkbox("Ignore case", &ignore_case_)) run();
    flags.Next(CheckboxWidth("Multiline: ^ and $ at every line"));
    if (ImGui::Checkbox("Multiline: ^ and $ at every line", &multiline_)) run();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextColored(t.text_faint, "%s", "ECMAScript (C++ std::regex): \\w and \\b know ASCII letters only.");
    ImGui::PopTextWrapPos();

    SideBySide("##regex", ImGuiTableFlags_SizingStretchSame | ImGuiTableFlags_Resizable, narrow,
        [&] {
        Label("Text");
        ImGui::InputTextMultiline("##regex_text", &regex_text_, ImVec2(-1, narrow ? 160.0f : -ImGui::GetFrameHeightWithSpacing() * 2.0f));
        ImGui::TextColored(t.text_faint, "%s", view::SizeText(regex_text_.size()).c_str());

        },
        [&] {
        if (!Running(regex_job_)) {
            const RegexJob* job = regex_job_.Answer(regex_key_);
            if (!job) {
                ColumnHint("No matches yet", "Type a pattern and text, then Test.");
            } else if (!job->result.is_valid_pattern) {
                ui::NoticeLine("The pattern: " + view::RegexSummary(job->result), t.error);
            } else {
                Label(view::RegexSummary(job->result).c_str());
                if (regex_rows_key_ != regex_key_) {
                    regex_rows_ = view::RegexRows(job->result, job->text);
                    regex_rows_key_ = regex_key_;
                }
                const auto& rows = regex_rows_;
                if (!rows.empty() &&
                    ImGui::BeginTable("##matches", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp,
                                      ImVec2(0, 220.0f))) {
                    ImGui::TableSetupScrollFreeze(0, 1);
                    ImGui::TableSetupColumn("Match", ImGuiTableColumnFlags_WidthStretch, 2.0f);
                    ImGui::TableSetupColumn("Line");
                    ImGui::TableSetupColumn("Column");
                    ImGui::TableSetupColumn("Characters");
                    ImGui::TableSetupColumn("Groups", ImGuiTableColumnFlags_WidthStretch, 1.5f);
                    ImGui::TableHeadersRow();
                    ImGuiListClipper clipper;
                    clipper.Begin(static_cast<int>(rows.size()));
                    while (clipper.Step())
                    for (int r = clipper.DisplayStart; r < clipper.DisplayEnd; ++r) {
                        const auto& row = rows[static_cast<size_t>(r)];
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::TextUnformatted(row.text.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%d", row.line);
                        ImGui::TableNextColumn();
                        ImGui::Text("%d", row.column);
                        ImGui::TableNextColumn();
                        ImGui::Text("%d-%d", row.start, row.end);
                        ImGui::TableNextColumn();
                        std::string groups;
                        for (size_t g = 0; g < row.groups.size(); ++g) groups += (g ? ", " : "") + row.groups[g];
                        ImGui::TextUnformatted(groups.c_str());
                    }
                    ImGui::EndTable();
                }
                if (!rows.empty() && ui::SecondaryButton("Copy matches")) {
                    std::string all;
                    for (const auto& row : rows) all += row.text + "\n";
                    Copy(all);
                }
                ImGui::Spacing();
                if (ImGui::CollapsingHeader("Replace")) {
                    ImGui::SetNextItemWidth(260.0f);
                    if (ImGui::InputTextWithHint("##replacement", "$& the match, $1 a group", &replacement_,
                                                 ImGuiInputTextFlags_EnterReturnsTrue))
                        rerun = true;
                    ImGui::SameLine();
                    if (ui::SecondaryButton("Replace all")) rerun = true;  // not "Replace": the header above has that id
                    if (job->replaced.success) {
                        ImGui::InputTextMultiline("##replaced", const_cast<std::string*>(&job->replaced.replaced_text),
                                                  ImVec2(-1, 120.0f), ImGuiInputTextFlags_ReadOnly);
                        if (ui::SecondaryButton("Copy the result")) Copy(job->replaced.replaced_text);
                    }
                }
            }
        }
        });
    if (rerun) run();
}

}  // namespace cyxwiz
