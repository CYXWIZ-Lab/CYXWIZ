// Preferences > Tools (TOFIX129 B9-1, moved from General): how Tools >
// Utilities and Monitoring open, and the diagnostics switches. Saved at once;
// the windows also have "Save as my default".

#include "toolbar.h"

#include "../../core/engine_config.h"
#include "../../core/utilities_presentation.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"

#include <imgui.h>

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz {

namespace {
// A setting's name in the left column, its control after it
void Label(const char* text, float column) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(text);
    ImGui::SameLine(column);
}

void Heading(const char* text) {
    ImGui::Spacing();
    ImGui::Text("%s", text);
    ImGui::Separator();
    ImGui::Spacing();
}
}  // namespace

void ToolbarPanel::RenderToolsPreferences() {
    auto& engine_config = core::EngineConfig::Instance();
    const float column = ImGui::CalcTextSize("Keep readings for").x + ImGui::GetStyle().ItemSpacing.x * 3.0f;

    // ---- Tools > Utilities (TOFIX129 D12)
    Heading("Utilities");
    {
        core::UtilitiesConfig u = engine_config.GetUtilities();
        bool changed = false;
        const auto combo = [&changed](const char* id, std::string& value,
                                      const std::vector<std::pair<std::string, std::string>>& items) {
            std::string shown = value;
            for (const auto& [key, text] : items)
                if (key == value) shown = text;
            ImGui::SetNextItemWidth(180.0f);
            if (ImGui::BeginCombo(id, shown.c_str())) {
                for (const auto& [key, text] : items) {
                    if (ImGui::Selectable(text.c_str(), key == value)) {
                        value = key;
                        changed = true;
                    }
                }
                ImGui::EndCombo();
            }
        };
        std::vector<std::pair<std::string, std::string>> tabs;
        for (const auto tab : utilities_view::kTabs) tabs.push_back({utilities_view::TabId(tab), utilities_view::TabLabel(tab)});
        Label("Opens on", column);
        combo("##util_tab", u.tab, tabs);
        Label("Calculator", column);
        changed |= ui::NumberField("##util_digits", &u.digits, 1, 17, 1, 40.0f, "Significant digits (1 to 17)");
        ImGui::SameLine();
        ImGui::TextDisabled("digits shown");
        ImGui::SameLine();
        changed |= ImGui::Checkbox("Degrees##util_degrees", &u.degrees);
        std::vector<std::pair<std::string, std::string>> categories;
        for (const auto& c : Utilities::GetUnitCategories()) categories.push_back({c, utilities_view::CategoryLabel(c)});
        Label("Unit category", column);
        combo("##util_units", u.unit_category, categories);
        Label("Hash", column);
        combo("##util_hash", u.hash_algorithm,
              {{"md5", "MD5"}, {"sha1", "SHA-1"}, {"sha256", "SHA-256"}, {"sha512", "SHA-512"}});
        Label("JSON indent", column);
        changed |= ui::NumberField("##util_indent", &u.json_indent, 0, 8, 1, 36.0f);
        Label("Regex", column);
        changed |= ImGui::Checkbox("Ignore case##util_icase", &u.regex_ignore_case);
        ImGui::SameLine();
        changed |= ImGui::Checkbox("Multiline##util_multiline", &u.regex_multiline);
        if (changed) {
            engine_config.SetUtilities(u);
            engine_config.Save();
        }
        ImGui::TextDisabled("How Tools > Utilities opens when the Engine starts.");
    }

    // ---- Tools > Monitoring (TOFIX129 D13)
    Heading("Monitoring");
    {
        core::MonitorConfig m = engine_config.GetMonitor();
        bool changed = false;
        Label("Sample every", column);
        ui::NumberFieldOptions interval;
        interval.min = 0.25;
        interval.max = 60.0;
        interval.step = 1.0;
        interval.format = "%.2g";
        interval.width = 56.0f;
        interval.hint = "Seconds between the Monitor window's readings";
        changed |= ui::NumberField("##monitor_interval", &m.interval_seconds, interval);
        ImGui::SameLine();
        ImGui::TextDisabled("s");
        static const double kKeepSeconds[] = {300.0, 900.0, 3600.0};
        static const char* const kKeepLabels[] = {"5 min", "15 min", "1 hour"};
        int keep = 1;
        for (int i = 0; i < 3; ++i)
            if (m.keep_seconds == kKeepSeconds[i]) keep = i;
        Label("Keep readings for", column);
        if (ui::SegmentedControl("##monitor_keep", kKeepLabels, 3, &keep)) {
            m.keep_seconds = kKeepSeconds[keep];
            changed = true;
        }
        static const char* const kTabIds[] = {"system", "memory", "loaded_data", "frames"};
        static const char* const kTabLabels[] = {"System", "Memory", "Loaded data", "Frames"};
        int tab = 0;
        for (int i = 0; i < 4; ++i)
            if (m.tab == kTabIds[i]) tab = i;
        Label("Opens on", column);
        if (ui::SegmentedControl("##monitor_tab", kTabLabels, 4, &tab)) {
            m.tab = kTabIds[tab];
            changed = true;
        }
        if (changed) {
            engine_config.SetMonitor(m);
            engine_config.Save();
        }
        ImGui::TextDisabled("How Tools > Monitoring opens when the Engine starts.");
    }

    // ---- Background Tasks (TOFIX129 C6)
    Heading("Background Tasks");
    {
        core::MonitorConfig m = engine_config.GetMonitor();
        Label("List", column);
        if (ImGui::Checkbox("Show finished tasks##tasks_finished", &m.show_finished_tasks)) {
            engine_config.SetMonitor(m);
            engine_config.Save();
        }
        ImGui::TextDisabled("How Background Tasks opens when the Engine starts.");
    }

    // ---- Tools > Diagnostics: on at once, and kept
    Heading("Diagnostics");
    {
        bool switched = false;
        Label("Log", column);
        if (idle_log_ptr_) {
            switched |= ImGui::Checkbox("Frame-rate changes##diag_pace", idle_log_ptr_);
            ui::Tooltip("One log line when the Engine slows or speeds its frames:\n"
                        "idle (10 a second), not in front (30 a second), minimised.");
            ImGui::SameLine();
        }
        if (verbose_python_log_ptr_) {
            bool verbose = verbose_python_log_ptr_->load();
            if (ImGui::Checkbox("Every Python command##diag_python", &verbose)) {
                verbose_python_log_ptr_->store(verbose);
                switched = true;
            }
            ui::Tooltip("Each command the Engine sends to Python, with how long it took.");
        }
        if (frame_overlay_ptr_) {
            Label("Overlay", column);
            switched |= ImGui::Checkbox("Frame time overlay##diag_overlay", frame_overlay_ptr_);
            ui::Tooltip("The whole frame time and the Engine's own interface-building time,\n"
                        "at the top right of the window. More in Tools > Monitoring > Frames.");
        }
        if (switched) SaveDiagnostics();
    }
}

}  // namespace cyxwiz
