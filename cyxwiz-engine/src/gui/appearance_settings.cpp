#include "appearance_settings.h"

#include "../core/appearance_options.h"
#include "../core/engine_config.h"
#include "dock_style.h"
#include "editor_fonts.h"

#include <imgui.h>
#include <imnodes.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <iterator>
#include <utility>
#include <cctype>
#include <fstream>
#include <spdlog/spdlog.h>

namespace gui {

namespace {

bool g_font_rebuild_requested = false;
// The pack the canvas draws with, read for every node every frame: kept here
// instead of reading the Engine settings each time.
int g_node_icon_pack = cyxwiz::appearance::kDefaultNodeIconPack;

cyxwiz::core::AppearanceConfig Normalized(cyxwiz::core::AppearanceConfig config) {
    config.theme_preset = cyxwiz::appearance::NormalizeThemePreset(
        config.theme_preset, static_cast<int>(ThemePreset::COUNT));
    config.ui_text_px = cyxwiz::appearance::NormalizeUiTextPx(config.ui_text_px);
    config.code_font_scale = cyxwiz::appearance::NormalizeCodeScale(config.code_font_scale);
    config.node_icon_pack = cyxwiz::appearance::NormalizeNodeIconPack(config.node_icon_pack);
    return config;
}

cyxwiz::core::AppearanceConfig Current() {
    return Normalized(cyxwiz::core::EngineConfig::Instance().GetAppearance());
}

void Store(const cyxwiz::core::AppearanceConfig& config) {
    auto& engine_config = cyxwiz::core::EngineConfig::Instance();
    engine_config.SetAppearance(config);
    if (!engine_config.Save())
        spdlog::warn("Appearance: could not save the Engine settings");
}

void ApplySidebar(bool left) {
    GetDockStyle().SetSidebarPosition(left ? SidebarPosition::Left : SidebarPosition::Right);
}

}  // namespace

bool ApplyThemeFile(const std::filesystem::path& path, std::string* error);

void ApplyStartupAppearance() {
    const auto config = Current();
    GetTheme().ApplyPreset(static_cast<ThemePreset>(config.theme_preset));
    if (!config.custom_theme_file.empty()) {
        std::string error;
        if (!ApplyThemeFile(config.custom_theme_file, &error)) {
            spdlog::warn("Appearance: custom theme not applied ({}); using the preset", error);
            auto cleared = config;
            cleared.custom_theme_file.clear();
            Store(cleared);
        }
    }
    cyxwiz::gui::g_code_font_scale = config.code_font_scale;
    ApplySidebar(config.sidebar_left);
    g_node_icon_pack = config.node_icon_pack;
    spdlog::debug("Appearance: theme {}, interface {} px, code scale {:.1f}, sidebar {}, node icons {}",
                 Theme::GetPresetName(static_cast<ThemePreset>(config.theme_preset)),
                 config.ui_text_px, config.code_font_scale,
                 config.sidebar_left ? "left" : "right",
                 cyxwiz::appearance::kNodeIconPacks[static_cast<size_t>(config.node_icon_pack)]);
}

ThemePreset CurrentThemePreset() { return GetTheme().GetCurrentPreset(); }

void SetThemePreset(ThemePreset preset) {
    GetTheme().ApplyPreset(preset);
    auto config = Current();
    config.theme_preset = static_cast<int>(preset);
    config.custom_theme_file.clear();  // a preset replaces any custom theme
    Store(config);
}

int UiTextPixels() { return Current().ui_text_px; }

void SetUiTextPixels(int pixels) {
    auto config = Current();
    const int normalized = cyxwiz::appearance::NormalizeUiTextPx(pixels);
    if (config.ui_text_px == normalized)
        return;
    config.ui_text_px = normalized;
    Store(config);
    g_font_rebuild_requested = true;
}

float CodeTextScale() { return cyxwiz::gui::g_code_font_scale; }

void SetCodeTextScale(float scale) {
    auto config = Current();
    config.code_font_scale = cyxwiz::appearance::NormalizeCodeScale(scale);
    cyxwiz::gui::g_code_font_scale = config.code_font_scale;
    Store(config);
}

bool SidebarOnLeft() { return Current().sidebar_left; }

void SetSidebarOnLeft(bool left) {
    auto config = Current();
    config.sidebar_left = left;
    ApplySidebar(left);
    Store(config);
}

int NodeIconPack() { return g_node_icon_pack; }

void SetNodeIconPack(int pack) {
    auto config = Current();
    config.node_icon_pack = cyxwiz::appearance::NormalizeNodeIconPack(pack);
    g_node_icon_pack = config.node_icon_pack;
    Store(config);
}

void ResetAppearance() {
    const auto before = Current();
    const cyxwiz::core::AppearanceConfig defaults;
    GetTheme().ApplyPreset(static_cast<ThemePreset>(defaults.theme_preset));
    cyxwiz::gui::g_code_font_scale = defaults.code_font_scale;
    ApplySidebar(defaults.sidebar_left);
    g_node_icon_pack = defaults.node_icon_pack;
    Store(defaults);
    if (before.ui_text_px != defaults.ui_text_px)
        g_font_rebuild_requested = true;
}

bool ConsumeFontRebuildRequest() {
    const bool requested = g_font_rebuild_requested;
    g_font_rebuild_requested = false;
    return requested;
}

bool IsLightTheme() {
    const ImVec4 bg = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    return cyxwiz::appearance::IsLightBackground(bg.x, bg.y, bg.z);
}

// ---- Custom themes ----------------------------------------------------------

namespace {

using json = nlohmann::json;

json ColorJson(const ImVec4& c, int id) {
    return json{{"id", id}, {"r", c.x}, {"g", c.y}, {"b", c.z}, {"a", c.w}};
}

bool ReadColor(const json& item, int count, int* id, ImVec4* color) {
    if (!item.is_object() || !item.contains("id")) return false;
    *id = item.value("id", -1);
    if (*id < 0 || *id >= count) return false;
    *color = ImVec4(item.value("r", 0.0f), item.value("g", 0.0f), item.value("b", 0.0f),
                    item.value("a", 1.0f));
    return true;
}

json Rgb(const ImVec4& c) { return json::array({c.x, c.y, c.z}); }

bool ReadRgb(const json& j, ImVec4* out) {
    if (!j.is_array() || j.size() != 3) return false;
    for (const auto& v : j)
        if (!v.is_number()) return false;
    *out = ImVec4(j[0].get<float>(), j[1].get<float>(), j[2].get<float>(), 1.0f);
    return true;
}

// The theme's own colours (TOFIX129 A4): accent, node families, data types,
// status words. Kept with a custom theme so it looks the same when reloaded.
json EngineColoursJson(const cyxwiz::ui::EngineColours& e) {
    json j;
    j["accent"] = Rgb(e.accent);
    j["node"] = json::array();
    for (const auto& c : e.node) j["node"].push_back(Rgb(c));
    j["data"] = json::array();
    for (const auto& c : e.data) j["data"].push_back(Rgb(c));
    j["status"] = {{"success", Rgb(e.success)}, {"warning", Rgb(e.warning)}, {"caution", Rgb(e.caution)},
                   {"error", Rgb(e.error)}, {"critical", Rgb(e.critical)}, {"info", Rgb(e.info)},
                   {"pending", Rgb(e.pending)}, {"running", Rgb(e.running)}};
    return j;
}

// Over `e`: every colour the file names; the others stay as they are.
void ReadEngineColours(const json& j, cyxwiz::ui::EngineColours* e) {
    if (j.contains("accent")) ReadRgb(j["accent"], &e->accent);
    if (j.contains("node") && j["node"].is_array())
        for (size_t i = 0; i < j["node"].size() && i < std::size(e->node); ++i) ReadRgb(j["node"][i], &e->node[i]);
    if (j.contains("data") && j["data"].is_array())
        for (size_t i = 0; i < j["data"].size() && i < std::size(e->data); ++i) ReadRgb(j["data"][i], &e->data[i]);
    if (j.contains("status") && j["status"].is_object()) {
        const auto& st = j["status"];
        const std::pair<const char*, ImVec4*> fields[] = {
            {"success", &e->success}, {"warning", &e->warning}, {"caution", &e->caution}, {"error", &e->error},
            {"critical", &e->critical}, {"info", &e->info}, {"pending", &e->pending}, {"running", &e->running}};
        for (const auto& [key, out] : fields)
            if (st.contains(key)) ReadRgb(st[key], out);
    }
}

}  // namespace

bool ApplyThemeFile(const std::filesystem::path& path, std::string* error) {
    std::ifstream file(path);
    if (!file.is_open()) {
        if (error) *error = "cannot open " + path.string();
        return false;
    }
    const json j = json::parse(file, nullptr, false);
    if (!j.is_object()) {
        if (error) *error = "not a theme file (invalid JSON)";
        return false;
    }
    ImGuiStyle& style = ImGui::GetStyle();
    if (j.contains("imgui_colors") && j["imgui_colors"].is_array()) {
        for (const auto& item : j["imgui_colors"]) {
            int id = -1;
            ImVec4 color;
            if (ReadColor(item, ImGuiCol_COUNT, &id, &color)) style.Colors[id] = color;
        }
    }
    if (j.contains("style") && j["style"].is_object()) {
        const auto& st = j["style"];
        const auto f = [&st](const char* key, float& out) {
            if (st.contains(key) && st[key].is_number()) out = st[key].get<float>();
        };
        const auto v = [&st](const char* key, ImVec2& out) {
            if (st.contains(key) && st[key].is_array() && st[key].size() == 2) {
                out.x = st[key][0].get<float>();
                out.y = st[key][1].get<float>();
            }
        };
        f("window_rounding", style.WindowRounding);
        f("frame_rounding", style.FrameRounding);
        f("popup_rounding", style.PopupRounding);
        f("scrollbar_rounding", style.ScrollbarRounding);
        f("grab_rounding", style.GrabRounding);
        f("tab_rounding", style.TabRounding);
        f("window_border", style.WindowBorderSize);
        f("frame_border", style.FrameBorderSize);
        f("popup_border", style.PopupBorderSize);
        f("child_border", style.ChildBorderSize);
        v("window_padding", style.WindowPadding);
        v("frame_padding", style.FramePadding);
        v("item_spacing", style.ItemSpacing);
        f("scrollbar_size", style.ScrollbarSize);
        f("grab_min_size", style.GrabMinSize);
        f("indent_spacing", style.IndentSpacing);
    }
    if (j.contains("imnodes_colors") && j["imnodes_colors"].is_array()) {
        auto& nodes = ImNodes::GetStyle();
        for (const auto& item : j["imnodes_colors"]) {
            int id = -1;
            ImVec4 color;
            if (ReadColor(item, ImNodesCol_COUNT, &id, &color))
                nodes.Colors[id] = ImGui::ColorConvertFloat4ToU32(color);
        }
    }
    if (j.contains("engine_colors") && j["engine_colors"].is_object()) {
        cyxwiz::ui::EngineColours engine = GetTheme().GetEngineColours();
        ReadEngineColours(j["engine_colors"], &engine);
        GetTheme().SetEngineColours(engine);
    }
    return true;
}

std::filesystem::path CustomThemesDirectory() {
    const auto config_path = cyxwiz::core::EngineConfig::Instance().GetConfigPath();
    return config_path.has_parent_path() ? config_path.parent_path() / "themes"
                                         : std::filesystem::path("themes");
}

std::vector<std::filesystem::path> ListCustomThemes() {
    std::vector<std::filesystem::path> themes;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(CustomThemesDirectory(), ec)) {
        if (entry.is_regular_file(ec) && entry.path().extension() == ".json")
            themes.push_back(entry.path());
    }
    std::sort(themes.begin(), themes.end());
    return themes;
}

bool SaveCustomTheme(const std::string& name, std::filesystem::path* saved_path,
                     std::string* error) {
    std::string clean;
    for (const char ch : name) {
        const bool ok = std::isalnum(static_cast<unsigned char>(ch)) != 0 || ch == '-' ||
                        ch == '_' || ch == ' ';
        if (ok) clean.push_back(ch);
    }
    while (!clean.empty() && clean.back() == ' ') clean.pop_back();
    while (!clean.empty() && clean.front() == ' ') clean.erase(clean.begin());
    if (clean.empty()) {
        if (error) *error = "Use letters, digits, spaces, '-' or '_' in the name.";
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(CustomThemesDirectory(), ec);
    const auto path = CustomThemesDirectory() / (clean + ".json");

    const ImGuiStyle& style = ImGui::GetStyle();
    json j;
    j["name"] = clean;
    j["version"] = "1.0";
    j["base_preset"] = Theme::GetPresetName(GetTheme().GetCurrentPreset());
    j["imgui_colors"] = json::array();
    for (int i = 0; i < ImGuiCol_COUNT; ++i)
        j["imgui_colors"].push_back(ColorJson(style.Colors[i], i));
    j["style"] = {
        {"window_rounding", style.WindowRounding}, {"frame_rounding", style.FrameRounding},
        {"popup_rounding", style.PopupRounding}, {"scrollbar_rounding", style.ScrollbarRounding},
        {"grab_rounding", style.GrabRounding}, {"tab_rounding", style.TabRounding},
        {"window_border", style.WindowBorderSize}, {"frame_border", style.FrameBorderSize},
        {"popup_border", style.PopupBorderSize}, {"child_border", style.ChildBorderSize},
        {"window_padding", {style.WindowPadding.x, style.WindowPadding.y}},
        {"frame_padding", {style.FramePadding.x, style.FramePadding.y}},
        {"item_spacing", {style.ItemSpacing.x, style.ItemSpacing.y}},
        {"scrollbar_size", style.ScrollbarSize}, {"grab_min_size", style.GrabMinSize},
        {"indent_spacing", style.IndentSpacing}};
    j["imnodes_colors"] = json::array();
    const auto& nodes = ImNodes::GetStyle();
    for (int i = 0; i < ImNodesCol_COUNT; ++i)
        j["imnodes_colors"].push_back(ColorJson(ImGui::ColorConvertU32ToFloat4(nodes.Colors[i]), i));
    j["engine_colors"] = EngineColoursJson(GetTheme().GetEngineColours());

    std::ofstream file(path);
    if (!file.is_open()) {
        if (error) *error = "Cannot write " + path.string();
        return false;
    }
    file << j.dump(2);
    file.close();
    auto config = Current();
    config.custom_theme_file = path.string();
    Store(config);
    if (saved_path) *saved_path = path;
    spdlog::info("Appearance: saved custom theme {}", path.string());
    return true;
}

bool LoadCustomTheme(const std::filesystem::path& path, std::string* error) {
    // A custom theme is applied over the current base preset.
    GetTheme().ApplyPreset(GetTheme().GetCurrentPreset());
    if (!ApplyThemeFile(path, error)) return false;
    auto config = Current();
    config.custom_theme_file = path.string();
    Store(config);
    spdlog::info("Appearance: loaded custom theme {}", path.string());
    return true;
}

std::string ActiveCustomThemeName() {
    const auto file = Current().custom_theme_file;
    return file.empty() ? std::string() : std::filesystem::path(file).stem().string();
}

void ReapplySavedTheme() {
    const auto config = Current();
    GetTheme().ApplyPreset(static_cast<ThemePreset>(config.theme_preset));
    if (!config.custom_theme_file.empty()) ApplyThemeFile(config.custom_theme_file, nullptr);
}

}  // namespace gui
