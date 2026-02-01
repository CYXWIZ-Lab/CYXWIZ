#include "mj_env_browser_panel.h"

#include <imgui.h>
#include <algorithm>

namespace cyxwiz::plugin::mujoco {

void MjEnvBrowserPanel::Render(MjEnvLibrary& library,
                                const MjEnvManager& env,
                                bool* visible) {
    if (!visible || !*visible) return;

    // Process deferred load BEFORE any ImGui rendering.
    // LoadEnvironment switches GL contexts which corrupts ImGui's GL state mid-frame.
    if (!pending_load_path_.empty()) {
        std::string path = std::move(pending_load_path_);
        std::string id = std::move(pending_load_id_);
        pending_load_path_.clear();
        pending_load_id_.clear();
        if (load_callback_ && load_callback_(path)) {
            loaded_env_id_ = id;
        }
    }

    ImGui::SetNextWindowSize(ImVec2(520, 480), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Environment Library", visible)) {
        ImGui::End();
        return;
    }

    RenderSearchBar();
    RenderCategoryTabs(library);
    ImGui::Separator();

    // Determine filter text (lowercase)
    std::string filter = search_buf_;
    std::transform(filter.begin(), filter.end(), filter.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    // Environment cards
    float panel_width = ImGui::GetContentRegionAvail().x;
    float card_width = 240.0f;
    int cols = std::max(1, static_cast<int>(panel_width / card_width));

    int col = 0;
    for (const auto& info : library.GetAll()) {
        // Category filter
        if (!selected_category_.empty() && info.category != selected_category_)
            continue;

        // Text search filter
        if (!filter.empty()) {
            std::string name_lower = info.name;
            std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            std::string desc_lower = info.description;
            std::transform(desc_lower.begin(), desc_lower.end(), desc_lower.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (name_lower.find(filter) == std::string::npos &&
                desc_lower.find(filter) == std::string::npos)
                continue;
        }

        if (col > 0) ImGui::SameLine();

        bool is_loaded = (env.IsLoaded() && loaded_env_id_ == info.id);
        RenderEnvCard(info, library, is_loaded);

        col++;
        if (col >= cols) col = 0;
    }

    ImGui::End();
}

void MjEnvBrowserPanel::RenderSearchBar() {
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - 4.0f);
    ImGui::InputTextWithHint("##env_search", "Search environments...", search_buf_, sizeof(search_buf_));
}

void MjEnvBrowserPanel::RenderCategoryTabs(const MjEnvLibrary& library) {
    // "All" tab
    bool all_selected = selected_category_.empty();
    if (all_selected) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
    }
    if (ImGui::SmallButton("All")) {
        selected_category_.clear();
    }
    if (all_selected) ImGui::PopStyleColor();

    for (const auto& cat : library.GetCategories()) {
        ImGui::SameLine();
        bool sel = (selected_category_ == cat);
        if (sel) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(ImGuiCol_ButtonActive));
        }
        if (ImGui::SmallButton(cat.c_str())) {
            selected_category_ = (selected_category_ == cat) ? "" : cat;
        }
        if (sel) ImGui::PopStyleColor();
    }
}

void MjEnvBrowserPanel::RenderEnvCard(const EnvInfo& info,
                                       const MjEnvLibrary& library,
                                       bool is_loaded) {
    ImGui::PushID(info.id.c_str());

    float card_w = 230.0f;
    ImVec2 card_start = ImGui::GetCursorScreenPos();

    // Card background
    ImDrawList* dl = ImGui::GetWindowDrawList();
    ImVec2 card_end(card_start.x + card_w, card_start.y + 140.0f);
    ImU32 bg_col = is_loaded ? IM_COL32(40, 80, 60, 255) : IM_COL32(45, 45, 50, 255);
    dl->AddRectFilled(card_start, card_end, bg_col, 6.0f);
    dl->AddRect(card_start, card_end, IM_COL32(80, 80, 90, 255), 6.0f);

    // Content inside card
    ImGui::BeginGroup();
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 8));

    // Name
    ImGui::TextColored(ImVec4(0.9f, 0.9f, 1.0f, 1.0f), "%s", info.name.c_str());

    // Category badge
    ImGui::SameLine(card_w - 90);
    ImGui::TextDisabled("%s", info.category.c_str());

    // Description (truncated)
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 30));
    ImGui::PushTextWrapPos(card_start.x + card_w - 10);
    ImGui::TextDisabled("%s", info.description.c_str());
    ImGui::PopTextWrapPos();

    // Metadata row
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 80));
    ImGui::Text("Obs: %d", info.obs_dim);
    ImGui::SameLine();
    ImGui::Text("Act: %d", info.act_dim);
    ImGui::SameLine();
    ImGui::Text("Steps: %d", info.max_steps);

    // Load button
    ImGui::SetCursorScreenPos(ImVec2(card_start.x + 10, card_start.y + 108));
    if (is_loaded) {
        ImGui::TextColored(ImVec4(0.3f, 0.9f, 0.5f, 1.0f), "Loaded");
    } else {
        if (ImGui::Button("Load", ImVec2(card_w - 20, 22))) {
            // Defer load to next frame — LoadEnvironment switches GL contexts
            pending_load_path_ = library.GetAssetPath(info);
            pending_load_id_ = info.id;
        }
    }

    ImGui::EndGroup();

    // Advance cursor past card
    ImGui::SetCursorScreenPos(ImVec2(card_start.x, card_end.y + 6));
    ImGui::Dummy(ImVec2(card_w, 0));

    ImGui::PopID();
}

} // namespace cyxwiz::plugin::mujoco
