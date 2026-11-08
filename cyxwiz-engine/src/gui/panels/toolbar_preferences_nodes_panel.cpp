// Preferences > Nodes panel (TOFIX178 T4, approved Nodes mockup): whether the
// Nodes panel shows Recent, how many, and Clear Recent. Engine-wide, applied
// and saved at once; the favourites and recent nodes themselves are kept by
// the Nodes panel (EngineConfig nodes_panel).

#include "toolbar.h"

#include "../../core/engine_config.h"
#include "../../core/node_metadata_registry.h"
#include "../ui_buttons.h"
#include "../ui_widgets.h"

#include <imgui.h>

namespace cyxwiz {

void ToolbarPanel::RenderNodesPanelPreferences() {
    auto& engine_config = core::EngineConfig::Instance();
    const core::NodesPanelConfig saved = engine_config.GetNodesPanel();
    core::NodesPanelConfig settings = saved;

    ImGui::Spacing();
    ImGui::Text("Recent");
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Checkbox("Show Recent", &settings.show_recent);
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("The last nodes you added, newest first, above Studio Tools.");
    ImGui::PopTextWrapPos();

    ImGui::Spacing();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Show the last");
    ImGui::SameLine();
    ui::NumberField("##recent_count", &settings.recent_count, 1, core::kNodesPanelMaxRecent, 1, 60.0f);
    ImGui::SameLine();
    ImGui::TextUnformatted("nodes");

    ImGui::Spacing();
    if (ui::SecondaryButton("Clear Recent", !saved.recent.empty(), "Recent is empty")) {
        auto& registry = NodeMetadataRegistry::Instance();
        registry.SetUserNodes(registry.FavoriteTypes(), {});
        settings.recent.clear();
    }

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("Favourites and Recent are saved with your Engine settings and apply to every project. "
                        "Star a node in the Nodes panel, or right-click it, to keep it in Favourites.");
    ImGui::PopTextWrapPos();

    if (!(settings == saved)) {
        engine_config.SetNodesPanel(settings);
        engine_config.Save();
    }
}

}  // namespace cyxwiz
