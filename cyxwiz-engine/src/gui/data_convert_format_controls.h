#pragma once

#include "../core/data_convert_formats.h"
#include <imgui.h>

namespace gui {

// Names, rather than list indices, own serialized selection. Merely rendering
// an alias, unknown value, or unavailable optional adapter cannot change it.
inline bool RenderDataConvertFormatCombo(const char* id, std::string& value,
                                         cyxwiz::data_convert::Direction direction,
                                         cyxwiz::data_convert::Features features) {
    namespace dc = cyxwiz::data_convert;
    const auto* info = dc::Find(value);
    const auto normalized = dc::Normalize(value);
    const bool automatic = normalized.empty() || normalized == "auto";
    const bool supported = automatic || (info && dc::Available(info->format, direction, features));
    const std::string label = automatic ? "Auto" : supported ? info->label : value + " (unavailable)";
    bool changed = false;
    if (ImGui::BeginCombo(id, label.c_str())) {
        if (ImGui::Selectable("Auto", automatic)) { value = "auto"; changed = true; }
        for (const auto& choice : dc::kFormats) {
            if (!dc::Available(choice.format, direction, features)) continue;
            if (ImGui::Selectable(choice.label, info == &choice)) {
                value = choice.name;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

} // namespace gui
