// texture.h - images for ImGui (TOFIX139 P7). A loaded PNG becomes a GL
// texture the draw list can show.
#pragma once

#include <imgui.h>

#include <string>

namespace cyxwiz::servernode::gui {

struct Texture {
    unsigned int id = 0;  // GL texture name; 0 = not loaded
    int width = 0;
    int height = 0;
    ImVec4 edge_color = ImVec4(0, 0, 0, 1);  // the image's top-left pixel (its background)

    bool Loaded() const { return id != 0; }
    ImTextureID ImGuiId() const { return static_cast<ImTextureID>(static_cast<intptr_t>(id)); }
};

// Loads a PNG (any channel count) into an RGBA texture. Returns false and
// logs when the file is missing or unreadable.
bool LoadTexture(const std::string& path, Texture& out);

// Finds a node resource next to the executable or in the source tree (the same
// places the window icon is looked for). Empty when not found.
std::string FindResourceFile(const std::string& file_name);

}  // namespace cyxwiz::servernode::gui
