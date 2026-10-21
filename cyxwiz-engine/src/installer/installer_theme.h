#pragma once

#include <imgui.h>

#include <filesystem>
#include <string>

struct GLFWwindow;

namespace cyxwiz::installer::gui {

struct InstallerVisualAssets {
  ImFont *body_font = nullptr;
  ImFont *heading_font = nullptr;
  ImTextureID logo_texture = 0;
  int logo_width = 0;
  int logo_height = 0;
};

ImVec4 InstallerBrandAccent();
ImVec4 InstallerCanvasColor();

void ApplyInstallerTheme(float scale);

InstallerVisualAssets
LoadInstallerVisualAssets(GLFWwindow *window,
                          const std::filesystem::path &executable_directory,
                          float scale, std::string &warning);

void DestroyInstallerVisualAssets(InstallerVisualAssets &assets);

} // namespace cyxwiz::installer::gui
