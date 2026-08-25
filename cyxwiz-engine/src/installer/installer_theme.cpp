#include "installer_theme.h"

#include "gui/icons.h"

// glad must own the OpenGL declarations before GLFW includes platform headers.
// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <system_error>

namespace cyxwiz::installer::gui {
namespace {

constexpr ImVec4 Rgb(int red, int green, int blue, int alpha = 255) {
  return ImVec4(
      static_cast<float>(red) / 255.0f, static_cast<float>(green) / 255.0f,
      static_cast<float>(blue) / 255.0f, static_cast<float>(alpha) / 255.0f);
}

std::filesystem::path
ResourcePath(const std::filesystem::path &executable_directory,
             const std::filesystem::path &relative) {
  const std::array candidates = {executable_directory / "resources" / relative,
                                 executable_directory / ".." / "resources" /
                                     relative,
                                 executable_directory / ".." / ".." /
                                     "cyxwiz-engine" / "resources" / relative};
  std::error_code error;
  for (const auto &candidate : candidates) {
    if (std::filesystem::is_regular_file(candidate, error)) {
      return candidate;
    }
    error.clear();
  }
  return {};
}

void AppendWarning(std::string &warning, const std::string &message) {
  if (!warning.empty())
    warning += "\n";
  warning += message;
}

} // namespace

ImVec4 InstallerBrandAccent() { return Rgb(124, 91, 246); }
ImVec4 InstallerCanvasColor() { return Rgb(7, 15, 29); }

void ApplyInstallerTheme(float scale) {
  ImGui::StyleColorsDark();
  auto &style = ImGui::GetStyle();
  style.WindowPadding = ImVec2(24.0f, 20.0f);
  style.FramePadding = ImVec2(12.0f, 8.0f);
  style.CellPadding = ImVec2(12.0f, 9.0f);
  style.ItemSpacing = ImVec2(10.0f, 10.0f);
  style.ItemInnerSpacing = ImVec2(8.0f, 6.0f);
  style.WindowRounding = 10.0f;
  style.ChildRounding = 9.0f;
  style.FrameRounding = 6.0f;
  style.PopupRounding = 9.0f;
  style.ScrollbarRounding = 9.0f;
  style.GrabRounding = 6.0f;
  style.TabRounding = 6.0f;
  style.TabBorderSize = 0.0f;
  style.TabBarBorderSize = 0.0f;
  style.WindowBorderSize = 0.0f;
  style.ChildBorderSize = 0.0f;
  style.FrameBorderSize = 0.0f;
  style.ScaleAllSizes(std::max(1.0f, scale));

  const ImVec4 accent = InstallerBrandAccent();
  const ImVec4 canvas = InstallerCanvasColor();
  auto &colors = style.Colors;
  colors[ImGuiCol_Text] = Rgb(233, 240, 250);
  colors[ImGuiCol_TextDisabled] = Rgb(139, 158, 184);
  colors[ImGuiCol_WindowBg] = canvas;
  colors[ImGuiCol_ChildBg] = canvas;
  colors[ImGuiCol_PopupBg] = Rgb(13, 27, 46);
  colors[ImGuiCol_Border] = Rgb(43, 66, 95);
  colors[ImGuiCol_BorderShadow] = Rgb(0, 0, 0, 0);
  colors[ImGuiCol_FrameBg] = Rgb(16, 35, 58);
  colors[ImGuiCol_FrameBgHovered] = Rgb(24, 50, 81);
  colors[ImGuiCol_FrameBgActive] = Rgb(29, 60, 96);
  colors[ImGuiCol_TitleBg] = Rgb(7, 15, 29);
  colors[ImGuiCol_TitleBgActive] = Rgb(7, 15, 29);
  colors[ImGuiCol_MenuBarBg] = Rgb(9, 20, 36);
  colors[ImGuiCol_ScrollbarBg] = Rgb(8, 18, 32);
  colors[ImGuiCol_ScrollbarGrab] = Rgb(48, 73, 104);
  colors[ImGuiCol_ScrollbarGrabHovered] = Rgb(61, 92, 130);
  colors[ImGuiCol_ScrollbarGrabActive] = Rgb(75, 111, 154);
  colors[ImGuiCol_CheckMark] = accent;
  colors[ImGuiCol_SliderGrab] = accent;
  colors[ImGuiCol_SliderGrabActive] = Rgb(145, 116, 252);
  colors[ImGuiCol_Button] = accent;
  colors[ImGuiCol_ButtonHovered] = Rgb(143, 113, 252);
  colors[ImGuiCol_ButtonActive] = Rgb(102, 72, 218);
  colors[ImGuiCol_Header] = Rgb(53, 42, 91);
  colors[ImGuiCol_HeaderHovered] = Rgb(74, 56, 128);
  colors[ImGuiCol_HeaderActive] = Rgb(91, 67, 159);
  colors[ImGuiCol_Separator] = Rgb(37, 58, 84);
  colors[ImGuiCol_SeparatorHovered] = accent;
  colors[ImGuiCol_SeparatorActive] = accent;
  colors[ImGuiCol_ResizeGrip] = Rgb(124, 91, 246, 80);
  colors[ImGuiCol_ResizeGripHovered] = Rgb(124, 91, 246, 160);
  colors[ImGuiCol_ResizeGripActive] = Rgb(124, 91, 246, 210);
  colors[ImGuiCol_Tab] = Rgb(14, 30, 50);
  colors[ImGuiCol_TabHovered] = Rgb(65, 50, 111);
  colors[ImGuiCol_TabSelected] = Rgb(82, 60, 145);
  colors[ImGuiCol_TabDimmed] = Rgb(10, 22, 38);
  colors[ImGuiCol_TabDimmedSelected] = Rgb(47, 38, 78);
  colors[ImGuiCol_TableHeaderBg] = Rgb(17, 35, 58);
  colors[ImGuiCol_TableBorderStrong] = Rgb(43, 66, 95);
  colors[ImGuiCol_TableBorderLight] = Rgb(31, 50, 73);
  colors[ImGuiCol_TableRowBg] = Rgb(11, 24, 41);
  colors[ImGuiCol_TableRowBgAlt] = Rgb(14, 29, 49);
  colors[ImGuiCol_NavHighlight] = accent;
}

InstallerVisualAssets
LoadInstallerVisualAssets(GLFWwindow *window,
                          const std::filesystem::path &executable_directory,
                          float scale, std::string &warning) {
  InstallerVisualAssets assets;
  auto &io = ImGui::GetIO();
  const auto body_path =
      ResourcePath(executable_directory, "fonts/Inter-Regular.ttf");
  const auto heading_path =
      ResourcePath(executable_directory, "fonts/Inter-Bold.ttf");
  const auto icon_font_path =
      ResourcePath(executable_directory, "fonts/fa-solid-900.ttf");
  const float body_size = 16.0f * std::max(1.0f, scale);
  const float heading_size = 25.0f * std::max(1.0f, scale);
  if (!body_path.empty()) {
    assets.body_font =
        io.Fonts->AddFontFromFileTTF(body_path.string().c_str(), body_size);
    io.FontDefault = assets.body_font;
  }
  bool icons_loaded = false;
  if (assets.body_font && !icon_font_path.empty()) {
    ImFontConfig config;
    config.MergeMode = true;
    config.PixelSnapH = true;
    config.GlyphMinAdvanceX = body_size;
    static constexpr ImWchar ranges[] = {ICON_MIN_FA, ICON_MAX_FA, 0};
    icons_loaded = io.Fonts->AddFontFromFileTTF(
                       icon_font_path.string().c_str(), body_size, &config,
                       ranges) != nullptr;
  }
  if (!heading_path.empty()) {
    assets.heading_font = io.Fonts->AddFontFromFileTTF(
        heading_path.string().c_str(), heading_size);
  }
  if (!assets.body_font || !assets.heading_font || !icons_loaded) {
    AppendWarning(warning, "Packaged typography or icon fonts were not found; "
                           "using the built-in UI font.");
  }

  const auto logo_path = ResourcePath(executable_directory, "cyxwiz.png");
  int channels = 0;
  unsigned char *pixels =
      logo_path.empty()
          ? nullptr
          : stbi_load(logo_path.string().c_str(), &assets.logo_width,
                      &assets.logo_height, &channels, 4);
  if (pixels) {
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, assets.logo_width,
                 assets.logo_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    assets.logo_texture = static_cast<ImTextureID>(texture);

    GLFWimage icon{assets.logo_width, assets.logo_height, pixels};
    glfwSetWindowIcon(window, 1, &icon);
    stbi_image_free(pixels);
  } else {
    AppendWarning(
        warning,
        "The CyxWiz logo could not be loaded from the packaged resources.");
  }
  return assets;
}

void DestroyInstallerVisualAssets(InstallerVisualAssets &assets) {
  if (assets.logo_texture != 0) {
    const auto texture = static_cast<GLuint>(assets.logo_texture);
    glDeleteTextures(1, &texture);
  }
  assets = {};
}

} // namespace cyxwiz::installer::gui
