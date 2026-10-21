// window_frame.cpp - the OS window frame follows the screen (TOFIX139 P7).
#include "gui/window_frame.h"

#include <GLFW/glfw3.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <dwmapi.h>
#pragma comment(lib, "dwmapi.lib")
#endif

#include <cmath>

namespace cyxwiz::servernode::gui {

void SetWindowFrameColor(const ImVec4& color) {
#ifdef _WIN32
    static COLORREF last = 0xFFFFFFFF;
    const auto channel = [](float v) { return static_cast<int>(std::lround(v * 255.0f)) & 0xFF; };
    const COLORREF rgb = static_cast<COLORREF>(channel(color.x) | (channel(color.y) << 8) | (channel(color.z) << 16));
    if (rgb == last) return;
    GLFWwindow* window = glfwGetCurrentContext();
    if (!window) return;
    HWND hwnd = glfwGetWin32Window(window);
    if (!hwnd) return;
    last = rgb;
    COLORREF value = rgb;
    DwmSetWindowAttribute(hwnd, 35, &value, sizeof(value));  // DWMWA_CAPTION_COLOR
    DwmSetWindowAttribute(hwnd, 34, &value, sizeof(value));  // DWMWA_BORDER_COLOR
#else
    (void)color;
#endif
}

}  // namespace cyxwiz::servernode::gui
