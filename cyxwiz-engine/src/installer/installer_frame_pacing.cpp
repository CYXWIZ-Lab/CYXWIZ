#include "installer_frame_pacing.h"

#include <GLFW/glfw3.h>

namespace cyxwiz::installer::gui {
namespace {

constexpr double kBusyFrameWaitSeconds = 1.0 / 60.0;
constexpr double kIdleFrameWaitSeconds = 1.0 / 30.0;
constexpr double kMinimizedFrameWaitSeconds = 0.25;

}  // namespace

void WaitForInstallerFrame(
    GLFWwindow* window,
    bool operation_running) {
    if (!window) return;
    const bool minimized =
        glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE;
    const double wait_seconds = minimized
        ? kMinimizedFrameWaitSeconds
        : (operation_running ? kBusyFrameWaitSeconds
                             : kIdleFrameWaitSeconds);
    glfwWaitEventsTimeout(wait_seconds);
}

}  // namespace cyxwiz::installer::gui
