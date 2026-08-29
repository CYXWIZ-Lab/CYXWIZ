#pragma once

struct GLFWwindow;

namespace cyxwiz::installer::gui {

void WaitForInstallerFrame(
    GLFWwindow* window,
    bool operation_running);

}  // namespace cyxwiz::installer::gui
