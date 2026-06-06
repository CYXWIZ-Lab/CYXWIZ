// Toolbar file dialog wrappers.

#include "toolbar.h"
#include "../../core/file_dialogs.h"

#include <string>

namespace cyxwiz {

std::string ToolbarPanel::OpenFolderDialog() {
    auto result = FileDialogs::SelectFolder("Select Project Location");
    return result.value_or("");
}

std::string ToolbarPanel::OpenFileDialog(const char* filter, const char* title) {
    (void)filter;  // Filter format differs from nfd, using generic dialog
    auto result = FileDialogs::OpenFile(title, {{"All Files", "*"}});
    return result.value_or("");
}

} // namespace cyxwiz
