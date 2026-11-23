// Toolbar file dialog wrappers.

#include "toolbar.h"
#include "../../core/file_dialogs.h"

#include <string>

namespace cyxwiz {

std::string ToolbarPanel::OpenFolderDialog() {
    auto result = FileDialogs::SelectFolder("Select Project Location");
    return result.value_or("");
}

} // namespace cyxwiz
