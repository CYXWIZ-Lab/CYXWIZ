#pragma once

// Shows a file or folder in the system file browser (Explorer selects it;
// macOS reveals it; Linux opens its folder). False when it could not start.
// Shared by the Asset Browser and the Plugin Manager's Python bundles.

#include <string>

namespace cyxwiz {

bool LaunchAssetInFileBrowser(const std::string& absolute_path, bool is_directory);

}  // namespace cyxwiz
