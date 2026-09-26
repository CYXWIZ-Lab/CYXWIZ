#pragma once

#include <cstdlib>
#include <filesystem>

namespace cyxwiz::servernode {

// Where remote jobs put their fetched dataset files: CYXWIZ_NODE_DATA_DIR if
// set (put it on a data drive), else the temp folder.
inline std::filesystem::path NodeDataRoot() {
    if (const char* configured = std::getenv("CYXWIZ_NODE_DATA_DIR"); configured && *configured) {
        return configured;
    }
    return std::filesystem::temp_directory_path() / "cyxwiz-node";
}

}  // namespace cyxwiz::servernode
