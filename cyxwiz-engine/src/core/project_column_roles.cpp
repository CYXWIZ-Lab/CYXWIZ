// The open project's column-role store (column_role_store.h).

#include "column_role_store.h"
#include "project_manager.h"

#include <spdlog/spdlog.h>

#include <algorithm>

namespace cyxwiz {

ColumnRoleStore& ProjectColumnRoles() {
    static ColumnRoleStore store;
    static std::string loaded_root = "\x01";
    const std::string root = ProjectManager::Instance().GetProjectRoot();
    if (root != loaded_root) {
        loaded_root = root;
        store = ColumnRoleStore(ColumnRoleStore::ProjectFile(root));
        std::string error;
        if (!store.Load(&error)) spdlog::warn("Column roles: {}", error);
    }
    return store;
}

std::string RoleSourceKey(const std::string& source_path, const std::string& dataset_name) {
    if (source_path.empty()) return dataset_name;
    std::string key = source_path;
    std::replace(key.begin(), key.end(), '\\', '/');
    return key;
}

}  // namespace cyxwiz
