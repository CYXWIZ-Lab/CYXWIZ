#pragma once

// The column roles set in Data Studio, saved with the project (TOFIX134 P3,
// owner answer 2 of 2026-10-02): <project>/datasets/column_roles.json, one
// entry per dataset source (the file it was loaded from, or its name), so
// the roles survive reloads and every consumer (dashboards, Data Studio
// Profile and Visualize, Plot nodes) reads the same ones. Roles for columns
// that are gone are kept (re-applied if the column returns). Also keeps the
// texts a column uses for "missing" (N/A, -, none).

#include "dataset_contract.h"

#include <map>
#include <string>
#include <vector>

namespace cyxwiz {

struct DatasetUserSettings {
    std::map<std::string, ColumnRole> roles;                       // column -> role
    std::map<std::string, std::vector<std::string>> missing_text;  // column -> texts that mean missing
    std::string schema_fingerprint;                                // when last saved
};

class ColumnRoleStore {
public:
    // `file` is the JSON path (the Engine uses ProjectFile()).
    explicit ColumnRoleStore(std::string file = {});
    static std::string ProjectFile(const std::string& project_root);

    // Reads the file (a missing file is an empty store). false + error on a broken file
    // (the store is then empty and Save would not overwrite it).
    bool Load(std::string* error = nullptr);
    bool Save(std::string* error = nullptr) const;

    const DatasetUserSettings* Find(const std::string& source_key) const;
    void SetRole(const std::string& source_key, const std::string& column, ColumnRole role);
    void ClearRole(const std::string& source_key, const std::string& column);  // back to inferred
    void SetMissingText(const std::string& source_key, const std::string& column, std::vector<std::string> texts);
    void SetSchema(const std::string& source_key, const std::string& fingerprint);
    // The user roles for BuildContract.
    std::map<std::string, ColumnRole> RolesFor(const std::string& source_key) const;

    const std::string& File() const { return file_; }
    bool Broken() const { return broken_; }

private:
    std::string file_;
    std::map<std::string, DatasetUserSettings> datasets_;
    bool broken_ = false;
};

// The open project's store (UI thread): loaded on first use and again when
// another project opens; empty (and not saved) without a project.
ColumnRoleStore& ProjectColumnRoles();
// The settings key of a dataset: the file it came from, else its name.
std::string RoleSourceKey(const std::string& source_path, const std::string& dataset_name);

}  // namespace cyxwiz
