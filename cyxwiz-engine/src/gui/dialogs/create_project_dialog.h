#pragma once

// The one "Create a new project" dialog (TOFIX129 A2-1), used by the start
// page and by File > New Project. Templates, default location, path
// preview, name checks, the exists warning and the error text all come
// from core/start_page_presentation.

#include <string>

namespace cyxwiz {

class CreateProjectDialog {
public:
    enum class Result { None, Created, Cancelled };

    CreateProjectDialog();

    // Opens the dialog on the next Render. `template_index` preselects a
    // template and fills the name when it is empty.
    void Open(int template_index = 0);
    bool IsOpen() const { return open_; }

    // Call every frame. Returns Created once the project exists and is
    // open in the ProjectManager; CreatedProjectPath() then names its file.
    Result Render();
    const std::string& CreatedProjectPath() const { return created_path_; }

private:
    void SelectTemplate(int index);
    bool TryCreate();

    bool open_ = false;
    bool request_open_ = false;
    bool focus_name_ = false;
    bool submit_ = false;
    int template_index_ = 0;
    char name_[256] = {};
    char location_[512] = {};
    std::string error_;
    std::string created_path_;
};

}  // namespace cyxwiz
