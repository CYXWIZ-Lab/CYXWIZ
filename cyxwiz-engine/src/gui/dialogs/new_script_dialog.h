#pragma once

// The one "New script" dialog (TOFIX129 A2-4), used by File > New Script,
// Script > New Script and the Asset Browser's New > Script. Name, type,
// folder (the project's scripts folder by default, never the Engine's own
// folder), path preview, name checks and the replace warning come from
// core/start_page_presentation.

#include <string>

namespace cyxwiz {

class NewScriptDialog {
public:
    enum class Result { None, Created, Cancelled };

    // Opens on the next Render. `folder` empty = the project's scripts folder,
    // or no folder when no project is open (the user chooses one).
    void Open(const std::string& folder = "");
    bool IsOpen() const { return open_; }

    Result Render();
    const std::string& CreatedPath() const { return created_path_; }

private:
    bool TryCreate();

    bool open_ = false;
    bool request_open_ = false;
    bool focus_name_ = false;
    bool python_ = true;
    char name_[256] = {};
    char folder_[512] = {};
    std::string error_;
    std::string created_path_;
};

}  // namespace cyxwiz
