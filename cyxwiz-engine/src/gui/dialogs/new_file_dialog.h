#pragma once

// The one "New script" / "New file" dialog (TOFIX129 A2-4), used by
// File > New Script, Script > New Script and the Asset Browser's New > Script
// and New > File. Name, type, folder (never the Engine's own folder), path
// preview, name checks and the replace warning come from
// core/start_page_presentation.

#include <string>

namespace cyxwiz {

class NewFileDialog {
public:
    enum class Result { None, Created, Cancelled };
    // Script: a Python or CyxWiz script. File: a text file (Text, Markdown,
    // JSON, YAML, CSV, or the extension typed in the name).
    enum class Mode { Script, File };

    // Opens on the next Render. `folder` empty = the project's scripts folder
    // for a script, the project folder for a file, or no folder when no
    // project is open (the user chooses one).
    void Open(Mode mode, const std::string& folder = "");
    bool IsOpen() const { return open_; }

    Result Render();
    const std::string& CreatedPath() const { return created_path_; }

private:
    bool TryCreate();
    std::string FileName() const;

    Mode mode_ = Mode::Script;
    bool open_ = false;
    bool request_open_ = false;
    bool focus_name_ = false;
    bool python_ = true;
    int text_kind_ = 0;  // startpage::TextFileKind
    char name_[256] = {};
    char folder_[512] = {};
    std::string error_;
    std::string created_path_;
};

}  // namespace cyxwiz
