// Script Editor file content and dialog helpers.

#include "script_editor.h"
#include "../../core/file_dialogs.h"

#include <fstream>
#include <sstream>
#include <string>

namespace cyxwiz {

// Helper functions

bool ScriptEditorPanel::LoadFileContent(const std::string& filepath, std::string& content) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    content = buffer.str();
    file.close();
    return true;
}

bool ScriptEditorPanel::SaveFileContent(const std::string& filepath, const std::string& content) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file << content;
    file.close();
    return true;
}

std::string ScriptEditorPanel::OpenFileDialog() {
    auto result = FileDialogs::OpenScript();
    return result.value_or("");
}

std::string ScriptEditorPanel::SaveFileDialog() {
    auto result = FileDialogs::SaveScript();
    return result.value_or("");
}

} // namespace cyxwiz
