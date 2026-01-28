#pragma once

#include <string>
#include <vector>
#include <TextEditor.h>

namespace gui {

// Supported syntax languages
enum class SyntaxLanguage {
    None,       // Plain text with line numbers
    Python,
    Cpp,
    XML,
    YAML,
    JSON,
    CSS,
    HTML,
    SQL,
    Markdown,
    INI,
    Shell
};

/**
 * Text preview with syntax highlighting and line numbers
 * Uses ImGuiColorTextEdit for syntax-highlighted files
 * Uses simple ImGui table for plain text
 */
class TextPreviewRenderer {
public:
    TextPreviewRenderer();
    ~TextPreviewRenderer();

    // Set content to display
    void SetContent(const std::string& content, SyntaxLanguage lang = SyntaxLanguage::None);
    void SetContentFromLines(const std::vector<std::string>& lines);

    // Render the text preview
    void Render();

    // Detect language from file extension
    static SyntaxLanguage DetectLanguage(const std::string& filepath);

    // Get language name
    static const char* GetLanguageName(SyntaxLanguage lang);

private:
    TextEditor editor_;
    bool is_initialized_ = false;
    SyntaxLanguage current_lang_ = SyntaxLanguage::None;
    std::string current_content_;
    bool use_syntax_ = false;  // true = TextEditor, false = plain line numbers

    // Language definitions
    static TextEditor::LanguageDefinition CreateCppLanguage();
    static TextEditor::LanguageDefinition CreateXMLLanguage();
    static TextEditor::LanguageDefinition CreateYAMLLanguage();
    static TextEditor::LanguageDefinition CreateJSONLanguage();
    static TextEditor::LanguageDefinition CreatePythonLanguage();
    static TextEditor::LanguageDefinition CreateShellLanguage();
    static TextEditor::LanguageDefinition CreateSQLLanguage();

    void ApplyLanguage(SyntaxLanguage lang);
    void RenderPlainText();
};

} // namespace gui
