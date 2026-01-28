#include "text_preview.h"
#include <imgui.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <algorithm>
#include <sstream>

namespace gui {

TextPreviewRenderer::TextPreviewRenderer() {
    editor_.SetReadOnly(true);
    editor_.SetShowWhitespaces(false);
}

TextPreviewRenderer::~TextPreviewRenderer() = default;

void TextPreviewRenderer::SetContent(const std::string& content, SyntaxLanguage lang) {
    if (content == current_content_ && lang == current_lang_) return;

    current_content_ = content;
    current_lang_ = lang;
    use_syntax_ = (lang != SyntaxLanguage::None);

    if (use_syntax_) {
        ApplyLanguage(lang);
        editor_.SetText(content);
    }

    is_initialized_ = true;
}

void TextPreviewRenderer::SetContentFromLines(const std::vector<std::string>& lines) {
    std::string content;
    for (const auto& line : lines) {
        content += line + "\n";
    }
    SetContent(content, SyntaxLanguage::None);
}

void TextPreviewRenderer::Render() {
    if (!is_initialized_) {
        ImGui::TextWrapped("No content loaded.");
        return;
    }

    if (use_syntax_) {
        // Render with TextEditor (syntax highlighted)
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        editor_.Render("##TextPreview", ImGui::GetContentRegionAvail());
        ImGui::PopStyleVar();
    } else {
        RenderPlainText();
    }
}

void TextPreviewRenderer::RenderPlainText() {
    // Split into lines
    std::vector<std::string> lines;
    std::istringstream stream(current_content_);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    if (lines.empty()) {
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "(empty)");
        return;
    }

    // Status line
    ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1.0f), "%zu lines", lines.size());
    ImGui::Separator();

    ImGui::BeginChild("##PlainTextContent", ImVec2(0, 0), false);

    // Calculate line number width
    int digits = 1;
    size_t n = lines.size();
    while (n >= 10) { digits++; n /= 10; }
    float line_num_width = ImGui::CalcTextSize("0").x * (digits + 2);

    // Use clipper for performance with large files
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(lines.size()));

    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
            // Line number (dimmed)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
            ImGui::Text("%*d", digits, i + 1);
            ImGui::PopStyleColor();

            ImGui::SameLine(line_num_width);

            // Separator line
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            ImGui::Text("|");
            ImGui::PopStyleColor();

            ImGui::SameLine(line_num_width + ImGui::CalcTextSize("| ").x);

            // Content
            if (i < static_cast<int>(lines.size())) {
                ImGui::TextUnformatted(lines[i].c_str());
            }
        }
    }

    ImGui::EndChild();
}

SyntaxLanguage TextPreviewRenderer::DetectLanguage(const std::string& filepath) {
    std::string ext = std::filesystem::path(filepath).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".py" || ext == ".pyw" || ext == ".pyx")
        return SyntaxLanguage::Python;
    if (ext == ".cpp" || ext == ".c" || ext == ".h" || ext == ".hpp" || ext == ".cc" || ext == ".cxx" ||
        ext == ".java" || ext == ".cs" || ext == ".js" || ext == ".ts" || ext == ".jsx" || ext == ".tsx" ||
        ext == ".rs" || ext == ".go")
        return SyntaxLanguage::Cpp;  // C-family syntax
    if (ext == ".xml" || ext == ".html" || ext == ".htm" || ext == ".svg" || ext == ".xhtml")
        return SyntaxLanguage::XML;
    if (ext == ".yaml" || ext == ".yml" || ext == ".toml")
        return SyntaxLanguage::YAML;
    if (ext == ".json" || ext == ".geojson")
        return SyntaxLanguage::JSON;
    if (ext == ".css" || ext == ".scss")
        return SyntaxLanguage::CSS;
    if (ext == ".sql")
        return SyntaxLanguage::SQL;
    if (ext == ".sh" || ext == ".bash" || ext == ".bat" || ext == ".ps1" || ext == ".cmake")
        return SyntaxLanguage::Shell;
    if (ext == ".md" || ext == ".markdown")
        return SyntaxLanguage::Markdown;
    if (ext == ".ini" || ext == ".cfg" || ext == ".conf")
        return SyntaxLanguage::INI;

    return SyntaxLanguage::None;
}

const char* TextPreviewRenderer::GetLanguageName(SyntaxLanguage lang) {
    switch (lang) {
        case SyntaxLanguage::Python:   return "Python";
        case SyntaxLanguage::Cpp:      return "C/C++";
        case SyntaxLanguage::XML:      return "XML/HTML";
        case SyntaxLanguage::YAML:     return "YAML";
        case SyntaxLanguage::JSON:     return "JSON";
        case SyntaxLanguage::CSS:      return "CSS";
        case SyntaxLanguage::SQL:      return "SQL";
        case SyntaxLanguage::Shell:    return "Shell";
        case SyntaxLanguage::HTML:     return "HTML";
        case SyntaxLanguage::Markdown: return "Markdown";
        case SyntaxLanguage::INI:      return "INI";
        default:                       return "Plain Text";
    }
}

void TextPreviewRenderer::ApplyLanguage(SyntaxLanguage lang) {
    switch (lang) {
        case SyntaxLanguage::Python:
            editor_.SetLanguageDefinition(CreatePythonLanguage());
            break;
        case SyntaxLanguage::Cpp:
            editor_.SetLanguageDefinition(CreateCppLanguage());
            break;
        case SyntaxLanguage::XML:
        case SyntaxLanguage::HTML:
            editor_.SetLanguageDefinition(CreateXMLLanguage());
            break;
        case SyntaxLanguage::YAML:
        case SyntaxLanguage::INI:
            editor_.SetLanguageDefinition(CreateYAMLLanguage());
            break;
        case SyntaxLanguage::JSON:
            editor_.SetLanguageDefinition(CreateJSONLanguage());
            break;
        case SyntaxLanguage::SQL:
            editor_.SetLanguageDefinition(CreateSQLLanguage());
            break;
        case SyntaxLanguage::Shell:
            editor_.SetLanguageDefinition(CreateShellLanguage());
            break;
        default:
            editor_.SetLanguageDefinition(TextEditor::LanguageDefinition::C());
            break;
    }
}

TextEditor::LanguageDefinition TextPreviewRenderer::CreatePythonLanguage() {
    static TextEditor::LanguageDefinition lang;
    static bool inited = false;
    if (!inited) {
        lang.mName = "Python";
        lang.mCaseSensitive = true;
        lang.mAutoIndentation = true;
        lang.mSingleLineComment = "#";
        lang.mCommentStart = "\"\"\"";
        lang.mCommentEnd = "\"\"\"";

        static const char* const keywords[] = {
            "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
            "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is",
            "lambda", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
            "while", "with", "yield", "async", "await"
        };
        for (auto& k : keywords) lang.mKeywords.insert(k);

        static const char* const ids[] = {
            "abs", "all", "any", "bin", "bool", "dict", "enumerate", "filter", "float",
            "format", "getattr", "hasattr", "int", "isinstance", "len", "list", "map",
            "max", "min", "next", "open", "print", "range", "repr", "set", "sorted",
            "str", "sum", "super", "tuple", "type", "zip"
        };
        for (auto& i : ids) {
            TextEditor::Identifier id;
            id.mDeclaration = "Built-in";
            lang.mIdentifiers.insert(std::make_pair(std::string(i), id));
        }
        inited = true;
    }
    return lang;
}

TextEditor::LanguageDefinition TextPreviewRenderer::CreateCppLanguage() {
    return TextEditor::LanguageDefinition::CPlusPlus();
}

TextEditor::LanguageDefinition TextPreviewRenderer::CreateXMLLanguage() {
    static TextEditor::LanguageDefinition lang;
    static bool inited = false;
    if (!inited) {
        lang.mName = "XML";
        lang.mCaseSensitive = true;
        lang.mCommentStart = "<!--";
        lang.mCommentEnd = "-->";
        lang.mSingleLineComment = "";

        static const char* const keywords[] = {
            "xml", "version", "encoding", "xmlns", "xsi", "type", "name",
            "id", "class", "style", "src", "href", "alt", "title"
        };
        for (auto& k : keywords) lang.mKeywords.insert(k);
        inited = true;
    }
    return lang;
}

TextEditor::LanguageDefinition TextPreviewRenderer::CreateYAMLLanguage() {
    static TextEditor::LanguageDefinition lang;
    static bool inited = false;
    if (!inited) {
        lang.mName = "YAML";
        lang.mCaseSensitive = true;
        lang.mSingleLineComment = "#";

        static const char* const keywords[] = {
            "true", "false", "null", "yes", "no", "on", "off"
        };
        for (auto& k : keywords) lang.mKeywords.insert(k);
        inited = true;
    }
    return lang;
}

TextEditor::LanguageDefinition TextPreviewRenderer::CreateJSONLanguage() {
    static TextEditor::LanguageDefinition lang;
    static bool inited = false;
    if (!inited) {
        lang.mName = "JSON";
        lang.mCaseSensitive = true;
        lang.mSingleLineComment = "";

        static const char* const keywords[] = {
            "true", "false", "null"
        };
        for (auto& k : keywords) lang.mKeywords.insert(k);
        inited = true;
    }
    return lang;
}

TextEditor::LanguageDefinition TextPreviewRenderer::CreateShellLanguage() {
    static TextEditor::LanguageDefinition lang;
    static bool inited = false;
    if (!inited) {
        lang.mName = "Shell";
        lang.mCaseSensitive = true;
        lang.mSingleLineComment = "#";

        static const char* const keywords[] = {
            "if", "then", "else", "elif", "fi", "case", "esac", "for", "while", "do",
            "done", "in", "function", "return", "exit", "export", "source", "local",
            "echo", "read", "set", "unset", "shift", "eval", "exec", "cd", "pwd", "ls"
        };
        for (auto& k : keywords) lang.mKeywords.insert(k);
        inited = true;
    }
    return lang;
}

TextEditor::LanguageDefinition TextPreviewRenderer::CreateSQLLanguage() {
    return TextEditor::LanguageDefinition::SQL();
}

} // namespace gui
