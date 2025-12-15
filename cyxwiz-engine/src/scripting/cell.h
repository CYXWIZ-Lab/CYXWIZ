#pragma once

#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <random>
#include <TextEditor.h>

// Use GLAD for cross-platform OpenGL loading
#include <glad/glad.h>

namespace cyxwiz {

/**
 * Cell type enumeration
 */
enum class CellType {
    Code,       // Python code cell (executable)
    Markdown,   // Markdown documentation cell
    Raw         // Raw text cell (no execution/rendering)
};

/**
 * Cell execution state
 */
enum class CellState {
    Idle,       // Not running
    Queued,     // Waiting to run
    Running,    // Currently executing
    Success,    // Completed successfully
    Error       // Completed with error
};

/**
 * Output type for cell outputs
 */
enum class OutputType {
    Text,       // Plain text output (stdout)
    Error,      // Error/traceback output (stderr)
    Stream,     // Stream output (stdout/stderr with name)
    Image,      // Base64-encoded image (PNG)
    Plot,       // Matplotlib plot (rendered as image)
    Table,      // Tabular data
    Html,       // HTML content
    Markdown    // Rendered markdown
};

/**
 * Single output from a cell execution
 */
struct CellOutput {
    OutputType type = OutputType::Text;
    std::string data;               // Output content (text or base64 for images)
    std::string mime_type;          // MIME type (e.g., "text/plain", "image/png")
    std::string name;               // Stream name (stdout/stderr) or output name

    // For image/plot outputs
    GLuint texture_id = 0;          // OpenGL texture ID (0 = not loaded)
    int width = 0;                  // Image width
    int height = 0;                 // Image height
    std::vector<unsigned char> image_data;  // Raw PNG data (alternative to base64 in data)

    CellOutput() = default;

    CellOutput(OutputType t, const std::string& d, const std::string& mime = "text/plain")
        : type(t), data(d), mime_type(mime) {}

    // Text output helper
    static CellOutput Text(const std::string& text) {
        return CellOutput(OutputType::Text, text, "text/plain");
    }

    // Error output helper
    static CellOutput Error(const std::string& error) {
        return CellOutput(OutputType::Error, error, "text/plain");
    }

    // Stream output helper
    static CellOutput Stream(const std::string& text, const std::string& stream_name) {
        CellOutput out(OutputType::Stream, text, "text/plain");
        out.name = stream_name;
        return out;
    }

    // Plot output helper
    static CellOutput Plot(const std::string& base64_png, int w, int h) {
        CellOutput out(OutputType::Plot, base64_png, "image/png");
        out.width = w;
        out.height = h;
        return out;
    }
};

/**
 * Single cell in the notebook-style editor
 */
struct Cell {
    std::string id;                         // Unique cell identifier
    CellType type = CellType::Code;         // Cell type
    std::string source;                     // Cell source content
    std::vector<CellOutput> outputs;        // Execution outputs
    int execution_count = 0;                // In [ ] number (0 = not run yet)
    CellState state = CellState::Idle;      // Current execution state

    // UI state
    bool collapsed = false;                 // Cell input collapsed
    bool output_collapsed = false;          // Output area collapsed
    float editor_height = 100.0f;           // Height of editor area
    bool is_selected = false;               // Currently selected
    bool is_editing = false;                // Currently in edit mode

    // Editor instance (for code cells)
    TextEditor editor;

    // Breakpoints (line numbers)
    std::vector<int> breakpoints;

    Cell() {
        id = GenerateId();
    }

    Cell(CellType t, const std::string& src = "")
        : type(t), source(src) {
        id = GenerateId();
        if (t == CellType::Code) {
            SetupCodeEditor();
        }
    }

    // Generate unique cell ID
    static std::string GenerateId() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 15);
        static const char* hex = "0123456789abcdef";

        std::string id = "cell-";
        for (int i = 0; i < 8; ++i) {
            id += hex[dis(gen)];
        }
        return id;
    }

    // Setup code editor with Python syntax
    void SetupCodeEditor() {
        auto lang = TextEditor::LanguageDefinition::CPlusPlus();
        lang.mKeywords.clear();

        static const char* const py_keywords[] = {
            "and", "as", "assert", "break", "class", "continue", "def", "del",
            "elif", "else", "except", "False", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "None", "nonlocal", "not", "or",
            "pass", "raise", "return", "True", "try", "while", "with", "yield",
            "async", "await", "print", "len", "range", "str", "int", "float",
            "list", "dict", "set", "tuple", "bool", "type", "self", "cls"
        };

        for (const auto& k : py_keywords) {
            lang.mKeywords.insert(k);
        }

        lang.mSingleLineComment = "#";
        lang.mCommentStart = "\"\"\"";
        lang.mCommentEnd = "\"\"\"";
        lang.mName = "Python";

        editor.SetLanguageDefinition(lang);
        editor.SetShowWhitespaces(false);
        editor.SetTabSize(4);
        editor.SetText(source);
    }

    // Sync source from editor
    void SyncSourceFromEditor() {
        if (type == CellType::Code || type == CellType::Markdown) {
            source = editor.GetText();
        }
    }

    // Sync editor from source
    void SyncEditorFromSource() {
        if (type == CellType::Code || type == CellType::Markdown) {
            editor.SetText(source);
        }
    }

    // Clear outputs
    void ClearOutputs() {
        // Clean up textures
        for (auto& output : outputs) {
            if (output.texture_id != 0) {
                glDeleteTextures(1, &output.texture_id);
            }
        }
        outputs.clear();
        state = CellState::Idle;
    }

    // Add output
    void AddOutput(const CellOutput& output) {
        outputs.push_back(output);
    }

    // Check if cell has any output
    bool HasOutput() const {
        return !outputs.empty();
    }

    // Get display text for cell type
    const char* GetTypeLabel() const {
        switch (type) {
            case CellType::Code: return "Code";
            case CellType::Markdown: return "Markdown";
            case CellType::Raw: return "Raw";
            default: return "Unknown";
        }
    }

    // Get execution count display string
    std::string GetExecutionLabel() const {
        if (state == CellState::Running) {
            return "[*]";
        } else if (execution_count > 0) {
            return "[" + std::to_string(execution_count) + "]";
        } else {
            return "[ ]";
        }
    }
};

/**
 * Cell marker strings for .cyx file format
 */
namespace CellMarkers {
    constexpr const char* CODE = "%%code";
    constexpr const char* MARKDOWN = "%%markdown";
    constexpr const char* RAW = "%%raw";
    constexpr const char* LEGACY_SECTION = "%%";  // Legacy section marker
}

} // namespace cyxwiz
