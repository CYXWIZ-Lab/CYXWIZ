#pragma once

// Undefine Windows macros that conflict with our method names
#ifdef CreateDialog
#undef CreateDialog
#endif
#ifdef CreateDialogA
#undef CreateDialogA
#endif
#ifdef CreateDialogW
#undef CreateDialogW
#endif

#include <string>
#include <map>
#include <functional>
#include <memory>
#include <imgui.h>
#include "node_editor.h"

namespace gui {

/**
 * NodeConfigDialog - Base class for KNIME-style node configuration dialogs
 *
 * These dialogs provide detailed configuration for complex nodes with:
 * - Tabbed interface (Settings, Advanced, Preview, etc.)
 * - File browsers with preview
 * - Real-time data preview tables
 * - Validation feedback
 */
class NodeConfigDialog {
public:
    NodeConfigDialog(const std::string& title, MLNode* node);
    virtual ~NodeConfigDialog() = default;

    // Render the dialog (call in main render loop when open)
    // Returns true if dialog should remain open, false if closed
    bool Render();

    // Open/close the dialog
    void Open();
    void Close();
    bool IsOpen() const { return is_open_; }

    // Apply changes to the node
    virtual void Apply() = 0;

    // Reset to original values
    virtual void Reset() = 0;

    // Get dialog size (can be overridden for larger dialogs)
    virtual ImVec2 GetDefaultSize() const { return ImVec2(600, 500); }

protected:
    // Render the dialog content (implemented by derived classes)
    virtual void RenderContent() = 0;

    // Render common tabs (can be used by derived classes)
    void RenderSettingsTab();
    void RenderAdvancedTab();
    void RenderPreviewTab();

    // Utility methods for common UI patterns
    bool FileSelector(const char* label, std::string& path, const char* filter = "All Files\0*.*\0");
    void ValidationMessage(const std::string& message, bool is_error = true);

    std::string title_;
    MLNode* node_;
    bool is_open_ = false;
    bool first_open_ = true;

    // Tab management
    int current_tab_ = 0;

    // Changed flag for apply/reset
    bool has_changes_ = false;

    // Original parameter backup for reset
    std::map<std::string, std::string> original_params_;
};

/**
 * CSVReaderDialog - Configuration dialog for CSV/File Input nodes
 *
 * Features:
 * - File path selection with browser
 * - Column delimiter selection
 * - Header row options
 * - Data preview table
 * - Row/column limit settings
 */
class CSVReaderDialog : public NodeConfigDialog {
public:
    CSVReaderDialog(MLNode* node);

    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(800, 600); }

protected:
    void RenderContent() override;

private:
    void RenderFileSettingsTab();
    void RenderTransformationTab();
    void RenderAdvancedSettingsTab();
    void RenderLimitRowsTab();
    void RenderPreviewTab();

    void LoadPreview();

    // Settings
    char file_path_[512] = {};
    int delimiter_idx_ = 0;  // 0=comma, 1=semicolon, 2=tab, 3=custom
    char custom_delimiter_[8] = {};
    bool has_header_ = true;
    bool skip_empty_lines_ = true;
    int skip_first_n_ = 0;
    int limit_rows_ = 1000;

    // Preview data
    std::vector<std::string> preview_headers_;
    std::vector<std::vector<std::string>> preview_rows_;
    bool preview_loaded_ = false;
    std::string preview_error_;
};

/**
 * TokenizerDialog - Configuration dialog for Text Tokenization nodes
 *
 * Features:
 * - Tokenization method selection (word, character, subword)
 * - Vocabulary settings
 * - Special tokens configuration
 * - Token preview with highlighting
 */
class TokenizerDialog : public NodeConfigDialog {
public:
    TokenizerDialog(MLNode* node);

    void Apply() override;
    void Reset() override;
    ImVec2 GetDefaultSize() const override { return ImVec2(700, 550); }

protected:
    void RenderContent() override;

private:
    void RenderTokenizerTab();
    void RenderVocabularyTab();
    void RenderSpecialTokensTab();
    void RenderPreviewTab();

    // Settings
    int tokenizer_type_ = 0;  // 0=word, 1=char, 2=subword
    int max_vocab_size_ = 10000;
    int min_frequency_ = 2;
    bool lowercase_ = true;
    bool remove_punctuation_ = false;

    char pad_token_[32] = "[PAD]";
    char unk_token_[32] = "[UNK]";
    char sos_token_[32] = "[SOS]";
    char eos_token_[32] = "[EOS]";

    // Preview
    char sample_text_[1024] = "Hello world! This is a sample text for tokenization preview.";
    std::vector<std::string> preview_tokens_;
};

/**
 * FilterDialog - Configuration dialog for Filter/Query nodes
 *
 * Features:
 * - Column selection
 * - Condition builder
 * - Multiple condition support (AND/OR)
 * - Result preview
 */
class FilterDialog : public NodeConfigDialog {
public:
    FilterDialog(MLNode* node);

    void Apply() override;
    void Reset() override;

protected:
    void RenderContent() override;

private:
    struct FilterCondition {
        int column_idx = 0;
        int operator_idx = 0;  // 0=equals, 1=not equals, 2=>, 3=<, 4=>=, 5=<=, 6=contains
        char value[256] = {};
        int join_type = 0;  // 0=AND, 1=OR
    };

    std::vector<FilterCondition> conditions_;
    std::vector<std::string> available_columns_;
};

/**
 * NodeConfigDialogFactory - Creates appropriate dialog for a node type
 */
class NodeConfigDialogFactory {
public:
    static NodeConfigDialogFactory& Instance();

    // Register a dialog creator for a node type
    using DialogCreator = std::function<std::unique_ptr<NodeConfigDialog>(MLNode*)>;
    void RegisterDialog(NodeType type, DialogCreator creator);

    // Check if a node type has a configuration dialog
    bool HasDialog(NodeType type) const;

    // Create a dialog for a node
    std::unique_ptr<NodeConfigDialog> CreateDialog(MLNode* node);

private:
    NodeConfigDialogFactory();
    std::map<NodeType, DialogCreator> creators_;
};

// Helper function to check if a node type should have "Open Dialog" button
bool ShouldShowOpenDialogButton(NodeType type);

} // namespace gui
