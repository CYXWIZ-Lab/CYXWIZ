#pragma once

#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <functional>

namespace scripting {

class ScriptingEngine;

/**
 * Completion item for auto-complete suggestions
 */
struct CompletionItem {
    enum class Kind {
        Keyword,      // Python keywords (def, class, if, etc.)
        Builtin,      // Built-in functions (print, len, range, etc.)
        Module,       // Import modules
        Function,     // User-defined or library function
        Class,        // Class name
        Variable,     // Variable name
        Method,       // Object method
        Property,     // Object property
        Snippet       // Code snippet template
    };

    std::string label;           // Display text
    std::string insert_text;     // Text to insert (may include template)
    std::string detail;          // Type info or short description
    std::string documentation;   // Full documentation
    Kind kind;
    int score;                   // Relevance score for sorting

    CompletionItem() : kind(Kind::Keyword), score(0) {}
    CompletionItem(const std::string& lbl, Kind k, const std::string& det = "")
        : label(lbl), insert_text(lbl), detail(det), kind(k), score(0) {}
};

/**
 * ScriptManager - Handles auto-completion and code intelligence for Python scripts
 *
 * Features:
 * - Keyword and builtin completion (offline, always available)
 * - Python introspection for object attributes (when interpreter available)
 * - Code snippets for common patterns
 * - Module import suggestions
 */
class ScriptManager {
public:
    ScriptManager();
    ~ScriptManager();

    /**
     * Initialize with scripting engine for Python introspection
     */
    void Initialize(ScriptingEngine* engine);

    /**
     * Get completions for current cursor position
     * @param code Full code text
     * @param cursor_pos Cursor position in code
     * @param line Current line text
     * @param column Cursor column in line
     * @return List of completion items sorted by relevance
     */
    std::vector<CompletionItem> GetCompletions(
        const std::string& code,
        size_t cursor_pos,
        const std::string& line,
        int column
    );

    /**
     * Get completions for object attributes (e.g., "np." -> ["array", "zeros", ...])
     * @param object_expr Expression before the dot
     * @return List of attribute completions
     */
    std::vector<CompletionItem> GetAttributeCompletions(const std::string& object_expr);

    /**
     * Get function signature/documentation
     * @param func_name Function name
     * @return Documentation string or empty if not found
     */
    std::string GetFunctionSignature(const std::string& func_name);

    /**
     * Check if character should trigger auto-completion
     */
    bool ShouldTriggerCompletion(char c) const;

    /**
     * Get word at/before cursor position
     */
    static std::string GetWordAtCursor(const std::string& line, int column);

    /**
     * Extract the object expression before a dot (e.g., "self.model" from "self.model.")
     */
    static std::string GetObjectBeforeDot(const std::string& line, int column);

private:
    ScriptingEngine* scripting_engine_ = nullptr;

    // Static keyword/builtin database
    std::unordered_set<std::string> keywords_;
    std::unordered_set<std::string> builtins_;
    std::unordered_map<std::string, std::string> builtin_signatures_;
    std::vector<CompletionItem> snippets_;
    std::unordered_set<std::string> common_modules_;

    // Initialize static databases
    void InitializeKeywords();
    void InitializeBuiltins();
    void InitializeSnippets();
    void InitializeModules();

    // Completion helpers
    std::vector<CompletionItem> GetKeywordCompletions(const std::string& prefix);
    std::vector<CompletionItem> GetBuiltinCompletions(const std::string& prefix);
    std::vector<CompletionItem> GetSnippetCompletions(const std::string& prefix);
    std::vector<CompletionItem> GetModuleCompletions(const std::string& prefix);

    // Python introspection (requires GIL)
    std::vector<CompletionItem> GetPythonCompletions(
        const std::string& code,
        const std::string& prefix
    );

    // Score and sort completions
    void ScoreCompletions(std::vector<CompletionItem>& items, const std::string& prefix);
    static int CalculateMatchScore(const std::string& item, const std::string& prefix);
};

/**
 * Get icon character for completion kind (for UI display)
 */
const char* GetCompletionKindIcon(CompletionItem::Kind kind);

/**
 * Get color for completion kind (ImGui ImU32 color)
 */
unsigned int GetCompletionKindColor(CompletionItem::Kind kind);

} // namespace scripting
