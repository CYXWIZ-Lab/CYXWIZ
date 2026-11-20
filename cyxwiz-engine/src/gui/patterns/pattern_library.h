#pragma once

#include "pattern.h"
#include "../node_editor.h"
#include <map>
#include <vector>
#include <string>
#include <functional>
#include <filesystem>

namespace gui::patterns {

// Callback type for node creation - creates a node with proper pins based on type
// This allows PatternLibrary to delegate node creation to NodeEditor
using NodeCreatorCallback = std::function<MLNode(NodeType type, const std::string& name)>;

class PatternLibrary {
public:
    // Singleton access
    static PatternLibrary& Instance();

    // Delete copy/move constructors
    PatternLibrary(const PatternLibrary&) = delete;
    PatternLibrary& operator=(const PatternLibrary&) = delete;
    PatternLibrary(PatternLibrary&&) = delete;
    PatternLibrary& operator=(PatternLibrary&&) = delete;

    // Initialization
    void Initialize();
    bool IsInitialized() const { return initialized_; }

    // Load patterns from directories
    void LoadBuiltinPatterns();
    void LoadUserPatterns(const std::string& directory);
    bool LoadPatternFromFile(const std::string& filepath);

    // Query patterns
    std::vector<Pattern> GetAllPatterns() const;
    std::vector<Pattern> GetByCategory(PatternCategory category) const;
    std::vector<Pattern> Search(const std::string& query) const;
    const Pattern* GetPattern(const std::string& id) const;
    size_t GetPatternCount() const { return patterns_.size(); }
    // Changes whenever a pattern is read, saved or deleted (views rebuild then).
    size_t Revision() const { return revision_; }

    // Get all categories that have patterns
    std::vector<PatternCategory> GetAvailableCategories() const;

    // Instantiate a pattern into nodes and links. The callback creates each
    // node with its real pins (NodeEditor::CreateNode). All or nothing: on a
    // refusal the outputs are empty and `error` (when given) says why in
    // plain words (not ready, an unknown or template node type, a link to a
    // pin the node does not have).
    bool InstantiatePatternWithCreator(
        const std::string& pattern_id,
        const std::map<std::string, std::string>& params,
        std::vector<MLNode>& out_nodes,
        std::vector<NodeLink>& out_links,
        int& next_node_id,
        int& next_link_id,
        ImVec2 base_position,
        NodeCreatorCallback node_creator,
        std::string* error = nullptr
    );

    // Writes the selected nodes as one of My patterns (<user folder>/<file
    // name from `name`>.cyxgraph, replacing one of the same name) and reads it
    // in: node types by their file names, places relative to the selection
    // (`positions` in canvas grid space), links with their pins. A selection
    // with a Data Input and an Output is a full graph; otherwise each node an
    // outside link fed gets a Block input. Returns the new pattern's id, or
    // empty with the reason in `error`.
    std::string SavePatternFromSelection(
        const std::vector<MLNode>& nodes,
        const std::vector<NodeLink>& links,
        const std::map<int, ImVec2>& positions,
        const std::vector<int>& selected_ids,
        const std::string& name,
        const std::string& description,
        PatternCategory category,
        std::string* error = nullptr
    );
    // The file one of My patterns named `name` is saved in.
    std::filesystem::path UserPatternFile(const std::string& name) const;

    // Moves one of My patterns' files to the Recycle Bin (Trash) and drops
    // it from the library. False, with the reason, when it could not.
    bool DeleteUserPattern(const std::string& id, std::string* error = nullptr);

    // The user's own patterns (My patterns): the Engine sets it to
    // <user settings folder>/patterns before Initialize. Empty: none are read.
    const std::string& GetUserPatternsDirectory() const { return user_patterns_dir_; }
    void SetUserPatternsDirectory(const std::string& dir) { user_patterns_dir_ = dir; }

    // Get builtin patterns directory
    const std::string& GetBuiltinPatternsDirectory() const { return builtin_patterns_dir_; }

private:
    PatternLibrary();
    ~PatternLibrary() = default;

    bool LoadPatternFile(const std::string& filepath, bool user);

    // Parse pattern JSON
    bool ParsePatternJson(const std::string& json_content, Pattern& out_pattern);

    // Substitute parameters in a template string
    std::string SubstituteParams(
        const std::string& template_str,
        const std::map<std::string, std::string>& params
    ) const;

    // Convert node type string to NodeType enum
    NodeType StringToNodeType(const std::string& type_str) const;

    // Storage
    std::map<std::string, Pattern> patterns_;
    std::string user_patterns_dir_;
    std::string builtin_patterns_dir_;
    bool initialized_ = false;
    size_t revision_ = 0;
};

} // namespace gui::patterns
