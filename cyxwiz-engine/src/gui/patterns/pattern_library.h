#pragma once

#include "pattern.h"
#include "../node_editor.h"
#include <map>
#include <vector>
#include <string>
#include <functional>

namespace gui::patterns {

// Forward declaration for instantiation callback
class PatternLibrary;

// Callback type for pattern instantiation (legacy)
using PatternInstantiateCallback = std::function<void(
    const std::vector<MLNode>& nodes,
    const std::vector<NodeLink>& links
)>;

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

    // Get all categories that have patterns
    std::vector<PatternCategory> GetAvailableCategories() const;

    // Instantiate a pattern into nodes and links
    // Legacy version - creates simplified nodes with only 1 input/1 output pin
    bool InstantiatePattern(
        const std::string& pattern_id,
        const std::map<std::string, std::string>& params,
        std::vector<MLNode>& out_nodes,
        std::vector<NodeLink>& out_links,
        int& next_node_id,
        int& next_pin_id,
        int& next_link_id,
        ImVec2 base_position = ImVec2(0, 0)
    );

    // New version with node creator callback - creates nodes with proper pins
    // The callback should use NodeEditor::CreateNode() to create nodes
    bool InstantiatePatternWithCreator(
        const std::string& pattern_id,
        const std::map<std::string, std::string>& params,
        std::vector<MLNode>& out_nodes,
        std::vector<NodeLink>& out_links,
        int& next_node_id,
        int& next_link_id,
        ImVec2 base_position,
        NodeCreatorCallback node_creator
    );

    // Save a selection of nodes as a custom pattern
    bool SavePatternFromSelection(
        const std::vector<MLNode>& nodes,
        const std::vector<NodeLink>& links,
        const std::vector<int>& selected_ids,
        const std::string& name,
        const std::string& description,
        PatternCategory category,
        const std::string& save_path = ""
    );

    // Get/set user patterns directory
    const std::string& GetUserPatternsDirectory() const { return user_patterns_dir_; }
    void SetUserPatternsDirectory(const std::string& dir) { user_patterns_dir_ = dir; }

    // Get builtin patterns directory
    const std::string& GetBuiltinPatternsDirectory() const { return builtin_patterns_dir_; }

private:
    PatternLibrary();
    ~PatternLibrary() = default;

    // Parse pattern JSON
    bool ParsePatternJson(const std::string& json_content, Pattern& out_pattern);

    // Substitute parameters in a template string
    std::string SubstituteParams(
        const std::string& template_str,
        const std::map<std::string, std::string>& params
    ) const;

    // Convert node type string to NodeType enum
    NodeType StringToNodeType(const std::string& type_str) const;

    // Create an MLNode from a PatternNode with parameter substitution
    MLNode CreateNodeFromPattern(
        const PatternNode& pattern_node,
        const std::map<std::string, std::string>& params,
        int& node_id,
        int& pin_id
    ) const;

    // Storage
    std::map<std::string, Pattern> patterns_;
    std::string user_patterns_dir_;
    std::string builtin_patterns_dir_;
    bool initialized_ = false;
};

} // namespace gui::patterns
