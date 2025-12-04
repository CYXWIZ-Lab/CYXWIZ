#pragma once

#include "../panel.h"
#include "../patterns/pattern.h"
#include "../patterns/pattern_library.h"
#include <functional>
#include <map>
#include <string>

// Forward declarations in global gui namespace
namespace gui {
class NodeEditor;
struct MLNode;
struct NodeLink;
}

namespace cyxwiz {

class PatternBrowserPanel : public Panel {
public:
    PatternBrowserPanel();
    ~PatternBrowserPanel() override = default;

    void Render() override;
    const char* GetIcon() const override { return "\xef\x86\xb3"; }  // cubes icon

    // Set callback for pattern insertion
    using InsertPatternCallback = std::function<void(
        const std::vector<::gui::MLNode>& nodes,
        const std::vector<::gui::NodeLink>& links
    )>;
    void SetInsertCallback(InsertPatternCallback callback) { insert_callback_ = std::move(callback); }

    // Set node editor reference for direct insertion
    void SetNodeEditor(::gui::NodeEditor* editor) { node_editor_ = editor; }

    // Open with specific pattern selected
    void OpenWithPattern(const std::string& pattern_id);

private:
    void RenderSearchBar();
    void RenderCategoryTabs();
    void RenderPatternList();
    void RenderPatternCard(const ::gui::patterns::Pattern& pattern);
    void RenderParameterInputs(const ::gui::patterns::Pattern& pattern);
    void InsertPattern(const std::string& pattern_id);

    // UI State
    char search_buffer_[256] = "";
    ::gui::patterns::PatternCategory selected_category_ = ::gui::patterns::PatternCategory::Basic;
    std::string selected_pattern_id_;
    std::string expanded_pattern_id_;  // Pattern card that's expanded to show parameters

    // Parameter values for the expanded pattern
    std::map<std::string, std::string> param_values_;

    // Callbacks
    InsertPatternCallback insert_callback_;
    ::gui::NodeEditor* node_editor_ = nullptr;

    // Category button colors
    static constexpr int NUM_CATEGORIES = 7;
};

} // namespace cyxwiz
