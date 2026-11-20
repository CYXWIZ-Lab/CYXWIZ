#pragma once

#include "../panel.h"
#include "../patterns/pattern.h"
#include "../../core/pattern_browser_presentation.h"
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

// The Pattern Browser (TOFIX178 4e, approved mockup): category chips, search
// across every category, a card per pattern with its kind, node count and
// settings, Insert with its result or failure said under it, the question
// before a full graph lands on a canvas that has nodes, and My patterns
// (show in folder, delete). The words come from core/pattern_browser_presentation.
class PatternBrowserPanel : public Panel {
public:
    PatternBrowserPanel();
    ~PatternBrowserPanel() override = default;

    void Render() override;
    const char* GetIcon() const override { return ICON_FA_CUBES; }

    // The canvas patterns go into.
    void SetNodeEditor(::gui::NodeEditor* editor) { node_editor_ = editor; }

    // Shows the browser with this pattern's card open (a Nodes panel example,
    // a pattern just saved).
    void OpenWithPattern(const std::string& pattern_id);

private:
    enum class Stage { None, Choose, ConfirmDelete, Done, Failed };
    enum class Where { Below, Beside, Replace };

    void RebuildEntries();
    void RenderChips(const pattern_browser::ListView& view);
    void RenderList(const pattern_browser::ListView& view);
    void RenderCard(const ::gui::patterns::Pattern& pattern, const pattern_browser::Entry& entry);
    void RenderSettings(const ::gui::patterns::Pattern& pattern);
    void RenderInsertLine(const ::gui::patterns::Pattern& pattern, const pattern_browser::Entry& entry);
    void RenderStage(const ::gui::patterns::Pattern& pattern, const pattern_browser::Entry& entry);
    void RenderFolderLine();

    void Open(const ::gui::patterns::Pattern& pattern);
    void Insert(const ::gui::patterns::Pattern& pattern, const pattern_browser::Entry& entry,
                std::optional<Where> where);
    void SetStage(const std::string& id, Stage stage, std::string text = {});
    std::string GraphName() const;  // the canvas graph's name, empty when it has no file

    ::gui::NodeEditor* node_editor_ = nullptr;

    // The library's patterns as the model reads them, rebuilt when it changes.
    std::vector<pattern_browser::Entry> entries_;
    size_t entries_revision_ = static_cast<size_t>(-1);

    char search_[256] = "";
    std::string chip_ = "Basic";
    std::string open_id_;                        // the card that is open
    std::map<std::string, std::string> params_;  // its settings
    bool scroll_to_open_ = false;

    // What the open card says under its button.
    std::string stage_id_;
    Stage stage_ = Stage::None;
    std::string stage_text_;
};

}  // namespace cyxwiz
