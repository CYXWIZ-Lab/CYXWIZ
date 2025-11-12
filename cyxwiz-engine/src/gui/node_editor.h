#pragma once

namespace gui {

class NodeEditor {
public:
    NodeEditor();
    ~NodeEditor();

    void Render();

private:
    void ShowToolbar();
    void RenderNodes();
    void HandleInteractions();

    bool show_window_;
};

} // namespace gui
