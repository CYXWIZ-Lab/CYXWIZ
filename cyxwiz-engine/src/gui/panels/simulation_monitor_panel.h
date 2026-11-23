#pragma once

#include "../panel.h"

namespace gui {
class NodeEditor;
}

namespace cyxwiz {

/**
 * Simulation Monitor (TOFIX195 S2.4): the open graph's run and every Signal
 * Scope's trace in one dockable window. It draws what the graph simulation
 * session holds and sends its buttons through NodeEditor's command path,
 * the one the canvas toolbar and the Simulation menu use; it owns no worker
 * and samples nothing, so closing it loses nothing.
 */
class SimulationMonitorPanel : public Panel {
public:
    SimulationMonitorPanel();

    void Render() override;
    const char* GetIcon() const override;

    void SetNodeEditor(::gui::NodeEditor* editor) { node_editor_ = editor; }
    // Show it and bring it forward in its next frame (a run started).
    void ShowAndFocus();

private:
    void RenderBody();
    void RenderScopes();

    ::gui::NodeEditor* node_editor_ = nullptr;
    bool focus_requested_ = false;
};

}  // namespace cyxwiz
