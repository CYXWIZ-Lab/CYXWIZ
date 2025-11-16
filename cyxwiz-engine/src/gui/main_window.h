#pragma once

#include <memory>

namespace gui {

class NodeEditor;
class Console;
class Viewport;
class Properties;

} // namespace gui

namespace cyxwiz {
class ToolbarPanel;
class AssetBrowserPanel;
class TrainingDashboardPanel;
class PlotTestControlPanel;
class CommandWindowPanel;
class ScriptEditorPanel;
} // namespace cyxwiz

namespace scripting {
class ScriptingEngine;
} // namespace scripting

namespace gui {

class MainWindow {
public:
    MainWindow();
    ~MainWindow();

    void Render();
    void ResetDockLayout();
    Console* GetConsole() { return console_.get(); }
    cyxwiz::PlotTestControlPanel* GetPlotTestControl() { return plot_test_control_.get(); }

private:
    void RenderDockSpace();
    void BuildInitialDockLayout();
    void ShowAboutDialog();

    // Original panels
    std::unique_ptr<NodeEditor> node_editor_;
    std::unique_ptr<Console> console_;
    std::unique_ptr<Viewport> viewport_;
    std::unique_ptr<Properties> properties_;

    // New panel system
    std::unique_ptr<cyxwiz::ToolbarPanel> toolbar_;
    std::unique_ptr<cyxwiz::AssetBrowserPanel> asset_browser_;
    std::unique_ptr<cyxwiz::TrainingDashboardPanel> training_dashboard_;
    std::unique_ptr<cyxwiz::PlotTestControlPanel> plot_test_control_;
    std::unique_ptr<cyxwiz::CommandWindowPanel> command_window_;
    std::unique_ptr<cyxwiz::ScriptEditorPanel> script_editor_;

    // Scripting engine (shared between panels)
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;

    bool show_about_dialog_;
    bool show_demo_window_;
    bool first_time_layout_;
};

} // namespace gui
