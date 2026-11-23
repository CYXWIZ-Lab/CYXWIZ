#pragma once

// Make a Python node (TOFIX231, approved mockup 2026-11-23): a form that
// writes a node's Python source, checks its gradients and creates the bundle
// in the open project's nodes folder, approved as the user's, so the node is
// in the Nodes panel at once. The kind (Activation, Layer, Loss) fixes the
// pins; the user writes the method bodies. The form and the source it writes
// are core/python_node_maker; the Python calls are gui/python_node_calls,
// shared with Script Editor > Create node from script.

#include "../panel.h"
#include "../../core/python_node_maker.h"

#include <filesystem>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace scripting {
class ScriptingEngine;
}

namespace cyxwiz {

class PythonNodeMakerPanel : public Panel {
public:
    PythonNodeMakerPanel();

    void Render() override;

    // Shows the window in front.
    void Open();

    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine) { engine_ = std::move(engine); }
    // Opens a file in the Script Editor (Open in Script Editor, a source not
    // made here).
    void SetOpenInScriptEditorCallback(std::function<void(const std::string&)> callback) {
        open_in_script_editor_ = std::move(callback);
    }

private:
    enum class Result { None, Busy, Passed, Failed, CheckError, Created, Error };

    void RenderTopRow();
    void RenderKind();
    void RenderDetails();
    void RenderSettings();
    void RenderSetting(node_maker::Setting& setting, size_t index, bool& remove);
    void RenderPins();
    void RenderCode();
    void RenderFooter(float height);
    void RenderNewConfirm();

    // Where the source goes: <project>/nodes_src/<stem>.py, or a temporary
    // folder for a check without a project. Empty without a class name.
    std::filesystem::path SourcePath() const;
    std::string ProjectName() const;
    bool HasProject() const;
    // The source written for Check, Create and Open in Script Editor.
    bool WriteSource(std::filesystem::path& written, std::string& error);
    // Another node's source already at the path this form would write.
    bool SourceTakenByAnother() const;

    void StartCheck();
    void StartCreate();
    void Collect();
    void Reset(const node_maker::NodeForm& form, const std::filesystem::path& file);
    // The form differs from what its file holds (or, new, from the starter).
    bool Dirty() const;
    bool Edited() const;

    node_maker::NodeForm form_;
    bool class_follows_name_ = false;  // the class name is made from the display name
    std::filesystem::path file_;       // the source this form was opened from or last written to
    std::string saved_source_;         // what file_ holds, for "not created yet"
    bool created_ = false;

    std::shared_ptr<scripting::ScriptingEngine> engine_;
    std::function<void(const std::string&)> open_in_script_editor_;

    enum class Call { None, Check, Create } call_ = Call::None;
    std::future<std::pair<bool, std::string>> pending_;
    Result result_ = Result::None;
    std::string result_title_;
    std::string result_detail_;
    std::string created_group_;

    bool confirm_new_ = false;
    bool bring_to_front_ = false;
};

}  // namespace cyxwiz
