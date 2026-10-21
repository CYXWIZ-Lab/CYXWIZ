#pragma once

// Script Editor > Run > Create node from script... (TOFIX125 P4 step 4.5,
// Part C10, mockup row 6 "CreateNode"): reads the saved script with
// pycyxwiz.nodes, lists the nodes it defines, checks their gradients,
// writes the bundle into <project>/nodes and approves it ("created": the
// user wrote it). Python calls run on a worker thread under the interpreter
// lock; the dialog polls them. Words come from core/python_bundle_presentation.

#include "../../core/python_bundle_presentation.h"

#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <utility>

namespace scripting {
class ScriptingEngine;
}

namespace cyxwiz {

class CreateNodeDialog {
public:
    // Opens the dialog for a saved Python script.
    void Open(const std::filesystem::path& script, std::shared_ptr<scripting::ScriptingEngine> engine);
    // Every frame.
    void Render();
    bool IsOpen() const { return open_ || popup_requested_; }

private:
    void StartInspect();
    void StartCheck();
    void StartWrite();
    void Collect();

    std::filesystem::path script_;
    std::shared_ptr<scripting::ScriptingEngine> engine_;
    CreateNodeInputs inputs_;
    char name_buf_[128] = {};
    bool open_ = false;
    bool popup_requested_ = false;
    // A Python call in flight: (ok, JSON or the bundle folder / the error).
    std::future<std::pair<bool, std::string>> pending_;
    enum class Call { None, Inspect, Check, Write } call_ = Call::None;
};

}  // namespace cyxwiz
