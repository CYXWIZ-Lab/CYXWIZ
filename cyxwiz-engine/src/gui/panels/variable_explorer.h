#pragma once

// Variable Explorer (TOFIX133 P5, approved board 9): the shared Variables
// view in a dock window beside the Console, with a scope picker (the Python
// session or an open notebook).

#include "../panel.h"
#include "../variables_view.h"

#include <memory>

namespace scripting {
class ScriptingEngine;
}

namespace cyxwiz {

class VariableExplorerPanel : public Panel {
public:
    VariableExplorerPanel();
    ~VariableExplorerPanel() override = default;

    void Render() override;
    const char* GetIcon() const override;

    void SetScriptingEngine(std::shared_ptr<scripting::ScriptingEngine> engine);
    VariablesView& View() { return view_; }

private:
    std::shared_ptr<scripting::ScriptingEngine> scripting_engine_;
    VariablesView view_;
};

}  // namespace cyxwiz
