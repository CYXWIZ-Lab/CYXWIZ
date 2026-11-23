// The Studio toolbar's code actions (Generate into the Script Editor, Save as
// .py); the code itself comes from core/code_export (GUI-free).
#include "node_editor.h"
#include "../plugin/plugin_manager.h"
#include "panels/script_editor.h"
#include "../core/code_export.h"
#include "../core/file_dialogs.h"
#include "../core/project_manager.h"
#include "../core/studio_toolbar_presentation.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <fstream>
#include <string>

namespace gui {

namespace {

const char* FrameworkName(cyxwiz::CodeFramework framework) {
    switch (framework) {
        case cyxwiz::CodeFramework::PyTorch: return "PyTorch";
        case cyxwiz::CodeFramework::TensorFlow: return "TensorFlow";
        case cyxwiz::CodeFramework::Keras: return "Keras";
        case cyxwiz::CodeFramework::PyCyxWiz: return "PyCyxWiz";
    }
    return "PyTorch";
}

int LineCount(const std::string& code) {
    return static_cast<int>(std::count(code.begin(), code.end(), '\n')) + (code.empty() || code.back() == '\n' ? 0 : 1);
}

} // namespace

std::string NodeEditor::CodeProblem() {
    if (nodes_.empty()) return "The canvas is empty. Add a node first.";
    std::string error;
    if (!ValidateGraph(error)) return error.empty() ? std::string("The graph is not complete.") : error;
    return {};
}

std::string NodeEditor::GenerateCodeText(cyxwiz::CodeFramework framework) {
    cyxwiz::CodeExportOptions options;
    options.plugin_dir = [](const std::string& provider) {
        const auto* loaded = cyxwiz::plugin::PluginManager::Instance().GetLoadedPlugin(provider);
        return loaded ? loaded->plugin_dir.string() : std::string();
    };
    return cyxwiz::ExportGraphCode(nodes_, links_, framework, options).code;
}

namespace {

// What Generate and Save as .py report.
cyxwiz::studio_toolbar::CodeResult StartResult(cyxwiz::CodeFramework framework) {
    cyxwiz::studio_toolbar::CodeResult result;
    result.framework = FrameworkName(framework);
    result.at = cyxwiz::LocalClockText("%H:%M");
    return result;
}

}  // namespace

// On the interface thread: the code is built from the graph as it is now, so
// no worker reads the graph while it is being edited.
void NodeEditor::GenerateCodeIntoEditor() {
    namespace st = cyxwiz::studio_toolbar;
    st::CodeResult result = StartResult(selected_framework_);
    const std::string problem = CodeProblem();
    const std::string code = problem.empty() ? GenerateCodeText(selected_framework_) : std::string();
    st::CodeInputs in;
    in.framework = result.framework;
    in.graph = current_file_path_.empty() ? std::string() : GetGraphStem();
    const std::string file_name = st::CodeMenuFor(in).file_name;
    if (!problem.empty()) {
        result.error = problem;
    } else if (code.empty()) {
        result.error = "the graph has a cycle, or no node in it writes " + result.framework + " code";
    } else if (!script_editor_) {
        result.error = "the Script Editor is not available";
    } else {
        script_editor_->LoadGeneratedCode(code, file_name);
        script_editor_->SetVisible(true);
        result.ok = true;
        result.lines = LineCount(code);
        result.where = file_name;
    }
    code_result_ = st::CodeResultText(result);
    code_result_ok_ = result.ok;
    if (result.ok) spdlog::info("{}", code_result_);
    else spdlog::warn("{}", code_result_);
}

void NodeEditor::SaveCodeAsPython() {
    namespace st = cyxwiz::studio_toolbar;
    st::CodeResult result = StartResult(selected_framework_);
    result.to_file = true;
    const std::string problem = CodeProblem();
    const std::string code = problem.empty() ? GenerateCodeText(selected_framework_) : std::string();
    if (!problem.empty() || code.empty()) {
        result.error = problem.empty() ? "the graph has a cycle, or no node in it writes " + result.framework + " code"
                                       : problem;
    } else {
        st::CodeInputs in;
        in.framework = result.framework;
        in.graph = current_file_path_.empty() ? std::string() : GetGraphStem();
        const std::string file_name = st::CodeMenuFor(in).file_name;
        auto& project = cyxwiz::ProjectManager::Instance();
        const std::string start = project.HasActiveProject() ? project.GetScriptsPath() : std::string();
        const std::string title = "Save " + result.framework + " code";
        const auto path = cyxwiz::FileDialogs::SaveFile(title.c_str(), {{"Python", "py"}},
                                                        start.empty() ? nullptr : start.c_str(), file_name.c_str());
        if (!path) return;  // cancelled: nothing to report
        std::ofstream file(*path, std::ios::binary | std::ios::trunc);
        if (!file || !(file << code)) {
            result.error = "cannot write " + *path;
        } else {
            result.ok = true;
            result.lines = LineCount(code);
            result.where = *path;
        }
    }
    code_result_ = st::CodeResultText(result);
    code_result_ok_ = result.ok;
    if (result.ok) spdlog::info("{}", code_result_);
    else spdlog::warn("{}", code_result_);
}

} // namespace gui
