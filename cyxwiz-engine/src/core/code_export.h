#pragma once

// The Python code a graph exports to: PyTorch, TensorFlow, Keras or PyCyxWiz
// (TOFIX228 1.5a, moved out of the Node Editor). GUI-free: it reads the
// graph's nodes and links only, so the Studio toolbar (Generate, Save as .py),
// the --export-code command and the training oracle all get the same text.
// An RL graph exports Stable-Baselines3 PPO for PyTorch and PyCyxWiz.

#include "graph_model.h"

#include <functional>
#include <string>
#include <vector>

namespace cyxwiz {

enum class CodeFramework {
    PyTorch,
    TensorFlow,
    Keras,
    PyCyxWiz
};

struct CodeExportOptions {
    // "# Generated at: <local time>" in the header; false writes
    // "(not stamped)", so the same graph always gives the same text.
    bool timestamp = true;
    // RL graphs: the folder of the loaded plugin that provides an environment
    // ("mujoco" -> its plugin folder); empty or unset when none is loaded.
    std::function<std::string(const std::string& provider)> plugin_dir;
};

// A PyTorch module of the exported GeneratedModel and the node it comes
// from: `self.layer3 = nn.Linear(...)` is {node id, "layer3"}.
struct ExportedModule {
    int node_id = 0;
    std::string name;
};

struct GraphCodeExport {
    // Empty when the graph has no nodes or a cycle. A graph the export cannot
    // write gives a script that raises ValueError with the reason.
    std::string code;
    std::vector<ExportedModule> modules;  // PyTorch only, in model order
};

GraphCodeExport ExportGraphCode(const std::vector<::gui::MLNode>& nodes, const std::vector<::gui::NodeLink>& links,
                                CodeFramework framework, const CodeExportOptions& options = {});

// The local time now in strftime `format` (the export header, the toolbar's
// result line).
std::string LocalClockText(const char* format);

}  // namespace cyxwiz
