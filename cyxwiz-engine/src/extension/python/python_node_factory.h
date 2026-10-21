#pragma once

// The Engine side of Python nodes (TOFIX125 P4 step 4.3, Part C3/C8): a
// factory per bundle node that the extension registry calls like a plugin's.
//
// - InferOutputShape needs no Python for shape_rule same_as_input: the
//   settings are checked against the manifest's rules here, with the
//   messages pycyxwiz.nodes gives. Other nodes ask Python.
// - CreateModule imports the bundle's node.py under a private module name
//   that includes its hash, calls pycyxwiz.nodes.create, and wraps the
//   Python module in a C++ Module that keeps the object alive and takes the
//   interpreter lock to release it.
// - A loss node (P7): CreateLoss calls pycyxwiz.nodes._engine_create_loss
//   and wraps the Python loss the same way.
//
// The interpreter is started by the Engine's scripting engine on the UI
// thread, never here: when it is not running, the call fails with a reason
// and PythonNodeRuntime records that Python is wanted.

#include "../../core/extension_node_contract.h"
#include "../../core/python_node_bundle.h"

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cyxwiz {

class PythonNodeRuntime {
public:
    static PythonNodeRuntime& Instance();

    // True when the interpreter runs. Otherwise false with the reason, and
    // Wanted() becomes true so the Engine can start Python.
    bool Ready(std::string& error);
    bool Wanted() const { return wanted_.load(); }
    void ClearWanted() { wanted_.store(false); }

    // False when this Engine was built without Python.
    static bool Available();

private:
    std::atomic<bool> wanted_{false};
};

// "cyxwiz_node_<id>_<first 16 hash digits>": the private import name.
std::string PythonBundleModuleName(const std::string& bundle_id, const std::string& content_hash);

// Checks settings against the node's declared settings: known names, type,
// range ("min-max", either side optional), choices. Empty when they are
// accepted; else the reason, worded as pycyxwiz.nodes words it.
std::string CheckPythonNodeSettings(const ExtensionNodeDescriptor& descriptor,
                                    const std::map<std::string, std::string>& settings);

std::shared_ptr<ITensorNodeFactory> MakePythonTensorNodeFactory(const PythonBundle& bundle,
                                                               const PythonBundleNode& node);
std::shared_ptr<ITensorLossFactory> MakePythonLossFactory(const PythonBundle& bundle,
                                                         const PythonBundleNode& node);

}  // namespace cyxwiz
