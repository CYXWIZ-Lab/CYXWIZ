# Troubleshooting extension nodes

> Checked against plugin API 1.2.0 and pycyxwiz 0.2.0.

Problems are grouped by where you see them. Messages in *italics* are the
Engine's own words.

## C++ plugins

### The plugin does not appear in the Plugin Manager

- Plugins load from the Plugin Manager (**Install...** with the plugin's
  folder, or **Refresh**), or at startup when they were loaded before.
- **Refresh** only searches two folders: `plugins` in the Engine program's
  own folder, and your user plugin folder (`%APPDATA%\cyxwiz\plugins\` on
  Windows, `~/.cyxwiz/plugins/` on Linux and macOS). In an installed Engine
  the program folder is replaced when the Engine updates, so put your
  plugins in the user folder, or load them with **Install...** from any
  folder.
- The plugin folder must hold a `plugin.json`, and the library named in its
  `platforms` entry must exist at that path (for example
  `bin/my_nodes.dll`).
- `api_version` must have the Engine's major number and a minor number no
  higher than the Engine's (1.2 today).
- **Install...** shows why a plugin did not load: *Failed to load plugin
  from &lt;folder&gt;: &lt;reason&gt;*. **Refresh** and startup write the
  reason to the **Console**. A plugin that loaded but failed to start shows
  its error on its card, with **Retry**.

### *&lt;name&gt; &lt;version&gt; is already loaded from &lt;folder&gt;*

A plugin id loads once. To load a new build of a plugin that is loaded:
**Unload** it, rebuild, then **Install...** again. While a model still uses
the plugin's nodes, Windows keeps its library open (the rebuild cannot
replace the file) and an unload keeps the old code in memory: close the
Engine instead, rebuild, and start it again.

### *Plugin installed, but N of its nodes were not added*

The plugin loaded, but the Engine refused some of the nodes it offered. The
list gives each reason:

- *it needs exactly one input and one output pin*: trainable nodes take one
  tensor and give one.
- *provider id '...' may only use a-z, 0-9, '.', '-' and '_'*, *node type
  name '...' may only use letters, digits and '_'*: fix the id in
  `plugin.json` or the node's type name.
- *node type '...' is already registered by '...'*: another plugin or a
  Python node bundle already uses this type id.

`cyxwiz-plugin-check` reports the same reasons as *FAIL node not added*.

### *The plugin was built with ...; this Engine with ...*

The plugin was built differently from this Engine; the message names each
difference (compiler, standard library, build type, pointer size, SDK or
ArrayFire version). Rebuild the plugin against the SDK of this Engine, with
the compiler the SDK names. See the [quickstart](plugin-quickstart.md) and
the [developer guide](plugin-developer-guide.md).

### *The plugin has no build fingerprint*

The plugin was built with an SDK older than plugin API 1.2. Rebuild it
against the current CyxWiz SDK; nothing in its code needs to change.

### `cyxwiz-plugin-check` grants a permission

*permission FileSystem: granted for this check only (the Engine asks the
user)*: the plugin's `plugin.json` asks for a permission the Engine
approves with a dialog. The check has no dialog, so it grants the
permission for its own run and goes on; the Engine still asks when the
plugin loads there.

## Python node bundles

### A bundle's nodes do not appear

The Engine looks for `.cyxnode` folders every 2 seconds in three places, in
this order:

1. `nodes/` in the open project (only while a project is open);
2. your user folder: `%APPDATA%\cyxwiz\nodes\` on Windows,
   `~/.cyxwiz/nodes/` on Linux and macOS;
3. `nodes/` inside each loaded C++ plugin's folder.

The Plugin Manager lists every bundle it found, and the ones it could not
read with the reason:

- *node.py changed since the bundle was written; bundle it again with
  pycyxwiz.nodes.bundle()*: the file no longer matches the hash in
  `node.json`. Bundle it again. If git changed the line endings, add
  `**/*.cyxnode/** -text` to your `.gitattributes`.
- *another bundle with the id ... is already loaded*: the first place in
  the list above wins.
- *the id ... is already used by a loaded plugin*: a C++ plugin owns the
  id.

### *the code of Python node bundle ... is not approved yet; approve it in the Plugin Manager to run it*

A bundle runs only after you approve its code. Use **Approve code...** on
its card (Plugin Manager, Node Browser, or Properties of one of its nodes).
An approval belongs to the bundle's id and the hash of its code: editing
and bundling again asks again.

### *Python is starting for the Python nodes (it needs an open project); try again in a moment*

The Engine starts Python on first use. Wait a moment and compile or train
again. Python starts only with a project open.

### *Python is not available in this build of CyxWiz*

This Engine was built without Python. Installed Engines include it; a
source build needs Python 3.12 or 3.13 when it is configured.

### Create node from script finds no node

*No node in ...: mark a class with @nodes.node, @nodes.activation or
@nodes.loss*. The script must define the classes itself (a class imported
from another file is left out) and decorate them. The whole script runs
when it is read, so keep training or test code under
`if __name__ == "__main__":`.

## In a graph

### A node shows "Not installed"

The graph was saved with a node whose extension is not loaded now. The node
keeps its pins, links and settings.

- A C++ plugin: open the **Plugin Manager** and load the plugin
  (**Refresh**, or **Install...**). If you unticked **Load when the Engine
  starts**, it stays unloaded after a restart until you load it.
- A Python node: the bundle is not where the Engine looks, could not be
  read, or its id clashes (see [A bundle's nodes do not appear](#a-bundles-nodes-do-not-appear)).

A not-installed node between the data and the loss stops training; remove
it or install its extension.

### *... is not the one this graph was saved with*

The extension is loaded, but the node differs from what the graph saved:
another version, other pins, or, for a Python node, *Its code changed
since the graph was saved, with the same version number.* Check the node's
settings and connections, then save the graph again. When you change a
node's code, pins or settings, raise its version.

### A setting is refused

Properties shows the node's own reason under the field, for example
*initial_a must be greater than 0*, and the compile shows it on the node.
Change the value; the message goes away when the node accepts it.

### The gradient check fails

*Failed: largest relative error 5.0e-01 in input (tolerance 2.0e-02)* means
the node's backward step does not match its forward step. Training can still
run, but the node may learn wrongly.

- **Where** (under **Details**) says which gradient is wrong: `input`, or
  `parameter <name>`.
- Compare the backward step with the derivative of the forward step by
  hand. A common slip is a missing factor (for Softsign, dividing by
  `1 + |x|` instead of `(1 + |x|)²`).
- For a loss, `predictions` is the place. The usual slip is the mean
  scaling: `backward` must divide as `forward` averages (by the element
  count for a mean over everything, by the row count for a mean over rows).
- If you are not the author, tell the node's author.

*Could not check* means the node could not be built or run for the check;
the reason follows. While the node refuses a setting (the orange message
under the field), the check is off: fix the setting first. Otherwise the
module failed on the check's inputs (batches of samples shaped [3] and
[4, 5]), and the node may need a specific input shape.

### Training stops with *Extension node '...' (...): ...*

The Engine's guard caught a broken rule of the [node contract](node-contract.md).
The end of the message says which one; the
[developer guide](plugin-developer-guide.md) lists the checks.

### Training stops with *Extension loss '...' (...): ...*

The same guard around a loss node. *forward must return one value*: return
a Tensor of one element (end with `.mean()` or `.sum()`). *backward returned
gradient shape ...*: return a gradient with the predictions' shape.

### *Remote nodes do not run extension nodes yet*

Graphs with extension nodes train on this computer only, for now.

## Exporting and importing

### *ONNX cannot describe its computation*

ONNX export writes the built-in layers it knows. An extension node's
computation is its own code, so a model with one would come out as a
different model; the Engine refuses instead and names the node. The export
dialog does not offer ONNX for such a model. Export a CyxWiz package
(`.cyxmodel`), which keeps the graph and the extension node's weights, or
Safetensors (weights only).

### Generated code says *code export cannot include an extension node's own code yet*

PyTorch, TensorFlow, Keras and pycyxwiz code export cannot contain an
extension node's code yet, so the generated script stops with this message
and the node's name instead of leaving the layer out. The same happens, with
"is not installed", when the extension is not loaded (for example with
`cyxwiz-engine --export-code`, which loads no extensions). Train the graph in
CyxWiz, or export a CyxWiz package.

### Importing a model says *Extension node '...' is not installed*

A CyxWiz package keeps the graph and the extension node's weights, but not
the extension itself. Install and approve the plugin or Python node bundle
the model was trained with, then import again. If the extension is installed
but cannot create its module (for example Python is still starting), the
import message ends with the extension's own reason.

The package's `manifest.json` lists the extensions it needs under
`extensions` (type id, version and code hash), and the export always includes
the graph for such a model. When the installed extension is another version
or other code than the model was trained with, the weights still load and the
import adds a warning: *Extension node '...' is not the one this graph was
saved with*. Check the model's results, or install the version it was trained
with.
