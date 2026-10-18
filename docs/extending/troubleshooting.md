# Troubleshooting extension nodes

> Checked 2026-10-17 against plugin API 1.2.0.

## The plugin does not appear in the Plugin Manager

- Plugins load from the Plugin Manager (**Install...** with the plugin's
  folder, or **Refresh**), or at startup when they were loaded before.
- **Refresh** only searches `plugins` beside the Engine program (in a source
  build `<build folder>/bin/Release/plugins/`) and your user plugin folder
  (`%APPDATA%\cyxwiz\plugins\`, `~/.cyxwiz/plugins/`). Starting the Engine
  from another folder does not change this. A plugin elsewhere loads with
  **Install...**.
- The plugin folder must hold a `plugin.json`.
- The library named in `platforms` must exist at that path (for example
  `bin/my_nodes.dll`).
- `api_version` must have the Engine's major number and a minor number no
  higher than the Engine's (1.2 today).
- A plugin that could not be loaded is reported in the **Console** ("Failed to load plugin from ..." with the reason). One that loaded but failed to start shows its error on its card, with **Retry**.

## "The plugin was built with ...; this Engine with ..."

The plugin was built differently from this Engine (another build type,
compiler, SDK or ArrayFire version). The message names each difference.
Rebuild the plugin with the same settings, most often **Release** instead of
Debug. See the developer guide, *Build fingerprint*.

## "The plugin has no build fingerprint"

The plugin was built with an SDK older than plugin API 1.2. Rebuild it
against the current CyxWiz SDK; nothing in its code needs to change.

## A node shows "Not installed"

The graph was saved with a node whose plugin is not loaded now. Open the
**Plugin Manager**, load the plugin (**Refresh**, or **Install...**), and the
node comes back with its saved settings. If you unticked **Load when the
Engine starts**, the plugin stays unloaded after a restart until you load it.

A not-installed node between the data and the loss stops training; remove it
or load its plugin.

## "is not the one this graph was saved with"

The plugin is loaded, but the node's version or pins differ from what the
graph saved. Check the node's settings and connections, then save the graph
again.

## A setting is refused

Properties shows the plugin's own reason under the field, for example
*initial_a must be greater than 0*, and the compile shows it on the node.
Change the value; the message goes away when the plugin accepts it.

## The gradient check fails

*Failed: largest relative error 5.0e-01 in input (tolerance 2.0e-02)* means
the node's backward step does not match its forward step. Training can still
run, but the node may learn wrongly.

- **Where** (under **Details**) says which gradient is wrong: `input`, or
  `parameter <name>`.
- Compare `Backward` with the derivative of `Forward` by hand. A common slip
  is a missing factor (for Softsign, dividing by `1 + |x|` instead of
  `(1 + |x|)²`).
- If you are not the author, tell the plugin's author.

*Could not check* means the node could not be built or run for the check;
the reason follows. While the plugin refuses a setting (the orange message
under the field), the check is off: fix the setting first. Otherwise the
module failed on the check's inputs (batch shapes [2, 3] and [3, 4, 5]), and
the node may need a specific input shape.

## Training stops with "Extension node '...' (...): ..."

The Engine's guard caught a broken rule. The end of the message says which
one; the [developer guide](plugin-developer-guide.md#6-what-the-engine-checks-for-you)
lists them.

## "Remote nodes do not run extension nodes yet"

Graphs with extension nodes train on this computer only, for now.

## "ONNX cannot describe its computation"

ONNX export writes the built-in layers it knows. An extension node's
computation is its own code, so a model with one would come out as a
different model; the Engine refuses instead and names the node. The export
dialog does not offer ONNX for such a model. Export a CyxWiz package
(`.cyxmodel`), which keeps the graph and the extension node's weights, or
Safetensors (weights only).

## Generated code says "code export cannot include an extension node's own code yet"

PyTorch, TensorFlow, Keras and pycyxwiz code export cannot contain an
extension node's code yet, so the generated script stops with this message
and the node's name instead of leaving the layer out. The same happens, with
"is not installed", when the extension is not loaded (for example with
`cyxwiz-engine --export-code`, which loads no extensions). Train the graph in
CyxWiz, or export a CyxWiz package.

## Importing a model says "Extension node '...' is not installed"

A CyxWiz package keeps the graph and the extension node's weights, but not
the extension itself. Install and approve the plugin or Python node bundle
the model was trained with, then import again. If the extension is installed
but cannot create its module (for example Python is not running yet), the
import message ends with the extension's own reason.
