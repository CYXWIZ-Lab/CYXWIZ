# Troubleshooting extension nodes

> Checked 2026-10-16 against plugin API 1.1.0.

## The plugin does not appear in the Plugin Manager

- Press **Refresh**: plugins load from the Plugin Manager, or at startup when
  they were loaded before.
- The folder must be under `plugins/` in the folder the Engine starts in, or
  under your user plugin folder (`%APPDATA%\cyxwiz\plugins\`,
  `~/.cyxwiz/plugins/`), and hold a `plugin.json`.
- The library named in `platforms` must exist at that path (for example
  `bin/my_nodes.dll`).
- `api_version` must have the Engine's major number and a minor number no
  higher than the Engine's (1.1 today).
- A plugin that could not be loaded is reported in the **Console** ("Failed to load plugin from ..." with the reason). One that loaded but failed to start shows its error on its card, with **Retry**.

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

*Could not check* means the module could not run on the check's inputs
(batch shapes [2, 3] and [3, 4, 5]); the reason follows. Fix the settings, or
the node may need a specific input shape.

## Training stops with "Extension node '...' (...): ..."

The Engine's guard caught a broken rule. The end of the message says which
one; the [developer guide](plugin-developer-guide.md#6-what-the-engine-checks-for-you)
lists them.

## "Remote nodes do not run extension nodes yet"

Graphs with extension nodes train on this computer only, for now.
