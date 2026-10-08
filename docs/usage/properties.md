# Properties

The Properties panel (right column; sidebar **Properties**, or View > Panels)
shows the node selected on the CyxWiz Studio canvas: what it is, its
settings and where each value comes from, what the compiler built for it,
and what the graph file stores. With nothing selected it shows the graph in
one line: name, node and link counts, and the compile state with the first
error's node.

## The header

Icon and name. Click the name to rename the node; Enter or clicking away
saves it. Under it, one line: type · category · #id · implementation status
(quiet when *Implemented*; *Planned*, *Deprecated* and *External* are
coloured) and the type's badge when it has one (for example *Coming Soon*).

Nodes that are set up in a window of their own have the primary button:
**Open Dialog...** (Data Input, Data Output, Data Convert, Data Loader, Data
Split, Embedding, Tokenizer, CSV and Excel readers, Filter rows), **Open Plot
window** (Plot) or **Open Dashboard** (Dashboard).

**Actions** holds everything else:

- **Apply preset**: the built-in presets of the type (Dense: Small / Medium /
  Large; Conv2D: VGG-, ResNet-, MobileNet-style; Adam: Default / Fast
  learning / Fine-tuning) and the ones you saved. Hover a preset to see its
  values.
- **Save current as preset...**: saves the node's settings under a name, for
  this node type, in `%APPDATA%\CyxWiz\node_presets.json` (Linux and macOS:
  `~/.cyxwiz/node_presets.json`). A saved preset with a built-in's name
  replaces it. **Delete saved preset** removes one.
- **Reset all settings to defaults**, **Copy settings as JSON** (name, type,
  parameters), **Remove unused keys** (stored keys the rules no longer map to
  a setting), **Show node in canvas**.

## Settings

One row per setting: the label, the editor, and a status mark at the right
when the Engine's rules know the setting:

- ✓ the value is used as shown;
- *dialog* the value is set in the node's dialog;
- *default* the node has no value of its own, the default applies;
- *alias* the value came from an older key that the rules still read;
- *missing*, *conflict*, *stale*, *unsupported* need attention.

Hover the mark for the source key, the owner (Compiler, Runtime, Loader,
Materializer, ...) and the rule's message. **▸ Details** under the rows shows
them under every row. Settings are grouped as their node type declares
(Training, Generation previews, Advanced settings, ...); a group's words are
dropped from its labels ("Every epochs" under Generation previews). *Reset*
appears on a row whose value differs from its default. A value is written to
the node only when you change it: looking at a node changes nothing.

Settings the rules know but no editor shows (for example a Data Input's
label column, owned by the loader) are listed read-only below the rows. A
Data Input also gets a **Data** row: the file, whether it is loaded, and the
rows × columns (or images and classes) once it is.

Plot and Dashboard nodes show the settings saved by their window, in words;
they are changed in the window.

## As compiled

The card shows what the graph compiler built for this node from the latest
background compile of the canvas: the status (*Compiled*, *Compile failed*,
*Compiling...*, *Not compiled yet*), the node's role, its own compiler
issues, and for a model layer the input and output shapes per sample and per
batch, the output memory per batch, the learnable parameters and their
memory. **▸ Details** adds the layer's parameters and the node type's support
(local training, remote training, export). The shapes come from the
compiler's dimension rules, not from the panel.

A Concatenate fed by an Embedding also gets the *Word + POS fusion* card with
the compiler's verdict and the fix when it will not compile.

## Advanced

Position and connections, then **Raw parameters**: every key the graph file
stores for this node, its value, which setting it maps to, the cleanup
advice, and **Remove** where the rules allow it.

## K-Means

A K-Means node keeps its executor section (Configuration, Results) below.
