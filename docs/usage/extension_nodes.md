# Your own nodes: Python node bundles and C++ plugins

An **extension node** is a node CyxWiz does not ship: written in Python with
pycyxwiz (a *Python node bundle*) or in C++ (a *plugin*). It is added from the
node editor, connected like a built-in layer, saved in `.cyxgraph` files and
trained. This page is how to use them in the Engine; how to write one is in
[docs/extending](../extending/README.md).

> Checked against plugin API 1.4.0 and pycyxwiz 0.2.0.

## Try the examples first

Every installation carries ten built-in layers and four losses rewritten
as Python nodes: Dense, Layer Norm, ReLU, Leaky ReLU, ELU, Sigmoid, Tanh,
Softplus, GELU and SiLU; Cross Entropy, MSE, L1 and Huber. They are in the plugin SDK of the CPU base,
`sdk/examples/python-nodes/builtin-layers/` (in the sources
`examples/python_nodes/builtin_layers/`).

1. Copy `nodes/builtin-layers.cyxnode` from that folder into your project's
   `nodes/` folder (create `nodes/` in the project folder if it is not
   there; a new project does not have it).
2. **Apps > Plugin Manager**, section *Python node bundles*: on the
   *builtin-layers* card press **Approve code...**, read what it adds (and
   **View code** if you like), and approve.
3. The fourteen nodes are in the Nodes panel and in search under *Python
   examples*, as "Dense (Python)", "ReLU (Python)", "MSE (Python)" and so on.
4. Open `blobs_python_layers.cyxgraph` from the example folder (with
   `blobs.csv` beside it), open the Data Input and apply the file, and press
   **Train**. The graph is Data Input -> Split -> Loader -> Dense 16 -> ReLU
   -> Layer Norm -> Dense 4 -> Cross Entropy -> Adam -> Output, every layer
   and the loss in Python. It is set to 20 epochs with early stopping: it
   reaches about 97% training and 96% validation accuracy on the four
   classes by epoch 6, usually stops after about 11 epochs when validation
   loss stops improving, and keeps the best epoch.

Each example computes the same numbers as the built-in node of the same name,
so you can swap one for the other in any graph.

A loss node is wired like a built-in loss: the last layer to
**Predictions**, the Loader's labels to **Targets**, **Loss** to the
optimizer. Cross Entropy is for classes; MSE, L1 and Huber predict a number
(give the last layer one unit).

## Python node bundles

A bundle is a folder `<name>.cyxnode` holding `node.json` and `node.py`. The
Engine finds bundles, without starting Python, in:

| Folder | Scope |
|---|---|
| `<project>/nodes/` | This project |
| `%APPDATA%\cyxwiz\nodes\` (`~/.cyxwiz/nodes/` elsewhere) | Every project of this user |
| `nodes/` inside a loaded plugin's folder | Delivered with that plugin |

The Engine looks every 2 seconds: new, edited and removed bundles show
within 2 seconds, without a restart.

- **Approval.** A bundle's code runs only after you approve that exact code.
  Until then its nodes can be placed and saved, and training says "the code
  of Python node bundle ... is not approved yet". Approve from the Plugin
  Manager card or from the node's Properties (Gradient check card). An
  approval belongs to the bundle's id and code hash, not to its folder:
  changed code needs a new approval, and a copy of approved code elsewhere
  is approved too. **Actions > Revoke approval** takes one back.
- **Make one in a form.** **Nodes > Make a Python node...** (or **More >
  Make a Python node...** on the Studio toolbar) writes the Python for you;
  see [Make a Python node](#make-a-python-node) below.
- **Make one from a script.** In the Script Editor, open a saved `.py`
  file that defines nodes with `@nodes.activation`, `@nodes.node` or
  `@nodes.loss`, then
  **Run > Create node from script...**: check gradients, name the bundle,
  **Create nodes**. It is written to `<project>/nodes/` and approved for you,
  each time you create it.
- **Change a node.** A node is ordinary Python: open its code with **View
  code** (Plugin Manager card). To change it, edit the `.py` file you wrote
  it in (for the examples, `builtin_layers.py`) and make the bundle again
  (**Create node from script**, or `nodes.bundle(...)`): `node.json` and
  `node.py` are written together and the new code needs approval once.
  Raise the node's version when you change its settings, pins or code.
- **A bundle that cannot be read** is listed in the Plugin Manager with the
  reason, and graphs show its nodes as **Not installed**. Editing `node.py`
  inside the bundle by hand does this: "node.py changed since the bundle was
  written", because the bundle's record no longer matches its code.
- **One id, one bundle.** Two bundles with the same id do not both load; the
  second says "another bundle with the id ... is already loaded". The
  project's `nodes` folder wins over your user folder, which wins over a
  plugin's. An id that a loaded C++ plugin uses is refused: "the id ... is
  already used by a loaded plugin".
- **Python** starts by itself when an open graph uses an approved Python
  node, or when the first one is built; it needs an open project. Until it
  runs, compiling says "Python is starting for the Python nodes (it needs an
  open project); try again in a moment". A CyxWiz built without Python says
  "Python is not available in this build of CyxWiz". Python nodes run in the
  open project's Python environment: after opening another project, Python
  moves to its environment by itself, or, when it cannot (see
  [Notebooks: Changing project](notebooks.md)), they say "Python runs another
  project's environment; restart the Engine (File > Restart) ..." until you
  restart.
- **Errors.** An error in a node's code stops training: the training status
  says *Training failed:* with the node's name and the Python traceback, and
  the Console (Logs) has the full text. How to find the cause:
  [python-nodes.md](../extending/python-nodes.md) section 10.

## Make a Python node

**Nodes > Make a Python node...** opens a form that writes a node's Python
source, checks it and creates its bundle. You write only the arithmetic.

1. **Kind.** It decides the pins, which you do not edit:
   - **Activation**: Input to Output, element by element. You write `f(x)`
     and its derivative `df(x)`; an activation has no settings.
   - **Layer**: Input to Output. You write `forward(self, x)` and
     `backward(self, grad_output)`. **Shown as** Layer or Activation picks
     its place in the Nodes panel; **Output shape** *I compute it* adds
     `output_shape(self, input_shape)` for a layer that changes the shape.
   - **Loss**: Predictions and Targets to Loss. You write `forward` and
     `backward` on `(predictions, targets)`; **Targets** is *Classes
     (one-hot)* or *Values*.
2. **Details.** The name in the Nodes panel, the class name (made from the
   name until you type your own), the version, the group it is listed in
   under **Plugins** in the Nodes panel, and a description.
3. **Settings** (Layer and Loss): each has a name, a type (number, whole
   number, on / off, text or choice), a default, and lowest / highest for
   numbers. They show in the node's Properties; the code reads them as
   `self.<name>`.
4. **Code.** The decorator and the settings are written from the form and
   shown above the method boxes; type the bodies of the methods. Each kind
   starts from a node that works (Soft Clip, Swish with a `beta` setting,
   Log-cosh), so **Check** passes before you change anything.
5. **Check** runs the gradient check with the default settings and says
   whether `backward` matches `forward` (largest relative error, limit
   0.02), or the Python error.
6. **Create node** writes the source to `<project>/nodes_src/<file>.py` and
   the bundle to `<project>/nodes/<file>.cyxnode`, approved for you; the
   node is in the Nodes panel at once, under **Plugins**, in its group. A node whose check
   fails is still created, with the result kept on it.

- **Open** lists the project's `nodes_src` sources. A source the form did not
  write, or one changed outside it (an import, a helper, a comment), opens in
  the Script Editor: make that node with **Run > Create node from
  script...**. **Open in Script Editor** writes the source and opens the
  whole file there. A source changed on disk after you opened it here is not
  overwritten: Check and Create say so until you open it again.
- **New**, **Open** and a different **Kind** ask first when the form holds
  changes not created yet; a Kind starts its code over.
- **Check** and **Create node** need an open project (Python runs in the
  project's environment); without one you can still write the node.
  Create is blocked, with the reason, while a field is wrong,
  when another node's source already has that file name, or when the
  project's `nodes` folder has a bundle of that name made another way.
- A Check or Create result stays only while the form still writes the
  source it ran on.
- Each node is its own bundle, with the id `project.<project>.<file>`: the
  same on every Create, so graphs that use the node keep it. Change the
  class name and you make a new node.

## C++ plugins

1. **Apps > Plugin Manager > Install...**, choose the folder that holds the
   plugin's `plugin.json`, and **Install**. Or copy the plugin folder into
   `%APPDATA%\cyxwiz\plugins\` (`~/.cyxwiz/plugins/` elsewhere) and press
   **Refresh**.
2. The card lists what the plugin adds; **Load when the Engine starts** is
   ticked. **Unload** removes it from the list.
3. **Apps > Loaded plugins > <plugin>** opens the Nodes panel on its nodes.

To build your own plugin against the SDK in the CPU base (`sdk/`, Apache
License 2.0), start with `sdk/template/README.md` and
[plugin-quickstart.md](../extending/plugin-quickstart.md); check a built
plugin with `cyxwiz-plugin-check`.

## Using an extension node

- Connect it between layers as you would ReLU or Dense.
- **Properties** shows where it comes from (*Python node* or the plugin, with
  id and version), its settings with their descriptions (a refused value is
  explained under its field), and a **Gradient check** card: **Check
  gradient** compares its backward step with a numerical gradient (for a
  loss, on one-hot or value targets as the loss declares).
- A loss node is the graph's training objective: training reports
  accuracy for a classification loss and regression metrics for one that
  predicts a number. A loss that returns more than one value, or a gradient
  of another shape than the predictions, stops training with its name.
- **Info** lists its bundle or plugin, approval and code hash.
- A graph whose node is missing on this computer shows it as **Not
  installed**, keeps its settings and pins, and does not train until the
  bundle or plugin is installed.

## Saving and exporting

| Action | Result |
|---|---|
| Save the graph (`.cyxgraph`) | Works; nodes record their bundle or plugin and code hash |
| Export a CyxWiz package (`.cyxmodel`) | Works; the node's weights are saved and the manifest lists the extension node types |
| Import that package | The node is rebuilt from the installed bundle or plugin. Not installed: refused. Different code than at training: loads with a warning |
| Export to ONNX or to code (PyTorch, TensorFlow, Keras, pycyxwiz) | Refused with the node's name |
| Deploy on, or train on, a Server Node | Refused: the Server Node does not run extension nodes yet |

## Not yet

- Nodes with several inputs or outputs (a loss has its fixed two).
- Losses from C++ plugins, losses on class indices or 0/1 targets, losses
  with weights.
- Autograd: a node's backward step is written by hand.

Problems ("not installed", refused settings, failed gradient checks):
[troubleshooting.md](../extending/troubleshooting.md).
