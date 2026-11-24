# Patterns

A pattern is a ready-made graph or part of a graph that you insert into the
canvas. Open the **Pattern Browser** with **View > Panels > Patterns panel**
(Ctrl+Shift+P), or double-click a pattern under **Examples** at the end of
the Nodes panel.

## Finding and inserting a pattern

- **Chips** under the search field pick a category: Basic, CNN, RNN,
  Transformer, Generative, Reinforcement learning, Blocks and My patterns,
  each with how many patterns it holds. The chips wrap onto a second line in
  a narrow panel.
- **Search** looks in every category at once: names, descriptions, tags.
  Esc clears it.
- **Each card** shows the pattern's name, **Full graph** or **Block**, its
  number of nodes and two lines of its description. Click a card to open it:
  the whole description, its **Settings** (type a number, or use - and +),
  and **Insert full graph** or **Insert block**.
- **After Insert**, a green line under the button says what was added and
  what to do next, until the canvas changes (Undo, a delete). When nothing could be added, an orange line says why (a
  link in the pattern to a pin its node does not have, a node type the
  Engine does not know, a run using the graph) and the canvas is unchanged.

A **block** goes below the graph on the canvas. A **full graph** brings its
own Data Input, loss and optimizer, so on a canvas that already has nodes the
browser asks first:

- **Replace the canvas**: the graph on the canvas is removed and the pattern
  takes its place. Undo on the canvas brings the graph back: click the
  canvas, then Ctrl+Z or **Edit > Undo** (double-click its Data Input and
  click Apply again to reload its data). The graph's saved file is not
  touched until you save.
- **Add beside it**: the pattern goes to the right of the graph. The canvas
  then holds two graphs; delete the one you do not want before you train.
- **Cancel**: nothing changes.

## Full graphs and blocks

Patterns come in two kinds.

- **A full graph** is a whole model, from data to training:
  Data Input -> (Text Tokenizer) -> Data Split -> DataLoader -> (image steps)
  -> layers -> loss -> optimizer -> Output. After inserting it, double-click
  **Data Input**, point it at your data, click **Apply**, then **Train**.
  Each pattern's description says what data it expects.
- **A block** is a few layers to put inside your own graph. It starts with a
  **Block input** node (an Identity layer): link the layer before the block
  to it, and link the block's last layer to whatever comes next. The block's
  description says what input shape it needs.

## The shipped patterns

| Pattern | Kind | Data it expects |
| --- | --- | --- |
| Basic MLP, MLP with Dropout | Full graph | A CSV table: number columns and a `label` column of classes |
| Linear Regression | Full graph | A CSV table: number columns and a number `label` column (MSE loss, SGD) |
| Skip Connections and Merges | Full graph | A CSV table with a `label` column; shows a residual Add, a skip through Concatenate, self-attention with its residual, and an Add of three branches |
| Basic CNN, CNN Image Classifier, LeNet-5 | Full graph | An image folder with one subfolder per class; Resize 32 x 32 and Normalize (mean 0.5, std 0.5) |
| AlexNet, VGG-16, MobileNet v1 | Full graph | An image folder with one subfolder per class; Resize 256, Center Crop 224 and Normalize with the ImageNet statistics |
| ResNet-18, MobileNetV2, MobileNetV3 Small | Full graph | The same image folder and image steps; the whole network as torchvision builds it, ready for its published weights (below) |
| Basic RNN, Basic GRU, Basic LSTM, Bidirectional LSTM, Transformer Encoder | Full graph | A CSV with a `text` column and a `label` column; the Text Tokenizer keeps `max_length` tokens |
| Transformer Decoder (next token) | Full graph | A CSV of token ids: `sequence_length` token columns and a `next` column (the token that follows); predicts the next token |
| Causal LM Generation Starter | Full graph | A CSV of token ids, one sequence per row; trained on the next token at every position |
| PPO (MLP 64-64) | Full graph | No data: an Environment node (CartPole by default). See [Training a policy (PPO)](rl_training.md) |
| Conv-BN-ReLU Block, Residual Block | Block | A feature map [height, width, channels]; the Residual Block's input must have `filters` channels |
| Multi-Head Attention Block | Block | A sequence [length, d_model] |

The column names (`label`, `text`, `next`) are only where the patterns start:
change them in the Data Input dialog to match your file.

## Starting from torchvision's weights

ResNet-18, MobileNetV2 and MobileNetV3 Small are torchvision's networks
layer for layer, and every layer is named as torchvision names it
(`layer1.0.conv1`, `features.3.conv.1.0`). So torchvision's published
ImageNet weights fill them:

1. Download the file: `resnet18-f37072fd.pth`, `mobilenet_v2-b0353104.pth`
   or `mobilenet_v3_small-047dcff4.pth`, from
   `https://download.pytorch.org/models/`.
2. Double-click **DataLoader**, open the **Checkpoints** tab, set
   **Start from** to **A weights file** and pick the file.

Two pattern settings are for this:

- **num_classes**: 1000 keeps the ImageNet head, which then takes its
  weights too. Any other number gives a new head (`fc`, `classifier.1` or
  `classifier.3`), which starts fresh.
- **freeze_backbone**: freezes every other layer that holds weights, so only
  the head trains. Turn **Frozen** off on the layers later to fine-tune the
  whole network.

See [Starting from published weights](../user-guide/training/start-from-pretrained-weights.md) for how layers
are matched and what the match report says.

## Patterns that cannot be used yet

Some patterns need a kind of training the Engine does not have yet. They are
listed greyed under **Not ready yet** at the end of their category, with
the reason on hover and in the open card; their Insert button stays off:

| Pattern | Needs |
| --- | --- |
| DCGAN, Basic GAN | Training a generator against a discriminator (GAN training) |
| Basic Autoencoder | Training against the input itself (a reconstruction target) |
| VAE Encoder, VAE Decoder | VAE sampling and the KL term |
| U-Net | Segmentation masks as labels (one class per pixel) |
| LSTM Seq2Seq | Two-input sequence-to-sequence training (an encoder input and a shifted decoder input) |

Each becomes a full graph when that training is built.

## My patterns: saving your own

1. Select the nodes on the canvas (drag a box, or Ctrl+click).
2. Right-click the canvas and choose **Actions > Save selection as
   pattern...**.
3. The dialog says what is kept: the nodes, the links between them, and
   which links leave the selection and are not kept. A selection that holds
   a Data Input and an Output saves as a **full graph**. Any other selection
   saves as a **block**, with a **Block input** in front of each node that
   an outside link fed.
4. Give it a **Name** (and, if you like, a description and a category), then
   click **Save**. Node settings are kept as they are on the canvas. Saving
   under a name you already used replaces that pattern; the dialog warns you
   first.

A notice on the canvas confirms the save, with **Show in Pattern Browser**.
Saved patterns are listed under **My patterns**, whatever category you
picked, and search finds them too. They are files in your settings folder,
`%APPDATA%\CyxWiz\patterns` on Windows (`~/.cyxwiz/patterns` on Linux and
macOS); the line under My patterns shows the folder, with **Open folder**.
An open card of yours has **Show in folder** and **Delete**: Delete asks
first, then moves the file to the Recycle Bin (the Trash on Linux and
macOS), so you can get it back.

## Limits

- A saved pattern keeps the settings it was saved with; it has no Settings
  of its own to change before inserting.
- Plugin nodes cannot be saved in a pattern yet: the dialog says which node
  stopped the save.

## For developers: the pattern file

Patterns are `.cyxgraph` files in `cyxwiz-engine/resources/patterns`, in the
template format: `id` (the file name), `name`, `kind` (`graph` or `block`),
`needs` (why it cannot be used yet; left out when it can), `description`,
`sample_shape` (one sample at the first layer, for example `[32, 32, 3]`),
`category`, `tags`, `parameters` (each `$name` in the template takes the
value the user sets), and `template` with `nodes` (registry type names) and
`links` (by node id and pin index). A user's patterns are the same format in
the settings folder's `patterns` folder; the library knows each as
`user:<file name>`, so one never replaces a shipped pattern.

Inserting is all or nothing: `PatternLibrary::InstantiatePatternWithCreator`
refuses the whole pattern, with the reason, when a node type is unknown or a
template, or a link names a node or pin that is not there. What the browser
and the Save dialog show (chips, list, insert words, the selection summary)
is `core/pattern_browser_presentation`, checked by
`test_pattern_browser_presentation` against the shipped files;
`test_pattern_template_guard` saves a selection, reads it back, inserts and
deletes it.

ResNet-18, MobileNetV2 and MobileNetV3 Small are written by
`scripts/examples/torch_to_cyxgraph.py` (run it from the repository root;
it needs PyTorch and torchvision) from torchvision's own networks; edit the
script, not the files.

`test_shipped_patterns` inserts every shipped pattern through the Engine's
pattern library and node factory. It checks that every link lands on a real
pin, that every full graph compiles, that every block compiles inside a small
graph, and that the patterns that cannot be used yet refuse to insert. It
also trains the small ones (Basic MLP, Basic CNN, LeNet-5, Basic RNN) for an
epoch on synthetic data, and checks that the three torchvision networks keep
every layer that holds weights under its torchvision name (41, 105 and 79
layers), with the head taking `num_classes` and the rest `freeze_backbone`.
