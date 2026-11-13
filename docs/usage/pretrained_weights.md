# Starting from published weights

A graph can start training from a weight file someone else published, such as
torchvision's ImageNet weights, instead of from random weights. Layers whose
names and shapes match the file take its weights; the others start fresh.

## Choosing the file

1. Double-click the **Data Loader** node and open the **Checkpoints** tab.
2. Set **Start from** to **A weights file**.
3. **Browse...** to the file, or type its path (a relative path is resolved
   against the project).

**Start from** has three answers, and a run uses exactly one:

| Start from | What the run starts with |
| --- | --- |
| Random weights | Every layer starts fresh (the default). |
| A checkpoint of an earlier run | A checkpoint folder of this Engine; layer names and shapes must match exactly. See [Fine-tuning language models](fine_tuning_language_models.md). |
| A weights file | A published file; matched layer by layer as described below. |

Optimizer state, the learning-rate schedule, epochs and the seed always start
fresh.

## Files that can be read

- **PyTorch state-dict** (`.pth`, `.pt`), as `torch.save(model.state_dict(), path)`
  writes it. A training checkpoint that keeps its state-dict under
  `state_dict`, `model_state_dict`, `model`, `net` or `params` is read too.
- **safetensors** (`.safetensors`).

PyTorch does not need to be installed. The file is read as data and nothing in
it is ever run: a file that holds anything other than tensors (a whole pickled
model, an optimizer, or code) is refused with the reason. Files in PyTorch's
format from before version 1.6 are refused; load them in a current PyTorch and
save the state-dict again.

Float16 and float64 tensors are converted to float32. Integer tensors (such as
BatchNorm's `num_batches_tracked` counter) are not weights and are left out.

## Example

`examples/cyxgraph/cats_dogs_resnet18_finetune.cyxgraph` is torchvision's
ResNet-18 with every node named as torchvision names the layer. Started from
`resnet18-f37072fd.pth` with the backbone frozen, three epochs on cats vs dogs
gave 86.29 % test accuracy; its README has the steps and the numbers.

## How layers are matched

A layer takes the entries whose name is its **node's name** followed by the
parameter: the node `layer1.0.conv1` takes `layer1.0.conv1.weight`. Rename the
nodes of your graph to the names in the file (the match report lists the
file's names).

| Layer | Entries it takes |
| --- | --- |
| Conv2D, Depthwise Conv2D, ConvTranspose2D, Conv1D | `weight`, and `bias` when the node's **Bias** is on |
| BatchNorm | `weight`, `bias`, `running_mean`, `running_var` |
| Dense | `weight`, `bias` |
| Other layers | entries named as the Engine names the layer's parameters, with the same shapes |

PyTorch's layouts are converted:

- Convolution kernels are stored `[out, in, h, w]` in the file and
  `[h, w, in, out]` in the Engine.
- A Dense that reads a flattened feature map (Conv ... -> Flatten -> Dense, as
  VGG and LeNet end) has its weight's columns reordered: PyTorch flattens
  channels first, the Engine channels last. Only activations and Dropout may
  sit between the Flatten and the Dense for this to be recognised.

A tensor saved as an expanded view (one stored value repeated) is refused;
save a plain copy of it.

A layer loads only when **every** one of its parameters is in the file with
the right shape. This is why the settings matter:

- A convolution that a BatchNorm follows has no bias in torchvision's files.
  Turn the node's **Bias** off, or the layer starts fresh because the file has
  no bias for it.
- A classifier with another number of classes (the file has 1000, your graph
  2) starts fresh. That is what fine-tuning wants: the new head is trained on
  your data.

## The summary and the match report

Under the file field, **What the file gives this graph** shows, before any
training:

- how many layers take their weights,
- the layers that start fresh and why (the first three; the report has all),
- how many entries of the file are not used.

**Match report** opens a table with one row per layer: the node, the layer
type, the file's entries with their shapes, the shape in the graph and the
result. The chips filter it to **Loaded**, **Starts fresh** or **In the file
only** (entries no layer took, each with its reason, such as "no node is named
fc").

The summary needs a compiled graph; it appears a moment after the graph
compiles without errors.

A run stops at the start, with the reason, when the file cannot be read or
**no** layer matches. The training log has one line with the summary, and
checkpoints of the run record the file they started from.

## Freezing: layers that stay as loaded

Every layer node that holds weights has a **Frozen** setting in Properties
(group Fine-tuning; off by default). A frozen layer keeps its weights while
the rest of the model trains: it gets no gradient step, no weight decay and no
optimizer memory. Its weights are still saved in checkpoints and exports.

**BatchNorm** also has **When frozen**:

- **Keep its statistics** (default): the layer normalises with its loaded
  running mean and variance, as in evaluation. This is the usual choice for
  fine-tuning on a small dataset.
- **Keep updating statistics**: the weights stay, but the running mean and
  variance keep following the new data. This is what PyTorch does when only
  `requires_grad` is turned off.

The node's **As compiled** card shows a frozen layer's weights as
"Weights: N frozen" instead of "Learnable parameters".

On the canvas a frozen node is dimmed and carries a snowflake mark, and the
line under the canvas reads "Trainable N · Frozen M" once anything is frozen.

### Freezing many layers at once

**Nodes > Freeze**, or right-click the canvas > **Actions > Freeze**:

| Command | What it does |
| --- | --- |
| Freeze selected / Unfreeze selected | The selected nodes that hold weights. |
| Freeze every layer loaded from the file | The layers the Data Loader's weights file fills (the "Loaded" rows of the match report). |
| Freeze all but the head | Every layer with weights except the last one of the model. |
| Freeze all but the last N layers... | Asks for N; layers are counted from the output back, and the last N are unfrozen. |
| Unfreeze all | Every layer trains again. |

Each command is one undo step. The last three need a graph that compiles
without errors, because the order of the layers comes from the compiled
model. The commands are not available while training runs.

The default for a new BatchNorm node's **When frozen** is set in
**Preferences > Training**.

The usual two runs:

1. Start from the weights file, freeze the layers that took weights, and train
   the new head.
2. Turn **Frozen** off, set **Start from** to **A checkpoint of an earlier
   run** (the first run's best checkpoint), lower the learning rate, and train
   everything.

Embedding keeps its own **Freeze Loaded Weights** setting for a table loaded
from a file.
