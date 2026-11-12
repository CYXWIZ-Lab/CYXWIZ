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

## Layers with weights that stay as loaded

A layer's weights can be kept unchanged while the rest trains (freezing). The
setting for it on the layer nodes is not available yet.
