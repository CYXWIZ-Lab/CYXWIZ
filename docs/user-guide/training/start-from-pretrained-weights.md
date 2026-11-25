---
title: Start from pretrained weights
description: Start training from a PyTorch or safetensors weight file, such as torchvision's ImageNet weights, matched to the graph layer by layer.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - freeze-layers.md
  - fine-tune-language-models.md
  - checkpoints-and-early-stopping.md
redirect_from:
  - ../../usage/pretrained_weights.md
---

A graph can start training from a weight file someone else published, such
as torchvision's ImageNet weights, instead of from random weights. Layers
whose names and shapes match the file take its weights; the others start
fresh. The file is a local file you downloaded; the Engine does not fetch
weights from the internet.

## Before you start

- A `.pth`, `.pt` or `.safetensors` file of the network (see
  [Files that can be read](#files-that-can-be-read)).
- A graph whose layer nodes are named as the file names its parameters (see
  [How layers are matched](#how-layers-are-matched)). For ResNet-18,
  MobileNetV2 and MobileNetV3 Small, the Patterns panel has ready graphs.

## Choose the file

1. Double-click the **Data Loader** node and open the **Checkpoints** tab.
2. Set **Start from** to **A weights file**.
3. **Browse...** to the file, or type its path (a relative path is resolved
   against the project).
4. Compile the graph (F7) and read **What the file gives this graph** under
   the file field (next sections).
5. Optional: freeze the layers that took weights. See
   [Freeze layers while fine-tuning](freeze-layers.md).
6. **Train** (F5).

**Start from** has three answers, and a run uses exactly one:

| Start from | What the run starts with |
| --- | --- |
| **Random weights** | Every layer starts fresh (the default). |
| **A checkpoint of an earlier run** | A checkpoint folder of this Engine; layer names and shapes must match exactly. See [Fine-tune a language model](fine-tune-language-models.md). |
| **A weights file** | A published file, matched layer by layer. |

Optimizer state, the learning-rate schedule, epochs and the seed always
start fresh.

## Files that can be read

- **PyTorch state-dict** (`.pth`, `.pt`), as
  `torch.save(model.state_dict(), path)` writes it. A training checkpoint
  that keeps its state-dict under `state_dict`, `model_state_dict`, `model`,
  `net` or `params` is read too.
- **safetensors** (`.safetensors`).

PyTorch does not need to be installed. The file is read as data and nothing
in it is ever run: a file that holds anything other than tensors (a whole
pickled model, an optimizer, or code) is refused with the reason. Files in
PyTorch's format from before version 1.6 are read too (torchvision's
MobileNetV2 file uses it).

Float16 and float64 tensors are converted to float32. Integer tensors (such
as BatchNorm's `num_batches_tracked` counter) are not weights and are left
out. A tensor saved as an expanded view (one stored value repeated) is
refused; save a plain copy of it.

## How layers are matched

A layer takes the entries whose name is its **node's name** followed by the
parameter: the node `layer1.0.conv1` takes `layer1.0.conv1.weight`. Rename
the nodes of your graph to the names in the file (the match report lists the
file's names).

| Layer | Entries it takes |
| --- | --- |
| Conv2D, Depthwise Conv2D, ConvTranspose2D, Conv1D | `weight`, and `bias` when the node's **Bias** is on |
| BatchNorm | `weight`, `bias`, `running_mean`, `running_var` |
| Squeeze-Excitation | `fc1.weight`, `fc1.bias`, `fc2.weight`, `fc2.bias` (as torchvision's `SqueezeExcitation`) |
| Dense | `weight`, `bias` |
| Other layers | Entries named as the Engine names the layer's parameters, with the same shapes |

PyTorch's layouts are converted. Convolution kernels are stored
`[out, in, h, w]` in the file and `[h, w, in, out]` in the Engine. A Dense
that reads a flattened feature map (Conv ... -> Flatten -> Dense, as VGG and
LeNet end) has its weight's columns reordered, because PyTorch flattens
channels first and the Engine channels last; only activations and Dropout
may sit between the Flatten and the Dense for this to be recognised.

A layer loads only when every one of its parameters is in the file with the
right shape, so the settings matter:

- A convolution that a BatchNorm follows has no bias in torchvision's files.
  Turn the node's **Bias** off, or the layer starts fresh.
- A classifier with another number of classes (the file has 1000, your
  graph 2) starts fresh. That is what fine-tuning wants: the new head is
  trained on your data.

## Read the summary and the match report

Once the graph compiles without errors, **What the file gives this graph**
shows how many layers take their weights, the layers that start fresh and
why (the first three), and how many entries of the file are not used.

**Match report** opens a table with one row per layer: the node, the layer
type, the file's entries with their shapes, the shape in the graph and the
result. The chips filter it to **Loaded**, **Starts fresh** or **In the file
only** (entries no layer took, each with its reason, such as *no node is
named fc*).

A run stops at the start, with the reason, when the file cannot be read or
no layer matches. The training log has one line with the summary, and the
run's checkpoints record the file they started from.

## Networks checked against PyTorch

The Engine's tests build these networks from canvas nodes, start them from
torchvision's published ImageNet files, and compare their outputs with
torchvision's:

| Network | File | Layers with weights |
| --- | --- | --- |
| ResNet-18 | `resnet18-f37072fd.pth` | 41 |
| MobileNetV2 | `mobilenet_v2-b0353104.pth` | 105 |
| MobileNetV3 small | `mobilenet_v3_small-047dcff4.pth` | 79 |

The same three networks are in the Patterns panel as full graphs, named for
these files; see [Patterns](../../main-window/pattern-browser.md#starting-from-torchvisions-weights).
Their graphs are written by `scripts/examples/torch_to_cyxgraph.py`, which
turns a PyTorch network into canvas nodes named as PyTorch names its
modules. It handles convolutions (depthwise too), BatchNorm, ReLU, ReLU6,
Hardswish, Hardsigmoid, SiLU, max and global average pooling, Linear,
Dropout, Squeeze-Excitation and residual additions, and refuses anything
else with the module's name.

## Example

`examples/cyxgraph/cats_dogs_resnet18_finetune.cyxgraph` is torchvision's
ResNet-18 with every node named as torchvision names the layer. Started from
`resnet18-f37072fd.pth` with the backbone frozen, three epochs on cats vs
dogs gave 86.29 % test accuracy; `examples/cyxgraph/cats_dogs_resnet18_finetune.md`
has the steps and the numbers.

## Related

- [Freeze layers while fine-tuning](freeze-layers.md)
- [Fine-tune a language model](fine-tune-language-models.md)
- [Checkpoints, early stopping and resume](checkpoints-and-early-stopping.md)
- [Training](index.md)
