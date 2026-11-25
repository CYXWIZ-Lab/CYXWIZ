---
title: Export a trained model
description: Write a trained model to a CyxWiz package or a weights file with File > Export Model, and open a model file's graph again.
type: how-to
area: deployment
audience: user
status: stable
since: 1.0.7
owner: deployment
last_reviewed: 2026-11-25
related:
  - ../../concepts/model-formats.md
  - ../../tutorials/save-export-and-use.md
  - ../training/checkpoints-and-early-stopping.md
redirect_from:
  - ../../usage/export_import.md
---

Use this when a graph has trained and you want to keep the model, use its
weights in another framework, or open it again later. What each format
holds is explained in [Model formats](../../concepts/model-formats.md).

## Before you start

- A graph that has finished training in this session. Without a trained
  model the dialog says **No trained model** and asks you to train first.

## Export the model

1. Choose **File > Export Model** (also in **Deploy > Export Model**) and
   pick the format, or press Ctrl+E for a CyxWiz package. The **Export
   Model** dialog opens.
2. Under **Format**, choose one:

   | Format | What the file holds |
   | --- | --- |
   | **CyxWiz package (.cyxmodel)** | The weights with the graph, configuration and, as you choose, the training history. Its graph opens again in CyxWiz. A CyxWiz package is how any trained model is saved; there is no separate Save Model. |
   | **Weights only (.safetensors)** | The parameter tensors only, readable by PyTorch and the safetensors library. No graph. |

3. Check **File**. It starts as `<project>/exports/<graph name><extension>`;
   switching the format changes the extension. **Browse...** picks another
   file.
4. For a package, set **Include**: **Graph** and **Training history**
   (loss and accuracy per epoch).
   - **Graph** is ticked and cannot be cleared for a graph with branches
     (Split, Add, Concatenate, multi-input layers) or with an extension
     node: the graph is what rebuilds the model around the weights. Hover
     it for the reason.
   - The optimizer state is not saved in a package; training checkpoints
     keep it.
5. Optional: change **Name** (it starts as the trained model's name, else
   the graph's file name), and fill **Author (optional)** and **Description
   (optional)**. Left empty, they keep what the trained model already
   carries.
6. Click **Export**. The file is written in the background.

The result lists the file, its size, the parameters and layers, and the
time, with **Show in folder**, **Export another** and **Close**. If the
export fails, the reason is shown with **Try again**.

## Open a model file's graph

1. Choose **File > Open Model File...** (Ctrl+I).
2. Click **Browse...** (it starts in the project's `exports` folder), or
   type the file's path and press Enter. The file is read in the
   background, once per chosen file.
3. Read what the dialog shows:
   - the format (*CyxWiz package (CYXW v3)* or *Weights only
     (safetensors)*);
   - **Name**, **Author** and **Description** when the file has them,
     **Parameters** (for example "519,719 in 6 layers"), **Trained**
     (epochs, final accuracy and loss), **Contains** (graph, weights,
     training history, tokenizer, vocabulary) and **Size**;
   - the layers with their shapes (the first 40, then "and N more").
4. Click **Open its graph**. The package's graph loads onto the canvas and
   the dialog closes. A package of a graph with branches loads back as that
   graph, with the same outputs.

The weights are not loaded: train the graph, or load one of its training
checkpoints with **Train > Test > Load Checkpoint for Testing**. A
weights-only file has no graph, so **Open its graph** stays off and says
why. To start a graph from a weights file instead, see
[Start from pretrained weights](../training/start-from-pretrained-weights.md).

## ONNX and GGUF

- **Export**: the released packages are built without the ONNX exporter.
  **File > Export Model > ONNX model (.onnx)** is shown greyed with "ONNX
  export is not compiled into this build.", and the dialog offers only the
  package and weights only. **GGUF LLM package (.gguf)** is shown greyed
  too: the Engine does not write GGUF. See
  [ONNX](../../concepts/model-formats.md#onnx).
- **Open Model File**: the file picker lists `.onnx` and `.gguf` files,
  but the Engine cannot read them. The dialog says **This file could not
  be read**, with a reason saying that reading this format is not
  implemented. Only CyxWiz packages and weights-only files open.

## If something goes wrong

- **No trained model**: train the graph in this session first; the
  trained model is what gets exported.
- **This file could not be read** with "[CW-F-0201] Unknown model format":
  the file is not a model the Engine knows.

## Related

- [Model formats](../../concepts/model-formats.md)
- [Save, export and use a model](../../tutorials/save-export-and-use.md)
- [Checkpoints and early stopping](../training/checkpoints-and-early-stopping.md)
