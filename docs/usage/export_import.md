# Exporting a model

Train the graph, then **File > Export Model** (Ctrl+E). Without a trained
model the dialog says *No trained model* and asks you to train first.

## Format

Pick one; only the formats this build can write are shown:

| Format | What the file holds |
| --- | --- |
| CyxWiz package (`.cyxmodel`) | The weights with the graph, configuration and, as you choose, the training history. Its graph opens again in CyxWiz. |
| Weights only (`.safetensors`) | The parameter tensors only, readable by other frameworks (PyTorch, safetensors). No graph. |
| ONNX (`.onnx`) | A model for inference in other runtimes, with an opset version. Shown only when the build includes the ONNX exporter. |

## File and details

- **File** starts as `<project>/exports/<graph name><extension>`; switching
  the format changes the extension. **Browse...** picks another file.
- **Include** (package only): Graph, Training history. The optimizer state
  is not saved in a package yet; checkpoints keep it.
- **Name** starts as the trained model's name, else the graph's file name.
  **Author** and **Description** are optional; left empty, they keep what the
  trained model already carries.

**Export** writes the file in the background. The result lists the file, its
size, the parameters and layers, and the time, with **Show in folder**,
**Export another** and **Close**. If it fails, the reason is shown with
**Try again**.

Example (sentiment LSTM with word vectors, 1 epoch): `exports/
sentiment_analysis_lstm_word_vectors.cyxmodel`, 2.07 MB, 519,719 parameters
in 6 layers.

# Opening a model file

**File > Open Model File...** (Ctrl+I) reads a model file and opens its graph
in Studio. Choose the file with **Browse...** (it starts in the project's
`exports` folder) or type its path and press Enter. The file is read in the
background, once per chosen file, and the dialog shows:

- the format (*CyxWiz package (CYXW v3)*, *Weights only (safetensors)*, ONNX,
  GGUF);
- Name, Author and Description when the file has them, Parameters ("519,719
  in 6 layers"), Trained (epochs, final accuracy and loss), Contains (graph,
  weights, training history, tokenizer, vocabulary) and Size;
- the layers with their shapes (the first 40, then "and N more").

**Open its graph** loads the package's graph onto the canvas and closes the
dialog. The weights are not loaded: train the graph, or load one of its
training checkpoints with **Train > Test > Load Checkpoint for Testing**. A
weights-only file has no graph, so the button stays off and says why. A file
that is not a model says *This file could not be read* with the reason
("[CW-F-0201] Unknown model format").
