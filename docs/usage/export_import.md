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
