# Model Summary

**Tools > Model > Model Summary** lists the model on the canvas layer by
layer: the shape each layer puts out, its learnable parameters and its
FLOPs. Click **Refresh** after changing the graph (or turn on
**Options > Auto-refresh**).

## Tabs

| Tab | Shows |
| --- | --- |
| Layer Table | One row per node in graph order: output shape, parameters, FLOPs (and memory with **Options > Show Memory**) |
| Summary | Totals: parameters (trainable and not), FLOPs, activation memory |
| Shape Flow | The shapes from input to output as a chain of boxes |

**Options > Batch Size** scales the FLOPs and memory; parameters do not
depend on it. **Export** saves the summary as text or JSON.

## Where the shapes come from

The first shape is one sample as the compile reads it from the data: for
example `(64, 64, 3)` after a Resize 64x64 on colour images, or `(784)` for
a 784-column table. Each layer is then sized by what reaches it, so a
Dense layer after a Flatten counts the flattened width, and a Conv2D with
padding `same` keeps the height and width (`valid` shrinks them by
kernel - 1).

If the graph has not compiled yet, the first shape falls back to the
Dataset Input node's height, width and channels (28 x 28 x 1 when it has
none).

**Tools > Model > Architecture Diagram** draws the same analysis.

## Example

The cats and dogs CNN (Resize 64x64, three Conv 3x3 `same` blocks of 32,
64 and 128 filters with BatchNorm and 2x2 pooling, then Flatten and
Dense 2):

| Layer | Output | Parameters |
| --- | --- | --- |
| Conv 32 | (64, 64, 32) | 896 |
| BatchNorm 1 | (64, 64, 32) | 64 |
| Pool 1 | (32, 32, 32) | - |
| Conv 64 | (32, 32, 64) | 18,496 |
| Conv 128 | (16, 16, 128) | 73,856 |
| Pool 3 | (8, 8, 128) | - |
| Flatten | (8192) | - |
| Dense 2 | (2) | 16,386 |

110,082 learnable parameters in all. BatchNorm's running mean and variance
are not learned; they show under **Show Non-trainable**.
