# Neural Architecture Search

**Tools > Model > Neural Architecture Search** proposes changes to the model
on the canvas: a layer more or less, another layer type, other sizes, other
activations. It scores each version and can put the version you pick back on
the canvas.

The score is an estimate made without training: it looks at the graph's
shape (size, depth, variety of layers, skip connections). It does not say
how well a model will learn. To compare versions for real, train them
(Train, or Data Studio > Results > Cross-validation).

## Score the model on the canvas

1. Open the window and, under **Current Architecture Score**, click
   **Load from CyxWiz Studio**.
2. The table shows the overall score and its parts:

| Part | What it rewards |
| --- | --- |
| Complexity | fewer parameters |
| Efficiency | more computation per parameter |
| Depth | about 10 trainable layers |
| Diversity | several kinds of layer |
| Connectivity | skip connections: links drawn as skips, and merge nodes (Add, Concatenate, ...) that join two paths |

Below it: the parameter and FLOP counts, the number of model layers
(layers, activations, merges; not the data nodes, the loss or the output)
and a summary such as "Layers: 3x Dense, 2x ReLU".

The counts come from the same analysis as **Model Summary**
([model_summary.md](model_summary.md)): the first layer sees the input size
set under **Input Configuration** (Features, or Channels x Height x Width),
and every later layer is sized by what reaches it. A 784 -> 128 -> 64 -> 10
MLP shows 109,386 parameters, the number the compiled model holds.

## Change the model

Open **Mutation Controls**. Each button changes the loaded copy, not the
canvas, and scores it again:

| Button | What it does |
| --- | --- |
| Add Layer | puts a new layer in front of a trainable layer and links it in. Beside a Dense layer it adds a Dense, BatchNorm or Dropout; beside a convolution, the same convolution, BatchNorm or Dropout |
| Remove Layer | removes a trainable layer and links what fed it to what it fed |
| Swap Layer | replaces a trainable layer with another type from the same choices as Add Layer; its links stay |
| Change Units | multiplies a layer's units or filters by 0.5 to 2 |
| Change Activation | changes one activation to ReLU, LeakyReLU, GELU, Swish or Mish |
| Random | one of the above |

Remove, Swap and Change Units never touch the **head**, the last trainable
layer before the loss: its size is the number of classes (or outputs) the
loss expects.

Click **Apply to CyxWiz Studio** to put the changed model on the canvas. It
replaces the graph:

- nodes that are still there keep their place, settings and group;
- a new node is placed between the nodes it links to (a swapped node stays
  where it was);
- **Ctrl+Z** takes the canvas back.

After applying, click **Load from CyxWiz Studio** again before the next
mutation, so the window works on the canvas as it is now.

## Suggestions

**Architecture Suggestions > Generate Suggestions** builds five small
models for the task under Input Configuration (Classification, Regression
or Image Classification). Click one to see its layers, then **Apply
Selected** to add it to the canvas beside your graph. It is a separate
model: link it to your data, or delete the old model, before training.

## Evolutionary search

**Evolutionary Search** starts from the loaded model and repeats, for the
number of generations set under **Show Config**: mutate the versions, score
them, keep the best (**Elite Count** of them unchanged). **Population
Size**, **Mutation Rate** and **Crossover Rate** set how many versions and
how often each step happens. Crossover keeps one of two parents whole; it
does not mix layers from both.

When it ends, **Apply Best Architecture** puts the best version on the
canvas the same way **Apply to CyxWiz Studio** does.

## Example

Open `examples/cyxgraph/mnist_mlp.cyxgraph` (Data Input, Split, DataLoader,
Normalize, then Dense 128, ReLU, Dense 64, ReLU, Dense 10, CrossEntropy,
Adam, Output):

1. Load from CyxWiz Studio: "Layers: 3x Dense, 2x ReLU", 5 model layers.
2. Add Layer, then Apply: a BatchNorm or Dropout appears between two
   layers; the graph has 13 nodes and still compiles.
3. Ctrl+Z, Load again, Remove Layer, then Apply: one of the first two Dense
   layers goes; Dense 10 stays, so CrossEntropy still gets 10 outputs.

## Limits

- The score does not train. A version that scores higher may learn worse.
- The search window is a floating window with its own layout; it is due a
  redesign to the Engine's current style.
