---
title: Search for a better architecture
description: Change the model by hand or by a search in Tools > Model > Neural Architecture Search, score each version and train the best ones.
type: how-to
area: training
audience: user
status: experimental
since: 1.0.8
owner: training
last_reviewed: 2026-11-25
related:
  - hyperparameter-search.md
  - ../../main-window/model-window.md
redirect_from:
  - ../../usage/neural_architecture_search.md
---

**Tools > Model > Neural Architecture Search** proposes changes to the model
on the canvas and scores each version. You change the model by hand or let a
search find versions, train the best ones the way **Train** does, and put
any version on the canvas.

## Before you start

- The graph on the canvas compiles and its Data Input is loaded. The window
  reads the input and output sizes from the graph, for example *784 features
  in, 10 classes out*; until the data is loaded it says so, and it reads the
  graph again when the compile changes.
- **Read the graph again** (top right) reads the canvas as it is now. When
  the canvas changes after the window read it (an edit, an undo), a notice
  says so.

## Read the quick score

The card at the top scores the model without training. It looks only at the
graph's shape:

| Part | What it rewards |
| --- | --- |
| Size | Fewer parameters (1 million: 0.25) |
| Efficiency | More computation per parameter |
| Depth | About 10 trainable layers |
| Variety | Several kinds of trainable layer |
| Skip links | Skip links and merge nodes (Add, Concatenate, ...) that join two paths |

The line under the parts gives the parameter count, the model layers
(layers, activations, merges; not the data nodes or the loss) and the skip
connections, for example *Parameters 109,386 · model layers: 3x Dense, 2x
ReLU · no skip connections*. The counts are those of the
[Model window](../../main-window/model-window.md)'s Summary: each version is
compiled the way the canvas compiles, its parameters come from the built
model and its multiply-adds from each compiled layer (two FLOPs each). A
version that does not compile is not scored, and the reason is shown.

The quick score does not say how well a model will learn. Use **Train the
best** to compare versions for real.

## Change the model by hand

Each button changes the version in the window, not the canvas, and adds a
step to the list with what it did and the new score (hover a button for its
rule):

| Button | What it does |
| --- | --- |
| **Add layer** | Puts a new layer in front of a trainable layer and links it in: beside a Dense layer a Dense, BatchNorm or Dropout; beside a convolution the same convolution, BatchNorm or Dropout |
| **Remove layer** | Removes a trainable layer other than the head; what fed it feeds what it fed |
| **Swap layer** | Replaces a trainable layer other than the head with another type that fits there; its links stay |
| **Change units** | Multiplies a layer's units or filters by 0.5 to 2, not the head |
| **Change activation** | Changes one activation to ReLU, LeakyReLU, GELU, Swish or Mish |
| **Random** | One of the other changes, picked at random |

A step reads like *Added BatchNorm before Dense #9* or *Removed Dense #7
(ReLU feeds ReLU)*. When a change finds nothing to change, the window says
so and adds no step.

The **head** is the last trainable layer before the loss. Its size is the
number of classes (or outputs) the loss expects, so Remove, Swap and Change
units never touch it. The note under the buttons names it.

- **Apply to the graph** replaces the graph on the canvas with this version.
  Nodes that are still there keep their place, settings and group; a new
  node is placed between the nodes it links to, below them when that place
  is taken. **Ctrl+Z** takes the canvas back.
- **Back to the graph's version** drops the steps.

## Search automatically

The search starts from the version in the window and repeats for a number of
rounds. In each round it changes and mixes the versions, scores them with
the quick score, and keeps the best ones.

| Field | What it sets | Default |
| --- | --- | --- |
| **Versions per round** | How many versions each round scores | 20 |
| **Rounds** | How many rounds | 10 |
| **Kept unchanged** | The best versions carried to the next round as they are | 2 |
| **Changed per round** | The share of new versions that get a random change | 30 % |
| **Made from two** | The share of new versions made from two parents (one of the two is kept whole; layers are not mixed) | 50 % |

1. Set the fields and select **Start the search**. **Stop** ends it after the
   current round.
2. Read the result line: the rounds, the versions scored and how many
   different versions were kept (up to 10, best first).

## Train the best versions

**Train the best N** (1 to 10) trains the top N versions for the
DataLoader's epochs, the way Train does. Train's checks and data preparation
run first, and the data is prepared once for all the versions. Each version
trains as a fresh model with its own scratch checkpoints, removed afterwards.
The model in use and the Training Dashboard's runs are not touched.

The table gives, per version: its layers in order (*Dense 192, BatchNorm,
ReLU, Dense 10*), the quick score, the validation loss and accuracy (MAE for
numbers) on the graph's validation part, and **Apply**, which puts that
version on the canvas (Ctrl+Z undoes it).

While versions train, the line above the table shows the version and the
epoch, and **Stop** ends the training. A version that fails shows *Failed*;
hover it for the reason. Train is not available while versions train, and
training does not start while Train, a hyperparameter search, a
cross-validation or a test runs. If a check stops the training before it
starts, the reason shows above the table.

**Preferences > Search** sets the defaults for the search fields and for
Train the best. The window takes them when it opens and when they change,
but not while it searches or trains. **Reset** puts the Engine's defaults
back.

## Example

Open `examples/cyxgraph/mnist_mlp.cyxgraph` (Data Input, Split, DataLoader,
Normalize, Dense 128, ReLU, Dense 64, ReLU, Dense 10, CrossEntropy, Adam,
Output). Load `mnist_test.csv` in its Data Input with the label column
`label` (`python examples/cyxgraph/model_tools/get_mnist.py` downloads it).

1. The header shows *784 features in, 10 classes out*. The quick score is
   0.25 with 109,386 parameters, and the note names the head (Dense #9, the
   10 outputs).
2. Select **Add layer** twice: two steps, such as *Added Dense 197 before
   Dense #9*. **Apply to the graph** gives 14 nodes, the new ones below the
   chain. Ctrl+Z takes the canvas back.
3. **Start the search** (20 versions, 10 rounds): 200 versions are scored
   and 10 different ones are kept, in a few seconds.
4. **Train the best 3** (10 epochs each, about 10 s per version) gave 87.3 %, 87.4 % and 88.3 % validation accuracy in one run.

## Limits

- The quick score does not train, and it favours small models: a version
  that scores higher may learn worse. Train the best to compare.
- The search changes one layer at a time, and its mixing keeps one parent
  whole.

## Related

- [Search for better hyperparameters](hyperparameter-search.md)
- [The Model window](../../main-window/model-window.md)
- [Training](index.md)
