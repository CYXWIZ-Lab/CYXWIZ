# Neural Architecture Search

**Tools > Model > Neural Architecture Search** proposes changes to the model
on the canvas and scores each version. Changes can be made by hand or found
by a search, and the best versions can be trained the way **Train** does.
Any version can be put on the canvas.

The window reads the graph on the canvas and its compile. The input and
output sizes come from the graph, for example "784 features in, 10 classes
out". The input size is known once the graph's data is loaded (its Data
Input). Until then the window says so, and it reads the graph again when the
compile changes. **Read the graph again** (top right) reads the canvas as it
is now. When the canvas changes after the window read it (an edit, an
undo), a notice says so.

## The quick score

The card at the top scores the model without training. It looks only at
the graph's shape:

| Part | What it rewards |
| --- | --- |
| Size | fewer parameters (1 million: 0.25) |
| Efficiency | more computation per parameter |
| Depth | about 10 trainable layers |
| Variety | several kinds of trainable layer |
| Skip links | skip links and merge nodes (Add, Concatenate, ...) that join two paths |

The line under the parts gives the parameter count, the model layers
(layers, activations, merges; not the data nodes or the loss) and the skip
connections. For example: "Parameters 109,386 · model layers: 3x Dense,
2x ReLU · no skip connections". The counts are the Model window's
(Summary, [model_tools.md](model_tools.md#summary)): each version is compiled the way
the canvas compiles, its parameters come from the built model and its
multiply-adds from each compiled layer (two FLOPs each). A version that
does not compile is not scored, and the reason is shown.

The quick score does not say how well a model will learn. Use **Train the
best** (below) to compare versions for real.

## Change the model

Each button changes the version in the window, not the canvas, and adds a
step to the list with what it did and the new score:

| Button | What it does |
| --- | --- |
| Add layer | puts a new layer in front of a trainable layer and links it in. Beside a Dense layer it adds a Dense, BatchNorm or Dropout; beside a convolution, the same convolution, BatchNorm or Dropout |
| Remove layer | removes a trainable layer other than the head; what fed it feeds what it fed |
| Swap layer | replaces a trainable layer other than the head with another type that fits there; its links stay |
| Change units | multiplies a layer's units or filters by 0.5 to 2, not the head |
| Change activation | changes one activation to ReLU, LeakyReLU, GELU, Swish or Mish |
| Random | one of the above |

Hover a button to see this. A step reads like "Added BatchNorm before Dense
#9" or "Removed Dense #7 (ReLU feeds ReLU)". When several nodes share a
name, the step adds the node's number. When a change finds nothing to change,
the window says so and adds no step.

The **head** is the last trainable layer before the loss. Its size is the
number of classes (or outputs) the loss expects, so Remove, Swap and Change
units never touch it. The note under the buttons names it.

- **Apply to the graph** replaces the graph on the canvas with this
  version. Nodes that are still there keep their place, settings and group.
  A new node is placed between the nodes it links to, below them when that
  place is taken. **Ctrl+Z** takes the canvas back.
- **Back to the graph's version** drops the steps.

## Search automatically

The search starts from the version in the window and repeats for a number of
rounds. In each round it changes and mixes the versions, scores them with
the quick score, and keeps the best ones:

| Field | What it sets | Default |
| --- | --- | --- |
| Versions per round | how many versions each round scores | 20 |
| Rounds | how many rounds | 10 |
| Kept unchanged | the best versions carried to the next round as they are | 2 |
| Changed per round | the share of new versions that get a random change | 30 % |
| Made from two | the share of new versions made from two parents (one of the two is kept whole; layers are not mixed) | 50 % |

**Start the search** runs it, and **Stop** ends it after the current
round. The result line gives the rounds, the versions scored and how many
different versions were kept (up to 10, best first).

### Train the best

**Train the best N** (1 to 10) trains the top N versions **for the
DataLoader's epochs, the way Train does**:

- Train's checks and data preparation run first. The data is prepared once
  for all the versions.
- Each version then trains as a fresh model with its own scratch
  checkpoints, which are removed afterwards.
- The model in use and the Training Dashboard's runs are not touched.

The table gives, per version:

- its layers in order (for example "Dense 192, BatchNorm, ReLU, Dense 10");
- the quick score;
- the validation loss and accuracy (MAE for numbers), on the graph's
  validation part;
- **Apply**, which puts that version on the canvas (Ctrl+Z undoes it).

While versions train, the line above the table shows the version and the
epoch, and **Stop** ends the training. A version that fails shows "Failed";
hover it for the reason. Train is not available while versions train, and
training does not start while Train, a hyperparameter search, a
cross-validation or a test runs. If a check stops the training before it
starts, the reason shows above the table.

## Defaults

**Preferences > Search** sets the defaults for the search fields and for
**Train the best**. The window takes them when it opens, and again when they
change, but not while it searches or trains. **Reset** puts the Engine's
defaults back.

## Example

Open `examples/cyxgraph/mnist_mlp.cyxgraph` (Data Input, Split, DataLoader,
Normalize, then Dense 128, ReLU, Dense 64, ReLU, Dense 10, CrossEntropy,
Adam, Output). Load `mnist_test.csv` in its Data Input, with the label
column `label`.

1. The header shows "784 features in, 10 classes out". The quick score is
   0.25 with 109,386 parameters, and the note names the head (Dense #9, the
   10 outputs).
2. Click Add layer twice: two steps, such as "Added Dense 197 before
   Dense #9". Apply to the graph: 14 nodes, the new ones below the chain.
   Ctrl+Z takes the canvas back.
3. Start the search (20 versions, 10 rounds): 200 versions are scored and
   10 different ones are kept, in a few seconds.
4. Train the best 3 (10 epochs each, about 10 s per version on this PC)
   gave 87.3 %, 87.4 % and 88.3 % validation accuracy.

## Limits

- The quick score does not train, and it favours small models: a version
  that scores higher may learn worse. Train the best to compare.
- The search changes one layer at a time, and its mixing keeps one parent
  whole.
