# The node contract

What every extension node must guarantee, whichever route it comes from: a
C++ plugin or a Python node. The Engine relies on these rules; where it can,
it checks them and stops with a message that names the node.

> Checked against plugin API 1.2.0 and pycyxwiz 0.2.0.

Each rule names the C++ member ([developer guide](plugin-developer-guide.md))
and the Python one ([python-modules.md](python-modules.md),
[python-nodes.md](python-nodes.md)).

## Identity

| Rule | C++ | Python | Why |
|---|---|---|---|
| The type id is `<provider id>:<TypeName>` | provider id = `plugin.json` `id`; type name = `TensorNodeInfo::type_name` | provider id = the decorator's `id`; type name = its `name` | Graphs save the type id; it is how a node is found again |
| Provider ids use `a-z 0-9 . - _`; type names use letters, digits and `_` | checked when the plugin loads; a node that breaks it is not added and the reason is shown | checked by the decorator (`ValueError`) | |
| Never rename the provider id or a type name once graphs use them | | | A renamed node opens as "Not installed" |
| Raise the node's version when its pins, settings or code change | `TensorNodeInfo::version` | the decorator's `version` | The Engine warns when a graph was saved with another version; for a Python node it also warns when the code changed under the same version |

## Shapes

| Rule | C++ | Python | Why |
|---|---|---|---|
| Shapes describe **one sample**, without the batch | `InferOutputShape(input_shape, settings)`, `CreateModule(input_shape, ...)` | `output_shape(self, input_shape)`, `build(self, input_shape)` | The compiler and the model builder work per sample |
| The output shape is the input shape unless you say otherwise | always implement `InferOutputShape` | leave `output_shape` out and the node keeps the input's shape | |
| Computing the output shape is pure: same input, same answer, no side effects, cheap | `InferOutputShape` | `output_shape` runs on a fresh module that was never built: do not rely on `build` there | It runs after every graph edit |
| Refuse bad settings with a message that names the setting | return an `ExtensionShapeResult` with `ok = false` and the message | `Param(min=, max=, choices=)` checks values; raise `ValueError` for anything else | The message appears on the node and under the setting in Properties |
| The module's output per sample equals the shape you announced | checked on every forward pass | checked on every forward pass | |

## The module

| Rule | C++ | Python | Why |
|---|---|---|---|
| The forward step receives the whole batch; dimension 0 is the batch, and the batch size must not change | `Forward(x)` | `forward(self, x)` | Modules are chained over the batch tensor |
| The backward step returns the gradient for the input of the last forward, with that input's shape | `Backward(g)` | `backward(self, grad_output)` | The result is passed to the previous module; checked |
| Learnable tensors and their gradients use the same names and shapes | `GetParameters`, `GetGradients`, `SetParameters` with the same keys; a module that ignores `SetParameters` never learns | `add_parameter(name, value)` in `__init__` or `build`; `set_gradient(name, g)` in `backward`; read with `parameter(name)` | The optimizer reads both, steps, and writes back. Checked |
| Parameter names are part of saved models | | | A trained model saves its parameters by name; renaming one breaks loading older `.cyxmodel` packages |
| Report whether there are parameters | `HasParameters()` | automatic once `add_parameter` was called | Modules without parameters are skipped by the optimizer |
| Do not change the input tensor | | | Other modules may hold it |
| Use tensor operations, not raw device arrays | `cyxwiz::Tensor` ([tensor-operations.md](tensor-operations.md)) | `pycyxwiz.Tensor` and the pycyxwiz functions | Same results on every device |
| Honour training mode when behaviour differs in training (as Dropout does) | `SetTraining(bool)` | `set_training(self, training)` / `is_training(self)` | Evaluation and inference switch it off |
| Same seed, same result | | | Reproducible runs |

## Losses (Python only)

| Rule | Why |
|---|---|
| A loss has the fixed pins Predictions (Tensor), Targets (Labels) and Loss | The graph connects it like the built-in losses |
| `forward(self, predictions, targets)` returns the loss as one value; `backward(self, predictions, targets)` returns the gradient for `predictions`, with their shape | The trainer starts the backward pass from it |
| `predictions` are the last layer's raw outputs (a classification loss applies its own softmax); `targets` are one-hot rows for `target="categorical"`, values for `target="continuous"` | The decorator's `target` tells the Engine which labels to feed |
| A loss has no input shape, no `build` and no learnable parameters; it may have settings (`nodes.Param`) | It is the training objective, not a layer |

[python-nodes.md](python-nodes.md) explains what `predictions` and
`targets` hold for classification and regression.

## The gradient

| Rule | Why |
|---|---|
| You write the backward step; the Engine has no autograd for extension nodes yet | Decision D1 of TOFIX125 |
| It must pass the gradient check (relative error at most 0.02 by default) | A wrong gradient trains silently badly; the check catches it before training |

The check compares your backward step with central differences of your
forward step on random inputs. In C++ it is `cyxwiz::CheckModuleGradients`
and `cyxwiz-plugin-check` ([developer guide](plugin-developer-guide.md));
in Python `pycyxwiz.gradcheck`, `pycyxwiz.gradcheck_loss` and
`nodes.gradcheck` ([python-modules.md](python-modules.md)). Properties runs
it for a node on the canvas.
