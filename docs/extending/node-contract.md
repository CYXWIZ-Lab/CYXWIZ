# The node contract

What every extension node must guarantee, whichever route it comes from (C++
plugin now, Python node later). The Engine relies on these rules; where it
can, it checks them and stops with a message that names the node.

> Checked 2026-10-16 against plugin API 1.2.0.

## Identity

| Rule | Why |
|---|---|
| The type id is `<provider id>:<TypeName>`. Provider ids use `a-z 0-9 . - _`; type names use letters, digits and `_` | Graphs save the type id; it is how a node is found again |
| Never rename the provider id or a type name once graphs use them | A renamed node opens as "Not installed" |
| Raise the node's `version` when its pins or settings change | The Engine warns when a graph was saved with another version |

## Shapes

| Rule | Why |
|---|---|
| `InferOutputShape` and `CreateModule` get **one sample's** shape, without the batch | The compiler and the model builder work per sample |
| `InferOutputShape` is pure: same input, same answer, no side effects, cheap | It runs after every graph edit, from any thread |
| Refuse bad settings in `InferOutputShape` with a message that names the setting | The message appears on the node and under the setting in Properties |
| The module's output per sample must equal what `InferOutputShape` returned | Checked on every forward pass |

## The module

| Rule | Why |
|---|---|
| `Forward(x)` receives the whole batch; dimension 0 is the batch, and the batch size must not change | Modules are chained over the batch tensor |
| `Backward(g)` returns the gradient for the input of the last `Forward`, with that input's shape | The result is passed to the previous module; checked |
| `GetParameters`, `GetGradients` and `SetParameters` use the same keys, with matching shapes | The optimizer reads both, steps, and writes back with `SetParameters`; a module that ignores `SetParameters` never learns. Checked |
| `HasParameters()` is true when there are parameters | Modules returning false are skipped by the optimizer |
| Do not change the input tensor | Other modules may hold it |
| Use `cyxwiz::Tensor` operations | Same results on every device, and the same rule as Python nodes |
| Honour `SetTraining` when behaviour differs in training (as Dropout does) | Evaluation and inference call it |
| Same seed, same result | Reproducible runs |

## The gradient

| Rule | Why |
|---|---|
| You write the backward step; the Engine has no autograd for extension nodes yet | Decision D1 of TOFIX125 |
| It must pass the gradient check (relative error at most 0.02) | A wrong gradient trains silently badly; the check catches it before training |

See the [developer guide](plugin-developer-guide.md#7-the-gradient-check)
for how the check works.
