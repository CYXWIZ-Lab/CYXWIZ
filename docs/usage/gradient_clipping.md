# Gradient clipping

Large gradients can throw a model's weights far off in one step (a common
cause of a loss that suddenly jumps or turns NaN, especially in language
models and recurrent networks). Clipping by the global norm limits how far
one step can move: when the gradients of all the model's parameters together
are longer than a limit, they are scaled down to that length.

## Where to set it

Every optimizer node (**Adam**, **AdamW**, **NAdam**, **SGD**, **RMSprop**,
**Adagrad**) has **Gradient clip norm** (`grad_clip_norm`) in Properties:

| Value | Effect |
|---|---|
| `0` (default) | No clipping |
| `1.0` | The usual value for language models |
| Any positive number | The limit on the global norm |

PPO has its own setting on the **PPO** node, **Max gradient norm**
(`max_grad_norm`, default 0.5), over the policy and value networks together;
see [Training a policy (PPO)](rl_training.md).

## What it computes

The same as PyTorch's `torch.nn.utils.clip_grad_norm_`, called before each
optimizer step:

1. The global norm: the square root of the sum of the squares of every
   gradient element of every parameter.
2. `coef = clip_norm / (norm + 1e-6)`, limited to at most 1.
3. Every gradient is multiplied by `coef`. When the norm is under the limit,
   `coef` is 1 and nothing changes.

With clipping on, the training log shows the norm before clipping on each
reported batch (`grad_norm=...`), so you can see how often it bites. The
PyTorch code the Studio generates (Code Gen mode, Generate) has the same
`clip_grad_norm_` line before `optimizer.step()` in its training-loop
outline.

## When the gradients are not finite

As in PyTorch (its default, `error_if_nonfinite=False`), a non-finite norm is
not refused:

- **A NaN in any gradient** makes the norm NaN, so every gradient becomes
  NaN, and the step makes the weights NaN. The loss then reads NaN from the
  next batch on.
- **An infinite gradient** makes the norm infinite: the finite gradients
  become 0 and only the infinite one turns NaN.

A NaN loss usually means the learning rate is too high, the data holds NaN
or infinite values, or a layer overflows. Lower the learning rate, check the
data (Data Studio > Explore > Missing), or turn on clipping before the
problem appears; clipping cannot repair gradients that are already NaN.

## Checked against PyTorch

`gradient_clipping_pytorch_parity` compares the Engine's three clip paths
(the optimizer nodes' clip, PPO's per-tensor path, and the clip inside the
fused Adam step PPO uses on an NVIDIA GPU) with `clip_grad_norm_` followed by
`torch.optim.Adam`: a clip that bites, one that does not, a NaN norm and an
infinite norm, on the CPU and on CUDA.
