---
redirect_from:
  - ../../../usage/lr_schedulers.md
  - ../../../usage/gradient_clipping.md
related:
  - ../../../user-guide/training/losses-and-optimizers.md
  - ../../../user-guide/training/checkpoints-and-early-stopping.md
  - regularization.md
---

## Learning-rate schedulers

A scheduler node changes the optimizer's learning rate as training goes on.
Connect the optimizer's **State** output to the scheduler's **Optimizer**
input:

```text
Data -> Dense ... -> Loss -> SGD / Adam / AdamW ... --State--> Step LR
```

The optimizer's `learning_rate` is the starting rate; the scheduler sets it
from there. Each node matches its `torch.optim.lr_scheduler` counterpart:

| Node | Rate after `epoch` completed epochs | PyTorch |
| --- | --- | --- |
| [Step LR](../training/step-lr.md) | learning_rate x gamma^floor(epoch / step_size) | `StepLR` |
| [Cosine LR](../training/cosine-lr.md) | eta_min + (learning_rate - eta_min) x (1 + cos(pi x epoch / T_max)) / 2 | `CosineAnnealingLR` |
| [Exponential LR](../training/exponential-lr.md) | learning_rate x gamma^epoch | `ExponentialLR` |
| [Warmup LR](../training/warmup-lr.md) | learning_rate x (start_factor + (1 - start_factor) x min(epoch / warmup_epochs, 1)) | `LinearLR(start_factor, 1.0, warmup_epochs)` |
| [Reduce LR](../training/reduce-lr.md) | cut by `factor` when the validation loss stops improving | `ReduceLROnPlateau(mode='min', threshold_mode='abs')` |

Step, Cosine, Exponential and Warmup step after every completed epoch.
Reduce LR steps after every epoch that ran validation, on that epoch's
validation loss: an epoch improves when its loss is below the best so far
minus `threshold`, and after more than `patience` epochs without improvement
the rate becomes max(rate x factor, min_lr).

Example: SGD with learning_rate 0.0625 and Step LR (step_size 2, gamma 0.5)
trains epochs 1-2 at 0.0625, epochs 3-4 at 0.03125 and epochs 5-6 at
0.015625. Cosine LR past `T_max` rises again, as in PyTorch; set `T_max` to
the run's epoch count to end at `eta_min`.

### The optimizer's own schedule

Every optimizer node also has **LR Schedule** (`lr_schedule`), which changes
the rate after every optimizer update instead of every epoch, with no
scheduler node. The Engine counts the run's updates itself (epochs x
batches, with gradient accumulation).

| `lr_schedule` | The rate over the run |
| --- | --- |
| `none` | `learning_rate` throughout |
| `warmup_cosine` | a linear rise from 0 to `learning_rate` over `warmup_ratio` of the updates, then a cosine down to `min_lr_ratio x learning_rate` |
| `warmup_linear` | the same rise, then a straight line down |
| `warmup_constant` | the rise, then constant |
| `one_cycle` | `torch.optim.lr_scheduler.OneCycleLR`: a cosine from `learning_rate / 25` up to `learning_rate` over `warmup_ratio` of the updates, then a cosine down to `learning_rate / 250,000` |

`one_cycle` is the usual schedule for CNNs trained with SGD: set
`learning_rate` to the peak (for example 0.1), `warmup_ratio` to 0.3,
momentum 0.9 with Nesterov, and a small `weight_decay`. It follows torch
with `cycle_momentum=False` (the momentum stays as set); `min_lr_ratio` is
not used.

### Rules the compiler checks

Each refusal is reported on the scheduler node:

- The scheduler is connected to the optimizer that trains the model, and
  there is one scheduler per optimizer.
- A scheduler node and the optimizer's own `lr_schedule` are not used
  together: use one or the other.
- Reduce LR needs validation data: Data Split `val_ratio` above 0, or a Dev
  dataset.
- `step_size`, `T_max` and `warmup_epochs` are whole numbers of at least 1;
  `start_factor` is above 0 and at most 1.

Resume checkpoints store the scheduler's state, so a resumed run continues
the schedule where it stopped. The PyTorch code export writes the scheduler
and steps it once per epoch (`scheduler.step(val_loss)` for Reduce LR). The
learning rate after every epoch is checked against torch's for each node,
including Reduce LR on the run's own validation losses and a resumed run.

## Gradient clipping

Large gradients can throw a model's weights far off in one step, a common
cause of a loss that suddenly jumps or turns NaN, especially in language
models and recurrent networks. Clipping by the global norm limits how far
one step can move: when the gradients of all the model's parameters together
are longer than a limit, they are scaled down to that length.

Every optimizer node ([Adam](../training/adam.md),
[AdamW](../training/adamw.md), [NAdam](../training/nadam.md),
[SGD](../training/sgd.md), [RMSprop](../training/rmsprop.md),
[Adagrad](../training/adagrad.md)) has **Gradient Clip Norm**
(`grad_clip_norm`): 0 is off, 1.0 is the usual value for language models,
and any positive number is the limit. The PPO node has its own **Max
gradient norm** (`max_grad_norm`), over the policy and value networks
together.

### What it computes

The same as PyTorch's `torch.nn.utils.clip_grad_norm_`, before each
optimizer step:

1. The global norm: the square root of the sum of the squares of every
   gradient element of every parameter.
2. `coef = clip_norm / (norm + 1e-6)`, limited to at most 1.
3. Every gradient is multiplied by `coef`. Under the limit, `coef` is 1 and
   nothing changes.

With clipping on, sequence (language-model) training prints the norm before
clipping on each reported batch of its log (`grad_norm=...`), so you can see
how often it bites. The PyTorch code export writes the same
`clip_grad_norm_` call in its example training loop.

### When the gradients are not finite

As in PyTorch (its default, `error_if_nonfinite=False`), a non-finite norm is
not refused:

- **A NaN in any gradient** makes the norm NaN, so every gradient becomes
  NaN and the step makes the weights NaN. The loss reads NaN from the next
  batch on.
- **An infinite gradient** makes the norm infinite: the finite gradients
  become 0 and only the infinite one turns NaN.

A NaN loss usually means the learning rate is too high, the data holds NaN
or infinite values, or a layer overflows. Lower the learning rate, check the
data in Data Studio (**Explore > Missing**), or turn on clipping before the
problem appears; clipping cannot repair gradients that are already NaN.

The optimizer nodes' clip, PPO's clip and the clip inside the fused Adam
step PPO uses on an NVIDIA GPU are checked against `clip_grad_norm_`
followed by `torch.optim.Adam`: a clip that bites, one that does not, a NaN
norm and an infinite norm, on the CPU and on CUDA.
