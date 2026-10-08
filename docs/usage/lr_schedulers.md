# Learning-rate schedulers

A scheduler node changes the optimizer's learning rate as training goes on.
Connect the optimizer's **State** output to the scheduler's **Optimizer**
input:

```
Data -> Dense ... -> Loss -> SGD / Adam / AdamW ... --State--> Step LR
```

The optimizer's `learning_rate` is the starting rate; the scheduler sets it
from there. Each node matches its `torch.optim.lr_scheduler` counterpart.

## The nodes

| Node | Settings | Rate (epoch = completed epochs) | PyTorch |
| --- | --- | --- | --- |
| Step LR | `step_size` (10), `gamma` (0.1) | learning_rate x gamma^floor(epoch / step_size) | `StepLR` |
| Cosine LR | `T_max` (100), `eta_min` (0) | eta_min + (learning_rate - eta_min) x (1 + cos(pi x epoch / T_max)) / 2 | `CosineAnnealingLR` |
| Exponential LR | `gamma` (0.95) | learning_rate x gamma^epoch | `ExponentialLR` |
| Warmup LR | `warmup_epochs` (5), `start_factor` (0.1) | learning_rate x (start_factor + (1 - start_factor) x min(epoch / warmup_epochs, 1)) | `LinearLR(start_factor, 1.0, warmup_epochs)` |
| Reduce LR | `factor` (0.1), `patience` (10), `threshold` (0.0001), `min_lr` (0) | cuts the rate by `factor` when the validation loss stops improving | `ReduceLROnPlateau(mode='min', threshold_mode='abs')` |

When it steps:

- Step, Cosine, Exponential and Warmup step after every completed epoch.
- Reduce LR steps after every epoch that ran validation, on that epoch's
  validation loss. An epoch "improves" when its loss is below the best so far
  minus `threshold`; after more than `patience` epochs without improvement the
  rate becomes max(rate x factor, min_lr).

Example: SGD with learning_rate 0.0625 and Step LR (step_size 2, gamma 0.5)
trains epochs 1-2 at 0.0625, epochs 3-4 at 0.03125, epochs 5-6 at 0.015625.

Cosine LR past `T_max` rises again, as in PyTorch; set `T_max` to the run's
epoch count to end at `eta_min`.

## Rules the compiler checks

- The scheduler must be connected to the optimizer that trains the model.
- One scheduler per optimizer.
- Not together with the optimizer's own `lr_schedule` (its per-update
  warmup/decay setting): use one or the other.
- Reduce LR needs validation data: Data Split `val_ratio` above 0, or a Dev
  dataset.
- `step_size`, `T_max` and `warmup_epochs` are whole numbers of at least 1;
  `gamma` and `eta_min` are not negative; `factor` is below 1; `start_factor`
  is above 0 and at most 1.

Each refusal is reported on the scheduler node.

## Resume and export

- Resume checkpoints store the scheduler's state; a resumed run continues the
  schedule where it stopped. A checkpoint from a run without a scheduler cannot
  resume a run that has one.
- The PyTorch export writes the scheduler (`scheduler = optim.lr_scheduler...`)
  and steps it once per epoch (`scheduler.step(val_loss)` for Reduce LR). The
  TensorFlow and Keras exports do not write the graph's optimizer yet, so they
  leave the scheduler out too.

## Checked against PyTorch

`scheduler_node_pytorch_parity` compiles a graph with each node, trains it and
compares the learning rate after every epoch with torch's
(`tests/computation_truth/fixtures/scheduler_node_pytorch.json`). It also
checks Reduce LR against the run's own validation losses and a resumed run
against torch's sequence.
