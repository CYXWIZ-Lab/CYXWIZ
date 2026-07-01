# Training checkpoints and early stopping

## What early stopping means

Early stopping means training stopped before the configured maximum epoch because validation performance stopped improving.

Example:

- Configured epochs: `30`
- Stopped epoch: `11`
- Terminal status: `early_stopped`
- Terminal reason: `validation_loss_plateau_patience_8`

This should be read as:

The model was allowed to train for up to 30 epochs, but validation loss did not improve for 8 validation checks, so the engine stopped at epoch 11.

This is different from a normal completed run.

## Dashboard status

The Training Dashboard should distinguish these states:

- `TRAINING`: training is still active.
- `COMPLETED`: all configured epochs finished normally.
- `EARLY STOPPED`: training stopped because an early-stop rule triggered.
- `CANCELLED`: the user stopped the run.
- `FAILED`: training ended because of an error.

If the dashboard shows `EARLY STOPPED`, the epoch display can still show something like `11 / 30`. That means the run stopped at epoch 11 out of the configured 30 epochs.

## Best validation checkpoint

When `save_best_checkpoint=true`, the engine saves the best validation model during training.

The best checkpoint is selected by validation loss.

For example, a run may finish or early-stop at epoch 11, but the best checkpoint may be epoch 3 if epoch 3 had the best validation loss.

In that case:

- final observed training state may show epoch 11
- best saved model may be epoch 3
- exported trained model should use the restored best checkpoint when restore succeeds

This prevents exporting the overfit final epoch when a better validation checkpoint exists.

## Checkpoint directory

The DataLoader property `checkpoint_dir` is the optional base location where checkpoints are saved.

If `checkpoint_dir` is empty, the engine uses the default run-local checkpoint path:

```text
<engine working directory>/.cyxwiz/checkpoints/<run_id>/
```

For a Release build run from `build/bin/Release`, this usually looks like:

```text
build/bin/Release/.cyxwiz/checkpoints/<run_id>/
```

If `checkpoint_dir` is set, the engine saves checkpoints under:

```text
<checkpoint_dir>/<run_id>/
```

The best checkpoint is stored under:

```text
<checkpoint_root>/<run_id>/best/
```

Typical files:

```text
best/metadata.json
best/model/manifest.json
best/model/layer0.weight.bin
best/model/layer0.bias.bin
```

## Important checkpoint fields

`metadata.json` records the checkpoint facts:

- checkpoint epoch
- global step
- training loss
- training accuracy
- validation loss
- validation accuracy
- learning rate
- metric history

These values explain which model was selected as best and why.

## Practical reading of an overfitting run

Example from the sentiment TF-IDF graph:

- final train accuracy: about `90.5%`
- best validation accuracy: about `73.6%`
- best checkpoint epoch: `3`
- terminal status: `early_stopped`
- terminal reason: `validation_loss_plateau_patience_8`

Interpretation:

The model learned the training set strongly, but validation stopped improving early. The engine stopped training to avoid keeping a worse overfit model, then restored the best validation checkpoint.

## Recommended usage

- Keep `save_best_checkpoint=true` for normal supervised training.
- Use validation split when early stopping is enabled.
- Treat `EARLY STOPPED` as a successful controlled stop, not a crash.
- Look at best validation checkpoint metrics, not only final training accuracy.
- Set `checkpoint_dir` only when you want checkpoints saved outside the default `.cyxwiz/checkpoints` run folder.
