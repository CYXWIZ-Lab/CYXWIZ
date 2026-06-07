# To Fix 12 - Fine Tuning and Sentiment Model Improvement

This note defines how we should think about fine tuning in CyxWiz and
what we can improve next in the engine without confusing retraining,
checkpointing, and pretrained-model fine tuning.

Current relationship to `tofix19.md`: this file tracks current
sentiment/text-classifier improvement strategy. Pretrained transformer
import, decoder/generative training, and broad PyTorch-parity model
families are tracked in `tofix19.md`.

Short version:
- In the current engine, fine tuning means training the existing graph
  better.
- We can already tune architecture, optimizer settings, checkpoints,
  validation behavior, and test evaluation.
- We do not yet have full pretrained transformer fine tuning inside the
  engine.
- The next work should improve comparability, control, and evaluation
  rather than hardcoding more training behavior into the backend.

## What Fine Tuning Means Here

There are two different meanings that should stay separate.

### 1. Engine-level fine tuning

This is what the current CyxWiz engine already does:
- train the current graph from scratch
- resume from saved checkpoints
- change node parameters and training configuration
- compare different runs on the same dataset split
- export the best checkpoint from training

This is the practical meaning of fine tuning for the current sentiment
project.

### 2. Pretrained-model fine tuning

This is the classic ML meaning:
- load a pretrained model such as BERT, RoBERTa, DistilBERT, or
  BERTweet
- attach a classification head
- freeze some layers or train all layers
- adapt the pretrained weights to the target sentiment dataset

The current backend does not yet provide that as a first-class path.

## Current Engine State

Already implemented or available:
- train/validation/test split for the text dataset
- DataLoader UI controls for training policy
- save-best-checkpoint support
- early stopping support
- pause, continue, and cooperative early stop in the training dashboard
- test execution path after training
- GRU and LSTM recurrent layers
- bidirectional recurrent support in the graph model
- Studio debugger run records for training progress and backend failures
- model export from the current trained session

That means the engine can already support real training experiments,
but it still needs better control and comparison tooling.

## What We Can Test Now

We can already run controlled comparisons on the same split:

1. GRU vs LSTM
2. bidirectional on vs off
3. one layer vs multiple layers
4. different hidden sizes
5. different dropout values
6. different batch sizes
7. different learning rates
8. different `num_workers`
9. early stopping patience values
10. best validation checkpoint vs final epoch checkpoint

For sentiment specifically, the deciding metrics should be:
- validation accuracy
- validation loss
- held-out test accuracy
- training time per epoch
- total training stability

Training accuracy alone is not enough.

## What We Should Improve In The Engine

### 1. Make training control fully user-owned

Training policy should stay in the UI and graph settings, not hidden as
hardcoded backend values.

The backend should accept:
- batch size
- epochs
- learning rate
- dropout
- hidden size
- number of layers
- bidirectional flag
- num workers
- save-best-checkpoint flag
- early stopping patience
- checkpoint directory

### 2. Make evaluation explicit

After training, the user should be able to run:
- validation evaluation
- test-set evaluation
- sample-level inference

The engine should show:
- the checkpoint used
- the exact split used
- the resulting accuracy and loss

### 3. Keep best-checkpoint behavior visible

When training improves and then degrades, the engine should make it
obvious which checkpoint is best.

The deployable model should come from:
- best validation epoch, or
- a user-selected checkpoint

not just the last batch of the last epoch.

### 4. Add comparison tooling

The Studio debugger and training dashboard should let us compare runs:
- GRU vs LSTM
- different bidirectional settings
- different training lengths
- different validation curves
- different test outcomes

### 5. Separate speed problems from model quality

If a model is slow, the debugger should tell us whether the bottleneck
is:
- GPU path
- CPU fallback
- preprocessing
- recurrent layer shape/layout
- UI polling

That matters because a slow model is not always a bad model.

## What We Still Do Not Have

Not yet implemented as a first-class engine feature:
- pretrained transformer import and fine tuning
- layer freezing controls
- per-layer learning rate schedules
- automatic hyperparameter search
- benchmark-run management across multiple model families
- architecture-specific preset scoring for transformer models

## Suggested Next Order

1. Keep the current train/val/test split and checkpoint flow stable.
2. Compare GRU and LSTM on the same dataset and same training budget.
3. Keep best-validation checkpointing as the default deployment path.
4. Use the held-out test set only after training is complete.
5. Expose more training controls in the UI if any setting still feels
   hardcoded.
6. Decide later whether the next model family should be transformer-
   based fine tuning rather than more recurrent tuning.

## What This Note Is Not

This is not the Studio debugger architecture note in `tofix9.md`.
It is also not the memory cleanup note in `tofix10.md`, or the text
vocabulary workflow note in `tofix11.md`.

This file exists to keep model-improvement strategy explicit so we can
compare runs and extend the engine without mixing the training policy
into unrelated backend changes.
