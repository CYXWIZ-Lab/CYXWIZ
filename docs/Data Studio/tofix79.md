# To Fix 79 - Objective-Family Dispatch for Target-Free Estimators and RL

## Status

Open - Track70 follow-up. This ticket extends compiler/training dispatch; it
does not claim that every unsupervised or reinforcement-learning algorithm is
implemented.

## Decision statement

Replace the assumption that every `Train` operation is a tensor model plus
target plus loss plus optimizer with an explicit objective-family contract.

The initial families are:

```text
TensorSupervised
TensorGraphGeneratedTargets
EstimatorFit
EnvironmentInteraction
```

Each family owns its required graph roles, executor, progress/metrics schema,
checkpoint/artifact type, and test/evaluation contract.

## Current truth

The compiler now records whether an objective requires targets and can resolve
targets from dataset columns, dataset structure, graph generation, or an
environment. Time-series windows and causal language modeling can therefore
generate targets without a raw label column.

Execution is still split:

- neural training uses the tensor loss/optimizer path;
- target-free estimators such as PCA or K-Means run as Data Studio pipeline
  operators;
- reinforcement learning uses a separate Python-driven route;
- unsupported objective families must currently fail honestly.

The next step is typed dispatch, not removal of all validation.

## Family contracts

### TensorSupervised

Requires model predictions, compatible targets, loss, optimizer, and an
evaluation metric contract.

### TensorGraphGeneratedTargets

Uses the tensor training executor but obtains targets from a declared graph
producer. The compiler validates producer shape, alignment, and leakage rules.

### EstimatorFit

Requires a dataset and estimator configuration, but no label, tensor loss, or
optimizer. Produces a fitted estimator artifact plus optional transformed data,
assignments, scores, or components according to the estimator type.

### EnvironmentInteraction

Requires a typed environment transition/step contract:

```text
observation, action, reward, next_observation, terminated, truncated
```

It uses an RL executor and policy/algorithm artifact contract. A dataset label
pin is neither required nor accepted as a substitute.

## Implementation phases

1. Add `ObjectiveFamily` and an executor capability registry to the compiled
   plan.
2. Move existing supervised and graph-generated-target routes behind explicit
   family validation without changing their behavior.
3. Route one target-free reference estimator, preferably K-Means or PCA,
   through `EstimatorFit` with background progress, cancellation, artifact
   identity, and evaluation.
4. Define the environment interface and adapt one existing RL route without
   embedding Python-specific concepts in the core contract.
5. Extend Dashboard, Tasks, checkpoints/artifacts, and persisted run history
   with family-specific metrics and actions.

## Compiler rules

- Missing targets are an error only for families that require targets.
- A tensor loss or optimizer connected to `EstimatorFit` is a topology error.
- `EnvironmentInteraction` requires an environment capability and an RL
  algorithm; a Data Input label cannot satisfy it.
- The selected family and executor must be implemented in the current build.
- No family may silently fall back to supervised tensor training.

## Acceptance criteria

- A supervised missing-target graph still fails with the existing corrective
  diagnostic.
- A time-series graph-generated-target model still compiles and trains.
- A target-free reference estimator runs from `Train` without a fabricated
  label/loss/optimizer and produces a reloadable artifact.
- Cancellation and task progress work for the estimator executor.
- An RL graph without a compatible environment fails at compile time; the
  reference RL route consumes the complete transition contract.
- Run history and the Dashboard identify the objective family and show only
  meaningful metrics for it.

## Non-goals

- exposing unimplemented estimators or RL algorithms in the GUI;
- one executor containing branches for every learning paradigm;
- treating clustering assignments as supervised ground-truth labels;
- using an unrestricted script as an implicit engine environment contract.

## Dependencies

This ticket consumes the role/target language from To Fix 78 and should use
the persistence contracts from To Fix 75 and To Fix 76 for resumable or
reloadable family-specific artifacts.
