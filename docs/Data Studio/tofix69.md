# To Fix 69 - Single-Active Model Branches and Controlled Algorithm Comparison

## Status

Open - design and planning ticket. No implementation is authorized by this
document.

## Decision statement

Allow one CyxWiz canvas to contain several complete candidate model branches
that share the same data and partition policy. Exactly **one** branch is the
active training target for a run. All other candidate branches remain visible
and editable but are idle: they consume no training compute and cannot affect
the compiled training configuration.

After each completed run, CyxWiz adds the result to the existing lightweight
run-comparison ledger. The user selects another branch and runs it against the
same frozen data split to make a fair model-selection decision.

This is controlled, sequential comparison. It is not parallel branch
execution, automatic hyperparameter search, or a new experiment-management
service.

## Problem

Today a user can keep several model ideas on one canvas, but training discovery
is graph-wide and does not provide an explicit, durable answer to:

```text
Which model branch is allowed to train right now?
```

That creates ambiguity when Model A, Model B, and Model C are connected to the
same prepared dataset. Rewiring or deleting branches to run each candidate is
slow, risks accidental changes to data policy, and makes comparisons harder to
reproduce.

## User workflow

```text
                         shared data contract
Data Input -> Partition Policy -> Data Loader
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
   Model A: MLP      Model B: LSTM    Model C: GRU
       Active             Idle             Idle
        |                  x                x
       Loss               Loss             Loss
```

1. Build candidate branches on one canvas.
2. Mark one branch as **Active for Training**.
3. Train it. Only that branch is compiled and executed.
4. CyxWiz records its architecture, data/split identity, validation metrics,
   checkpoint, final-test availability, elapsed time, and run status.
5. Select Model B or Model C as active and run again.
6. Compare candidates using the same frozen train/dev policy.
7. Select the winning branch using validation metrics; run the protected test
   partition only for the chosen final candidate, or clearly disclose that
   repeated test evaluation weakens the test set's independence.

## Core model

```text
Canvas graph             : all user-authored nodes and links
Candidate training branch: one valid model-to-loss-to-optimizer route
Active branch            : the sole candidate selected for the next run
Idle branch              : a candidate explicitly excluded from compilation
Run comparison ledger    : completed-run records; already exists
```

The active state is graph metadata, not an inferred property of whichever
node happens to be visually closest to the Data Loader. It must be serialized
with the graph and included in run provenance.

Suggested compact contract:

```text
TrainingBranchId = stable branch root/output identity

GraphTrainingSelection
  active_branch_id: optional TrainingBranchId
  selection_revision: integer
```

There must be exactly zero or one active branch. Zero means training is blocked
with a clear action: select a valid branch. More than one is an invalid graph
state and must be rejected, not resolved by node order.

## Canvas UX

### Branch state

- The active candidate has a visible `Active` badge and normal link styling.
- Idle candidates are dimmed and display `Idle`.
- The user-requested `x` appears on the branch's training-entry link as an
  **exclude-from-run** control. It does not delete the link or change graph
  topology. Hover text must say `Idle for training; still part of canvas`.
- Selecting `Set Active` on an idle branch atomically makes it active and
  makes the previously active branch idle.
- A user can reactivate an idle branch without rebuilding or reloading data.

Do not use a literal delete affordance without explanatory text. An `x` on a
normal link commonly means "delete link"; the visual control therefore needs
a state label, undo support, and a separate normal link-delete action.

### Entry points

Support the same action from:

- right-click branch context menu: `Set Active for Training`;
- branch header/badge: `Active` or `Idle` toggle;
- Training Dashboard: active branch name and `Change` action;
- optional keyboard command after the primary UI is proven clear.

The normal Run button runs the selected branch only. It must display the
branch name before launch, for example:

```text
Run Training — Model B: LSTM
```

## Compiler and runtime contract

```text
validate graph
  -> resolve GraphTrainingSelection.active_branch_id
  -> validate the selected branch reaches one model, loss, optimizer,
     output contract, and shared dataset policy
  -> compile selected branch only
  -> build train/dev/test runtime batchers from the shared partition manifest
  -> train selected branch
  -> validate during training
  -> optionally run final test according to test-evaluation policy
  -> append one comparison record
```

Idle branches must be excluded before layer extraction, shape inference,
optimizer selection, backend placement planning, memory estimation, and model
construction. They must not cause errors merely because they are incomplete.
An active branch must still fail normally if it is incomplete or invalid.

The run record must contain at least:

- graph ID and graph revision;
- active branch ID and user-visible branch name;
- dataset fingerprint and partition-manifest fingerprint;
- model architecture summary and training policy;
- validation metric used for selection;
- checkpoint policy and checkpoint used;
- test-evaluation status and final test metrics when run;
- elapsed time, device/backend placement, and run outcome.

## Comparison policy

For model selection, validation/dev performance is the primary rank:

```text
same Train + Dev split
  Model A -> validation result
  Model B -> validation result
  Model C -> validation result
  choose winner -> final protected test evaluation
```

The existing run-comparison ledger should be extended, not replaced. It must
show whether candidate rows share the same dataset and partition fingerprint.
Rows with different data policies are useful historical results but must be
marked `not directly comparable`.

Do not automatically rank candidates primarily by repeated test accuracy. A
test set repeatedly used to choose a winner becomes a development set and
overstates final generalization.

## Interaction with explicit Train / Dev / Test datasets

This ticket depends on the future partition-policy design:

```text
Training source [required]
Dev source      [optional; preserve if supplied]
Test source     [optional; preserve if supplied]
```

All candidate models in a comparison must use the same resolved partition
manifest. For APS, the external test file remains untouched while each model
trains on the same training/dev policy.

## Compatibility and migration

- Existing graphs load with no active selection and keep current behavior only
  until the user opens training; then show `Choose active model branch` if
  more than one valid candidate exists.
- A graph with exactly one valid training branch may be auto-selected once and
  persisted on next save.
- Never delete or rewrite idle links during migration.
- Graph serialization must use stable node IDs, not display names or node
  positions.
- If the selected branch is deleted, clear the selection and block training.

## Non-goals

- Parallel training of A/B/C branches.
- Automatic grid/random/Bayesian hyperparameter search.
- An MLflow/W&B-style persistent experiment platform.
- Sharing weights, gradients, optimizers, or hidden state across candidates.
- Automatically training every visible branch.
- Treating an idle branch as a disabled graph feature for inference or export.
- Replacing the existing run-comparison ledger.

## Risks and safeguards

| Risk | Safeguard |
| --- | --- |
| User mistakes `x` for delete | Label it `Idle`, retain a distinct delete-link action, support undo. |
| Idle invalid branch blocks training | Exclude idle branches before selected-path validation. |
| Two models quietly train | Typed zero-or-one active selection; reject ambiguity. |
| Comparison is scientifically unfair | Record and compare dataset/partition fingerprints. |
| Test data becomes tuning data | Validation-first ranking; explicit repeated-test warning. |
| State grows into an experiment framework | Persist only selection and append to the existing ledger. |

## Acceptance criteria

1. One canvas can contain at least three complete candidate model branches.
2. Exactly one branch is visibly active and trainable at a time.
3. Switching active branch does not alter data links, node settings, or idle
   branch topology.
4. Compiler/runtime receives only the selected branch; incomplete idle
   branches do not block a valid active run.
5. An incomplete active branch blocks launch with a branch-specific error.
6. Each completed run writes branch identity plus data/partition fingerprints
   into the existing comparison ledger.
7. The comparison UI identifies which records are directly comparable and
   ranks candidate selection by validation metric.
8. Test evaluation is visibly distinguished from validation selection.
9. Save/load preserves selection; legacy graphs migrate safely.
10. Existing one-model graphs retain their current train/validation/test
    behavior.

## Test plan

- Graph serialization round-trip for active, idle, and no-selection states.
- Compiler test: only active branch layers, loss, and optimizer are extracted.
- Compiler test: invalid idle branch does not fail a valid selected run.
- Compiler test: invalid active branch fails with its branch name/ID.
- Launch test: branch identity reaches TrainingConfiguration and the run ledger.
- UI test: `Set Active` is exclusive and undoable; normal link deletion remains
  distinct from idling.
- Ledger test: same and mismatched partition fingerprints are classified
  correctly; validation-first ranking is deterministic.
- Regression tests for ordinary one-model graphs and externally supplied APS
  test data.

## Delivery phases

1. Define branch identity, serialized selection schema, migration behavior,
   and compiler selection boundary. No visual `x` yet.
2. Implement compile/launch exclusion plus focused tests.
3. Add minimal canvas badges/context-menu selection and clear launch summary.
4. Add branch/data/partition provenance to the existing run ledger and direct
   comparability indicators.
5. Add the `Idle` link control only after it is proven visually distinct from
   link deletion.
6. Consider sequential `Run selected candidates` only after manual switching,
   cancellation, checkpoint isolation, and comparison semantics are stable.

## Relationship to existing work

- `done12.md` owns the existing lightweight completed-run comparison ledger.
  This ticket extends it; it must not introduce a second ledger.
- The planned explicit Train/Dev/Test partition-policy work owns external
  dataset roles and split manifests.
- Existing graph compiler selected-path behavior is a foundation, but it must
  become explicit and user-controlled for multiple candidate models.
- `BACKEND_EXTERN/` is unrelated: candidate branches may later use different
  backends, but backend selection is not part of this first delivery.
