# To Fix 78 - Non-Tabular Dataset Roles and Typed Partition Contracts

## Status

Open - Track70 follow-up. Current Track70 role guarantees are proven for
Arrow/Parquet tabular datasets; this ticket extends the contract by modality.

## Decision statement

Make Train, Dev/Validation, Test, and Inference roles operate on a typed
dataset-partition interface that does not assume an Arrow table or a label
column.

Do not convert every modality into tabular Arrow merely to reuse the current
role resolver. Each adapter keeps its natural sample representation while
sharing role identity, partition provenance, and ownership rules.

## Shared role contract

Every supported dataset adapter provides:

```text
DatasetDescriptor
  dataset_id
  modality
  sample_count
  schema_or_sample_contract
  target_contract
  source_fingerprint

PartitionDescriptor
  role
  origin                supplied | derived
  dataset_id
  selection_fingerprint
  sample_count
```

The shared resolver decides which source owns each role and whether Train may
be split. Modality adapters implement deterministic selection and batch
construction.

## Required semantics

- Train only: derive Train/Dev/Test according to the configured policy.
- Train plus external Test: derive only Train/Dev; preserve all supplied Test
  samples.
- Train plus external Dev and Test: preserve both and use all Train samples for
  fitting.
- Dev/Test/Inference sources cannot be used for fitting preprocessing or
  balancing.
- Split, shuffle, balance, augmentation, and drop-last apply only where the
  typed policy permits them.
- Prefetch wrappers own or retain their actual modality source; replacing a
  role cannot leave a wrapper around a destroyed source.
- Target contracts may be column-based, structured, graph-generated, or absent.

## Modality slices

Implement one adapter at a time behind the shared contract:

1. image classification folders/manifests;
2. text/sequence datasets;
3. audio datasets;
4. tensor/HDF5 datasets when To Fix 72 provides the supported source slice;
5. time-series sources whose windows generate targets.

The first slice should be the non-tabular adapter with the strongest existing
batcher and test foundation. Do not modify all adapters in one change.

## Compiler and UI behavior

- Data Input role help is generated from the selected modality and objective.
- The compiler reports role, source, sample count, target origin, and partition
  origin without using legacy `label column` language where it does not apply.
- Unsupported role/modality combinations fail before materialization.
- The Training Dashboard and run manifest display the effective role mapping.

## Acceptance criteria

- At least one non-tabular modality passes the same Train-only, external-Test,
  and external-Dev/Test role matrix already proven for Arrow/Parquet.
- Supplied Test samples are consumed exactly once and in full.
- Deterministic split fingerprints survive restart and are recorded in run
  history.
- Prefetch lifetime tests consume batches after role resolution without stale
  ownership.
- Role-specific preprocessing and augmentation cannot fit or mutate state on
  evaluation sources.
- Existing tabular role tests remain unchanged and pass.

## Non-goals

- a single monolithic dataset class for every modality;
- requiring labels for target-free or environment objectives;
- implicit joins between unrelated sources;
- adding HDF5 layouts beyond the explicit To Fix 72 support contract.
