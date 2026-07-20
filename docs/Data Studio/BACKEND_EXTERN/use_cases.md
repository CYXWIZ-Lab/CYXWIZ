# BackendExtern Use Cases

## UC-1: Curated zero-shot tabular inference

A data analyst loads a CSV in Data Studio, selects an approved external tabular
model, maps the target column, and runs inference. The graph returns
predictions and a report showing the exact runtime/model provenance.

Value: a visual workflow gains a strong pre-trained model without claiming that
CyxWiz trained or natively executed it.

## UC-2: Engineer brings a project script

An engineer uses the existing Script Editor and project Python venv to run a
custom PyTorch experiment. CyxWiz captures output/plots as it does today.

Value: no new provider implementation is required. Limitation: dependencies,
model correctness, and reproducibility are user-managed unless the engineer
creates a managed provider.

## UC-3: Repeat a regulated analysis

A project records a supported runtime ID, exact model revision, artifact
hashes, device policy, and result schema. A colleague opens it later and is
either able to install the same runtime or receives a deliberate blocked state
with a migration path.

Value: better auditability than a notebook that depends on an undocumented
environment.

## UC-4: Framework failure while native work remains available

A JAX worker fails during GPU initialization. The external node reports the
failure and offers CPU/repair guidance. The Engine and ArrayFire-native nodes
remain available.

Value: external capability is additive, not a single point of failure.

## Not a first-release use case

Training arbitrary user PyTorch/JAX models from native graph nodes is deferred.
It requires a separate training, checkpoint, optimizer, and artifact contract;
it must not be smuggled into an inference integration.

