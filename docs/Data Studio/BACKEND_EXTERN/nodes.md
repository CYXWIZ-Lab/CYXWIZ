# Proposed CyxWiz Nodes and UI Surfaces

## Node design rule

Nodes express stable user concepts. They must not expose raw framework objects,
Python code, checkpoint directories, or framework-specific tensor pointers.

## Initial nodes

### External Model Runtime

An optional configuration/source node that resolves a named managed runtime.
It may be implicit in a curated model node for the first release.

Inputs: none.

Outputs: `ExternalRuntimeHandle` (internal control type only).

Properties: runtime ID, device policy, health state, support date.

### External Model Predict

Generic internal executor node, not initially exposed as a broad user palette
node. It binds an approved provider/model adapter to typed input data.

Inputs: table/tensor data, optional labels/context, optional runtime handle.

Outputs: predictions, probabilities/scores, run report.

Properties: provider ID, operation, model revision, device policy, timeout.

### TabFM Zero-Shot

The first user-facing curated node, only after commercial licensing allows its
selected model artifacts. Inputs are a Data Studio table and target/context
selection. Outputs are predictions, probabilities where supported, and a run
report. It presents `PyTorch` or `JAX/Flax` as implementation choices but
defaults to `Auto` based on the pinned runtime policy.

### External Run Report

An inspectable artifact node/panel providing runtime ID, provider/framework
versions, model revision/license classification, device, input/result hashes,
duration, warnings, and error codes. This is important for trustworthy UX and
support bundles.

## Compiler/materializer rules

- Native nodes keep their current compile/materialize route.
- An external node compiles only when its provider schema is installed and its
  pins/parameters validate.
- Missing runtime is a clear `install-needed` preflight state, not a fallback
  to a native node.
- An unsupported external node in a saved graph remains visible with its
  properties and fails closed; it is never silently deleted or converted.
- External outputs are reintroduced as normal typed CyxWiz table/tensor/scalar
  artifacts only after result validation.

## UI surfaces

1. Data Studio model chooser: native, managed external, user script categories.
2. Runtime Manager: installed runtimes, size, versions, health, device report,
   support end date, uninstall action.
3. Model provenance view: source/revision/license/terms status and cache data.
4. Task/run view: progress, logs, cancellation, and result report.

