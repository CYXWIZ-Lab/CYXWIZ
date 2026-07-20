# To Fix 66 - Hugging Face Open-Weight Import and Framework Bridge Plugins

## Status

Open — design and staged implementation ticket.

## Goal

Let a CyxWiz user discover and download permitted open-weight model artifacts
from the Hugging Face Hub, then use them only through an explicit, truthful
compatibility path. Add optional PyTorch, JAX, and Flax bridge plugins without
making any of those large framework runtimes a required CyxWiz engine
dependency.

This ticket is about **download, inspection, and controlled interchange**. It
does not claim that arbitrary Hub repositories can run in CyxWiz.

## Why weights alone are not enough

A model repository normally contains more than tensor values:

- architecture/configuration files (for example `config.json`);
- tensor shards and an index/manifest;
- tokenizer and vocabulary assets;
- generation configuration or processor assets;
- model-specific Python code in some repositories;
- a license, gated-access terms, and an exact revision.

CyxWiz must never infer an architecture merely from tensor names or load a
repository's Python code automatically. A model becomes usable only when a
known adapter validates the architecture, tensor names, shapes, dtypes,
tokenizer contract, and supported task.

## Essential first capability

The smallest useful first slice is a **Hugging Face Hub Import** plugin that:

1. accepts an explicit repository ID and optional immutable revision/commit;
2. shows license, gating, and file metadata before download;
3. downloads selected declared artifacts into a CyxWiz-managed cache;
4. records provenance and integrity information in a local import manifest;
5. inspects `safetensors` and supported configuration files without executing
   remote code;
6. either produces a clear “not supported by CyxWiz yet” result or hands a
   validated package to a registered model adapter.

Start with public repositories and `safetensors`. Do not begin with pickle
based `.bin`, `.pt`, or `.pth` files, `trust_remote_code`, automatic model
conversion, training, or arbitrary repository scripts.

## Proposed architecture

```
Studio model browser
        |
        v
huggingface-hub plugin -- manifest/provenance --> local model cache
        |                                               |
        | supported files only                           v
        +------------------------------------------ model adapter registry
                                                        |
                                                        +--> native .cyxmodel
                                                        +--> optional framework bridge
```

The engine owns only stable, framework-neutral concepts:

- `ModelArtifact`: local path, size, SHA-256, format, repository/revision, and
  declared license information;
- `ModelDescriptor`: architecture family, task, input/output contract,
  tokenizer requirement, and adapter ID;
- `ModelImportResult`: imported, unsupported, needs-authentication, rejected,
  or failed, with a user-readable reason;
- an adapter registry that chooses a named adapter only after validation.

The Hub client, network credentials, parsing of framework-specific artifacts,
and framework runtimes remain inside plugins.

## Download and cache contract

### Repository selection

- Require `owner/model` and default to a resolved commit hash, not a floating
  branch, once the user confirms an import.
- Display repository ID, resolved revision, license field when present, gated
  status, file list, total selected size, and requested permissions.
- Gated/private access uses a user-provided Hugging Face token held by the
  platform credential store. Never put tokens in graph files, `.cyxmodel`,
  logs, support bundles, or plugin manifests.
- The user accepts any repository terms in their own Hugging Face account; the
  plugin must not bypass gating or license restrictions.

### Cache layout and provenance

Store files below a dedicated cache root, keyed by repository and resolved
revision. Keep an adjacent CyxWiz-owned JSON manifest containing:

```json
{
  "source": "huggingface",
  "repository": "org/model",
  "revision": "resolved-commit-hash",
  "downloaded_at": "ISO-8601 timestamp",
  "files": [{"path": "model.safetensors", "sha256": "...", "bytes": 0}],
  "license": "declared license or unknown",
  "adapter": "none",
  "status": "downloaded-not-imported"
}
```

Use atomic temporary downloads, checksum verification when the Hub metadata
provides a digest, size limits, cancellation, and resumable transfers. The
cache must be user-visible and removable through CyxWiz cache management; it
must not silently modify the source repository.

### Safe parsing policy

- Prefer `safetensors`, whose tensor metadata can be inspected without
  deserializing executable Python objects.
- Treat pickle-backed PyTorch checkpoints as untrusted data and reject them in
  the first slice.
- Never enable `trust_remote_code` or import repository Python implicitly.
- Parse JSON only through bounded, schema-validated readers; impose file-size,
  tensor-count, rank, dtype, and allocation limits before materializing data.

## Import compatibility policy

Downloading succeeds independently from importing. The UI must distinguish:

| State | Meaning |
| --- | --- |
| Downloaded | Artifacts and provenance are cached; no execution claim. |
| Inspectable | Files passed safe format/schema checks. |
| Supported | A specific CyxWiz adapter recognized and validated the contract. |
| Imported | A native or bridge-owned runnable package was created. |
| Unsupported | The repository is retained in cache but cannot run; show the missing adapter/feature. |

The first adapters should target a very small, explicit set of architectures
already represented and tested by the engine. Each adapter must validate every
required tensor name, shape, dtype, layer configuration, tokenizer asset, and
output contract before conversion. Partial imports are failures, not best-effort
loads. Unsupported architecture fields fail closed with precise diagnostics.

When an adapter creates `.cyxmodel`, embed the immutable source provenance and
adapter version, but do not copy access tokens. Do not market an imported model
as equivalent to the original framework model unless output parity tests prove
the stated task and configuration.

## Optional framework bridge plugins

### Shared boundary

Implement each bridge as a separate plugin with its own manifest, isolated
Python environment, capability declaration, and versioned RPC/JSON contract.
The engine passes artifact paths plus a validated `ModelDescriptor`; it does not
link to framework headers or expose internal tensor memory to plugin code.

Each plugin may provide only these initial operations:

- inspect a locally cached artifact;
- report installed framework, version, device availability, and supported
  formats;
- run a declared, bounded inference probe with explicit input/output tensors;
- export a portable interchange artifact when the plugin can prove it.

Long-running framework work runs in a plugin worker process. A crash, CUDA
failure, or dependency conflict must fail that operation and leave the engine
alive. Do not make framework plugins per-frame callbacks.

### PyTorch bridge (`cyxwiz.pytorch`)

Purpose: inspect PyTorch-compatible artifacts and run explicitly supported
TorchScript or locally defined, allow-listed model adapters. It may also serve
as an oracle during development, but numerical-reference tests stay outside the
production runtime.

First supported transport: TorchScript or a tested export path. Raw Python
state dictionaries require a known architecture adapter and must not be loaded
with unsafe pickle defaults. GPU selection, dtype, and device fallback are
reported as operation metadata.

### JAX bridge (`cyxwiz.jax`)

Purpose: execute JAX-side probes or export supported JAX parameter pytrees only
through explicit serialization adapters. JAX is a numerical runtime, not by
itself a standard model-package format, so the plugin must declare the accepted
serialization format and model family rather than accepting a generic
“JAX checkpoint.”

The plugin owns JAX/JAXLIB and accelerator compatibility. It must report CPU,
CUDA, or TPU availability without changing the CyxWiz engine compute backend.

### Flax bridge (`cyxwiz.flax`)

Purpose: add Flax model/module and parameter-tree knowledge on top of the JAX
bridge. Flax is not an independent tensor runtime: its manifest depends on a
compatible JAX bridge version, but it remains a separate optional plugin so
users who only need JAX do not install Flax.

The first Flax adapter supports only a pinned, documented checkpoint format and
an allow-listed module/configuration pair. It must reject unknown collection
names, mutable state, custom modules, and remote code.

## Plugin manifests and permissions

All four plugins use the existing CyxWiz plugin lifecycle, but need narrow
additional capabilities:

- `network.model_repository` — Hub metadata and artifact downloads;
- `filesystem.model_cache` — read/write only within the managed cache;
- `process.framework_worker` — launch the plugin's isolated worker;
- `compute.gpu` — optional, user-approved accelerator access.

The Hub plugin needs network and cache permissions. Framework bridges need a
worker and cache permission; GPU access is optional. No plugin receives broad
filesystem, shell, source-code execution, or engine-internal pointer access.

Suggested plugin layout:

```
plugins/
  huggingface_hub/
    plugin.json
    worker/
  pytorch_bridge/
    plugin.json
    worker/
  jax_bridge/
    plugin.json
    worker/
  flax_bridge/
    plugin.json              # depends on cyxwiz.jax
    worker/
```

Use one narrowly versioned request/response schema shared by workers. Do not
create a generic “execute arbitrary Python” plugin API.

## Delivery order

1. Define engine-neutral artifact, descriptor, import-result, and adapter
   registry interfaces with unit tests. No network or framework dependency.
2. Implement the Hub plugin for metadata, authenticated public/gated download,
   deterministic cache/provenance, cancellation, and `safetensors` inspection.
3. Add one native adapter for a small already-tested model family and prove
   deterministic output parity with a pinned source revision.
4. Add the PyTorch worker bridge with an inference-probe contract and crash/
   timeout containment.
5. Add the JAX bridge with one explicit serialization adapter.
6. Add the Flax bridge as a dependent plugin with one allow-listed Flax
   checkpoint/module pair.
7. Add Studio flows only after the underlying results and failure states are
   available through a headless API.

## Validation and acceptance criteria

- A Hub download records the repository, resolved revision, selected files,
  hashes, size, license metadata, and adapter result.
- Cancellation and interrupted downloads leave no artifact presented as valid.
- Gated/private tokens are absent from logs, exported packages, diagnostics,
  and graph files.
- Invalid JSON, unsafe pickle formats, malformed safetensors, unbounded
  metadata, and unknown architectures are rejected safely and explain why.
- A supported adapter validates all required artifacts and matches pinned
  reference outputs within stated tolerances before it is exposed as supported.
- Each installed framework bridge reports its own version/device capabilities;
  missing frameworks leave CyxWiz usable and produce a clear install-needed
  result.
- Worker timeout/crash and CUDA failures do not crash the engine or corrupt the
  cache.
- Tests cover cache/provenance, permission denial, revision pinning, adapter
  mismatch, and every declared supported model path.

## Non-goals

- Do not bundle PyTorch, JAX, Flax, or their GPU runtimes into the core engine.
- Do not support every Hugging Face architecture, pipeline, or checkpoint.
- Do not execute Hub repository code, enable `trust_remote_code`, or accept
  arbitrary Python callbacks.
- Do not treat a successful download as a successful CyxWiz model import.
- Do not promise training/fine-tuning through framework plugins in the first
  implementation.
- Do not replace the native CyxWiz model/runtime path with framework bridges.

## Relationship to existing work

- `tofix28.md` establishes the rule that PyTorch is a reference/oracle, not a
  required runtime dependency.
- `tofix51.md` explicitly deferred Hugging Face checkpoint loading; this ticket
  owns that deferred work.
- The existing plugin architecture supplies lifecycle and permission concepts;
  this ticket narrows the model-artifact and isolated-worker contracts needed
  for framework integrations.
