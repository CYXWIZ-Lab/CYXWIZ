# LTS Operations and Support Policy

## Support units

The unit of support is a tuple, not a framework name:

```text
(Engine release, protocol major, runtime ID, provider ID, model revision,
 platform, device class)
```

For example, support `cyxwiz-tabular-2026.1` on Windows x64 CPU before claiming
that "CyxWiz supports JAX".

## Runtime lifecycle

| State | Meaning |
| --- | --- |
| Available | Catalogued, installable, not necessarily installed |
| Installed | Files/lock verified |
| Healthy | Worker protocol and minimum dependency probe pass |
| Supported | Within LTS window and tested platform/device tuple |
| Deprecated | Still runnable, no new features |
| Retired | Kept only where legally/technically feasible; migration required |

Do not mutate an installed runtime in place. Security patches produce a new
runtime revision with a new immutable identity, while affected users receive a
clear advisory and upgrade path.

## Compatibility matrix

Publish and test a matrix for only the combinations the product promises:

```text
Windows x64 / CPU / runtime 2026.1 / provider X
Windows x64 / NVIDIA CUDA / runtime 2026.1 / provider X
```

Do not infer macOS, Linux, AMD, TPU, or multi-GPU support from framework
marketing. Add a row only when CI/hardware validation and support ownership
exist.

## Project persistence

Projects persist runtime ID, provider ID/version, model source/revision,
artifact hashes, device policy, and input/output schema. They never persist
access tokens, unrestricted local paths, or unvalidated framework objects.

## Diagnostics and support bundle

Capture Engine/protocol/runtime/provider/framework versions, device report,
worker exit code, sanitized logs, model provenance, and result schema. Redact
tokens, cookie values, environment secrets, and user table contents by default.

## Upgrade policy

- New project: choose current supported runtime.
- Existing project: continue exact pinned runtime.
- Upgrade: explicit project migration with compatibility probe and rollback.
- Unsupported old runtime: state the reason and offer documented migration; do
  not silently run the project on latest dependencies.

