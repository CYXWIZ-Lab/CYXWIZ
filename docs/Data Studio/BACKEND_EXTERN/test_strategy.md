# BackendExtern Test Strategy

## Test philosophy

The control plane must be testable without installing PyTorch, JAX, Flax, CUDA,
or downloading a real model. Framework tests belong to a small, pinned runtime
test lane. This keeps normal Engine CI fast and prevents external dependencies
from obscuring core regressions.

## Test layers

| Layer | Runs where | Examples |
| --- | --- | --- |
| Pure C++ unit tests | normal Engine CI | manifest parsing, schema, path/hash validation, run state transitions |
| Fake-worker integration tests | normal Engine CI | start/event/result/cancel/crash/timeout scenarios |
| Graph contract tests | normal Engine CI | node pins, missing provider, fail-closed materialization, result type checks |
| Managed-runtime smoke | dedicated pinned environment | worker health, one CPU inference fixture |
| GPU compatibility lane | dedicated hardware | one supported runtime/framework/device tuple, diagnostics, containment |
| Release acceptance | release candidate | install/upgrade/uninstall, license/SBOM, support-bundle redaction |

## Required fake-worker cases

- healthy startup followed by valid result;
- unknown protocol major;
- malformed event and malformed result;
- worker exits before response;
- worker ignores cancellation;
- worker returns wrong result hash/schema;
- timeout boundary;
- Engine shutdown while run is active;
- concurrent run request policy (explicitly reject or queue; decide before code).

## Regression gates

- Existing native GraphCompiler, materializer, executor, and ArrayFire tests
  pass unchanged.
- BackendExtern unavailable or disabled still permits all existing native use.
- An external failure never changes a native backend placement record.
- A saved graph with an unavailable external node stays inspectable and cannot
  report a false successful execution.

