# Contributing to CyxWiz

Thank you for helping improve CyxWiz. This is a source-available project; contributions and use remain subject to [LICENSE](LICENSE).

## Before starting

For a substantial change, open an issue describing the user problem, affected component, proposed contract, and validation plan. Security vulnerabilities must follow [SECURITY.md](SECURITY.md), not a public issue.

Keep each change focused. Avoid combining cleanup, behavior changes, generated files, and dependency upgrades in one review.

## Component ownership

- Shared computation, tensor, model, layer, loss, optimizer, data, and device behavior belongs in `cyxwiz-backend/`.
- Desktop interaction and presentation belong in `cyxwiz-engine/`.
- Cross-process messages belong in `cyxwiz-protocol/`.
- Worker process behavior belongs in `cyxwiz-server-node/`.
- Optional integrations belong in `plugins/` and must fail closed when unavailable.

Do not duplicate backend algorithms in the GUI. Extend the narrowest existing public contract and update every consumer.

See [docs/project-structure.md](docs/project-structure.md) for the dependency direction.

## Local setup

Follow [INSTALL.md](INSTALL.md), then configure a Debug build:

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
```

Use the matching Linux or macOS preset on those platforms.

## Engineering standards

- Use C++20 and preserve the style of the surrounding code.
- Prefer small, explicit interfaces and single-purpose files.
- Reuse existing types, error contracts, task infrastructure, and runtime services.
- Keep UI rendering separate from long-running I/O and computation.
- Treat device selection as an executable contract: surface actual placement, fallback, and failure.
- Validate serialized input and fail with actionable errors.
- Preserve backward compatibility for persisted graphs and protocol messages, or provide an explicit migration.
- Do not add dependencies when a small existing abstraction is sufficient.
- Never commit credentials, private strategy documents, datasets, checkpoints, build outputs, caches, logs, or third-party material without confirmed redistribution rights.

Public documentation must describe behavior observed in the current tree. Mark experimental or partial paths plainly.

## Tests

Add the smallest test that proves the changed contract. Prefer:

- backend unit tests for computation and data behavior;
- engine tests for graph compilation, materialization, task lifecycle, and persistence;
- protocol compatibility tests for message changes;
- focused integration tests for component boundaries.

Run the relevant suite:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

The Windows Catch2 binary is normally `build/bin/Debug/cyxwiz-tests.exe`.

For device-sensitive work, report the requested backend, resolved backend, physical device, fallback reason, and test evidence. A successful build or GUI preference is not enough to prove GPU execution.

## Change checklist

- [ ] The change has one clear responsibility.
- [ ] Public contracts and persisted formats remain compatible or include migration.
- [ ] Relevant tests pass.
- [ ] Affected targets build.
- [ ] User-visible behavior and limitations are documented truthfully.
- [ ] No generated, secret, private, or unlicensed material is included.
- [ ] The diff contains no unrelated formatting or cleanup.

## Commits and pull requests

Write an imperative commit subject that states the outcome, for example `Fix CSV schema promotion during paging`. In the pull request include:

- problem and user impact;
- design and affected contracts;
- validation commands and results;
- device/platform coverage;
- remaining limitations or follow-up work.

Reviewers should be able to reproduce the evidence without relying on machine-local files.
