# Contributing to CyxWiz

Thank you for helping improve CyxWiz. CyxWiz is licensed under the [GNU AGPL v3.0 or later](LICENSE) with a [commercial licence](LICENSE-COMMERCIAL.md) available. Because CYXWIZ offers both, outside contributions need a signed Contributor Licence Agreement that lets CYXWIZ license them under both. The agreement is being prepared; please open an issue before sending a pull request.

Everyone who takes part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Where to start

The contributor documentation is in [docs/internals/](docs/internals/index.md):

- [Architecture overview](docs/internals/architecture/overview.md) and the short map in [ARCHITECTURE.md](ARCHITECTURE.md)
- [Code layout](docs/internals/code-layout.md): what each folder owns and the dependency direction
- [Build from source](docs/internals/build-from-source.md), with the platform details in [INSTALL.md](INSTALL.md)
- [Tests](docs/internals/testing.md) and [the parity oracle](docs/internals/parity-oracle.md)
- [Coding standards](docs/internals/coding-standards.md)
- [Add a built-in node](docs/internals/add-a-node.md) and [Add a panel](docs/internals/add-a-panel.md)
- [Decision records](docs/internals/decisions/README.md)

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

See [Code layout](docs/internals/code-layout.md) for the dependency direction.

## Local setup

Follow [Build from source](docs/internals/build-from-source.md) and [INSTALL.md](INSTALL.md), then configure a Debug build:

```powershell
cmake --preset windows-debug -DCYXWIZ_BUILD_TESTS=ON
cmake --build --preset windows-debug
```

Use the matching Linux or macOS preset on those platforms.

## Engineering standards

The full rules are in [Coding standards](docs/internals/coding-standards.md). In short:

- Use C++20 and preserve the style of the surrounding code.
- Prefer small, explicit interfaces and single-purpose files.
- Reuse existing types, error contracts, task infrastructure, and runtime services.
- Keep UI rendering separate from long-running I/O and computation.
- Treat device selection as an executable contract: surface actual placement, fallback, and failure.
- Validate serialized input and fail with actionable errors.
- Preserve backward compatibility for persisted graphs and protocol messages, or provide an explicit migration.
- Do not add dependencies when a small existing abstraction is sufficient.
- Never commit credentials, private strategy documents, datasets, checkpoints, build outputs, caches, logs, or third-party material without confirmed redistribution rights.

## Documentation

Docs change in the same commit as the behaviour they describe. The docs live in [docs/](docs/README.md), which explains the tree, the page types and the front matter; page templates are in `docs/internals/docs/templates/`, and agents follow the docs skill in `.claude/skills/cyxwiz-docs/SKILL.md`.

- A user-visible change updates its page and adds a release-note fragment in `docs/release-notes/next/`.
- Describe behaviour observed in the current tree. Mark experimental or partial paths plainly.
- Run the check before you hand in a change:

```bash
python scripts/docs/check_docs.py
```

## Tests

Add the smallest test that proves the changed contract. Prefer:

- backend unit tests for computation and data behavior;
- engine tests for graph compilation, materialization, task lifecycle, and persistence;
- protocol compatibility tests for message changes;
- focused integration tests for component boundaries;
- a parity test against the PyTorch or scikit-learn reference for numerical changes.

Run the relevant suite:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

ctest does not register every test executable. After a change to shared code, also run the affected `test_*` executables directly; see [Tests](docs/internals/testing.md).

For device-sensitive work, report the requested backend, resolved backend, physical device, fallback reason, and test evidence. A successful build or GUI preference is not enough to prove GPU execution.

## Change checklist

- [ ] The change has one clear responsibility.
- [ ] Public contracts and persisted formats remain compatible or include migration.
- [ ] Relevant tests pass.
- [ ] Affected targets build.
- [ ] User-visible behavior and limitations are documented truthfully, and `python scripts/docs/check_docs.py` passes.
- [ ] A confirmed bug users can meet (ours or in a dependency such as ArrayFire) that is not
      fixed in this change is listed in [KNOWN_ISSUES.md](KNOWN_ISSUES.md), with a reproducer
      and the upstream report where there is one. Remove the entry when the fix ships.
- [ ] No generated, secret, private, or unlicensed material is included.
- [ ] The diff contains no unrelated formatting or cleanup.

## Commits and pull requests

Write a commit subject that states the outcome, for example `Fix CSV schema promotion during paging` or `fix(patterns): a pattern file opens through the pattern library`. In the pull request include:

- problem and user impact;
- design and affected contracts;
- validation commands and results;
- device/platform coverage;
- remaining limitations or follow-up work.

Reviewers should be able to reproduce the evidence without relying on machine-local files.
