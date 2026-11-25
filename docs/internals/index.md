---
title: Contribute to CyxWiz
description: The entry point for contributors - how the code is laid out and runs, how to build and test it, how to extend it, and how to write its docs.
type: landing
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - architecture/overview.md
  - code-layout.md
  - build-from-source.md
  - testing.md
---

These pages are for people who change CyxWiz itself: the Engine, the
compute library, the protocol, the server node and the plugin host. They
explain how the code works and the rules it follows, so you can make a
change that fits. Start with the overview and the code layout, build it,
run the tests, then pick the how-to for your change.

New to the project? Read [CONTRIBUTING.md](../../CONTRIBUTING.md) at the
repository root for how to propose a change, and the [ecosystem overview](../concepts/ecosystem.md) for what the
parts are.

## How it is built

- [Architecture overview](architecture/overview.md): the system, its
  programs and the Engine's components in C4-style diagrams, and the
  sequence of a training run.
- [Code layout](code-layout.md): every folder, the dependency direction and
  the entry points.
- [Threading and background work](architecture/threading.md): the UI
  thread, the task pool, the training thread, Python, and the
  frozen-snapshot rule.
- [Plugin system internals](architecture/plugin-system.md): discovery,
  checks, permissions, crash isolation, registries and Python node bundles.
- [Compute back ends and neural providers](architecture/compute-back-ends.md):
  ArrayFire, the native providers, and why LibTorch is only the test
  oracle.
- [The protocol](protocol.md): every gRPC service, who serves and calls it,
  and the rules for changing a message.
- [Decision records](decisions/README.md): why things are the way they are.

## Build and test

- [Build CyxWiz from source](build-from-source.md): presets, options, fast
  incremental builds.
- [Tests and checks](testing.md): what exists, how to run all of it, the
  source and docs checks, CI.
- [The parity oracle](parity-oracle.md): the comparison with PyTorch,
  scikit-learn and SciPy, and how to add a case.

## Change it

- [Add a node](add-a-node.md): register a node type, give it pins and
  parameters, test it.
- [Add a panel](add-a-panel.md): a presentation model, a thin ImGui
  renderer, the theme and widget rules.
- [Coding standards](coding-standards.md): C++, Python and GUI rules.
- [Release process](release-process.md): versions, packages, GPU packs and
  release notes.

## Write the docs

The docs live in `docs/` and change in the same commit as the code they
describe. [How the documentation is organised](../README.md) gives the tree,
the front matter and the workflow; the page templates are in
`docs/internals/docs/templates/`:
[tutorial](docs/templates/tutorial.md), [how-to](docs/templates/how-to.md),
[reference](docs/templates/reference.md),
[explanation](docs/templates/explanation.md),
[troubleshooting](docs/templates/troubleshooting.md) and
[release-note fragment](docs/templates/release-note-fragment.md). Run
`python scripts/docs/check_docs.py` before you hand a change in.

## Where to look next

- [The CyxWiz ecosystem](../concepts/ecosystem.md)
- [How graphs run](../concepts/how-graphs-run.md)
- [The data pipeline](../concepts/data-pipeline.md)
- [Security](../security/index.md)
- [Troubleshooting](../troubleshooting/index.md)
- [Error messages](../reference/error-messages.md)
- [Glossary](../glossary.md)
