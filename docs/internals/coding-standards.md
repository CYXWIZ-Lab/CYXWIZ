---
title: Coding standards
description: The C++, Python and GUI rules CyxWiz code follows, the checks that enforce them, and how to write commits.
type: reference
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - add-a-node.md
  - add-a-panel.md
  - testing.md
  - code-layout.md
  - decisions/README.md
---

The rules every change to the CyxWiz Engine, backend, Server Node and
plugins follows. Reviewers check them; the checks listed under each rule
enforce what can be enforced. Where this page and the surrounding code
disagree, follow this page and fix the code in its own commit.

## Language and build

| Rule | Detail |
| --- | --- |
| C++20 | `CMAKE_CXX_STANDARD 20` for every target. |
| Warnings | MSVC `/W4`, GCC and Clang `-Wall -Wextra -Wpedantic`. CyxWiz-owned source has zero warnings on Windows, Linux and macOS; the compiler warning gate in CI fails on any new one (`scripts/compiler_warnings/`, baseline in `baseline.json`). Silence a warning in third-party headers with a scoped pragma around the include, never by lowering the flags. |
| Cross-platform | Every change builds on Windows (MSVC), Linux (GCC) and macOS (Apple Clang). Use `std::filesystem::path` and forward slashes, never hard-coded drive letters or separators. Guard platform code with `_WIN32`, `__APPLE__` and `__linux__`. |
| Dependencies | Through vcpkg (`vcpkg.json`, overlay ports in `vcpkg-ports/`) and ArrayFire. Do not add a dependency when a small existing abstraction does the job. |

## Naming and layout

As the code is written today:

| Item | Style | Example |
| --- | --- | --- |
| Files | `snake_case.cpp` / `.h`, one concern per file | `graph_node_factory.cpp` |
| Classes, structs, enums | `PascalCase` | `NodeMetadataRegistry`, `enum class PipelineBand` |
| Functions and methods | `PascalCase` | `RegisterNode`, `GetAllMetadata` |
| Local variables, parameters | `snake_case` | `next_pin_id` |
| Members | `snake_case_` with a trailing underscore | `metadata_`, `visible_` |
| Constants | `kPascalCase` | `kNodeTypeImportNames` |
| Namespaces | `cyxwiz` (core and backend), `cyxwiz::ui` (widgets), `gui` (editor model) | |
| Headers | `#pragma once` | |
| Indent | 4 spaces, no tabs | |
| Logging | `spdlog` | `spdlog::info("{}: ...", name)` |

Large classes are split across files by concern, with the class name as a
prefix: `node_editor_io.cpp`, `node_editor_nodes.cpp`,
`main_window_shortcuts.cpp`. Files that are part of the plugin SDK start
with `// SPDX-License-Identifier: Apache-2.0`.

## Design rules

- **Find before you write.** Search for an existing implementation before a
  fix or a feature; extend the narrowest existing contract and update every
  consumer.
- **Compute belongs in the backend.** Tensor, model, data and device code
  lives in `cyxwiz-backend`; the Engine translates graphs into backend
  calls and never keeps a second copy of an algorithm.
- **GPU first, written once.** Compute is written on ArrayFire and stays
  device resident. The CPU fallback is ArrayFire's CPU back end, not a
  hand-written CPU loop. See [Decision: ArrayFire for compute](decisions/0001-arrayfire-for-compute.md).
- **Avoid known ArrayFire traps.** On oneAPI, indexed `af::min` and `af::max`
  along dimension 1 are wrong; use the helpers in
  `cyxwiz-backend/src/algorithms/arrayfire_indexed.h`. Draw random numbers
  through `cyxwiz/portable_random.h`, never `std::shuffle` or
  `std::*_distribution` (`scripts/checks/check_random_sources.py`, ctest
  `random_sources_check`). See [Seeds and reproducibility](../concepts/seeds-and-reproducibility.md).
- **Nodes are real operations.** Data flows through a node; a node is never
  a bag of settings the compiler scans for. See
  [Add a built-in node](add-a-node.md).
- **The truth on screen.** Properties, the compile report and logs say what
  will actually run, including the device and any fallback. A fallback is
  never silent.
- **Fail closed, with the reason.** When something cannot run, refuse and
  say why and what to do, rather than run something else.
- **No orphan code.** A change that makes a function, panel, route or file
  unused deletes it in the same change.
- **No legacy loader shims.** Old graphs are rebuilt on the current
  contract; the loader refuses a retired node with the name of its
  replacement instead of translating it. See
  [Decision: no legacy loader shims](decisions/0004-no-legacy-loader-shims.md).
- **Smart defaults over knobs.** Detect good defaults and keep them; add a
  setting only when users need it.
- **Fix what you find.** A bug spotted on the way is fixed in its own commit,
  not only logged.

## Threads and processes

- The UI thread draws; it never blocks on I/O or compute. Long work runs as
  an `AsyncTaskManager` task on a snapshot taken on the UI thread, and its
  result is drained on a later frame. See
  [Threading and async tasks](architecture/threading.md).
- The Engine never calls `system()` or `_popen()`: on Windows each call
  flashes a console window. Run a helper program with
  `cyxwiz::core::RunHidden` (`cyxwiz-engine/src/core/child_process.h`),
  which uses no shell and no console window.

## GUI rules

- **The theme is the only colour source.** Read colours from
  `cyxwiz::ui::CurrentTokens()` (`gui/ui_tokens.h`); never type a colour
  literal or add a second palette. Status words, colours and icons come from
  `StatusStyleFor()`.
- **Shared widgets only.** Buttons are the `cyxwiz::ui` buttons
  (`gui/ui_buttons.h`), never a default `ImGui::Button`. A disabled control
  explains why in its tooltip.
- **Typed number fields.** Numbers are typed with - / + steps
  (`ui::NumberField`), never dragged or slid.
- **Few lines.** No outlines on fields or buttons; the window has the only
  border. Borders and scrollbars use the surface tone.
- **Nothing hidden or clipped.** Toolbars wrap (`ui::FlowRow`); a window at
  its smallest size still shows every button.
- **Saved defaults.** Every option a user can change has a saved default in
  Preferences.
- **Words are tested.** A screen's wording and states come from a
  presentation model in `core/*_presentation.*` with its own test. See
  [Add a panel](add-a-panel.md).
- **Icons.** Every icon macro must draw the glyph it names
  (`scripts/checks/check_icon_headers.py`, ctest `icon_headers_check`).

## Strings and comments

- User-facing strings say what happened and what to do, in plain words.
- Internal issue numbers, private paths, machine names and addresses never
  appear in string literals (UI text, log messages, test messages). They
  may appear in comments only. `scripts/check_private_details.py` checks
  tracked files against a private, git-ignored list of patterns.
- Comments explain why, not what. Keep them true when the code changes.

## Python

- Release and docs tooling (`redist/scripts/`, `scripts/docs/`) uses the
  standard library only, so it runs on any Python 3 without installing
  packages.
- The `pycyxwiz` bindings are pybind11 code in `cyxwiz-backend/python/`;
  their docstrings are the Python API reference, which is generated from
  them.
- Example scripts in `examples/` run as shown and use committed data or say
  how to get it.

## Tests

A change adds the smallest test that proves its contract, and a numerical
change is checked against the PyTorch or scikit-learn reference. ctest does
not run every test executable; run the affected `test_*` executables
directly. See [Tests](testing.md) and [The parity oracle](parity-oracle.md).

## Commits

- One responsibility per commit; no unrelated formatting.
- The subject states the outcome, in the imperative or as a fact, with an
  optional area prefix: `fix(patterns): a pattern file opens through the
  pattern library`, `docs: keyboard shortcuts reference`.
- Docs change in the same commit as the behaviour, with a release-note
  fragment; see [How the docs are organised](../README.md).
- Commit only the files you mean to (explicit paths), never secrets, build
  outputs, datasets, logs or private material.

## Related

- [Decision records](decisions/README.md)
- [Code layout](code-layout.md)
- [Tests](testing.md)
