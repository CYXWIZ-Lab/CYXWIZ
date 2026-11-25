---
title: Tests and checks
description: The kinds of tests in the repository, where they live, how to run all of them, and the source and docs checks that guard a change.
type: explanation
area: internals
audience: contributor
status: stable
since: 1.0.7
owner: internals
last_reviewed: 2026-11-25
related:
  - parity-oracle.md
  - build-from-source.md
  - coding-standards.md
  - add-a-node.md
---

A change is tested when the smallest test that proves its contract passes,
on the devices it claims. This page lists what exists and how to run it. The
numerical side, the comparison with PyTorch and scikit-learn, has its own
page: [The parity oracle](parity-oracle.md).

## What kinds of tests there are

| Kind | Where | What it proves |
| --- | --- | --- |
| Backend unit tests | `tests/unit/`, one Catch2 executable `cyxwiz-tests` | Tensors, layers, losses, devices, the C API, the `.cyxmodel` format |
| Device smoke probes | `tests/smoke/` (`cyxwiz-route-probe`, `cyxwiz-device-inventory`) | A route activates and runs the operations it claims |
| Engine contract tests | `cyxwiz-engine/tests/test_*.cpp`, one executable each | Graph compile, pipeline runs, batchers, training lifecycle, persistence, tasks, presentation models |
| Computation truth and oracle | `cyxwiz-engine/tests/computation_truth/`, `tests/oracle/` | Results match PyTorch, scikit-learn or SciPy fixtures ([details](parity-oracle.md)) |
| Node contract gates | `test_node_contract_gates` (`tests/node_contract_inventory.cpp`) | Every registered node has pins, parameters, an implementation status and a workflow it is tested in |
| Example graphs | `test_example_graphs_compile` | Every committed example graph compiles |
| Python tests | `cyxwiz-backend/python/tests/`, `cyxwiz-engine/python_tools/tests/` | `pycyxwiz` modules and nodes; the Engine's Python tools |
| Plugin checks | `cyxwiz-plugin-check`, `plugins/*/tests/` | A plugin folder opens and declares the permissions it uses; plugin behaviour |
| Server node tests | `cyxwiz-server-node/tests/` | Job execution service, dataset fetchers, security, sign-in presentation, IPC reconnect |
| Release packaging | `redist/tests/` (Python `unittest`) | Package layout, GPU pack contracts, installer bundles, runtime closure |
| Benchmarks | `tests/benchmarks/`, `cyxwiz-bench-*`, `cyxwiz-training-profile` | Speed; not pass or fail |

Screens are tested through their presentation models: a plain C++ model
(`core/*_presentation.{h,cpp}`) holds the text, state and rules of a panel,
and a `test_*_presentation` executable checks it without ImGui. See
[Add a panel](add-a-panel.md).

## Run the tests

Build with `-DCYXWIZ_BUILD_TESTS=ON` (the default), then from the build
tree:

```powershell
ctest --test-dir build -C Debug --output-on-failure
```

On Linux and macOS drop `-C Debug`. Labels select a group, for example
`ctest -L oracle` for the oracle tests.

**ctest does not run every test.** Each Engine test is its own executable,
and only part of them are registered with `add_test`. After a change to
shared code, run every built `test_*` executable directly as well:

```powershell
Get-ChildItem build\bin\Debug\test_*.exe | ForEach-Object {
  & $_.FullName
  if ($LASTEXITCODE -ne 0) { "FAILED: $($_.Name)" }
}
```

```bash
for t in build/linux-debug/bin/test_*; do "$t" || echo "FAILED: $t"; done
```

A few executables take arguments that ctest passes (for example
`test_transformer_dropout` takes a device and an output file); run those
through ctest.

Other suites:

```bash
python -m unittest redist.tests.test_package_release   # release packaging, one module at a time
python scripts/checks/check_random_sources.py .    # also ctest random_sources_check
```

## Devices

A test proves only the devices it ran on. Pick the ArrayFire back end with
`CYXWIZ_TEST_ARRAYFIRE_BACKEND=cpu|cuda|opencl|oneapi` (and
`CYXWIZ_OPENCL_TEST_DEVICE=N` for an OpenCL device). Oracle tests print a
`device:` line first; check it before trusting a result. An unavailable GPU
route is a skip, never a pass on another device. When you report results
for device-sensitive work, give the requested back end, the resolved back
end, the physical device and any fallback reason.

## Source checks

| Check | Runs as | Catches |
| --- | --- | --- |
| `scripts/checks/check_random_sources.py` | ctest `random_sources_check` | `std::shuffle`, `std::*_distribution` and non-Philox ArrayFire engines, which differ between compilers or devices |
| `scripts/checks/check_icon_headers.py` | ctest `icon_headers_check` | An icon macro that does not draw the glyph its comment names |
| `scripts/checks/check_oracle_coverage.py`, `check_oracle_fixtures.py` | ctest `oracle_coverage`, `oracle_fixture_manifest` | A public component with no oracle row; a fixture changed by hand |
| `scripts/compiler_warnings/report.py` | CI | New compiler warnings against `baseline.json` |
| `scripts/check_private_details.py` | by hand before a push | Private details (addresses, machine names) from a local pattern list |

## Docs checks

| Check | Runs as | Catches |
| --- | --- | --- |
| `scripts/docs/check_docs.py` | by hand and in review | Missing or invalid front matter in the docs tree, broken internal links, ticket numbers, local paths and addresses in published pages |
| `scripts/docs/docs_drift.py` | by hand | Commits since each area's docs were last checked (`docs/sync.yaml`) |
| `scripts/docs/generate_extension_reference.py --check` | ctest `extension_docs_reference` | The generated Python and SDK reference out of date with the code |
| `scripts/docs/check_extension_docs.py` | ctest `extension_docs_samples` | Plugin and Python-node guide samples that no longer match their source |

## Continuous integration

The workflows are in `.github/workflows/`:

- `oracle-cpu.yml`, on every pull request and every push to the integration
  branch: builds the oracle targets and runs `ctest -L oracle` on ArrayFire
  CPU, on Windows and Linux, from the committed fixtures (no GPU, no
  PyTorch).
- `cpu-base-native.yml`, `cpu-base-posix.yml`: build and check the CPU base
  packages.
- `compiler-warnings.yml`: the warning report.
- `installer-*.yml`, `release.yml`: installer candidates and releases (see
  [Release process](release-process.md)).

GPU routes and live PyTorch runs are not in CI. They run on maintainers'
machines with `scripts/oracle/nightly_oracle.py`
([The parity oracle](parity-oracle.md#nightly-runs-on-gpu-machines)).

## Writing a test

- Test the contract at the layer that owns it: backend math in the backend,
  graph compile and persistence in the Engine, message changes at the
  protocol boundary.
- Prefer a committed fixture to a live reference, and committed example
  data to generated data.
- A new public layer, loss, optimizer, scheduler or batcher needs a row in
  `oracle_coverage.json` ([The parity oracle](parity-oracle.md)).
- A new node needs a workflow it is tested in, or the node contract gates
  fail ([Add a node](add-a-node.md)).
- Register the executable with `add_test` so ctest runs it.

## Related

- [The parity oracle](parity-oracle.md)
- [Build from source](build-from-source.md)
- [Coding standards](coding-standards.md)
