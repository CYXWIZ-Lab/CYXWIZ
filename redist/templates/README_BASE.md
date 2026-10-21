# CyxWiz CPU Base {{VERSION}}

This is a signed runtime component for `{{PLATFORM}}`, not a standalone
portable application. It contains the CyxWiz Engine and ArrayFire
{{ARRAYFIRE_VERSION}} unified/CPU closure. {{PYTHON_RUNTIME}}

Install it beneath the CyxWiz runtime root as:

```text
runtime/base/<base-pack-id>/
```

The installer or Backend Manager must validate its signed manifest and hashes,
then atomically select it in `runtime/active-runtime.json`. Start CyxWiz with
the app-level `{{BOOTSTRAPPER}}`; do not invoke the Engine or
modify `PATH` directly.

The bootstrapper accepts only the base and optional packs named by the active
runtime state. Hardware drivers remain host prerequisites.

Known problems in the GPU runtimes CyxWiz uses, and what to do about them, are listed in
`KNOWN_ISSUES.md` in this package.

The `sdk/` folder is the CyxWiz plugin SDK for C++ plugins with trainable
nodes: headers, the backend's import library, `find_package(CyxWizSDK)` and a
template plugin, under the Apache License 2.0 (`sdk/LICENSE`, `sdk/NOTICE`).
`cyxwiz-plugin-check` loads a plugin folder the way the Engine does and checks
its nodes. Start with `sdk/template/README.md`.
