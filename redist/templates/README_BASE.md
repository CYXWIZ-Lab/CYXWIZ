# CyxWiz CPU Base {{VERSION}}

This is a signed runtime component for `{{PLATFORM}}`, not a standalone
portable application. It contains the CyxWiz Engine, Python runtime, and
ArrayFire {{ARRAYFIRE_VERSION}} unified/CPU closure.

Install it beneath the CyxWiz runtime root as:

```text
runtime/base/<base-pack-id>/
```

The installer or Backend Manager must validate its signed manifest and hashes,
then atomically select it in `runtime/active-runtime.json`. Start CyxWiz with
the app-level `cyxwiz-runtime-bootstrapper.exe`; do not invoke the Engine or
modify `PATH` directly.

The bootstrapper accepts only the base and optional packs named by the active
runtime state. Hardware drivers remain host prerequisites.
