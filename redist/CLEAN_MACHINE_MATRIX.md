# Clean-Machine Release Matrix

This matrix separates portable package evidence from physical accelerator
qualification. A hosted CPU runner or a mocked device record never counts as
CUDA, OpenCL, or oneAPI execution evidence.

## Hosted installer evidence

The `Installer Native Build` workflow transfers each staged standalone
installer to a fresh Windows x64, Ubuntu x64, or macOS arm64 runner. The fresh
runner verifies package-local dependencies, starts the installer in bounded
smoke mode, checks the maintenance helper's fail-closed invocation, and runs
the packaged lifecycle contract executables.

Each runner uploads `cyxwiz-installer-evidence-<platform>` containing:

- installed and validation-only byte sizes plus the SHA-256, role, and size of
  every staged file;
- dependency-audit status and duration;
- observed exit code and duration for every startup or contract check; and
- explicit `not_run` records for CUDA, OpenCL, and oneAPI.

This proves the standalone installer artifact. It does not prove that a full
CPU-base Engine package launches or trains.

## Hosted Windows CPU-base evidence

The `Windows CPU Base Native` workflow builds the production CPU-base ZIP and
transfers only that artifact to a fresh Windows runner. The runner installs it
into the versioned runtime layout, deliberately contaminates inherited
ArrayFire, Python, and `PATH` variables, and launches the real Engine through
the production bootstrapper's bounded `--package-smoke` mode. It then runs the
package-local route probe on ArrayFire CPU for tensor conversion, BCE forward
and backward, and a dense forward/backward benchmark.

The evidence artifact records installed bytes, file hashes, Engine/probe
durations, dense benchmark timing, and the package-relative locations of the
loaded ArrayFire unified, CPU, and oneMKL runtime modules. CUDA, OpenCL, and
oneAPI remain explicit `not_run` entries. CI emits an unsigned signing request;
it is qualification evidence, not a publishable release artifact.

Latest qualifying checkpoint: GitHub Actions run `32460137920` passed on a
separate fresh Windows runner. The archive is 307,438,064 bytes and installs
317 files totaling 884,984,243 bytes. Engine bootstrap smoke completed in
1,373.634 ms; the five CPU route checks completed in 22.971-160.888 ms; and
`dense-compute-v1` reported a 6.0449 ms median iteration. The module audit
resolved `arrayfire/bin/af.dll`, `arrayfire/bin/afcpu.dll`, and
`arrayfire/bin/mkl_rt.2.dll` inside the installed base while contaminated
development paths were present. This closes the Windows CPU-base row with
bounded CPU execution evidence; it does not claim a full optimizer loop or
physical accelerator qualification.

## Failure-contract coverage

| Release condition | Native contract |
| --- | --- |
| Upgrade, atomic activation, and rollback | `test_backend_pack_lifecycle_service` |
| Repair and removal | `test_backend_pack_installer`, `test_backend_pack_maintenance` |
| Disk budget exhausted | `test_backend_pack_delivery`, `test_backend_pack_installer` |
| Interrupted download, install, repair, or removal | delivery, installer, and maintenance contracts |
| Corrupt archive or component | delivery and installer contracts |
| Revoked signing key or invalid signature | `test_backend_pack_metadata_verifier` |
| Unsupported application downgrade | `test_backend_pack_metadata_verifier` |

These deterministic failure injections prove transaction invariants. Release
qualification must still exercise disk exhaustion and interruption on a clean
VM before closing the release-matrix item.

## Hardware evidence still required

For each row below, retain the package manifest, device inventory, bounded
route-probe output, requested/effective route truth, installed bytes, and
startup/probe durations. An unavailable provider is a skip or rejection, not a
pass through CPU fallback.

| Machine | Required result |
| --- | --- |
| Clean supported host, CPU base only | Engine launch and bounded CPU training pass with development paths removed |
| Catalog-supported NVIDIA system | CUDA package installs and the CUDA route passes |
| Each supported OpenCL vendor/provider class | Matching OpenCL package and route pass |
| Catalog-supported Intel system | oneAPI package and route pass |
| Intel UHD 630 validation host | oneAPI is rejected while the independently installed OpenCL route remains available |

Do not copy ticket names, local fixture IDs, absolute paths, or machine-unique
identifiers into production-facing output.
