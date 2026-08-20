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
