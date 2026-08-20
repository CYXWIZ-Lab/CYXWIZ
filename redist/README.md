# CyxWiz Redistribution

This directory contains the source tooling for CyxWiz release packages.
Generated staging directories and archives are written under `redist/output/`
and are not source files.

The frozen base/backend-pack layout, signed metadata schemas, trust policy,
and ownership boundaries are defined in
[`BACKEND_PACK_CONTRACT.md`](BACKEND_PACK_CONTRACT.md). The current
`base` and `pack` profiles emit these artifacts directly. The legacy
`minimal` and `full` profiles remain available during migration, with `full`
composed from the same CPU-base and optional-backend closure functions.

## Package Profiles

### Minimal

The minimal package contains the CyxWiz Engine, backend library, resources,
Python bindings, and runtime libraries emitted by the Release build. It
intentionally excludes ArrayFire, external Python, and accelerator drivers.
Customers install a matching ArrayFire distribution and Python 3.12.

### Full

The full profile adds an app-local Python runtime, ArrayFire unified and CPU
runtimes, explicitly selected accelerator packs, and ArrayFire license notices.
CPU is always present. Accelerator packs are optional and named in both the
archive and `PACKAGE_MANIFEST.json`.

| Pack | ArrayFire plugin | Host prerequisite |
|---|---|---|
| `cpu` | CPU | Supported CPU and OS |
| `cuda` | CUDA | Compatible NVIDIA driver |
| `oneapi` | oneAPI/SYCL | Compatible Intel/SYCL provider and driver |
| `opencl` | OpenCL | Vendor OpenCL ICD/provider and driver |

A full package is self-contained for application runtimes, not hardware
drivers. It does not claim that every backend can execute on every machine.

### Base and optional backend packs

The `base` profile emits the Engine, embedded Python, ArrayFire unified/CPU
runtime, and required notices. The `pack` profile emits exactly one optional
`cuda`, `opencl`, or `oneapi` plugin closure. Base and pack archives are
deterministic ZIPs with explicit runtime-set, companion-base, platform, and
ArrayFire ABI identities.

Each archive is accompanied by `.zip.signed.json` canonical signature input
and `.zip.manifest.json`. Without `--signature`, the manifest intentionally has
an empty signature list and is a signing request, not a publishable artifact.

## Implementation

Both shell families call one standard-library implementation:

- `scripts/package_release.py`
- `scripts/sign_pack_manifest.py`
- `bootstrapper/` native Windows launcher and runtime-state resolver
- `scripts/package_minimal.bat` and `package_minimal.sh`
- `scripts/package_full.bat` and `package_full.sh`

The base and optional-pack profiles call `package_release.py` directly so the
release command explicitly names the artifact being produced.

### Windows runtime bootstrapper

The installed app-level `cyxwiz-runtime-bootstrapper.exe` reads only
`runtime/active-runtime.json`, resolves one versioned base and its explicitly
selected optional packs, and launches the Engine from that base. It replaces
the inherited developer `PATH`, removes ArrayFire/Python path overrides, and
the Engine installs `SetDefaultDllDirectories`/`AddDllDirectory` restrictions
before it initializes optional runtimes. Direct Engine launch remains
available for development builds when `CYXWIZ_ACTIVE_RUNTIME_ROOT` is absent.

The Engine's bounded `--package-smoke` diagnostic is accepted only through an
active runtime layout. It verifies that inherited ArrayFire/Python overrides
were removed, activates the exact ArrayFire CPU route, executes a small tensor
calculation, reports machine-readable route truth, and exits before graphical
initialization. Release automation pairs it with the package-local route probe
for the denser forward/backward execution check.

Bootstrap failures are printed to stderr and appended to the package-local
`runtime/bootstrapper.log`. A base archive is a runtime component and no longer
contains the legacy PATH-mutating batch launcher; activation and the app-level
bootstrapper are required.

Repair is intentionally split from the minimal launcher. After the Engine
queues an exact pack Repair and exits, the bootstrapper dispatches the sibling
`cyxwiz-backend-pack-installer` (`.exe` on Windows). That helper verifies the signed catalog
and pack, stages an immutable replacement, and qualifies it through the
isolated route probe before activation. It does not link the backend or
ArrayFire DLLs, and a failed qualification leaves the pack inactive and the
request available for retry.

The packaged desktop application also includes `cyxwiz-installer` (`.exe` on
Windows), a standalone graphical component manager. Recommended, CPU-only,
and Custom package selection live there; the Engine Backend Manager launches
it for Install and Update while retaining runtime and qualification reporting.
The UI, selection model, signed HTTPS delivery, immutable staging, isolated
qualification, and exact helper dispatch are cross-platform and provided
through narrow OS boundaries. Windows uses WinHTTP/process APIs; Linux and
macOS use certificate-verifying HTTPS plus `fork`/`exec`. Linux hardware
recommendation reads stable kernel vendor IDs. macOS remains conservative and
recommends CPU-only unless a later native classifier can prove an eligible
accelerator. Release publication still requires the clean-machine matrix for
each shipped OS.

This keeps validation, backend closure rules, manifests, hashes, and README
rendering consistent across platforms.

## Release Prerequisites

1. Build the Engine and backend in Release configuration.
2. Install Python 3 to run the packaging tool.
3. For a Windows full package, extract the official Python 3.12 embeddable
   package into a dedicated runtime directory. Do not point at a developer
   Python installation or virtual environment.
4. For a full package, install the exact ArrayFire distribution being shipped.
5. Review redistribution terms for every bundled component.

For the verified Windows CPU + oneAPI package, use notices from the exact
runtime releases present in the selected ArrayFire installation. The Intel
notice directory must contain:

- an `mkl`-named path with a oneMKL license and
  `third-party-programs*.txt` files;
- for oneAPI, a `dpcpp`, `compiler`, or `sycl`-named path containing the Intel
  compiler license/EULA, `credist.txt`, and third-party program notices.

The packager rejects a generic or incomplete notice directory.

The default build directory is `build/bin/Release`. Override it with
`--build-dir` when using a preset-specific build tree.

## Commands

Windows minimal:

```batch
redist\scripts\package_minimal.bat --version 0.2.0
```

Windows full CPU plus oneAPI:

```batch
redist\scripts\package_full.bat --version 0.2.0 ^
  --arrayfire-dir "C:\Program Files\ArrayFire\v3" ^
  --python-dir "C:\Python312-embed" ^
  --intel-runtime-license-dir "C:\release-notices\intel" ^
  --backends cpu,oneapi
```

Windows CPU base:

```batch
py -3 redist\scripts\package_release.py base --version 0.2.0 ^
  --arrayfire-dir "C:\Program Files\ArrayFire\v3" ^
  --python-dir "C:\Python312-embed" ^
  --intel-runtime-license-dir "C:\release-notices\intel"
```

Windows OpenCL pack:

```batch
py -3 redist\scripts\package_release.py pack --version 0.2.0 ^
  --backend opencl ^
  --arrayfire-dir "C:\Program Files\ArrayFire\v3"
```

The packager prints the canonical `.signed.json` path. Sign the prepared
manifest as an explicit release step:

```batch
py -3 redist\scripts\sign_pack_manifest.py ^
  redist\output\cyxwiz-af-opencl-3.10.0-1-win64.zip.manifest.json ^
  --private-key "D:\release-secrets\backend-packs-ed25519.pem" ^
  --key-id release-2026
```

The signer uses OpenSSL Ed25519, checks that `.signed.json` exactly matches the
canonical manifest body, verifies its newly generated signature before an
atomic manifest replacement, and never copies the private key. Keep production
keys outside the repository, build trees, output directories, and application
packages. Use the same `--runtime-set-id` and matching `--base-pack-id` when
preparing companion artifacts.

Example notice layout:

```text
C:\release-notices\intel\
  onemkl-2025.2.0.627\
    license.txt
    share\doc\mkl\licensing\third-party-programs*.txt
  dpcpp-compiler-runtime-2025.2.0\
    credist.txt
    compiler\Intel Developer Tools EULA.rtf
    compiler\c\third-party-programs.txt
```

Linux/macOS minimal:

```sh
./redist/scripts/package_minimal.sh --version 0.2.0
```

Self-contained full-package runtime closure validation is currently enabled
only on Windows. The Unix full wrapper fails explicitly until the Linux/macOS
ArrayFire and system-library closures have their own clean-machine evidence.

Run `package_release.py --help` for all inputs. Supported environment
variables are `CYXWIZ_PACKAGING_PYTHON`, `ARRAYFIRE_DIR`, `PYTHON_EMBED`,
`INTEL_RUNTIME_LICENSE_DIR`, and `NVIDIA_RUNTIME_LICENSE_DIR`. Intel runtime
notices are mandatory for every Windows full profile because ArrayFire CPU
packages MKL. CUDA also requires `--nvidia-runtime-license-dir`. Use
`--stage-only` to inspect contents without archiving.

### CMake Targets

CMake exposes the same validated packaging implementation through these
targets:

```powershell
cmake --build build --config Release --target cyxwiz-package-minimal
cmake --build build --config Release --target cyxwiz-package-full
```

Both targets call `scripts/package_release.py`; CMake does not maintain a
second DLL list. They reject non-Release configurations. Configure
full-package inputs with
`CYXWIZ_PACKAGE_ARRAYFIRE_DIR`, `CYXWIZ_PACKAGE_PYTHON_DIR`,
`CYXWIZ_PACKAGE_INTEL_NOTICES_DIR`, `CYXWIZ_PACKAGE_NVIDIA_NOTICES_DIR`, and
`CYXWIZ_PACKAGE_FULL_BACKENDS`. The full target is omitted when a required
input directory is absent. Runtime components are copied only from these
declared roots and recorded in `PACKAGE_MANIFEST.json`; the developer `PATH`
is not a package input.

## Validation Rules

Packaging exits non-zero before archive creation when:

- the Release Engine, backend library, resources, or CyxWiz license is missing;
- a full package lacks Python or cannot identify its version;
- a full package does not contain a Python 3.12 executable and license;
- the Windows Python root is not an embeddable distribution containing
  `python312.dll`, `python312.zip`, and `python312._pth`;
- ArrayFire unified/CPU files or licenses are missing;
- a selected backend plugin or required runtime group is missing;
- required Intel/NVIDIA runtime redistribution notices are absent;
- a README template contains unresolved values.

The Windows oneAPI pack validates and copies the matching plugin, SYCL runtime,
oneMKL SYCL libraries, Unified Runtime loader/adapters, and related Intel files
from one ArrayFire installation. CUDA files are also sourced from that selected
ArrayFire installation to avoid version mixing.

Every staged package contains `PACKAGE_MANIFEST.json` with package identity,
detected dependency versions, external prerequisites, and each staged
component's size, source class, and SHA-256 hash (excluding the manifest
itself). CyxWiz, Python, ArrayFire, and vcpkg license notices are staged with
their associated payloads. Unknown values stay unknown.

## Release Verification

Before publication:

1. Inspect the manifest and third-party notices.
2. Extract on a clean machine or VM.
3. Remove development ArrayFire/Python paths from the environment.
4. Launch through the packaged launcher.
5. Verify backend inventory with the Engine runtime console.
6. Run a bounded operation on every advertised backend.
7. Confirm requested and effective backend/device truth agree.
8. Confirm missing packs and missing drivers produce distinct errors.
9. Record archive size and clean-machine evidence in the release ticket.

A successful GUI launch proves dependency loading only. It does not prove that
a selected accelerator executed computation.

The portable installer workflow and the remaining physical-hardware evidence
are defined in [CLEAN_MACHINE_MATRIX.md](CLEAN_MACHINE_MATRIX.md). Hosted
runners publish installed-size and timing evidence without claiming an
accelerator route that was not physically exercised.

## Official Downloads

- ArrayFire: https://arrayfire.com/download/
- Python 3.12.8 embeddable x64:
  https://www.python.org/ftp/python/3.12.8/python-3.12.8-embed-amd64.zip
- Intel oneAPI DPC++/C++ runtime versions:
  https://www.intel.com/content/www/us/en/developer/articles/tool/compilers-redistributable-libraries-by-version.html
- Intel oneMKL 2025.2.0.627 Windows redistributable package:
  https://www.nuget.org/packages/intelmkl.redist.win-x64/2025.2.0.627
- Microsoft VC++ runtime: https://aka.ms/vs/17/release/vc_redist.x64.exe
- NVIDIA drivers: https://www.nvidia.com/download/index.aspx
- Intel drivers: https://www.intel.com/content/www/us/en/download-center/home.html
- AMD drivers: https://www.amd.com/en/support/download/drivers.html

Do not bundle hardware drivers inside CyxWiz archives. Do not mix `af.dll`
with backend plugins from a different ArrayFire build.

For the validated `0.2.0` CPU + oneAPI build, the prerequisite SHA-256 values
were:

```text
python-3.12.8-embed-amd64.zip
  8d3f33be9eb810f23c102f08475af2854e50484b8e4e06275e937be61ce3d2fb
w_dpcpp_cpp_runtime_p_2025.2.0.768.exe
  6e6e0eb23c6d8bc166fce9a1a12c758910edd2912743e37a1097e2daa6e87216
intelmkl.redist.win-x64.2025.2.0.627.nupkg
  42bf35a13581aa03ecbee62e83e2c6397a45f13ae8aa657c1727fd0335e52c9e
```

Recompute and review hashes for later dependency versions; do not carry these
values forward as approval for a different release.
