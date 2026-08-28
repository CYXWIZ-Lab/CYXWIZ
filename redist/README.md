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
- `scripts/prepare_backend_pack_repository.py`
- `bootstrapper/` native desktop launchers and shared runtime-state resolver
- `scripts/package_minimal.bat` and `package_minimal.sh`
- `scripts/package_full.bat` and `package_full.sh`

The base and optional-pack profiles call `package_release.py` directly so the
release command explicitly names the artifact being produced.

Standalone installer packaging also requires an initial signed metadata
bundle. Configure `CYXWIZ_INSTALLER_BOOTSTRAP_METADATA_DIR` with a directory
containing `trust/` and `catalogs/`; CMake stages those files below the
installer's adjacent `runtime/`. `verify_installer_package.py` fails closed if
the CPU base, an optional backend pack, catalog, trust store, or corresponding
cached manifest is absent. The native CI fixture uses ephemeral keys and
non-routable package URLs and is test evidence only, never release metadata.

The optional first-stage setup boundary consumes a separate deterministic
installer bundle; it never owns package selection or product lifecycle logic.
`package_installer_bundle.py` accepts a fully staged installer and emits one
bounded ZIP, an exact component inventory in a schema-1 descriptor, and the
canonical detached signing input. The descriptor pins the CyxWiz and bundle
versions, release channel, platform, architecture, minimum setup version,
validity window, archive digest, and every extracted file digest. Links,
case-folded path collisions, missing GUI/helper/bootstrap metadata, oversized
payloads, in-stage output, mutation during packaging, and replacement of an
existing versioned artifact fail closed. The archive and signing input publish
before the unsigned descriptor, so interruption cannot publish an
authoritative incomplete descriptor.

Release signing is a separate explicit operation:

```text
python redist/scripts/package_installer_bundle.py <stage> <output> ...
python redist/scripts/sign_installer_bundle.py <descriptor> \
  --private-key <ed25519.pem> --key-id <trusted-installer-key>
```

Ordinary builds never receive the private key. The resulting signed contract
is the input for the small setup verifier/acquirer/launcher; the consolidated
`cyxwiz-installer` remains the only graphical installation experience.

The manually dispatched `Installer Alpha Candidate` workflow is the bounded
pre-publication build. It requires the protected `cyxwiz-alpha` environment,
the repository variable `CYXWIZ_ALPHA_CANDIDATE_ENABLED=true`, and the
repository secret `CYXWIZ_INSTALLER_TRUST_STORE_B64`. That secret contains
only the base64-encoded public `trusted-keys.json`; private signing keys are
not inputs to this workflow. For the requested immutable tag it builds Windows
x64, Linux x64, macOS Intel, and macOS Apple Silicon setup packages with the
public trust root embedded and the exact future GitHub Release descriptor URL
compiled in. Release-configured setup and clean installer-stage artifacts use
the `cyxwiz-release-*` artifact prefix. The workflow deliberately does not
create a tag or GitHub Release. Publication remains blocked until the signed
base/optional-pack matrix, signed installer descriptors, final bootstrap
metadata, and platform code signatures have passed their own release gate.

### Native runtime bootstrapper

The installed app-level `cyxwiz-runtime-bootstrapper` (`.exe` on Windows)
reads only
`runtime/active-runtime.json`, resolves one versioned base and its explicitly
selected optional packs, and launches the Engine from that base. Windows
replaces inherited developer `PATH`, and the Engine installs
`SetDefaultDllDirectories`/`AddDllDirectory` restrictions before initializing
optional runtimes. Linux and macOS replace inherited dynamic-loader paths with
the exact active base/pack directories and launch through `fork`/`exec` without
a shell. All platforms remove ArrayFire/Python overrides, propagate exact
runtime identity, support `--installer`, and apply queued Repair only after the
Engine exits. Direct Engine launch remains available for development builds
when `CYXWIZ_ACTIVE_RUNTIME_ROOT` is absent.

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
queues an exact pack Repair and exits, the bootstrapper dispatches the exact
`cyxwiz-backend-pack-installer` (`.exe` on Windows) from the active signed base.
That helper verifies the signed catalog and pack, stages an immutable
replacement, and qualifies it through the isolated route probe before
activation. It does not link the backend or ArrayFire DLLs, and a failed
qualification leaves the pack inactive and the request available for retry.

Fresh delivery rechecks and atomically publishes the signed base's stable
`cyxwiz-product-removal-finalizer` first and
`cyxwiz-runtime-bootstrapper` second beside `runtime/` before generation-1
activation. Publishing the finalizer first cannot strand a launcher without
its removal closure.
Its `--installer` mode resolves the same versioned `cyxwiz-installer` GUI from
the active base, allowing later platform registration to point at one stable
launcher without copying the GUI or dependency closure into the product root.

After the signed CPU base is qualified and activated, the exact delivery helper
registers that stable launcher with the selected install scope. Windows creates
Engine and Installer Start Menu links and an Apps & Features entry; Linux writes
current-user or system desktop entries; macOS publishes Engine and Installer
application bundles in the product root. Every maintenance entry resolves the
installed GUI through `--installer`, and none changes the machine-wide loader
environment. The Apps & Features uninstall command reopens that GUI through
the stable bootstrapper. In maintenance mode the GUI offers full removal only
for an exact installed runtime with a valid ownership receipt and finalizer.
After explicit acknowledgement, it queues the version-pinned removal request
and exits with the private removal code. The stable bootstrapper then schedules
the detached finalizer and retains the lifetime token until its own process
exits, ensuring installed binaries are released before unregistration,
quarantine, and cleanup begin. Direct base-GUI launch cannot enable this flow.

The matching unregistration boundary is idempotent and removes only the exact
CyxWiz Start Menu/Apps & Features, desktop-entry, or application-bundle
integration for the selected scope. Linux and macOS fail closed when a managed
entry contains unknown changes or files. This boundary does not recursively
delete the product root; runtime deletion remains a separate transaction.

Successful product registration also creates the bounded hidden receipt
`.cyxwiz-installation.json` in the normalized product root. It records a unique
installation ID, that exact root, and the selected scope. Repair registration
preserves the ID and scope; a missing, malformed, redirected, or scope-mismatched
receipt blocks unregistration and future full-product removal. The receipt is
ownership evidence, not sufficient deletion authority by itself: a deferred
removal finalizer must also revalidate the active runtime set and generation
immediately before deleting any product-owned files.

The shared removal-authorization boundary performs that second check without
deleting anything. It accepts only a direct, normalized, non-root product
directory with an exact regular stable launcher, captures the receipt ID and
complete resolved active-runtime identity, and requires the receipt, scope,
runtime set, generation, base, and optional-pack inventory to remain identical
when revalidated. The later finalizer must consume this typed authorization;
path strings supplied directly to a delete operation are not sufficient.

Confirmed removal is handed across processes through the bounded hidden
`.cyxwiz-removal-request.json` file in the product root. Queueing first captures
fresh typed authorization, then atomically publishes an exact schema containing
the install root, scope, receipt ID, runtime set, generation, base, and selected
optional packs. The stable bootstrapper and detached finalizer load that same
schema and repeat live authorization validation; copying the request to another
root or changing the active runtime makes it stale. A new explicit confirmation
may replace a corrupt or stale request, but it never replaces ownership evidence.

The CPU base also carries the dependency-isolated
`cyxwiz-product-removal-finalizer` (`.exe` on Windows). Its first operation is
to block on an inherited read-only lifetime pipe; the stable bootstrapper owns
the only write end, so EOF is tied to process exit rather than a reusable PID.
Only after EOF does the finalizer reload the durable request and repeat live
authorization validation. The Windows finalizer uses the static CRT and depends
only on system `KERNEL32.dll`, `ADVAPI32.dll`, `ole32.dll`, and `SHELL32.dll`,
allowing it to run from a temporary directory after the product runtime is
released. After authorization it performs the ordered removal transaction:
exact native unregistration, atomic quarantine, then bounded no-follow cleanup.
If quarantine fails, exact native registration is restored. Cleanup failure
keeps the partially cleaned quarantine and its recovery evidence; it never
restores a partial product tree.

The detached scheduler copies that exact, 16 MiB-bounded, non-redirected
executable into an exclusive temporary directory and launches it with only the read token
inherited. Windows uses an explicit process handle list; POSIX uses a detached
double-fork/`exec` boundary. A bounded result marker proves the child stayed
blocked until the parent token closed. The handoff contract uses a dedicated
non-destructive child, so tests never mutate a developer's native registration.

Removal request schema 2 also pins the exact CyxWiz product version from the
active signed base pack's bounded, non-redirected `RUNTIME_VERSIONS.json`.
Registration and removal share the same safe-version rule. Schema 1 removal
requests fail closed because they cannot prove which version of native OS
registration may be removed or restored.

Native unregistration preflights ownership before changing external state.
Windows verifies the exact install root, version, uninstall commands, and both
Start Menu shortcut identities before deleting fixed CyxWiz entries. Linux
validates both managed desktop entries before removing either one; macOS
likewise preflights both exact managed application bundles. An entry belonging
to another installation or containing unmanaged changes fails closed.

After every running installed process has released the product, the removal
transaction revalidates authorization and atomically renames the exact product
root to a deterministic sibling `.cyxwiz-removing-<install-id>` quarantine. It
never overwrites or merges an existing quarantine. The moved receipt continues
to name the original root and must match the pinned ID and scope; failed
post-rename validation attempts an immediate rollback. Quarantine does not
recursively delete content—bounded no-follow cleanup remains a separate finalizer
stage so interruption cannot be mistaken for successful removal.

The cleanup boundary now preflights that quarantine before deleting payload.
It is bounded to 256 directory levels and one million entries. POSIX traverses
with `openat`/`fstatat(AT_SYMLINK_NOFOLLOW)`/`unlinkat`, pins inode and device
identity, removes links as links, and refuses another filesystem device.
Windows pins directories without delete sharing, rejects redirected traversal,
and removes files, directories, junctions, and reparse entries through exact
handles. Both platforms keep the removal request and ownership receipt while
payload cleanup is incomplete, remove the request next, and remove the receipt
only after all other entries are gone. An interrupted cleanup can therefore be
retried from the deterministic quarantine without following external targets.

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

Configure `CYXWIZ_INSTALLER_CATALOG_URL` at CMake configure time with the
direct HTTPS URL of the production signed catalog. Leaving it empty keeps the
packaged verified catalog available and makes online Refresh report that no
source is configured. Local integration runs can supply
`--catalog-url <https-url>` without changing the package. Refresh downloads
bounded catalog and manifest documents off the render thread, verifies the
complete snapshot with the packaged trust store, and publishes the catalog
only after every eligible manifest is trusted. Verified online metadata is
stored below the selected runtime root, not beside the installer executable.
Failure retains the previous verified catalog.

After local verification, the same installer shows each route's typed result,
a bounded failure reason and next action, plus benchmark medians when present.
It identifies a best measured configuration only when two or more active,
verified routes have comparable fixed-benchmark evidence. Internal evidence
keys and engineering ticket names are never customer-facing result text.

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

The signer requires OpenSSL 3 with Ed25519 `pkeyutl` support, checks that
`.signed.json` exactly matches the
canonical manifest body, verifies its newly generated signature before an
atomic manifest replacement, and never copies the private key. Keep production
keys outside the repository, build trees, output directories, and application
packages. Use the same `--runtime-set-id` and matching `--base-pack-id` when
preparing companion artifacts.

After every selected base/backend manifest is signed and its archive remains
beside it, assemble the exact hosted and installer-bootstrap repository:

```bat
py -3 redist\scripts\prepare_backend_pack_repository.py ^
  --manifest redist\output\cyxwiz-base-0.2.0-1-win64.zip.manifest.json ^
  --manifest redist\output\cyxwiz-af-opencl-3.10.0-1-win64.zip.manifest.json ^
  --trust-root D:\release-trust\trusted-keys.json ^
  --catalog-private-key D:\release-secrets\catalog-ed25519.pem ^
  --catalog-key-id catalog-2026 ^
  --pack-key-id release-2026 ^
  --catalog-id cyxwiz-alpha-2026-08 ^
  --generated-utc 2026-08-25T12:00:00Z ^
  --expires-utc 2026-09-25T12:00:00Z ^
  --minimum-client-version 0.2.0 ^
  --base-url https://packages.example.com/cyxwiz/alpha ^
  --output redist\output\alpha-repository
```

The command verifies trust roles, every pack signature, archive size and
SHA-256, companion base, runtime set, platform, architecture, and ArrayFire ABI
before signing the catalog. The configured private catalog key must match the
app-bundled public trust root. Inputs with multiple valid pack signatures
require an explicit `--pack-key-id` so key rotation cannot change catalog
authority implicitly.

Publish only `alpha-repository/hosted/` at the exact non-redirecting HTTPS base
URL. Configure the installer with
`CYXWIZ_INSTALLER_BOOTSTRAP_METADATA_DIR=.../alpha-repository/bootstrap` and
`CYXWIZ_INSTALLER_CATALOG_URL=https://packages.example.com/cyxwiz/alpha/catalogs/current.json`.
The bootstrap tree intentionally excludes the large pack archives. The
catalog-signing private key is read from its external path and is never copied
to either output tree.

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
