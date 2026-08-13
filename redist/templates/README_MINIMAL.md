# CyxWiz Engine Minimal Distribution

- CyxWiz version: `{{VERSION}}`
- Platform: `{{PLATFORM}}`
- ArrayFire backends: `{{BACKENDS}}`

## Package Contract

This package contains the CyxWiz application payload emitted by the Release
build. It does not include ArrayFire or Python. Install those prerequisites
before launching the Engine.

The bundled component files and SHA-256 hashes are recorded in
`PACKAGE_MANIFEST.json` (excluding the manifest itself). Do not infer package
contents from a fixed DLL count.
Third-party notices for bundled vcpkg runtimes are under
`THIRD_PARTY_LICENSES/vcpkg`.

## Required Downloads

### Python 3.12

Download the official 64-bit Python 3.12 installer:

https://www.python.org/downloads/

Project creation and Python bindings require the matching Python 3.12 ABI.

### Microsoft Visual C++ Runtime on Windows

Install the current supported x64 redistributable:

https://aka.ms/vs/17/release/vc_redist.x64.exe

### ArrayFire

Download the official ArrayFire package:

https://arrayfire.com/download/

CyxWiz is currently validated against the ArrayFire 3.10 release family.
Install one matching distribution containing `af.dll` (or `libaf`) and the
backend plugin you intend to use. Never combine the unified library with
plugins from another build or architecture.

| Selection | Required plugin | Required system capability |
|---|---|---|
| ArrayFire CPU | `afcpu` | Supported CPU |
| CUDA | `afcuda` | Compatible NVIDIA driver |
| oneAPI | `afoneapi` | Compatible SYCL provider and device driver |
| OpenCL | `afopencl` | Vendor OpenCL ICD/provider and driver |

ArrayFire CPU executes ArrayFire operations on the CPU. It is not CyxWiz's
native C++ CPU fallback.

The complete Intel oneAPI compiler toolkit is not normally required when the
ArrayFire installation supplies matching SYCL and oneMKL redistributable
runtimes. A compatible driver/provider is still required. CUDA likewise needs
a compatible NVIDIA driver; a developer toolkit is needed only when the
selected package does not supply its runtime or for CUDA development.

Driver downloads:

- NVIDIA: https://www.nvidia.com/download/index.aspx
- Intel: https://www.intel.com/content/www/us/en/download-center/home.html
- AMD: https://www.amd.com/en/support/download/drivers.html

## Configure Library Search Paths

### Windows

The normal ArrayFire installer uses:

```text
C:\Program Files\ArrayFire\v3\lib
```

Add that directory to persistent user/system `PATH`, or set it for the
packaged launcher:

```batch
set "CYXWIZ_ARRAYFIRE_DIR=C:\Program Files\ArrayFire\v3\lib"
start_cyxwiz.bat
```

The `set` command affects only the current terminal.

### Linux

```sh
export LD_LIBRARY_PATH=/opt/arrayfire/lib:${LD_LIBRARY_PATH:-}
./cyxwiz
```

### macOS

```sh
export DYLD_LIBRARY_PATH=/opt/arrayfire/lib:${DYLD_LIBRARY_PATH:-}
./cyxwiz
```

## Verify Execution

1. Launch with `start_cyxwiz.bat` on Windows or `./cyxwiz` on Unix.
2. Open the Engine runtime console.
3. Run `show device backends`.
4. Run `show device available`.
5. For oneAPI, run `show device oneapi`.
6. Select the intended device and run a bounded training/execution preflight.
7. Confirm requested and effective backend/device agree.
8. Confirm no undeclared native fallback occurred.

Opening the GUI proves only that startup libraries loaded. It does not prove
that computation ran on the selected backend.

## Troubleshooting

### `af.dll` or `libaf` is missing

ArrayFire is absent from the loader path. Install it and configure the path
above. On Windows, `where af.dll` should resolve the intended build.

### Backend plugin is missing

The unified library loaded, but the selected `afcpu`, `afcuda`,
`afoneapi`, or `afopencl` plugin is unavailable. Install the matching
plugin from the same ArrayFire distribution.

### A transitive runtime is missing

Reinstall the complete matching ArrayFire package. Do not download individual
DLLs from unofficial sites or mix runtime files from different releases.

### Backend exists but no compatible device is found

Install or update the vendor driver/provider. A backend plugin cannot replace a
missing hardware driver.

### oneAPI reports metadata error 301

`oneapi::devprop not supported` means optional metadata is unsupported. It
does not by itself mean oneAPI is absent. Use inventory plus the bounded
execution probe.

### Requested and effective devices differ

Treat this as fallback or activation failure. Review the runtime verdict before
training. Strict mode should reject before native fallback.

## Support Evidence

Include the package manifest, CyxWiz/ArrayFire versions, requested/effective
device, device commands, bounded preflight result, and concise activation log.
