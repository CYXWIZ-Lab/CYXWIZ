# CyxWiz Engine Full Distribution

- CyxWiz version: `{{VERSION}}`
- Platform: `{{PLATFORM}}`
- ArrayFire version: `{{ARRAYFIRE_VERSION}}`
- Packaged ArrayFire backends: `{{BACKENDS}}`

## Package Contract

This archive contains an app-local CyxWiz runtime, Python runtime, ArrayFire
unified library, ArrayFire CPU backend, and only the accelerator packs listed
above.

The Python directory originates from the official Python 3.12 Windows
embeddable distribution, not a copied developer installation.

Hardware drivers and providers supplied by the operating system or hardware
vendor are not bundled. A packaged backend can execute only when the target
machine provides compatible hardware and drivers.

Windows also requires the Microsoft Visual C++ 2015-2022 x64 Redistributable:

https://aka.ms/vs/17/release/vc_redist.x64.exe

`PACKAGE_MANIFEST.json` is the authority for component contents, versions,
sizes, source classes, and SHA-256 hashes (excluding the manifest itself).

## Start

Windows:

```batch
start_cyxwiz.bat
```

Linux or macOS:

```sh
chmod +x cyxwiz
./cyxwiz
```

Use the launcher so CyxWiz resolves app-local ArrayFire and Python runtimes.
Direct execution may select unrelated libraries from the global loader path.

## Backend Meaning

| Pack | Compute path | Host prerequisite |
|---|---|---|
| `cpu` | ArrayFire CPU | Supported CPU and OS |
| `cuda` | ArrayFire CUDA | Compatible NVIDIA driver |
| `oneapi` | ArrayFire oneAPI/SYCL | Compatible SYCL provider and driver |
| `opencl` | ArrayFire OpenCL | Vendor OpenCL ICD/provider and driver |

ArrayFire CPU is not CyxWiz native C++ fallback. Native fallback is a separate,
explicitly reported compatibility path.

The oneAPI pack includes the redistributable SYCL/oneMKL closure selected by
release engineering. It does not include the Intel compiler toolkit or a
hardware driver. The CUDA pack likewise excludes the NVIDIA display driver.

## Verify Execution

1. Launch through the packaged launcher.
2. Open the Engine runtime console.
3. Run `show device backends`.
4. Run `show device available`.
5. Run the backend-specific device command when applicable.
6. Select a device and run the bounded execution/training preflight.
7. Confirm requested and effective backend/device agree.
8. Confirm the verdict reports no undeclared native fallback.

A successful GUI launch proves dependency loading only. It is not accelerator
execution evidence.

## Troubleshooting

### Missing backend pack

Check the backend list above and the manifest. Install a CyxWiz artifact that
names the required pack; do not copy a plugin from another ArrayFire release.

### Pack is present but no device is available

Install or update the hardware vendor's driver/provider:

- NVIDIA: https://www.nvidia.com/download/index.aspx
- Intel: https://www.intel.com/content/www/us/en/download-center/home.html
- AMD: https://www.amd.com/en/support/download/drivers.html

### oneAPI metadata error 301

`oneapi::devprop not supported` is a metadata limitation, not proof that the
plugin is missing. Use device inventory plus the bounded execution probe.

### Requested and effective devices differ

Review the run's fallback/activation verdict. Do not assume the GUI preference
is the device that executed.

### Python project creation fails

Verify the `python/` directory exists in the manifest and launch through the
provided script.

## Integrity and Licenses

The package contains the CyxWiz `LICENSE` and ArrayFire notices under
`THIRD_PARTY_LICENSES/ArrayFire`. Notices for bundled vcpkg runtimes are under
`THIRD_PARTY_LICENSES/vcpkg`; Intel/NVIDIA runtime notices use their named
directories; and Python's license is retained in `python/`. Verify files
against the manifest before publication or support analysis.

When reporting a problem, include the manifest, backend inventory,
requested/effective device, bounded preflight result, and activation log.
