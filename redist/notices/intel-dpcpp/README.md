# Intel DPC++ redistributables list

`credist.txt` is the Redistributables list referenced by the Intel End User
License Agreement for Developer Tools (Version April 2023). The oneAPI backend
pack redistributes `sycl8.dll` and its Unified Runtime companions from the
ArrayFire 3.10 install, and ships this list with the Intel notices.

Source: Intel oneAPI DPC++/C++ Compiler 2025.1 (`<oneapi>/compiler/2025.1/
licensing/credist.txt`), SHA-256
`bf0c16122c70aa5fd9d79b3273a17fb8f8e4203b5cc6cbd1a6b53cbf1ffd1922`. It is kept
here because Intel publishes it only inside the compiler installers; the other
Intel notices (DPC++ license and third-party programs, TBB, UMF, oneMKL) are
downloaded by CI from pinned, hash-checked packages.
