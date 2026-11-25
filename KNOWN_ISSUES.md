# Known issues

The full list, with the affected devices, the details and the status of each entry, is in the
CyxWiz documentation: `docs/known-issues.md` in the source repository
(https://github.com/CYXWIZ-Lab/CYXWIZ/blob/HEAD/docs/known-issues.md). This file ships with every
release package, next to `LICENSE`, so the short list below is here when you are offline.

Last updated: 2026-11-25.

## GPU runtimes

- **ArrayFire oneAPI: indexed min / max along rows return wrong results** (Intel GPUs, oneAPI,
  ArrayFire 3.10.0). No effect in CyxWiz. In your own ArrayFire code on oneAPI, reduce along dim 0
  instead of dim 1. Upstream: arrayfire/arrayfire#3732.
- **ArrayFire oneAPI: indexing a slice with an index array ignores the slice's offset** (Intel
  GPUs, oneAPI, ArrayFire 3.10.0). No effect in CyxWiz. In your own code, use `af::lookup` or
  `copy()` the slice first. Upstream: arrayfire/arrayfire#3733.
- **oneAPI does not run on Intel UHD 630.** Use OpenCL on that GPU; the Engine recommends it on
  Intel GPUs.

## Precision

- **GPUs without float64 compute analysis in float32** (for example Intel Iris Xe). Results agree
  with the references to float32 precision. For float64, use NVIDIA CUDA or OpenCL, or the CPU.

## Text

- **Right-to-left text is joined only in the dashboard and plots.** Read Arabic values in a
  dashboard or plot; Hebrew letters show as "?". The data itself is unchanged.

## Reporting a new issue

Open an issue with the device and compute route (Preferences > Devices), the CyxWiz version
(Help > About CyxWiz..., Copy details), and what you expected and what you saw.
