---
title: GPU and driver problems
description: Fix compute routes that fail verification, training that runs on the CPU, oneAPI failures and float32-only GPUs.
type: troubleshooting
area: install
audience: user
status: stable
since: 1.0.7
owner: install
last_reviewed: 2026-11-25
related:
  - ../install/gpu-packs.md
  - ../concepts/gpu-providers.md
  - ../known-issues.md
  - compile-and-training.md
---

The Engine computes through ArrayFire on a **compute route**: a device plus
a runtime (CPU, CUDA, OpenCL or oneAPI). Training uses only routes that
passed verification on your machine. Routes and their status are in
**File > Preferences... > Devices**; see
[Add a GPU pack and verify it](../install/gpu-packs.md).

## No GPU in Preferences > Devices

**Cause:** GPU packs are published for Windows x64 only. Linux and macOS
installs compute on the CPU. On Windows, no GPU pack is installed yet.

**Fix:** on Windows, install a pack with the **CyxWiz Installer**
(**Individual components** tab), then verify it. See
[Add a GPU pack and verify it](../install/gpu-packs.md).

## "Not verified yet. Verify the routes on this device before training on it."

**Cause:** the route was detected but has no saved verification result.
Training does not use it until it passes.

**Fix:** select **Verify all**, or **Verify again** on the route's row.

## "Needs driver update"

The route's summary ends with "Install or update the device driver, then
verify again."

**Cause:** the device driver or a runtime library the route needs is
missing, for example no NVIDIA display driver for CUDA, or no OpenCL driver.
The pack does not install drivers.

**Fix:**

1. Install the newest driver from your GPU vendor (NVIDIA or Intel).
2. Restart the computer.
3. Select **Verify again** on the route.

## "Crashed during verification in ...; ... The Engine kept running and this route is disabled."

The route shows **Not supported on this device**.

**Cause:** the GPU runtime crashed while it ran one of the qualification
operations. Verification runs in a separate process, so the Engine was not
affected.

**Fix:** update the driver and verify again. If it still crashes, use
another route on the same GPU (OpenCL on an Intel GPU) or the CPU. Check
[Known issues](../known-issues.md) for your device.

## "Timed out in ... after N s" or "Failed in ..."

The route shows **Failed**.

**Cause:** an operation of the qualification did not finish in time or gave
a wrong result. **Details** on the route names the operation, the failure
stage and a recommended action.

**Fix:** follow the recommended action in **Details**. Usually: update the
provider or driver and verify again, or use another verified route.

## "Saved verification results belong to a different runtime. Run Verify All again."

**Cause:** a pack or the Engine was updated after the last verification.

**Fix:** select **Verify all**.

## "No route on this device passed verification; training uses a verified route on another device."

**Cause:** every route of this GPU failed or was never verified, so the
Engine trains on another verified route, often the CPU.

**Fix:** verify the routes of this device. If they fail, see the entries
above.

## Training uses the CPU although a pack is installed

**Cause:** the GPU route is not verified, or another route is selected.

**Fix:**

1. Open **File > Preferences... > Devices** and verify the GPU route.
2. Select the circle in front of it, then **Use this device**.

The compile report's **Where each layer runs** tab shows, before training,
whether each layer runs on the GPU. See
[Compile and run a graph](../user-guide/graph-editor/compile-and-run.md).

## "Waiting for the GPU: another training run is using ..."

**Cause:** another training run, in this Engine or another one on the
computer, is using the same GPU. A GPU runs one training at a time.

**Fix:** wait; the run starts when the GPU is free ("The GPU is free: the run
starts on ..." in the Console). To start now, stop the other run.

## oneAPI fails on Intel UHD 630

**Cause:** the oneAPI runtime fails while it activates older Intel
integrated graphics (UHD 630, 8th and 9th generation Core), before any work
runs.

**Fix:** use OpenCL on that GPU. The Engine recommends OpenCL on Intel GPUs,
and it is verified on the UHD 630.

## Results on an Intel Iris Xe differ from the CPU in the 5th digit

**Cause:** GPUs without double-precision support, such as Intel Iris Xe
(OpenCL and oneAPI), run classic ML, clustering, data transforms,
dimensionality reduction and linear algebra in float32 instead of float64.
Results agree with the references to float32 precision (5e-5 relative), and
iterative methods such as t-SNE agree less closely.

**Fix:** for float64 results, use a device that has it: an NVIDIA GPU
(CUDA or OpenCL) or the CPU route. See
[Numerical accuracy](../concepts/numerical-accuracy.md).

## Your own ArrayFire code gives wrong results on oneAPI

**Cause:** two ArrayFire 3.10.0 defects on oneAPI: indexed `af::max` and
`af::min` along dim 1, and indexing a slice with an index array. The Engine
avoids both.

**Fix:** in your plugins and scripts, reduce along dim 0 (transpose first),
and index slices with `af::lookup`. Details in
[Known issues](../known-issues.md).

## Related

- [Add a GPU pack and verify it](../install/gpu-packs.md)
- [What runs where on a GPU](../concepts/gpu-providers.md)
- [Compute back ends](../concepts/compute-back-ends.md)
- [Known issues](../known-issues.md)
