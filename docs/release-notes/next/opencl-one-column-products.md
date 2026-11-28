kind: fixed
area: training
page: usage/gpu_providers.md

On an OpenCL device, a layer with one output (a Dense layer of 1 unit, as in most single-value regression models) could give NaN or wrong values from its first batch, most often after an earlier run in the same session had diverged to NaN. The cause is in the OpenCL math library ArrayFire uses: for a product with one result column it read the old contents of the result's memory. Every matrix product in the Engine now avoids that path on OpenCL; CUDA and the CPU were not affected. Statistics and time-series checks that had their own workaround for random NaN rows on OpenCL use the same fix.
