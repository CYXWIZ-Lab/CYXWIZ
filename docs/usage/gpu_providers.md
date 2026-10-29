# What runs where on a GPU

CyxWiz computes with **ArrayFire** on the device you choose in Preferences >
Devices (CUDA, OpenCL or CPU). A few layers ArrayFire cannot run well on a
GPU are run by CyxWiz's own **providers** on the same device; nothing else
changes for you.

| On | Who runs it |
|---|---|
| Every layer, by default | ArrayFire on the chosen device |
| RNN / LSTM / GRU on an NVIDIA GPU (CUDA) | `cyxwiz.nvidia-cublas-cell` |
| Conv2D on an NVIDIA GPU (CUDA) | `cyxwiz.nvidia-cublas-cell` |
| RNN / LSTM / GRU on an OpenCL GPU | `cyxwiz.opencl-cell` |

## The NVIDIA provider

`cyxwiz.nvidia-cublas-cell` is CyxWiz's own provider for NVIDIA GPUs (not
cuDNN): CyxWiz's CUDA kernels plus NVIDIA cuBLAS for the matrix multiplies.
For Conv2D it works on the GPU data ArrayFire already holds, so no data goes
back to the host. The Console shows when it is used:

> Conv2D: convolution via cyxwiz.nvidia-cublas-cell (3 -> 16, kernel 3, stride 1, padding 1)

It is needed for Conv2D on CUDA because ArrayFire cannot compile its CUDA
convolution kernel for real image sizes. It comes with the **CUDA GPU pack**.

## Without it

A CNN (any graph with Conv2D) trained on CUDA where the NVIDIA provider is
not installed does not stop half-way. Training starts on ArrayFire CPU
instead, and the Console says why:

> Conv2D cannot run on ArrayFire CUDA here: ArrayFire cannot compile
> its CUDA convolution kernel and the NVIDIA provider
> (cyxwiz.nvidia-cublas-cell) is not installed. Install the CUDA GPU pack, or
> choose ArrayFire OpenCL on this GPU in Preferences > Devices (faster than
> the CPU)

With the execution policy **Strict (stays on the selected device)** in
Preferences the run does not start and shows the same reason.

ArrayFire OpenCL runs Conv2D on the same NVIDIA GPU without the provider;
it is slower than the provider and faster than the CPU.

For diagnosis, `CYXWIZ_DISABLE_NEURAL_PROVIDERS=1` in the environment turns
every provider off for that Engine (each layer then runs on ArrayFire alone).
