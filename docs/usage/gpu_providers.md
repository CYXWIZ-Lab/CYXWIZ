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
| Attention (MultiHeadAttention with a causal or no mask, a decoder block's self-attention) on CUDA | `cyxwiz.nvidia-cublas-cell` |
| Attention on an OpenCL GPU | `cyxwiz.opencl-cell` |
| Attention on oneAPI | `cyxwiz.oneapi-sycl-attention` |
| PPO's networks (Dense + activations, up to 256 units), its sampling, loss, GAE and Adam on an NVIDIA GPU (CUDA) | `cyxwiz.nvidia-cublas-cell` |

## The NVIDIA provider

`cyxwiz.nvidia-cublas-cell` is CyxWiz's own provider for NVIDIA GPUs (not
cuDNN): CyxWiz's CUDA kernels plus NVIDIA cuBLAS for the matrix multiplies.
For Conv2D it works on the GPU data ArrayFire already holds, so no data goes
back to the host. The Console shows when it is used:

> Conv2D: convolution via cyxwiz.nvidia-cublas-cell (3 -> 16, kernel 3, stride 1, padding 1)

It is needed for Conv2D on CUDA because ArrayFire cannot compile its CUDA
convolution kernel for real image sizes. It comes with the **CUDA GPU pack**.

For PPO it runs the policy and value networks whole and together (one GPU
launch forward for both, two backward), reads each minibatch in place, and
runs PPO's own steps as single launches: about a seventeenth of the GPU
launches of the layer-by-layer path, and three to four times faster. The
Console says which path a run takes:

> PPO: the networks run as fused small-network kernels (cyxwiz.nvidia-cublas-cell)

and the RL Training Dashboard's device chip says it on hover ("Runs as ...").
The run's trace (Studio Debugger, Runtime) records it as `PPO.ComputePath`
at the start, and again as a warning if the run falls back to ArrayFire
operations partway through.

## Why small networks are slow on a GPU, and what CyxWiz does about it

Every GPU operation costs a few microseconds to launch, however little it
computes. A PPO network is small (two layers of 64 units for CartPole, 256
for HalfCheetah), and during the rollout it runs on one row per environment
copy (8 rows for CartPole), so each multiply, bias and activation is far less
work than its launch. Run layer by layer, the GPU spends most of its time
waiting for the next launch: measured on a GTX 1050 Ti, about 23,600 launches
a second with the GPU 11 % busy. On top of that, every environment step waits
for the GPU, because the environment runs on the CPU and needs the actions
back.

This is not particular to CyxWiz. Any framework that runs one GPU kernel per
operation has it: PyTorch in its default (eager) mode runs each `Linear` as
its own matrix multiply and each activation as its own kernel, which is why
Stable-Baselines3 warns against running MLP policies on the GPU and suggests
the CPU. PyTorch's ways around it are a fused or multi-tensor Adam,
`torch.compile` (which merges the element-wise operations but keeps one
matrix multiply per layer) and CUDA Graphs (which replay a recorded step).

On an NVIDIA GPU with the CUDA GPU pack, CyxWiz removes the launches instead:

- each network runs **whole in one kernel** (every layer, bias and
  activation), and the policy and value networks share that one launch; the
  backward pass is two launches for both;
- Adam updates **every tensor of both networks in two launches** (one for the
  gradient norm used by clipping, one for the update);
- PPO's sampling, loss and advantage (GAE) steps are **one launch each**;
- each minibatch is **read in place**, without copying its columns.

Measured on a GTX 1050 Ti (GPU launches counted per PPO update):

| | Layer by layer | CyxWiz kernels |
|---|---|---|
| CartPole, GPU launches per update | 4,595 | 268 |
| HalfCheetah, GPU launches per update | 41,590 | 2,317 |
| CartPole, seconds to a mean return of 475 (seed 1) | 31.9 | 7.9 |

HalfCheetah's 1,000,000 steps take 10-13 minutes on that GPU with the
kernels (about 77 minutes on the CPU, with three runs sharing the computer).

What is left is the environment's round trip: the actions go back to the CPU
every step, as in any framework with CPU environments. On OpenCL and oneAPI,
and on CUDA without the CUDA GPU pack, PPO runs layer by layer and the
advice above holds: small networks may train as fast on the CPU.

## Strict and the providers

The execution policy **Strict (stays on the selected device)** means: every
layer runs on the device the run is on, by the route the run declared before
its first batch. Whether ArrayFire or a CyxWiz provider computes a layer does
not matter; both run on that device. So an LSTM, a GRU or a CNN trains under
Strict on CUDA with the NVIDIA provider.

Before the first batch the run asks the providers about each layer, for the
device the run is on, with the shapes it will train: a layer whose forward
and backward a provider serves is placed on that provider (the Compile
report's Placement, and the run's trace, name it); every other layer on
ArrayFire.

During the run:

- **The provider runs the layer**: counted in the run's trace (Studio
  Debugger, Runtime: provider groups).
- **The provider declines a batch** (a shape it does not take, for example a
  shorter last batch): ArrayFire runs that batch on the same device. The
  trace counts it and warns once per layer.
- **The provider fails** (it tried and broke): a Strict run stops at that
  batch and names the provider, the operation and the layer:

  > provider_route_changed: cyxwiz.nvidia-cublas-cell lstm_forward failed in LSTM (...): ...

  A Compatibility run (the default) runs the layer on ArrayFire instead and
  the trace records the failure.

## One training run per GPU

A training run on a GPU (a model graph's Train, cross-validation, or PPO)
holds that GPU for the whole run, across every Engine and tool you run on
the computer. A second run on the same GPU waits until the first ends and
says why: in the Console, on the Studio toolbar, on the Training or RL
Training Dashboard, and in cross-validation's progress line:

> Waiting for the GPU: another training run is using NVIDIA GeForce GTX 1050 Ti

Stop ends the wait (the run is cancelled). A run on the CPU, or on another
GPU (also a second card of the same model), does not wait. The same card is
one GPU whether a run reaches it through CUDA or OpenCL. Runs started under
another user account of the computer are not always seen. Many small runs
at once on one GPU are what strain a graphics driver, so the Engine never
stacks them.

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
