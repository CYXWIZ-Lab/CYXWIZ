# Backend compute frameworks: what executes tensors in CyxWiz

Status: decision record, ruled 2026-09-24. Internal detail and evidence live
in the private engineering tickets tofix68 (native neural providers),
tofix114 (LibTorch question) and done28 (PyTorch reference policy).

## The decision

CyxWiz executes tensors through **ArrayFire** (CPU, CUDA, OpenCL backends)
plus **native neural providers** that register behind one interface
(`INeuralNetworkProvider`, `cyxwiz-backend/include/cyxwiz/neural_provider.h`)
and are selected per run by the device the run targets:

| Executor | Role | Selected when |
| --- | --- | --- |
| ArrayFire | general tensor execution, transformer blocks, dense/conv/norm | default for every layer it can run |
| `cyxwiz.nvidia-cublas-cell` | recurrent training (RNN/LSTM/GRU, stacked, bidirectional) and Conv2D forward / backward on CUDA | run targets a CUDA device |
| `cyxwiz.opencl-cell` | the same recurrent contract on OpenCL GPUs | run targets an OpenCL device and hidden >= 16 |
| CPU reference | small deterministic tests, portable fallback | strict mode off and nothing else can run |

**LibTorch is not a backend compute framework of the engine.** It is the
numerical oracle for parity tests: `test_computation_truth_transformer_primitives`
links it when `CYXWIZ_HAS_PYTORCH` is set. No engine compute target
(`cyxwiz-backend`, `cyxwiz-engine`, `pycyxwiz`) may link it.

One other linker exists and is out of this decision's scope: the server
node's model loader (`cyxwiz-server-node/src/model_loader.cpp`) can import
external TorchScript models for serving, the same way it imports ONNX and
GGUF models. That is model-format interoperability for models built
elsewhere, not execution of CyxWiz graphs, and it is off unless the build
finds LibTorch. It shares the `CYXWIZ_ENABLE_PYTORCH` option with the
oracle test; split the option when the server node loader is next touched
so a developer can enable the oracle without building the loader.

## The NVIDIA provider (`cyxwiz.nvidia-cublas-cell`)

CyxWiz's own provider for NVIDIA GPUs (tofix68), not cuDNN: our CUDA kernels,
compiled at run time with NVRTC (fused RNN / LSTM / GRU cells, im2col for 2D
convolution), with NVIDIA cuBLAS for the matrix multiplies.

- **Recurrent layers**: ArrayFire's JIT cannot run recurrent training at
  useful sizes on any backend (it flattens the expression tree into one
  kernel); the provider runs the cells.
- **Conv2D** (TOFIX140 A1b): ArrayFire 3.10's CUDA backend cannot compile its
  own convolution kernel (unwrap / moddims) for real image sizes
  (`NVRTC_ERROR_COMPILATION`); ArrayFire's OpenCL and CPU backends run Conv2D
  fine. The provider runs the convolution device-resident: on ArrayFire's CUDA
  stream, on ArrayFire's device buffers, with no host copies (the v2
  `device_resident` boundary; the recurrent ops still use the v1 boundary
  with explicit host copies). The Console says "Conv2D: convolution via
  cyxwiz.nvidia-cublas-cell".
- **Built** with `CYXWIZ_ENABLE_NVIDIA_DNN_PROVIDER=ON` and the CUDA Toolkit
  (CUDA runtime, cuBLAS, NVRTC, driver API) as a module of its own,
  `cyxwiz-nvidia-provider` (`.dll` / `.so`), beside the backend in `bin`
  (TOFIX178 1.0(c), option 1, 2026-10-28). `cyxwiz-backend` links no CUDA
  library, so the base runs on every machine. The release workflow
  (`cpu-base-native.yml`) builds the module against pinned CUDA 12.6
  redistributable archives, and `package_release.py` ships it only in the
  CUDA GPU pack (`runtime/`, beside afcuda); a CUDA pack without it fails to
  package.
- **Loaded** by the provider registry when it starts: by name (the pack's
  `runtime/` is on the Engine's PATH), then from beside `cyxwiz-backend`. The
  module exports `cyxwiz_nvidia_provider_abi` (must equal
  `CYXWIZ_NEURAL_PROVIDER_MODULE_ABI`, `neural_provider.h`),
  `cyxwiz_nvidia_provider_register` and
  `cyxwiz_nvidia_provider_device_memory`. A missing module, or one that
  cannot load (no NVIDIA driver: `nvcuda.dll`), counts as not installed
  (a Debug log line); a loaded one logs "NVIDIA neural provider module
  loaded: <path>".
- **Without it**, a CUDA run of a model with Conv2D does not start on CUDA:
  under the Compatibility policy it moves to ArrayFire CPU and says why;
  under Strict it refuses with the reason (TOFIX178 1.0(d)).
- `CYXWIZ_DISABLE_NEURAL_PROVIDERS=1` hides every provider for a process (the
  portable path everywhere), for diagnosis and tests.

## Why not LibTorch as an additional backend

- **The gaps a provider exists for are closed without it.** Recurrent
  training, which ArrayFire's JIT could not run, is served by the two native
  tenants with CPU parity, leak gates and an end-to-end training gate. The
  ArrayFire CUDA transformer block runs fully on the GPU at scale
  (S512/D256: 16-20x over ArrayFire CPU, zero fallbacks), so a transformer
  provider would be a speed option over a working path.
- **The v1 provider boundary neutralizes LibTorch's strengths.** Providers
  execute single ops with explicit host copies; CyxWiz keeps autograd,
  optimizer and checkpoint ownership. At that boundary LibTorch cannot bring
  fused autograd, subgraph execution or mixed precision, and pays the same
  copies our own tenants pay.
- **Cost is concrete.** LibTorch 2.7.0 is 1.1 GB (CPU) or 4.7 GB across 36
  DLLs (CUDA 12.6), pins a CUDA+MSVC toolchain per PyTorch release, bundles
  cuBLAS/cuDNN DLLs with the same names ArrayFire's CUDA backend loads
  (first-loaded wins; a mismatch is a crash or a silent numerical change),
  and needs an installer, ABI and redistribution review of its own.
- **The modern transformer variants do not need it.** RMSNorm, RoPE/ALiBi,
  GLU-family FFNs, norm position and bias policy are small primitives that
  are implemented natively and validated against PyTorch fixtures, which is
  what the done28 reference policy already prescribes.
- **Policy.** done28: PyTorch is the numerical reference, kept outside the
  runtime boundary, never a required engine dependency.

## What would change the decision

Any one of these reopens tofix114 phase 2:

1. A measured ArrayFire failure or regression at a shape the product
   needs, for a variant the native path cannot fix within its budget,
   recorded as placement evidence first.
2. A decision to move to subgraph-level provider binding (the v2
   boundary), which is where LibTorch's fusion and autograd would pay.
3. A device platform neither ArrayFire nor the two native tenants can
   serve.

If reopened, the shape is fixed: LibTorch enters as a **third tenant**
behind the existing provider interface, serving the transformer-block
family only (forward and backward, LibTorch autograd inside the op,
gradients returned as CyxWiz tensors), registered for CPU and CUDA
targets, shipped as an **optional pack** that the core installer never
requires. Selection would be capability-gap by default (ArrayFire first;
the tenant is queried only for a contract the native path declares
unsupported) with an explicit per-run override, and the placement audit
would show requested versus effective provider so the switch is never
silent. Duplicate palette nodes per backend are not permitted.

## How to keep the oracle live

The oracle compares against LibTorch only when the build finds it; otherwise
the truth test uses baked fixtures. For a developer build:

```
cmake -S . -B build-oracle -DCYXWIZ_ENABLE_PYTORCH=ON -DTORCH_DIR=<libtorch>/libtorch
```

and put `<libtorch>/libtorch/lib` on PATH when running
`test_computation_truth_transformer_primitives`. Use a dedicated build
directory; the reconfigure rebuilds broadly.
