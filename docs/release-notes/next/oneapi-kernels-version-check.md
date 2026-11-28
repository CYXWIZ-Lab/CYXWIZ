kind: fixed
area: training
page: usage/gpu_providers.md

The oneAPI provider checks the version of its kernel library before any attention call uses it. A library from another Engine version used to fail at a run's first attention call, which stopped a Strict run; it is now refused with the reason `provider_abi_mismatch` and what to do, and the layer computes on ArrayFire on the same device. The library is loaded and checked once per Engine start, the first time anything asks for a neural provider (usually the first Compile or run), and never again while Compile asks the providers about a graph. The provider is registered only if ArrayFire's oneAPI backend is available at that moment. After installing or replacing the library, restart the Engine.
