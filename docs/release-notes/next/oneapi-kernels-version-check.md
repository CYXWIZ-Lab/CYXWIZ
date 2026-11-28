kind: fixed
area: training
page: usage/gpu_providers.md

The oneAPI provider checks the version of its kernel library before any attention call uses it. A library from another Engine version used to fail at a run's first attention call, which stopped a Strict run; it is now refused with the reason `provider_abi_mismatch` and what to do, and the layer computes on ArrayFire on the same device.
