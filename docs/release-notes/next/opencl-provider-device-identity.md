kind: fixed
area: training
page: usage/gpu_providers.md

The OpenCL provider now finds the GPU a run chose by the device itself, not by its position in a list. ArrayFire and the provider order OpenCL devices differently on a computer with GPUs of several vendors, so a recurrent layer could run on another GPU than the one chosen in Preferences > Devices. An OpenCL CPU device is now declined with that reason.
