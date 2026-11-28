kind: fixed
area: training
page: usage/gpu_providers.md

The OpenCL provider now finds the GPU a run chose by the device itself, not by its position in a list. ArrayFire and the provider order OpenCL devices differently on a computer with GPUs of several vendors, so a recurrent layer could run on another GPU than the one chosen in Preferences > Devices. It matches the device by its handle, then by its PCI address or UUID, then by its name and vendor (a driver that reaches the GPU through another OpenCL loader no longer has it declined for the rest of the session), and only last by position, with a warning in the log. An OpenCL CPU device is now declined with that reason.
