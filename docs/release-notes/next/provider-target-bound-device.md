kind: fixed
area: training
page: usage/gpu_providers.md

Neural providers now take their device from the device the run bound: training, Run Test, RL training, policy playback and the stress test. A thread whose ArrayFire device differs from it is never served by a provider; ArrayFire runs the operation and the run's trace warns with the reason `target_mismatch`. Compile's placement report asks the providers about the device chosen in Preferences > Devices instead of the device the window happened to be on, and a failed device query is reported instead of silently counting as the CPU. Policy playback and the stress test are now observed and follow the Strict policy as training does.
