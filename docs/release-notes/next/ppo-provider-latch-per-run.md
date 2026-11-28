kind: fixed
area: training
page: usage/gpu_providers.md

A PPO sampling, loss or advantage kernel that failed on the GPU provider in one run no longer stays on ArrayFire for every later run of the same Engine session. The fallback now lasts for that run only, as it does for the fused networks and the joint Adam step; the next run asks the provider again.
